//! Coupling layer; supports affine (Glow) and additive (NICE/RealNVP)
//! parameterisations selected at config time via [`CouplingType`]. Splits
//! channels into two halves along the channel axis.
//!
//! Convention: **first half** `xa` is unchanged; **second half** `xb` is transformed.
//! The conditioning network maps `xa` → either `(raw_scale, shift)` for affine
//! (`out_channels_factor = 2`) or just `shift` for additive (`factor = 1`),
//! with `ConvBlock` (`in_channels = C/2`).
//!
//! ## Affine (`CouplingType::Affine`)
//!
//! The raw scale output is passed through a **scaled sigmoid** that bakes the lower
//! bound directly into the parameterisation:
//!     `s = SCALE_MIN + (1 − SCALE_MIN) · σ(raw_scale + SCALE_BIAS)`
//! with `SCALE_BIAS = 2.0` and `SCALE_MIN ≈ 0.082` (matches the previous
//! `clamp_min(LOG_S_MIN = −2.5)` floor). This keeps `s ∈ [SCALE_MIN, 1)` with
//! everywhere-non-zero gradient (the previous `clamp_min` masked gradients past the
//! floor), so the optimiser can always pull saturated elements back into the live
//! region. Initialisation: `σ(2) ≈ 0.881` ⇒ `s ≈ 0.891`, layer is near identity.
//! Bounding `s` keeps `1/s ≤ 1/SCALE_MIN ≈ 12` in `inverse`, preventing the
//! reconstruction blow-up the unbounded `exp(log_s)` formulation suffers once any
//! `|log_s|` grows past ~3 in f32 (see `INVERT_DIAGNOSTICS_PLAN.md`).
//!
//! Forward: `yb = xb * s + shift`. Log-determinant: `sum(log s)` (shift depends only on `xa`,
//! so it does not change the Jacobian w.r.t. `xb`).
//!
//! `shift` is bounded via `tanh` so large unconstrained conv outputs cannot blow up `inverse`,
//! which uses `(yb - shift) / s` over dozens of steps in `f32`.
//!
//! ## Additive (`CouplingType::Additive`)
//!
//! `yb = xb + t(xa)`, `log_det = 0` (volume-preserving). The conditioning conv
//! emits `half` channels (no scale head). The shift `t` is **unbounded** — unlike
//! the affine path, additive's inverse `xb = yb − t` has no division, so a
//! magnitude cap on `t` would buy no extra invertibility and only cost
//! expressivity (and gradient flow in any saturated tanh regime). All convs in
//! the conditioning net use Salimans–Kingma weight normalisation; the Lipschitz
//! of `t` is not hard-bounded.
//!
//! **Checkpoint compatibility:** affine and additive checkpoints are not
//! interchangeable — `out_channels_factor` differs and the record load fails.
//! See the README "Migrating pre-additive checkpoints" section.

use burn::{
    config::Config,
    module::{Ignored, Module},
    prelude::Backend,
    tensor::{activation, Device},
    Tensor,
};

use crate::models::blocks::conv::{ConvBlock, ConvBlockConfig};

/// Selects between the affine and additive coupling parameterisations.
///
/// **Affine** (`y_b = s · x_b + t`): full Glow coupling with bounded scale via
/// scaled sigmoid (see module-level doc). Higher capacity per layer.
///
/// **Additive** (`y_b = x_b + t`): NICE-style coupling with `s = 1`, `log_det = 0`.
/// Strictly more stable in the inverse direction — `y_b − t` contains no
/// division, so worst-case `f32` reconstruction error per step is just `|t|`'s
/// rounding error rather than `1/s`-amplified. Lower capacity per layer; pair
/// with more steps if needed.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum CouplingType {
    Affine,
    Additive,
}

// ── Coupling parameterisation constants (affine path) ──────────────────────
//
// These three constants jointly bound the per-step Lipschitz constant of the
// **affine** coupling map (and its inverse) so a deep stack stays numerically
// invertible in `f32`. None of them apply to additive coupling: additive's
// inverse is `yb − t` (no division), so bounding `|t|` via tanh would only cost
// expressivity / gradient flow in saturated regions. All convs in the additive
// conditioning net use the same WeightNorm as affine.
//
// * Inside ActNorm + InvConv the activations are normalised to ≈unit variance.
//   Therefore a `tanh`-based bound at ±4σ on the affine `shift` covers
//   >99.99% of clean signal — empirically well beyond anything the network
//   needs to represent.
// * `s` lives in `[SCALE_MIN, 1)` via a scaled sigmoid. The lower bound
//   `SCALE_MIN ≈ 0.0821` matches the previous `exp(LOG_S_MIN = −2.5)` floor but
//   is applied **inside the parameterisation** (smooth) rather than as a hard
//   `clamp_min` (which masks gradients past the floor). Per-element worst-case
//   inverse amplification is `1/SCALE_MIN ≈ 12`.
// * `SCALE_BIAS = 2` puts `σ(2) ≈ 0.881` and therefore `s ≈ 0.891` at init
//   (`raw_s = 0` from zero-initialised conv3). The layer is **near-identity**,
//   not exactly identity — the asymptotic identity `s = 1` is unreachable
//   under a smooth sigmoid. This trades a 12% scale at init for a smooth
//   gradient surface; rosinality / Glow reference checkpoints converge from
//   here within a handful of steps.
// * The combined inverse-direction worst case per affine coupling step is
//   `|xb_inv| ≤ (|yb| + SHIFT_BOUND) / SCALE_MIN ≈ (|yb| + 4) / 0.082 ≈ 12·|yb| + 49`.
//   With a stack of K · L ≈ 32 couplings the cumulative bound is loose but
//   stays finite, which is the property `inverse` needs.

const SCALE_BIAS: f32 = 2.0;
const SCALE_MIN: f32 = 0.0821;
const SHIFT_BOUND: f32 = 4.0;

#[derive(Config, Debug)]
pub struct CouplingConfig {
    /// Total channel count; must be even. Each of `xa` and `xb` uses `num_channels / 2`.
    pub num_channels: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
}

impl CouplingConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Coupling<B> {
        assert_eq!(
            self.num_channels % 2,
            0,
            "Coupling: num_channels must be even"
        );
        let half = self.num_channels / 2;

        // Affine: factor=2 (raw_s + shift). Additive: factor=1 (shift only).
        let out_factor = match self.coupling_type {
            CouplingType::Affine => 2_usize,
            CouplingType::Additive => 1_usize,
        };

        let net = ConvBlockConfig::new(half)
            .with_hidden_features(self.hidden_features)
            .with_out_channels_factor(out_factor)
            .init(device);
        Coupling {
            net,
            coupling_type: Ignored(self.coupling_type),
        }
    }
}

#[derive(Module, Debug)]
pub struct Coupling<B: Backend> {
    net: ConvBlock<B>,
    /// Captured at init from `CouplingConfig::coupling_type`. The conv shape
    /// already encodes this (factor 2 vs 1) but storing it explicitly keeps the
    /// `scale_and_shift` branches readable and lets diagnostic helpers report it.
    coupling_type: Ignored<CouplingType>,
}

impl<B: Backend> Coupling<B> {
    /// Run the conditioning net on `xa` and return `(s_opt, log_s_opt, shift)`.
    ///
    /// **Affine** (`s = SCALE_MIN + (1 − SCALE_MIN) · σ(raw_s + SCALE_BIAS)`):
    /// `s ∈ [SCALE_MIN, 1)`, `log_s ∈ [log(SCALE_MIN), 0)`. Both are returned as
    /// `Some(_)`. The sigmoid is materialised via `log_sigmoid + exp` for a
    /// numerically robust path even when the conv net pushes `raw_s + SCALE_BIAS`
    /// very negative.
    ///
    /// **Additive** (`s = 1`, `log_s = 0`): both options are `None` and the conv
    /// only emits `shift`. Callers must treat `None` as the identity scale: skip
    /// the `xb * s` multiply and skip the log-det sum (it would be a sum over an
    /// all-zeros tensor). This keeps additive coupling allocation- and
    /// multiply-free in the common path.
    fn scale_and_shift(
        &self,
        xa: Tensor<B, 4>,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>, Tensor<B, 4>) {
        let half = xa.dims()[1];
        let st = self.net.forward(xa);

        match *self.coupling_type {
            CouplingType::Affine => {
                let raw_s = st.clone().narrow(1, 0, half);
                let shift_raw = st.narrow(1, half, half);
                let shift = activation::tanh(shift_raw).mul_scalar(SHIFT_BOUND);

                let log_sig = activation::log_sigmoid(raw_s.add_scalar(SCALE_BIAS));
                let sig = log_sig.exp();
                let s = sig.mul_scalar(1.0 - SCALE_MIN).add_scalar(SCALE_MIN);
                let log_s = s.clone().log();
                (Some(s), Some(log_s), shift)
            }
            CouplingType::Additive => {
                // Conv emitted exactly `half` channels (out_channels_factor = 1).
                // No magnitude cap on `t`: the inverse `yb − t` has no division,
                // so bounding `|t|` would not improve numerical invertibility.
                (None, None, st)
            }
        }
    }

    /// Returns the coupling type captured at init. Useful for diagnostics and
    /// resume-time validation.
    pub fn coupling_type(&self) -> CouplingType {
        *self.coupling_type
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let [b, c, _h, _w] = x.dims();
        debug_assert_eq!(c % 2, 0);
        let half = c / 2;

        let xa = x.clone().narrow(1, 0, half);
        let xb = x.narrow(1, half, half);
        let device = xb.device();

        let (s_opt, log_s_opt, shift) = self.scale_and_shift(xa.clone());

        // Additive coupling has Jacobian determinant 1; skip the all-zeros sum and the
        // `xb * 1` multiply rather than allocate full-size `ones_like` / `zeros_like`.
        let log_det = match log_s_opt {
            Some(log_s) => log_s.sum_dims_squeeze::<1, _>(&[1, 2, 3]),
            None => Tensor::zeros([b], &device),
        };
        let yb = match s_opt {
            Some(s) => xb * s + shift,
            None => xb + shift,
        };
        let y = Tensor::cat(vec![xa, yb], 1);

        (y, log_det)
    }

    /// `(min, max)` of the `log_s` actually applied in `forward`/`inverse` (post-sigmoid).
    /// `log_s ∈ [log(SCALE_MIN), 0)` for affine; `log_s ≡ 0` for additive (returned as
    /// scalar zero tensors so the existing diagnostic plumbing keeps working).
    pub fn log_s_extrema_on_input(&self, x: Tensor<B, 4>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let [_b, c, _h, _w] = x.dims();
        debug_assert_eq!(c % 2, 0);
        let half = c / 2;
        let xa = x.clone().narrow(1, 0, half);
        let (_, log_s_opt, _) = self.scale_and_shift(xa);
        match log_s_opt {
            Some(log_s) => (log_s.clone().min(), log_s.max()),
            None => {
                let device = x.device();
                (Tensor::zeros([1], &device), Tensor::zeros([1], &device))
            }
        }
    }

    /// `max(|shift|)` actually applied in `forward`/`inverse` for the given input. With
    /// `tanh` bounding, this should always be `≤ SHIFT_BOUND`; an outlier here would point
    /// to a regression in the gating logic rather than a learning issue.
    pub fn shift_abs_max_on_input(&self, x: Tensor<B, 4>) -> Tensor<B, 1> {
        let [_b, c, _h, _w] = x.dims();
        debug_assert_eq!(c % 2, 0);
        let half = c / 2;
        let xa = x.narrow(1, 0, half);
        let (_, _, shift) = self.scale_and_shift(xa);
        shift.abs().max()
    }

    pub fn inverse(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, c, _, _] = y.dims();
        debug_assert_eq!(c % 2, 0);
        let half = c / 2;

        let ya = y.clone().narrow(1, 0, half);
        let yb = y.narrow(1, half, half);

        let (s_opt, _, shift) = self.scale_and_shift(ya.clone());
        let xb = match s_opt {
            // Affine: `(yb − shift) / s`. Bound `1/s ≤ 1/SCALE_MIN ≈ 12`.
            Some(s) => (yb - shift) / s,
            // Additive: `(yb − shift)` — no division, exact in f32.
            None => yb - shift,
        };
        Tensor::cat(vec![ya, xb], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{check_closeness, Distribution, Tensor};
    use rstest::*;

    type B = NdArray;

    fn has_nan(t: Tensor<B, 1>) -> bool {
        t.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .any(|v| v.is_nan())
    }

    #[fixture]
    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[rstest]
    #[case::affine_4(CouplingType::Affine, 4, 2, 8, 8)]
    #[case::affine_8(CouplingType::Affine, 8, 1, 4, 4)]
    #[case::additive_4(CouplingType::Additive, 4, 2, 8, 8)]
    #[case::additive_8(CouplingType::Additive, 8, 1, 4, 4)]
    fn coupling_forward_inverse_roundtrip(
        device: NdArrayDevice,
        #[case] coupling_type: CouplingType,
        #[case] channels: usize,
        #[case] batch: usize,
        #[case] h: usize,
        #[case] w: usize,
    ) {
        let layer = CouplingConfig::new(channels)
            .with_coupling_type(coupling_type)
            .init(&device);
        let x = Tensor::<B, 4>::random(
            [batch, channels, h, w],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let (y, _) = layer.forward(x.clone());
        let x2 = layer.inverse(y);
        check_closeness(&x2, &x);
        assert!(
            x2.all_close(x, Some(1e-4), Some(1e-4)),
            "inverse(forward(x)) should recover x ({coupling_type:?})"
        );
    }

    /// Additive coupling has Jacobian determinant 1 → log_det must be (numerically)
    /// zero. Tighter than affine, where `log_det` can drift on the order of `1e-3`
    /// per layer due to the `log(s)` reduction. We assert `< 1e-9` rather than
    /// strict equality: with the `Option<Tensor>` skip-path the value is currently
    /// produced by `Tensor::zeros` and equality holds, but the bound stays valid
    /// if a future implementation rebuilds it from `log(constant 1.0)` and accrues
    /// rounding noise.
    #[rstest]
    fn additive_coupling_log_det_is_zero(device: NdArrayDevice) {
        let layer = CouplingConfig::new(8)
            .with_coupling_type(CouplingType::Additive)
            .init::<B>(&device);
        let x = Tensor::<B, 4>::random([2, 8, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (_, log_det) = layer.forward(x);
        let max_abs: f32 = log_det.abs().max().into_scalar();
        assert!(
            max_abs < 1e-9,
            "additive log_det must be (numerically) zero, got max|log_det|={max_abs}"
        );
    }

    #[rstest]
    fn affine_coupling_log_det_finite(device: NdArrayDevice) {
        let layer = CouplingConfig::new(8).init(&device);
        let x = Tensor::<B, 4>::random([2, 8, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (_, log_det) = layer.forward(x);
        assert!(!has_nan(log_det), "log_det should not be NaN");
    }

    /// Even when the conditioning net's pre-activations are forced large by a wide-σ input,
    /// the bounded `shift` must stay within `[-SHIFT_BOUND, SHIFT_BOUND]`.
    #[rstest]
    fn affine_coupling_shift_is_bounded(device: NdArrayDevice) {
        let layer = CouplingConfig::new(8).init(&device);
        let x = Tensor::<B, 4>::random([2, 8, 4, 4], Distribution::Normal(0.0, 5.0), &device);
        let max_abs: f32 = layer.shift_abs_max_on_input(x).into_scalar();
        assert!(
            max_abs.is_finite() && max_abs <= SHIFT_BOUND + 1e-5,
            "shift |{max_abs}| should be ≤ SHIFT_BOUND={SHIFT_BOUND}"
        );
    }
}
