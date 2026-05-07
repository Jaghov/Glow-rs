//! Composable Glow flow step: ActNorm → invertible 1×1 conv → affine coupling.

use burn::{config::Config, module::Module, prelude::Backend, Tensor};

use super::actnorm::{ActNorm, ActnormConfig};
use super::coupling::{Coupling, CouplingConfig, CouplingType};
use super::invconv::{InvConv1x1, InvConv1x1Config, TriangularInverse};

#[derive(Config, Debug)]
pub struct GlowStepConfig {
    pub num_channels: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
}

impl GlowStepConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GlowStep<B> {
        assert_eq!(
            self.num_channels % 2,
            0,
            "GlowStep: num_channels must be even (coupling splits channels)"
        );
        GlowStep {
            actnorm: ActnormConfig::new(self.num_channels).init(device),
            invconv: InvConv1x1Config::new(self.num_channels).init(device),
            coupling: CouplingConfig::new(self.num_channels)
                .with_hidden_features(self.hidden_features)
                .with_coupling_type(self.coupling_type)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct GlowStep<B: Backend> {
    actnorm: ActNorm<B>,
    invconv: InvConv1x1<B>,
    coupling: Coupling<B>,
}

impl<B: Backend> GlowStep<B> {
    /// Data-dependent ActNorm initialization: call once with a representative batch
    /// before the first training forward pass.
    pub fn init_actnorm(&mut self, example: Tensor<B, 4>) {
        self.actnorm.init(example);
    }

    pub fn actnorm_ref(&self) -> &ActNorm<B> {
        &self.actnorm
    }
    pub fn invconv_ref(&self) -> &InvConv1x1<B> {
        &self.invconv
    }
    pub fn coupling_ref(&self) -> &Coupling<B> {
        &self.coupling
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let [b, c, _, _] = x.dims();
        let (x, ld1) = self.actnorm.forward(x);
        // ActNorm returns one `(h*w)*weight[c]` per channel; reduce to a scalar per batch row.
        let ld1 = ld1.reshape([b, c]).sum_dims_squeeze::<1, _>(&[1]);
        let (x, ld2) = self.invconv.forward(x);
        let (x, ld3) = self.coupling.forward(x);
        let log_det = ld1 + ld2 + ld3;
        (x, log_det)
    }
}

impl<B: Backend + TriangularInverse> GlowStep<B> {
    pub fn inverse(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.coupling.inverse(y);
        let y = self.invconv.inverse(y);
        self.actnorm.inverse(y)
    }
}

/// Per-step reconstruction-diagnostic snapshot. Each tensor is shape `[1]`; reading them as
/// f32 forces a host sync, so callers should reduce/print all rows in one batch.
pub struct GlowStepDiag<B: Backend> {
    pub actnorm_weight_min: Tensor<B, 1>,
    pub actnorm_weight_max: Tensor<B, 1>,
    pub invconv_log_s_min: Tensor<B, 1>,
    pub invconv_log_s_max: Tensor<B, 1>,
    /// `log_s` from coupling, evaluated on the post-actnorm/invconv `h` actually fed to coupling.
    pub coupling_log_s_min: Tensor<B, 1>,
    pub coupling_log_s_max: Tensor<B, 1>,
    /// `max(|shift|)` from coupling on the same `h`. With `tanh` bounding this is `≤ SHIFT_BOUND`;
    /// values riding the bound across many steps suggest cumulative pressure on the inverse path.
    pub coupling_shift_abs_max: Tensor<B, 1>,
}

impl<B: Backend> GlowStep<B> {
    /// Run one forward step and also return per-component diagnostic extrema. The returned
    /// `h` matches `forward(x).0`; logdet is dropped because diagnostics don't need it.
    pub fn forward_with_diag(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, GlowStepDiag<B>) {
        let (an_min, an_max) = self.actnorm.weight_extrema();
        let (ic_min, ic_max) = self.invconv.log_w_s_extrema();
        let (h, _) = self.actnorm.forward(x);
        let (h, _) = self.invconv.forward(h);
        let (cp_min, cp_max) = self.coupling.log_s_extrema_on_input(h.clone());
        let cp_shift = self.coupling.shift_abs_max_on_input(h.clone());
        let (h, _) = self.coupling.forward(h);
        (
            h,
            GlowStepDiag {
                actnorm_weight_min: an_min,
                actnorm_weight_max: an_max,
                invconv_log_s_min: ic_min,
                invconv_log_s_max: ic_max,
                coupling_log_s_min: cp_min,
                coupling_log_s_max: cp_max,
                coupling_shift_abs_max: cp_shift,
            },
        )
    }
}

impl<B: Backend> GlowBlock<B> {
    /// Walk every step in the block, collecting `GlowStepDiag` rows. Mirrors `forward`.
    pub fn forward_with_diag(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<GlowStepDiag<B>>) {
        let mut h = super::squeeze2d(x, 2);
        let mut rows = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let (h2, diag) = step.forward_with_diag(h);
            h = h2;
            rows.push(diag);
        }
        (h, rows)
    }
}

// ── SplitBlock ────────────────────────────────────────────────────────────────

/// Zero-parameter channel-split layer used between multi-scale Glow levels.
///
/// `forward(x)` splits channels in half: `(z = x[.., :C/2, .., ..], h = x[.., C/2:, .., ..])`.
/// `inverse(z, h)` concatenates them back: `cat([z, h], dim=1)`.
pub struct SplitBlock;

impl SplitBlock {
    pub fn forward<B: Backend>(x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_, c, _, _] = x.dims();
        debug_assert_eq!(c % 2, 0, "SplitBlock: channel count must be even");
        let half = c / 2;
        (x.clone().narrow(1, 0, half), x.narrow(1, half, half))
    }

    pub fn inverse<B: Backend>(z: Tensor<B, 4>, h: Tensor<B, 4>) -> Tensor<B, 4> {
        Tensor::cat(vec![z, h], 1)
    }
}

// ── GlowBlock ────────────────────────────────────────────────────────────────

/// Configuration for one level of the multi-scale Glow architecture.
#[derive(Config, Debug)]
pub struct GlowBlockConfig {
    /// Channel count AFTER squeeze (= 4 × C_in for squeeze factor 2).
    pub num_channels: usize,
    /// Number of GlowSteps per block (K in the paper).
    pub num_steps: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
}

impl GlowBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GlowBlock<B> {
        assert!(self.num_steps >= 1, "GlowBlock: num_steps must be >= 1");
        let steps = (0..self.num_steps)
            .map(|_| {
                GlowStepConfig::new(self.num_channels)
                    .with_hidden_features(self.hidden_features)
                    .with_coupling_type(self.coupling_type)
                    .init(device)
            })
            .collect();
        GlowBlock { steps }
    }
}

/// One level of the Glow model: `squeeze2d` followed by K `GlowStep`s.
///
/// Input shape: `[B, C_in, H, W]`
/// Output shape after forward: `[B, 4*C_in, H/2, W/2]`
#[derive(Module, Debug)]
pub struct GlowBlock<B: Backend> {
    steps: Vec<GlowStep<B>>,
}

impl<B: Backend> GlowBlock<B> {
    /// Data-dependent ActNorm initialisation; call once before the first training step.
    pub fn init_actnorm(&mut self, example: Tensor<B, 4>) {
        let device = example.device();
        let mut h = super::squeeze2d(example, 2);
        for step in &mut self.steps {
            step.init_actnorm(h.clone());
            let (h2, _) = step.forward(h);
            // Detach from the autodiff graph — these passes are only for distribution
            // estimation, not training, so we must not let the graph accumulate over K steps.
            h = Tensor::from_data(h2.into_data(), &device);
        }
    }

    pub fn steps_for_diag(&self) -> &[GlowStep<B>] {
        &self.steps
    }

    /// `[B, C_in, H, W]` → `([B, 4*C_in, H/2, W/2], log_det [B])`
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let mut h = super::squeeze2d(x, 2);
        let mut log_det_acc: Option<Tensor<B, 1>> = None;
        for step in &self.steps {
            let (h2, ld) = step.forward(h);
            h = h2;
            log_det_acc = Some(match log_det_acc {
                None => ld,
                Some(acc) => acc + ld,
            });
        }
        (
            h,
            log_det_acc.expect("GlowBlock must have at least one step"),
        )
    }
}

impl<B: Backend + TriangularInverse> GlowBlock<B> {
    /// `[B, 4*C_in, H/2, W/2]` → `[B, C_in, H, W]`
    pub fn inverse(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = y;
        for step in self.steps.iter().rev() {
            h = step.inverse(h);
        }
        super::unsqueeze2d(h, 2)
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

    fn has_nan_log_det(t: Tensor<B, 1>) -> bool {
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
    #[case::twelve_ch(12)]
    #[case::sixteen_ch(16)]
    fn glow_step_is_invertible(device: NdArrayDevice, #[case] channels: usize) {
        let step = GlowStepConfig::new(channels).init(&device);
        let input = Tensor::<B, 4>::random(
            [4, channels, 12, 12],
            Distribution::Normal(0.0, 0.5),
            &device,
        );
        let (z, log_det) = step.forward(input.clone());
        assert!(!has_nan_log_det(log_det));
        let reconstructed = step.inverse(z);
        check_closeness(&reconstructed, &input);
        assert!(
            reconstructed.all_close(input, Some(1e-4), Some(1e-4)),
            "GlowStep inverse(forward(x)) should recover x (use cargo test -- --nocapture and burn::tensor::check_closeness if debugging)"
        );
    }

    #[rstest]
    fn glow_block_init_actnorm_does_not_compound_variance(device: NdArrayDevice) {
        // K=8 with std≈0.29 input (simulating dequantized images, Uniform[-0.5, 0.5]).
        // Broken code: all steps init on same input → weight≈1.25/step → output RMS ≈ 0.29 * 3.5^8 ≈ 25k.
        // Fixed code:  each step sees N(0,1) after the first → output RMS ≈ 1.
        let mut block = GlowBlockConfig::new(12, 8)
            .with_hidden_features(32)
            .init(&device);

        let input = Tensor::<B, 4>::random([4, 3, 8, 8], Distribution::Uniform(-0.5, 0.5), &device);
        block.init_actnorm(input.clone());
        let (z, log_det) = block.forward(input);

        let z_vals = z.into_data().to_vec::<f32>().unwrap();
        assert!(
            z_vals.iter().all(|v| v.is_finite()),
            "forward output contains Inf/NaN after init_actnorm"
        );

        let rms: f32 = (z_vals.iter().map(|v| v * v).sum::<f32>() / z_vals.len() as f32).sqrt();
        assert!(
            rms < 100.0,
            "output RMS={rms:.1} — variance is compounding; init_actnorm is not sequential"
        );

        let ld_vals = log_det.into_data().to_vec::<f32>().unwrap();
        assert!(
            ld_vals.iter().all(|v| v.is_finite()),
            "log_det contains Inf/NaN after init_actnorm"
        );
    }

    #[rstest]
    fn glow_step_log_det_not_all_zero(device: NdArrayDevice) {
        let step = GlowStepConfig::new(12).init(&device);
        let input = Tensor::<B, 4>::random([2, 12, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (_, log_det) = step.forward(input);
        let sum: f32 = log_det
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.abs())
            .sum();
        assert!(sum > 1e-6, "expected non-trivial log-det sum, got {sum}");
    }
}
