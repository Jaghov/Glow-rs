use std::f32::consts::TAU;

use burn::{config::Config, module::Module, prelude::Backend, Tensor};

mod actnorm;
mod blocks;
mod coupling;
mod invconv;
mod preprocess;

pub use actnorm::{ActNorm, ActnormConfig};
pub use blocks::{GlowBlock, GlowBlockConfig, GlowStep, GlowStepConfig, GlowStepDiag, SplitBlock};
pub use coupling::{Coupling, CouplingConfig, CouplingType};
pub use invconv::{InvConv1x1, InvConv1x1Config, TriangularInverse};
pub use preprocess::{Dequantize, DequantizeConfig};

// ── squeeze / unsqueeze ───────────────────────────────────────────────────────

/// `[B, C, H, W]` → `[B, C*f², H/f, W/f]` (space-to-depth).
#[inline]
pub(crate) fn squeeze2d<B: Backend>(tensor: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = tensor.dims();
    assert!(height % factor == 0 && width % factor == 0);
    let x = tensor.reshape([
        batch,
        channels,
        height / factor,
        factor,
        width / factor,
        factor,
    ]);
    // [B,C,H/f,f,W/f,f] → [B,C,f,f,H/f,W/f]
    let x = x.permute([0, 1, 3, 5, 2, 4]);
    x.reshape([
        batch,
        channels * factor * factor,
        height / factor,
        width / factor,
    ])
}

/// `[B, C*f², H/f, W/f]` → `[B, C, H, W]` (depth-to-space; inverse of `squeeze2d`).
#[inline]
pub(crate) fn unsqueeze2d<B: Backend>(tensor: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = tensor.dims();
    let f2 = factor * factor;
    debug_assert_eq!(
        channels % f2,
        0,
        "unsqueeze2d: channels not divisible by factor²"
    );
    let c_out = channels / f2;
    // [B, C*f², H/f, W/f] → [B, C, f, f, H/f, W/f]
    let x = tensor.reshape([batch, c_out, factor, factor, height, width]);
    // inverse of permute [0,1,3,5,2,4] is [0,1,4,2,5,3]
    let x = x.permute([0, 1, 4, 2, 5, 3]);
    // [B, C, H/f, f, W/f, f] → [B, C, H, W]
    x.reshape([batch, c_out, height * factor, width * factor])
}

// ── Likelihood helpers ────────────────────────────────────────────────────────

/// Isotropic Gaussian log p(z) summed over all latent dimensions; returns `[B]`.
pub fn log_p_z<B: Backend>(zs: &[Tensor<B, 4>]) -> Tensor<B, 1> {
    let log_two_pi = TAU.ln();
    let mut sum: Option<Tensor<B, 1>> = None;
    for z in zs {
        let [_, c, h, w] = z.dims();
        let d = (c * h * w) as f32;
        // -0.5 * (||z||² + D·log2π)
        let z_sq_sum = (z.clone() * z.clone()).sum_dims_squeeze::<1, _>(&[1, 2, 3]);
        let contrib = z_sq_sum.add_scalar(d * log_two_pi).mul_scalar(-0.5_f32);
        sum = Some(match sum {
            None => contrib,
            Some(acc) => acc + contrib,
        });
    }
    sum.expect("zs must not be empty")
}

/// `log p(x) = Σ log|det J| + log p(z)`, shape `[B]`.
pub fn log_likelihood<B: Backend>(model: &Glow<B>, x: Tensor<B, 4>) -> Tensor<B, 1> {
    let (zs, log_det) = model.forward(x);
    log_det + log_p_z(&zs)
}

/// Forward pass shared by the NLL loss and any side-tasks that need access to the
/// continuous latent (e.g. FD regularization). Returns `(x_cont, zs, log_p)`:
///
/// * `x_cont` — the dequantized continuous image fed into Glow (`[B, C, H, W]`).
/// * `zs`     — per-scale latents from `Glow::forward`.
/// * `log_p`  — `log p(image)` per sample (`[B]`), already including the dequant penalty.
///
/// Callers wanting just `log_p` can use [`log_prob_pixels`].
#[doc(hidden)]
pub fn forward_for_training<B: Backend>(
    model: &Glow<B>,
    dequantize: &Dequantize<B>,
    pixels_f32: Tensor<B, 4>,
    training: bool,
) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>, Tensor<B, 1>) {
    let (x_cont, pen) = if training {
        dequantize.forward_train(pixels_f32)
    } else {
        let y = dequantize.forward(pixels_f32);
        let pen = dequantize.discretization_penalty(&y.dims(), &y.device());
        (y, pen)
    };
    let (zs, log_det) = model.forward(x_cont.clone());
    let log_p = log_det + log_p_z(&zs) + pen;
    (x_cont, zs, log_p)
}

/// Full `log p(image)` per sample: Glow (continuous latent) + uniform dequantization term.
///
/// When `training` is true, uses [`Dequantize::forward_train`] (noise + penalty). Otherwise
/// [`Dequantize::forward`] with the same penalty (eval / validation).
pub fn log_prob_pixels<B: Backend>(
    model: &Glow<B>,
    dequantize: &Dequantize<B>,
    pixels_f32: Tensor<B, 4>,
    training: bool,
) -> Tensor<B, 1> {
    let (_x_cont, _zs, log_p) = forward_for_training(model, dequantize, pixels_f32, training);
    log_p
}

// ── Glow (top-level model) ───────────────────────────────────────────────────

/// Hyperparameters for the full multi-scale Glow model.
#[derive(Config, Debug)]
pub struct GlowConfig {
    /// Input channels (3 for RGB CelebA).
    pub in_channels: usize,
    /// Number of levels L; each level halves spatial dimensions via squeeze.
    #[config(default = "3")]
    pub num_levels: usize,
    /// GlowSteps per level (K in the paper).
    #[config(default = "4")]
    pub num_steps: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
    /// Coupling parameterisation. See [`CouplingType`].
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
}

impl GlowConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Glow<B> {
        assert!(self.num_levels >= 1);
        let blocks = (0..self.num_levels)
            .map(|l| {
                // channels entering block l (before squeeze): 2^l * in_channels
                // channels after squeeze (factor=2):          4 * 2^l * in_channels
                let c_after_squeeze = 4 * (1_usize << l) * self.in_channels;
                GlowBlockConfig::new(c_after_squeeze, self.num_steps)
                    .with_hidden_features(self.hidden_features)
                    .with_coupling_type(self.coupling_type)
                    .init(device)
            })
            .collect();
        Glow { blocks }
    }
}

/// Full multi-scale Glow model: L `GlowBlock`s with `SplitBlock`s between non-final levels.
#[derive(Module, Debug)]
pub struct Glow<B: Backend> {
    blocks: Vec<GlowBlock<B>>,
}

impl<B: Backend> Glow<B> {
    /// Data-dependent ActNorm initialisation; call once before the first training step.
    pub fn init_actnorm(&mut self, example: Tensor<B, 4>) {
        let num_blocks = self.blocks.len();
        let mut h = example;
        for (l, block) in self.blocks.iter_mut().enumerate() {
            block.init_actnorm(h.clone());
            let (h2, _) = block.forward(h);
            h = h2;
            if l < num_blocks - 1 {
                let (_, h2) = SplitBlock::forward(h);
                h = h2;
            }
        }
    }

    /// Returns `(zs, total_log_det)`.
    ///
    /// `zs[l]` is the latent extracted at level `l`; `zs[L-1]` is the full last-level output.
    pub fn forward(&self, x: Tensor<B, 4>) -> (Vec<Tensor<B, 4>>, Tensor<B, 1>) {
        let mut h = x;
        let mut total_log_det: Option<Tensor<B, 1>> = None;
        let mut zs: Vec<Tensor<B, 4>> = Vec::with_capacity(self.blocks.len());

        for (l, block) in self.blocks.iter().enumerate() {
            let (h2, ld) = block.forward(h);
            h = h2;
            total_log_det = Some(match total_log_det {
                None => ld,
                Some(acc) => acc + ld,
            });
            if l < self.blocks.len() - 1 {
                let (z, h2) = SplitBlock::forward(h);
                zs.push(z);
                h = h2;
            }
        }
        zs.push(h);
        (
            zs,
            total_log_det.expect("Glow must have at least one block"),
        )
    }
}

/// Per-step diagnostic row: `(level, step_index, GlowStepDiag)`.
pub struct GlowInvertDiagRow<B: Backend> {
    pub level: usize,
    pub step_idx: usize,
    pub diag: GlowStepDiag<B>,
}

impl<B: Backend + TriangularInverse> Glow<B> {
    /// Reconstruct `x` from latent tensors; `zs` must have length `num_levels`.
    pub fn inverse(&self, zs: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let l_last = self.blocks.len() - 1;
        let mut h = self.blocks[l_last].inverse(zs[l_last].clone());
        for l in (0..l_last).rev() {
            // zs[l] is the split-off first half; h is the second-half path
            let full = SplitBlock::inverse(zs[l].clone(), h);
            h = self.blocks[l].inverse(full);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{Distribution, Tensor};
    use rstest::*;

    type B = NdArray;

    #[fixture]
    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    // ── squeeze / unsqueeze ───────────────────────────────────────────────────

    #[rstest]
    fn squeeze_then_unsqueeze_is_identity(device: NdArrayDevice) {
        let input = Tensor::<B, 4>::random([2, 3, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let squeezed = squeeze2d(input.clone(), 2);
        assert_eq!(squeezed.dims(), [2, 12, 4, 4]);
        let recovered = unsqueeze2d(squeezed, 2);
        assert_eq!(recovered.dims(), [2, 3, 8, 8]);
        assert!(
            recovered.all_close(input, Some(1e-6), Some(1e-6)),
            "unsqueeze2d(squeeze2d(x)) should be identity"
        );
    }

    #[rstest]
    fn unsqueeze_then_squeeze_is_identity(device: NdArrayDevice) {
        let input = Tensor::<B, 4>::random([2, 12, 4, 4], Distribution::Normal(0.0, 1.0), &device);
        let unsqueezed = unsqueeze2d(input.clone(), 2);
        assert_eq!(unsqueezed.dims(), [2, 3, 8, 8]);
        let recovered = squeeze2d(unsqueezed, 2);
        assert_eq!(recovered.dims(), [2, 12, 4, 4]);
        assert!(
            recovered.all_close(input, Some(1e-6), Some(1e-6)),
            "squeeze2d(unsqueeze2d(x)) should be identity"
        );
    }

    // ── GlowBlock ─────────────────────────────────────────────────────────────

    #[rstest]
    #[case::three_ch_two_steps(3, 2)]
    #[case::four_ch_three_steps(4, 3)]
    fn glow_block_is_invertible(
        device: NdArrayDevice,
        #[case] in_channels: usize,
        #[case] num_steps: usize,
    ) {
        let c_after_squeeze = 4 * in_channels;
        let block = GlowBlockConfig::new(c_after_squeeze, num_steps)
            .with_hidden_features(32)
            .init(&device);

        let input = Tensor::<B, 4>::random(
            [2, in_channels, 8, 8],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (z, log_det) = block.forward(input.clone());

        assert_eq!(z.dims(), [2, c_after_squeeze, 4, 4]);
        assert!(!log_det
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .any(|v| v.is_nan()));

        let reconstructed = block.inverse(z);
        assert_eq!(reconstructed.dims(), [2, in_channels, 8, 8]);
        assert!(
            reconstructed.all_close(input, Some(1e-4), Some(1e-4)),
            "GlowBlock: inverse(forward(x)) should recover x"
        );
    }

    // ── Glow ─────────────────────────────────────────────────────────────────

    #[rstest]
    fn glow_forward_shapes_and_nll_finite(device: NdArrayDevice) {
        // 2 levels, 2 steps, 3 in_channels, 8×8 spatial
        let model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(32)
            .init::<B>(&device);

        let input = Tensor::<B, 4>::random([2, 3, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (zs, log_det) = model.forward(input.clone());

        // level 0 split: z_0=[2,6,4,4];  level 1 (last): z_1=[2,24,2,2]
        assert_eq!(zs.len(), 2);
        assert_eq!(zs[0].dims(), [2, 6, 4, 4]);
        assert_eq!(zs[1].dims(), [2, 24, 2, 2]);
        assert_eq!(log_det.dims(), [2]);

        let nll = log_likelihood(&model, input);
        assert_eq!(nll.dims(), [2]);
        assert!(
            !nll.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .any(|v| v.is_nan()),
            "log_likelihood must not contain NaN"
        );
    }

    #[rstest]
    fn glow_is_invertible(device: NdArrayDevice) {
        let model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(32)
            .init::<B>(&device);

        let input = Tensor::<B, 4>::random([2, 3, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (zs, _) = model.forward(input.clone());
        let reconstructed = model.inverse(zs);
        assert!(
            reconstructed.all_close(input, Some(1e-4), Some(1e-4)),
            "Glow: inverse(forward(x)) should recover x"
        );
    }

    /// Additive coupling end-to-end: `Glow::inverse(forward(x)) == x` to better tolerance
    /// than affine because there's no `1/s` amplification in any of the K · L couplings.
    #[rstest]
    fn glow_additive_round_trip(device: NdArrayDevice) {
        let model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(32)
            .with_coupling_type(CouplingType::Additive)
            .init::<B>(&device);

        let input = Tensor::<B, 4>::random([2, 3, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        // NOTE: Glow log_det is the sum of ActNorm + InvConv + Coupling. Only the
        // coupling contribution is identically zero in additive mode; the ActNorm
        // and InvConv terms remain non-zero. We assert log_det finiteness here,
        // not equality to zero. The "log_det == 0" property is asserted at the
        // coupling-layer level in `additive_coupling_log_det_is_zero`.
        let (zs, log_det) = model.forward(input.clone());
        let max_abs_logdet: f32 = log_det.abs().max().into_scalar();
        assert!(
            max_abs_logdet.is_finite(),
            "additive Glow log_det must be finite; got {max_abs_logdet}"
        );

        let reconstructed = model.inverse(zs);
        assert!(
            reconstructed.all_close(input, Some(1e-5), Some(1e-5)),
            "additive Glow: inverse(forward(x)) should round-trip tighter than affine"
        );
    }

    /// Wide-σ inputs (5σ above the training distribution): additive + SN should still
    /// round-trip cleanly. Catches catastrophic SN/parametrisation regressions.
    #[rstest]
    fn glow_additive_invert_under_wide_inputs(device: NdArrayDevice) {
        let model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(32)
            .with_coupling_type(CouplingType::Additive)
            .init::<B>(&device);
        let input = Tensor::<B, 4>::random([2, 3, 8, 8], Distribution::Normal(0.0, 5.0), &device);
        let (zs, _) = model.forward(input.clone());
        let recon = model.inverse(zs);
        let max_abs: f32 = (recon - input).abs().max().into_scalar();
        assert!(
            max_abs.is_finite() && max_abs < 1e-3,
            "additive+SN must round-trip wide-σ inputs cleanly; got max_abs={max_abs}"
        );
    }

    /// One Adam step on CPU autodiff: ensures dequantize + Glow + backward + optim compose.
    #[rstest]
    fn glow_training_step_ndarray_smoke(device: NdArrayDevice) {
        use burn::backend::Autodiff;
        use burn::optim::{AdamConfig, GradientsParams, Optimizer};
        use burn::prelude::ElementConversion;

        type AD = Autodiff<NdArray>;

        let deq = DequantizeConfig::new(8).init::<AD>(&device);
        let mut model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(16)
            .init::<AD>(&device);

        let x = Tensor::<AD, 4>::random([2, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        model.init_actnorm(x.clone());

        let logp = log_prob_pixels(&model, &deq, x.clone(), true);
        let loss = logp.clone().mean().neg();
        let grads = loss.backward();
        let mut optim = AdamConfig::new().init::<AD, Glow<AD>>();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(1e-3, model, grads);

        let logp_after = log_prob_pixels(&model, &deq, x, true);
        let m: f32 = logp_after.mean().into_scalar().elem();
        assert!(m.is_finite(), "post-step log-prob mean must be finite");
    }

    /// Gradient accumulation invariant: K micro-batches with `loss / K` summed via
    /// [`GradientsAccumulator`] must produce the same Adam update as a single forward
    /// over the concatenated B·K batch. Compared via post-step forward loss.
    #[rstest]
    fn grad_accumulation_matches_single_large_batch(device: NdArrayDevice) {
        use burn::backend::Autodiff;
        use burn::optim::{AdamConfig, GradientsAccumulator, GradientsParams, Optimizer};
        use burn::prelude::ElementConversion;

        type AD = Autodiff<NdArray>;
        const K: usize = 3;
        const B: usize = 2;

        let deq = DequantizeConfig::new(8).init::<AD>(&device);

        let mut model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(8)
            .init::<AD>(&device);

        // Deterministic init so both paths start from byte-identical weights.
        let init_x =
            Tensor::<AD, 4>::random([B, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        model.init_actnorm(deq.forward(init_x));

        let micros: Vec<Tensor<AD, 4>> = (0..K)
            .map(|_| {
                Tensor::<AD, 4>::random([B, 3, 8, 8], Distribution::Uniform(0., 255.), &device)
            })
            .collect();

        let model_a = model.clone();
        let model_b = model;

        // Path A: K micro-batches → loss/K → accumulate → one Adam step.
        // `training=false` keeps dequantize deterministic so both paths see the
        // same continuous inputs.
        let mut optim_a = AdamConfig::new().init::<AD, Glow<AD>>();
        let mut accum = GradientsAccumulator::<Glow<AD>>::new();
        for x in &micros {
            let logp = log_prob_pixels(&model_a, &deq, x.clone(), false);
            let loss = logp.mean().neg().mul_scalar(1.0 / K as f32);
            let grads = GradientsParams::from_grads(loss.backward(), &model_a);
            accum.accumulate(&model_a, grads);
        }
        let model_a = optim_a.step(1e-3, model_a, accum.grads());

        // Path B: one forward over the concatenated B·K batch.
        let mut optim_b = AdamConfig::new().init::<AD, Glow<AD>>();
        let big = Tensor::cat(micros.clone(), 0);
        let logp = log_prob_pixels(&model_b, &deq, big, false);
        let loss = logp.mean().neg();
        let grads = GradientsParams::from_grads(loss.backward(), &model_b);
        let model_b = optim_b.step(1e-3, model_b, grads);

        // Compare via a fresh forward on identical inputs.
        let probe = Tensor::cat(micros, 0);
        let nats_a: f32 = log_prob_pixels(&model_a, &deq, probe.clone(), false)
            .mean()
            .neg()
            .into_scalar()
            .elem();
        let nats_b: f32 = log_prob_pixels(&model_b, &deq, probe, false)
            .mean()
            .neg()
            .into_scalar()
            .elem();

        assert!(nats_a.is_finite() && nats_b.is_finite());
        let diff = (nats_a - nats_b).abs();
        let scale = nats_a.abs().max(nats_b.abs()).max(1.0);
        assert!(
            diff / scale < 1e-4,
            "accumulated K={K} update should match B·K batch update; nats_a={nats_a}, nats_b={nats_b} (rel diff {})",
            diff / scale
        );
    }

    /// AdamW state must round-trip through `Recorder::record` / `Recorder::load`. After
    /// reloading, the next optimiser step on the loaded record must produce the same
    /// parameters as continuing on the original optimiser — i.e. Adam moments survive
    /// the save/load cycle, not just the parameter list. Mirrors Burn's own
    /// `test_adamw_optimizer_save_load_state` plus a numerical equivalence check.
    #[rstest]
    fn adamw_state_save_load_roundtrip(device: NdArrayDevice) {
        use burn::backend::Autodiff;
        use burn::optim::adaptor::OptimizerAdaptor;
        use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
        use burn::prelude::ElementConversion;
        use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

        type AD = Autodiff<NdArray>;
        type Optim = OptimizerAdaptor<AdamW, Glow<AD>, AD>;
        type OptimRecord = <Optim as Optimizer<Glow<AD>, AD>>::Record;

        let deq = DequantizeConfig::new(8).init::<AD>(&device);
        let mut model = GlowConfig::new(3)
            .with_num_levels(2)
            .with_num_steps(2)
            .with_hidden_features(8)
            .init::<AD>(&device);
        let init_x =
            Tensor::<AD, 4>::random([2, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        model.init_actnorm(deq.forward(init_x));

        let mut optim: Optim = AdamWConfig::new().init::<AD, Glow<AD>>();

        let step = |optim: &mut Optim, model: Glow<AD>, x: &Tensor<AD, 4>| -> Glow<AD> {
            let logp = log_prob_pixels(&model, &deq, x.clone(), false);
            let loss = logp.mean().neg();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            optim.step(1e-3, model, grads)
        };

        // First step builds non-trivial Adam moments.
        let x1 = Tensor::<AD, 4>::random([2, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        model = step(&mut optim, model, &x1);

        // Save optim state to disk; load into a fresh optim; both must produce the
        // same model after a second step on identical inputs.
        let tmp =
            std::env::temp_dir().join(format!("glow_rs_optim_roundtrip_{}", std::process::id()));
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(optim.to_record(), tmp.clone())
            .expect("save optim record");

        let mut optim_loaded: Optim = AdamWConfig::new().init::<AD, Glow<AD>>();
        let record: OptimRecord = recorder
            .load(tmp.clone(), &device)
            .expect("load optim record");
        optim_loaded = optim_loaded.load_record(record);

        let _ = std::fs::remove_file(tmp.with_extension("bin"));

        let x2 = Tensor::<AD, 4>::random([2, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        let model_a = step(&mut optim, model.clone(), &x2);
        let model_b = step(&mut optim_loaded, model, &x2);

        let probe = Tensor::<AD, 4>::random([2, 3, 8, 8], Distribution::Uniform(0., 255.), &device);
        let nats_a: f32 = log_prob_pixels(&model_a, &deq, probe.clone(), false)
            .mean()
            .neg()
            .into_scalar()
            .elem();
        let nats_b: f32 = log_prob_pixels(&model_b, &deq, probe, false)
            .mean()
            .neg()
            .into_scalar()
            .elem();
        assert!(nats_a.is_finite() && nats_b.is_finite());
        let diff = (nats_a - nats_b).abs();
        let scale = nats_a.abs().max(nats_b.abs()).max(1.0);
        assert!(
            diff / scale < 1e-5,
            "optim state round-trip should be lossless: nats_a={nats_a}, nats_b={nats_b}"
        );
    }
}
