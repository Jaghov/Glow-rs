use std::sync::{atomic::AtomicBool, Arc};

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    prelude::Backend,
    tensor::Shape,
    Tensor,
};

#[derive(Config, Debug)]
pub struct ActnormConfig {
    num_channels: usize,
}

impl ActnormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActNorm<B> {
        ActNorm {
            weight: Param::from_tensor(Tensor::zeros(Shape::new([self.num_channels]), device)),
            bias: Param::from_tensor(Tensor::zeros(Shape::new([self.num_channels]), device)),
            initialized: Ignored(Arc::new(AtomicBool::new(false))),
        }
    }
}

#[derive(Module, Debug)]
pub struct ActNorm<B: Backend> {
    /// vector of length num_channels
    weight: Param<Tensor<B, 1>>,
    /// vector of length num_channels
    bias: Param<Tensor<B, 1>>,
    /// Initiliser
    initialized: Ignored<Arc<AtomicBool>>,
}

impl<B: Backend> ActNorm<B> {
    #[cold]
    pub fn init(&mut self, example: Tensor<B, 4>) {
        let device = example.device();
        let [b, c, ..] = example.dims();
        let mean = example
            .clone()
            .mean_dims(&[0, 2, 3])
            .squeeze_dims::<1>(&[0, 2, 3]);

        let var = example
            .reshape([b as isize, c as isize, -1])
            .var(2)
            .mean_dim(0)
            .squeeze_dims::<1>(&[0, 2]);

        let stdev = (var + 1e-6).sqrt();
        // Break autodiff graph (stats must become plain parameter initial values).
        let weight_init = Tensor::from_data(stdev.log().neg().into_data(), &device);
        let bias_init = Tensor::from_data(mean.neg().into_data(), &device);
        (self.weight, self.bias) = (
            Param::from_tensor(weight_init),
            Param::from_tensor(bias_init),
        );

        self.initialized
            .0
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let [b, .., h, w] = x.dims();
        let log_det_jacobian = (h * w) as f32 * self.weight.val();
        (
            (x + self.bias.val().unsqueeze_dims(&[0, 2, 3]))
                .mul(self.weight.val().exp().unsqueeze_dims(&[0, 2, 3])),
            Tensor::repeat_dim(log_det_jacobian, 0, b),
        )
    }
    pub fn inverse(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        x.mul(self.weight.val().neg().exp().unsqueeze_dims(&[0, 2, 3]))
            - self.bias.val().unsqueeze_dims(&[0, 2, 3])
    }

    /// Diagnostic: `(min, max)` of the per-channel log-scale `weight`.
    /// `inverse` applies `exp(-weight)`; large positive `weight` = aggressive shrink,
    /// large negative `weight` = aggressive amplification on the inverse path.
    pub fn weight_extrema(&self) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let w = self.weight.val();
        (w.clone().min(), w.max())
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

    const ATOL: f32 = 1e-5;

    fn tensor_to_vec(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    fn max_abs_diff(a: Tensor<B, 4>, b: Tensor<B, 4>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────

    #[fixture]
    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    // ── Test 1: default config is identity ──────────────────────────────────────
    // weight=zeros → exp(0)=1, bias=zeros → (x + 0) * 1 = x

    #[rstest]
    #[case(4, 2, 8, 8)]
    #[case(6, 1, 16, 16)]
    fn actnorm_default_init_is_identity(
        device: NdArrayDevice,
        #[case] channels: usize,
        #[case] batch: usize,
        #[case] h: usize,
        #[case] w: usize,
    ) {
        let actnorm = ActnormConfig::new(channels).init(&device);

        let x = Tensor::<B, 4>::random(
            [batch, channels, h, w],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (z, _log_det) = actnorm.forward(x.clone());

        check_closeness(&z, &x);
        z.to_data().assert_eq(&x.to_data(), true);
    }

    // log-det should also be zero for the identity (sum of zeros * h*w = 0)
    #[rstest]
    fn actnorm_default_logdet_is_zero(device: NdArrayDevice) {
        let actnorm = ActnormConfig::new(4).init(&device);
        let x = Tensor::<B, 4>::random([2, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let (_, log_det) = actnorm.forward(x);

        let zeros = log_det.zeros_like();
        check_closeness(&log_det, &zeros);
        let log_det_vals = log_det.abs().into_data();
        log_det_vals.assert_within_range_inclusive(0f32..=1e-5f32);
    }

    // ── Test 2: data-dependent init changes weight and bias ─────────────────────

    #[rstest]
    fn actnorm_weights_change_after_data_init(device: NdArrayDevice) {
        let mut actnorm = ActnormConfig::new(4).init(&device);

        let weight_before = actnorm.weight.val();
        let bias_before = actnorm.bias.val();

        // Use a distribution with non-zero mean and non-unit variance so
        // both weight and bias are guaranteed to shift
        let example =
            Tensor::<B, 4>::random([8, 4, 16, 16], Distribution::Normal(3.0, 2.0), &device);
        actnorm.init(example);

        let weight_after = actnorm.weight.val();
        let bias_after = actnorm.bias.val();

        check_closeness(&weight_before, &weight_after);
        check_closeness(&bias_before, &bias_after);
        let w_diff = (weight_before - weight_after).abs().max().into_scalar();
        let b_diff = (bias_before - bias_after).abs().max().into_scalar();
        assert!(
            w_diff > 1e-4_f32,
            "weight should change after data-dependent init, max |Δ| = {w_diff}"
        );
        assert!(
            b_diff > 1e-4_f32,
            "bias should change after data-dependent init, max |Δ| = {b_diff}"
        );
    }

    // After init on zero-mean unit-variance data, actnorm should again be ≈ identity
    #[rstest]
    fn actnorm_data_init_on_unit_normal_is_near_identity(device: NdArrayDevice) {
        let mut actnorm = ActnormConfig::new(4).init(&device);

        // Large batch so sample statistics ≈ true statistics (mean=0, std=1)
        let example =
            Tensor::<B, 4>::random([512, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        actnorm.init(example.clone());

        // weight = -log(std) ≈ -log(1) = 0  →  scale ≈ 1
        // bias   = -mean     ≈ 0
        let weight_vals = tensor_to_vec(actnorm.weight.val());
        let bias_vals = tensor_to_vec(actnorm.bias.val());

        for w in &weight_vals {
            assert!(
                w.abs() < 0.1_f32,
                "weight should be near 0 for unit-normal data, got {w}"
            );
        }
        for b in &bias_vals {
            assert!(
                b.abs() < 0.1_f32,
                "bias should be near 0 for zero-mean data, got {b}"
            );
        }
    }

    // ── Test 3: invertibility ───────────────────────────────────────────────────

    #[rstest]
    // (channels, batch, h, w)
    #[case(4, 2, 8, 8)]
    #[case(6, 4, 16, 16)]
    #[case(12, 1, 4, 4)]
    fn actnorm_is_invertible_after_data_init(
        device: NdArrayDevice,
        #[case] channels: usize,
        #[case] batch: usize,
        #[case] h: usize,
        #[case] w: usize,
    ) {
        let mut actnorm = ActnormConfig::new(channels).init(&device);

        let example = Tensor::<B, 4>::random(
            [batch, channels, h, w],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        actnorm.init(example.clone());

        let (z, _) = actnorm.forward(example.clone());
        let reconstructed = actnorm.inverse(z);

        check_closeness(&reconstructed, &example);
        assert!(
            max_abs_diff(example, reconstructed) < ATOL,
            "ActNorm inverse(forward(x)) should recover x within atol={ATOL}"
        );
    }

    // Invertibility should also hold before data_init (identity ∘ identity = identity)
    #[rstest]
    fn actnorm_is_invertible_before_data_init(device: NdArrayDevice) {
        let actnorm = ActnormConfig::new(4).init(&device);
        let x = Tensor::<B, 4>::random([2, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);

        let (z, _) = actnorm.forward(x.clone());
        let reconstructed = actnorm.inverse(z);

        check_closeness(&reconstructed, &x);
        assert!(
            max_abs_diff(x, reconstructed) < ATOL,
            "ActNorm should be invertible even before data-dependent init"
        );
    }
}
