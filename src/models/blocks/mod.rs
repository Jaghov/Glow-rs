pub mod conv;
pub mod linear;

use burn::{
    config::Config,
    tensor::{backend::Backend, Tensor},
};

// ─────────────────────────────────────────────────────────────
// Activation
//
// A plain Rust enum (no trainable params). Stored via
// Ignored<Activation> inside Module structs so that Burn's
// Module derive does not try to track it as a parameter.
// ─────────────────────────────────────────────────────────────

#[derive(Config, Debug, PartialEq)]
pub enum Activation {
    Sigmoid,
    Cos,
    Tanh,
    Relu,
    LeakyRelu,
}

impl Activation {
    /// Works for any tensor rank D.
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Sigmoid => burn::tensor::activation::sigmoid(x),
            Activation::Cos => x.cos(),
            Activation::Tanh => burn::tensor::activation::tanh(x),
            Activation::Relu => burn::tensor::activation::relu(x),
            Activation::LeakyRelu => burn::tensor::activation::leaky_relu(x, 0.2),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// WeightNormConv2d
//
// Burn has no built-in weight_norm wrapper, so we implement it
// manually by storing the direction (v) and magnitude (g) as
// separate learned parameters and computing:
//   w = g * v / ‖v‖₂   (per output channel)
// then calling the functional conv2d.
// ─────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    use burn::tensor::{check_closeness, Distribution, Tensor};
    use rstest::*;

    type B = NdArray;

    // ── Shared fixture ──────────────────────────────────────────────────────────
    // #[fixture] injects this automatically into any test that takes NdArrayDevice
    // as an argument, by matching on the parameter name.

    #[fixture]
    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    // ── Activation: shape ───────────────────────────────────────────────────────

    #[rstest]
    #[case::sigmoid(Activation::Sigmoid)]
    #[case::cos(Activation::Cos)]
    #[case::tanh(Activation::Tanh)]
    #[case::relu(Activation::Relu)]
    #[case::leaky_relu(Activation::LeakyRelu)]
    fn activation_preserves_shape(device: NdArrayDevice, #[case] act: Activation) {
        let x = Tensor::<B, 2>::random([4, 16], Distribution::Normal(0.0, 1.0), &device);
        let y = act.forward(x);
        assert_eq!(y.dims(), [4, 16]);
    }

    // ── Activation: known scalar values ────────────────────────────────────────

    #[rstest]
    #[case::relu_clips_neg(Activation::Relu,      -2.0_f32,  0.0_f32,  1e-6)]
    #[case::relu_passes_pos(Activation::Relu, 3.0_f32, 3.0_f32, 1e-6)]
    #[case::sigmoid_at_zero(Activation::Sigmoid, 0.0_f32, 0.5_f32, 1e-6)]
    #[case::tanh_at_zero(Activation::Tanh, 0.0_f32, 0.0_f32, 1e-6)]
    #[case::cos_at_zero(Activation::Cos, 0.0_f32, 1.0_f32, 1e-6)]
    #[case::leaky_neg_slope(Activation::LeakyRelu, -1.0_f32, -0.2_f32, 1e-6)]
    fn activation_known_values(
        device: NdArrayDevice,
        #[case] act: Activation,
        #[case] input: f32,
        #[case] expected: f32,
        #[case] tol: f32,
    ) {
        let x = Tensor::<B, 1>::from_floats([input], &device);
        let y = act.forward(x);
        let expected_t = Tensor::<B, 1>::from_floats([expected], &device);
        check_closeness(&y, &expected_t);
        let val = y.into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            (val - expected).abs() < tol,
            "activation: got {val:.6}, expected {expected:.6}"
        );
    }
}
