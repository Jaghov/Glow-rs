// ─────────────────────────────────────────────────────────────
// MLP
//
// Used for the factored (non-image) path in AffineCoupling.
// Input/output are 2-D tensors [Batch, Features].
// ─────────────────────────────────────────────────────────────

use crate::models::blocks::{Activation, Module};

use burn::{
    config::Config,
    module::Ignored,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::Device,
    Tensor,
};

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub in_dim: usize,
    pub out_dim: usize,
    #[config(default = "128")]
    pub hid_dim: usize,
    #[config(default = "Activation::Sigmoid")]
    pub activation: Activation,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MLP<B> {
        MLP {
            fc1: LinearConfig::new(self.in_dim, self.hid_dim).init(device),
            fc2: LinearConfig::new(self.hid_dim, self.hid_dim).init(device),
            fc_out: LinearConfig::new(self.hid_dim, self.out_dim).init(device),
            // Activation has no trainable params; wrap in Ignored<> so the
            // Module derive does not attempt to track or move it.
            activation: Ignored(self.activation.clone()),
        }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc_out: Linear<B>,
    activation: Ignored<Activation>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.activation.0.forward(self.fc1.forward(x));
        let out = self.activation.0.forward(self.fc2.forward(out));
        self.fc_out.forward(out)
    }
}

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

    // ── MLP: output shape across all activations ────────────────────────────────

    #[rstest]
    #[case::sigmoid(Activation::Sigmoid, 8, 16, 4)]
    #[case::tanh(Activation::Tanh, 16, 32, 2)]
    #[case::relu(Activation::Relu, 4, 8, 1)]
    #[case::leaky_relu(Activation::LeakyRelu, 12, 24, 3)]
    #[case::cos(Activation::Cos, 6, 12, 5)]
    fn mlp_output_shape(
        device: NdArrayDevice,
        #[case] act: Activation,
        #[case] in_dim: usize,
        #[case] out_dim: usize,
        #[case] batch: usize,
    ) {
        let mlp = MLPConfig::new(in_dim, out_dim)
            .with_activation(act)
            .init(&device);
        let x = Tensor::<B, 2>::random([batch, in_dim], Distribution::Normal(0.0, 1.0), &device);
        assert_eq!(mlp.forward(x).dims(), [batch, out_dim]);
    }

    // ── MLP: custom hidden dim ──────────────────────────────────────────────────

    #[rstest]
    #[case(32)]
    #[case(64)]
    #[case(256)]
    fn mlp_custom_hidden_dim(device: NdArrayDevice, #[case] hid_dim: usize) {
        let mlp = MLPConfig::new(8, 4)
            .with_hid_dim(hid_dim)
            .with_activation(Activation::Relu)
            .init(&device);
        let x = Tensor::<B, 2>::random([2, 8], Distribution::Normal(0.0, 1.0), &device);
        assert_eq!(mlp.forward(x).dims(), [2, 4]);
    }

    // ── MLP: numerical sanity ───────────────────────────────────────────────────

    #[rstest]
    fn mlp_output_is_finite(device: NdArrayDevice) {
        let mlp = MLPConfig::new(8, 16).init(&device);
        let x = Tensor::<B, 2>::random([4, 8], Distribution::Normal(0.0, 1.0), &device);
        let values = mlp.forward(x).into_data().to_vec::<f32>().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "MLP output contains NaN or Inf"
        );
    }
}
