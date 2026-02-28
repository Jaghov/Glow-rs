mod conv;
mod linear;

use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::{Linear, LinearConfig, Relu},
    tensor::{backend::Backend, module::conv2d, ops::ConvOptions, Device, Distribution, Tensor},
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

#[derive(Config, Debug)]
pub struct WeightNormConv2dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    #[config(default = "[0, 0]")]
    pub padding: [usize; 2],
    #[config(default = "true")]
    pub bias: bool,
}

impl WeightNormConv2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WeightNormConv2d<B> {
        let [kh, kw] = self.kernel_size;

        // v: Normal(0, 0.05) to match the Python init
        let v = Tensor::<B, 4>::random(
            [self.out_channels, self.in_channels, kh, kw],
            Distribution::Normal(0.0, 0.05),
            device,
        );

        // g: initialised as the per-output-channel L2 norm of v
        let g = {
            let flat = v
                .clone()
                .reshape([self.out_channels, self.in_channels * kh * kw]);
            flat.powf_scalar(2.0_f32)
                .sum_dim(1)
                .sqrt()
                .reshape([self.out_channels, 1, 1, 1])
        };

        let bias = self
            .bias
            .then(|| Param::from_tensor(Tensor::zeros([self.out_channels], device)));

        WeightNormConv2d {
            v: Param::from_tensor(v),
            g: Param::from_tensor(g),
            bias,
            stride: self.stride,
            padding: self.padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WeightNormConv2d<B: Backend> {
    v: Param<Tensor<B, 4>>, // direction  [out, in, kH, kW]
    g: Param<Tensor<B, 4>>, // magnitude  [out, 1,  1,  1 ]
    bias: Option<Param<Tensor<B, 1>>>,
    stride: [usize; 2],
    padding: [usize; 2],
}

impl<B: Backend> WeightNormConv2d<B> {
    /// Reconstruct the normalised kernel: w = g * (v / ‖v‖₂)
    fn calc_weight(&self) -> Tensor<B, 4> {
        let v = self.v.val();
        let g = self.g.val();
        let [out_ch, in_ch, kh, kw] = v.dims();

        let v_norm = v
            .clone()
            .reshape([out_ch, in_ch * kh * kw])
            .powf_scalar(2.0_f32)
            .sum_dim(1)
            .sqrt()
            .reshape([out_ch, 1, 1, 1]);

        v * g / v_norm
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let w = self.calc_weight();
        let bias = self.bias.as_ref().map(|b| b.val());

        conv2d(
            x,
            w,
            bias,
            ConvOptions::new(self.stride, self.padding, [1, 1], 1),
        )
    }
}

// ─────────────────────────────────────────────────────────────
// ConvBlock
//
// [Conv3x3-WN → ReLU → Conv1x1-WN → ReLU → Conv3x3-WN]
// Output channels = 2 * in_channels (for affine-coupling splits).
// ─────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    pub in_channels: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ConvBlock<B> {
        let h = self.hidden_features;

        // conv1: 3×3, padding=1 — in → hidden
        let conv1 = WeightNormConv2dConfig::new(self.in_channels, h, [3, 3])
            .with_padding([1, 1])
            .init(device);

        // conv2: 1×1 — hidden → hidden
        let conv2 = WeightNormConv2dConfig::new(h, h, [1, 1]).init(device);

        // conv3: 3×3, padding=1 — hidden → 2*in (log_scale + shift)
        let conv3 = WeightNormConv2dConfig::new(h, 2 * self.in_channels, [3, 3])
            .with_padding([1, 1])
            .init(device);

        ConvBlock {
            conv1,
            conv2,
            conv3,
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: WeightNormConv2d<B>,
    conv2: WeightNormConv2d<B>,
    conv3: WeightNormConv2d<B>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.activation.forward(self.conv1.forward(x));
        let out = self.activation.forward(self.conv2.forward(out));
        self.conv3.forward(out)
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
        let val = act.forward(x).into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            (val - expected).abs() < tol,
            "activation: got {val:.6}, expected {expected:.6}"
        );
    }

    // ── WeightNormConv2d: output shape ──────────────────────────────────────────
    // Parameterised over: (in_ch, out_ch, kernel, stride, padding, H_in, W_in, H_out, W_out)

    #[rstest]
    #[case::same_padding(   3,  8, [3,3], [1,1], [1,1], 16, 16, 16, 16)]
    #[case::no_padding(     3,  8, [3,3], [1,1], [0,0], 16, 16, 14, 14)]
    #[case::pointwise_conv( 3, 16, [1,1], [1,1], [0,0],  8,  8,  8,  8)]
    #[case::strided(        8, 16, [3,3], [2,2], [1,1], 16, 16,  8,  8)]
    fn wn_conv2d_output_shape(
        device: NdArrayDevice,
        #[case] in_ch: usize,
        #[case] out_ch: usize,
        #[case] kernel: [usize; 2],
        #[case] stride: [usize; 2],
        #[case] padding: [usize; 2],
        #[case] h_in: usize,
        #[case] w_in: usize,
        #[case] h_out: usize,
        #[case] w_out: usize,
    ) {
        let conv = WeightNormConv2dConfig::new(in_ch, out_ch, kernel)
            .with_stride(stride)
            .with_padding(padding)
            .init(&device);

        let x = Tensor::<B, 4>::random(
            [2, in_ch, h_in, w_in],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(conv.forward(x).dims(), [2, out_ch, h_out, w_out]);
    }

    // ── WeightNormConv2d: weight norm property ──────────────────────────────────
    // Verifies that for every output channel c: ||w_c||₂ == g_c

    #[rstest]
    fn wn_conv2d_weight_norm_property(device: NdArrayDevice) {
        let (out_ch, in_ch) = (8, 3);
        let conv = WeightNormConv2dConfig::new(in_ch, out_ch, [3, 3])
            .with_padding([1, 1])
            .init::<B>(&device);

        let w = conv.calc_weight(); // [out_ch, in_ch, 3, 3]
        let g = conv.g.val(); // [out_ch, 1,    1, 1]

        let w_norm = w
            .reshape([out_ch, in_ch * 9])
            .powf_scalar(2.0_f32)
            .sum_dim(1)
            .sqrt()
            .reshape([out_ch, 1, 1, 1]);

        let max_diff = (w_norm - g).abs().max().into_scalar();
        assert!(
            max_diff < 1e-5_f32,
            "weight norm property violated: max |‖w_c‖ - g_c| = {max_diff}"
        );
    }

    // ── WeightNormConv2d: no bias variant ───────────────────────────────────────

    #[rstest]
    fn wn_conv2d_no_bias(device: NdArrayDevice) {
        let conv = WeightNormConv2dConfig::new(3, 8, [3, 3])
            .with_padding([1, 1])
            .with_bias(false)
            .init(&device);
        let x = Tensor::<B, 4>::random([1, 3, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        assert_eq!(conv.forward(x).dims(), [1, 8, 8, 8]);
        assert!(conv.bias.is_none());
    }

    // ── ConvBlock: output shape ─────────────────────────────────────────────────
    // Output must always be [B, 2 * in_channels, H, W]

    #[rstest]
    #[case::base(4, 512, 2, 16, 16)]
    #[case::small(8, 256, 1, 8, 8)]
    #[case::tiny(12, 128, 3, 4, 4)]
    fn conv_block_output_shape(
        device: NdArrayDevice,
        #[case] in_ch: usize,
        #[case] hidden: usize,
        #[case] batch: usize,
        #[case] h: usize,
        #[case] w: usize,
    ) {
        let block = ConvBlockConfig::new(in_ch)
            .with_hidden_features(hidden)
            .init(&device);
        let x = Tensor::<B, 4>::random(
            [batch, in_ch, h, w],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(block.forward(x).dims(), [batch, 2 * in_ch, h, w]);
    }

    // ── ConvBlock: numerical sanity ─────────────────────────────────────────────

    #[rstest]
    fn conv_block_output_is_finite(device: NdArrayDevice) {
        let block = ConvBlockConfig::new(4).init(&device);
        let x = Tensor::<B, 4>::random([2, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        let values = block.forward(x).into_data().to_vec::<f32>().unwrap();
        assert!(
            values.iter().all(|v| v.is_finite()),
            "ConvBlock output contains NaN or Inf"
        );
    }
}
