use burn::{
    config::Config,
    module::{Module, Param},
    nn::Relu,
    tensor::{
        backend::Backend, module::conv2d, ops::ConvOptions, Device, Distribution, Tensor,
    },
};

#[derive(Config, Debug)]
/// Config for [`WeightNormConv2d`]
pub struct WeightNormConv2dConfig {
    /// Number in input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel
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
/// Paretramised convolution
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

/// ─────────────────────────────────────────────────────────────
/// ConvBlock
///
/// \[Conv3x3 → ReLU → Conv1x1 → ReLU → Conv3x3\]
///
/// All three convs use Salimans–Kingma weight normalisation. Output channels
/// = `out_channels_factor * in_channels`; affine coupling sets factor=2 (raw_s
/// + shift) and additive coupling sets factor=1 (shift only).
/// ─────────────────────────────────────────────────────────────
#[derive(Config, Debug)]
pub struct ConvBlockConfig {
    pub in_channels: usize,
    #[config(default = "512")]
    pub hidden_features: usize,
    /// Final conv output channel count = `out_channels_factor * in_channels`.
    /// Affine coupling needs 2 (raw_s + shift); additive needs 1 (shift only).
    #[config(default = "2")]
    pub out_channels_factor: usize,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ConvBlock<B> {
        let h = self.hidden_features;
        let out_c = self.out_channels_factor * self.in_channels;

        let conv1 = WeightNormConv2dConfig::new(self.in_channels, h, [3, 3])
            .with_padding([1, 1])
            .init::<B>(device);
        let conv2 = WeightNormConv2dConfig::new(h, h, [1, 1])
            .with_padding([0, 0])
            .init::<B>(device);
        // conv3 is zero-initialised (`g = 0`). With raw_s ≈ 0 the scaled sigmoid
        // in `coupling.rs` gives s ≈ 0.891 (not 1.0), so the coupling layer is
        // **near-identity** at init — see the constants block in
        // `src/models/flow/coupling.rs` for the full rationale.
        let mut conv3 = WeightNormConv2dConfig::new(h, out_c, [3, 3])
            .with_padding([1, 1])
            .init::<B>(device);
        conv3.g = Param::from_tensor(Tensor::zeros_like(&conv3.g.val()));

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

    // ── WeightNormConv2d: output shape ──────────────────────────────────────────
    // Parameterised over: (in_ch, out_ch, kernel, stride, padding, H_in, W_in, H_out, W_out)

    #[allow(clippy::too_many_arguments)]
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

        check_closeness(&w_norm, &g);
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
    // Output must always be [B, out_channels_factor * in_channels, H, W]

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

    /// Additive conv block must produce `in_channels` outputs (factor=1), not
    /// `2 * in_channels`. Catches accidental factor-2 regressions in the additive path.
    #[rstest]
    fn conv_block_additive_factor_one(device: NdArrayDevice) {
        let block = ConvBlockConfig::new(4)
            .with_out_channels_factor(1)
            .init::<B>(&device);
        let x = Tensor::<B, 4>::random([2, 4, 8, 8], Distribution::Normal(0.0, 1.0), &device);
        assert_eq!(block.forward(x).dims(), [2, 4, 8, 8]);
    }
}
