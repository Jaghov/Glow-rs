use std::marker::PhantomData;

use burn::{config::Config, prelude::Backend, tensor::Shape, Tensor};

#[derive(Config, Debug)]
pub struct DequantizeConfig {
    pixel_depth: u32,
}

impl DequantizeConfig {
    pub fn init<B: Backend>(&self, _device: &B::Device) -> Dequantize<B> {
        assert!(
            self.pixel_depth <= 8,
            "Pixel depth should be <= 8 got '{}' instead",
            self.pixel_depth
        );
        Dequantize::<B> {
            pixel_depth: self.pixel_depth,
            _backend: PhantomData,
        }
    }
}
/// Change of variables formula is defined only in the continuous space
/// To get it to work for discrete functions such as in image pixels
/// We have to give a minimum penalty to the loss to correct the log_det_jacobian and keep the model from collapsing.
pub struct Dequantize<B: Backend> {
    /// Number of bits to represent a pixel
    pixel_depth: u32,
    _backend: PhantomData<B>,
}

impl<B: Backend> Dequantize<B> {
    /// This version introduces noise and adds the dequantisation Jacocobian cost.
    pub fn forward_train(&self, image_batch: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let missing_precision = 8u32 - self.pixel_depth;
        let scale = 2u32.pow(missing_precision) as f32;
        // floor() is what makes pixel_depth < 8 actually quantize; for pixel_depth = 8
        // the input is already integer-valued so this is a no-op.
        let y = (image_batch / scale).floor();

        // Add uniform noise
        let y = y.random_like(burn::tensor::Distribution::Uniform(0., 1.)) + y;

        let y = y / 2u32.pow(self.pixel_depth) as f32 - 0.5;
        let c = self.discretization_penalty(&y.dims(), &y.device());

        (y, c)
    }
    /// This version purely performs the forward flow, without the Jacobian computation per batch item
    /// or adding the unifrom noise.
    pub fn forward(&self, image_batch: Tensor<B, 4>) -> Tensor<B, 4> {
        let missing_precision = 8u32 - self.pixel_depth;
        let scale = 2u32.pow(missing_precision) as f32;
        let y = (image_batch / scale).floor();

        let y = y / 2u32.pow(self.pixel_depth) as f32 - 0.5;

        y
    }

    /// Log-factor contribution from uniform dequantization (same for train and eval), shape `[B]`.
    pub fn discretization_penalty(&self, dims: &[usize], device: &B::Device) -> Tensor<B, 1> {
        let batch_size = dims[0];
        let num_elements = dims.iter().skip(1).product::<usize>() as f32;
        let bins = 2u32.pow(self.pixel_depth) as f32;
        let penalty = -num_elements * bins.ln();
        Tensor::full(Shape::new([batch_size]), penalty, device)
    }

    pub fn inverse(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        // Recover the quantized level x_q ∈ [0, n_levels - 1] by inverting the
        // forward map, then expand back to 8-bit pixel space. floor() cancels
        // the U[0,1) noise added by forward_train exactly.
        let n_levels = 2u32.pow(self.pixel_depth) as f32;
        let coarsen = 2u32.pow(8u32 - self.pixel_depth) as f32;
        let x_q = ((y + 0.5) * n_levels).floor().clamp(0.0, n_levels - 1.0);
        x_q * coarsen
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{check_closeness, Distribution, Tensor, Tolerance};
    use rstest::{fixture, rstest};

    type B = NdArray;
    #[fixture]
    fn preprocess() -> Dequantize<B> {
        DequantizeConfig { pixel_depth: 8 }.init(&Default::default())
    }

    fn dummy_image(batch: usize) -> Tensor<B, 4> {
        // Real CelebA pixels arrive as integer-valued floats (u8 cast to f32);
        // floor matches that distribution so quantization round-trips are exact.
        Tensor::<B, 4>::random(
            [batch, 3, 4, 4],
            Distribution::Uniform(0., 256.),
            &Default::default(),
        )
        .floor()
        .clamp(0., 255.)
    }

    #[rstest]
    fn jitters_during_training(preprocess: Dequantize<NdArray>) {
        let input = dummy_image(2);

        let (train_out, _) = preprocess.forward_train(input.clone());
        let eval_out = preprocess.forward(input.clone());

        // Training mode adds U[0,1) dequant noise; eval mode does not.
        assert!(!train_out.clone().all_close(eval_out, None, Some(1e-6)));

        // The inverse must cancel that noise exactly: floor((x_q + u)) = x_q.
        let reconstructed = preprocess.inverse(train_out);
        check_closeness(&reconstructed, &input);
        reconstructed
            .into_data()
            .assert_approx_eq::<f32>(&input.into_data(), Tolerance::absolute(1e-4));
    }

    #[rstest]
    fn no_jitter_in_eval_mode(preprocess: Dequantize<NdArray>) {
        let input = dummy_image(8);

        let output = preprocess.forward(input.clone());

        let reconstructed = preprocess.inverse(output);

        check_closeness(&reconstructed, &input);
        // assert!(reconstructed.all_close(input, None, None));
        reconstructed
            .into_data()
            .assert_approx_eq::<f32>(&input.into_data(), Tolerance::absolute(0.5));
    }

    #[rstest]
    fn output_range(preprocess: Dequantize<NdArray>) {
        let input = dummy_image(2);

        let (output, _) = preprocess.forward_train(input);

        let max = output.clone().max().into_scalar();
        let min = output.min().into_scalar();

        assert!(max < 0.5);
        assert!(min > -0.5);
    }

    #[rstest]
    fn log_det_shape(preprocess: Dequantize<NdArray>) {
        let input = dummy_image(4);

        let (_, log_det) = preprocess.forward_train(input);

        assert_eq!(log_det.dims(), [4]);
    }

    #[rstest]
    fn log_det_value(preprocess: Dequantize<NdArray>) {
        let input = dummy_image(2);

        let (_, log_det) = preprocess.forward_train(input.clone());

        let dims = input.dims();
        let num_elements = dims[1] * dims[2] * dims[3];

        let expected = log_det.full_like(-(num_elements as f32) * (8f32 * std::f32::consts::LN_2));

        check_closeness(&log_det, &expected);
        assert!(log_det.all_close(expected, None, None));
    }
}
