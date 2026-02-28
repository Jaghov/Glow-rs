use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::data::dataloader::DataLoaderBuilder;
use burn::prelude::Backend;
use burn::Tensor;
use glow_rs::dataset::{
    bouncingball::BouncingBallBatcher,
    celeba::{CelebABatcher, CelebADataset},
};

use rerun::TimeColumn;

use glow_rs::dataset::bouncingball::BouncingBallDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = LibTorchDevice::Cuda(0);

    let rec = rerun::RecordingStreamBuilder::new("rerun_example_image").spawn()?;

    // // Log celeb
    // view_celeb::<LibTorch>(&device, &rec);

    // Log Bouncing Ball
    view_bball::<LibTorch>(&device, &rec)?;

    // Wait for all logging to finish before handing data to the viewer

    // Show storage
    rec.flush_blocking().unwrap();

    Ok(())
}

fn view_celeb<B: Backend>(device: &B::Device, rec: &rerun::RecordingStream) {
    let batch_size = 6;

    let batcher_train = CelebABatcher::<B>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(0)
        .num_workers(1)
        .build(CelebADataset::test());

    let batch = dataloader
        .iter()
        .next()
        .expect("should be non-empty")
        .images;
    let [_t, .., c, h, w] = batch.dims();

    let data = batch.clone().permute([0, 2, 3, 1]).to_data();
    let buffer = data
        .convert_dtype(burn::tensor::DType::U8)
        .into_vec()
        .unwrap();

    let image_size = h * w * c;
    for (i, chunk) in buffer.chunks(image_size).enumerate() {
        let img = rerun::Image::from_rgb24(chunk, [w as u32, h as u32]);
        rec.log(format!("image/{}", i), &img)
            .expect("No log errors");
    }
}
fn view_bball<B: Backend>(
    device: &B::Device,
    rec: &rerun::RecordingStream,
) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 1;

    let batcher_train = BouncingBallBatcher::<B>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(0)
        .num_workers(1)
        .build(BouncingBallDataset::test());

    let batch = dataloader
        .iter()
        .next()
        .expect("should be non-empty")
        .image_sequences;
    let [_b, t, c, h, w] = batch.dims();

    let time: Vec<i64> = (0..t as i64).collect();
    let timeline = TimeColumn::new_sequence("time", time);

    let format = rerun::components::ImageFormat::rgb8([w as u32, h as u32]);
    rec.log_static("images", &rerun::Image::update_fields().with_format(format))?;
    let batch = batch * 256; // Rescale intensity from [0,1] back to [0, 255]

    let data = batch
        .clone()
        .permute([0, 1, 3, 4, 2])
        .squeeze_dim::<4>(0)
        .to_data();
    let buffer = data
        .convert_dtype(burn::tensor::DType::U8)
        .into_vec()
        .unwrap();

    let image_size = h * w * c;
    rec.send_columns(
        "images",
        [timeline],
        rerun::Image::update_fields()
            .with_many_buffer(buffer.chunks(image_size))
            .columns_of_unit_batches()?,
    )?;

    Ok(())
}

// fn tensor_to_image<B: Backend>(tensor: Tensor<B, 4>) -> Vec<u8> {
//     let data = tensor.permute([0, 2, 3, 1]).to_data();
//     let buffer = data
//         .convert_dtype(burn::tensor::DType::U8)
//         .into_vec()
//         .unwrap();
//     buffer
// }
