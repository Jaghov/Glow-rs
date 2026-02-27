mod dataset;

use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::data::dataloader::DataLoaderBuilder;
use burn::tensor::Tensor;
use dataset::celeba::CelebABatcher;

use dataset::celeba::CelebADataset;
use rerun::external::crossbeam::epoch::Pointable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = LibTorchDevice::Cuda(0);

    let batch_size = 6;

    let rec = rerun::RecordingStreamBuilder::new("rerun_example_image")
        .enabled(true)
        .spawn()?;

    let batcher_train = CelebABatcher::<LibTorch>::new(device);
    let dataloader = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(0)
        .num_workers(1)
        .build(CelebADataset::test());

    let batch = dataloader.iter().next().unwrap().images;
    let [t, .., c, h, w] = batch.dims();

    let data = batch.clone().permute([0, 2, 3, 1]).to_data();
    let buffer = data
        .convert_dtype(burn::tensor::DType::U8)
        .into_vec()
        .unwrap();
    // Cleaner loop if separate paths are needed
    let image_size = h * w * c;
    for (i, chunk) in buffer.chunks(image_size).enumerate() {
        let img = rerun::Image::from_rgb24(chunk.to_vec(), [w as u32, h as u32]);
        rec.log(format!("image/{}", i), &img)?;
    }

    // Compute the size

    // let image = rerun::Image::from_rgb24(item, [h as u32, w as u32]);

    // rec.log("image", &image)?;

    Ok(())
}
