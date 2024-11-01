mod dataset;

use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::tensor::Tensor;
use dataset::CelebADataset;

fn main() {
    println!("number of devices: {}", tch::Cuda::device_count());
    println!("{}", tch::Cuda::cudnn_is_available());
    assert!(
        tch::utils::has_cuda(),
        "Could not detect valid CUDA configuration"
    );

    let device = LibTorchDevice::Cuda(0);

    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<LibTorch<f32>, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::ones_like(&tensor_1);

    // Print the element-wise addition of the two tensors.
    println!("{}", tensor_1 + tensor_2);

    let dataset = CelebADataset::test();

    // let batcher_train = BouncingBallBatcher::<LibTorch>::new(LibTorchDevice::Cpu);

    // let dataloader_train = DataLoaderBuilder::new(batcher_train)
    //     .batch_size(2)
    //     .shuffle(0)
    //     .num_workers(1)
    //     .build(BouncingBallDataset::train());
    // let item = dataloader_train.iter().next();
    // println!("{:?}", item);
    // read_npz_file();
}
