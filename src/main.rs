use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::tensor::Tensor;

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
}

// extern crate tch;

// use tch::{Cuda, Device};

// fn main() {
//     if !Cuda::is_available() {
//         panic!("Could not detect valid CUDA configuration. No CUDA devices found!");
//     } else {
//         println!("CUDA is available! Device count: {}", Cuda::device_count());
//         for device_index in 0..Cuda::device_count() {
//             println!(
//                 "Device {}: {:?}",
//                 device_index,
//                 Device::Cuda(device_index as usize)
//             );
//         }
//     }
// }

// use burn::backend::libtorch::{LibTorch, LibTorchDevice, TchTensor};
// use burn::backend::Autodiff;
// use burn::tensor::Device;
// use burn::tensor::{
//     backend::{AutodiffBackend, Backend},
//     Data, Shape, Tensor,
// };
// use tch::{Device as TchDevice, Kind};

// fn random_orthogonal_tensor<B: Backend>(
//     shape: &[usize],
//     device: &B::Device,
// ) -> TchTensor<f32, 2> {
//     // Create a random tensor using tch
//     let tch_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
//     let random_tensor = tch::Tensor::randn(&tch_shape, (Kind::Float, TchDevice::Cuda(0)));
//     // Perform QR decomposition
//     let (q, _r) = tch::Tensor::linalg_qr(&random_tensor, "reduced");
//     // Convert tch tensor to burn tensor with gradients
//     {
//         let tensor = TchTensor::new(q);
//         Tensor::new(tensor)
//     }
// }

// fn main() {
//     type MyBackend = LibTorch<f32>;
//     let device = LibTorchDevice::Cuda(0);

//     // Create a random orthogonal tensor
//     let shape = &[3, 3];
//     let random_tensor: Tensor<MyBackend, 2> = random_orthogonal_tensor(shape, &device);

//     // Print the tensor
//     println!("Random Orthogonal Tensor:");
//     println!("{}", random_tensor);

//     // Verify orthogonality
//     let transpose = random_tensor.clone().transpose();
//     let product = random_tensor.matmul(transpose);

//     println!("\nVerification (should be close to identity matrix):");
//     println!("{}", product);

//     let determinant = tensor_to_tch(&product).det();
//     let determinant: Tensor<MyBackend, 1> = tch_to_tensor(&determinant, &device);

//     println!("\nHere is the determinant:");
//     println!("{}", determinant);

//     // Compute gradients
//     let loss = determinant.sum();
//     let gradients = loss.backward();

//     println!("\nGradients of the random orthogonal tensor:");
//     println!("{}", random_tensor.grad(&gradients));
// }
