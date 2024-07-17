use burn::backend::libtorch::{LibTorch, LibTorchDevice, TchTensor};
use burn::backend::Autodiff;
use burn::tensor::Device;
use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    Data, Shape, Tensor,
};
use tch::{Device as TchDevice, Kind};

fn random_orthogonal_tensor<B: AutodiffBackend>(
    shape: &[usize],
    device: &B::Device,
) -> TchTensor<f32, 2> {
    // Create a random tensor using tch
    let tch_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let random_tensor = tch::Tensor::randn(&tch_shape, (Kind::Float, TchDevice::Cpu));
    // Perform QR decomposition
    let (q, _r) = tch::Tensor::linalg_qr(&random_tensor, "reduced");
    // Convert tch tensor to burn tensor with gradients
    TchTensor::new(q)
}

fn main() {
    type MyBackend = Autodiff<LibTorch<f32>>;
    let device = LibTorchDevice::Cuda(0);

    // Create a random orthogonal tensor
    let shape = &[3, 3];
    let random_tensor: Tensor<MyBackend, 2> = random_orthogonal_tensor(shape, &device);

    // Print the tensor
    println!("Random Orthogonal Tensor:");
    println!("{}", random_tensor);

    // Verify orthogonality
    let transpose = random_tensor.clone().transpose();
    let product = random_tensor.matmul(transpose);

    println!("\nVerification (should be close to identity matrix):");
    println!("{}", product);

    let determinant = tensor_to_tch(&product).det();
    let determinant: Tensor<MyBackend, 1> = tch_to_tensor(&determinant, &device);

    println!("\nHere is the determinant:");
    println!("{}", determinant);

    // Compute gradients
    let loss = determinant.sum();
    let gradients = loss.backward();

    println!("\nGradients of the random orthogonal tensor:");
    println!("{}", random_tensor.grad(&gradients));
}
