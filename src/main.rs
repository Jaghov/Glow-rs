use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::tensor::Device;
use burn::tensor::{backend::Backend, Data, Shape, Tensor};
use tch::{Device as TchDevice, Kind, Tensor as TchTensor};

fn tensor_to_tch<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> TchTensor {
    let shape: Vec<i64> = tensor.shape().dims.iter().map(|&x| x as i64).collect();
    let data: Vec<f32> = tensor.to_data().convert().value;
    TchTensor::from_slice(&data).reshape(&shape)
}

fn tch_to_tensor<B: Backend, const D: usize>(
    tch_tensor: &TchTensor,
    device: &B::Device,
) -> Tensor<B, D> {
    let shape: Shape<D> = Shape::from(
        tch_tensor
            .size()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>(),
    );
    let data: Vec<f32> = tch_tensor.flatten(0, -1).try_into().unwrap();
    Tensor::<B, D>::from_data(Data::new(data, shape).convert(), device)
}

fn random_orthogonal_tensor<B: Backend>(shape: &[usize]) -> Tensor<B, 2> {
    let device = B::Device::default();

    // Create a random tensor using tch
    let tch_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let random_tensor = TchTensor::randn(&tch_shape, (Kind::Float, TchDevice::Cpu));

    // Perform QR decomposition
    let (q, _r) = TchTensor::linalg_qr(&random_tensor, "reduced");

    // Convert tch tensor to burn tensor
    tch_to_tensor(&q, &device)
}

fn main() {
    type MyBackend = LibTorch<f32>;

    // Create a random orthogonal tensor
    let shape = &[3, 3];
    let random_tensor: Tensor<MyBackend, 2> = random_orthogonal_tensor(shape);

    // Print the tensor
    println!("Random Orthogonal Tensor:");
    println!("{}", random_tensor);

    // Verify orthogonality
    let transpose = random_tensor.clone().transpose();
    let product = random_tensor.matmul(transpose);

    println!("\nVerification (should be close to identity matrix):");
    println!("{}", product);

    let determinant = tensor_to_tch(&product).det();
    let determinant: Tensor<MyBackend, 1> = tch_to_tensor(&determinant, &LibTorchDevice::Cpu);

    println!("\nHere is the determinant:");
    println!("{}", determinant);
}
// use burn::autodiff::Autodiff;
// use burn::backend::libtorch::LibTorch;
// use burn::prelude::Module; // For the forward method
// use burn::tensor::Device;
// use burn::tensor::{
//     backend::{AutodiffBackend, Backend},
//     Data, Shape, Tensor,
// };
// use tch::{Device as TchDevice, Kind, Tensor as TchTensor};

// fn tensor_to_tch<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> TchTensor {
//     let shape: Vec<i64> = tensor.shape().dims.iter().map(|&x| x as i64).collect();
//     let data: Vec<f32> = tensor.to_data().convert().value;
//     TchTensor::from_slice(&data).reshape(&shape)
// }

// fn tch_to_tensor<B: Backend, const D: usize>(
//     tch_tensor: &TchTensor,
//     device: &B::Device,
// ) -> Tensor<B, D> {
//     let shape: Shape<D> = Shape::from(
//         tch_tensor
//             .size()
//             .iter()
//             .map(|&x| x as usize)
//             .collect::<Vec<_>>(),
//     );
//     let data: Vec<f32> = tch_tensor.flatten(0, -1).try_into().unwrap();
//     Tensor::<B, D>::from_data(Data::new(data, shape).convert(), device)
// }

// fn random_orthogonal_tensor<B: AutodiffBackend>(shape: &[usize]) -> Tensor<B, 2> {
//     let device = B::Device::default();

//     // Create a random tensor using tch
//     let tch_shape: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
//     let random_tensor = TchTensor::randn(&tch_shape, (Kind::Float, TchDevice::Cpu));

//     // Perform QR decomposition
//     let (q, _r) = random_tensor.qr(false);

//     // Convert tch tensor to burn tensor with gradients
//     tch_to_tensor(&q, &device).trace()
// }

// // A simple model that uses our random orthogonal tensor
// struct SimpleModel<B: AutodiffBackend> {
//     weight: Tensor<B, 2>,
// }

// impl<B: AutodiffBackend> Module<Tensor<B, 2>> for SimpleModel<B> {
//     fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
//         input.matmul(self.weight.clone())
//     }
// }

// fn main() {
//     type MyBackend = Autodiff<LibTorch<f32>>;

//     // Create a random orthogonal tensor
//     let shape = &[3, 3];
//     let random_tensor: Tensor<MyBackend, 2> = random_orthogonal_tensor(shape);

//     // Create a simple model
//     let model = SimpleModel {
//         weight: random_tensor,
//     };

//     // Create some input data
//     let input = Tensor::<MyBackend, 2>::random(
//         [3, 3],
//         burn::tensor::Distribution::Default,
//         Device::default(),
//     );

//     // Forward pass
//     let output = model.forward(input);

//     // Compute loss (for example, sum of all elements)
//     let loss = output.sum();

//     // Backward pass
//     let gradients = loss.backward();

//     // Print the gradients
//     println!("Gradients of the weight:");
//     println!("{}", model.weight.grad(&gradients));

//     // You can also use an optimizer to update the weights
//     // let mut optimizer = Adam::new(model, AdamConfig::default());
//     // optimizer.step(&gradients);

//     println!(
//         "\nNote: This example demonstrates backpropagation through the random orthogonal tensor."
//     );
//     println!("In a real scenario, you'd typically use this within a larger neural network and training loop.");
// }
