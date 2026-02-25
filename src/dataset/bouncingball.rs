use burn::{
    backend::{
        ndarray::{NdArrayTensor, NdArrayTensorFloat},
        NdArray,
    },
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    tensor::{backend::Backend, Device, Tensor, TensorPrimitive},
};
use ndarray::Array4;
use ndarray_npz::NpzReader;
use std::fs::{read_dir, File};

#[derive(Clone, Debug)]
pub struct BouncingBallBatch<B: Backend> {
    image_sequences: Tensor<B, 5>,
}

#[derive(Clone, Debug)]
pub struct BouncingBallItem {
    image_sequence: Array4<f32>,
}

type BallDataset = InMemDataset<BouncingBallItem>;

pub struct BouncingBallDataset {
    dataset: BallDataset,
}

/// Passed dataloader builder along with dataset to return a dataloader for training
/// ```
/// # Example use
/// let batcher_train = BouncingBallBatcher::<LibTorch>::new(LibTorchDevice::Cpu);
///
/// let dataloader_train = DataLoaderBuilder::new(batcher_train)
///     .batch_size(2)
///     .shuffle(0)
///     .num_workers(1)
///     .build(BouncingBallDataset::train());
/// let item = dataloader_train.iter().next();
/// println!("{:?}", item);
/// ````
#[derive(Clone, Debug)]
pub struct BouncingBallBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<B, BouncingBallItem, BouncingBallBatch<B>> for BouncingBallBatcher<B> {
    fn batch(&self, items: Vec<BouncingBallItem>, device: &Device<B>) -> BouncingBallBatch<B> {
        let image_sequence = items
            .iter()
            .map(|data| {
                Tensor::<NdArray, 5>::from_primitive(TensorPrimitive::Float(
                    NdArrayTensorFloat::F32(NdArrayTensor {
                        array: data.image_sequence.clone().into_dyn().into_shared(),
                    }),
                ))
                .permute([0, 1, 4, 2, 3])
            })
            .collect();
        let image_sequences = Tensor::cat(image_sequence, 0);
        let image_sequences = Tensor::<B, 5>::from_data(image_sequences.into_data(), &self.device);

        BouncingBallBatch { image_sequences }
    }
}

impl<B: Backend> BouncingBallBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl BouncingBallDataset {
    pub fn train() -> Self {
        Self::new("train")
    }
    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let file_iterator = read_dir(format!("data/images_{split}_N_5000_T_100_dim_latent_2_dim_obs_2_resolution_32_state_3_sparsity_0.0_net_cosine_seed_24/" )).unwrap();

        let items: Vec<_> = file_iterator
            .filter_map(|f| f.ok())
            .filter_map(|seq| {
                let mut npz = NpzReader::new(File::open(seq.path()).ok()?).ok()?;
                let a1: Array4<f64> = npz.by_index(0).ok()?;
                let a1: Array4<f32> = a1.mapv(|x| x as f32);
                Some(BouncingBallItem { image_sequence: a1 })
            })
            .collect();

        let dataset = InMemDataset::new(items);
        // let dataset = ShuffledDataset::new(dataset,);

        Self { dataset }
    }
}

impl Dataset<BouncingBallItem> for BouncingBallDataset {
    fn get(&self, index: usize) -> Option<BouncingBallItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// pub fn read_npz_file() {
//     let mut file_iterator = read_dir("data/images_train_N_5000_T_100_dim_latent_2_dim_obs_2_resolution_32_state_3_sparsity_0.0_net_cosine_seed_24/").unwrap();
//     // println!("{:?}", file_iterator.unwrap());
//     let Some(Ok(sequence)) = file_iterator.next() else {
//         return;
//     };
//     let mut npz = NpzReader::new(File::open(sequence.path()).unwrap()).unwrap();

//     let a1: Array4<f64> = npz.by_index(0).unwrap();
//     let a2: Array4<f32> = a1.mapv(|x| x as f32);

//     println!("{:?}", a2);
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn can_read() {}
// }
