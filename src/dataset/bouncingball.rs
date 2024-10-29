use burn::{
    backend::{ndarray::NdArrayTensor, NdArray},
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    tensor::{backend::Backend, Tensor, TensorPrimitive},
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

pub struct BouncingBallBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<BouncingBallItem, BouncingBallBatch<B>> for BouncingBallBatcher<B> {
    fn batch(&self, items: Vec<BouncingBallItem>) -> BouncingBallBatch<B> {
        let image_sequence = items
            .iter()
            .map(|item| {
                Tensor::<NdArray, 5>::from_primitive(TensorPrimitive::Float(NdArrayTensor::new(
                    item.image_sequence.clone().into_dyn().into_shared(),
                )))
            })
            .map(|item_nd| {
                Tensor::<B, 5>::from_data(item_nd.into_data(), &self.device)
                    .permute([0, 1, 4, 2, 3])
                // shape (B, T, C, H,  W)
            })
            .collect();
        let image_sequences = Tensor::cat(image_sequence, 0);

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

pub fn read_npz_file() {
    let mut file_iterator = read_dir("data/images_train_N_5000_T_100_dim_latent_2_dim_obs_2_resolution_32_state_3_sparsity_0.0_net_cosine_seed_24/").unwrap();
    // println!("{:?}", file_iterator.unwrap());
    let Some(Ok(sequence)) = file_iterator.next() else {
        return;
    };
    let mut npz = NpzReader::new(File::open(sequence.path()).unwrap()).unwrap();

    let a1: Array4<f64> = npz.by_index(0).unwrap();
    let a2: Array4<f32> = a1.mapv(|x| x as f32);

    println!("{:?}", a2);
}

impl Dataset<BouncingBallItem> for BouncingBallDataset {
    fn get(&self, index: usize) -> Option<BouncingBallItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
