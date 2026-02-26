use std::io::Cursor;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::transform::Mapper;
use burn::data::dataset::transform::MapperDataset;
use burn::data::dataset::Dataset;
use burn::data::dataset::HuggingfaceDatasetLoader;
use burn::data::dataset::SqliteDataset;
use burn::prelude::*;
use image::codecs::png::PngDecoder;
use image::ImageDecoder;
use serde::Deserialize;

const WIDTH: usize = 128;
const HEIGHT: usize = 128;
const CHANNELS: usize = 3;

#[derive(Deserialize, Debug, Clone)]
pub struct CelebAItemRaw {
    // TODO Make private after done testing
    pub image_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CelebAItem {
    pub image: [[[u8; WIDTH]; HEIGHT]; CHANNELS],
}

#[derive(Debug, Clone)]
pub struct CelebABatch<B: Backend> {
    pub images: Tensor<B, 4, Int>,
}

/// Passed dataloader builder along with dataset to return a dataloader for training
/// # Example use
/// ```
/// use burn::{
///     backend::libtorch::{LibTorch, LibTorchDevice},
///     data::dataloader::DataLoaderBuilder,
/// };
/// use glow_rs::dataset::*;
/// let batcher_train = CelebABatcher::<LibTorch>::new(LibTorchDevice::Cuda(0));
///
/// let dataloader_train = DataLoaderBuilder::new(batcher_train)
///     .batch_size(2)
///     .shuffle(0)
///     .num_workers(1)
///     .build(CelebADataset::train());
/// let item = dataloader_train.iter().next();
/// println!("{:?}", item);
/// ````
pub struct CelebABatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CelebABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, CelebAItem, CelebABatch<B>> for CelebABatcher<B> {
    fn batch(&self, items: Vec<CelebAItem>, device: &B::Device) -> CelebABatch<B> {
        let images = items
            .into_iter()
            .map(|item| Tensor::<B, 3, Int>::from_data(item.image, device).unsqueeze_dim::<4>(0))
            .collect();
        let batch = Tensor::cat(images, 0);

        CelebABatch { images: batch }
    }
}

struct BytesToImage;

impl Mapper<CelebAItemRaw, CelebAItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &CelebAItemRaw) -> CelebAItem {
        // Decode png as Pixel intensities
        let decoder = PngDecoder::new(Cursor::new(&item.image_bytes)).unwrap();
        let mut img: Vec<u8> = vec![0; decoder.total_bytes() as usize];
        decoder.read_image(&mut img).unwrap();

        // Ensure the image dimensions are correct.
        debug_assert_eq!(img.len(), WIDTH * HEIGHT * CHANNELS);

        // Convert the image to a 2D array of floats.
        let mut image_array = [[[0u8; WIDTH]; HEIGHT]; CHANNELS];
        for (i, pixel) in img.iter().enumerate() {
            let color = i % CHANNELS;
            let x = (i / CHANNELS) % WIDTH;
            let y = (i / CHANNELS) / HEIGHT;
            image_array[color][y][x] = *pixel as u8;
        }

        CelebAItem { image: image_array }
    }
}

type MappedDataset = MapperDataset<SqliteDataset<CelebAItemRaw>, BytesToImage, CelebAItemRaw>;

pub struct CelebADataset {
    dataset: MappedDataset,
}

impl CelebADataset {
    pub fn train() -> Self {
        Self::new("train")
    }
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let dataset: SqliteDataset<CelebAItemRaw> =
            HuggingfaceDatasetLoader::new("tglcourse/CelebA-faces-cropped-128")
                .with_base_dir("data/celeba")
                .dataset(split)
                .unwrap();
        let dataset = MapperDataset::new(dataset, BytesToImage);

        CelebADataset { dataset }
    }
}

impl Dataset<CelebAItem> for CelebADataset {
    fn get(&self, index: usize) -> Option<CelebAItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
