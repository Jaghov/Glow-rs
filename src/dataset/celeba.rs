use burn::data::dataset::transform::Mapper;
use burn::data::dataset::transform::MapperDataset;
use burn::data::dataset::Dataset;
use burn::data::dataset::HuggingfaceDatasetLoader;
use burn::data::dataset::SqliteDataset;
use burn::prelude::*;
use serde::Deserialize;

const WIDTH: usize = 128;
const HEIGHT: usize = 128;
const CHANNELS: usize = 3;

#[derive(Deserialize, Debug, Clone)]
struct CelebAItemRaw {
    image_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct CelebAItem {
    pub image: [[[f32; WIDTH]; HEIGHT]; CHANNELS],
}

pub struct CelebABatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CelebABatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

// Bytes to image
// Bytes to image
struct BytesToImage;

impl Mapper<CelebAItemRaw, CelebAItem> for BytesToImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (2D array image).
    fn map(&self, item: &CelebAItemRaw) -> CelebAItem {
        // Ensure the image dimensions are correct.
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT * CHANNELS);

        // Convert the image to a 2D array of floats.
        let mut image_array = [[[0f32; WIDTH]; HEIGHT]; CHANNELS];
        for (i, pixel) in item.image_bytes.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            let z = i / CHANNELS;
            image_array[z][y][x] = *pixel as f32;
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
                .with_base_dir("dataset/celeba")
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
// struct CelebAItem

// Impl mapper from Item Raw to Item

// Finally a CelebA dataset
