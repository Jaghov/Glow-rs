use std::io::Cursor;

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

// pub fn load_raw_dataset() -> SqliteDataset<CelebAItemRaw> {
//     let dataset: SqliteDataset<CelebAItemRaw> =
//         HuggingfaceDatasetLoader::new("tglcourse/CelebA-faces-cropped-128")
//             .with_base_dir("dataset/celeba")
//             .dataset("test")
//             .unwrap();
//     dataset
// }
// Bytes to image
// Bytes to image
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
        let mut image_array = [[[0f32; WIDTH]; HEIGHT]; CHANNELS];
        for (i, pixel) in img.iter().enumerate() {
            let color = i % CHANNELS;
            let x = (i / CHANNELS) % WIDTH;
            let y = (i / CHANNELS) / HEIGHT;
            image_array[color][y][x] = *pixel as f32;
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
