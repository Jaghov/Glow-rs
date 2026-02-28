//! Passed dataloader builder along with dataset to return a dataloader for training
//! # Example use
//! ```
//! use burn::{
//!     backend::libtorch::{LibTorch, LibTorchDevice},
//!     data::dataloader::DataLoaderBuilder,
//! };
//! use glow_rs::dataset::*;
//! let batcher_train = CelebABatcher::<LibTorch>::new(LibTorchDevice::Cuda(0));
//!
//! let dataloader_train = DataLoaderBuilder::new(batcher_train)
//!     .batch_size(2)
//!     .shuffle(0)
//!     .num_workers(1)
//!     .build(CelebADataset::train());
//! let item = dataloader_train.iter().next();
//! println!("{:?}", item);
//! ```
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

/// Width of each image in pixels after cropping.
const WIDTH: usize = 128;

/// Height of each image in pixels after cropping.
const HEIGHT: usize = 128;

/// Number of colour channels per image (R, G, B).
const CHANNELS: usize = 3;

/// A collated mini-batch of celebrity face images ready for model consumption.
///
/// `images` has shape `[B, C, H, W]` — batch × channels × height × width —
/// where each value is an integer pixel intensity in the range `[0, 255]`.
#[derive(Debug, Clone)]
pub struct CelebABatch<B: Backend> {
    /// Batched images tensor of shape `[batch, channels, height, width]`.
    pub images: Tensor<B, 4, Int>,
}

/// Collates individual [`CelebAItem`]s into a [`CelebABatch`] for a given backend.
///
/// Constructed once and passed to [`DataLoaderBuilder::new`].  The batcher
/// stacks each item's CHW tensor along a new leading batch dimension and
/// concatenates them into a single `[B, C, H, W]` tensor.
pub struct CelebABatcher<B: Backend> {
    /// Target device on which the output tensors will be allocated.
    device: B::Device,
}

impl<B: Backend> CelebABatcher<B> {
    /// Creates a new batcher that places tensors on `device`.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, CelebAItem, CelebABatch<B>> for CelebABatcher<B> {
    /// Converts a `Vec` of [`CelebAItem`]s into a single [`CelebABatch`].
    ///
    /// Each item is converted to a rank-3 integer tensor of shape `[C, H, W]`,
    /// unsqueezed to `[1, C, H, W]`, then all items are concatenated along
    /// dimension `0` to produce the final `[B, C, H, W]` batch tensor.
    fn batch(&self, items: Vec<CelebAItem>, device: &B::Device) -> CelebABatch<B> {
        let images = items
            .into_iter()
            .map(|item| Tensor::<B, 3, Int>::from_data(item.image, device).unsqueeze_dim::<4>(0))
            .collect();
        let batch = Tensor::cat(images, 0);

        CelebABatch { images: batch }
    }
}
/// Convenience alias for the lazily-mapped dataset type used internally by
/// [`CelebADataset`].
type MappedDataset = MapperDataset<SqliteDataset<CelebAItemRaw>, BytesToImage, CelebAItemRaw>;

/// Dataset of 128 × 128 pixel cropped celebrity face images sourced from the
/// [`tglcourse/CelebA-faces-cropped-128`](https://huggingface.co/datasets/tglcourse/CelebA-faces-cropped-128)
/// HuggingFace dataset.
///
/// Images are downloaded once and cached as a local SQLite database under
/// `data/celeba/`.  Access is provided via the [`Dataset`] trait; individual
/// items are decoded lazily on retrieval.
///
/// # Example
/// ```
/// use glow_rs::dataset::CelebADataset;
/// use burn::data::dataset::Dataset;
///
/// let train = CelebADataset::train();
/// println!("Training samples: {}", train.len());
/// ```
pub struct CelebADataset {
    /// The underlying lazily-mapped SQLite dataset.
    dataset: MappedDataset,
    /// The transformation applied to each raw item on access.
    transformation: BytesToImage,
}

impl CelebADataset {
    /// Returns a dataset over the **training** split.
    ///
    /// Equivalent to `CelebADataset::new("train")`.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Returns a dataset over the **test** split.
    ///
    /// Equivalent to `CelebADataset::new("test")`.
    pub fn test() -> Self {
        Self::new("test")
    }

    /// Loads the given `split` (`"train"` or `"test"`) from the local SQLite
    /// cache, downloading via HuggingFace if the cache does not yet exist.
    ///
    /// The dataset is stored under `data/celeba/`.  A [`BytesToImage`] mapper
    /// with `quantize_factor = 2` is attached for lazy decoding.
    ///
    /// # Panics
    /// Panics if the HuggingFace download or SQLite initialisation fails.
    fn new(split: &str) -> Self {
        let dataset: SqliteDataset<CelebAItemRaw> =
            HuggingfaceDatasetLoader::new("tglcourse/CelebA-faces-cropped-128")
                .with_base_dir("data/celeba")
                .dataset(split)
                .unwrap();
        let map = BytesToImage { quantize_factor: 2 };
        let dataset = MapperDataset::new(dataset, map.clone());

        CelebADataset {
            dataset,
            transformation: map,
        }
    }
}

impl Dataset<CelebAItem> for CelebADataset {
    /// Returns the decoded [`CelebAItem`] at the given `index`, or `None` if
    /// `index` is out of bounds.
    fn get(&self, index: usize) -> Option<CelebAItem> {
        self.dataset.get(index)
    }

    /// Returns the total number of samples in this split.
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// [`Mapper`] that decodes a PNG byte buffer into a channels-first pixel array.
///
/// Applied lazily by [`MapperDataset`] when a dataset item is first accessed.
/// The `quantize_factor` field is reserved for future use — pixel values are
/// currently stored at full `u8` precision (see note in [`BytesToImage::map`]).
#[derive(Clone)]
struct BytesToImage {
    /// Intended divisor for reducing pixel depth (e.g. `2` → 128 levels).
    // TODO : this field is not yet applied during mapping; pixels are
    // stored at their original `u8` intensity.  See [`BytesToImage::map`].
    quantize_factor: u8,
}

impl Mapper<CelebAItemRaw, CelebAItem> for BytesToImage {
    /// Decodes a PNG-encoded [`CelebAItemRaw`] into a [`CelebAItem`].
    ///
    /// # Panics
    /// Panics if the PNG is corrupt or if the decoded byte count does not
    /// equal `WIDTH * HEIGHT * CHANNELS` (checked with [`debug_assert_eq!`]).
    // Steps
    //  1. Wraps `image_bytes` in a [`Cursor`] and feeds it to [`PngDecoder`].
    //  2. Reads the full image into a flat `Vec<u8>` in interleaved RGB order
    //     (`R₀G₀B₀ R₁G₁B₁ …`).
    //  3. Re-indexes each byte into the CHW array
    //     `image[channel][row][col]`.
    fn map(&self, item: &CelebAItemRaw) -> CelebAItem {
        let decoder = PngDecoder::new(Cursor::new(&item.image_bytes)).unwrap();
        let mut img: Vec<u8> = vec![0; decoder.total_bytes() as usize];
        decoder.read_image(&mut img).unwrap();

        debug_assert_eq!(img.len(), WIDTH * HEIGHT * CHANNELS);

        let mut image_array = [[[0u8; WIDTH]; HEIGHT]; CHANNELS];
        for (i, pixel) in img.iter().enumerate() {
            let color = i % CHANNELS;
            let x = (i / CHANNELS) % WIDTH;
            let y = (i / CHANNELS) / HEIGHT;

            // TODO: apply quantize_factor here (currently unused).
            let _quantized_pixel = self.quantize_factor;
            image_array[color][y][x] = *pixel;
        }

        CelebAItem { image: image_array }
    }
}

/// Raw dataset record as deserialized from the SQLite backing store.
///
/// The `image_bytes` field contains the raw PNG-encoded bytes for a single
/// celebrity face, exactly as stored by the HuggingFace dataset loader.
#[derive(Deserialize, Debug, Clone)]
struct CelebAItemRaw {
    /// PNG-encoded bytes of a single celebrity face image.
    pub image_bytes: Vec<u8>,
}

/// A single decoded and channel-separated image sample.
///
/// Pixel values are stored in **channels-first** (CHW) layout:
#[derive(Debug, Clone)]
pub struct CelebAItem {
    /// Pixel intensities laid out as `[C, H, W]` — channels × height × width.
    pub image: [[[u8; WIDTH]; HEIGHT]; CHANNELS],
}
