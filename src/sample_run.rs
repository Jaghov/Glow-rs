//! Sampling: load a trained checkpoint, draw z ~ N(0,I), invert through Glow, save image grid.

use std::path::Path;

use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{Distribution, Tensor};
use ndarray::Array3;

use crate::models::flow::{DequantizeConfig, GlowConfig};
use crate::train_run::CheckpointMeta;

/// Returns the latent tensor shape at each Glow level for a given input spatial size.
///
/// - Levels `0..L-1` (split levels): `[n, C·2^(l+1), H/2^(l+1), W/2^(l+1)]`
/// - Level `L-1` (last):             `[n, C·2^(l+2), H/2^(l+1), W/2^(l+1)]`
pub fn latent_shapes(
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    num_levels: usize,
    n: usize,
) -> Vec<[usize; 4]> {
    (0..num_levels)
        .map(|l| {
            let h = in_height >> (l + 1);
            let w = in_width >> (l + 1);
            let c_shift = if l < num_levels - 1 { l + 1 } else { l + 2 };
            let c = in_channels << c_shift;
            [n, c, h, w]
        })
        .collect()
}

/// `[N, 3, H, W]` float pixels in `[0, 255]` → row-major `HWC` `u8` mosaic with `cols` tiles per row.
pub(crate) fn rgb_grid_hwc_u8_from_nchw(
    pixels: Tensor<LibTorch, 4>,
    cols: usize,
) -> Result<Array3<u8>, String> {
    let [n, c, h, w] = pixels.dims();
    if c != 3 {
        return Err(format!("rgb_grid: expected 3 channels, got {c}"));
    }

    let data = pixels
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| e.to_string())?;

    let rows = n.div_ceil(cols);
    let grid_w = cols * w;
    let grid_h = rows * h;
    let mut buf = vec![0u8; grid_h * grid_w * 3];

    for (idx, img) in data.chunks(c * h * w).enumerate() {
        let col = idx % cols;
        let row = idx / cols;
        for ih in 0..h {
            for iw in 0..w {
                let r = img[ih * w + iw].clamp(0.0, 255.0) as u8;
                let g = img[h * w + ih * w + iw].clamp(0.0, 255.0) as u8;
                let b = img[2 * h * w + ih * w + iw].clamp(0.0, 255.0) as u8;
                let gy = row * h + ih;
                let gx = col * w + iw;
                let o = (gy * grid_w + gx) * 3;
                buf[o] = r;
                buf[o + 1] = g;
                buf[o + 2] = b;
            }
        }
    }

    Array3::from_shape_vec((grid_h, grid_w, 3), buf).map_err(|e| e.to_string())
}

/// Save an `HWC` `u8` RGB mosaic as PNG.
fn save_rgb_grid_png(arr: &Array3<u8>, path: &Path) -> Result<(), String> {
    let (grid_h, grid_w, _) = arr.dim();
    let raw = arr
        .as_slice()
        .ok_or_else(|| "save_rgb_grid_png: non-contiguous buffer".to_string())?
        .to_vec();
    let grid = image::RgbImage::from_raw(grid_w as u32, grid_h as u32, raw).ok_or_else(|| {
        format!(
            "save_rgb_grid_png: invalid buffer for {}×{}",
            grid_w, grid_h
        )
    })?;

    std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).map_err(|e| e.to_string())?;
    grid.save(path).map_err(|e| e.to_string())
}

/// Load a trained checkpoint and generate `num_samples` images, saved as a PNG grid.
pub fn run_sampling(
    checkpoint: &Path,
    output_dir: &Path,
    num_samples: usize,
    temperature: f32,
    device: LibTorchDevice,
) -> Result<(), String> {
    // ── load metadata ─────────────────────────────────────────────────────────
    let meta_path = checkpoint.with_extension("meta.json");
    let meta = CheckpointMeta::load(&meta_path)?;

    println!(
        "Loaded checkpoint (step {}, epoch {})",
        meta.global_step, meta.epoch
    );

    // ── build model ───────────────────────────────────────────────────────────
    let glow_cfg = GlowConfig::new(meta.in_channels)
        .with_num_levels(meta.num_levels)
        .with_num_steps(meta.num_steps)
        .with_hidden_features(meta.hidden_features);

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = glow_cfg
        .init::<LibTorch>(&device)
        .load_file(checkpoint, &recorder, &device)
        .map_err(|e| format!("load model weights: {e}"))?;

    let dq = DequantizeConfig::new(meta.pixel_depth).init::<LibTorch>(&device);

    // ── generate latent samples ───────────────────────────────────────────────
    let shapes = latent_shapes(
        meta.in_channels,
        meta.in_height,
        meta.in_width,
        meta.num_levels,
        num_samples,
    );
    let zs: Vec<Tensor<LibTorch, 4>> = shapes
        .iter()
        .map(|&shape| {
            Tensor::<LibTorch, 4>::random(
                shape,
                Distribution::Normal(0.0, temperature as f64),
                &device,
            )
        })
        .collect();

    println!("Sampling {num_samples} images at temperature {temperature:.2} …");

    // ── invert through model ──────────────────────────────────────────────────
    let continuous = model.inverse(zs);
    let pixels = dq.inverse(continuous);

    // ── save grid ─────────────────────────────────────────────────────────────
    let cols = (num_samples as f64).sqrt().ceil() as usize;
    let out_path = output_dir.join("samples.png");
    let grid_arr = rgb_grid_hwc_u8_from_nchw(pixels, cols)?;
    save_rgb_grid_png(&grid_arr, &out_path)?;
    println!("Saved image grid → {out_path:?}");

    #[cfg(feature = "rerun")]
    {
        use rerun::RecordingStreamBuilder;
        if let Ok(rec) = RecordingStreamBuilder::new("glow-rs-sample").spawn() {
            rec.set_time_sequence("sample", 0_i64);
            if let Ok(img) =
                rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, grid_arr)
            {
                let _ = rec.log("sample/grid", &img);
            }
            let _ = rec.log(
                "sample/path",
                &rerun::TextLog::new(out_path.display().to_string()),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_shapes_two_levels() {
        // C=3, H=W=128, L=2, n=4
        let shapes = latent_shapes(3, 128, 128, 2, 4);
        assert_eq!(shapes[0], [4, 6, 64, 64]); // split level
        assert_eq!(shapes[1], [4, 24, 32, 32]); // last level
    }

    #[test]
    fn latent_shapes_three_levels() {
        // C=3, H=W=128, L=3, n=1
        let shapes = latent_shapes(3, 128, 128, 3, 1);
        assert_eq!(shapes[0], [1, 6, 64, 64]);
        assert_eq!(shapes[1], [1, 12, 32, 32]);
        assert_eq!(shapes[2], [1, 48, 16, 16]);
    }
}
