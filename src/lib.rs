//! Glow normalising flow on the [Burn](https://burn.dev) ML framework.
//!
//! This crate exposes the model and its building blocks as a library; the
//! training loop, CLI, dataset loaders, and live-visualisation are all opt-in
//! via Cargo features so a downstream `cargo add glow` (or git dependency)
//! pulls only what's needed for inference.
//!
//! # Quick start
//!
//! ```ignore
//! use glow::prelude::*;
//! use burn::backend::NdArray;
//! use burn::tensor::Tensor;
//!
//! type B = NdArray;
//! let device = Default::default();
//!
//! // 3 colour channels, 4 levels, K=32 steps per level.
//! let model: Glow<B> = GlowConfig::new(3, 4, 32).init(&device);
//! let dq = DequantizeConfig::new().init();
//!
//! // x_u8: [N, 3, H, W] in [0, 255]
//! # let x_u8: Tensor<B, 4> = unimplemented!();
//! let (x, ld_dq) = dq.forward(x_u8);
//! let (zs, ld_flow) = model.forward(x);
//! let nll = (-(log_p_z(&zs) + ld_flow + ld_dq)).mean();
//! ```
//!
//! # Cargo features
//!
//! | Feature    | Pulls in                                        | Use when                                  |
//! |------------|-------------------------------------------------|-------------------------------------------|
//! | *(none)*   | `burn` (autodiff), `serde`                      | inference / custom training loop          |
//! | `dataset`  | `image`, `ndarray`, `burn/dataset`              | CelebA / bouncing-ball loaders            |
//! | `training` | `dataset` + `toml` + `serde_json` + `burn/tch`  | the bundled training loop, LibTorch       |
//! | `cli`      | `training` + `clap`                             | the `glow` binary (`train` / `sample`)    |
//! | `rerun`    | `training` + `rerun`                            | live training dashboards                  |
//! | `fd_reg`   | nothing extra                                   | finite-differences regulariser            |
//!
//! # Public API
//!
//! See [`prelude`] for the canonical inference surface. The full module tree
//! ([`models::flow`], [`models::flow::actnorm`], etc.) is also reachable for
//! consumers that want to compose individual flow layers.

pub mod models;

#[cfg(feature = "dataset")]
pub mod dataset;

#[cfg(feature = "training")]
pub mod training_config;
#[cfg(feature = "training")]
pub mod train_run;
#[cfg(feature = "training")]
pub mod sample_run;

/// Canonical inference API.
///
/// `use glow::prelude::*;` brings in everything you need to build a Glow model,
/// run forward / inverse, and compute log-likelihoods.
pub mod prelude {
    pub use crate::models::flow::{
        log_likelihood, log_p_z, log_prob_pixels, CouplingType, Dequantize,
        DequantizeConfig, Glow, GlowConfig, TriangularInverse,
    };
}
