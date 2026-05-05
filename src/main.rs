//! CLI entry: `glow train` / `glow sample`.

use std::path::PathBuf;

use burn::backend::libtorch::LibTorchDevice;
use clap::{Parser, Subcommand, ValueEnum};

use glow_flow::models::flow::CouplingType;
use glow_flow::sample_run::run_sampling;
use glow_flow::train_run::run_training;
use glow_flow::training_config::TrainingConfig;

#[derive(Parser)]
#[command(
    name = "glow",
    version,
    about = "Glow-style normalizing flow for images (Burn + LibTorch).",
    long_about = "Subcommands implement training on CelebA and sampling from a saved checkpoint.\n\
\n\
Training reads a TOML config (see `train.toml` in the repo) with tables `[model]`, `[optimizer]`, \
`[data]`, and `[run]`. Only a few fields can be overridden from the CLI; everything else stays in the file.\n\
\n\
Examples:\n\
  glow train --config train.toml\n\
  glow train --config train.toml --device cpu --batch-size 8\n\
  glow sample --checkpoint checkpoints/best --output-dir out --num-samples 16"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train Glow on CelebA 128×128 (LibTorch; `--device cuda` by default).
    ///
    /// Effective batch size is `batch_size * grad_accum_steps` from `[data]` in the TOML.
    /// Use `--batch-size` to override only the per-step micro-batch size.
    Train(TrainArgs),
    /// Decode a saved checkpoint: z ~ N(0,I) → inverse Glow → dequantize → PNG grid.
    Sample(SampleArgs),
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum DeviceArg {
    Cuda,
    Cpu,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum CouplingArg {
    /// y_b = s · x_b + t (Glow paper default; expressive, scaled-sigmoid bounded).
    Affine,
    /// y_b = x_b + t (NICE/RealNVP; volume-preserving, more invertible-friendly).
    Additive,
}

impl From<CouplingArg> for CouplingType {
    fn from(value: CouplingArg) -> Self {
        match value {
            CouplingArg::Affine => CouplingType::Affine,
            CouplingArg::Additive => CouplingType::Additive,
        }
    }
}

#[derive(Parser)]
struct TrainArgs {
    /// Path to training TOML (`[model]`, `[optimizer]`, `[data]`, `[run]`).
    ///
    /// If omitted, embedded defaults from `TrainingConfig` are used (same as an empty merge).
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    /// LibTorch device: CUDA GPU 0, or CPU.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cuda)]
    device: DeviceArg,
    /// Resume weights from a checkpoint **base path** (no `.bin`); reads `<base>.meta.json` for step/epoch.
    #[arg(long, value_name = "BASE")]
    resume: Option<PathBuf>,
    /// Override `[data].batch_size` (micro-batch per forward); pair with `grad_accum_steps` in TOML for effective batch.
    #[arg(long, value_name = "N")]
    batch_size: Option<usize>,
    /// Override `[run].checkpoint_dir`. Files are namespaced per coupling type, so
    /// the actual on-disk layout becomes `<DIR>/{affine,additive}/{best,latest,step_*}`.
    #[arg(long, value_name = "DIR")]
    checkpoint_dir: Option<PathBuf>,
    /// Override `[run].max_epochs`.
    #[arg(long, value_name = "N")]
    max_epochs: Option<usize>,
    /// Override `[optimizer].learning_rate`.
    ///
    /// On resume this is the **only** way to override the LR persisted in
    /// `<checkpoint>.meta.json` — editing the TOML alone has no effect because
    /// the meta beats the config (precedence: CLI > meta > config). Does not
    /// auto-scale when you change batch or accumulation.
    #[arg(long, value_name = "LR")]
    learning_rate: Option<f64>,
    /// Override `[model].coupling_type`. `affine` uses scaled-sigmoid affine coupling
    /// (Glow default); `additive` uses NICE-style coupling (`y_b = x_b + t`,
    /// log_det = 0). Additive checkpoints land under a separate sub-directory and are
    /// **not** weight-compatible with affine checkpoints (different conv-block widths).
    #[arg(long, value_enum, value_name = "KIND")]
    coupling_type: Option<CouplingArg>,
}

#[derive(Parser)]
struct SampleArgs {
    /// Checkpoint **base path** without `.bin` (same path passed to `save_file` when training).
    #[arg(long, value_name = "BASE")]
    checkpoint: PathBuf,
    /// Directory for the output PNG grid.
    #[arg(long, default_value = "samples", value_name = "DIR")]
    output_dir: PathBuf,
    /// Number of images in the grid (layout is roughly square).
    #[arg(long, default_value_t = 16, value_name = "N")]
    num_samples: usize,
    /// LibTorch device for loading weights and running inverse.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cuda)]
    device: DeviceArg,
    /// Scale factor on latent standard deviation before inverse (1.0 = model default).
    #[arg(long, default_value_t = 1.0, value_name = "T")]
    temperature: f32,
}

fn resolve_device(d: DeviceArg) -> LibTorchDevice {
    match d {
        DeviceArg::Cuda => LibTorchDevice::Cuda(0),
        DeviceArg::Cpu => LibTorchDevice::Cpu,
    }
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(args) => {
            let mut cfg = if let Some(ref p) = args.config {
                TrainingConfig::from_toml_path(p)?
            } else {
                TrainingConfig::default()
            };
            if let Some(bs) = args.batch_size {
                cfg.data.batch_size = bs;
            }
            if let Some(dir) = args.checkpoint_dir {
                cfg.run.checkpoint_dir = dir.to_string_lossy().into_owned();
            }
            if let Some(e) = args.max_epochs {
                cfg.run.max_epochs = e;
            }
            if let Some(ct) = args.coupling_type {
                cfg.model.coupling_type = ct.into();
            }
            // Don't pre-apply --learning-rate into `cfg`: `run_training` resolves the
            // effective LR with the right precedence (CLI > checkpoint meta > config),
            // which depends on knowing whether the user explicitly passed --learning-rate.
            run_training(
                cfg,
                resolve_device(args.device),
                args.resume,
                args.learning_rate,
            )
        }
        Commands::Sample(args) => run_sampling(
            &args.checkpoint,
            &args.output_dir,
            args.num_samples,
            args.temperature,
            resolve_device(args.device),
        ),
    }
}
