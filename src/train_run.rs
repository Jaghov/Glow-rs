//! CUDA training loop: CelebA → dequantize → Glow NLL, checkpoints, validation.

use std::path::{Path, PathBuf};

use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::backend::Autodiff;
use burn::data::dataloader::DataLoaderBuilder;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion, Module};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::Tensor;

use crate::dataset::celeba::{CelebABatcher, CelebADataset};
use crate::models::flow::{Dequantize, DequantizeConfig, Glow, TriangularInverse};
use crate::training_config::{SaveOptim, TrainingConfig};

type TrainBackend = Autodiff<LibTorch>;

/// Optimizer state for the AdamW optimiser used during training. The adaptor record
/// holds per-parameter Adam moments (`m`, `v`, step counters) keyed by `ParamId`, so
/// the model and optimiser checkpoints must be loaded against architecturally identical
/// modules — a code change that adds/removes parameters invalidates the optim record.
type TrainOptim = OptimizerAdaptor<AdamW, Glow<TrainBackend>, TrainBackend>;
type TrainOptimRecord = <TrainOptim as Optimizer<Glow<TrainBackend>, TrainBackend>>::Record;

/// Provenance of the effective learning rate used by the training loop.
/// Logged once at startup so a resumed run is unambiguous.
#[derive(Debug, Clone, Copy)]
enum LrSource {
    /// `--learning-rate` was passed on the CLI.
    Cli,
    /// Loaded from `<base>.meta.json::learning_rate` on resume.
    Meta,
    /// `[optimizer].learning_rate` from the TOML config (default fallback).
    Config,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    pub global_step: u64,
    pub epoch: usize,
    pub best_val_bpd: Option<f32>,
    // model architecture — needed to reconstruct the model for sampling
    pub in_channels: usize,
    pub in_height: usize,
    pub in_width: usize,
    pub num_levels: usize,
    pub num_steps: usize,
    pub hidden_features: usize,
    pub pixel_depth: u32,
    /// Learning rate actually used for `optim.step` at the moment this checkpoint was
    /// written. `Option` for backward compatibility with checkpoints written before this
    /// field was added; on resume, `None` falls back to the value from the TOML config.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    /// Coupling parameterisation used for this run. Stored as a string (`"affine"` or
    /// `"additive"`) for forward compatibility — old checkpoints have `None` and are
    /// assumed to be affine. On resume, a mismatch with the active TOML config is a
    /// hard error: the conv shape differs between the two coupling types.
    #[serde(default)]
    pub coupling_type: Option<String>,
}

impl CheckpointMeta {
    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| e.to_string())
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let s = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, s).map_err(|e| e.to_string())
    }
}

fn make_meta(
    cfg: &TrainingConfig,
    global_step: u64,
    epoch: usize,
    best_val_bpd: Option<f32>,
    learning_rate: f64,
) -> CheckpointMeta {
    CheckpointMeta {
        global_step,
        epoch,
        best_val_bpd,
        in_channels: cfg.model.in_channels,
        in_height: cfg.model.in_height,
        in_width: cfg.model.in_width,
        num_levels: cfg.model.num_levels,
        num_steps: cfg.model.num_steps,
        hidden_features: cfg.model.hidden_features,
        pixel_depth: cfg.model.pixel_depth,
        learning_rate: Some(learning_rate),
        coupling_type: Some(coupling_type_to_str(cfg.model.coupling_type).to_string()),
    }
}

fn coupling_type_to_str(ct: crate::models::flow::CouplingType) -> &'static str {
    match ct {
        crate::models::flow::CouplingType::Affine => "affine",
        crate::models::flow::CouplingType::Additive => "additive",
    }
}

fn meta_path_for_model_base(model_base: &Path) -> PathBuf {
    model_base.with_extension("meta.json")
}

/// Sibling base path for the optimiser record (Burn appends `.bin` for `BinFileRecorder`).
/// e.g. `checkpoints/best` → `checkpoints/best.optim`.
fn optim_base_for_model_base(model_base: &Path) -> PathBuf {
    let stem = model_base
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut out = model_base.to_path_buf();
    out.set_file_name(format!("{stem}.optim"));
    out
}

fn bits_per_dim_from_mean_nats(mean_nats: f32, c: usize, h: usize, w: usize) -> f32 {
    let dims = (c * h * w) as f32;
    mean_nats / (dims * std::f32::consts::LN_2)
}

#[cfg(feature = "rerun")]
fn send_train_rerun_video_blueprint(rec: &rerun::RecordingStream) {
    use rerun::blueprint::components::{LoopMode, PlayState};
    use rerun::blueprint::{Blueprint, BlueprintActivation, Spatial2DView, TimePanel, Vertical};

    let view = Spatial2DView::new("Training samples")
        .with_origin("/")
        .with_contents(["/train/sample_grid"]);

    let blueprint = Blueprint::new(Vertical::new([view.into()])).with_time_panel(
        TimePanel::new()
            .with_timeline("train_frames")
            .with_fps(4.0)
            .with_playback_speed(1.0)
            .with_play_state(PlayState::Following)
            .with_loop_mode(LoopMode::All),
    );

    if let Err(e) = blueprint.send(rec, BlueprintActivation::default()) {
        eprintln!("rerun: failed to send viewer blueprint (video layout): {e}");
    }
}

#[cfg(feature = "rerun")]
fn log_training_samples_to_rerun(
    rec: &rerun::RecordingStream,
    train_frame: i64,
    global_step: i64,
    model: &Glow<TrainBackend>,
    dequantize: &crate::models::flow::Dequantize<LibTorch>,
    cfg: &TrainingConfig,
    device: &LibTorchDevice,
) {
    use burn::tensor::{Distribution, Tensor};

    const N: usize = 16;

    let model_eval = model.valid();
    let shapes = crate::sample_run::latent_shapes(
        cfg.model.in_channels,
        cfg.model.in_height,
        cfg.model.in_width,
        cfg.model.num_levels,
        N,
    );
    let zs: Vec<Tensor<LibTorch, 4>> = shapes
        .iter()
        .map(|&shape| Tensor::<LibTorch, 4>::random(shape, Distribution::Normal(0.0, 1.0), device))
        .collect();
    let continuous = model_eval.inverse(zs);
    let pixels = dequantize.inverse(continuous);
    let cols = (N as f64).sqrt().ceil() as usize;
    let Ok(grid) = crate::sample_run::rgb_grid_hwc_u8_from_nchw(pixels, cols) else {
        return;
    };
    let Ok(image) = rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGB, grid) else {
        return;
    };
    // Dense `train_frames` index → even playback in the time panel (video-like scrub/play).
    // `step` keeps the real optimizer step for reference on a second timeline.
    rec.set_timepoint(rerun::TimePoint::from([
        ("train_frames", rerun::TimeCell::from_sequence(train_frame)),
        ("step", rerun::TimeCell::from_sequence(global_step)),
    ]));
    let _ = rec.log("train/sample_grid", &image);
}

/// Round-trip diagnostic report in raw pixel space (`[0, 255]` floats).
struct InvertReport {
    /// Mean squared error of `pixels - inverse(forward(pixels))` across all elements / batches.
    mean_mse: f32,
    /// 99th percentile of per-element absolute error; surfaces tail-pixel outliers that the
    /// mean smooths over (a single rogue dimension can balloon `mean_mse` on its own).
    p99_abs: f32,
}

/// Glow round-trip on each dequantized batch, then back to pixels; aggregates mean MSE and
/// P99 absolute error over **all elements of all batches**. Reusing the same `batches` slice
/// across calls makes the metric comparable step-to-step.
fn round_trip_pixel_report<B: Backend + TriangularInverse>(
    model: &Glow<B>,
    dq: &Dequantize<B>,
    batches: &[Tensor<B, 4>],
) -> InvertReport {
    let mut sum_sq = 0.0_f64;
    let mut all_abs: Vec<f32> = Vec::new();

    for pixels in batches {
        let y = dq.forward(pixels.clone());
        let (zs, _) = model.forward(y);
        let y_hat = model.inverse(zs);
        let pixels_hat = dq.inverse(y_hat);
        let diff: Vec<f32> = (pixels.clone() - pixels_hat)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();
        for v in &diff {
            let d = *v as f64;
            sum_sq += d * d;
            all_abs.push(v.abs());
        }
    }

    if all_abs.is_empty() {
        return InvertReport {
            mean_mse: f32::NAN,
            p99_abs: f32::NAN,
        };
    }

    let mean_mse = (sum_sq / all_abs.len() as f64) as f32;
    // Partial sort would suffice but element count is bounded (a couple of `[B, 3, 128, 128]`
    // batches → <300k floats), so a single sort is simple and fast enough off the hot path.
    all_abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (((all_abs.len() - 1) as f64) * 0.99).round() as usize;
    let p99_abs = all_abs[idx];

    InvertReport { mean_mse, p99_abs }
}

/// Build a fixed list of validation pixel batches reused on every invert check, so the
/// reported metric is comparable across optimizer steps. Empty when `invert_check_batches`
/// is `0` (caller should then skip the check entirely).
fn build_invert_check_batches(
    cfg: &TrainingConfig,
    device: &LibTorchDevice,
) -> Vec<Tensor<LibTorch, 4>> {
    let n = cfg.run.invert_check_batches;
    if n == 0 {
        return Vec::new();
    }
    let loader = DataLoaderBuilder::<LibTorch, _, _>::new(CelebABatcher)
        .batch_size(cfg.data.batch_size)
        .shuffle(0)
        .num_workers(cfg.data.num_workers)
        .set_device(device.clone())
        .build(CelebADataset::test());
    loader
        .iter()
        .take(n)
        .map(|batch| batch.images.float())
        .collect()
}

/// Print compact per-step invert diagnostics: largest |coupling log_s| (the leading cause
/// of f32 cancellation in `inverse`), plus ActNorm and InvConv parameter ranges. One forward
/// over `y`; only call when invertibility metrics already justify the cost.
fn print_invert_diagnostics<B: Backend + TriangularInverse>(
    global_step: u64,
    model: &Glow<B>,
    y: Tensor<B, 4>,
) {
    let rows = model.collect_invert_diagnostics(y);
    let mut worst_abs_log_s = f32::NEG_INFINITY;
    let mut worst_loc = (0usize, 0usize);
    let mut an_w_lo = f32::INFINITY;
    let mut an_w_hi = f32::NEG_INFINITY;
    let mut ic_s_lo = f32::INFINITY;
    let mut ic_s_hi = f32::NEG_INFINITY;
    let mut worst_shift = f32::NEG_INFINITY;
    let mut worst_shift_loc = (0usize, 0usize);
    for r in &rows {
        let cp_min: f32 = r.diag.coupling_log_s_min.clone().into_scalar().elem();
        let cp_max: f32 = r.diag.coupling_log_s_max.clone().into_scalar().elem();
        let max_abs = cp_min.abs().max(cp_max.abs());
        if max_abs > worst_abs_log_s {
            worst_abs_log_s = max_abs;
            worst_loc = (r.level, r.step_idx);
        }
        let shift_abs: f32 = r.diag.coupling_shift_abs_max.clone().into_scalar().elem();
        if shift_abs > worst_shift {
            worst_shift = shift_abs;
            worst_shift_loc = (r.level, r.step_idx);
        }
        an_w_lo = an_w_lo.min(
            r.diag
                .actnorm_weight_min
                .clone()
                .into_scalar()
                .elem::<f32>(),
        );
        an_w_hi = an_w_hi.max(
            r.diag
                .actnorm_weight_max
                .clone()
                .into_scalar()
                .elem::<f32>(),
        );
        ic_s_lo = ic_s_lo.min(r.diag.invconv_log_s_min.clone().into_scalar().elem::<f32>());
        ic_s_hi = ic_s_hi.max(r.diag.invconv_log_s_max.clone().into_scalar().elem::<f32>());
    }
    println!(
        "step {global_step} invert_diag: \
         coupling_max|log_s|={worst_abs_log_s:.3} @ (level={}, step={}); \
         coupling_max|shift|={worst_shift:.3} @ (level={}, step={}); \
         actnorm_weight=[{an_w_lo:.3},{an_w_hi:.3}]; \
         invconv_log_w_s=[{ic_s_lo:.3},{ic_s_hi:.3}]",
        worst_loc.0, worst_loc.1, worst_shift_loc.0, worst_shift_loc.1,
    );
}

/// Per-sublayer round-trip diagnostic. Emits one line per sublayer plus a summary of which
/// (level, step, kind) first crosses 1e-3 — the threshold above which f32 rounding alone
/// can't explain the drift.
fn print_roundtrip_diag<B: Backend + TriangularInverse>(
    global_step: u64,
    model: &Glow<B>,
    y: Tensor<B, 4>,
) {
    let rows = model.roundtrip_diag(y);
    let mut first_bad: Option<&crate::models::flow::RoundtripRow> = None;
    let mut worst: Option<&crate::models::flow::RoundtripRow> = None;
    for r in &rows {
        if first_bad.is_none() && r.max_abs_err > 1e-3 && r.kind != "block" {
            first_bad = Some(r);
        }
        if worst.map_or(true, |w| r.max_abs_err > w.max_abs_err) {
            worst = Some(r);
        }
        let step_label = if r.step_idx == usize::MAX {
            "*".to_string()
        } else {
            r.step_idx.to_string()
        };
        println!(
            "step {global_step} roundtrip level={} step={} kind={} max_abs_err={:.3e}",
            r.level, step_label, r.kind, r.max_abs_err
        );
    }
    if let Some(r) = first_bad {
        let step_label = if r.step_idx == usize::MAX {
            "*".to_string()
        } else {
            r.step_idx.to_string()
        };
        println!(
            "step {global_step} roundtrip first_bad: level={} step={} kind={} err={:.3e}",
            r.level, step_label, r.kind, r.max_abs_err
        );
    }
    if let Some(r) = worst {
        let step_label = if r.step_idx == usize::MAX {
            "*".to_string()
        } else {
            r.step_idx.to_string()
        };
        println!(
            "step {global_step} roundtrip worst: level={} step={} kind={} err={:.3e}",
            r.level, step_label, r.kind, r.max_abs_err
        );
    }
}

/// Logging, validation, invert check, best/step checkpoints. Returns whether validation ran.
#[allow(clippy::too_many_arguments)]
fn training_hooks_on_optimizer_step(
    global_step: u64,
    epoch: usize,
    train_nats: f32,
    learning_rate: f64,
    x: &Tensor<TrainBackend, 4>,
    model: &Glow<TrainBackend>,
    optim: &TrainOptim,
    dequantize_infer: &crate::models::flow::Dequantize<LibTorch>,
    invert_batches: &[Tensor<LibTorch, 4>],
    cfg: &TrainingConfig,
    device: &LibTorchDevice,
    recorder: &BinFileRecorder<FullPrecisionSettings>,
    checkpoint_dir: &Path,
    best_val_bpd: &mut Option<f32>,
) -> Result<bool, String> {
    let mut ran_val = false;
    // Caller passes NaN when the windowed nats wasn't synced (skipping the host read on
    // non-log steps). Only emit the log line when a real value is available.
    if global_step % cfg.run.log_every as u64 == 0 && train_nats.is_finite() {
        let [_, c, h, w] = x.dims();
        let bpd = bits_per_dim_from_mean_nats(train_nats, c, h, w);
        println!("step {global_step} epoch {epoch} train_nats={train_nats:.4} train_bpd={bpd:.4}");
    }

    if global_step % cfg.run.val_every as u64 == 0 {
        ran_val = true;
        let (mean_nats, mean_bpd) = evaluate(
            model,
            dequantize_infer,
            cfg.data.batch_size,
            cfg.data.num_workers,
            cfg.data.val_max_batches,
            device.clone(),
        )?;
        println!("step {global_step} val_mean_nats={mean_nats:.4} val_mean_bpd={mean_bpd:.4}");

        if !invert_batches.is_empty() {
            let report = round_trip_pixel_report(&model.valid(), dequantize_infer, invert_batches);
            println!(
                "step {global_step} invert_pixel_mse={:.6} p99_abs_err={:.4}",
                report.mean_mse, report.p99_abs
            );
            if cfg.run.invert_diagnostics {
                let y = dequantize_infer.forward(invert_batches[0].clone());
                print_invert_diagnostics(global_step, &model.valid(), y.clone());
                print_roundtrip_diag(global_step, &model.valid(), y);
            }

            let abort = !report.mean_mse.is_finite()
                || cfg
                    .run
                    .invert_abort_tol
                    .is_some_and(|t| report.mean_mse > t);
            if !abort && cfg.run.invert_warn_tol.is_some_and(|t| report.mean_mse > t) {
                println!(
                    "step {global_step} WARN invert_pixel_mse={:.6} > warn_tol={:.6}",
                    report.mean_mse,
                    cfg.run.invert_warn_tol.unwrap()
                );
            }
            if abort {
                let base = checkpoint_dir.join(format!("step_{global_step:06}_diverged"));
                save_checkpoint(
                    model,
                    optim,
                    recorder,
                    &base,
                    make_meta(cfg, global_step, epoch, *best_val_bpd, learning_rate),
                    CheckpointKind::Diverged,
                    cfg.run.save_optim,
                )?;
                return Err(format!(
                    "step {global_step}: invertibility broken (pixel_mse={:.6}, abort_tol={:?}); checkpoint saved to {base:?}",
                    report.mean_mse, cfg.run.invert_abort_tol
                ));
            }
        }

        let improved = best_val_bpd.map_or(true, |b| mean_bpd < b);
        if improved {
            *best_val_bpd = Some(mean_bpd);
            let base = checkpoint_dir.join("best");
            save_checkpoint(
                model,
                optim,
                recorder,
                &base,
                make_meta(cfg, global_step, epoch, *best_val_bpd, learning_rate),
                CheckpointKind::Best,
                cfg.run.save_optim,
            )?;
            println!("saved best checkpoint (bpd={mean_bpd:.4})");
        }
    }

    if global_step % cfg.run.checkpoint_every as u64 == 0 {
        let base = checkpoint_dir.join(format!("step_{global_step:06}"));
        save_checkpoint(
            model,
            optim,
            recorder,
            &base,
            make_meta(cfg, global_step, epoch, *best_val_bpd, learning_rate),
            CheckpointKind::Step,
            cfg.run.save_optim,
        )?;
        println!("saved checkpoint {base:?}");
    }

    Ok(ran_val)
}

/// Run training with merged file config and LibTorch device.
///
/// `cli_lr_override` carries an explicit user-provided learning rate (from `--learning-rate`).
/// On resume, precedence is: CLI override > checkpoint meta `learning_rate` > config value.
/// Without resume, precedence is simply: CLI override > config value.
pub fn run_training(
    cfg: TrainingConfig,
    device: LibTorchDevice,
    resume_base: Option<PathBuf>,
    cli_lr_override: Option<f64>,
) -> Result<(), String> {
    crate::disable_tf32();

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    // Namespace checkpoints by coupling type so affine and additive runs don't
    // overwrite each other in a shared `checkpoint_dir`. The user's TOML value is
    // treated as the parent; actual files land in `<dir>/{affine|additive}/...`.
    // To resume, point `--resume` at the same nested path
    // (e.g. `--resume checkpoints/additive/latest`).
    let coupling_subdir = coupling_type_to_str(cfg.model.coupling_type);
    let checkpoint_dir = PathBuf::from(&cfg.run.checkpoint_dir).join(coupling_subdir);
    std::fs::create_dir_all(&checkpoint_dir).map_err(|e| e.to_string())?;
    println!("checkpoints will be written to {checkpoint_dir:?}");

    let dequant_cfg = DequantizeConfig::new(cfg.model.pixel_depth);
    let dequantize_train = dequant_cfg.init::<TrainBackend>(&device);
    let dequantize_infer = dequant_cfg.init::<LibTorch>(&device);

    let glow_cfg = cfg.glow_config();
    let mut model = glow_cfg.init::<TrainBackend>(&device);

    let (mut global_step, mut start_epoch, mut best_val_bpd, mut resume_weights) =
        (0_u64, 0_usize, None::<f32>, false);
    let mut resumed_meta_lr: Option<f64> = None;

    if let Some(ref base) = resume_base {
        model = model
            .load_file(base, &recorder, &device)
            .map_err(|e| format!("load checkpoint: {e}"))?;
        let meta_path = meta_path_for_model_base(base);
        let meta = CheckpointMeta::load(&meta_path)?;
        global_step = meta.global_step;
        start_epoch = meta.epoch;
        best_val_bpd = meta.best_val_bpd;
        resumed_meta_lr = meta.learning_rate;

        // Coupling-type mismatch produces a confusing record-load error (the conv
        // shape differs between affine and additive). Catch it up-front with a
        // clear message.
        let active_ct = coupling_type_to_str(cfg.model.coupling_type);
        match meta.coupling_type.as_deref() {
            Some(meta_ct) if meta_ct.eq_ignore_ascii_case(active_ct) => {
                // matches — no-op
            }
            Some(meta_ct) => {
                return Err(format!(
                    "checkpoint coupling_type {meta_ct:?} does not match active config {active_ct:?}; \
                     model record shapes are incompatible — train fresh or change the TOML"
                ));
            }
            None => {
                // Old checkpoint predating the additive option: was definitionally
                // affine. Resuming as affine is fine; resuming as additive requires
                // weight surgery the loader can't perform, so abort early.
                if cfg.model.coupling_type == crate::models::flow::CouplingType::Additive {
                    return Err(
                        "checkpoint predates the `coupling_type` field (assumed affine), \
                         but config requests additive coupling — these are not weight-compatible. \
                         Train fresh, or set `model.coupling_type = \"affine\"` to continue."
                            .to_string(),
                    );
                }
                println!(
                    "note: checkpoint predates `coupling_type` field; assuming affine (matches active config)."
                );
            }
        }
        resume_weights = true;
    }

    let mut adam = AdamWConfig::new();
    if let Some(wd) = cfg.optimizer.weight_decay {
        adam = adam.with_weight_decay(wd);
    }
    if let Some(norm) = cfg.optimizer.grad_clip_norm {
        adam = adam.with_grad_clipping(Some(GradientClippingConfig::Norm(norm)));
    }
    let mut optim: TrainOptim = adam.init::<TrainBackend, Glow<TrainBackend>>();

    // Resume optimiser state if the resume checkpoint has a sibling `.optim.bin` file.
    // Older checkpoints without one fall through with a freshly initialised optimiser
    // (Adam moments restart from zero — same as a cold resume).
    if let Some(ref base) = resume_base {
        let optim_base = optim_base_for_model_base(base);
        let optim_file = optim_base.with_extension("bin");
        if optim_file.exists() {
            let record: TrainOptimRecord = recorder
                .load(optim_base.clone(), &device)
                .map_err(|e| format!("load optim: {e}"))?;
            optim = optim.load_record(record);
            println!("resumed optimiser state from {optim_file:?}");
        } else {
            println!("no optimiser checkpoint at {optim_file:?}; AdamW moments reset to zero");
        }
    }

    // Resolve effective learning rate.
    // Precedence: CLI override > checkpoint meta > config (TOML). The decision to let
    // the meta beat the TOML on resume is deliberate: the common case is "I want to
    // keep training from where I left off"; users who want to override the meta have
    // `--learning-rate` for that explicitly (see CLI doc).
    let (learning_rate, lr_source) = match (cli_lr_override, resumed_meta_lr) {
        (Some(lr), _) => (lr, LrSource::Cli),
        (None, Some(lr)) => (lr, LrSource::Meta),
        (None, None) => (cfg.optimizer.learning_rate, LrSource::Config),
    };
    if resume_base.is_some() || cli_lr_override.is_some() {
        let suffix = match lr_source {
            LrSource::Cli => "from --learning-rate (overrides any checkpoint meta)",
            LrSource::Meta => "from checkpoint meta",
            LrSource::Config => "from config (no meta in this checkpoint)",
        };
        println!("learning_rate={learning_rate} {suffix}");
    }

    // `DataLoaderBuilder` defaults to `LibTorchDevice::default()` (CPU) unless we set this;
    // batches must match the model device (e.g. CUDA) or conv and other ops panic.
    let train_loader = DataLoaderBuilder::new(CelebABatcher {})
        .batch_size(cfg.data.batch_size)
        .shuffle(cfg.data.shuffle_seed)
        .num_workers(cfg.data.num_workers)
        .set_device(device.clone())
        .build(CelebADataset::train());

    // Fixed validation batches reused on every invert check so the metric is comparable
    // across optimizer steps. Built once; held in device memory for the rest of training.
    let invert_batches = build_invert_check_batches(&cfg, &device);

    let mut pending_actnorm = !resume_weights;

    #[cfg(feature = "rerun")]
    let rerun_rec = rerun::RecordingStreamBuilder::new("glow-rs-train")
        .spawn()
        .ok();
    #[cfg(feature = "rerun")]
    let mut rerun_train_frame: i64 = 0;
    #[cfg(feature = "rerun")]
    if let Some(ref rec) = rerun_rec {
        send_train_rerun_video_blueprint(rec);
    }

    let accum_steps = cfg.data.grad_accum_steps.max(1);
    let inv_accum = 1.0 / accum_steps as f32;
    let mut grad_accum = GradientsAccumulator::<Glow<TrainBackend>>::new();
    let mut micro_ix = 0usize;
    // On-device running sum of per-micro-batch mean nats; only synced to host when we log.
    let mut nats_window: Option<Tensor<LibTorch, 1>> = None;
    // Same accumulator for the FD reg term (when enabled); summed only over micro-batches
    // where FD actually fired, so the printed value reflects the accumulated contribution.
    #[cfg(feature = "fd_reg")]
    let mut fd_window: Option<Tensor<LibTorch, 1>> = None;
    #[cfg(feature = "fd_reg")]
    let mut fd_window_count: u32 = 0;

    for epoch in start_epoch..cfg.run.max_epochs {
        for batch in train_loader.iter() {
            let x = batch.images.float();

            if pending_actnorm {
                // Init on dequantized values — the same distribution the model will train on.
                // Raw pixels are [0,255]; dequantized output is ~[-0.5, 0.5].
                // Initialising on raw pixels would set wildly wrong scales and bias the log-det
                // by ~H*W*C*log(74) ≈ 200k per sample, causing immediate NaN on the first backward.
                let x_deq = dequantize_train.forward(x.clone());
                model.init_actnorm(x_deq);
                pending_actnorm = false;
            }

            // Resolve FD reg schedule once per micro-batch. Decision is keyed off the
            // *upcoming* optimiser step (`global_step + 1`) so that all micro-batches
            // within an accumulation window share the same FD on/off state.
            #[cfg(feature = "fd_reg")]
            let (fd_lambda_eff, do_fd) = {
                let next_step = global_step + 1;
                let warmup = cfg.regularization.fd_warmup_steps.max(1);
                let warmup_factor = (next_step as f32 / warmup as f32).min(1.0);
                let lambda = cfg.regularization.fd_lambda * warmup_factor;
                let every_n = cfg.regularization.fd_every_n_steps.max(1);
                let do_fd = lambda > 0.0 && (next_step % every_n == 0);
                (lambda, do_fd)
            };

            // Forward pass; the FD branch (when enabled & active) reuses `x_cont` and
            // `zs` to avoid a second `model.forward`. When FD is disabled at compile
            // time we call `log_prob_pixels` directly so the side outputs of
            // `forward_for_training` aren't even allocated as named bindings.
            #[cfg(feature = "fd_reg")]
            let (fd_inputs, logp) = {
                let (x_cont, zs, logp) =
                    forward_for_training(&model, &dequantize_train, x.clone(), true);
                if do_fd {
                    (Some((x_cont, zs)), logp)
                } else {
                    // Drop x_cont and zs explicitly so the only handles held into the
                    // autodiff graph past this line are the ones backward() actually
                    // walks (via `logp`). Functionally equivalent to letting them go
                    // out of scope at end of body, but signals intent.
                    drop((x_cont, zs));
                    (None, logp)
                }
            };
            #[cfg(not(feature = "fd_reg"))]
            let logp = log_prob_pixels(&model, &dequantize_train, x.clone(), true);

            let nats_mean = logp.mean().neg();

            // Detached, inner-backend copy → can be added without touching the autodiff graph
            // and without forcing a host sync each micro-batch.
            let nats_value = nats_mean.clone().inner();
            nats_window = Some(match nats_window.take() {
                None => nats_value,
                Some(prev) => prev + nats_value,
            });

            #[cfg(feature = "fd_reg")]
            let fd_term = if let Some((x_cont, zs)) = fd_inputs {
                use burn::tensor::Distribution;
                let eps = cfg.regularization.fd_epsilon;
                let zs_perturbed: Vec<_> = zs
                    .into_iter()
                    .map(|z| {
                        let eta = z.clone().random_like(Distribution::Normal(0.0, 1.0));
                        z + eta.mul_scalar(eps)
                    })
                    .collect();
                let x_tilde = model.inverse_autodiff(zs_perturbed);
                let diff = x_tilde - x_cont;
                // mean ||x̃ − x||² · (λ / ε²); absorbing 1/ε² keeps λ on a Hutchinson-like scale.
                let scale = fd_lambda_eff / (eps * eps);
                let term = (diff.clone() * diff).mean().mul_scalar(scale);
                let term_value = term.clone().inner();
                fd_window = Some(match fd_window.take() {
                    None => term_value,
                    Some(prev) => prev + term_value,
                });
                fd_window_count += 1;
                Some(term)
            } else {
                None
            };

            #[cfg(feature = "fd_reg")]
            let total_loss = match fd_term {
                Some(t) => nats_mean.clone() + t,
                None => nats_mean.clone(),
            };
            #[cfg(not(feature = "fd_reg"))]
            let total_loss = nats_mean.clone();

            let loss = total_loss.mul_scalar(inv_accum);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            grad_accum.accumulate(&model, grads);

            micro_ix += 1;
            if micro_ix < accum_steps {
                continue;
            }

            micro_ix = 0;
            global_step += 1;
            let accumulated = grad_accum.grads();
            model = optim.step(learning_rate, model, accumulated);

            // One host sync per optimizer step, and only when we actually need to log it.
            let needs_log = global_step % cfg.run.log_every as u64 == 0;
            let train_nats = if needs_log {
                let sum = nats_window.take().expect("non-empty window after step");
                sum.into_scalar().elem::<f32>() * inv_accum
            } else {
                nats_window = None;
                f32::NAN
            };

            #[cfg(feature = "fd_reg")]
            {
                // The FD window aggregates `fd_window_count` per-microbatch FD terms
                // until the next log step. Three cases:
                //   1. needs_log: drain the window (sync to host once, print, reset).
                //   2. !needs_log && do_fd: a fresh FD term was just appended this
                //      step; keep accumulating, do not reset count.
                //   3. !needs_log && !do_fd: no FD term this step. The window is still
                //      valid (count matches sum) — do nothing.
                // The earlier "reset count when window is None" branch existed only to
                // recover from a programming mistake that can no longer happen with
                // the current code path; dropped for clarity.
                if needs_log {
                    if let Some(sum) = fd_window.take() {
                        let count = fd_window_count.max(1) as f32;
                        let mean = sum.into_scalar().elem::<f32>() / count;
                        println!(
                            "step {global_step} fd_term={mean:.6} fd_lambda={fd_lambda_eff:.6} fd_fired={fd_window_count}/{accum_steps}"
                        );
                    }
                    fd_window_count = 0;
                }
            }

            #[cfg_attr(not(feature = "rerun"), allow(unused_variables))]
            let ran_val = training_hooks_on_optimizer_step(
                global_step,
                epoch,
                train_nats,
                learning_rate,
                &x,
                &model,
                &optim,
                &dequantize_infer,
                &invert_batches,
                &cfg,
                &device,
                &recorder,
                &checkpoint_dir,
                &mut best_val_bpd,
            )?;
            #[cfg(feature = "rerun")]
            if ran_val {
                if let Some(ref rec) = rerun_rec {
                    log_training_samples_to_rerun(
                        rec,
                        rerun_train_frame,
                        global_step as i64,
                        &model,
                        &dequantize_infer,
                        &cfg,
                        &device,
                    );
                    rerun_train_frame += 1;
                }
            }
        }

        // Drop any partial accumulation window at epoch end. With well-tuned `accum_steps`
        // this is at most `accum_steps - 1` micro-batches per epoch — negligible vs. the
        // complexity of correctly rescaling, re-syncing nats, and re-firing hooks.
        if micro_ix > 0 {
            let _ = grad_accum.grads();
            micro_ix = 0;
            nats_window = None;
            #[cfg(feature = "fd_reg")]
            {
                fd_window = None;
                fd_window_count = 0;
            }
        }

        let base = checkpoint_dir.join("latest");
        save_checkpoint(
            &model,
            &optim,
            &recorder,
            &base,
            make_meta(&cfg, global_step, epoch + 1, best_val_bpd, learning_rate),
            CheckpointKind::Latest,
            cfg.run.save_optim,
        )?;
    }

    Ok(())
}

/// Tag identifying which checkpoint slot is being written. Used to apply the
/// `[run].save_optim` knob: only matching slots get the AdamW sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointKind {
    /// `best.{bin,meta.json}` — written when val bpd improves.
    Best,
    /// `latest.{bin,meta.json}` — written at end of every epoch.
    Latest,
    /// `step_NNNNNN.{bin,meta.json}` — periodic archive at `[run].checkpoint_every`.
    Step,
    /// `step_NNNNNN_diverged.{bin,meta.json}` — emergency save before abort.
    Diverged,
}

fn should_save_optim(kind: CheckpointKind, policy: SaveOptim) -> bool {
    match (kind, policy) {
        (CheckpointKind::Diverged, _) => true, // always include for post-mortem resume
        (_, SaveOptim::All) => true,
        (CheckpointKind::Latest, SaveOptim::Latest) => true,
        (CheckpointKind::Latest, SaveOptim::Best) => true,
        (CheckpointKind::Best, SaveOptim::Best) => true,
        _ => false,
    }
}

fn save_checkpoint(
    model: &Glow<TrainBackend>,
    optim: &TrainOptim,
    recorder: &BinFileRecorder<FullPrecisionSettings>,
    base: &Path,
    meta: CheckpointMeta,
    kind: CheckpointKind,
    save_optim_policy: SaveOptim,
) -> Result<(), String> {
    model
        .clone()
        .save_file(base, recorder)
        .map_err(|e| format!("save checkpoint: {e}"))?;
    // AdamW state (per-parameter moments) — required to resume training without
    // resetting optimiser dynamics. Sibling file: `<base>.optim.bin`. Only emitted
    // when the policy in `[run].save_optim` covers this checkpoint kind.
    if should_save_optim(kind, save_optim_policy) {
        let optim_base = optim_base_for_model_base(base);
        recorder
            .record(optim.to_record(), optim_base)
            .map_err(|e| format!("save optim: {e}"))?;
    }
    meta.save_to_file(&meta_path_for_model_base(base))
        .map_err(|e| format!("save meta: {e}"))?;
    Ok(())
}

fn evaluate(
    model: &Glow<TrainBackend>,
    dequantize: &crate::models::flow::Dequantize<LibTorch>,
    batch_size: usize,
    num_workers: usize,
    max_batches: Option<usize>,
    device: LibTorchDevice,
) -> Result<(f32, f32), String> {
    let model_eval = model.valid();
    let val_loader = DataLoaderBuilder::<TrainBackend, _, _>::new(CelebABatcher)
        .batch_size(batch_size)
        .shuffle(0)
        .num_workers(num_workers)
        .set_device(device)
        .build(CelebADataset::test());

    let mut sum_nats = 0_f64;
    let mut n_img = 0_u64;
    let mut c_hw = (3usize, 128usize, 128usize);

    for (i, batch) in val_loader.iter().enumerate() {
        if max_batches.is_some_and(|m| i >= m) {
            break;
        }
        let x = batch.images.float();
        let d = x.dims();
        c_hw = (d[1], d[2], d[3]);
        let x_inner = x.inner();
        let logp = log_prob_pixels(&model_eval, dequantize, x_inner, false);
        let b = logp.dims()[0];
        let batch_sum: f32 = (-logp).sum().into_scalar().elem();
        sum_nats += f64::from(batch_sum);
        n_img += b as u64;
    }

    if n_img == 0 {
        return Err("validation: zero images".to_string());
    }

    let mean_nats = (sum_nats / n_img as f64) as f32;
    let (c, h, w) = c_hw;
    let mean_bpd = bits_per_dim_from_mean_nats(mean_nats, c, h, w);
    Ok((mean_nats, mean_bpd))
}
