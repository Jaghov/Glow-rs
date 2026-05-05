//! TOML training configuration (model, optimizer, data, run schedule).
//!
//! Subsections use Burn [`Config`] for defaults and `::new()`. The file loader merges optional
//! TOML tables with those defaults so keys can be omitted (Burn’s `Config` serde layer does not).

use std::path::Path;

use burn::config::Config;
use serde::Deserialize;

use crate::models::flow::{CouplingType, GlowConfig};

/// Fully resolved training configuration (ready for `train_run`).
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub model: ModelSettings,
    pub optimizer: OptimizerSettings,
    pub data: DataSettings,
    pub run: RunSettings,
    /// Finite-differences regularizer (Behrmann/Vicol "Exploding Inverses"). Penalises the
    /// inverse Lipschitz of the full Glow flow during training. Available only with the
    /// `fd_reg` cargo feature; absent on default builds.
    #[cfg(feature = "fd_reg")]
    pub regularization: RegularizationSettings,
}

/// Finite-differences regularizer parameters. Only present with `fd_reg` feature.
#[cfg(feature = "fd_reg")]
#[derive(Config, Debug)]
pub struct RegularizationSettings {
    /// Target FD coefficient applied after warmup. `0.0` disables the FD term even when the
    /// feature is compiled in (useful for A/B comparisons against a non-FD baseline without
    /// rebuilding).
    #[config(default = "1.0_f32")]
    pub fd_lambda: f32,
    /// Number of optimiser steps over which `lambda` ramps linearly from `0` to `fd_lambda`.
    /// Without warmup the FD term dominates the loss on a freshly-initialised model and can
    /// destabilise training before the NLL has a chance to take effect.
    #[config(default = "5000")]
    pub fd_warmup_steps: u64,
    /// Perturbation magnitude `ε` used in `f⁻¹(z + ε·η)`. Recommended range 0.01–0.1.
    /// Smaller `ε` is more faithful to the local Jacobian but suffers from f32 noise; larger
    /// `ε` stretches the linearisation and starts measuring non-local inverse behaviour.
    #[config(default = "0.05_f32")]
    pub fd_epsilon: f32,
    /// Apply the FD term on every Nth optimiser step. `1` = every step (highest cost). Values
    /// >1 throttle the activation memory cost (FD ≈ doubles activations on the steps it runs).
    #[config(default = "1")]
    pub fd_every_n_steps: u64,
}

#[derive(Config, Debug)]
pub struct ModelSettings {
    #[config(default = "3")]
    pub in_channels: usize,
    #[config(default = "128")]
    pub in_height: usize,
    #[config(default = "128")]
    pub in_width: usize,
    #[config(default = "2")]
    pub num_levels: usize,
    #[config(default = "2")]
    pub num_steps: usize,
    #[config(default = "128")]
    pub hidden_features: usize,
    #[config(default = "8")]
    pub pixel_depth: u32,
    /// Coupling parameterisation: `Affine` (default) or `Additive`.
    #[config(default = "CouplingType::Affine")]
    pub coupling_type: CouplingType,
}

#[derive(Config, Debug)]
pub struct OptimizerSettings {
    #[config(default = "1e-4")]
    pub learning_rate: f64,
    pub weight_decay: Option<f32>,
    /// Gradient clip by global norm (`Norm(max_norm)` in Burn). `None` disables clipping.
    #[config(default = "Some(5.0f32)")]
    pub grad_clip_norm: Option<f32>,
}

#[derive(Config, Debug)]
pub struct DataSettings {
    #[config(default = "4")]
    pub batch_size: usize,
    /// Number of micro-batches to accumulate before each optimizer step. Effective batch size
    /// is `batch_size * grad_accum_steps`. Any partial window at epoch end is dropped.
    ///
    /// Note: increasing `grad_accum_steps` reduces gradient noise but does **not** auto-scale
    /// the learning rate. Consider raising `optimizer.learning_rate` (linear or √-rule) when
    /// going from effective batch B to B·K.
    #[config(default = "1")]
    pub grad_accum_steps: usize,
    #[config(default = "2")]
    pub num_workers: usize,
    #[config(default = "42")]
    pub shuffle_seed: u64,
    /// Cap validation batches per eval (None = full pass).
    #[config(default = "Some(20usize)")]
    pub val_max_batches: Option<usize>,
}

/// Which checkpoints carry the AdamW optimiser sidecar (`<base>.optim.bin`).
///
/// AdamW state is roughly 2× model size; persisting it on every `step_NNNNNN`
/// checkpoint balloons disk usage linearly. Only `latest` and `best` need it
/// for resume — periodic step checkpoints are mostly archival.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum SaveOptim {
    /// Write `.optim.bin` only alongside `latest` (default).
    Latest,
    /// Write `.optim.bin` alongside both `best` and `latest`.
    Best,
    /// Write `.optim.bin` alongside every checkpoint (bigger disk; unbounded resume).
    All,
}

#[derive(Config, Debug)]
pub struct RunSettings {
    #[config(default = "10")]
    pub max_epochs: usize,
    #[config(default = "10")]
    pub log_every: usize,
    #[config(default = "100")]
    pub val_every: usize,
    #[config(default = "500")]
    pub checkpoint_every: usize,
    #[config(default = "\"checkpoints\".to_string()")]
    pub checkpoint_dir: String,
    /// Soft threshold on Glow round-trip **mean squared error in pixel space** `[0,255]`.
    /// Above this, training prints a warning but continues. `None` disables the warning.
    #[config(default = "Some(50f32)")]
    pub invert_warn_tol: Option<f32>,
    /// Hard threshold on Glow round-trip pixel MSE; above this (or non-finite) training aborts
    /// and saves a `*_diverged` checkpoint. RMSE ≈ √tol, so `5000` ≈ `±70` per pixel — visibly
    /// catastrophic on `[0,255]`. `None` disables the abort.
    #[config(default = "Some(5000f32)")]
    pub invert_abort_tol: Option<f32>,
    /// Number of fixed validation batches to average the round-trip pixel MSE over. Larger
    /// values reduce noise; the same batches are reused on every invert check so the metric
    /// is comparable across steps.
    #[config(default = "2")]
    pub invert_check_batches: usize,
    /// When the invert check runs at validation, also walk the model and print per-step
    /// diagnostics (coupling `log_s` extrema, ActNorm `weight` range, InvConv `log_w_s`).
    /// Costs one extra forward on the same fixed val batches.
    #[config(default = "false")]
    pub invert_diagnostics: bool,
    /// Which checkpoints get an `<base>.optim.bin` sidecar. See [`SaveOptim`].
    /// Default: `Latest` (suitable to resume training after a crash without bloating
    /// the periodic `step_*` archive).
    #[config(default = "SaveOptim::Latest")]
    pub save_optim: SaveOptim,
}

#[derive(Default, Deserialize)]
struct TrainingConfigToml {
    #[serde(default)]
    model: Option<ModelSettingsToml>,
    #[serde(default)]
    optimizer: Option<OptimizerSettingsToml>,
    #[serde(default)]
    data: Option<DataSettingsToml>,
    #[serde(default)]
    run: Option<RunSettingsToml>,
    #[cfg(feature = "fd_reg")]
    #[serde(default)]
    regularization: Option<RegularizationSettingsToml>,
}

#[cfg(feature = "fd_reg")]
#[derive(Default, Deserialize)]
struct RegularizationSettingsToml {
    #[serde(default)]
    fd_lambda: Option<f32>,
    #[serde(default)]
    fd_warmup_steps: Option<u64>,
    #[serde(default)]
    fd_epsilon: Option<f32>,
    #[serde(default)]
    fd_every_n_steps: Option<u64>,
}

#[derive(Default, Deserialize)]
struct ModelSettingsToml {
    #[serde(default)]
    in_channels: Option<usize>,
    #[serde(default)]
    in_height: Option<usize>,
    #[serde(default)]
    in_width: Option<usize>,
    #[serde(default)]
    num_levels: Option<usize>,
    #[serde(default)]
    num_steps: Option<usize>,
    #[serde(default)]
    hidden_features: Option<usize>,
    #[serde(default)]
    pixel_depth: Option<u32>,
    /// `"affine"` (default) or `"additive"`. Case-insensitive. Other values are
    /// rejected by [`parse_coupling_type`] with a clear error.
    #[serde(default)]
    coupling_type: Option<String>,
}

fn parse_coupling_type(raw: Option<&str>) -> Result<Option<CouplingType>, String> {
    match raw.map(|s| s.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(s) if s == "affine" => Ok(Some(CouplingType::Affine)),
        Some(s) if s == "additive" => Ok(Some(CouplingType::Additive)),
        Some(other) => Err(format!(
            "model.coupling_type: unknown variant {other:?}; expected \"affine\" or \"additive\""
        )),
    }
}

#[derive(Default, Deserialize)]
struct OptimizerSettingsToml {
    #[serde(default)]
    learning_rate: Option<f64>,
    #[serde(default)]
    weight_decay: Option<f32>,
    #[serde(default)]
    grad_clip_norm: Option<f32>,
}

#[derive(Default, Deserialize)]
struct DataSettingsToml {
    #[serde(default)]
    batch_size: Option<usize>,
    #[serde(default)]
    grad_accum_steps: Option<usize>,
    #[serde(default)]
    num_workers: Option<usize>,
    #[serde(default)]
    shuffle_seed: Option<u64>,
    #[serde(default)]
    val_max_batches: Option<usize>,
}

#[derive(Default, Deserialize)]
struct RunSettingsToml {
    #[serde(default)]
    max_epochs: Option<usize>,
    #[serde(default)]
    log_every: Option<usize>,
    #[serde(default)]
    val_every: Option<usize>,
    #[serde(default)]
    checkpoint_every: Option<usize>,
    #[serde(default)]
    checkpoint_dir: Option<String>,
    #[serde(default)]
    invert_warn_tol: Option<f32>,
    #[serde(default)]
    invert_abort_tol: Option<f32>,
    #[serde(default)]
    invert_check_batches: Option<usize>,
    #[serde(default)]
    invert_diagnostics: Option<bool>,
    /// `"latest"` (default), `"best"`, or `"all"`. Case-insensitive.
    #[serde(default)]
    save_optim: Option<String>,
}

fn parse_save_optim(raw: Option<&str>) -> Result<Option<SaveOptim>, String> {
    match raw.map(|s| s.trim().to_ascii_lowercase()) {
        None => Ok(None),
        Some(s) if s == "latest" => Ok(Some(SaveOptim::Latest)),
        Some(s) if s == "best" => Ok(Some(SaveOptim::Best)),
        Some(s) if s == "all" => Ok(Some(SaveOptim::All)),
        Some(other) => Err(format!(
            "run.save_optim: unknown variant {other:?}; expected \"latest\", \"best\", or \"all\""
        )),
    }
}

impl TrainingConfigToml {
    fn merge(self) -> Result<TrainingConfig, String> {
        Ok(TrainingConfig {
            model: merge_model_settings(self.model)?,
            optimizer: merge_optimizer_settings(self.optimizer),
            data: merge_data_settings(self.data),
            run: merge_run_settings(self.run)?,
            #[cfg(feature = "fd_reg")]
            regularization: merge_regularization_settings(self.regularization),
        })
    }
}

#[cfg(feature = "fd_reg")]
fn merge_regularization_settings(
    raw: Option<RegularizationSettingsToml>,
) -> RegularizationSettings {
    let b = RegularizationSettings::new();
    let r = raw.unwrap_or_default();
    RegularizationSettings {
        fd_lambda: r.fd_lambda.unwrap_or(b.fd_lambda),
        fd_warmup_steps: r.fd_warmup_steps.unwrap_or(b.fd_warmup_steps),
        fd_epsilon: r.fd_epsilon.unwrap_or(b.fd_epsilon),
        fd_every_n_steps: r.fd_every_n_steps.unwrap_or(b.fd_every_n_steps),
    }
}

fn merge_model_settings(raw: Option<ModelSettingsToml>) -> Result<ModelSettings, String> {
    let b = ModelSettings::new();
    let r = raw.unwrap_or_default();
    let coupling_type = parse_coupling_type(r.coupling_type.as_deref())?.unwrap_or(b.coupling_type);
    Ok(ModelSettings {
        in_channels: r.in_channels.unwrap_or(b.in_channels),
        in_height: r.in_height.unwrap_or(b.in_height),
        in_width: r.in_width.unwrap_or(b.in_width),
        num_levels: r.num_levels.unwrap_or(b.num_levels),
        num_steps: r.num_steps.unwrap_or(b.num_steps),
        hidden_features: r.hidden_features.unwrap_or(b.hidden_features),
        pixel_depth: r.pixel_depth.unwrap_or(b.pixel_depth),
        coupling_type,
    })
}

fn merge_optimizer_settings(raw: Option<OptimizerSettingsToml>) -> OptimizerSettings {
    let b = OptimizerSettings::new();
    let r = raw.unwrap_or_default();
    OptimizerSettings {
        learning_rate: r.learning_rate.unwrap_or(b.learning_rate),
        weight_decay: r.weight_decay.or(b.weight_decay),
        grad_clip_norm: r.grad_clip_norm.or(b.grad_clip_norm),
    }
}

fn merge_data_settings(raw: Option<DataSettingsToml>) -> DataSettings {
    let b = DataSettings::new();
    let r = raw.unwrap_or_default();
    DataSettings {
        batch_size: r.batch_size.unwrap_or(b.batch_size),
        grad_accum_steps: r.grad_accum_steps.unwrap_or(b.grad_accum_steps),
        num_workers: r.num_workers.unwrap_or(b.num_workers),
        shuffle_seed: r.shuffle_seed.unwrap_or(b.shuffle_seed),
        val_max_batches: r.val_max_batches.or(b.val_max_batches),
    }
}

fn merge_run_settings(raw: Option<RunSettingsToml>) -> Result<RunSettings, String> {
    let b = RunSettings::new();
    let r = raw.unwrap_or_default();
    let save_optim = parse_save_optim(r.save_optim.as_deref())?.unwrap_or(b.save_optim);
    Ok(RunSettings {
        max_epochs: r.max_epochs.unwrap_or(b.max_epochs),
        log_every: r.log_every.unwrap_or(b.log_every),
        val_every: r.val_every.unwrap_or(b.val_every),
        checkpoint_every: r.checkpoint_every.unwrap_or(b.checkpoint_every),
        checkpoint_dir: r.checkpoint_dir.unwrap_or(b.checkpoint_dir),
        invert_warn_tol: r.invert_warn_tol.or(b.invert_warn_tol),
        invert_abort_tol: r.invert_abort_tol.or(b.invert_abort_tol),
        invert_check_batches: r.invert_check_batches.unwrap_or(b.invert_check_batches),
        invert_diagnostics: r.invert_diagnostics.unwrap_or(b.invert_diagnostics),
        save_optim,
    })
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelSettings::new(),
            optimizer: OptimizerSettings::new(),
            data: DataSettings::new(),
            run: RunSettings::new(),
            #[cfg(feature = "fd_reg")]
            regularization: RegularizationSettings::new(),
        }
    }
}

impl TrainingConfig {
    pub fn from_toml_path(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let raw: TrainingConfigToml = toml::from_str(&s).map_err(|e| e.to_string())?;
        raw.merge()
    }

    pub fn glow_config(&self) -> GlowConfig {
        GlowConfig::new(self.model.in_channels)
            .with_num_levels(self.model.num_levels)
            .with_num_steps(self.model.num_steps)
            .with_hidden_features(self.model.hidden_features)
            .with_coupling_type(self.model.coupling_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_empty_yields_config_new_defaults() {
        let c = TrainingConfigToml::default().merge().expect("merge");
        assert_eq!(c.model.in_channels, 3);
        assert_eq!(c.optimizer.learning_rate, 1e-4);
        assert_eq!(c.optimizer.grad_clip_norm, Some(5.0));
        assert_eq!(c.data.batch_size, 4);
        assert_eq!(c.data.grad_accum_steps, 1);
        assert_eq!(c.run.checkpoint_dir, "checkpoints");
        assert_eq!(c.model.coupling_type, CouplingType::Affine);
    }

    #[test]
    fn toml_partial_sections_merge_with_new() {
        let s = r#"
            [model]
            num_levels = 3
            [optimizer]
            learning_rate = 0.001
        "#;
        let raw: TrainingConfigToml = toml::from_str(s).expect("partial TOML");
        let c = raw.merge().expect("merge");
        assert_eq!(c.model.in_channels, 3);
        assert_eq!(c.model.num_levels, 3);
        assert_eq!(c.model.num_steps, 2);
        assert!((c.optimizer.learning_rate - 0.001).abs() < 1e-9);
        assert_eq!(c.optimizer.grad_clip_norm, Some(5.0));
    }

    #[test]
    fn burn_config_new_matches_merged_empty() {
        let a = TrainingConfig::default();
        let b = TrainingConfigToml::default().merge().expect("merge");
        assert_eq!(a.model.in_channels, b.model.in_channels);
        assert_eq!(a.optimizer.grad_clip_norm, b.optimizer.grad_clip_norm);
        assert_eq!(a.data.val_max_batches, b.data.val_max_batches);
        assert_eq!(a.model.coupling_type, b.model.coupling_type);
    }

    #[test]
    fn toml_coupling_type_additive_parses() {
        let s = r#"
            [model]
            coupling_type = "additive"
        "#;
        let raw: TrainingConfigToml = toml::from_str(s).expect("TOML");
        let c = raw.merge().expect("merge");
        assert_eq!(c.model.coupling_type, CouplingType::Additive);
    }

    #[test]
    fn toml_coupling_type_unknown_errors() {
        let s = r#"
            [model]
            coupling_type = "lol"
        "#;
        let raw: TrainingConfigToml = toml::from_str(s).expect("TOML");
        let err = raw.merge().expect_err("unknown variant should error");
        assert!(err.contains("coupling_type"), "error mentions field: {err}");
    }
}
