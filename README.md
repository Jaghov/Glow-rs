# Glow-rs

A Rust implementation of [Glow](https://arxiv.org/abs/1807.03039) — a generative normalizing flow for images — built with [Burn](https://burn.dev).

Targets CelebA 128×128. Uses **NdArray** for CPU tests and **LibTorch + CUDA** for training and sampling.

---

## Build & test

```bash
cargo build
cargo test --lib          # runs all unit + integration tests on CPU (NdArray)
cargo test --lib -- --nocapture   # with output (useful for debugging)
```

---

## Training

GPU training requires LibTorch with CUDA. See [Burn's LibTorch setup guide](https://burn.dev/book/getting-started.html) for installation steps.

```bash
# Train with default config (2 levels, 2 steps, lr=1e-4) on CUDA:
cargo run -- train

# Train with a custom TOML config:
cargo run -- train --config train.toml

# Train on CPU (slow — for smoke-testing only):
cargo run -- train --device cpu

# Resume from a checkpoint (path includes the coupling-type sub-dir):
cargo run -- train --resume checkpoints/affine/latest

# Override individual hyperparameters:
cargo run -- train --batch-size 8 --max-epochs 100 --learning-rate 5e-5

# Train with Rerun (see “Optional Rerun visualisation” below):
cargo run --features rerun -- train

# Switch to additive (NICE-style) coupling:
cargo run -- train --coupling-type additive

```

### Example `train.toml`

```toml
[model]
in_channels   = 3
in_height     = 128
in_width      = 128
num_levels    = 3
num_steps     = 16
hidden_features = 512
pixel_depth   = 8
# Coupling parameterisation: "affine" (Glow default, scaled-sigmoid bounded scale)
# or "additive" (NICE/RealNVP, y_b = x_b + t; volume-preserving, log_det = 0).
coupling_type = "affine"

[optimizer]
learning_rate  = 1e-4
grad_clip_norm = 5.0

[data]
batch_size  = 16
num_workers = 4

[run]
max_epochs       = 50
checkpoint_dir   = "checkpoints"
checkpoint_every = 1000
val_every        = 500


#### Coupling types

- **`affine`** — `y_b = s · x_b + t`, where `s = SCALE_MIN + (1 − SCALE_MIN) · σ(raw_s + bias)` is bounded in `[~0.082, 1)` via a scaled sigmoid (no `clamp_min` gradient masking). Best image quality; this is what the original Glow paper uses.
- **`additive`** — `y_b = x_b + t`, log-det = 0 (NICE/RealNVP). The conv block predicts only the shift, so it has half as many output channels. More numerically robust to invert (no `1/s` factor in the inverse).

See [docs/ADDITIVE_COUPLING.md](docs/ADDITIVE_COUPLING.md) for the design-doc walkthrough (motivation, when to pick which, history of the removed spectral-norm path).

#### Checkpoints

Checkpoints are namespaced by coupling type so affine and additive runs can share `checkpoint_dir` without overwriting each other:

```
<checkpoint_dir>/
├── affine/
│   ├── best.{bin,meta.json}
│   ├── latest.{bin,meta.json}
│   └── step_NNNNNN.{bin,meta.json}
└── additive/
    ├── best.{bin,meta.json}
    └── ...
```

To resume, point `--resume` at the nested base path, e.g. `--resume checkpoints/additive/latest`. The two coupling types are not weight-compatible (different conv-block output widths), and the meta sidecar records the type so a mismatched resume aborts with a clear error.


#### Migrating pre-additive checkpoints

The coupling-rework drop is BREAKING in three ways. Old checkpoints from before this change will not work as-is:

1. **`tanh`-bounded `shift`** — `forward` now applies `shift = SHIFT_BOUND · tanh(raw)` instead of using `raw` directly. Pre-bound checkpoints compute a different forward map even with identical conv weights. There is no automatic remap; train fresh.
2. **Scaled-sigmoid `s`** — `s` is now produced by `SCALE_MIN + (1 − SCALE_MIN) · σ(raw + SCALE_BIAS)` rather than `exp(clamp_min(raw, LOG_S_MIN))`. Same checkpoint-incompatibility consequence as (1).
3. **Checkpoint sub-directory layout** — `checkpoints/best.bin` is now `checkpoints/{affine,additive}/best.bin`. Move existing files manually if you want to keep them under the new layout, e.g. `mkdir -p checkpoints/affine && mv checkpoints/best.* checkpoints/latest.* checkpoints/affine/`. Resume needs the nested path: `--resume checkpoints/affine/latest`.

A checkpoint that lacks the `coupling_type` meta field is assumed to be affine. Resuming such a checkpoint with `model.coupling_type = "affine"` works (and warns); resuming with `"additive"` aborts with an error because the conv shapes don't match.

---

## Sampling

Given a trained checkpoint, generate a grid of images:

```bash
# 16 samples on CUDA (default) — note checkpoints are nested under the coupling type:
cargo run -- sample --checkpoint checkpoints/affine/best

# Custom output dir, sample count, temperature, and device:
cargo run -- sample \
  --checkpoint checkpoints/affine/best \
  --output-dir samples \
  --num-samples 25 \
  --temperature 0.8 \
  --device cpu
```

Output is saved to `<output-dir>/samples.png`.

### Optional Rerun visualisation

Enable the `rerun` Cargo feature to stream data to the [Rerun](https://rerun.io) viewer (install the [Rerun CLI](https://www.rerun.io/docs/getting-started/installing-viewer) if you use a separate viewer). The dependency is optional so default builds stay lean.

**Sampling** — logs the sample grid as an image (`sample/grid`), plus a text entry with the PNG path (`sample/path`):

```bash
cargo run --features rerun -- sample --checkpoint checkpoints/affine/best
```

**Training** — starts a recording named `glow-rs-train`. On each validation step (whenever `run.val_every` fires), the code draws **16** latents, runs the model in eval mode through `inverse`, and logs a 4×4 mosaic to `train/sample_grid`. Each log uses two timelines: **`train_frames`** (0, 1, 2, … for consecutive validation snapshots) and **`step`** (the real global training step). A default [blueprint](https://www.rerun.io/docs/how-to/reusable-blueprints) opens a **2D** view on that entity, sets the active timeline to **`train_frames`**, enables **follow/play**-friendly defaults (4 fps, loop), so scrubbing the time cursor or pressing play animates the grids like a short video. The `step` timeline is still available in the UI if you want to seek by optimizer step. Logging costs one inverse pass per validation, not every optimizer step.

```bash
cargo run --features rerun -- train
cargo run --features rerun -- train --config train.toml
```

---

## Architecture

```
Pixels [B, C, H, W]  ∈  [0, 255]
  → Dequantize        (uniform noise + log-det penalty)
  → L × GlowBlock:
      squeeze2d  →  K × GlowStep (ActNorm → InvConv1×1 → AffineCoupling)
      SplitBlock →  z_l extracted at each non-final level
  → Isotropic Gaussian prior  log p(z)

log p(x) = Σ log|det J| + log p(z)
```
