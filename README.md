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
