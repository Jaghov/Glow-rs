use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{linalg, ops::ConvOptions, Float as BurnFloat, TensorData},
    Tensor,
};
use std::sync::{Arc, Mutex};

// ── TriangularInverse trait ───────────────────────────────────────────────────

/// Compute W⁻¹ for W = P L U using the **factored** form W⁻¹ = U⁻¹ · L⁻¹ · Pᵀ.
///
/// Inverts each factor separately on the host (small n×n channel matrix):
///
/// * **Pᵀ** — exact: `w_p` is a frozen one-hot permutation matrix, transpose = inverse.
/// * **L⁻¹** — forward substitution on a unit lower triangular `L` (`diag = 1`); no pivoting,
///   no division by anything other than 1, exact up to `f32` rounding.
/// * **U⁻¹** — back substitution on `U`; the only numerical sensitivity is `1 / U_ii`, i.e.
///   `exp(-log_w_s_i)`, which is exactly the conditioning the model already exposes via the
///   log-det. A near-singular layer is a model-level problem, not a solver artefact.
///
/// Compared to the previous full-matrix Gauss–Jordan over the assembled `W`, this version:
/// * does roughly half the floating-point work,
/// * never re-pivots a matrix that is already factored,
/// * localises NaN propagation to the factor that is actually corrupted (almost always `U`
///   when `log_w_s` blows up).
///
/// `NdArray` and `LibTorch` share the same host implementation; on-device triangular solves
/// are blocked on Burn's default fusion stack not exposing raw Cube launches.
pub trait TriangularInverse: Backend {
    fn invert_plu(p: Tensor<Self, 2>, l: Tensor<Self, 2>, u: Tensor<Self, 2>) -> Tensor<Self, 2> {
        let [n, _] = l.dims();
        let device = l.device();

        let l_data = l
            .into_data()
            .to_vec::<f32>()
            .expect("invert_plu: l f32 conversion failed");
        let u_data = u
            .into_data()
            .to_vec::<f32>()
            .expect("invert_plu: u f32 conversion failed");
        let p_data = p
            .into_data()
            .to_vec::<f32>()
            .expect("invert_plu: p f32 conversion failed");

        let nan_kernel = || {
            Tensor::from_data(TensorData::new(vec![f32::NAN; n * n], [n, n]), &device)
        };

        // Early bail-out: a non-finite factor can only produce a non-finite kernel.
        // Returning a NaN kernel up-front avoids wasting host work on substitutions that
        // would just propagate the NaN anyway.
        if !l_data.iter().all(|x| x.is_finite()) || !u_data.iter().all(|x| x.is_finite()) {
            return nan_kernel();
        }

        // L⁻¹: solve L · X = I column-by-column via forward substitution.
        // L is unit lower triangular (diag = 1, zero strictly above diagonal) so each step
        // is x[i] = -Σ_{k=j..i} L[i][k] · X[k][j] with no division. Output is also unit
        // lower triangular; entries strictly above the diagonal stay zero.
        let mut l_inv = vec![0.0f32; n * n];
        for j in 0..n {
            l_inv[j * n + j] = 1.0;
            for i in (j + 1)..n {
                let mut s = 0.0f32;
                for k in j..i {
                    s += l_data[i * n + k] * l_inv[k * n + j];
                }
                l_inv[i * n + j] = -s;
            }
        }

        // U⁻¹: solve U · X = I column-by-column via back substitution.
        // diag(U) = exp(log_w_s) · w_s_sign; a tiny pivot (≈ 0) means the layer is
        // singular — fail loudly with NaN instead of returning a wildly amplified inverse.
        let mut u_inv = vec![0.0f32; n * n];
        for j in 0..n {
            for i in (0..=j).rev() {
                let mut s = if i == j { 1.0f32 } else { 0.0f32 };
                for k in (i + 1)..=j {
                    s -= u_data[i * n + k] * u_inv[k * n + j];
                }
                let pivot = u_data[i * n + i];
                if !pivot.is_finite() || pivot.abs() < 1e-12 {
                    return nan_kernel();
                }
                u_inv[i * n + j] = s / pivot;
            }
        }

        // M = U⁻¹ · L⁻¹ (host matmul; n ≤ ~96 in practice).
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            for k in 0..n {
                let a = u_inv[i * n + k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..n {
                    m[i * n + j] += a * l_inv[k * n + j];
                }
            }
        }

        // out = M · Pᵀ. With P stored row-major as a one-hot, (Pᵀ)[k][j] = P[j][k],
        // so out[i][j] = Σ_k M[i][k] · P[j][k]. Permutation has at most one non-zero per
        // row of P, so this loop also runs O(n²) effective work despite the O(n³) shape.
        let mut out = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    s += m[i * n + k] * p_data[j * n + k];
                }
                out[i * n + j] = s;
            }
        }

        Tensor::from_data(TensorData::new(out, [n, n]), &device)
    }
}

// Backend-specific impls, gated by their `backend-*` features. The trait body
// is empty (host-side `into_data` / `from_data` round-trip works on any backend),
// so consumers using a different backend (e.g. `wgpu`, `cuda`) just write
// `impl TriangularInverse for MyBackend {}` in their own crate.
//
// `cfg(test)` is included so the in-crate test suite (which uses NdArray) finds
// the impl without forcing inference users to enable `backend-ndarray`.
#[cfg(any(test, feature = "backend-ndarray"))]
impl TriangularInverse for burn::backend::NdArray {}

#[cfg(feature = "backend-tch")]
impl TriangularInverse for burn::backend::LibTorch {}

#[derive(Config, Debug)]
pub struct InvConv1x1Config {
    pub num_channels: usize,
}

impl InvConv1x1Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> InvConv1x1<B> {
        let shape = burn::tensor::Shape::new([self.num_channels, self.num_channels]);
        let weight: Tensor<B, 2> = Initializer::Orthogonal { gain: 1.0 }
            .init(shape, device)
            .into_value();

        let (w_lu, w_p) = linalg::lu_decomposition(weight);
        let w_p = w_p.one_hot::<2>(self.num_channels).float();

        let w_s = linalg::diag::<B, 2, 1, BurnFloat>(w_lu.clone());
        let w_s_sign = w_s.clone().sign();
        let log_w_s = Param::from_tensor(Tensor::from_data(w_s.abs().log().into_data(), device));
        let w_lu = Param::from_tensor(Tensor::from_data(w_lu.into_data(), device));
        // Frozen params: persisted via the Module record so resume preserves the LU
        // factorisation (mirrors PyTorch `register_buffer`); marked non-trainable so
        // AdamW won't drift them off the {0,1} / {±1} domain they live on.
        let w_p = Param::from_tensor(Tensor::from_data(w_p.into_data(), device))
            .set_require_grad(false);
        let w_s_sign = Param::from_tensor(Tensor::from_data(w_s_sign.into_data(), device))
            .set_require_grad(false);
        let conv_options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

        InvConv1x1 {
            w_p,
            w_lu,
            log_w_s,
            w_s_sign,
            conv_options: Ignored(conv_options),
            inverse_cache: Ignored(Arc::new(Mutex::new(None))),
        }
    }
}

/// Persistence note: only `Param<...>` fields round-trip through `Module::save_file` /
/// `load_file`. Burn implements `Module` for raw `Tensor<B, D>` with `type Record =
/// ConstantRecord` (empty) — i.e. a raw tensor field is **not** persisted; its loaded
/// value is whatever `init` produced on the receiving module. There is no buffer type;
/// the only distinction is `Param<Tensor>` (saved) vs `Tensor` (constant). Always wrap
/// state that must survive resume in `Param`, even if it's frozen via `set_require_grad(false)`.
#[derive(Module, Debug)]
pub struct InvConv1x1<B: Backend> {
    /// Permutation matrix from the initial LU factorisation. Persisted (`Param` is saved by
    /// the `Module` record) but frozen so AdamW never updates it; otherwise resume would
    /// silently regenerate a random permutation and the loaded `w_lu` would refer to a
    /// different factorisation. See `register_buffer("w_p", ...)` in the rosinality reference.
    w_p: Param<Tensor<B, 2>>,
    w_lu: Param<Tensor<B, 2>>,
    log_w_s: Param<Tensor<B, 1>>,
    /// Signs of the initial diagonal of `U` (`{-1, +1}`). Same persistence rationale as
    /// `w_p`: trained `log_w_s` only stores `log|U_ii|`, so the sign vector must round-trip
    /// or the kernel reconstruction flips signs on load.
    w_s_sign: Param<Tensor<B, 1>>,
    conv_options: Ignored<ConvOptions<2>>,
    // Cache for the inverted kernel. Must be cleared (`clear_inverse_cache`) after any
    // parameter update — there is no version counter on `Param` to auto-invalidate.
    // Stored as `TensorData` (backend-independent) so the field type stays compatible
    // with `Module` derive's autodiff `valid()` mapping.
    inverse_cache: Ignored<Arc<Mutex<Option<TensorData>>>>,
}

impl<B: Backend> InvConv1x1<B> {
    /// Build a fresh identity matrix on the device hosting `w_p`. Cheaper than carrying
    /// a persistent `ident` tensor: a raw `Tensor<B, D>` field would *not* be saved by
    /// the `Module` record (Burn's `Module for Tensor` uses `ConstantRecord`), so it
    /// would silently fall back to whatever `init` produced anyway — making the field
    /// only a foot-gun for future maintainers who reassign it.
    #[inline]
    fn ident(&self) -> Tensor<B, 2> {
        let n = self.w_p.val().dims()[0];
        Tensor::eye(n, &self.w_p.val().device())
    }

    #[inline]
    fn construct_kernel(&self) -> Tensor<B, 2> {
        let ident = self.ident();
        self.w_p
            .val()
            .matmul(self.w_lu.val().tril(-1) + ident.clone())
            .matmul(
                self.w_lu.val().triu(1)
                    + (self.log_w_s.val().exp() * self.w_s_sign.val()).unsqueeze::<2>() * ident,
            )
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let kernel = self.construct_kernel();
        let [b, .., h, w] = input.dims();
        let log_det_jacobian = (h * w) as f32 * self.log_w_s.val().sum();
        let out = burn::tensor::module::conv2d(
            input,
            kernel.unsqueeze_dims(&[2, 3]),
            None,
            self.conv_options.0.clone(),
        );
        (out, Tensor::repeat_dim(log_det_jacobian, 0, b))
    }
}

impl<B: Backend> InvConv1x1<B> {
    /// Drop any cached inverse kernel. Call after every optimizer step (or any other
    /// mutation of `w_lu` / `log_w_s`) — otherwise `inverse` will keep returning the
    /// kernel built from stale parameters.
    pub fn clear_inverse_cache(&self) {
        *self.inverse_cache.0.lock().unwrap() = None;
    }

    /// Diagnostic: `(min, max)` of `log_w_s` (log of `|diag(U)|`). Very negative values mean
    /// the channel-mixing matrix is approaching singular; `invert_plu` will amplify error.
    pub fn log_w_s_extrema(&self) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let s = self.log_w_s.val();
        (s.clone().min(), s.max())
    }
}

impl<B: Backend + TriangularInverse> InvConv1x1<B> {
    #[inline]
    fn construct_inverse_kernel(&self) -> Tensor<B, 2> {
        let device = self.w_p.val().device();
        let mut guard = self.inverse_cache.0.lock().unwrap();
        if let Some(cached) = guard.as_ref() {
            return Tensor::from_data(cached.clone(), &device);
        }
        let ident = self.ident();
        let l = self.w_lu.val().tril(-1) + ident.clone();
        let u = self.w_lu.val().triu(1)
            + (self.log_w_s.val().exp() * self.w_s_sign.val()).unsqueeze::<2>() * ident;
        let kernel = B::invert_plu(self.w_p.val(), l, u);
        *guard = Some(kernel.to_data());
        kernel
    }

    pub fn inverse(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let kernel = self.construct_inverse_kernel();
        burn::tensor::module::conv2d(
            input,
            kernel.unsqueeze_dims(&[2, 3]),
            None,
            self.conv_options.clone().0,
        )
    }
}

/// Differentiable inverse path for finite-differences regularization.
///
/// The default `inverse` routes through host Gauss–Jordan / triangular substitution and
/// `Tensor::from_data`, which **breaks the autodiff graph**. The FD regularizer needs to
/// backprop `||f⁻¹(f(x) + εη) − x||²` through the inverse, so this module-private path
/// constructs `W⁻¹ = U⁻¹ · L⁻¹ · Pᵀ` using only Burn ops, with **Newton–Schulz polishing
/// iterations seeded by the host-computed triangular inverses**.
///
/// ## Why polishing instead of cold-start Newton–Schulz
///
/// `X_{k+1} = X_k(2I − A X_k)` only converges when `||I − A X_0||_2 < 1`. Starting from
/// `X_0 = I` (or `X_0 = D⁻¹` for U) works for small n but not for n ≈ 96 with our
/// orthogonal init: the spectral norm of the strict-triangular part of the LU factor
/// can exceed 1, sending the iterates to overflow before the nilpotency-driven
/// cancellation kicks in.
///
/// With `X_0` already (numerically) equal to the true inverse, the first iteration's
/// residual is at floating-point rounding (~1e-7) and the second drops to below f32
/// resolution. **Critically, the gradient still flows correctly through the iteration**:
/// `X_0` is a detached leaf (no autodiff edges), and each Newton–Schulz step is a pure
/// Burn-op matmul that propagates `∂L⁻¹/∂L = −L⁻¹ (∂L) L⁻¹` exactly via autodiff.
///
/// ## Memory & compute
///
/// Each call: 2 host syncs (one per factor, n×n f32) + 4 matmuls of [n,n] in the
/// autodiff graph. For n ≤ 96 this is microseconds and a few MB of activations per
/// `InvConv1x1` call — dwarfed by the conv-net activations elsewhere.
#[cfg(feature = "fd_reg")]
impl<B: Backend> InvConv1x1<B> {
    /// Two polishing iterations: one is mathematically sufficient when `X_0` is the exact
    /// inverse (gradient via implicit differentiation is one Newton step), two covers f32
    /// rounding in the host-computed initial guess.
    const NEWTON_ITERS_POLISH: usize = 2;

    /// Newton–Schulz polishing: `X_{k+1} = X_k · (2I − A · X_k)`. Quadratic convergence
    /// in the residual `E_k = I − A X_k`; with `X_0 ≈ A⁻¹`, two steps is plenty.
    fn newton_schulz_polish(
        a: Tensor<B, 2>,
        x0: Tensor<B, 2>,
        iters: usize,
        ident: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut x = x0;
        for _ in 0..iters {
            let ax = a.clone().matmul(x.clone());
            let two_i = ident.clone().mul_scalar(2.0_f32);
            x = x.matmul(two_i - ax);
        }
        x
    }

    /// Forward substitution: invert a unit lower triangular L (diag = 1, strict lower
    /// from `w_lu`). Exact in `f32` up to rounding; never divides by anything since the
    /// diagonal is 1 by construction.
    fn host_invert_unit_lower(l: &[f32], n: usize) -> Vec<f32> {
        let mut inv = vec![0.0f32; n * n];
        for j in 0..n {
            inv[j * n + j] = 1.0;
            for i in (j + 1)..n {
                let mut s = 0.0f32;
                for k in j..i {
                    s += l[i * n + k] * inv[k * n + j];
                }
                inv[i * n + j] = -s;
            }
        }
        inv
    }

    /// Back substitution: invert an upper triangular U with arbitrary positive-or-negative
    /// diagonal. Returns `None` if any pivot is non-finite or below singular threshold —
    /// the autodiff caller substitutes a NaN initial guess in that case so the iteration
    /// produces NaN rather than a wildly wrong polish.
    fn host_invert_upper(u: &[f32], n: usize) -> Option<Vec<f32>> {
        let mut inv = vec![0.0f32; n * n];
        for j in 0..n {
            for i in (0..=j).rev() {
                let mut s = if i == j { 1.0f32 } else { 0.0f32 };
                for k in (i + 1)..=j {
                    s -= u[i * n + k] * inv[k * n + j];
                }
                let pivot = u[i * n + i];
                if !pivot.is_finite() || pivot.abs() < 1e-12 {
                    return None;
                }
                inv[i * n + j] = s / pivot;
            }
        }
        Some(inv)
    }

    /// Construct `W⁻¹` differentiably using the factored form `W⁻¹ = U⁻¹ · L⁻¹ · Pᵀ`.
    ///
    /// Unlike [`construct_inverse_kernel`](Self::construct_inverse_kernel) this path
    /// keeps the autodiff graph alive end-to-end — gradients flow back to `w_lu` and
    /// `log_w_s` (`w_p` and `w_s_sign` are frozen via `set_require_grad(false)`).
    fn construct_inverse_kernel_autodiff(&self) -> Tensor<B, 2> {
        let ident = self.ident();
        let n = ident.dims()[0];
        let device = self.w_p.val().device();

        // L = tril(w_lu, -1) + I (unit lower triangular). Autodiff-alive.
        let l = self.w_lu.val().tril(-1) + ident.clone();
        // U = triu(w_lu, 1) + diag(exp(log_w_s) · sign). Autodiff-alive.
        let diag_u = self.log_w_s.val().exp() * self.w_s_sign.val();
        let u = self.w_lu.val().triu(1) + diag_u.unsqueeze::<2>() * ident.clone();

        // Host initial guesses for the polishing iteration (autodiff-detached leaves).
        let l_data = l
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("autodiff inverse: l f32 conversion");
        let u_data = u
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("autodiff inverse: u f32 conversion");
        let l_inv_host = Self::host_invert_unit_lower(&l_data, n);
        let u_inv_host = Self::host_invert_upper(&u_data, n).unwrap_or_else(|| vec![f32::NAN; n * n]);
        let l_inv_init: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(l_inv_host, [n, n]), &device);
        let u_inv_init: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(u_inv_host, [n, n]), &device);

        // Polish: brings the gradient information into the autodiff graph.
        let l_inv = Self::newton_schulz_polish(l, l_inv_init, Self::NEWTON_ITERS_POLISH, ident.clone());
        let u_inv = Self::newton_schulz_polish(u, u_inv_init, Self::NEWTON_ITERS_POLISH, ident);

        // Pᵀ: permutation transpose. `w_p` has require_grad=false so no grad flows.
        let p_t: Tensor<B, 2> = self.w_p.val().swap_dims(0, 1);

        u_inv.matmul(l_inv).matmul(p_t)
    }

    /// Differentiable counterpart to [`inverse`](Self::inverse). Use only on the FD
    /// regularization path during training; the default `inverse` is faster and
    /// cache-aware for sampling / invert checks.
    pub fn inverse_autodiff(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let kernel = self.construct_inverse_kernel_autodiff();
        burn::tensor::module::conv2d(
            input,
            kernel.unsqueeze_dims(&[2, 3]),
            None,
            self.conv_options.clone().0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::check_closeness;
    use Module;

    type TestBackend = NdArray;

    const ATOL: f32 = 1e-4;

    fn max_abs_diff(a: Tensor<TestBackend, 2>, b: Tensor<TestBackend, 2>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    fn make_layer(num_channels: usize) -> InvConv1x1<TestBackend> {
        let device = Default::default();
        InvConv1x1Config { num_channels }.init(&device)
    }

    #[test]
    fn test_kernel_times_inverse_is_identity() {
        let layer = make_layer(4);
        let device = Default::default();
        let w = layer.construct_kernel();
        let w_inv = layer.construct_inverse_kernel();
        let product = w.matmul(w_inv);
        let ident: Tensor<TestBackend, 2> = Tensor::eye(4, &device);
        check_closeness(&product, &ident);
        assert!(
            max_abs_diff(product, ident) < ATOL,
            "W * W⁻¹ should be identity"
        );
    }

    #[test]
    fn test_inverse_times_kernel_is_identity() {
        let layer = make_layer(4);
        let device = Default::default();
        let w = layer.construct_kernel();
        let w_inv = layer.construct_inverse_kernel();
        let product = w_inv.matmul(w);
        let ident: Tensor<TestBackend, 2> = Tensor::eye(4, &device);
        check_closeness(&product, &ident);
        assert!(
            max_abs_diff(product, ident) < ATOL,
            "W⁻¹ * W should be identity"
        );
    }

    #[test]
    fn test_forward_inverse_roundtrip() {
        let layer = make_layer(4);
        let device = Default::default();
        let x: Tensor<TestBackend, 4> = Tensor::random(
            [2, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = layer.forward(x.clone());
        let x_rec = layer.inverse(y);
        check_closeness(&x_rec, &x);
        let diff = (x - x_rec).abs().max().into_scalar();
        assert!(
            diff < ATOL,
            "inverse(forward(x)) should recover x, diff={diff}"
        );
    }

    #[test]
    fn test_forward_inverse_roundtrip_large_channels() {
        // n=96 previously caused Neumann series overflow → NaN. Gauss-Jordan handles it exactly.
        let layer = make_layer(96);
        let device = Default::default();
        let x: Tensor<TestBackend, 4> = Tensor::random(
            [1, 96, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = layer.forward(x.clone());
        let x_rec = layer.inverse(y);
        let diff = (x - x_rec).abs().max().into_scalar();
        assert!(
            diff.is_finite() && diff < 1e-3,
            "inverse(forward(x)) should recover x for n=96, diff={diff}"
        );
    }

    #[test]
    fn test_inverse_non_finite_for_singular_matrix() {
        // A singular weight matrix should produce Inf or NaN, not silently wrong values.
        let device = Default::default();
        let mut layer = make_layer(4);
        layer.log_w_s = Param::from_tensor(Tensor::full([4], f32::NEG_INFINITY, &device));

        let x: Tensor<TestBackend, 4> = Tensor::random(
            [2, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = layer.forward(x);
        let x_rec = layer.inverse(y);

        let vals = x_rec.into_data().to_vec::<f32>().unwrap();
        assert!(
            vals.iter().any(|v| !v.is_finite()),
            "singular matrix should produce Inf/NaN in the inverse output"
        );
    }

    /// Regression: `w_p` and `w_s_sign` are `register_buffer`-style state. They must
    /// round-trip through `save_file`/`load_file` so resume reconstructs the exact same
    /// LU factorisation. Without `Param` wrapping these were silently re-randomised on
    /// load, producing garbage activations and eventual NaN — see the 2026-05-04 bug.
    #[test]
    fn invconv_resume_roundtrip_preserves_buffers() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let layer = make_layer(8);
        let kernel_before = layer.construct_kernel();
        let inv_before = layer.construct_inverse_kernel();

        let tmp = std::env::temp_dir().join(format!(
            "glow_rs_invconv_resume_{}",
            std::process::id()
        ));
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        layer
            .clone()
            .save_file(&tmp, &recorder)
            .expect("save_file should succeed");

        // Fresh `init` produces a random `w_p` / `w_s_sign`; if `load_file` doesn't
        // restore them, `kernel_after` will diverge wildly from `kernel_before`.
        let reloaded: InvConv1x1<TestBackend> = make_layer(8)
            .load_file(&tmp, &recorder, &device)
            .expect("load_file should succeed");
        let kernel_after = reloaded.construct_kernel();
        let inv_after = reloaded.construct_inverse_kernel();

        let with_ext = tmp.with_extension("bin");
        let _ = std::fs::remove_file(&with_ext);

        let kernel_diff = max_abs_diff(kernel_before, kernel_after);
        let inv_diff = max_abs_diff(inv_before, inv_after);
        assert!(
            kernel_diff < 1e-6,
            "kernel must match after save/load round-trip, diff={kernel_diff}"
        );
        assert!(
            inv_diff < 1e-6,
            "inverse kernel must match after save/load round-trip, diff={inv_diff}"
        );
    }

    #[test]
    fn test_inverse_cache_returns_stale_until_cleared() {
        let device = Default::default();
        let mut layer = make_layer(4);

        let kernel_orig = layer.construct_inverse_kernel();

        // Mutate log_w_s; without invalidation, the cache should still hand back the
        // kernel computed from the original parameters.
        let bumped = layer.log_w_s.val() + Tensor::full([4], 0.5, &device);
        layer.log_w_s = Param::from_tensor(bumped);

        let kernel_stale = layer.construct_inverse_kernel();
        let stale_diff = max_abs_diff(kernel_orig.clone(), kernel_stale);
        assert!(
            stale_diff < 1e-9,
            "cache should return identical stale kernel before clear, diff={stale_diff}"
        );

        layer.clear_inverse_cache();
        let kernel_fresh = layer.construct_inverse_kernel();
        let fresh_diff = max_abs_diff(kernel_orig, kernel_fresh);
        assert!(
            fresh_diff > 1e-3,
            "cache should be invalidated after clear, diff={fresh_diff}"
        );
    }

    /// Differentiable inverse parity: `W · W_inv_autodiff ≈ I` for both small and large n.
    /// Without this, the FD reg loss would silently get the wrong inverse.
    #[cfg(feature = "fd_reg")]
    #[test]
    fn autodiff_inverse_kernel_matches_identity_n4() {
        let layer = make_layer(4);
        let device = Default::default();
        let w = layer.construct_kernel();
        let w_inv = layer.construct_inverse_kernel_autodiff();
        let product = w.matmul(w_inv);
        let ident: Tensor<TestBackend, 2> = Tensor::eye(4, &device);
        let diff = max_abs_diff(product, ident);
        assert!(
            diff < 1e-4,
            "autodiff inverse for n=4 should give identity; diff={diff}"
        );
    }

    /// n=96 is the largest channel count we hit in CelebA-128 + num_levels=3. Newton–Schulz
    /// must still reach f32 precision at that size with the configured iteration counts.
    #[cfg(feature = "fd_reg")]
    #[test]
    fn autodiff_inverse_kernel_matches_identity_n96() {
        let layer = make_layer(96);
        let device = Default::default();
        let w = layer.construct_kernel();
        let w_inv = layer.construct_inverse_kernel_autodiff();
        let product = w.matmul(w_inv);
        let ident: Tensor<TestBackend, 2> = Tensor::eye(96, &device);
        let diff = max_abs_diff(product, ident);
        assert!(
            diff < 1e-3,
            "autodiff inverse for n=96 should give identity; diff={diff}"
        );
    }

    /// FD regularization needs gradients to flow through the inverse back to `w_lu` and
    /// `log_w_s`. The non-autodiff path uses `from_data` and would produce zero gradients.
    /// Verify the autodiff path produces a non-zero gradient on `w_lu`.
    #[cfg(feature = "fd_reg")]
    #[test]
    fn autodiff_inverse_kernel_produces_grad_on_w_lu() {
        use burn::backend::Autodiff;
        use burn::module::AutodiffModule;
        use burn::prelude::ElementConversion;

        type AD = Autodiff<NdArray>;

        let device = <AD as Backend>::Device::default();
        let layer: InvConv1x1<AD> = InvConv1x1Config { num_channels: 6 }.init::<AD>(&device);

        let w = layer.construct_kernel();
        let w_inv = layer.construct_inverse_kernel_autodiff();
        let ident: Tensor<AD, 2> = Tensor::eye(6, &device);
        let residual = w.matmul(w_inv) - ident;
        let loss = (residual.clone() * residual).sum();

        let grads = loss.backward();
        let grad_w_lu = layer.w_lu.val().grad(&grads).expect("w_lu must have grad");
        let abs_sum: f32 = grad_w_lu.abs().sum().into_scalar().elem();
        assert!(
            abs_sum.is_finite() && abs_sum > 0.0,
            "w_lu must receive non-zero finite gradient through differentiable inverse; abs_sum={abs_sum}"
        );

        // Sanity: `valid()` must still produce a usable inference module after we touched
        // the autodiff path. Catches accidental graph-leaking field changes.
        let _ = layer.valid();
    }

    #[cfg(feature = "backend-tch")]
    mod cuda_invert_plu {
        use super::super::TriangularInverse;
        use super::*;
        use burn::backend::libtorch::{LibTorch, LibTorchDevice};

        #[test]
        #[ignore = "requires CUDA; run: cargo test invert_plu_cuda_matches_ndarray -- --ignored --nocapture"]
        fn invert_plu_cuda_matches_ndarray() {
            let n = 8_usize;
            let cpu = Default::default();
            let gpu = LibTorchDevice::default();

            let layer = InvConv1x1Config { num_channels: n }.init::<NdArray>(&cpu);
            let p = layer.w_p.val();
            let ident = layer.ident();
            let l = layer.w_lu.val().tril(-1) + ident.clone();
            let u = layer.w_lu.val().triu(1)
                + (layer.log_w_s.val().exp() * layer.w_s_sign.val()).unsqueeze::<2>() * ident;

            let w_inv_nd = TriangularInverse::invert_plu(p.clone(), l.clone(), u.clone());

            let p_c = Tensor::<LibTorch, 2>::from_data(p.into_data(), &gpu);
            let l_c = Tensor::<LibTorch, 2>::from_data(l.into_data(), &gpu);
            let u_c = Tensor::<LibTorch, 2>::from_data(u.into_data(), &gpu);
            let w_inv_cuda = TriangularInverse::invert_plu(p_c, l_c, u_c);

            let w_inv_ref_on_gpu = Tensor::<LibTorch, 2>::from_data(w_inv_nd.into_data(), &gpu);
            assert!(
                w_inv_cuda.all_close(w_inv_ref_on_gpu, Some(1e-4), Some(1e-4)),
                "Cuda::invert_plu should match NdArray reference"
            );
        }
    }
}
