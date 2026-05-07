use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::Initializer,
    prelude::Backend,
    tensor::{activation, linalg, ops::ConvOptions, Float as BurnFloat, TensorData},
    Tensor,
};

// Soft bound on `log_w_s` (log of `|diag(U)|` in the PLU factorisation).
// Effective value is `LOG_W_S_MAX * tanh(raw / LOG_W_S_MAX)`, keeping pivots
// in `[exp(-MAX), exp(MAX)]` ≈ `[0.05, 20.1]` for MAX = 3. Prevents
// near-singular matrices (tiny pivots) and huge inverse amplification across
// a deep stack of InvConv layers.
const LOG_W_S_MAX: f32 = 3.0;

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
    /// Invert a small n×n matrix on-device. The default pulls to host and uses
    /// Gauss–Jordan; backends with native linalg can override (see LibTorch impl).
    fn invert_matrix(w: Tensor<Self, 2>) -> Tensor<Self, 2> {
        let [n, _] = w.dims();
        let device = w.device();
        let w_data = w
            .into_data()
            .to_vec::<f32>()
            .expect("invert_matrix: f32 conversion failed");
        gauss_jordan_inverse::<Self>(&w_data, n, &device)
    }

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

        let nan_kernel =
            || Tensor::from_data(TensorData::new(vec![f32::NAN; n * n], [n, n]), &device);

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

/// Gauss–Jordan full-matrix inverse on the host. Given a flat row-major `n×n`
/// matrix, returns `Tensor<B, 2>` on `device`. Returns a NaN matrix if the
/// matrix is singular or contains non-finite values.
fn gauss_jordan_inverse<B: Backend>(data: &[f32], n: usize, device: &B::Device) -> Tensor<B, 2> {
    let nan_kernel = || Tensor::from_data(TensorData::new(vec![f32::NAN; n * n], [n, n]), device);
    if data.iter().any(|v| !v.is_finite()) {
        return nan_kernel();
    }

    let mut aug = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = data[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return nan_kernel();
        }
        if max_row != col {
            for k in 0..(2 * n) {
                aug.swap(col * 2 * n + k, max_row * 2 * n + k);
            }
        }

        let pivot = aug[col * 2 * n + col];
        let inv_pivot = 1.0 / pivot;
        for k in 0..(2 * n) {
            aug[col * 2 * n + k] *= inv_pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for k in 0..(2 * n) {
                aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Tensor::from_data(TensorData::new(inv, [n, n]), device)
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

// Cast to fp64 before inversion: a per-layer 1×1 conv weight has condition
// number ~exp(LOG_W_S_MAX·2); compounded across 96 layers an f32 inverse drifts
// enough to dominate the round-trip MSE. fp64 inversion (then cast back) costs
// ~nothing on a [C,C] matrix — sizes here are at most a few hundred.
#[cfg(feature = "backend-tch")]
fn invert_via_fp64(tch_t: &tch::Tensor) -> tch::Tensor {
    let dtype = tch_t.kind();
    let in_f64 = tch_t.to_kind(tch::Kind::Double);
    let inv_f64 = tch::Tensor::linalg_inv(&in_f64);
    inv_f64.to_kind(dtype)
}

#[cfg(any(feature = "backend-tch"))]
impl TriangularInverse for burn::backend::LibTorch {
    fn invert_matrix(w: Tensor<Self, 2>) -> Tensor<Self, 2> {
        use burn::backend::libtorch::TchTensor;
        use burn::tensor::TensorPrimitive;
        match w.into_primitive() {
            TensorPrimitive::Float(tch_t) => {
                let tch_inv = invert_via_fp64(&tch_t.tensor);
                Tensor::from_primitive(TensorPrimitive::Float(TchTensor::new(tch_inv)))
            }
            _ => unreachable!("invert_matrix: expected Float primitive"),
        }
    }
}
// `invert_matrix` is only used on the reverse path (sampling / round-trip
// check); no gradient flows through it. So the Autodiff variant detaches to
// the inner LibTorch tensor, runs the fp64 inversion there, and re-wraps via
// `Tensor::from_inner` (no-grad).
#[cfg(feature = "backend-tch")]
impl TriangularInverse for backend::Autodiff<burn::backend::LibTorch> {
    fn invert_matrix(w: Tensor<Self, 2>) -> Tensor<Self, 2> {
        let inner = w.inner();
        let inv = <burn::backend::LibTorch as TriangularInverse>::invert_matrix(inner);
        Tensor::from_inner(inv)
    }
}

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
        let w_p =
            Param::from_tensor(Tensor::from_data(w_p.into_data(), device)).set_require_grad(false);
        let w_s_sign = Param::from_tensor(Tensor::from_data(w_s_sign.into_data(), device))
            .set_require_grad(false);
        let conv_options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

        InvConv1x1 {
            w_p,
            w_lu,
            log_w_s,
            w_s_sign,
            conv_options: Ignored(conv_options),
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
}

impl<B: Backend> InvConv1x1<B> {
    /// Bounded log-scale: `LOG_W_S_MAX * tanh(raw / LOG_W_S_MAX)`.
    /// At init (orthogonal matrix, `log_w_s` near 0) this is near-identity.
    #[inline]
    fn effective_log_w_s(&self) -> Tensor<B, 1> {
        activation::tanh(self.log_w_s.val().div_scalar(LOG_W_S_MAX)).mul_scalar(LOG_W_S_MAX)
    }

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
        let eff = self.effective_log_w_s();
        self.w_p
            .val()
            .matmul(self.w_lu.val().tril(-1) + ident.clone())
            .matmul(
                self.w_lu.val().triu(1)
                    + (eff.exp() * self.w_s_sign.val()).unsqueeze::<2>() * ident,
            )
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 1>) {
        let kernel = self.construct_kernel();
        let [b, .., h, w] = input.dims();
        let log_det_jacobian = (h * w) as f32 * self.effective_log_w_s().sum();
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
    /// Diagnostic: `(min, max)` of the effective (post-tanh) `log_w_s`.
    pub fn log_w_s_extrema(&self) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let s = self.effective_log_w_s();
        (s.clone().min(), s.max())
    }

    /// Diagnostic: max absolute value of the off-diagonal elements of L and U.
    pub fn off_diag_abs_max(&self) -> (f32, f32) {
        let w_lu = self.w_lu.val();
        let l_data = w_lu.clone().tril(-1).abs().max().into_data();
        let u_data = w_lu.triu(1).abs().max().into_data();
        let l_off = l_data.to_vec::<f32>().unwrap()[0];
        let u_off = u_data.to_vec::<f32>().unwrap()[0];
        (l_off, u_off)
    }

    /// Diagnostic: forward kernel W and its Frobenius norm.
    pub fn kernel_frobenius_norm(&self) -> f32 {
        let w = self.construct_kernel();
        let data = w.powf_scalar(2.0).sum().sqrt().into_data();
        data.to_vec::<f32>().unwrap()[0]
    }
}

impl<B: Backend + TriangularInverse> InvConv1x1<B> {
    #[inline]
    fn construct_inverse_kernel(&self) -> Tensor<B, 2> {
        let w = self.construct_kernel();
        B::invert_matrix(w)
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::record::{BinFileRecorder, FullPrecisionSettings};
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
    fn test_extreme_log_w_s_stays_invertible() {
        // With the tanh soft-clamp, even extreme raw log_w_s values are bounded
        // and the layer stays invertible (no NaN/Inf). This is the desired
        // behaviour: singularity is unreachable by design.
        let device = Default::default();
        let mut layer = make_layer(4);
        layer.log_w_s = Param::from_tensor(Tensor::full([4], -100.0_f32, &device));

        let x: Tensor<TestBackend, 4> = Tensor::random(
            [2, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (y, _) = layer.forward(x.clone());
        let x_rec = layer.inverse(y);

        let vals = x_rec.clone().into_data().to_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "extreme log_w_s should still produce finite inverse thanks to tanh bound"
        );
        let diff = (x - x_rec).abs().max().into_scalar();
        assert!(
            diff < 1e-3,
            "round-trip should still be accurate, got diff={diff}"
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

        let tmp =
            std::env::temp_dir().join(format!("glow_rs_invconv_resume_{}", std::process::id()));
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
