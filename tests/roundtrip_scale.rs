#![cfg(feature = "backend-ndarray")]
/// Test round-trip error at increasing scale to find where drift appears.
use burn::backend::NdArray;
use burn::tensor::{Distribution, Tensor};
use glow_flow::models::flow::{CouplingType, GlowConfig};

type B = NdArray;

fn roundtrip_error(levels: usize, steps: usize, hidden: usize, spatial: usize) -> (f32, f32) {
    let device = Default::default();
    let model = GlowConfig::new(3)
        .with_num_levels(levels)
        .with_num_steps(steps)
        .with_hidden_features(hidden)
        .with_coupling_type(CouplingType::Additive)
        .init::<B>(&device);

    let input = Tensor::<B, 4>::random(
        [1, 3, spatial, spatial],
        Distribution::Normal(0.0, 0.3),
        &device,
    );
    let (zs, _) = model.forward(input.clone());
    let recon = model.inverse(zs);
    let diff = recon - input;
    let max_abs: f32 = diff.clone().abs().max().into_scalar();
    let mse: f32 = diff.powf_scalar(2.0).mean().into_scalar();
    (max_abs, mse)
}

#[test]
fn roundtrip_scaling_sweep() {
    // Small: 2 levels × 2 steps = 4 total steps
    let (ma, mse) = roundtrip_error(2, 2, 32, 16);
    eprintln!("2L×2S (4 steps):  max_abs={ma:.2e}  mse={mse:.2e}");
    assert!(ma < 1e-4, "2L×2S: max_abs={ma}");

    // Medium: 2 levels × 8 steps = 16 total steps
    let (ma, mse) = roundtrip_error(2, 8, 64, 16);
    eprintln!("2L×8S (16 steps): max_abs={ma:.2e}  mse={mse:.2e}");
    assert!(ma < 1e-3, "2L×8S: max_abs={ma}");

    // Larger: 3 levels × 12 steps = 36 total steps
    let (ma, mse) = roundtrip_error(3, 12, 128, 32);
    eprintln!("3L×12S (36 steps): max_abs={ma:.2e}  mse={mse:.2e}");

    // Near-production: 4 levels × 12 steps = 48 total steps
    let (ma, mse) = roundtrip_error(4, 12, 128, 32);
    eprintln!("4L×12S (48 steps): max_abs={ma:.2e}  mse={mse:.2e}");
}
