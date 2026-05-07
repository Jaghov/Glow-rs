#![cfg(feature = "backend-tch")]
/// Test round-trip on LibTorch backend to isolate GPU vs CPU precision.
use burn::backend::libtorch::LibTorch;
use burn::tensor::{Distribution, Tensor};
use glow_flow::models::flow::{CouplingType, GlowConfig};

type B = LibTorch;

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
fn roundtrip_tch_scaling() {
    let (ma, mse) = roundtrip_error(2, 2, 32, 16);
    eprintln!("tch 2L×2S:  max_abs={ma:.2e}  mse={mse:.2e}");

    let (ma, mse) = roundtrip_error(2, 8, 64, 16);
    eprintln!("tch 2L×8S:  max_abs={ma:.2e}  mse={mse:.2e}");

    let (ma, mse) = roundtrip_error(3, 12, 128, 32);
    eprintln!("tch 3L×12S: max_abs={ma:.2e}  mse={mse:.2e}");

    let (ma, mse) = roundtrip_error(4, 12, 128, 32);
    eprintln!("tch 4L×12S: max_abs={ma:.2e}  mse={mse:.2e}");

    // Near production
    let (ma, mse) = roundtrip_error(4, 24, 384, 64);
    eprintln!("tch 4L×24S×384h×64px: max_abs={ma:.2e}  mse={mse:.2e}");

    // Full production
    let (ma, mse) = roundtrip_error(4, 24, 384, 128);
    eprintln!("tch 4L×24S×384h×128px: max_abs={ma:.2e}  mse={mse:.2e}");
    assert!(
        ma < 0.01,
        "full production round-trip max_abs should be small, got {ma}"
    );
}
