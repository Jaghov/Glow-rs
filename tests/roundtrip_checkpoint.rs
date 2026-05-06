/// Load the actual trained checkpoint and test round-trip on both CPU and GPU.
use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{Distribution, Tensor};
use glow_flow::models::flow::{CouplingType, GlowConfig};

type B = LibTorch;

fn test_roundtrip(device: LibTorchDevice, label: &str) {
    let checkpoint = std::path::Path::new("/tmp/glow_stability_test/additive/best");
    if !checkpoint.with_extension("bin").exists() {
        eprintln!("SKIP: checkpoint not found");
        return;
    }

    let glow_cfg = GlowConfig::new(3)
        .with_num_levels(4)
        .with_num_steps(24)
        .with_hidden_features(384)
        .with_coupling_type(CouplingType::Additive);

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = glow_cfg
        .init::<B>(&device)
        .load_file(checkpoint, &recorder, &device)
        .expect("load checkpoint");

    let input = Tensor::<B, 4>::random(
        [1, 3, 128, 128],
        Distribution::Normal(0.0, 0.3),
        &device,
    );

    let (zs, _) = model.forward(input.clone());
    let recon = model.inverse(zs);
    let diff = recon - input;
    let max_abs: f32 = diff.clone().abs().max().into_scalar();
    let mse: f32 = diff.powf_scalar(2.0).mean().into_scalar();

    eprintln!("{label}: max_abs={max_abs:.2e}  mse={mse:.2e}");
}

#[test]
fn checkpoint_roundtrip_cpu() {
    test_roundtrip(LibTorchDevice::Cpu, "CPU");
}

#[test]
fn checkpoint_roundtrip_gpu() {
    test_roundtrip(LibTorchDevice::Cuda(0), "GPU");
}
