//! TEMP runner (Phase 1 verification): train a watercolour style LoRA on
//! corpus/style/watercolour/ against SD3.5 MMDiT and write the safetensors.
//! Proves the trainer end-to-end before the `plakat style train` subcommand.
//!
//!   cargo run --release --features metal --bin style_train_run
use anyhow::Result;
use candle_core::Device;
use plakat::pipelines::sd3::{train_style_lora, StyleTrainRequest, Variant};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let device = Device::new_metal(0)?;
    let dir = std::path::Path::new("corpus/style/watercolour");
    let mut images: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("jpeg") | Some("jpg") | Some("png")
            )
        })
        .collect();
    images.sort();
    println!("style-train: {} source images", images.len());

    train_style_lora(StyleTrainRequest {
        variant: Variant::Sd35Medium,
        repo: "stabilityai/stable-diffusion-3.5-medium".into(),
        device,
        images,
        trigger: "wcstyle watercolour painting illustration".into(),
        rank: 16,
        steps: 90,
        lr: 1.5e-4,
        size: 256,
        out: "corpus/style/watercolour.safetensors".into(),
        checkpoint_every: None,
        log_every: 10,
    })
    .await
}
