//! TEMP runner: train an SD 1.5 watercolour style LoRA on
//! corpus/style/watercolour/ → corpus/style/watercolour-sd15.safetensors.
//! Proves the SD UNet trainer before the `--base sd15` subcommand wiring.
//!
//!   cargo run --release --features metal --bin sd_train_run
use anyhow::Result;
use candle_core::Device;
use plakat::pipelines::sd_train::trainer::{train_style_lora_sd, SdStyleTrainRequest};

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
    println!("sd-style-train: {} source images", images.len());

    train_style_lora_sd(SdStyleTrainRequest {
        model: "sd15".into(),
        device,
        images,
        trigger: "wcstyle watercolour painting illustration".into(),
        rank: 16,
        steps: 150,
        lr: 1e-4,
        size: 512,
        out: "corpus/style/watercolour-sd15.safetensors".into(),
        checkpoint_every: None,
    })
    .await
}
