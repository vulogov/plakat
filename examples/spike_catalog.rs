//! Spike: build the style catalog from a fixture directory.
//!
//! Reads a fixtures directory laid out as:
//!
//! ```text
//! <fixtures>/
//! ├── <style_id_1>/      ← any image files (.jpg/.png), at least one per style
//! ├── <style_id_2>/
//! └── holdout/           ← skipped; reserved for smoke-test queries
//! ```
//!
//! For each style subdirectory, every image is encoded through CLIP-H
//! to a 1024-d pooled embedding, L2-normalized, downcast to f16, and
//! written into `<out>/exemplars.safetensors` under the key
//! `<style_id>/<idx>`. A companion `<out>/catalog.json` is emitted with
//! the spike's hand-written routing metadata.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example spike_catalog -- \
//!     --fixtures tests/fixtures/style_catalog \
//!     --out      assets/style_catalog
//! ```
//!
//! Re-running overwrites both output files in-place.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Tensor};
use clap::Parser;
use serde_json::json;

use plakat::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};

#[derive(Parser, Debug)]
#[command(about = "Build the style-detection spike catalog from a fixture directory.")]
struct Args {
    /// Fixtures directory. Each subdirectory becomes a style; images
    /// inside it become its exemplars. The `holdout` subdirectory is
    /// skipped (reserved for smoke-test queries).
    #[arg(long, value_name = "DIR")]
    fixtures: PathBuf,

    /// Output directory for `catalog.json` + `exemplars.safetensors`.
    /// Created if it doesn't exist.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Override device (auto | cuda[:N] | metal | cpu).
    #[arg(long, default_value = "auto")]
    device: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = plakat::device::select(&args.device)?;

    if !args.fixtures.is_dir() {
        bail!("--fixtures {} is not a directory", args.fixtures.display());
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    eprintln!("==> loading CLIP-H image encoder");
    let weights =
        plakat::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors")
            .await?;
    let encoder = ImageEncoder::load(&weights, &device, DType::F32)?;

    // Discover styles. Sorted alphabetically for deterministic catalog order.
    let mut style_dirs: Vec<PathBuf> = std::fs::read_dir(&args.fixtures)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.file_name().map_or(false, |n| n != "holdout"))
        .collect();
    style_dirs.sort();

    if style_dirs.is_empty() {
        bail!(
            "no style subdirectories found under {} (other than `holdout`)",
            args.fixtures.display()
        );
    }

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut catalog_styles: Vec<serde_json::Value> = Vec::with_capacity(style_dirs.len());

    for style_dir in &style_dirs {
        let style_id = style_dir
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("non-UTF8 directory name {}", style_dir.display()))?
            .to_string();

        let mut images: Vec<PathBuf> = std::fs::read_dir(style_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("jpg" | "jpeg" | "png")))
            .collect();
        images.sort();

        if images.is_empty() {
            bail!("style '{}' has no image files", style_id);
        }

        eprintln!("==> encoding style '{}' ({} exemplars)", style_id, images.len());

        let mut exemplar_keys = Vec::with_capacity(images.len());

        for (idx, img_path) in images.iter().enumerate() {
            let key = format!("{}/{:02}", style_id, idx);
            eprintln!("    {} ← {}", key, img_path.display());

            // Reuse the runtime encode helper end-to-end — guarantees the
            // catalog is built with the exact preprocess/encode/normalize
            // path that the detector will use at query time.
            let emb = plakat::style::encode_reference_photo(&encoder, img_path, &device)
                .with_context(|| format!("encoding {}", img_path.display()))?;

            // (1024,) f32 → (1024,) f16 for storage.
            let stored = emb.to_dtype(DType::F16)?;
            tensors.insert(key.clone(), stored);
            exemplar_keys.push(key);
        }

        catalog_styles.push(spike_style_metadata(&style_id, exemplar_keys)?);
    }

    // ----- Write the safetensors -----
    let exemplars_path = args.out.join("exemplars.safetensors");
    candle_core::safetensors::save(&tensors, &exemplars_path)
        .with_context(|| format!("writing {}", exemplars_path.display()))?;
    eprintln!(
        "==> wrote {} ({} tensors)",
        exemplars_path.display(),
        tensors.len()
    );

    // ----- Write catalog.json -----
    let catalog = json!({
        "schema_version": 1,
        "encoder": {
            "id": "clip-h-laion2b",
            "embed_dim": 1024,
            "exemplars_file": "exemplars.safetensors",
            "preprocess": "clip-standard-224",
        },
        "detection": {
            "aggregation": "top3-mean",
            "min_confidence": 0.22,
            "margin_over_runner_up": 0.02,
        },
        "styles": catalog_styles,
    });
    let catalog_path = args.out.join("catalog.json");
    std::fs::write(&catalog_path, serde_json::to_string_pretty(&catalog)?)
        .with_context(|| format!("writing {}", catalog_path.display()))?;
    eprintln!("==> wrote {}", catalog_path.display());

    Ok(())
}

/// Hand-curated routing metadata per spike style. The full real catalog
/// (post-spike) sources this from a curator's YAML; here it's inlined
/// for the two styles we ship in the spike.
fn spike_style_metadata(
    style_id: &str,
    exemplar_keys: Vec<String>,
) -> Result<serde_json::Value> {
    let (display_name, description) = match style_id {
        "watercolor" => (
            "Watercolor",
            "Wet-on-wet pigment washes, ink lineart, visible paper texture.",
        ),
        "photorealistic" => (
            "Photorealistic",
            "Photographic realism; lens characteristics; lighting physicality.",
        ),
        other => bail!(
            "unknown spike style '{}' — extend spike_style_metadata() with its routing",
            other
        ),
    };

    // Spike: empty `models` section. The real catalog populates per-base
    // LoRA refs + trigger phrases here; the spike only validates the
    // detection path, so routing is left for a follow-up.
    Ok(json!({
        "id": style_id,
        "display_name": display_name,
        "description": description,
        "exemplar_keys": exemplar_keys,
        "models": {},
    }))
}
