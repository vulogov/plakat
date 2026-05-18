//! Spike smoke test: prove the cosine pipeline carries enough style
//! signal end-to-end on real images.
//!
//! Validates three claims:
//!
//! 1. A held-out watercolor (not in the exemplars) picks the
//!    `watercolor` style with score above `min_confidence`.
//! 2. A held-out NASA photograph (not in the exemplars) picks the
//!    `photorealistic` style.
//! 3. Re-encoding an exemplar at runtime scores ≈ 1.0 against its own
//!    style — catches L2-normalization / tensor-shape / catalog-key bugs.
//!
//! These tests download CLIP-H weights (~2.5 GB) on first run. Once
//! cached locally they're fast. Marked `#[ignore]` so the default
//! `cargo test` invocation doesn't spend network bandwidth or
//! gigabytes of disk on a CI box that doesn't have the model cached.
//! Run explicitly: `cargo test --test style_detect_smoke -- --ignored`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Device};

use plakat::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};
use plakat::style::{detect_style, encode_reference_photo, StyleCatalog};

const CATALOG_DIR: &str = "assets/style_catalog";
const FIXTURES: &str = "tests/fixtures/style_catalog";

struct Fixtures {
    catalog: StyleCatalog,
    encoder: ImageEncoder,
    device: Device,
}

async fn load() -> Result<Fixtures> {
    let device = Device::Cpu;
    let catalog = StyleCatalog::load(Path::new(CATALOG_DIR), &device)?;
    catalog.assert_encoder("clip-h-laion2b")?;

    let weights =
        plakat::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors")
            .await?;
    let encoder = ImageEncoder::load(&weights, &device, DType::F32)?;

    Ok(Fixtures {
        catalog,
        encoder,
        device,
    })
}

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(FIXTURES).join(rel)
}

#[tokio::test]
#[ignore = "downloads CLIP-H (~2.5 GB) on first run; not for default CI"]
async fn watercolor_holdout_picks_watercolor() -> Result<()> {
    let f = load().await?;

    let photo = fixture("holdout/watercolor_sargent_willows.jpg");
    let emb = encode_reference_photo(&f.encoder, &photo, &f.device)?;
    let result = detect_style(&f.catalog, &emb, 5)?;

    eprintln!("watercolor holdout scores:");
    for m in &result.top {
        eprintln!("  {:>16}  {:.4}", m.style_id, m.score);
    }

    assert_eq!(
        result.picked.as_deref(),
        Some("watercolor"),
        "expected pick='watercolor', got picked={:?}, top={:?}",
        result.picked,
        result.top.iter().map(|m| (&m.style_id, m.score)).collect::<Vec<_>>(),
    );
    assert!(
        result.top[0].score > 0.25,
        "top score {} is suspiciously low",
        result.top[0].score
    );
    Ok(())
}

#[tokio::test]
#[ignore = "downloads CLIP-H (~2.5 GB) on first run; not for default CI"]
async fn photo_holdout_picks_photorealistic() -> Result<()> {
    let f = load().await?;

    let photo = fixture("holdout/photo_apollo17_earth.jpg");
    let emb = encode_reference_photo(&f.encoder, &photo, &f.device)?;
    let result = detect_style(&f.catalog, &emb, 5)?;

    eprintln!("photo holdout scores:");
    for m in &result.top {
        eprintln!("  {:>16}  {:.4}", m.style_id, m.score);
    }

    assert_eq!(
        result.picked.as_deref(),
        Some("photorealistic"),
        "expected pick='photorealistic', got picked={:?}, top={:?}",
        result.picked,
        result.top.iter().map(|m| (&m.style_id, m.score)).collect::<Vec<_>>(),
    );
    Ok(())
}

#[tokio::test]
#[ignore = "downloads CLIP-H (~2.5 GB) on first run; not for default CI"]
async fn identity_exemplar_beats_holdout() -> Result<()> {
    let f = load().await?;

    // Encode an image that IS in the catalog (one of watercolor's
    // exemplars). It should score *higher* than the watercolor holdout
    // — re-encoding the same image gives a cosine of ~1.0 against its
    // own stored embedding, even though Top3Mean aggregation dilutes
    // that with the next-2-best exemplars from the same style.
    let exemplar = fixture("watercolor/01_durer_hare.jpg");
    let exemplar_emb = encode_reference_photo(&f.encoder, &exemplar, &f.device)?;
    let exemplar_result = detect_style(&f.catalog, &exemplar_emb, 5)?;

    let holdout = fixture("holdout/watercolor_sargent_willows.jpg");
    let holdout_emb = encode_reference_photo(&f.encoder, &holdout, &f.device)?;
    let holdout_result = detect_style(&f.catalog, &holdout_emb, 5)?;

    let exemplar_top = &exemplar_result.top[0];
    let holdout_top = &holdout_result.top[0];

    eprintln!(
        "exemplar score: {:.4} ({})",
        exemplar_top.score, exemplar_top.style_id
    );
    eprintln!(
        "holdout score:  {:.4} ({})",
        holdout_top.score, holdout_top.style_id
    );

    assert_eq!(exemplar_top.style_id, "watercolor");
    assert_eq!(holdout_top.style_id, "watercolor");
    assert!(
        exemplar_top.score > holdout_top.score,
        "exemplar score {} not greater than holdout score {} — \
         L2 norm or catalog-key bug suspected",
        exemplar_top.score,
        holdout_top.score
    );
    Ok(())
}
