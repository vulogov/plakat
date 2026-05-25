use anyhow::{Context, Result, anyhow};
use image::{ImageBuffer, Rgb};
use std::path::Path;

use crate::imaging::metadata::GenerationMetadata;

pub fn save_rgb_u8(buf: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    save_rgb_u8_inner(buf, width, height, path, None)
}

/// v0.17 phase 3: save the RGB buffer as PNG **with** an
/// Auto1111-compatible `parameters` tEXt chunk + a sibling JSON
/// sidecar carrying the structured equivalent.
///
/// Sidecar path: same stem as the PNG, `.json` extension. Existing
/// files are overwritten. Sidecar write failure is non-fatal —
/// emits a warning and continues so an image generation never
/// fails for a metadata-write reason (filesystem flake, read-only
/// out-dir, etc.).
/// v0.18: read the Auto1111 `parameters` tEXt chunk from a PNG.
/// Returns `None` when the PNG has no `parameters` chunk (most
/// non-plakat PNGs) or fails to decode. Used by
/// `plakat metadata <FILE>` and any future tooling that wants to
/// inspect a v0.17+ output's embedded recipe.
pub fn read_parameters_chunk(path: &Path) -> Result<Option<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let file = std::io::BufReader::new(file);
    let decoder = png::Decoder::new(file);
    let reader = decoder
        .read_info()
        .with_context(|| format!("decoding {}", path.display()))?;
    for chunk in &reader.info().uncompressed_latin1_text {
        if chunk.keyword == "parameters" {
            return Ok(Some(chunk.text.clone()));
        }
    }
    Ok(None)
}

pub fn save_rgb_u8_with_metadata(
    buf: &[u8],
    width: u32,
    height: u32,
    path: &Path,
    metadata: &GenerationMetadata,
) -> Result<()> {
    save_rgb_u8_inner(buf, width, height, path, Some(metadata))?;
    // Write the JSON sidecar. Best-effort.
    let json_path = path.with_extension("json");
    match metadata.to_json_pretty() {
        Ok(json) => {
            if let Err(e) = std::fs::write(&json_path, json) {
                tracing::warn!(
                    target: "plakat",
                    "metadata sidecar write failed for {}: {e}",
                    json_path.display()
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                target: "plakat",
                "metadata JSON serialization failed for {}: {e}",
                json_path.display()
            );
        }
    }
    Ok(())
}

fn save_rgb_u8_inner(
    buf: &[u8],
    width: u32,
    height: u32,
    path: &Path,
    metadata: Option<&GenerationMetadata>,
) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 3;
    if buf.len() != expected {
        return Err(anyhow!(
            "buffer size mismatch: got {}, expected {} for {}x{}",
            buf.len(),
            expected,
            width,
            height
        ));
    }
    match metadata {
        None => {
            // Fast path — no metadata to embed, defer to the
            // `image` crate. Byte-identical to the pre-phase-3
            // output.
            let img: ImageBuffer<Rgb<u8>, _> =
                ImageBuffer::from_raw(width, height, buf.to_vec())
                    .ok_or_else(|| anyhow!("failed to construct ImageBuffer"))?;
            img.save(path)?;
            Ok(())
        }
        Some(meta) => write_png_with_text_chunk(buf, width, height, path, meta),
    }
}

fn write_png_with_text_chunk(
    buf: &[u8],
    width: u32,
    height: u32,
    path: &Path,
    metadata: &GenerationMetadata,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let file = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    // A1111 / Civitai / ComfyUI all read this key.
    encoder
        .add_text_chunk(
            "parameters".to_string(),
            metadata.to_a1111_parameters_string(),
        )
        .with_context(|| "add `parameters` tEXt chunk")?;
    let mut writer = encoder
        .write_header()
        .with_context(|| format!("PNG header for {}", path.display()))?;
    writer
        .write_image_data(buf)
        .with_context(|| format!("PNG body for {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::metadata::GenerationMetadata;

    // Tests use the public `read_parameters_chunk` helper directly
    // (promoted from this module-private helper in v0.18 to power
    // `plakat metadata <FILE>`).
    fn read_parameters_chunk(path: &Path) -> Option<String> {
        super::read_parameters_chunk(path).ok().flatten()
    }

    #[test]
    fn save_with_metadata_writes_parameters_chunk_and_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let png_path = tmp.path().join("test.png");
        let json_path = tmp.path().join("test.json");

        // 2×2 solid-red image.
        let buf = vec![255u8, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
        let mut meta = GenerationMetadata::new(
            "a red square",
            "sd15",
            42,
            28,
            7.5,
            "euler-a",
            2,
            2,
        );
        meta.negative = "blurry".to_string();
        meta.loras = vec!["my/style:0.7".into()];

        save_rgb_u8_with_metadata(&buf, 2, 2, &png_path, &meta).unwrap();

        // PNG side: tEXt chunk readable + matches expected A1111
        // serialization.
        let chunk = read_parameters_chunk(&png_path).expect("parameters chunk present");
        assert!(chunk.contains("a red square"));
        assert!(chunk.contains("Negative prompt: blurry"));
        assert!(chunk.contains("Seed: 42"));
        assert!(chunk.contains("Model: sd15"));
        assert!(chunk.contains("LoRAs: my/style:0.7"));

        // Sidecar side: JSON file written and parses back.
        let json_text = std::fs::read_to_string(&json_path).expect("sidecar exists");
        let parsed: GenerationMetadata =
            serde_json::from_str(&json_text).expect("sidecar parses");
        assert_eq!(parsed.prompt, "a red square");
        assert_eq!(parsed.negative, "blurry");
        assert_eq!(parsed.loras, vec!["my/style:0.7".to_string()]);
    }

    #[test]
    fn save_without_metadata_produces_chunk_free_png() {
        let tmp = tempfile::tempdir().unwrap();
        let png_path = tmp.path().join("plain.png");
        let buf = vec![0u8, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255];
        save_rgb_u8(&buf, 2, 2, &png_path).unwrap();
        // No `parameters` chunk should be present.
        assert!(read_parameters_chunk(&png_path).is_none());
        // No sidecar should exist.
        let json_path = png_path.with_extension("json");
        assert!(!json_path.exists());
    }
}
