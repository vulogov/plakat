use anyhow::{Context, Result, anyhow};
use image::{ImageBuffer, Rgb};
use std::path::Path;
use std::str::FromStr;

use crate::imaging::metadata::GenerationMetadata;

/// v0.19: which container to write generated images into. PNG
/// stays the default (carries the v0.17 Auto1111 tEXt chunk for
/// drag-and-drop compatibility with A1111 / Civitai / ComfyUI);
/// WebP is opt-in via `--format webp` and trades the embedded
/// chunk for ~30% smaller files. The JSON sidecar is written for
/// both formats, so the recipe is recoverable via
/// `plakat metadata` / `plakat clone` regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Png,
    Webp,
}

impl OutputFormat {
    /// File extension WITHOUT the leading dot.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Webp => "webp",
        }
    }
}

impl FromStr for OutputFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "webp" => Ok(Self::Webp),
            other => Err(anyhow!(
                "unknown output format {other:?}; supported: png, webp"
            )),
        }
    }
}

pub fn save_rgb_u8(buf: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    save_rgb_u8_inner(buf, width, height, path, None, None)
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
/// The JSON-sidecar path for an image: the FULL filename plus `.json` (e.g.
/// `a.png` → `a.png.json`). Appending — rather than `with_extension("json")`, which drops
/// the image extension — keeps `a.png` and `a.webp` sidecars distinct when both formats
/// share a stem. Writer and reader must agree, so both use this.
pub fn sidecar_path(image_path: &Path) -> std::path::PathBuf {
    let mut s = image_path.as_os_str().to_owned();
    s.push(".json");
    std::path::PathBuf::from(s)
}

/// v2.8: patch an image's `.json` sidecar with an aesthetic `score`, preserving every other field.
/// Best-effort: no sidecar → skip (a bare score without the gen params isn't useful). Used by
/// `--score` / `--keep-best` / `plakat rank` so the collection manager has an on-disk sort key.
pub fn patch_sidecar_score(image_path: &Path, score: f64) -> Result<()> {
    let sidecar = sidecar_path(image_path);
    if !sidecar.exists() {
        return Ok(());
    }
    let text = std::fs::read_to_string(&sidecar)
        .with_context(|| format!("reading sidecar {}", sidecar.display()))?;
    let mut meta: crate::imaging::metadata::GenerationMetadata =
        serde_json::from_str(&text).with_context(|| format!("parsing sidecar {}", sidecar.display()))?;
    meta.score = Some(score);
    let json = meta
        .to_json_pretty()
        .with_context(|| format!("serialising sidecar {}", sidecar.display()))?;
    std::fs::write(&sidecar, json)
        .with_context(|| format!("writing sidecar {}", sidecar.display()))?;
    Ok(())
}

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
    // ETCH-1 (6.7.0): when `--etch` is on — L1 embeds the pixel etch into the buffer, L0 derives the
    // manifest written into the PNG `etch` tEXt chunk + JSON sidecar. Off by default → both are no-ops and
    // the path is byte-identical. (RGB save carries no alpha, so L1 embeds everywhere.)
    let l1 = crate::etch::l1_embed_rgb(buf, width, height, None, metadata);
    let buf: &[u8] = l1.as_deref().unwrap_or(buf);
    let etch_json = crate::etch::l0_manifest_json(metadata);
    save_rgb_u8_inner(buf, width, height, path, Some(metadata), etch_json.as_deref())?;
    // L3: enqueue this image for CLIP fingerprinting (drained once at end-of-run in `etch::l3_flush`).
    crate::etch::l3_enqueue(path, metadata);
    // Write the JSON sidecar. Best-effort.
    let json_path = sidecar_path(path);
    match metadata.to_json_pretty() {
        Ok(json) => {
            let json = match &etch_json {
                Some(e) => inject_etch_into_sidecar(&json, e),
                None => json,
            };
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

/// Insert the L0 `etch` object into the sidecar JSON (best-effort — returns the original on any parse
/// failure so a malformed injection never loses the recipe).
pub(crate) fn inject_etch_into_sidecar(pretty_json: &str, etch_json: &str) -> String {
    match (
        serde_json::from_str::<serde_json::Value>(pretty_json),
        serde_json::from_str::<serde_json::Value>(etch_json),
    ) {
        (Ok(mut v), Ok(e)) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("etch".to_string(), e);
            }
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| pretty_json.to_string())
        }
        _ => pretty_json.to_string(),
    }
}

pub(crate) fn save_rgb_u8_inner(
    buf: &[u8],
    width: u32,
    height: u32,
    path: &Path,
    metadata: Option<&GenerationMetadata>,
    etch_json: Option<&str>,
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
    // v0.19: extension-driven format routing. `.png` (the default)
    // takes the metadata-aware PNG tEXt chunk path; `.webp` and
    // other extensions fall through to the image-crate save
    // (no embedded chunk — WebP's EXIF / XMP slots aren't part
    // of the A1111 / Civitai metadata convention). The JSON
    // sidecar is written for both formats by the caller's
    // `save_rgb_u8_with_metadata` wrapper, so the recipe stays
    // recoverable via `plakat metadata` / `plakat clone`.
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    let is_png = ext.as_deref() == Some("png");

    match (metadata, is_png) {
        (None, _) | (Some(_), false) => {
            // Fast path — no metadata embed. Defer to the `image`
            // crate which auto-detects format from the path's
            // extension. WebP outputs land here regardless of the
            // metadata arg.
            let img: ImageBuffer<Rgb<u8>, _> =
                ImageBuffer::from_raw(width, height, buf.to_vec())
                    .ok_or_else(|| anyhow!("failed to construct ImageBuffer"))?;
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("creating output dir {}", parent.display())
                    })?;
                }
            }
            img.save(path)
                .with_context(|| format!("writing {}", path.display()))?;
            Ok(())
        }
        (Some(meta), true) => write_png_with_text_chunk(buf, width, height, path, meta, etch_json),
    }
}

fn write_png_with_text_chunk(
    buf: &[u8],
    width: u32,
    height: u32,
    path: &Path,
    metadata: &GenerationMetadata,
    etch_json: Option<&str>,
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
    // ETCH-1 L0: the `etch` provenance chunk (6.7.0), when `--etch` is on.
    if let Some(etch) = etch_json {
        encoder
            .add_text_chunk("etch".to_string(), etch.to_string())
            .with_context(|| "add `etch` tEXt chunk")?;
    }
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
        let json_path = tmp.path().join("test.png.json");

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

    // v0.19 — WebP output via extension-driven routing.

    #[test]
    fn output_format_from_str_round_trip() {
        assert_eq!(
            "png".parse::<OutputFormat>().unwrap(),
            OutputFormat::Png
        );
        assert_eq!(
            "webp".parse::<OutputFormat>().unwrap(),
            OutputFormat::Webp
        );
        // Case-insensitive.
        assert_eq!(
            "WEBP".parse::<OutputFormat>().unwrap(),
            OutputFormat::Webp
        );
        // Unknown bails with the supported list.
        let err = "jpeg".parse::<OutputFormat>().unwrap_err();
        assert!(format!("{err}").contains("png, webp"));
    }

    #[test]
    fn output_format_extension_matches_variant() {
        assert_eq!(OutputFormat::Png.extension(), "png");
        assert_eq!(OutputFormat::Webp.extension(), "webp");
    }

    #[test]
    fn save_rgb_u8_writes_webp_when_path_has_webp_extension() {
        let tmp = tempfile::tempdir().unwrap();
        let webp_path = tmp.path().join("plain.webp");
        // 4×4 solid-red so the encoded file is non-trivial.
        let buf: Vec<u8> = (0..16).flat_map(|_| [255u8, 0, 0]).collect();
        save_rgb_u8(&buf, 4, 4, &webp_path).unwrap();
        assert!(webp_path.exists());
        // Round-trip read: the file is actually a WebP (image crate
        // would error otherwise), and it decodes to the same 4×4
        // RGB tensor we wrote.
        let decoded = image::open(&webp_path).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (4, 4));
        // Lossy compression — sample one pixel and confirm red is
        // dominant. Don't insist on byte-exact (WebP at default
        // quality isn't lossless).
        let p = decoded.get_pixel(2, 2).0;
        assert!(p[0] > 200, "red channel should dominate, got {p:?}");
        assert!(p[1] < 80, "green channel should be near-zero, got {p:?}");
        assert!(p[2] < 80, "blue channel should be near-zero, got {p:?}");
    }

    #[test]
    fn save_with_metadata_skips_chunk_for_webp_writes_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let webp_path = tmp.path().join("test.webp");
        let sidecar = tmp.path().join("test.webp.json");
        let buf: Vec<u8> = (0..4).flat_map(|_| [0u8, 128, 255]).collect();
        let meta = GenerationMetadata::new(
            "a fox",
            "sd15",
            42,
            28,
            7.5,
            "euler-a",
            2,
            2,
        );
        save_rgb_u8_with_metadata(&buf, 2, 2, &webp_path, &meta).unwrap();
        assert!(webp_path.exists());
        // The PNG tEXt chunk path doesn't apply to WebP — but the
        // JSON sidecar SHOULD still be written (the recipe stays
        // recoverable via `plakat metadata --json-only`).
        assert!(sidecar.exists());
        // Confirm the chunk-read returns "no chunk" on the WebP.
        // The shadowed test helper above wraps the super:: function's
        // `Result<Option<_>>` into `Option<_>`, treating both error
        // and "no chunk" as the absence we expect for a WebP file.
        assert!(
            read_parameters_chunk(&webp_path).is_none(),
            "WebP outputs must not carry the PNG tEXt chunk"
        );
    }
}
