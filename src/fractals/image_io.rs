//! PNG output for fractals — writes the RGB buffer plus the embedded `fractalspec`
//! tEXt chunk so every render carries the recipe that reproduces it (`--fractal-clone`).
//! Mirrors `imaging::io::write_png_with_text_chunk`. RFC FRACTALS-1, Phase 1.

use anyhow::{Context, Result};
use std::path::Path;

use super::spec::{FractalSpec, SPEC_CHUNK_KEYWORD};

/// Save a packed `RGB8` buffer as a PNG with the spec embedded as a `fractalspec`
/// tEXt chunk (and a `Software` marker). Creates parent directories as needed.
pub fn save_png_with_spec(
    buf: &[u8],
    width: u32,
    height: u32,
    spec: &FractalSpec,
    path: &Path,
) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 3;
    if buf.len() != expected {
        anyhow::bail!(
            "fractal buffer size mismatch: got {}, expected {} for {}x{}",
            buf.len(),
            expected,
            width,
            height
        );
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output dir {}", parent.display()))?;
        }
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let spec_json = spec.to_json()?;
    encoder
        .add_text_chunk(SPEC_CHUNK_KEYWORD.to_string(), spec_json)
        .context("adding fractalspec tEXt chunk")?;
    encoder
        .add_text_chunk("Software".to_string(), "plakat fractals".to_string())
        .context("adding Software tEXt chunk")?;
    let mut writer = encoder.write_header().context("writing PNG header")?;
    writer.write_image_data(buf).context("writing PNG image data")?;
    writer.finish().context("finalizing PNG")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::{read_spec_chunk, FractalKind};

    #[test]
    fn save_then_clone_round_trips_the_spec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frac.png");
        let spec = FractalSpec {
            kind: FractalKind::Julia,
            width: 8,
            height: 6,
            julia_c: [-0.7, 0.27],
            ..FractalSpec::default()
        };
        let buf = vec![0u8; 8 * 6 * 3];
        save_png_with_spec(&buf, 8, 6, &spec, &path).unwrap();

        let recovered = read_spec_chunk(&path).unwrap().expect("chunk present");
        assert_eq!(recovered, spec);
    }

    #[test]
    fn plain_png_has_no_spec_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.png");
        // A PNG written by the image crate carries no fractalspec chunk.
        image::RgbImage::new(4, 4).save(&path).unwrap();
        assert!(read_spec_chunk(&path).unwrap().is_none());
    }

    #[test]
    fn buffer_size_mismatch_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.png");
        let spec = FractalSpec { width: 8, height: 8, ..FractalSpec::default() };
        assert!(save_png_with_spec(&[0u8; 10], 8, 8, &spec, &path).is_err());
    }
}
