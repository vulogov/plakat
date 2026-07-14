//! Image loading + thumbnail cache (RFC PHOTOS-1 §3, §24).
//!
//! Standard raster formats go through `image`; camera RAW through `rawloader` + a fast 2×2-quad
//! demosaic (each Bayer quad → one RGB pixel, with black/white-level normalisation, white balance,
//! and gamma 2.2). Thumbnails are cached under the XDG cache dir, keyed by `sha256(path + mtime)`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb};

use super::library::is_raw_ext;

fn ext_lower(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

/// Decode any supported image to an RGB `DynamicImage`. RAW files are fully demosaiced (2×2-quad).
pub fn load(path: &Path) -> Result<DynamicImage> {
    if is_raw_ext(&ext_lower(path)) {
        Ok(DynamicImage::ImageRgb8(demosaic_raw(path, usize::MAX)?))
    } else {
        Ok(image::open(path).with_context(|| format!("decoding {}", path.display()))?)
    }
}

/// Decode a thumbnail (longest side ≈ `size`). For RAW, the demosaic is strided so only ~`size`
/// output pixels are computed (the RFC's fast path) — full debayer is reserved for `load`.
pub fn thumbnail(path: &Path, size: u32) -> Result<DynamicImage> {
    if is_raw_ext(&ext_lower(path)) {
        let img = demosaic_raw(path, size as usize)?;
        Ok(DynamicImage::ImageRgb8(img).thumbnail(size, size))
    } else {
        Ok(image::open(path)
            .with_context(|| format!("decoding {}", path.display()))?
            .thumbnail(size, size))
    }
}

/// 2×2-quad demosaic of a RAW file → half-resolution (or strided) RGB. `target_long` bounds the
/// output's long side by striding whole quads (≈ nearest-neighbour downscale in the Bayer domain).
fn demosaic_raw(path: &Path, target_long: usize) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let raw = rawloader::decode_file(path)
        .map_err(|e| anyhow::anyhow!("RAW decode {}: {e:?}", path.display()))?;
    let data = match raw.data {
        rawloader::RawImageData::Integer(d) => d,
        rawloader::RawImageData::Float(_) => {
            anyhow::bail!("float RAW not supported for {}", path.display())
        }
    };
    let (w, h, cfa) = (raw.width, raw.height, raw.cfa);
    anyhow::ensure!(w >= 2 && h >= 2 && data.len() >= w * h, "malformed RAW dims");

    // Stride whole quads so the output long side is ≈ target_long (each quad = 1 output pixel).
    let qw = w / 2;
    let qh = h / 2;
    let long = qw.max(qh);
    let stride = if target_long == usize::MAX || long <= target_long {
        1
    } else {
        (long / target_long).max(1)
    };
    let ow = (qw / stride).max(1);
    let oh = (qh / stride).max(1);

    // Per-channel (R,G,B) black/white levels + white-balance, normalised to green.
    let bl = raw.blacklevels;
    let wl = raw.whitelevels;
    let wb_g = if raw.wb_coeffs[1] > 0.0 { raw.wb_coeffs[1] } else { 1.0 };
    let wb = [
        raw.wb_coeffs[0] / wb_g,
        1.0,
        raw.wb_coeffs[2] / wb_g,
    ];
    let norm = |v: u16, c: usize| -> f32 {
        let b = bl[c] as f32;
        let range = (wl[c] as f32 - b).max(1.0);
        (((v as f32 - b) / range).clamp(0.0, 1.0) * wb[c]).clamp(0.0, 1.0)
    };
    let to_srgb = |lin: f32| (lin.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u8;

    let mut out = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(ow as u32, oh as u32);
    for oy in 0..oh {
        let qy = (oy * stride).min(qh - 1);
        for ox in 0..ow {
            let qx = (ox * stride).min(qw - 1);
            // The 2×2 quad's four sensor pixels.
            let (r0, c0) = (qy * 2, qx * 2);
            let mut acc = [0f32; 3];
            let mut cnt = [0u32; 3];
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let (row, col) = (r0 + dr, c0 + dc);
                    let color = cfa.color_at(row, col);
                    let ch = if color <= 2 { color } else { 1 }; // emerald → green
                    acc[ch] += norm(data[row * w + col], ch);
                    cnt[ch] += 1;
                }
            }
            let px = Rgb([
                to_srgb(acc[0] / cnt[0].max(1) as f32),
                to_srgb(acc[1] / cnt[1].max(1) as f32),
                to_srgb(acc[2] / cnt[2].max(1) as f32),
            ]);
            out.put_pixel(ox as u32, oy as u32, px);
        }
    }
    Ok(out)
}

/// XDG cache dir for photo thumbnails: `<cache>/plakat/photos/thumbs`.
pub fn thumb_cache_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "plakat")
        .map(|d| d.cache_dir().join("photos").join("thumbs"))
        .unwrap_or_else(|| std::env::temp_dir().join("plakat-photos-thumbs"))
}

/// Cache path for a thumbnail: `sha256(abs_path + mtime)` → `<hex>.png`.
pub fn thumb_cache_path(path: &Path, size: u32) -> PathBuf {
    use sha2::{Digest, Sha256};
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(size.to_le_bytes());
    hasher.update(mtime.to_le_bytes());
    let hex = format!("{:x}", hasher.finalize());
    thumb_cache_dir().join(format!("{hex}.png"))
}

/// Return a cached thumbnail path, rendering + caching it on miss.
pub fn get_or_render_thumb(path: &Path, size: u32) -> Result<PathBuf> {
    let cache = thumb_cache_path(path, size);
    if cache.exists() {
        return Ok(cache);
    }
    let thumb = thumbnail(path, size)?;
    if let Some(parent) = cache.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    thumb.save(&cache).with_context(|| format!("caching thumb {}", cache.display()))?;
    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_thumbnail_and_cache_key() {
        let dir = std::env::temp_dir().join(format!("plakat-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.png");
        DynamicImage::ImageRgb8(ImageBuffer::from_pixel(64, 32, Rgb([10, 20, 30]))).save(&p).unwrap();

        let th = thumbnail(&p, 16).unwrap();
        assert!(th.width() <= 16 && th.height() <= 16);
        // Cache key is stable + mtime-sensitive (same call → same path).
        assert_eq!(thumb_cache_path(&p, 16), thumb_cache_path(&p, 16));
        assert_ne!(thumb_cache_path(&p, 16), thumb_cache_path(&p, 32));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
