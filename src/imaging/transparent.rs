//! Color-key transparency: read upper-left pixel, make every matching pixel
//! fully transparent. Pure CPU image processing; no model load.

use anyhow::{Result, anyhow};
use image::{Rgba, RgbaImage};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct Report {
    pub width: u32,
    pub height: u32,
    pub key_rgb: [u8; 3],
    pub transparent_pixels: u64,
    pub total_pixels: u64,
}

/// Make every pixel within `tolerance` (per-channel max diff) of the upper-left
/// pixel fully transparent. Matched pixels keep their RGB; alpha is set to 0.
/// Non-matched pixels preserve their original alpha (255 for RGB inputs).
pub fn make_transparent(in_path: &Path, out_path: &Path, tolerance: u8) -> Result<Report> {
    if let Some(ext) = out_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
    {
        if matches!(ext.as_str(), "jpg" | "jpeg" | "bmp") {
            return Err(anyhow!(
                "output .{ext} doesn't support alpha — use a .png or .webp output path"
            ));
        }
    }

    let img = image::open(in_path)?.to_rgba8();
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return Err(anyhow!("empty image: {}", in_path.display()));
    }

    let (out, hit, key_rgb) = chroma_key_image(img, tolerance);
    let (kr, kg, kb) = (key_rgb[0], key_rgb[1], key_rgb[2]);

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    out.save(out_path)?;

    Ok(Report {
        width: w,
        height: h,
        key_rgb: [kr, kg, kb],
        transparent_pixels: hit,
        total_pixels: (w as u64) * (h as u64),
    })
}

/// In-memory variant of [`make_transparent`]: chroma-key the upper-
/// left pixel of `img` to alpha=0 using `tolerance` per-channel diff.
/// Returns the modified image, the number of pixels that matched, and
/// the keyed RGB.
///
/// Reused by `artefacts::compositing` as the auto-fallback when a
/// user-supplied artefact PNG has no alpha channel.
pub fn chroma_key_image(img: RgbaImage, tolerance: u8) -> (RgbaImage, u64, [u8; 3]) {
    let w = img.width();
    let h = img.height();
    if w == 0 || h == 0 {
        return (img, 0, [0, 0, 0]);
    }
    let key = img.get_pixel(0, 0).0;
    let (kr, kg, kb) = (key[0], key[1], key[2]);
    let tol = tolerance as i16;

    let mut out = RgbaImage::new(w, h);
    let mut hit: u64 = 0;
    for (x, y, px) in img.enumerate_pixels() {
        let [r, g, b, a] = px.0;
        let matches = (r as i16 - kr as i16).abs() <= tol
            && (g as i16 - kg as i16).abs() <= tol
            && (b as i16 - kb as i16).abs() <= tol;
        if matches {
            out.put_pixel(x, y, Rgba([r, g, b, 0]));
            hit += 1;
        } else {
            out.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }
    (out, hit, [kr, kg, kb])
}
