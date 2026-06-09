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

/// Flood-fill the background to fully transparent, starting from the image
/// corners: a pixel joins the background when it's within `tolerance` (per
/// channel) of an already-background NEIGHBOUR. Smooth gradients / soft shadows
/// are removed (each step is small) while a sharp subject edge stops the fill —
/// robust on real, studio-lit renders where a single corner-colour key fails.
/// Matched pixels keep their RGB with alpha 0; everything else is untouched.
pub fn make_transparent(
    in_path: &Path,
    out_path: &Path,
    tolerance: u8,
    crop: bool,
) -> Result<Report> {
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

    let (mut out, hit, key_rgb) = flood_key_image(img, tolerance);
    let (kr, kg, kb) = (key_rgb[0], key_rgb[1], key_rgb[2]);

    // Crop to the non-transparent bounding box so a downstream compositor
    // (e.g. the artefact library) scales the *subject*, not the mostly-
    // transparent full frame — otherwise a centred subject lands tiny.
    if crop {
        if let Some((x0, y0, cw, ch)) = opaque_bbox(&out) {
            out = image::imageops::crop_imm(&out, x0, y0, cw, ch).to_image();
        }
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    out.save(out_path)?;

    Ok(Report {
        width: out.width(),
        height: out.height(),
        key_rgb: [kr, kg, kb],
        transparent_pixels: hit,
        total_pixels: (w as u64) * (h as u64),
    })
}

/// Bounding box `(x, y, w, h)` of pixels with non-zero alpha, or `None` if the
/// image is fully transparent.
fn opaque_bbox(img: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut any = false;
    for (x, y, px) in img.enumerate_pixels() {
        if px.0[3] != 0 {
            any = true;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    any.then(|| (x0, y0, x1 - x0 + 1, y1 - y0 + 1))
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

/// Corner flood-fill variant of [`chroma_key_image`]: removes only the
/// background **connected to the four corners**, growing a pixel into the
/// background when it's within `tolerance` (per channel) of an already-background
/// *neighbour*. Follows smooth gradients / soft shadows yet stops at a sharp
/// subject edge, and the subject's interior keeps colours that happen to match
/// the background. Used by [`make_transparent`].
pub fn flood_key_image(img: RgbaImage, tolerance: u8) -> (RgbaImage, u64, [u8; 3]) {
    use std::collections::VecDeque;
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return (img, 0, [0, 0, 0]);
    }
    let key = img.get_pixel(0, 0).0;
    let tol = tolerance as i16;
    let close = |a: &[u8; 4], b: &[u8; 4]| {
        (a[0] as i16 - b[0] as i16).abs() <= tol
            && (a[1] as i16 - b[1] as i16).abs() <= tol
            && (a[2] as i16 - b[2] as i16).abs() <= tol
    };
    // Seed constraint: a background pixel must ALSO stay within this generous
    // distance of the corner colour. This lets the fill follow a chroma gradient
    // / soft shadow but stops it creeping through an anti-aliased edge into a
    // far-coloured subject — which the neighbour test alone would consume (e.g.
    // a smoothly-shaded red apple, all within `tolerance` step-to-step).
    let seed_tol = (tol * 4).clamp(40, 96);
    let within_seed = |p: &[u8; 4]| {
        (p[0] as i16 - key[0] as i16).abs() <= seed_tol
            && (p[1] as i16 - key[1] as i16).abs() <= seed_tol
            && (p[2] as i16 - key[2] as i16).abs() <= seed_tol
    };
    let mut bg = vec![false; w * h];
    let mut q: VecDeque<(usize, usize)> = VecDeque::new();
    for (cx, cy) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
        let p = cy * w + cx;
        if !bg[p] {
            bg[p] = true;
            q.push_back((cx, cy));
        }
    }
    while let Some((x, y)) = q.pop_front() {
        let cur = img.get_pixel(x as u32, y as u32).0;
        let mut ns = [(0usize, 0usize); 4];
        let mut n = 0;
        if x > 0 {
            ns[n] = (x - 1, y);
            n += 1;
        }
        if x + 1 < w {
            ns[n] = (x + 1, y);
            n += 1;
        }
        if y > 0 {
            ns[n] = (x, y - 1);
            n += 1;
        }
        if y + 1 < h {
            ns[n] = (x, y + 1);
            n += 1;
        }
        for &(nx, ny) in &ns[..n] {
            let np = ny * w + nx;
            if bg[np] {
                continue;
            }
            let nc = img.get_pixel(nx as u32, ny as u32).0;
            if close(&cur, &nc) && within_seed(&nc) {
                bg[np] = true;
                q.push_back((nx, ny));
            }
        }
    }
    let mut out = img;
    let mut hit = 0u64;
    for (i, &is_bg) in bg.iter().enumerate() {
        if is_bg {
            out.get_pixel_mut((i % w) as u32, (i / w) as u32).0[3] = 0;
            hit += 1;
        }
    }
    (out, hit, [key[0], key[1], key[2]])
}
