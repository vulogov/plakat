//! Weight-free **painterly brush-stroke** synthesis (RFC — naturalize media pass).
//!
//! `paper_texture` (mod.rs) already gives wet media their paper tooth + pigment granulation — the
//! *surface*. This module adds the thing that grain alone can't fake: **human-like brush strokes**
//! that follow the forms. It's classic non-photorealistic rendering, all deterministic image
//! processing (Sobel gradient field, Kuwahara edge-preserving flatten, gradient-aligned smear,
//! edge pooling, hatching) — no weights, so it composes with the rest of naturalize and never
//! drifts a cohesive image the way an img2img repaint would.
//!
//! Strokes run **along the isophotes** (perpendicular to the luma gradient), the way a painter
//! lays pigment along a form's contour. Opt-in per medium (`--medium <kind>` / auto-detected);
//! each medium composes the building blocks differently.

use image::RgbImage;

use super::{box_blur_gray, lum, noise};

/// A hand painting medium. Each drives a distinct composition of the building blocks below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Medium {
    Watercolor,
    Oil,
    Gouache,
    Ink,
    Pastel,
}

impl Medium {
    /// Parse a `--medium` value (accepts the common spellings). `None` for non-painterly media.
    pub fn parse(s: &str) -> Option<Medium> {
        match s.trim().to_ascii_lowercase().as_str() {
            "watercolor" | "watercolour" | "ink-wash" | "ink wash" => Some(Medium::Watercolor),
            "oil" | "oil painting" | "acrylic" => Some(Medium::Oil),
            "gouache" => Some(Medium::Gouache),
            "ink" | "pen" | "pen and ink" => Some(Medium::Ink),
            "pastel" | "chalk" | "charcoal" => Some(Medium::Pastel),
            _ => None,
        }
    }

    /// Detect a medium named anywhere in a descriptive style phrase (e.g. the CLIP-detected
    /// "soft wet-on-wet watercolor illustration…" or a `--style` string). `None` if none matches.
    pub fn detect(s: &str) -> Option<Medium> {
        let l = s.to_lowercase();
        let has = |k: &str| l.contains(k);
        if has("watercolor") || has("watercolour") || has("ink-wash") || has("ink wash") || has("wet-on-wet") {
            Some(Medium::Watercolor)
        } else if has("oil") || has("acrylic") || has("impasto") {
            Some(Medium::Oil)
        } else if has("gouache") {
            Some(Medium::Gouache)
        } else if has("pen and ink") || has("cross-hatch") || has("ink drawing") || has("pen-and-ink") {
            Some(Medium::Ink)
        } else if has("pastel") || has("chalk") || has("charcoal") {
            Some(Medium::Pastel)
        } else {
            None
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Medium::Watercolor => "watercolor",
            Medium::Oil => "oil",
            Medium::Gouache => "gouache",
            Medium::Ink => "ink",
            Medium::Pastel => "pastel",
        }
    }
}

/// Apply the medium's brush-stroke pass at `strength` (0..1). Returns a new image; `strength <= 0`
/// (or a 1×1 image) is a clone. Deterministic.
pub fn apply_brush(src: &RgbImage, medium: Medium, strength: f32) -> RgbImage {
    let s = strength.clamp(0.0, 1.0);
    let (w, h) = (src.width() as usize, src.height() as usize);
    if s <= 0.0 || w < 3 || h < 3 {
        return src.clone();
    }
    let luma: Vec<f32> = src.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    let (mag, angle) = sobel(&luma, w, h);
    match medium {
        // Wet media: no flattening — pigment pools at edges and bleeds slightly along the forms.
        Medium::Watercolor => {
            let mut out = stroke_smear(src, &angle, &mag, w, h, 2, 0.5 * s);
            edge_darken(&mut out, &mag, w, h, 0.55 * s);
            out
        }
        // Thick media: flatten into painted regions, then drag directional strokes + raise impasto.
        Medium::Oil => {
            let flat = kuwahara(src, 3);
            let mut out = stroke_smear(&flat, &angle, &mag, w, h, 4, 0.9 * s);
            impasto(&mut out, &luma, &angle, w, h, 0.5 * s);
            out
        }
        Medium::Gouache => {
            let flat = kuwahara(src, 2);
            stroke_smear(&flat, &angle, &mag, w, h, 3, 0.7 * s)
        }
        // Ink: crisp, high-contrast, with directional hatching in the shadows.
        Medium::Ink => {
            let mut out = src.clone();
            hatching(&mut out, &luma, &angle, w, h, 0.8 * s);
            edge_darken(&mut out, &mag, w, h, 0.4 * s);
            out
        }
        // Pastel: soft directional strokes with a heavier chalky grain.
        Medium::Pastel => {
            let mut out = stroke_smear(src, &angle, &mag, w, h, 3, 0.6 * s);
            chalk_grain(&mut out, w, h, 0.5 * s);
            out
        }
    }
}

/// Sobel gradient magnitude + angle of the luma field. Angle is the gradient direction (radians);
/// strokes run perpendicular to it.
fn sobel(luma: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let at = |x: i32, y: i32| luma[(y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize];
    let mut mag = vec![0f32; w * h];
    let mut ang = vec![0f32; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = -at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1)
                + at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1)
                + at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1);
            let i = y as usize * w + x as usize;
            mag[i] = (gx * gx + gy * gy).sqrt();
            ang[i] = gy.atan2(gx);
        }
    }
    (mag, ang)
}

/// Drag colour **along the stroke direction** (perpendicular to the gradient) by averaging a short
/// line of samples — the core "brush stroke". `half` = samples each side; `amount` blends the
/// smeared value over the original. Cross-stroke noise breaks the line into individual bristles.
fn stroke_smear(src: &RgbImage, angle: &[f32], mag: &[f32], w: usize, h: usize, half: i32, amount: f32) -> RgbImage {
    let mut out = src.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // Stroke runs perpendicular to the gradient; flatter regions (low mag) get a longer,
            // freer stroke, hard edges keep their crispness (short stroke).
            let dir = angle[i] + std::f32::consts::FRAC_PI_2;
            let (dx, dy) = (dir.cos(), dir.sin());
            // Per-stroke bristle jitter so the line isn't mechanically straight.
            let jitter = (noise(x as u32, y as u32) - 0.5) * 0.6;
            let edge_hold = (1.0 - (mag[i] / 90.0)).clamp(0.2, 1.0); // hold hard edges
            let reach = (half as f32 * edge_hold).max(1.0);
            let (mut r, mut g, mut b, mut n) = (0f32, 0f32, 0f32, 0f32);
            let mut t = -reach;
            while t <= reach {
                let sx = (x as f32 + t * dx + jitter * dy).round() as i32;
                let sy = (y as f32 + t * dy - jitter * dx).round() as i32;
                if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                    let p = src.get_pixel(sx as u32, sy as u32);
                    r += p.0[0] as f32;
                    g += p.0[1] as f32;
                    b += p.0[2] as f32;
                    n += 1.0;
                }
                t += 1.0;
            }
            if n > 0.0 {
                let px = out.get_pixel_mut(x as u32, y as u32);
                let orig = [px.0[0] as f32, px.0[1] as f32, px.0[2] as f32];
                let sm = [r / n, g / n, b / n];
                for c in 0..3 {
                    px.0[c] = (orig[c] * (1.0 - amount) + sm[c] * amount).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// Kuwahara edge-preserving flatten (radius `r`): each pixel takes the mean of its lowest-variance
/// quadrant → flat painted regions with crisp boundaries (the oil/gouache base). Deterministic.
fn kuwahara(src: &RgbImage, r: i32) -> RgbImage {
    let (w, h) = (src.width() as i32, src.height() as i32);
    let get = |x: i32, y: i32| {
        let p = src.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
        (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)
    };
    let mut out = src.clone();
    // The four quadrants (dx range, dy range) around the pixel.
    let quads = [(-r, 0, -r, 0), (0, r, -r, 0), (-r, 0, 0, r), (0, r, 0, r)];
    for y in 0..h {
        for x in 0..w {
            let (mut best_var, mut best) = (f32::MAX, (0f32, 0f32, 0f32));
            for &(x0, x1, y0, y1) in &quads {
                let (mut sr, mut sg, mut sb, mut sl, mut sl2, mut n) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
                for dy in y0..=y1 {
                    for dx in x0..=x1 {
                        let (r_, g_, b_) = get(x + dx, y + dy);
                        let l = lum(r_, g_, b_);
                        sr += r_; sg += g_; sb += b_; sl += l; sl2 += l * l; n += 1.0;
                    }
                }
                let var = sl2 / n - (sl / n) * (sl / n);
                if var < best_var {
                    best_var = var;
                    best = (sr / n, sg / n, sb / n);
                }
            }
            let px = out.get_pixel_mut(x as u32, y as u32);
            px.0[0] = best.0.clamp(0.0, 255.0) as u8;
            px.0[1] = best.1.clamp(0.0, 255.0) as u8;
            px.0[2] = best.2.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Watercolor **edge pooling**: pigment darkens where it dries at a boundary — darken proportional
/// to the local gradient magnitude.
fn edge_darken(out: &mut RgbImage, mag: &[f32], w: usize, h: usize, amount: f32) {
    for y in 0..h {
        for x in 0..w {
            let m = (mag[y * w + x] / 120.0).clamp(0.0, 1.0);
            if m <= 0.0 {
                continue;
            }
            let k = 1.0 - amount * m * 0.5; // up to ~50% darker at the strongest edges
            let px = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px.0[c] = (px.0[c] as f32 * k).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Oil **impasto**: raise the highlights along the stroke ridges (a fake specular from the paint's
/// relief). Brightens where the along-stroke luma is a local ridge.
fn impasto(out: &mut RgbImage, luma: &[f32], angle: &[f32], w: usize, h: usize, amount: f32) {
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let dir = angle[i] + std::f32::consts::FRAC_PI_2;
            let (dx, dy) = (dir.cos(), dir.sin());
            let sample = |t: f32| {
                let sx = (x as f32 + t * dx).round().clamp(0.0, w as f32 - 1.0) as usize;
                let sy = (y as f32 + t * dy).round().clamp(0.0, h as f32 - 1.0) as usize;
                luma[sy * w + sx]
            };
            let ridge = luma[i] - 0.5 * (sample(1.5) + sample(-1.5));
            if ridge > 2.0 {
                let lift = amount * (ridge / 30.0).clamp(0.0, 1.0) * 22.0;
                let px = out.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    px.0[c] = (px.0[c] as f32 + lift).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// Ink **hatching**: lay directional dark lines in the shadow regions (denser where darker).
fn hatching(out: &mut RgbImage, luma: &[f32], angle: &[f32], w: usize, h: usize, amount: f32) {
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let shadow = (1.0 - luma[i] / 130.0).clamp(0.0, 1.0); // only in the darks
            if shadow <= 0.0 {
                continue;
            }
            // Line coordinate across the stroke direction → periodic hatch.
            let dir = angle[i] + std::f32::consts::FRAC_PI_2;
            let across = x as f32 * dir.sin() - y as f32 * dir.cos();
            let hatch = ((across * 0.7).sin() * 0.5 + 0.5).powi(3); // thin dark lines
            let darken = amount * shadow * hatch * 60.0;
            let px = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px.0[c] = (px.0[c] as f32 - darken).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Pastel **chalk grain**: a coarser, brighter-biased grain that reads as dry pigment sitting on
/// tooth (distinct from the fine film grain naturalize already adds).
fn chalk_grain(out: &mut RgbImage, w: usize, h: usize, amount: f32) {
    // A slightly blurred noise field → clumps like chalk dust rather than per-pixel snow.
    let raw: Vec<f32> = (0..w * h).map(|k| noise((k % w) as u32, (k / w) as u32)).collect();
    let clumped = box_blur_gray(&raw, w, h, 1);
    for y in 0..h {
        for x in 0..w {
            let d = (clumped[y * w + x] - 0.5) * amount * 26.0;
            let px = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px.0[c] = (px.0[c] as f32 + d).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn ramp() -> RgbImage {
        // A diagonal luma ramp with a sharp vertical edge — gives the passes real gradients to work on.
        ImageBuffer::from_fn(48, 48, |x, _y| {
            let v = if x < 24 { 60 } else { 200 } as u8;
            Rgb([v, v, v])
        })
    }

    #[test]
    fn parse_maps_common_media() {
        assert_eq!(Medium::parse("watercolour"), Some(Medium::Watercolor));
        assert_eq!(Medium::parse("Oil Painting"), Some(Medium::Oil));
        assert_eq!(Medium::parse("charcoal"), Some(Medium::Pastel));
        assert_eq!(Medium::parse("photo"), None);
    }

    #[test]
    fn brush_is_a_noop_at_zero_and_changes_pixels_when_on() {
        let img = ramp();
        // strength 0 → identical.
        assert_eq!(apply_brush(&img, Medium::Oil, 0.0), img);
        // strength > 0 → the image actually changes (strokes/flatten/edge work).
        for m in [Medium::Watercolor, Medium::Oil, Medium::Gouache, Medium::Ink, Medium::Pastel] {
            let out = apply_brush(&img, m, 0.8);
            let changed = out.pixels().zip(img.pixels()).filter(|(a, b)| a != b).count();
            assert!(changed > 50, "{:?} changed only {changed} px", m);
            assert_eq!(out.dimensions(), img.dimensions());
        }
    }

    #[test]
    fn kuwahara_flattens_a_noisy_region_toward_its_mean() {
        // A noisy flat gray patch → Kuwahara should reduce its variance (flatten).
        let noisy: RgbImage = ImageBuffer::from_fn(32, 32, |x, y| {
            let n = ((x * 7 + y * 13) % 40) as i32 - 20;
            let v = (128 + n).clamp(0, 255) as u8;
            Rgb([v, v, v])
        });
        let var = |im: &RgbImage| {
            let ls: Vec<f32> = im.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
            let mean = ls.iter().sum::<f32>() / ls.len() as f32;
            ls.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / ls.len() as f32
        };
        assert!(var(&kuwahara(&noisy, 3)) < var(&noisy), "Kuwahara should flatten the patch");
    }

    #[test]
    fn deterministic() {
        let img = ramp();
        assert_eq!(apply_brush(&img, Medium::Watercolor, 0.7), apply_brush(&img, Medium::Watercolor, 0.7));
    }
}
