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

/// Parse `medium=<kind>` from a naturalize **spec** (`generate --naturalize`, scenario `naturalize:`,
/// compiled `naturalize:`) → the painting [`Medium`]. `None` when absent or not a painting medium.
pub fn medium_from_spec(spec: &str) -> Option<Medium> {
    spec.split_whitespace().find_map(|t| t.strip_prefix("medium=")).and_then(Medium::parse)
}

/// Parse `brush=<N>` stroke strength from a spec; `None` if absent (caller defaults to 0.6).
pub fn brush_from_spec(spec: &str) -> Option<f32> {
    spec.split_whitespace().find_map(|t| t.strip_prefix("brush=").and_then(|v| v.parse::<f32>().ok()))
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
        // Wet media stay SOFT by design — no placed strokes; pigment pools at edges and bleeds a
        // little along the forms, over the paper granulation (`--paper`). Watercolor is diffuse.
        Medium::Watercolor => {
            let mut out = stroke_smear(src, &angle, &mag, w, h, 2, 0.5 * s);
            edge_darken(&mut out, &mag, w, h, 0.55 * s);
            out
        }
        // Thick opaque media: STROKE-BASED RENDERING — place discrete, textured, gradient-aligned
        // brush marks (the shared engine below), tuned per medium. Oil also gets impasto highlights.
        Medium::Oil => {
            // Kuwahara first → flat painted regions; strokes then sample those cleaner colours.
            let flat = kuwahara(src, 2);
            let mut out = render_strokes(&flat, &stroke_style(medium), s);
            let ol: Vec<f32> = out.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
            impasto(&mut out, &ol, &angle, w, h, 0.5 * s);
            out
        }
        Medium::Gouache => render_strokes(src, &stroke_style(medium), s),
        Medium::Pastel => {
            let mut out = render_strokes(src, &stroke_style(medium), s);
            chalk_grain(&mut out, w, h, 0.5 * s);
            out
        }
        // Ink: crisp, high-contrast, with directional hatching in the shadows.
        Medium::Ink => {
            let mut out = src.clone();
            hatching(&mut out, &luma, &angle, w, h, 0.8 * s);
            edge_darken(&mut out, &mag, w, h, 0.4 * s);
            out
        }
    }
}

/// Per-medium brush parameters for the stroke-based renderer.
#[derive(Debug, Clone, Copy)]
struct StrokeStyle {
    width: f32,      // brush width (px)
    length: f32,     // max stroke length (px)
    spacing: f32,    // seed grid spacing (< width → overlapping strokes)
    opacity: f32,    // per-stroke coverage over the under-painting
    bristle: f32,    // along-stroke bristle streak strength (0..1)
    edge_soft: f32,  // cross-stroke alpha falloff (soft vs crisp edge)
    color_tol: f32,  // stop growing a stroke when the luma deviates this much (holds boundaries)
    base_blur: i32,  // under-painting blur radius (shows between strokes)
}

fn stroke_style(m: Medium) -> StrokeStyle {
    match m {
        // Long, textured, curved strokes over a soft under-painting — classic oil.
        Medium::Oil => StrokeStyle { width: 5.0, length: 18.0, spacing: 3.2, opacity: 0.92, bristle: 0.5, edge_soft: 0.45, color_tol: 42.0, base_blur: 3 },
        // Shorter, flatter, opaque, crisper edges — gouache is matte and covering.
        Medium::Gouache => StrokeStyle { width: 4.0, length: 10.0, spacing: 3.0, opacity: 1.0, bristle: 0.22, edge_soft: 0.18, color_tol: 32.0, base_blur: 2 },
        // Short, soft, chalky strokes with the under-painting showing through — pastel.
        Medium::Pastel => StrokeStyle { width: 3.0, length: 8.0, spacing: 2.4, opacity: 0.7, bristle: 0.75, edge_soft: 0.6, color_tol: 55.0, base_blur: 1 },
        // Not stroke-filled (watercolor/ink handle themselves) — a neutral fallback.
        _ => StrokeStyle { width: 4.0, length: 12.0, spacing: 3.0, opacity: 0.85, bristle: 0.4, edge_soft: 0.4, color_tol: 40.0, base_blur: 2 },
    }
}

/// **Stroke-based rendering** (Hertzmann-style, weight-free + deterministic): lay discrete brush
/// marks on the canvas. Each mark is seeded on an overlapping grid (jittered), grown **along the
/// isophote** (perpendicular to the luma gradient, so it follows the form) until the colour drifts
/// (a boundary), then stamped as an oriented, bristle-textured, soft-edged capsule in the colour
/// sampled at its seed. The strokes sit over a blurred under-painting, so gaps read as
/// brushwork, not a filter — and different regions (sky vs stone vs figures) get differently
/// oriented strokes. `strength` cross-blends the painted canvas with the original.
fn render_strokes(src: &RgbImage, st: &StrokeStyle, strength: f32) -> RgbImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let src_f: Vec<[f32; 3]> = src.pixels().map(|p| [p.0[0] as f32, p.0[1] as f32, p.0[2] as f32]).collect();
    let luma: Vec<f32> = src_f.iter().map(|c| lum(c[0], c[1], c[2])).collect();
    let (_, angle) = sobel(&luma, w, h);

    // Under-painting: a blurred copy per channel, so bare gaps between strokes look painted.
    let mut canvas: Vec<[f32; 3]> = {
        let ch = |k: usize| box_blur_gray(&src_f.iter().map(|c| c[k]).collect::<Vec<_>>(), w, h, st.base_blur);
        let (r, g, b) = (ch(0), ch(1), ch(2));
        (0..w * h).map(|i| [r[i], g[i], b[i]]).collect()
    };

    let idx = |x: i32, y: i32| (y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize;
    let step = st.spacing.max(1.0);
    let half_len = st.length * 0.5;
    let half_w = (st.width * 0.5).max(0.6);

    // Seed grid (deterministic order → later strokes layer over earlier, like real painting).
    let mut gy = step * 0.5;
    while gy < h as f32 {
        let mut gx = step * 0.5;
        while gx < w as f32 {
            let jx = (noise(gx as u32, gy as u32) - 0.5) * step;
            let jy = (noise((gy as u32) ^ 0x1234, (gx as u32) ^ 0x5678) - 0.5) * step;
            let (sxf, syf) = ((gx + jx).clamp(0.0, w as f32 - 1.0), (gy + jy).clamp(0.0, h as f32 - 1.0));
            let seed = idx(sxf as i32, syf as i32);
            let color = src_f[seed];
            let seed_l = luma[seed];

            // Grow a poly-line both ways from the seed, following the isophote field.
            let mut pts: Vec<(f32, f32)> = vec![(sxf, syf)];
            for &sign in &[1.0f32, -1.0] {
                let (mut x, mut y) = (sxf, syf);
                let (mut pdx, mut pdy) = {
                    let a = angle[seed] + std::f32::consts::FRAC_PI_2;
                    (a.cos() * sign, a.sin() * sign)
                };
                let mut t = 0.0;
                while t < half_len {
                    let i = idx(x as i32, y as i32);
                    let a = angle[i] + std::f32::consts::FRAC_PI_2;
                    let (mut dx, mut dy) = (a.cos(), a.sin());
                    if dx * pdx + dy * pdy < 0.0 { dx = -dx; dy = -dy; } // keep direction consistent
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0.0 || ny < 0.0 || nx >= w as f32 || ny >= h as f32 {
                        break;
                    }
                    if (luma[idx(nx as i32, ny as i32)] - seed_l).abs() > st.color_tol {
                        break; // hit a boundary — a stroke stays within one region
                    }
                    pts.push((nx, ny));
                    x = nx; y = ny; pdx = dx; pdy = dy; t += 1.0;
                }
            }

            // Stamp the stroke: an oriented capsule of `width`, bristle-streaked, tapering at the ends.
            let n = pts.len() as f32;
            for (k, &(cx, cy)) in pts.iter().enumerate() {
                // End taper: full alpha in the middle, fading over the last ~30%.
                let along = (k as f32 / n.max(1.0) - 0.5).abs() * 2.0; // 0 centre → 1 ends
                let taper = (1.0 - ((along - 0.7).max(0.0) / 0.3)).clamp(0.0, 1.0);
                // Cross-stroke normal for bristle streaks.
                let (dx, dy) = if pts.len() > 1 {
                    let j = k.min(pts.len() - 2);
                    (pts[j + 1].0 - pts[j].0, pts[j + 1].1 - pts[j].1)
                } else {
                    (1.0, 0.0)
                };
                let (ndx, ndy) = (-dy, dx); // across the stroke
                let r = half_w.ceil() as i32;
                for oy in -r..=r {
                    for ox in -r..=r {
                        let (fx, fy) = (cx + ox as f32, cy + oy as f32);
                        if fx < 0.0 || fy < 0.0 || fx >= w as f32 || fy >= h as f32 {
                            continue;
                        }
                        // Perpendicular distance from the stroke centre-line.
                        let perp = (ox as f32 * ndx + oy as f32 * ndy) / (ndx * ndx + ndy * ndy).sqrt().max(1e-3);
                        let pd = perp.abs() / half_w;
                        if pd > 1.0 {
                            continue;
                        }
                        // Soft cross-section + bristle streaks (noise along the stroke normal).
                        let soft = (1.0 - pd).powf(1.0 + st.edge_soft * 2.0);
                        let bristle = 1.0 - st.bristle * noise((perp * 3.0) as u32 ^ (k as u32), (cx + cy) as u32);
                        let alpha = (st.opacity * soft * bristle * taper).clamp(0.0, 1.0);
                        let px = &mut canvas[idx(fx as i32, fy as i32)];
                        for c in 0..3 {
                            px[c] = px[c] * (1.0 - alpha) + color[c] * alpha;
                        }
                    }
                }
            }
            gx += step;
        }
        gy += step;
    }

    // Cross-blend the painted canvas with the original by strength.
    let s = strength.clamp(0.0, 1.0);
    let mut out = src.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let px = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px.0[c] = (src_f[i][c] * (1.0 - s) + canvas[i][c] * s).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
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
    fn spec_parses_medium_and_brush() {
        assert_eq!(medium_from_spec("photo paper=0.6 medium=watercolor brush=0.7"), Some(Medium::Watercolor));
        assert_eq!(brush_from_spec("photo medium=oil brush=0.7"), Some(0.7));
        assert_eq!(medium_from_spec("photo paper=0.6"), None); // no medium= → off
        assert_eq!(brush_from_spec("photo medium=oil"), None); // caller defaults to 0.6
    }

    #[test]
    fn deterministic() {
        let img = ramp();
        assert_eq!(apply_brush(&img, Medium::Watercolor, 0.7), apply_brush(&img, Medium::Watercolor, 0.7));
    }
}
