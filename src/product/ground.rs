//! The grounding algorithm (RFC PRODUCT-1 §"Ground", proven in G0.1) — deterministic, weight-free. Turns
//! a subject **alpha matte** into a physically-plausible contact shadow + floor reflection so the product
//! sits on the ground instead of floating. No GPU.

use image::{Rgba, RgbaImage};

/// Which contact-shadow model to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowKind {
    /// A soft symmetric pool under the base — the default packshot look.
    Soft,
    /// A directional cast that rakes away from the light (dramatic / low-camera).
    Hard,
    None,
}
impl ShadowKind {
    pub fn parse(s: Option<&str>) -> ShadowKind {
        match s.map(|x| x.to_ascii_lowercase()).as_deref() {
            Some("hard") | Some("cast") => ShadowKind::Hard,
            Some("none") | Some("off") => ShadowKind::None,
            _ => ShadowKind::Soft,
        }
    }
}

/// Which floor reflection to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionKind {
    /// A dim gloss reflection (default).
    Gloss,
    /// A brighter mirror reflection (glossy / lacquered products).
    Mirror,
    None,
}
impl ReflectionKind {
    pub fn parse(s: Option<&str>) -> ReflectionKind {
        match s.map(|x| x.to_ascii_lowercase()).as_deref() {
            Some("mirror") => ReflectionKind::Mirror,
            Some("none") | Some("off") => ReflectionKind::None,
            _ => ReflectionKind::Gloss,
        }
    }
    fn dim(self) -> f32 {
        match self {
            ReflectionKind::Gloss => 0.45,
            ReflectionKind::Mirror => 0.72,
            ReflectionKind::None => 0.0,
        }
    }
}

/// The horizontal light-offset factor from a key-light direction: light from the left throws the shadow
/// right (+x), from the right throws it left (−x), from the top/front stays symmetric.
pub fn key_offset(key_dir: Option<&str>) -> f32 {
    match key_dir.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("top-left") | Some("left") => 1.0,
        Some("top-right") | Some("right") => -1.0,
        _ => 0.0, // top / front / unknown → symmetric pool
    }
}

/// Separable box blur (repeat for a gaussian-ish penumbra) on an `w×h` f32 buffer.
fn box_blur(src: &[f32], w: usize, h: usize, r: usize, passes: usize) -> Vec<f32> {
    let mut buf = src.to_vec();
    let mut tmp = vec![0.0f32; w * h];
    for _ in 0..passes {
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut n) = (0.0, 0.0);
                for dx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                    acc += buf[y * w + dx];
                    n += 1.0;
                }
                tmp[y * w + x] = acc / n;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut n) = (0.0, 0.0);
                for dy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    acc += tmp[dy * w + x];
                    n += 1.0;
                }
                buf[y * w + x] = acc / n;
            }
        }
    }
    buf
}

/// Build the contact-shadow buffer (`0..1` darkness) from the subject alpha. Higher parts of the subject
/// project further from the light + down the floor and fade; the result is clamped to the ground plane so
/// the blur can't bleed a halo above the contact line (the G0 fix), then softened. `key` is [-1,1] from
/// [`key_offset`]; `soft` scales the penumbra.
pub fn contact_shadow(alpha: &[f32], w: usize, h: usize, ground_y: usize, kind: ShadowKind, key: f32, soft: f32) -> Vec<f32> {
    if kind == ShadowKind::None {
        return vec![0.0; w * h];
    }
    // per-model projection + penumbra, scaled to the canvas.
    let (mut kx, ky, blur_frac, opacity) = match kind {
        ShadowKind::Soft => (0.10 * key, 0.14, 0.028 * soft.max(0.2), 0.9),
        ShadowKind::Hard => (0.55 * key, 0.10, 0.013 * soft.max(0.2), 0.85),
        ShadowKind::None => unreachable!(),
    };
    // a hard cast under a symmetric top light still needs a direction to read — rake gently right.
    if kind == ShadowKind::Hard && key.abs() < 0.01 {
        kx = 0.4;
    }
    let mut acc = vec![0.0f32; w * h];
    let hspan = ground_y.max(1) as f32;
    for y in 0..ground_y.min(h) {
        for x in 0..w {
            let a = alpha[y * w + x];
            if a <= 0.01 {
                continue;
            }
            let d = (ground_y as f32 - y as f32).max(0.0); // height above the contact line
            let gx = x as f32 + kx * d;
            let gy = ground_y as f32 + ky * d;
            let fade = (1.0 - d / hspan * 0.6).clamp(0.0, 1.0);
            let (ix, iy) = (gx.round() as isize, gy.round() as isize);
            if ix >= 0 && (ix as usize) < w && iy >= 0 && (iy as usize) < h {
                acc[iy as usize * w + ix as usize] += a * fade;
            }
        }
    }
    let blur_r = ((h as f32 * blur_frac) as usize).max(2);
    let blurred = box_blur(&acc, w, h, blur_r, 3);
    blurred
        .iter()
        .enumerate()
        .map(|(i, v)| if i / w < ground_y { 0.0 } else { (v * opacity).clamp(0.0, 1.0) })
        .collect()
}

/// Build the floor-reflection layer: the subject flipped about the foot-line, foreshortened by the camera
/// `squash`, fading over `falloff` of the subject height. Returns an RGBA layer to composite under the
/// subject. `subject` is the placed subject (canvas-sized RGBA).
pub fn reflection(subject: &RgbaImage, ground_y: usize, kind: ReflectionKind, squash: f32, falloff: f32) -> RgbaImage {
    let (w, h) = (subject.width(), subject.height());
    let mut r = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    let dim = kind.dim();
    if kind == ReflectionKind::None {
        return r;
    }
    // subject height (for the fade span): distance from foot-line up to the top opaque row.
    let subj_h = (0..ground_y as u32)
        .find(|&y| (0..w).any(|x| subject.get_pixel(x, y).0[3] > 0))
        .map(|top| ground_y as f32 - top as f32)
        .unwrap_or(ground_y as f32);
    for y in ground_y as u32..h {
        let below = (y - ground_y as u32) as f32;
        let src_y = ground_y as f32 - below / squash.max(0.1); // mirror + foreshorten
        if src_y < 0.0 {
            continue;
        }
        let sy = src_y.round() as u32;
        if sy >= h {
            continue;
        }
        // Q3 (SEAMS-1): PERSPECTIVE falloff — a real floor reflection dims faster with distance than a
        // linear ramp (which reads as a full mirror). Raise the linear fade to a >1 power so the
        // reflection concentrates near the contact line and fades off realistically.
        let t = (below / (falloff.max(0.05) * subj_h)).clamp(0.0, 1.0);
        let fade = (1.0 - t).powf(1.7);
        // A subtle depth-increasing horizontal blur (gloss): sample a small run and average, widening
        // with depth so the far reflection is softer than the sharp contact line.
        let blur_r = (t * 3.0).round() as i32;
        for x in 0..w {
            let p = if blur_r > 0 {
                let (mut sr, mut sg, mut sb, mut sa, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
                for dx in -blur_r..=blur_r {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let q = subject.get_pixel(xx, sy);
                    sr += q.0[0] as u32;
                    sg += q.0[1] as u32;
                    sb += q.0[2] as u32;
                    sa += q.0[3] as u32;
                    n += 1;
                }
                Rgba([(sr / n) as u8, (sg / n) as u8, (sb / n) as u8, (sa / n) as u8])
            } else {
                *subject.get_pixel(x, sy)
            };
            if p.0[3] > 0 {
                let a = (p.0[3] as f32 / 255.0 * fade * dim * 255.0).round() as u8;
                if a > 0 {
                    r.put_pixel(x, y, Rgba([p.0[0], p.0[1], p.0[2], a]));
                }
            }
        }
    }
    r
}

/// The camera-angle foreshorten factor for the reflection (eye-level → thin, low/hero → taller).
pub fn camera_squash(angle: Option<&str>) -> f32 {
    match angle.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("hero") | Some("low") => 1.4,
        Some("top") | Some("flatlay") => 3.5,
        Some("three-quarter") => 2.4,
        _ => 2.0, // eye
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn box_subject(w: usize, h: usize, ground_y: usize) -> (Vec<f32>, RgbaImage) {
        let mut img = RgbaImage::from_pixel(w as u32, h as u32, Rgba([0, 0, 0, 0]));
        let (x0, x1, y0) = (w / 3, 2 * w / 3, ground_y.saturating_sub(h / 3));
        for y in y0..ground_y {
            for x in x0..x1 {
                img.put_pixel(x as u32, y as u32, Rgba([200, 40, 40, 255]));
            }
        }
        let alpha = img.pixels().map(|p| p.0[3] as f32 / 255.0).collect();
        (alpha, img)
    }

    #[test]
    fn shadow_is_grounded_and_anchored() {
        let (w, h, gy) = (240, 300, 216);
        let (alpha, _) = box_subject(w, h, gy);
        let sh = contact_shadow(&alpha, w, h, gy, ShadowKind::Soft, key_offset(Some("top-left")), 0.5);
        // no shadow above the contact line (grounded — the G0 fix).
        let above: f32 = (0..gy).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| sh[y * w + x]).sum();
        assert_eq!(above, 0.0, "shadow stays on the floor");
        // densest just below contact, softer far away.
        let band = |y0: usize, y1: usize| -> f32 {
            let mut s = 0.0;
            for y in y0..y1 {
                for x in 0..w {
                    s += sh[y * w + x];
                }
            }
            s
        };
        assert!(band(gy, gy + 6) > band(gy + 50, gy + 56), "anchored at the base");
    }

    #[test]
    fn reflection_falloff_is_perspective_not_linear() {
        // The reflection must dim FASTER than linear with depth (perspective) — a mid-depth row's
        // brightness is below what a straight linear ramp would give, so it doesn't read as a full mirror.
        let (w, h, gy) = (240, 300, 216);
        let (_, subj) = box_subject(w, h, gy);
        let falloff = 0.6f32;
        let refl = reflection(&subj, gy, ReflectionKind::Mirror, camera_squash(Some("eye")), falloff);
        // subject height in this fixture = h/3 = 100 → fade span = falloff*subj_h.
        let subj_h = (h / 3) as f32;
        let mid_below = 0.5 * falloff * subj_h; // the half-way point of the fade span
        let y = gy + mid_below.round() as usize;
        let max_a = (0..w as u32).map(|x| refl.get_pixel(x, y as u32).0[3]).max().unwrap_or(0);
        // A LINEAR fade at t=0.5 gives 0.5·dim·255; perspective (0.5^1.7 ≈ 0.31·dim·255) is clearly less.
        let dim = ReflectionKind::Mirror.dim();
        let linear = (255.0 * 0.5 * dim).round() as i32;
        assert!(max_a > 0, "reflection present at mid-depth");
        assert!((max_a as i32) < linear * 3 / 4, "perspective dims faster than linear (got {max_a}, linear {linear})");
    }

    #[test]
    fn reflection_starts_at_the_foot_line() {
        let (w, h, gy) = (240, 300, 216);
        let (_, subj) = box_subject(w, h, gy);
        let refl = reflection(&subj, gy, ReflectionKind::Gloss, camera_squash(Some("eye")), 0.55);
        let top = (gy as u32..h as u32).find(|&y| (0..w as u32).any(|x| refl.get_pixel(x, y).0[3] > 0));
        assert!(top.map(|y| y <= gy as u32 + 2).unwrap_or(false), "reflection aligned to foot-line: {top:?}");
    }
}
