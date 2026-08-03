//! The print/ink scorecard (RFC BOOKART-1 §9). Measures a *finished* ornament (transparent RGBA)
//! against its resolved plan: is it truly B/W, is the alpha clean, is it symmetric within tolerance, is
//! the ink coverage sane, is it the right size? Pure, no weights (the stray-glyph probe, which needs
//! OWL-ViT, is deferred to the render wiring). Drives `bookart verify` + later rejection sampling.

use crate::bookart::compile::RenderPlan;
use image::RgbaImage;

/// Thresholds a finished ornament should clear.
pub const CHROMA_MAX: f32 = 0.02; // finished ink is neutral → almost no chroma
pub const ALPHA_PARTIAL_MAX: f32 = 0.35; // a clean edge has few partial-alpha px
pub const SYMMETRY_RMS_MAX: f32 = 0.15; // for symmetric ornaments
pub const INK_MIN: f32 = 0.005; // not blank
pub const INK_MAX: f32 = 0.75; // not a solid slab

#[derive(Debug, Clone)]
pub struct Scorecard {
    /// Fraction of visible px with saturation > 0.10 (truly B/W → ~0).
    pub chroma_frac: f32,
    /// Fraction of px with partial alpha (8 < α < 247) — soft-edge fringe / halo.
    pub alpha_partial_frac: f32,
    /// Bilateral fold RMS (0 = perfect); `None` when the ornament isn't declared symmetric.
    pub symmetry_rms: Option<f32>,
    /// Fraction of opaque (α > 128) px — the ink coverage.
    pub ink_coverage: f32,
    /// Does the image match the plan's print canvas (px)?
    pub resolution_ok: bool,
    /// Per-probe pass flags folded into one verdict (resolution excluded — it's B2's job).
    pub passes: bool,
    pub notes: Vec<String>,
}

fn saturation(p: [u8; 4]) -> f32 {
    let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    if mx <= 0.0 { 0.0 } else { (mx - mn) / mx }
}

/// Bilateral fold RMS over the alpha channel (mirror about the vertical mid-axis).
fn bilateral_rms(img: &RgbaImage) -> f32 {
    let (w, h) = (img.width(), img.height());
    let (mut acc, mut n) = (0.0f64, 0u64);
    for y in 0..h {
        for x in 0..w / 2 {
            let a = img.get_pixel(x, y).0[3] as f64;
            let b = img.get_pixel(w - 1 - x, y).0[3] as f64;
            let d = (a - b) / 255.0;
            acc += d * d;
            n += 1;
        }
    }
    (acc / n.max(1) as f64).sqrt() as f32
}

/// Score a finished ornament against its plan.
pub fn score(finished: &RgbaImage, plan: &RenderPlan) -> Scorecard {
    let (mut colored, mut vis, mut partial, mut opaque, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for p in finished.pixels() {
        let a = p.0[3];
        n += 1;
        if a > 10 {
            vis += 1;
            if saturation(p.0) > 0.10 {
                colored += 1;
            }
        }
        if a > 8 && a < 247 {
            partial += 1;
        }
        if a > 128 {
            opaque += 1;
        }
    }
    let chroma_frac = if vis > 0 { colored as f32 / vis as f32 } else { 0.0 };
    let alpha_partial_frac = partial as f32 / n.max(1) as f32;
    let ink_coverage = opaque as f32 / n.max(1) as f32;

    let is_symmetric = plan.symmetry.starts_with("bilateral");
    let symmetry_rms = is_symmetric.then(|| bilateral_rms(finished));

    let resolution_ok = finished.dimensions() == (plan.page.w_px, plan.page.h_px);

    let mut notes = Vec::new();
    let mut pass = true;
    if chroma_frac > CHROMA_MAX {
        notes.push(format!("chroma {chroma_frac:.3} > {CHROMA_MAX} — not neutral B/W"));
        pass = false;
    }
    if alpha_partial_frac > ALPHA_PARTIAL_MAX {
        notes.push(format!("alpha-halo {alpha_partial_frac:.3} > {ALPHA_PARTIAL_MAX} — soft/haloed edges"));
        pass = false;
    }
    if let Some(rms) = symmetry_rms {
        if rms > SYMMETRY_RMS_MAX {
            notes.push(format!("symmetry RMS {rms:.3} > {SYMMETRY_RMS_MAX} — asymmetric (needs the symmetry engine)"));
            pass = false;
        }
    }
    if ink_coverage < INK_MIN {
        notes.push(format!("ink coverage {ink_coverage:.3} < {INK_MIN} — near-blank"));
        pass = false;
    } else if ink_coverage > INK_MAX {
        notes.push(format!("ink coverage {ink_coverage:.3} > {INK_MAX} — a solid slab, not ornament"));
        pass = false;
    }

    Scorecard { chroma_frac, alpha_partial_frac, symmetry_rms, ink_coverage, resolution_ok, passes: pass, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bookart::{compile::resolve, finish::finish_ornament, BookArtSpec};
    use image::{Rgb, RgbImage};

    fn symmetric_raw() -> RgbImage {
        // a bilaterally-symmetric black mark on white
        let mut raw = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
        for y in 20..44 {
            for x in 28..36 {
                raw.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        raw
    }

    #[test]
    fn finished_symmetric_ornament_passes() {
        let plan = resolve(&BookArtSpec::default()); // divider, bilateral
        let rgba = finish_ornament(&symmetric_raw(), &plan);
        let sc = score(&rgba, &plan);
        assert_eq!(sc.chroma_frac, 0.0, "neutral ink → no chroma");
        assert!(sc.symmetry_rms.unwrap() < SYMMETRY_RMS_MAX, "symmetric: {:?}", sc.symmetry_rms);
        assert!(sc.ink_coverage > INK_MIN);
        assert!(sc.passes, "{:?}", sc.notes);
    }

    #[test]
    fn asymmetric_render_flags_symmetry() {
        let mut raw = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255]));
        for y in 0..64 {
            for x in 0..12 {
                raw.put_pixel(x, y, Rgb([0, 0, 0])); // ink only on the left
            }
        }
        let plan = resolve(&BookArtSpec::default());
        let sc = score(&finish_ornament(&raw, &plan), &plan);
        assert!(sc.symmetry_rms.unwrap() > SYMMETRY_RMS_MAX);
        assert!(!sc.passes);
    }

    #[test]
    fn blank_render_is_flagged() {
        let raw = RgbImage::from_pixel(64, 64, Rgb([255, 255, 255])); // no ink
        let plan = resolve(&BookArtSpec::default());
        let sc = score(&finish_ornament(&raw, &plan), &plan);
        assert!(sc.ink_coverage < INK_MIN);
        assert!(!sc.passes);
    }
}
