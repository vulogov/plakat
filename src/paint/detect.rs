//! Automatic failure detection (RFC PAINT-1 §12.2). Two failure modes are diagnosable without human judgement,
//! and both belong in the harness rather than being eyeballed:
//!
//! * **Cutout** — objects reading as pasted despite correct values. Signature: the hard-edge fraction of the
//!   seam table is above the cap (§7.4). Pure function of the seam table.
//! * **Halo** — a bright or dark rim tracing every object. Signature: a value SPIKE in a narrow band along the
//!   plane boundaries (a boundary pixel that is a local extremum against both sides). Pure function of the
//!   painted value field + the plane map.

use crate::paint::seam::{EdgeClass, Seam};

/// The CUTOUT signature: the fraction of total seam length classified HARD (§7.4). A painting where every
/// boundary is a cut-in reads as collage. Above the configured hard-edge cap → a cutout risk.
pub fn cutout_fraction(seams: &[Seam]) -> f32 {
    let total: usize = seams.iter().map(|s| s.length).sum();
    if total == 0 {
        return 0.0;
    }
    let hard: usize = seams.iter().filter(|s| s.class == EdgeClass::Hard).map(|s| s.length).sum();
    hard as f32 / total as f32
}

/// True when the cutout fraction exceeds the cap — the picture risks reading as pasted.
pub fn is_cutout(seams: &[Seam], hard_cap: f32) -> bool {
    cutout_fraction(seams) > hard_cap
}

/// The HALO signature: the fraction of plane-boundary pixels whose painted value is a local EXTREMUM against
/// both bordering planes — a bright/dark rim tracing the object. `value` is the painted luma field `[0,1]`;
/// `plane` the plane index map; `margin` how far the extremum must exceed its cross-boundary neighbours.
pub fn halo_fraction(value: &[f32], plane: &[u32], w: u32, h: u32, margin: f32) -> f32 {
    let (wi, hi) = (w as i32, h as i32);
    let at = |x: i32, y: i32| y as usize * w as usize + x as usize;
    let mut boundary = 0usize;
    let mut rim = 0usize;
    for y in 0..hi {
        for x in 0..wi {
            let i = at(x, y);
            // A boundary pixel borders a different plane in some 4-neighbour direction. A halo pixel spikes
            // against its OWN plane's INTERIOR (the rim is brighter/darker than the mass it belongs to) — NOT
            // merely against the other plane, which would flag every legitimate value step.
            let mut own: Vec<f32> = Vec::new();
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= wi || ny >= hi {
                    continue;
                }
                if plane[at(nx, ny)] != plane[i] {
                    // Step INTO our own plane (opposite the boundary) and sample the interior value there.
                    let (ox, oy) = ((x - 2 * dx).clamp(0, wi - 1), (y - 2 * dy).clamp(0, hi - 1));
                    if plane[at(ox, oy)] == plane[i] {
                        own.push(value[at(ox, oy)]);
                    }
                }
            }
            if own.is_empty() {
                continue;
            }
            boundary += 1;
            let v = value[i];
            let bright = own.iter().all(|&c| v > c + margin);
            let dark = own.iter().all(|&c| v < c - margin);
            if bright || dark {
                rim += 1;
            }
        }
    }
    if boundary == 0 {
        0.0
    } else {
        rim as f32 / boundary as f32
    }
}

/// True when a strong rim traces the objects — masking has won over overspray; check the dilation direction
/// (§7.2).
pub fn is_halo(value: &[f32], plane: &[u32], w: u32, h: u32, margin: f32, frac_cap: f32) -> bool {
    halo_fraction(value, plane, w, h, margin) > frac_cap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::seam::Seam;

    fn seam(near: u32, far: u32, len: usize, class: EdgeClass) -> Seam {
        Seam { plane_near: near, plane_far: far, length: len, mean_contrast: 0.5, peak_saliency: 0.5, class }
    }

    #[test]
    fn cutout_is_the_hard_edge_fraction() {
        let seams = vec![seam(0, 1, 30, EdgeClass::Hard), seam(1, 2, 70, EdgeClass::Soft)];
        assert!((cutout_fraction(&seams) - 0.3).abs() < 1e-4);
        assert!(is_cutout(&seams, 0.2), "0.3 > cap 0.2 → cutout");
        assert!(!is_cutout(&seams, 0.4), "0.3 < cap 0.4 → fine");
    }

    #[test]
    fn halo_detects_a_rim_and_ignores_a_clean_boundary() {
        // 6×2, two planes split down the middle. Value: plane A dark 0.2, plane B light 0.7.
        let plane = vec![0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1];
        // A clean boundary (values match their planes) → no rim.
        let clean = vec![0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.2, 0.2, 0.2, 0.7, 0.7, 0.7];
        assert!(halo_fraction(&clean, &plane, 6, 2, 0.05) < 0.01, "smooth boundary → no halo");
        // A bright RIM: the boundary columns spike to 0.95 (brighter than both sides).
        let rim = vec![0.2, 0.2, 0.95, 0.95, 0.7, 0.7, 0.2, 0.2, 0.95, 0.95, 0.7, 0.7];
        assert!(halo_fraction(&rim, &plane, 6, 2, 0.05) > 0.5, "a bright rim along the seam → halo");
    }
}
