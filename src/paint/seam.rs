//! The seam table and stroke-termination policies (RFC PAINT-1 §7). This is where cut-paste-blur compositing
//! failed, so the mechanism is deliberately different: **nothing is ever clipped to a mask — the seam is a
//! stroke TERMINATION policy.** A painter paints the background *through* where an object will be, then paints
//! the object *over* it, and the boundary comes into existence as a consequence of where the last stroke
//! stopped. Mask-first produces cutouts; stroke-termination-first produces edges.
//!
//! All pure functions of the plane index map (§3.1 numbering: 0 = nearest). Seam extraction here aggregates
//! per plane-pair; splitting seams into connected polylines + the T-junction validator (§7.5) are a refinement.

/// How a stroke crosses a seam (§7.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeClass {
    /// The stroke ends exactly on the seam — a cut-in. Focal points only; capped to a fraction of total length.
    Hard,
    /// The stroke crosses by ~half a brush radius, pickup blends. Most of the picture.
    Soft,
    /// The stroke crosses freely — hair, foliage, shadow-side, shared value.
    Lost,
}

/// One seam between two planes.
#[derive(Clone, Debug)]
pub struct Seam {
    /// The nearer plane (the occluder — lower index).
    pub plane_near: u32,
    /// The farther plane.
    pub plane_far: u32,
    /// Boundary length in pixels.
    pub length: usize,
    /// Mean value contrast across the seam `[0,1]`.
    pub mean_contrast: f32,
    /// Peak saliency along the seam `[0,1]`.
    pub peak_saliency: f32,
    pub class: EdgeClass,
}

/// Extract the seam table from a plane index map. One seam per plane-pair, with its length, mean value
/// contrast, and peak saliency. `value` and `saliency` are `[0,1]` per pixel (saliency may be empty → 0).
pub fn extract_seams(plane: &[u32], w: u32, h: u32, value: &[f32], saliency: &[f32]) -> Vec<Seam> {
    use std::collections::HashMap;
    let (wi, hi) = (w as i32, h as i32);
    let mut acc: HashMap<(u32, u32), (usize, f32, f32)> = HashMap::new(); // (len, contrast_sum, peak_sal)
    let at = |x: i32, y: i32| y as usize * w as usize + x as usize;
    let sal = |i: usize| saliency.get(i).copied().unwrap_or(0.0);
    for y in 0..hi {
        for x in 0..wi {
            let i = at(x, y);
            for (dx, dy) in [(1, 0), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= wi || ny >= hi {
                    continue;
                }
                let j = at(nx, ny);
                if plane[i] != plane[j] {
                    let key = (plane[i].min(plane[j]), plane[i].max(plane[j]));
                    let e = acc.entry(key).or_insert((0, 0.0, 0.0));
                    e.0 += 1;
                    e.1 += (value[i] - value[j]).abs();
                    e.2 = e.2.max(sal(i)).max(sal(j));
                }
            }
        }
    }
    let mut seams: Vec<Seam> = acc
        .into_iter()
        .map(|((near, far), (len, csum, psal))| Seam {
            plane_near: near,
            plane_far: far,
            length: len,
            mean_contrast: if len > 0 { csum / len as f32 } else { 0.0 },
            peak_saliency: psal,
            class: EdgeClass::Soft,
        })
        .collect();
    // Stable order: longest first (deterministic).
    seams.sort_by(|a, b| b.length.cmp(&a.length).then(a.plane_near.cmp(&b.plane_near)));
    seams
}

/// The fraction of total seam length below which a seam's contrast makes it a LOST edge.
const LOST_CONTRAST: f32 = 0.08;

/// Classify seams (§7.4): the highest-saliency seams become HARD cut-ins up to `hard_cap` of total length
/// (default 0.15 — a painting where every boundary is cut in reads as collage); low-contrast seams are LOST;
/// the rest SOFT. Mutates the table's `class` in place.
pub fn classify_seams(seams: &mut [Seam], hard_cap: f32) {
    let total: usize = seams.iter().map(|s| s.length).sum();
    if total == 0 {
        return;
    }
    // Low-contrast → lost.
    for s in seams.iter_mut() {
        s.class = if s.mean_contrast < LOST_CONTRAST { EdgeClass::Lost } else { EdgeClass::Soft };
    }
    // Hard cut-ins: highest saliency first, until the cap on total length is reached.
    let mut order: Vec<usize> = (0..seams.len()).filter(|&i| seams[i].class != EdgeClass::Lost).collect();
    order.sort_by(|&a, &b| seams[b].peak_saliency.partial_cmp(&seams[a].peak_saliency).unwrap_or(std::cmp::Ordering::Equal));
    let cap = (hard_cap.clamp(0.0, 1.0) * total as f32) as usize;
    let mut hard_len = 0usize;
    for i in order {
        if hard_len + seams[i].length <= cap {
            seams[i].class = EdgeClass::Hard;
            hard_len += seams[i].length;
        }
    }
}

/// Binary dilation of a mask by Chebyshev radius `r` (a box structuring element). Pure, O(w·h·r).
fn dilate(mask: &[bool], w: u32, h: u32, r: i32) -> Vec<bool> {
    if r <= 0 {
        return mask.to_vec();
    }
    let (wi, hi) = (w as i32, h as i32);
    let mut out = vec![false; mask.len()];
    for y in 0..hi {
        for x in 0..wi {
            let mut any = false;
            'k: for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < wi && ny < hi && mask[(ny * wi + nx) as usize] {
                        any = true;
                        break 'k;
                    }
                }
            }
            out[(y * wi + x) as usize] = any;
        }
    }
    out
}

/// The paint region for plane `p`, given the medium's opacity (§7.2 / §7.3). Paint order is far → near, so when
/// plane `p` is painted the FARTHER planes (index > p) are already down and the NEARER ones (index < p) are
/// still bare.
///
/// * **Opaque** — forward overspray: `V_p ∪ (dilate(V_p, r) ∩ already_painted)`. The region grows only INTO
///   already-painted (farther) territory, which gets covered — never backward into finished nearer work. This
///   asymmetry is what eliminates halos.
/// * **Transparent** — reservation: `V_p \ dilate(nearer, margin)`. The far wash paints around the nearer
///   objects (which come later), reserving their silhouette so the later wash meets them cleanly.
pub fn paint_region(plane: &[u32], w: u32, h: u32, p: u32, r_max: i32, opaque: bool) -> Vec<bool> {
    let vp: Vec<bool> = plane.iter().map(|&q| q == p).collect();
    if opaque {
        let already: Vec<bool> = plane.iter().map(|&q| q > p).collect();
        let grown = dilate(&vp, w, h, r_max);
        vp.iter().zip(grown.iter().zip(already.iter())).map(|(&v, (&g, &a))| v || (g && a)).collect()
    } else {
        let nearer: Vec<bool> = plane.iter().map(|&q| q < p).collect();
        let margin = (r_max / 3).max(1);
        let reserve = dilate(&nearer, w, h, margin);
        vp.iter().zip(reserve.iter()).map(|(&v, &res)| v && !res).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 4×2 image: left half plane 1 (far), right half plane 0 (near). Boundary down the middle.
    fn split_map() -> (Vec<u32>, u32, u32) {
        (vec![1, 1, 0, 0, 1, 1, 0, 0], 4, 2)
    }

    #[test]
    fn seam_records_the_plane_pair_and_occluder() {
        let (plane, w, h) = split_map();
        let value = vec![0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8];
        let seams = extract_seams(&plane, w, h, &value, &[]);
        assert_eq!(seams.len(), 1, "one plane-pair → one seam");
        assert_eq!((seams[0].plane_near, seams[0].plane_far), (0, 1), "plane 0 is the nearer occluder");
        assert_eq!(seams[0].length, 2, "two vertical boundary edges");
        assert!((seams[0].mean_contrast - 0.6).abs() < 1e-4, "value contrast across the seam");
    }

    #[test]
    fn classification_caps_hard_edges() {
        // Three seams of equal length; only the highest-saliency one may go hard under a small cap.
        let mut seams = vec![
            Seam { plane_near: 0, plane_far: 1, length: 10, mean_contrast: 0.5, peak_saliency: 0.9, class: EdgeClass::Soft },
            Seam { plane_near: 1, plane_far: 2, length: 10, mean_contrast: 0.5, peak_saliency: 0.4, class: EdgeClass::Soft },
            Seam { plane_near: 0, plane_far: 2, length: 10, mean_contrast: 0.02, peak_saliency: 0.8, class: EdgeClass::Soft },
        ];
        classify_seams(&mut seams, 0.34); // cap ≈ 10 of 30 px → one hard seam
        assert_eq!(seams[0].class, EdgeClass::Hard, "highest-saliency, high-contrast → hard");
        assert_eq!(seams[1].class, EdgeClass::Soft, "over the cap → soft");
        assert_eq!(seams[2].class, EdgeClass::Lost, "low contrast → lost even though salient");
    }

    #[test]
    fn opaque_region_oversprays_forward_only() {
        let (plane, w, h) = split_map();
        // Painting the NEAR plane 0 with overspray: it may grow into the already-painted far plane (1)…
        let near = paint_region(&plane, w, h, 0, 1, true);
        assert!(near.iter().zip(&plane).all(|(&inreg, &pl)| pl != 0 || inreg), "the near plane's own pixels are in region");
        assert!(near.iter().enumerate().any(|(i, &r)| r && plane[i] == 1), "overspray reaches into the far (already-painted) plane");
        // Painting the FAR plane 1 does NOT grow into the nearer plane 0 (which is bare, not yet painted).
        let far = paint_region(&plane, w, h, 1, 1, true);
        assert!(!far.iter().enumerate().any(|(i, &r)| r && plane[i] == 0), "no backward overspray into bare nearer work");
    }

    #[test]
    fn transparent_region_reserves_the_nearer_silhouette() {
        let (plane, w, h) = split_map();
        // The far plane 1, transparent: it reserves a margin around the nearer plane 0, so near its boundary
        // some of plane 1's own pixels are withheld.
        let far = paint_region(&plane, w, h, 1, 3, false);
        let plane1_painted = far.iter().enumerate().filter(|(i, r)| **r && plane[*i] == 1).count();
        let plane1_total = plane.iter().filter(|&&q| q == 1).count();
        assert!(plane1_painted < plane1_total, "some of the far wash is reserved back from the near silhouette");
    }
}
