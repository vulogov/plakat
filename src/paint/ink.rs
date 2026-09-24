//! PEN-AND-INK drawing (RFC PAINT-1 §8, the density mark model): a pen DRAWS the composition — it does not
//! paint masses. The drawing has two parts, both derived from the armature (structure, not detail) with no
//! weights and no tuned per-scene constants:
//!
//! * **Contours** — the structure's boundaries traced as continuous lines: an edge map (gradient magnitude,
//!   thinned to one pixel, kept by hysteresis against the image's own gradient distribution), linked into
//!   chains and drawn as pen lines. This is what makes the sheet read as *that* composition.
//! * **Tone by hatching** — value is built by LINE SPACING, the way a pen builds it: where the reference is
//!   darker than a threshold, parallel lines at a fixed spacing cover a fixed fraction of the paper; each darker
//!   threshold adds another set at a different angle (cross-hatching). Lines are long and straight (a slight
//!   hand waver), clipped to the tone region they shade, so the hatch itself draws the masses' edges. The lights
//!   are left as paper.
//!
//! Everything here is geometry: polylines in pixel space. The painter rasterises them with the pen and records
//! each as a stroke, so the drawing replays byte-exact from its score like any painting.

use image::RgbImage;

/// A planned drawing: contour polylines and hatch segments (with the tone layer each belongs to).
pub struct Drawing {
    pub contours: Vec<Vec<[f32; 2]>>,
    pub hatch: Vec<(usize, Vec<[f32; 2]>)>,
    /// Pen width (px) for contours and for hatch lines.
    pub contour_width: f32,
    pub hatch_width: f32,
}

/// Perceptual value (gamma-encoded luma): a pen drawing's mid-grey is what the eye calls mid-grey.
fn value_map(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| 0.299 * p.0[0] as f32 / 255.0 + 0.587 * p.0[1] as f32 / 255.0 + 0.114 * p.0[2] as f32 / 255.0).collect()
}

/// 3×3 Gaussian smooth (σ≈1) so the edge map reads structure, not pixel noise.
fn smooth3(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    let k = [1.0f32, 2.0, 1.0];
    let at = |x: i64, y: i64| v[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, ky) in k.iter().enumerate() {
                for (i, kx) in k.iter().enumerate() {
                    acc += kx * ky * at(x as i64 + i as i64 - 1, y as i64 + j as i64 - 1);
                }
            }
            out[y * w + x] = acc / 16.0;
        }
    }
    out
}

/// Sobel gradient (gx, gy, magnitude).
fn sobel(v: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let at = |x: i64, y: i64| v[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
    let (mut gx, mut gy, mut mag) = (vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let sx = -at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1) + at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1);
            let sy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1);
            let i = y as usize * w + x as usize;
            gx[i] = sx;
            gy[i] = sy;
            mag[i] = (sx * sx + sy * sy).sqrt();
        }
    }
    (gx, gy, mag)
}

/// Deterministic hash in [-0.5, 0.5] (the painter's jitter, duplicated here to keep the module self-contained).
fn jitter(seed: u64, k: u64) -> f32 {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z as f32 / u64::MAX as f32) - 0.5
}

/// The one-pixel-thin, hysteresis-kept edge map of a value field. `strength` (0..1) opens the map: at 0 only the
/// strongest 3% of gradients count, at 1 the strongest ~15%; the weak threshold is 40% of the strong one, so a
/// contour that fades keeps going as long as it is connected to a strong start.
fn edge_map(v: &[f32], w: usize, h: usize, strength: f32) -> Vec<bool> {
    let (gx, gy, mag) = sobel(v, w, h);
    // Non-maximum suppression along the gradient (4 quantised directions) → one-pixel edges.
    let mut thin = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let m = mag[i];
            if m <= 1e-4 {
                continue;
            }
            let (ax, ay) = (gx[i].abs(), gy[i].abs());
            let (n1, n2) = if ax >= 2.414 * ay {
                (mag[i - 1], mag[i + 1])
            } else if ay >= 2.414 * ax {
                (mag[i - w], mag[i + w])
            } else if (gx[i] > 0.0) == (gy[i] > 0.0) {
                (mag[i - w - 1], mag[i + w + 1])
            } else {
                (mag[i - w + 1], mag[i + w - 1])
            };
            if m >= n1 && m >= n2 {
                thin[i] = m;
            }
        }
    }
    let mut sorted: Vec<f32> = thin.iter().copied().filter(|&m| m > 1e-4).collect();
    if sorted.is_empty() {
        return vec![false; w * h];
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let keep = 0.03 + 0.12 * strength.clamp(0.0, 1.0);
    let hi = sorted[((1.0 - keep) * (sorted.len() as f32 - 1.0)) as usize];
    let lo = hi * 0.4;
    // Hysteresis: strong seeds, weak pixels kept only when connected to a strong one.
    let mut edge = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..w * h {
        if thin[i] >= hi && !edge[i] {
            edge[i] = true;
            stack.push(i);
            while let Some(j) = stack.pop() {
                let (x, y) = ((j % w) as i64, (j / w) as i64);
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        let n = ny as usize * w + nx as usize;
                        if !edge[n] && thin[n] >= lo {
                            edge[n] = true;
                            stack.push(n);
                        }
                    }
                }
            }
        }
    }
    edge
}

/// Link edge pixels into chains (polylines), walking 8-connected neighbours and preferring to continue straight.
/// Chains shorter than `min_len` pixels are noise and dropped.
fn trace_chains(edge: &[bool], w: usize, h: usize, min_len: usize) -> Vec<Vec<[f32; 2]>> {
    let mut used = vec![false; w * h];
    let mut chains = Vec::new();
    let neighbours = |i: usize| -> Vec<usize> {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        let mut out = Vec::with_capacity(8);
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                    out.push(ny as usize * w + nx as usize);
                }
            }
        }
        out
    };
    // Walk from a start pixel in one direction until the chain ends.
    let walk = |start: usize, used: &mut Vec<bool>| -> Vec<usize> {
        let mut path = vec![start];
        used[start] = true;
        let mut cur = start;
        let mut last_dir: Option<(i64, i64)> = None;
        loop {
            let (cx, cy) = ((cur % w) as i64, (cur / w) as i64);
            let mut best: Option<(usize, f32)> = None;
            for n in neighbours(cur) {
                if !edge[n] || used[n] {
                    continue;
                }
                let (nx, ny) = ((n % w) as i64, (n / w) as i64);
                let d = (nx - cx, ny - cy);
                // Prefer the neighbour that continues the current direction (dot product), then 4-connected.
                let score = match last_dir {
                    Some((lx, ly)) => (lx * d.0 + ly * d.1) as f32,
                    None => if d.0 == 0 || d.1 == 0 { 1.0 } else { 0.5 },
                };
                if best.map(|(_, s)| score > s).unwrap_or(true) {
                    best = Some((n, score));
                }
            }
            match best {
                Some((n, _)) => {
                    let (nx, ny) = ((n % w) as i64, (n / w) as i64);
                    last_dir = Some((nx - cx, ny - cy));
                    used[n] = true;
                    path.push(n);
                    cur = n;
                }
                None => break,
            }
        }
        path
    };
    for i in 0..w * h {
        if !edge[i] || used[i] {
            continue;
        }
        // Grow both ways from the start so a chain begun mid-edge is whole.
        let fwd = walk(i, &mut used);
        used[i] = false;
        let mut back = walk(i, &mut used);
        back.reverse();
        back.pop(); // the start pixel, already in fwd
        let mut chain: Vec<usize> = back;
        chain.extend(fwd);
        if chain.len() < min_len {
            continue;
        }
        // Smooth (3-tap) and thin the polyline to every other pixel — a drawn line, not a pixel staircase.
        let pts: Vec<[f32; 2]> = chain.iter().map(|&j| [(j % w) as f32 + 0.5, (j / w) as f32 + 0.5]).collect();
        let n = pts.len();
        let mut out = Vec::with_capacity(n / 2 + 2);
        for (j, p) in pts.iter().enumerate() {
            if j % 2 == 1 && j + 1 < n {
                continue;
            }
            let (a, b) = (pts[j.saturating_sub(1)], pts[(j + 1).min(n - 1)]);
            out.push([(a[0] + p[0] + b[0]) / 3.0, (a[1] + p[1] + b[1]) / 3.0]);
        }
        chains.push(out);
    }
    chains
}

/// Plan the drawing. `contour` (0..1) opens the edge map (the medium's contour strength); `budget` caps the
/// number of marks — contours take at most 40% of it, the hatch widens its spacing to fit the rest.
pub fn plan(input: &RgbImage, contour: f32, budget: usize, seed: u64) -> Drawing {
    let (w, h) = (input.width() as usize, input.height() as usize);
    let long = w.max(h) as f32;
    let value = smooth3(&value_map(input), w, h);
    let contour_width = (long / 800.0).clamp(1.0, 2.5);
    let hatch_width = (long / 1000.0).clamp(0.8, 2.0);

    // CONTOURS.
    let mut contours = Vec::new();
    if contour > 0.0 {
        let edge = edge_map(&value, w, h, contour);
        let min_len = (long / 150.0).max(6.0) as usize;
        contours = trace_chains(&edge, w, h, min_len);
        // Longest first, so a cap keeps the composition's main lines.
        contours.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let cap = budget * 2 / 5;
        contours.truncate(cap);
    }

    // TONE LAYERS. Thresholds are the image's own value distribution (a high-key sheet still shades its
    // relatively darker passages) with absolute floors so an open sky is never hatched for being the darkest
    // thing in a bright picture. Classic angles: 45°, 135°, 0°, 90°.
    let darkness: Vec<f32> = value.iter().map(|v| 1.0 - v).collect();
    let mut sorted = darkness.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |f: f32| sorted[((sorted.len() as f32 - 1.0) * f) as usize];
    let thresh = [q(0.45).max(0.18), q(0.68).max(0.34), q(0.84).max(0.50), q(0.94).max(0.66)];
    let angles = [0.7854f32, 2.3562, 0.0, 1.5708];
    // Spacing: fine enough to read as tone; widened until the marks fit the remaining budget.
    let remaining = budget.saturating_sub(contours.len()).max(1);
    let mut spacing = (long / 180.0).max(3.0);
    let est = |sp: f32| -> f32 {
        // segments ≈ covered pixels / (spacing × mean segment length ≈ 6·spacing)
        thresh.iter().map(|&t| darkness.iter().filter(|&&d| d >= t).count() as f32).sum::<f32>() / (sp * sp * 6.0)
    };
    while est(spacing) > remaining as f32 && spacing < long / 20.0 {
        spacing *= 1.15;
    }
    let mut hatch: Vec<(usize, Vec<[f32; 2]>)> = Vec::new();
    let diag = ((w * w + h * h) as f32).sqrt();
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    'layers: for (li, (&t, &ang)) in thresh.iter().zip(&angles).enumerate() {
        // A slight per-layer hand rotation (±3°) so the sheets of lines don't read as ruled.
        let ang = ang + 0.05 * jitter(seed ^ 0x1AC7, li as u64);
        let (d, n) = ([ang.cos(), ang.sin()], [-ang.sin(), ang.cos()]);
        let mut o = -diag * 0.5;
        let mut line = 0u64;
        while o <= diag * 0.5 {
            // Per-line spacing waver (±15%) and a slow bow — a hand, not a ruler.
            let off = o + spacing * 0.15 * jitter(seed ^ 0x1AC8, line + (li as u64) << 20);
            let mut seg: Vec<[f32; 2]> = Vec::new();
            let mut s = -diag * 0.5;
            let bow = 0.6 * jitter(seed ^ 0x1AC9, line + 7);
            while s <= diag * 0.5 {
                let wob = bow * (s / diag * std::f32::consts::PI).sin() * spacing;
                let (x, y) = (cx + d[0] * s + n[0] * (off + wob), cy + d[1] * s + n[1] * (off + wob));
                let inside = x >= 0.0 && y >= 0.0 && x < w as f32 && y < h as f32 && darkness[y as usize * w + x as usize] >= t;
                if inside {
                    seg.push([x, y]);
                } else if seg.len() >= 4 {
                    hatch.push((li, std::mem::take(&mut seg)));
                    if contours.len() + hatch.len() >= budget {
                        break 'layers;
                    }
                } else {
                    seg.clear();
                }
                s += 1.0;
            }
            if seg.len() >= 4 {
                hatch.push((li, seg));
            }
            o += spacing;
            line += 1;
        }
    }
    Drawing { contours, hatch, contour_width, hatch_width }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_masses(w: u32, h: u32) -> RgbImage {
        // A dark square on a light ground: one closed contour, hatch inside only.
        RgbImage::from_fn(w, h, |x, y| if x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4 { image::Rgb([40, 40, 45]) } else { image::Rgb([235, 232, 226]) })
    }

    #[test]
    fn contours_trace_the_boundary_and_hatch_stays_inside_the_dark_mass() {
        let img = two_masses(120, 120);
        let d = plan(&img, 0.6, 20_000, 7);
        assert!(!d.contours.is_empty(), "the square's boundary is traced");
        let longest = d.contours.iter().map(|c| c.len()).max().unwrap();
        assert!(longest > 40, "a contour runs along the boundary, not in dashes ({longest} points)");
        assert!(!d.hatch.is_empty(), "the dark mass is hatched");
        // The value field is smoothed by a pixel before thresholding, so allow a 1px halo at the boundary.
        let outside = d.hatch.iter().flat_map(|(_, s)| s.iter()).filter(|p| !(p[0] > 29.0 && p[0] < 91.0 && p[1] > 29.0 && p[1] < 91.0)).count();
        assert_eq!(outside, 0, "no hatch on the light ground");
        let layers: std::collections::BTreeSet<usize> = d.hatch.iter().map(|(l, _)| *l).collect();
        assert!(layers.len() >= 3, "a 0.84-dark mass is cross-hatched by several layers ({layers:?})");
    }

    #[test]
    fn a_flat_light_sheet_stays_paper() {
        let img = RgbImage::from_pixel(64, 64, image::Rgb([230, 228, 220]));
        let d = plan(&img, 0.6, 5_000, 7);
        assert!(d.contours.is_empty() && d.hatch.is_empty(), "nothing to draw on blank paper");
    }

    #[test]
    fn deterministic() {
        let img = two_masses(90, 70);
        let a = plan(&img, 0.5, 8_000, 3);
        let b = plan(&img, 0.5, 8_000, 3);
        assert_eq!(a.hatch.len(), b.hatch.len());
        assert_eq!(a.contours.len(), b.contours.len());
        assert_eq!(a.hatch.first().map(|(_, s)| s.clone()), b.hatch.first().map(|(_, s)| s.clone()));
    }
}
