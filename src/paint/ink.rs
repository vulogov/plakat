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

/// How the tone is hatched. A PEN rules straight lines at fixed angles; an ENGRAVING (Dürer's burin) cuts
/// lines that FOLLOW THE FORM — curved along the isophotes of the value, cross-hatched in the darks — finer and
/// denser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HatchStyle {
    /// Line spacing = sheet's long side / this.
    pub spacing_div: f32,
    /// Lines follow the form's isophotes (streamlines) instead of ruled fixed angles.
    pub follow_form: bool,
    /// Tone layers: the darkness QUANTILES at which each hatch layer starts (the image's own distribution)…
    pub quantiles: [f32; 4],
    /// …with absolute darkness FLOORS so an open sky is never hatched for being the darkest thing in a bright
    /// picture. An engraving starts its lightest hatch earlier than a pen (Dürer's skies carry a few lines).
    pub floors: [f32; 4],
    /// Line width = long side / this.
    pub width_div: f32,
    /// ENGRAVING tone: six crossings — darkness floors at which each layer starts (the plate's continuous tone),
    /// and each layer's angle off the form's direction (a Dürer dark is four or five crossings deep).
    pub eng_floors: [f32; 6],
    pub eng_offsets: [f32; 6],
}

impl HatchStyle {
    /// SUMI-E: the drawing model with a BRUSH — bold contour strokes, and tone only where the value is truly
    /// mid-to-dark, laid as a few wide form-following strokes (grey wash, then black); the paper is the light.
    pub const SUMI: HatchStyle = HatchStyle { spacing_div: 70.0, follow_form: true, quantiles: [0.50, 0.72, 0.88, 0.97], floors: [0.35, 0.55, 0.72, 0.86], width_div: 110.0, eng_floors: [1.01; 6], eng_offsets: [0.0; 6] };
    /// Contours only — no tone layer ever fires (thresholds above full darkness).
    pub const NONE: HatchStyle = HatchStyle { spacing_div: 180.0, follow_form: false, quantiles: [1.0, 1.0, 1.0, 1.0], floors: [1.01, 1.01, 1.01, 1.01], width_div: 1000.0, eng_floors: [1.01; 6], eng_offsets: [0.0; 6] };
    pub const PEN: HatchStyle = HatchStyle { spacing_div: 180.0, follow_form: false, quantiles: [0.45, 0.68, 0.84, 0.94], floors: [0.18, 0.34, 0.50, 0.66], width_div: 1000.0, eng_floors: [1.01; 6], eng_offsets: [0.0; 6] };
    /// Dürer's plate is ELABORATE: every surface but the true highlight carries line. The first (form-following)
    /// hatch starts almost at the paper, the crossings come in with the tone, and the burin's line swells with
    /// the darkness it cuts (see `Drawing::hatch_widths`).
    /// The six floors here are placeholders: `plan_with` derives the engraving's floors from the ink COVERAGE
    /// each crossing adds (width / spacing), so the plate's tone matches the picture's value — a passage gets
    /// its k-th crossing once it is darker than k crossings would make it.
    pub const ENGRAVING: HatchStyle = HatchStyle { spacing_div: 300.0, follow_form: true, quantiles: [0.15, 0.42, 0.66, 0.85], floors: [0.12, 0.28, 0.48, 0.66], width_div: 1400.0, eng_floors: [0.0; 6], eng_offsets: [0.0, 0.0, 1.5708, 1.5708, 0.7854, 2.3562] };
}

/// A planned drawing: contour polylines and hatch segments (with the tone layer each belongs to).
pub struct Drawing {
    pub contours: Vec<Vec<[f32; 2]>>,
    pub hatch: Vec<(usize, Vec<[f32; 2]>)>,
    /// Pen width (px) for contours and for hatch lines.
    pub contour_width: f32,
    pub hatch_width: f32,
    /// Per-hatch-line width (same index as `hatch`) when the medium varies it — an engraver's line SWELLS with
    /// the tone it cuts (thin in the lights, full in the darks). `None` = every line `hatch_width`.
    pub hatch_widths: Option<Vec<f32>>,
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

/// [`edge_map`] restricted to a region: the strong/weak thresholds come from the gradients INSIDE the mask (so a
/// soft-lit face keeps its own strongest `keep` fraction of edges — brow, nose, mouth, beard, glasses — instead
/// of losing them all to the sheet's foliage), and only masked pixels are returned.
fn edge_map_masked(v: &[f32], w: usize, h: usize, mask: &[f32], keep: f32) -> Vec<bool> {
    let (gx, gy, mag) = sobel(v, w, h);
    let mut thin = vec![0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if mask[i] <= 0.5 {
                continue;
            }
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
    let hi = sorted[((1.0 - keep.clamp(0.02, 0.6)) * (sorted.len() as f32 - 1.0)) as usize];
    let lo = hi * 0.35;
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
    plan_with(input, contour, budget, seed, HatchStyle::PEN)
}

/// [`plan`] with an explicit hatch style.
pub fn plan_with(input: &RgbImage, contour: f32, budget: usize, seed: u64, style: HatchStyle) -> Drawing {
    plan_detailed(input, None, contour, budget, seed, style)
}

/// [`plan_with`] with a DETAIL tier: inside `detail` (a 0..1 mask — the head, from the face detector), the
/// contours are traced from the full-resolution `source` (the armature has already simplified a face away)
/// and kept by the face's OWN edge distribution, not the sheet's — an engraver draws every fold of a face while
/// the foliage behind it gets a few strokes. Dürer's faces are recognizable because they are DRAWN, not shaded.
pub fn plan_detailed(input: &RgbImage, detail: Option<(&RgbImage, &[f32])>, contour: f32, budget: usize, seed: u64, style: HatchStyle) -> Drawing {
    let (w, h) = (input.width() as usize, input.height() as usize);
    let long = w.max(h) as f32;
    let value = smooth3(&value_map(input), w, h);
    // An engraved line is finer than a pen's.
    // A brush drawing (sumi) draws its contours with a loaded brush, an engraving with a fine burin, a pen in between.
    let contour_width = if style.width_div < 300.0 { (long / 220.0).clamp(2.0, 8.0) } else if style.follow_form { (long / 1000.0).clamp(0.9, 2.0) } else { (long / 800.0).clamp(1.0, 2.5) };
    let hatch_width = if style.width_div < 300.0 { (long / style.width_div).clamp(3.0, 16.0) } else { (long / style.width_div).clamp(0.7, 2.0) };

    // CONTOURS.
    let mut contours = Vec::new();
    if contour > 0.0 {
        let mut edge = edge_map(&value, w, h, contour);
        // DETAIL tier: the head's edges from the SOURCE, thresholded against the head's own gradients.
        if let Some((src, mask)) = detail {
            if src.width() as usize == w && src.height() as usize == h && mask.len() == w * h {
                let sv = smooth3(&value_map(src), w, h);
                let head_edge = edge_map_masked(&sv, w, h, mask, 0.30);
                for i in 0..w * h {
                    if mask[i] > 0.5 {
                        edge[i] = head_edge[i];
                    }
                }
            }
        }
        // An engraving keeps its SHORT chains too — a window bar, an eye, a hand — a pen drawing drops them.
        let min_len = if style.follow_form { (long / 320.0).max(5.0) as usize } else { (long / 150.0).max(6.0) as usize };
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
    let (qs, fl) = (style.quantiles, style.floors);
    let thresh = [q(qs[0]).max(fl[0]), q(qs[1]).max(fl[1]), q(qs[2]).max(fl[2]), q(qs[3]).max(fl[3])];
    let angles = [0.7854f32, 2.3562, 0.0, 1.5708];
    // Spacing: fine enough to read as tone; widened until the marks fit the remaining budget.
    let remaining = budget.saturating_sub(contours.len()).max(1);
    let mut spacing = (long / style.spacing_div).max(2.0);
    let est = |sp: f32| -> f32 {
        // segments ≈ covered pixels / (spacing × mean segment length ≈ 6·spacing)
        thresh.iter().map(|&t| darkness.iter().filter(|&&d| d >= t).count() as f32).sum::<f32>() / (sp * sp * 6.0)
    };
    while est(spacing) > remaining as f32 && spacing < long / 20.0 {
        spacing *= 1.15;
    }
    let mut hatch_widths: Option<Vec<f32>> = None;
    let hatch = if style.follow_form {
        // Tone by coverage: one crossing at width w and spacing s inks w/s of the paper; k crossings ink
        // 1 − (1 − w/s)^k. The k-th crossing starts where the picture is at least that dark (a little earlier,
        // so the lightest tone carries a few lines rather than none).
        // TONE BY LINE WEIGHT FIRST (the burin cuts deeper for a darker passage): the first, form-following
        // layer runs over every surface but the highlight and carries the tone in its WIDTH (a hair in the
        // lights, a full cut in the darks); a crossing is added only where even full-weight lines cannot carry
        // the tone — the k-th crossing where k full-weight layers would ink the paper that dark.
        // Tone floors are a fact of the LINE WIDTH and SPACING of the pass that lays them (a finer pass at half
        // the spacing inks twice the paper per crossing, so its crossings must start twice as dark).
        let floors_for = |width: f32, sp: f32| -> [f32; 6] {
            let c_full = (width * 2.2 / sp).clamp(0.15, 0.6);
            let mut f = [0f32; 6];
            for (k, v) in f.iter_mut().enumerate() {
                // A HIGHLIGHT (value above ~0.88) is bare paper; everything else carries at least a hair line.
                *v = if k == 0 { (width * 0.35 / sp).clamp(0.10, 0.2) } else { 1.0 - (1.0 - c_full).powi(k as i32) };
            }
            f
        };
        let floors = floors_for(hatch_width, spacing);
        let cap = budget.saturating_sub(contours.len());
        let lines = match detail {
            Some((src, mask)) if src.width() as usize == w && src.height() as usize == h && mask.len() == w * h => {
                // THE HEAD IS ENGRAVED FROM THE SOURCE: its value (not the armature's simplification) sets the
                // tone, the direction and where a cut stops — so brow, eye socket, nose and beard are modelled —
                // at half the spacing with SHORT cuts (the fine burin work of a portrait); the body's hatch is kept
                // out of the head so the two passes do not cross.
                let sv = smooth3(&value_map(src), w, h);
                let head_dark: Vec<f32> = (0..w * h).map(|i| if mask[i] > 0.5 { 1.0 - sv[i] } else { 0.0 }).collect();
                let body_dark: Vec<f32> = (0..w * h).map(|i| if mask[i] > 0.5 { 0.0 } else { darkness[i] }).collect();
                let mut all = engrave_hatch(&value, &body_dark, w, h, &floors, &style.eng_offsets, spacing, 40.0, 3.0, seed, cap);
                // The portrait pass: half the spacing, a finer line (0.7×), and floors honest to that geometry —
                // the same tone as the body, cut with a finer burin.
                let mut head_floors = floors_for(hatch_width * 0.7, spacing * 0.5);
                // A Dürer face is mostly PAPER on its lit side: the first cuts begin in the half-tone, and the
                // features and the shadow side carry the crossings.
                head_floors[0] = head_floors[0].max(0.35);
                let head = engrave_hatch(&sv, &head_dark, w, h, &head_floors, &style.eng_offsets, spacing * 0.5, 8.0, 12.0, seed ^ 0x4EAD, cap.saturating_sub(all.len()));
                all.extend(head);
                all
            }
            _ => engrave_hatch(&value, &darkness, w, h, &floors, &style.eng_offsets, spacing, 40.0, 3.0, seed, cap),
        };
        // The swelling line: width from the mean darkness along the cut — a light passage is a hair, a dark one
        // a full burin stroke. Contours keep the plate's firm line.
        let dark_for_width: Vec<f32> = match detail {
            Some((src, mask)) if src.width() as usize == w && src.height() as usize == h && mask.len() == w * h => {
                let sv = smooth3(&value_map(src), w, h);
                (0..w * h).map(|i| if mask[i] > 0.5 { 1.0 - sv[i] } else { darkness[i] }).collect()
            }
            _ => darkness.clone(),
        };
        let head_scale: Vec<f32> = match detail {
            Some((_, mask)) if mask.len() == w * h => mask.iter().map(|&m| if m > 0.5 { 0.7 } else { 1.0 }).collect(),
            _ => vec![1.0; w * h],
        };
        hatch_widths = Some(lines.iter().map(|(_, pts)| {
            let mid = pts[pts.len() / 2];
            let hs = head_scale[(mid[1] as usize).min(h - 1) * w + (mid[0] as usize).min(w - 1)];
            let d = hs * pts.iter().map(|p| dark_for_width[(p[1] as usize).min(h - 1) * w + (p[0] as usize).min(w - 1)]).sum::<f32>() / pts.len().max(1) as f32;
            hatch_width * (0.3 + 1.7 * d)
        }).collect());
        lines
    } else {
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
        hatch
    };
    Drawing { contours, hatch, contour_width, hatch_width, hatch_widths }
}

/// Separable box blur of a field (radius `r`).
fn box_blur(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
            let mut acc = 0.0;
            for xx in x0..=x1 {
                acc += v[y * w + xx];
            }
            tmp[y * w + x] = acc / (x1 - x0 + 1) as f32;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            let mut acc = 0.0;
            for yy in y0..=y1 {
                acc += tmp[yy * w + x];
            }
            out[y * w + x] = acc / (y1 - y0 + 1) as f32;
        }
    }
    out
}

/// ENGRAVED HATCH: tone by lines that FOLLOW THE FORM. The direction field is the isophote of the value smoothed
/// at the hatch scale (so lines wrap a cheek, a sleeve, a wall — not pixel texture); each tone layer rotates it
/// (along the form, across it, then the two diagonals) so the darks are cross-hatched. Lines are traced as
/// streamlines from a jittered seed grid, clipped to the layer's tone region, and kept EVENLY SPACED by an
/// occupancy grid one spacing wide (a cell holds one line per layer — Jobard–Lefer). Deterministic.
#[allow(clippy::too_many_arguments)]
fn engrave_hatch(value: &[f32], darkness: &[f32], w: usize, h: usize, thresh: &[f32; 6], offsets: &[f32; 6], spacing: f32, run: f32, form: f32, seed: u64, cap: usize) -> Vec<(usize, Vec<[f32; 2]>)> {
    // The FORM's direction, not the texture's: the structure tensor of the gradient, smoothed over several
    // spacings, gives one dominant orientation per neighbourhood (a wall, a sleeve, a cheek), so the lines run
    // long and parallel instead of breaking at every blade of grass.
    let r = (spacing * 1.5).round().max(2.0) as usize;
    let sm = box_blur(value, w, h, r);
    let (gx, gy, _) = sobel(&sm, w, h);
    let n = w * h;
    let (mut jxx, mut jyy, mut jxy) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        jxx[i] = gx[i] * gx[i];
        jyy[i] = gy[i] * gy[i];
        jxy[i] = gx[i] * gy[i];
    }
    // Smoothed over MANY spacings: one direction per SURFACE (a wall, a sleeve, a trunk), not per wrinkle.
    // Over ~3 spacings: the LOCAL form (a face, a hand, a trunk keeps its own direction); wider smoothing made
    // every surface follow the picture's lighting gradient — one sweep of lines across the whole sheet.
    // `form` = the scale (in spacings) of the FORM the lines follow: the body's local objects at 3, a portrait's
    // head-scale volumes (brow, cheek, jaw) at 12 — not the skin's texture.
    let rt = (spacing * form).round().max(3.0) as usize;
    let (jxx, jyy, jxy) = (box_blur(&jxx, w, h, rt), box_blur(&jyy, w, h, rt), box_blur(&jxy, w, h, rt));
    // FLAT vs FORM: a wall or a road hardly changes value across the wide window (its slow lighting gradient is
    // not a form) — it is RULED with straight parallel lines; a form (a trunk, a sleeve, a cheek) varies, and its
    // lines wrap it. Value spread over the wide window decides, not gradient coherence (a smooth gradient is
    // perfectly coherent, which is what streaked the walls with wood-grain).
    let rf = (spacing * 12.0).round().max(6.0) as usize;
    let mean = box_blur(value, w, h, rf);
    let sq: Vec<f32> = value.iter().map(|v| v * v).collect();
    let mean2 = box_blur(&sq, w, h, rf);
    let spread: Vec<f32> = mean.iter().zip(&mean2).map(|(m, m2)| (m2 - m * m).max(0.0).sqrt()).collect();
    // Gradient orientation θ → isophote (perpendicular) as a unit vector; coherence as the magnitude proxy.
    let mut iso = vec![[0.7071f32, 0.7071f32]; n];
    let mut mag = vec![0f32; n];
    for i in 0..n {
        let theta = 0.5 * (2.0 * jxy[i]).atan2(jxx[i] - jyy[i]);
        let disc = ((jxx[i] - jyy[i]).powi(2) + 4.0 * jxy[i] * jxy[i]).sqrt();
        mag[i] = disc / (jxx[i] + jyy[i] + 1e-9);
        iso[i] = [-theta.sin(), theta.cos()];
    }
    let cell = spacing.max(1.5);
    let (cw, ch) = ((w as f32 / cell).ceil() as usize + 1, (h as f32 / cell).ceil() as usize + 1);
    // Long, straight cuts: a burin line runs across the whole surface it shades.
    let max_len = (spacing * run).max(12.0) as usize;
    let seed_step = (spacing * 0.75).max(1.0);
    let mut out: Vec<(usize, Vec<[f32; 2]>)> = Vec::new();
    for (li, (&t, &off)) in thresh.iter().zip(offsets.iter()).enumerate() {
        let mut occ = vec![u32::MAX; cw * ch];
        let (so, co) = off.sin_cos();
        // A FLAT surface (no coherent form) is hatched with straight parallel lines at the plate's base angle
        // (45°), as Dürer hatches a wall or a floor; a curved form's lines wrap it.
        let dir_at = |x: f32, y: f32| -> [f32; 2] {
            let i = (y as usize).min(h - 1) * w + (x as usize).min(w - 1);
            let (dx, dy) = if mag[i] > 0.15 && spread[i] > 0.07 { (iso[i][0], iso[i][1]) } else { (0.7071f32, 0.7071f32) };
            [dx * co - dy * so, dx * so + dy * co]
        };
        let mut line_id: u32 = 0;
        let mut k = 0u64;
        // A repeated angle (the previous layer's) lays its lines BETWEEN the previous ones: same direction,
        // half a spacing over — the engraver's way of deepening a tone before crossing it.
        let repeat = li > 0 && (offsets[li] - offsets[li - 1]).abs() < 1e-3;
        let shift = if repeat { spacing * 0.5 } else { 0.0 };
        let mut sy = shift;
        while sy < h as f32 {
            let mut sx = shift;
            while sx < w as f32 {
                k += 1;
                let x0 = sx + jitter(seed ^ 0xD0E1, k) * seed_step * 0.5;
                let y0 = sy + jitter(seed ^ 0xD0E2, k.wrapping_add(1)) * seed_step * 0.5;
                sx += seed_step;
                if x0 < 0.0 || y0 < 0.0 || x0 >= w as f32 || y0 >= h as f32 {
                    continue;
                }
                if darkness[y0 as usize * w + x0 as usize] < t {
                    continue;
                }
                let c0 = (y0 / cell) as usize * cw + (x0 / cell) as usize;
                if occ[c0] != u32::MAX {
                    continue;
                }
                let mut path: Vec<[f32; 2]> = Vec::new();
                for sign in [-1.0f32, 1.0] {
                    let (mut x, mut y) = (x0, y0);
                    let mut prev: Option<[f32; 2]> = None;
                    let mut pts = Vec::new();
                    for _ in 0..max_len {
                        let mut d = dir_at(x, y);
                        match prev {
                            Some(pv) => {
                                if pv[0] * d[0] + pv[1] * d[1] < 0.0 {
                                    d = [-d[0], -d[1]];
                                }
                            }
                            None => d = [d[0] * sign, d[1] * sign],
                        }
                        x += d[0];
                        y += d[1];
                        if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
                            break;
                        }
                        // No slack below the layer's threshold: with the lightest layer starting near the
                        // paper's own value, any slack lets lines wander onto the ground.
                        let dn = darkness[y as usize * w + x as usize];
                        if dn < t {
                            break;
                        }
                        // A cut stops at a SILHOUETTE: the tone jumps between one step and the next.
                        if let Some(_) = prev {
                            let dp = darkness[(y - d[1]) as usize * w + (x - d[0]) as usize];
                            if (dn - dp).abs() > 0.12 {
                                break;
                            }
                        }
                        let c = (y / cell) as usize * cw + (x / cell) as usize;
                        if occ[c] != u32::MAX && occ[c] != line_id {
                            break;
                        }
                        occ[c] = line_id;
                        pts.push([x, y]);
                        prev = Some(d);
                    }
                    if sign < 0.0 {
                        pts.reverse();
                        path = pts;
                        path.push([x0, y0]);
                    } else {
                        path.extend(pts);
                    }
                }
                occ[c0] = line_id;
                // A line shorter than a couple of spacings is a dot, not a cut — an engraving has none.
                if path.len() >= (spacing * 2.5).max(4.0) as usize {
                    out.push((li, path));
                    line_id += 1;
                    if out.len() >= cap {
                        return out;
                    }
                }
            }
            sy += seed_step;
        }
    }
    out
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
    fn engraving_hatch_follows_the_form_and_stays_in_the_dark_mass() {
        let img = two_masses(120, 120);
        let d = plan_with(&img, 0.6, 20_000, 7, HatchStyle::ENGRAVING);
        assert!(!d.hatch.is_empty(), "the dark mass is engraved");
        let outside = d.hatch.iter().flat_map(|(_, s)| s.iter()).filter(|p| !(p[0] > 29.0 && p[0] < 91.0 && p[1] > 29.0 && p[1] < 91.0)).count();
        assert_eq!(outside, 0, "no lines on the light ground");
        let layers: std::collections::BTreeSet<usize> = d.hatch.iter().map(|(l, _)| *l).collect();
        assert!(layers.len() >= 3, "cross-hatched ({layers:?})");
        let longest = d.hatch.iter().map(|(_, s)| s.len()).max().unwrap();
        assert!(longest >= 12, "lines run, they are not dots ({longest})");
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
