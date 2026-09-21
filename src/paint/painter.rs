//! The coarse-to-fine painter (RFC PAINT-1 P0) — and the filter-gate it exists to test.
//!
//! Given a reference IMAGE, paint it stroke by stroke under PAINT-1's hard constraints: a finite, inviolable
//! stroke **budget**; a limited **palette** under Kubelka-Munk mixing; a **minimum brush size**; and strokes
//! that follow an **orientation field** and are laid with the bristle brush (deposit + pickup). Crucially the
//! reference each brush layer paints from is BLURRED proportionally to the brush size, so a large brush has no
//! detail to trace — the constraints, not a matching objective, decide when a pass is done.
//!
//! This is the P0 make-or-break: if the output reads as a painterly filter with the budget and palette
//! constraints active, the core thesis is wrong. [`traceability`] measures how close the output is to the
//! full-resolution reference — a filter scores high, a painting lower (structure kept, surface invented).

use image::{imageops, RgbImage};

use crate::paint::canvas::Canvas;
use crate::paint::color::{self, Srgb};
use crate::paint::mixer;
use crate::paint::palette::Palette;
use crate::paint::score::{ScoreHeader, StrokeRecord, StrokeScore};
use crate::paint::stroke::{BrushConfig, Stroke};

/// One pass of the painter: a stage painted at a brush radius with its own stroke budget. When a plan (P1.3)
/// supplies passes, they drive the painter; otherwise the coarse→fine `brush_sizes` do.
#[derive(Clone, Debug)]
pub struct PassSpec {
    pub radius: f32,
    pub budget: usize,
    pub stage: String,
}

/// Parameters for a paint-from-image run.
#[derive(Clone, Debug)]
pub struct PaintParams {
    pub palette: Palette,
    /// Total stroke budget across all layers — inviolable.
    pub budget: usize,
    /// Explicit stage passes (from a compiled plan). `None` → derive passes from `brush_sizes`.
    pub passes: Option<Vec<PassSpec>>,
    /// Brush radii, coarse → fine. A radius below `min_brush` is skipped.
    pub brush_sizes: Vec<f32>,
    /// The smallest brush allowed, so the finest pass still cannot chase pixel detail.
    pub min_brush: f32,
    /// ARMATURE resolution (RFC §1.1/§5.1): the reference is coarsened to this longest-side before painting, so
    /// structure survives but DETAIL does not — the brush must invent the surface rather than trace it. `None`
    /// keeps the reference full-resolution (the P0 from-image behaviour).
    pub armature_side: Option<u32>,
    /// Charge multiplier for a stroke's load (how much paint the brush holds vs its footprint).
    pub charge: f32,
    /// Medium name recorded in the score header (physics still comes from `brush` in this slice).
    pub medium: String,
    /// RESERVE (surface-white media): a luma threshold in `[0,1]` — cells brighter than this get NO stroke, so
    /// the paper/ground shows through (watercolour whites, §6.3). `None` → paint everywhere.
    pub reserve: Option<f32>,
    /// DENSITY mark model (§8.6): build value by black hatch marks whose count scales with darkness, instead
    /// of loaded continuous strokes (pen-ink). Uses the palette's darkest pigment.
    pub density: bool,
    /// A TONED ground (imprimatura) to prime the canvas with — for opaque media, so light passages show.
    /// `None` = a white ground / paper.
    pub ground: Option<Srgb>,
    /// NEGATIVE PAINTING (§8.7): a protect mask (`w*h`, true = leave unpainted) — strokes are not seeded inside
    /// it, so a shape is DEFINED by painting the space around it. Generalises the `reserve`. `None` = paint all.
    pub protect: Option<Vec<bool>>,
    pub seed: u64,
    pub brush: BrushConfig,
}

impl PaintParams {
    /// A sensible default over a palette at a stroke budget.
    pub fn new(palette: Palette, budget: usize) -> Self {
        Self { palette, budget, passes: None, brush_sizes: vec![28.0, 14.0, 7.0], min_brush: 4.0, armature_side: None, charge: 6.0, medium: "oil-direct".into(), reserve: None, density: false, ground: None, protect: None, seed: 42, brush: BrushConfig::default() }
    }
}

/// The result of a paint run.
pub struct PaintResult {
    pub canvas: Canvas,
    /// How many strokes were actually laid (≤ budget).
    pub strokes: usize,
    /// The replayable stroke score — the canonical artifact.
    pub score: StrokeScore,
    /// Stages the critic rejected (rolled back) — empty without a critic.
    pub rejected: Vec<String>,
}

fn luma_map(img: &RgbImage) -> Vec<f32> {
    img.pixels().map(|p| color::linear_luma(color::srgb_to_linear(p.0))).collect()
}

/// Sobel gradient of a luma map → `(gx, gy, magnitude)` per pixel.
fn sobel(luma: &[f32], w: u32, h: u32) -> (Vec<f32>, Vec<f32>) {
    let (wi, hi) = (w as i32, h as i32);
    let at = |x: i32, y: i32| -> f32 {
        let x = x.clamp(0, wi - 1) as usize;
        let y = y.clamp(0, hi - 1) as usize;
        luma[y * w as usize + x]
    };
    let mut gx = vec![0f32; luma.len()];
    let mut gy = vec![0f32; luma.len()];
    for y in 0..hi {
        for x in 0..wi {
            let i = (y as usize) * w as usize + x as usize;
            gx[i] = at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1);
            gy[i] = at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1);
        }
    }
    (gx, gy)
}

/// The isophote (stroke) direction at a pixel: perpendicular to the luminance gradient. In a flat region the
/// gradient vanishes, so we fall back to horizontal.
fn stroke_dir(gx: f32, gy: f32) -> [f32; 2] {
    let m = (gx * gx + gy * gy).sqrt();
    if m < 1e-4 {
        [1.0, 0.0]
    } else {
        // perpendicular to (gx,gy) is (-gy,gx)
        [-gy / m, gx / m]
    }
}

/// Deterministic per-index jitter in `[-0.5,0.5]` (a hashed LCG — no RNG dependency, reproducible).
fn jitter(seed: u64, k: u64) -> f32 {
    let mut z = seed.wrapping_add(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z as f32 / u64::MAX as f32) - 0.5
}

/// sRGB distance (linear-RGB Euclidean) — cheap, for the placement error map.
fn rgb_dist(a: Srgb, b: Srgb) -> f32 {
    let (la, lb) = (color::srgb_to_linear(a), color::srgb_to_linear(b));
    ((la[0] - lb[0]).powi(2) + (la[1] - lb[1]).powi(2) + (la[2] - lb[2]).powi(2)).sqrt()
}

/// Grow a stroke in ONE direction (`sign` = +1 forward, −1 backward) from the seed along the orientation
/// field, ending when the reference colour drifts too far from the stroke's colour or the half-length cap hits.
fn grow_half(x0: f32, y0: f32, sign: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb, protect: Option<&[bool]>) -> Vec<[f32; 2]> {
    let (w, h) = (reference.width(), reference.height());
    let max_len = (radius * 2.5).max(radius + 1.0);
    let step = (radius * 0.6).max(1.0);
    let mut pts = Vec::new();
    let mut last = [0f32, 0f32];
    let (mut x, mut y) = (x0, y0);
    let mut travelled = 0.0;
    while travelled < max_len {
        let (ix, iy) = (x.round().clamp(0.0, w as f32 - 1.0) as usize, y.round().clamp(0.0, h as f32 - 1.0) as usize);
        let i = iy * w as usize + ix;
        // Negative painting: the stroke terminates at the protected shape's edge (paint AROUND it).
        if let Some(m) = protect {
            if m.get(i).copied().unwrap_or(false) {
                break;
            }
        }
        let mut d = stroke_dir(gx[i], gy[i]);
        d = [d[0] * sign, d[1] * sign];
        // Keep the direction from flipping 180° between steps.
        if travelled > 0.0 && last[0] * d[0] + last[1] * d[1] < 0.0 {
            d = [-d[0], -d[1]];
        }
        x += d[0] * step;
        y += d[1] * step;
        if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 {
            break;
        }
        // Stop where the reference no longer matches this stroke's colour (Hertzmann's rule), after a minimum
        // length so strokes read as paint, not one-step scribble.
        let here = reference.get_pixel(x as u32, y as u32).0;
        if rgb_dist(here, color0) > 0.18 && travelled > step * 1.5 {
            break;
        }
        pts.push([x, y]);
        last = d;
        travelled += step;
    }
    pts
}

/// Grow a stroke through the seed in BOTH directions (Hertzmann), so a seed mid-feature paints the whole
/// isophote it sits on, not just the half below it.
fn grow_path(x0: f32, y0: f32, radius: f32, gx: &[f32], gy: &[f32], reference: &RgbImage, color0: Srgb, protect: Option<&[bool]>) -> Vec<[f32; 2]> {
    let mut back = grow_half(x0, y0, -1.0, radius, gx, gy, reference, color0, protect);
    back.reverse();
    let fwd = grow_half(x0, y0, 1.0, radius, gx, gy, reference, color0, protect);
    back.push([x0, y0]);
    back.extend(fwd);
    back
}

/// Coarsen an image to an ARMATURE: downsample to `side` (longest edge) then upsample back, smoothly — so the
/// structure survives but the fine detail is gone. The brush then invents the surface instead of tracing it.
fn coarsen(img: &RgbImage, side: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let scale = side as f32 / w.max(h) as f32;
    if scale >= 1.0 {
        return img.clone();
    }
    let (dw, dh) = ((w as f32 * scale).round().max(1.0) as u32, (h as f32 * scale).round().max(1.0) as u32);
    let small = imageops::resize(img, dw, dh, imageops::FilterType::Triangle);
    imageops::resize(&small, w, h, imageops::FilterType::Triangle)
}

/// A pass-level CRITIC (RFC PAINT-1 §10.1): scores a rendered canvas so the painter can accept or reject a
/// whole PASS. Injected as a closure so the loop logic is testable offline (the aesthetic/CLIP scorer is wired
/// by the CLI). Higher is better.
pub type PassCritic<'a> = dyn Fn(&RgbImage) -> f32 + 'a;

/// Paint a reference image under the PAINT-1 constraints. See [`paint_critiqued`] for the pass-level critic.
pub fn paint_from_image(input: &RgbImage, p: &PaintParams) -> PaintResult {
    paint_inner(input, p, None, 0.0)
}

/// Paint with a pass-level CRITIC (§10.1): after each stage pass the canvas is scored; a pass that does not
/// improve the score by at least `margin` is REJECTED (the canvas + score are rolled back) and its stage
/// recorded as tabu, so the loop never keeps a configuration that made the painting worse. Pass-level only —
/// per-stroke scoring is prohibitively expensive and rejected outright (§10.1, N4).
pub fn paint_critiqued(input: &RgbImage, p: &PaintParams, critic: &PassCritic, margin: f32) -> PaintResult {
    paint_inner(input, p, Some(critic), margin)
}

fn paint_inner(input: &RgbImage, p: &PaintParams, critic: Option<&PassCritic>, margin: f32) -> PaintResult {
    let (w, h) = (input.width(), input.height());
    // The reference the strokes read is a low-resolution ARMATURE — structure without detail (§1.1). The output
    // canvas stays full size; only the thing being painted FROM is coarsened.
    let armature_owned;
    let input: &RgbImage = match p.armature_side.filter(|&s| s > 0) {
        Some(s) => {
            armature_owned = coarsen(input, s);
            &armature_owned
        }
        None => input,
    };
    let mut canvas = match p.ground {
        Some(tone) => Canvas::toned(w, h, p.palette, tone, 0.85),
        None => Canvas::white(w, h, p.palette, 0.85),
    };
    let n = p.palette.pigments.len();
    // Mixture cache keyed on the quantised reference colour — thousands of strokes sample similar colours.
    let mut cache: std::collections::HashMap<u32, Vec<f32>> = std::collections::HashMap::new();
    let mut mixture_for = |target: Srgb, palette: &Palette, charge: f32| -> Vec<f32> {
        let key = ((target[0] as u32 >> 2) << 12) | ((target[1] as u32 >> 2) << 6) | (target[2] as u32 >> 2);
        let base = cache.entry(key).or_insert_with(|| {
            let m = mixer::solve_mixture(palette, target, 3);
            let mut v = vec![0f32; n];
            for (&idx, &w) in m.pigments.iter().zip(m.weights.iter()) {
                v[idx] = w;
            }
            v
        });
        base.iter().map(|c| c * charge).collect()
    };

    let mut score = StrokeScore {
        header: ScoreHeader { version: 1, palette: p.palette.name.to_string(), medium: p.medium.clone(), seed: p.seed, width: w, height: h, tooth: 0.85, ground: p.ground, brush: p.brush },
        strokes: Vec::new(),
    };

    // The passes to run: an explicit plan (P1.3), else coarse→fine from brush_sizes (P0).
    let sizes: Vec<f32> = p.brush_sizes.iter().copied().filter(|&r| r >= p.min_brush).collect();
    let passes: Vec<PassSpec> = p.passes.clone().unwrap_or_else(|| {
        sizes.iter().enumerate().map(|(i, &r)| PassSpec { radius: r, budget: p.budget, stage: if i == 0 { "block-in".into() } else { "restate".into() } }).collect()
    });

    let mut placed = 0usize;
    let mut k = 0u64;
    let mut rejected: Vec<String> = Vec::new();
    for (layer, pass) in passes.iter().enumerate() {
        let radius = pass.radius.max(p.min_brush);
        // The first pass is a block-in: it covers the whole canvas so no white ground survives. Later passes
        // only restate where the canvas is still wrong.
        let block_in = layer == 0;
        if placed >= p.budget {
            break;
        }
        let mut in_pass = 0usize;
        // Critic: snapshot the canvas + score BEFORE the pass, so a pass that hurts can be rolled back.
        let snapshot = critic.map(|c| (canvas.clone(), score.strokes.len(), placed, c(&canvas.to_image())));
        // The reference this pass paints from is blurred ∝ the brush — a coarse brush has no detail to trace.
        let reference = imageops::blur(input, radius * 0.7);
        let luma = luma_map(&reference);
        let (gx, gy) = sobel(&luma, w, h);
        let grid = (radius * 0.9).max(1.5);

        let cols = ((w as f32) / grid).ceil() as u32;
        let rows = ((h as f32) / grid).ceil() as u32;
        for gyi in 0..rows {
            for gxi in 0..cols {
                if placed >= p.budget || in_pass >= pass.budget {
                    break;
                }
                k += 1;
                let jx = jitter(p.seed, k) * grid;
                let jy = jitter(p.seed, k.wrapping_add(1)) * grid;
                let cx = (gxi as f32 + 0.5) * grid + jx;
                let cy = (gyi as f32 + 0.5) * grid + jy;
                if cx < 0.0 || cy < 0.0 || cx >= w as f32 || cy >= h as f32 {
                    continue;
                }
                let (ix, iy) = (cx as u32, cy as u32);
                // NEGATIVE PAINTING: never seed a stroke inside the protected shape — paint around it.
                if let Some(mask) = &p.protect {
                    if mask.get(iy as usize * w as usize + ix as usize).copied().unwrap_or(false) {
                        continue;
                    }
                }
                let target = reference.get_pixel(ix, iy).0;
                let tluma = color::linear_luma(color::srgb_to_linear(target));
                // RESERVE the whites for a surface-white medium: bright cells keep the paper (no stroke enters).
                if let Some(rt) = p.reserve {
                    if tluma > rt {
                        continue;
                    }
                }
                // Later layers only restate where the canvas is still notably wrong; the block-in covers all.
                if !block_in && !p.density && rgb_dist(canvas.color_at(ix, iy), target) < 0.06 {
                    continue;
                }
                // DENSITY mark model: build value with black hatch marks whose count scales with darkness.
                if p.density {
                    let n = density_marks(&mut canvas, &mut score, &mut placed, &mut in_pass, pass.budget, p.budget, cx, cy, radius, tluma, &gx, &gy, p, &pass.stage);
                    k = k.wrapping_add(n as u64);
                    continue;
                }
                let load = mixture_for(target, &p.palette, p.charge);
                let path = grow_path(cx, cy, radius, &gx, &gy, &reference, target, p.protect.as_deref());
                let s = Stroke { path, width0: radius, width1: (radius * 0.6).max(p.min_brush * 0.6), load, pressure: 1.0, wetness: 1.0 };
                s.rasterize(&mut canvas, &p.brush);
                // Record the stroke into the score (mix as pigment name → value, for the non-zero pigments).
                let mix: Vec<(String, f32)> = s.load.iter().enumerate().filter(|(_, v)| **v > 0.0).map(|(i, v)| (p.palette.pigments[i].name.to_string(), *v)).collect();
                placed += 1;
                in_pass += 1;
                score.strokes.push(StrokeRecord {
                    id: placed as u32,
                    wipe: false,
                    stage: pass.stage.clone(),
                    spline: s.path,
                    w0: s.width0,
                    w1: s.width1,
                    taper: 0.4,
                    mix,
                    wet: s.wetness,
                    press: s.pressure,
                });
            }
        }

        // Critic verdict: if the pass didn't improve the score by `margin`, ROLL BACK the canvas + score and
        // record the stage as tabu (§10.1). The block-in is never rejected — the canvas must be covered.
        if let (Some(c), Some((snap_canvas, snap_len, snap_placed, before))) = (critic, snapshot) {
            let after = c(&canvas.to_image());
            if !block_in && after < before + margin {
                canvas = snap_canvas;
                score.strokes.truncate(snap_len);
                placed = snap_placed;
                rejected.push(pass.stage.clone());
            }
        }
    }
    if !rejected.is_empty() {
        tracing::info!(target: "plakat", "paint critic: rejected {} pass(es): {}", rejected.len(), rejected.join(", "));
    }
    PaintResult { canvas, strokes: placed, score, rejected }
}

/// The density mark model (§8.6): at a cell, lay black hatch marks whose count scales with the target
/// darkness — value is built by mark DENSITY, not pigment concentration. Darker cells add a crosshatch. No
/// pickup, single bristle. Records each mark into the score; respects the budget. Returns the count laid.
#[allow(clippy::too_many_arguments)]
fn density_marks(
    canvas: &mut Canvas,
    score: &mut StrokeScore,
    placed: &mut usize,
    in_pass: &mut usize,
    pass_budget: usize,
    total_budget: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    tluma: f32,
    gx: &[f32],
    gy: &[f32],
    p: &PaintParams,
    stage: &str,
) -> usize {
    let darkness = (1.0 - tluma).clamp(0.0, 1.0);
    let n = (darkness * 6.0).round() as usize;
    if n == 0 {
        return 0;
    }
    let (w, h) = (canvas.w, canvas.h);
    let ci = (cy.round().clamp(0.0, h as f32 - 1.0) as usize) * w as usize + (cx.round().clamp(0.0, w as f32 - 1.0) as usize);
    let dir = stroke_dir(gx[ci], gy[ci]);
    let perp = [-dir[1], dir[0]];
    let ink = crate::paint::canvas::darkest_pigment(&p.palette);
    let mut load = vec![0f32; p.palette.pigments.len()];
    load[ink] = p.charge;
    let mut brush = p.brush;
    brush.k_pickup = 0.0;
    brush.bristles = 1;
    let half = radius * 0.5;
    let mut laid = 0usize;
    for m in 0..n {
        if *placed >= total_budget || *in_pass >= pass_budget {
            break;
        }
        let off = ((m as f32 + 0.5) / n as f32 - 0.5) * radius;
        // Crosshatch the darkest cells: alternate marks run perpendicular.
        let (d, spread) = if m % 2 == 1 && darkness >= 0.6 { (perp, dir) } else { (dir, perp) };
        let (mx, my) = (cx + spread[0] * off, cy + spread[1] * off);
        let a = [mx - d[0] * half, my - d[1] * half];
        let b = [mx + d[0] * half, my + d[1] * half];
        let s = Stroke { path: vec![a, b], width0: 1.5, width1: 1.5, load: load.clone(), pressure: 1.0, wetness: 0.0 };
        s.rasterize(canvas, &brush);
        *placed += 1;
        *in_pass += 1;
        laid += 1;
        score.strokes.push(StrokeRecord {
            id: *placed as u32,
            wipe: false,
            stage: stage.to_string(),
            spline: s.path,
            w0: 1.5,
            w1: 1.5,
            taper: 0.0,
            mix: vec![(p.palette.pigments[ink].name.to_string(), p.charge)],
            wet: 0.0,
            press: 1.0,
        });
    }
    laid
}

/// Traceability (RFC PAINT-1 §12.1): the mean linear-luma correlation between a painted image and its
/// reference at full resolution. High (→1) means the painting traced the reference (a filter); a real painting
/// keeps the structure but invents the surface, so it correlates moderately, not perfectly.
pub fn traceability(painted: &RgbImage, reference: &RgbImage) -> f32 {
    if painted.dimensions() != reference.dimensions() {
        return f32::NAN;
    }
    let a = luma_map(painted);
    let b = luma_map(reference);
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::palette;

    fn gradient_img(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, _y| {
            let t = x as f32 / w as f32;
            image::Rgb([(40.0 + 180.0 * t) as u8, (60.0 + 120.0 * t) as u8, (30.0 + 60.0 * t) as u8])
        })
    }

    #[test]
    fn painter_respects_the_budget_and_is_deterministic() {
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 120);
        p.brush_sizes = vec![16.0, 8.0];
        let a = paint_from_image(&img, &p);
        assert!(a.strokes <= 120, "budget honoured ({} ≤ 120)", a.strokes);
        assert!(a.strokes > 0, "some strokes laid");
        let b = paint_from_image(&img, &p);
        assert_eq!(a.canvas.to_image().into_raw(), b.canvas.to_image().into_raw(), "deterministic");
    }

    #[test]
    fn reserve_keeps_bright_cells_as_paper() {
        // A gradient dark→light; reserving above luma 0.6 leaves the bright right side unpainted (near white).
        let img = gradient_img(80, 30);
        let mut p = PaintParams::new(palette::SUMI, 400);
        p.brush_sizes = vec![10.0];
        p.reserve = Some(0.55);
        let out = paint_from_image(&img, &p).canvas.to_image();
        // The brightest column stays near the paper white (reserved); the dark left gets painted.
        let right = out.get_pixel(78, 15).0;
        assert!(right[0] > 235 && right[1] > 235, "bright side reserved (paper): {right:?}");
    }

    #[test]
    fn density_marks_build_value_by_count() {
        // Pen-ink density over a dark→light gradient: the dark side accumulates more black marks (lower luma)
        // than the light side.
        let img = gradient_img(80, 40);
        let mut p = PaintParams::new(palette::SUMI, 4000);
        p.brush_sizes = vec![8.0];
        p.density = true;
        let out = paint_from_image(&img, &p).canvas.to_image();
        // Density is statistical — average over a patch on each side rather than sampling one pixel.
        let patch_mean = |x0: u32, x1: u32| -> f32 {
            let mut s = 0.0;
            let mut n = 0.0;
            for y in 8..32 {
                for x in x0..x1 {
                    s += out.get_pixel(x, y).0[0] as f32;
                    n += 1.0;
                }
            }
            s / n
        };
        let dark = patch_mean(2, 18);
        let light = patch_mean(62, 78);
        assert!(dark < light - 15.0, "hatch density darkens the dark side more ({dark:.0} vs {light:.0})");
    }

    #[test]
    fn critic_rejects_passes_that_dont_improve() {
        let img = gradient_img(48, 48);
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![16.0, 8.0, 4.0]; // → passes: block-in, restate, restate
        // A FLAT critic: no pass ever improves the score, so every non-block-in pass is rolled back.
        let flat = |_img: &RgbImage| 0.5f32;
        let r = paint_critiqued(&img, &p, &flat, 0.01);
        assert_eq!(r.rejected.len(), 2, "the two restate passes rejected; the block-in is never rejected");
        // An IMPROVING critic (rewards coverage): passes are kept.
        let rewarding = |img: &RgbImage| img.pixels().map(|px| 255 - px.0[0] as i32).sum::<i32>() as f32;
        let r2 = paint_critiqued(&img, &p, &rewarding, 0.0);
        assert!(r2.rejected.len() < 2, "passes that improve the score are kept ({} rejected)", r2.rejected.len());
    }

    #[test]
    fn negative_painting_leaves_the_protected_shape() {
        // Paint around a protected central square — the shape stays (near) the ground while the surround is
        // painted, defining the shape by negative space.
        let img = gradient_img(60, 60);
        let mut p = PaintParams::new(palette::EARTH, 600);
        p.brush_sizes = vec![8.0];
        let mut mask = vec![false; 60 * 60];
        for y in 22..38 {
            for x in 22..38 {
                mask[y * 60 + x] = true;
            }
        }
        p.protect = Some(mask);
        let c = paint_from_image(&img, &p).canvas;
        // The protected centre is (near) the bare white ground; a surround pixel is painted (has height).
        assert!(c.height[30 * 60 + 30] < 0.05, "protected shape left unpainted");
        assert!(c.height[30 * 60 + 5] > 0.0 || c.height[10 * 60 + 30] > 0.0, "the surround is painted");
    }

    #[test]
    fn a_low_res_armature_paints_from_structure_not_detail() {
        // A detailed reference (fine checker over a gradient). Painting from a low-res ARMATURE keeps the broad
        // structure (still correlates) but the finest checker detail is gone, so it traces LESS than painting
        // the full-resolution reference.
        let img = image::RgbImage::from_fn(96, 96, |x, y| {
            let t = (x as f32 / 96.0 * 200.0) as u8;
            let checker = if (x / 3 + y / 3) % 2 == 0 { 40 } else { 0 };
            image::Rgb([t.saturating_add(checker), (120 + checker) as u8, (60 + checker) as u8])
        });
        let mut full = PaintParams::new(palette::EARTH, 400);
        full.brush_sizes = vec![18.0, 9.0];
        let mut arm = full.clone();
        arm.armature_side = Some(24);
        let tr_full = traceability(&paint_from_image(&img, &full).canvas.to_image(), &img);
        let tr_arm = traceability(&paint_from_image(&img, &arm).canvas.to_image(), &img);
        assert!(tr_arm > 0.1, "some broad structure survives, not random (corr {tr_arm})");
        assert!(tr_arm <= tr_full + 0.02, "the armature traces no MORE of the detailed reference than full-res ({tr_arm} vs {tr_full})");
    }

    #[test]
    fn the_score_faithfully_records_the_paint() {
        // Replaying a paint's own score at native size reproduces the canvas byte-for-byte — the score IS the
        // painting (A4).
        let img = gradient_img(64, 48);
        let mut p = PaintParams::new(palette::EARTH, 150);
        p.brush_sizes = vec![16.0, 8.0];
        let result = paint_from_image(&img, &p);
        let painted = result.canvas.to_image().into_raw();
        let replayed = result.score.replay(64, 48).unwrap().to_image().into_raw();
        assert_eq!(painted, replayed, "score replay == the original paint at native size");
        assert_eq!(result.score.strokes.len(), result.strokes, "one record per stroke laid");
    }

    #[test]
    fn painting_keeps_structure_without_perfectly_tracing() {
        // The painted output should correlate with the reference (structure survives) but NOT perfectly
        // (surface is invented, budget/palette constrain it) — the filter-gate signal.
        let img = gradient_img(80, 60);
        let mut p = PaintParams::new(palette::EARTH, 300);
        p.brush_sizes = vec![20.0, 10.0];
        let out = paint_from_image(&img, &p).canvas.to_image();
        let tr = traceability(&out, &img);
        assert!(tr > 0.5, "structure survives (corr {tr})");
        assert!(tr < 0.999, "not a pixel-perfect trace (corr {tr})");
    }
}
