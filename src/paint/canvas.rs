//! The pigment canvas (RFC PAINT-1 §8.1). The canvas does NOT carry RGB — it carries, per pixel, a **pigment
//! concentration vector** over the active palette plus paint **height**, **wetness**, and a static **tooth**.
//! sRGB is computed only on output, by Kubelka-Munk mixing the concentration vector. This is what makes mixing,
//! glazing, pickup, and mud *physical* rather than faked: a heavy stroke shifts the concentration ratio and
//! dominates (opaque), a thin one barely perturbs it and the ground shows through (glaze).
//!
//! Colour is a function of the concentration *ratio*, so laying more of the same pigment doesn't darken —
//! only introducing other pigment does. Total concentration is tracked as **saturation** (how full the tooth
//! is), which throttles further deposit so paint build-up is bounded.

use image::RgbImage;

use crate::paint::color::{self, LinRgb, Srgb};
use crate::paint::palette::Palette;
use crate::paint::pigment;

/// A thin priming's worth of ground pigment — used only to derive the ground COLOUR in the constructors.
pub const GROUND_CONC: f32 = 0.25;

/// Film-build opacity: how fast a pixel's paint HIDES the ground as concentration accumulates. Opacity is
/// `1 − e^(−OPACITY_K · total_concentration)`, so a thin glaze lets the ground show (light) and a built stroke
/// becomes opaque — which is what restores deep darks, bright lights, and saturated colour. Without this the
/// ground stays a permanent proportion of every pixel and the whole painting washes out to mid-value.
const OPACITY_K: f32 = 1.6;

/// A pigment canvas over a fixed palette basis.
#[derive(Clone)]
pub struct Canvas {
    pub w: u32,
    pub h: u32,
    palette: Palette,
    /// Per-pixel concentration over the palette: `w*h*n`, row-major, pixel-major. This is DEPOSITED paint only
    /// — the ground is NOT baked in here; it shows through via opacity where the paint is thin. It is the
    /// pigment RATIO of the visible film: a new deposit COVERS what lies beneath (see [`Canvas::deposit`]), so
    /// this is not a running sum over every stroke ever laid.
    conc: Vec<f32>,
    /// Per-pixel paint AMOUNT (the sum of every deposit, minus wipes), row-major. Drives the film-build opacity
    /// over the ground and the tooth saturation; kept apart from `conc` so covering a light block-in with a
    /// dark restatement changes the film's colour without thinning it.
    film: Vec<f32>,
    /// The film at the last `mark_film` (empty = never marked), for a wash brush's film cap.
    film_mark: Vec<f32>,
    /// The pigment as it was at the last `mark_film` (transmittance film only — there deposits ADD, so the
    /// difference is exactly what the pass since laid; see `flow`).
    conc_mark: Vec<f32>,
    /// Per-pixel paint height (impasto), row-major.
    pub height: Vec<f32>,
    /// Per-pixel wet pigment available for pickup, row-major (0 = dry).
    pub wetness: Vec<f32>,
    /// Per-pixel static surface tooth in `[0,1]` (how much the brush catches), row-major.
    pub tooth: Vec<f32>,
    /// The ground's linear reflectance — shown through where the paint film is thin (glaze) and hidden where
    /// it's built up (opaque). Stored separately so it can never dilute the paint's own colour as a proportion.
    ground_lin: LinRgb,
    /// BODY / opacity multiplier on the film-build (1 = opaque media; lower = transparent, the ground glows
    /// through more even as paint builds — watercolour, ink).
    opacity: f32,
    /// Hiding power (the film-build's `OPACITY_K`). Measured: a full block-in already hides 96% of the ground at
    /// the default, so this is NOT the lever for a light/grey painting (that was the palette gamut).
    opacity_k: f32,
    /// TRANSMITTANCE film (transparent media): the film is a stack of pigment densities and the colour is the
    /// ground seen through it — `ground × Π R_i^(density_i)` (Beer–Lambert) — so a dark is the SAME hue at a
    /// higher density, a tint keeps its hue, and two glazes multiply. Off: the covering film-build model.
    transmittance: bool,
    /// `ln R_i` per pigment (linear reflectance), filled when `transmittance` is on.
    ln_r: Vec<[f32; 3]>,
    /// A CLIP: while set, strokes lay paint only where it is true (a wash brush working INSIDE a shape —
    /// the shape's edge stays exact however the strokes overshoot). `None` = no clip.
    clip: Option<Vec<bool>>,
    n: usize,
}

/// A 3×3 box blur of a per-pixel field (edge-clamped).
fn box_blur3(f: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum += f[ny * w + nx];
                }
            }
            out[y * w + x] = sum / 9.0;
        }
    }
    out
}

/// Pigment drift (see `Canvas::bleed_with`): the share of a cell's load that crosses to a neighbour per step
/// at full difference, the load difference (as a fraction of the wet cells' mean load) that counts as full, and
/// the drift steps taken after each bloom iteration (each step travels one pixel). Calibrated on a 1024²
/// watercolour: rate 0.25 is stable (0.5 piled pigment into a stipple); 4 steps give a visible but unbroken
/// full-scale effect (+1 moves the picture by DSSIM ~0.02, -1 ~0.004 — gathering feeds itself, spreading
/// equalises, so the light side is inherently the gentler one).
const DRIFT_RATE: f32 = 0.25;
const DRIFT_TAU: f32 = 0.3;
const DRIFT_STEPS: usize = 4;
/// The most of its load a cell may give away in ONE drift step, over all four neighbours together. Without it
/// a cell between darker neighbours could be drained to bare paper (white specks in a stipple of dark dots —
/// seen at rate 0.5); with it the failure is impossible whatever the rate.
const DRIFT_MAX_OUT: f32 = 0.5;

impl Canvas {
    /// A canvas primed with a ground: the given concentration vector at every pixel (length = palette size).
    /// `tooth` is uniform. `wetness`/`height` start at zero.
    pub fn with_ground(w: u32, h: u32, palette: Palette, ground: &[f32], tooth: f32) -> Self {
        let n = palette.pigments.len();
        let px = (w * h) as usize;
        let mut g = vec![0f32; n];
        for i in 0..n.min(ground.len()) {
            g[i] = ground[i].max(0.0);
        }
        // The ground contributes only its COLOUR (reflectance), stored separately; the canvas starts bare (no
        // deposited pigment) so an opaque stroke hides it by film build rather than mixing with it forever.
        let gsum: f32 = g.iter().sum();
        let ground_lin = if gsum > 0.0 { pigment::mix_linear(palette.pigments, &g) } else { color::srgb_to_linear([255, 255, 255]) };
        Self { w, h, palette, conc: vec![0.0; px * n], film: vec![0.0; px], film_mark: Vec::new(), conc_mark: Vec::new(), height: vec![0.0; px], wetness: vec![0.0; px], tooth: vec![tooth.clamp(0.0, 1.0); px], ground_lin, opacity: 1.0, opacity_k: OPACITY_K, transmittance: false, ln_r: Vec::new(), clip: None, n }
    }

    /// Mean film-build opacity over the canvas (0 = bare ground everywhere, 1 = fully hidden) — how much of the
    /// picture is paint versus ground showing through. A diagnostic for the "too light / too grey" tell.
    pub fn mean_film_alpha(&self) -> f32 {
        let px = (self.w * self.h) as usize;
        if px == 0 {
            return 0.0;
        }
        let mut acc = 0f64;
        for &total in &self.film {
            acc += (1.0 - (-self.opacity_k * self.opacity * total.max(0.0)).exp()) as f64;
        }
        (acc / px as f64) as f32
    }

    /// Set the clip to the inside of `rings` (NaN-separated polygons, even-odd, pixel centres); fewer than
    /// three points clears it.
    pub fn set_clip_rings(&mut self, rings: &[[f32; 2]]) {
        let (w, h) = (self.w as usize, self.h as usize);
        let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
        let mut ring: Vec<[f32; 2]> = Vec::new();
        let flush = |ring: &mut Vec<[f32; 2]>, edges: &mut Vec<([f32; 2], [f32; 2])>| {
            if ring.len() >= 3 {
                for i in 0..ring.len() {
                    edges.push((ring[i], ring[(i + 1) % ring.len()]));
                }
            }
            ring.clear();
        };
        for pt in rings {
            if pt[0].is_nan() || pt[1].is_nan() {
                flush(&mut ring, &mut edges);
            } else {
                ring.push(*pt);
            }
        }
        flush(&mut ring, &mut edges);
        if edges.is_empty() {
            self.clip = None;
            return;
        }
        let mut mask = vec![false; w * h];
        let mut xs: Vec<f32> = Vec::new();
        for y in 0..h {
            let yc = y as f32 + 0.5;
            xs.clear();
            for (a, b) in &edges {
                let (ya, yb) = (a[1], b[1]);
                if (ya <= yc && yb > yc) || (yb <= yc && ya > yc) {
                    let t = (yc - ya) / (yb - ya);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            if xs.len() < 2 {
                continue;
            }
            xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let xa = (pair[0] - 0.5).ceil().max(0.0) as i64;
                let xb = (pair[1] - 0.5).floor().min(w as f32 - 1.0) as i64;
                for x in xa..=xb {
                    mask[y * w + x as usize] = true;
                }
            }
        }
        self.clip = Some(mask);
    }

    /// Whether row-major pixel `p` is outside the current clip (never, when no clip is set).
    pub fn clipped(&self, p: usize) -> bool {
        self.clip.as_ref().is_some_and(|c| !c[p])
    }

    /// Switch the TRANSMITTANCE film on (see the field). Builder-style.
    pub fn with_transmittance(mut self, on: bool) -> Self {
        self.transmittance = on;
        self.ln_r = if on {
            self.palette
                .pigments
                .iter()
                .map(|p| {
                    let r = color::srgb_to_linear(p.masstone);
                    [r[0].clamp(0.004, 1.0).ln(), r[1].clamp(0.004, 1.0).ln(), r[2].clamp(0.004, 1.0).ln()]
                })
                .collect()
        } else {
            Vec::new()
        };
        self
    }

    /// Whether the transmittance film is on.
    pub fn is_transmittance(&self) -> bool {
        self.transmittance
    }

    /// `ln R` per pigment, for the transmittance solver (empty unless the film is on).
    pub fn ln_reflectance(&self) -> &[[f32; 3]] {
        &self.ln_r
    }

    /// The ground's linear reflectance (the paper, or the toned priming).
    pub fn ground_linear(&self) -> LinRgb {
        self.ground_lin
    }

    /// The linear reflectance of a pixel.
    pub fn linear_at(&self, x: u32, y: u32) -> LinRgb {
        color::srgb_to_linear(self.color_at(x, y))
    }

    /// Set the BODY / opacity multiplier (1 = opaque; lower = transparent). Builder-style.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.1, 1.0);
        self
    }

    /// A canvas primed with a WHITE ground (the lightest pigment in the palette), i.e. bare paper / gessoed
    /// panel. The default for opaque media before an imprimatura. The ground carries only a MODEST
    /// concentration — a thin priming — so an opaque stroke's deposit overcomes it (covers), rather than the
    /// substrate contributing forever in the concentration-ratio mix.
    pub fn white(w: u32, h: u32, palette: Palette, tooth: f32) -> Self {
        let white = lightest_pigment(&palette);
        let mut ground = vec![0f32; palette.pigments.len()];
        ground[white] = GROUND_CONC;
        Self::with_ground(w, h, palette, &ground, tooth)
    }

    /// A canvas primed with a TONED ground (imprimatura): the given tone, solved into the palette. Opaque media
    /// work on a mid-tone so light passages SHOW (light paint on a white ground is invisible) and the picture
    /// reads as one keyed surface rather than sparse marks on paper.
    pub fn toned(w: u32, h: u32, palette: Palette, tone: Srgb, tooth: f32) -> Self {
        let m = crate::paint::mixer::solve_mixture(&palette, tone, 3);
        let mut ground = vec![0f32; palette.pigments.len()];
        for (&i, &wt) in m.pigments.iter().zip(m.weights.iter()) {
            ground[i] = wt * GROUND_CONC;
        }
        Self::with_ground(w, h, palette, &ground, tooth)
    }

    pub fn palette(&self) -> &Palette {
        &self.palette
    }
    pub fn n_pigments(&self) -> usize {
        self.n
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize * self.w as usize + x as usize) * self.n
    }

    /// The concentration vector at a pixel.
    pub fn conc_at(&self, x: u32, y: u32) -> &[f32] {
        let i = self.idx(x, y);
        &self.conc[i..i + self.n]
    }

    /// Total paint AMOUNT at a pixel (every deposit ever laid here, minus wipes) — the "saturation" that
    /// throttles further deposit and the film thickness that hides the ground.
    pub fn saturation_at(&self, x: u32, y: u32) -> f32 {
        self.film[y as usize * self.w as usize + x as usize]
    }

    /// Deposit a concentration delta at a pixel and raise its height. Paint COVERS: the deposit hides the film
    /// beneath it by the same film-build law that hides the ground (`1 − e^(−k · opacity · amount)`), so an
    /// opaque medium's restatement replaces the colour under it (a dark laid over the light block-in reads dark)
    /// while a thin or transparent deposit (a glaze, watercolour body) still mixes with what is there. Without
    /// this, every stroke ever laid stayed a permanent proportion of the pixel — a dark mass restated over a
    /// light block-in could only nudge the average, leaving every shadow mass with a light core and a dark rim.
    /// The paint amount (`film`) keeps accumulating, so covering never thins the film or re-exposes the ground.
    pub fn deposit(&mut self, x: u32, y: u32, delta: &[f32], height: f32) {
        let i = self.idx(x, y);
        let amount: f32 = delta.iter().take(self.n).map(|d| d.max(0.0)).sum();
        if self.transmittance {
            // Densities ADD: a glaze over a glaze is both.
            for c in 0..self.n {
                self.conc[i + c] += delta.get(c).copied().unwrap_or(0.0).max(0.0);
            }
        } else {
            let hide = 1.0 - (-self.opacity_k * self.opacity * amount).exp();
            for c in 0..self.n {
                self.conc[i + c] = self.conc[i + c] * (1.0 - hide) + delta.get(c).copied().unwrap_or(0.0).max(0.0);
            }
        }
        let p = y as usize * self.w as usize + x as usize;
        self.film[p] += amount;
        self.height[p] += height.max(0.0);
    }

    /// Remember the film as it is now (see `BrushConfig::film_cap`): the start of a wash pass.
    pub fn mark_film(&mut self) {
        self.film_mark = self.film.clone();
        if self.transmittance {
            self.conc_mark = self.conc.clone();
        }
    }

    /// The film laid at row-major pixel `p` since the last `mark_film` (all of it when never marked).
    pub fn film_since_mark(&self, p: usize) -> f32 {
        self.film[p] - self.film_mark.get(p).copied().unwrap_or(0.0)
    }

    /// WIPE / scrape back at a pixel (RFC PAINT-1 §8.7): remove a `strength` fraction of the accumulated
    /// pigment and height, exposing the ground/earlier work beneath (Sargent scraping a head, Turner wiping,
    /// watercolour lifting). A structural move, not error correction.
    pub fn wipe(&mut self, x: u32, y: u32, strength: f32) {
        let s = strength.clamp(0.0, 1.0);
        let i = self.idx(x, y);
        // Scale the deposited paint down — thinning the film re-exposes the ground through the opacity model, so
        // there's no need to inject ground pigment back into the mix.
        for c in 0..self.n {
            self.conc[i + c] *= 1.0 - s;
        }
        let p = y as usize * self.w as usize + x as usize;
        self.film[p] *= 1.0 - s;
        self.height[p] *= 1.0 - s;
    }

    /// DRY the whole canvas between passes: scale every pixel's wetness by `keep` (0 = bone dry, 1 = leave fully
    /// wet). A brush picks up canvas pigment in proportion to wetness, so a canvas that never dries lets each pass
    /// lift and re-deposit the masses beneath — the mechanism behind the muddy/smeared look. Drying between passes
    /// lets the block-in set before the restatement, so later marks read as fresh overlays, not stirred mud.
    pub fn dry(&mut self, keep: f32) {
        let k = keep.clamp(0.0, 1.0);
        if k >= 1.0 {
            return;
        }
        for w in &mut self.wetness {
            *w *= k;
        }
    }

    /// Wet-into-wet BLEED (watercolour / ink-wash): diffuse the deposited pigment into WET neighbours, so
    /// colours fuse and bloom at the edges. `strength` (0..1) scales both the spread and how far it reaches;
    /// only WET pixels bleed, so dry paint keeps its edge. Deterministic — replay reproduces it from the same
    /// wetness state. A no-op at `strength == 0` (the dry media).
    pub fn bleed(&mut self, strength: f32) {
        self.bleed_with(strength, 0.0);
    }

    /// `bleed` with a signed PIGMENT DIFFUSION (`diffuse` in -1..+1). In a real wash the pigment does not only
    /// fuse: it TRAVELS. +1 favours the darks — pigment migrates from a lighter wet cell into its darker, more
    /// loaded neighbour (the darks charge up, the lights stay clean, the edge stays crisp on the light side);
    /// -1 favours the lights — pigment spreads out of the loaded passages into the lighter wet paper around them
    /// (feathered halos, softened edges). 0 = the plain isotropic bleed, byte-identical to before. The drift is
    /// mass-conserving (what one cell loses its neighbour gains), gated by BOTH cells' wetness (dry or reserved
    /// paper neither gives nor takes), and runs after each bloom iteration so it reaches as far as the bloom.
    pub fn bleed_with(&mut self, strength: f32, diffuse: f32) {
        let s = strength.clamp(0.0, 1.0);
        if s <= 0.0 {
            return;
        }
        let d = diffuse.clamp(-1.0, 1.0);
        let (w, h) = (self.w as usize, self.h as usize);
        let n = self.n;
        let iters = (1.0 + 5.0 * s).round() as usize; // more strength → farther bloom
        for _ in 0..iters {
            let src = self.conc.clone();
            let src_film = self.film.clone();
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let a = s * self.wetness[p].clamp(0.0, 1.0);
                    if a <= 1e-4 {
                        continue;
                    }
                    // The paint amount blooms with the pigment, so the wash's edge thins as it spreads.
                    let mut fsum = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                            let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                            fsum += src_film[ny * w + nx];
                        }
                    }
                    self.film[p] = self.film[p] * (1.0 - a) + (fsum / 9.0) * a;
                    for c in 0..n {
                        let mut sum = 0.0;
                        let mut cnt = 0.0;
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                                let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                                sum += src[(ny * w + nx) * n + c];
                                cnt += 1.0;
                            }
                        }
                        let avg = sum / cnt;
                        let idx = p * n + c;
                        self.conc[idx] = self.conc[idx] * (1.0 - a) + avg * a;
                    }
                }
            }
            if d.abs() > 1e-4 {
                for _ in 0..DRIFT_STEPS {
                    self.pigment_drift(s, d, DRIFT_RATE);
                }
            }
        }
    }

    /// THE FLUID STAGE of a watercolour (`flow`): water on paper is a continuous FILM, not the brush's
    /// footprint, and only the pigment still IN the water moves — what dried in the earlier passes stays
    /// where it was laid. The pass's own deposit (since `mark_film`; the transmittance film adds, so the
    /// difference is exact) is the wet layer; it is split into WASHES — the regions where one pigment leads
    /// the mix (a watercolourist lays the sky, the wall, the lit window as separate washes, each of its own
    /// colour) — and inside each wash the fresh pigment DIFFUSES by `strength` over `radius` px (a blur
    /// confined to the wash, so the lanes' gaps are filled: the plain bleed, run on the lanes' own wetness,
    /// diffused pigment in a lattice of wet cells over dry gaps and printed a honeycomb) and SETTLES toward
    /// the wash's edge as it dries, deeper there by `rim` (0..1): the tide line between two washes, the
    /// cauliflower against the dry paper. And the pigment GRANULATES in the water by `grain` (0..1): a
    /// granulating pigment (an earth; read from the masstone's chroma, a staining dye does not) settles
    /// into the paper's tooth where the wash POOLED — the heavier the water, the heavier the mottle — so the
    /// grain is a fact of the wash, not a uniform screen over the sheet. And the water is SELECTIVE by
    /// `selective` (0..1): a painter floods the soft masses wet-in-wet and keeps the architecture wet-on-dry,
    /// so the diffusion holds back near the picture's HARD EDGES — read from the canvas itself (the colour
    /// the sheet shows at this moment, so a replay sees the same edges), not from the source. Deterministic;
    /// a no-op at strength 0 or on a non-transmittance canvas.
    pub fn flow(&mut self, strength: f32, radius: f32, rim: f32, grain: f32, selective: f32) {
        let s = strength.clamp(0.0, 1.0);
        if s <= 0.0 || radius < 0.5 || !self.transmittance {
            return;
        }
        let (w, h, n) = (self.w as usize, self.h as usize, self.n);
        let px = w * h;
        let r = radius.round().max(1.0) as usize;
        let film_mark = |p: usize| self.film_mark.get(p).copied().unwrap_or(0.0);
        // The pass's own deposit: the pigment that is still wet.
        let fresh_film: Vec<f32> = (0..px).map(|p| (self.film[p] - film_mark(p)).max(0.0)).collect();
        if !fresh_film.iter().any(|&f| f > 1e-5) {
            return;
        }
        let mut fresh = vec![0f32; px * n];
        for i in 0..px * n {
            fresh[i] = (self.conc[i] - self.conc_mark.get(i).copied().unwrap_or(0.0)).max(0.0);
        }
        // 1. The washes: the wet layer smoothed over the radius (the water joins the lanes), each pixel
        //    assigned to the pigment that leads the smoothed mix there. The water's REACH beyond the deposit
        //    is not one radius all along the edge: it runs further where the paper's fibres carry it and
        //    stops short where they do not (capillary action), so the wash ends in FINGERS, not in the
        //    smooth support of a blur. The reach is a slow value noise at the radius' scale, and it gates
        //    where the smoothed water counts as wet.
        let wet_b = Self::box_blur_f(&fresh_film, w, h, r);
        let reach: Vec<f32> = (0..px).map(|p| {
            let (x, y) = (p % w, p / w);
            let fx = x as f32 / (r as f32 * 1.2);
            let fy = y as f32 / (r as f32 * 1.2);
            0.25 + 0.75 * value_noise(fx, fy, 0xF1BE_5EED)
        }).collect();
        // A deposit's smoothed water falls from its level at the deposit to ~0 one radius out; the finger gate
        // asks for more of it where the reach is short. Judged against the local deposit level so a thin
        // wash fingers as a heavy one does.
        let level = {
            let mut m: Vec<f32> = fresh_film.iter().map(|&f| if f > 1e-5 { 1.0 } else { 0.0 }).collect();
            let mut t = m.clone();
            for y in 0..h { for x in 0..w { let mut mx = 0f32; for d in x.saturating_sub(r)..=(x + r).min(w - 1) { mx = mx.max(m[y * w + d]); } t[y * w + x] = mx; } }
            for y in 0..h { for x in 0..w { let mut mx = 0f32; for d in y.saturating_sub(r)..=(y + r).min(h - 1) { mx = mx.max(t[d * w + x]); } m[y * w + x] = mx; } }
            Self::box_blur_f(&m, w, h, r)
        };
        let wet_here = |p: usize| -> bool { fresh_film[p] > 1e-5 || (wet_b[p] > 1e-6 && level[p] > 1.0 - reach[p]) };
        let mut ch = vec![0f32; px];
        let mut lead: Vec<i16> = vec![-1; px];
        let mut lead_v = vec![0f32; px];
        let mut present = vec![false; n];
        for c in 0..n {
            for p in 0..px { ch[p] = fresh[p * n + c]; }
            if !ch.iter().any(|&v| v > 0.0) { continue; }
            present[c] = true;
            let bl = Self::box_blur_f(&ch, w, h, r);
            for p in 0..px {
                if wet_here(p) && bl[p] > lead_v[p] {
                    lead_v[p] = bl[p];
                    lead[p] = c as i16;
                }
            }
        }
        // 2. Per wash: the pigment diffuses inside it (a blur confined to the wash: blur(fresh × mask) /
        //    blur(mask)), mixed in by `s × coverage`, and at the wash's edge — where its coverage falls —
        //    the pigment the water carried out settles, deeper.
        //    The rim is NOT a contour drawn round every wash: a tide line forms where the water meets a
        //    DIFFERENT load — the lit window against the wall, a wash against dry paper — not between two
        //    washes of the same weight laid together (they blend). So the band is weighted by the LOAD
        //    CONTRAST across the edge, varies along it with the paper (a tide line is ragged, never a
        //    uniform stroke), and pools heavier along a wash's LOWER edge — the water runs down.
        let k = rim.clamp(0.0, 1.0);
        let load_b = Self::box_blur_f(&fresh_film, w, h, r);
        //    The backrun is ONE-SIDED: the wetter wash pushes its pigment into the drier one, and the line
        //    forms on the DRIER side of the boundary — where the load is below the local mean of the two
        //    sides — sharp there, nothing on the wet side. (Symmetric, both sides deepened and the line
        //    read as a drawn border.)
        let load_wide = Self::box_blur_f(&load_b, w, h, r);
        // A backrun needs a WASH on the drier side to run into: a thin dark mass (a branch, a mullion) on
        // bare paper has no drier wash on either flank, and the rule below — "the load is below the
        // boundary's mean" — was true on BOTH its sides, so every branch grew a dark rim all round
        // (cloisonné). The drier side must itself be wet: its own, unsmoothed, deposit above a floor set
        // by the sheet (a tenth of the mean wet load).
        let wet_floor = {
            let (mut sum, mut cnt) = (0f32, 0usize);
            for p in 0..px { if fresh_film[p] > 1e-5 { sum += fresh_film[p]; cnt += 1; } }
            if cnt > 0 { sum / cnt as f32 * 0.1 } else { 0.0 }
        };
        let own_b = Self::box_blur_f(&fresh_film, w, h, (r / 3).max(1));
        let contrast: Vec<f32> = (0..px)
            .map(|p| {
                let (x, y) = (p % w, p / w);
                if x == 0 || y == 0 || x + 1 >= w || y + 1 >= h { return 0.0; }
                // Wet enough to be a wash, and thinner than the boundary's mean (this is the wash side,
                // not the mass's own edge pixels).
                if own_b[p] < wet_floor || own_b[p] > load_wide[p] { return 0.0; }
                let gx = load_b[p + 1] - load_b[p - 1];
                let gy = load_b[p + w] - load_b[p - w];
                let g = (gx * gx + gy * gy).sqrt() * r as f32 * 0.5;
                let c = (g / (load_b[p] + 1e-4) * 1.5).clamp(0.0, 1.0);
                // The drier side: this pixel's load sits below the boundary's mean.
                let drier = ((load_wide[p] - load_b[p]) / (load_wide[p] + 1e-4) * 4.0).clamp(0.0, 1.0);
                c * drier
            })
            .collect();
        // Mobility: a staining dye travels in the water, an earth settles where it was laid — the pigments
        // SEPARATE at a wash's edge instead of moving as one tint (read from the masstone's chroma).
        let mobility: Vec<f32> = (0..n)
            .map(|c| {
                let m = self.palette.pigments.get(c).map(|pg| pg.masstone).unwrap_or([128, 128, 128]);
                let (mx, mn) = (m[0].max(m[1]).max(m[2]) as f32, m[0].min(m[1]).min(m[2]) as f32);
                let chroma = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
                0.7 + 0.6 * chroma
            })
            .collect();
        // The rim weight at a pixel: the load contrast there (the band is the contrast's own width, ~r
        // either side of the boundary), ragged with the paper, heavier where the water runs DOWN into it
        // (the load falls going down: the wash's lower edge).
        let edge: Vec<f32> = (0..px)
            .map(|p| {
                if k <= 0.0 || contrast[p] <= 0.0 { return 0.0; }
                let (x, y) = (p % w, p / w);
                let ragged = 0.4 + 1.2 * value_noise(x as f32 / (r as f32 * 1.5), y as f32 / (r as f32 * 1.5), 0x7ADE_11E5);
                let below = if y + 1 < h && y > 0 { (load_b[p - w] - load_b[p + w]).max(0.0) * r as f32 / (load_b[p] + 1e-4) } else { 0.0 };
                let run = 1.0 + 0.6 * below.min(1.0);
                k * contrast[p].powf(1.5) * ragged * run
            })
            .collect();
        // Where the water POOLED: the local load against the pass's mean wet load (0..2).
        let gq = grain.clamp(0.0, 1.0);
        let pool: Vec<f32> = if gq > 0.0 {
            let (mut sum, mut cnt) = (0f32, 0usize);
            for p in 0..px { if load_b[p] > 1e-5 { sum += load_b[p]; cnt += 1; } }
            let mean = if cnt > 0 { sum / cnt as f32 } else { 1.0 };
            (0..px).map(|p| (load_b[p] / mean.max(1e-6)).min(2.0)).collect()
        } else {
            Vec::new()
        };
        // Where the sheet has a HARD EDGE: the luma of what it shows now, smoothed at r/2 (a one-pixel lane
        // gap is diluted away by that — it is what the water must fill — while a boundary between two
        // masses keeps its full step), then the value RANGE over r/2: a crisp boundary spans ≥ 0.2, a wash's
        // fade far less. The band reaches r/2 either side, so the water stops short of the edge.
        let sel = selective.clamp(0.0, 1.0);
        let hold: Vec<f32> = if sel > 0.0 {
            let mut lum = vec![0f32; px];
            for p in 0..px {
                let mut out = self.ground_lin;
                for (i, rr) in self.ln_r.iter().enumerate() {
                    let d = self.conc[p * n + i].max(0.0);
                    if d > 0.0 { for c in 0..3 { out[c] *= (d * rr[c]).exp(); } }
                }
                lum[p] = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            }
            let rd = (r / 2).max(1);
            let lb = Self::box_blur_f(&lum, w, h, rd);
            let (mut mx, mut mn) = (lb.clone(), lb.clone());
            let mut t = lb.clone();
            for y in 0..h { for x in 0..w { let mut m = 0f32; for d in x.saturating_sub(rd)..=(x + rd).min(w - 1) { m = m.max(lb[y * w + d]); } t[y * w + x] = m; } }
            for y in 0..h { for x in 0..w { let mut m = 0f32; for d in y.saturating_sub(rd)..=(y + rd).min(h - 1) { m = m.max(t[d * w + x]); } mx[y * w + x] = m; } }
            for y in 0..h { for x in 0..w { let mut m = f32::MAX; for d in x.saturating_sub(rd)..=(x + rd).min(w - 1) { m = m.min(lb[y * w + d]); } t[y * w + x] = m; } }
            for y in 0..h { for x in 0..w { let mut m = f32::MAX; for d in y.saturating_sub(rd)..=(y + rd).min(h - 1) { m = m.min(t[d * w + x]); } mn[y * w + x] = m; } }
            let e: Vec<f32> = (0..px).map(|p| ((mx[p] - mn[p] - 0.06) / 0.14).clamp(0.0, 1.0)).collect();
            // Widened by r/2 more: the water stops a brush-radius short of the edge.
            for y in 0..h { for x in 0..w { let mut m = 0f32; for d in x.saturating_sub(rd)..=(x + rd).min(w - 1) { m = m.max(e[y * w + d]); } t[y * w + x] = m; } }
            for y in 0..h { for x in 0..w { let mut m = 0f32; for d in y.saturating_sub(rd)..=(y + rd).min(h - 1) { m = m.max(t[d * w + x]); } mx[y * w + x] = m; } }
            Self::box_blur_f(&mx, w, h, rd).into_iter().map(|v| sel * v).collect()
        } else {
            Vec::new()
        };
        let held = |p: usize| -> f32 { if hold.is_empty() { 1.0 } else { 1.0 - hold[p] } };
        let mut mask = vec![0f32; px];
        let mut masked = vec![0f32; px];
        let mut out_film = fresh_film.clone();
        let mut out = fresh.clone();
        for c in 0..n {
            if !present[c] { continue; }
            let mut any = false;
            for p in 0..px { mask[p] = if lead[p] == c as i16 { any = true; 1.0 } else { 0.0 }; }
            if !any { continue; }
            let cov = Self::box_blur_f(&mask, w, h, r);
            for p in 0..px { masked[p] = fresh_film[p] * mask[p]; }
            let num = Self::box_blur_f(&masked, w, h, r);
            for p in 0..px {
                if mask[p] <= 0.0 { continue; }
                let a = s * cov[p].min(1.0) * held(p);
                let d = num[p] / cov[p].max(1e-6);
                out_film[p] = fresh_film[p] * (1.0 - a) + d * a * (1.0 + edge[p]);
            }
            for cc in 0..n {
                if !present[cc] { continue; }
                let mut any = false;
                for p in 0..px { masked[p] = fresh[p * n + cc] * mask[p]; any |= masked[p] > 0.0; }
                if !any { continue; }
                let num = Self::box_blur_f(&masked, w, h, r);
                let mob = mobility[cc];
                // The granulating share: what does not travel settles.
                let settle = gq * ((1.3 - mob) / 0.6).clamp(0.0, 1.0);
                for p in 0..px {
                    if mask[p] <= 0.0 { continue; }
                    let a = (s * cov[p].min(1.0) * mob * held(p)).min(1.0);
                    let d = num[p] / cov[p].max(1e-6);
                    let i = p * n + cc;
                    let mut v = fresh[i] * (1.0 - a) + d * a * (1.0 + edge[p]);
                    if settle > 0.0 {
                        let g = paper_grain(p % w, p / w, 0x6_1A1_0000) - 0.5;
                        v *= (1.0 + settle * pool[p] * g * 2.4).max(0.0);
                    }
                    out[i] = v;
                }
            }
        }
        for p in 0..px {
            self.film[p] = film_mark(p) + out_film[p];
            for c in 0..n {
                let i = p * n + c;
                self.conc[i] = self.conc_mark.get(i).copied().unwrap_or(0.0) + out[i];
            }
        }
    }

    /// WET COLLISION (`collide`, an oil's stage): when a loaded wet stroke lands beside or over another, the
    /// paint MOVES — the colours drag into each other along the stroke's direction (marbling), the new
    /// stroke's bead plows the old paint's height sideways, and nothing of it is a blur. Run after a broad
    /// pass, on the wet paint only (a dried pass does not move), with the direction read from the canvas
    /// itself — the striation of the height field (its structure tensor over `radius`), so a replay sees the
    /// same drags. `strength` (0..1) is how far the paint is carried (up to `radius` px); `face` masks the
    /// pixels that must not smear (the faces: their features soften under any drag). Opaque media only;
    /// a no-op at strength 0.
    pub fn collide(&mut self, strength: f32, radius: f32, face: Option<&[f32]>, hold_at: f32) {
        let s = strength.clamp(0.0, 1.0);
        if s <= 0.0 || radius < 1.0 || self.transmittance {
            return;
        }
        let (w, h, n) = (self.w as usize, self.h as usize, self.n);
        let px = w * h;
        let r = radius.round().max(1.0) as usize;
        // 1. The direction the paint runs: the height field's striation. Structure tensor of the height's
        //    gradient, smoothed over r; the stroke runs ALONG the striation (perpendicular to the gradient).
        let hb = Self::box_blur_f(&self.height, w, h, 1);
        let (mut jxx, mut jyy, mut jxy) = (vec![0f32; px], vec![0f32; px], vec![0f32; px]);
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p = y * w + x;
                let gx = (hb[p + 1] - hb[p - 1]) * 0.5;
                let gy = (hb[p + w] - hb[p - w]) * 0.5;
                jxx[p] = gx * gx;
                jyy[p] = gy * gy;
                jxy[p] = gx * gy;
            }
        }
        let jxx = Self::box_blur_f(&jxx, w, h, r);
        let jyy = Self::box_blur_f(&jyy, w, h, r);
        let jxy = Self::box_blur_f(&jxy, w, h, r);
        // 2. Where the collision happens: wet paint beside wet paint of a DIFFERENT colour or height — the
        //    wetness smoothed over r is the "both wet" measure, the local height range the "a ridge meets
        //    paint" measure.
        let wet_b = Self::box_blur_f(&self.wetness, w, h, r);
        // 3. Advect: each pixel pulls colour and height from a point `d` back along the striation, where
        //    d = s × r × wet × contrast; the colour mixes (marbling), the height moves with it (the plow).
        let old_conc = self.conc.clone();
        let old_h = self.height.clone();
        let old_w = self.wetness.clone();
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let wet = old_w[p].clamp(0.0, 1.0) * wet_b[p].clamp(0.0, 1.0);
                if wet < 0.02 {
                    continue;
                }
                // The hold mask carries a LEVEL: ≥ `hold_at` holds the paint still (faces always; the
                // subject on the fine passes, where a drag would soften what the fine brushes resolved).
                let fm = face.map(|m| m.get(p).copied().unwrap_or(0.0)).unwrap_or(0.0);
                if fm >= hold_at {
                    continue;
                }
                // The striation direction from the tensor: the eigenvector of the SMALLER eigenvalue.
                let (a, b, c) = (jxx[p], jyy[p], jxy[p]);
                let coh = (((a - b) * (a - b) + 4.0 * c * c).sqrt()) / (a + b + 1e-9);
                if coh < 0.15 {
                    continue;
                }
                let theta = 0.5 * (2.0 * c).atan2(a - b); // gradient direction
                let (ux, uy) = (-theta.sin(), theta.cos()); // along the striation
                // The drag follows the slope's sign so paint runs off the ridge, not into it: pull from the
                // higher side.
                let gx = (hb[(p + 1).min(px - 1)] - hb[p.saturating_sub(1)]) * 0.5;
                let gy = (hb[(p + w).min(px - 1)] - hb[p.saturating_sub(w)]) * 0.5;
                let sign = if gx * ux + gy * uy >= 0.0 { 1.0 } else { -1.0 };
                let d = s * r as f32 * wet * coh.min(1.0);
                let sx = (x as f32 + ux * d * sign).clamp(0.0, w as f32 - 1.0);
                let sy = (y as f32 + uy * d * sign).clamp(0.0, h as f32 - 1.0);
                let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (tx, ty) = (sx - x0 as f32, sy - y0 as f32);
                let wts = [((y0 * w + x0), (1.0 - tx) * (1.0 - ty)), ((y0 * w + x1), tx * (1.0 - ty)), ((y1 * w + x0), (1.0 - tx) * ty), ((y1 * w + x1), tx * ty)];
                // How much of the sampled paint comes in: the wetness (dry paint under a wet stroke does not
                // move) — and the mix is in CONCENTRATION, so the colours marble by Kubelka–Munk, not alpha.
                let mix = (0.85 * wet).min(0.85);
                let mut hsum = 0f32;
                for c in 0..n {
                    let mut v = 0f32;
                    for (q, wq) in &wts {
                        v += old_conc[q * n + c] * wq;
                    }
                    self.conc[p * n + c] = old_conc[p * n + c] * (1.0 - mix) + v * mix;
                }
                for (q, wq) in &wts {
                    hsum += old_h[*q] * wq;
                }
                self.height[p] = old_h[p] * (1.0 - mix) + hsum * mix;
            }
        }
    }

    /// `box_blur_f` for callers outside the canvas (the painter's fields).
    pub fn box_blur_pub_impl(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
        Self::box_blur_f(src, w, h, r)
    }

    /// A box blur of a scalar field, separable, radius `r` (window clamped at the edges, mean over the
    /// cells actually inside).
    fn box_blur_f(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
        let pass = |line: &[f32], out: &mut [f32]| {
            let len = line.len();
            let mut prefix = vec![0f32; len + 1];
            for i in 0..len { prefix[i + 1] = prefix[i] + line[i]; }
            for i in 0..len {
                let a = i.saturating_sub(r);
                let b = (i + r).min(len - 1);
                out[i] = (prefix[b + 1] - prefix[a]) / (b - a + 1) as f32;
            }
        };
        let mut t = vec![0f32; w * h];
        for y in 0..h { pass(&src[y * w..(y + 1) * w], &mut t[y * w..(y + 1) * w]); }
        let mut out = vec![0f32; w * h];
        let mut col = vec![0f32; h];
        let mut colo = vec![0f32; h];
        for x in 0..w {
            for y in 0..h { col[y] = t[y * w + x]; }
            pass(&col, &mut colo);
            for y in 0..h { out[y * w + x] = colo[y]; }
        }
        out
    }

    /// One step of directional pigment transport between wet neighbours (see `bleed_with`). Each 4-neighbour
    /// pair is visited once; the amount moved is a fraction of the SOURCE cell's pigment set by the pair's
    /// wetness, `|d|`, and how different the two loads are — so an even wash does not drift, a wash against a
    /// dark passage does. Bounded so a cell can never go negative (≤ ¼ of its load per pair, 4 pairs).
    fn pigment_drift(&mut self, s: f32, d: f32, rate: f32) {
        let (w, h, n) = (self.w as usize, self.h as usize, self.n);
        let toward_dark = d > 0.0;
        let mag = d.abs().min(1.0);
        let cell_load: Vec<f32> = (0..w * h).map(|p| self.conc[p * n..p * n + n].iter().sum::<f32>()).collect();
        // The drift follows the WASH's level, not the pixel's: the load field smoothed over a few pixels. Judged
        // pixel by pixel, the transport fed on pixel-scale noise and piled pigment into a checkerboard; judged on
        // the wash level it moves pigment across the real boundaries between a loaded passage and a thin one.
        let dark = box_blur3(&box_blur3(&cell_load, w, h), w, h);
        // The load difference that counts as a real boundary: a fraction of the wet cells' MEAN load — a fact of
        // this painting, so a thin wash and a loaded one drift alike, and the tiny differences inside one even
        // wash are left alone (no clumping out of noise).
        let (mut sum, mut cnt) = (0.0f32, 0usize);
        for p in 0..w * h {
            if self.wetness[p] > 1e-3 {
                sum += dark[p];
                cnt += 1;
            }
        }
        let scale = (sum / cnt.max(1) as f32) * DRIFT_TAU + 1e-4;
        let src = self.conc.clone();
        let src_film = self.film.clone();
        // One pair's flow: who gives, who takes, and what share of the giver's load (before the cap).
        let flow = |p: usize, q: usize, wet: &[f32]| -> Option<(usize, usize, f32)> {
            let a = s * mag * wet[p].min(wet[q]).clamp(0.0, 1.0);
            if a <= 1e-4 {
                return None;
            }
            let dd = dark[q] - dark[p];
            if dd.abs() <= 1e-6 {
                return None;
            }
            // Pigment leaves the lighter cell for the darker one (toward the dark), or the darker for the
            // lighter (toward the light).
            let (from, to) = if (dd > 0.0) == toward_dark { (p, q) } else { (q, p) };
            Some((from, to, a * rate * (dd.abs() / scale).clamp(0.0, 1.0)))
        };
        // Pass 1: each cell's intended total outflow, so pass 2 can hold it to `DRIFT_MAX_OUT`.
        let mut out = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    if let Some((from, _, f)) = flow(p, ny * w + nx, &self.wetness) {
                        out[from] += f;
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let Some((from, to, f)) = flow(p, ny * w + nx, &self.wetness) else { continue };
                    let f = if out[from] > DRIFT_MAX_OUT { f * (DRIFT_MAX_OUT / out[from]) } else { f };
                    for c in 0..n {
                        let m = src[from * n + c] * f;
                        self.conc[from * n + c] = (self.conc[from * n + c] - m).max(0.0);
                        self.conc[to * n + c] += m;
                    }
                    let mf = src_film[from] * f;
                    self.film[from] = (self.film[from] - mf).max(0.0);
                    self.film[to] += mf;
                }
            }
        }
    }

    /// A WASH: fill the area inside `rings` (polygons separated by a `[NaN, NaN]` point; even-odd, so a hole
    /// ring inside an outer ring stays unfilled) with `load` at every pixel, wet at `wet`. The watercolour's
    /// area mark. `feather` (px) is the WET EDGE: the wash BLOOMS OUTWARD past its boundary over that distance
    /// with a falling deposit (0 = a hard, dried edge), so it meets its neighbour and the paper with a soft
    /// bleed, not a cut — and never with a pale rim inside (an inward ramp stacked into halos).
    /// Pixel centres are tested (x + 0.5, y + 0.5); the deposit is the same covering law a stroke uses, at
    /// zero height (a wash stands off nothing).
    pub fn fill_rings(&mut self, rings: &[[f32; 2]], load: &[f32], wet: f32, feather: f32) {
        let (w, h) = (self.w as i64, self.h as i64);
        let mut edges: Vec<([f32; 2], [f32; 2])> = Vec::new();
        let mut ring: Vec<[f32; 2]> = Vec::new();
        let flush = |ring: &mut Vec<[f32; 2]>, edges: &mut Vec<([f32; 2], [f32; 2])>| {
            if ring.len() >= 3 {
                for i in 0..ring.len() {
                    edges.push((ring[i], ring[(i + 1) % ring.len()]));
                }
            }
            ring.clear();
        };
        for pt in rings {
            if pt[0].is_nan() || pt[1].is_nan() {
                flush(&mut ring, &mut edges);
            } else {
                ring.push(*pt);
            }
        }
        flush(&mut ring, &mut edges);
        if edges.is_empty() {
            return;
        }
        let f = feather.max(0.0).round().min(250.0) as usize;
        let fi = f as i64;
        let x0 = (edges.iter().map(|e| e.0[0].min(e.1[0])).fold(f32::INFINITY, f32::min).floor() as i64 - fi).max(0);
        let x1 = (edges.iter().map(|e| e.0[0].max(e.1[0])).fold(f32::NEG_INFINITY, f32::max).ceil() as i64 + fi).min(w);
        let y0 = (edges.iter().map(|e| e.0[1].min(e.1[1])).fold(f32::INFINITY, f32::min).floor() as i64 - fi).max(0);
        let y1 = (edges.iter().map(|e| e.0[1].max(e.1[1])).fold(f32::NEG_INFINITY, f32::max).ceil() as i64 + fi).min(h);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let (bw, bh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        // 1. The inside mask over the bounding box (scanline, even-odd, pixel centres).
        let mut mask = vec![false; bw * bh];
        let mut xs: Vec<f32> = Vec::new();
        for y in y0..y1 {
            let yc = y as f32 + 0.5;
            xs.clear();
            for (a, b) in &edges {
                let (ya, yb) = (a[1], b[1]);
                if (ya <= yc && yb > yc) || (yb <= yc && ya > yc) {
                    let t = (yc - ya) / (yb - ya);
                    xs.push(a[0] + t * (b[0] - a[0]));
                }
            }
            if xs.len() < 2 {
                continue;
            }
            xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let xa = (pair[0] - 0.5).ceil().max(x0 as f32) as i64;
                let xb = (pair[1] - 0.5).floor().min(x1 as f32 - 1.0) as i64;
                for x in xa..=xb {
                    mask[(y - y0) as usize * bw + (x - x0) as usize] = true;
                }
            }
        }
        // 2. The wet edge: distance OUTSIDE the boundary by successive dilation, up to `feather` px.
        let mut dist = vec![0u8; bw * bh];
        if f > 0 {
            let mut cur = mask.clone();
            for d in 1..=f {
                let mut next = cur.clone();
                for y in 0..bh {
                    for x in 0..bw {
                        let i = y * bw + x;
                        if cur[i] {
                            continue;
                        }
                        let touch = (x > 0 && cur[i - 1]) || (x + 1 < bw && cur[i + 1]) || (y > 0 && cur[i - bw]) || (y + 1 < bh && cur[i + bw]);
                        if touch {
                            next[i] = true;
                            dist[i] = d as u8;
                        }
                    }
                }
                cur = next;
            }
        }
        // 3. Deposit: full inside; outside, a bloom falling with the distance (a bleed is thinner than the wash).
        let mut scaled = vec![0f32; load.len()];
        for y in 0..bh {
            for x in 0..bw {
                let i = y * bw + x;
                let k = if mask[i] {
                    1.0
                } else if dist[i] > 0 {
                    let t = 1.0 - dist[i] as f32 / (f as f32 + 1.0);
                    0.6 * t * t
                } else {
                    continue;
                };
                for (c, v) in load.iter().enumerate() {
                    scaled[c] = v * k;
                }
                let (px, py) = ((x as i64 + x0) as u32, (y as i64 + y0) as u32);
                self.deposit(px, py, &scaled, 0.0);
                if mask[i] {
                    let j = self.idx(px, py) / self.n;
                    self.wetness[j] = self.wetness[j].max(wet.clamp(0.0, 1.0));
                }
            }
        }
    }

    /// Clear the deposited paint (and height, wetness) wherever `mask` is true, re-exposing the ground. Used to
    /// PRIME a composition layer's footprint before painting it, so a nearer element paints fresh and opaquely
    /// OCCLUDES the farther layers beneath — the colour model mixes by concentration RATIO, so without this a
    /// thin new layer would be dominated by the thick paint already there and fail to cover.
    pub fn clear_mask(&mut self, mask: &[bool]) {
        for p in 0..(self.w as usize * self.h as usize).min(mask.len()) {
            if mask[p] {
                for c in 0..self.n {
                    self.conc[p * self.n + c] = 0.0;
                }
                self.film[p] = 0.0;
                self.height[p] = 0.0;
                self.wetness[p] = 0.0;
            }
        }
    }

    /// The sRGB colour at a pixel: the paint's own Kubelka-Munk colour (from deposited pigment) composited OVER
    /// the ground with a film-build opacity that grows with the amount of paint laid. Thin paint → the ground
    /// shows (glaze/light); built paint → opaque, so darks stay dark, lights bright, and colour saturated.
    pub fn color_at(&self, x: u32, y: u32) -> Srgb {
        let conc = self.conc_at(x, y);
        let total = self.film[y as usize * self.w as usize + x as usize].max(0.0);
        if total <= 1e-4 {
            return color::linear_to_srgb(self.ground_lin);
        }
        if self.transmittance {
            let mut out = self.ground_lin;
            for (i, r) in self.ln_r.iter().enumerate() {
                let d = conc.get(i).copied().unwrap_or(0.0).max(0.0);
                if d > 0.0 {
                    for c in 0..3 {
                        out[c] *= (d * r[c]).exp();
                    }
                }
            }
            return color::linear_to_srgb(out);
        }
        let paint = pigment::mix_linear(self.palette.pigments, conc);
        let alpha = 1.0 - (-self.opacity_k * self.opacity * total).exp();
        let mut out = [0f32; 3];
        for c in 0..3 {
            out[c] = alpha * paint[c] + (1.0 - alpha) * self.ground_lin[c];
        }
        color::linear_to_srgb(out)
    }

    /// A copy of the current canvas state — for timelapse frames.
    pub fn snapshot(&self) -> Canvas {
        self.clone()
    }

    /// The impasto height as a 16-bit grayscale image, normalised to the canvas's peak height (0 if flat).
    pub fn height_image(&self) -> image::ImageBuffer<image::Luma<u16>, Vec<u16>> {
        let peak = self.height.iter().copied().fold(0.0_f32, f32::max);
        let scale = if peak > 0.0 { 65535.0 / peak } else { 0.0 };
        image::ImageBuffer::from_fn(self.w, self.h, |x, y| {
            let v = self.height[y as usize * self.w as usize + x as usize] * scale;
            image::Luma([v.round().clamp(0.0, 65535.0) as u16])
        })
    }

    /// Render the whole canvas to an sRGB image (no impasto relight — that is a later output stage).
    pub fn to_image(&self) -> RgbImage {
        let mut img = RgbImage::new(self.w, self.h);
        for y in 0..self.h {
            for x in 0..self.w {
                let c = self.color_at(x, y);
                img.put_pixel(x, y, image::Rgb(c));
            }
        }
        img
    }

    /// Backwards-compatible impasto-only relight (used where no other material stage applies).
    pub fn to_image_relit(&self, impasto: f32) -> RgbImage {
        self.to_image_finished(&Finish { impasto, ..Default::default() })
    }

    /// Render WITH the MEDIUM'S MATERIAL FINISH (RFC §8.5 output stages) — the way the *paint itself* behaves,
    /// beyond how it was applied:
    /// - **chroma**: saturation range (oil vivid, gouache/watercolour muted);
    /// - **dry_shift**: the value change on drying (+ watercolour dries lighter; − gouache dries to a matte,
    ///   compressed mid);
    /// - **granulate**: pigment settling into the paper's tooth — the mottled watercolour / graphite grain;
    /// - **impasto** + **sheen**: thick paint relit from stroke height (a raking light + a glossy specular).
    ///
    /// All are deterministic (grain seeded) and recorded in the score, so `replay` reproduces the material
    /// exactly. `Finish::default()` (all neutral) reproduces `to_image`.
    pub fn to_image_finished(&self, f: &Finish) -> RgbImage {
        let mut img = self.to_image();
        let (w, h) = (self.w as usize, self.h as usize);
        let material = (f.chroma - 1.0).abs() > 1e-3 || f.dry_shift.abs() > 1e-3 || f.granulate > 1e-3;
        if material {
            for y in 0..h {
                for x in 0..w {
                    // How much PAINT is here (vs bare ground) — material stages act on paint, not the paper.
                    let total = self.film[y * w + x].max(0.0);
                    let paint = (1.0 - (-1.6 * total).exp()).clamp(0.0, 1.0);
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    let mut c = [p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, p.0[2] as f32 / 255.0];
                    // CHROMA — push away from (or toward) the pixel's own luma.
                    if (f.chroma - 1.0).abs() > 1e-3 {
                        let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                        for v in c.iter_mut() {
                            *v = (l + (*v - l) * f.chroma).clamp(0.0, 1.0);
                        }
                    }
                    // DRY SHIFT — +: dries lighter (watercolour); −: dries to a matte, compressed mid (gouache).
                    if f.dry_shift.abs() > 1e-3 {
                        let s = f.dry_shift * paint;
                        if s >= 0.0 {
                            for v in c.iter_mut() {
                                *v = (*v + s * (1.0 - *v)).clamp(0.0, 1.0);
                            }
                        } else {
                            let k = -s;
                            for v in c.iter_mut() {
                                *v = (*v * (1.0 - k) + 0.5 * k - 0.04 * k).clamp(0.0, 1.0);
                            }
                        }
                    }
                    // GRANULATION — the paper's tooth holds pigment unevenly: a MOTTLE (some grains lighter, some
                    // darker) where paint sits, with NO net darkening (a centred grain), so it reads as texture,
                    // not grey noise.
                    if f.granulate > 1e-3 && paint > 0.05 {
                        let d = f.granulate * paint * (paper_grain(x, y, f.seed) - 0.5) * 0.7;
                        for v in c.iter_mut() {
                            *v = (*v * (1.0 - d)).clamp(0.0, 1.0);
                        }
                    }
                    for i in 0..3 {
                        p.0[i] = (c[i] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        // EDGE POOLING — a watercolour wash dries into a darker pigment RING at its boundary (the "cauliflower" /
        // edge-bloom). Darken where the PAINT AMOUNT changes fastest — the rim of a wash — scaled by `edge_pool`.
        if f.edge_pool > 1e-3 {
            let mut amt = vec![0f32; w * h];
            for (idx, a) in amt.iter_mut().enumerate() {
                let total = self.film[idx].max(0.0);
                *a = (1.0 - (-1.6 * total).exp()).clamp(0.0, 1.0);
            }
            for y in 0..h {
                for x in 0..w {
                    let xl = x.saturating_sub(1);
                    let xr = (x + 1).min(w - 1);
                    let yt = y.saturating_sub(1);
                    let yb = (y + 1).min(h - 1);
                    let gx = amt[y * w + xr] - amt[y * w + xl];
                    let gy = amt[yb * w + x] - amt[yt * w + x];
                    let g = (gx * gx + gy * gy).sqrt();
                    let here = amt[y * w + x];
                    let d = (f.edge_pool * g * 1.6 * here).clamp(0.0, 0.5);
                    if d < 1e-3 {
                        continue;
                    }
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    for cc in 0..3 {
                        p.0[cc] = (p.0[cc] as f32 * (1.0 - d)).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        // IMPASTO + SHEEN relight from the paint HEIGHT — and the CANVAS WEAVE beneath it.
        if f.impasto > 1e-4 || f.sheen > 1e-4 || f.weave > 1e-4 {
            // Normalise the relief against a ROBUST height (the 98th percentile), not the single peak: one
            // heavy crossing of strokes set the scale for the whole sheet and flattened every other ridge.
            let peak = if f.relief_robust {
                let mut v: Vec<f32> = self.height.iter().copied().filter(|h| *h > 0.0).collect();
                if v.is_empty() { 1e-4 } else { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v[(v.len() - 1) * 98 / 100].max(1e-4) }
            } else {
                self.height.iter().copied().fold(0.0_f32, f32::max).max(1e-4)
            };
            let (lx, ly) = (0.55_f32, 0.83_f32);
            let gain = 1.6 * f.impasto.clamp(0.0, 1.0);
            let spec = 0.9 * f.sheen.clamp(0.0, 1.0);
            // The weave: a plain LINEN, not a grid — the warp and weft domain-warped by a slow noise (the
            // cloth stretched unevenly on its bars), interlocked over-and-under (a checker of bumps where
            // one thread crosses the other), each thread's thickness wandering along its length (slubs),
            // and a fibrous micro-roughness over all. Period ~1/400 of the short side (a medium canvas).
            // Seen where the paint is THIN — the cover falls off with the LOCAL paint height (a ridged
            // stroke is a comb of peaks and furrows and the furrows are paint too), so a built-up passage
            // hides the threads entirely and a glaze or the bare ground shows them.
            let weave = f.weave.clamp(0.0, 1.0);
            let period = (w.min(h) as f32 / 400.0).max(3.0);
            let kw = std::f32::consts::TAU / period;
            let (h50, local_h) = if weave > 0.0 {
                // The local height: the max over a 3-px window, lightly smoothed.
                let r3 = 3usize;
                let mut t = self.height.clone();
                let mut mx = self.height.clone();
                for y in 0..h { for x in 0..w { let mut m = 0f32; for d in x.saturating_sub(r3)..=(x + r3).min(w - 1) { m = m.max(self.height[y * w + d]); } t[y * w + x] = m; } }
                for y in 0..h { for x in 0..w { let mut m = 0f32; for d in y.saturating_sub(r3)..=(y + r3).min(h - 1) { m = m.max(t[d * w + x]); } mx[y * w + x] = m; } }
                let lh = Self::box_blur_f(&mx, w, h, 2);
                let mut v: Vec<f32> = lh.iter().copied().filter(|h| *h > 0.0).collect();
                let med = if v.is_empty() { 1e-4 } else { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v[v.len() / 2].max(1e-4) };
                (med, lh)
            } else {
                (1.0, Vec::new())
            };
            let linen = |x: f32, y: f32| -> f32 {
                // Domain warp: a slow drift of the cloth, ± a period over ~40 periods.
                // (±1.5 periods over ~12: at a 10 px period on a 4096 sheet the earlier ±1 over 40 was a
                // straight grid to the eye — a dot screen on the sky.)
                let wx = (value_noise(x / (period * 12.0), y / (period * 12.0), f.seed ^ 0x11EA) - 0.5) * 3.0 * period
                    + (value_noise(x / (period * 3.5), y / (period * 3.5), f.seed ^ 0x77A1) - 0.5) * 0.8 * period;
                let wy = (value_noise(x / (period * 12.0) + 7.3, y / (period * 12.0) + 3.1, f.seed ^ 0x2BEE) - 0.5) * 3.0 * period
                    + (value_noise(x / (period * 3.5) + 2.2, y / (period * 3.5) + 9.1, f.seed ^ 0x88B2) - 0.5) * 0.8 * period;
                let (u, v) = (x + wx, y + wy);
                let sx = (kw * u).sin();
                let sy = (kw * v).sin();
                // Slubs: a warp thread (running in y) varies along y, a weft thread along x.
                let tx = 0.55 + 0.9 * value_noise(u / (period * 0.9) + 11.0, v / (period * 6.0), f.seed ^ 0x3C0D);
                let ty = 0.55 + 0.9 * value_noise(u / (period * 6.0), v / (period * 0.9) + 5.0, f.seed ^ 0x4D1E);
                // Interlock (the checker of crossings) + the threads' own ridges + fibrous roughness.
                let interlock = sx * sy;
                let ridges = 0.35 * (sx.abs() * tx + sy.abs() * ty);
                let fibre = 0.12 * (value_noise(x / 1.7, y / 1.7, f.seed ^ 0x5F1B) - 0.5);
                0.6 * interlock * (tx + ty) * 0.5 + ridges + fibre
            };
            for y in 0..h {
                for x in 0..w {
                    let xl = x.saturating_sub(1);
                    let xr = (x + 1).min(w - 1);
                    let yt = y.saturating_sub(1);
                    let yb = (y + 1).min(h - 1);
                    let hx = (self.height[y * w + xr] - self.height[y * w + xl]) / peak;
                    let hy = (self.height[yb * w + x] - self.height[yt * w + x]) / peak;
                    let facing = hx * lx + hy * ly;
                    // Gate by the paint AMOUNT here: thin / flat passages (a wash, a bare background) barely stand
                    // off the surface, so they get little relief — only built-up strokes catch light. Keeps the
                    // background smooth instead of a canvas-weave grain.
                    let total = self.film[y * w + x].max(0.0);
                    let amt = (total / 2.0).clamp(0.0, 1.0);
                    // Diffuse impasto shading + a sharper glossy highlight on the near ridges (sheen).
                    let mut shade = gain * facing * amt;
                    if spec > 0.0 && facing > 0.0 {
                        shade += spec * facing * facing * amt;
                    }
                    // CAST SHADOW (with the striated relief, `ridges`): a thick stroke beside thin paint is
                    // a step, and the raking light throws the step's shadow onto the thin side — the
                    // micro-shadow that makes thick paint SIT ON the smooth passage instead of fading into
                    // it. Walk a few pixels toward the light; where the paint there stands higher than this
                    // pixel by more than the light's climb, this pixel is in its shadow.
                    if f.relief_robust && gain > 0.0 {
                        let mut occl = 0f32;
                        for k in 1..=5i32 {
                            let sx = (x as i32 + (lx * k as f32).round() as i32).clamp(0, w as i32 - 1) as usize;
                            let sy = (y as i32 + (ly * k as f32).round() as i32).clamp(0, h as i32 - 1) as usize;
                            let rise = (self.height[sy * w + sx] - self.height[y * w + x]) / peak - 0.06 * k as f32;
                            occl = occl.max(rise);
                        }
                        if occl > 0.0 {
                            shade -= 0.7 * gain * occl.min(1.0) * amt;
                        }
                    }
                    if weave > 0.0 {
                        let (xf, yf) = (x as f32, y as f32);
                        let dwx = (linen(xf + 1.0, yf) - linen(xf - 1.0, yf)) * 0.5;
                        let dwy = (linen(xf, yf + 1.0) - linen(xf, yf - 1.0)) * 0.5;
                        let facing_w = (dwx * lx + dwy * ly) * period / 2.2;
                        let cover = (-4.0 * local_h[y * w + x] / h50).exp();
                        shade += 0.9 * weave * facing_w * cover;
                    }
                    let shade = shade.clamp(-0.55, 0.85);
                    if shade.abs() < 1e-4 {
                        continue;
                    }
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    // The relief is MULTIPLICATIVE, so on a LIGHT passage (a bright sky, a pale wall) a given normal
                    // swings a large ABSOLUTE amount and the canvas weave reads as a hard textured BAND against the
                    // darker, smoother masses — a seam. Damp the relief toward the light end so the texture's
                    // magnitude is even across the value range (impasto still reads on the mid/dark brushwork).
                    let l = (p.0[0] as f32 * 0.299 + p.0[1] as f32 * 0.587 + p.0[2] as f32 * 0.114) / 255.0;
                    // (With the striated relief on — `ridges` — a near-white painting went flat under this damp:
                    // a snow scene's mane stood three times the sky's height and none of it showed. Then the
                    // relief is damped by half as much, and on a light passage the shadow side keeps its depth
                    // while the lit side is held — paint on white can only gain shadow, not glare.)
                    let ldamp = if f.relief_robust {
                        let d = (1.0 - 0.4 * (l - 0.45).max(0.0) / 0.55).clamp(0.5, 1.0);
                        if shade > 0.0 { d * 0.6 } else { d }
                    } else {
                        (1.0 - 0.8 * (l - 0.45).max(0.0) / 0.55).clamp(0.22, 1.0)
                    };
                    let shade = shade * ldamp;
                    for cc in 0..3 {
                        p.0[cc] = (p.0[cc] as f32 * (1.0 + shade)).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        // FINISH GRADE — a painting-safe tonal grade (the SAFE subset of a naturalize pass): CONTRAST (S-curve
        // around mid-grey), WARMTH (white balance), and CLARITY (gentle LOCAL contrast — a large-radius unsharp,
        // midtone punch, NOT edge sharpening that would re-introduce photographic detail). Whole-image, recorded
        // in the score so replay reproduces it.
        let grade = (f.contrast - 1.0).abs() > 1e-3 || f.warmth.abs() > 1e-3 || f.clarity > 1e-3;
        if grade {
            let contrast = f.contrast.clamp(0.3, 3.0);
            let warmth = f.warmth.clamp(-1.0, 1.0);
            let clarity = f.clarity.clamp(0.0, 1.0);
            let blurred: Option<RgbImage> = (clarity > 1e-3).then(|| image::imageops::blur(&img, (w.min(h) as f32 * 0.02).clamp(2.0, 20.0)));
            for y in 0..h {
                for x in 0..w {
                    let bp = blurred.as_ref().map(|b| b.get_pixel(x as u32, y as u32).0);
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    let mut c = [p.0[0] as f32 / 255.0, p.0[1] as f32 / 255.0, p.0[2] as f32 / 255.0];
                    if (contrast - 1.0).abs() > 1e-3 {
                        for v in c.iter_mut() {
                            *v = (0.5 + (*v - 0.5) * contrast).clamp(0.0, 1.0);
                        }
                    }
                    if warmth.abs() > 1e-3 {
                        c[0] = (c[0] + warmth * 0.12).clamp(0.0, 1.0);
                        c[2] = (c[2] - warmth * 0.12).clamp(0.0, 1.0);
                    }
                    if let Some(b) = bp {
                        for i in 0..3 {
                            let lo = b[i] as f32 / 255.0;
                            c[i] = (c[i] + clarity * 0.6 * (c[i] - lo)).clamp(0.0, 1.0);
                        }
                    }
                    for i in 0..3 {
                        p.0[i] = (c[i] * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        // PAPER EDGE — fade to bare paper at the borders with an IRREGULAR deckled edge (the torn-paper vignette a
        // watercolour sits in). The fade band's inner boundary wobbles per-pixel via the grain hash, so the edge
        // reads as torn paper, not a clean rectangle. Paper tone = a warm near-white.
        if f.paper_edge > 1e-3 {
            let paper = [249.0_f32, 246.0, 240.0];
            let band = (w.min(h) as f32 * (0.03 + 0.12 * f.paper_edge.clamp(0.0, 1.0))).max(2.0);
            for y in 0..h {
                for x in 0..w {
                    let d = x.min(w - 1 - x).min(y).min(h - 1 - y) as f32;
                    // Irregular inner boundary: the band width wobbles with a low-frequency hash of the position.
                    let wob = 0.55 + 0.9 * paper_grain(x / 3, y / 3, f.seed ^ 0x9E37);
                    let edge = band * wob;
                    if d >= edge {
                        continue;
                    }
                    let t = (1.0 - d / edge).clamp(0.0, 1.0);
                    let fade = t * t; // ease in toward the very border
                    let p = img.get_pixel_mut(x as u32, y as u32);
                    for cc in 0..3 {
                        p.0[cc] = (p.0[cc] as f32 * (1.0 - fade) + paper[cc] * fade).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        if f.old_paper {
            old_paper(&mut img, f.seed, f.plate_mark);
        }
        img
    }
}

/// The MEDIUM'S material finish, applied at output (§8.5). Neutral by default (= `to_image`).
#[derive(Clone, Copy, Debug)]
pub struct Finish {
    /// Impasto relief strength (0 flat → 1 thick, light-catching).
    pub impasto: f32,
    /// Scale the relief against a robust height (the 98th percentile) rather than the single peak — one
    /// heavy crossing no longer flattens every other ridge. Off = the old scale, byte for byte.
    pub relief_robust: bool,
    /// Saturation multiplier (1 neutral; >1 vivid oil; <1 muted gouache/watercolour).
    pub chroma: f32,
    /// Drying value shift (+ lighter watercolour; − matte-compressed gouache).
    pub dry_shift: f32,
    /// Granulation strength (paper-tooth pigment settling — watercolour, graphite).
    pub granulate: f32,
    /// Gloss sheen (specular highlight on ridges — oil).
    pub sheen: f32,
    /// Edge pooling (0..1): darken pigment at wash boundaries — the watercolour edge-bloom / "cauliflower" ring.
    pub edge_pool: f32,
    /// Paper edge (0..1): fade the painting to bare paper at the borders with an irregular DECKLED edge — the
    /// torn-paper vignette a watercolour sits in. 0 = full-bleed rectangle.
    pub paper_edge: f32,
    /// OLD PAPER: print the picture on an aged sheet (see [`old_paper`]).
    pub old_paper: bool,
    /// With `old_paper`: the PLATE MARK an intaglio press leaves round an engraving — an inked line a little
    /// inside the sheet's edge and a breath of plate tone within it.
    pub plate_mark: bool,
    /// CANVAS WEAVE (0 = none): the linen's threads under the paint, seen in the relight where the paint is
    /// thin or bare — a built-up passage covers them entirely.
    pub weave: f32,
    /// Finish grade — a painting-safe tonal grade (naturalize's safe subset), recorded for replay.
    /// CONTRAST (0.5..2, 1 = neutral): S-curve around mid-grey.
    pub contrast: f32,
    /// WARMTH (−1..1, 0 = neutral): white-balance shift — + warms (toward amber), − cools (toward blue).
    pub warmth: f32,
    /// CLARITY (0..1, 0 = off): gentle LOCAL contrast (large-radius unsharp) — midtone punch, NOT edge sharpening.
    pub clarity: f32,
    /// Grain seed (deterministic granulation for exact replay).
    pub seed: u64,
}

impl Default for Finish {
    fn default() -> Self {
        Self { impasto: 0.0, relief_robust: false, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, old_paper: false, plate_mark: false, weave: 0.0, contrast: 1.0, warmth: 0.0, clarity: 0.0, seed: 0 }
    }
}

/// A box blur of a scalar field (see `Canvas::box_blur_f`), for the painter's fields.
pub fn box_blur_pub(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    Canvas::box_blur_pub_impl(src, w, h, r)
}

/// Hashed value in `[0,1]` at an integer lattice point.
fn hash01(ix: i64, iy: i64, seed: u64) -> f32 {
    let mut z = seed.wrapping_add((ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_add((iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z as f32 / u64::MAX as f32
}

/// Smooth VALUE NOISE in `[0,1]`: hashed lattice points, smoothstep-interpolated. Correlated across neighbouring
/// pixels (unlike a per-pixel hash), so it reads as clustered mottle rather than static.
fn value_noise(fx: f32, fy: f32, seed: u64) -> f32 {
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let a = hash01(x0, y0, seed);
    let b = hash01(x0 + 1, y0, seed);
    let c = hash01(x0, y0 + 1, seed);
    let d = hash01(x0 + 1, y0 + 1, seed);
    let top = a + (b - a) * sx;
    let bot = c + (d - c) * sx;
    top + (bot - top) * sy
}

/// Deterministic paper-grain value in `[0,1]` at a pixel — the mottle granulation settles into. A LOW-FREQUENCY
/// value noise at the paper-tooth scale (a coarse cell plus a finer octave), NOT a per-pixel hash: real
/// granulation is pigment pooling in clusters across the tooth, so a per-pixel hash read as digital static.
/// The paper's RELIEF at a pixel (0 = a hollow, 1 = a standing fibre): the grain a dry brush skips over
/// (see `BrushConfig::skip`). One fixed seed — the sheet is the same sheet under every stroke and every
/// replay.
pub fn paper_relief(x: usize, y: usize) -> f32 {
    paper_grain(x, y, 0x5A9E_7001)
}

fn paper_grain(x: usize, y: usize, seed: u64) -> f32 {
    let coarse = value_noise(x as f32 / 5.0, y as f32 / 5.0, seed);
    let fine = value_noise(x as f32 / 2.2, y as f32 / 2.2, seed ^ 0x9E37_79B9);
    (coarse * 0.68 + fine * 0.32).clamp(0.0, 1.0)
}

/// OLD PAPER (`--oldpaper`): the finished picture as it would look PRINTED ON AN AGED SHEET. The picture is laid
/// over the sheet the way ink lies on paper — its white is the paper, its black a warm, slightly faded ink, and
/// every colour between is the paper seen through it — so any medium can sit on it. The sheet itself is cream
/// laid paper that has yellowed unevenly: a slow mottle, the fibres' grain, foxing (the brown blooms of damp)
/// in a few clusters, scattered specks, and edges darkened by handling. With `plate_mark`, the bevelled edge
/// of an intaglio plate is pressed a little inside the sheet's border: a broken line of ink it held, and a
/// breath of plate tone within. Everything is a deterministic function of the position, the sheet's size and
/// the seed, so a score replays to the same sheet.
pub fn old_paper(img: &mut RgbImage, seed: u64, plate_mark: bool) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let short = w.min(h) as f32;
    // The sheet's features are sized against a 1024-px sheet, so a replay at another size ages the same way.
    let unit = short / 1024.0;
    let fbm = |x: f32, y: f32, s: u64| 0.55 * value_noise(x, y, s) + 0.3 * value_noise(x * 2.1, y * 2.1, s ^ 0x51) + 0.15 * value_noise(x * 4.3, y * 4.3, s ^ 0xA7);
    let step = |a: f32, b: f32, v: f32| {
        let t = ((v - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    const PAPER: [f32; 3] = [238.0, 225.0, 198.0];
    const FOX: [f32; 3] = [188.0, 142.0, 86.0];
    const INK: [f32; 3] = [28.0, 23.0, 20.0];
    let inset = short * 0.035;
    let line = (short * 0.0022).max(1.0);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / unit, y as f32 / unit);
            // Uneven yellowing, the laid fibres, and the fine tooth.
            let mottle = fbm(u / 170.0, v / 170.0, seed ^ 0x01D_0001) - 0.5;
            let fibre = value_noise(u / 16.0, v / 2.4, seed ^ 0x01D_0002) - 0.5;
            let tooth = value_noise(u / 1.4, v / 1.4, seed ^ 0x01D_0003) - 0.5;
            // Laid paper: the mould's close wires leave fine ribs across the sheet, and its chain wires a thin
            // lighter line every inch or so along it.
            let laid = (v * std::f32::consts::TAU / 4.6 + 1.5 * (value_noise(u / 90.0, v / 90.0, seed ^ 0x01D_000D) - 0.5)).sin();
            let chain = 1.0 - (((u + 6.0 * (value_noise(v / 120.0, 0.5, seed ^ 0x01D_000E) - 0.5)).rem_euclid(96.0) - 48.0).abs() / 1.3).min(1.0);
            let light = 1.0 + 0.11 * mottle + 0.025 * fibre + 0.035 * tooth + 0.014 * laid + 0.022 * chain;
            // Foxing: blooms, but only in the few neighbourhoods where damp got in.
            let damp = step(0.56, 0.78, fbm(u / 300.0, v / 300.0, seed ^ 0x01D_0004));
            let bloom = step(0.45, 0.85, fbm(u / 55.0, v / 55.0, seed ^ 0x01D_0005) + 0.25 * (value_noise(u / 11.0, v / 11.0, seed ^ 0x01D_000C) - 0.5));
            let mut fox = 0.42 * damp * bloom;
            // Specks: a dark grain here and there.
            let (cx, cy) = ((u / 9.0).floor(), (v / 9.0).floor());
            let pick = value_noise(cx * 7.31 + 0.5, cy * 5.17 + 0.5, seed ^ 0x01D_0006);
            if pick > 0.955 {
                let (sx, sy) = ((cx + 0.2 + 0.6 * value_noise(cx + 0.5, cy + 9.5, seed ^ 0x01D_0007)) * 9.0, (cy + 0.2 + 0.6 * value_noise(cx + 4.5, cy + 0.5, seed ^ 0x01D_0008)) * 9.0);
                let r = 0.6 + 28.0 * (pick - 0.955);
                fox = fox.max(0.75 * (1.0 - (((u - sx).powi(2) + (v - sy).powi(2)).sqrt() / r)).clamp(0.0, 1.0));
            }
            // Handling: the edges are darker and browner, unevenly.
            let border = x.min(w - 1 - x).min(y).min(h - 1 - y) as f32 / short;
            let worn = step(0.09, 0.0, border + 0.05 * (fbm(u / 60.0, v / 60.0, seed ^ 0x01D_0009) - 0.5));
            fox = (fox + 0.3 * worn).min(0.9);
            let mut plate = 1.0f32;
            let mut held = 0.0f32;
            let mut press = 1.0f32;
            if plate_mark {
                // A hand-pulled impression: the ink is not laid evenly, and it breaks on the paper's tooth.
                press = (0.8 + 0.3 * fbm(u / 130.0, v / 130.0, seed ^ 0x01D_000F) + 0.16 * tooth).clamp(0.6, 1.0);
                // Signed distance into the plate (negative outside it), its edge wavering a little.
                let waver = unit * 1.6 * (value_noise(u / 45.0, v / 45.0, seed ^ 0x01D_000A) - 0.5);
                let into = (x.min(w - 1 - x).min(y).min(h - 1 - y)) as f32 - inset + waver;
                if into > 0.0 {
                    plate = 0.975;
                }
                // The ink the plate's edge held prints as a line that breaks where the edge was wiped clean.
                let wiped = step(0.3, 0.62, fbm(u / 70.0, v / 70.0, seed ^ 0x01D_000B));
                held = (1.0 - (into / line).abs()).clamp(0.0, 1.0) * (0.2 + 0.7 * wiped);
            }
            let px = img.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                let sheet = (PAPER[c] + (FOX[c] - PAPER[c]) * fox) * light * plate;
                let sheet = sheet + (INK[c] - sheet) * held;
                let through = 1.0 - (1.0 - px.0[c] as f32 / 255.0) * press;
                px.0[c] = (INK[c].min(sheet) + (sheet - INK[c].min(sheet)) * through).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// The index of the lightest pigment in a palette (highest CIELAB L*), used as "white"/ground.
pub fn lightest_pigment(palette: &Palette) -> usize {
    use crate::paint::color::srgb_to_lab;
    (0..palette.pigments.len())
        .max_by(|&a, &b| srgb_to_lab(palette.pigments[a].masstone).l.partial_cmp(&srgb_to_lab(palette.pigments[b].masstone).l).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0)
}

/// The index of the darkest pigment in a palette (lowest CIELAB L*) — the ink for a density medium.
pub fn darkest_pigment(palette: &Palette) -> usize {
    use crate::paint::color::srgb_to_lab;
    (0..palette.pigments.len())
        .min_by(|&a, &b| srgb_to_lab(palette.pigments[a].masstone).l.partial_cmp(&srgb_to_lab(palette.pigments[b].masstone).l).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0)
}

/// The index of the COOLEST pigment in a palette (most blue relative to red), used for a chromatic edge.
pub fn coolest_pigment(palette: &Palette) -> usize {
    (0..palette.pigments.len())
        .max_by(|&a, &b| {
            let ca = palette.pigments[a].masstone[2] as i32 - palette.pigments[a].masstone[0] as i32;
            let cb = palette.pigments[b].masstone[2] as i32 - palette.pigments[b].masstone[0] as i32;
            ca.cmp(&cb)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_paper_keeps_the_picture_and_ages_the_sheet() {
        // Left half white, right half black.
        let fresh = RgbImage::from_fn(200, 160, |x, _| if x < 100 { image::Rgb([255, 255, 255]) } else { image::Rgb([0, 0, 0]) });
        let mut aged = fresh.clone();
        old_paper(&mut aged, 7, false);
        let mean = |img: &RgbImage, x0: u32, x1: u32| {
            let mut s = [0f32; 3];
            for y in 40..120 {
                for x in x0..x1 {
                    for c in 0..3 {
                        s[c] += img.get_pixel(x, y).0[c] as f32;
                    }
                }
            }
            s.map(|v| v / (80 * (x1 - x0)) as f32)
        };
        let (paper, ink) = (mean(&aged, 30, 70), mean(&aged, 130, 170));
        assert!(paper[0] > paper[2] + 20.0 && paper[0] > 200.0, "white is now warm paper: {paper:?}");
        assert!(ink[0] < 45.0, "black is still ink: {ink:?}");
        let spread = (40..120).flat_map(|y| (30..70).map(move |x| (x, y))).map(|(x, y)| aged.get_pixel(x, y).0[1] as i32).fold((255, 0), |(lo, hi), v| (lo.min(v), hi.max(v)));
        assert!(spread.1 - spread.0 >= 4, "the sheet is uneven, not a flat tint: {spread:?}");
        let mut again = fresh.clone();
        old_paper(&mut again, 7, false);
        assert_eq!(aged, again, "the same seed ages the same sheet");
        let mut other = fresh.clone();
        old_paper(&mut other, 8, false);
        assert_ne!(aged, other, "another seed, another sheet");
    }

    #[test]
    fn a_plate_mark_is_pressed_inside_the_border() {
        let fresh = RgbImage::from_pixel(400, 400, image::Rgb([255, 255, 255]));
        let (mut plain, mut pressed) = (fresh.clone(), fresh);
        old_paper(&mut plain, 3, false);
        old_paper(&mut pressed, 3, true);
        // The mark sits 3.5% in: along that line the pressed sheet is darker than the plain one, on average.
        let along = |img: &RgbImage| (60..340).map(|x| (12..=16).map(|y| img.get_pixel(x, y).0[1] as f32).fold(255.0, f32::min)).sum::<f32>() / 280.0;
        assert!(along(&pressed) < along(&plain) - 20.0, "an inked line: {} against {}", along(&pressed), along(&plain));
        // Outside the plate the two sheets are the same sheet.
        assert_eq!(plain.get_pixel(200, 3), pressed.get_pixel(200, 3));
    }
    use crate::paint::color::{delta_e76, srgb_to_lab};
    use crate::paint::palette;

    #[test]
    fn diffusion_moves_pigment_toward_the_dark_or_the_light_and_conserves_it() {
        // Two wet cells: a loaded dark one and a thin light one. Zorn: [ochre, cad-red, black, white].
        let mk = || {
            let mut c = Canvas::white(2, 1, palette::ZORN, 0.7);
            c.deposit(0, 0, &[0.0, 0.0, 2.0, 0.0], 0.3);
            c.deposit(1, 0, &[0.0, 0.0, 0.4, 0.0], 0.1);
            c.wetness[0] = 1.0;
            c.wetness[1] = 1.0;
            c
        };
        let total = |c: &Canvas| c.conc_at(0, 0)[2] + c.conc_at(1, 0)[2];
        let mut iso = mk();
        iso.bleed_with(0.5, 0.0);
        let mut plain = mk();
        plain.bleed(0.5);
        assert_eq!(iso.conc_at(0, 0), plain.conc_at(0, 0), "diffuse 0 is byte-identical to the plain bleed");
        let mut dark = mk();
        dark.bleed_with(0.5, 1.0);
        let mut light = mk();
        light.bleed_with(0.5, -1.0);
        assert!(dark.conc_at(0, 0)[2] > iso.conc_at(0, 0)[2], "+1: the dark cell charges up ({} > {})", dark.conc_at(0, 0)[2], iso.conc_at(0, 0)[2]);
        assert!(light.conc_at(0, 0)[2] < iso.conc_at(0, 0)[2], "-1: the dark cell gives pigment to the light ({} < {})", light.conc_at(0, 0)[2], iso.conc_at(0, 0)[2]);
        assert!((total(&dark) - total(&iso)).abs() < 1e-4 && (total(&light) - total(&iso)).abs() < 1e-4, "the drift conserves pigment");
        // Dry paper neither gives nor takes.
        let mut dry = mk();
        dry.wetness[1] = 0.0;
        let before = dry.conc_at(1, 0)[2];
        dry.bleed_with(0.5, -1.0);
        assert!((dry.conc_at(1, 0)[2] - before).abs() < 1e-6, "a dry cell takes no drifting pigment");
    }

    #[test]
    fn a_cell_between_darker_neighbours_is_never_drained() {
        // A one-cell hole in a loaded wet sheet, pulled toward the dark by all four neighbours at an absurd
        // rate: the outflow cap keeps at least half of its load in one step (without it the cell is drained to
        // bare paper — the white-speck failure), and it still gives some.
        let mut c = Canvas::white(7, 7, palette::ZORN, 0.7);
        for y in 0..7 {
            for x in 0..7 {
                if (x, y) != (3, 3) {
                    c.deposit(x, y, &[0.0, 0.0, 3.0, 0.0], 0.3);
                }
            }
        }
        c.deposit(3, 3, &[0.0, 0.0, 0.3, 0.0], 0.1);
        for w in c.wetness.iter_mut() {
            *w = 1.0;
        }
        let before = c.conc_at(3, 3)[2];
        c.pigment_drift(1.0, 1.0, 4.0);
        let after = c.conc_at(3, 3)[2];
        assert!(after >= before * (1.0 - DRIFT_MAX_OUT) - 1e-6, "the centre keeps at least half: {after} of {before}");
        assert!(after < before, "…but does give some pigment to the darks");
    }

    #[test]
    fn a_wash_fills_its_rings_even_odd_with_a_hard_edge() {
        // A 20×20 sheet: an outer square with a square hole; the hole stays paper, the edge is exact.
        let mut c = Canvas::white(20, 20, palette::ZORN, 0.7);
        let outer = [[2.0, 2.0], [16.0, 2.0], [16.0, 16.0], [2.0, 16.0]];
        let hole = [[6.0, 6.0], [12.0, 6.0], [12.0, 12.0], [6.0, 12.0]];
        let mut rings: Vec<[f32; 2]> = outer.to_vec();
        rings.push([f32::NAN, f32::NAN]);
        rings.extend_from_slice(&hole);
        c.fill_rings(&rings, &[0.0, 0.0, 3.0, 0.0], 0.7, 0.0);
        let dark = |c: &Canvas, x: u32, y: u32| c.color_at(x, y)[0] < 120;
        assert!(dark(&c, 3, 3) && dark(&c, 15, 15) && dark(&c, 4, 10), "inside the outer ring is washed");
        assert!(!dark(&c, 8, 8) && !dark(&c, 11, 11), "the hole stays paper");
        assert!(!dark(&c, 1, 1) && !dark(&c, 17, 10) && !dark(&c, 10, 17), "outside stays paper");
        assert!(dark(&c, 2, 2) && !dark(&c, 16, 16), "pixel-centre rule: the top-left edge pixel is in, the bottom-right is out");
        assert!((c.wetness[3 * 20 + 3] - 0.7).abs() < 1e-6 && c.wetness[8 * 20 + 8] < 1e-6, "wet inside, dry in the hole");
        // A wet edge blooms OUTWARD: just outside the ring carries some pigment, further out none, inside full.
        let mut fz = Canvas::white(24, 24, palette::ZORN, 0.7);
        let sq = [[6.0, 6.0], [16.0, 6.0], [16.0, 16.0], [6.0, 16.0]];
        fz.fill_rings(&sq, &[0.0, 0.0, 3.0, 0.0], 0.7, 3.0);
        let v = |x: u32, y: u32| fz.color_at(x, y)[0];
        assert!(v(10, 10) < v(17, 10) && v(17, 10) < v(19, 10) && v(22, 10) > 235, "inside {} < bloom {} < faint {} < paper {}", v(10, 10), v(17, 10), v(19, 10), v(22, 10));
        assert!((v(10, 10) as i32 - v(6, 10) as i32).abs() < 3, "no pale rim inside the edge");
    }

    #[test]
    fn white_ground_reads_white() {
        let c = Canvas::white(4, 4, palette::ZORN, 0.7);
        let col = c.color_at(1, 1);
        assert!(col[0] > 235 && col[1] > 235 && col[2] > 225, "white ground is white: {col:?}");
    }

    #[test]
    fn a_heavy_deposit_dominates_the_ground() {
        // Zorn: [ochre, cad-red, black, white]. Ground is white; deposit a lot of red.
        let mut c = Canvas::white(2, 2, palette::ZORN, 0.7);
        c.deposit(0, 0, &[0.0, 6.0, 0.0, 0.0], 0.5);
        let col = c.color_at(0, 0);
        let d = delta_e76(srgb_to_lab(col), srgb_to_lab(palette::CADMIUM_RED.masstone));
        assert!(d < 25.0, "a heavy deposit reads mostly as the pigment: {col:?} (ΔE {d})");
        assert!(c.height[0] > 0.4, "height accumulated");
    }

    #[test]
    fn wipe_scrapes_pigment_and_height_back() {
        let mut c = Canvas::white(2, 2, palette::ZORN, 0.7);
        c.deposit(0, 0, &[0.0, 1.2, 0.0, 0.0], 0.6); // a red pass + impasto
        let before = srgb_to_lab(c.color_at(0, 0)).l;
        let h_before = c.height[0];
        c.wipe(0, 0, 0.8);
        assert!(srgb_to_lab(c.color_at(0, 0)).l > before + 3.0, "wiping back lightens toward the ground");
        assert!(c.height[0] < h_before, "impasto height scraped down");
    }

    #[test]
    fn a_thin_deposit_shifts_less_than_a_thick_one() {
        // Opacity is emergent: a thin glaze perturbs the ground less than a covering deposit of the same
        // pigment (both shift it — a strong tinter like red visibly tints even thin).
        let base = srgb_to_lab(Canvas::white(1, 1, palette::ZORN, 0.7).color_at(0, 0));
        let mut thin = Canvas::white(1, 1, palette::ZORN, 0.7);
        thin.deposit(0, 0, &[0.0, 0.05, 0.0, 0.0], 0.0);
        let mut thick = Canvas::white(1, 1, palette::ZORN, 0.7);
        thick.deposit(0, 0, &[0.0, 1.0, 0.0, 0.0], 0.0);
        let d_thin = delta_e76(base, srgb_to_lab(thin.color_at(0, 0)));
        let d_thick = delta_e76(base, srgb_to_lab(thick.color_at(0, 0)));
        assert!(d_thin > 0.0, "a glaze does tint");
        assert!(d_thin < d_thick, "thin glaze shifts less than a covering deposit ({d_thin} < {d_thick})");
    }
}

#[cfg(test)]
mod flow_tests {
    use super::*;
    use crate::paint::palette;

    #[test]
    fn the_box_blur_matches_a_naive_mean() {
        let (w, h) = (7usize, 5usize);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32).collect();
        let got = Canvas::box_blur_f(&src, w, h, 2);
        for y in 0..h {
            for x in 0..w {
                let (mut s, mut c) = (0f32, 0f32);
                for yy in y.saturating_sub(2)..=(y + 2).min(h - 1) { for xx in x.saturating_sub(2)..=(x + 2).min(w - 1) { s += src[yy * w + xx]; c += 1.0; } }
                assert!((got[y * w + x] - s / c).abs() < 1e-4, "at {x},{y}: {} vs {}", got[y * w + x], s / c);
            }
        }
    }

    #[test]
    fn the_flow_fills_the_gaps_between_lanes_and_leaves_a_rim() {
        // Two wet dots a few pixels apart on paper: after the flow the gap between them is painted (the
        // film joins them), and the film's edge is deeper than its interior.
        let pal = palette::EARTH;
        let mut c = Canvas::white(48, 48, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let mut load = vec![0f32; pal.pigments.len()];
        load[3] = 1.5;
        for &x in &[20u32, 26] { c.deposit(x, 24, &load, 0.0); }
        let gap_before = c.saturation_at(23, 24);
        c.flow(1.0, 4.0, 1.0, 0.0, 0.0);
        let gap_after = c.saturation_at(23, 24);
        assert!(gap_before == 0.0 && gap_after > 0.0, "the gap is painted by the film: {gap_before} → {gap_after}");
        assert!(c.saturation_at(23, 24) > c.saturation_at(23, 40), "and the paper beyond the film stays bare");
        let interior = c.saturation_at(23, 24);
        let edge: f32 = (0..48).map(|y| c.saturation_at(23, y)).fold(0.0, f32::max);
        assert!(edge >= interior, "the rim is at least as deep as the interior ({edge} vs {interior})");
    }

    #[test]
    fn two_washes_meet_in_a_tide_line() {
        // A heavy blue wash and a thin ochre wash laid side by side, wet: each keeps its own colour (the
        // diffusion is confined to the wash), and the BACKRUN forms on the DRIER side — the thin wash is
        // deeper along the line where they meet than in its interior, the heavy wash is not deepened.
        let pal = palette::EARTH;
        let mut c = Canvas::white(64, 32, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let n = c.n;
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        a[1] = 1.0;
        b[3] = 0.3;
        for y in 0..32u32 { for x in 0..32u32 { c.deposit(x, y, &a, 0.0); c.deposit(x + 32, y, &b, 0.0); } }
        c.flow(1.0, 4.0, 1.0, 0.0, 0.0);
        let at = |x: usize, y: usize, k: usize| c.conc[(y * 64 + x) * n + k];
        assert!(at(8, 16, 3) == 0.0 && at(56, 16, 1) == 0.0, "no pigment crosses into the other wash");
        let line: f32 = (32..40).map(|x| at(x, 16, 3)).fold(0.0, f32::max);
        assert!(line > at(56, 16, 3) * 1.15, "the thin wash is deeper at the meeting line: {line} vs {}", at(56, 16, 3));
        let heavy_line: f32 = (24..32).map(|x| at(x, 16, 1)).fold(0.0, f32::max);
        assert!(heavy_line <= at(8, 16, 1) * 1.02, "the heavy wash is not deepened: {heavy_line} vs {}", at(8, 16, 1));
    }

    #[test]
    fn a_thin_mass_on_bare_paper_grows_no_rim() {
        // A 2-px dark line on bare paper: nothing wet on either flank for a backrun to run into, so the
        // flow leaves no rim — the line's own pigment total is unchanged.
        let pal = palette::EARTH;
        let mut c = Canvas::white(64, 32, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let n = c.n;
        let mut ink = vec![0f32; n];
        ink[1] = 3.0;
        for x in 4..60u32 { for y in 15..17u32 { c.deposit(x, y, &ink, 0.0); } }
        let peak_before = c.conc.iter().cloned().fold(0f32, f32::max);
        c.flow(1.0, 4.0, 1.0, 0.0, 0.0);
        // The water may carry the line's pigment OUT (a softened line), but nothing is DEEPENED: no pixel
        // beside the line stands above what the line itself carried — no rim.
        let peak_after = c.conc.iter().cloned().fold(0f32, f32::max);
        assert!(peak_after <= peak_before * 1.01, "no rim added to a thin mass: peak {peak_before} → {peak_after}");
        let flank: f32 = (4..60).map(|x| c.conc[(13 * 64 + x) * n + 1]).fold(0f32, f32::max);
        let line: f32 = (4..60).map(|x| c.conc[(15 * 64 + x) * n + 1]).fold(0f32, f32::max);
        assert!(flank < line, "the flank stays lighter than the line: {flank} vs {line}");
    }

    #[test]
    fn two_washes_of_one_weight_blend_without_a_line() {
        // The same two washes at the same load: no tide line forms between them (they were laid together,
        // wet) — the rim is a fact of the load contrast, not a contour round every wash.
        let pal = palette::EARTH;
        let mut c = Canvas::white(64, 32, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let n = c.n;
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        a[1] = 1.0;
        b[3] = 1.0;
        for y in 0..32u32 { for x in 0..32u32 { c.deposit(x, y, &a, 0.0); c.deposit(x + 32, y, &b, 0.0); } }
        c.flow(1.0, 4.0, 1.0, 0.0, 0.0);
        let at = |x: usize, y: usize, k: usize| c.conc[(y * 64 + x) * n + k];
        let line: f32 = (26..36).map(|x| at(x, 16, 1)).fold(0.0, f32::max);
        assert!(line < at(8, 16, 1) * 1.05, "no line between equal washes: {line} vs {}", at(8, 16, 1));
    }

    #[test]
    fn the_grain_settles_where_the_wash_pooled() {
        // One earth pigment, a heavy wash on the left and a thin one on the right, granulating in the water:
        // the heavy wash mottles (its pigment varies across the tooth), the thin one much less.
        let pal = palette::EARTH;
        let mut c = Canvas::white(64, 32, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let n = c.n;
        let earth = (0..n).min_by_key(|&k| { let m = pal.pigments[k].masstone; (m[0].max(m[1]).max(m[2]) as i32 - m[0].min(m[1]).min(m[2]) as i32) * 255 / m[0].max(m[1]).max(m[2]).max(1) as i32 }).unwrap();
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        a[earth] = 1.6;
        b[earth] = 0.2;
        for y in 0..32u32 { for x in 0..32u32 { c.deposit(x, y, &a, 0.0); c.deposit(x + 32, y, &b, 0.0); } }
        let mut plain = c.clone();
        plain.flow(1.0, 3.0, 0.0, 0.0, 0.0);
        c.flow(1.0, 3.0, 0.0, 1.0, 0.0);
        let spread = |cv: &Canvas, x0: usize| -> f32 {
            let vals: Vec<f32> = (8..24).flat_map(|y| (x0..x0 + 16).map(move |x| (x, y))).map(|(x, y)| cv.conc[(y * 64 + x) * n + earth]).collect();
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt() / mean.max(1e-6)
        };
        assert!(spread(&plain, 8) < 0.01, "without grain the heavy wash is even: {}", spread(&plain, 8));
        assert!(spread(&c, 8) > 0.1, "with grain it mottles: {}", spread(&c, 8));
        assert!(spread(&c, 8) > spread(&c, 40) * 3.0, "and the thin wash far less: {} vs {}", spread(&c, 8), spread(&c, 40));
    }

    #[test]
    fn the_selective_flow_holds_back_at_a_hard_edge_and_floods_the_soft_mass() {
        // One wash of one pigment with a sharp heavy stripe through it: flooded (selective 0) the stripe's
        // edge softens; selective, the water holds back at that hard edge and the stripe keeps its edge,
        // while far from it the wash still diffuses (a gap in the lanes is still filled).
        let pal = palette::EARTH;
        let mk = || {
            let mut c = Canvas::white(96, 32, pal, 0.85).with_opacity(0.45).with_transmittance(true);
            let n = c.n;
            let (mut thin, mut heavy) = (vec![0f32; n], vec![0f32; n]);
            thin[1] = 0.3;
            heavy[1] = 3.0;
            for y in 0..32u32 { for x in 0..96u32 { if x != 70 { c.deposit(x, y, &thin, 0.0); } } }
            for y in 0..32u32 { for x in 20..28u32 { c.deposit(x, y, &heavy, 0.0); } }
            c
        };
        let (mut flood, mut sel) = (mk(), mk());
        flood.flow(1.0, 4.0, 0.0, 0.0, 0.0);
        sel.flow(1.0, 4.0, 0.0, 0.0, 1.0);
        let n = flood.n;
        let at = |c: &Canvas, x: usize| c.conc[(16 * 96 + x) * n + 1];
        // Just outside the stripe: the flooded wash received the stripe's pigment, the selective one far less.
        assert!(at(&sel, 30) < at(&flood, 30) * 0.5, "the water holds back at the hard edge: {} vs {}", at(&sel, 30), at(&flood, 30));
        // Far from it the lane gap is filled either way.
        assert!(at(&sel, 70) > 0.0 && at(&flood, 70) > 0.0, "the soft mass still floods");
    }

    #[test]
    fn the_weave_shows_under_thin_paint_and_is_covered_by_thick() {
        // One canvas, a thin glaze on the left and a built-up passage on the right, relit with the weave:
        // the left half's pixels vary with the threads, the right half's hardly at all.
        let pal = palette::ZORN;
        let mut c = Canvas::white(64, 32, pal, 0.9);
        let n = c.n;
        let (mut thin, mut thick) = (vec![0f32; n], vec![0f32; n]);
        thin[1] = 0.15;
        thick[1] = 3.0;
        for y in 0..32u32 { for x in 0..32u32 { c.deposit(x, y, &thin, 0.1); c.deposit(x + 32, y, &thick, 2.0); } }
        let img = c.to_image_finished(&Finish { weave: 1.0, impasto: 0.0, ..Default::default() });
        let spread = |x0: u32| -> f32 {
            let v: Vec<f32> = (8..24).flat_map(|y| (x0..x0 + 24).map(move |x| (x, y))).map(|(x, y)| img.get_pixel(x, y).0[1] as f32).collect();
            let m = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        assert!(spread(4) > 2.0, "the thin glaze shows the threads: {}", spread(4));
        assert!(spread(36) < spread(4) * 0.25, "the built-up paint covers them: {} vs {}", spread(36), spread(4));
    }

    #[test]
    fn the_collision_drags_wet_paint_along_its_striation_and_leaves_dry_paint_alone() {
        // Two wet ridged bands of different pigments running horizontally, touching: after the collision
        // the colours have mixed along the bands' direction (a pixel inside the red band carries some ochre
        // from its neighbour band), and a DRY band of the same layout does not move.
        let pal = palette::ZORN;
        let lay = |c: &mut Canvas, wet: f32| {
            let n = c.n;
            let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
            a[1] = 2.0; // cadmium red
            b[0] = 2.0; // ochre
            for x in 0..64u32 {
                for y in 8..16u32 { c.deposit(x, y, &a, 1.0 + 0.6 * ((x / 2) % 2) as f32); c.wetness[(y * 64 + x) as usize] = wet; }
                for y in 16..24u32 { c.deposit(x, y, &b, 1.0 + 0.6 * ((x / 2 + 1) % 2) as f32); c.wetness[(y * 64 + x) as usize] = wet; }
            }
        };
        let mut wetc = Canvas::white(64, 32, pal, 0.9);
        lay(&mut wetc, 1.0);
        let n = wetc.n;
        let before = wetc.conc.clone();
        wetc.collide(1.0, 4.0, None, 2.0);
        let moved: f32 = wetc.conc.iter().zip(&before).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved > 1.0, "wet paint moved: {moved}");
        let mut dryc = Canvas::white(64, 32, pal, 0.9);
        lay(&mut dryc, 0.0);
        let before_d = dryc.conc.clone();
        dryc.collide(1.0, 4.0, None, 2.0);
        assert_eq!(dryc.conc, before_d, "dry paint does not move");
        // Masked pixels do not move either.
        let mut maskc = Canvas::white(64, 32, pal, 0.9);
        lay(&mut maskc, 1.0);
        let before_m = maskc.conc.clone();
        let mask = vec![1.0f32; 64 * 32];
        maskc.collide(1.0, 4.0, Some(&mask), 1.0);
        assert_eq!(maskc.conc, before_m, "the face mask holds the paint still");
        let _ = n;
    }

    #[test]
    fn the_flow_leaves_the_dried_passes_where_they_were() {
        // A crisp dot laid in an earlier pass (marked = dried) and a wet wash laid after it: the flow moves
        // the wash, not the dot — the dot's pigment and its sharp edge survive exactly.
        let pal = palette::EARTH;
        let mut c = Canvas::white(48, 48, pal, 0.85).with_opacity(0.45).with_transmittance(true);
        let mut dot = vec![0f32; pal.pigments.len()];
        dot[1] = 2.0;
        c.deposit(10, 10, &dot, 0.0);
        c.mark_film();
        let n = c.n;
        let dot_before: Vec<f32> = c.conc[(10 * 48 + 10) * n..(10 * 48 + 11) * n].to_vec();
        let beside_before = c.saturation_at(11, 10);
        let mut wash = vec![0f32; pal.pigments.len()];
        wash[3] = 1.0;
        for x in 30..40u32 { for y in 30..34u32 { c.deposit(x, y, &wash, 0.0); } }
        c.flow(1.0, 3.0, 0.5, 0.0, 0.0);
        let dot_after: Vec<f32> = c.conc[(10 * 48 + 10) * n..(10 * 48 + 11) * n].to_vec();
        assert_eq!(dot_before, dot_after, "the dried dot does not move");
        assert_eq!(beside_before, c.saturation_at(11, 10), "nor does its edge soften");
        assert!(c.saturation_at(29, 32) > 0.0 && c.saturation_at(41, 32) > 0.0, "the wet wash spread beyond its footprint");
    }
}
