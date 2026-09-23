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
    /// — the ground is NOT baked in here; it shows through via opacity where the paint is thin.
    conc: Vec<f32>,
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
    n: usize,
}

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
        Self { w, h, palette, conc: vec![0.0; px * n], height: vec![0.0; px], wetness: vec![0.0; px], tooth: vec![tooth.clamp(0.0, 1.0); px], ground_lin, opacity: 1.0, n }
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

    /// Total pigment concentration at a pixel — the "saturation" that throttles further deposit.
    pub fn saturation_at(&self, x: u32, y: u32) -> f32 {
        self.conc_at(x, y).iter().sum()
    }

    /// Deposit a concentration delta at a pixel (adds to the accumulated pigment) and raise its height.
    pub fn deposit(&mut self, x: u32, y: u32, delta: &[f32], height: f32) {
        let i = self.idx(x, y);
        for c in 0..self.n {
            self.conc[i + c] += delta.get(c).copied().unwrap_or(0.0).max(0.0);
        }
        let p = y as usize * self.w as usize + x as usize;
        self.height[p] += height.max(0.0);
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
        let s = strength.clamp(0.0, 1.0);
        if s <= 0.0 {
            return;
        }
        let (w, h) = (self.w as usize, self.h as usize);
        let n = self.n;
        let iters = (1.0 + 5.0 * s).round() as usize; // more strength → farther bloom
        for _ in 0..iters {
            let src = self.conc.clone();
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let a = s * self.wetness[p].clamp(0.0, 1.0);
                    if a <= 1e-4 {
                        continue;
                    }
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
        let total: f32 = conc.iter().map(|c| c.max(0.0)).sum();
        if total <= 1e-4 {
            return color::linear_to_srgb(self.ground_lin);
        }
        let paint = pigment::mix_linear(self.palette.pigments, conc);
        let alpha = 1.0 - (-OPACITY_K * self.opacity * total).exp();
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
                    let total: f32 = self.conc[(y * w + x) * self.n..(y * w + x) * self.n + self.n].iter().map(|v| v.max(0.0)).sum();
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
                        let d = f.granulate * paint * (grain(x, y, f.seed) - 0.5) * 0.7;
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
                let total: f32 = self.conc[idx * self.n..idx * self.n + self.n].iter().map(|v| v.max(0.0)).sum();
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
        // IMPASTO + SHEEN relight from the paint HEIGHT.
        if f.impasto > 1e-4 || f.sheen > 1e-4 {
            let peak = self.height.iter().copied().fold(0.0_f32, f32::max).max(1e-4);
            let (lx, ly) = (0.55_f32, 0.83_f32);
            let gain = 1.6 * f.impasto.clamp(0.0, 1.0);
            let spec = 0.9 * f.sheen.clamp(0.0, 1.0);
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
                    let total: f32 = self.conc[(y * w + x) * self.n..(y * w + x) * self.n + self.n].iter().map(|v| v.max(0.0)).sum();
                    let amt = (total / 2.0).clamp(0.0, 1.0);
                    // Diffuse impasto shading + a sharper glossy highlight on the near ridges (sheen).
                    let mut shade = gain * facing * amt;
                    if spec > 0.0 && facing > 0.0 {
                        shade += spec * facing * facing * amt;
                    }
                    let shade = shade.clamp(-0.55, 0.85);
                    if shade.abs() < 1e-4 {
                        continue;
                    }
                    let p = img.get_pixel_mut(x as u32, y as u32);
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
                    let wob = 0.55 + 0.9 * grain(x / 3, y / 3, f.seed ^ 0x9E37);
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
        img
    }
}

/// The MEDIUM'S material finish, applied at output (§8.5). Neutral by default (= `to_image`).
#[derive(Clone, Copy, Debug)]
pub struct Finish {
    /// Impasto relief strength (0 flat → 1 thick, light-catching).
    pub impasto: f32,
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
        Self { impasto: 0.0, chroma: 1.0, dry_shift: 0.0, granulate: 0.0, sheen: 0.0, edge_pool: 0.0, paper_edge: 0.0, contrast: 1.0, warmth: 0.0, clarity: 0.0, seed: 0 }
    }
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
fn grain(x: usize, y: usize, seed: u64) -> f32 {
    let coarse = value_noise(x as f32 / 5.0, y as f32 / 5.0, seed);
    let fine = value_noise(x as f32 / 2.2, y as f32 / 2.2, seed ^ 0x9E37_79B9);
    (coarse * 0.68 + fine * 0.32).clamp(0.0, 1.0)
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
    use crate::paint::color::{delta_e76, srgb_to_lab};
    use crate::paint::palette;

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
