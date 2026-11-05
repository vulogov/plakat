//! THE SIGHT (7.17.0, set 1; Cargo feature `aqua-sight`): the semantic brain's eye for `--medium watercolour`.
//!
//! The water engine (`Canvas::aqua_inject_with` / `aqua_settle_full`) is final; what it still lacked was a brain
//! that SEES. The parser's focus (`aqua_engine_pass`) was the subject's matte when the plan had found a face and,
//! for every other picture, the top third of the Sobel gradient energy — a landscape, a ship, a tunnel painted by
//! where the pixels were busy, not by what they were. The sight is a per-pixel SALIENCY map `[0, 1]` fused from
//! the vision models the studio already runs — U²-Net's salient-object matte, Depth-Anything-V2's relative depth,
//! the SCRFD face mask — and the sheet's own low-level cue (the centre-weighted contrast field), handed to the
//! painter as `PaintParams::sight` and blended into the engine's focus by the key `aqua_sight`. The engine is
//! untouched: the sight changes where and how much the brush goes, never how the water behaves. The map never
//! enters the score — the brain's decisions are baked into every touch record, so the replay stays exact without
//! it. Off this feature (or at the key 0) the field stays `None` and every sheet is byte-identical; no other
//! medium reads it.
//!
//! The fusion (`fuse`) is a pure function, testable without weights: the cues combine as a NOISY-OR,
//! `S = 1 − Π(1 − w·c²)`, so any one strong cue makes a pixel salient and the cues never cancel each other — each
//! cue as its CONFIDENCE, squared, so a half-sure matte counts a quarter (U²-Net's soft tail over the water and the
//! waves' contrast no longer add up to a focus on the sea); the
//! result is normalised by its 99th percentile, clamped to `[0, 1]`, and a face is salient outright (the max with
//! the face mask). The weights: the matte 1, the face 1, the NEAR depth ½ (what stands ABOVE THE GROUND PLANE: a
//! row's far quarter is the plane — the sea, the floor, the road recede row by row — and the depth a pixel stands
//! nearer than its row's plane, rescaled to the 99th percentile, is the tier; `near_from_depth`), the low-level
//! contrast ½.
//!
//! THE CLASSES (set 3, the key `aqua_sight_class`): SegFormer-B0 (ADE20k, `pipelines::segformer`) folded into the
//! eight classes a watercolourist treats differently (`PainterClass`: sky, water, ground, foliage, built, thing,
//! animal, person — a class is a technique, not a name; `painter_class` is the table). Each class has a SALIENCE of
//! its own (the sky nothing, the water and the ground little, a thing much, a person or an animal everything): the
//! class enters the fusion twice, as a cue `salience × confidence` into the noisy-OR (a ship's sails are salient
//! because they are a ship, whatever the matte saw) and as a DAMPING `1 − key × confidence × (1 − salience)` on the
//! fused map before its normalisation (the sea's waves and the sky's clouds, however busy, are the sea and the sky).
//! And each class has a WATER, the technique: +1 one wet wash (the sky, the water), −1 a dry brush (the foliage), 0
//! the brush as it is — `SightMaps::wash`, handed to the painter as `PaintParams::sight_wash`.
//!
//! Set 4: the class table is the plan's to tune (`ClassMap::tune`, the plan's `aqua_sight_classes` — per class its
//! salience and its water), and the technique YIELDS TO THE SALIENCY: the wash is `water × confidence × (1 − S)`,
//! so a pixel the fused sight finds salient is painted as the brush is whatever SegFormer called it — a white fox's
//! fur called foliage is not dry-brushed, the sea the sight does not look at is.
//!
//! Set 5 (the painter's `aqua_sight_gradient`, `aqua_sight_barrier`): the base of a wash class — its low-pass — is
//! laid ONCE at its final tone and then counts as paid, the later touches and the drawing after the washes owing
//! only the picture's darks and contrasts over it (the debt cap and the high-pass gate in `aqua_engine_pass` and the
//! restate gate); and the boundary of a wash class is a CAPILLARY SEAM in the solver (`Canvas::add_capillary_seam`:
//! the flow slowed, the seam drying as a rim, the pigment pinned along it) instead of the masking fluid's wall.
//!
//! Set 6 (the painter's `aqua_sight_debt`, `aqua_sight_sessions`): the COMPOSITION by the sight — off the sight the
//! later sessions' debt is let go (the background a broad understated wash), and a session touches only where the
//! sight reaches key × its progress, the drawing after the washes only where it reaches key × the brush's fineness:
//! the finish's contrast between the subject and its ground leads the eye.

use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use std::path::Path;

/// The matte's weight in the noisy-OR.
pub const W_OBJECT: f32 = 1.0;
/// The near depth's weight.
pub const W_NEAR: f32 = 0.5;
/// The low-level contrast field's weight.
pub const W_LOW: f32 = 0.5;
/// The class cue's weight.
pub const W_CLASS: f32 = 1.0;

/// THE PAINTER'S CLASSES: what a watercolourist does differently, not what a thing is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PainterClass {
    Sky,
    Water,
    Ground,
    Foliage,
    Built,
    Thing,
    Animal,
    Person,
}

/// The classes in index order (`PainterClass::index`).
pub const CLASSES: [PainterClass; 8] = [
    PainterClass::Sky,
    PainterClass::Water,
    PainterClass::Ground,
    PainterClass::Foliage,
    PainterClass::Built,
    PainterClass::Thing,
    PainterClass::Animal,
    PainterClass::Person,
];

impl PainterClass {
    pub fn index(self) -> usize {
        CLASSES.iter().position(|c| *c == self).unwrap_or(5)
    }

    pub fn name(self) -> &'static str {
        match self {
            PainterClass::Sky => "sky",
            PainterClass::Water => "water",
            PainterClass::Ground => "ground",
            PainterClass::Foliage => "foliage",
            PainterClass::Built => "built",
            PainterClass::Thing => "thing",
            PainterClass::Animal => "animal",
            PainterClass::Person => "person",
        }
    }

    /// How salient the class is in itself — the painter's prior, before any cue of the picture.
    pub fn salience(self) -> f32 {
        match self {
            PainterClass::Sky => 0.05,
            PainterClass::Water => 0.2,
            PainterClass::Ground => 0.35,
            PainterClass::Foliage => 0.5,
            PainterClass::Built => 0.7,
            PainterClass::Thing => 0.85,
            PainterClass::Animal => 1.0,
            PainterClass::Person => 1.0,
        }
    }

    /// The class's technique as water: +1 one wet wash, 0 the brush as it is, −1 a dry brush.
    pub fn water(self) -> f32 {
        match self {
            PainterClass::Sky => 1.0,
            PainterClass::Water => 1.0,
            PainterClass::Ground => 0.3,
            PainterClass::Foliage => -0.5,
            PainterClass::Built => -0.2,
            PainterClass::Thing => 0.0,
            PainterClass::Animal => 0.0,
            PainterClass::Person => 0.0,
        }
    }

    /// The class's colour in the review dump.
    pub fn colour(self) -> [u8; 3] {
        match self {
            PainterClass::Sky => [120, 180, 255],
            PainterClass::Water => [30, 90, 200],
            PainterClass::Ground => [170, 130, 80],
            PainterClass::Foliage => [60, 160, 60],
            PainterClass::Built => [160, 160, 160],
            PainterClass::Thing => [255, 170, 40],
            PainterClass::Animal => [220, 60, 160],
            PainterClass::Person => [255, 40, 40],
        }
    }
}

/// ADE20k's label (as the checkpoint names it) to the painter's class; a name the table does not know is a thing.
pub fn painter_class(label: &str) -> PainterClass {
    let l = label.trim().to_ascii_lowercase();
    let l = l.split([';', ',']).next().unwrap_or("").trim();
    const SKY: &[&str] = &["sky"];
    const WATER: &[&str] = &["water", "sea", "river", "lake", "waterfall", "swimming pool", "fountain"];
    const GROUND: &[&str] = &[
        "floor", "road", "grass", "sidewalk", "earth", "field", "sand", "path", "runway", "dirt track", "land", "hill",
        "mountain", "rock", "rug",
    ];
    const FOLIAGE: &[&str] = &["tree", "plant", "palm", "flower"];
    const BUILT: &[&str] = &[
        "wall", "building", "ceiling", "house", "skyscraper", "fence", "railing", "column", "bridge", "tower", "hovel",
        "awning", "tent", "pier", "stairs", "stairway", "step", "bannister", "door", "windowpane", "grandstand", "booth",
        "canopy", "escalator", "pole", "streetlight", "base", "bar", "fireplace", "stage", "counter", "countertop",
        "kitchen island", "radiator", "shower", "hood", "screen door", "blind", "curtain",
    ];
    const PERSON: &[&str] = &["person"];
    const ANIMAL: &[&str] = &["animal"];
    let has = |t: &[&str]| t.contains(&l);
    if has(SKY) {
        PainterClass::Sky
    } else if has(WATER) {
        PainterClass::Water
    } else if has(GROUND) {
        PainterClass::Ground
    } else if has(FOLIAGE) {
        PainterClass::Foliage
    } else if has(BUILT) {
        PainterClass::Built
    } else if has(PERSON) {
        PainterClass::Person
    } else if has(ANIMAL) {
        PainterClass::Animal
    } else {
        PainterClass::Thing
    }
}

/// The class map at the paint size: the painter's class of every pixel and the confidence (the summed probability
/// of the ADE classes folded into it).
#[derive(Debug, Clone)]
pub struct ClassMap {
    pub class: Vec<u8>,
    pub conf: Vec<f32>,
    /// The classes' saliences, in `CLASSES` order (`PainterClass::salience` unless tuned).
    pub salience: [f32; 8],
    /// The classes' water, in `CLASSES` order (`PainterClass::water` unless tuned).
    pub water: [f32; 8],
}

/// The table's defaults, in `CLASSES` order.
pub fn default_salience() -> [f32; 8] {
    let mut a = [0.0f32; 8];
    for (i, c) in CLASSES.iter().enumerate() {
        a[i] = c.salience();
    }
    a
}

/// The table's defaults, in `CLASSES` order.
pub fn default_water() -> [f32; 8] {
    let mut a = [0.0f32; 8];
    for (i, c) in CLASSES.iter().enumerate() {
        a[i] = c.water();
    }
    a
}

/// Bilinear resample a `gw × gh` map to `w × h`.
fn resize_bilinear(src: &[f32], gw: usize, gh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    if gw == 0 || gh == 0 || w == 0 || h == 0 || src.len() != gw * gh {
        return out;
    }
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * gh as f32 / h as f32 - 0.5).clamp(0.0, (gh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(gh - 1);
        let ty = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * gw as f32 / w as f32 - 0.5).clamp(0.0, (gw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let tx = fx - x0 as f32;
            let top = src[y0 * gw + x0] * (1.0 - tx) + src[y0 * gw + x1] * tx;
            let bot = src[y1 * gw + x0] * (1.0 - tx) + src[y1 * gw + x1] * tx;
            out[y * w + x] = top * (1.0 - ty) + bot * ty;
        }
    }
    out
}

/// Fold the checkpoint's class probabilities (`probs[c * gh * gw + y * gw + x]`, `labels[c]` the names) into the
/// painter's classes, resample to `w × h`, and take the class with the largest summed probability at every pixel.
pub fn collapse_classes(probs: &[f32], labels: &[String], gw: usize, gh: usize, w: usize, h: usize) -> Option<ClassMap> {
    let c = labels.len();
    let cells = gw * gh;
    if c == 0 || cells == 0 || probs.len() != c * cells || w == 0 || h == 0 {
        return None;
    }
    let mut acc: Vec<Vec<f32>> = vec![vec![0.0f32; cells]; CLASSES.len()];
    for (ci, label) in labels.iter().enumerate() {
        let pc = painter_class(label).index();
        let src = &probs[ci * cells..(ci + 1) * cells];
        for (a, p) in acc[pc].iter_mut().zip(src) {
            *a += *p;
        }
    }
    let maps: Vec<Vec<f32>> = acc.iter().map(|m| resize_bilinear(m, gw, gh, w, h)).collect();
    let mut class = vec![0u8; w * h];
    let mut conf = vec![0.0f32; w * h];
    for i in 0..w * h {
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        for (k, m) in maps.iter().enumerate() {
            if m[i] > bv {
                bv = m[i];
                best = k;
            }
        }
        class[i] = best as u8;
        conf[i] = bv.clamp(0.0, 1.0);
    }
    Some(ClassMap { class, conf, salience: default_salience(), water: default_water() })
}

/// The class map SNAPPED TO THE PICTURE'S EDGES: SegFormer's grid is a 128th of the picture, its boundary a
/// staircase a cell wide and a cell off the hull's real edge; so every pixel takes its class probabilities from
/// the 4 × 4 cells around it weighted by a tent in space and by the likeness of the pixel's colour to the cell's
/// mean colour (a joint bilateral upsampling), the sea's grey next to the hull's brown taking the sea's cells.
/// Falls back to the plain resampling where the picture is not `w × h`.
pub fn collapse_classes_guided(probs: &[f32], labels: &[String], gw: usize, gh: usize, img: &RgbImage) -> Option<ClassMap> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let c = labels.len();
    let cells = gw * gh;
    if c == 0 || cells == 0 || probs.len() != c * cells || w == 0 || h == 0 {
        return None;
    }
    // The painter's class probabilities at the cells.
    let mut acc: Vec<[f32; 8]> = vec![[0.0; 8]; cells];
    for (ci, label) in labels.iter().enumerate() {
        let pc = painter_class(label).index();
        for (q, p) in probs[ci * cells..(ci + 1) * cells].iter().enumerate() {
            acc[q][pc] += *p;
        }
    }
    // The cells' mean colours (linear-ish: sRGB / 255 is near enough for a likeness).
    let mut mean: Vec<[f32; 3]> = vec![[0.0; 3]; cells];
    let mut count: Vec<f32> = vec![0.0; cells];
    for (i, px) in img.pixels().enumerate() {
        let (x, y) = (i % w, i / w);
        let (cx, cy) = ((x * gw / w).min(gw - 1), (y * gh / h).min(gh - 1));
        let q = cy * gw + cx;
        for k in 0..3 {
            mean[q][k] += px.0[k] as f32 / 255.0;
        }
        count[q] += 1.0;
    }
    for (m, n) in mean.iter_mut().zip(&count) {
        if *n > 0.0 {
            for v in m.iter_mut() {
                *v /= *n;
            }
        }
    }
    let sigma2 = 2.0 * 0.08f32 * 0.08;
    let mut class = vec![0u8; w * h];
    let mut conf = vec![0.0f32; w * h];
    for (i, px) in img.pixels().enumerate() {
        let (x, y) = (i % w, i / w);
        let fx = (x as f32 + 0.5) * gw as f32 / w as f32 - 0.5;
        let fy = (y as f32 + 0.5) * gh as f32 / h as f32 - 0.5;
        let (x0, y0) = (fx.floor() as i64 - 1, fy.floor() as i64 - 1);
        let col = [px.0[0] as f32 / 255.0, px.0[1] as f32 / 255.0, px.0[2] as f32 / 255.0];
        let mut sum = [0.0f32; 8];
        let mut wsum = 0.0f32;
        for cy in y0..y0 + 4 {
            if cy < 0 || cy >= gh as i64 {
                continue;
            }
            let ty = (1.0 - (fy - cy as f32).abs() / 2.0).max(0.0);
            for cx in x0..x0 + 4 {
                if cx < 0 || cx >= gw as i64 {
                    continue;
                }
                let tx = (1.0 - (fx - cx as f32).abs() / 2.0).max(0.0);
                let q = cy as usize * gw + cx as usize;
                let m = &mean[q];
                let d2 = (col[0] - m[0]).powi(2) + (col[1] - m[1]).powi(2) + (col[2] - m[2]).powi(2);
                let wgt = tx * ty * ((-d2 / sigma2).exp() + 1e-3);
                for k in 0..8 {
                    sum[k] += wgt * acc[q][k];
                }
                wsum += wgt;
            }
        }
        let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
        for (k, v) in sum.iter().enumerate() {
            if *v > bv {
                bv = *v;
                best = k;
            }
        }
        class[i] = best as u8;
        conf[i] = if wsum > 0.0 { (bv / wsum).clamp(0.0, 1.0) } else { 0.0 };
    }
    Some(ClassMap { class, conf, salience: default_salience(), water: default_water() })
}

impl ClassMap {
    /// The table tuned from the plan: `(class name, salience, water)`, each value optional, a name the table does not
    /// know ignored, the values clamped (the salience to `[0, 1]`, the water to `[−1, 1]`).
    pub fn tune(&mut self, overrides: &[(String, Option<f32>, Option<f32>)]) {
        for (name, sal, wat) in overrides {
            let n = name.trim().to_ascii_lowercase();
            if let Some(i) = CLASSES.iter().position(|c| c.name() == n) {
                if let Some(v) = sal {
                    self.salience[i] = v.clamp(0.0, 1.0);
                }
                if let Some(v) = wat {
                    self.water[i] = v.clamp(-1.0, 1.0);
                }
            }
        }
    }

    /// The class cue: the class's salience by its confidence.
    pub fn cue(&self) -> Vec<f32> {
        self.class.iter().zip(&self.conf).map(|(c, p)| self.salience[(*c as usize).min(7)] * p).collect()
    }

    /// The damping by the key: `1 − key × confidence × (1 − salience)`.
    pub fn damp(&self, key: f32) -> Vec<f32> {
        let k = key.clamp(0.0, 1.0);
        self.class.iter().zip(&self.conf).map(|(c, p)| 1.0 - k * p * (1.0 - self.salience[(*c as usize).min(7)])).collect()
    }

    /// The technique as water, by its confidence: in `[−1, 1]`.
    pub fn wash(&self) -> Vec<f32> {
        self.class.iter().zip(&self.conf).map(|(c, p)| self.water[(*c as usize).min(7)] * p).collect()
    }

    /// The classes' shares of the frame, largest first, for the report line.
    pub fn summary(&self) -> String {
        let mut counts = [0usize; 8];
        for c in &self.class {
            counts[(*c as usize).min(7)] += 1;
        }
        let n = self.class.len().max(1);
        let mut parts: Vec<(usize, &str)> = CLASSES.iter().map(|c| (counts[c.index()] * 100 / n, c.name())).filter(|(p, _)| *p >= 1).collect();
        parts.sort_by(|a, b| b.0.cmp(&a.0));
        parts.iter().map(|(p, name)| format!("{name} {p}%")).collect::<Vec<_>>().join(" · ")
    }
}

/// The sight's maps, every one `w × h` row-major in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct SightMaps {
    pub w: u32,
    pub h: u32,
    /// The fused saliency — what the painter reads.
    pub saliency: Vec<f32>,
    /// U²-Net's salient-object matte (1 = the object), when it ran and found a usable object.
    pub object: Option<Vec<f32>>,
    /// Depth-Anything-V2's relative depth (1 = nearest), when it ran.
    pub depth: Option<Vec<f32>>,
    /// The near tier — what stands above the ground plane (`near_from_depth`), when the depth ran.
    pub near: Option<Vec<f32>>,
    /// The low-level cue: the centre-weighted contrast field (`painter::focal_field`).
    pub low: Vec<f32>,
    /// The painter's class of every pixel and its confidence (`collapse_classes`), when SegFormer ran.
    pub classes: Option<ClassMap>,
    /// The technique as water by class and confidence, in `[−1, 1]` (`ClassMap::wash`), when SegFormer ran.
    pub wash: Option<Vec<f32>>,
    /// The names of the cues that went into the fusion, for the report line.
    pub sources: Vec<&'static str>,
}

fn percentile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[((sorted.len() as f32 - 1.0) * q.clamp(0.0, 1.0)) as usize]
}

/// A salient-object matte is usable when the object is neither a speck nor the whole frame: its coverage above
/// one half is between 2% and 98% of the pixels (the matting path's own gate).
pub fn object_usable(mask: &[f32]) -> bool {
    let covered = mask.iter().filter(|&&m| m > 0.5).count();
    let total = mask.len().max(1);
    covered > total / 50 && covered < total * 49 / 50
}

/// The NEAR tier of a relative depth map (1 = nearest): the depth above its median, rescaled so the 90th
/// percentile is 1 — the foreground alone, the middle distance fading in, the far nothing.
pub fn near_from_depth(depth: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w == 0 || h == 0 || depth.len() != w * h {
        return vec![0.0; depth.len()];
    }
    // THE GROUND PLANE: each row's 25th percentile of depth (its far quarter), smoothed over a 32nd of the height —
    // the sea at a row is the plane, a hull at that row stands nearer than it. (The global median made the whole
    // foreground sea the near tier.) A row more than three quarters object keeps its object as the plane; the
    // matte has it.
    let plane_raw: Vec<f32> = (0..h).map(|y| percentile(&depth[y * w..(y + 1) * w], 0.25)).collect();
    let r = (h / 32).max(1);
    let plane: Vec<f32> = (0..h)
        .map(|y| {
            let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
            plane_raw[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32
        })
        .collect();
    // Rescaled to the 99th percentile of what stands above the plane (a small boat is a twentieth of the frame; the
    // 90th would be zero); under a fiftieth of the range nothing stands above it.
    let above: Vec<f32> = depth.iter().enumerate().map(|(i, d)| (d - plane[i / w]).max(0.0)).collect();
    let p99 = percentile(&above, 0.99);
    if p99 < 0.02 {
        return vec![0.0; depth.len()];
    }
    above.iter().map(|a| (a / p99).clamp(0.0, 1.0)).collect()
}

/// The fusion: the noisy-OR of the weighted cues, normalised by the 99th percentile, a face salient outright.
/// Every map `w × h`; a cue of the wrong length is ignored.
pub fn fuse(w: u32, h: u32, object: Option<&[f32]>, near: Option<&[f32]>, face: Option<&[f32]>, low: &[f32]) -> Vec<f32> {
    fuse_with_class(w, h, object, near, face, low, None, None)
}

/// The fusion with the class map: the class cue (`ClassMap::cue`) one more cue in the noisy-OR, the class damping
/// (`ClassMap::damp`) multiplied into the fused map before its normalisation.
pub fn fuse_with_class(w: u32, h: u32, object: Option<&[f32]>, near: Option<&[f32]>, face: Option<&[f32]>, low: &[f32], class_cue: Option<&[f32]>, class_damp: Option<&[f32]>) -> Vec<f32> {
    let n = w as usize * h as usize;
    fn fit<'a>(m: Option<&'a [f32]>, n: usize) -> Option<&'a [f32]> {
        m.filter(|m| m.len() == n)
    }
    let (object, near, face) = (fit(object, n), fit(near, n), fit(face, n));
    let (class_cue, class_damp) = (fit(class_cue, n), fit(class_damp, n));
    let low_ok = low.len() == n;
    let mut s: Vec<f32> = (0..n)
        .map(|i| {
            let mut keep = 1.0f32;
            // Each cue as its confidence — squared: a half-sure cue counts a quarter.
            let conf = |v: f32| { let v = v.clamp(0.0, 1.0); v * v };
            if let Some(o) = object {
                keep *= 1.0 - W_OBJECT * conf(o[i]);
            }
            if let Some(d) = near {
                keep *= 1.0 - W_NEAR * conf(d[i]);
            }
            if low_ok {
                keep *= 1.0 - W_LOW * conf(low[i]);
            }
            if let Some(c) = class_cue {
                keep *= 1.0 - W_CLASS * conf(c[i]);
            }
            let s = 1.0 - keep;
            match class_damp {
                Some(d) => s * d[i].clamp(0.0, 1.0),
                None => s,
            }
        })
        .collect();
    let hi = percentile(&s, 0.99).max(1e-3);
    for v in &mut s {
        *v = (*v / hi).clamp(0.0, 1.0);
    }
    if let Some(f) = face {
        for (v, f) in s.iter_mut().zip(f) {
            *v = v.max(f.clamp(0.0, 1.0));
        }
    }
    s
}

/// The technique yielding to the saliency: the wash × (1 − S) — a salient pixel takes the brush as it is.
pub fn yield_to_saliency(wash: &[f32], saliency: &[f32]) -> Vec<f32> {
    if wash.len() != saliency.len() {
        return wash.to_vec();
    }
    wash.iter().zip(saliency).map(|(v, s)| v * (1.0 - s.clamp(0.0, 1.0))).collect()
}

/// The low-level cue: the centre-weighted contrast field of the picture itself.
pub fn low_level(img: &RgbImage) -> Vec<f32> {
    crate::paint::painter::focal_field(img)
}

fn to_gray(v: &[f32], w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| image::Luma([(v[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0).round() as u8]))
}

impl SightMaps {
    /// The share of the frame the fused map calls salient (above one half), in percent.
    pub fn salient_percent(&self) -> usize {
        self.saliency.iter().filter(|&&v| v >= 0.5).count() * 100 / self.saliency.len().max(1)
    }

    /// Write the maps as grey PNGs into `dir` (`sight_saliency.png`, `sight_object.png`, `sight_depth.png`,
    /// `sight_low.png`) — the review's montage.
    pub fn dump(&self, dir: &Path) {
        let _ = to_gray(&self.saliency, self.w, self.h).save(dir.join("sight_saliency.png"));
        let _ = to_gray(&self.low, self.w, self.h).save(dir.join("sight_low.png"));
        if let Some(o) = &self.object {
            let _ = to_gray(o, self.w, self.h).save(dir.join("sight_object.png"));
        }
        if let Some(d) = &self.depth {
            let _ = to_gray(d, self.w, self.h).save(dir.join("sight_depth.png"));
        }
        if let Some(d) = &self.near {
            let _ = to_gray(d, self.w, self.h).save(dir.join("sight_near.png"));
        }
        if let Some(c) = &self.classes {
            let img = RgbImage::from_fn(self.w, self.h, |x, y| {
                let i = (y * self.w + x) as usize;
                let col = CLASSES[(c.class[i] as usize).min(7)].colour();
                let p = c.conf[i].clamp(0.0, 1.0);
                image::Rgb([(col[0] as f32 * p) as u8, (col[1] as f32 * p) as u8, (col[2] as f32 * p) as u8])
            });
            let _ = img.save(dir.join("sight_class.png"));
        }
        if let Some(m) = &self.wash {
            let half: Vec<f32> = m.iter().map(|v| 0.5 + 0.5 * v).collect();
            let _ = to_gray(&half, self.w, self.h).save(dir.join("sight_wash.png"));
        }
    }
}

/// Build the sight for a picture: the models run on the auto device, each cue optional — a model that fails to
/// load or finds nothing usable is left out with a note, and the fusion goes on with the cues that came. `img`
/// is the picture at the paint size `w × h`; `input` its path (the depth pipeline reads the file); `face` the
/// painter's face mask when one was found; `class_k` the key `aqua_sight_class` — above 0 SegFormer runs and the
/// class map enters the fusion by it; `tune` the plan's class table (`ClassMap::tune`). Fails only when not even the
/// low-level cue can be built.
pub async fn build(input: &Path, img: &RgbImage, face: Option<&[f32]>, class_k: f32, tune: &[(String, Option<f32>, Option<f32>)]) -> Result<(SightMaps, Vec<String>)> {
    let (w, h) = img.dimensions();
    let n = w as usize * h as usize;
    let mut notes: Vec<String> = Vec::new();
    let mut sources: Vec<&'static str> = Vec::new();
    let device = crate::device::select("auto").context("selecting the device for the sight")?;
    let object: Option<Vec<f32>> = match crate::pipelines::matting::Matter::load(&device).await {
        Ok(matter) => match matter.matte(img) {
            Ok(alpha) => {
                let m: Vec<f32> = alpha.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
                if m.len() == n && object_usable(&m) {
                    sources.push("matte");
                    Some(m)
                } else {
                    notes.push("the matte found no usable object (under 2% or over 98% of the frame)".into());
                    None
                }
            }
            Err(e) => {
                notes.push(format!("the matte failed ({e})"));
                None
            }
        },
        Err(e) => {
            notes.push(format!("U²-Net did not load ({e})"));
            None
        }
    };
    let depth: Option<Vec<f32>> = match crate::pipelines::depth::DepthPipeline::load(device.clone()).await {
        Ok(pipe) => match pipe.depth_map(input, w, h) {
            Ok(d) if d.len() == n => {
                sources.push("depth");
                Some(d)
            }
            Ok(_) => {
                notes.push("the depth map came at the wrong size".into());
                None
            }
            Err(e) => {
                notes.push(format!("the depth failed ({e})"));
                None
            }
        },
        Err(e) => {
            notes.push(format!("Depth-Anything did not load ({e})"));
            None
        }
    };
    let near = depth.as_deref().map(|d| near_from_depth(d, w as usize, h as usize));
    // THE CLASSES: SegFormer on the device, and on the CPU should the device refuse it (the small model is cheap).
    let classes: Option<ClassMap> = if class_k > 0.0 {
        let run = |seg: &crate::pipelines::segformer::Segmenter| -> Result<ClassMap> {
            let out = seg.segment(img)?;
            collapse_classes_guided(&out.probs, seg.labels(), out.gw, out.gh, img).context("the class map came at the wrong size")
        };
        let first = match crate::pipelines::segformer::Segmenter::load(&device).await {
            Ok(seg) => run(&seg),
            Err(e) => Err(e),
        };
        match first {
            Ok(c) => Some(c),
            Err(e) if !device.is_cpu() => {
                notes.push(format!("SegFormer on the device failed ({e}) — on the CPU"));
                match crate::pipelines::segformer::Segmenter::load(&candle_core::Device::Cpu).await {
                    Ok(seg) => match run(&seg) {
                        Ok(c) => Some(c),
                        Err(e) => {
                            notes.push(format!("SegFormer failed ({e})"));
                            None
                        }
                    },
                    Err(e) => {
                        notes.push(format!("SegFormer did not load ({e})"));
                        None
                    }
                }
            }
            Err(e) => {
                notes.push(format!("SegFormer failed ({e})"));
                None
            }
        }
    } else {
        None
    };
    let classes = classes.map(|mut c| {
        c.tune(tune);
        c
    });
    if classes.is_some() {
        sources.push("classes");
    }
    let low = low_level(img);
    if low.len() != n {
        anyhow::bail!("the contrast field came at the wrong size");
    }
    sources.push("contrast");
    let face = face.filter(|f| f.len() == n && f.iter().any(|v| *v > 0.35));
    if face.is_some() {
        sources.push("faces");
    }
    let (cue, damp, wash) = match &classes {
        Some(c) => (Some(c.cue()), Some(c.damp(class_k)), Some(c.wash())),
        None => (None, None, None),
    };
    let saliency = fuse_with_class(w, h, object.as_deref(), near.as_deref(), face, &low, cue.as_deref(), damp.as_deref());
    // The technique yields to the saliency: what the sight looks at is painted as the brush is, whatever its class.
    let wash = wash.map(|m| yield_to_saliency(&m, &saliency));
    Ok((SightMaps { w, h, saliency, object, depth, near, low, classes, wash, sources }, notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(w: u32, h: u32, x0: u32, x1: u32) -> Vec<f32> {
        (0..w * h).map(|i| if (i % w) >= x0 && (i % w) < x1 { 1.0 } else { 0.0 }).collect()
    }

    #[test]
    fn the_matte_makes_its_object_salient_and_the_rest_dim() {
        let (w, h) = (40u32, 20u32);
        let low = vec![0.2f32; (w * h) as usize];
        let object = block(w, h, 10, 20);
        let s = fuse(w, h, Some(&object), None, None, &low);
        assert_eq!(s.len(), (w * h) as usize);
        assert!(s.iter().all(|v| (0.0..=1.0).contains(v)), "in [0, 1]");
        let inside = s[(5 * w + 15) as usize];
        let outside = s[(5 * w + 30) as usize];
        assert!(inside >= 0.99, "the object salient: {inside}");
        assert!(outside < 0.15, "the ground dim: {outside}");
    }

    #[test]
    fn the_cues_never_cancel_and_a_face_is_salient_outright() {
        let (w, h) = (40u32, 20u32);
        let low: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let near = block(w, h, 0, 10);
        let alone = fuse(w, h, None, None, None, &low);
        let with_near = fuse(w, h, None, Some(&near), None, &low);
        // The normalisation rescales the whole map, so a cue is judged by the ORDER it makes: the near pixel rises
        // against a far one, and the far pixels keep their order among themselves.
        let (i_in, i_out, j_out) = ((3 * w + 2) as usize, (3 * w + 20) as usize, (3 * w + 35) as usize);
        assert!(with_near[i_in] - with_near[i_out] > alone[i_in] - alone[i_out], "the near tier lifts its pixels against the rest");
        assert!(with_near[i_in] > alone[i_in], "…and above where it stood");
        assert!(with_near[j_out] > with_near[i_out], "the far pixels keep their order");
        let face = block(w, h, 30, 32);
        let with_face = fuse(w, h, None, None, Some(&face), &low);
        assert!(with_face[(3 * w + 31) as usize] >= 1.0 - 1e-6, "a face is salient outright");
    }

    #[test]
    fn the_near_tier_is_what_stands_above_the_ground_plane() {
        // A sea receding row by row (the depth rising down the frame, flat across a row) is the plane everywhere:
        // no pixel of it is near, however close its row. A hull standing nearer than its rows is the near tier.
        let (w, h) = (40usize, 40usize);
        let sea: Vec<f32> = (0..w * h).map(|i| (i / w) as f32 / (h - 1) as f32).collect();
        let near = near_from_depth(&sea, w, h);
        assert!(near.iter().all(|v| *v == 0.0), "the plane alone is nowhere near");
        let hull: Vec<f32> = sea.iter().enumerate().map(|(i, d)| if (i % w) >= 10 && (i % w) < 25 && (i / w) >= 20 && (i / w) < 28 { d + 0.3 } else { *d }).collect();
        let near = near_from_depth(&hull, w, h);
        assert!(near[24 * w + 15] > 0.9, "the hull stands above its row's plane: {}", near[24 * w + 15]);
        assert!(near[24 * w + 35] < 1e-3 && near[35 * w + 15] < 1e-3, "the sea beside and below it is the plane: {} {}", near[24 * w + 35], near[35 * w + 15]);
        assert!(near_from_depth(&sea, 7, 3).iter().all(|v| *v == 0.0), "a map of the wrong size is nowhere near");
        assert!(object_usable(&block(40, 20, 10, 20)));
        assert!(!object_usable(&block(40, 20, 0, 40)), "the whole frame is no object");
        assert!(!object_usable(&vec![0.0; 800]), "nothing is no object");
    }

    #[test]
    fn the_painters_table_folds_ades_labels_into_eight_techniques() {
        use PainterClass::*;
        for (label, want) in [
            ("sky", Sky), ("sea", Water), ("water", Water), ("river", Water), ("swimming pool", Water),
            ("pool table", Thing), ("grass", Ground), ("road", Ground), ("mountain", Ground), ("tree", Foliage),
            ("plant", Foliage), ("wall", Built), ("building", Built), ("skyscraper", Built), ("house", Built),
            ("person", Person), ("animal", Animal), ("boat", Thing), ("ship", Thing), ("car", Thing), (" bed ", Thing),
            ("Building;edifice", Built), ("zeppelin", Thing),
        ] {
            assert_eq!(painter_class(label), want, "{label}");
        }
        assert!(CLASSES.iter().enumerate().all(|(i, c)| c.index() == i), "the index order");
        assert!(CLASSES.iter().all(|c| (0.0..=1.0).contains(&c.salience()) && (-1.0..=1.0).contains(&c.water())));
    }

    #[test]
    fn the_class_map_is_the_summed_probability_at_the_paint_size_and_enters_the_fusion_twice() {
        // Two ADE labels on a 2 × 1 grid: the left cell sea 0.9 / boat 0.1, the right cell sea 0.2 / boat 0.8.
        let labels: Vec<String> = vec!["sea".into(), "boat".into()];
        let probs = vec![0.9f32, 0.2, 0.1, 0.8];
        let cm = collapse_classes(&probs, &labels, 2, 1, 40, 10).expect("a map");
        assert_eq!(cm.class.len(), 400);
        assert_eq!(cm.class[5], PainterClass::Water.index() as u8, "the left is water");
        assert_eq!(cm.class[35], PainterClass::Thing.index() as u8, "the right is a thing");
        assert!(cm.conf[5] > 0.85 && cm.conf[35] > 0.75, "the confidences: {} {}", cm.conf[5], cm.conf[35]);
        assert!(collapse_classes(&probs, &labels, 3, 1, 40, 10).is_none(), "a grid the probabilities do not fill is no map");
        let summary = cm.summary();
        assert!(summary.starts_with("water 5") || summary.starts_with("thing 5"), "{summary}");
        // The fusion: a busy sea (contrast 0.8) with no other cue drops with the class damping; a quiet thing (contrast
        // 0.1) rises with the class cue — the thing ends above the sea, where the contrast alone had it below.
        let (w, h) = (40u32, 10u32);
        let low: Vec<f32> = (0..w * h).map(|i| if (i % w) < 20 { 0.8 } else { 0.1 }).collect();
        let alone = fuse(w, h, None, None, None, &low);
        assert!(alone[5] > alone[35], "the contrast alone has the sea above the thing");
        let cue = cm.cue();
        let damp = cm.damp(1.0);
        let classed = fuse_with_class(w, h, None, None, None, &low, Some(&cue), Some(&damp));
        assert!(classed[35] > classed[5], "with the classes the thing is above the sea: {} vs {}", classed[35], classed[5]);
        assert!(classed[5] < 0.4, "the sea damped: {}", classed[5]);
        let undamped = fuse_with_class(w, h, None, None, None, &low, Some(&cue), Some(&cm.damp(0.0)));
        assert!(undamped[5] > classed[5], "the damping at the key 0 is none");
        let wash = cm.wash();
        assert!(wash[5] > 0.8 && wash[35].abs() < 1e-6, "the sea a wet wash, the thing the brush as it is: {} {}", wash[5], wash[35]);
    }

    #[test]
    fn the_guided_class_map_snaps_to_the_pictures_edges() {
        // Two cells, sea | boat, on a 40 × 8 picture whose colour edge stands at x = 14, six pixels before the cells'
        // boundary at 20: the plain resampling puts the boundary at the cells', the guided one at the picture's.
        let labels: Vec<String> = vec!["sea".into(), "boat".into()];
        let probs = vec![1.0f32, 0.0, 0.0, 1.0];
        let img = RgbImage::from_fn(40, 8, |x, _| if x < 14 { image::Rgb([90, 120, 170]) } else { image::Rgb([80, 50, 30]) });
        let plain = collapse_classes(&probs, &labels, 2, 1, 40, 8).unwrap();
        let guided = collapse_classes_guided(&probs, &labels, 2, 1, &img).unwrap();
        let water = PainterClass::Water.index() as u8;
        let thing = PainterClass::Thing.index() as u8;
        assert_eq!(plain.class[4 * 40 + 17], water, "the plain map keeps the sea to the cells' boundary");
        assert_eq!(guided.class[4 * 40 + 17], thing, "the guided map follows the colour edge: {:?}", &guided.class[160..200]);
        assert_eq!(guided.class[4 * 40 + 12], water);
        assert_eq!(guided.class[4 * 40 + 2], water);
        assert_eq!(guided.class[4 * 40 + 37], thing);
        assert!(guided.conf[4 * 40 + 2] > 0.9 && guided.conf[4 * 40 + 37] > 0.9, "sure far from the edge: {} {}", guided.conf[4 * 40 + 2], guided.conf[4 * 40 + 37]);
        assert!(collapse_classes_guided(&probs, &labels, 3, 1, &img).is_none());
    }

    #[test]
    fn the_class_table_is_the_plans_to_tune_and_the_technique_yields_to_the_saliency() {
        let labels: Vec<String> = vec!["sky".into(), "boat".into()];
        let probs = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut cm = collapse_classes(&probs, &labels, 2, 1, 4, 1).expect("a map");
        assert!(cm.wash()[0] > 0.99 && cm.cue()[0] < 0.1, "the sky a wet wash of no salience by default");
        cm.tune(&[("sky".into(), Some(0.9), Some(-1.0)), ("Thing".into(), None, Some(2.0)), ("dragon".into(), Some(1.0), Some(1.0))]);
        assert!(cm.wash()[0] < -0.99 && cm.cue()[0] > 0.89, "the sky tuned dry and salient: {} {}", cm.wash()[0], cm.cue()[0]);
        assert!(cm.wash()[3] > 0.99, "the thing's water clamped to 1: {}", cm.wash()[3]);
        assert!((cm.cue()[3] - PainterClass::Thing.salience()).abs() < 1e-6, "the thing's salience untouched");
        let wash = vec![1.0f32, -1.0, 0.5, 0.0];
        let sal = vec![0.0f32, 1.0, 0.5, 1.0];
        let y = yield_to_saliency(&wash, &sal);
        assert!((y[0] - 1.0).abs() < 1e-6 && y[1].abs() < 1e-6 && (y[2] - 0.25).abs() < 1e-6, "{y:?}");
        assert_eq!(yield_to_saliency(&wash, &sal[..2]), wash, "a saliency of the wrong length changes nothing");
    }

    #[test]
    fn a_cue_of_the_wrong_length_is_ignored() {
        let (w, h) = (8u32, 4u32);
        let low = vec![0.5f32; 32];
        let wrong = vec![1.0f32; 7];
        assert_eq!(fuse(w, h, Some(&wrong), Some(&wrong), Some(&wrong), &low), fuse(w, h, None, None, None, &low));
    }
}
