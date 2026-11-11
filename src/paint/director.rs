//! THE ART DIRECTOR WITH EYES (7.19.0) — `plakat paint from <img> --medium M --make-plan` and
//! `plakat paint plan <img> --medium M --director`.
//!
//! The v1 art director ([`super::plan::plan_from`]) reads a face count and two tonal numbers and writes the
//! STRUCTURAL keys of a plan; every dial of the medium it leaves to the medium's stock profile. The verification
//! cycles (7.5 → 7.18) settled, medium by medium, a HOUSE RECIPE — the plan the reviews accepted — and the rule
//! that a medium is tuned in its plan, never in the source. This module puts the two together and reads the
//! picture first:
//!
//! 1. **The recipe** (`assets/recipes/<medium>.hjson`): the medium's accepted plan, shipped with the binary as a
//!    plan, not as code. A key the recipe names is a dial the medium USES; the director moves no dial the
//!    medium does not use, and turns on none the recipe holds at 0 (a watercolour's `hotspot: 0` is the
//!    medium's decision, not a gap).
//! 2. **The facts** ([`SceneFacts`]): what the picture is — measured from its pixels (key, contrast, flat
//!    fields, chroma, points of light, structure) and read by the vision models already in the crate (the
//!    faces by SCRFD, the subject's matte by U²-Net, the far ground by Depth-Anything, the genre and the stuff
//!    — sky, water, foliage, stone — by CLIP's zero-shot over the sheet and its crops). Every model fact is
//!    optional: a model that cannot run leaves its fact unknown and the rule that needs it is skipped.
//! 3. **The rules** ([`rules`]): fact → dial, each universal (a nocturne is a nocturne whatever the picture),
//!    each written into the plan with its reason and its measurement, so the plan is read, argued with and
//!    edited like any other. Nothing scene-specific is baked in: a key without a fact behind it is not moved.
//! 4. **The model** (optional, `--make-plan-model <provider>`): a vision model sees the picture, the facts,
//!    the composed plan and the dial glossary, and proposes dial lines in the insights' own form
//!    (`key: value   # reason`). Its lines pass the same gate as the rules ([`accept_model_lines`]): only the
//!    medium's dials, never the structure, never the faces, values clamped — and are written as its own section.
//!
//! The painter never sees any of this: the director writes a plan file and `paint from` reads it as it reads
//! any `--plan <file>`. Without the flag not a byte of any medium changes.

use std::collections::BTreeMap;

use image::RgbImage;

use super::plan::{plan_from, Analysis, PaintPlan};

// ───────────────────────────────────────────── the recipes ──────────────────────────────────────────────

/// The house recipes, one per executable medium (`assets/recipes/<medium>.hjson`).
const RECIPES: &[(&str, &str)] = &[
    ("oil-direct", include_str!("../../assets/recipes/oil-direct.hjson")),
    ("oil-indirect", include_str!("../../assets/recipes/oil-indirect.hjson")),
    ("gouache", include_str!("../../assets/recipes/gouache.hjson")),
    ("watercolour", include_str!("../../assets/recipes/watercolour.hjson")),
    ("line-and-wash", include_str!("../../assets/recipes/line-and-wash.hjson")),
    ("early-book-illustration", include_str!("../../assets/recipes/early-book-illustration.hjson")),
    ("ink-wash", include_str!("../../assets/recipes/ink-wash.hjson")),
    ("japanese-ink", include_str!("../../assets/recipes/japanese-ink.hjson")),
    ("pen-ink", include_str!("../../assets/recipes/pen-ink.hjson")),
    ("durer", include_str!("../../assets/recipes/durer.hjson")),
    ("tempera", include_str!("../../assets/recipes/tempera.hjson")),
    ("pencil", include_str!("../../assets/recipes/pencil.hjson")),
    ("black-pencil", include_str!("../../assets/recipes/black-pencil.hjson")),
    ("pastel", include_str!("../../assets/recipes/pastel.hjson")),
    ("charcoal", include_str!("../../assets/recipes/charcoal.hjson")),
    ("acrylic", include_str!("../../assets/recipes/acrylic.hjson")),
    ("palette-knife", include_str!("../../assets/recipes/palette-knife.hjson")),
    ("pointillism", include_str!("../../assets/recipes/pointillism.hjson")),
];

/// The house recipe of a medium — its accepted plan text — or `None` for a name no recipe knows.
pub fn recipe_for(medium: &str) -> Option<&'static str> {
    let n = medium.trim();
    let n = if n.eq_ignore_ascii_case("watercolor") { "watercolour" } else { n };
    RECIPES.iter().find(|(m, _)| m.eq_ignore_ascii_case(n)).map(|(_, t)| *t)
}

// ─────────────────────────────────────────── the plan as lines ──────────────────────────────────────────

/// One line of a plan: a comment, a blank, or `key: value   # comment`.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    Comment(String),
    Blank,
    Key { key: String, value: String, comment: Option<String> },
}

/// Read a plan's text as lines. Braces (an HJSON object's) are dropped; a value runs to the first ` #` that
/// is not inside brackets or quotes, so an inline array or a quoted path keeps its `#`.
pub fn parse_entries(text: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        let s = line.trim_start();
        if s.is_empty() {
            out.push(Entry::Blank);
            continue;
        }
        if s == "{" || s == "}" {
            continue;
        }
        if s.starts_with('#') {
            out.push(Entry::Comment(s.trim_start_matches('#').trim_start().to_string()));
            continue;
        }
        let Some(colon) = s.find(':') else {
            out.push(Entry::Comment(s.to_string()));
            continue;
        };
        let key = s[..colon].trim();
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.push(Entry::Comment(s.to_string()));
            continue;
        }
        let rest = &s[colon + 1..];
        // The value ends at a `#` outside brackets and quotes.
        let (mut depth, mut quote, mut cut) = (0i32, false, rest.len());
        for (i, c) in rest.char_indices() {
            match c {
                '"' => quote = !quote,
                '[' | '{' if !quote => depth += 1,
                ']' | '}' if !quote => depth -= 1,
                '#' if !quote && depth <= 0 => {
                    cut = i;
                    break;
                }
                _ => {}
            }
        }
        let value = rest[..cut].trim().to_string();
        let comment = rest[cut..].trim_start_matches('#').trim();
        out.push(Entry::Key { key: key.to_string(), value, comment: (!comment.is_empty()).then(|| comment.to_string()) });
    }
    out
}

/// Write lines back as plan text.
pub fn entries_text(entries: &[Entry]) -> String {
    let mut o = String::new();
    for e in entries {
        match e {
            Entry::Blank => o.push('\n'),
            Entry::Comment(c) => {
                o.push_str("# ");
                o.push_str(c);
                o.push('\n');
            }
            Entry::Key { key, value, comment } => {
                o.push_str(key);
                o.push_str(": ");
                o.push_str(value);
                if let Some(c) = comment {
                    o.push_str("   # ");
                    o.push_str(c);
                }
                o.push('\n');
            }
        }
    }
    o
}

/// The keys a list of lines names, with their values (the last one wins, as a plan would read it).
pub fn values_of(entries: &[Entry]) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for e in entries {
        if let Entry::Key { key, value, .. } = e {
            m.insert(key.clone(), value.clone());
        }
    }
    m
}

fn parse_num(v: &str) -> Option<f32> {
    v.trim().parse::<f32>().ok()
}

// ────────────────────────────────────────────── the facts ───────────────────────────────────────────────

/// What the picture IS, as CLIP's zero-shot reads it over the whole sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Genre {
    Portrait,
    Figure,
    Landscape,
    Townscape,
    StillLife,
    Interior,
    Animal,
    Other,
}

impl Genre {
    /// The prompts CLIP scores the sheet against, one per genre.
    pub const LABELS: &'static [(Genre, &'static str)] = &[
        (Genre::Portrait, "a portrait photograph of a person's face"),
        (Genre::Figure, "a photograph of people, full figure"),
        (Genre::Landscape, "a landscape photograph of nature"),
        (Genre::Townscape, "a photograph of a town street with buildings"),
        (Genre::StillLife, "a still life photograph of objects on a table"),
        (Genre::Interior, "a photograph of a room's interior"),
        (Genre::Animal, "a photograph of an animal"),
        (Genre::Other, "an abstract pattern"),
    ];

    pub fn name(self) -> &'static str {
        match self {
            Genre::Portrait => "portrait",
            Genre::Figure => "figure",
            Genre::Landscape => "landscape",
            Genre::Townscape => "townscape",
            Genre::StillLife => "still life",
            Genre::Interior => "interior",
            Genre::Animal => "animal",
            Genre::Other => "other",
        }
    }

    /// The genre from CLIP's cosine scores (one per label, any scale): a softmax at CLIP's scale of 100 and the
    /// winner with its share.
    pub fn from_scores(scores: &[(Genre, f32)]) -> Option<(Genre, f32)> {
        if scores.is_empty() {
            return None;
        }
        let m = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
        let ex: Vec<f32> = scores.iter().map(|s| (100.0 * (s.1 - m)).exp()).collect();
        let sum: f32 = ex.iter().sum();
        let best = (0..ex.len()).max_by(|a, b| ex[*a].total_cmp(&ex[*b]))?;
        Some((scores[best].0, ex[best] / sum.max(1e-6)))
    }
}

/// The STUFF a crop of the picture can be, for the shares the rules read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stuff {
    Sky,
    Water,
    Foliage,
    Stone,
    Ground,
    Person,
    Other,
}

/// The prompts CLIP scores each crop against, with the stuff each one counts toward.
pub const STUFF_LABELS: &[(Stuff, &str)] = &[
    (Stuff::Sky, "the sky"),
    (Stuff::Sky, "clouds"),
    (Stuff::Water, "water"),
    (Stuff::Water, "the sea"),
    (Stuff::Water, "a river"),
    (Stuff::Foliage, "trees"),
    (Stuff::Foliage, "a forest"),
    (Stuff::Foliage, "grass"),
    (Stuff::Foliage, "bushes"),
    (Stuff::Stone, "a stone wall"),
    (Stuff::Stone, "rocks"),
    (Stuff::Stone, "a building"),
    (Stuff::Stone, "a cobbled street"),
    (Stuff::Ground, "a dirt road"),
    (Stuff::Ground, "sand"),
    (Stuff::Ground, "snow"),
    (Stuff::Ground, "a wooden floor"),
    (Stuff::Person, "a person"),
    (Stuff::Person, "clothing"),
    (Stuff::Person, "a face"),
    (Stuff::Other, "a table"),
    (Stuff::Other, "a window"),
    (Stuff::Other, "an animal's fur"),
    (Stuff::Other, "books on shelves"),
];

/// A crop's stuff shares from its cosine scores against [`STUFF_LABELS`] (softmax at CLIP's scale of 100,
/// summed per stuff). Returns `(sky, water, foliage, stone)`.
pub fn stuff_shares(scores: &[f32]) -> (f32, f32, f32, f32) {
    if scores.len() != STUFF_LABELS.len() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let m = scores.iter().copied().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = scores.iter().map(|s| (100.0 * (s - m)).exp()).collect();
    let sum: f32 = ex.iter().sum::<f32>().max(1e-6);
    let share = |what: Stuff| STUFF_LABELS.iter().zip(&ex).filter(|((s, _), _)| *s == what).map(|(_, e)| e).sum::<f32>() / sum;
    (share(Stuff::Sky), share(Stuff::Water), share(Stuff::Foliage), share(Stuff::Stone))
}

/// The local standard deviation of a field over a square window of half-width `r` (box sums of the value and
/// its square; the window clipped at the borders).
pub fn local_std(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let (w1, h1) = (w + 1, h + 1);
    let mut s = vec![0f64; w1 * h1];
    let mut q = vec![0f64; w1 * h1];
    for y in 0..h {
        let (mut rs, mut rq) = (0f64, 0f64);
        for x in 0..w {
            let val = v[y * w + x] as f64;
            rs += val;
            rq += val * val;
            s[(y + 1) * w1 + x + 1] = s[y * w1 + x + 1] + rs;
            q[(y + 1) * w1 + x + 1] = q[y * w1 + x + 1] + rq;
        }
    }
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(r), (x + r + 1).min(w));
            let cnt = ((y1 - y0) * (x1 - x0)) as f64;
            let sum = s[y1 * w1 + x1] - s[y0 * w1 + x1] - s[y1 * w1 + x0] + s[y0 * w1 + x0];
            let sq = q[y1 * w1 + x1] - q[y0 * w1 + x1] - q[y1 * w1 + x0] + q[y0 * w1 + x0];
            let mean = sum / cnt;
            out[y * w + x] = (sq / cnt - mean * mean).max(0.0).sqrt() as f32;
        }
    }
    out
}

/// The facts of a picture. The pixel facts are always known; a model's fact is `None` when the model could
/// not run (no weights, no device) and the rules that need it stand down.
#[derive(Clone, Debug, Default)]
pub struct SceneFacts {
    pub width: u32,
    pub height: u32,
    /// Mean luma in [0,1] (a nocturne sits under 0.3; a high key over 0.72).
    pub luma_mean: f32,
    /// Luma σ (under 0.12 a flat, foggy picture).
    pub luma_stddev: f32,
    /// The share of pixels under 0.12 of luma.
    pub dark_share: f32,
    /// The share of pixels over 0.88 of luma.
    pub light_share: f32,
    /// σ of the luma's 8-neighbour Laplacian (`plan::structure_of`): how much pixel-scale detail there is.
    pub structure: f32,
    /// The share of the sheet whose local luma σ (a 9×9 window at 1024 px) is under 0.02 — the calm fields: a
    /// sky, a wall, a still water (a night sea 0.48, a storm 0.54, a street 0.13 on the verification set).
    pub flat_share: f32,
    /// The share whose local σ is over 0.08 — the busy passages: foliage, rigging, cobbles, a crowd (a street
    /// 0.47, a snowy valley 0.38, a night sea 0.13).
    pub busy_share: f32,
    /// Mean chroma (the sRGB max−min, in [0,1]): under 0.06 near-monochrome, over 0.22 a wide palette.
    pub chroma_mean: f32,
    /// The share of chromatic pixels that are warm (red–yellow) rather than cool.
    pub warm_share: f32,
    /// Isolated points of light on a dark ground (`painter::point_lights`, the ground under 0.25 of luma) per
    /// megapixel of the sheet — the stars, the lamps, the glints on a night sea (268 on the verification set's
    /// night sea, 168 on its lamplit harbour, 40–70 on its daylight pictures).
    pub points_per_mpx: f32,
    /// Faces found (SCRFD, the animal boxes dropped). 0 when the detector did not run.
    pub faces: usize,
    /// The faces' share of the sheet (the focal mask's mean), 0 without a face.
    pub face_share: f32,
    /// The salient subject's share of the sheet (the U²-Net matte over 0.5), when the matter ran.
    pub subject_share: Option<f32>,
    /// The subject's centroid in sheet fractions, when the matter ran and found one.
    pub subject_centre: Option<(f32, f32)>,
    /// The far ground's share (Depth-Anything's RELATIVE depth under 0.35 — the nearest thing at 1, the farthest
    /// at 0; a portrait's wall is far as a landscape's distance is), when it ran.
    pub far_share: Option<f32>,
    /// The genre with CLIP's confidence in it, when CLIP ran.
    pub genre: Option<(Genre, f32)>,
    /// The stuff shares over the sheet's crops (CLIP), when CLIP ran.
    pub sky_share: Option<f32>,
    pub water_share: Option<f32>,
    pub foliage_share: Option<f32>,
    pub stone_share: Option<f32>,
    /// What each model reported (ran, or why not) — for the plan's header and the terminal.
    pub sources: Vec<String>,
}

impl SceneFacts {
    /// The pixel facts of an image (the model facts left unknown). The image may be a reduced copy: every
    /// fact here is a share or a density, scale-free; `structure` is NOT set here (the analysis measures it on
    /// the picture at its own size — `with_structure`).
    pub fn from_pixels(img: &RgbImage) -> SceneFacts {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let n = (w * h).max(1) as f32;
        let luma: Vec<f32> = img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).collect();
        let mean = luma.iter().sum::<f32>() / n;
        let sd = (luma.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / n).sqrt();
        let dark = luma.iter().filter(|l| **l < 0.12).count() as f32 / n;
        let light = luma.iter().filter(|l| **l > 0.88).count() as f32 / n;
        // The calm and the busy: the LOCAL σ of the luma over a window of a 128th of the long side (9×9 at
        // 1024 — the review cycles' own measure; a max−min range reads a photograph's grain as structure).
        let (mut flat, mut busy) = (0.0, 0.0);
        if w >= 4 && h >= 4 {
            let r = ((w.max(h) as f32 / 256.0).round() as usize).clamp(2, 32);
            let sd = local_std(&luma, w, h, r);
            flat = sd.iter().filter(|v| **v < 0.02).count() as f32 / n;
            busy = sd.iter().filter(|v| **v > 0.08).count() as f32 / n;
        }
        // Chroma: the sRGB max−min; warmth by the hue of the chromatic pixels.
        let (mut csum, mut warm, mut chromatic) = (0.0f32, 0usize, 0usize);
        for p in img.pixels() {
            let [r, g, b] = p.0;
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            let c = (mx - mn) as f32 / 255.0;
            csum += c;
            if c > 0.1 {
                chromatic += 1;
                if r >= b && (r as u16 + g as u16) > 2 * b as u16 {
                    warm += 1;
                }
            }
        }
        // The points of light: the point pass's own detector (a lone speck), counting only the lights ON A DARK
        // GROUND (the ground's luma under 0.25) — a star, a lamp, a glint on a night sea. A daylight photograph
        // is full of bright specks on a mid ground (a street 97 per Mpx, the night sea 268 before the filter).
        let points = if w >= 8 && h >= 8 {
            let r = ((w.min(h) as f32) / 512.0).round().clamp(2.0, 6.0) as i32;
            let lin = crate::paint::painter::luma_map(img);
            crate::paint::painter::point_lights(&lin, w as u32, h as u32, r, 0.02).iter().filter(|p| p.ground < 0.25).count()
        } else {
            0
        };
        SceneFacts {
            width: img.width(),
            height: img.height(),
            luma_mean: mean,
            luma_stddev: sd,
            dark_share: dark,
            light_share: light,
            structure: 0.0,
            flat_share: flat,
            busy_share: busy,
            chroma_mean: csum / n,
            warm_share: if chromatic > 0 { warm as f32 / chromatic as f32 } else { 0.0 },
            points_per_mpx: points as f32 / (n / 1.0e6),
            ..SceneFacts::default()
        }
    }

    /// Whether the picture is a nocturne (mean luma under 0.3 — the same line `plan_from` draws).
    pub fn nocturne(&self) -> bool {
        self.luma_mean < 0.3
    }

    /// The facts as lines for the plan's header, the terminal and the model.
    pub fn describe(&self) -> Vec<String> {
        let mut o = vec![
            format!("sheet {}×{} px", self.width, self.height),
            format!(
                "key: mean luma {:.2} ({}), σ {:.3} ({}), darks {:.0}% under 0.12, lights {:.0}% over 0.88",
                self.luma_mean,
                if self.nocturne() { "a nocturne" } else if self.luma_mean > 0.72 { "a high key" } else { "a middle key" },
                self.luma_stddev,
                if self.luma_stddev < 0.12 { "flat, foggy" } else if self.luma_stddev > 0.26 { "punchy" } else { "normal contrast" },
                self.dark_share * 100.0,
                self.light_share * 100.0
            ),
            format!(
                "surface: structure {:.3} (Laplacian σ; norm {:.3}), calm fields {:.0}% of the sheet (local σ under 0.02), busy passages {:.0}% (over 0.08)",
                self.structure,
                super::plan::STRUCTURE_NORM,
                self.flat_share * 100.0,
                self.busy_share * 100.0
            ),
            format!(
                "colour: mean chroma {:.3} ({}), {:.0}% of the colour warm",
                self.chroma_mean,
                if self.chroma_mean < 0.06 { "near-monochrome" } else if self.chroma_mean > 0.22 { "a wide palette" } else { "a moderate palette" },
                self.warm_share * 100.0
            ),
            format!("points of light on a dark ground: {:.1} per megapixel", self.points_per_mpx),
        ];
        if self.faces > 0 {
            o.push(format!("faces: {} (SCRFD), {:.1}% of the sheet", self.faces, self.face_share * 100.0));
        } else {
            o.push("faces: none".into());
        }
        match (self.subject_share, self.subject_centre) {
            (Some(s), Some((cx, cy))) => o.push(format!("subject (U²-Net matte): {:.0}% of the sheet, centred at ({:.2}, {:.2})", s * 100.0, cx, cy)),
            (Some(s), None) => o.push(format!("subject (U²-Net matte): {:.0}% of the sheet", s * 100.0)),
            _ => o.push("subject: unknown (the matter did not run)".into()),
        }
        match self.far_share {
            Some(f) => o.push(format!("depth (Depth-Anything): far ground {:.0}% of the sheet", f * 100.0)),
            None => o.push("depth: unknown (the depth model did not run)".into()),
        }
        match self.genre {
            Some((g, c)) => o.push(format!("genre (CLIP): {} ({:.0}%)", g.name(), c * 100.0)),
            None => o.push("genre: unknown (CLIP did not run)".into()),
        }
        if let (Some(sky), Some(water), Some(fol), Some(stone)) = (self.sky_share, self.water_share, self.foliage_share, self.stone_share) {
            o.push(format!("stuff (CLIP over crops): sky {:.0}%, water {:.0}%, foliage {:.0}%, stone and building {:.0}%", sky * 100.0, water * 100.0, fol * 100.0, stone * 100.0));
        }
        o
    }
}

// ────────────────────────────────────────────── the rules ───────────────────────────────────────────────

/// One dial the director moves, with its reason.
#[derive(Clone, Debug, PartialEq)]
pub struct Adjust {
    pub key: String,
    pub value: String,
    pub why: String,
}

/// The keys the director may set whether or not the recipe names them: the structure (`plan_from`'s own) and
/// the picture's palette count.
pub const STRUCTURAL: &[&str] = &[
    "medium", "palette", "style", "armature", "armature_face", "armature_body", "recede", "new", "semantic", "sam", "families",
    "commit_shadows", "value_key", "shadow_floor", "detail_len", "reserve", "budget", "pigments",
];

/// The keys a MODEL may never move: the structure, the medium's identity, the sheet's masks and ladders.
pub const PROTECTED: &[&str] = &[
    "medium", "palette", "style", "armature", "armature_face", "armature_body", "recede", "new", "semantic", "sam", "families",
    "budget", "hdr", "hair_mask", "min_brush", "blade_knives", "ladder_keep", "analysis", "outcome", "analysis_insights", "dump_passes",
    "technique", "infill", "silhouette_mode", "reserve", "detail_len", "commit_shadows",
];

/// The rules' working set: the plan's current values and the adjustments so far.
struct Table<'a> {
    values: &'a BTreeMap<String, String>,
    out: Vec<Adjust>,
}

impl Table<'_> {
    fn named(&self, key: &str) -> bool {
        self.values.contains_key(key) || STRUCTURAL.contains(&key)
    }
    fn current(&self, key: &str) -> Option<f32> {
        self.out.iter().rev().find(|a| a.key == key).and_then(|a| parse_num(&a.value)).or_else(|| self.values.get(key).and_then(|v| parse_num(v)))
    }
    fn set(&mut self, key: &str, value: impl std::fmt::Display, why: impl Into<String>) {
        if !self.named(key) {
            return;
        }
        let value = value.to_string();
        if let Some(i) = self.out.iter().position(|a| a.key == key) {
            self.out[i] = Adjust { key: key.into(), value, why: why.into() };
        } else {
            self.out.push(Adjust { key: key.into(), value, why: why.into() });
        }
    }
    /// Lift a dial the recipe has ON to at least `floor`. A dial the recipe does not name is not introduced, and
    /// one it holds at 0 is one the medium has turned OFF (a watercolour's `hotspot: 0`) — not the director's to
    /// turn on.
    fn raise(&mut self, key: &str, floor: f32, why: &str) {
        if !self.values.contains_key(key) {
            return;
        }
        let cur = self.current(key).unwrap_or(0.0);
        if cur > 0.0 && cur < floor - 1e-4 {
            self.set(key, fmt(floor), format!("{why} (was {})", fmt(cur)));
        }
    }
    /// Cap a dial the recipe names at `cap`.
    fn lower(&mut self, key: &str, cap: f32, why: &str) {
        if !self.values.contains_key(key) {
            return;
        }
        if let Some(cur) = self.current(key) {
            if cur > cap + 1e-4 {
                self.set(key, fmt(cap), format!("{why} (was {})", fmt(cur)));
            }
        }
    }
    /// Scale a dial the recipe names by `k`.
    fn scale(&mut self, key: &str, k: f32, why: &str) {
        if !self.values.contains_key(key) {
            return;
        }
        if let Some(cur) = self.current(key) {
            if cur > 0.0 && (k - 1.0).abs() > 1e-4 {
                self.set(key, fmt(cur * k), format!("{why} (was {})", fmt(cur)));
            }
        }
    }
}

fn fmt(v: f32) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() || s == "-0" { "0".into() } else { s }
}

/// The medium's family, as far as the rules care.
struct Family {
    luminous: bool,
    surface_white: bool,
    dots: bool,
}

fn family_of(medium: &str) -> Family {
    let m = crate::paint::medium::MediumProfile::by_name(medium);
    Family {
        luminous: m.as_ref().map(|m| m.mark.luminous).unwrap_or(false),
        surface_white: m.as_ref().map(|m| m.white_source == crate::paint::medium::WhiteSource::Surface).unwrap_or(false),
        dots: medium.eq_ignore_ascii_case("pointillism"),
    }
}

/// THE RULES: the picture's facts → the dials to move, over the plan's current values (`values`: the structure
/// from `plan_from` under the recipe). Each adjustment carries its reason and measurement. A dial the medium
/// does not use (the recipe does not name) is never introduced — except the structural keys.
pub fn rules(f: &SceneFacts, base: &PaintPlan, values: &BTreeMap<String, String>, medium: &str) -> Vec<Adjust> {
    let fam = family_of(medium);
    let mut t = Table { values, out: Vec::new() };

    // 1. THE KEY is measured, not inherited: the recipe's value_key was tuned on its own pictures; this picture's
    //    contrast sets it (plan_from's rule — a flat reference expanded, a punchy one left, a luminous medium's
    //    mild fixed key, a nocturne's none).
    if let Some(cur) = t.current("value_key") {
        if (cur - base.value_key).abs() > 0.05 {
            let why = base.notes.iter().find(|n| n.contains("value-key") || n.contains("key")).cloned().unwrap_or_else(|| format!("luma σ {:.3}", f.luma_stddev));
            t.set("value_key", fmt(base.value_key), format!("the picture's own key: {why} (recipe {})", fmt(cur)));
        }
    }

    // 2. A NOCTURNE (mean luma under 0.3): the night keeps its depth and its lights are the picture.
    if f.nocturne() {
        let m = format!("nocturne: mean luma {:.2}, darks {:.0}%", f.luma_mean, f.dark_share * 100.0);
        if !fam.luminous {
            t.lower("shadow_floor", 0.10, &format!("{m} — the shadow family may sit deeper than a daylight picture's"));
        }
        t.raise("point", 1.0, &format!("{m} — the lamps, the stars, the glints set with the point of the brush"));
        t.raise("point_field", 0.5, &format!("{m} — a lit street or a sky of stars is a field of points"));
        t.raise("glint", 0.6, &format!("{m} — the top of the values lifted toward white on the lights"));
        t.scale("cavity", 0.6, &format!("{m} — the relief's shadow in the dark reads as mud, not form"));
        if f.faces > 0 {
            t.scale("cavity_face", 0.6, &format!("{m} — and still less on the face"));
        }
    }

    // 3. A HIGH KEY (mean luma over 0.72): the paper and the lights carry it.
    if f.luma_mean > 0.72 {
        let m = format!("high key: mean luma {:.2}, lights {:.0}%", f.luma_mean, f.light_share * 100.0);
        if fam.surface_white {
            t.set("reserve", "0.95", format!("{m} — the paper only for the true whites, the light masses painted"));
        }
        t.raise("shadow_floor", 0.20, &format!("{m} — the darks are the picture's accents, lifted off black"));
    }

    // 4. FLAT, FOGGY (luma σ under 0.12, not a nocturne): soft edges, air, little to restate.
    if f.luma_stddev < 0.12 && !f.nocturne() {
        let m = format!("flat picture: luma σ {:.3}", f.luma_stddev);
        t.raise("lost", 0.3, &format!("{m} — edges dissolve into the ground as the fog dissolves them"));
        t.raise("aerial", 0.3, &format!("{m} — the far passages lose edge by depth"));
        t.raise("sfumato", 0.6, &format!("{m} — the transitions smoked"));
        t.scale("micro", 0.7, &format!("{m} — there is little fine structure to restate"));
        t.scale("detail_restate", 0.7, &format!("{m} — the fine passes disagree less with a soft picture"));
        t.raise("aqua_soft", 1.0, &format!("{m} — the washes' edges soft"));
        t.raise("aqua_bloom", 0.5, &format!("{m} — the blooms of a wet sheet"));
    }

    // 5. BUSY (structure over 0.45, or over 0.35 of the sheet busy): the brush adapts and comes back.
    let busy = f.structure > 0.45 || f.busy_share > 0.35;
    if busy {
        let m = format!("busy surface: structure {:.3}, busy passages {:.0}%", f.structure, f.busy_share * 100.0);
        t.raise("micro", 0.6, &format!("{m} — the fine brushes restate where the brushwork lost the structure"));
        t.raise("adapt", 0.7, &format!("{m} — one mark per feature, not a grid of dashes"));
        if let Some(cur) = t.current("detail_restate") {
            if t.values.contains_key("detail_restate") {
                t.set("detail_restate", fmt((cur * 1.3).min(0.2)), format!("{m} — the fine passes restate at a lower threshold (was {})", fmt(cur)));
            }
        }
        t.raise("aqua_detail", 0.4, &format!("{m} — the watercolour's fine brush takes the detail"));
        t.raise("aqua_crisp", 0.5, &format!("{m} — crisp wet-on-dry edges where the picture is sharp"));
    }

    // 6. CALM FIELDS (the flat share, or the sky's, over 0.45): veils, not dabs; no marbling in a plain sky.
    let calm = f.flat_share.max(f.sky_share.unwrap_or(0.0));
    if calm > 0.45 {
        let m = format!("calm fields: {:.0}% of the sheet flat{}", f.flat_share * 100.0, f.sky_share.map(|s| format!(", sky {:.0}%", s * 100.0)).unwrap_or_default());
        t.raise("smooth", 0.8, &format!("{m} — the calm passages laid in soft veils"));
        t.raise("settle", 0.4, &format!("{m} — the paint settles into the tooth on the broad fields"));
        t.raise("vary", 0.35, &format!("{m} — wider, longer marks where the picture is flat"));
        t.scale("collide", 0.6, &format!("{m} — marbling in a plain field reads as dirt"));
        t.scale("fray", 0.7, &format!("{m} — fewer torn tails over a smooth passage"));
        t.raise("aqua_soft", 0.8, &format!("{m} — soft wash edges over the fields"));
    }

    // 7. A FACE (SCRFD): a portrait painter does not rake the face, keeps the ground broad, and drags little
    //    wet paint across the modelling.
    if f.faces > 0 {
        let m = format!("{} face(s), {:.1}% of the sheet", f.faces, f.face_share * 100.0);
        if t.values.contains_key("cavity") && t.current("cavity").unwrap_or(0.0) > 0.0 {
            match t.current("cavity_face") {
                Some(cur) if cur > 0.3 => t.set("cavity_face", "0.3", format!("{m} — the relief's shadow cut on the skin (was {})", fmt(cur))),
                None if !t.values.contains_key("cavity_face") => t.out.push(Adjust { key: "cavity_face".into(), value: "0.3".into(), why: format!("{m} — the relief's shadow cut on the skin") }),
                _ => {}
            }
        }
        t.raise("periphery", 0.5, &format!("{m} — the attention on the face, the corners left broad"));
        t.scale("advect", 0.6, &format!("{m} — the wet drag blurs the face's modelling along the stroke"));
        if f.face_share > 0.08 {
            t.raise("loose", 0.3, &format!("{m}: a close portrait — the big brush for the ground, the small for the figure"));
        }
    } else if let Some(s) = f.subject_share {
        // 8. A SUBJECT without a face (the matte): compact → a focus; everywhere or nowhere → none.
        if (0.04..=0.5).contains(&s) {
            let m = format!("a compact subject: {:.0}% of the sheet", s * 100.0);
            t.raise("periphery", 0.4, &format!("{m} — the attention on it, the corners broad"));
            t.raise("loose", 0.3, &format!("{m} — the big brush for the ground"));
        } else {
            let m = format!("no compact subject: the matte covers {:.0}% of the sheet", s * 100.0);
            t.lower("periphery", 0.3, &format!("{m} — nothing to be peripheral to; the sheet finished evenly"));
        }
    }

    // 9. DEPTH (Depth-Anything, RELATIVE — the nearest thing at 1, the farthest at 0): when most of the sheet lies
    //    far behind the nearest plane — a landscape's distance, a portrait's wall — the far passages soften.
    if let Some(far) = f.far_share {
        if far > 0.45 {
            let m = format!("the far ground: {:.0}% of the sheet lies in the relative distance", far * 100.0);
            t.raise("aerial", 0.3, &format!("{m} — the far passages lose edge by depth"));
            t.raise("lost", 0.2, &format!("{m} — the far edges dissolve"));
            t.raise("nib_depth", 0.6, &format!("{m} — the nib's far strokes thinner and fewer"));
            t.raise("aqua_vignette", 0.7, &format!("{m} — the washes lighten toward the distance"));
        }
    }

    // 10. CHROMA: the painter's palette as wide as the picture's.
    if base.palette == "image" && !fam.luminous && !fam.dots {
        if f.chroma_mean < 0.06 {
            t.set("pigments", 8, format!("near-monochrome picture: mean chroma {:.3} — a limited palette of 8 pigments", f.chroma_mean));
        } else if f.chroma_mean > 0.22 {
            t.set("pigments", 20, format!("a wide palette: mean chroma {:.3} — 20 pigments derived from the picture", f.chroma_mean));
        }
    }

    // 11. POINTS OF LIGHT (the point pass's own detector, the lights on a dark ground): a night sea of stars
    //     reads 268 per Mpx, a lamplit harbour 168; a daylight street, a snowy valley, a portrait 40–70 (the
    //     bright specks in their shadows) — the line is drawn at 100.
    if f.points_per_mpx >= 100.0 {
        let m = format!("{:.0} points of light on a dark ground per megapixel", f.points_per_mpx);
        t.raise("point", 1.0, &format!("{m} — set last with the point of the brush"));
        t.raise("point_field", 0.5, &format!("{m} — a field of them accepted"));
        t.raise("hotspot", 0.6, &format!("{m} — the small bright plateaux re-modelled as their own marks"));
    }

    // 12. THE STUFF (CLIP over the crops).
    if let Some(water) = f.water_share {
        if water > 0.15 {
            let m = format!("water {:.0}% of the sheet", water * 100.0);
            t.raise("glint", 0.6, &format!("{m} — the glints on the water lifted toward white"));
            t.raise("hotspot", 0.8, &format!("{m} — the sun on the water as thick dots of white"));
            t.raise("smooth", 0.6, &format!("{m} — the still water in veils"));
        }
    }
    if let Some(fol) = f.foliage_share {
        if fol > 0.3 {
            let m = format!("foliage {:.0}% of the sheet", fol * 100.0);
            t.raise("fray", 0.5, &format!("{m} — the dry tails fall apart into leaves"));
            t.raise("vary", 0.35, &format!("{m} — the marks' lengths spread"));
            t.raise("haste", 0.5, &format!("{m} — fast, torn marks in the leaves"));
            t.raise("nib_tuft", 1.0, &format!("{m} — the nib's tufts in the grass"));
        }
    }
    let townscape = matches!(f.genre, Some((Genre::Townscape, c)) if c > 0.4);
    if townscape || f.stone_share.is_some_and(|s| s > 0.3) {
        let m = format!("stone and building{}{}", f.stone_share.map(|s| format!(" {:.0}% of the sheet", s * 100.0)).unwrap_or_default(), if townscape { ", a townscape" } else { "" });
        t.raise("adapt", 0.7, &format!("{m} — one mark per stone, the brush on the feature"));
        t.raise("chisel", 0.5, &format!("{m} — the flat brush's square marks on the masonry"));
        t.raise("micro", 0.5, &format!("{m} — the fine structure of the walls restated"));
    }
    match f.genre {
        Some((Genre::StillLife, c)) if c > 0.4 => {
            let m = format!("a still life ({:.0}%)", c * 100.0);
            t.raise("hotspot", 0.5, &format!("{m} — the highlights on glass and fruit re-modelled"));
            t.raise("glint", 0.5, &format!("{m} — the glints on the objects"));
        }
        Some((Genre::Interior, c)) if c > 0.4 => {
            let m = format!("an interior ({:.0}%)", c * 100.0);
            t.raise("shade", 0.5, &format!("{m} — the shadows off the subject calm and long"));
            t.raise("lost", 0.2, &format!("{m} — the edges in the dark lost"));
        }
        Some((Genre::Animal, c)) if c > 0.4 => {
            t.set("semantic", "true", format!("an animal ({:.0}%) — the part detector's fur and hair tier", c * 100.0));
        }
        Some((Genre::Landscape, c)) if c > 0.4 => {
            t.raise("aerial", 0.2, &format!("a landscape ({:.0}%) — the distance loses edge", c * 100.0));
        }
        _ => {}
    }
    // The knife and the dots have their own machinery: the brush dials above that their recipes do not name were
    // skipped by `raise`, so nothing is left to undo here.
    t.out
}

// ──────────────────────────────────────────── the composition ───────────────────────────────────────────

/// The director's plan: the text (header, structure, recipe, adjustments), the plan it parses to, and the
/// notes for the terminal.
#[derive(Clone, Debug)]
pub struct Directed {
    pub text: String,
    pub plan: PaintPlan,
    pub notes: Vec<String>,
    pub adjustments: Vec<Adjust>,
    /// The keys the plan names (recipe and structure), for the model's gate.
    pub keys: BTreeMap<String, String>,
}

/// Compose the plan for `a`'s medium from its facts: `plan_from`'s structure, the house recipe over it, the
/// rules over both. `source` names the picture in the header.
pub fn direct(a: &Analysis, f: &SceneFacts, source: &str) -> anyhow::Result<Directed> {
    let recipe_text = recipe_for(&a.medium);
    let recipe = recipe_text.map(parse_entries).unwrap_or_default();
    let recipe_values = values_of(&recipe);
    let recipe_new = recipe_values.get("new").map(|v| v.trim() == "true").unwrap_or(false);
    // The structure from the deterministic director, read as a NEW painting when the recipe paints one.
    let analysis = Analysis {
        faces: a.faces,
        luma_stddev: a.luma_stddev,
        luma_mean: a.luma_mean,
        medium: a.medium.clone(),
        palette: a.palette.clone(),
        short_side: a.short_side,
        long_side: a.long_side,
        surface_white: a.surface_white,
        structure: a.structure,
        from_scratch: a.from_scratch || recipe_new,
    };
    let base = plan_from(&analysis);
    let mut base_entries: Vec<Entry> = parse_entries(&base.to_hjson()).into_iter().filter(|e| matches!(e, Entry::Key { .. })).collect();
    // The recipe's palette is the medium's (sumi for the drawing media, divisionist for the dots) unless the
    // caller named one that is not the default.
    let palette_from_recipe = a.palette == "image" && recipe_values.contains_key("palette");
    base_entries.retain(|e| match e {
        Entry::Key { key, .. } => {
            // A key the recipe names, the recipe decides.
            if recipe_values.contains_key(key) && (key != "palette" || palette_from_recipe) {
                return false;
            }
            // A new painting counts its own coverage budget unless the recipe fixed one.
            if key == "budget" && (recipe_new || analysis.from_scratch) {
                return false;
            }
            true
        }
        _ => true,
    });
    let mut recipe_entries = recipe.clone();
    if !palette_from_recipe {
        recipe_entries.retain(|e| !matches!(e, Entry::Key { key, .. } if key == "palette"));
    }
    // The working values: structure under recipe.
    let mut values = values_of(&base_entries);
    values.extend(values_of(&recipe_entries));
    let adjustments = rules(f, &base, &values, &a.medium);

    // The text.
    let mut notes: Vec<String> = Vec::new();
    let mut out: Vec<Entry> = Vec::new();
    out.push(Entry::Comment(format!("plakat painting plan — written by the art director for “{source}” in {} ({}).", a.medium, if recipe_text.is_some() { "the house recipe adjusted to the picture" } else { "no house recipe for this medium: the structure and the picture's rules only" })));
    out.push(Entry::Comment("Every line below is a decision you can argue with: edit it and paint again with `--plan <this file>`.".into()));
    out.push(Entry::Blank);
    out.push(Entry::Comment("── the picture's facts ──".into()));
    for l in f.describe() {
        out.push(Entry::Comment(l));
    }
    for s in &f.sources {
        out.push(Entry::Comment(format!("source: {s}")));
    }
    out.push(Entry::Blank);
    out.push(Entry::Comment("── the structure (the deterministic director) ──".into()));
    for n in &base.notes {
        out.push(Entry::Comment(n.clone()));
    }
    if recipe_new || analysis.from_scratch {
        if recipe_values.contains_key("budget") {
            notes.push("budget: the recipe's".into());
        } else {
            out.push(Entry::Comment("new painting: no budget key — the budget is the coverage count of the planes as found".into()));
            notes.push("budget: the coverage count of the planes as found (no budget key)".into());
        }
    }
    let adjusted: std::collections::BTreeSet<&str> = adjustments.iter().map(|a| a.key.as_str()).collect();
    for e in &base_entries {
        if let Entry::Key { key, .. } = e {
            if adjusted.contains(key.as_str()) {
                continue;
            }
        }
        out.push(e.clone());
    }
    if !recipe_entries.is_empty() {
        out.push(Entry::Blank);
        out.push(Entry::Comment(format!("── the house recipe of {} ──", a.medium)));
        for e in &recipe_entries {
            match e {
                Entry::Key { key, value, comment } if adjusted.contains(key.as_str()) => {
                    // The recipe's line kept as a comment, so the reader sees what the picture moved.
                    out.push(Entry::Comment(format!("{key}: {value}   (recipe; moved below){}", comment.as_ref().map(|c| format!(" — {c}")).unwrap_or_default())));
                }
                other => out.push(other.clone()),
            }
        }
    }
    out.push(Entry::Blank);
    out.push(Entry::Comment("── the picture's adjustments (the director's rules; each with its fact) ──".into()));
    if adjustments.is_empty() {
        out.push(Entry::Comment("none: the recipe stands as it is for this picture".into()));
    }
    for adj in &adjustments {
        out.push(Entry::Key { key: adj.key.clone(), value: adj.value.clone(), comment: Some(adj.why.clone()) });
        notes.push(format!("{}: {}  — {}", adj.key, adj.value, adj.why));
    }
    let text = entries_text(&out);
    let plan = PaintPlan::parse(&text).map_err(|e| anyhow::anyhow!("the director's plan does not parse: {e}"))?;
    let mut keys = values;
    for adj in &adjustments {
        keys.insert(adj.key.clone(), adj.value.clone());
    }
    Ok(Directed { text, plan, notes, adjustments, keys })
}

// ──────────────────────────────────────────────── the model ─────────────────────────────────────────────

/// The system prompt for the model's pass, versioned with the binary.
pub const DIRECTOR_SYSTEM: &str = include_str!("../../assets/prompts/paint_director.md");

/// The model's brief: the system rules, the facts, the composed plan and the glossary of the dials it may move.
pub fn model_brief(d: &Directed, f: &SceneFacts, medium: &str) -> String {
    let mut glossary = String::new();
    for key in d.keys.keys() {
        if PROTECTED.contains(&key.as_str()) || key.contains("face") || key.contains("eyes") {
            continue;
        }
        if let Some(row) = crate::paint::report::control_row(key) {
            if !row.is_empty() {
                glossary.push_str(&format!("- `{key}` (now {}): {row}\n", d.keys[key]));
            } else {
                glossary.push_str(&format!("- `{key}` (now {})\n", d.keys[key]));
            }
        } else {
            glossary.push_str(&format!("- `{key}` (now {})\n", d.keys[key]));
        }
    }
    format!(
        "{DIRECTOR_SYSTEM}\n\nMedium: {medium}\n\n# THE PICTURE'S FACTS (measured)\n\n{}\n\n# THE PLAN SO FAR (the house recipe adjusted by the rules)\n\n```hjson\n{}```\n\n# THE DIALS YOU MAY MOVE (the only ones)\n\n{glossary}",
        f.describe().iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n"),
        d.text
    )
}

/// The model's lines through the gate: only `key: value` lines whose key the plan names and that is not
/// protected or a face's, with a numeric value (clamped to [0,1], or to [0, 2× the current value] for a dial
/// that stands above 1 — a count, an angle, a pigment). Returns the accepted adjustments and the rejected
/// lines with the reason each was refused.
pub fn accept_model_lines(text: &str, d: &Directed) -> (Vec<Adjust>, Vec<String>) {
    let mut body = text;
    if let Some(i) = text.find("```") {
        let after = &text[i + 3..];
        let after = after.strip_prefix("hjson").unwrap_or(after);
        body = after.split("```").next().unwrap_or(after);
    }
    let (mut ok, mut no) = (Vec::new(), Vec::new());
    for e in parse_entries(body) {
        let Entry::Key { key, value, comment } = e else { continue };
        if !d.keys.contains_key(&key) {
            no.push(format!("{key}: {value} — not a dial of this medium's plan"));
            continue;
        }
        if PROTECTED.contains(&key.as_str()) {
            no.push(format!("{key}: {value} — the structure is not the model's to move"));
            continue;
        }
        if key.contains("face") || key.contains("eyes") {
            no.push(format!("{key}: {value} — the faces stay as the recipe and the rules set them"));
            continue;
        }
        let Some(v) = parse_num(&value) else {
            no.push(format!("{key}: {value} — not a number"));
            continue;
        };
        let cur = parse_num(&d.keys[&key]).unwrap_or(0.0);
        let clamped = if cur > 1.0 { v.clamp(0.0, cur * 2.0) } else { v.clamp(0.0, 1.0) };
        if (clamped - cur).abs() < 1e-4 {
            continue;
        }
        let why = comment.unwrap_or_else(|| "the model's proposal".into());
        let why = if (clamped - v).abs() > 1e-4 { format!("{why} (asked {}, clamped)", fmt(v)) } else { why };
        ok.push(Adjust { key, value: fmt(clamped), why });
    }
    (ok, no)
}

/// Write the model's accepted lines into the plan as their own section and re-parse it.
pub fn apply_model(d: &Directed, label: &str, accepted: &[Adjust], rejected: &[String]) -> anyhow::Result<Directed> {
    // A plan reads a key once (the parser refuses a duplicate): the line the model moves is kept as a comment.
    let moved: std::collections::BTreeSet<&str> = accepted.iter().map(|a| a.key.as_str()).collect();
    let entries: Vec<Entry> = parse_entries(&d.text)
        .into_iter()
        .map(|e| match e {
            Entry::Key { key, value, comment } if moved.contains(key.as_str()) => Entry::Comment(format!("{key}: {value}   (moved below by the model){}", comment.map(|c| format!(" — {c}")).unwrap_or_default())),
            other => other,
        })
        .collect();
    let mut text = entries_text(&entries);
    text.push_str(&format!("\n# ── the model's adjustments ({label}; the picture seen, the lines gated: the medium's dials only, never the structure or the faces) ──\n"));
    if accepted.is_empty() {
        text.push_str("# none accepted\n");
    }
    for a in accepted {
        text.push_str(&format!("{}: {}   # {}\n", a.key, a.value, a.why));
    }
    for r in rejected {
        text.push_str(&format!("# refused: {r}\n"));
    }
    let plan = PaintPlan::parse(&text).map_err(|e| anyhow::anyhow!("the plan with the model's lines does not parse: {e}"))?;
    let mut keys = d.keys.clone();
    let mut adjustments = d.adjustments.clone();
    let mut notes = d.notes.clone();
    for a in accepted {
        keys.insert(a.key.clone(), a.value.clone());
        adjustments.push(a.clone());
        notes.push(format!("{}: {}  — model: {}", a.key, a.value, a.why));
    }
    Ok(Directed { text, plan, notes, adjustments, keys })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis(medium: &str, luma_mean: f32, luma_stddev: f32, faces: usize) -> Analysis {
        let surface_white = crate::paint::medium::MediumProfile::by_name(medium).map(|m| m.white_source == crate::paint::medium::WhiteSource::Surface).unwrap_or(false);
        Analysis { faces, luma_stddev, luma_mean, medium: medium.into(), palette: "image".into(), short_side: 1536, long_side: 2048, surface_white, structure: 0.33, from_scratch: false }
    }

    fn facts(luma_mean: f32, luma_stddev: f32) -> SceneFacts {
        SceneFacts { width: 2048, height: 1536, luma_mean, luma_stddev, structure: 0.33, chroma_mean: 0.12, ..Default::default() }
    }

    #[test]
    fn every_executable_medium_has_a_recipe_that_parses() {
        for m in crate::paint::medium::EXECUTABLE {
            let text = recipe_for(m).unwrap_or_else(|| panic!("no recipe for {m}"));
            let plan = PaintPlan::parse(text).unwrap_or_else(|e| panic!("{m}'s recipe does not parse: {e}"));
            let v = values_of(&parse_entries(text));
            if let Some(named) = v.get("medium") {
                assert_eq!(named, m, "{m}'s recipe names another medium");
                assert_eq!(plan.medium, *m);
            }
            // No two lines name the same key: the plan reads as written.
            let entries = parse_entries(text);
            let keys: Vec<&String> = entries.iter().filter_map(|e| if let Entry::Key { key, .. } = e { Some(key) } else { None }).collect();
            let mut seen = std::collections::HashSet::new();
            for k in &keys {
                assert!(seen.insert(k.as_str()), "{m}'s recipe names {k} twice");
            }
        }
        assert!(recipe_for("watercolor").is_some() && recipe_for("no-such-medium").is_none());
    }

    #[test]
    fn the_lines_round_trip_and_keep_an_inline_array_and_a_comment() {
        let text = "# a note\nmedium: oil-direct\nblade_knives: [120, 56, 26]   # the knives\n\nimpasto: 0.45 # thick\n{\n}\n";
        let e = parse_entries(text);
        assert_eq!(e[0], Entry::Comment("a note".into()));
        assert_eq!(e[2], Entry::Key { key: "blade_knives".into(), value: "[120, 56, 26]".into(), comment: Some("the knives".into()) });
        assert_eq!(e[3], Entry::Blank);
        assert_eq!(e[4], Entry::Key { key: "impasto".into(), value: "0.45".into(), comment: Some("thick".into()) });
        assert_eq!(e.len(), 5, "the braces are dropped");
        let back = entries_text(&e);
        assert_eq!(values_of(&parse_entries(&back)), values_of(&e));
        assert!(PaintPlan::parse(&back).is_ok());
    }

    #[test]
    fn the_local_sigma_is_zero_on_a_flat_field_and_half_on_a_fine_checker() {
        let flat = vec![0.4f32; 20 * 12];
        assert!(local_std(&flat, 20, 12, 4).iter().all(|v| *v < 1e-6));
        let checker: Vec<f32> = (0..20 * 12).map(|i| if (i % 20 + i / 20) % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let sd = local_std(&checker, 20, 12, 4);
        // A 9×9 window over a 1-px checker holds 41 of one and 40 of the other: σ ≈ 0.5.
        assert!((sd[6 * 20 + 10] - 0.5).abs() < 0.01, "{}", sd[6 * 20 + 10]);
        // The corner's clipped window too.
        assert!((sd[0] - 0.5).abs() < 0.02, "{}", sd[0]);
    }

    #[test]
    fn the_pixel_facts_tell_a_nocturne_with_points_from_a_flat_grey() {
        // A night: near-black with a handful of bright specks.
        let mut night = RgbImage::from_pixel(256, 192, image::Rgb([12, 12, 20]));
        for (x, y) in [(40u32, 50u32), (120, 30), (200, 150), (70, 170)] {
            night.put_pixel(x, y, image::Rgb([255, 250, 230]));
            night.put_pixel(x + 1, y, image::Rgb([240, 235, 220]));
        }
        let f = SceneFacts::from_pixels(&night);
        assert!(f.nocturne(), "mean luma {}", f.luma_mean);
        assert!(f.dark_share > 0.95);
        assert!(f.points_per_mpx > 0.0, "the specks are points of light");
        assert!(f.chroma_mean < 0.06, "near-monochrome: {}", f.chroma_mean);
        // A flat mid grey: calm everywhere, no points, no chroma.
        let grey = RgbImage::from_pixel(256, 192, image::Rgb([128, 128, 128]));
        let g = SceneFacts::from_pixels(&grey);
        assert!(!g.nocturne() && g.luma_mean <= 0.72);
        assert!(g.flat_share > 0.99 && g.busy_share == 0.0, "flat {} busy {}", g.flat_share, g.busy_share);
        assert_eq!(g.points_per_mpx, 0.0);
        assert_eq!(g.warm_share, 0.0);
        // A warm, busy checker: chroma wide, warm, busy.
        let busy = RgbImage::from_fn(256, 192, |x, y| if (x / 4 + y / 4) % 2 == 0 { image::Rgb([230, 120, 30]) } else { image::Rgb([30, 30, 30]) });
        let b = SceneFacts::from_pixels(&busy);
        assert!(b.busy_share > 0.9, "busy {}", b.busy_share);
        assert!(b.chroma_mean > 0.22 && b.warm_share > 0.9, "chroma {} warm {}", b.chroma_mean, b.warm_share);
        assert!(f.describe().iter().any(|l| l.contains("a nocturne")));
    }

    #[test]
    fn a_nocturne_in_oil_deepens_the_floor_and_sets_the_point_and_the_text_parses() {
        let a = analysis("oil-direct", 0.18, 0.15, 0);
        let mut f = facts(0.18, 0.15);
        f.dark_share = 0.7;
        let d = direct(&a, &f, "night.png").unwrap();
        let v = values_of(&parse_entries(&d.text));
        assert_eq!(v["shadow_floor"], "0.1", "{}", d.text);
        assert_eq!(parse_num(&v["point"]), Some(1.0), "the recipe's point stands at 1 already; a nocturne keeps it");
        assert_eq!(v["cavity"], "0.3", "cavity 0.5 × 0.6");
        assert_eq!(v["medium"], "oil-direct");
        assert_eq!(v["new"], "true");
        assert!(!v.contains_key("budget"), "a new painting counts its coverage budget");
        assert_eq!(d.plan.shadow_floor, 0.1);
        assert_eq!(d.plan.cavity, Some(0.3));
        assert!(d.plan.from_scratch);
        // Every recipe key survives into the plan, moved or not.
        for (k, _) in values_of(&parse_entries(recipe_for("oil-direct").unwrap())) {
            assert!(v.contains_key(&k), "the recipe's {k} is lost");
        }
        // Each key once.
        let keys: Vec<String> = parse_entries(&d.text).into_iter().filter_map(|e| if let Entry::Key { key, .. } = e { Some(key) } else { None }).collect();
        let mut seen = std::collections::HashSet::new();
        for k in &keys {
            assert!(seen.insert(k.clone()), "{k} is named twice in\n{}", d.text);
        }
        assert!(d.text.contains("nocturne: mean luma 0.18"), "the reason is written");
    }

    #[test]
    fn a_daylight_picture_keeps_the_recipe_but_its_own_key() {
        let a = analysis("oil-direct", 0.55, 0.26, 0);
        let f = facts(0.55, 0.26);
        let d = direct(&a, &f, "day.png").unwrap();
        let v = values_of(&parse_entries(&d.text));
        // The recipe's value_key 0.30 yields to the measured one (σ 0.26 → 0).
        assert_eq!(v["value_key"], "0");
        assert_eq!(v["shadow_floor"], "0.16");
        assert_eq!(v["cavity"], "0.5", "nothing moved it");
        assert!(d.adjustments.iter().all(|x| x.key == "value_key"), "{:?}", d.adjustments);
    }

    #[test]
    fn a_face_softens_the_cavity_and_the_drag_and_a_flat_picture_loses_its_edges() {
        let a = analysis("oil-direct", 0.5, 0.10, 1);
        let mut f = facts(0.5, 0.10);
        f.faces = 1;
        f.face_share = 0.12;
        let d = direct(&a, &f, "portrait.png").unwrap();
        let v = values_of(&parse_entries(&d.text));
        assert_eq!(v["cavity_face"], "0.3");
        assert_eq!(v["advect"], "0.3", "0.5 × 0.6");
        assert!(!v.contains_key("lost"), "a flat picture raises `lost` only where the recipe names it");
    }

    #[test]
    fn a_flat_picture_moves_only_the_dials_the_recipe_names() {
        // oil-direct names no `lost`, `aerial` 0.15, `sfumato` none: the flat rule raises aerial only.
        let a = analysis("oil-direct", 0.5, 0.10, 0);
        let d = direct(&a, &facts(0.5, 0.10), "fog.png").unwrap();
        let keys: Vec<&str> = d.adjustments.iter().map(|x| x.key.as_str()).collect();
        assert!(keys.contains(&"aerial"), "{keys:?}");
        assert!(!keys.contains(&"lost") && !keys.contains(&"sfumato"), "{keys:?}");
        let recipe = values_of(&parse_entries(recipe_for("oil-direct").unwrap()));
        for k in &keys {
            assert!(recipe.contains_key(*k) || STRUCTURAL.contains(k), "{k} is neither the recipe's nor structural");
        }
    }

    #[test]
    fn a_medium_without_a_full_recipe_gets_the_structure_and_nothing_it_cannot_use() {
        let a = analysis("durer", 0.2, 0.3, 0);
        let mut f = facts(0.2, 0.3);
        f.points_per_mpx = 40.0;
        f.water_share = Some(0.5);
        let d = direct(&a, &f, "x.png").unwrap();
        let v = values_of(&parse_entries(&d.text));
        assert_eq!(v["medium"], "durer");
        assert_eq!(v["micro"], "0.5");
        assert!(v.contains_key("budget"), "not a new painting: the structural budget stands");
        assert!(!v.contains_key("point") && !v.contains_key("glint"), "{}", d.text);
        for x in &d.adjustments {
            assert!(STRUCTURAL.contains(&x.key.as_str()) || ["micro", "adapt", "point_field"].contains(&x.key.as_str()), "{x:?}");
        }
    }

    #[test]
    fn every_medium_directs_to_a_plan_that_parses_for_several_pictures() {
        let pictures = [(0.18f32, 0.12f32, 0usize), (0.5, 0.26, 1), (0.8, 0.09, 0), (0.45, 0.5, 2)];
        for m in crate::paint::medium::EXECUTABLE {
            for (lm, sd, faces) in pictures {
                let a = analysis(m, lm, sd, faces);
                let mut f = facts(lm, sd);
                f.faces = faces;
                f.face_share = if faces > 0 { 0.1 } else { 0.0 };
                f.structure = if sd > 0.4 { 0.5 } else { 0.3 };
                f.flat_share = if sd < 0.1 { 0.6 } else { 0.2 };
                f.subject_share = Some(0.3);
                f.far_share = Some(0.5);
                f.genre = Some((Genre::Landscape, 0.7));
                f.sky_share = Some(0.3);
                f.water_share = Some(0.2);
                f.foliage_share = Some(0.4);
                f.stone_share = Some(0.1);
                let d = direct(&a, &f, "p.png").unwrap_or_else(|e| panic!("{m}: {e}"));
                assert_eq!(d.plan.medium, *m);
                // The drawing media keep their sumi palette from the recipe; the dots their divisionist.
                let v = values_of(&parse_entries(&d.text));
                let recipe = values_of(&parse_entries(recipe_for(m).unwrap()));
                if let Some(p) = recipe.get("palette") {
                    assert_eq!(&v["palette"], p, "{m}");
                }
            }
        }
    }

    #[test]
    fn a_named_palette_beats_the_recipe_s() {
        let mut a = analysis("pencil", 0.5, 0.2, 0);
        a.palette = "earth".into();
        let d = direct(&a, &facts(0.5, 0.2), "p.png").unwrap();
        assert_eq!(d.plan.palette, "earth");
        let d2 = direct(&analysis("pencil", 0.5, 0.2, 0), &facts(0.5, 0.2), "p.png").unwrap();
        assert_eq!(d2.plan.palette, "sumi", "the default `image` yields to the recipe's sumi");
    }

    #[test]
    fn the_model_s_lines_pass_the_gate_or_do_not() {
        let a = analysis("oil-direct", 0.5, 0.2, 1);
        let mut f = facts(0.5, 0.2);
        f.faces = 1;
        let d = direct(&a, &f, "p.png").unwrap();
        let reply = "## Insights\n- blah\n\n## Recommendations\n```hjson\nimpasto: 0.7   # thicker on the lights (lights 20%)\narmature: 100   # finer\ncavity_face: 0.9\nglaze: 0.5\nblade_light: 400\nsmooth: 1.4   # over the cap\nsmooth: abc\npigments: 12   # fewer colours\n```\n- could not judge likeness\n";
        let (ok, no) = accept_model_lines(reply, &d);
        let keys: Vec<&str> = ok.iter().map(|x| x.key.as_str()).collect();
        assert_eq!(keys, vec!["impasto", "blade_light", "smooth"], "{ok:?}");
        assert_eq!(ok[0].value, "0.7");
        assert_eq!(ok[1].value, "400", "blade_light stands at 225: a value under 2× passes as asked");
        assert_eq!(ok[2].value, "1", "smooth clamped to 1");
        assert!(ok[2].why.contains("clamped"));
        assert!(no.iter().any(|r| r.starts_with("armature:")), "{no:?}");
        assert!(no.iter().any(|r| r.starts_with("cavity_face:")), "{no:?}");
        assert!(no.iter().any(|r| r.starts_with("glaze:")), "oil-direct's recipe has no glaze: {no:?}");
        assert!(no.iter().any(|r| r.contains("not a number")), "{no:?}");
        let d2 = apply_model(&d, "test", &ok, &no).unwrap();
        assert_eq!(d2.plan.impasto, Some(0.7));
        assert!(d2.text.contains("impasto: 0.45   (moved below by the model)"), "{}", d2.text);
        assert!(no.iter().any(|r| r.starts_with("pigments:")), "a key the plan does not name is not the model's: {no:?}");
        assert_eq!(d2.plan.armature, d.plan.armature);
        assert!(d2.text.contains("# refused: armature: 100"));
        assert!(model_brief(&d, &f, "oil-direct").contains("`impasto` (now 0.45)"));
        assert!(!model_brief(&d, &f, "oil-direct").contains("`cavity_face`"), "the face's dials are not offered");
    }

    #[test]
    fn the_director_prompt_pins_its_rules() {
        for must in ["ONLY", "faces", "key: value", "## Recommendations", "structure"] {
            assert!(DIRECTOR_SYSTEM.contains(must), "system prompt lost: {must}");
        }
    }

    #[test]
    fn the_genre_and_the_stuff_read_their_scores() {
        let scores: Vec<(Genre, f32)> = Genre::LABELS.iter().map(|(g, _)| (*g, if *g == Genre::Townscape { 0.30 } else { 0.22 })).collect();
        let (g, c) = Genre::from_scores(&scores).unwrap();
        assert_eq!(g, Genre::Townscape);
        assert!(c > 0.9, "{c}");
        let s: Vec<f32> = STUFF_LABELS.iter().map(|(st, _)| if *st == Stuff::Sky { 0.3 } else { 0.2 }).collect();
        let (sky, water, fol, stone) = stuff_shares(&s);
        assert!(sky > 0.9 && water < 0.05 && fol < 0.05 && stone < 0.05, "{sky} {water} {fol} {stone}");
        assert_eq!(stuff_shares(&[0.1, 0.2]), (0.0, 0.0, 0.0, 0.0));
    }
}

#[cfg(test)]
mod calibration {
    //! `PLAKAT_DIRECTOR_FACTS="a.jpg;b.jpg" cargo test --lib paint::director::calibration -- --ignored --nocapture`:
    //! the pixel facts of real pictures, for reading the rules' thresholds against them.
    #[test]
    #[ignore]
    fn print_the_facts_of_the_pictures_named_in_the_environment() {
        let Ok(list) = std::env::var("PLAKAT_DIRECTOR_FACTS") else { return };
        for path in list.split(';').filter(|p| !p.is_empty()) {
            let img = image::open(path).expect("opening").to_rgb8();
            let (w, h) = img.dimensions();
            let scale = (1024.0 / w.max(h) as f32).min(1.0);
            let small = if scale < 1.0 { image::imageops::resize(&img, ((w as f32 * scale) as u32).max(1), ((h as f32 * scale) as u32).max(1), image::imageops::FilterType::Triangle) } else { img.clone() };
            let f = super::SceneFacts::from_pixels(&small);
            let lin = crate::paint::painter::luma_map(&small);
            let (sw, sh) = small.dimensions();
            let mpx = (sw as f32 * sh as f32) / 1.0e6;
            let mut pts = String::new();
            for (r, alone) in [(2i32, 0.02f32), (2, 0.09), (3, 0.02)] {
                let pl = crate::paint::painter::point_lights(&lin, sw, sh, r, alone);
                pts.push_str(&format!(" r{r}/alone{alone}: {:.0} (ground<0.25: {:.0}, <0.15: {:.0})", pl.len() as f32 / mpx, pl.iter().filter(|p| p.ground < 0.25).count() as f32 / mpx, pl.iter().filter(|p| p.ground < 0.15).count() as f32 / mpx));
            }
            println!("{path}: luma {:.2} σ {:.3} flat {:.2} busy {:.2} chroma {:.3} warm {:.2} points/Mpx{pts}", f.luma_mean, f.luma_stddev, f.flat_share, f.busy_share, f.chroma_mean, f.warm_share);
        }
    }
}
