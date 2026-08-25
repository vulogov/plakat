//! `plakat naturalize` — make AI / computer output **less sloppy: a genuinely better picture** (RFC
//! QUALITY-1/2). The goal is NOT to disguise a render as human-made — it's to *fix* the things that make
//! AI output look cheap. Two halves, both weight-free, deterministic, no GPU:
//!   1. **Quality improvement** (the core — see [`polish`] + [`micro_texture`]): gray-world white balance,
//!      robust auto-levels, vibrance (tame the oversaturation, lift dull colour), unsharp, and variance-
//!      gated **micro-texture** (pores / micro-wrinkles that break plastic AI skin). This makes the colour,
//!      contrast and detail objectively better.
//!   2. **Analog finish** (optional, secondary — see [`apply`]): a light film grain / grade / vignette so
//!      the frame isn't uniformly razor-clean. Chromatic aberration stays minimal (it's a *degradation*).
//!
//! Structural errors (bad reflections, incoherent geometry, plastic hands) are a model-capability limit —
//! the corrective focuses (`--geometry`/`--anatomy`, model-backed) and the hi-res fix address those.

pub mod refine;

use image::{GrayImage, Luma, Rgb, RgbImage};

/// The analog-imperfection strengths (each roughly `0.0..=1.0`; higher = more).
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Film grain amount (luminance-weighted, noisier in the mids/shadows).
    pub grain: f32,
    /// Chromatic aberration — radial R-outward / B-inward channel shift, growing with r².
    pub aberration: f32,
    /// Radial corner darkening.
    pub vignette: f32,
    /// Highlight bloom / halation.
    pub bloom: f32,
    /// Desaturation toward luminance (the oversaturation tell is the loudest one).
    pub desaturate: f32,
    /// Warm film lift in the shadows (+R / −B).
    pub warm: f32,
    /// Radial defocus — a faint edge softness so the frame isn't uniformly razor-sharp.
    pub defocus: f32,
    /// **Quality improvement** ("de-slop") strength — a genuine correction pass (gray-world white
    /// balance + robust auto-levels + vibrance + unsharp) that runs FIRST, making the colours and detail
    /// objectively better before any analog look. `0` = off. This is the part that makes a *better*
    /// picture, not a disguised one; see [`polish`].
    pub polish: f32,
    /// **Micro-texture** strength — fine pore / micro-wrinkle detail added ONLY where the image is
    /// unnaturally smooth (variance-gated), modulated to mid-tones. Breaks the plastic "perfect skin" AI
    /// tell that colour/grade alone can't touch. High on the `People` focus. See [`micro_texture`].
    pub micro: f32,
}

impl Default for Params {
    fn default() -> Self {
        Preset::Subtle.params()
    }
}

/// Named strength bundles. **All aim at contemporary realism, not a retro/"vintage" look** — the goal is
/// to read as a genuine human-made image, so the grade only *desaturates* (kills the AI oversaturation
/// tell); the warm lift and vignette stay small (a strong warm grade + heavy vignette read as an applied
/// *filter*, which is its own artifact, not naturalness).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// Barely-there — a light denial of digital perfection (the default). Grain + a hint of aberration +
    /// a touch of desaturation; essentially no vignette/warmth.
    Subtle,
    /// A real-camera look — film grain + lens aberration + a slight vignette + desaturation. Neutral grade.
    Photo,
    /// For painterly renders — a canvas-like grain + more desaturation + a faint defocus. Neutral grade.
    Painting,
}

impl Preset {
    pub fn parse(s: &str) -> Option<Preset> {
        match s.trim().to_ascii_lowercase().as_str() {
            "subtle" => Some(Preset::Subtle),
            "photo" => Some(Preset::Photo),
            "painting" => Some(Preset::Painting),
            _ => None,
        }
    }
    pub fn params(self) -> Params {
        match self {
            // Polish-forward (make a BETTER picture), analog kept light — chromatic aberration in
            // particular is a *degradation* (colour fringing), so it stays minimal.
            Preset::Subtle => Params { grain: 0.10, aberration: 0.02, vignette: 0.03, bloom: 0.03, desaturate: 0.05, warm: 0.0, defocus: 0.0, polish: 0.55, micro: 0.15 },
            Preset::Photo => Params { grain: 0.18, aberration: 0.05, vignette: 0.07, bloom: 0.06, desaturate: 0.08, warm: 0.03, defocus: 0.0, polish: 0.70, micro: 0.25 },
            Preset::Painting => Params { grain: 0.22, aberration: 0.03, vignette: 0.06, bloom: 0.05, desaturate: 0.10, warm: 0.03, defocus: 0.04, polish: 0.60, micro: 0.15 },
        }
    }
}

/// A **content focus** (RFC QUALITY-1): each subject has a characteristic AI tell, so pre-tuning the pass
/// to it de-AIs more effectively. A focus is a full target [`Params`] the base is blended toward (see
/// [`blend_focus`]); the weight `N` says how strongly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// AI **skin** is plastic / waxy / over-saturated → more desaturation + fine grain, minimal aberration
    /// & vignette (fringing/darkening look wrong on faces).
    People,
    /// AI **skies** band and are too-smooth → fine de-banding grain, little else.
    Sky,
    /// AI **foliage** is a cloud-like repeating mush → stronger broadband grain + a little defocus variation
    /// to break the uniform texture.
    Vegetation,
    /// AI **cityscapes** have razor-clean geometry / repeating windows → grain + edge chromatic aberration
    /// (lens) + a touch of defocus.
    Cityscape,
    /// General **landscape / scenery** → grain + a gentle atmospheric vignette + desaturation (haze).
    Landscape,
    /// **Seascape** surface → specular-highlight bloom + fine surface grain, cool/desaturated. (Fixing the
    /// water *reflections* is structural — a P2 model refine, not this analog pass.)
    Sea,
    /// **Riverscape** surface → like [`Sea`](Focus::Sea) with a touch of defocus.
    River,
    /// **Mechanical apparatus / transports** surface → metal specular bloom + edge chromatic aberration
    /// (lens on hard edges) + grain. (Fixing mechanical *structure* is a P2 refine.)
    Mechanics,
    /// **Household / indoor** scene → soft indoor grain + gentle vignette, low aberration.
    Household,
    /// **Animal** (fur / feather) — AI fur is a too-smooth gradient → heavy micro-texture + grain.
    Animal,
    /// **Food** — AI food is over-saturated and plastic-glossy → desaturate + tame bloom + micro.
    Food,
    /// **Interior / architectural render** — flat, even CGI light → auto-levels contrast + vignette depth.
    Interior,
    /// **Textile / fabric** — AI cloth is a smooth sheen → weave micro-texture + grain.
    Textile,
    /// **Foliage macro / close-up botanical** — fine leaf/petal detail → micro + broadband grain.
    FoliageMacro,
}

impl Focus {
    /// Parse an **analog** focus name (the weight-free surface-look ones). The corrective focuses
    /// (`geometry`/`anatomy`/`no-twins`) are model-backed and handled separately — not here.
    pub fn parse(s: &str) -> Option<Focus> {
        match s.trim().to_ascii_lowercase().as_str() {
            "people" | "portrait" => Some(Focus::People),
            "sky" => Some(Focus::Sky),
            "vegetation" | "foliage" => Some(Focus::Vegetation),
            "cityscape" | "city" => Some(Focus::Cityscape),
            "landscape" => Some(Focus::Landscape),
            "sea" | "seascape" | "ocean" => Some(Focus::Sea),
            "river" | "riverscape" => Some(Focus::River),
            "mechanics" | "mechanical" | "transport" => Some(Focus::Mechanics),
            "household" | "indoor" => Some(Focus::Household),
            "animal" | "fur" | "feather" => Some(Focus::Animal),
            "food" => Some(Focus::Food),
            "interior" | "render" => Some(Focus::Interior),
            "textile" | "fabric" | "cloth" => Some(Focus::Textile),
            "foliage-macro" | "foliagemacro" | "botanical" | "macro" => Some(Focus::FoliageMacro),
            _ => None,
        }
    }

    fn profile(self) -> Params {
        match self {
            // polish is the "make it better" knob; cityscape/mechanics push it higher (crisper structure).
            // People: HEAVY micro-texture — pores/micro-wrinkles are the fix for plastic AI skin.
            Focus::People => Params { grain: 0.16, aberration: 0.02, vignette: 0.03, bloom: 0.05, desaturate: 0.12, warm: 0.03, defocus: 0.0, polish: 0.55, micro: 0.85 },
            Focus::Sky => Params { grain: 0.14, aberration: 0.02, vignette: 0.04, bloom: 0.08, desaturate: 0.06, warm: 0.02, defocus: 0.0, polish: 0.55, micro: 0.20 },
            Focus::Vegetation => Params { grain: 0.22, aberration: 0.04, vignette: 0.05, bloom: 0.05, desaturate: 0.10, warm: 0.03, defocus: 0.04, polish: 0.62, micro: 0.30 },
            Focus::Cityscape => Params { grain: 0.16, aberration: 0.06, vignette: 0.06, bloom: 0.08, desaturate: 0.07, warm: 0.03, defocus: 0.0, polish: 0.75, micro: 0.25 },
            Focus::Landscape => Params { grain: 0.16, aberration: 0.04, vignette: 0.10, bloom: 0.08, desaturate: 0.10, warm: 0.04, defocus: 0.0, polish: 0.62, micro: 0.25 },
            Focus::Sea => Params { grain: 0.16, aberration: 0.03, vignette: 0.06, bloom: 0.12, desaturate: 0.10, warm: 0.0, defocus: 0.0, polish: 0.60, micro: 0.20 },
            Focus::River => Params { grain: 0.18, aberration: 0.03, vignette: 0.06, bloom: 0.09, desaturate: 0.09, warm: 0.03, defocus: 0.02, polish: 0.62, micro: 0.20 },
            Focus::Mechanics => Params { grain: 0.16, aberration: 0.05, vignette: 0.06, bloom: 0.12, desaturate: 0.07, warm: 0.03, defocus: 0.0, polish: 0.75, micro: 0.30 },
            Focus::Household => Params { grain: 0.16, aberration: 0.04, vignette: 0.06, bloom: 0.08, desaturate: 0.09, warm: 0.05, defocus: 0.0, polish: 0.60, micro: 0.25 },
            // 6.13 (RFC QUALITY-4 P3):
            Focus::Animal => Params { grain: 0.22, aberration: 0.02, vignette: 0.04, bloom: 0.05, desaturate: 0.10, warm: 0.03, defocus: 0.0, polish: 0.58, micro: 0.70 },
            Focus::Food => Params { grain: 0.18, aberration: 0.02, vignette: 0.05, bloom: 0.04, desaturate: 0.16, warm: 0.04, defocus: 0.0, polish: 0.65, micro: 0.35 },
            Focus::Interior => Params { grain: 0.16, aberration: 0.03, vignette: 0.10, bloom: 0.06, desaturate: 0.08, warm: 0.05, defocus: 0.0, polish: 0.72, micro: 0.25 },
            Focus::Textile => Params { grain: 0.20, aberration: 0.02, vignette: 0.05, bloom: 0.04, desaturate: 0.08, warm: 0.03, defocus: 0.0, polish: 0.58, micro: 0.55 },
            Focus::FoliageMacro => Params { grain: 0.24, aberration: 0.03, vignette: 0.05, bloom: 0.05, desaturate: 0.10, warm: 0.03, defocus: 0.02, polish: 0.62, micro: 0.45 },
        }
    }
}

fn to_arr(p: &Params) -> [f32; 9] {
    [p.grain, p.aberration, p.vignette, p.bloom, p.desaturate, p.warm, p.defocus, p.polish, p.micro]
}
fn from_arr(a: [f32; 9]) -> Params {
    Params { grain: a[0], aberration: a[1], vignette: a[2], bloom: a[3], desaturate: a[4], warm: a[5], defocus: a[6], polish: a[7], micro: a[8] }
}

/// Blend the base params toward each active [`Focus`] by its weight — a weighted average where the base
/// always carries weight 1 (so with no focuses the base is returned unchanged, and a single `focus:1.0`
/// gives the midpoint of base and that profile). Weights ≤ 0 are ignored.
pub fn blend_focus(base: Params, focuses: &[(Focus, f32)]) -> Params {
    let mut acc = to_arr(&base);
    let mut wsum = 1.0f32;
    for (f, w) in focuses {
        let w = w.max(0.0);
        if w <= 0.0 {
            continue;
        }
        let pf = to_arr(&f.profile());
        for i in 0..9 {
            acc[i] += pf[i] * w;
        }
        wsum += w;
    }
    for v in acc.iter_mut() {
        *v /= wsum;
    }
    from_arr(acc)
}

/// The named **preset library** (RFC QUALITY-7 P3) — curated de-slop recipes beyond the three base presets,
/// each a full naturalize spec. `(name, spec, one-line description)`.
pub fn preset_library() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        ("portrait", "photo people=1.2 micro=1", "photoreal faces — heavy skin micro-texture, tame oversaturation"),
        ("landscape", "photo sky=1 vegetation=0.8 landscape=1", "scenery — de-band sky, break cloud-foliage mush, atmosphere"),
        ("product", "photo polish=0.8 desaturate=0.05 micro=0.2", "clean product shot — strong colour correction, minimal grade"),
        ("anime", "subtle polish=0.7 micro=0.2 grain=0.08 aberration=0", "cel/anime — polish + faint grain, no aberration"),
        ("film", "photo grain=0.3 desaturate=0.12 warm=0.06 vignette=0.12", "filmic look — visible grain, warm grade, vignette"),
        ("restore", "polish=0.9 desaturate=0.1 micro=0.3 grain=0.06", "de-slop pass — strong polish + micro, light finish"),
    ]
}

/// A naturalize preset **spec tuned to a model family's tells** (RFC QUALITY-8 P1) — SDXL over-saturates,
/// SD 1.5 is soft, Flux is clean-but-plastic, etc. Matched by substring on the model name from metadata.
/// `None` for an unrecognised model (caller falls back to analysis).
pub fn model_preset(model: &str) -> Option<&'static str> {
    let m = model.to_ascii_lowercase();
    // order matters — check the more specific tokens first.
    if m.contains("flux") {
        Some("photo polish=0.72 micro=0.5 desaturate=0.08 grain=0.14") // clean but plastic → micro + light grain
    } else if m.contains("sd35") || m.contains("sd3.5") || m.contains("sd3") || m.contains("stable-diffusion-3") {
        Some("photo polish=0.72 desaturate=0.1 micro=0.35")
    } else if m.contains("sdxl") || m.contains("xl") || m.contains("lightning") || m.contains("hyper") {
        Some("photo desaturate=0.16 polish=0.78 micro=0.3 aberration=0") // SDXL over-saturates (no aberration — it's a degradation)
    } else if m.contains("sd15") || m.contains("sd1.5") || m.contains("v1-5") || m.contains("stable-diffusion-v1") || m.contains("dreamshaper") {
        Some("photo polish=0.75 micro=0.4 grain=0.14 aberration=0") // SD 1.5 is soft → lean on unsharp + micro + grain
    } else if m.contains("pixart") {
        Some("photo polish=0.72 desaturate=0.1 micro=0.3")
    } else if m.contains("cascade") {
        Some("photo polish=0.7 desaturate=0.1 micro=0.3 bloom=0.1")
    } else if m.contains("sana") {
        Some("photo polish=0.72 micro=0.35 desaturate=0.08")
    } else {
        None
    }
}

/// Resolve `name` as a library preset spec, if it is one.
pub fn library_preset(name: &str) -> Option<&'static str> {
    let n = name.trim().to_ascii_lowercase();
    preset_library().iter().find(|(k, _, _)| *k == n).map(|(_, spec, _)| *spec)
}

/// The **base params** for a `--preset` value: a library preset (full spec) or one of the three base
/// presets (`subtle`/`photo`/`painting`). `preset=None` → `subtle`; unknown name → `None`.
pub fn base_params(preset: Option<&str>) -> Option<Params> {
    match preset {
        None => Some(Preset::Subtle.params()),
        Some(name) => library_preset(name).map(from_spec).or_else(|| Preset::parse(name).map(|p| p.params())),
    }
}

/// Parse a compact naturalize **spec** string (for `generate --naturalize` / a scenario `naturalize:`
/// field) into analog [`Params`]: a preset name and/or `key=value` tokens (space/comma separated), where
/// `key` is an analog focus (`vegetation=1`) or a param (`grain=0.3`). Empty / "on" / "true" → the default
/// `subtle` preset. Example: `"photo vegetation=1 sky=0.5 grain=0.35"`.
/// Extract an explicit `paper=N` amount from a naturalize **spec** (RFC QUALITY-5 P3). The paper pass isn't
/// part of the analog [`Params`], so the spec-consuming paths (`generate --naturalize`, scenario, api) read
/// it separately and apply [`paper_texture`]. `None` if absent / unparseable.
pub fn paper_from_spec(spec: &str) -> Option<f32> {
    spec.split([' ', ',', ';'])
        .map(str::trim)
        .find_map(|t| t.strip_prefix("paper=").and_then(|v| v.parse::<f32>().ok()))
}

pub fn from_spec(spec: &str) -> Params {
    let mut base = Preset::Subtle;
    let mut focuses: Vec<(Focus, f32)> = Vec::new();
    let mut overrides: Vec<(String, f32)> = Vec::new();
    for tok in spec.split([' ', ',', ';']).map(str::trim).filter(|t| !t.is_empty()) {
        if let Some((k, v)) = tok.split_once('=') {
            let val: f32 = v.trim().parse().unwrap_or(0.0);
            if let Some(f) = Focus::parse(k) {
                focuses.push((f, val));
            } else {
                overrides.push((k.trim().to_ascii_lowercase(), val));
            }
        } else if let Some(p) = Preset::parse(tok) {
            base = p;
        }
        // bare "on"/"true"/unknown tokens just keep the default preset.
    }
    let mut p = blend_focus(base.params(), &focuses);
    for (k, v) in overrides {
        match k.as_str() {
            "grain" => p.grain = v,
            "aberration" => p.aberration = v,
            "vignette" => p.vignette = v,
            "bloom" => p.bloom = v,
            "desaturate" => p.desaturate = v,
            "warm" => p.warm = v,
            "defocus" => p.defocus = v,
            "polish" => p.polish = v,
            "micro" => p.micro = v,
            _ => {}
        }
    }
    p
}

fn lum(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Deterministic value noise in `[-1,1]` from integer coords (no RNG → same image every run).
fn noise(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bilinear(img: &RgbImage, ch: usize, x: f32, y: f32) -> f32 {
    let (w, h) = (img.width(), img.height());
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: u32, yy: u32| img.get_pixel(xx, yy).0[ch] as f32;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

fn box_blur_gray(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut acc, mut n) = (0.0, 0.0);
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        acc += src[yy as usize * w + xx as usize];
                        n += 1.0;
                    }
                }
            }
            out[y * w + x] = acc / n;
        }
    }
    out
}

/// A weight-free **AI-tell score** in `0..1` (higher = reads more "AI-generated"): the two loudest,
/// cheaply-measurable tells — **oversaturation** (mean HSV saturation) and **texture over-smoothness** (a
/// low high-frequency-to-contrast ratio: AI output is too clean / uniformly smooth). Feeds selection
/// (`rank --ai-tells`) so a batch can be pruned to the most human-looking frames, and lets `naturalize`
/// report the before/after delta.
pub fn ai_tell_score(img: &RgbImage) -> f32 {
    analyze(img).ai_tell
}

/// The decomposed de-slop **analysis** of an image (RFC QUALITY-6 P1) — the drivers behind
/// [`ai_tell_score`], for the `naturalize --report` scorecard.
#[derive(Debug, Clone, Copy)]
pub struct Analysis {
    /// Combined AI-tell score, `0..1` (higher = reads more AI-generated).
    pub ai_tell: f32,
    /// Mean HSV saturation, `0..1` (the oversaturation tell).
    pub saturation: f32,
    /// Texture over-smoothness tell, `0..1` (1 = dead-smooth / too clean).
    pub smoothness_tell: f32,
    /// Global luminance contrast (std/255) — low = washed/muddy.
    pub contrast: f32,
}

/// Decompose the weight-free AI-tell drivers of an image: **oversaturation** (mean HSV saturation) and
/// **texture over-smoothness** (low high-frequency-to-contrast ratio → too clean). Feeds `ai_tell_score`
/// and the `--report` scorecard.
pub fn analyze(img: &RgbImage) -> Analysis {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return Analysis { ai_tell: 0.0, saturation: 0.0, smoothness_tell: 0.0, contrast: 0.0 };
    }
    let sat: f32 = img
        .pixels()
        .map(|p| {
            let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
            let mx = r.max(g).max(b);
            if mx <= 0.0 { 0.0 } else { (mx - r.min(g).min(b)) / mx }
        })
        .sum::<f32>()
        / (w * h) as f32;
    let l: Vec<f32> = img.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    let (mut hf, mut n) = (0.0f32, 0.0f32);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut acc = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    acc += l[((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize];
                }
            }
            hf += (l[y * w + x] - acc / 9.0).abs();
            n += 1.0;
        }
    }
    let hf = hf / n.max(1.0);
    let mean = l.iter().sum::<f32>() / l.len() as f32;
    let std = (l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l.len() as f32).sqrt();
    let ratio = (hf / std.max(1.0)).min(0.5) / 0.5; // 0 (dead smooth) .. 1 (grainy/detailed)
    let smoothness_tell = 1.0 - ratio;
    let ai_tell = (0.6 * sat + 0.4 * smoothness_tell).clamp(0.0, 1.0);
    Analysis { ai_tell, saturation: sat, smoothness_tell, contrast: std / 255.0 }
}

/// Whether a style/medium string names a **wet media** (watercolour / gouache / ink-wash) — triggers
/// auto-paper (QUALITY-5) and the `--paper` recommendation.
pub fn is_wet_media(style: &str) -> bool {
    let s = style.to_ascii_lowercase();
    ["watercolor", "watercolour", "gouache", "ink-wash", "ink wash", "wet-on-wet", "wet on wet"]
        .iter()
        .any(|k| s.contains(k))
}

/// A recommended de-slop **recipe** (the `naturalize` flags to run) from an [`Analysis`] + optional detected
/// medium — the actionable half of the `--report` scorecard.
pub fn recommend(a: &Analysis, medium: Option<&str>) -> Vec<String> {
    let mut r = vec!["--polish 0.7".to_string()]; // polish always helps (WB + levels + vibrance + unsharp)
    if a.saturation > 0.5 {
        r.push("--desaturate 0.2".to_string());
    }
    if a.smoothness_tell > 0.45 {
        r.push("--micro 0.5".to_string());
    }
    if a.contrast < 0.14 {
        r.push("--polish 0.9".to_string()); // washed → lean harder on auto-levels (replaces the 0.7)
        r.remove(0);
    }
    if medium.map(is_wet_media).unwrap_or(false) {
        r.push("--paper 0.6".to_string());
    }
    r
}

/// Which corner a ghost signature sits in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corner {
    BottomRight,
    BottomLeft,
    TopRight,
    TopLeft,
}
impl Corner {
    pub fn parse(s: &str) -> Option<Corner> {
        match s.trim().to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "br" | "bottomright" => Some(Corner::BottomRight),
            "bl" | "bottomleft" => Some(Corner::BottomLeft),
            "tr" | "topright" => Some(Corner::TopRight),
            "tl" | "topleft" => Some(Corner::TopLeft),
            _ => None,
        }
    }
}

/// Weight-free **ghost-signature removal** (RFC QUALITY-1): a faint training-data signature smudge lives in
/// a corner, so dissolve that corner region into its own blurred background (feathered so there's no seam).
/// Scoped to a **foreign** artifact — it never touches plakat's own etch (the L1 pixel mark is spread over
/// the whole frame and survives; L0 is metadata). `strength` in `0..1`.
pub fn designature(src: &RgbImage, corner: Corner, strength: f32) -> RgbImage {
    let (w, h) = (src.width(), src.height());
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return src.clone();
    }
    // corner box ~16% of the short side.
    let box_px = ((w.min(h) as f32) * 0.16) as u32;
    let (bx0, by0) = match corner {
        Corner::BottomRight => (w.saturating_sub(box_px), h.saturating_sub(box_px)),
        Corner::BottomLeft => (0, h.saturating_sub(box_px)),
        Corner::TopRight => (w.saturating_sub(box_px), 0),
        Corner::TopLeft => (0, 0),
    };
    let (bx1, by1) = (bx0 + box_px, by0 + box_px);
    // heavy blur of just the corner region (per channel).
    let r = (box_px / 5).max(3) as i32;
    let mut out = src.clone();
    for ch in 0..3 {
        for y in by0..by1.min(h) {
            for x in bx0..bx1.min(w) {
                let (mut acc, mut n) = (0.0f32, 0.0f32);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx >= 0 && yy >= 0 && (xx as u32) < w && (yy as u32) < h {
                            acc += src.get_pixel(xx as u32, yy as u32).0[ch] as f32;
                            n += 1.0;
                        }
                    }
                }
                let blurred = acc / n;
                // feather: full effect deep in the corner, fading to 0 at the box's inner edges.
                let fx = match corner {
                    Corner::BottomRight | Corner::TopRight => (x - bx0) as f32 / box_px as f32,
                    _ => 1.0 - (x - bx0) as f32 / box_px as f32,
                };
                let fy = match corner {
                    Corner::BottomRight | Corner::BottomLeft => (y - by0) as f32 / box_px as f32,
                    _ => 1.0 - (y - by0) as f32 / box_px as f32,
                };
                let feather = (fx.min(1.0).max(0.0) * fy.min(1.0).max(0.0)) * s;
                let orig = src.get_pixel(x, y).0[ch] as f32;
                out.get_pixel_mut(x, y).0[ch] = (orig * (1.0 - feather) + blurred * feather).round() as u8;
            }
        }
    }
    out
}

/// Apply the naturalize pass. Returns a new image the same size as `src`.
/// Weight-free **quality improvement** ("de-slop") — the part that makes a genuinely *better* picture, not
/// a disguised one. Four honest corrections, each scaled by `strength` (0..1), applied in order:
///   1. **gray-world white balance** — neutralise the AI colour cast (average of the frame → grey),
///   2. **robust auto-levels** — stretch a muddy / washed histogram to true black & white using the 0.5 /
///      99.5 luminance percentiles (ignoring outliers), so contrast reads clean,
///   3. **vibrance** — tame blown-out oversaturation *and* lift dull colours toward a natural mid, for
///      better colour without the plastic AI look,
///   4. **unsharp mask** — crisp the soft AI mush so edges / structure read sharply.
/// Deterministic, no GPU. Returns the improved image; `strength <= 0` is a passthrough.
/// The per-image **colour** parameters fitted by the polish pass (stages 1–3: white-balance gains +
/// auto-level black/white points/scale). Every stage here is *pointwise* in (r,g,b) — so once fitted on
/// an image, the whole colour grade is representable as a 3-D LUT (see [`export_cube_image`]). The
/// spatial unsharp (stage 4) is deliberately NOT part of this — it can't be a colour LUT.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PolishColor {
    s: f32,
    gr: f32,
    gg: f32,
    gb: f32,
    lo: f32,
    hi: f32,
    level_on: bool,
}

impl PolishColor {
    /// Fit the colour stages on `src` at strength `s`: gray-world WB gains from the channel means, then
    /// the 0.5 / 99.5 luminance percentiles (measured *after* WB, as the pass applies them) for auto-levels.
    pub(crate) fn fit(src: &RgbImage, s: f32) -> Self {
        let n = (src.width() as usize * src.height() as usize).max(1) as f32;
        let (mut sr, mut sg, mut sb) = (0.0f32, 0.0f32, 0.0f32);
        for p in src.pixels() {
            sr += p.0[0] as f32;
            sg += p.0[1] as f32;
            sb += p.0[2] as f32;
        }
        let (mr, mg, mb) = (sr / n, sg / n, sb / n);
        let grey = (mr + mg + mb) / 3.0;
        let gain = |m: f32| {
            let raw = if m > 1.0 { grey / m } else { 1.0 };
            let clamped = raw.clamp(0.85, 1.15); // never more than ±15% cast correction
            1.0 + s * (clamped - 1.0)
        };
        let (gr, gg, gb) = (gain(mr), gain(mg), gain(mb));
        // Histogram of WB-adjusted luminance → percentiles (matches the sequential pass exactly).
        let mut hist = [0u32; 256];
        for p in src.pixels() {
            let r = (p.0[0] as f32 * gr).clamp(0.0, 255.0);
            let g = (p.0[1] as f32 * gg).clamp(0.0, 255.0);
            let b = (p.0[2] as f32 * gb).clamp(0.0, 255.0);
            hist[lum(r, g, b).clamp(0.0, 255.0) as usize] += 1;
        }
        let total = n;
        let pct = |target: f32| -> f32 {
            let mut acc = 0.0f32;
            for (v, &c) in hist.iter().enumerate() {
                acc += c as f32;
                if acc / total >= target {
                    return v as f32;
                }
            }
            255.0
        };
        let (lo, hi) = (pct(0.005), pct(0.995));
        Self { s, gr, gg, gb, lo, hi, level_on: hi - lo > 8.0 }
    }

    /// Apply the pointwise colour stages 1–3 (WB → auto-levels → vibrance) to one pixel (0..255 in/out).
    pub(crate) fn map_pixel(&self, mut r: f32, mut g: f32, mut b: f32) -> [f32; 3] {
        // 1. white balance
        r = (r * self.gr).clamp(0.0, 255.0);
        g = (g * self.gg).clamp(0.0, 255.0);
        b = (b * self.gb).clamp(0.0, 255.0);
        // 2. ratio-preserving auto-levels (same luminance gain to all three channels)
        if self.level_on {
            let scale = 255.0 / (self.hi - self.lo);
            let l = lum(r, g, b);
            if l > 1.0 {
                let stretched = ((l - self.lo) * scale).clamp(0.0, 255.0);
                let gain = 1.0 + self.s * (stretched / l - 1.0);
                r = (r * gain).clamp(0.0, 255.0);
                g = (g * gain).clamp(0.0, 255.0);
                b = (b * gain).clamp(0.0, 255.0);
            }
        }
        // 3. vibrance — pull saturation toward a natural mid, around per-pixel luminance (hue-preserving)
        let l = lum(r, g, b);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let sat = if mx > 1.0 { (mx - mn) / mx } else { 0.0 };
        let factor = 1.0 + self.s * 0.5 * (0.45 - sat);
        r = (l + (r - l) * factor).clamp(0.0, 255.0);
        g = (l + (g - l) * factor).clamp(0.0, 255.0);
        b = (l + (b - l) * factor).clamp(0.0, 255.0);
        [r, g, b]
    }
}

pub fn polish(src: &RgbImage, strength: f32) -> RgbImage {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return src.clone();
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    // Stages 1–3 are the fitted, pointwise colour grade (shared with the per-image LUT export).
    let pc = PolishColor::fit(src, s);
    let mut r = vec![0.0f32; w * h];
    let mut g = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for (i, px) in src.pixels().enumerate() {
        let [nr, ng, nb] = pc.map_pixel(px.0[0] as f32, px.0[1] as f32, px.0[2] as f32);
        r[i] = nr;
        g[i] = ng;
        b[i] = nb;
    }

    // 4. unsharp mask — sharpen against a blurred luminance so soft AI mush gains edge definition. This
    //    is SPATIAL (depends on neighbours) → not part of the colour LUT.
    let luma: Vec<f32> = (0..r.len()).map(|i| lum(r[i], g[i], b[i])).collect();
    let blur = box_blur_gray(&luma, w, h, 2);
    let amount = s * 0.8;
    for i in 0..r.len() {
        let hp = luma[i] - blur[i]; // high-frequency detail
        let add = hp * amount;
        for ch in [&mut r, &mut g, &mut b] {
            ch[i] = (ch[i] + add).clamp(0.0, 255.0);
        }
    }

    let mut out = RgbImage::new(w as u32, h as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        *px = Rgb([r[i] as u8, g[i] as u8, b[i] as u8]);
    }
    out
}

/// A square erosion (min) or dilation (max) — the morphology primitive for the wire detector.
fn box_morph(src: &[f32], w: usize, h: usize, r: i32, dilate: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = if dilate { f32::MIN } else { f32::MAX };
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        let s = src[yy as usize * w + xx as usize];
                        v = if dilate { v.max(s) } else { v.min(s) };
                    }
                }
            }
            out[y * w + x] = v;
        }
    }
    out
}

/// Weight-free **thin-line ("wire") mask** — detect floating catenary / power lines that OWL-ViT (an
/// *object* detector) is blind to. A morphological top-hat finds thin high-contrast ridges (dark OR bright)
/// **only where the surrounding region is smooth** (sky / plain wall), so it targets the floating-wire tell
/// without eating the dense tram / building / foliage structure below. `sensitivity` in `0..1` (higher =
/// catch fainter wires). Returns a white-on-black mask (255 = wire → inpaint away).
pub fn wire_mask(img: &RgbImage, sensitivity: f32) -> GrayImage {
    let s = sensitivity.clamp(0.0, 1.0);
    let (w, h) = (img.width() as usize, img.height() as usize);
    let gray: Vec<f32> = img.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    // per-pixel "sky-like" flag — bluish OR bright (excludes the warm cobblestones / red tram / lit
    // shopfronts). Blurred so a dark wire pixel inherits its bright-sky surroundings.
    let sky_raw: Vec<f32> = img
        .pixels()
        .map(|p| {
            let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
            let bright = lum(r, g, b) > 150.0;
            let bluish = b > r + 6.0 && b > 70.0;
            if bright || bluish { 1.0 } else { 0.0 }
        })
        .collect();
    let sky = box_blur_gray(&sky_raw, w, h, 4);

    // closing (dilate→erode) fills thin DARK lines; opening (erode→dilate) removes thin BRIGHT lines.
    let r = 2;
    let closing = box_morph(&box_morph(&gray, w, h, r, true), w, h, r, false);
    let opening = box_morph(&box_morph(&gray, w, h, r, false), w, h, r, true);

    // Smoothness gate from the CLOSING (dark wire already filled) — local std over a wider window; low std
    // = uniform background (sky), which is where floating wires live.
    let bg = &closing;
    let mut mask = GrayImage::from_pixel(w as u32, h as u32, Luma([0]));
    let gate_r = 4i32;
    let thresh = 8.0 + (1.0 - s) * 26.0; // higher sensitivity → lower ridge threshold
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let bt = (closing[i] - gray[i]).max(0.0); // dark-line response
            let wt = (gray[i] - opening[i]).max(0.0); // bright-line response
            let resp = bt.max(wt);
            if resp < thresh {
                continue;
            }
            // local std of the background over gate_r → smoothness.
            let (mut m, mut m2, mut n) = (0.0f32, 0.0f32, 0.0f32);
            for dy in -gate_r..=gate_r {
                for dx in -gate_r..=gate_r {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        let v = bg[yy as usize * w + xx as usize];
                        m += v;
                        m2 += v * v;
                        n += 1.0;
                    }
                }
            }
            let std = (m2 / n - (m / n) * (m / n)).max(0.0).sqrt();
            // accept only thin lines set against SMOOTH, SKY-LIKE surroundings — this is what separates a
            // floating wire from cobblestone grout / tram frames / lit shopfront edges.
            if std < 16.0 && sky[i] > 0.55 {
                mask.put_pixel(x as u32, y as u32, Luma([255]));
            }
        }
    }
    // ELONGATION filter — the key discriminator between a wire and pervasive ink speckle: keep only
    // connected components that are LONG (bbox diagonal) and THIN (sparse fill of their bbox). This drops
    // the short splatter/foliage/cobblestone marks that a tophat also fires on.
    let mut bin: Vec<bool> = mask.pixels().map(|p| p.0[0] > 127).collect();
    let diag = ((w * w + h * h) as f32).sqrt();
    let min_len = 0.18 * diag; // a wire spans a good fraction of the frame
    let mut visited = vec![false; w * h];
    let mut keep = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if !bin[start] || visited[start] {
            continue;
        }
        // flood-fill this component (8-connectivity), tracking bbox + size.
        stack.clear();
        stack.push(start);
        visited[start] = true;
        let mut comp: Vec<usize> = Vec::new();
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        while let Some(p) = stack.pop() {
            comp.push(p);
            let (px, py) = (p % w, p / w);
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (px as i32 + dx, py as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let np = ny as usize * w + nx as usize;
                        if bin[np] && !visited[np] {
                            visited[np] = true;
                            stack.push(np);
                        }
                    }
                }
            }
        }
        let bw = (x1 - x0 + 1) as f32;
        let bh = (y1 - y0 + 1) as f32;
        let bbox_diag = (bw * bw + bh * bh).sqrt();
        let fill = comp.len() as f32 / (bw * bh).max(1.0);
        // long AND thin (low bbox fill → a line, not a blob).
        if bbox_diag >= min_len && fill < 0.25 {
            for &p in &comp {
                keep[p] = true;
            }
        }
    }
    bin = keep;

    // dilate the surviving lines by 1px so the inpaint fully covers the wire + its anti-aliased edge.
    let m: Vec<f32> = bin.iter().map(|&b| if b { 255.0 } else { 0.0 }).collect();
    let grown = box_morph(&m, w, h, 1, true);
    for (idx, px) in mask.pixels_mut().enumerate() {
        px.0[0] = if grown[idx] > 127.0 { 255 } else { 0 };
    }
    mask
}

/// Weight-free **micro-texture** — the fix for plastic AI skin (and any unnaturally smooth surface):
/// real skin has pores and micro-wrinkles, never a perfect gradient. Adds fine high-frequency detail
/// **only where the image is too smooth** (gated by local variance, so already-textured regions like hair
/// or fabric are left alone) and **only in mid-tones** (where skin lives; blown highlights / deep shadows
/// stay clean). `amount` in `0..1`. Deterministic (hash noise). Returns the textured image.
pub fn micro_texture(src: &RgbImage, amount: f32) -> RgbImage {
    let a = amount.clamp(0.0, 2.0);
    if a <= 0.0 {
        return src.clone();
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    let luma: Vec<f32> = src.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    // local 3×3 std-dev → smoothness gate (low variance = plastic → full texture).
    let blur = box_blur_gray(&luma, w, h, 1);
    let mut out = src.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // local variance estimate from the |detail| against the 3×3 mean.
            let mut var = 0.0f32;
            let mut n = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        let d = luma[yy as usize * w + xx as usize] - blur[i];
                        var += d * d;
                        n += 1.0;
                    }
                }
            }
            let std = (var / n.max(1.0)).sqrt();
            // smoothness: 1 where flat (std≈0), →0 by std≈12 (already-detailed regions).
            let smooth = (1.0 - (std / 12.0)).clamp(0.0, 1.0);
            // mid-tone weight: a bump peaking around L≈150 (skin), fading at shadows/highlights.
            let l = luma[i];
            let mid = (1.0 - ((l - 150.0) / 110.0).powi(2)).clamp(0.0, 1.0);
            // fine two-octave pore noise (per-pixel + a half-offset octave).
            let fine = 0.7 * noise(x as u32, y as u32) + 0.3 * noise(x as u32 ^ 0x5bd1, y as u32 ^ 0x9f37);
            let delta = a * smooth * mid * fine * 9.0; // ±~9 luma at full weight
            let px = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px.0[c] = (px.0[c] as f32 + delta).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Weight-free **watercolor-paper / pigment authenticity** (RFC QUALITY-4) — the fix for the "simulated
/// media" tell (procedural bleeds/splatters that don't behave like real wet-on-wet pigment). Models three
/// physical effects, each scaled by `amount` (0..1), applied only where there IS pigment (washes, not the
/// bare paper), so a photo left alone stays a photo:
///   1. **paper tooth** — a cold-press height field; pigment settles into the valleys (darker) and skips
///      the peaks (lighter), so a flat wash gains real granular tooth,
///   2. **granulation** — fine pigment speckle correlated with pigment density,
///   3. **edge pooling** — pigment migrates to the rim of a wash as it dries → darker edges (the signature
///      wet-on-wet cauliflower/backrun look).
/// Deterministic (hash noise). Apply to genuine watercolor/ink-wash art; opt-in.
pub fn paper_texture(src: &RgbImage, amount: f32) -> RgbImage {
    let a = amount.clamp(0.0, 2.5);
    if a <= 0.0 {
        return src.clone();
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    // white noise → two blurred octaves = a smooth cold-press paper height field in ~[-1,1].
    let raw: Vec<f32> = (0..w * h).map(|i| noise((i % w) as u32, (i / w) as u32)).collect();
    let fine = box_blur_gray(&raw, w, h, 1); // ~3px tooth
    let coarse = box_blur_gray(&raw, w, h, 5); // broad unevenness / cockling
    // renormalise the combined field to ~unit amplitude (blur shrinks it).
    let paper: Vec<f32> = (0..w * h).map(|i| 0.6 * fine[i] + 0.4 * coarse[i]).collect();
    let pstd = (paper.iter().map(|v| v * v).sum::<f32>() / paper.len() as f32).sqrt().max(1e-3);

    let luma: Vec<f32> = src.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    // edge magnitude of luminance → where pigment pools (wash boundaries).
    let blur_l = box_blur_gray(&luma, w, h, 2);

    let mut out = src.clone();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            // "pigment density" — 0 on bare white paper, →1 in dark/saturated washes.
            let px = src.get_pixel(x as u32, y as u32).0;
            let mx = px[0].max(px[1]).max(px[2]) as f32;
            let mn = px[0].min(px[1]).min(px[2]) as f32;
            let sat = if mx > 1.0 { (mx - mn) / mx } else { 0.0 };
            let darkness = (1.0 - luma[i] / 255.0).clamp(0.0, 1.0);
            // any non-white wash carries pigment (a light-blue sky wash still granulates), so give it a
            // floor and weight saturation strongly.
            let pigment = (0.45 * darkness + 0.8 * sat + 0.12).clamp(0.0, 1.0);
            if darkness < 0.04 && sat < 0.05 {
                continue; // leave bare white paper (and flat photos) alone
            }
            // 1. paper tooth: valleys hold pigment (darker), peaks resist (lighter).
            let tooth = (paper[i] / pstd) * a * 22.0 * pigment;
            // 2. granulation: fine speckle in the pigment, scaled by density.
            let gran = noise(x as u32 ^ 0x2f1b, y as u32 ^ 0x8c3d) * a * 12.0 * pigment;
            // 3. edge pooling: darken where this pixel is darker than its neighbourhood edge (wash rim).
            let edge = (blur_l[i] - luma[i]).max(0.0); // inside-edge (this pixel darker than blur)
            let pool = -(edge / 255.0) * a * 50.0 * pigment;
            let delta = -tooth + gran + pool; // tooth valleys darken → subtract the signed field
            let opx = out.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                opx.0[c] = (opx.0[c] as f32 + delta).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Composite per-region de-slop (RFC QUALITY-6 P3): apply `base` params over the whole frame, then blend
/// each region's own [`apply`] result in weighted by its (feathered) mask — so a frame with several subjects
/// (sky + people + foliage) gets each subject its own profile. Later regions win where masks overlap.
pub fn apply_with_regions(src: &RgbImage, base: &Params, regions: &[(GrayImage, Params)]) -> RgbImage {
    let mut out = apply(src, base);
    let (w, h) = (out.width(), out.height());
    for (mask, params) in regions {
        if mask.width() != w || mask.height() != h {
            continue;
        }
        let region = apply(src, params);
        for y in 0..h {
            for x in 0..w {
                let a = mask.get_pixel(x, y).0[0] as f32 / 255.0;
                if a <= 0.0 {
                    continue;
                }
                let (o, r) = (out.get_pixel_mut(x, y), region.get_pixel(x, y));
                for c in 0..3 {
                    o.0[c] = (o.0[c] as f32 * (1.0 - a) + r.0[c] as f32 * a).round() as u8;
                }
            }
        }
    }
    out
}

/// A feathered white rectangle mask (normalized `0..1` coords) for [`apply_with_regions`] — soft edges so
/// the region seam is invisible.
pub fn feathered_rect(w: u32, h: u32, x0: f32, y0: f32, x1: f32, y1: f32, feather_px: f32) -> GrayImage {
    let (px0, py0) = (x0.clamp(0.0, 1.0) * w as f32, y0.clamp(0.0, 1.0) * h as f32);
    let (px1, py1) = (x1.clamp(0.0, 1.0) * w as f32, y1.clamp(0.0, 1.0) * h as f32);
    let f = feather_px.max(1.0);
    let mut m = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            // signed distance inside the rectangle (positive inside), clamped through the feather band.
            let dx = (x as f32 - px0).min(px1 - x as f32);
            let dy = (y as f32 - py0).min(py1 - y as f32);
            let d = dx.min(dy);
            let a = ((d / f) + 0.0).clamp(0.0, 1.0);
            m.put_pixel(x, y, Luma([(a * 255.0) as u8]));
        }
    }
    m
}

/// A weight-free **sky mask** (RFC QUALITY-6 P3): the smooth, bluish/bright region in the upper frame —
/// where AI-sky banding lives. Feathered.
pub fn sky_mask(img: &RgbImage) -> GrayImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let luma: Vec<f32> = img.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
    let blur = box_blur_gray(&luma, w, h, 3);
    let mut raw = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let p = img.get_pixel(x as u32, y as u32).0;
            let (r, _g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
            let bright = luma[i] > 150.0;
            let bluish = b > r + 6.0 && b > 70.0;
            let smooth = (luma[i] - blur[i]).abs() < 12.0;
            // fade the vertical prior: full weight at the top, ~0 by 65% down.
            let vfade = (1.0 - (y as f32 / h as f32) / 0.65).clamp(0.0, 1.0);
            raw[i] = if (bright || bluish) && smooth { vfade } else { 0.0 };
        }
    }
    let soft = box_blur_gray(&raw, w, h, 4); // feather
    let mut m = GrayImage::new(w as u32, h as u32);
    for (i, px) in m.pixels_mut().enumerate() {
        px.0[0] = (soft[i].clamp(0.0, 1.0) * 255.0) as u8;
    }
    m
}

/// The **fixed film grade** (desaturate-toward-luma + warm shadow lift) as a pure per-pixel colour function
/// — mirrors [`apply`]'s grade step, minus the spatial vignette/grain. This is the LUT-able part of
/// naturalize. RGB in `0..255`. (RFC QUALITY-7 P2.)
pub fn grade_pixel(c: [f32; 3], desaturate: f32, warm: f32) -> [f32; 3] {
    let l = lum(c[0], c[1], c[2]);
    let mut o = c;
    for v in o.iter_mut() {
        *v = *v * (1.0 - desaturate) + l * desaturate;
    }
    let shadow = (1.0 - l / 255.0).clamp(0.0, 1.0);
    o[0] += warm * 14.0 * shadow;
    o[2] -= warm * 10.0 * shadow;
    [o[0].clamp(0.0, 255.0), o[1].clamp(0.0, 255.0), o[2].clamp(0.0, 255.0)]
}

/// Export the fixed grade as a standard Adobe/Resolve **`.cube`** 3-D LUT (values `0..1`), so the colour
/// grade can be applied in a video editor. Honest scope: only the *fixed* transform — the polish
/// white-balance/auto-levels are per-image adaptive and NOT captured here. `size` clamped to `2..=64`.
pub fn export_cube(desaturate: f32, warm: f32, size: usize) -> String {
    let size = size.clamp(2, 64);
    let mut s = format!(
        "# plakat naturalize grade LUT (desaturate={desaturate:.3}, warm={warm:.3})\n# WB/auto-levels are per-image and NOT in this static LUT.\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n"
    );
    let d = (size - 1) as f32;
    // .cube ordering: red index varies fastest, then green, then blue.
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let inp = [r as f32 / d * 255.0, g as f32 / d * 255.0, b as f32 / d * 255.0];
                let o = grade_pixel(inp, desaturate, warm);
                s.push_str(&format!("{:.6} {:.6} {:.6}\n", o[0] / 255.0, o[1] / 255.0, o[2] / 255.0));
            }
        }
    }
    s
}

/// Export the **per-image adaptive** colour grade as a `.cube` 3-D LUT (RFC QUALITY-9 P1): fit the polish
/// colour stages (white-balance + auto-levels + vibrance) on `src`, then bake *those fitted, pointwise*
/// scalars — plus the fixed film grade (`desaturate` + `warm`) — onto the identity lattice. Unlike
/// [`export_cube`], this captures the WB/auto-levels the fixed grade could not (closing the 6.16 limit).
/// Honest carve-out: the polish **unsharp** (and micro-texture) are *spatial* — they are not colour LUTs
/// and are excluded; this is the full **colour** grade for `src`, not the sharpening. `size` → `2..=64`.
pub fn export_cube_image(src: &RgbImage, polish: f32, desaturate: f32, warm: f32, size: usize) -> String {
    let size = size.clamp(2, 64);
    let ps = polish.clamp(0.0, 1.0);
    let pc = PolishColor::fit(src, ps);
    let mut s = format!(
        "# plakat naturalize per-image colour LUT (polish={ps:.3}, desaturate={desaturate:.3}, warm={warm:.3})\n# per-image WB+auto-levels+vibrance are baked; spatial unsharp/micro are NOT (colour only).\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n"
    );
    let d = (size - 1) as f32;
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let (ir, ig, ib) = (r as f32 / d * 255.0, g as f32 / d * 255.0, b as f32 / d * 255.0);
                // polish colour stages (only when strength > 0), then the fixed grade.
                let c = if ps > 0.0 { pc.map_pixel(ir, ig, ib) } else { [ir, ig, ib] };
                let o = grade_pixel(c, desaturate, warm);
                s.push_str(&format!("{:.6} {:.6} {:.6}\n", o[0] / 255.0, o[1] / 255.0, o[2] / 255.0));
            }
        }
    }
    s
}

pub fn apply(src: &RgbImage, p: &Params) -> RgbImage {
    // 0. QUALITY IMPROVEMENT first — make a better picture (colour + detail), THEN any analog look.
    let src = if p.polish > 0.0 { polish(src, p.polish) } else { src.clone() };
    // 0b. micro-texture — break plastic smoothness (skin pores / micro-wrinkles) before the film look.
    let src = if p.micro > 0.0 { micro_texture(&src, p.micro) } else { src };
    let src = &src;
    let (w, h) = (src.width(), src.height());
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let maxr = (cx * cx + cy * cy).sqrt().max(1.0);

    // 1. chromatic aberration — R outward / B inward along the radius, ∝ r².
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr;
            let shift = p.aberration * r * r * 10.0;
            let (ux, uy) = if r > 1e-4 { (dx / (r * maxr), dy / (r * maxr)) } else { (0.0, 0.0) };
            let rr = if p.aberration > 0.0 { bilinear(src, 0, x as f32 + ux * shift, y as f32 + uy * shift) } else { src.get_pixel(x, y).0[0] as f32 };
            let gg = src.get_pixel(x, y).0[1] as f32;
            let bb = if p.aberration > 0.0 { bilinear(src, 2, x as f32 - ux * shift, y as f32 - uy * shift) } else { src.get_pixel(x, y).0[2] as f32 };
            out.put_pixel(x, y, Rgb([rr.clamp(0.0, 255.0) as u8, gg as u8, bb.clamp(0.0, 255.0) as u8]));
        }
    }

    // 2. radial defocus — blend a blurred copy in, weighted by r² (sharp centre, soft edges).
    if p.defocus > 0.0 {
        let luma: Vec<f32> = out.pixels().map(|px| lum(px.0[0] as f32, px.0[1] as f32, px.0[2] as f32)).collect();
        let _ = &luma; // keep the per-channel blur below independent
        for ch in 0..3 {
            let chan: Vec<f32> = out.pixels().map(|px| px.0[ch] as f32).collect();
            let blur = box_blur_gray(&chan, w as usize, h as usize, 2);
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                    let r = (dx * dx + dy * dy).sqrt() / maxr;
                    let mix = (p.defocus * r * r).clamp(0.0, 1.0);
                    let i = (y * w + x) as usize;
                    let v = chan[i] * (1.0 - mix) + blur[i] * mix;
                    out.get_pixel_mut(x, y).0[ch] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    // 3. bloom — screen a blurred highlight mask back in.
    if p.bloom > 0.0 {
        let hi: Vec<f32> = out.pixels().map(|px| ((lum(px.0[0] as f32, px.0[1] as f32, px.0[2] as f32) - 200.0).max(0.0)) / 55.0).collect();
        let blur = box_blur_gray(&hi, w as usize, h as usize, 4);
        for y in 0..h {
            for x in 0..w {
                let add = (blur[(y * w + x) as usize] * p.bloom * 60.0).min(60.0);
                let px = out.get_pixel_mut(x, y);
                for c in 0..3 {
                    px.0[c] = (px.0[c] as f32 + add).min(255.0) as u8;
                }
            }
        }
    }

    // 4. per-pixel: grade (desaturate + warm lift) → vignette → luminance-weighted grain.
    for y in 0..h {
        for x in 0..w {
            let px = out.get_pixel_mut(x, y);
            let mut c = [px.0[0] as f32, px.0[1] as f32, px.0[2] as f32];
            let l = lum(c[0], c[1], c[2]);
            for v in c.iter_mut() {
                *v = *v * (1.0 - p.desaturate) + l * p.desaturate;
            }
            let shadow = (1.0 - l / 255.0).clamp(0.0, 1.0);
            c[0] += p.warm * 14.0 * shadow;
            c[2] -= p.warm * 10.0 * shadow;
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr;
            let vig = 1.0 - p.vignette * r * r;
            for v in c.iter_mut() {
                *v *= vig;
            }
            let g_amp = p.grain * 22.0 * (0.4 + 0.6 * shadow);
            c[0] += noise(x, y) * g_amp;
            c[1] += noise(x + 7, y + 3) * g_amp * 0.9;
            c[2] += noise(x + 13, y + 11) * g_amp * 0.9;
            px.0 = [c[0].clamp(0.0, 255.0) as u8, c[1].clamp(0.0, 255.0) as u8, c[2].clamp(0.0, 255.0) as u8];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic over-clean, over-saturated image: a flat saturated gradient + a uniform high-freq tile.
    fn ai_clean() -> RgbImage {
        let (w, h) = (200u32, 200u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / w as f32;
                let mut g = 40.0 + (1.0 - t) * 120.0;
                if x < w / 2 && y < h / 2 {
                    g += (((x / 3 + y / 3) % 2) as f32) * 24.0 - 12.0; // uniform checker
                }
                img.put_pixel(x, y, Rgb([(20.0 + t * 40.0) as u8, g as u8, (200.0 - t * 60.0) as u8]));
            }
        }
        img
    }

    fn mean_sat(img: &RgbImage) -> f32 {
        let s: f32 = img.pixels().map(|p| {
            let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
            let mx = r.max(g).max(b);
            if mx <= 0.0 { 0.0 } else { (mx - r.min(g).min(b)) / mx }
        }).sum();
        s / (img.width() * img.height()) as f32
    }

    fn flat_var(img: &RgbImage) -> f32 {
        // high-freq energy in a flat sub-region (grain raises it).
        let (w, _h) = (img.width() as i32, img.height() as i32);
        let l = |x: i32, y: i32| { let p = img.get_pixel(x as u32, y as u32); lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32) };
        let (mut e, mut n) = (0.0f32, 0.0f32);
        for y in 120..180 {
            for x in 120..180 {
                let mut acc = 0.0;
                for dy in -1..=1 { for dx in -1..=1 { acc += l(x + dx, y + dy); } }
                let d = l(x, y) - acc / 9.0;
                e += d * d; n += 1.0; let _ = w;
            }
        }
        e / n
    }

    fn corr(a: &RgbImage, b: &RgbImage) -> f32 {
        let la: Vec<f32> = a.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
        let lb: Vec<f32> = b.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
        let (ma, mb) = (la.iter().sum::<f32>() / la.len() as f32, lb.iter().sum::<f32>() / lb.len() as f32);
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for i in 0..la.len() { let (x, y) = (la[i] - ma, lb[i] - mb); num += x * y; da += x * x; db += y * y; }
        num / (da.sqrt() * db.sqrt()).max(1e-6)
    }

    #[test]
    fn photo_preset_degrades_fingerprint_but_preserves_structure() {
        let src = ai_clean();
        let out = apply(&src, &Preset::Photo.params());
        assert!(mean_sat(&out) < mean_sat(&src) - 0.01, "saturation drops");
        assert!(flat_var(&out) > flat_var(&src) + 1.0, "grain raises flat-region variance");
        assert!(corr(&src, &out) > 0.9, "structure preserved");
    }

    #[test]
    fn focus_blends_toward_the_subject_profile() {
        let base = Preset::Subtle.params();
        // no focus → unchanged.
        let same = blend_focus(base, &[]);
        assert_eq!(to_arr(&same), to_arr(&base));
        // vegetation focus raises grain (breaks the cloud-foliage mush); people focus raises desaturation
        // (waxy skin) without adding much aberration.
        let veg = blend_focus(base, &[(Focus::Vegetation, 1.0)]);
        assert!(veg.grain > base.grain, "vegetation raises grain");
        let ppl = blend_focus(base, &[(Focus::People, 1.0)]);
        assert!(ppl.desaturate > base.desaturate && ppl.aberration <= base.aberration + 1e-3, "people → desaturate up, aberration not up");
    }

    #[test]
    fn ai_tell_score_drops_after_naturalize() {
        let src = ai_clean();
        let before = ai_tell_score(&src);
        let after = ai_tell_score(&apply(&src, &Preset::Photo.params()));
        assert!(after < before, "naturalize lowers the AI-tell score ({before:.3} → {after:.3})");
    }

    #[test]
    fn wire_mask_catches_thin_line_on_smooth_bg_not_texture() {
        // smooth bright "sky" with a 1px dark diagonal "wire" + a textured lower band.
        let (w, h) = (80u32, 80u32);
        let mut img = RgbImage::from_pixel(w, h, Rgb([200, 205, 215]));
        for i in 0..80u32 {
            let x = i.min(w - 1);
            let y = (i / 2).min(h - 1);
            img.put_pixel(x, y, Rgb([40, 40, 45])); // thin dark wire across the smooth sky
        }
        // textured lower band (should NOT be flagged despite high local contrast).
        for y in 60..80 {
            for x in 0..80 {
                let v = (((x * 7 + y * 13) % 100) as u8).saturating_add(80);
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let mask = wire_mask(&img, 0.6);
        let on_wire = (0..40u32).filter(|&i| mask.get_pixel(i, i / 2).0[0] > 127).count();
        assert!(on_wire > 10, "wire pixels flagged (got {on_wire})");
        // the textured band is not a thin line in smooth surroundings → few/no hits.
        let in_texture = (60..80u32).flat_map(|y| (0..80u32).map(move |x| (x, y))).filter(|&(x, y)| mask.get_pixel(x, y).0[0] > 127).count();
        assert!(in_texture < 40, "textured region mostly ignored (got {in_texture})");
    }

    #[test]
    fn polish_neutralises_a_cast_and_stretches_contrast() {
        // a washed, blue-cast, low-contrast image (all luminance in a narrow mid band).
        let (w, h) = (64u32, 64u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / w as f32;
                // narrow 110..150 luminance band + a strong blue cast (B ≫ R).
                let base = 110.0 + t * 40.0;
                img.put_pixel(x, y, Rgb([(base * 0.75) as u8, base as u8, (base * 1.25).min(255.0) as u8]));
            }
        }
        let before_contrast = {
            let l: Vec<f32> = img.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
            let m = l.iter().sum::<f32>() / l.len() as f32;
            (l.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / l.len() as f32).sqrt()
        };
        let out = polish(&img, 1.0);
        // white balance pulls the channel means together (cast neutralised).
        let ch_mean = |im: &RgbImage, c: usize| im.pixels().map(|p| p.0[c] as f32).sum::<f32>() / (im.width() * im.height()) as f32;
        let spread = |im: &RgbImage| (ch_mean(im, 0) - ch_mean(im, 2)).abs();
        assert!(spread(&out) < spread(&img), "white balance narrows the R/B cast ({:.1} → {:.1})", spread(&img), spread(&out));
        // auto-levels widens the luminance contrast.
        let after_contrast = {
            let l: Vec<f32> = out.pixels().map(|p| lum(p.0[0] as f32, p.0[1] as f32, p.0[2] as f32)).collect();
            let m = l.iter().sum::<f32>() / l.len() as f32;
            (l.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / l.len() as f32).sqrt()
        };
        assert!(after_contrast > before_contrast * 1.3, "auto-levels lifts contrast ({before_contrast:.1} → {after_contrast:.1})");
        // deterministic + passthrough at 0.
        assert_eq!(polish(&img, 0.0).into_raw(), img.clone().into_raw(), "strength 0 is a passthrough");
        assert_eq!(polish(&img, 1.0).into_raw(), out.into_raw(), "polish is deterministic");
    }

    #[test]
    fn designature_dissolves_the_corner_only() {
        let mut img = ai_clean();
        // a faint thin "signature" stroke in the bottom-right (like a real ghost mark).
        for x in 172..196 {
            img.put_pixel(x, 190, Rgb([15, 15, 15]));
            img.put_pixel(x, 191, Rgb([20, 20, 20]));
        }
        // local std-dev over a region (measures how much the stroke stands out).
        let std = |im: &RgbImage, x0: u32, y0: u32| -> f32 {
            let vals: Vec<f32> = (y0..y0 + 20).flat_map(|y| (x0..x0 + 20).map(move |x| (x, y))).map(|(x, y)| lum(im.get_pixel(x, y).0[0] as f32, im.get_pixel(x, y).0[1] as f32, im.get_pixel(x, y).0[2] as f32)).collect();
            let m = vals.iter().sum::<f32>() / vals.len() as f32;
            (vals.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / vals.len() as f32).sqrt()
        };
        let cleaned = designature(&img, Corner::BottomRight, 0.95);
        assert!(std(&cleaned, 176, 180) < std(&img, 176, 180) * 0.8, "signature stroke dissolved (variance dropped)");
        assert_eq!(cleaned.get_pixel(5, 5), img.get_pixel(5, 5), "opposite corner untouched");
    }

    #[test]
    fn from_spec_parses_preset_focuses_and_overrides() {
        // bare / unknown → subtle default.
        assert_eq!(to_arr(&from_spec("")), to_arr(&Preset::Subtle.params()));
        // preset + combined focuses + a param override.
        let p = from_spec("photo vegetation=1 sky=1 grain=0.5");
        assert_eq!(p.grain, 0.5, "explicit grain override wins");
        let base_photo = blend_focus(Preset::Photo.params(), &[(Focus::Vegetation, 1.0), (Focus::Sky, 1.0)]);
        // aberration/vignette come from the blended photo+focuses (grain was overridden).
        assert!((p.aberration - base_photo.aberration).abs() < 1e-4);
    }

    #[test]
    fn presets_parse_and_apply_is_deterministic() {
        assert_eq!(Preset::parse("Photo"), Some(Preset::Photo));
        assert_eq!(Preset::parse("painting"), Some(Preset::Painting));
        assert!(Preset::parse("vintage").is_none(), "no retro/vintage preset — naturalize aims at realism");
        assert!(Preset::parse("bogus").is_none());
        let src = ai_clean();
        let a = apply(&src, &Preset::Subtle.params());
        let b = apply(&src, &Preset::Subtle.params());
        assert!(a.pixels().zip(b.pixels()).all(|(x, y)| x == y), "deterministic (no RNG)");
    }

    #[test]
    fn per_image_lut_captures_wb_the_fixed_lut_cannot() {
        // A blue-cast image → the per-image polish must fit a WB correction the fixed grade lacks.
        let mut cast = RgbImage::new(64, 64);
        for (x, _y, px) in cast.enumerate_pixels_mut() {
            *px = Rgb([60, 90, (150 + x % 40) as u8]);
        }
        let fixed = export_cube(0.1, 0.05, 9);
        let per_img = export_cube_image(&cast, 0.8, 0.1, 0.05, 9);
        assert_ne!(fixed, per_img, "per-image LUT must differ from the fixed grade (WB is captured)");
        // Well-formed: size³ colour rows + header, values parse in 0..=1.
        let rows: Vec<&str> = per_img.lines().filter(|l| !l.starts_with('#') && !l.contains("LUT_3D_SIZE") && !l.starts_with("DOMAIN")).filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(rows.len(), 9 * 9 * 9, "one entry per lattice point");
        for row in &rows {
            for tok in row.split_whitespace() {
                let v: f32 = tok.parse::<f32>().expect("cube value parses");
                assert!((0.0..=1.0).contains(&v), "cube values in 0..=1");
            }
        }
        // The LUT's colour stages must match polish's stages 1–3 at the identity nodes (no interpolation
        // error at a node): map_pixel then grade == the exported row.
        let pc = PolishColor::fit(&cast, 0.8);
        let expect = grade_pixel(pc.map_pixel(0.0, 0.0, 0.0), 0.1, 0.05); // node (0,0,0)
        let first = rows[0].split_whitespace().map(|t| t.parse::<f32>().unwrap() * 255.0).collect::<Vec<_>>();
        for k in 0..3 {
            assert!((first[k] - expect[k]).abs() < 0.5, "node value matches map_pixel∘grade");
        }
    }
}
