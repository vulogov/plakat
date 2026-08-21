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

use image::{Rgb, RgbImage};

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

/// Parse a compact naturalize **spec** string (for `generate --naturalize` / a scenario `naturalize:`
/// field) into analog [`Params`]: a preset name and/or `key=value` tokens (space/comma separated), where
/// `key` is an analog focus (`vegetation=1`) or a param (`grain=0.3`). Empty / "on" / "true" → the default
/// `subtle` preset. Example: `"photo vegetation=1 sky=0.5 grain=0.35"`.
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
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    // oversaturation
    let sat: f32 = img
        .pixels()
        .map(|p| {
            let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
            let mx = r.max(g).max(b);
            if mx <= 0.0 { 0.0 } else { (mx - r.min(g).min(b)) / mx }
        })
        .sum::<f32>()
        / (w * h) as f32;
    // over-smoothness: high-frequency energy vs global contrast. Low hf/contrast → too clean → high tell.
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
    let contrast = (l.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / l.len() as f32).sqrt().max(1.0);
    let ratio = (hf / contrast).min(0.5) / 0.5; // 0 (dead smooth) .. 1 (grainy/detailed)
    let smoothness_tell = 1.0 - ratio;
    (0.6 * sat + 0.4 * smoothness_tell).clamp(0.0, 1.0)
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
pub fn polish(src: &RgbImage, strength: f32) -> RgbImage {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return src.clone();
    }
    let (w, h) = (src.width() as usize, src.height() as usize);
    let n = (w * h) as f32;
    let mut r: Vec<f32> = src.pixels().map(|p| p.0[0] as f32).collect();
    let mut g: Vec<f32> = src.pixels().map(|p| p.0[1] as f32).collect();
    let mut b: Vec<f32> = src.pixels().map(|p| p.0[2] as f32).collect();

    // 1. gray-world white balance — pull each channel mean toward the overall grey, blended by s and
    //    clamped so a genuinely-tinted scene (sunset) isn't flattened to grey.
    let (mr, mg, mb) = (r.iter().sum::<f32>() / n, g.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let grey = (mr + mg + mb) / 3.0;
    let gain = |m: f32| {
        let raw = if m > 1.0 { grey / m } else { 1.0 };
        let clamped = raw.clamp(0.85, 1.15); // never more than ±15% cast correction
        1.0 + s * (clamped - 1.0)
    };
    let (gr, gg, gb) = (gain(mr), gain(mg), gain(mb));
    for i in 0..r.len() {
        r[i] = (r[i] * gr).clamp(0.0, 255.0);
        g[i] = (g[i] * gg).clamp(0.0, 255.0);
        b[i] = (b[i] * gb).clamp(0.0, 255.0);
    }

    // 2. robust auto-levels — map the 0.5 / 99.5 luminance percentiles to 0 / 255 (blended by s), so a
    //    washed-out histogram gains real black and white points without crushing.
    let mut hist = [0u32; 256];
    for i in 0..r.len() {
        let l = lum(r[i], g[i], b[i]).clamp(0.0, 255.0) as usize;
        hist[l] += 1;
    }
    let total = r.len() as f32;
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
    if hi - lo > 8.0 {
        let scale = 255.0 / (hi - lo);
        // Ratio-preserving: stretch LUMINANCE and apply the same per-pixel gain to all three channels, so
        // contrast lifts without shifting colour balance or saturation (a per-channel stretch would
        // multiply the R/G/B gaps and amplify any cast).
        for i in 0..r.len() {
            let l = lum(r[i], g[i], b[i]);
            if l <= 1.0 {
                continue;
            }
            let stretched = ((l - lo) * scale).clamp(0.0, 255.0);
            let gain = 1.0 + s * (stretched / l - 1.0);
            for ch in [&mut r, &mut g, &mut b] {
                ch[i] = (ch[i] * gain).clamp(0.0, 255.0);
            }
        }
    }

    // 3. vibrance — adjust saturation toward a natural target: compress high saturation (the oversaturation
    //    tell), gently lift low saturation. Operate around per-pixel luminance so hue is preserved.
    for i in 0..r.len() {
        let l = lum(r[i], g[i], b[i]);
        let mx = r[i].max(g[i]).max(b[i]);
        let mn = r[i].min(g[i]).min(b[i]);
        let sat = if mx > 1.0 { (mx - mn) / mx } else { 0.0 };
        // factor > 1 boosts, < 1 tames. Natural mid ~0.45; pull toward it.
        let target_pull = 0.45 - sat; // + if dull, − if oversaturated
        let factor = 1.0 + s * 0.5 * target_pull;
        for ch in [&mut r, &mut g, &mut b] {
            ch[i] = (l + (ch[i] - l) * factor).clamp(0.0, 255.0);
        }
    }

    // 4. unsharp mask — sharpen against a blurred luminance so soft AI mush gains edge definition.
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
}
