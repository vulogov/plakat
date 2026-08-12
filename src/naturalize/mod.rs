//! `plakat naturalize` — the analog post-pass (RFC QUALITY-1, proven in G0.1). Stamps physical-media
//! imperfections onto a finished image — film grain, chromatic aberration, vignette, bloom, a desaturating
//! film grade, optional defocus — to break the digital-clean, over-saturated fingerprint that reads as
//! "AI-generated". Deterministic, weight-free, no GPU.
//!
//! It reduces the machine *fingerprint*; it does **not** fix physical-reasoning errors (bad reflections,
//! impossible geometry) — that's a model-capability limit, addressed by the hi-res fix in P2.

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
            Preset::Subtle => Params { grain: 0.16, aberration: 0.12, vignette: 0.05, bloom: 0.05, desaturate: 0.08, warm: 0.0, defocus: 0.0 },
            Preset::Photo => Params { grain: 0.30, aberration: 0.35, vignette: 0.12, bloom: 0.10, desaturate: 0.12, warm: 0.05, defocus: 0.0 },
            Preset::Painting => Params { grain: 0.36, aberration: 0.10, vignette: 0.10, bloom: 0.08, desaturate: 0.20, warm: 0.05, defocus: 0.10 },
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
            Focus::People => Params { grain: 0.30, aberration: 0.05, vignette: 0.04, bloom: 0.06, desaturate: 0.18, warm: 0.03, defocus: 0.0 },
            Focus::Sky => Params { grain: 0.28, aberration: 0.05, vignette: 0.06, bloom: 0.10, desaturate: 0.10, warm: 0.02, defocus: 0.0 },
            Focus::Vegetation => Params { grain: 0.42, aberration: 0.10, vignette: 0.08, bloom: 0.06, desaturate: 0.14, warm: 0.03, defocus: 0.08 },
            Focus::Cityscape => Params { grain: 0.30, aberration: 0.30, vignette: 0.10, bloom: 0.10, desaturate: 0.10, warm: 0.03, defocus: 0.05 },
            Focus::Landscape => Params { grain: 0.30, aberration: 0.14, vignette: 0.16, bloom: 0.10, desaturate: 0.16, warm: 0.04, defocus: 0.02 },
            Focus::Sea => Params { grain: 0.30, aberration: 0.10, vignette: 0.10, bloom: 0.15, desaturate: 0.16, warm: 0.0, defocus: 0.02 },
            Focus::River => Params { grain: 0.32, aberration: 0.10, vignette: 0.10, bloom: 0.11, desaturate: 0.14, warm: 0.03, defocus: 0.05 },
            Focus::Mechanics => Params { grain: 0.28, aberration: 0.24, vignette: 0.10, bloom: 0.15, desaturate: 0.10, warm: 0.03, defocus: 0.03 },
            Focus::Household => Params { grain: 0.28, aberration: 0.10, vignette: 0.10, bloom: 0.10, desaturate: 0.12, warm: 0.05, defocus: 0.02 },
        }
    }
}

fn to_arr(p: &Params) -> [f32; 7] {
    [p.grain, p.aberration, p.vignette, p.bloom, p.desaturate, p.warm, p.defocus]
}
fn from_arr(a: [f32; 7]) -> Params {
    Params { grain: a[0], aberration: a[1], vignette: a[2], bloom: a[3], desaturate: a[4], warm: a[5], defocus: a[6] }
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
        for i in 0..7 {
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

/// Apply the naturalize pass. Returns a new image the same size as `src`.
pub fn apply(src: &RgbImage, p: &Params) -> RgbImage {
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
