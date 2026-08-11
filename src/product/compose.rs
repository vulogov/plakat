//! Canvas + sweep + composite (RFC PRODUCT-1) — weight-free. Places a subject cutout on a studio sweep,
//! grounds it (contact shadow + reflection, [`super::ground`]), composites in z-order, and writes the
//! `shot.meta.json` sidecar. With a supplied cutout this produces a finished packshot with no GPU.

use super::ground::{self, ReflectionKind, ShadowKind};
use super::spec::ProductSpec;
use image::{DynamicImage, Rgb, Rgba, RgbImage, RgbaImage};

/// The resolved plan for one shot: canvas + background + the placed subject + grounding parameters.
#[derive(Debug, Clone)]
pub struct Plan {
    pub w: u32,
    pub h: u32,
    pub bg: Bg,
    pub subject_scale: f32,
    pub anchor_bottom: bool,
    pub ground_y: u32,
    pub shadow: ShadowKind,
    pub reflection: ReflectionKind,
    pub key: f32,
    pub softness: f32,
    pub falloff: f32,
    pub camera_angle: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Bg {
    Flat(Rgb<u8>),
    /// vertical gradient top→bottom.
    Gradient(Rgb<u8>, Rgb<u8>),
}

fn parse_color(s: &str) -> Option<Rgb<u8>> {
    match s.trim().to_ascii_lowercase().as_str() {
        "white" => Some(Rgb([255, 255, 255])),
        "black" => Some(Rgb([16, 16, 16])),
        "grey" | "gray" => Some(Rgb([210, 210, 212])),
        other => {
            let c: Vec<u8> = other.split(',').filter_map(|p| p.trim().parse::<u8>().ok()).collect();
            (c.len() == 3).then(|| Rgb([c[0], c[1], c[2]]))
        }
    }
}

fn resolve_bg(bg: Option<&str>) -> Bg {
    match bg.map(|s| s.trim().to_ascii_lowercase()) {
        None => Bg::Flat(Rgb([255, 255, 255])),
        Some(s) if s == "white" => Bg::Flat(Rgb([255, 255, 255])),
        Some(s) if s == "grey-sweep" || s == "gray-sweep" || s == "sweep" => Bg::Gradient(Rgb([255, 255, 255]), Rgb([225, 225, 228])),
        Some(s) if s == "scene" => Bg::Gradient(Rgb([255, 255, 255]), Rgb([225, 225, 228])), // P3 generates; P1 → a sweep
        Some(s) if s.starts_with("gradient:") => {
            let rest = &s["gradient:".len()..];
            let mut it = rest.splitn(2, ',').map(str::trim);
            let top = it.next().and_then(parse_color).unwrap_or(Rgb([255, 255, 255]));
            let bot = it.next().and_then(parse_color).unwrap_or(Rgb([225, 225, 228]));
            Bg::Gradient(top, bot)
        }
        Some(s) => parse_color(&s).map(Bg::Flat).unwrap_or(Bg::Flat(Rgb([255, 255, 255]))),
    }
}

fn canvas_dims(size: Option<&str>, px: u32) -> (u32, u32) {
    let px = px.clamp(256, 4096);
    match size.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("portrait") => (((px as f32) * 0.8) as u32, px),
        Some("landscape") => (px, ((px as f32) * 0.75) as u32),
        _ => (px, px), // square
    }
}

/// Resolve a [`ProductSpec`] → a [`Plan`].
pub fn resolve(spec: &ProductSpec) -> Plan {
    let canvas = spec.canvas.clone().unwrap_or_default();
    let (w, h) = canvas_dims(canvas.size.as_deref(), canvas.px.unwrap_or(1024));
    let bg = resolve_bg(canvas.bg.as_deref());
    let subject = spec.subject.clone().unwrap_or_default();
    let scale = subject.scale.unwrap_or(0.7).clamp(0.1, 0.98);
    let anchor_bottom = !matches!(subject.anchor.as_deref(), Some("center") | Some("centre"));
    let ground_frac = if anchor_bottom { 0.82 } else { 0.5 };
    let ground_y = (h as f32 * ground_frac) as u32;
    let g = spec.ground.clone().unwrap_or_default();
    let lighting = spec.lighting.clone().unwrap_or_default();
    let camera = spec.camera.clone().unwrap_or_default();
    Plan {
        w,
        h,
        bg,
        subject_scale: scale,
        anchor_bottom,
        ground_y,
        shadow: ShadowKind::parse(g.shadow.as_deref()),
        reflection: ReflectionKind::parse(g.reflection.as_deref()),
        key: ground::key_offset(lighting.key_dir.as_deref()),
        softness: g.softness.unwrap_or(0.5),
        falloff: g.falloff.unwrap_or(0.6),
        camera_angle: camera.angle,
    }
}

/// The background sweep for a plan.
fn sweep(plan: &Plan) -> RgbImage {
    let mut img = RgbImage::new(plan.w, plan.h);
    match plan.bg {
        Bg::Flat(c) => {
            for p in img.pixels_mut() {
                *p = c;
            }
        }
        Bg::Gradient(a, b) => {
            for y in 0..plan.h {
                let t = y as f32 / plan.h.max(1) as f32;
                let mix = |i: usize| (a.0[i] as f32 * (1.0 - t) + b.0[i] as f32 * t).round() as u8;
                let c = Rgb([mix(0), mix(1), mix(2)]);
                for x in 0..plan.w {
                    img.put_pixel(x, y, c);
                }
            }
        }
    }
    img
}

/// Trim a cutout to its alpha bounding box (so `scale` refers to the product, not its padding).
fn trim_to_alpha(img: &RgbaImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    let mut any = false;
    for (x, y, p) in img.enumerate_pixels() {
        if p.0[3] > 8 {
            any = true;
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if !any {
        return img.clone();
    }
    image::imageops::crop_imm(img, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image()
}

/// Place the (trimmed) subject cutout onto a canvas-sized transparent layer at the plan's scale + anchor.
fn place_subject(plan: &Plan, cutout: &RgbaImage) -> RgbaImage {
    let trimmed = trim_to_alpha(cutout);
    let (sw, sh) = trimmed.dimensions();
    let target_h = (plan.h as f32 * plan.subject_scale).max(1.0);
    let ratio = target_h / sh.max(1) as f32;
    let (tw, th) = ((sw as f32 * ratio).round().max(1.0) as u32, target_h.round() as u32);
    let scaled = image::imageops::resize(&trimmed, tw, th, image::imageops::FilterType::Lanczos3);
    let mut layer = RgbaImage::from_pixel(plan.w, plan.h, Rgba([0, 0, 0, 0]));
    let x = ((plan.w as i64 - tw as i64) / 2).max(0) as u32;
    let y = if plan.anchor_bottom {
        plan.ground_y.saturating_sub(th)
    } else {
        ((plan.h as i64 - th as i64) / 2).max(0) as u32
    };
    image::imageops::overlay(&mut layer, &scaled, x as i64, y as i64);
    layer
}

fn over(dst: &mut RgbImage, src: &RgbaImage) {
    for (x, y, p) in src.enumerate_pixels() {
        let a = p.0[3] as f32 / 255.0;
        if a <= 0.0 {
            continue;
        }
        let d = dst.get_pixel(x, y).0;
        let blend = |i: usize| (p.0[i] as f32 * a + d[i] as f32 * (1.0 - a)).round() as u8;
        dst.put_pixel(x, y, Rgb([blend(0), blend(1), blend(2)]));
    }
}

fn darken(dst: &mut RgbImage, shadow: &[f32], max_dark: f32) {
    let w = dst.width();
    for (i, s) in shadow.iter().enumerate() {
        if *s <= 0.0 {
            continue;
        }
        let (x, y) = (i as u32 % w, i as u32 / w);
        let k = 1.0 - s * max_dark;
        let d = dst.get_pixel(x, y).0;
        dst.put_pixel(x, y, Rgb([(d[0] as f32 * k) as u8, (d[1] as f32 * k) as u8, (d[2] as f32 * k) as u8]));
    }
}

/// Compose the packshot: sweep ← reflection ← shadow ← subject. `subject_cutout` is a transparent PNG.
pub fn compose(plan: &Plan, subject_cutout: &DynamicImage) -> RgbImage {
    let (w, h) = (plan.w as usize, plan.h as usize);
    let placed = place_subject(plan, &subject_cutout.to_rgba8());
    let alpha: Vec<f32> = placed.pixels().map(|p| p.0[3] as f32 / 255.0).collect();

    let mut out = sweep(plan);
    // reflection (under everything but the sweep)
    let refl = ground::reflection(&placed, plan.ground_y as usize, plan.reflection, ground::camera_squash(plan.camera_angle.as_deref()), plan.falloff);
    over(&mut out, &refl);
    // contact shadow
    let shadow = ground::contact_shadow(&alpha, w, h, plan.ground_y as usize, plan.shadow, plan.key, plan.softness);
    darken(&mut out, &shadow, 0.55);
    // subject on top (pixel-exact)
    over(&mut out, &placed);
    out
}

/// The `shot.meta.json` sidecar: the resolved rig / camera / ground / canvas, for reproducibility.
pub fn meta_json(spec: &ProductSpec, plan: &Plan) -> String {
    let shadow = match plan.shadow {
        ShadowKind::Soft => "soft",
        ShadowKind::Hard => "hard",
        ShadowKind::None => "none",
    };
    let reflection = match plan.reflection {
        ReflectionKind::Gloss => "gloss",
        ReflectionKind::Mirror => "mirror",
        ReflectionKind::None => "none",
    };
    serde_json::to_string_pretty(&serde_json::json!({
        "schema": super::SCHEMA_VERSION,
        "canvas": { "w": plan.w, "h": plan.h },
        "subject": { "scale": plan.subject_scale, "anchor": if plan.anchor_bottom { "bottom" } else { "center" } },
        "camera": { "angle": plan.camera_angle.clone().unwrap_or_else(|| "eye".into()) },
        "ground": { "shadow": shadow, "reflection": reflection, "ground_y": plan.ground_y },
        "lighting": { "key": plan.key },
        "seed": spec.seed,
    }))
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cutout(w: u32, h: u32) -> DynamicImage {
        let mut img = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
        for y in h / 4..h * 3 / 4 {
            for x in w / 3..w * 2 / 3 {
                img.put_pixel(x, y, Rgba([30, 160, 170, 255]));
            }
        }
        DynamicImage::ImageRgba8(img)
    }

    #[test]
    fn composes_a_grounded_packshot() {
        let spec = ProductSpec::from_hjson(r#"{ canvas: { px: 400, bg: "white" }, subject: { scale: 0.5, anchor: "bottom" }, ground: { shadow: "soft", reflection: "gloss" } }"#).unwrap();
        let plan = resolve(&spec);
        assert_eq!((plan.w, plan.h), (400, 400));
        let shot = compose(&plan, &cutout(300, 300));
        assert_eq!(shot.dimensions(), (400, 400));
        // there is a dark contact shadow band on the floor just below the subject.
        let gy = plan.ground_y;
        let row_lum = |y: u32| -> f32 { (0..plan.w).map(|x| shot.get_pixel(x, y).0[0] as f32).sum::<f32>() / plan.w as f32 };
        assert!(row_lum(gy + 4) < 250.0, "a shadow darkens the floor below the subject ({})", row_lum(gy + 4));
        // the top of the canvas (above the product) is the clean white sweep.
        assert!(row_lum(4) > 250.0, "clean sweep at the top");
        // sidecar carries the resolved ground.
        let js: serde_json::Value = serde_json::from_str(&meta_json(&spec, &plan)).unwrap();
        assert_eq!(js["ground"]["shadow"], "soft");
    }
}
