//! P3 — per-panel scene art + character consistency (RFC COMIC-1 §2). Each panel is one
//! [`crate::api::Generate`] at the panel aspect. Every named character injects a stable identity
//! description so the same person recurs panel to panel: a `persona:` cast member compiles its
//! `PersonaSpec` through the deterministic `persona` layer (the reason a comic needs persona at all); a
//! `describe:` member contributes its seed-locked text. Generated panels flow back through the P1
//! composite + P2 lettering — and, when a face detector is configured, balloons become face-aware
//! (masks off faces + tails toward the nearest face), closing the P2 deferral.
//!
//! Needs a model (this is the one part of `comic` that does); the layout/lettering front half stays
//! weight-free.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::balloon::{self, Rectf};
use super::layout::{PanelRect, Plan};
use super::spec::{ComicSpec, Panel};

/// The default shared art style — a clean comic-book look — when the spec sets no `style`.
const DEFAULT_STYLE: &str =
    "comic book art, bold clean ink lineart, flat cel shading, vibrant colors, dynamic composition";
/// Always excluded: we letter *separately*, so the model must not draw its own bubbles/text.
const BASE_NEG: &str =
    "speech bubble, text, words, letters, caption box, watermark, signature, lowres, blurry, deformed, extra limbs";

/// A resolved character identity to inject into panel prompts.
#[derive(Debug, Clone)]
pub struct CharDesc {
    pub name: String,
    pub positive: String,
    pub negative: String,
}

/// Resolve every cast member to a stable identity description (persona-compiled or seed-locked text).
pub fn resolve_cast(spec: &ComicSpec) -> Result<HashMap<String, CharDesc>> {
    use crate::persona::{compile, lexicon::Lexicon, spec::PersonaSpec};
    let model = spec.model.as_deref().unwrap_or("sdxl");
    let class = compile::EncoderClass::from_model(model);
    let lex = Lexicon::skeleton();
    let mut map = HashMap::new();
    for c in &spec.cast {
        if c.name.trim().is_empty() {
            continue;
        }
        let (positive, negative) = if let Some(p) = c.persona.as_deref().filter(|s| !s.trim().is_empty()) {
            let ps = PersonaSpec::load(Path::new(p)).with_context(|| format!("cast `{}`: loading persona {p}", c.name))?;
            // Bare attribute description (no "photo of …" portrait framing — the character is going *into*
            // a scene, not a headshot). Fold the collection negatives in too.
            let emitted = compile::emit(&compile::resolve(&ps, &lex), class);
            let mut neg = emitted.negative.clone();
            for x in compile::collection_negatives(&ps) {
                if !neg.contains(&x) {
                    if !neg.is_empty() {
                        neg.push_str(", ");
                    }
                    neg.push_str(&x);
                }
            }
            (emitted.positive, neg)
        } else if let Some(d) = c.describe.as_deref().filter(|s| !s.trim().is_empty()) {
            (d.trim().to_string(), String::new())
        } else {
            (String::new(), String::new())
        };
        map.insert(c.name.clone(), CharDesc { name: c.name.clone(), positive, negative });
    }
    Ok(map)
}

/// Build the (positive, negative) prompt for one panel: scene → present characters' identities → shared
/// style; negatives = base (no drawn text) + each present character's exclusions.
pub fn panel_prompt(spec: &ComicSpec, panel: &Panel, cast: &HashMap<String, CharDesc>) -> (String, String) {
    let style = spec.style.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(DEFAULT_STYLE);
    let mut pos = String::new();
    if let Some(scene) = panel.scene.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        pos.push_str(scene);
    }
    let mut clauses = Vec::new();
    for name in &panel.chars {
        match cast.get(name) {
            Some(cd) if !cd.positive.is_empty() => clauses.push(format!("{name} is {}", cd.positive)),
            _ => clauses.push(name.clone()),
        }
    }
    if !clauses.is_empty() {
        if !pos.is_empty() {
            pos.push_str(". ");
        }
        pos.push_str(&clauses.join(". "));
    }
    if !pos.is_empty() {
        pos.push_str(", ");
    }
    pos.push_str(style);

    let mut neg = String::from(BASE_NEG);
    for name in &panel.chars {
        if let Some(cd) = cast.get(name) {
            if !cd.negative.is_empty() {
                neg.push_str(", ");
                neg.push_str(&cd.negative);
            }
        }
    }
    (pos, neg)
}

fn model_base(model: &str) -> u32 {
    let m = model.to_ascii_lowercase();
    if m.starts_with("sd15") || m.starts_with("sd21") || m == "sd" || m.contains("1.5") {
        512
    } else {
        1024
    }
}

/// A model-friendly generation size matching the panel's aspect (longest side = the model's base, snapped
/// to /64, clamped). The result is cover-fit into the panel rect at composite time, so it need not match
/// the panel pixel size exactly — only the aspect.
pub fn panel_size(rect: &PanelRect, model: &str) -> (u32, u32) {
    let base = model_base(model) as f32;
    let ar = rect.w.max(1) as f32 / rect.h.max(1) as f32;
    let (w, h) = if ar >= 1.0 { (base, base / ar) } else { (base * ar, base) };
    let snap = |v: f32| (((v / 64.0).round().max(1.0) as u32) * 64).clamp(512, 1536);
    (snap(w), snap(h))
}

/// Generate the scene art for one page's panels → `out_dir/<prefix>panel_NN.png` (reading index). Panels
/// with no entry yield `None`. Deterministic: each panel seeds off `seed_base + reading-index` (the caller
/// advances `seed_base` by the panel count so seeds are unique across pages).
pub async fn render_page_panels(spec: &ComicSpec, plan: &Plan, panels: &[super::spec::Panel], cast: &HashMap<String, CharDesc>, device: Option<&str>, out_dir: &Path, seed_base: u64, prefix: &str) -> Result<Vec<Option<PathBuf>>> {
    let model = spec.model.as_deref().unwrap_or("sdxl").to_string();
    let steps = spec.steps.unwrap_or(30);
    let mut out = Vec::with_capacity(plan.panels.len());
    for r in &plan.panels {
        let Some(panel) = panels.get(r.panel) else {
            out.push(None);
            continue;
        };
        let (pos, neg) = panel_prompt(spec, panel, cast);
        let (w, h) = panel_size(r, &model);
        let seed = seed_base.wrapping_add(r.index as u64);
        let mut g = crate::api::Generate::new(&model).prompt(pos).negative(neg).size(w, h).steps(steps).seed(seed).count(1);
        if let Some(d) = device {
            g = g.device(d);
        }
        let imgs = g.run().await.with_context(|| format!("rendering {prefix}panel #{}", r.index))?;
        let path = out_dir.join(format!("{prefix}panel_{:02}.png", r.index));
        imgs.first().context("panel render produced no image")?.save(&path)?;
        out.push(Some(path));
    }
    Ok(out)
}

/// Options for [`render_spec`] — the one orchestration the CLI, `api::Comic`, the scenario task, and the
/// Bund word all share.
#[derive(Debug, Clone, Default)]
pub struct RenderOpts {
    /// Device selector for generation + face detection (`None` → `auto`).
    pub device: Option<String>,
    /// Keep the generated per-panel PNGs here (else a temp dir, discarded).
    pub panels_out: Option<PathBuf>,
    /// Draw the balloons/captions after compositing (`false` → scene art only).
    pub letter: bool,
}

/// What a full render produced. Aggregate totals across all pages; `pages`/`sidecars` list each page's
/// PNG + sidecar in order (`page`/`sidecar` are the first, for the common single-page case).
#[derive(Debug, Clone)]
pub struct Report {
    pub page: PathBuf,
    pub sidecar: PathBuf,
    pub pages: Vec<PathBuf>,
    pub sidecars: Vec<PathBuf>,
    pub panels_rendered: usize,
    pub panels_total: usize,
    pub lines_placed: usize,
    pub lines_requested: usize,
    pub faces: usize,
}

/// The path for logical page `i` of `n`: single page → `out` unchanged; multi-page → `out` stem + `_NN`
/// with the same extension (`page.png` → `page_00.png`, `page_01.png`, …).
pub fn page_path(out: &Path, i: usize, n: usize) -> PathBuf {
    if n <= 1 {
        return out.to_path_buf();
    }
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("png");
    out.with_file_name(format!("{stem}_{i:02}.{ext}"))
}

/// The full flagship: for every logical page, generate its panels' scene art → composite → (optionally)
/// letter face-aware → write the page PNG + its `panels.json` sidecar. Shared core behind `comic render`,
/// `api::Comic`, the scenario `type: comic` task, and `plakat.comic.render`. Multi-page (`pages:` in the
/// spec) writes `out_00.png, out_01.png, …`; a single-page spec writes `out` unchanged.
pub async fn render_spec(spec: &ComicSpec, out: &Path, opts: &RenderOpts) -> Result<Report> {
    let device = opts.device.as_deref();
    let cast = resolve_cast(spec)?;
    let base_seed = spec.seed.unwrap_or(0);
    let logical = spec.logical_pages();
    let n_pages = logical.len();

    // per-panel PNGs → a kept dir or a temp dir (shared across pages; a per-page filename prefix keeps
    // them distinct).
    let tmp = if opts.panels_out.is_none() { Some(tempfile::tempdir().context("temp dir for panels")?) } else { None };
    let panels_dir = match (&opts.panels_out, &tmp) {
        (Some(d), _) => {
            std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
            d.clone()
        }
        (None, Some(t)) => t.path().to_path_buf(),
        _ => unreachable!(),
    };

    let mut rep = Report {
        page: PathBuf::new(),
        sidecar: PathBuf::new(),
        pages: Vec::with_capacity(n_pages),
        sidecars: Vec::with_capacity(n_pages),
        panels_rendered: 0,
        panels_total: 0,
        lines_placed: 0,
        lines_requested: 0,
        faces: 0,
    };
    let mut seed_cursor = base_seed;

    for (pi, lp) in logical.iter().enumerate() {
        let plan = super::layout::resolve_page(spec, lp);
        let bw = plan.border as f32;
        let prefix = if n_pages <= 1 { String::new() } else { format!("p{pi:02}_") };

        let paths = render_page_panels(spec, &plan, &lp.panels, &cast, device, &panels_dir, seed_cursor, &prefix).await?;
        seed_cursor = seed_cursor.wrapping_add(lp.panels.len().max(1) as u64);

        // paths are in reading order; compose indexes by page-panel index — bridge the two.
        let mut imgs: Vec<Option<image::DynamicImage>> = vec![None; lp.panels.len().max(1)];
        for (r, p) in plan.panels.iter().zip(paths.iter()) {
            if let Some(pp) = p {
                imgs[r.panel] = image::open(pp).ok();
            }
        }
        let mut page = super::page::compose(&plan, &imgs);

        if opts.letter {
            let mut faces: HashMap<usize, Vec<Rectf>> = HashMap::new();
            for (r, p) in plan.panels.iter().zip(paths.iter()) {
                let Some(pp) = p else { continue };
                let boxes = detect_faces(pp, device).await;
                if boxes.is_empty() {
                    continue;
                }
                if let Ok((sw, sh)) = image::image_dimensions(pp) {
                    let (iw, ih) = ((r.w as f32 - 2.0 * bw).max(1.0), (r.h as f32 - 2.0 * bw).max(1.0));
                    faces.insert(r.index, boxes.iter().map(|b| cover_fit_box(*b, sw as f32, sh as f32, iw, ih)).collect());
                }
            }
            rep.faces += faces.values().map(Vec::len).sum::<usize>();
            let (pl, rq) = letter_faceaware(&mut page, &plan, &lp.panels, &faces);
            rep.lines_placed += pl;
            rep.lines_requested += rq;
        }

        let page_out = page_path(out, pi, n_pages);
        page.save(&page_out).with_context(|| format!("writing {}", page_out.display()))?;
        let sidecar = page_out.with_extension("panels.json");
        std::fs::write(&sidecar, super::page::panels_json(&plan)).with_context(|| format!("writing {}", sidecar.display()))?;

        rep.panels_rendered += imgs.iter().filter(|o| o.is_some()).count();
        rep.panels_total += lp.panels.len();
        rep.pages.push(page_out);
        rep.sidecars.push(sidecar);
    }

    rep.page = rep.pages.first().cloned().unwrap_or_else(|| out.to_path_buf());
    rep.sidecar = rep.sidecars.first().cloned().unwrap_or_else(|| out.with_extension("panels.json"));
    Ok(rep)
}

// ---- face-aware lettering (closes the P2 deferral) ----

/// Detect faces in `path`, best-effort. Returns their `[x1,y1,x2,y2]` boxes in the image's own pixels, or
/// an empty vec when no detector is configured / detection fails (the caller falls back to P2 defaults).
pub async fn detect_faces(path: &Path, device: Option<&str>) -> Vec<[f32; 4]> {
    use crate::pipelines::scrfd::{resolve_scrfd_weights, SCRFDConfig, SCRFDDetector};
    let Some(weights) = resolve_scrfd_weights().await.ok().flatten() else {
        return Vec::new();
    };
    let dev = match crate::api::device(device.unwrap_or("auto")) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    let Ok(det) = SCRFDDetector::load(&weights, SCRFDConfig::default(), &dev, candle_core::DType::F32) else {
        return Vec::new();
    };
    match det.detect(path) {
        Ok(faces) => faces.iter().filter(|f| f.score >= 0.4).map(|f| f.bbox).collect(),
        Err(_) => Vec::new(),
    }
}

/// Map a face box from generated-image pixels into panel-interior coordinates under the same cover-fit
/// (scale-to-fill + centre-crop) the compositor uses, so masks/tails line up with what's on the page.
pub fn cover_fit_box(b: [f32; 4], sw: f32, sh: f32, iw: f32, ih: f32) -> Rectf {
    let scale = (iw / sw).max(ih / sh);
    let (rw, rh) = (sw * scale, sh * scale);
    let (ox, oy) = ((rw - iw) / 2.0, (rh - ih) / 2.0);
    let x0 = (b[0] * scale - ox).clamp(0.0, iw);
    let y0 = (b[1] * scale - oy).clamp(0.0, ih);
    let x1 = (b[2] * scale - ox).clamp(0.0, iw);
    let y1 = (b[3] * scale - oy).clamp(0.0, ih);
    Rectf { x: x0, y: y0, w: (x1 - x0).max(1.0), h: (y1 - y0).max(1.0) }
}

/// Letter a composited `page`, using per-panel detected faces (interior-local coordinates, keyed by
/// panel *reading index*) as balloon-exclusion masks + tail targets. Panels without faces fall back to
/// the P2 open-space defaults. Returns (lines placed, lines requested).
pub fn letter_faceaware(page: &mut image::RgbImage, plan: &Plan, panels: &[super::spec::Panel], faces: &HashMap<usize, Vec<Rectf>>) -> (usize, usize) {
    let (mut placed, mut requested) = (0usize, 0usize);
    let bw = plan.border as f32;
    for r in &plan.panels {
        let Some(panel) = panels.get(r.panel) else { continue };
        let mut lines = balloon::lines_for_panel(panel);
        requested += lines.len();
        let (iw, ih) = ((r.w as f32 - 2.0 * bw).max(1.0), (r.h as f32 - 2.0 * bw).max(1.0));
        let panel_faces = faces.get(&r.index).map(Vec::as_slice).unwrap_or(&[]);
        // aim each balloon's tail at the face nearest its preferred x (fallback: largest face).
        if !panel_faces.is_empty() {
            for ln in &mut lines {
                if ln.kind == balloon::Kind::Caption {
                    continue;
                }
                let want_x = ln.anchor.pref_x(iw).unwrap_or(iw * 0.5);
                let target = panel_faces
                    .iter()
                    .min_by(|a, b| ((a.x + a.w / 2.0) - want_x).abs().partial_cmp(&((b.x + b.w / 2.0) - want_x).abs()).unwrap())
                    .map(|f| (f.x + f.w / 2.0, f.y + f.h / 2.0));
                ln.speaker = target;
            }
        }
        let laid = balloon::place(iw, ih, panel_faces, &lines);
        placed += laid.len();
        let (ox, oy) = ((r.x as f32 + bw) as i32, (r.y as f32 + bw) as i32);
        for b in &laid {
            balloon::draw(page, b, ox, oy);
        }
    }
    (placed, requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(scene: &str, chars: &[&str]) -> (ComicSpec, Panel) {
        let mut spec = ComicSpec::default();
        spec.cast = vec![super::super::spec::CastMember { name: "mika".into(), persona: None, describe: Some("a woman with short black hair and a red scarf".into()) }];
        let panel = Panel { scene: Some(scene.into()), chars: chars.iter().map(|s| s.to_string()).collect(), caption: None, balloons: vec![] };
        (spec, panel)
    }

    #[test]
    fn prompt_injects_character_identity_and_style() {
        let (spec, panel) = spec_with("a neon alley at night", &["mika"]);
        let cast = resolve_cast(&spec).unwrap();
        let (pos, neg) = panel_prompt(&spec, &panel, &cast);
        assert!(pos.contains("neon alley"), "scene present: {pos}");
        assert!(pos.contains("mika is a woman with short black hair"), "identity injected: {pos}");
        assert!(pos.contains("comic book art"), "shared style appended: {pos}");
        assert!(neg.contains("speech bubble") && neg.contains("text"), "drawn text excluded: {neg}");
    }

    #[test]
    fn unknown_char_falls_back_to_bare_name() {
        let (spec, panel) = spec_with("a rooftop", &["ghost"]);
        let cast = resolve_cast(&spec).unwrap();
        let (pos, _) = panel_prompt(&spec, &panel, &cast);
        assert!(pos.contains("ghost"), "bare name kept: {pos}");
    }

    #[test]
    fn panel_size_matches_aspect_and_snaps() {
        let wide = PanelRect { index: 0, panel: 0, x: 0, y: 0, w: 1200, h: 400 };
        let (w, h) = panel_size(&wide, "sdxl");
        assert!(w > h, "wide panel → landscape gen size {w}x{h}");
        assert_eq!(w % 64, 0, "snapped to 64");
        assert_eq!(h % 64, 0);
        let (w2, _) = panel_size(&wide, "sd15");
        assert!(w2 <= 1024, "sd15 base is smaller: {w2}");
    }

    #[test]
    fn page_path_single_vs_multi() {
        let out = std::path::Path::new("out/page.png");
        assert_eq!(page_path(out, 0, 1), std::path::PathBuf::from("out/page.png"));
        assert_eq!(page_path(out, 0, 3), std::path::PathBuf::from("out/page_00.png"));
        assert_eq!(page_path(out, 2, 3), std::path::PathBuf::from("out/page_02.png"));
    }

    #[test]
    fn cover_fit_box_maps_into_interior() {
        // a face in the middle of a 1024×768 render → still central in a 400×300 interior.
        let r = cover_fit_box([460.0, 340.0, 560.0, 440.0], 1024.0, 768.0, 400.0, 300.0);
        assert!(r.x > 100.0 && r.x < 300.0 && r.y > 80.0 && r.y < 240.0, "central mapping: {r:?}");
        assert!(r.w > 0.0 && r.h > 0.0);
    }
}
