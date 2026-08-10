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
///
/// 6.8.2 D2: `art_ids` accumulates `panel.id → rendered art path` across pages; a panel with `reuse: "@id"`
/// yields that path (no generation), so an establishing shot repeats *identically* book-wide.
pub async fn render_page_panels(spec: &ComicSpec, plan: &Plan, panels: &[super::spec::Panel], cast: &HashMap<String, CharDesc>, device: Option<&str>, out_dir: &Path, seed_base: u64, prefix: &str, art_ids: &mut HashMap<String, PathBuf>) -> Result<Vec<Option<PathBuf>>> {
    let model = spec.model.as_deref().unwrap_or("sdxl").to_string();
    let steps = spec.steps.unwrap_or(30);
    let mut out = Vec::with_capacity(plan.panels.len());
    for r in &plan.panels {
        let Some(panel) = panels.get(r.panel) else {
            out.push(None);
            continue;
        };
        // D2: reuse a labelled panel's already-rendered art instead of generating.
        if let Some(key) = panel.reuse.as_deref().map(|s| s.trim_start_matches('@').trim()).filter(|s| !s.is_empty()) {
            match art_ids.get(key) {
                Some(src) => {
                    out.push(Some(src.clone()));
                    continue;
                }
                None => tracing::warn!(target: "plakat", "comic: panel reuse `@{key}` not found (unrendered / unknown id) — generating instead"),
            }
        }
        let (pos, neg) = panel_prompt(spec, panel, cast);
        let (w, h) = panel_size(r, &model);
        let seed = seed_base.wrapping_add(r.index as u64);
        let mut g = crate::api::Generate::new(&model).prompt(pos).negative(neg).size(w, h).steps(steps).seed(seed).count(1);
        if let Some(d) = device {
            g = g.device(d);
        }
        // M2 style-lock: the same LoRA on every panel of every page → one look book-wide.
        if let Some(lora) = spec.style_lora.as_deref().filter(|s| !s.trim().is_empty()) {
            g = g.lora(lora, spec.style_lora_scale.unwrap_or(0.8));
        }
        let imgs = g.run().await.with_context(|| format!("rendering {prefix}panel #{}", r.index))?;
        let path = out_dir.join(format!("{prefix}panel_{:02}.png", r.index));
        imgs.first().context("panel render produced no image")?.save(&path)?;
        // D2: register this panel's id so later panels can reuse its art.
        if let Some(id) = panel.id.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            art_ids.insert(id.to_string(), path.clone());
        }
        out.push(Some(path));
    }
    Ok(out)
}

/// M2 reference-lock context: the loaded face-swapper + a per-character ArcFace identity latent (built
/// once from each character's reference face). Present only when the face-swap weights resolved.
struct LockCtx {
    swapper: crate::pipelines::faceswap::FaceSwapper,
    latents: HashMap<String, candle_core::Tensor>,
    references: Vec<PathBuf>,
}

/// Build the reference-lock context (best-effort): load the face-swapper, build/collect each lockable
/// character's reference face, and embed it into an identity latent. Returns `Ok(None)` when the weights
/// aren't available (→ identity stays description-level) or nothing is lockable.
async fn build_lock_ctx(spec: &ComicSpec, device: Option<&str>, dir: &Path) -> Result<Option<LockCtx>> {
    let dev = crate::api::device(device.unwrap_or("auto"))?;
    let swapper = match crate::pipelines::faceswap::FaceSwapper::load_resolved(&dev, candle_core::DType::F32).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target: "plakat", "comic: face-lock unavailable ({e}); identity stays description-level");
            return Ok(None);
        }
    };
    let refs = build_cast_references(spec, device, dir).await?;
    if refs.is_empty() {
        return Ok(None);
    }
    let mut latents = HashMap::new();
    let mut references = Vec::new();
    for (name, path) in &refs {
        match swapper.source_latent(path) {
            Ok(t) => {
                latents.insert(name.clone(), t);
                references.push(path.clone());
            }
            Err(e) => tracing::warn!(target: "plakat", "comic: no usable face in reference for `{name}` ({e}); it stays description-level"),
        }
    }
    if latents.is_empty() {
        return Ok(None);
    }
    Ok(Some(LockCtx { swapper, latents, references }))
}

/// D3: run one restore-faces (ADetailer) pass over `panels` (small swapped faces), refining each face
/// with a light img2img so the swap crisps up. Best-effort — returns how many panels were touched, 0 on
/// any setup failure (missing SD/SCRFD weights). Loads the model once.
async fn restore_small_faces(spec: &ComicSpec, device: Option<&str>, panels: &[PathBuf]) -> usize {
    let dev = match crate::api::device(device.unwrap_or("auto")) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let mut cfg = crate::pipelines::adetailer::Config::defaults();
    cfg.model = spec.model.as_deref().unwrap_or("sdxl").to_string();
    cfg.working_size = model_base(&cfg.model);
    cfg.strength = 0.35; // crisp detail without drifting the swapped identity
    cfg.device = dev;
    match crate::pipelines::adetailer::refine_files(&cfg, panels, None).await {
        Ok(_n) => panels.len(),
        Err(e) => {
            tracing::warn!(target: "plakat", "comic: restore-faces pass skipped ({e})");
            0
        }
    }
}

/// Return the indices of `centroids` (face x-positions) in reading order: left→right for `ltr`,
/// right→left for `rtl`. Stable for equal x. Position i of the result ↔ character i in `panel.chars`.
fn reading_order(centroids: &[f32], rtl: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..centroids.len()).collect();
    idx.sort_by(|&a, &b| {
        let o = centroids[a].partial_cmp(&centroids[b]).unwrap_or(std::cmp::Ordering::Equal);
        if rtl { o.reverse() } else { o }
    });
    idx
}

/// The outcome of locking one panel: how many faces were swapped, and whether any swapped face was
/// **small** relative to the panel (→ a candidate for the D3 restore pass).
#[derive(Debug, Clone, Copy, Default)]
struct LockResult {
    locked: usize,
    small: bool,
}

/// A swapped face shorter than this fraction of the panel height is "small" (distant / group shot) and
/// benefits from a restore-faces refine (6.8.2 D3).
const SMALL_FACE_FRAC: f32 = 0.22;

/// Face-swap the locked characters' references onto a rendered panel, in place. Handles **multiple**
/// characters (6.8.2 D1): detected faces are matched to `panel.chars` by **reading-order position** — the
/// author controls who's where by the order of `chars` (left→right for `ltr`, right→left for `rtl`).
/// Swaps chain onto the same scene. Returns a [`LockResult`]. Best-effort.
fn lock_panel(ctx: &LockCtx, panel: &super::spec::Panel, path: &Path, rtl: bool) -> LockResult {
    // characters present that have a reference, in author (= reading) order.
    let present: Vec<&String> = panel.chars.iter().filter(|c| ctx.latents.contains_key(c.as_str())).collect();
    if present.is_empty() {
        return LockResult::default();
    }
    let Ok(mut faces) = ctx.swapper.detect(path) else { return LockResult::default() };
    if faces.is_empty() {
        return LockResult::default();
    }
    // order faces spatially by face-centroid x so position i ↔ character i.
    let order = reading_order(&faces.iter().map(|f| (f.bbox[0] + f.bbox[2]) * 0.5).collect::<Vec<_>>(), rtl);
    let ordered: Vec<_> = order.into_iter().map(|i| faces[i].clone()).collect();
    faces = ordered;
    let Ok(mut scene) = image::open(path).map(|i| i.to_rgb8()) else { return LockResult::default() };
    let ph = scene.height() as f32;
    let n = present.len().min(faces.len());
    let mut res = LockResult::default();
    for i in 0..n {
        let latent = &ctx.latents[present[i]];
        match ctx.swapper.swap_into(&scene, faces[i].landmarks, latent) {
            Ok(swapped) => {
                scene = swapped;
                res.locked += 1;
                if (faces[i].bbox[3] - faces[i].bbox[1]) < SMALL_FACE_FRAC * ph {
                    res.small = true;
                }
            }
            Err(e) => tracing::warn!(target: "plakat", "comic: face-swap failed for `{}` on {} ({e})", present[i], path.display()),
        }
    }
    if res.locked > 0 && scene.save(path).is_err() {
        return LockResult::default();
    }
    res
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
    /// M2 reference-lock: build a per-character reference face (once) and face-swap it onto every panel so
    /// identity holds book-wide (best-effort — needs the face-swap weights; falls back to description-level
    /// when absent). Off unless requested.
    pub lock: bool,
    /// D3: after locking, run a restore-faces (ADetailer) refine over panels whose swapped face is small,
    /// to crisp the detail. Best-effort (needs the restore pipeline). Off unless requested.
    pub restore: bool,
}

/// A stable per-name seed contribution (FNV-1a) so a character's reference portrait is reproducible
/// without depending on iteration order or a RNG.
fn name_salt(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn safe_name(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

/// Build a reference face image for each lockable cast member (M2). An explicit `reference:` wins;
/// otherwise a canonical portrait is rendered **once** from the persona/describe (deterministic per name)
/// and cached in `dir`. Members with `lock: false` or no identity are skipped.
pub async fn build_cast_references(spec: &ComicSpec, device: Option<&str>, dir: &Path) -> Result<HashMap<String, PathBuf>> {
    use crate::persona::{compile, lexicon::Lexicon, spec::PersonaSpec};
    let model = spec.model.as_deref().unwrap_or("sdxl").to_string();
    let lex = Lexicon::skeleton();
    let steps = spec.steps.unwrap_or(30);
    let base_seed = spec.seed.unwrap_or(0);
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let mut refs = HashMap::new();
    for c in &spec.cast {
        if c.lock == Some(false) || c.name.trim().is_empty() {
            continue;
        }
        // an explicit reference face wins — no render.
        if let Some(r) = c.reference.as_deref().filter(|s| !s.trim().is_empty()) {
            refs.insert(c.name.clone(), PathBuf::from(r));
            continue;
        }
        // else render one canonical portrait from the persona/describe.
        let (prompt, neg) = if let Some(p) = c.persona.as_deref().filter(|s| !s.trim().is_empty()) {
            let ps = PersonaSpec::load(Path::new(p)).with_context(|| format!("cast `{}`: loading persona {p}", c.name))?;
            let comp = compile::compile_for_model(&ps, &lex, &model);
            (format!("{}, character reference portrait, front view, head and shoulders, plain neutral background", comp.positive), comp.negative)
        } else if let Some(d) = c.describe.as_deref().filter(|s| !s.trim().is_empty()) {
            (format!("portrait headshot of {d}, front view, head and shoulders, plain neutral background, character reference sheet"), String::new())
        } else {
            continue;
        };
        let neg = if neg.trim().is_empty() {
            "multiple people, full body, cropped, text, watermark, blurry, deformed".to_string()
        } else {
            format!("{neg}, multiple people, full body, text, watermark")
        };
        let seed = base_seed ^ name_salt(&c.name);
        // Portrait at the model's native square (sd15 → 512, sdxl → 1024) so it renders fast + coherent.
        let sz = model_base(&model);
        let mut g = crate::api::Generate::new(&model).prompt(prompt).negative(neg).size(sz, sz).steps(steps).seed(seed).count(1);
        if let Some(d) = device {
            g = g.device(d);
        }
        if let Some(lora) = spec.style_lora.as_deref().filter(|s| !s.trim().is_empty()) {
            g = g.lora(lora, spec.style_lora_scale.unwrap_or(0.8));
        }
        let imgs = g.run().await.with_context(|| format!("rendering reference portrait for `{}`", c.name))?;
        let path = dir.join(format!("ref_{}.png", safe_name(&c.name)));
        imgs.first().context("reference portrait produced no image")?.save(&path)?;
        refs.insert(c.name.clone(), path);
    }
    Ok(refs)
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
    /// M2: how many panels had their character's face locked to a reference (0 when `lock` is off or the
    /// face-swap weights weren't available).
    pub faces_locked: usize,
    /// M2: the per-character reference face images used (the "cast reference sheet").
    pub references: Vec<PathBuf>,
    /// D3: how many small swapped faces were refined by the restore pass.
    pub faces_restored: usize,
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

    // M2 reference-lock: build the cast reference sheet + face-swapper once, up front (best-effort).
    let lock_ctx = if opts.lock { build_lock_ctx(spec, device, &panels_dir).await? } else { None };

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
        faces_locked: 0,
        references: lock_ctx.as_ref().map(|c| c.references.clone()).unwrap_or_default(),
        faces_restored: 0,
    };
    let mut seed_cursor = base_seed;
    let mut art_ids: HashMap<String, PathBuf> = HashMap::new(); // D2: panel.id → rendered art, book-wide
    let mut restore_panels: Vec<PathBuf> = Vec::new(); // D3: panels with a small swapped face

    // Phase A — render + lock every page's panels. Compositing is deferred to Phase B so the D3 restore
    // pass (which mutates panel images) runs BEFORE the page is assembled.
    let mut pending: Vec<(usize, super::layout::Plan, Vec<Option<PathBuf>>)> = Vec::with_capacity(n_pages);
    for (pi, lp) in logical.iter().enumerate() {
        let plan = super::layout::resolve_page(spec, lp);
        let prefix = if n_pages <= 1 { String::new() } else { format!("p{pi:02}_") };

        let paths = render_page_panels(spec, &plan, &lp.panels, &cast, device, &panels_dir, seed_cursor, &prefix, &mut art_ids).await?;
        seed_cursor = seed_cursor.wrapping_add(lp.panels.len().max(1) as u64);

        // M2/D1: lock each panel's characters to their reference faces (multi-character: matched by
        // reading-order position). Reused panels (D2) already carry a locked source → skip.
        if let Some(ctx) = &lock_ctx {
            let rtl = plan.reading.eq_ignore_ascii_case("rtl");
            for (r, p) in plan.panels.iter().zip(paths.iter()) {
                if let (Some(pp), Some(panel)) = (p, lp.panels.get(r.panel)) {
                    if panel.reuse.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false) {
                        continue;
                    }
                    let res = lock_panel(ctx, panel, pp, rtl);
                    rep.faces_locked += res.locked;
                    if opts.restore && res.small {
                        restore_panels.push(pp.clone());
                    }
                }
            }
        }
        pending.push((pi, plan, paths));
    }

    // D3 — one restore-faces (ADetailer) pass over the small swapped faces, before compositing.
    if !restore_panels.is_empty() {
        restore_panels.sort();
        restore_panels.dedup();
        rep.faces_restored = restore_small_faces(spec, device, &restore_panels).await;
    }

    // Phase B — composite + letter each page (now that panel images are final).
    for (pi, plan, paths) in &pending {
        let lp = &logical[*pi];
        let bw = plan.border as f32;
        // paths are in reading order; compose indexes by page-panel index — bridge the two.
        let mut imgs: Vec<Option<image::DynamicImage>> = vec![None; lp.panels.len().max(1)];
        for (r, p) in plan.panels.iter().zip(paths.iter()) {
            if let Some(pp) = p {
                imgs[r.panel] = image::open(pp).ok();
            }
        }
        let mut page = super::page::compose(plan, &imgs);

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
            let (pl, rq) = letter_faceaware(&mut page, plan, &lp.panels, &faces);
            rep.lines_placed += pl;
            rep.lines_requested += rq;
        }

        let page_out = page_path(out, *pi, n_pages);
        page.save(&page_out).with_context(|| format!("writing {}", page_out.display()))?;
        let sidecar = page_out.with_extension("panels.json");
        std::fs::write(&sidecar, super::page::panels_json(plan)).with_context(|| format!("writing {}", sidecar.display()))?;

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
        spec.cast = vec![super::super::spec::CastMember { name: "mika".into(), persona: None, describe: Some("a woman with short black hair and a red scarf".into()), reference: None, lock: None }];
        let panel = Panel { scene: Some(scene.into()), chars: chars.iter().map(|s| s.to_string()).collect(), caption: None, balloons: vec![], id: None, reuse: None };
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
    fn reading_order_maps_faces_to_chars_by_position() {
        // three faces at x = 300 (mid), 60 (left), 500 (right).
        let cx = [300.0f32, 60.0, 500.0];
        assert_eq!(reading_order(&cx, false), vec![1, 0, 2], "ltr: left→right");
        assert_eq!(reading_order(&cx, true), vec![2, 0, 1], "rtl: right→left");
    }

    #[test]
    fn name_salt_is_deterministic_and_distinct() {
        assert_eq!(name_salt("mira"), name_salt("mira"), "stable per name");
        assert_ne!(name_salt("mira"), name_salt("bot"), "distinct names → distinct salt");
        assert_eq!(safe_name("Mira Vex!"), "Mira_Vex_");
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
