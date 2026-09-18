//! LAYERED-1 S1 — the **draft** stage. Each layer is rendered ALONE (binding is trivial with one subject),
//! at its box's aspect ratio snapped to the draft family's native area, from the layer's full prompt + the
//! anchored global palette + light + a plain-context suffix. The backdrop is rendered full-canvas. Drafts
//! are cached by a content hash so editing one layer re-renders only that layer.
//!
//! `global.medium` is NEVER sent to a draft: only low frequencies survive into the guide, so medium and
//! technique are the finish's business, while palette and light ARE anchored (and must be right).

use anyhow::{Context, Result};
use image::RgbImage;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::layered::plan::{self, Backdrop, Global, LayerPlan};
use crate::pipelines::scheduler::SchedulerKind;

const PLAIN_CONTEXT: &str = "full view, plain uncluttered surroundings";
const DRAFT_NEGATIVE: &str = "cropped, cut off, multiple subjects, busy background, clutter, text, watermark";

/// Everything that changes a draft's pixels — the cache key.
#[derive(Clone, Debug)]
pub struct DraftSpec {
    pub model: String,
    pub prompt: String,
    pub negative: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub scheduler: SchedulerKind,
}

impl DraftSpec {
    /// A content hash over every pixel-affecting input → the cache filename.
    pub fn cache_key(&self) -> String {
        let mut h = Sha256::new();
        h.update(format!(
            "layered-draft/v1|{}|{}|{}|{}|{}x{}|{}|{:?}",
            self.model, self.prompt, self.negative, self.seed, self.width, self.height, self.steps, self.scheduler
        ));
        format!("{:x}", h.finalize())
    }
}

/// Draft prompt for a subject layer: its own full prompt + the anchored palette + light + a plain context.
/// **No `medium`** — only low frequencies survive the guide.
pub fn draft_prompt(layer_prompt: &str, g: &Global) -> String {
    join_nonempty(&[layer_prompt, g.palette.as_deref().unwrap_or(""), g.light.as_deref().unwrap_or(""), PLAIN_CONTEXT])
}

/// Backdrop prompt: the environment + palette + light (no subjects, no medium).
pub fn backdrop_prompt(b: &Backdrop, g: &Global) -> String {
    join_nonempty(&[b.prompt.as_deref().unwrap_or(""), g.palette.as_deref().unwrap_or(""), g.light.as_deref().unwrap_or("")])
}

fn join_nonempty(parts: &[&str]) -> String {
    parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", ")
}

fn round8(v: f32) -> u32 {
    (((v / 8.0).round() as i64 * 8).clamp(256, 1536)) as u32
}

/// Snap a layer box to a draft render size: keep the box aspect, target the draft family's native AREA
/// (`native_side²`), round each side to a multiple of 8, clamp to `[256, 1536]`.
pub fn box_to_draft_size(bbox: &[f32; 4], out_w: u32, out_h: u32, native_side: u32) -> (u32, u32) {
    let bw = ((bbox[2] - bbox[0]).max(0.02) * out_w as f32).max(1.0);
    let bh = ((bbox[3] - bbox[1]).max(0.02) * out_h as f32).max(1.0);
    let aspect = bw / bh;
    let area = (native_side as f32) * (native_side as f32);
    (round8((area * aspect).sqrt()), round8((area / aspect).sqrt()))
}

/// Backdrop draft size: the output canvas rounded to a multiple of 8 (full-canvas render).
pub fn backdrop_size(out_w: u32, out_h: u32) -> (u32, u32) {
    (round8(out_w as f32), round8(out_h as f32))
}

/// A stable per-layer seed: the plan's draft seed offset by a hash of the layer id (so each layer draws its
/// own noise, reproducibly, and editing one id doesn't disturb the others).
pub fn derive_seed(base: u64, id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    base.wrapping_add(h.finish())
}

fn drafts_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PLAKAT_CACHE_DIR") {
        return PathBuf::from(p).join("layered").join("drafts");
    }
    directories::BaseDirs::new()
        .map(|b| b.home_dir().join(".cache").join("plakat").join("layered").join("drafts"))
        .unwrap_or_else(|| PathBuf::from(".plakat-cache/layered/drafts"))
}

/// The on-disk cache path for a draft spec.
pub fn cache_path(spec: &DraftSpec) -> PathBuf {
    drafts_dir().join(format!("{}.png", spec.cache_key()))
}

/// Render (or load from cache) ONE draft on `device` (a `--device` spec, e.g. `"cpu"`/`"metal"`/`"auto"`).
/// Loads the cached PNG when present; otherwise renders via the resident generate path, caches it, and
/// returns it.
pub async fn render_draft(spec: &DraftSpec, device: &str) -> Result<RgbImage> {
    let path = cache_path(spec);
    if path.exists() {
        return Ok(image::open(&path).with_context(|| format!("loading cached draft {}", path.display()))?.to_rgb8());
    }
    let images = crate::api::Generate::new(&spec.model)
        .prompt(&spec.prompt)
        .negative(&spec.negative)
        .size(spec.width, spec.height)
        .seed(spec.seed)
        .steps(spec.steps)
        .scheduler(spec.scheduler)
        .device(device)
        .run()
        .await
        .with_context(|| format!("rendering draft on {}", spec.model))?;
    let img = images.into_iter().next().ok_or_else(|| anyhow::anyhow!("draft render produced no image"))?;
    std::fs::create_dir_all(drafts_dir()).with_context(|| "creating the drafts cache dir")?;
    img.save(&path)?;
    RgbImage::from_raw(img.width(), img.height(), img.pixels().to_vec()).ok_or_else(|| anyhow::anyhow!("draft buffer size mismatch"))
}

/// Options for [`render_all`].
pub struct DraftOpts {
    pub out_w: u32,
    pub out_h: u32,
    /// The draft model (fast preset welcome).
    pub model: String,
    pub steps: usize,
    pub scheduler: SchedulerKind,
    /// The draft family's native side (area target for layer drafts), e.g. 1024 for SDXL, 512 for SD1.5.
    pub native_side: u32,
    /// Render/refresh only these layer ids (plus the backdrop). `None` = all.
    pub only: Option<Vec<String>>,
    /// The `--device` spec the drafts render on (so they match the finish device). Default `"auto"`.
    pub device: String,
}

/// The rendered draft set: the full-canvas backdrop + a draft per (rendered) layer.
pub struct DraftSet {
    pub backdrop: RgbImage,
    pub layers: Vec<(String, RgbImage)>,
}

/// Build a draft spec for a layer (or the backdrop when `bbox` is `None`).
fn spec_for(prompt: String, seed: u64, w: u32, h: u32, o: &DraftOpts) -> DraftSpec {
    DraftSpec { model: o.model.clone(), prompt, negative: DRAFT_NEGATIVE.to_string(), seed, width: w, height: h, steps: o.steps, scheduler: o.scheduler }
}

/// S1: render (or load) the backdrop + every selected layer draft.
pub async fn render_all(plan: &LayerPlan, o: &DraftOpts) -> Result<DraftSet> {
    let g = &plan.global;
    let base_seed = plan.draft.seed.unwrap_or(0);

    // Backdrop — full canvas.
    let bd = plan.backdrop.clone().unwrap_or_default();
    let (bw, bh) = backdrop_size(o.out_w, o.out_h);
    let bspec = spec_for(backdrop_prompt(&bd, g), derive_seed(base_seed, "__backdrop__"), bw, bh, o);
    let backdrop = render_draft(&bspec, &o.device).await.context("rendering the backdrop draft")?;

    // Layers — each alone, at its box aspect.
    let mut layers = Vec::new();
    for l in &plan.layers {
        if let Some(only) = &o.only {
            if !only.iter().any(|id| id.eq_ignore_ascii_case(&l.id)) {
                continue;
            }
        }
        let bbox = plan::layer_box(l);
        let (w, h) = box_to_draft_size(&bbox, o.out_w, o.out_h, o.native_side);
        let seed = l.seed.unwrap_or_else(|| derive_seed(base_seed, &l.id));
        let spec = spec_for(draft_prompt(l.prompt.as_deref().unwrap_or(""), g), seed, w, h, o);
        let img = render_draft(&spec, &o.device).await.with_context(|| format!("rendering layer {:?}", l.id))?;
        layers.push((l.id.clone(), img));
    }
    Ok(DraftSet { backdrop, layers })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn global() -> Global {
        Global { palette: Some("muted autumn".into()), light: Some("overcast".into()), medium: Some("oil painting".into()) }
    }

    #[test]
    fn draft_prompt_anchors_palette_light_not_medium() {
        let p = draft_prompt("a boy in a yellow raincoat", &global());
        assert!(p.contains("a boy in a yellow raincoat"), "keeps the layer prompt");
        assert!(p.contains("muted autumn") && p.contains("overcast"), "anchors palette + light");
        assert!(p.contains("plain uncluttered"), "plain-context suffix");
        assert!(!p.contains("oil painting"), "medium is NOT sent to drafts");
    }

    #[test]
    fn backdrop_prompt_has_environment_and_look() {
        let b = Backdrop { prompt: Some("cobblestone square".into()), weight: None, window: None };
        let p = backdrop_prompt(&b, &global());
        assert!(p.starts_with("cobblestone square"), "environment first: {p}");
        assert!(p.contains("muted autumn") && p.contains("overcast"));
        assert!(!p.contains("oil painting"), "no medium");
    }

    #[test]
    fn box_size_keeps_aspect_area_and_multiple_of_8() {
        // A wide box (2:1) on a 1024-native family → ~1024² area at 2:1.
        let (w, h) = box_to_draft_size(&[0.0, 0.0, 0.8, 0.4], 1000, 1000, 1024);
        assert_eq!(w % 8, 0);
        assert_eq!(h % 8, 0);
        assert!(w > h, "wider than tall");
        let aspect = w as f32 / h as f32;
        assert!((aspect - 2.0).abs() < 0.15, "aspect ≈ 2:1 (got {aspect})");
        let area = (w * h) as f32;
        assert!((area.sqrt() - 1024.0).abs() < 120.0, "area ≈ native (side {})", area.sqrt());
    }

    #[test]
    fn cache_key_is_deterministic_and_sensitive() {
        let a = DraftSpec { model: "sdxl".into(), prompt: "a cat".into(), negative: "n".into(), seed: 7, width: 1024, height: 1024, steps: 8, scheduler: SchedulerKind::default() };
        let mut b = a.clone();
        assert_eq!(a.cache_key(), b.cache_key(), "same inputs → same key");
        b.prompt = "a dog".into();
        assert_ne!(a.cache_key(), b.cache_key(), "prompt change → new key");
        let mut c = a.clone();
        c.seed = 8;
        assert_ne!(a.cache_key(), c.cache_key(), "seed change → new key");
        assert!(cache_path(&a).to_string_lossy().ends_with(".png"));
    }

    #[test]
    fn derive_seed_is_stable_and_per_id() {
        assert_eq!(derive_seed(100, "boy"), derive_seed(100, "boy"), "stable");
        assert_ne!(derive_seed(100, "boy"), derive_seed(100, "dog"), "per-id");
        assert_ne!(derive_seed(100, "boy"), derive_seed(101, "boy"), "base shifts it");
    }
}
