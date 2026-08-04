//! The render orchestrator — the weights-bearing half of bookart (RFC BOOKART-1 §5.3). Both the CLI
//! (`bookart render` / `illustrate` / `kit` / `manuscript`) and the library facade
//! `plakat::api::BookArt` drive [`render_spec`]: resolve a spec → route to a tier (procedural /
//! diffusion / composite) → finish → symmetry → exact page canvas → an **in-memory** transparent,
//! page-sized ornament (+ optional born-vector SVG + a print/ink scorecard). No file I/O — the caller
//! decides where the bytes go. Extracted from the CLI in 6.1.0 (A1) so every automation surface shares
//! one render core.

use crate::bookart::compile::{self, RenderPlan};
use crate::bookart::scorecard::{self, Scorecard};
use crate::bookart::spec::BookArtSpec;
use crate::bookart::{finish, geometry, procedural};
use anyhow::{Context, Result};
use console::style;
use std::path::PathBuf;

/// Knobs for a bookart render.
#[derive(Debug, Clone)]
pub struct RenderOpts {
    /// Base model for the diffusion/composite tiers (the origin LoRAs are sd15).
    pub model: String,
    pub seed: u64,
    pub steps: usize,
    /// Also produce born-vector SVG (procedural tier). Honoured together with the spec's `output.formats`.
    pub svg: bool,
    /// Diffusion-tier rejection sampling: try up to N seeds, keep the first that clears the scorecard.
    pub attempts: u32,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self { model: "sd15".into(), seed: 0, steps: 28, svg: false, attempts: 1 }
    }
}

/// The in-memory result of a render.
pub struct Rendered {
    /// The transparent, exactly-page-sized ornament.
    pub page: image::RgbaImage,
    /// Born-vector SVG (procedural tier + SVG requested), else `None`.
    pub svg: Option<String>,
    pub plan: RenderPlan,
    pub scorecard: Scorecard,
    /// Number of placed pieces (e.g. 4 for a corner).
    pub pieces: usize,
}

/// A /8-snapped working generation size from a layout rect's aspect (512 short side, cap 768).
fn gen_size(rw: u32, rh: u32) -> (u32, u32) {
    let ar = rw.max(1) as f32 / rh.max(1) as f32;
    let (mut w, mut h) = if ar >= 1.0 { (512.0 * ar, 512.0) } else { (512.0, 512.0 / ar) };
    let scale = (768.0 / w.max(h)).min(1.0);
    w *= scale;
    h *= scale;
    let snap = |v: f32| (((v / 8.0).round() * 8.0) as u32).clamp(256, 768);
    (snap(w), snap(h))
}

/// Generate one diffusion render (with the origin LoRA if not `generic`) at `w×h`. Returns the raw RGB
/// plus the temp path it was written to (kept for an optional U2Net matte; the caller deletes it).
async fn diffuse(model: &str, plan: &RenderPlan, w: u32, h: u32, steps: usize, seed: u64) -> Result<(image::RgbImage, PathBuf)> {
    let mut prompt = plan.prompt.clone();
    let mut builder = crate::api::Generate::new(model).negative(&plan.negative).size(w, h).steps(steps).seed(seed);
    if plan.origin != "generic" {
        prompt = format!("{prompt}, bookart_{} style", plan.origin);
        builder = builder.lora(format!("vulogov98/plakat-bookart#{}-sd15.safetensors", plan.origin), 1.0);
        println!("  {} origin LoRA: bookart_{} (sd15)", style("↳").cyan(), plan.origin);
    } else {
        println!("  {} generic line-art path (no LoRA)", style("↳").dim());
    }
    println!("  {} diffusion {w}×{h}, {steps} steps, seed {seed}…", style("→").cyan());
    let imgs = builder.prompt(&prompt).run().await.context("diffusion render")?;
    let img = imgs.into_iter().next().context("diffusion produced no image")?;
    let tmp = std::env::temp_dir().join(format!("bookart_diff_{seed}_{w}x{h}.png"));
    img.save(&tmp)?;
    let raw = image::open(&tmp).context("reopening the diffusion render")?.to_rgb8();
    Ok((raw, tmp))
}

/// U2Net matte → a solid silhouette in the ink tint (the `transparency: matte` mode).
async fn matte_silhouette(path: &std::path::Path, raw: &image::RgbImage, plan: &RenderPlan) -> Result<image::RgbaImage> {
    let device = crate::api::device("auto")?;
    let (_fg, mask) = crate::pipelines::matting::matte(path, &device).await.context("U2Net matte")?;
    let mask = if mask.dimensions() != raw.dimensions() { image::imageops::resize(&mask, raw.width(), raw.height(), image::imageops::FilterType::Triangle) } else { mask };
    let tint = finish::parse_tint(&plan.tint);
    let mut out = image::RgbaImage::new(raw.width(), raw.height());
    for (x, y, p) in out.enumerate_pixels_mut() {
        *p = image::Rgba([tint[0], tint[1], tint[2], mask.get_pixel(x, y).0[0]]);
    }
    Ok(out)
}

/// A stable 64-bit fingerprint (FNV-1a, hex) of a resolved plan's identity — the "spec-hash" recorded
/// in the sidecar so two ornaments that would render identically share a hash regardless of surface.
/// Dependency-free and stable across builds (unlike `DefaultHasher`).
fn spec_hash(plan: &RenderPlan) -> String {
    let id = format!(
        "{}|{}|{}|{}|{}|{}|{:.3}|{}|{}|{}x{}@{}",
        plan.origin, plan.technique, plan.tier, plan.ornament_kind, plan.symmetry,
        plan.tint, plan.ink_weight, plan.motif.join(","), plan.prompt,
        plan.page.w_px, plan.page.h_px, plan.page.dpi,
    );
    let mut h: u64 = 0xcbf29ce484222325;
    for b in id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Build the reproducibility recipe for a rendered ornament (RFC BOOKART-1 §20, A5). Carries the
/// origin / technique / tier / ornament / symmetry / page + a stable **spec-hash** in `extras`, the
/// origin LoRA in `loras`, and the diffusion knobs — so the PNG `tEXt` chunk + `.json` sidecar make
/// the ornament searchable and re-runnable, and `--import` lands it in an album already curated.
pub fn recipe_metadata(plan: &RenderPlan, model: &str, seed: u64, steps: usize) -> crate::imaging::metadata::GenerationMetadata {
    use crate::imaging::metadata::GenerationMetadata;
    // Procedural ornaments have no prompt — synthesise a descriptive one so the recipe still reads.
    let prompt = if plan.prompt.trim().is_empty() {
        format!("bookart {} {} ({} ornament)", plan.origin, plan.ornament_kind, plan.tier)
    } else {
        plan.prompt.clone()
    };
    let mut m = GenerationMetadata::new(prompt, model, seed, steps, 0.0, "bookart", plan.page.w_px, plan.page.h_px);
    if !plan.negative.trim().is_empty() {
        m.negative = plan.negative.clone();
    }
    if plan.origin != "generic" {
        m.loras = vec![format!("vulogov98/plakat-bookart#{}-sd15.safetensors", plan.origin)];
    }
    m.extras = vec![
        ("Bookart origin".into(), plan.origin.clone()),
        ("Bookart technique".into(), plan.technique.clone()),
        ("Bookart tier".into(), plan.tier.clone()),
        ("Bookart ornament".into(), plan.ornament_kind.clone()),
        ("Bookart symmetry".into(), plan.symmetry.clone()),
        // ASCII `x` (not `×`) so the Latin-1 PNG tEXt chunk stays pure-ASCII and portable.
        ("Bookart page".into(), format!("{}x{} @ {} DPI", plan.page.w_px, plan.page.h_px, plan.page.dpi)),
        ("Bookart spec-hash".into(), spec_hash(plan)),
    ];
    m
}

/// Render a spec to an in-memory ornament — the single render core behind every bookart surface.
pub async fn render_spec(spec: &BookArtSpec, opts: &RenderOpts) -> Result<Rendered> {
    let plan = compile::resolve(spec);
    let tb = geometry::text_block(&plan.page, spec);
    let layout = geometry::layout_for(&plan.ornament_kind, &tb);
    let r0 = layout.rects[0];
    let variant = (opts.seed % 8) as u32; // diversifies procedural ornament across a set/manuscript

    let ornament = match plan.tier.as_str() {
        // B3: vector-native, no weights.
        "procedural" => {
            let gray = procedural::generate(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h, variant);
            finish::finish_procedural(&gray, &plan)
        }
        // B4: diffusion + optional matte, with B6 scorecard rejection sampling.
        "diffusion" => {
            let (gw, gh) = gen_size(r0.w, r0.h);
            let tries = opts.attempts.max(1);
            let (mut best, mut fewest) = (None, usize::MAX);
            for i in 0..tries {
                let (raw, tmp) = diffuse(&opts.model, &plan, gw, gh, opts.steps, opts.seed + i as u64).await?;
                let finished = if plan.transparency_mode == "matte" {
                    println!("  {} U2Net matte → silhouette", style("↳").cyan());
                    matte_silhouette(&tmp, &raw, &plan).await?
                } else {
                    finish::finish_ornament(&raw, &plan)
                };
                let _ = std::fs::remove_file(&tmp);
                let sc = scorecard::score(&finished, &plan);
                if sc.passes {
                    if tries > 1 {
                        println!("  {} scorecard PASS on attempt {}/{}", style("✓").green(), i + 1, tries);
                    }
                    best = Some(finished);
                    break;
                }
                if sc.notes.len() < fewest {
                    fewest = sc.notes.len();
                    best = Some(finished);
                }
                if tries > 1 {
                    println!("  {} attempt {}/{} FAIL ({} issue(s)), retrying…", style("·").yellow(), i + 1, tries, sc.notes.len());
                }
            }
            best.context("no diffusion image")?
        }
        // B5: composite — procedural frame + diffusion line-art inlay.
        "composite" => {
            let (frame_paths, (wx, wy, ww, wh)) = procedural::frame(&plan.symmetry, r0.w, r0.h);
            let width = (r0.w.min(r0.h) as f32 * 0.004).max(1.5);
            let frame_rgba = finish::finish_procedural(&procedural::rasterise(&frame_paths, r0.w, r0.h, width), &plan);
            let (gw, gh) = gen_size(ww, wh);
            let (raw, tmp) = diffuse(&opts.model, &plan, gw, gh, opts.steps, opts.seed).await?;
            let _ = std::fs::remove_file(&tmp);
            let inlay_gray = finish::binarize::binarise(&finish::to_luma(&raw), "xdog", plan.ink_weight);
            let inlay = finish::alpha::to_transparent(&inlay_gray, "luminance", finish::parse_tint(&plan.tint), 0.0);
            let mut canvas = image::RgbaImage::from_pixel(r0.w, r0.h, image::Rgba([0, 0, 0, 0]));
            let pic = image::imageops::resize(&inlay, ww.max(1), wh.max(1), image::imageops::FilterType::Lanczos3);
            image::imageops::overlay(&mut canvas, &pic, wx as i64, wy as i64);
            image::imageops::overlay(&mut canvas, &frame_rgba, 0, 0);
            println!("  {} composite: procedural frame + diffusion inlay", style("↳").cyan());
            canvas
        }
        other => anyhow::bail!("unknown render tier `{other}`"),
    };

    // Symmetry (no-op for `none`); skipped for `composite` (frame already symmetric; picture is a scene).
    let orn = if plan.tier == "composite" { ornament } else { geometry::symmetrize(&ornament, &plan.symmetry) };
    let page = finish::canvas::place_on_canvas(&orn, &plan.page, &layout);
    let sc = scorecard::score(&page, &plan);

    // Opt-in born-vector SVG (§7.5) — procedural only; the raster trace is a documented fast-follow (B1).
    let want_svg = opts.svg || plan.formats.iter().any(|f| f == "svg");
    let svg = if want_svg && plan.tier == "procedural" {
        let paths = procedural::generate_paths(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h, variant);
        let stroke = (r0.w.min(r0.h) as f32 * 0.004).max(1.5);
        let all: Vec<_> = layout.rects.iter().flat_map(|r| finish::vector::transform_to_rect(&paths, r, r0.w, r0.h)).collect();
        Some(finish::vector::polylines_to_svg(&all, plan.page.w_px, plan.page.h_px, plan.page.dpi, stroke, finish::parse_tint(&plan.tint)))
    } else {
        None
    };

    Ok(Rendered { page, svg, plan, scorecard: sc, pieces: layout.rects.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_render_core_is_page_sized_transparent_with_svg() {
        // The procedural tier needs no weights, so the shared render core is fully CI-testable.
        let spec = BookArtSpec::from_hjson(r#"{"origin":"russian","ornament":{"type":"border"}}"#).unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let r = rt.block_on(render_spec(&spec, &RenderOpts { svg: true, ..Default::default() })).unwrap();
        assert_eq!(r.plan.tier, "procedural");
        assert_eq!(r.page.dimensions(), (r.plan.page.w_px, r.plan.page.h_px), "page-sized");
        assert_eq!(r.page.get_pixel(0, 0).0[3], 0, "corner transparent");
        assert!(r.page.pixels().any(|p| p.0[3] > 200), "has ink");
        assert!(r.svg.as_deref().is_some_and(|s| s.contains("<svg")), "born-vector SVG emitted");
        assert!(r.scorecard.chroma_frac < 0.01, "neutral B/W");
    }

    #[test]
    fn recipe_carries_bookart_provenance_and_stable_spec_hash() {
        let spec = BookArtSpec::from_hjson(r#"{"origin":"russian","technique":"line","ornament":{"type":"border"}}"#).unwrap();
        let plan = compile::resolve(&spec);
        let m = recipe_metadata(&plan, "sd15", 7, 28);
        // Non-generic origin → the hosted origin LoRA is recorded for reproducibility.
        assert_eq!(m.loras, vec!["vulogov98/plakat-bookart#russian-sd15.safetensors".to_string()]);
        let get = |k: &str| m.extras.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.as_str());
        assert_eq!(get("Bookart origin"), Some("russian"));
        assert_eq!(get("Bookart technique"), Some("line"));
        assert_eq!(get("Bookart ornament"), Some("border"));
        let h = get("Bookart spec-hash").unwrap().to_string();
        assert_eq!(h.len(), 16, "16-hex FNV-1a fingerprint");
        // Stable: same plan → same hash.
        assert_eq!(spec_hash(&compile::resolve(&spec)), h);
        // Sensitive: a different origin → a different hash + no LoRA for `generic`.
        let generic = BookArtSpec::from_hjson(r#"{"origin":"generic","ornament":{"type":"border"}}"#).unwrap();
        let gm = recipe_metadata(&compile::resolve(&generic), "sd15", 7, 28);
        assert!(gm.loras.is_empty(), "generic path records no origin LoRA");
        assert_ne!(spec_hash(&compile::resolve(&generic)), h, "origin change moves the hash");
    }
}
