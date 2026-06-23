//! `plakat multiperson` — place specific personas into a generated scene.
//!
//! M1 (placement → existing Form-2 inpaint): the user gives each persona a
//! *relative location* in words (`at: "left closer front"`); unpinned personas
//! are placed by a scene-aware LLM so they fit naturally. Each placement →
//! a screen region → the scene base is generated, then each persona is inpainted
//! into their region with their identity (reusing the portrait pipeline).
//!
//! See `Documentation/RFC_MULTIPERSON_REVIEW.md` for the design + milestones.

pub mod analyser;
pub mod placement;
pub mod prompt;

pub use placement::{Distance, Facing, Placement, Position};
pub use prompt::MultipersonPrompt;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::PathBuf;

use crate::pipelines::ip_adapter::{IdentityKind, WeightedPhoto};
use crate::pipelines::portrait;

/// One persona to place into the scene.
pub struct Person {
    pub label: String,
    pub photos: Vec<WeightedPhoto>,
    /// Relative location (`at:`). `None` → auto-placed by the scene analyser.
    pub placement: Option<Placement>,
    /// Explicit pixel region override `[x0,y0,x1,y1]`; wins over `placement`.
    pub bbox: Option<[f32; 4]>,
    /// Behavioural prompt ("laughing, leaning forward").
    pub prompt: Option<String>,
    pub face_strength: Option<f32>,
    pub face_bbox: Option<[f32; 4]>,
    pub face_landmarks: Option<[[f32; 2]; 5]>,
}

/// A `plakat multiperson` request (Form B — inline prose + people).
pub struct MultipersonRequest {
    pub scene: String,
    pub people: Vec<Person>,
    pub model: String,
    pub identity: IdentityKind,
    pub style: Option<String>,
    pub negative: String,
    pub layout_provider: String,
    pub enhancer: Option<String>,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub count: u32,
    pub out_dir: PathBuf,
    pub scheduler: crate::pipelines::scheduler::SchedulerKind,
    pub device: Device,
    pub dry_run: bool,
}

/// A persona resolved to a concrete screen region + how it's conditioned.
struct Resolved {
    idx: usize,
    bbox: [f32; 4],
    facing_phrase: Option<&'static str>,
    /// Sort key: render farther personas first, closer ones last (occlusion).
    order_y: f32,
    source: &'static str, // "at" | "bbox" | "auto"
}

impl MultipersonRequest {
    /// Resolve every persona to a screen region: explicit bbox > `at:` location
    /// > scene-aware LLM auto-placement (for any persona left unpinned).
    async fn resolve(&self) -> Vec<Resolved> {
        let mp = MultipersonPrompt::parse(&self.scene);
        let base_seed = self.seed.unwrap_or(0);

        // Pinned personas first (bbox or `at:`); collect their placements so the
        // analyser can avoid those zones for the unpinned ones.
        let occupied: Vec<Placement> =
            self.people.iter().filter_map(|p| p.placement).collect();
        let unpinned: Vec<usize> = self
            .people
            .iter()
            .enumerate()
            .filter(|(_, p)| p.placement.is_none() && p.bbox.is_none())
            .map(|(i, _)| i)
            .collect();

        let auto = analyser::auto_place(
            &self.layout_provider,
            &self.device,
            mp.for_analyser(),
            unpinned.len(),
            &occupied,
            base_seed,
        )
        .await;
        let auto_map: std::collections::HashMap<usize, Placement> =
            unpinned.iter().copied().zip(auto).collect();

        let mut out = Vec::with_capacity(self.people.len());
        for (i, p) in self.people.iter().enumerate() {
            let (bbox, facing, source) = if let Some(b) = p.bbox {
                (b, None, "bbox")
            } else if let Some(pl) = p.placement {
                (pl.bbox(), Some(pl.facing_phrase()), "at")
            } else {
                let pl = auto_map.get(&i).copied().unwrap_or_default();
                (pl.bbox(), Some(pl.facing_phrase()), "auto")
            };
            out.push(Resolved {
                idx: i,
                order_y: (bbox[1] + bbox[3]) * 0.5,
                bbox,
                facing_phrase: facing,
                source,
            });
        }
        // Farther (higher on screen, smaller order_y) first; closer last.
        out.sort_by(|a, b| a.order_y.partial_cmp(&b.order_y).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

/// Run a multiperson generation (Form B). M1: placement → scene base →
/// per-persona inpaint (reusing the portrait pipeline).
pub async fn run(req: MultipersonRequest) -> Result<()> {
    anyhow::ensure!(req.people.len() >= 2, "multiperson needs at least 2 people; got {}", req.people.len());

    let resolved = req.resolve().await;
    let mp = MultipersonPrompt::parse(&req.scene);
    let enhancer_base_raw = mp.enhancer_base(None, req.style.as_deref(), None);

    // ---- dry-run: print the placement plan, no models ----
    if req.dry_run {
        println!("multiperson · {} people · model {} · identity {:?}", req.people.len(), req.model, req.identity);
        println!("scene  → analyser: \"{}\"", mp.for_analyser());
        println!("base   → enhancer: \"{enhancer_base_raw}\"");
        println!("placement (render order: farther → closer):");
        for r in &resolved {
            let p = &req.people[r.idx];
            let b = r.bbox;
            println!(
                "  {:<10} [{}] bbox [{:.2},{:.2},{:.2},{:.2}]{}",
                p.label, r.source, b[0], b[1], b[2], b[3],
                r.facing_phrase.map(|f| format!("  ({f})")).unwrap_or_default(),
            );
        }
        return Ok(());
    }

    // ---- load the portrait pipeline (SD backbone + identity encoder) ----
    let pipe = portrait::Pipeline::load(portrait::LoadRequest {
        model: req.model.clone(),
        device: req.device.clone(),
        loras: Vec::new(),
        lora_scale: 1.0,
        identity: Some(req.identity),
        shared_clip_h: None,
    })
    .await
    .context("loading portrait backbone for multiperson")?;
    let dtype = pipe.core().dtype;
    let (lh, lw) = (req.height as usize / 8, req.width as usize / 8);

    // Optionally enhance the shared scene+style base once (per-region prompts
    // append the persona + facing on top of it).
    let base_prompt = match &req.enhancer {
        Some(alias) if !alias.is_empty() && !alias.eq_ignore_ascii_case("none") => {
            match crate::llm::enhance(
                alias,
                req.device.clone(),
                crate::prompt::SYSTEM,
                &enhancer_base_raw,
                crate::llm::EnhanceOpts { seed: req.seed.unwrap_or(0), temperature: 0.0, max_new_tokens: 160 },
            )
            .await
            {
                Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
                _ => enhancer_base_raw.clone(),
            }
        }
        _ => enhancer_base_raw.clone(),
    };

    std::fs::create_dir_all(&req.out_dir).ok();
    let base_seed = req.seed.unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));

    for n in 0..req.count.max(1) {
        let seed = base_seed.wrapping_add(n as u64);
        let mk_req = |prompt: String, photos: Vec<WeightedPhoto>, face_strength: f32,
                      face_bbox, face_landmarks| portrait::GenRequest {
            prompt,
            negative: req.negative.clone(),
            photos,
            width: req.width,
            height: req.height,
            count: 1,
            steps: req.steps,
            guidance: req.guidance,
            seed: Some(seed),
            out_dir: req.out_dir.clone(),
            scheduler: req.scheduler,
            refine: None,
            refine_strength: 0.0,
            face_strength,
            face_bbox,
            face_landmarks,
        };

        // 1) scene base — text-only, all personas absent.
        let base_req = mk_req(base_prompt.clone(), Vec::new(), 0.0, None, None);
        let mut latents = pipe
            .generate_latents_one(&base_req, seed, &[])
            .context("multiperson: scene base generation")?;

        // 2) inpaint each persona into their region, farther → closer.
        for r in &resolved {
            let p = &req.people[r.idx];
            let mask = crate::pipelines::tiled::region_mask(r.bbox, lh, lw, &req.device, dtype)?;
            let region_prompt = mp.region_prompt(&base_prompt, p.prompt.as_deref(), r.facing_phrase);
            let preq = mk_req(
                region_prompt,
                p.photos.clone(),
                p.face_strength.unwrap_or(0.8),
                p.face_bbox,
                p.face_landmarks,
            );
            latents = pipe
                .inpaint_latents_one(&latents, &mask, &preq, seed, &[], None)
                .with_context(|| format!("multiperson: inpaint persona '{}'", p.label))?;
        }

        let out = req.out_dir.join(format!("plakat-multiperson-{seed}.png"));
        pipe.save_image(&latents, &out)?;
        write_sidecar(&req, &resolved, &mp, &base_prompt, seed, &out)?;
        crate::ui::progress::println(&format!("  {} {}", console::style("✓").green().bold(), out.display()));
    }
    Ok(())
}

fn write_sidecar(
    req: &MultipersonRequest,
    resolved: &[Resolved],
    mp: &MultipersonPrompt,
    base_prompt: &str,
    seed: u64,
    out: &std::path::Path,
) -> Result<()> {
    let persons: Vec<serde_json::Value> = resolved
        .iter()
        .map(|r| {
            let p = &req.people[r.idx];
            serde_json::json!({
                "label": p.label,
                "placement_source": r.source,
                "bbox": r.bbox,
                "facing": r.facing_phrase,
                "behavioral_prompt": p.prompt,
                "face_strength": p.face_strength.unwrap_or(0.8),
            })
        })
        .collect();
    let v = serde_json::json!({
        "command": "multiperson",
        "scene_clause": mp.for_analyser(),
        "style_clause": mp.style_clause,
        "base_prompt": base_prompt,
        "model": req.model,
        "identity": format!("{:?}", req.identity),
        "layout_provider": req.layout_provider,
        "seed": seed,
        "size": format!("{}x{}", req.width, req.height),
        "steps": req.steps,
        "guidance": req.guidance,
        "persons": persons,
    });
    std::fs::write(out.with_extension("json"), serde_json::to_string_pretty(&v)?)?;
    Ok(())
}
