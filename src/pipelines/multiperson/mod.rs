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
pub mod pose;
pub mod prompt;
pub mod scenario_task;

pub use placement::{Distance, Facing, Placement, Position};
pub use prompt::MultipersonPrompt;

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::PathBuf;

use crate::pipelines::ip_adapter::{IdentityKind, WeightedPhoto};
use crate::pipelines::portrait;
use crate::pipelines::scrfd::{Face, SCRFDConfig, SCRFDDetector};

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
    /// Figure height relative to a full-grown adult, for the `--pose` skeleton.
    /// `1.0` = adult; `~0.7` = a child/teen so they render shorter. `None` → 1.0.
    pub scale: Option<f32>,
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
    /// Composite identity path: generate the scene background with any model, then
    /// matte + place each persona's actual photo. Exact identity, model-agnostic.
    pub composite: bool,
    /// With composite: relight each persona to the scene lighting (IC-Light) before
    /// placing — the real integration step.
    pub relight: bool,
    /// With composite: optional img2img harmonize strength over the final image.
    pub harmonize: Option<f32>,
    /// Pin each figure's position/pose with a synthetic OpenPose ControlNet (one
    /// skeleton per persona region) during scene generation, so persona↔figure
    /// binding holds. Applies to the `--swap` path.
    pub pose: bool,
    /// Face-swap identity path: generate one coherent scene, then swap each
    /// detected face with the persona matched to its placement region.
    pub swap: bool,
    /// After `--swap`, run a low-strength ADetailer-style detail pass on the
    /// swapped faces (sharpens small scene faces). Off by default — at higher
    /// strength img2img can drift the swapped identity.
    pub restore_faces: bool,
    /// Run the identity face-refinement pass after the body inpaint (detect each
    /// face with SCRFD, re-inpaint the face crop at high identity strength). This
    /// is what actually makes the personas *look like* their reference photos —
    /// identity injected over a whole body region is too diluted. Off → body-only.
    pub refine_faces: bool,
    /// IP-Adapter identity scale for the face-refinement pass. High — the face
    /// crop is plus-face's sweet spot.
    pub refine_face_strength: f32,
    /// Denoise strength of the face-refinement repaint (0..1). LOW keeps the
    /// existing face's scale/framing and just nudges identity + detail (the
    /// ADetailer-style light touch); high regenerates it (and can distort).
    pub refine_denoise: f32,
}

/// A persona resolved to a concrete screen region + how it's conditioned.
struct Resolved {
    idx: usize,
    bbox: [f32; 4],
    facing_phrase: Option<&'static str>,
    /// Facing enum (for synthetic OpenPose skeletons). `Front` for explicit bbox.
    facing: Facing,
    /// Figure height scale for the `--pose` skeleton (child < 1.0).
    scale: f32,
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
            let (bbox, facing_phrase, facing, source) = if let Some(b) = p.bbox {
                (b, None, Facing::Front, "bbox")
            } else if let Some(pl) = p.placement {
                (pl.bbox(), Some(pl.facing_phrase()), pl.facing, "at")
            } else {
                let pl = auto_map.get(&i).copied().unwrap_or_default();
                (pl.bbox(), Some(pl.facing_phrase()), pl.facing, "auto")
            };
            out.push(Resolved {
                idx: i,
                order_y: (bbox[1] + bbox[3]) * 0.5,
                bbox,
                facing_phrase,
                facing,
                scale: p.scale.unwrap_or(1.0).clamp(0.4, 1.0),
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
    anyhow::ensure!(!req.people.is_empty(), "multiperson needs at least one person");

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

    // ---- composite identity path: any-model background + matted real photos ----
    if req.composite {
        return run_composite(&req, &resolved, &mp).await;
    }

    // ---- face-swap identity path: one coherent scene, then swap each face ----
    if req.swap {
        return run_swap(&req, &resolved, &mp).await;
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

    // Optional SCRFD detector for the identity-refinement pass. The refinement
    // itself ALWAYS runs when enabled (it's the identity fix) — by default it
    // locates each face geometrically from the persona's placement region (which
    // is exact), needing no extra weights. If the user has configured SCRFD we use
    // its detected boxes instead for tighter alignment with the rendered face.
    let detector = if req.refine_faces {
        match crate::pipelines::scrfd::resolve_scrfd_weights().await {
            Ok(Some(path)) => match SCRFDDetector::load(
                &path,
                SCRFDConfig::default(),
                &req.device,
                candle_core::DType::F32,
            ) {
                Ok(d) => Some(d),
                Err(e) => {
                    crate::ui::progress::println(&format!(
                        "  {} SCRFD load failed ({e}); face-refine will use geometric boxes",
                        console::style("!").yellow().bold()
                    ));
                    None
                }
            },
            _ => None, // no env config → geometric face boxes (still refines).
        }
    } else {
        None
    };

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

        // 2) inpaint each persona into their region, farther → closer. The region
        //    prompt is SINGLE-person (not the whole-scene "three people" clause),
        //    so the identity isn't diluted across the crowd.
        for r in &resolved {
            let p = &req.people[r.idx];
            let mask = crate::pipelines::tiled::region_mask(r.bbox, lh, lw, &req.device, dtype)?;
            let region_prompt = mp.single_region_prompt(p.prompt.as_deref(), r.facing_phrase);
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

        // 3) identity face-refinement: identity injected over a whole body region
        //    is weak (the face is small in a tall mask). Re-inpaint just each
        //    persona's face crop at high identity strength with a portrait framing
        //    — plus-face's sweet spot. Face boxes come from SCRFD when configured,
        //    else geometrically from the (exact) placement region. This is what
        //    makes the outputs actually resemble the source photos.
        if req.refine_faces {
            latents = refine_persona_faces(
                &pipe, detector.as_ref(), latents, &resolved, &req, &mp, seed, lh, lw, dtype,
                &mk_req,
            )
            .with_context(|| format!("multiperson: face-refinement pass (seed {seed})"))?;
        }

        let out = req.out_dir.join(format!("plakat-multiperson-{seed}.png"));
        pipe.save_image(&latents, &out)?;
        write_sidecar(&req, &resolved, &mp, &base_prompt, seed, &out)?;
        crate::ui::progress::println(&format!("  {} {}", console::style("✓").green().bold(), out.display()));
    }
    Ok(())
}

/// Composite identity path — the model-agnostic, exact-identity route. Generate
/// the scene background with ANY text-to-image model, then matte each persona's
/// actual photo (U2Net, no face model) and place it at its `--at` region (scaled
/// to the region, farther personas behind closer ones). Optionally img2img the
/// finished composite to harmonise lighting/style. No IP-Adapter / face encoder,
/// so it works on every model `generate` supports.
async fn run_composite(
    req: &MultipersonRequest,
    resolved: &[Resolved],
    mp: &MultipersonPrompt,
) -> Result<()> {
    use crate::pipelines::{matting, t2i};

    let base_prompt = mp.enhancer_base(None, req.style.as_deref(), None);
    std::fs::create_dir_all(&req.out_dir).ok();
    let base_seed = req.seed.unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));

    // IC-Light relighter (optional) — relights each person to the scene lighting
    // before compositing, so they belong in the scene instead of looking pasted.
    let relighter = if req.relight {
        Some(
            crate::pipelines::ic_light::Pipeline::load(req.device.clone())
                .await
                .context("loading IC-Light for --relight")?,
        )
    } else {
        None
    };

    for n in 0..req.count.max(1) {
        let seed = base_seed.wrapping_add(n as u64);

        // 1) Scene background — any model (no people needed; they're composited).
        let bg_dir = tempfile::Builder::new().prefix("plakat-mp-bg-").tempdir()?;
        t2i::run(t2i::Request::simple(
            base_prompt.clone(),
            req.model.clone(),
            req.width,
            req.height,
            req.steps,
            Some(seed),
            req.device.clone(),
            bg_dir.path().to_path_buf(),
        ))
        .await
        .context("multiperson --composite: background generation")?;
        let bg_path = std::fs::read_dir(bg_dir.path())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
            .context("background render produced no PNG")?;
        let mut canvas = image::open(&bg_path)?.to_rgba8();

        // 2) Matte + place each persona (resolved is sorted farther → closer, so
        //    compositing in order puts nearer people in front).
        for r in resolved {
            let p = &req.people[r.idx];
            let photo = &p.photos.first().context("persona has no photo")?.path;

            // Relight the person to the scene's lighting first (IC-Light), then
            // re-matte the relit subject. Otherwise matte the raw photo. The
            // `_relit` holder keeps the temp file alive until matting reads it.
            let _relit: Option<tempfile::NamedTempFile>;
            let matte_src: std::path::PathBuf = if let Some(ic) = &relighter {
                let (buf, w, h) = ic
                    .relight(photo, &base_prompt, &req.negative, 512, 640, req.steps, 2.0, seed)
                    .with_context(|| format!("relighting persona '{}'", p.label))?;
                let rt = tempfile::Builder::new().prefix("plakat-mp-relit-").suffix(".png").tempfile()?;
                crate::imaging::io::save_rgb_u8(&buf, w, h, rt.path())?;
                let path = rt.path().to_path_buf();
                _relit = Some(rt);
                crate::ui::progress::println(&format!(
                    "  {} relit {}",
                    console::style("·").cyan(),
                    p.label
                ));
                path
            } else {
                _relit = None;
                photo.clone()
            };

            let tmp = tempfile::Builder::new().prefix("plakat-mp-cut-").suffix(".png").tempfile()?;
            matting::cutout(&matte_src, tmp.path(), true, &req.device)
                .await
                .with_context(|| format!("matting persona '{}'", p.label))?;
            let cut = image::open(tmp.path())?.to_rgba8();

            // Fit the cut-out inside the placement region, preserving aspect.
            let (bw, bh) = (
                (r.bbox[2] - r.bbox[0]) * req.width as f32,
                (r.bbox[3] - r.bbox[1]) * req.height as f32,
            );
            let (cw, ch) = (cut.width() as f32, cut.height() as f32);
            let scale = (bw / cw).min(bh / ch);
            let (nw, nh) = ((cw * scale).round().max(1.0) as u32, (ch * scale).round().max(1.0) as u32);
            let resized = image::imageops::resize(&cut, nw, nh, image::imageops::FilterType::Lanczos3);
            // Centre horizontally in the region; anchor to the region top.
            let px = (r.bbox[0] * req.width as f32 + (bw - nw as f32) * 0.5).round() as i64;
            let py = (r.bbox[1] * req.height as f32).round() as i64;
            crate::cli::compose::composite(&mut canvas, &resized, px, py, 1.0);
            crate::ui::progress::println(&format!(
                "  {} placed {}",
                console::style("·").cyan(),
                p.label
            ));
        }

        let out = req.out_dir.join(format!("plakat-multiperson-{seed}.png"));
        image::DynamicImage::ImageRgba8(canvas).to_rgb8().save(&out)?;

        // 3) Optional harmonisation: a light img2img over the whole composite so
        //    the placed people share the scene's lighting/style.
        if let Some(strength) = req.harmonize {
            let h_dir = tempfile::Builder::new().prefix("plakat-mp-harm-").tempdir()?;
            let hreq = crate::pipelines::img2img::Request {
                prompt: base_prompt.clone(),
                negative: req.negative.clone(),
                model: req.model.clone(),
                device: req.device.clone(),
                loras: Vec::new(),
                lora_scale: 1.0,
                input: out.clone(),
                mask: None,
                mask_feather: 0,
                mask_invert: false,
                width: req.width,
                height: req.height,
                count: 1,
                steps: req.steps,
                guidance: req.guidance,
                scheduler: req.scheduler,
                strength: strength.clamp(0.05, 0.95),
                seed: Some(seed),
                out_dir: h_dir.path().to_path_buf(),
                controls: Vec::new(),
            };
            crate::pipelines::img2img::run(hreq)
                .await
                .context("multiperson --composite: harmonize pass")?;
            if let Some(h) = std::fs::read_dir(h_dir.path())?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
            {
                std::fs::copy(&h, &out)?;
            }
        }

        write_sidecar(req, resolved, mp, &base_prompt, seed, &out)?;
        crate::ui::progress::println(&format!(
            "  {} {}",
            console::style("✓").green().bold(),
            out.display()
        ));
    }
    Ok(())
}

/// Face-swap identity path: generate ONE coherent scene (plain text-to-image,
/// which composes properly), detect every face, match each persona to the face in
/// its placement region, and swap that face with the persona's source identity.
/// Personas with no face in their region are reported, not silently dropped.
async fn run_swap(
    req: &MultipersonRequest,
    resolved: &[Resolved],
    mp: &MultipersonPrompt,
) -> Result<()> {
    use crate::pipelines::faceswap::FaceSwapper;
    use crate::pipelines::{controlnet::ControlSpec, t2i};

    let base_prompt = mp.enhancer_base(None, req.style.as_deref(), None);

    // Optional OpenPose composition: one synthetic skeleton per persona region,
    // pinned so the model places a figure exactly where each persona goes — fixing
    // the persona↔figure binding. Rendered once, reused each generation.
    let _pose_tmp; // keep the pose-map file alive for the whole run
    let pose_map_path: Option<std::path::PathBuf> = if req.pose {
        let regions: Vec<([f32; 4], Facing, f32)> =
            resolved.iter().map(|r| (r.bbox, r.facing, r.scale)).collect();
        let map = pose::render_pose_map(&regions, req.width, req.height);
        let t = tempfile::Builder::new().prefix("plakat-mp-pose-").suffix(".png").tempfile()?;
        map.save(t.path())?;
        let path = t.path().to_path_buf();
        _pose_tmp = Some(t);
        crate::ui::progress::println(&format!(
            "  {} OpenPose composition: {} skeleton(s) pinned",
            console::style("·").cyan(),
            regions.len()
        ));
        Some(path)
    } else {
        _pose_tmp = None;
        None
    };

    let swapper = FaceSwapper::load_resolved(&req.device, candle_core::DType::F32)
        .await
        .context("loading face-swap models")?;

    // Source identity latent per persona (from their first reference photo).
    let mut latents: Vec<candle_core::Tensor> = Vec::with_capacity(req.people.len());
    for p in &req.people {
        let path = &p.photos.first().context("persona has no photo")?.path;
        latents.push(
            swapper
                .source_latent(path)
                .with_context(|| format!("embedding source identity for '{}'", p.label))?,
        );
    }

    std::fs::create_dir_all(&req.out_dir).ok();
    let base_seed = req.seed.unwrap_or_else(|| rand::random::<u64>() & (u32::MAX as u64));
    let (w, h) = (req.width as f32, req.height as f32);

    for n in 0..req.count.max(1) {
        let seed = base_seed.wrapping_add(n as u64);

        // Generate the scene via the model-agnostic t2i path so any model works
        // and the OpenPose ControlNet (if --pose) can pin the figures.
        let gen_dir = tempfile::Builder::new().prefix("plakat-mp-scene-").tempdir()?;
        let mut treq = t2i::Request::simple(
            base_prompt.clone(),
            req.model.clone(),
            req.width,
            req.height,
            req.steps,
            Some(seed),
            req.device.clone(),
            gen_dir.path().to_path_buf(),
        );
        treq.negative = req.negative.clone();
        treq.guidance = req.guidance;
        treq.scheduler = req.scheduler;
        if let Some(pp) = &pose_map_path {
            treq.controls = vec![ControlSpec {
                kind: crate::pipelines::controlnet::ControlKind::OpenPose,
                image: Some(pp.clone()),
                from: None,
                video: None,
                strength: 1.0,
                start: 0.0,
                end: 1.0,
            }];
        }
        t2i::run(treq).await.context("multiperson --swap: scene generation")?;
        let scene_path = req.out_dir.join(format!("plakat-multiperson-{seed}.png"));
        let gen_png = std::fs::read_dir(gen_dir.path())?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
            .context("scene generation produced no PNG")?;
        std::fs::copy(&gen_png, &scene_path)?;

        let faces = swapper.detect(&scene_path)?;
        crate::ui::progress::println(&format!(
            "  scene has {} detected face(s); {} persona(s) to place",
            faces.len(),
            req.people.len()
        ));

        let pairs = match_personas_to_faces(&faces, resolved, w, h);
        let mut scene_img = image::open(&scene_path)?.to_rgb8();
        let mut swapped_labels = Vec::new();
        for (ri, fi) in &pairs {
            let p = &req.people[resolved[*ri].idx];
            scene_img = swapper
                .swap_into(&scene_img, faces[*fi].landmarks, &latents[resolved[*ri].idx])
                .with_context(|| format!("swapping persona '{}'", p.label))?;
            swapped_labels.push(p.label.clone());
            crate::ui::progress::println(&format!(
                "  {} swapped {}",
                console::style("·").cyan(),
                p.label
            ));
        }
        // Report personas with no matching face.
        for r in resolved {
            let label = &req.people[r.idx].label;
            if !swapped_labels.contains(label) {
                crate::ui::progress::println(&format!(
                    "  {} no face in {}'s region — left unswapped",
                    console::style("!").yellow().bold(),
                    label
                ));
            }
        }

        scene_img.save(&scene_path)?;

        // Optional low-strength ADetailer detail pass on the swapped faces.
        if req.restore_faces {
            let mut cfg = crate::pipelines::adetailer::Config::defaults();
            cfg.model = req.model.clone();
            cfg.device = req.device.clone();
            cfg.strength = 0.18; // light — sharpen without drifting the swapped face
            cfg.confidence = 0.35; // catch small/painterly scene faces
            cfg.scheduler = req.scheduler;
            match crate::pipelines::adetailer::refine_files(&cfg, &[scene_path.clone()], None).await {
                Ok(n) => crate::ui::progress::println(&format!(
                    "  {} face-restore refined {n} face(s)",
                    console::style("·").cyan()
                )),
                Err(e) => crate::ui::progress::println(&format!(
                    "  {} face-restore skipped ({e})",
                    console::style("!").yellow().bold()
                )),
            }
        }

        write_sidecar(req, resolved, mp, &base_prompt, seed, &scene_path)?;
        crate::ui::progress::println(&format!(
            "  {} {}",
            console::style("✓").green().bold(),
            scene_path.display()
        ));
    }
    Ok(())
}

/// Greedily match each persona to the detected face nearest its expected face
/// point (placement-region centre, near the top). Returns `(resolved index, face
/// index)` pairs; personas beyond the available faces are simply not matched.
fn match_personas_to_faces(
    faces: &[Face],
    resolved: &[Resolved],
    w: f32,
    h: f32,
) -> Vec<(usize, usize)> {
    let mut used = vec![false; faces.len()];
    let mut out = Vec::new();
    for (ri, r) in resolved.iter().enumerate() {
        let ex = (r.bbox[0] + r.bbox[2]) * 0.5;
        let ey = r.bbox[1] + 0.18 * (r.bbox[3] - r.bbox[1]);
        let mut best: Option<usize> = None;
        let mut best_d = f32::MAX;
        for (fi, f) in faces.iter().enumerate() {
            if used[fi] {
                continue;
            }
            let fcx = (f.bbox[0] + f.bbox[2]) * 0.5 / w;
            let fcy = (f.bbox[1] + f.bbox[3]) * 0.5 / h;
            let d = (fcx - ex).powi(2) + (fcy - ey).powi(2);
            if d < best_d {
                best_d = d;
                best = Some(fi);
            }
        }
        if let Some(fi) = best {
            used[fi] = true;
            out.push((ri, fi));
        }
    }
    out
}

/// A documented, plakat-compatible SCRFD (converted `scrfd_10g_bnkps`). Used as
/// the default for the multiperson face-refine pass when neither
/// `PLAKAT_SCRFD_WEIGHTS` nor `PLAKAT_SCRFD_HF` is configured.
/// Identity face-refinement: re-inpaint just each persona's face crop with their
/// reference photos at high identity strength and a portrait framing. Face-focused
/// conditioning is where plus-face transfers identity well; the whole-body inpaint
/// can't. Face boxes come from SCRFD when a detector is supplied (tighter alignment
/// with the rendered face), else geometrically from the persona's placement region.
#[allow(clippy::too_many_arguments)]
fn refine_persona_faces<F>(
    pipe: &portrait::Pipeline,
    detector: Option<&SCRFDDetector>,
    mut latents: candle_core::Tensor,
    resolved: &[Resolved],
    req: &MultipersonRequest,
    mp: &MultipersonPrompt,
    seed: u64,
    lh: usize,
    lw: usize,
    dtype: candle_core::DType,
    mk_req: &F,
) -> Result<candle_core::Tensor>
where
    F: Fn(String, Vec<WeightedPhoto>, f32, Option<[f32; 4]>, Option<[[f32; 2]; 5]>) -> portrait::GenRequest,
{
    let (w, h) = (req.width as f32, req.height as f32);

    // Determine each persona's face box. SCRFD (if present) detects the actual
    // rendered faces; otherwise we derive a box from the exact placement region.
    let assignments: Vec<(usize, [f32; 4])> = match detector {
        Some(det) => {
            // Decode the current latents to a temp PNG so SCRFD can detect faces.
            let tmp = tempfile::Builder::new()
                .prefix("plakat-mp-faces-")
                .suffix(".png")
                .tempfile()
                .context("creating face-refine tempfile")?;
            pipe.save_image(&latents, tmp.path()).context("decoding render for face detection")?;
            let detected = det.detect(tmp.path()).context("SCRFD detect on render")?;
            let faces: Vec<&Face> = detected.iter().filter(|f| f.score >= 0.3).collect();
            if faces.is_empty() {
                crate::ui::progress::println(&format!(
                    "  {} SCRFD found no faces — falling back to geometric face boxes",
                    console::style("!").yellow().bold()
                ));
                resolved.iter().enumerate().map(|(ri, r)| (ri, geometric_face_box(&r.bbox))).collect()
            } else {
                assign_faces_to_personas(&faces, resolved, w, h)
            }
        }
        None => resolved
            .iter()
            .enumerate()
            .map(|(ri, r)| (ri, geometric_face_box(&r.bbox)))
            .collect(),
    };

    for (ri, face_bbox) in assignments {
        let r = &resolved[ri];
        let p = &req.people[r.idx];
        // SAFETY: clamp the face box to this persona's placement region. The
        // refine inpaint fully regenerates the masked area, so an over-large or
        // mis-detected box would otherwise repaint a frame-filling face. Bounding
        // it to the region the persona already occupies makes that impossible.
        let Some(face_bbox) = intersect_bbox(&face_bbox, &r.bbox) else {
            crate::ui::progress::println(&format!(
                "  {} face-refine skipped {} (no face inside its region)",
                console::style("!").yellow().bold(),
                p.label
            ));
            continue;
        };
        let mask = crate::pipelines::tiled::region_mask(face_bbox, lh, lw, &req.device, dtype)?;
        let prompt = mp.face_region_prompt(r.facing_phrase);
        let preq = mk_req(
            prompt,
            p.photos.clone(),
            req.refine_face_strength,
            p.face_bbox,
            p.face_landmarks,
        );
        // LOW-strength masked repaint (ADetailer-style): re-noise the face region
        // only partway and denoise from there, so the existing face's scale and
        // framing are preserved while the persona's identity is pushed in. Full
        // strength (inpaint_latents_one) would redraw it free-form and could blow
        // the face up — that was the earlier "nightmare". Seed decorrelated from
        // the body pass.
        latents = pipe
            .blend_latents_one(
                &latents,
                &mask,
                &preq,
                req.refine_denoise,
                seed.wrapping_add(0x9e37_79b9),
                &[],
                None,
            )
            .with_context(|| format!("multiperson: face-refine persona '{}'", p.label))?;
        crate::ui::progress::println(&format!(
            "  {} face-refined {}",
            console::style("·").cyan(),
            p.label
        ));
    }
    Ok(latents)
}

/// Greedily match each detected face to the persona whose region best explains it.
/// For every persona (in render order) we take its expected face point — the
/// horizontal centre of its body region, near the top — and claim the nearest
/// still-unused detected face. Returns `(index into `resolved`, padded normalised
/// face bbox)` pairs. Robust to extra/spurious detections (they go unclaimed).
fn assign_faces_to_personas(
    faces: &[&Face],
    resolved: &[Resolved],
    w: f32,
    h: f32,
) -> Vec<(usize, [f32; 4])> {
    let mut used = vec![false; faces.len()];
    let mut out = Vec::with_capacity(resolved.len());
    for (ri, r) in resolved.iter().enumerate() {
        let ex = (r.bbox[0] + r.bbox[2]) * 0.5;
        let ey = r.bbox[1] + 0.18 * (r.bbox[3] - r.bbox[1]);
        let mut best: Option<usize> = None;
        let mut best_d = f32::MAX;
        for (fi, f) in faces.iter().enumerate() {
            if used[fi] {
                continue;
            }
            let fcx = (f.bbox[0] + f.bbox[2]) * 0.5 / w;
            let fcy = (f.bbox[1] + f.bbox[3]) * 0.5 / h;
            let d = (fcx - ex).powi(2) + (fcy - ey).powi(2);
            if d < best_d {
                best_d = d;
                best = Some(fi);
            }
        }
        if let Some(fi) = best {
            used[fi] = true;
            out.push((ri, pad_norm_bbox(faces[fi], w, h, 0.15)));
        }
    }
    out
}

/// A face box derived geometrically from a persona's body region — the top-centre
/// slab where a standing/seated figure's head sits. Used when SCRFD isn't
/// configured: placement is exact, so the head is reliably at the top-centre of
/// the region. Width ≈ half the body width, height ≈ the top ~26%.
fn geometric_face_box(body: &[f32; 4]) -> [f32; 4] {
    let (bw, bh) = (body[2] - body[0], body[3] - body[1]);
    let cx = (body[0] + body[2]) * 0.5;
    let half_w = bw * 0.27; // ~54% of body width
    let y0 = body[1] + bh * 0.03;
    let y1 = body[1] + bh * 0.29;
    [
        (cx - half_w).clamp(0.0, 1.0),
        y0.clamp(0.0, 1.0),
        (cx + half_w).clamp(0.0, 1.0),
        y1.clamp(0.0, 1.0),
    ]
}

/// Intersection of two normalised `[x0,y0,x1,y1]` boxes, or `None` if they don't
/// overlap in a usable region (degenerate / vanishingly small).
fn intersect_bbox(a: &[f32; 4], b: &[f32; 4]) -> Option<[f32; 4]> {
    let x0 = a[0].max(b[0]);
    let y0 = a[1].max(b[1]);
    let x1 = a[2].min(b[2]);
    let y1 = a[3].min(b[3]);
    if x1 - x0 > 0.02 && y1 - y0 > 0.02 {
        Some([x0, y0, x1, y1])
    } else {
        None
    }
}

/// Pixel face bbox → normalised, padded by `pad` of its size each side, clamped.
fn pad_norm_bbox(f: &Face, w: f32, h: f32, pad: f32) -> [f32; 4] {
    let (x0, y0, x1, y1) = (f.bbox[0] / w, f.bbox[1] / h, f.bbox[2] / w, f.bbox[3] / h);
    let (dx, dy) = ((x1 - x0) * pad, (y1 - y0) * pad);
    [
        (x0 - dx).clamp(0.0, 1.0),
        (y0 - dy).clamp(0.0, 1.0),
        (x1 + dx).clamp(0.0, 1.0),
        (y1 + dy).clamp(0.0, 1.0),
    ]
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
        "face_refine": req.refine_faces,
        "refine_face_strength": req.refine_face_strength,
        "persons": persons,
    });
    std::fs::write(out.with_extension("json"), serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x0: f32, y0: f32, x1: f32, y1: f32) -> Face {
        Face { bbox: [x0, y0, x1, y1], landmarks: [[0.0; 2]; 5], score: 0.9 }
    }

    fn resolved_at(bbox: [f32; 4]) -> Resolved {
        Resolved { idx: 0, bbox, facing_phrase: None, facing: Facing::Front, scale: 1.0, order_y: 0.0, source: "at" }
    }

    #[test]
    fn pad_norm_bbox_normalises_and_pads() {
        // 100×100 px image; a 20px face at (40,30)-(60,50).
        let b = pad_norm_bbox(&face(40.0, 30.0, 60.0, 50.0), 100.0, 100.0, 0.5);
        // centre preserved; padded by 0.5×0.2 = 0.1 each side.
        assert!((b[0] - 0.30).abs() < 1e-5);
        assert!((b[2] - 0.70).abs() < 1e-5);
        assert!(b.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn assign_matches_faces_to_nearest_persona_by_region() {
        // Two personas: one on the left, one on the right.
        let resolved = vec![
            resolved_at([0.05, 0.30, 0.45, 0.95]), // left
            resolved_at([0.55, 0.30, 0.95, 0.95]), // right
        ];
        // Two detected faces near the top of each band (1000×1000 px).
        let left_face = face(200.0, 350.0, 300.0, 450.0); // cx 0.25
        let right_face = face(700.0, 350.0, 800.0, 450.0); // cx 0.75
        // Pass them out of order to prove matching is by position, not index.
        let faces = vec![&right_face, &left_face];
        let pairs = assign_faces_to_personas(&faces, &resolved, 1000.0, 1000.0);
        assert_eq!(pairs.len(), 2);
        // persona 0 (left) should get the left face → cx ≈ 0.25.
        let p0 = pairs.iter().find(|(ri, _)| *ri == 0).unwrap();
        let cx0 = (p0.1[0] + p0.1[2]) * 0.5;
        assert!((cx0 - 0.25).abs() < 0.06, "left persona matched cx {cx0}");
        // persona 1 (right) should get the right face → cx ≈ 0.75.
        let p1 = pairs.iter().find(|(ri, _)| *ri == 1).unwrap();
        let cx1 = (p1.1[0] + p1.1[2]) * 0.5;
        assert!((cx1 - 0.75).abs() < 0.06, "right persona matched cx {cx1}");
    }

    #[test]
    fn geometric_face_box_sits_top_centre_of_region() {
        // A tall body region on the left half.
        let fb = geometric_face_box(&[0.10, 0.20, 0.50, 0.95]);
        // horizontally centred on the body (cx 0.30).
        let cx = (fb[0] + fb[2]) * 0.5;
        assert!((cx - 0.30).abs() < 1e-5, "cx {cx}");
        // top of the box near the top of the region, well above its vertical mid.
        assert!(fb[1] >= 0.20 && fb[1] < 0.30);
        assert!(fb[3] < (0.20 + 0.95) * 0.5, "face box stays in upper region");
        assert!(fb.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn intersect_clamps_face_to_region_and_rejects_disjoint() {
        // A face box that spills well outside a small centre region is clamped
        // back inside it — the anti-frame-fill safety.
        let region = [0.38, 0.09, 0.62, 0.59];
        let huge = [0.05, 0.02, 0.95, 0.95];
        let clamped = intersect_bbox(&huge, &region).unwrap();
        assert_eq!(clamped, region);
        // Disjoint boxes → None (skip refine).
        assert!(intersect_bbox(&[0.0, 0.0, 0.1, 0.1], &[0.8, 0.8, 0.9, 0.9]).is_none());
    }

    #[test]
    fn assign_never_reuses_a_face() {
        let resolved = vec![
            resolved_at([0.05, 0.30, 0.45, 0.95]),
            resolved_at([0.10, 0.30, 0.50, 0.95]), // overlapping; both want same face
        ];
        let only = face(200.0, 350.0, 300.0, 450.0);
        let faces = vec![&only];
        let pairs = assign_faces_to_personas(&faces, &resolved, 1000.0, 1000.0);
        // Only one face exists → only one persona is matched.
        assert_eq!(pairs.len(), 1);
    }
}
