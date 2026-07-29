//! `plakat persona` — controllable synthetic-person composition (RFC PERSONA-1, the 5.0.0 flagship).
//!
//! P0 first slice: `new` (scaffold a spec) + `lint` (validate without weights). The resolver, geometry,
//! details, casting, scorecard and TUI subcommands land across later phases (ROADMAP_5.0.0). Fully
//! additive — nothing here touches existing behaviour.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::persona::lint::{self, Level};
use crate::persona::PersonaSpec;

#[derive(Args, Debug)]
pub struct PersonaArgs {
    #[command(subcommand)]
    pub cmd: PersonaCmd,
}

#[derive(Subcommand, Debug)]
pub enum PersonaCmd {
    /// Scaffold a new persona spec (a valid, partial `PersonaSpec` HJSON you then edit or `--tui`).
    New(NewArgs),
    /// Validate a persona spec — schema, scalar ranges, contradictions, the age gate. No weights, no
    /// network. Exits non-zero on any error so it can gate CI.
    Lint(LintArgs),
    /// Show what a spec resolves to for a model family: the salience-ranked prompt-routed attributes,
    /// the compiled positive/negative prompt, and (on CLIP) which attributes the token budget dropped.
    Show(ShowArgs),
    /// Measure a rendered image against a spec (the scorecard, RFC §12). P1: the landmark probe —
    /// SCRFD detects the face, PIPNet-98 aligns it, and geometric ratio metrics are reported.
    Verify(VerifyArgs),
    /// Rasterise the geometry maps a spec resolves to (the Layer-2 geometry engine, RFC §10) — mesh /
    /// wireframe / depth / pose-skeleton / region-masks / dentition / figure. Pure, no weights.
    Geometry(GeometryArgs),
    /// Composite a spec's localized details (marks / jewelry) onto a rendered image (RFC §8) —
    /// anchored through the realised landmarks, deterministic. `--harmonise` blends them into skin.
    Composite(CompositeArgs),
    /// Build a family calibration table (RFC §13): `--bootstrap` regenerates the provisional seed, or
    /// `--from <dir>` measures a rendered sweep (the offline compute job) → priors + curves + grades.
    Calibrate(CalibrateArgs),
    /// Cast a persona (RFC §11.1): render candidates, composite the persona's details, score them
    /// against the spec, keep the best, and validate identity coherence → a stored reference set.
    Cast(CastArgs),
    /// Render a cast persona into a scene (RFC §11.5, the universal Tier-B path): generate → swap the
    /// canonical face in → restore → composite the persona's details AFTER the swap → save.
    Render(RenderArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Output path for the new spec (`.hjson`).
    pub out: PathBuf,
    /// How much of the schema to scaffold: `quick` (identity + core structure), `standard`, `full`.
    #[arg(long, default_value = "quick")]
    pub depth: String,
    /// Slug name for the persona (also the People-library folder name).
    #[arg(long, default_value = "unnamed")]
    pub name: String,
    /// Apparent age in years (must be >= 18; see §23.1).
    #[arg(long, default_value_t = 30)]
    pub age: u32,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Model family to compile for (its encoder class shapes the prompt). Default `sdxl`.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to measure against the spec.
    #[arg(long)]
    pub image: PathBuf,
    /// Model family whose calibration table scores the geometric scalars (§13). Default `sdxl`.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct GeometryArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Directory to write the map PNG(s) into (created if missing).
    #[arg(long, default_value = "persona-geometry")]
    pub out: PathBuf,
    /// Which map(s): `all` or a comma list of
    /// mesh,wireframe,depth,skeleton,masks,dentition,figure.
    #[arg(long, default_value = "all")]
    pub map: String,
    /// Output edge length in pixels (face maps are square; the figure map is 3:4).
    #[arg(long, default_value_t = 512)]
    pub size: u32,
    /// Mesh drawing convention: `mediapipe` (per-feature colour) or `generic` (white).
    #[arg(long = "mesh-style", default_value = "mediapipe")]
    pub mesh_style: String,
    /// Seed for the asymmetry perturbation (§10.2).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Pre-distort the deformation through this family's calibration curves (§13.2), so a requested
    /// scalar lands at its value once realised. Omit to emit the raw requested geometry.
    #[arg(long)]
    pub calibrate: Option<String>,
}

#[derive(Args, Debug)]
pub struct CompositeArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to composite the details onto.
    #[arg(long)]
    pub image: PathBuf,
    /// Output path for the composited image.
    #[arg(long, default_value = "composited.png")]
    pub out: PathBuf,
    /// Also write the union affected-region mask here (for inspection / external harmonisation).
    #[arg(long)]
    pub mask: Option<PathBuf>,
    /// Seed for the procedural overlays (§8.3).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Run the harmonisation img2img pass (§8.4) over the affected region to blend the overlays into
    /// skin. Requires model weights; the deterministic composite runs regardless.
    #[arg(long)]
    pub harmonise: bool,
    /// Model for the harmonisation pass.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Harmonisation denoise strength — low, so the overlays are blended, not regenerated.
    #[arg(long = "harmonise-strength", default_value_t = 0.25)]
    pub harmonise_strength: f32,
}

#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// Model family the table is for (e.g. `sd15`, `sdxl`).
    pub family: String,
    /// Output path for the table (`.hjson`).
    #[arg(long)]
    pub out: PathBuf,
    /// Regenerate the provisional bootstrap (priors from the geometry mean-template, grades from the
    /// lexicon defaults). No renders required.
    #[arg(long)]
    pub bootstrap: bool,
    /// Measure a rendered sweep directory (files `<attr>__<requested>__<seed>.png`) → real priors +
    /// curves + grades (§13.1/§13.2). The render half is the offline compute job.
    #[arg(long)]
    pub from: Option<PathBuf>,
    /// Prompt / sampler / steps / size recorded in the measurement identity (§13.1).
    #[arg(long, default_value = "a portrait photograph of a person, neutral expression, front facing")]
    pub prompt: String,
    #[arg(long, default_value = "euler")]
    pub sampler: String,
    #[arg(long, default_value_t = 30)]
    pub steps: u32,
    #[arg(long, default_value_t = 1024)]
    pub size: u32,
}

#[derive(Args, Debug)]
pub struct CastArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// Persona directory to write the reference set into (default `persona-<name>`).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Casting family (its encoder shapes the prompt). Default `sdxl`.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Number of candidates to render.
    #[arg(long, default_value_t = 8)]
    pub count: u32,
    /// How many top-scoring candidates to keep in the reference set.
    #[arg(long = "keep-best", default_value_t = 4)]
    pub keep_best: u32,
    /// Render size (square).
    #[arg(long, default_value_t = 768)]
    pub size: u32,
    /// Denoise steps per candidate.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,
    /// Base seed; candidate `i` renders at `seed + i`.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Skip the detail compositing pass (§8.4) on the references.
    #[arg(long)]
    pub no_details: bool,
    /// Also score aesthetics (LAION) as a secondary sort key (loads an extra model).
    #[arg(long)]
    pub aesthetic: bool,
    /// Rejection sampling (§12.3): keep rendering (up to `--max-attempts`) until `--keep-best`
    /// candidates score at least this. `0` (default) disables it — render exactly `--count`.
    #[arg(long = "min-score", default_value_t = 0.0)]
    pub min_score: f32,
    /// Cap on total render attempts when rejection sampling. Defaults to `--count` when unset.
    #[arg(long = "max-attempts", default_value_t = 0)]
    pub max_attempts: u32,
    /// Tier label recorded in the set (§11.4): `B` (universal swap) by default.
    #[arg(long, default_value = "B")]
    pub tier: String,
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    /// Persona directory produced by `persona cast` (holds `reference_set.json`).
    pub persona: PathBuf,
    /// Scene / setting prompt (the persona's appearance is merged in from its spec).
    #[arg(long)]
    pub scene: String,
    /// Model family to render the scene on (any — identity comes from the swap).
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output path.
    #[arg(long, default_value = "render.png")]
    pub out: PathBuf,
    /// Override the persona spec (default: `<persona>/spec.hjson`).
    #[arg(long)]
    pub spec: Option<PathBuf>,
    #[arg(long, default_value_t = 768)]
    pub size: u32,
    #[arg(long, default_value_t = 30)]
    pub steps: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Skip the face-restoration pass after the swap.
    #[arg(long)]
    pub no_restore: bool,
    /// Skip compositing the persona's details after the swap.
    #[arg(long)]
    pub no_details: bool,
    /// Identity tier (§11.4): `auto` (A where an IP-Adapter exists, else B), `A` (IP-Adapter-Plus-Face
    /// from the reference set), or `B` (native render → face swap → restore).
    #[arg(long, default_value = "auto")]
    pub tier: String,
}

pub async fn run(args: PersonaArgs) -> Result<()> {
    match args.cmd {
        PersonaCmd::New(a) => run_new(a),
        PersonaCmd::Lint(a) => run_lint(a),
        PersonaCmd::Show(a) => run_show(a),
        PersonaCmd::Verify(a) => run_verify(a).await,
        PersonaCmd::Geometry(a) => run_geometry(a),
        PersonaCmd::Composite(a) => run_composite(a).await,
        PersonaCmd::Calibrate(a) => run_calibrate(a).await,
        PersonaCmd::Cast(a) => run_cast(a).await,
        PersonaCmd::Render(a) => run_render(a).await,
    }
}

async fn run_render(a: RenderArgs) -> Result<()> {
    use crate::persona::casting::{self, ReferenceSet};
    use crate::persona::{compile, detail, lexicon::Lexicon, scorecard};

    let set = ReferenceSet::load(&a.persona)
        .with_context(|| format!("loading reference set from {} (run `persona cast` first)", a.persona.display()))?;
    let Some(canonical) = set.canonical().cloned() else {
        anyhow::bail!("reference set is empty");
    };
    let canonical_img = a.persona.join(&canonical.image);

    // The spec drives the appearance prompt + the post-swap detail composite.
    let spec_path = a.spec.clone().unwrap_or_else(|| a.persona.join("spec.hjson"));
    let spec = PersonaSpec::load(&spec_path).ok();
    let appearance = spec.as_ref().map(|s| compile::compile_for_model(s, &Lexicon::skeleton(), &a.model)).unwrap_or_else(|| compile::compile_for_model(&PersonaSpec::default(), &Lexicon::skeleton(), &a.model));
    let prompt = if appearance.positive.is_empty() {
        a.scene.clone()
    } else {
        format!("{}, {}", a.scene, appearance.positive)
    };

    println!("{}  {} into a scene  (model {})", style("persona render").bold(), set.persona, a.model);
    if !set.coherence.passes {
        println!("  {} reference set identity coherence is below threshold (min cos {:.3}) — identity anchoring is weaker than ideal; consider re-casting with tighter conditioning", style("note:").yellow(), set.coherence.min_cosine);
    }

    let device = candle_core::Device::Cpu;
    let tier = resolve_render_tier(&a.tier, &a.model);

    if tier == "A" {
        // Tier A (§11.4): IP-Adapter-Plus-Face from the reference set — identity from the adapter,
        // no swap. The face-reference generalises across the scene the sampler produces.
        let kind = identity_kind_for(&a.model).expect("tier A implies an adapter");
        let n = set.references.len().min(4);
        println!("  {} Tier A: generating with the face adapter ({n} reference photo(s))…", style("→").cyan());
        let mut portrait = crate::api::Portrait::new(&a.model)
            .prompt(&prompt)
            .negative(&appearance.negative)
            .identity(kind)
            .size(a.size, a.size)
            .steps(a.steps)
            .seed(a.seed);
        for r in set.references.iter().take(4) {
            portrait = portrait.photo(a.persona.join(&r.image), r.centroid_cosine.max(0.1));
        }
        let imgs = portrait.run().await.context("Tier A portrait render")?;
        let Some(img) = imgs.into_iter().next() else {
            anyhow::bail!("portrait render produced no image");
        };
        img.save(&a.out)?;
        println!("  {} rendered via the face adapter", style("✓").green());
    } else {
        // Tier B (§11.5, universal): native render → face swap → restore.
        println!("  {} Tier B: generating the scene…", style("→").cyan());
        let imgs = crate::api::Generate::new(&a.model)
            .prompt(&prompt)
            .negative(&appearance.negative)
            .seed(a.seed)
            .size(a.size, a.size)
            .steps(a.steps)
            .run()
            .await
            .context("scene render")?;
        let Some(scene_img) = imgs.into_iter().next() else {
            anyhow::bail!("scene render produced no image");
        };
        scene_img.save(&a.out)?;

        let swapper = crate::pipelines::faceswap::FaceSwapper::load_resolved(&device, candle_core::DType::F32).await?;
        let faces = swapper.detect(&a.out).context("detecting the scene face")?;
        let Some(target) = faces.into_iter().next() else {
            println!("  {} no face in the scene render — leaving it un-swapped ({})", style("·").yellow(), a.out.display());
            return Ok(());
        };
        // Region-escalation ladder (§14.1): measure the face's frame area; a small face swaps + restores
        // poorly, so escalate the restore to native resolution at higher strength.
        let face_area = casting::area_fraction(target.bbox, a.size, a.size);
        let face_dec = casting::decide(casting::EscalationRegion::Face, face_area, casting::EscalationRegion::Face.default_threshold());
        let latent = swapper.source_latent(&canonical_img).context("embedding the canonical reference face")?;
        let scene_rgb = image::open(&a.out)?.to_rgb8();
        let swapped = swapper.swap_into(&scene_rgb, target.landmarks, &latent).context("face swap")?;
        swapped.save(&a.out)?;
        println!("  {} swapped in {} (canonical, centroid cos {:.3})", style("✓").green(), canonical.image.display(), canonical.centroid_cosine);
        if face_dec.escalate {
            println!("  {} face is {:.1}% of the frame (< {:.0}%) — escalating the restore to native resolution (§14.1)", style("ladder:").cyan(), face_area * 100.0, face_dec.threshold * 100.0);
        }

        // Restore the swapped region — gentle by default, native-res + stronger when the face is small.
        if !a.no_restore {
            println!("  {} restoring the swapped face…", style("→").cyan());
            let sdxl = a.model.starts_with("sdxl");
            let native = if sdxl { 1024 } else { 512 };
            let restore = crate::cli::restore_faces::RestoreFacesArgs {
                inputs: vec![a.out.clone()],
                model: if sdxl { "sdxl".into() } else { "sd15".into() },
                strength: if face_dec.escalate { 0.45 } else { 0.35 },
                padding: 0.25,
                feather: 0.25,
                confidence: 0.5,
                working_size: if face_dec.escalate { native } else { 512 },
            };
            if let Err(e) = crate::cli::restore_faces::run(restore, device.clone()).await {
                println!("  {} restore skipped: {e}", style("·").yellow());
            }
        }
    }

    // 4. Detail compositing runs AFTER the swap (hard ordering §11.5 — the swap would erase marks
    //    composited before it) against the REALISED landmarks.
    if !a.no_details {
        if let Some(spec) = &spec {
            let has_details = spec.marks.as_ref().is_some_and(|m| !m.is_empty())
                || spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).is_some_and(|i| !i.is_empty())
                || spec.piercings.as_ref().is_some_and(|p| !p.is_empty());
            if has_details {
                let (det, pip) = scorecard::load_probes(&device).await?;
                if let Some(m) = scorecard::measure_landmarks(&a.out, &det, &pip)? {
                    let base = image::open(&a.out)?.to_rgb8();
                    let r = detail::composite_details(&base, spec, &m, a.seed);
                    r.image.save(&a.out)?;
                    println!("  {} composited {} detail(s) after the swap", style("✓").green(), r.placed);

                    // Mouth escalation advisory (§14.1) — the dentition inpaint itself is P7.
                    if crate::persona::geometry::open_mouth(spec) {
                        let lip: Vec<(f32, f32)> = (76..88).map(|i| m.landmarks[i]).collect();
                        let (cw, ch) = (m.crop.width() as f32, m.crop.height() as f32);
                        let w = (lip.iter().map(|p| p.0).fold(f32::MIN, f32::max) - lip.iter().map(|p| p.0).fold(f32::MAX, f32::min)) * cw;
                        let h = (lip.iter().map(|p| p.1).fold(f32::MIN, f32::max) - lip.iter().map(|p| p.1).fold(f32::MAX, f32::min)) * ch;
                        let mouth_area = (w * h / (a.size as f32 * a.size as f32)).clamp(0.0, 1.0);
                        if casting::decide(casting::EscalationRegion::Mouth, mouth_area, casting::EscalationRegion::Mouth.default_threshold()).escalate {
                            println!("  {} mouth is {:.2}% of the frame — dentition would need a mouth-region inpaint (§8.7, P7)", style("ladder:").cyan(), mouth_area * 100.0);
                        }
                    }
                    // Hand jewelry advisory (§8.5) — hand-region refine needs hand landmarks (unreliable).
                    let has_hand_jewelry = spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).is_some_and(|it| {
                        it.iter().any(|x| matches!(x.site.as_deref(), Some("left-wrist" | "right-wrist" | "left-hand" | "right-hand" | "finger")))
                    });
                    if has_hand_jewelry {
                        println!("  {} hand jewelry present — hand-region escalation is best-effort and not wired (§8.5)", style("·").yellow());
                    }
                } else {
                    println!("  {} could not re-detect the face for detail compositing", style("·").yellow());
                }
            }
        }
    }

    println!("{} {}", style("done:").bold(), a.out.display());
    Ok(())
}

fn fnv_hash(s: &str) -> String {
    let mut acc: u64 = 1469598103934665603;
    for b in s.bytes() {
        acc = (acc ^ b as u64).wrapping_mul(1099511628211);
    }
    format!("{acc:016x}")
}

/// The IP-Adapter-Plus-Face identity variant for a family, or `None` where no adapter exists (§11.4).
fn identity_kind_for(model: &str) -> Option<crate::pipelines::ip_adapter::IdentityKind> {
    use crate::pipelines::ip_adapter::IdentityKind;
    if model.starts_with("sdxl") || model == "pony" {
        Some(IdentityKind::PlusFaceSdxl)
    } else if model == "sd15" || model == "sd21" {
        Some(IdentityKind::PlusFace)
    } else {
        None
    }
}

/// Resolve the render tier (§11.4): `auto` → `A` where an adapter exists, else `B`; an explicit `A`
/// falls back to `B` (with no adapter) rather than failing.
fn resolve_render_tier(requested: &str, model: &str) -> String {
    let has_adapter = identity_kind_for(model).is_some();
    match requested {
        "A" if has_adapter => "A".into(),
        "A" => {
            println!("  {} no face adapter for `{model}` — falling back to Tier B (swap)", style("·").yellow());
            "B".into()
        }
        "B" => "B".into(),
        _ => {
            if has_adapter {
                "A".into()
            } else {
                "B".into()
            }
        }
    }
}

async fn run_cast(a: CastArgs) -> Result<()> {
    use crate::persona::casting::{Reference, ReferenceSet, COHERENCE_THRESHOLD};
    use crate::persona::{compile, detail, geometry, lexicon::Lexicon, scorecard};

    let spec = PersonaSpec::load(&a.spec)?;
    // Lint is advisory here (casting is exploratory) — report errors, don't abort.
    let findings = crate::persona::lint::lint(&spec);
    let errs = findings.iter().filter(|f| f.level == Level::Error).count();
    if errs > 0 {
        println!("{} {errs} lint error(s) — casting anyway; run `persona lint` for detail", style("warning:").yellow());
    }

    let lex = Lexicon::skeleton();
    let compiled = compile::compile_for_model(&spec, &lex, &a.model);
    let name = spec.identity.as_ref().and_then(|i| i.name.as_deref()).unwrap_or("persona").to_string();
    let persona_dir = a.out.clone().unwrap_or_else(|| PathBuf::from(format!("persona-{name}")));
    let cand_dir = persona_dir.join("candidates");
    let refs_dir = persona_dir.join("references");
    std::fs::create_dir_all(&cand_dir)?;
    std::fs::create_dir_all(&refs_dir)?;
    // Stash the spec so `persona render` can composite details + merge the appearance prompt.
    std::fs::copy(&a.spec, persona_dir.join("spec.hjson")).ok();

    let conditioning_hash = fnv_hash(&format!("{}|{}|{:?}", compiled.positive, compiled.negative, geometry::geometry_values(&spec)));
    let detail_plan_hash = fnv_hash(&format!("{:?}", spec.marks));

    // Probes (CPU) + ArcFace + calibration; aesthetic is opt-in.
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let arcface = crate::pipelines::identity_quality::IdentityScorer::load_resolved(&device).await?;
    let table = crate::persona::calibration::CalibrationTable::bundled(&a.model);
    let aesthetic = if a.aesthetic {
        Some(crate::pipelines::aesthetic::AestheticScorer::load(&device).await?)
    } else {
        None
    };

    println!("{}  {} → {}  (model {}, {} candidates)", style("persona cast").bold(), name, persona_dir.display(), a.model, a.count);
    if !compiled.positive.is_empty() {
        println!("  {} {}", style("prompt:").dim(), compiled.positive);
    }

    struct Cand {
        image: PathBuf,
        seed: u64,
        score: f32,
        aesthetic: Option<f32>,
        embedding: Vec<f32>,
    }
    let mut cands: Vec<Cand> = Vec::new();
    let mut no_face = 0u32;
    let mut rejected = 0u32;

    // Rejection sampling (§12.3): without `--min-score` this renders exactly `--count`; with it, keep
    // going (up to `--max-attempts`) until `--keep-best` candidates clear the bar.
    let target = a.keep_best as usize;
    let cap = if a.max_attempts > 0 { a.max_attempts.max(a.count) } else { a.count };
    let mut i = 0u32;
    while i < cap {
        if a.min_score > 0.0 {
            if cands.iter().filter(|c| c.score >= a.min_score).count() >= target {
                break;
            }
        } else if cands.len() + no_face as usize >= a.count as usize {
            break;
        }
        let attempt = i;
        i += 1;
        let seed = a.seed + attempt as u64;
        let raw = cand_dir.join(format!("cand_{attempt}_raw.png"));
        // Render one candidate (Tier-B: prompt only; geometry-CN casting is a follow-on).
        let imgs = crate::api::Generate::new(&a.model)
            .prompt(&compiled.positive)
            .negative(&compiled.negative)
            .seed(seed)
            .size(a.size, a.size)
            .steps(a.steps)
            .run()
            .await
            .with_context(|| format!("rendering candidate {attempt}"))?;
        let Some(img) = imgs.into_iter().next() else { continue };
        img.save(&raw)?;

        let Some(m) = scorecard::measure_landmarks(&raw, &detector, &pipnet)? else {
            println!("  {} candidate {attempt}: no face detected — skipped", style("·").yellow());
            no_face += 1;
            continue;
        };

        // Composite the persona's details onto the reference (§11.1 step 3) unless disabled.
        let ref_img = cand_dir.join(format!("cand_{attempt}.png"));
        if a.no_details {
            std::fs::copy(&raw, &ref_img)?;
        } else {
            let base = image::open(&raw)?.to_rgb8();
            let r = detail::composite_details(&base, &spec, &m, seed);
            r.image.save(&ref_img)?;
        }

        let sc = scorecard::score_render(&spec, &m, table.as_ref());
        let score = sc.aggregate().unwrap_or(0.0);
        if a.min_score > 0.0 && score < a.min_score {
            rejected += 1;
        }
        let Some(embedding) = arcface.embed(&ref_img)? else {
            println!("  {} candidate {attempt}: face not embeddable — skipped", style("·").yellow());
            no_face += 1;
            continue;
        };
        let aes = aesthetic.as_ref().map(|s| s.score_path(&ref_img)).transpose()?;
        println!(
            "  {} candidate {attempt}: score {:.2}{}  (seed {seed})",
            style("✓").green(),
            score,
            aes.map(|v| format!(", aesthetic {v:.1}")).unwrap_or_default()
        );
        cands.push(Cand { image: ref_img, seed, score, aesthetic: aes, embedding });
    }

    if cands.is_empty() {
        anyhow::bail!("no usable candidates ({no_face} had no embeddable face) — try a different model/prompt or more --count");
    }

    // Rank: spec conformance primary, aesthetics a distant secondary (§11.1 step 5).
    cands.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| y.aesthetic.unwrap_or(0.0).partial_cmp(&x.aesthetic.unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal))
    });
    cands.truncate(a.keep_best as usize);

    // Move the kept candidates into references/ and build the set.
    let mut references = Vec::new();
    for (rank, c) in cands.iter().enumerate() {
        let dst = refs_dir.join(format!("ref_{rank}.png"));
        std::fs::copy(&c.image, &dst)?;
        references.push(Reference {
            id: rank,
            image: PathBuf::from("references").join(format!("ref_{rank}.png")),
            seed: c.seed,
            model: a.model.clone(),
            view: "frontal".into(),
            expression: "neutral".into(),
            conditioning_hash: conditioning_hash.clone(),
            detail_plan_hash: detail_plan_hash.clone(),
            embedding: c.embedding.clone(),
            score: Some(c.score),
            aesthetic: c.aesthetic,
            centroid_cosine: 0.0,
        });
    }
    let set = ReferenceSet::assemble(&name, &a.model, &a.tier, references, COHERENCE_THRESHOLD);
    set.save(&persona_dir)?;

    // Report.
    println!("\n{}", style("── cast ──").bold());
    let rej = if a.min_score > 0.0 { format!(", {rejected} below min-score {:.2}", a.min_score) } else { String::new() };
    println!("  kept {} / {i} rendered ({no_face} unusable{rej}) → {}", set.references.len(), ReferenceSet::manifest_path(&persona_dir).display());
    let coh = &set.coherence;
    let glyph = if coh.passes { style("✓").green() } else { style("✗").red() };
    println!("  {glyph} identity coherence: mean cos {:.3}, min cos {:.3} (threshold {:.2})", coh.mean_cosine, coh.min_cosine, coh.threshold);
    if !coh.passes {
        println!("  {} the cast produced several different people — re-run with tighter conditioning (more --count, a stronger family, or geometry conditioning)", style("warning:").yellow());
    }
    if let Some(c) = set.canonical() {
        println!("  canonical face: {} (centroid cos {:.3})", c.image.display(), c.centroid_cosine);
    }
    Ok(())
}

/// (attr, metric-extractor, invert-for-scalar) for the aligner-measured geometric scalars.
type CalibMetric = (&'static str, fn(&crate::persona::scorecard::FaceMetrics) -> f32, bool);
/// The three aligner-measured geometric scalars.
const CALIB_METRICS: &[CalibMetric] = &[
    ("eyes.spacing", |m| m.interpupillary_over_facewidth, false),
    ("mouth.width", |m| m.mouth_over_facewidth, false),
    ("face.width", |m| m.face_aspect, true),
];

fn median_p5_p95(xs: &mut [f32]) -> crate::persona::calibration::Prior {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f32| xs[((q * (xs.len() - 1) as f32).round() as usize).min(xs.len() - 1)];
    crate::persona::calibration::Prior { median: at(0.5), p5: at(0.05), p95: at(0.95) }
}

async fn run_calibrate(a: CalibrateArgs) -> Result<()> {
    use crate::persona::calibration::{self as cal, MeasurementIdentity, Prior};
    use crate::persona::lexicon::Lexicon;
    use std::collections::BTreeMap;

    let lex = Lexicon::skeleton();
    let harmonise: BTreeMap<String, f32> = [("mole", 0.25), ("scar", 0.30), ("birthmark", 0.28), ("freckles", 0.35)]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    let (priors, samples, provisional, population, spont, hit) = if a.bootstrap {
        // Priors = the geometry engine's own mean-template metrics (0.5 configuration).
        use crate::persona::geometry::{mean_template, topology::*};
        let t = mean_template(false);
        let d = |i: usize, j: usize| ((t[i].0 - t[j].0).powi(2) + (t[i].1 - t[j].1).powi(2)).sqrt();
        let fw = CONTOUR.clone().map(|i| t[i].0).fold(f32::NEG_INFINITY, f32::max) - CONTOUR.clone().map(|i| t[i].0).fold(f32::INFINITY, f32::min);
        let fh = CONTOUR.clone().map(|i| t[i].1).fold(f32::NEG_INFINITY, f32::max) - CONTOUR.clone().map(|i| t[i].1).fold(f32::INFINITY, f32::min);
        let ipd = d(PUPIL_RIGHT, PUPIL_LEFT) / fw;
        let mw = d(MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT) / fw;
        let asp = fh / fw;
        let mut priors = BTreeMap::new();
        priors.insert("eyes.spacing".to_string(), Prior { median: ipd, p5: ipd * 0.85, p95: ipd * 1.15 });
        priors.insert("mouth.width".to_string(), Prior { median: mw, p5: mw * 0.85, p95: mw * 1.15 });
        priors.insert("face.width".to_string(), Prior { median: asp, p5: asp * 0.85, p95: asp * 1.15 });
        // Curves seeded from each attribute's lexicon control grade.
        let mut samples = BTreeMap::new();
        for (attr, _, _) in CALIB_METRICS {
            let grade = lex.get(attr).and_then(|e| e.control.clone()).unwrap_or_else(|| "moderate".into());
            let half = match grade.as_str() {
                "strong" => 0.38,
                "weak" => 0.10,
                "experimental" => 0.02,
                _ => 0.20, // moderate
            };
            let s = vec![(0.0, 0.5 - half), (0.25, 0.5 - half * 0.5), (0.5, 0.5), (0.75, 0.5 + half * 0.5), (1.0, 0.5 + half)];
            samples.insert(attr.to_string(), (s, 0.03));
        }
        (priors, samples, true, 0u32, 0.12, 0.65)
    } else if let Some(dir) = &a.from {
        // Measure a rendered sweep: files `<attr>__<requested>__<seed>.png`.
        let device = candle_core::Device::Cpu;
        let (det, pip) = crate::persona::scorecard::load_probes(&device).await?;
        // attr → requested → Vec<realised metric>
        let mut by: BTreeMap<String, BTreeMap<u32, Vec<f32>>> = BTreeMap::new();
        let mut count = 0u32;
        for entry in std::fs::read_dir(dir).with_context(|| format!("reading sweep dir {}", dir.display()))? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let parts: Vec<&str> = stem.split("__").collect();
            if parts.len() < 2 {
                continue;
            }
            let (attr, req) = (parts[0].to_string(), parts[1].parse::<f32>().ok());
            let Some(req) = req else { continue };
            let Some(&(_, extract, _)) = CALIB_METRICS.iter().find(|(p, _, _)| *p == attr) else { continue };
            if let Some(m) = crate::persona::scorecard::measure_landmarks(&path, &det, &pip)? {
                by.entry(attr).or_default().entry((req * 1000.0).round() as u32).or_default().push(extract(&m));
                count += 1;
            }
        }
        if count == 0 {
            anyhow::bail!("no measurable `<attr>__<requested>__<seed>.png` renders found in {}", dir.display());
        }
        // Priors: from the requested≈0.5 population per attr (fallback: all samples).
        let mut priors = BTreeMap::new();
        let mut samples = BTreeMap::new();
        for (attr, steps) in &by {
            let invert = CALIB_METRICS.iter().find(|(p, _, _)| p == attr).map(|(_, _, i)| *i).unwrap_or(false);
            let mut neutral: Vec<f32> = steps.get(&500).cloned().unwrap_or_else(|| steps.values().flatten().copied().collect());
            let prior = median_p5_p95(&mut neutral);
            priors.insert(attr.clone(), prior);
            let mut pts: Vec<(f32, f32)> = Vec::new();
            let mut vars: Vec<f32> = Vec::new();
            for (req_k, vals) in steps {
                let mut v = vals.clone();
                let med = median_p5_p95(&mut v).median;
                let mut realised = prior.normalise(med);
                if invert {
                    realised = 1.0 - realised;
                }
                pts.push((*req_k as f32 / 1000.0, realised));
                // spread as a rough variance proxy
                let spread = v.iter().map(|x| (prior.normalise(*x) - realised).abs()).fold(0.0, f32::max);
                vars.push(spread);
            }
            let var = if vars.is_empty() { 0.0 } else { vars.iter().sum::<f32>() / vars.len() as f32 };
            samples.insert(attr.clone(), (pts, var));
        }
        println!("  measured {count} renders across {} attribute(s)", by.len());
        (priors, samples, false, count, 0.0, 0.0)
    } else {
        anyhow::bail!("specify --bootstrap (regenerate the seed) or --from <dir> (measure a sweep)");
    };

    let identity = MeasurementIdentity {
        population,
        prompt: a.prompt.clone(),
        sampler: a.sampler.clone(),
        steps: a.steps,
        size: a.size,
        aligner: cal::table::CURRENT_ALIGNER.into(),
        topology: cal::table::CURRENT_TOPOLOGY.into(),
        lexicon_version: cal::table::CURRENT_LEXICON.into(),
        provisional,
    };
    let table = cal::assemble(a.family.clone(), identity, priors, samples, harmonise, spont, hit);
    std::fs::write(&a.out, table.to_hjson()).with_context(|| format!("writing {}", a.out.display()))?;

    println!("{}  {} → {}", style("persona calibrate").bold(), a.family, a.out.display());
    for (attr, curve) in &table.curves {
        println!("  {} {attr}: grade {} (slope {:.2})", style("·").dim(), curve.grade.as_str(), curve.slope);
    }
    if provisional {
        println!("  {} provisional bootstrap — replace with a measured sweep via `--from <dir>`", style("note:").dim());
    }
    Ok(())
}

async fn run_composite(a: CompositeArgs) -> Result<()> {
    use crate::persona::{detail, scorecard};
    let spec = PersonaSpec::load(&a.spec)?;
    let n_marks = spec.marks.as_ref().map(|m| m.len()).unwrap_or(0);
    let n_piercings = spec.piercings.as_ref().map(|p| p.len()).unwrap_or(0);
    let n_jewelry = spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).map(|i| i.len()).unwrap_or(0);
    let n_details = n_marks + n_piercings + n_jewelry;
    if n_details == 0 {
        println!("{}  spec has no marks / piercings / jewelry — nothing to composite", style("·").dim());
        // still a no-op copy so downstream steps have an output.
        image::open(&a.image)?.to_rgb8().save(&a.out)?;
        println!("  {} {}", style("✓").green(), a.out.display());
        return Ok(());
    }

    // Detect + align the rendered face (the realised landmarks the anchors resolve through, §8.4).
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let Some(m) = scorecard::measure_landmarks(&a.image, &detector, &pipnet)? else {
        println!("{}  no face detected in {} — cannot anchor details", style("✗").red(), a.image.display());
        return Ok(());
    };

    let base = image::open(&a.image)?.to_rgb8();
    let r = detail::composite_details(&base, &spec, &m, a.seed);
    println!(
        "{}  {}  (face score {:.2})",
        style("persona composite").bold(),
        a.image.display(),
        m.detection_score
    );
    println!(
        "  placed {} / {n_details} detail(s); light ({:+.2}, {:+.2})",
        r.placed,
        r.light.dx,
        r.light.dy
    );
    for c in &r.culled {
        println!("  {} culled {}: {}", style("·").yellow(), c.kind, c.reason);
    }

    r.image.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
    println!("  {} {}", style("✓").green(), a.out.display());
    if let Some(mp) = &a.mask {
        r.mask.save(mp).with_context(|| format!("writing {}", mp.display()))?;
        println!("  {} {} (affected-region mask)", style("✓").green(), mp.display());
    }

    // Optional harmonisation: low-strength masked img2img over the affected region (§8.4).
    if a.harmonise {
        use crate::api::Img2img;
        let tmp = std::env::temp_dir();
        let comp_path = tmp.join(format!("persona_harm_{}.png", a.seed));
        let mask_path = tmp.join(format!("persona_harm_mask_{}.png", a.seed));
        r.image.save(&comp_path)?;
        r.mask.save(&mask_path)?;
        println!("  {} harmonising over the affected region (model {}, strength {:.2})…", style("→").cyan(), a.model, a.harmonise_strength);
        let prompt = "a portrait photograph, natural skin, detailed skin texture";
        let out = Img2img::new(&a.model, &comp_path)
            .prompt(prompt)
            .mask(&mask_path)
            .mask_feather(6)
            .strength(a.harmonise_strength)
            .seed(a.seed)
            .run()
            .await?;
        if let Some(img) = out.into_iter().next() {
            img.save(&a.out)?;
            println!("  {} {} (harmonised)", style("✓").green(), a.out.display());
        }
        let _ = std::fs::remove_file(&comp_path);
        let _ = std::fs::remove_file(&mask_path);
    }
    Ok(())
}

fn run_geometry(a: GeometryArgs) -> Result<()> {
    use crate::persona::geometry as geo;
    let spec = PersonaSpec::load(&a.spec)?;
    let sz = a.size.clamp(64, 4096);
    let mesh_style = match a.mesh_style.as_str() {
        "generic" => geo::MeshStyle::Generic,
        _ => geo::MeshStyle::MediaPipe,
    };

    // resolve the face geometry from the spec (optionally pre-distorted through a family's curves).
    let mut values = geo::geometry_values(&spec);
    if let Some(family) = &a.calibrate {
        match crate::persona::calibration::CalibrationTable::bundled(family) {
            Some(t) => {
                let corrected = crate::persona::calibration::predistort_geometry(&mut values, &t);
                if corrected.is_empty() {
                    println!("  {} calibrate({family}): no geometric scalars to correct", style("·").dim());
                } else {
                    println!("  {} calibrate({family}): {}", style("pre-distort").cyan(), corrected.join(", "));
                }
            }
            None => println!("  {} no calibration table for `{family}` — emitting raw geometry", style("·").yellow()),
        }
    }
    let open = geo::open_mouth(&spec);
    let d = geo::resolve(&values, open, a.seed);

    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let want: Vec<&str> = if a.map == "all" {
        vec!["mesh", "wireframe", "depth", "skeleton", "masks", "dentition", "figure"]
    } else {
        a.map.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect()
    };

    let name = spec.identity.as_ref().and_then(|i| i.name.as_deref()).unwrap_or("persona");
    println!("{}  {}  (seed {}, {}²)", style("persona geometry").bold(), name, a.seed, sz);
    if !d.warnings.is_empty() {
        for w in &d.warnings {
            println!("  {} {}: {}", style("validity").yellow(), w.kind, w.detail);
        }
    }

    let mut wrote = 0;
    let save = |path: PathBuf, r: image::DynamicImage| -> Result<()> {
        r.save(&path).with_context(|| format!("writing {}", path.display()))?;
        println!("  {} {}", style("✓").green(), path.display());
        Ok(())
    };
    for m in &want {
        let p = |suffix: &str| a.out.join(format!("{name}_{suffix}.png"));
        match *m {
            "mesh" => {
                save(p("mesh"), geo::mesh_map(&d.landmarks, sz, mesh_style).into())?;
                wrote += 1;
            }
            "wireframe" | "wire" => {
                save(p("wireframe"), geo::wireframe(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "depth" => {
                save(p("depth"), geo::depth_proxy(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "skeleton" | "pose" => {
                save(p("skeleton"), geo::face_skeleton(&d.landmarks, sz).into())?;
                wrote += 1;
            }
            "masks" | "mask" => {
                // composite the region masks into one colour-coded image.
                let mut img = image::RgbImage::new(sz, sz);
                for (r, c) in [
                    (geo::Region::Face, [40u8, 40, 60]),
                    (geo::Region::BrowRight, [200, 160, 60]),
                    (geo::Region::BrowLeft, [200, 160, 60]),
                    (geo::Region::EyeRight, [80, 200, 255]),
                    (geo::Region::EyeLeft, [80, 200, 255]),
                    (geo::Region::Nose, [120, 255, 120]),
                    (geo::Region::Mouth, [255, 90, 90]),
                ] {
                    let mask = geo::region_mask(&d.landmarks, sz, r);
                    for (x, y, px) in mask.enumerate_pixels() {
                        if px.0[0] > 0 {
                            img.put_pixel(x, y, image::Rgb(c));
                        }
                    }
                }
                save(p("masks"), img.into())?;
                wrote += 1;
            }
            "dentition" | "teeth" => {
                save(p("dentition"), geo::dentition_hint(&d.landmarks, sz, 6).into())?;
                wrote += 1;
            }
            "figure" => match geo::figure_params(&spec) {
                Some(fp) => {
                    let (w, h) = (sz, sz * 4 / 3);
                    let fig = geo::resolve_figure(&fp, a.seed);
                    save(p("figure_skeleton"), geo::figure_skeleton_map(&fig, w, h).into())?;
                    save(p("figure_silhouette"), geo::silhouette_mask(&fig, w, h).into())?;
                    wrote += 2;
                }
                None => {
                    if a.map != "all" {
                        println!("  {} figure: spec has no `figure:` block — nothing to render", style("·").dim());
                    }
                }
            },
            other => println!("  {} unknown map `{other}` (skipped)", style("?").yellow()),
        }
    }
    println!("{} {wrote} map(s) → {}", style("done:").bold(), a.out.display());
    Ok(())
}

async fn run_verify(a: VerifyArgs) -> Result<()> {
    use crate::persona::scorecard::{self, AttrScore, DetailSubscore, Scorecard};
    use crate::persona::spec::Color;
    let spec = PersonaSpec::load(&a.spec)?;
    let lex = crate::persona::lexicon::Lexicon::skeleton();
    let weight = |p: &str| lex.get(p).map(|e| e.control_weight() * e.class_weight()).unwrap_or(0.7);
    let mut sc = Scorecard::default();

    // Measurement runs on CPU (the aligner is tiny + deterministic; avoids GPU contention).
    let device = candle_core::Device::Cpu;
    let (detector, pipnet) = scorecard::load_probes(&device).await?;
    let Some(m) = scorecard::measure_landmarks(&a.image, &detector, &pipnet)? else {
        println!("{}  no face detected in {}", style("✗").red(), a.image.display());
        return Ok(());
    };
    println!("{}  {}  (face score {:.2})", style("scorecard").bold(), a.image.display(), m.detection_score);

    // --- landmark metrics + geometric-scalar scoring vs the family calibration prior (§13.1) ---
    println!("\n{}", style("landmark metrics (WFLW-98):").bold());
    println!("  interpupillary / face-width   {:.3}   ({})", m.interpupillary_over_facewidth, style("eyes.spacing").dim());
    println!("  mouth-width / face-width      {:.3}   ({})", m.mouth_over_facewidth, style("mouth.width").dim());
    println!("  face aspect (h/w)             {:.3}   ({})", m.face_aspect, style("face.width⁻¹").dim());

    let table = crate::persona::calibration::CalibrationTable::bundled(&a.model);
    // (path, requested, realised-metric, invert) for the three scalar attrs the aligner measures.
    let scalars = [
        ("eyes.spacing", spec.eyes.as_ref().and_then(|e| e.spacing), m.interpupillary_over_facewidth, false),
        ("mouth.width", spec.mouth.as_ref().and_then(|mo| mo.width), m.mouth_over_facewidth, false),
        ("face.width", spec.face.as_ref().and_then(|f| f.width), m.face_aspect, true),
    ];
    let any_set = scalars.iter().any(|(_, req, _, _)| req.is_some());
    if any_set {
        println!("\n{}", style(format!("geometric scalars (vs {} prior):", a.model)).bold());
        if let Some(t) = &table {
            for s in t.staleness() {
                println!("  {} {}", style("staleness:").yellow(), s);
            }
        }
    }
    for (path, requested, metric, invert) in scalars {
        let Some(req) = requested else { continue };
        match table.as_ref().and_then(|t| scorecard::scalar_score(path, req, metric, t, invert)) {
            Some((_realised, score)) => {
                let glyph = if score.pass { style("✓").green() } else { style("✗").red() };
                println!("  {glyph} {path}: {}", score.note);
                sc.scored.push(score);
            }
            None => {
                println!("  {} {path}: no prior for {} (uncalibrated)", style("·").dim(), a.model);
                sc.pending_calibration.push(path.to_string());
            }
        }
    }

    // --- region_color (eyes.color scorable now via ΔE; skin reported) ---
    let colors = scorecard::measure_colors(&m);
    println!("\n{}", style("region colour (CIELAB):").bold());
    let fmt = |l: Option<[f32; 3]>| l.map(|v| format!("L{:.0} a{:+.0} b{:+.0}", v[0], v[1], v[2])).unwrap_or_else(|| "—".into());
    let eye_target = spec.eyes.as_ref().and_then(|e| e.color.as_ref()).and_then(|c| match c {
        Color::Named(n) => scorecard::color_name_to_lab(n),
        Color::Lab { lab } => Some(*lab),
    });
    match (colors.iris, eye_target) {
        (Some(iris), Some(t)) => {
            let de = scorecard::delta_e(iris, t);
            let pass = de < 20.0;
            println!("  iris  {}   → ΔE {:.1} to eyes.color   {}", fmt(Some(iris)), de, if pass { style("✓").green() } else { style("✗").red() });
            sc.scored.push(AttrScore { path: "eyes.color".into(), pass, weight: weight("eyes.color"), note: format!("ΔE {de:.1}") });
        }
        (iris, _) => {
            println!("  iris  {}   ({})", fmt(iris), style("eyes.color").dim());
            if spec.eyes.as_ref().and_then(|e| e.color.as_ref()).is_some() {
                sc.unmeasurable.push("eyes.color".into());
            }
        }
    }
    println!("  skin  {}   ({})", fmt(colors.skin), style("skin.tone").dim());

    // --- detail sub-score (marks via local_anomaly) ---
    let marks = spec.marks.as_deref().unwrap_or(&[]);
    let positional: Vec<_> = marks.iter().filter(|mk| mk.anchor.is_some()).collect();
    if !positional.is_empty() {
        println!("\n{}", style("localized details (local_anomaly):").bold());
        let (mut pres_sum, mut pos_sum, mut n) = (0.0f32, 0.0f32, 0usize);
        for (i, mk) in positional.iter().enumerate() {
            let anchor = mk.anchor.as_ref().unwrap();
            let kind = mk.kind.as_deref().unwrap_or("mark");
            let name = anchor.landmark.as_deref().or(anchor.region.as_deref()).unwrap_or("?");
            let Some(pos) = scorecard::resolve_anchor(anchor, &m) else {
                println!("  {} {kind}[{i}]: anchor {name:?} not in the WFLW-98 vocabulary", style("?").yellow());
                continue;
            };
            let radius = mk.size.unwrap_or(0.05) * m.face_w;
            let r = scorecard::local_anomaly(&m.crop, pos, radius.max(0.02));
            let glyph = if r.presence > 0.5 { style("✓").green() } else { style("✗").red() };
            println!("  {glyph} {kind}[{i}] @ {}  presence {:.2}  position-error {:.3}", style(name).dim(), r.presence, r.position_error);
            pres_sum += r.presence;
            pos_sum += r.position_error;
            n += 1;
        }
        if n > 0 {
            sc.detail = Some(DetailSubscore { presence_mean: pres_sum / n as f32, position_mean: pos_sum / n as f32, n });
        }
    }

    // --- detect (OWL-ViT, lazy): facial hair + glasses (scorable present/absent) ---
    let mut queries: Vec<(String, String, String, bool)> = Vec::new(); // (label, path, query, expected)
    if let Some(style) = spec.facial_hair.as_ref().and_then(|f| f.style.as_deref()) {
        queries.push((format!("facial_hair.style={style}"), "facial_hair.style".into(), scorecard::facial_hair_query(style).into(), style != "none"));
    }
    if spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).is_some_and(|it| it.iter().any(|x| x.kind.as_deref() == Some("glasses"))) {
        queries.push(("jewelry: glasses".into(), "jewelry.glasses".into(), "glasses".into(), true));
    }
    if !queries.is_empty() {
        let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(&device).await?;
        println!("\n{}", style("salient objects (detect / OWL-ViT):").bold());
        for (label, path, query, expected) in &queries {
            let r = scorecard::detect_probe(&owl, &a.image, query, 0.1)?;
            let pass = r.present == *expected;
            let glyph = if pass { style("✓").green() } else { style("✗").red() };
            println!("  {glyph} {label}: {} (score {:.2}, expected {})", if r.present { "present" } else { "absent" }, r.score, if *expected { "present" } else { "absent" });
            sc.scored.push(AttrScore { path: path.clone(), pass, weight: weight(path), note: String::new() });
        }
    }

    // --- aggregate (§12.2): weighted pass fraction over scored attrs, exclusions reported separately ---
    println!("\n{}", style("── scorecard ──").bold());
    match sc.aggregate() {
        Some(agg) => {
            let styled = if agg >= 0.8 { style(format!("{agg:.2}")).green() } else if agg >= 0.5 { style(format!("{agg:.2}")).yellow() } else { style(format!("{agg:.2}")).red() };
            let passed = sc.scored.iter().filter(|s| s.pass).count();
            println!("  score {styled}  ({}/{} scored attributes pass)", passed, sc.scored.len());
        }
        None => println!("  score {}  (no attributes scorable yet on this spec)", style("—").dim()),
    }
    if let Some(d) = &sc.detail {
        println!("  detail sub-score: presence {:.2}, position {:.3} over {} mark(s)", d.presence_mean, d.position_mean, d.n);
    }
    let ex = |label: &str, v: &[String]| {
        if !v.is_empty() {
            println!("  {} {}: {}", style("·").dim(), style(label).dim(), v.join(", "));
        }
    };
    ex("pending calibration (P4)", &sc.pending_calibration);
    ex("unmeasurable", &sc.unmeasurable);
    ex("non-manifesting", &sc.non_manifesting);
    Ok(())
}

fn run_show(a: ShowArgs) -> Result<()> {
    use crate::persona::compile::{self, EncoderClass};
    use crate::persona::lexicon::Lexicon;
    let spec = PersonaSpec::load(&a.spec)?;
    let lex = Lexicon::skeleton();
    let resolved = compile::resolve(&spec, &lex);
    let compiled = compile::compile_for_model(&spec, &lex, &a.model);

    println!("{}  {}  (model {}, encoder {})", style("persona").bold(), a.spec.display(), a.model, style(compiled.class).cyan());
    let _ = EncoderClass::from_model(&a.model);
    // Per-family controllability grades (§13.3) override the lexicon defaults where a table exists.
    let table = crate::persona::calibration::CalibrationTable::bundled(&a.model);
    if let Some(t) = &table {
        if t.identity.provisional {
            println!("  {} grades from a provisional bootstrap table — run `persona calibrate`", style("note:").dim());
        }
    }
    println!("\n{}", style("resolved attributes (salience-ranked):").bold());
    if resolved.is_empty() {
        println!("  (none — spec is empty or all-unknown)");
    }
    for r in &resolved {
        let emitted = compiled.emitted.iter().any(|p| p == &r.path);
        let mark = if r.phrase.is_empty() {
            style("neg").yellow()
        } else if emitted {
            style("✓").green()
        } else {
            style("dropped").red()
        };
        let phrase = if r.phrase.is_empty() { "(negative only)" } else { &r.phrase };
        let grade = table.as_ref().and_then(|t| t.grade(&r.path)).map(|g| format!(" [{}]", g.as_str())).unwrap_or_default();
        println!("  {mark} {:<22} sal {:.2}{}  {}", style(&r.path).dim(), r.salience, style(grade).magenta(), phrase);
    }
    if !compiled.dropped.is_empty() {
        println!("\n{} {}", style("dropped by budget:").red(), compiled.dropped.join(", "));
    }
    println!("\n{}\n  {}", style("positive:").bold().green(), if compiled.positive.is_empty() { "(empty)" } else { &compiled.positive });
    println!("{}\n  {}", style("negative:").bold().yellow(), if compiled.negative.is_empty() { "(empty)" } else { &compiled.negative });
    Ok(())
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let body = scaffold(&a.depth, &a.name, a.age)?;
    // Sanity: the scaffold must itself load + lint clean.
    let spec = PersonaSpec::from_hjson(&body)
        .with_context(|| "internal: generated scaffold did not parse")?;
    let findings = lint::lint(&spec);
    if lint::has_errors(&findings) {
        anyhow::bail!("internal: generated scaffold did not lint clean: {findings:?}");
    }
    if let Some(parent) = a.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    std::fs::write(&a.out, body).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  persona scaffold ({})  →  {}", style("✓").green(), a.depth, a.out.display());
    println!("   edit it, or run `plakat persona lint {}`", a.out.display());
    Ok(())
}

fn run_lint(a: LintArgs) -> Result<()> {
    let spec = PersonaSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    let (mut errs, mut warns, mut infos) = (0u32, 0u32, 0u32);
    for f in &findings {
        let (tag, styled) = match f.level {
            Level::Error => {
                errs += 1;
                ("error", style("error").red().bold())
            }
            Level::Warning => {
                warns += 1;
                ("warn", style("warn").yellow())
            }
            Level::Info => {
                infos += 1;
                ("info", style("info").cyan())
            }
        };
        let _ = tag;
        println!("  {styled} {}: {}", style(&f.path).bold(), f.message);
    }
    if findings.is_empty() {
        println!("{}  {} — clean", style("✓").green(), a.spec.display());
    } else {
        println!(
            "{}  {errs} error(s), {warns} warning(s), {infos} info",
            if errs > 0 { style("✗").red() } else { style("✓").green() },
        );
    }
    if errs > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Build a valid, partial persona scaffold at the requested depth. HJSON is brace-based; quoteless
/// strings run to end-of-line, so every value sits alone on its line.
fn scaffold(depth: &str, name: &str, age: u32) -> Result<String> {
    let mut s = String::new();
    s.push_str("{\n  schema: persona/1\n\n");
    s.push_str("  identity: {\n");
    s.push_str(&format!("    name: {name}\n"));
    s.push_str(&format!("    apparent_age: {age}\n"));
    s.push_str("    sex: androgynous          # female | male | androgynous\n");
    s.push_str("  }\n\n");
    s.push_str("  # Scalars are 0..1 where 0.5 is the model's own prior; leave a field OUT to mean\n");
    s.push_str("  # \"unknown\" (an explicit 0.5 asserts \"deliberately average\"). See PERSONA.md.\n\n");
    s.push_str("  face: {\n    # shape: oval             # oval|round|square|heart|diamond|oblong|triangular\n");
    s.push_str("    # width: 0.5\n  }\n\n");
    s.push_str("  eyes: {\n    # color: hazel\n    # spacing: 0.5\n  }\n");

    if depth == "standard" || depth == "full" {
        s.push_str("\n  nose: {\n    # profile: straight\n  }\n");
        s.push_str("\n  mouth: {\n    # width: 0.5\n  }\n");
        s.push_str("\n  skin: {\n    # tone: fitzpatrick-3     # fitzpatrick-1..6\n  }\n");
        s.push_str("\n  hair: {\n    # color: auburn\n    # length: shoulder        # buzz|crop|short|chin|shoulder|long|very-long\n  }\n");
        s.push_str("\n  figure: {\n    # height_cm: 170\n    # build: mesomorph\n  }\n");
    }
    if depth == "full" {
        s.push_str("\n  teeth: {\n    # visibility: auto\n    # alignment: even\n  }\n");
        // Collections: an empty list ASSERTS \"none\"; omit the key entirely to mean \"undecided\".
        s.push_str("\n  # marks: []                 # [] asserts \"no marks\"; omit for \"undecided\"\n");
        s.push_str("  # piercings: []\n");
        s.push_str("\n  jewelry: {\n    # identity_locked: false\n    # items: []\n  }\n");
        s.push_str("\n  defaults: {\n    # expression: neutral\n    # framing: headshot\n  }\n");
    }
    if !matches!(depth, "quick" | "standard" | "full") {
        anyhow::bail!("unknown --depth {depth:?} (quick|standard|full)");
    }
    s.push_str("\n  provenance: {\n    method: manual\n    lexicon_version: \"1.0\"\n  }\n}\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffolds_lint_clean_at_every_depth() {
        for depth in ["quick", "standard", "full"] {
            let body = scaffold(depth, "alice", 34).unwrap();
            let spec = PersonaSpec::from_hjson(&body)
                .unwrap_or_else(|e| panic!("depth {depth} did not parse: {e}\n{body}"));
            let f = lint::lint(&spec);
            assert!(!lint::has_errors(&f), "depth {depth} did not lint clean: {f:?}");
            assert_eq!(spec.schema_version(), Some(1));
            assert_eq!(spec.identity.unwrap().apparent_age, Some(34));
        }
    }

    #[test]
    fn scaffold_rejects_bad_depth() {
        assert!(scaffold("deep", "x", 30).is_err());
    }
}
