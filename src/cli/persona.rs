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
    /// Bake a per-base adapter (Tier C, §11.6) from a cast reference set — a textual-inversion token or
    /// a LoRA. Excludes swappable presentation jewelry by default; memory-gated.
    Bake(BakeArgs),
    /// Diff two persona specs by attribute class (§6.5) — reports which changes are structural (force a
    /// re-cast) vs surface / detail / presentation (cheap in-place repairs). No weights.
    Diff(DiffArgs),
    /// Repair one failing attribute on a render in place (RFC §12.4): mask the attribute's region →
    /// recomposite (detail) or regional inpaint (surface) → re-score → keep only on improvement.
    Repair(RepairArgs),
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
    /// Additional persona directories to place in the same scene (§14.2 multiperson). Repeatable;
    /// each is assigned to a left-to-right figure band and swapped only into its own face.
    #[arg(long = "with")]
    pub with: Vec<PathBuf>,
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

#[derive(Args, Debug)]
pub struct BakeArgs {
    /// Persona directory produced by `persona cast`.
    pub persona: PathBuf,
    /// Base model to bake for.
    #[arg(long)]
    pub base: String,
    /// `ti` (textual inversion — cheap, composable) or `lora` (stronger, heavier).
    #[arg(long, default_value = "lora")]
    pub method: String,
    /// Trigger token for the baked identity.
    #[arg(long, default_value = "sks")]
    pub token: String,
    /// Training steps (trainer default when 0).
    #[arg(long, default_value_t = 0)]
    pub steps: usize,
    /// Training resolution.
    #[arg(long, default_value_t = 512)]
    pub size: u32,
    /// Include worn presentation jewelry in the trained reference set (default: exclude, §11.6).
    #[arg(long)]
    pub keep_jewelry: bool,
}

#[derive(Args, Debug)]
pub struct DiffArgs {
    /// The original persona spec.
    pub old: PathBuf,
    /// The edited persona spec.
    pub new: PathBuf,
}

#[derive(Args, Debug)]
pub struct RepairArgs {
    /// Path to the persona spec (`.hjson`).
    pub spec: PathBuf,
    /// The rendered image to repair.
    #[arg(long)]
    pub image: PathBuf,
    /// The failing attribute path (e.g. `eyes.color`, `marks`, `face.width`).
    #[arg(long)]
    pub attr: String,
    /// Output path (default: `repaired.png`).
    #[arg(long, default_value = "repaired.png")]
    pub out: PathBuf,
    /// Model for a surface inpaint repair.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Inpaint denoise strength for a surface repair.
    #[arg(long, default_value_t = 0.4)]
    pub strength: f32,
    /// Seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
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
        PersonaCmd::Bake(a) => run_bake(a).await,
        PersonaCmd::Diff(a) => run_diff(a),
        PersonaCmd::Repair(a) => run_repair(a).await,
    }
}

/// Full-frame feathered mask over an attribute's feature region, from the realised landmarks (§12.4
/// mask source #1). Returns `None` for a section with no landmark region (caller falls back to detect).
fn feature_mask(section: &str, m: &crate::persona::scorecard::FaceMetrics, fw: u32, fh: u32) -> Option<image::GrayImage> {
    use crate::persona::geometry::topology::*;
    let idxs: Vec<usize> = match section {
        "eyes" => EYE_RIGHT.chain(EYE_LEFT).chain([PUPIL_RIGHT, PUPIL_LEFT]).collect(),
        "mouth" => LIP_OUTER.collect(),
        // dentition sits inside the inner-lip aperture (§8.7).
        "teeth" => LIP_INNER.collect(),
        "nose" => NOSE.collect(),
        "face" | "skin" => CONTOUR.collect(),
        _ => return None,
    };
    let (cw, ch) = (m.crop.width() as f32, m.crop.height() as f32);
    let pts: Vec<(f32, f32)> = idxs
        .iter()
        .map(|&i| (m.crop_origin.0 as f32 + m.landmarks[i].0 * cw, m.crop_origin.1 as f32 + m.landmarks[i].1 * ch))
        .collect();
    let (x0, y0, x1, y1) = pts.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |(ax, ay, bx, by), &(x, y)| {
        (ax.min(x), ay.min(y), bx.max(x), by.max(y))
    });
    // grow by 25% of the region size, then fill a feathered ellipse.
    let (gx, gy) = ((x1 - x0) * 0.25 + 4.0, (y1 - y0) * 0.25 + 4.0);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (rx, ry) = ((x1 - x0) / 2.0 + gx, (y1 - y0) / 2.0 + gy);
    let mut mask = image::GrayImage::from_pixel(fw, fh, image::Luma([0]));
    for y in 0..fh {
        for x in 0..fw {
            let (dx, dy) = ((x as f32 - cx) / rx.max(1.0), (y as f32 - cy) / ry.max(1.0));
            let d = (dx * dx + dy * dy).sqrt();
            if d <= 1.0 {
                let a = ((1.0 - d) / 0.3).clamp(0.0, 1.0);
                mask.put_pixel(x, y, image::Luma([(a * 255.0) as u8]));
            }
        }
    }
    Some(mask)
}

async fn run_repair(a: RepairArgs) -> Result<()> {
    use crate::persona::edit::{class_of, Class};
    use crate::persona::{compile, detail, lexicon::Lexicon, scorecard};

    let spec = PersonaSpec::load(&a.spec)?;
    let lex = Lexicon::skeleton();
    let class = class_of(&a.attr, &lex);
    let section = a.attr.split('.').next().unwrap_or(&a.attr).to_string();
    println!("{}  {} on {}  (class {})", style("persona repair").bold(), a.attr, a.image.display(), class.as_str());

    let device = candle_core::Device::Cpu;
    let (det, pip) = scorecard::load_probes(&device).await?;
    let Some(m) = scorecard::measure_landmarks(&a.image, &det, &pip)? else {
        println!("{}  no face detected — cannot repair", style("✗").red());
        return Ok(());
    };

    match class {
        // Structural geometry cannot be fixed in place — the sampler decided it (§12.4).
        Class::Structural => {
            println!("  {} `{}` is structural — it cannot be repaired in place; re-cast with the edited spec (`persona cast`)", style("⚠").yellow(), a.attr);
            std::fs::copy(&a.image, &a.out)?;
            Ok(())
        }
        // Details are deterministic — recomposite (§8.4). Cheapest repair.
        Class::Detail | Class::Presentation => {
            let base = image::open(&a.image)?.to_rgb8();
            let r = detail::composite_details(&base, &spec, &m, a.seed);
            r.image.save(&a.out)?;
            println!("  {} recomposited {} detail(s) → {}", style("✓").green(), r.placed, a.out.display());
            Ok(())
        }
        // Surface → regional inpaint with an attribute-focused prompt, kept only on improvement.
        Class::Surface => {
            let (fw, fh) = (m.crop.width().max(1), m.crop.height().max(1));
            let full = image::open(&a.image)?.to_rgb8();
            let Some(mask) = feature_mask(&section, &m, full.width(), full.height()) else {
                println!("  {} no landmark region for `{section}` — targeted surface repair not available (try detect-based repair)", style("·").yellow());
                std::fs::copy(&a.image, &a.out)?;
                return Ok(());
            };
            let _ = (fw, fh);
            // focused prompt = a dentition prompt for teeth (§8.7), else the attribute's resolved phrase.
            let resolved = compile::resolve(&spec, &lex);
            let phrase = if section == "teeth" {
                compile::dentition_prompt(&spec).unwrap_or_else(|| "teeth".into())
            } else {
                resolved.iter().find(|r| r.path == a.attr).map(|r| r.phrase.clone()).unwrap_or_else(|| a.attr.clone())
            };
            let eye_target = scorecard::eyes_color_target(&spec);
            let before = eye_target.and_then(|t| scorecard::measure_colors(&m).iris.map(|iris| scorecard::delta_e(iris, t)));

            let tmp = std::env::temp_dir().join(format!("persona_repair_{}.png", a.seed));
            let mtmp = std::env::temp_dir().join(format!("persona_repair_mask_{}.png", a.seed));
            full.save(&tmp)?;
            mask.save(&mtmp)?;
            println!("  {} inpainting the {section} region with \"{phrase}\"…", style("→").cyan());
            let out = crate::api::Img2img::new(&a.model, &tmp)
                .prompt(&phrase)
                .mask(&mtmp)
                .mask_feather(6)
                .strength(a.strength)
                .seed(a.seed)
                .run()
                .await
                .context("surface repair inpaint")?;
            let Some(img) = out.into_iter().next() else {
                anyhow::bail!("inpaint produced no image");
            };
            img.save(&a.out)?;

            // Re-score: accept only on improvement, else revert (§12.4).
            if let Some(before_de) = before {
                if let Some(m2) = scorecard::measure_landmarks(&a.out, &det, &pip)? {
                    let after = scorecard::eyes_color_target(&spec).and_then(|t| scorecard::measure_colors(&m2).iris.map(|iris| scorecard::delta_e(iris, t)));
                    if let Some(after_de) = after {
                        if after_de <= before_de {
                            println!("  {} accepted: ΔE {before_de:.1} → {after_de:.1}", style("✓").green());
                        } else {
                            std::fs::copy(&a.image, &a.out)?;
                            println!("  {} reverted: ΔE worsened {before_de:.1} → {after_de:.1} (kept the original)", style("✗").red());
                        }
                    }
                }
            } else {
                println!("  {} repaired → {} (no scorable target for `{}`, kept the inpaint)", style("✓").green(), a.out.display(), a.attr);
            }
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(&mtmp);
            Ok(())
        }
    }
}

fn run_diff(a: DiffArgs) -> Result<()> {
    use crate::persona::edit::{self, ChangeKind, Class};
    use crate::persona::lexicon::Lexicon;
    let old = std::fs::read_to_string(&a.old).with_context(|| format!("reading {}", a.old.display()))?;
    let new = std::fs::read_to_string(&a.new).with_context(|| format!("reading {}", a.new.display()))?;
    let lex = Lexicon::skeleton();
    let changes = edit::diff(&old, &new, &lex)?;

    println!("{}  {} → {}", style("persona diff").bold(), a.old.display(), a.new.display());
    if changes.is_empty() {
        println!("  {} no attribute changes", style("·").dim());
        return Ok(());
    }
    for cls in [Class::Structural, Class::Surface, Class::Detail, Class::Presentation] {
        let group: Vec<_> = changes.iter().filter(|c| c.class == cls).collect();
        if group.is_empty() {
            continue;
        }
        let head = match cls {
            Class::Structural => style(cls.as_str()).red(),
            Class::Surface => style(cls.as_str()).yellow(),
            Class::Detail => style(cls.as_str()).green(),
            Class::Presentation => style(cls.as_str()).cyan(),
        };
        println!("\n  {head} — {}", style(cls.invalidation()).dim());
        for c in group {
            let verb = match c.kind {
                ChangeKind::Added => "＋",
                ChangeKind::Removed => "－",
                ChangeKind::Changed => "→",
            };
            let val = match (&c.old, &c.new) {
                (Some(o), Some(n)) => format!("{o} {verb} {n}"),
                (None, Some(n)) => format!("{verb} {n}"),
                (Some(o), None) => format!("{verb} (was {o})"),
                _ => String::new(),
            };
            println!("    {} {}  {}", verb, style(&c.path).bold(), style(val).dim());
        }
    }
    let s = edit::summarize(&changes);
    println!("\n{}", style("── summary ──").bold());
    println!("  {} structural · {} surface · {} detail · {} presentation", s.structural, s.surface, s.detail, s.presentation);
    if s.invalidates_references() {
        println!("  {} {} structural change(s) will INVALIDATE the reference set + baked adapters — re-cast required", style("⚠").red(), s.structural);
    } else {
        println!("  {} no structural changes — repairs are in-place (inpaint / recomposite / per-render), the reference set stands", style("✓").green());
    }
    Ok(())
}

async fn run_bake(a: BakeArgs) -> Result<()> {
    use crate::persona::casting::ReferenceSet;
    use crate::persona::{detail, scorecard};

    let set = ReferenceSet::load(&a.persona)
        .with_context(|| format!("loading reference set from {} (run `persona cast` first)", a.persona.display()))?;
    if set.references.is_empty() {
        anyhow::bail!("reference set is empty — nothing to bake");
    }
    let spec = PersonaSpec::load(&a.persona.join("spec.hjson")).ok();

    // Presentation jewelry is excluded from the trained set by default (§11.6) unless identity_locked
    // or `--keep-jewelry`. Rebuild jewelry-free references from the raw candidates when needed.
    let identity_locked = spec.as_ref().and_then(|s| s.jewelry.as_ref()).and_then(|j| j.identity_locked).unwrap_or(false);
    let has_worn_jewelry = spec.as_ref().and_then(|s| s.jewelry.as_ref()).and_then(|j| j.items.as_ref()).is_some_and(|i| !i.is_empty());
    let exclude_jewelry = has_worn_jewelry && !identity_locked && !a.keep_jewelry;

    let bake_dir = a.persona.join("bake");
    std::fs::create_dir_all(&bake_dir)?;
    let train_dir = bake_dir.join("train");
    std::fs::create_dir_all(&train_dir)?;

    println!("{}  {} on {} via {}", style("persona bake").bold(), set.persona, a.base, a.method);

    // Build the training image set.
    let mut images: Vec<PathBuf> = Vec::new();
    if exclude_jewelry {
        println!("  {} excluding presentation jewelry from the trained set (§11.6; --keep-jewelry to include, or set jewelry.identity_locked)", style("note:").cyan());
        let device = candle_core::Device::Cpu;
        let (det, pip) = scorecard::load_probes(&device).await?;
        let spec = spec.as_ref().unwrap();
        for r in &set.references {
            let Some(raw_rel) = &r.raw_image else {
                println!("  {} ref {} has no stored raw — training on its composited image (may carry jewelry)", style("·").yellow(), r.id);
                images.push(a.persona.join(&r.image));
                continue;
            };
            let raw_path = a.persona.join(raw_rel);
            match scorecard::measure_landmarks(&raw_path, &det, &pip)? {
                Some(m) => {
                    let base = image::open(&raw_path)?.to_rgb8();
                    let out = train_dir.join(format!("train_{}.png", r.id));
                    detail::composite_details_opts(&base, spec, &m, r.seed, false).image.save(&out)?;
                    images.push(out);
                }
                None => images.push(a.persona.join(&r.image)),
            }
        }
    } else {
        for r in &set.references {
            images.push(a.persona.join(&r.image));
        }
    }
    println!("  training on {} reference image(s)", images.len());

    // Memory gate (§11.6 / §22): warn up front, and arm the training-mode guard around the run so a
    // sustained-pressure OOM aborts cleanly rather than getting Killed.
    let device = crate::api::device("auto").unwrap_or(candle_core::Device::Cpu);
    crate::hw::memory_preflight(&device, &a.base);
    let _guard = crate::memwatch::MemoryGuard::start_mode(&device, &format!("persona bake {}", a.base), crate::memwatch::Mode::Training);

    // Train.
    let artifact = bake_dir.join(format!("{}_{}.safetensors", a.base, a.method));
    match a.method.as_str() {
        "ti" => {
            let mut t = crate::api::EmbeddingTrain::new(&a.base, images, &a.token, &artifact);
            if a.steps > 0 {
                t = t.steps(a.steps);
            }
            t.size(a.size).run().await.context("textual-inversion bake")?;
        }
        "lora" => {
            let mut t = crate::api::StyleTrain::new(&a.base, images, &artifact).trigger(&a.token);
            if a.steps > 0 {
                t = t.steps(a.steps);
            }
            t.size(a.size).run().await.context("LoRA bake")?;
        }
        other => anyhow::bail!("unknown --method `{other}` (expected `ti` or `lora`)"),
    }

    // Record the inputs the bake was measured under, for invalidation (§11.6).
    let cond = set.references.first().map(|r| r.conditioning_hash.clone()).unwrap_or_default();
    let detail_hash = set.references.first().map(|r| r.detail_plan_hash.clone()).unwrap_or_default();
    let record = format!(
        "{{\n  base: {:?},\n  method: {:?},\n  token: {:?},\n  artifact: {:?},\n  jewelry_excluded: {exclude_jewelry},\n  conditioning_hash: {:?},\n  detail_plan_hash: {:?},\n  topology: {:?},\n  lexicon_version: {:?}\n}}\n",
        a.base, a.method, a.token, artifact.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        cond, detail_hash,
        crate::persona::calibration::table::CURRENT_TOPOLOGY, crate::persona::calibration::table::CURRENT_LEXICON,
    );
    std::fs::write(bake_dir.join(format!("{}_{}.bake.json", a.base, a.method)), record)?;

    println!("{} {} (invalidated by structural / baked-mark / lexicon / topology changes)", style("done:").bold(), artifact.display());
    Ok(())
}

async fn run_render(a: RenderArgs) -> Result<()> {
    use crate::persona::casting::{self, ReferenceSet};
    use crate::persona::{compile, detail, lexicon::Lexicon, scorecard};

    if !a.with.is_empty() {
        return run_render_multi(a).await;
    }

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

                    // Mouth-region dentition inpaint (§8.7): when teeth manifest, mask the inner-lip
                    // aperture and regenerate it with a dentition-focused prompt. (The geometry
                    // dentition hint as a ControlNet cond needs the lower-level t2i path; prompt-only
                    // here — the facade can't pass controls.)
                    if crate::persona::geometry::open_mouth(spec) {
                        if let (Some(dprompt), Some(mask)) = (compile::dentition_prompt(spec), feature_mask("teeth", &m, a.size, a.size)) {
                            let cur = image::open(&a.out)?.to_rgb8();
                            let tmp = std::env::temp_dir().join(format!("persona_teeth_{}.png", a.seed));
                            let mtmp = std::env::temp_dir().join(format!("persona_teeth_mask_{}.png", a.seed));
                            cur.save(&tmp)?;
                            mask.save(&mtmp)?;
                            println!("  {} dentition inpaint (\"{dprompt}\") over the mouth aperture (§8.7)…", style("→").cyan());
                            match crate::api::Img2img::new(&a.model, &tmp).prompt(&dprompt).mask(&mtmp).mask_feather(4).strength(0.5).seed(a.seed).run().await {
                                Ok(imgs) => {
                                    if let Some(img) = imgs.into_iter().next() {
                                        img.save(&a.out)?;
                                        println!("  {} teeth refined", style("✓").green());
                                    }
                                }
                                Err(e) => println!("  {} dentition inpaint skipped: {e}", style("·").yellow()),
                            }
                            let _ = std::fs::remove_file(&tmp);
                            let _ = std::fs::remove_file(&mtmp);
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

/// Multiperson render (§14.2): place several personas in one scene. Each is assigned a left-to-right
/// figure band; a detected face attributed to a band (above confidence) gets that persona's face
/// swapped in + its details composited — and NOTHING when attribution is below threshold, so figure
/// A's scar is never composited onto figure B (the catastrophic-failure guard).
async fn run_render_multi(a: RenderArgs) -> Result<()> {
    use crate::persona::casting::{self, ReferenceSet};
    use crate::persona::{compile, detail, lexicon::Lexicon, scorecard};

    // Load every persona (primary + each --with), in left-to-right order.
    let mut dirs = vec![a.persona.clone()];
    dirs.extend(a.with.iter().cloned());
    let lex = Lexicon::skeleton();
    struct P {
        dir: PathBuf,
        set: ReferenceSet,
        spec: Option<PersonaSpec>,
        appearance: String,
    }
    let mut personas: Vec<P> = Vec::new();
    for d in &dirs {
        let set = ReferenceSet::load(d).with_context(|| format!("loading reference set from {}", d.display()))?;
        if set.references.is_empty() {
            anyhow::bail!("reference set {} is empty", d.display());
        }
        let spec = PersonaSpec::load(&d.join("spec.hjson")).ok();
        let appearance = spec.as_ref().map(|s| compile::compile_for_model(s, &lex, &a.model).positive).unwrap_or_default();
        personas.push(P { dir: d.clone(), set, spec, appearance });
    }
    let n = personas.len();
    println!("{}  {} personas into a scene  (model {})", style("persona render").bold(), n, a.model);

    // Scene prompt: "N people" + each persona's appearance.
    let count_word = match n {
        2 => "two people".to_string(),
        3 => "three people".to_string(),
        _ => format!("{n} people"),
    };
    let appearances: Vec<String> = personas.iter().map(|p| p.appearance.clone()).filter(|s| !s.is_empty()).collect();
    let prompt = if appearances.is_empty() {
        format!("{count_word}, {}", a.scene)
    } else {
        format!("{count_word}, {}, {}", a.scene, appearances.join("; "))
    };

    println!("  {} generating the group scene…", style("→").cyan());
    let imgs = crate::api::Generate::new(&a.model)
        .prompt(&prompt)
        .seed(a.seed)
        .size(a.size, a.size)
        .steps(a.steps)
        .run()
        .await
        .context("group scene render")?;
    let Some(scene_img) = imgs.into_iter().next() else {
        anyhow::bail!("group scene render produced no image");
    };
    scene_img.save(&a.out)?;

    let device = candle_core::Device::Cpu;
    let swapper = crate::pipelines::faceswap::FaceSwapper::load_resolved(&device, candle_core::DType::F32).await?;
    let faces = swapper.detect(&a.out).context("detecting scene faces")?;
    if faces.is_empty() {
        println!("  {} no faces detected in the group scene — leaving it un-swapped", style("·").yellow());
        return Ok(());
    }
    println!("  detected {} face(s)", faces.len());

    // Figure bands: N equal vertical slices, left→right, mapped to the personas in order.
    let fw = a.size as f32;
    let bands: Vec<[f32; 4]> = (0..n)
        .map(|i| [fw * i as f32 / n as f32, 0.0, fw * (i + 1) as f32 / n as f32, a.size as f32])
        .collect();
    let face_boxes: Vec<[f32; 4]> = faces.iter().map(|f| f.bbox).collect();
    let assignments = casting::assign(&face_boxes, &bands, casting::ATTRIBUTION_CONFIDENCE_MIN);

    // Invert to persona(band) → face, and swap each persona's canonical face into its face.
    let mut scene = image::open(&a.out)?.to_rgb8();
    let mut swapped_faces: Vec<Option<usize>> = vec![None; n]; // band → face idx
    for asg in &assignments {
        if let Some(band) = asg.figure {
            if swapped_faces[band].is_none() {
                swapped_faces[band] = Some(asg.face);
            }
        }
    }
    for (band, face_idx) in swapped_faces.iter().enumerate() {
        let p = &personas[band];
        let Some(face_idx) = face_idx else {
            println!("  {} {}: no face attributed with confidence ≥ {:.2} — left absent, not misplaced (§14.2)", style("·").yellow(), p.set.persona, casting::ATTRIBUTION_CONFIDENCE_MIN);
            continue;
        };
        let target = &faces[*face_idx];
        let Some(canonical) = p.set.canonical() else { continue };
        let latent = swapper.source_latent(&p.dir.join(&canonical.image)).context("embedding canonical face")?;
        scene = swapper.swap_into(&scene, target.landmarks, &latent).context("face swap")?;
        println!("  {} {} → face {} (band {band})", style("✓").green(), p.set.persona, face_idx);
    }
    scene.save(&a.out)?;

    // Detail compositing per persona, each against ITS OWN assigned face's landmarks (§14.2).
    if !a.no_details {
        let (det, pip) = scorecard::load_probes(&device).await?;
        for (band, face_idx) in swapped_faces.iter().enumerate() {
            let (Some(face_idx), Some(spec)) = (face_idx, personas[band].spec.as_ref()) else { continue };
            let has_details = spec.marks.as_ref().is_some_and(|m| !m.is_empty())
                || spec.jewelry.as_ref().and_then(|j| j.items.as_ref()).is_some_and(|i| !i.is_empty())
                || spec.piercings.as_ref().is_some_and(|p| !p.is_empty());
            if !has_details {
                continue;
            }
            // measure only within this persona's face band, so its details anchor to its own face.
            let bbox = faces[*face_idx].bbox;
            let crop = casting::refine_crop(bbox, 0.6, a.size, a.size);
            let sub = image::imageops::crop_imm(&scene, crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]).to_image();
            let tmp = std::env::temp_dir().join(format!("persona_mp_{band}.png"));
            sub.save(&tmp)?;
            if let Some(m) = scorecard::measure_landmarks(&tmp, &det, &pip)? {
                let r = detail::composite_details(&sub, spec, &m, a.seed + band as u64);
                // paste the composited sub-crop back into the scene.
                image::imageops::overlay(&mut scene, &r.image, crop[0] as i64, crop[1] as i64);
                println!("  {} composited {} detail(s) for {}", style("✓").green(), r.placed, personas[band].set.persona);
            }
            let _ = std::fs::remove_file(&tmp);
        }
        scene.save(&a.out)?;
    }

    println!("{} {}", style("done:").bold(), a.out.display());
    Ok(())
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
        raw: PathBuf,
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
        cands.push(Cand { image: ref_img, raw, seed, score, aesthetic: aes, embedding });
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
        // keep the raw (un-composited) candidate so `bake` can build a jewelry-free set (§11.6).
        let raw_rel = if std::fs::copy(&c.raw, refs_dir.join(format!("ref_{rank}_raw.png"))).is_ok() {
            Some(PathBuf::from("references").join(format!("ref_{rank}_raw.png")))
        } else {
            None
        };
        references.push(Reference {
            id: rank,
            image: PathBuf::from("references").join(format!("ref_{rank}.png")),
            seed: c.seed,
            model: a.model.clone(),
            view: "frontal".into(),
            expression: "neutral".into(),
            conditioning_hash: conditioning_hash.clone(),
            detail_plan_hash: detail_plan_hash.clone(),
            raw_image: raw_rel,
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
