//! `plakat paint` CLI (RFC PAINT-1). P0 exposes `paint from <IMAGE>` — repaint an existing image in stroke
//! space, the make-or-break filter-gate — plus `paint palette <NAME>` to inspect a palette. The spec-driven
//! `paint <SPEC>`, `replay`, `export`, and the rest land across later phases. Everything here is GPU-free.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;

use crate::paint::painter::{self, PaintParams};
use crate::paint::palette::{self, Palette};

#[derive(Args, Debug)]
pub struct PaintArgs {
    /// A PaintSpec to paint (when no subcommand is given): `plakat paint <SPEC>`.
    pub spec: Option<PathBuf>,
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    /// Override the output size `WxH`.
    #[arg(long)]
    pub size: Option<String>,
    /// Override the stroke budget (total marks). When unset, the spec's `budget.strokes` is used, else a
    /// painting-scale default is derived from the canvas size and medium.
    #[arg(long)]
    pub strokes: Option<usize>,
    /// Fidelity register: `legible` (default — resolves & hardens edges) or `impressionist` (loose masses).
    #[arg(long, default_value = "legible")]
    pub style: String,
    /// Edge-HARDNESS strength (0..1) — how many boundaries are treated as HARD (strokes stop, masses meet crisply). Legible only.
    #[arg(long, default_value_t = 0.6)]
    pub define: f32,
    /// AERIAL PERSPECTIVE strength (0..1): veil distant passages toward the atmosphere so the background recedes
    /// and the foreground advances (depth "layers"). Uses the armature's depth map. 0 = flat single plane.
    #[arg(long, default_value_t = 0.55)]
    pub haze: f32,
    /// STROKE LENGTH multiplier (1.0 = default): longer = cleaner sweeping strokes; shorter = choppier.
    #[arg(long, default_value_t = 1.0)]
    pub stroke_length: f32,
    /// STROKE WIDTH multiplier (1.0 = default): wider = fewer/broader marks; narrower = finer/more marks.
    #[arg(long, default_value_t = 1.0)]
    pub stroke_width: f32,
    /// BLEED (0..1) wet-into-wet fusion — overrides the medium default (watercolour/ink bleed; oil/gouache don't).
    #[arg(long)]
    pub bleed: Option<f32>,
    /// OPACITY / body (0.1..1) — overrides the medium default (1 = opaque; low = transparent).
    #[arg(long)]
    pub opacity: Option<f32>,
    /// PICKUP (0..1) dirty-brush drag — overrides the medium default.
    #[arg(long)]
    pub pickup: Option<f32>,
    /// IMPASTO (0..1) textured thick-paint relief at output — overrides the medium default (oil high, flat 0).
    #[arg(long)]
    pub impasto: Option<f32>,
    /// Also print the traceability.
    #[arg(long)]
    pub report: bool,
    /// Apply the merge with N depth planes (recession + palette unity) before painting. Uses a CPU depth proxy
    /// for bring-up; the real per-figure armature is the GPU path. Default off (single plane).
    #[arg(long)]
    pub planes: Option<u32>,
    /// Pass-level CRITIC (§10.1): score each stage pass with the aesthetic predictor and roll back any pass
    /// that makes the painting worse. GPU.
    #[arg(long)]
    pub critic: bool,
    /// FAMILY-KEY the reference (§5.5.5): split light/shadow and enforce the family-separation invariant so the
    /// masses read solid (sharper subject). Opt-in.
    #[arg(long)]
    pub families: bool,
    /// FOCAL HARD-EDGE (§7.4): matte the subject (U2Net) and terminate strokes at its silhouette, so the
    /// subject stays crisp against the ground instead of smearing across it. GPU.
    #[arg(long)]
    pub crisp: bool,
    #[command(subcommand)]
    pub cmd: Option<PaintCmd>,
}

#[derive(Subcommand, Debug)]
pub enum PaintCmd {
    /// Scaffold a PaintSpec.
    New(NewArgs),
    /// Show a spec's compiled plan: stages, brush radii, per-stage budget.
    Show(SpecFileArgs),
    /// Validate a spec (medium executable, palette known, reference present).
    Lint(SpecFileArgs),
    /// Repaint an existing image in stroke space (P0 — the filter-gate). Writes the image + its stroke score.
    From(FromArgs),
    /// Re-render a stroke score to an image at any size (no GPU).
    Replay(ReplayArgs),
    /// Export derived products from a score: per-stage separations, a stages sheet, the impasto height.
    Export(ExportArgs),
    /// Render a stroke-by-stroke timelapse from a score (PNG frames; GIF when few enough).
    Timelapse(TimelapseArgs),
    /// Inspect a built-in palette's pigments.
    Palette(PaletteArgs),
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// The `.strokes` score.
    pub score: PathBuf,
    /// What to export: `separations` (one PNG per stage), `stages` (cumulative per stage), `height` (16-bit).
    #[arg(value_parser = ["separations", "stages", "height"])]
    pub target: String,
    /// Output directory (separations/stages) or file (height).
    #[arg(short, long, default_value = "export")]
    pub out: PathBuf,
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct TimelapseArgs {
    pub score: PathBuf,
    /// Output directory for the PNG frames.
    #[arg(short, long, default_value = "timelapse")]
    pub out: PathBuf,
    /// Emit a frame every N strokes.
    #[arg(long, default_value_t = 25)]
    pub every: usize,
    #[arg(long)]
    pub size: Option<String>,
}

/// The resolved arguments for painting a spec (from the top-level positional form).
pub struct SpecArgs {
    pub spec: PathBuf,
    pub out: Option<PathBuf>,
    pub size: Option<String>,
    pub report: bool,
    pub planes: Option<u32>,
    pub critic: bool,
    pub families: bool,
    pub crisp: bool,
    pub strokes: Option<usize>,
    pub style: String,
    pub define: f32,
    pub haze: f32,
    pub stroke_length: f32,
    pub stroke_width: f32,
    pub bleed: Option<f32>,
    pub opacity: Option<f32>,
    pub pickup: Option<f32>,
    pub impasto: Option<f32>,
}

/// Resize a per-pixel depth field from `(sw,sh)` to `(dw,dh)` and normalise it to `[0,1]` (min–max), so aerial
/// perspective works regardless of the raw depth range. Larger = nearer.
fn resize_depth(depth: &[f32], (sw, sh): (u32, u32), (dw, dh): (u32, u32)) -> Vec<f32> {
    let (mn, mx) = depth.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (mx - mn).max(1e-6);
    let norm = image::ImageBuffer::from_fn(sw, sh, |x, y| {
        let v = depth.get((y * sw + x) as usize).copied().unwrap_or(mn);
        image::Luma([(((v - mn) / span) * 65535.0).round().clamp(0.0, 65535.0) as u16])
    });
    let scaled = image::imageops::resize(&norm, dw, dh, image::imageops::FilterType::Triangle);
    scaled.pixels().map(|p| p.0[0] as f32 / 65535.0).collect()
}

/// Parse the fidelity register from the CLI string.
fn parse_style(s: &str) -> Result<crate::paint::painter::PaintStyle> {
    use crate::paint::painter::PaintStyle;
    match s.trim().to_ascii_lowercase().as_str() {
        "legible" | "legibility" => Ok(PaintStyle::Legible),
        "impressionist" | "impressionistic" | "loose" => Ok(PaintStyle::Impressionist),
        "fidelity" | "high-fidelity" | "realist" | "realistic" | "detailed" | "tight" => Ok(PaintStyle::Fidelity),
        other => anyhow::bail!("unknown --style {other:?} (use: legible | impressionist | fidelity)"),
    }
}

#[derive(Args, Debug)]
pub struct NewArgs {
    #[arg(default_value = "painting.paint.hjson")]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct SpecFileArgs {
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ReplayArgs {
    /// The `.strokes` score to render.
    pub score: PathBuf,
    #[arg(short, long, default_value = "replay.png")]
    pub out: PathBuf,
    /// Render size `WxH` (default: the score's native size).
    #[arg(long)]
    pub size: Option<String>,
    /// Stop after stroke N (truncation — an intermediate state of the painting).
    #[arg(long)]
    pub until: Option<u32>,
}

#[derive(Args, Debug)]
pub struct FromArgs {
    /// The reference image to repaint.
    pub input: PathBuf,
    /// Output path.
    #[arg(short, long, default_value = "painting.png")]
    pub out: PathBuf,
    /// Palette name (zorn / split-primary / verdaccio / earth / limited-landscape / sumi).
    #[arg(long, default_value = "zorn")]
    pub palette: String,
    /// Total stroke budget — inviolable.
    #[arg(long, default_value_t = 1500)]
    pub budget: usize,
    /// Brush radii, coarse → fine (comma-separated). Default: derived from the image size.
    #[arg(long, value_delimiter = ',')]
    pub brush: Option<Vec<f32>>,
    /// The smallest brush allowed (keeps the finest pass off pixel detail).
    #[arg(long, default_value_t = 4.0)]
    pub min_brush: f32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Print the traceability (correlation to the reference): high = a filter, lower = a painting.
    #[arg(long)]
    pub report: bool,
    /// Fidelity register: `legible` (default — resolves resolves features, draws edges hardens edges) or `impressionist` (loose masses).
    #[arg(long, default_value = "legible")]
    pub style: String,
    /// Edge-HARDNESS strength (0..1) — how many boundaries are treated as HARD (strokes stop, masses meet crisply). Legible only.
    #[arg(long, default_value_t = 0.6)]
    pub define: f32,
    /// AERIAL PERSPECTIVE strength (0..1) using a CPU depth proxy (central+low = near). 0 = flat.
    #[arg(long, default_value_t = 0.0)]
    pub haze: f32,
    /// STROKE LENGTH multiplier (1.0 = default): longer = cleaner sweeping strokes; shorter = choppier.
    #[arg(long, default_value_t = 1.0)]
    pub stroke_length: f32,
    /// STROKE WIDTH multiplier (1.0 = default): wider = fewer/broader marks; narrower = finer/more marks.
    #[arg(long, default_value_t = 1.0)]
    pub stroke_width: f32,
    /// BLEED (0..1) wet-into-wet fusion (default 0 for `from`).
    #[arg(long)]
    pub bleed: Option<f32>,
    /// OPACITY / body (0.1..1) — 1 = opaque, low = transparent.
    #[arg(long)]
    pub opacity: Option<f32>,
    /// PICKUP (0..1) dirty-brush drag.
    #[arg(long)]
    pub pickup: Option<f32>,
    /// IMPASTO (0..1) textured thick-paint relief at output.
    #[arg(long)]
    pub impasto: Option<f32>,
    /// CHROMA / saturation (1 neutral; >1 vivid; <1 muted).
    #[arg(long)]
    pub chroma: Option<f32>,
    /// DRY SHIFT (−0.4..0.4): + dries lighter (watercolour); − dries to a matte mid (gouache).
    #[arg(long)]
    pub dry_shift: Option<f32>,
    /// GRANULATION (0..1) paper-tooth pigment settling (watercolour / graphite grain).
    #[arg(long)]
    pub granulate: Option<f32>,
    /// SHEEN / gloss (0..1) specular on ridges.
    #[arg(long)]
    pub sheen: Option<f32>,
}

#[derive(Args, Debug)]
pub struct PaletteArgs {
    /// Palette name, or omit to list them all.
    pub name: Option<String>,
}

pub async fn run(args: PaintArgs) -> Result<()> {
    match args.cmd {
        Some(PaintCmd::New(a)) => run_new(a),
        Some(PaintCmd::Show(a)) => run_show(a, false),
        Some(PaintCmd::Lint(a)) => run_show(a, true),
        Some(PaintCmd::From(a)) => run_from(a),
        Some(PaintCmd::Replay(a)) => run_replay(a),
        Some(PaintCmd::Export(a)) => run_export(a),
        Some(PaintCmd::Timelapse(a)) => run_timelapse(a),
        Some(PaintCmd::Palette(a)) => run_palette(a),
        None => match args.spec {
            Some(spec) => run_spec(SpecArgs { spec, out: args.out, size: args.size, report: args.report, planes: args.planes, critic: args.critic, families: args.families, crisp: args.crisp, strokes: args.strokes, style: args.style, define: args.define, haze: args.haze, stroke_length: args.stroke_length, stroke_width: args.stroke_width, bleed: args.bleed, opacity: args.opacity, pickup: args.pickup, impasto: args.impasto }).await,
            None => anyhow::bail!("give a PaintSpec (`plakat paint <SPEC>`) or a subcommand (new / show / lint / from / replay / palette)"),
        },
    }
}

const SCAFFOLD: &str = r#"{
  paint: {
    version: 1

    // WHAT to paint: a prose `subject:` (the model renders an armature) OR a `reference:` image path.
    subject: "an old fisherman in a yellow oilskin hauling a net, grey sea, overcast sky, full figure"
    negative: "blurry, deformed hands, extra limbs, faceless, back turned, low detail"

    // ARMATURE MODEL — the armature is the CEILING for the painting, so the model choice matters most:
    //   sdxl   : balanced default, wide style range
    //   sd35   : best anatomy / composition (hands, faces); low-mem mode fits 24GB Metal
    //   sana   : light + fast, painterly — good for cheap drafts while you tune the spec
    //   sd15   : fastest / lightest, lower fidelity (quick drafts)
    //   pixart : artistic  ·  pony : stylized  ·  flux : coherent (note: GGUF Flux is broken on Metal)
    // Tip: draft the composition with a fast model (sana/sd15), then switch to sdxl/sd35 for the final.
    model: sdxl
    steps: 40

    // MEDIUM: oil-direct | oil-indirect | gouache | watercolour | ink-wash | pen-ink | tempera
    medium: watercolour
    // PALETTE: zorn | split-primary | verdaccio | earth | limited-landscape | sumi
    palette: limited-landscape

    // STYLE: fidelity (tracks the armature — recognizable & detailed) | legible | impressionist
    style: fidelity
    // define = edge hardness (0..1).  haze = aerial depth recession (0..1; use 0 with fidelity).
    define: 0.7
    haze: 0

    surface: { size: "768x1024" }
    seed: 42
  }
}
"#;

fn run_new(a: NewArgs) -> Result<()> {
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&a.out, SCAFFOLD).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  scaffolded a PaintSpec → {}", style("✓").green(), a.out.display());
    Ok(())
}

/// Load a spec + its reference image, returning `(spec, reference, plan)`.
fn load_spec(path: &std::path::Path, size_override: Option<&str>, strokes_override: Option<usize>) -> Result<(crate::paint::spec::PaintSpec, Option<image::RgbImage>, crate::paint::PaintPlan)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut spec = crate::paint::spec::PaintSpec::parse(&text)?;
    if let Some(s) = size_override {
        spec.surface.get_or_insert_with(Default::default).size = Some(s.to_string());
    }
    if let Some(n) = strokes_override {
        spec.budget.get_or_insert_with(Default::default).strokes = Some(n);
    }
    // A reference IMAGE, if given (resolved relative to the spec's directory). Otherwise the armature is
    // constructed from the `subject:` prose (the model as art director).
    let img = match spec.reference.clone() {
        Some(reference) => {
            let ref_path = path.parent().map(|d| d.join(&reference)).unwrap_or_else(|| reference.clone().into());
            Some(image::open(&ref_path).with_context(|| format!("opening reference {}", ref_path.display()))?.to_rgb8())
        }
        None => None,
    };
    let (w, h) = img.as_ref().map(|i| i.dimensions()).or_else(|| spec.size()).unwrap_or((1024, 1280));
    let plan = crate::paint::spec::compile(&spec, w, h)?;
    Ok((spec, img, plan))
}

fn run_show(a: SpecFileArgs, lint_only: bool) -> Result<()> {
    let (_, _, plan) = load_spec(&a.spec, None, None)?;
    println!(
        "{}  {}  ·  medium {}  ·  palette {}  ·  {}×{}  ·  budget {}  ·  {} stages",
        style("◆").cyan(),
        a.spec.display(),
        plan.medium.name,
        plan.palette.name,
        plan.size.0,
        plan.size.1,
        plan.budget,
        plan.stages.len(),
    );
    for pass in &plan.passes {
        println!("    {:<14} brush {:>5.1}px   budget {}", pass.stage, pass.radius, pass.budget);
    }
    if lint_only {
        println!("{}  spec is valid", style("✓").green());
    }
    Ok(())
}

async fn run_spec(a: SpecArgs) -> Result<()> {
    let (spec, img, plan) = load_spec(&a.spec, a.size.as_deref(), a.strokes)?;
    let out = a.out.clone().unwrap_or_else(|| a.spec.with_extension("").with_extension("png"));

    // COMPOSITION: when the scene is decomposed into elements, render a single COHERENT armature with the
    // LAYERED pipeline (elements are plan constraints anchored into one diffusion — shared light/perspective, no
    // cut-and-paste), then paint THAT through the normal reference path below.
    let img = if let Some(comp) = spec.composition.clone().filter(|c| !c.elements.is_empty()) {
        Some(render_layered_armature(&spec, &plan, &comp).await?)
    } else {
        img
    };

    let mut params = PaintParams::new(plan.palette, plan.budget);
    params.passes = Some(plan.passes.clone());
    params.medium = plan.medium.name.to_string();
    params.seed = plan.seed;
    // Painter controls are SCENE-authoritative: the HJSON value wins when set, else the CLI default. Keeps the
    // spec the control surface (no code-constant tuning per picture).
    params.style = parse_style(spec.style.as_deref().unwrap_or(&a.style))?;
    params.define = spec.define.unwrap_or(a.define).clamp(0.0, 1.0);
    params.stroke_len = spec.stroke_length.unwrap_or(a.stroke_length).clamp(0.2, 4.0);
    params.stroke_width = spec.stroke_width.unwrap_or(a.stroke_width).clamp(0.3, 3.0);
    let haze = spec.haze.unwrap_or(a.haze).clamp(0.0, 1.0);
    // TECHNIQUE behaviour: each control defaults to the MEDIUM's characteristic value, overridable by the spec
    // then the CLI — so watercolour bleeds and glows, oil is opaque and dirty, pen-ink is crisp, out of the box.
    params.bleed = spec.bleed.or(a.bleed).unwrap_or(plan.medium.bleed).clamp(0.0, 1.0);
    params.opacity = spec.opacity.or(a.opacity).unwrap_or(plan.medium.body).clamp(0.1, 1.0);
    params.impasto = spec.impasto.or(a.impasto).unwrap_or(plan.medium.impasto).clamp(0.0, 1.0);
    params.brush.k_pickup = spec.pickup.or(a.pickup).unwrap_or(plan.medium.pickup).clamp(0.0, 1.0);
    // MATERIAL physics: the paint's own behaviour, defaulting to the medium.
    params.chroma = spec.chroma.unwrap_or(plan.medium.chroma).clamp(0.3, 2.0);
    params.dry_shift = spec.dry_shift.unwrap_or(plan.medium.dry_shift).clamp(-0.4, 0.4);
    params.granulate = spec.granulate.unwrap_or(plan.medium.granulate).clamp(0.0, 1.0);
    params.sheen = spec.sheen.unwrap_or(plan.medium.sheen).clamp(0.0, 1.0);
    params.lift = spec.lift.unwrap_or(plan.medium.lift).clamp(0.0, 1.0);
    // Paint from a LOW-RES ARMATURE (§1.1): coarsen the reference so the brush invents the surface instead of
    // tracing detail. Sized to the WORKING canvas (below).
    // Paint at a normalized WORKING resolution so the stroke budget (a style control, §9.1) gives a consistent
    // DENSITY regardless of the output size; the score then replays at the requested output size (§G4). This is
    // what stops a big canvas reading as sparse scribble.
    let work = {
        let longest = plan.size.0.max(plan.size.1);
        // Paint near-native so the fine DETAIL passes can resolve real features (a face, a net, windows). The
        // old 640 cap forced an upscale that blurred the surface; the auto budget scales with area, so density
        // stays consistent without downscaling. Still capped so a huge output doesn't run unbounded on CPU.
        let work_max = 1024u32;
        if longest > work_max {
            let s = work_max as f32 / longest as f32;
            ((plan.size.0 as f32 * s).round() as u32, (plan.size.1 as f32 * s).round() as u32)
        } else {
            plan.size
        }
    };
    // No extra pre-coarsening: the working-size downscale below already reduces the reference (like the proven
    // P0 from-image path), and the per-layer blur removes finer detail. Pre-coarsening ON TOP double-smoothed
    // into mush.
    params.armature_side = None;
    // Surface-white media reserve their whites (paper shows through); density media build value by hatch marks.
    use crate::paint::medium::{MarkModel, WhiteSource};
    if plan.medium.white_source == WhiteSource::Surface {
        params.reserve = Some(0.72);
    }
    params.density = plan.medium.mark_model == MarkModel::Density;

    // Get the reference to paint from, and depth for the merge, one of two ways:
    //   • a supplied `reference:` image → optional CPU-proxy depth (merge only with --planes), or
    //   • the model as ART DIRECTOR: construct an armature from the `subject:` prose (SDXL render +
    //     Depth-Anything depth), and merge with that REAL depth. GPU.
    let (mut reference, depth, n_planes) = match img {
        Some(img) => {
            let r = if img.dimensions() == plan.size { img } else { image::imageops::resize(&img, plan.size.0, plan.size.1, image::imageops::FilterType::Triangle) };
            let depth = a.planes.filter(|&n| n > 1).map(|_| crate::paint::armature::depth_proxy(plan.size.0, plan.size.1));
            (r, depth, a.planes.unwrap_or(1))
        }
        None => {
            let subject = spec.subject.clone().filter(|s| !s.trim().is_empty()).context("PaintSpec: give a `reference:` image or a `subject:` (prose) to construct an armature")?;
            let device = crate::device::select("auto")?;
            // Armature quality is scene-authoritative: steps / model / negative come from the HJSON. More steps
            // = a clearer, more coherent armature (figure, net, sea), which is the biggest lever on legibility.
            let steps = spec.steps.unwrap_or(24).clamp(1, 100);
            let model = spec.model.clone().unwrap_or_else(|| "sdxl".into());
            let negative = spec.negative.clone().unwrap_or_default();
            println!("{}  armature: rendering \"{}\" ({}, {} steps) + estimating depth (Depth-Anything)…", style("◆").cyan(), subject, model, steps);
            let (colour, depth) = crate::paint::armature::construct(&subject, plan.size.0, plan.size.1, &model, steps, plan.seed, &negative, device).await?;
            // Merge only when asked (--planes): the recession merge is for multi-plane compositions; on a
            // single subject it would flatten the figure. Default paints the constructed reference directly.
            (colour, Some(depth), a.planes.unwrap_or(1))
        }
    };

    // The MERGE (P2): assign depth planes, apply atmospheric recession + palette unity, paint from the merged
    // reference — so the reference already carries plane structure, recession, and palette unity.
    if let (Some(depth), true) = (depth.as_ref(), n_planes > 1) {
        let colour: Vec<crate::paint::color::Srgb> = reference.pixels().map(|p| p.0).collect();
        let merged = crate::paint::armature::merged_reference(&colour, depth, &plan.palette, n_planes, 0.35);
        for (i, p) in reference.pixels_mut().enumerate() {
            p.0 = merged[i];
        }
        println!("{}  merge: {} depth planes · recession + palette unity", style("·").dim(), n_planes);
    }

    println!(
        "{}  paint {} → {}  ({}×{} · {} · {} · budget {} · {} stages)",
        style("◆").cyan(),
        a.spec.display(),
        out.display(),
        plan.size.0,
        plan.size.1,
        plan.medium.name,
        plan.palette.name,
        plan.budget,
        plan.stages.len(),
    );

    // Drop to the working resolution: scale the pass radii + min brush, resize the reference. Painting there
    // keeps the density right; the score replays to the output size afterward.
    if work != plan.size {
        let s = work.0 as f32 / plan.size.0 as f32;
        if let Some(passes) = params.passes.as_mut() {
            for pass in passes.iter_mut() {
                pass.radius *= s;
            }
        }
        params.min_brush *= s;
        reference = image::imageops::resize(&reference, work.0, work.1, image::imageops::FilterType::Triangle);
        println!("{}  working at {}×{} (budget density) → replay to {}×{}", style("·").dim(), work.0, work.1, plan.size.0, plan.size.1);
    }

    // AERIAL PERSPECTIVE: hand the painter the depth map (at working resolution) so the background recedes and
    // the foreground advances — foreground/background "layers". Only when a plane-merge isn't already applied.
    if haze > 0.0 && n_planes <= 1 {
        if let Some(d) = depth.as_ref() {
            params.depth = Some(resize_depth(d, plan.size, work));
            params.haze = haze;
            println!("{}  aerial perspective: depth recession (haze {:.2})", style("·").dim(), haze);
        }
    }

    // FAMILY-KEY (§5.5.5, opt-in): split light/shadow and enforce the invariant so the masses read solid.
    if a.families {
        let (w, h) = reference.dimensions();
        let colour: Vec<crate::paint::color::Srgb> = reference.pixels().map(|p| p.0).collect();
        let keyed = crate::paint::armature::key_families(&colour, w, h, 135.0, 40.0);
        for (i, p) in reference.pixels_mut().enumerate() {
            p.0 = keyed[i];
        }
        println!("{}  family split · invariant enforced (light/shadow masses)", style("·").dim());
    }
    // VALUE RE-KEY (§5.5.2): expand the reference's tonal range so the painting reads with real lights and
    // darks rather than collapsing toward the mid ground.
    {
        let colour: Vec<crate::paint::color::Srgb> = reference.pixels().map(|p| p.0).collect();
        let keyed = crate::paint::armature::value_key(&colour, 0.04, 0.96);
        for (i, p) in reference.pixels_mut().enumerate() {
            p.0 = keyed[i];
        }
    }
    // Ground: the proven path paints on a WHITE ground (like the coherent P0 portrait) — a mean-keyed toned
    // ground made every mid-tone stroke blend into it (flat grey). Opaque media get a faint warm imprimatura
    // (near-white) so light passages read as painted, not stark paper, without flattening the mid-tones.
    if plan.medium.opacity == crate::paint::medium::Opacity::Opaque {
        params.ground = Some([236, 230, 220]);
    }

    // FOCAL HARD-EDGE (§7.4): matte the subject and terminate strokes at its silhouette, so it stays crisp.
    if a.crisp {
        let device = crate::device::select("auto")?;
        let matter = crate::pipelines::matting::Matter::load(&device).await.context("loading U2Net for --crisp")?;
        let alpha = matter.matte(&reference).context("matting the subject")?;
        let mask: Vec<bool> = alpha.pixels().map(|p| p.0[0] > 128).collect();
        let total = mask.len().max(1);
        let covered = mask.iter().filter(|&&b| b).count();
        // Only use it if the matte actually found a subject (not everything / nothing).
        if covered > total / 50 && covered < total * 49 / 50 {
            params.region_mask = Some(mask);
            println!("{}  focal hard-edge: subject matted ({}% of the frame)", style("·").dim(), covered * 100 / total);
        } else {
            println!("{}  focal hard-edge: no clear subject matted — skipping", style("·").dim());
        }
    }

    let result = if a.critic {
        let device = crate::device::select("auto")?;
        let scorer = crate::pipelines::aesthetic::AestheticScorer::load(&device).await.context("loading the aesthetic critic")?;
        let tmp = tempfile::Builder::new().prefix("plakat-paint-critic-").tempdir().context("critic scratch dir")?;
        let tmp_path = tmp.path().join("pass.png");
        let score_fn = |img: &image::RgbImage| -> f32 {
            img.save(&tmp_path).ok();
            scorer.score_path(&tmp_path).unwrap_or(0.0)
        };
        println!("{}  critic: aesthetic pass accept/reject", style("◆").cyan());
        let r = painter::paint_critiqued(&reference, &params, &score_fn, 0.0);
        if !r.rejected.is_empty() {
            println!("{}  critic rolled back {} pass(es): {}", style("·").dim(), r.rejected.len(), r.rejected.join(", "));
        }
        r
    } else {
        painter::paint_from_image(&reference, &params)
    };
    // Output at the requested size — replay the score up from the working canvas (resolution-independent).
    let image_out = if work != plan.size { result.score.replay(plan.size.0, plan.size.1).context("replaying to output size")?.to_image_finished(&result.score.header.finish()) } else { result.canvas.to_image_finished(&result.score.header.finish()) };
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    image_out.save(&out).with_context(|| format!("writing {}", out.display()))?;
    let score_path = out.with_extension("strokes");
    std::fs::write(&score_path, result.score.to_text()).with_context(|| format!("writing {}", score_path.display()))?;
    // Sidecar recipe (§11.5).
    let sidecar = out.with_extension("json");
    let subject = spec.subject.clone().unwrap_or_default();
    let recipe = format!(
        "{{\n  \"medium\": \"{}\",\n  \"palette\": \"{}\",\n  \"size\": \"{}x{}\",\n  \"budget\": {},\n  \"strokes\": {},\n  \"seed\": {},\n  \"subject\": {}\n}}\n",
        plan.medium.name, plan.palette.name, plan.size.0, plan.size.1, plan.budget, result.strokes, plan.seed, serde_json::to_string(&subject).unwrap_or_else(|_| "\"\"".into()),
    );
    std::fs::write(&sidecar, recipe).ok();

    println!("{}  {} strokes → {}  ·  score → {}  ·  recipe → {}", style("✓").green(), result.strokes, out.display(), score_path.display(), sidecar.display());
    if a.report {
        let tr = painter::traceability(&image_out, &reference);
        println!("{}  traceability {:.3}", style("·").dim(), tr);
    }
    Ok(())
}

/// Build a LAYERED plan (HJSON) from the composition elements: the first element becomes the backdrop
/// (farthest), the rest become anchored layers placed by their `mask` hint and ordered by depth. The layered
/// pipeline anchors each as a low-frequency constraint inside one diffusion, so the scene is co-rendered
/// (shared light/perspective) — the opposite of cut-and-paste.
fn build_layer_plan_hjson(scene: &str, comp: &crate::paint::spec::CompositionSpec, w: u32, h: u32) -> String {
    let place_of = |mask: &str| -> &'static str {
        match mask.trim().to_ascii_lowercase().as_str() {
            "top" | "upper" | "sky" => "center-top",
            "bottom" | "lower" | "foreground" | "ground" => "center-bottom",
            "left" => "center-left",
            "right" => "center-right",
            _ => "center",
        }
    };
    let q = |s: &str| serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    let mut o = String::new();
    o.push_str("{\n");
    o.push_str(&format!("  size: \"{}x{}\"\n", w, h));
    o.push_str(&format!("  prompt: {}\n", q(scene)));
    let mut layers = comp.elements.iter();
    // First element = backdrop (the farthest plane); its weight/window set the backdrop anchor strength.
    if let Some(bg) = layers.next() {
        let bp = bg.subject.clone().unwrap_or_default();
        o.push_str(&format!("  backdrop: {{ prompt: {}", q(&bp)));
        if let Some(wt) = bg.weight {
            o.push_str(&format!(", weight: {:.3}", wt));
        }
        if let Some(wn) = bg.window {
            o.push_str(&format!(", window: {:.3}", wn));
        }
        o.push_str(" }\n");
    }
    o.push_str("  layers: [\n");
    let rest: Vec<_> = layers.collect();
    let n = rest.len().max(1);
    for (i, el) in rest.iter().enumerate() {
        let id = el.name.clone().unwrap_or_else(|| format!("layer-{i}"));
        let prompt = el.subject.clone().unwrap_or_default();
        // Placement: explicit `place` wins, else map the coarse `mask` hint.
        let place = el.place.clone().unwrap_or_else(|| place_of(el.mask.as_deref().unwrap_or("center")).to_string());
        // Nearer elements come later in the list → smaller depth (0 = nearest); an explicit depth overrides.
        let depth = el.depth.unwrap_or(0.75 * (1.0 - (i as f32 + 1.0) / (n as f32 + 1.0)));
        o.push_str(&format!("    {{ id: {}, prompt: {}, place: {}, depth: {:.3}", q(&id), q(&prompt), q(&place), depth));
        if let Some(sz) = el.size.as_deref() {
            o.push_str(&format!(", size: {}", q(sz)));
        }
        if let Some(wt) = el.weight {
            o.push_str(&format!(", weight: {:.3}", wt));
        }
        if let Some(wn) = el.window {
            o.push_str(&format!(", window: {:.3}", wn));
        }
        o.push_str(" }\n");
    }
    o.push_str("  ]\n}\n");
    o
}

/// Render a single COHERENT armature for a composition via the LAYERED pipeline (plan → drafts → one anchored
/// diffusion). Returns the finished image to be PAINTED. This replaces cut-and-paste compositing: no draft
/// pixel reaches the output; the elements are co-rendered into one scene.
async fn render_layered_armature(spec: &crate::paint::spec::PaintSpec, plan: &crate::paint::PaintPlan, comp: &crate::paint::spec::CompositionSpec) -> Result<image::RgbImage> {
    let (ow, oh) = plan.size;
    let scene = spec.subject.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| {
        comp.elements.iter().filter_map(|e| e.subject.clone()).collect::<Vec<_>>().join(", ")
    });
    let plan_hjson = build_layer_plan_hjson(&scene, comp, ow, oh);
    let tmp = tempfile::Builder::new().prefix("plakat-paint-layered-").tempdir().context("layered scratch dir")?;
    let plan_path = tmp.path().join("plan.hjson");
    std::fs::write(&plan_path, &plan_hjson).context("writing the layered plan")?;
    let armature_path = tmp.path().join("armature.png");
    let model = spec.model.clone().unwrap_or_else(|| "sdxl".into());
    let steps = spec.steps.unwrap_or(36).clamp(1, 100);
    println!(
        "{}  layered armature: {} elements → one coherent render ({}, {} steps · no cut-and-paste)…",
        style("◆").cyan(), comp.elements.len(), model, steps,
    );
    crate::api::Layered::from_plan(&model, &plan_path, &armature_path)
        .size(format!("{}x{}", ow, oh))
        .steps(steps)
        .seed(plan.seed)
        .run()
        .await
        .context("layered armature render")?;
    let img = image::open(&armature_path).context("opening the layered armature")?.to_rgb8();
    println!("{}  layered armature ready → painting", style("·").dim());
    Ok(img)
}


fn run_replay(a: ReplayArgs) -> Result<()> {
    let text = std::fs::read_to_string(&a.score).with_context(|| format!("reading {}", a.score.display()))?;
    let score = crate::paint::score::StrokeScore::parse(&text).context("parsing the stroke score")?;
    let (w, h) = match a.size.as_deref() {
        Some(s) => {
            let (ws, hs) = s.split_once(['x', 'X']).context("--size must be WxH")?;
            (ws.trim().parse().context("bad width")?, hs.trim().parse().context("bad height")?)
        }
        None => (score.header.width, score.header.height),
    };
    println!(
        "{}  replay {} → {}  ({}×{} · {} strokes · palette {} · medium {})",
        style("◆").cyan(),
        a.score.display(),
        a.out.display(),
        w,
        h,
        score.strokes.len(),
        score.header.palette,
        score.header.medium,
    );
    let canvas = match a.until {
        Some(n) => score.replay_filtered(w, h, |r| r.id <= n).context("replaying the score")?,
        None => score.replay(w, h).context("replaying the score")?,
    };
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    canvas.to_image_finished(&score.header.finish()).save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  rendered → {}", style("✓").green(), a.out.display());
    Ok(())
}

fn run_from(a: FromArgs) -> Result<()> {
    let palette = Palette::by_name(&a.palette)
        .with_context(|| format!("unknown palette {:?} — try: {}", a.palette, palette::ALL.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?;
    let img = image::open(&a.input).with_context(|| format!("opening {}", a.input.display()))?.to_rgb8();
    let (w, h) = img.dimensions();

    // Brush sizes: derived from the image if not given — a coarse block-in down to a fine restatement.
    let brush_sizes = a.brush.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| {
        let coarse = (w.max(h) as f32 / 18.0).max(a.min_brush * 2.0);
        vec![coarse, coarse * 0.5, coarse * 0.25]
    });

    let mut params = PaintParams::new(palette, a.budget);
    params.brush_sizes = brush_sizes.clone();
    params.min_brush = a.min_brush;
    params.seed = a.seed;
    params.style = parse_style(&a.style)?;
    params.define = a.define.clamp(0.0, 1.0);
    params.stroke_len = a.stroke_length.clamp(0.2, 4.0);
    params.stroke_width = a.stroke_width.clamp(0.3, 3.0);
    if let Some(b) = a.bleed {
        params.bleed = b.clamp(0.0, 1.0);
    }
    if let Some(o) = a.opacity {
        params.opacity = o.clamp(0.1, 1.0);
    }
    if let Some(pk) = a.pickup {
        params.brush.k_pickup = pk.clamp(0.0, 1.0);
    }
    if let Some(im) = a.impasto {
        params.impasto = im.clamp(0.0, 1.0);
    }
    if let Some(ch) = a.chroma {
        params.chroma = ch.clamp(0.3, 2.0);
    }
    if let Some(ds) = a.dry_shift {
        params.dry_shift = ds.clamp(-0.4, 0.4);
    }
    if let Some(gr) = a.granulate {
        params.granulate = gr.clamp(0.0, 1.0);
    }
    if let Some(sh) = a.sheen {
        params.sheen = sh.clamp(0.0, 1.0);
    }
    if a.haze > 0.0 {
        // A CPU depth proxy (central + low = near) so aerial perspective can be exercised without a depth model.
        params.depth = Some(crate::paint::armature::depth_proxy(w, h));
        params.haze = a.haze.clamp(0.0, 1.0);
    }

    println!(
        "{}  paint from {} → {}  ({}×{} px · palette {} · budget {} · brushes {})",
        style("◆").cyan(),
        a.input.display(),
        a.out.display(),
        w,
        h,
        palette.name,
        a.budget,
        brush_sizes.iter().map(|r| format!("{r:.0}")).collect::<Vec<_>>().join("→"),
    );

    let result = painter::paint_from_image(&img, &params);
    let out = result.canvas.to_image_finished(&result.score.header.finish());
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;

    // The canonical artifact: the replayable stroke score, next to the image.
    let score_path = a.out.with_extension("strokes");
    std::fs::write(&score_path, result.score.to_text()).with_context(|| format!("writing {}", score_path.display()))?;

    println!("{}  {} strokes laid → {}  ·  score → {}", style("✓").green(), result.strokes, a.out.display(), score_path.display());
    if a.report {
        let tr = painter::traceability(&out, &img);
        println!("{}  traceability {:.3} (→1 = traced/filter; a painting keeps structure but invents surface)", style("·").dim(), tr);
    }
    Ok(())
}

fn load_score(path: &std::path::Path, size: Option<&str>) -> Result<(crate::paint::score::StrokeScore, u32, u32)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let score = crate::paint::score::StrokeScore::parse(&text).context("parsing the stroke score")?;
    let (w, h) = match size {
        Some(s) => {
            let (ws, hs) = s.split_once(['x', 'X']).context("--size must be WxH")?;
            (ws.trim().parse().context("bad width")?, hs.trim().parse().context("bad height")?)
        }
        None => (score.header.width, score.header.height),
    };
    Ok((score, w, h))
}

fn run_export(a: ExportArgs) -> Result<()> {
    let (score, w, h) = load_score(&a.score, a.size.as_deref())?;
    match a.target.as_str() {
        "height" => {
            let canvas = score.replay(w, h).context("replaying for height")?;
            let out = if a.out.extension().is_some() { a.out.clone() } else { a.out.with_extension("png") };
            if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(p).ok();
            }
            canvas.height_image().save(&out).with_context(|| format!("writing {}", out.display()))?;
            println!("{}  impasto height (16-bit) → {}", style("✓").green(), out.display());
        }
        "separations" | "stages" => {
            std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
            let stages = score.stages();
            let cumulative = a.target == "stages";
            println!("{}  {} {} → {}/", style("◆").cyan(), stages.len(), a.target, a.out.display());
            for (i, stage) in stages.iter().enumerate() {
                // separations: this stage alone; stages: the painting through the end of this stage.
                let upto = i; // index in the stage order
                let canvas = score.replay_filtered(w, h, |r| {
                    let ri = stages.iter().position(|s| s == &r.stage).unwrap_or(usize::MAX);
                    if cumulative {
                        ri <= upto
                    } else {
                        r.stage == *stage
                    }
                })?;
                let safe: String = stage.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' }).collect();
                let path = a.out.join(format!("{:02}-{}.png", i + 1, safe));
                canvas.to_image().save(&path).with_context(|| format!("writing {}", path.display()))?;
                println!("    {}", path.display());
            }
        }
        other => anyhow::bail!("unknown export target {other:?}"),
    }
    Ok(())
}

fn run_timelapse(a: TimelapseArgs) -> Result<()> {
    let (score, w, h) = load_score(&a.score, a.size.as_deref())?;
    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let frames = score.replay_frames(w, h, a.every).context("replaying frames")?;
    println!("{}  timelapse {} → {}/ ({} frames, every {} strokes)", style("◆").cyan(), a.score.display(), a.out.display(), frames.len(), a.every);
    for (i, canvas) in frames.iter().enumerate() {
        let path = a.out.join(format!("frame-{i:04}.png"));
        canvas.to_image().save(&path).with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{}  {} frames → {}/  (assemble with: ffmpeg -i {}/frame-%04d.png out.mp4)", style("✓").green(), frames.len(), a.out.display(), a.out.display());
    Ok(())
}

fn run_palette(a: PaletteArgs) -> Result<()> {
    let show = |p: &Palette| {
        println!("{} {}", style("◆").cyan(), style(p.name).bold());
        for pig in p.pigments {
            let c = pig.masstone;
            println!("    {:<20} #{:02x}{:02x}{:02x}  rgb({}, {}, {})", pig.name, c[0], c[1], c[2], c[0], c[1], c[2]);
        }
    };
    match a.name {
        Some(name) => {
            let p = Palette::by_name(&name).with_context(|| format!("unknown palette {name:?}"))?;
            show(&p);
        }
        None => {
            for p in palette::ALL {
                show(p);
            }
        }
    }
    Ok(())
}
