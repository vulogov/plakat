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
    /// Also print the traceability.
    #[arg(long)]
    pub report: bool,
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
            Some(spec) => run_spec(SpecArgs { spec, out: args.out, size: args.size, report: args.report }),
            None => anyhow::bail!("give a PaintSpec (`plakat paint <SPEC>`) or a subcommand (new / show / lint / from / replay / palette)"),
        },
    }
}

const SCAFFOLD: &str = "{\n  paint: {\n    version: 1\n    // P1 paints from a reference image (the prose subject + armature land in P2).\n    reference: \"input.png\"\n    medium: oil-direct        // oil-direct | gouache\n    palette: zorn             // zorn | split-primary | verdaccio | earth | limited-landscape | sumi\n    surface: { size: \"1024x1280\" }\n    budget: { strokes: 1500 }\n    seed: 42\n  }\n}\n";

fn run_new(a: NewArgs) -> Result<()> {
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&a.out, SCAFFOLD).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{}  scaffolded a PaintSpec → {}", style("✓").green(), a.out.display());
    Ok(())
}

/// Load a spec + its reference image, returning `(spec, reference, plan)`.
fn load_spec(path: &std::path::Path, size_override: Option<&str>) -> Result<(crate::paint::spec::PaintSpec, image::RgbImage, crate::paint::PaintPlan)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut spec = crate::paint::spec::PaintSpec::parse(&text)?;
    if let Some(s) = size_override {
        spec.surface.get_or_insert_with(Default::default).size = Some(s.to_string());
    }
    let reference = spec.reference.clone().context("PaintSpec: `reference:` image is required in P1")?;
    // Resolve the reference relative to the spec's directory.
    let ref_path = path.parent().map(|d| d.join(&reference)).unwrap_or_else(|| reference.clone().into());
    let img = image::open(&ref_path).with_context(|| format!("opening reference {}", ref_path.display()))?.to_rgb8();
    let (w, h) = img.dimensions();
    let plan = crate::paint::spec::compile(&spec, w, h)?;
    Ok((spec, img, plan))
}

fn run_show(a: SpecFileArgs, lint_only: bool) -> Result<()> {
    let (_, _, plan) = load_spec(&a.spec, None)?;
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

fn run_spec(a: SpecArgs) -> Result<()> {
    let (spec, img, plan) = load_spec(&a.spec, a.size.as_deref())?;
    let out = a.out.clone().unwrap_or_else(|| a.spec.with_extension("").with_extension("png"));

    let mut params = PaintParams::new(plan.palette, plan.budget);
    params.passes = Some(plan.passes.clone());
    params.medium = plan.medium.name.to_string();
    params.seed = plan.seed;
    params.brush.k_pickup = plan.medium.pickup; // the medium's pickup drives the dirty brush
    // Surface-white media reserve their whites (paper shows through); density media build value by hatch marks.
    use crate::paint::medium::{MarkModel, WhiteSource};
    if plan.medium.white_source == WhiteSource::Surface {
        params.reserve = Some(0.72);
    }
    params.density = plan.medium.mark_model == MarkModel::Density;

    // The reference is resized to the plan's output size so the score's geometry is at output scale.
    let reference = if img.dimensions() == plan.size { img } else { image::imageops::resize(&img, plan.size.0, plan.size.1, image::imageops::FilterType::Triangle) };

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

    let result = painter::paint_from_image(&reference, &params);
    let image_out = result.canvas.to_image();
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
    let canvas = score.replay(w, h).context("replaying the score")?;
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    canvas.to_image().save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
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
    let out = result.canvas.to_image();
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
