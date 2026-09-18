//! `plakat layers` — plan-guided layered generation (RFC LAYERED-1). P1 offline subcommands: `new`
//! (scaffold a plan), `lint` (validate + size-class report), `show` (resolve + optionally draw the boxes).
//! The GPU stages (`draft`/`guide`/`render`) land in later P1 slices.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use image::{Rgb, RgbImage};
use std::path::PathBuf;

use crate::layered::lint::{self, Class, Severity};
use crate::layered::plan::{self, Layer, LayerPlan};

#[derive(Args, Debug)]
pub struct LayersArgs {
    #[command(subcommand)]
    pub cmd: LayersCmd,
}

#[derive(Subcommand, Debug)]
pub enum LayersCmd {
    /// Scaffold a new layer plan (a partial `LayerPlan` HJSON to edit). A PROMPT seeds the finish prompt.
    New(NewArgs),
    /// Validate a plan — schema, lint rules, and the size-class report for the finish model. Exits non-zero
    /// on any error so it can gate a render run.
    Lint(LintArgs),
    /// Print what a plan resolves to (per-layer box, depth, class). `--boxes out.png` draws the boxes,
    /// coloured by class, on a blank canvas.
    Show(ShowArgs),
    /// Render (or refresh from cache) the per-layer drafts + the backdrop → PNGs in a directory (S1).
    /// Each layer is rendered ALONE; drafts are content-cached, so `--only` re-renders just those.
    Draft(DraftArgs),
    /// Compose the guide from a draft directory (S2): matte each anchored subject, luma-normalise, lay
    /// back-to-front → `__guide.png`, and build the anchor maps → `__weight.png` / `__window.png`.
    Guide(GuideArgs),
    /// The full layered render (S1→S2→S3): draft → guide → one anchored finish trajectory → an image.
    /// Bring-up is the SD family (SD 1.5 / SDXL).
    Render(RenderArgs),
    /// Compare two images (e.g. the guide vs the finish) at the plan's low-frequency band — whole-canvas and
    /// per-layer-box agreement (MAE + correlation), with an optional difference heatmap.
    Diff(DiffArgs),
    /// Score a PLAIN vs a LAYERED render of a plan (layout / placement / semantic) and report the delta +
    /// a go/no-go verdict (P2). Measurement only — the renders are produced separately.
    Eval(EvalArgs),
    /// Aggregate `eval` over a directory of cases (each a subdir with `plan.hjson` + `guide.png` +
    /// `plain.png` + `layered.png`) into one corpus go/no-go (P2).
    Sweep(SweepArgs),
    /// Check (OWL-ViT) that each anchored/hinted layer's subject is present in its box (S4). Exits non-zero
    /// on any miss, so it can gate a repair. Detection only — no diffusion.
    Verify(VerifyArgs),
    /// Re-assert failing layers with a masked img2img pass (S4). `--layers` names the targets, or `--auto`
    /// verifies first and repairs whatever missed. Renders.
    Repair(RepairArgs),
    /// Regenerate every lifted-class (tiny) subject at a workable resolution and composite it back (S5).
    /// Renders.
    Lift(LiftArgs),
    /// Decompose a prose description into a layer plan HJSON via an LLM (P4): global look + backdrop +
    /// independent subject layers with placements. Edit + lint before rendering.
    Plan(PlanArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Optional finish prompt to seed the scaffold.
    pub prompt: Option<String>,
    /// Output plan file (`.hjson`).
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Output size `WxH`. Default 1216x832.
    #[arg(long, default_value = "1216x832")]
    pub size: String,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// Finish model (drives the size-class report + geometry). Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    #[arg(long)]
    pub size: Option<String>,
    /// Draw the layer boxes (coloured by class, laid back-to-front by depth) → this PNG.
    #[arg(long)]
    pub boxes: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct DraftArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// Directory to write the drafts into (`__backdrop.png` + `<id>.png`).
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Draft model (overrides the plan's `draft.model`; default sdxl).
    #[arg(long)]
    pub draft_model: Option<String>,
    /// Draft steps (a fast preset welcome). Default 8.
    #[arg(long, default_value_t = 8)]
    pub steps: usize,
    /// Only (re)render these layer ids (comma-separated), plus the backdrop.
    #[arg(long)]
    pub only: Option<String>,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct GuideArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// The draft directory produced by `plakat layers draft` (`__backdrop.png` + `<id>.png`).
    #[arg(long)]
    pub drafts: PathBuf,
    /// Directory to write the guide + maps into (`__guide.png`, `__weight.png`, `__window.png`).
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Finish model (drives the size classes + latent geometry). Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// Output image path.
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Finish model (SD family for bring-up). Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Draft model (overrides the plan's `draft.model`; default sdxl).
    #[arg(long)]
    pub draft_model: Option<String>,
    /// Draft steps (a fast preset welcome). Default 8.
    #[arg(long, default_value_t = 8)]
    pub draft_steps: usize,
    /// Finish steps. Default 30.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,
    /// Finish CFG guidance. Default 7.0.
    #[arg(long, default_value_t = 7.0)]
    pub guidance: f64,
    /// Seed (used for the finish trajectory AND the guide noise). Default 0.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Window-close ramp `r`. Default 0.1.
    #[arg(long, default_value_t = 0.1)]
    pub ramp: f32,
    /// Also write the intermediate drafts + guide + anchor maps into this directory.
    #[arg(long)]
    pub keep: Option<PathBuf>,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct DiffArgs {
    /// The layer plan HJSON (drives the low-frequency band + the layer boxes).
    pub plan: PathBuf,
    /// First image (e.g. the guide `__guide.png`).
    #[arg(long)]
    pub a: PathBuf,
    /// Second image (e.g. the finish output).
    #[arg(long)]
    pub b: PathBuf,
    /// Model whose geometry sets the low-frequency band. Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Write a difference heatmap (layer boxes drawn) to this PNG.
    #[arg(long, short = 'o')]
    pub out: Option<PathBuf>,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
}

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// The guide image (the intended layout — both renders are scored against it).
    #[arg(long)]
    pub guide: PathBuf,
    /// The PLAIN (single-prompt) render.
    #[arg(long)]
    pub plain: PathBuf,
    /// The LAYERED render.
    #[arg(long)]
    pub layered: PathBuf,
    /// Model whose geometry drives the classes + low-frequency band. Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
    /// The overall-delta margin for the single-case verdict. Default 0.02.
    #[arg(long, default_value_t = 0.02)]
    pub margin: f32,
    /// Skip the CLIP semantic metric.
    #[arg(long)]
    pub no_clip: bool,
    /// Skip the U2Net placement metric.
    #[arg(long)]
    pub no_saliency: bool,
}

#[derive(Args, Debug)]
pub struct SweepArgs {
    /// A directory of cases: each subdir has `plan.hjson` + `guide.png` + `plain.png` + `layered.png`.
    pub dir: PathBuf,
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    #[arg(long)]
    pub size: Option<String>,
    /// The mean-overall-delta margin for the corpus go/no-go. Default 0.02.
    #[arg(long, default_value_t = 0.02)]
    pub margin: f32,
    #[arg(long)]
    pub no_clip: bool,
    #[arg(long)]
    pub no_saliency: bool,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// The finished image to check.
    #[arg(long)]
    pub image: PathBuf,
    /// Model whose geometry drives the size classes. Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output size override `WxH` (default: the plan's `size`).
    #[arg(long)]
    pub size: Option<String>,
    /// OWL-ViT detection score threshold. Default 0.1.
    #[arg(long, default_value_t = 0.1)]
    pub threshold: f32,
    /// Max detections per query. Default 8.
    #[arg(long, default_value_t = 8)]
    pub max_dets: usize,
}

#[derive(Args, Debug)]
pub struct RepairArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// The finished image to repair.
    #[arg(long)]
    pub image: PathBuf,
    /// Output image path.
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Model (SD family). Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    #[arg(long)]
    pub size: Option<String>,
    /// Comma-separated layer ids to repair. Omit with `--auto` to repair whatever verify flags.
    #[arg(long)]
    pub layers: Option<String>,
    /// Verify first (OWL-ViT) and repair the layers that missed.
    #[arg(long)]
    pub auto: bool,
    /// Inpaint strength `[0,1]`. Default 0.6.
    #[arg(long, default_value_t = 0.6)]
    pub strength: f32,
    #[arg(long, default_value_t = 24)]
    pub steps: usize,
    #[arg(long, default_value_t = 7.0)]
    pub guidance: f64,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 8)]
    pub feather: u32,
    /// OWL-ViT threshold for `--auto`. Default 0.1.
    #[arg(long, default_value_t = 0.1)]
    pub threshold: f32,
}

#[derive(Args, Debug)]
pub struct LiftArgs {
    /// The layer plan HJSON.
    pub plan: PathBuf,
    /// The finished image to lift small subjects into.
    #[arg(long)]
    pub image: PathBuf,
    /// Output image path.
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Model (SD family). Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    #[arg(long)]
    pub size: Option<String>,
    /// img2img strength for the regenerate. Default 0.7.
    #[arg(long, default_value_t = 0.7)]
    pub strength: f32,
    #[arg(long, default_value_t = 24)]
    pub steps: usize,
    #[arg(long, default_value_t = 7.0)]
    pub guidance: f64,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Short-side working resolution the crop is upscaled to. Default 384.
    #[arg(long, default_value_t = 384)]
    pub work: u32,
    #[arg(long, default_value_t = 6)]
    pub feather: u32,
}

#[derive(Args, Debug)]
pub struct PlanArgs {
    /// The prose description to decompose into a plan.
    pub prose: String,
    /// Output plan file (`.hjson`).
    #[arg(long, short = 'o')]
    pub out: PathBuf,
    /// Finish model whose geometry drives the size-class report. Default sdxl.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
    /// Output size `WxH`. Default 1216x832.
    #[arg(long, default_value = "1216x832")]
    pub size: String,
    /// The layout LLM alias (enhance provider stack). Default: the enhance default.
    #[arg(long)]
    pub provider: Option<String>,
    /// The draft model written into the plan. Default sdxl-lightning.
    #[arg(long, default_value = "sdxl-lightning")]
    pub draft_model: String,
    /// Seed for the LLM decode. Default 0.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub async fn run(args: LayersArgs, device: candle_core::Device) -> Result<()> {
    match args.cmd {
        LayersCmd::New(a) => run_new(a),
        LayersCmd::Lint(a) => run_lint(a),
        LayersCmd::Show(a) => run_show(a),
        LayersCmd::Draft(a) => run_draft(a, device).await,
        LayersCmd::Guide(a) => run_guide(a, device).await,
        LayersCmd::Render(a) => run_render(a, device).await,
        LayersCmd::Diff(a) => run_diff(a),
        LayersCmd::Eval(a) => run_eval(a, device).await,
        LayersCmd::Sweep(a) => run_sweep(a, device).await,
        LayersCmd::Verify(a) => run_verify(a, device).await,
        LayersCmd::Repair(a) => run_repair(a, device).await,
        LayersCmd::Lift(a) => run_lift(a, device).await,
        LayersCmd::Plan(a) => run_plan(a, device).await,
    }
}

async fn run_plan(a: PlanArgs, device: candle_core::Device) -> Result<()> {
    let (w, h) = plan::parse_size(Some(&a.size), (1216, 832));
    let provider = a.provider.clone().unwrap_or_else(|| crate::llm::DEFAULT_ALIAS.to_string());
    println!("{}  planning \"{}\" ({} · {}×{} px)…", style("◆").cyan(), a.prose.chars().take(60).collect::<String>(), provider, w, h);
    let hjson = crate::layered::planner::plan_prose(&a.prose, w, h, &a.draft_model, &provider, &device, a.seed).await?;

    // Lint the produced plan (report, but always write — the author edits before rendering).
    let p = plan::parse(&hjson)?;
    let geom = lint::geometry_for_model(&a.model);
    let issues = lint::lint(&p, &geom, w, h);
    let (errs, warns) = (issues.iter().filter(|i| i.severity == Severity::Error).count(), issues.iter().filter(|i| i.severity == Severity::Warn).count());

    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&a.out, &hjson).with_context(|| format!("writing {}", a.out.display()))?;

    println!("{} {}  ({} layer(s))", style("wrote").green(), a.out.display(), p.layers.len());
    for l in &p.layers {
        println!("    {} {:<14} {:<8} \"{}\"", style("·").dim(), l.id, lint::layer_class(l, &geom, w, h).label(), l.prompt.as_deref().unwrap_or(""));
    }
    if errs > 0 || warns > 0 {
        println!("  {} {} error(s), {} warning(s) — run {} {}", style("lint:").dim(), errs, warns, style("plakat layers lint").dim(), a.out.display());
    }
    println!("\n{}  edit if needed, then: {} {} -o out.png", style("→").dim(), style("plakat layers render").dim(), a.out.display());
    Ok(())
}

/// Load OWL-ViT and verify a plan against an image; returns the report (used by `verify` + `repair --auto`).
async fn verify_image(plan: &LayerPlan, geom: &crate::pipelines::noise_space::LatentGeometry, w: u32, h: u32, image: &std::path::Path, threshold: f32, max_dets: usize, device: &candle_core::Device) -> Result<crate::layered::verify::Report> {
    use crate::layered::verify::{self, Det};
    let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(device).await.context("loading OWL-ViT")?;
    let detect = |query: &str| -> Result<Vec<Det>> {
        let dets = owl.detect_all(image, query, threshold, max_dets)?;
        Ok(dets.into_iter().map(|d| Det { x0: d.x0, y0: d.y0, x1: d.x1, y1: d.y1, score: d.score }).collect())
    };
    verify::verify(plan, geom, w, h, &detect)
}

async fn run_verify(a: VerifyArgs, device: candle_core::Device) -> Result<()> {
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    println!("{}  verify {} against {} ({} · {}×{} px)", style("◆").cyan(), a.plan.display(), a.image.display(), a.model, w, h);
    let report = verify_image(&p, &geom, w, h, &a.image, a.threshold, a.max_dets, &device).await?;
    for v in &report.layers {
        if v.class == Class::Lifted || v.query.is_empty() {
            println!("    {} {:<14} {} (skipped)", style("·").dim(), v.id, style(v.class.label()).dim());
            continue;
        }
        let mark = if v.found { style("✓").green() } else { style("✗").red() };
        println!("    {} {:<14} {:<8} score {:.3}  iou {:.2}  \"{}\"", mark, v.id, v.class.label(), v.score, v.iou, v.query);
    }
    let verdict = if report.pass() {
        style(format!("PASS — {}/{} checked layer(s) present", report.checked, report.checked)).green().bold()
    } else {
        style(format!("FAIL — {} missing: {}", report.failures.len(), report.failures.join(", "))).red().bold()
    };
    println!("\n{}  {}", style("→").dim(), verdict);
    if !report.pass() {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_repair(a: RepairArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::repair::{repair, RepairOpts};
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    // Resolve the targets: explicit --layers, else --auto verify, else all anchored/hinted.
    let targets: Vec<String> = if let Some(list) = &a.layers {
        list.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else if a.auto {
        let report = verify_image(&p, &geom, w, h, &a.image, a.threshold, 8, &device).await?;
        println!("{}  auto-repair: verify flagged {} layer(s): {}", style("◆").cyan(), report.failures.len(), report.failures.join(", "));
        report.failures
    } else {
        anyhow::bail!("name the layers to repair with --layers <id,..>, or use --auto to repair whatever verify flags");
    };
    if targets.is_empty() {
        println!("{}  nothing to repair — copying through", style("·").dim());
    }
    let opts = RepairOpts {
        model: a.model.clone(),
        out: a.out.clone(),
        strength: a.strength,
        steps: a.steps,
        guidance: a.guidance,
        seed: a.seed,
        mask_feather: a.feather,
        device: crate::device::spec_of(&device).to_string(),
    };
    println!("{}  repairing {} layer(s) in {} → {}", style("◆").cyan(), targets.len(), a.image.display(), a.out.display());
    repair(&a.image, &p, &targets, &opts).await?;
    println!("{}  {}", style("✓ repaired").green(), a.out.display());
    Ok(())
}

async fn run_lift(a: LiftArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::lift::{lift, LiftOpts};
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    let n_lifted = p
        .layers
        .iter()
        .filter(|l| crate::layered::lint::layer_class(l, &geom, w, h) == Class::Lifted && !l.prompt.as_deref().map(str::trim).unwrap_or("").is_empty())
        .count();
    let opts = LiftOpts {
        model: a.model.clone(),
        out: a.out.clone(),
        strength: a.strength,
        steps: a.steps,
        guidance: a.guidance,
        seed: a.seed,
        work: a.work,
        feather: a.feather,
        device: crate::device::spec_of(&device).to_string(),
    };
    println!("{}  lifting {} lifted-class subject(s) in {} → {}", style("◆").cyan(), n_lifted, a.image.display(), a.out.display());
    lift(&a.image, &p, &geom, &opts).await?;
    println!("{}  {}", style("✓ lifted").green(), a.out.display());
    Ok(())
}

/// The metric models (loaded once; each optional). No diffusion — safe to run offline.
struct Scorers {
    matter: Option<crate::pipelines::matting::Matter>,
    clip: Option<(crate::pipelines::aesthetic::AestheticScorer, crate::pipelines::clip_adherence::ClipAdherence)>,
}

impl Scorers {
    async fn load(device: &candle_core::Device, saliency: bool, clip: bool) -> Result<Self> {
        let matter = if saliency { Some(crate::pipelines::matting::Matter::load(device).await.context("loading U2Net")?) } else { None };
        let clip = if clip {
            let aes = crate::pipelines::aesthetic::AestheticScorer::load(device).await.context("loading the CLIP image tower")?;
            let txt = crate::pipelines::clip_adherence::ClipAdherence::load(device).await.context("loading the CLIP text tower")?;
            Some((aes, txt))
        } else {
            None
        };
        Ok(Self { matter, clip })
    }

    fn score(&self, plan: &LayerPlan, geom: &crate::pipelines::noise_space::LatentGeometry, w: u32, h: u32, img: &RgbImage, guide: &RgbImage) -> Result<crate::layered::eval::Score> {
        use crate::layered::eval;
        let layout = eval::layout_adherence(plan, img, guide, geom, w, h)?;
        let placement = match &self.matter {
            Some(m) => Some(eval::placement(plan, &m.matte(img)?, geom, w, h)),
            None => None,
        };
        let semantic = match &self.clip {
            Some((aes, cl)) => {
                let scorer = |crop: &RgbImage, text: &str| -> Result<f32> {
                    let tmp = tempfile::Builder::new().prefix("plakat-eval-").suffix(".png").tempfile().context("crop temp file")?;
                    crop.save(tmp.path()).context("saving crop")?;
                    let emb = aes.image_embedding(tmp.path())?;
                    cl.adherence(&emb, text)
                };
                Some(eval::semantic(plan, img, geom, w, h, &scorer)?)
            }
            None => None,
        };
        Ok(eval::combine(Some(layout), placement, semantic))
    }
}

fn fopt(o: Option<f32>) -> String {
    o.map(|v| format!("{v:.3}")).unwrap_or_else(|| "  –  ".into())
}
fn fdelta(o: Option<f32>) -> String {
    o.map(|v| format!("{v:+.3}")).unwrap_or_else(|| "  –  ".into())
}

async fn run_eval(a: EvalArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::eval;
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    let guide = crate::layered::diff::load_rgb(&a.guide)?;
    let plain = crate::layered::diff::load_rgb(&a.plain)?;
    let layered = crate::layered::diff::load_rgb(&a.layered)?;
    let scorers = Scorers::load(&device, !a.no_saliency, !a.no_clip).await?;

    let sp = scorers.score(&p, &geom, w, h, &plain, &guide).context("scoring the plain render")?;
    let sl = scorers.score(&p, &geom, w, h, &layered, &guide).context("scoring the layered render")?;
    let d = eval::delta(&sp, &sl);

    println!("{}  eval {} ({} · {}×{} px)", style("◆").cyan(), a.plan.display(), a.model, w, h);
    println!("  {:<9} layout {}  placement {}  semantic {}  → overall {}", "plain", fopt(sp.layout), fopt(sp.placement), fopt(sp.semantic), style(fopt(Some(sp.overall))).bold());
    println!("  {:<9} layout {}  placement {}  semantic {}  → overall {}", "layered", fopt(sl.layout), fopt(sl.placement), fopt(sl.semantic), style(fopt(Some(sl.overall))).bold());
    println!("  {:<9} layout {}  placement {}  semantic {}  → overall {}", "Δ", fdelta(d.layout), fdelta(d.placement), fdelta(d.semantic), style(fdelta(Some(d.overall))).bold());

    let (verdict, _, _) = eval::decide(&[d.overall], a.margin);
    let vstyle = match verdict {
        eval::Verdict::Go => style(verdict.label()).green().bold(),
        eval::Verdict::NoGo => style(verdict.label()).red().bold(),
        eval::Verdict::Inconclusive => style(verdict.label()).yellow().bold(),
    };
    println!("\n{}  {}  (overall Δ {:+.3}, margin {:.3})", style("→").dim(), vstyle, d.overall, a.margin);
    Ok(())
}

async fn run_sweep(a: SweepArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::eval;
    let scorers = Scorers::load(&device, !a.no_saliency, !a.no_clip).await?;
    let mut cases: Vec<std::path::PathBuf> = std::fs::read_dir(&a.dir)
        .with_context(|| format!("reading {}", a.dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    cases.sort();
    anyhow::ensure!(!cases.is_empty(), "no case subdirectories in {}", a.dir.display());

    println!("{}  sweep {} ({} case(s) · {} · margin {:.3})", style("◆").cyan(), a.dir.display(), cases.len(), a.model, a.margin);
    let mut deltas = Vec::new();
    for case in &cases {
        let name = case.file_name().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let plan_path = case.join("plan.hjson");
        let guide_path = ["guide.png", "__guide.png"].iter().map(|f| case.join(f)).find(|p| p.exists());
        let (plain_path, layered_path) = (case.join("plain.png"), case.join("layered.png"));
        let (Some(guide_path), true, true) = (guide_path, plain_path.exists() && plan_path.exists(), layered_path.exists()) else {
            println!("  {} {:<16} skipped (needs plan.hjson + guide.png + plain.png + layered.png)", style("·").dim(), name);
            continue;
        };
        let (p, geom, w, h) = load(&plan_path, &a.model, &a.size)?;
        let guide = crate::layered::diff::load_rgb(&guide_path)?;
        let sp = scorers.score(&p, &geom, w, h, &crate::layered::diff::load_rgb(&plain_path)?, &guide)?;
        let sl = scorers.score(&p, &geom, w, h, &crate::layered::diff::load_rgb(&layered_path)?, &guide)?;
        let d = eval::delta(&sp, &sl).overall;
        deltas.push(d);
        let mark = if d > 0.0 { style("✓").green() } else { style("✗").red() };
        println!("  {} {:<16} plain {:.3}  layered {:.3}  Δ {:+.3}", mark, name, sp.overall, sl.overall, d);
    }

    let (verdict, mean, win) = eval::decide(&deltas, a.margin);
    let vstyle = match verdict {
        eval::Verdict::Go => style(verdict.label()).green().bold(),
        eval::Verdict::NoGo => style(verdict.label()).red().bold(),
        eval::Verdict::Inconclusive => style(verdict.label()).yellow().bold(),
    };
    println!("\n{}  corpus: mean Δ {:+.3} · win-rate {:.0}% ({} case(s)) → {}", style("→").dim(), mean, win * 100.0, deltas.len(), vstyle);
    Ok(())
}

fn run_diff(a: DiffArgs) -> Result<()> {
    use crate::layered::diff;
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    let (ia, ib) = (diff::load_rgb(&a.a)?, diff::load_rgb(&a.b)?);
    let report = diff::compare(&p, &ia, &ib, &geom, w, h)?;
    println!(
        "{}  diff {} vs {} ({} · {}×{} px · low-pass {}px)",
        style("◆").cyan(),
        a.a.display(),
        a.b.display(),
        a.model,
        w,
        h,
        diff::lowpass_radius(&geom)
    );
    let o = &report.overall;
    println!("  {}  overall   MAE {:.4}  corr {:+.3}", style("·").dim(), o.mae, o.corr);
    for (id, s) in &report.layers {
        // Flag layers whose region drifted more than the canvas as a whole.
        let mark = if s.corr < 0.5 || s.mae > o.mae * 1.5 { style("⚠").yellow() } else { style("·").dim() };
        println!("  {}  layer {:<12} MAE {:.4}  corr {:+.3}", mark, id, s.mae, s.corr);
    }
    if let Some(out) = &a.out {
        // Colourise the heatmap boxes: draw each layer box over the grayscale diff.
        let mut rgb = image::RgbImage::new(w, h);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            let g = report.heatmap.get_pixel(x, y).0[0];
            *p = image::Rgb([g, g, g]);
        }
        for l in &p.layers {
            let bb = plan::layer_box(l);
            diff_box_outline(&mut rgb, bb, w, h, image::Rgb([255, 80, 80]));
        }
        rgb.save(out).with_context(|| format!("saving {}", out.display()))?;
        println!("  {} {}  (difference heatmap)", style("✓").green(), out.display());
    }
    Ok(())
}

/// Draw a 1px rectangle outline for a normalised box on an RGB image.
fn diff_box_outline(img: &mut image::RgbImage, b: [f32; 4], w: u32, h: u32, colour: image::Rgb<u8>) {
    let x0 = (b[0] * w as f32).round().clamp(0.0, w as f32 - 1.0) as u32;
    let y0 = (b[1] * h as f32).round().clamp(0.0, h as f32 - 1.0) as u32;
    let x1 = (b[2] * w as f32).round().clamp(0.0, w as f32 - 1.0) as u32;
    let y1 = (b[3] * h as f32).round().clamp(0.0, h as f32 - 1.0) as u32;
    for x in x0..=x1 {
        img.put_pixel(x, y0, colour);
        img.put_pixel(x, y1, colour);
    }
    for y in y0..=y1 {
        img.put_pixel(x0, y, colour);
        img.put_pixel(x1, y, colour);
    }
}

async fn run_render(a: RenderArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::render::{render, RenderOpts};
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    let draft_model = a.draft_model.clone().or_else(|| p.draft.model.clone()).unwrap_or_else(|| "sdxl".into());
    let opts = RenderOpts {
        model: a.model.clone(),
        out: a.out.clone(),
        draft_model,
        draft_steps: a.draft_steps,
        steps: a.steps,
        guidance: a.guidance,
        seed: a.seed,
        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
        ramp: a.ramp,
        keep: a.keep.clone(),
    };
    println!(
        "{}  layered render {} → {}  ({} finish · {} draft · {}×{} px · seed {})",
        style("◆").cyan(),
        a.plan.display(),
        a.out.display(),
        a.model,
        opts.draft_model,
        w,
        h,
        a.seed
    );
    render(&p, &geom, w, h, device, &opts).await?;
    println!("{}  {}", style("✓ finished").green(), a.out.display());
    Ok(())
}

/// Load a draft set (`__backdrop.png` + one `<id>.png` per plan layer) from a directory.
fn load_draft_set(dir: &std::path::Path, p: &LayerPlan) -> Result<crate::layered::draft::DraftSet> {
    let bpath = dir.join("__backdrop.png");
    let backdrop = image::open(&bpath).with_context(|| format!("loading backdrop draft {}", bpath.display()))?.to_rgb8();
    let mut layers = Vec::new();
    for l in &p.layers {
        let path = dir.join(format!("{}.png", sanitize(&l.id)));
        if path.exists() {
            let img = image::open(&path).with_context(|| format!("loading layer draft {}", path.display()))?.to_rgb8();
            layers.push((l.id.clone(), img));
        }
    }
    Ok(crate::layered::draft::DraftSet { backdrop, layers })
}

async fn run_guide(a: GuideArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::guide;
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    let drafts = load_draft_set(&a.drafts, &p)?;
    if drafts.layers.is_empty() {
        println!("{}  no layer drafts found in {} — run `plakat layers draft` first", style("⚠").yellow(), a.drafts.display());
    }
    println!("{}  composing guide for {} ({} · {}×{} px · {} draft layer(s))…", style("◆").cyan(), a.plan.display(), a.model, w, h, drafts.layers.len());

    // Load U2Net once; matte each anchored subject off its draft.
    let matter = crate::pipelines::matting::Matter::load(&device).await.context("loading the U2Net matter")?;
    let g = guide::build(&p, &drafts, &geom, w, h, &device, |img| matter.matte(img))?;

    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let gpath = a.out.join("__guide.png");
    g.canvas.save(&gpath).with_context(|| format!("saving {}", gpath.display()))?;
    println!("  {} {}  (composed guide, {} anchored subject(s))", style("✓").green(), gpath.display(), g.placed.len());

    // Map visualisations, upscaled from latent res to the canvas for viewing.
    for (name, map) in [("__weight.png", &g.weight), ("__window.png", &g.window_end)] {
        let gray = guide::map_to_gray(map)?;
        let up = image::imageops::resize(&gray, w, h, image::imageops::FilterType::Nearest);
        let path = a.out.join(name);
        up.save(&path).with_context(|| format!("saving {}", path.display()))?;
        println!("  {} {}", style("✓").green(), path.display());
    }

    println!("\n{}  guide + anchor maps → {}", style("→").dim(), a.out.display());
    Ok(())
}

async fn run_draft(a: DraftArgs, device: candle_core::Device) -> Result<()> {
    use crate::layered::draft::{render_all, DraftOpts};
    let (p, _geom, w, h) = load(&a.plan, "sdxl", &a.size)?;
    let model = a.draft_model.clone().or_else(|| p.draft.model.clone()).unwrap_or_else(|| "sdxl".into());
    // The draft family's native side (area target for layer drafts).
    let native_side = match crate::compile::classify_model(&model) {
        crate::compile::ModelFamily::Sd15 => 512,
        _ => 1024,
    };
    let only = a.only.as_ref().map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect::<Vec<_>>());
    let opts = DraftOpts {
        out_w: w,
        out_h: h,
        model: model.clone(),
        steps: a.steps,
        scheduler: crate::pipelines::scheduler::SchedulerKind::default(),
        native_side,
        only,
        device: crate::device::spec_of(&device).to_string(),
    };
    println!("{}  drafting {} on {} ({}×{} canvas, {} steps)…", style("◆").cyan(), a.plan.display(), model, w, h, a.steps);
    let set = render_all(&p, &opts).await?;
    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let bpath = a.out.join("__backdrop.png");
    set.backdrop.save(&bpath).with_context(|| format!("saving {}", bpath.display()))?;
    println!("  {} {}  ({}×{})", style("✓").green(), bpath.display(), set.backdrop.width(), set.backdrop.height());
    for (id, img) in &set.layers {
        let path = a.out.join(format!("{}.png", sanitize(id)));
        img.save(&path).with_context(|| format!("saving {}", path.display()))?;
        println!("  {} {}  ({}×{})", style("✓").green(), path.display(), img.width(), img.height());
    }
    println!("\n{}  {} draft(s) + backdrop → {}", style("→").dim(), set.layers.len(), a.out.display());
    Ok(())
}

pub(crate) fn sanitize(id: &str) -> String {
    id.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Build a scaffold plan (valid HJSON) for the given size + finish prompt.
fn scaffold(size: &str, prompt: &str) -> String {
    let prompt = prompt.replace('"', "'");
    format!(
        "{{\n  \
         version: 1\n  \
         size: \"{size}\"\n  \
         global: {{\n    \
         palette: \"TODO: colour palette (anchored into the drafts)\"\n    \
         light:   \"TODO: lighting (anchored into the drafts)\"\n    \
         medium:  \"TODO: oil painting / photo / engraving — FINISH only\"\n  }}\n  \
         // The FINISH prompt (the model renders this): keep the STYLE, drop per-object attribute detail.\n  \
         prompt: \"{prompt}\"\n  \
         backdrop: {{ prompt: \"TODO: the environment\", weight: 0.6, window: 0.25 }}\n  \
         layers: [\n    \
         // One INDEPENDENT subject per layer (interacting subjects stay in ONE layer). Give a box\n    \
         // [x0,y0,x1,y1] in [0,1], OR `place: \"center-left mid front\"`.\n    \
         {{ id: \"subject\", prompt: \"TODO: one subject, full detail\", box: [0.25, 0.2, 0.75, 0.9], depth: 0.3 }}\n  \
         ]\n  \
         draft: {{ model: \"sdxl-lightning\", seed: 7 }}\n}}\n",
        size = size,
        prompt = prompt,
    )
}

fn run_new(a: NewArgs) -> Result<()> {
    let prompt = a.prompt.unwrap_or_else(|| "a scene with a subject and a backdrop".into());
    let template = scaffold(&a.size, &prompt);
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&a.out, &template).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{} {}  — edit the TODOs, then: {} {}", style("wrote").green(), a.out.display(), style("plakat layers lint").dim(), a.out.display());
    Ok(())
}

/// Resolve (plan, geometry, output size) shared by lint + show.
fn load(planned: &PathBuf, model: &str, size_override: &Option<String>) -> Result<(LayerPlan, crate::pipelines::noise_space::LatentGeometry, u32, u32)> {
    let text = std::fs::read_to_string(planned).with_context(|| format!("reading {}", planned.display()))?;
    let p = plan::parse(&text)?;
    let geom = lint::geometry_for_model(model);
    let size = size_override.clone().or_else(|| p.size.clone());
    let (w, h) = plan::parse_size(size.as_deref(), (1216, 832));
    Ok((p, geom, w, h))
}

fn run_lint(a: LintArgs) -> Result<()> {
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    println!("{}  lint {} ({} · {}×{} px · {} layer(s))", style("◆").cyan(), a.plan.display(), a.model, w, h, p.layers.len());
    let issues = lint::lint(&p, &geom, w, h);
    let (mut errs, mut warns) = (0usize, 0usize);
    for i in &issues {
        match i.severity {
            Severity::Error => {
                errs += 1;
                println!("    {} {}", style("✗").red(), i.message);
            }
            Severity::Warn => {
                warns += 1;
                println!("    {} {}", style("⚠").yellow(), i.message);
            }
            Severity::Info => println!("    {} {}", style("·").dim(), style(&i.message).dim()),
        }
    }
    let verdict = if errs > 0 {
        style(format!("FAIL — {errs} error(s), {warns} warning(s)")).red().bold()
    } else if warns > 0 {
        style(format!("PASS with {warns} warning(s)")).yellow()
    } else {
        style("PASS — clean".into()).green().bold()
    };
    println!("\n{}  {}", style("→").dim(), verdict);
    anyhow::ensure!(lint::is_clean(&issues), "layers lint failed: {errs} error(s)");
    Ok(())
}

fn run_show(a: ShowArgs) -> Result<()> {
    let (p, geom, w, h) = load(&a.plan, &a.model, &a.size)?;
    println!("{}  {} · finish model {} · {}×{} px", style("◆").cyan(), a.plan.display(), a.model, w, h);
    if let Some(pr) = &p.prompt {
        println!("  {} {}", style("finish:").dim(), pr);
    }
    if let Some(b) = &p.backdrop {
        println!(
            "  {} {}  (weight {:.2}, window {:.2})",
            style("backdrop:").dim(),
            b.prompt.as_deref().unwrap_or("—"),
            b.weight.unwrap_or(0.6),
            b.window.unwrap_or(0.25),
        );
    }
    // Layers back-to-front by depth.
    let mut order: Vec<&Layer> = p.layers.iter().collect();
    order.sort_by(|x, y| plan::layer_depth(y).partial_cmp(&plan::layer_depth(x)).unwrap_or(std::cmp::Ordering::Equal));
    for l in &order {
        let bx = plan::layer_box(l);
        let c = lint::layer_class(l, &geom, w, h);
        let window = l.window.map(|v| format!("{v:.2}")).unwrap_or_else(|| if c == Class::Anchored { "0.40".into() } else { "—".into() });
        println!(
            "  {:<14} {} depth {:.2}  weight {:.2}  window {}  box [{:.2},{:.2},{:.2},{:.2}]",
            style(&l.id).bold(),
            class_tag(c),
            plan::layer_depth(l),
            l.weight.unwrap_or(1.0),
            window,
            bx[0], bx[1], bx[2], bx[3],
        );
    }
    if let Some(out) = &a.boxes {
        draw_boxes(&p, &geom, w, h, out)?;
        println!("\n{} {}  (boxes, coloured by class)", style("wrote").green(), out.display());
    }
    Ok(())
}

fn class_tag(c: Class) -> console::StyledObject<&'static str> {
    match c {
        Class::Anchored => style("anchored").green(),
        Class::Hinted => style("hinted  ").yellow(),
        Class::Lifted => style("lifted  ").red(),
    }
}

fn class_color(c: Class) -> Rgb<u8> {
    match c {
        Class::Anchored => Rgb([46, 125, 50]),
        Class::Hinted => Rgb([230, 140, 0]),
        Class::Lifted => Rgb([200, 40, 40]),
    }
}

/// Draw each layer's box (coloured by class, back-to-front by depth) on a half-scale blank canvas.
fn draw_boxes(p: &LayerPlan, geom: &crate::pipelines::noise_space::LatentGeometry, w: u32, h: u32, out: &PathBuf) -> Result<()> {
    let cw = (w / 2).clamp(64, 2048);
    let ch = (h / 2).clamp(64, 2048);
    let mut img = RgbImage::from_pixel(cw, ch, Rgb([248, 248, 246]));

    let mut order: Vec<&Layer> = p.layers.iter().collect();
    order.sort_by(|x, y| plan::layer_depth(y).partial_cmp(&plan::layer_depth(x)).unwrap_or(std::cmp::Ordering::Equal));
    for l in &order {
        let b = plan::layer_box(l);
        let color = class_color(lint::layer_class(l, geom, w, h));
        let x0 = (b[0] * cw as f32).round() as i64;
        let y0 = (b[1] * ch as f32).round() as i64;
        let x1 = (b[2] * cw as f32).round() as i64;
        let y1 = (b[3] * ch as f32).round() as i64;
        rect_outline(&mut img, x0, y0, x1, y1, color, 2);
        // A filled class tag at the top-left corner of the box.
        fill_rect(&mut img, x0, y0, x0 + 10, y0 + 10, color);
    }
    img.save(out).with_context(|| format!("writing {}", out.display()))?;
    Ok(())
}

fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn fill_rect(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb<u8>) {
    for y in y0..y1 {
        for x in x0..x1 {
            put(img, x, y, c);
        }
    }
}

fn rect_outline(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb<u8>, t: i64) {
    for k in 0..t {
        for x in x0..=x1 {
            put(img, x, y0 + k, c);
            put(img, x, y1 - k, c);
        }
        for y in y0..=y1 {
            put(img, x0 + k, y, c);
            put(img, x1 - k, y, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_is_valid_lintable_hjson() {
        let s = scaffold("1216x832", "a rainy market \"square\"");
        let p = plan::parse(&s).expect("scaffold parses");
        assert_eq!(p.layers.len(), 1);
        assert_eq!(plan::parse_size(p.size.as_deref(), (0, 0)), (1216, 832));
        // A quote in the prompt was sanitised so the HJSON stays valid.
        assert!(p.prompt.as_deref().unwrap().contains("'square'"));
        // It lints without an Error (the scaffold's single box is valid).
        let g = lint::geometry_for_model("sdxl");
        assert!(lint::is_clean(&lint::lint(&p, &g, 1216, 832)), "scaffold lints clean");
    }
}
