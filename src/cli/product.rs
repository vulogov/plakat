//! `plakat product` (RFC PRODUCT-1) — P1 slice: `new` (scaffold), `lint` (validate, no weights), `show`
//! (the resolved plan), `render` (a supplied cutout → sweep + grounding → a finished packshot, no GPU).
//! Relight + subject-from-photo/prompt land in P2.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::product::compose::Bg;
use crate::product::ground::{ReflectionKind, ShadowKind};
use crate::product::lint::{self, Level};
use crate::product::{compose, render, ProductSpec};

#[derive(Args, Debug)]
pub struct ProductArgs {
    #[command(subcommand)]
    pub cmd: ProductCmd,
}

#[derive(Subcommand, Debug)]
pub enum ProductCmd {
    /// Scaffold a new product spec (`.hjson`).
    New(NewArgs),
    /// Validate a spec — schema, vocabulary, a subject source. No weights.
    Lint(SpecArg),
    /// Show the resolved plan: canvas, background, subject placement, grounding.
    Show(SpecArg),
    /// A supplied cutout → sweep + contact shadow + reflection → a finished packshot. **No GPU.**
    Render(RenderArgs),
}

#[derive(Args, Debug)]
pub struct SpecArg {
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct NewArgs {
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    pub spec: PathBuf,
    /// Output packshot PNG.
    #[arg(long)]
    pub out: PathBuf,
    /// Override `subject.image` with this cutout / photo (a `subject.photo`/`prompt` in the spec is matted
    /// / generated with a model).
    #[arg(long)]
    pub subject: Option<PathBuf>,
    /// Relight the subject to the `lighting` rig (IC-Light). On by default when the spec has a `lighting:`
    /// block; this forces it on even without one.
    #[arg(long)]
    pub relight: bool,
    /// Force relight off (keep the subject's own light) even when a `lighting:` block is present.
    #[arg(long)]
    pub no_relight: bool,
    /// Device for the model steps (matte / generate / relight).
    #[arg(long, default_value = "auto")]
    pub device: String,
}

pub async fn run(args: ProductArgs) -> Result<()> {
    match args.cmd {
        ProductCmd::New(a) => run_new(a),
        ProductCmd::Lint(a) => run_lint(a),
        ProductCmd::Show(a) => run_show(a),
        ProductCmd::Render(a) => run_render(a).await,
    }
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let template = format!(
        "{{\n  schema: \"{schema}\"\n\n  // A transparent cutout PNG is the weight-free path (kept pixel-exact).\n  subject: {{ image: \"product.png\", scale: 0.7, anchor: \"bottom\" }}\n\n  canvas: {{ size: \"square\", px: 1024, bg: \"grey-sweep\" }}\n  camera: {{ angle: \"eye\" }}\n  ground: {{ shadow: \"soft\", reflection: \"gloss\", softness: 0.5, falloff: 0.6 }}\n\n  // Relight (IC-Light) is P2 and opt-in; leave `lighting` out to keep the cutout's own light.\n  // lighting: {{ rig: \"three-point\", key_dir: \"top-left\" }}\n\n  seed: 7\n}}\n",
        schema = crate::product::SCHEMA_VERSION,
    );
    std::fs::write(&a.out, &template).with_context(|| format!("writing {}", a.out.display()))?;
    let spec = ProductSpec::from_hjson(&template).context("scaffold failed to parse")?;
    let _ = compose::resolve(&spec); // must resolve
    println!("{} {}  (grey-sweep packshot template — drop in a cutout at `subject.image`)", style("wrote").green(), a.out.display());
    Ok(())
}

fn run_lint(a: SpecArg) -> Result<()> {
    let spec = ProductSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    let (mut errs, mut warns) = (0, 0);
    for f in &findings {
        match f.level {
            Level::Error => {
                errs += 1;
                println!("{} {}: {}", style("error").red().bold(), style(&f.path).bold(), f.message);
            }
            Level::Warn => {
                warns += 1;
                println!("{} {}: {}", style("warn").yellow(), style(&f.path).bold(), f.message);
            }
        }
    }
    if errs == 0 {
        println!("{} {} ({} warning(s))", style("✓ lint ok").green(), a.spec.display(), warns);
        Ok(())
    } else {
        anyhow::bail!("{errs} error(s), {warns} warning(s)")
    }
}

fn run_show(a: SpecArg) -> Result<()> {
    let spec = ProductSpec::load(&a.spec)?;
    let plan = compose::resolve(&spec);
    let b = |s: &str| style(s.to_string()).bold();
    let shadow = match plan.shadow {
        ShadowKind::Soft => "soft",
        ShadowKind::Hard => "hard",
        ShadowKind::None => "none",
    };
    let reflection = match plan.reflection {
        ReflectionKind::Gloss => "gloss",
        ReflectionKind::Mirror => "mirror",
        ReflectionKind::None => "none",
    };
    let bg = match &plan.bg {
        Bg::Flat(c) => format!("flat {:?}", c.0),
        Bg::Gradient(a, b) => format!("gradient {:?}→{:?}", a.0, b.0),
    };
    println!("{}", b("plakat product — resolved plan"));
    println!("  canvas     {}×{} px · bg {}", plan.w, plan.h, bg);
    println!("  subject    scale {:.2} · anchor {} · ground_y {}", plan.subject_scale, if plan.anchor_bottom { "bottom" } else { "center" }, plan.ground_y);
    println!("  camera     {}", plan.camera_angle.as_deref().unwrap_or("eye"));
    println!("  ground     shadow {shadow} (key {:+.0}, soft {:.2}) · reflection {reflection} (falloff {:.2})", plan.key, plan.softness, plan.falloff);
    println!("  subject src {}", spec.subject_image().unwrap_or("(none — set subject.image)"));
    Ok(())
}

async fn run_render(a: RenderArgs) -> Result<()> {
    let spec = ProductSpec::load(&a.spec)?;
    let errs: Vec<_> = lint::lint(&spec).into_iter().filter(|f| f.level == Level::Error).collect();
    if !errs.is_empty() {
        for f in &errs {
            println!("{} {}: {}", style("error").red().bold(), style(&f.path).bold(), f.message);
        }
        anyhow::bail!("{} lint error(s) — fix before render", errs.len());
    }
    // relight: --no-relight forces off; --relight forces on; else a `lighting:` block opts in (RFC Q3).
    let relight = if a.no_relight { false } else { a.relight || spec.lighting.is_some() };
    let opts = render::RenderOpts { subject: a.subject.clone(), relight, device: Some(a.device.clone()) };
    let rep = render::render_spec(&spec, &a.out, &opts).await?;
    let mode = if rep.weight_free { "weight-free".to_string() } else { format!("subject: {}{}", rep.subject_source, if rep.relit { " · relit" } else { "" }) };
    println!("{} {}  ({}×{} px · {mode})", style("wrote").green(), rep.shot.display(), rep.w, rep.h);
    println!("  {} {}", style("sidecar").cyan(), rep.sidecar.display());
    Ok(())
}
