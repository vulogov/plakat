//! `plakat comic` (RFC COMIC-1) — P1 slice: `new` (scaffold), `lint` (validate, no weights), `show` (the
//! resolved page/panel plan), `layout` (composite SUPPLIED panel images into a bordered page + sidecar,
//! no GPU). Scene generation + balloons land in later phases.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::comic::lint::{self, Level};
use crate::comic::{layout, page, ComicSpec};

#[derive(Args, Debug)]
pub struct ComicArgs {
    #[command(subcommand)]
    pub cmd: ComicCmd,
}

#[derive(Subcommand, Debug)]
pub enum ComicCmd {
    /// Scaffold a new comic spec (`.hjson`).
    New(NewArgs),
    /// Validate a spec — schema, vocabulary, cast cross-references. No weights.
    Lint(SpecArg),
    /// Show the resolved plan: page size, panel rects, reading order, cast.
    Show(SpecArg),
    /// Composite supplied panel images (`--panels <dir>`) into a bordered page + `panels.json`. **No GPU.**
    Layout(LayoutArgs),
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
pub struct LayoutArgs {
    pub spec: PathBuf,
    /// Output page PNG.
    #[arg(long)]
    pub out: PathBuf,
    /// A directory of panel images (sorted by name → panel order). Missing panels draw a placeholder.
    #[arg(long)]
    pub panels: Option<PathBuf>,
}

pub async fn run(args: ComicArgs) -> Result<()> {
    match args.cmd {
        ComicCmd::New(a) => run_new(a),
        ComicCmd::Lint(a) => run_lint(a),
        ComicCmd::Show(a) => run_show(a),
        ComicCmd::Layout(a) => run_layout(a),
    }
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let template = format!(
        "{{\n  schema: \"{schema}\"\n  page: {{ size: \"us-letter\", dpi: 300, gutter: 24, border: 6, bg: \"white\" }}\n  reading: \"ltr\"\n  layout: {{ rows: [[1,1],[1],[1,1,1]] }}\n\n  cast: [\n    {{ name: \"hero\", describe: \"a weary detective in a long coat\" }}\n  ]\n\n  panels: [\n    {{ scene: \"a rain-slick neon alley, wide establishing shot\", caption: \"Tuesday. 3 a.m.\" }}\n    {{ scene: \"the detective lights a cigarette\", chars: [\"hero\"], balloons: [ {{ by: \"hero\", say: \"Someone's been here.\", at: \"top-left\" }} ] }}\n    {{ scene: \"a shadow moves behind a dumpster\" }}\n    {{ scene: \"close on the detective's eyes, narrowed\", chars: [\"hero\"] }}\n    {{ scene: \"a hand reaches from the dark\" }}\n    {{ scene: \"the alley, empty now\", caption: \"Gone.\" }}\n  ]\n\n  model: \"sdxl\"  seed: 7  steps: 30\n}}\n",
        schema = crate::comic::SCHEMA_VERSION,
    );
    std::fs::write(&a.out, &template).with_context(|| format!("writing {}", a.out.display()))?;
    let spec = ComicSpec::from_hjson(&template).context("scaffold failed to parse")?;
    let errs = lint::lint(&spec).into_iter().filter(|f| f.level == Level::Error).count();
    println!("{} {}  (6-panel us-letter template)", style("wrote").green(), a.out.display());
    if errs > 0 {
        anyhow::bail!("scaffold has {errs} lint error(s) — this is a bug");
    }
    Ok(())
}

fn run_lint(a: SpecArg) -> Result<()> {
    let spec = ComicSpec::load(&a.spec)?;
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
    let spec = ComicSpec::load(&a.spec)?;
    let plan = layout::resolve(&spec);
    let b = |s: &str| style(s.to_string()).bold();
    println!("{}", b("plakat comic — resolved plan"));
    println!("  page       {}×{} px @ {} DPI · bg {:?}", plan.w, plan.h, plan.dpi, plan.bg);
    println!("  reading    {} · gutter {} · border {}", plan.reading, plan.gutter, plan.border);
    println!("  cast       {}", if spec.cast.is_empty() { "(none)".into() } else { spec.cast.iter().map(|c| c.name.clone()).collect::<Vec<_>>().join(", ") });
    println!("  panels     {}", plan.panels.len());
    for r in &plan.panels {
        let scene = spec.panels.get(r.panel).and_then(|p| p.scene.as_deref()).unwrap_or("(empty)");
        let short: String = scene.chars().take(48).collect();
        println!("    {} #{:<2} {:>4}×{:<4} at ({:>4},{:>4})  {}", style("·").cyan(), r.index, r.w, r.h, r.x, r.y, style(short).dim());
    }
    Ok(())
}

fn run_layout(a: LayoutArgs) -> Result<()> {
    let spec = ComicSpec::load(&a.spec)?;
    let plan = layout::resolve(&spec);
    // load supplied panel images (sorted by name → panel index).
    let mut imgs: Vec<Option<image::DynamicImage>> = vec![None; spec.panels.len().max(1)];
    if let Some(dir) = &a.panels {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()).as_deref(), Some("png" | "jpg" | "jpeg" | "webp")))
            .collect();
        files.sort();
        for (i, f) in files.iter().enumerate() {
            if i >= imgs.len() {
                break;
            }
            imgs[i] = image::open(f).ok();
        }
    }
    let pageimg = page::compose(&plan, &imgs);
    pageimg.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
    let side = a.out.with_extension("panels.json");
    std::fs::write(&side, page::panels_json(&plan)).with_context(|| format!("writing {}", side.display()))?;
    let filled = imgs.iter().filter(|o| o.is_some()).count();
    println!("{} {}  ({} panels, {}/{} filled · {}×{} px)", style("wrote").green(), a.out.display(), plan.panels.len(), filled, spec.panels.len(), plan.w, plan.h);
    println!("  {} {}", style("sidecar").cyan(), side.display());
    Ok(())
}
