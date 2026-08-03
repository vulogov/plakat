//! `plakat bookart` — B/W book-ornament composition (RFC BOOKART-1).
//!
//! B0 first slice: `new` (scaffold a spec), `lint` (validate without weights), `show` (what a spec
//! resolves to — tier, symmetry, canvas, prompt). The finisher, geometry, procedural/diffusion render
//! tiers, scorecard, kit and manuscript subcommands land across later phases (ROADMAP_BOOKART_1). Fully
//! additive — nothing here touches existing behaviour.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::bookart::lint::{self, Level};
use crate::bookart::{compile, BookArtSpec};

#[derive(Args, Debug)]
pub struct BookartArgs {
    #[command(subcommand)]
    pub cmd: BookartCmd,
}

#[derive(Subcommand, Debug)]
pub enum BookartCmd {
    /// Scaffold a new bookart spec (a valid, partial `BookArtSpec` HJSON you then edit).
    New(NewArgs),
    /// Validate a bookart spec — schema, vocabulary, ranges, page, contradictions. No weights, no
    /// network. Exits non-zero on any error so it can gate CI.
    Lint(LintArgs),
    /// Show what a spec resolves to: origin/technique/motif, the render tier, symmetry, the print
    /// canvas (px @ DPI), the finisher chain, and the compiled prompt/negative.
    Show(ShowArgs),
    /// Finish a raw render (binarise → transparency) per a spec and score it (RFC §7/§9): chroma
    /// purity, alpha cleanliness, symmetry, ink coverage. `--out` writes the transparent PNG.
    Verify(VerifyArgs),
    /// Render an ornament to a transparent, page-sized PNG. **B3: the `procedural` tier** (border /
    /// corner / divider / fleuron / rosette — vector-native, no weights); diffusion/composite land in B4/B5.
    Render(RenderArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Output path for the new spec (`.hjson`).
    pub out: PathBuf,
    /// Illustration tradition (`russian`/`english`/`japanese`/`generic`/…).
    #[arg(long, default_value = "generic")]
    pub origin: String,
    /// Drawing technique (`line`/`woodcut`/`engraving`/…).
    #[arg(long, default_value = "line")]
    pub technique: String,
    /// Ornament type (`headpiece`/`tailpiece`/`divider`/`vignette`/…).
    #[arg(long = "type", default_value = "headpiece")]
    pub kind: String,
    /// Page size (`a4`/`a5`/`a6`/`letter`/…).
    #[arg(long, default_value = "a5")]
    pub page: String,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub spec: PathBuf,
    /// The render to finish + score (a raw diffusion/procedural render, or a finished PNG).
    #[arg(long)]
    pub image: PathBuf,
    /// Write the finished transparent PNG here.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Treat `--image` as already finished (score as-is; skip binarise + transparency).
    #[arg(long, default_value_t = false)]
    pub finished: bool,
    /// Apply the plan's symmetry (bilateral / radial:N) to the finished ornament (§6.3).
    #[arg(long, default_value_t = false)]
    pub symmetrize: bool,
    /// Place the ornament onto the exact page-size canvas at its layout rect (§6.4); `--out` is then
    /// page-sized with the DPI recorded.
    #[arg(long, default_value_t = false)]
    pub page: bool,
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    pub spec: PathBuf,
    /// Output PNG (transparent, page-sized).
    #[arg(long)]
    pub out: PathBuf,
}

pub async fn run(args: BookartArgs) -> Result<()> {
    match args.cmd {
        BookartCmd::New(a) => run_new(a),
        BookartCmd::Lint(a) => run_lint(a),
        BookartCmd::Show(a) => run_show(a),
        BookartCmd::Verify(a) => run_verify(a),
        BookartCmd::Render(a) => run_render(a),
    }
}

fn run_render(a: RenderArgs) -> Result<()> {
    use crate::bookart::{finish, geometry, procedural};
    let spec = BookArtSpec::load(&a.spec)?;
    let plan = compile::resolve(&spec);
    if plan.tier != "procedural" {
        anyhow::bail!(
            "`bookart render` supports the `procedural` tier only for now (this spec resolves to `{}`); the diffusion/composite tiers land in B4/B5. Use a geometric ornament (border/corner/divider/fleuron/rosette) or set `ornament.tier: procedural`.",
            plan.tier
        );
    }
    let tb = geometry::text_block(&plan.page, &spec);
    let layout = geometry::layout_for(&plan.ornament_kind, &tb);
    let r0 = layout.rects[0];
    let gray = procedural::generate(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h);
    let orn = geometry::symmetrize(&finish::finish_procedural(&gray, &plan), &plan.symmetry);
    let page = finish::canvas::place_on_canvas(&orn, &plan.page, &layout);
    finish::canvas::save_png_dpi(&page, &a.out, plan.page.dpi)?;
    println!(
        "{} {}  ({} × {} px @ {} DPI · {} · {} · {} piece(s))",
        style("wrote").green(),
        a.out.display(),
        page.width(),
        page.height(),
        plan.page.dpi,
        plan.ornament_kind,
        plan.symmetry,
        layout.rects.len()
    );
    Ok(())
}

fn run_new(a: NewArgs) -> Result<()> {
    let template = format!(
        "{{\n  schema: \"{schema}\"\n  origin: \"{origin}\"\n  technique: \"{technique}\"\n  motif: [\"firebird\", \"oak-leaf\"]\n\
         ink: {{ color: \"black\", weight: 0.6, transparency: \"luminance\" }}\n  page: {{ size: \"{page}\", dpi: 300, bleed_mm: 3 }}\n\
         transparent: true\n  output: {{ formats: [\"png\"], tint: \"black\" }}\n\n  ornament: {{\n    type: \"{kind}\"\n    symmetry: \"bilateral\"\n    tier: \"auto\"\n    prompt: \"a firebird among oak branches\"\n  }}\n}}\n",
        schema = crate::bookart::SCHEMA_VERSION,
        origin = a.origin,
        technique = a.technique,
        page = a.page,
        kind = a.kind,
    );
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    std::fs::write(&a.out, template).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{} {}", style("wrote").green(), a.out.display());
    // Lint the scaffold so the user starts from a clean bill.
    let spec = BookArtSpec::load(&a.out)?;
    print_findings(&lint::lint(&spec));
    Ok(())
}

fn run_lint(a: LintArgs) -> Result<()> {
    let spec = BookArtSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    print_findings(&findings);
    if lint::has_errors(&findings) {
        anyhow::bail!("lint failed with errors");
    }
    println!("{} {}", style("ok").green(), a.spec.display());
    Ok(())
}

fn run_show(a: ShowArgs) -> Result<()> {
    let spec = BookArtSpec::load(&a.spec)?;
    let p = compile::resolve(&spec);
    println!("{}  {}  (schema {})", style("bookart show").bold(), a.spec.display(), if p.schema_ok { "ok" } else { "mismatch" });
    println!("  {:14} {} × {}", style("origin/tech").dim(), p.origin, p.technique);
    println!("  {:14} {}", style("motif").dim(), if p.motif.is_empty() { "—".into() } else { p.motif.join(", ") });
    println!("  {:14} {}", style("ornament").dim(), p.ornament_kind);
    println!("  {:14} {}", style("render tier").dim(), p.tier);
    println!("  {:14} {}", style("symmetry").dim(), p.symmetry);
    println!(
        "  {:14} {} × {} px @ {} DPI  ({:.0}×{:.0} mm, bleed {:.0} mm, size {})",
        style("canvas").dim(),
        p.page.w_px,
        p.page.h_px,
        p.page.dpi,
        p.page.w_mm,
        p.page.h_mm,
        p.page.bleed_mm,
        p.page.size_name
    );
    println!(
        "  {:14} {} (mode {}, binariser {}, ink {} @ {:.2}, tint {})",
        style("finisher").dim(),
        if p.transparent { "transparent" } else { "opaque" },
        p.transparency_mode,
        p.binariser,
        p.ink_color,
        p.ink_weight,
        p.tint
    );
    println!("  {:14} {}", style("formats").dim(), p.formats.join(", "));
    if p.prompt.is_empty() {
        println!("  {:14} {}", style("prompt").dim(), style("(procedural tier — no prompt)").italic());
    } else {
        println!("  {:14} {}", style("prompt").dim(), p.prompt);
        println!("  {:14} {}", style("negative").dim(), p.negative);
    }
    Ok(())
}

fn run_verify(a: VerifyArgs) -> Result<()> {
    let spec = BookArtSpec::load(&a.spec)?;
    let plan = compile::resolve(&spec);
    let img = image::open(&a.image).with_context(|| format!("opening {}", a.image.display()))?;
    let mut rgba = if a.finished {
        img.to_rgba8()
    } else {
        crate::bookart::finish::finish_ornament(&img.to_rgb8(), &plan)
    };
    // Symmetry engine (§6.3): a geometric guarantee the finisher can't provide.
    if a.symmetrize {
        rgba = crate::bookart::geometry::symmetrize(&rgba, &plan.symmetry);
    }
    // Canvas sizing (§6.4): place onto the exact page canvas at the ornament's layout rect.
    if a.page {
        let tb = crate::bookart::geometry::text_block(&plan.page, &spec);
        let layout = crate::bookart::geometry::layout_for(&plan.ornament_kind, &tb);
        rgba = crate::bookart::finish::canvas::place_on_canvas(&rgba, &plan.page, &layout);
    }
    if let Some(out) = &a.out {
        if a.page {
            crate::bookart::finish::canvas::save_png_dpi(&rgba, out, plan.page.dpi).with_context(|| format!("writing {}", out.display()))?;
        } else {
            rgba.save(out).with_context(|| format!("writing {}", out.display()))?;
        }
        println!("{} {}", style("wrote").green(), out.display());
    }
    let sc = crate::bookart::scorecard::score(&rgba, &plan);
    let verdict = if sc.passes { style("PASS").green() } else { style("FAIL").red() };
    println!("{}  {}  ({}, {} × {})", style("bookart verify").bold(), a.image.display(), verdict, rgba.width(), rgba.height());
    println!("  {:16} {:.3}", style("chroma").dim(), sc.chroma_frac);
    println!("  {:16} {:.3}", style("alpha-halo").dim(), sc.alpha_partial_frac);
    println!("  {:16} {}", style("symmetry RMS").dim(), sc.symmetry_rms.map(|r| format!("{r:.3}")).unwrap_or_else(|| "— (not symmetric)".into()));
    println!("  {:16} {:.3}", style("ink coverage").dim(), sc.ink_coverage);
    println!("  {:16} {}", style("resolution").dim(), if sc.resolution_ok { "matches page".into() } else { format!("{}×{} (page is {}×{}; sizing is B2)", rgba.width(), rgba.height(), plan.page.w_px, plan.page.h_px) });
    for n in &sc.notes {
        println!("  {} {}", style("!").yellow(), n);
    }
    Ok(())
}

fn print_findings(findings: &[lint::Finding]) {
    for f in findings {
        let (tag, sty) = match f.level {
            Level::Error => ("error", style("✗").red()),
            Level::Warn => ("warn", style("!").yellow()),
            Level::Info => ("info", style("·").dim()),
        };
        println!("  {sty} {tag} {}: {}", style(&f.path).cyan(), f.message);
    }
}
