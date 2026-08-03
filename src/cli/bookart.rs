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
    /// Base model for the diffusion tier (the origin LoRAs are sd15).
    #[arg(long, default_value = "sd15")]
    pub model: String,
    /// Seed (diffusion tier).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Denoise steps (diffusion tier).
    #[arg(long, default_value_t = 28)]
    pub steps: usize,
}

pub async fn run(args: BookartArgs) -> Result<()> {
    match args.cmd {
        BookartCmd::New(a) => run_new(a),
        BookartCmd::Lint(a) => run_lint(a),
        BookartCmd::Show(a) => run_show(a),
        BookartCmd::Verify(a) => run_verify(a),
        BookartCmd::Render(a) => run_render(a).await,
    }
}

/// A working generation size for the diffusion tier from a layout rect's aspect: ~512 short side,
/// longest side capped at 768, snapped to /8 (sd15-friendly).
fn gen_size(rw: u32, rh: u32) -> (u32, u32) {
    let ar = rw.max(1) as f32 / rh.max(1) as f32;
    let (mut w, mut h) = if ar >= 1.0 { (512.0 * ar, 512.0) } else { (512.0, 512.0 / ar) };
    let scale = (768.0 / w.max(h)).min(1.0);
    w *= scale;
    h *= scale;
    let snap = |v: f32| (((v / 8.0).round() * 8.0) as u32).clamp(256, 768);
    (snap(w), snap(h))
}

/// Generate one diffusion render (with the origin LoRA if not `generic`) at `w×h`. Returns the raw RGB
/// plus the temp path it was written to (kept for an optional U2Net matte; the caller deletes it).
async fn diffuse(model: &str, plan: &crate::bookart::RenderPlan, w: u32, h: u32, steps: usize, seed: u64) -> Result<(image::RgbImage, PathBuf)> {
    let mut prompt = plan.prompt.clone();
    let mut builder = crate::api::Generate::new(model).negative(&plan.negative).size(w, h).steps(steps).seed(seed);
    if plan.origin != "generic" {
        prompt = format!("{prompt}, bookart_{} style", plan.origin);
        builder = builder.lora(format!("vulogov98/plakat-bookart#{}-sd15.safetensors", plan.origin), 1.0);
        println!("  {} origin LoRA: bookart_{} (sd15)", style("↳").cyan(), plan.origin);
    } else {
        println!("  {} generic line-art path (no LoRA)", style("↳").dim());
    }
    println!("  {} diffusion {w}×{h}, {steps} steps, seed {seed}…", style("→").cyan());
    let imgs = builder.prompt(&prompt).run().await.context("diffusion render")?;
    let img = imgs.into_iter().next().context("diffusion produced no image")?;
    let tmp = std::env::temp_dir().join(format!("bookart_diff_{seed}_{w}x{h}.png"));
    img.save(&tmp)?;
    let raw = image::open(&tmp).context("reopening the diffusion render")?.to_rgb8();
    Ok((raw, tmp))
}

/// U2Net matte → a solid silhouette in the ink tint (the `transparency: matte` mode).
async fn matte_silhouette(path: &std::path::Path, raw: &image::RgbImage, plan: &crate::bookart::RenderPlan) -> Result<image::RgbaImage> {
    let device = crate::api::device("auto")?;
    let (_fg, mask) = crate::pipelines::matting::matte(path, &device).await.context("U2Net matte")?;
    let mask = if mask.dimensions() != raw.dimensions() { image::imageops::resize(&mask, raw.width(), raw.height(), image::imageops::FilterType::Triangle) } else { mask };
    let tint = crate::bookart::finish::parse_tint(&plan.tint);
    let mut out = image::RgbaImage::new(raw.width(), raw.height());
    for (x, y, p) in out.enumerate_pixels_mut() {
        *p = image::Rgba([tint[0], tint[1], tint[2], mask.get_pixel(x, y).0[0]]);
    }
    Ok(out)
}

async fn run_render(a: RenderArgs) -> Result<()> {
    use crate::bookart::{finish, geometry, procedural};
    let spec = BookArtSpec::load(&a.spec)?;
    let plan = compile::resolve(&spec);
    let tb = geometry::text_block(&plan.page, &spec);
    let layout = geometry::layout_for(&plan.ornament_kind, &tb);
    let r0 = layout.rects[0];

    // Produce the ornament at its layout resolution, per the resolved render tier.
    let ornament = match plan.tier.as_str() {
        // B3: vector-native, no weights.
        "procedural" => {
            let gray = procedural::generate(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h);
            finish::finish_procedural(&gray, &plan)
        }
        // B4: diffusion — generic line-art path + the origin LoRA (if not `generic`); optional U2Net matte.
        "diffusion" => {
            let (gw, gh) = gen_size(r0.w, r0.h);
            let (raw, tmp) = diffuse(&a.model, &plan, gw, gh, a.steps, a.seed).await?;
            let finished = if plan.transparency_mode == "matte" {
                println!("  {} U2Net matte → silhouette", style("↳").cyan());
                matte_silhouette(&tmp, &raw, &plan).await?
            } else {
                finish::finish_ornament(&raw, &plan)
            };
            let _ = std::fs::remove_file(&tmp);
            finished
        }
        // B5: composite — a procedural frame with a diffusion line-art picture inlaid into its window
        // (the persona geometry + detail-composite analog).
        "composite" => {
            let (frame_paths, (wx, wy, ww, wh)) = procedural::frame(&plan.symmetry, r0.w, r0.h);
            let width = (r0.w.min(r0.h) as f32 * 0.004).max(1.5);
            let frame_rgba = finish::finish_procedural(&procedural::rasterise(&frame_paths, r0.w, r0.h, width), &plan);
            let (gw, gh) = gen_size(ww, wh);
            let (raw, tmp) = diffuse(&a.model, &plan, gw, gh, a.steps, a.seed).await?;
            let _ = std::fs::remove_file(&tmp);
            // the inlay is always finished as clean LINE art (transparent paper), never a solid slab.
            let inlay_gray = finish::binarize::binarise(&finish::to_luma(&raw), "xdog", plan.ink_weight);
            let inlay = finish::alpha::to_transparent(&inlay_gray, "luminance", finish::parse_tint(&plan.tint), 0.0);
            let mut canvas = image::RgbaImage::from_pixel(r0.w, r0.h, image::Rgba([0, 0, 0, 0]));
            let pic = image::imageops::resize(&inlay, ww.max(1), wh.max(1), image::imageops::FilterType::Lanczos3);
            image::imageops::overlay(&mut canvas, &pic, wx as i64, wy as i64);
            image::imageops::overlay(&mut canvas, &frame_rgba, 0, 0);
            println!("  {} composite: procedural frame + diffusion inlay", style("↳").cyan());
            canvas
        }
        other => anyhow::bail!("unknown render tier `{other}`"),
    };

    // Symmetry (a no-op for `none`); skipped for `composite` — the frame is already symmetric and the
    // inlaid picture is a scene we don't want mirror-doubled.
    let orn = if plan.tier == "composite" { ornament } else { geometry::symmetrize(&ornament, &plan.symmetry) };
    let page = finish::canvas::place_on_canvas(&orn, &plan.page, &layout);
    finish::canvas::save_png_dpi(&page, &a.out, plan.page.dpi)?;
    println!(
        "{} {}  ({} × {} px @ {} DPI · {} · {} · {} · {} piece(s))",
        style("wrote").green(),
        a.out.display(),
        page.width(),
        page.height(),
        plan.page.dpi,
        plan.tier,
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
