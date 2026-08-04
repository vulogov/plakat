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
    /// Render an ornament to a transparent, page-sized PNG per a spec (all three tiers: procedural /
    /// diffusion / composite). `--svg` also emits born-vector SVG (procedural).
    Render(RenderArgs),
    /// Illustrate a single B/W plate from a prompt (a standalone frontispiece / spot via the diffusion
    /// tier) — the quick path when you don't want to author a spec.
    Illustrate(IllustrateArgs),
    /// Render a coherent **kit** — a matched set of ornaments from a spec's `kit` block, sharing one
    /// origin/technique, motif DNA, and seed lineage. Emits a directory + contact sheet + manifest,
    /// and a CLIP style-coherence score (§10, flagship).
    Kit(KitArgs),
    /// **Manuscript-aware** set (§11, flagship): parse a book's chapters (Markdown headings or a plain
    /// list) → a frontispiece + a seed-varied headpiece & tailpiece per chapter, in one hand. Emits a
    /// directory + manifest + contact sheet, and optional LaTeX includes.
    Manuscript(ManuscriptArgs),
    /// Build a contact sheet / page-proof from a directory of ornament PNGs.
    Proof(ProofArgs),
    /// Classify the changes between two specs (§9): which edits are cheap `post` ops, which need a
    /// `re-raster`, and which force a full `re-gen`.
    Diff(DiffArgs),
    /// Apply a cheap `post`-class edit to a *finished* ornament PNG — recolour the ink (`--tint`) or
    /// re-apply symmetry (`--symmetry`) — with no re-render. Other changes need `bookart render`.
    Edit(EditArgs),
    /// Lineage: blend two traditions into a new spec (origin of A × technique of B, motifs unioned).
    Blend(BlendArgs),
    /// Trace a raster ornament into a compact SVG (B1; needs the `bookart-trace` feature). The
    /// procedural tier is already born-vector — this is for scanned / diffusion / composite art.
    Vectorize(VectorizeArgs),
    /// List the origins × techniques × ornament vocabulary, which origins ship a hosted LoRA, and the
    /// status of the optional `assets/bookart/lexicon.hjson` override.
    Origins(OriginsArgs),
}

#[derive(Args, Debug)]
pub struct OriginsArgs {
    /// Also print each origin's prompt scaffold + default technique + motifs.
    #[arg(long, default_value_t = false)]
    pub details: bool,
}

#[derive(Args, Debug)]
pub struct VectorizeArgs {
    /// Input raster (PNG/…); alpha is honoured (flattened onto white before tracing).
    pub image: PathBuf,
    /// Output SVG path.
    #[arg(long)]
    pub out: PathBuf,
    /// Ink colour for the traced paths (`black`/`sepia`/`#rrggbb`).
    #[arg(long, default_value = "black")]
    pub tint: String,
    /// DPI the raster was rendered at — sets the SVG's physical (mm) print size.
    #[arg(long, default_value_t = 300)]
    pub dpi: u32,
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
    /// Also emit a born-vector SVG (procedural tier only; §7.5). Otherwise honours the spec's
    /// `output.formats`.
    #[arg(long, default_value_t = false)]
    pub svg: bool,
    /// Rejection sampling (diffusion tier): try up to N seeds, keep the first that clears the scorecard.
    #[arg(long, default_value_t = 1)]
    pub attempts: u32,
    /// Also land the ornament (+ its recipe sidecar) in a `plakat photos` album at this path.
    #[arg(long)]
    pub import: Option<PathBuf>,
    /// C2: also cache the pre-finish gray + plan (`<out>.raw.png`/`.plan.json`) so `bookart edit
    /// --ink-weight/--transparency` can re-finish without re-rendering.
    #[arg(long = "cache-raw", default_value_t = false)]
    pub cache_raw: bool,
}

#[derive(Args, Debug)]
pub struct IllustrateArgs {
    /// The illustration prompt (a B/W plate suitable as a frontispiece / spot).
    pub prompt: String,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "generic")]
    pub origin: String,
    #[arg(long, default_value = "line")]
    pub technique: String,
    #[arg(long, default_value = "a5")]
    pub page: String,
    /// Ornament framing (`frontispiece` page-fill, or `vignette` centred spot).
    #[arg(long = "type", default_value = "frontispiece")]
    pub kind: String,
    #[arg(long, default_value = "sd15")]
    pub model: String,
    #[arg(long, default_value_t = 28)]
    pub steps: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long, default_value_t = 1)]
    pub attempts: u32,
    /// Also land the plate (+ its recipe sidecar) in a `plakat photos` album at this path.
    #[arg(long)]
    pub import: Option<PathBuf>,
    /// C2: also cache the pre-finish gray + plan so `bookart edit --ink-weight/--transparency` works.
    #[arg(long = "cache-raw", default_value_t = false)]
    pub cache_raw: bool,
}

#[derive(Args, Debug)]
pub struct KitArgs {
    /// A spec with a `kit: { ornaments: [...] }` block.
    pub spec: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "sd15")]
    pub model: String,
    #[arg(long, default_value_t = 28)]
    pub steps: usize,
    /// Also emit born-vector SVG per procedural ornament.
    #[arg(long, default_value_t = false)]
    pub svg: bool,
    /// Skip the CLIP style-coherence probe (avoids loading the ~1.7 GB CLIP model).
    #[arg(long = "no-coherence", default_value_t = false)]
    pub no_coherence: bool,
}

#[derive(Args, Debug)]
pub struct ManuscriptArgs {
    /// A manuscript: Markdown (chapters = `#`/`##` headings) or a plain one-title-per-line list.
    pub book: PathBuf,
    /// A spec supplying the shared style (origin/technique/motif/page); its `kit.seed` seeds the lineage.
    #[arg(long)]
    pub kit: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value = "sd15")]
    pub model: String,
    #[arg(long, default_value_t = 24)]
    pub steps: usize,
    #[arg(long, default_value_t = false)]
    pub svg: bool,
    /// Also emit a LaTeX include file (`\input{includes.tex}`).
    #[arg(long, default_value_t = false)]
    pub latex: bool,
}

#[derive(Args, Debug)]
pub struct ProofArgs {
    /// A directory of ornament PNGs.
    pub dir: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Args, Debug)]
pub struct DiffArgs {
    pub old: PathBuf,
    pub new: PathBuf,
}

#[derive(Args, Debug)]
pub struct EditArgs {
    /// A finished ornament PNG.
    pub image: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// Recolour the ink (`black` / `sepia` / `#rrggbb`).
    #[arg(long)]
    pub tint: Option<String>,
    /// Re-apply symmetry (`bilateral` / `radial:N`).
    #[arg(long)]
    pub symmetry: Option<String>,
    /// C2 (needs `render --cache-raw`): re-finish at a new ink weight `[0,1]` — re-runs the binariser +
    /// transparency on the cached gray, no re-sampling.
    #[arg(long = "ink-weight")]
    pub ink_weight: Option<f32>,
    /// C2 (needs `render --cache-raw`): re-finish with a new transparency mode (`luminance`/`threshold`/`fade`).
    #[arg(long)]
    pub transparency: Option<String>,
    /// C2 (needs `render --cache-raw`): re-finish with a new edge fade `[0,1]`.
    #[arg(long)]
    pub fade: Option<f32>,
}

#[derive(Args, Debug)]
pub struct BlendArgs {
    pub a: PathBuf,
    pub b: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

pub async fn run(args: BookartArgs) -> Result<()> {
    match args.cmd {
        BookartCmd::New(a) => run_new(a),
        BookartCmd::Lint(a) => run_lint(a),
        BookartCmd::Show(a) => run_show(a),
        BookartCmd::Verify(a) => run_verify(a),
        BookartCmd::Render(a) => run_render(a).await,
        BookartCmd::Illustrate(a) => run_illustrate(a).await,
        BookartCmd::Kit(a) => run_kit(a).await,
        BookartCmd::Manuscript(a) => run_manuscript(a).await,
        BookartCmd::Proof(a) => run_proof(a),
        BookartCmd::Diff(a) => run_diff(a),
        BookartCmd::Edit(a) => run_edit(a),
        BookartCmd::Blend(a) => run_blend(a),
        BookartCmd::Vectorize(a) => run_vectorize(a),
        BookartCmd::Origins(a) => run_origins(a),
    }
}

fn run_diff(a: DiffArgs) -> Result<()> {
    use crate::bookart::edit::{self, EditClass};
    let load = |p: &PathBuf| -> Result<serde_json::Value> {
        deser_hjson::from_str(&std::fs::read_to_string(p)?).map_err(|e| anyhow::anyhow!("parsing {}: {e}", p.display()))
    };
    let (old, new) = (load(&a.old)?, load(&a.new)?);
    let changes = edit::diff(&old, &new);
    if changes.is_empty() {
        println!("{} the specs are identical", style("=").dim());
        return Ok(());
    }
    println!("{}  {} → {}", style("bookart diff").bold(), a.old.display(), a.new.display());
    for c in &changes {
        let tag = match c.class {
            EditClass::Post => style("post").green(),
            EditClass::Reraster => style("re-raster").yellow(),
            EditClass::Regen => style("re-gen").red(),
        };
        println!("  {:9} {}: {} → {}", tag, style(&c.path).cyan(), c.old.as_deref().unwrap_or("∅"), c.new.as_deref().unwrap_or("∅"));
    }
    let worst = edit::worst(&changes).unwrap();
    println!("\n{} cheapest sufficient action: {}", style("→").bold(), style(worst.label()).bold());
    Ok(())
}

fn run_edit(a: EditArgs) -> Result<()> {
    // C2: ink-weight / transparency / fade need the pre-finish gray — route to the refinish path.
    if a.ink_weight.is_some() || a.transparency.is_some() || a.fade.is_some() {
        return run_refinish(&a);
    }
    let mut rgba = image::open(&a.image).with_context(|| format!("opening {}", a.image.display()))?.to_rgba8();
    let mut ops = Vec::new();
    if let Some(tint) = &a.tint {
        let t = crate::bookart::finish::parse_tint(tint);
        for p in rgba.pixels_mut() {
            if p.0[3] > 0 {
                p.0[0] = t[0];
                p.0[1] = t[1];
                p.0[2] = t[2];
            }
        }
        ops.push(format!("re-tint {tint}"));
    }
    if let Some(sym) = &a.symmetry {
        rgba = crate::bookart::geometry::symmetrize(&rgba, sym);
        ops.push(format!("symmetry {sym}"));
    }
    if ops.is_empty() {
        anyhow::bail!("nothing to edit — pass `--tint` and/or `--symmetry` (the `post` class). Origin/motif/page changes need `bookart render` (see `bookart diff`).");
    }
    rgba.save(&a.out)?;
    println!("{} {}  [{}]", style("wrote").green(), a.out.display(), ops.join(", "));
    Ok(())
}

/// C2: re-finish a cached ornament at a new ink weight / transparency / fade — the finisher only, no
/// re-sampling. Reads `<image>.raw.png` + `<image>.plan.json` written by `render --cache-raw`, patches
/// the plan, then re-runs finish → symmetry → page canvas.
fn run_refinish(a: &EditArgs) -> Result<()> {
    use crate::bookart::{compile::RenderPlan, finish, geometry};
    let (gray_path, plan_path) = raw_cache_paths(&a.image);
    if !gray_path.exists() || !plan_path.exists() {
        anyhow::bail!(
            "ink-weight/transparency/fade edits need the raw cache ({} + {}). Re-run `bookart render <spec> \
             --out {} --cache-raw` first (these are `post` edits only when the pre-finish gray is cached).",
            gray_path.display(), plan_path.display(), a.image.display()
        );
    }
    let gray = image::open(&gray_path).with_context(|| format!("opening {}", gray_path.display()))?.to_luma8();
    let mut plan: RenderPlan = serde_json::from_str(&std::fs::read_to_string(&plan_path)?)
        .with_context(|| format!("parsing {}", plan_path.display()))?;
    let mut ops = Vec::new();
    if let Some(w) = a.ink_weight { plan.ink_weight = w.clamp(0.0, 1.0); ops.push(format!("ink-weight {w}")); }
    if let Some(t) = &a.transparency { plan.transparency_mode = t.clone(); ops.push(format!("transparency {t}")); }
    if let Some(f) = a.fade { plan.fade = f.clamp(0.0, 1.0); ops.push(format!("fade {f}")); }
    if let Some(tint) = &a.tint { plan.tint = tint.clone(); ops.push(format!("tint {tint}")); }
    // Re-finish from the cached gray: procedural skips binarise (born-clean); diffusion re-binarises.
    let orn = if plan.tier == "procedural" {
        finish::finish_procedural(&gray, &plan)
    } else {
        finish::finish_from_gray(&gray, &plan)
    };
    let sym = a.symmetry.clone().unwrap_or_else(|| plan.symmetry.clone());
    let orn = geometry::symmetrize(&orn, &sym);
    let tb = geometry::text_block(&plan.page, &BookArtSpec::default());
    let layout = geometry::layout_for(&plan.ornament_kind, &tb);
    let page = finish::canvas::place_on_canvas(&orn, &plan.page, &layout);
    finish::canvas::save_png_dpi(&page, &a.out, plan.page.dpi)?;
    println!("{} {}  [re-finish: {}]  (no re-render)", style("wrote").green(), a.out.display(), ops.join(", "));
    Ok(())
}

fn run_blend(a: BlendArgs) -> Result<()> {
    let (sa, sb) = (BookArtSpec::load(&a.a)?, BookArtSpec::load(&a.b)?);
    let origin = sa.origin.clone().unwrap_or_else(|| "generic".into());
    let technique = sb.technique.clone().or_else(|| sa.technique.clone()).unwrap_or_else(|| "line".into());
    let mut motif = sa.motif.clone().unwrap_or_default();
    for m in sb.motif.clone().unwrap_or_default() {
        if !motif.contains(&m) {
            motif.push(m);
        }
    }
    let kind = sa.ornament.as_ref().and_then(|o| o.kind.clone()).unwrap_or_else(|| "vignette".into());
    let page = sa.page.as_ref().and_then(|p| p.size.clone()).unwrap_or_else(|| "a5".into());
    let motif_json = motif.iter().map(|m| format!("\"{m}\"")).collect::<Vec<_>>().join(", ");
    let prompt_line = sa.ornament.as_ref().and_then(|o| o.prompt.clone()).map(|p| format!("\n    prompt: \"{p}\"")).unwrap_or_default();
    let spec = format!(
        "{{\n  schema: \"bookart/1\"\n  origin: \"{origin}\"\n  technique: \"{technique}\"\n  motif: [{motif_json}]\n  page: {{ size: \"{page}\" }}\n  ornament: {{\n    type: \"{kind}\"{prompt_line}\n  }}\n}}\n"
    );
    std::fs::write(&a.out, &spec).with_context(|| format!("writing {}", a.out.display()))?;
    println!("{} {}  (blend: origin {} × technique {}, {} motif(s))", style("wrote").green(), a.out.display(), origin, technique, motif.len());
    print_findings(&lint::lint(&BookArtSpec::load(&a.out)?));
    Ok(())
}

async fn run_manuscript(a: ManuscriptArgs) -> Result<()> {
    use crate::bookart::spec::{BookArtSpec, Ornament};
    use crate::bookart::{kit, manuscript};
    let text = std::fs::read_to_string(&a.book).with_context(|| format!("reading {}", a.book.display()))?;
    let chapters = manuscript::parse_chapters(&text);
    if chapters.is_empty() {
        anyhow::bail!("no chapters found in {} (Markdown `#` headings, or one title per line)", a.book.display());
    }
    let theme = BookArtSpec::load(&a.kit)?;
    let base_seed = theme.kit.as_ref().and_then(|k| k.seed).unwrap_or(0);
    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    println!("{}  {} chapter(s), origin {} → {}", style("bookart manuscript").bold(), chapters.len(), theme.origin.as_deref().unwrap_or("generic"), a.out.display());

    // Per-ornament spec = the shared theme + one ornament (tier overridable).
    let mk = |kind: &str, tier: Option<&str>| BookArtSpec {
        schema: theme.schema.clone(),
        origin: theme.origin.clone(),
        technique: theme.technique.clone(),
        motif: theme.motif.clone(),
        ink: theme.ink.clone(),
        page: theme.page.clone(),
        transparent: theme.transparent,
        output: theme.output.clone(),
        ornament: Some(Ornament { kind: Some(kind.into()), tier: tier.map(String::from), ..Default::default() }),
        kit: None,
    };

    // Frontispiece (once).
    let front = "frontispiece.png";
    println!("\n{} frontispiece…", style("→").cyan());
    do_render(mk("frontispiece", Some("diffusion")), &a.out.join(front), &a.model, base_seed, a.steps, a.svg, 1, None, false).await.context("frontispiece")?;

    let mut all_files = vec![front.to_string()];
    let mut tex_ch = Vec::new();
    let mut man_ch = Vec::new();
    for (i, ch) in chapters.iter().enumerate() {
        let n = i + 1;
        let (hseed, tseed) = (kit::ornament_seed(base_seed, i * 2 + 1), kit::ornament_seed(base_seed, i * 2 + 2));
        let (hfile, tfile) = (format!("ch{n:02}_headpiece.png"), format!("ch{n:02}_tailpiece.png"));
        println!("\n{} [ch {n}/{}] {}  (headpiece seed {hseed})", style("→").cyan(), chapters.len(), ch.title);
        // headpiece: a procedural ornamental band (застАвка), varied per chapter by the seed lineage —
        // clean airy line-work, not a heavy woodcut block. The pictorial motif lives in the frontispiece.
        do_render(mk("headpiece", Some("procedural")), &a.out.join(&hfile), &a.model, hseed, a.steps, a.svg, 1, None, false).await.with_context(|| format!("chapter {n} headpiece"))?;
        // tailpiece: a procedural cul-de-lampe, also varied per chapter.
        do_render(mk("tailpiece", Some("procedural")), &a.out.join(&tfile), &a.model, tseed, a.steps, a.svg, 1, None, false).await.with_context(|| format!("chapter {n} tailpiece"))?;
        all_files.push(hfile.clone());
        all_files.push(tfile.clone());
        tex_ch.push((ch.title.clone(), hfile.clone(), tfile.clone()));
        man_ch.push(serde_json::json!({ "chapter": n, "title": ch.title, "first_letter": ch.first_letter, "headpiece": hfile, "tailpiece": tfile, "headpiece_seed": hseed }));
    }

    // Contact sheet.
    let mut thumbs = Vec::new();
    for f in &all_files {
        let page = image::open(a.out.join(f))?.to_rgba8();
        thumbs.push(kit::thumb_on_white(&kit::crop_to_content(&page), 300));
    }
    kit::contact_sheet(&thumbs, 3).save(a.out.join("contact_sheet.png"))?;

    // Manifest (+ optional LaTeX).
    let manifest = serde_json::json!({ "schema": "bookart-manuscript/1", "origin": theme.origin, "technique": theme.technique, "motif": theme.motif, "seed": base_seed, "frontispiece": front, "chapters": man_ch });
    std::fs::write(a.out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    if a.latex {
        std::fs::write(a.out.join("includes.tex"), manuscript::latex_includes(front, &tex_ch))?;
        println!("{} LaTeX includes → includes.tex", style("↳").cyan());
    }
    println!("{} {} chapter(s), {} asset(s) + contact sheet + manifest → {}", style("done:").bold(), chapters.len(), all_files.len(), a.out.display());
    Ok(())
}

fn run_proof(a: ProofArgs) -> Result<()> {
    use crate::bookart::kit;
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&a.dir)
        .with_context(|| format!("reading {}", a.dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png") && p.file_name().and_then(|n| n.to_str()) != Some("contact_sheet.png"))
        .collect();
    pngs.sort();
    if pngs.is_empty() {
        anyhow::bail!("no ornament PNGs in {}", a.dir.display());
    }
    let mut thumbs = Vec::new();
    for f in &pngs {
        let page = image::open(f)?.to_rgba8();
        thumbs.push(kit::thumb_on_white(&kit::crop_to_content(&page), 320));
    }
    kit::contact_sheet(&thumbs, 3).save(&a.out)?;
    println!("{} {} ornament(s) → {}", style("proof").green(), pngs.len(), a.out.display());
    Ok(())
}

async fn run_kit(a: KitArgs) -> Result<()> {
    use crate::bookart::kit;
    let spec = BookArtSpec::load(&a.spec)?;
    let kitspec = spec.kit.clone().context("this spec has no `kit` block — add `kit: { ornaments: [...] }`")?;
    let ornaments = kitspec.ornaments.clone().unwrap_or_default();
    if ornaments.is_empty() {
        anyhow::bail!("the kit has no ornaments");
    }
    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let base_seed = kitspec.seed.unwrap_or(0);
    println!("{}  {} ornament(s), origin {} → {}", style("bookart kit").bold(), ornaments.len(), spec.origin.as_deref().unwrap_or("generic"), a.out.display());

    // Render each ornament sharing origin/technique/motif + a deterministic seed lineage.
    let (mut files, mut kinds, mut seeds): (Vec<PathBuf>, Vec<String>, Vec<u64>) = (vec![], vec![], vec![]);
    for (i, orn) in ornaments.iter().enumerate() {
        let seed_i = kit::ornament_seed(base_seed, i);
        let kind = orn.kind.clone().unwrap_or_else(|| "divider".into());
        let per = crate::bookart::spec::BookArtSpec {
            schema: spec.schema.clone(),
            origin: spec.origin.clone(),
            technique: spec.technique.clone(),
            motif: spec.motif.clone(),
            ink: spec.ink.clone(),
            page: spec.page.clone(),
            transparent: spec.transparent,
            output: spec.output.clone(),
            ornament: Some(orn.clone()),
            kit: None,
        };
        let file = a.out.join(format!("{i:02}_{kind}.png"));
        println!("\n{} [{}/{}] {kind}  (seed {seed_i})", style("→").cyan(), i + 1, ornaments.len());
        do_render(per, &file, &a.model, seed_i, a.steps, a.svg, 1, None, false).await.with_context(|| format!("kit ornament {i} ({kind})"))?;
        files.push(file);
        kinds.push(kind);
        seeds.push(seed_i);
    }

    // Contact sheet: crop each page to its ink, thumb on white, tile.
    let mut thumbs = Vec::new();
    for f in &files {
        let page = image::open(f)?.to_rgba8();
        thumbs.push(kit::thumb_on_white(&kit::crop_to_content(&page), 320));
    }
    let sheet_path = a.out.join("contact_sheet.png");
    kit::contact_sheet(&thumbs, 3).save(&sheet_path)?;
    println!("\n{} contact sheet → {}", style("↳").cyan(), sheet_path.display());

    // CLIP style-coherence (opt-out).
    let coherence = if a.no_coherence {
        None
    } else {
        match kit_coherence(&files).await {
            Ok((min, mean)) => {
                println!("{} kit coherence: min {min:.3}, mean {mean:.3}  (CLIP style similarity across the set)", style("↳").cyan());
                Some((min, mean))
            }
            Err(e) => {
                println!("{} coherence skipped: {e}", style("·").yellow());
                None
            }
        }
    };

    // Manifest.
    let manifest = serde_json::json!({
        "schema": "bookart-kit/1",
        "origin": spec.origin,
        "technique": spec.technique,
        "motif": spec.motif,
        "seed": base_seed,
        "ornaments": files.iter().zip(&kinds).zip(&seeds)
            .map(|((f, k), s)| serde_json::json!({ "file": f.file_name().and_then(|n| n.to_str()), "type": k, "seed": s }))
            .collect::<Vec<_>>(),
        "coherence": coherence.map(|(min, mean)| serde_json::json!({ "min": min, "mean": mean })),
    });
    std::fs::write(a.out.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    println!("{} {} ornament(s) + contact sheet + manifest → {}", style("done:").bold(), files.len(), a.out.display());
    Ok(())
}

/// Embed each kit ornament (cropped to content, on white) with CLIP and return (min, mean) pairwise
/// cosine — the style-coherence of the set.
async fn kit_coherence(files: &[PathBuf]) -> Result<(f32, f32)> {
    use crate::bookart::kit;
    let device = crate::api::device("auto")?;
    let embedder = crate::pipelines::clip_embed::ClipEmbedder::load(&device).await?;
    let mut embs = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let page = image::open(f)?.to_rgba8();
        let thumb = kit::thumb_on_white(&kit::crop_to_content(&page), 224);
        let tmp = std::env::temp_dir().join(format!("bookart_kit_emb_{i}.png"));
        thumb.save(&tmp)?;
        let e = embedder.embed_image(&tmp)?;
        let _ = std::fs::remove_file(&tmp);
        embs.push(e);
    }
    Ok(kit::pairwise_min_mean(&embs))
}

/// A working generation size for the diffusion tier from a layout rect's aspect: ~512 short side,
/// longest side capped at 768, snapped to /8 (sd15-friendly).
async fn run_render(a: RenderArgs) -> Result<()> {
    let spec = BookArtSpec::load(&a.spec)?;
    do_render(spec, &a.out, &a.model, a.seed, a.steps, a.svg, a.attempts, a.import.as_deref(), a.cache_raw).await
}

async fn run_illustrate(a: IllustrateArgs) -> Result<()> {
    use crate::bookart::spec::{BookArtSpec, Ornament, Page};
    // A single B/W plate: synthesise a diffusion-tier spec from the prompt + flags.
    let spec = BookArtSpec {
        schema: Some(crate::bookart::SCHEMA_VERSION.into()),
        origin: Some(a.origin),
        technique: Some(a.technique),
        page: Some(Page { size: Some(a.page), ..Default::default() }),
        ornament: Some(Ornament { kind: Some(a.kind), tier: Some("diffusion".into()), prompt: Some(a.prompt), ..Default::default() }),
        ..Default::default()
    };
    do_render(spec, &a.out, &a.model, a.seed, a.steps, false, a.attempts, a.import.as_deref(), a.cache_raw).await
}

/// The shared render entry (used by `render`, `illustrate`, `kit`, `manuscript`): drive the library
/// render core ([`crate::bookart::render::render_spec`]), then write the PNG (+ opt-in SVG) to disk.
async fn do_render(spec: BookArtSpec, out: &std::path::Path, model: &str, seed: u64, steps: usize, svg: bool, attempts: u32, import: Option<&std::path::Path>, cache_raw: bool) -> Result<()> {
    use crate::bookart::render::{recipe_metadata, render_spec, RenderOpts};
    let r = render_spec(&spec, &RenderOpts { model: model.into(), seed, steps, svg, attempts }).await?;
    // A5: attach the reproducibility recipe (origin/technique/spec-hash) as a PNG tEXt chunk + `.json`
    // sidecar, so the ornament is searchable, re-runnable, and `--import`-ready.
    let meta = recipe_metadata(&r.plan, model, seed, steps);
    crate::bookart::finish::canvas::save_png_dpi_with_metadata(&r.page, out, r.plan.page.dpi, &meta)?;
    // C2: cache the pre-finish gray + the resolved plan so `bookart edit --ink-weight/--transparency`
    // can re-finish without re-sampling (procedural/diffusion only; composite/matte have no single gray).
    if cache_raw {
        write_raw_cache(out, &r)?;
    }
    println!(
        "{} {}  ({} × {} px @ {} DPI · {} · {} · {} · {} piece(s))",
        style("wrote").green(),
        out.display(),
        r.page.width(),
        r.page.height(),
        r.plan.page.dpi,
        r.plan.tier,
        r.plan.ornament_kind,
        r.plan.symmetry,
        r.pieces
    );
    let want_svg = svg || r.plan.formats.iter().any(|f| f == "svg");
    if let Some(svg_str) = &r.svg {
        let svg_path = out.with_extension("svg");
        std::fs::write(&svg_path, svg_str).with_context(|| format!("writing {}", svg_path.display()))?;
        println!("  {} born-vector SVG → {}", style("↳").cyan(), svg_path.display());
    } else if want_svg && r.plan.tier != "procedural" {
        // B1: the pixel tiers can only be *traced* (the procedural tier is born-vector above).
        maybe_trace_svg(&r.page, out, &r.plan)?;
    }
    if let Some(album) = import {
        import_ornament(out, album)?;
    }
    Ok(())
}

/// C2: paths of the raw-refinish cache next to an ornament PNG.
fn raw_cache_paths(out: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let mut gray = out.as_os_str().to_owned();
    gray.push(".raw.png");
    let mut plan = out.as_os_str().to_owned();
    plan.push(".plan.json");
    (std::path::PathBuf::from(gray), std::path::PathBuf::from(plan))
}

/// C2: write the pre-finish gray + the resolved plan next to the ornament (for `bookart edit`).
fn write_raw_cache(out: &std::path::Path, r: &crate::bookart::render::Rendered) -> Result<()> {
    let Some(gray) = &r.raw_gray else {
        println!("  {} --cache-raw skipped: the `{}` tier has no single gray to re-finish", style("·").yellow(), r.plan.tier);
        return Ok(());
    };
    let (gray_path, plan_path) = raw_cache_paths(out);
    gray.save(&gray_path).with_context(|| format!("writing {}", gray_path.display()))?;
    std::fs::write(&plan_path, serde_json::to_string(&r.plan)?).with_context(|| format!("writing {}", plan_path.display()))?;
    println!("  {} raw cache → {} (edit ink-weight/transparency without re-render)", style("↳").cyan(), gray_path.display());
    Ok(())
}

/// B3: `bookart origins` — list the vocabulary, LoRA hosting, and override status (mirrors
/// `plakat style` / the `doctor` bookart section).
fn run_origins(a: OriginsArgs) -> Result<()> {
    use crate::bookart::lexicon;
    println!("{}", style("bookart origins").bold());
    for origin in lexicon::all_origins() {
        let hosted = lexicon::has_hosted_lora(&origin);
        let custom = !lexicon::ORIGINS.contains(&origin.as_str());
        let tag = if hosted {
            style("[hosted LoRA]").green().to_string()
        } else if origin == "generic" {
            style("[LoRA-free path]").dim().to_string()
        } else {
            style("[scaffold only]").yellow().to_string()
        };
        let custom_tag = if custom { style(" (custom)").cyan().to_string() } else { String::new() };
        println!("  {:<10} {tag}{custom_tag}", style(&origin).bold());
        if a.details {
            let (scaffold, tech, motifs) = lexicon::origin_scaffold_dyn(&origin);
            println!("      scaffold: {}", style(&scaffold).dim());
            println!("      default technique: {tech}   ·   motifs: {}", motifs.join(", "));
        }
    }
    println!("\n{}", style("techniques").bold());
    for t in lexicon::TECHNIQUES {
        println!("  {:<12} → binariser {} · {}", style(t).bold(), lexicon::technique_binariser(t), style(lexicon::technique_prompt(t)).dim());
    }
    println!("\n{}", style("ornaments").bold());
    for k in lexicon::ORNAMENTS {
        println!("  {:<12} tier {:<10} symmetry {}", style(k).bold(), lexicon::default_tier(k), lexicon::default_symmetry(k));
    }
    let path = lexicon::override_path();
    println!();
    match lexicon::lexicon_override() {
        Some(ov) => println!("{} lexicon override: {} ({} custom origin(s))", style("✓").green(), path.display(), ov.origins.len()),
        None => println!("{} no lexicon override ({} — built-in vocabulary). Add one to define custom traditions.", style("·").dim(), path.display()),
    }
    Ok(())
}

/// B1: trace a diffusion/composite page to SVG when `--svg` is asked for on a pixel tier. With the
/// `bookart-trace` feature it writes the traced SVG; without it, a one-line note (the PNG is the
/// deliverable — §7.5).
#[cfg(feature = "bookart-trace")]
fn maybe_trace_svg(page: &image::RgbaImage, out: &std::path::Path, plan: &crate::bookart::compile::RenderPlan) -> Result<()> {
    let tint = crate::bookart::finish::parse_tint(&plan.tint);
    let svg = crate::bookart::finish::trace::trace_rgba(page, tint, plan.page.dpi).context("tracing the render to SVG")?;
    let svg_path = out.with_extension("svg");
    std::fs::write(&svg_path, &svg).with_context(|| format!("writing {}", svg_path.display()))?;
    println!("  {} traced SVG → {} ({:.1} KB)", style("↳").cyan(), svg_path.display(), svg.len() as f32 / 1024.0);
    Ok(())
}

#[cfg(not(feature = "bookart-trace"))]
fn maybe_trace_svg(_page: &image::RgbaImage, _out: &std::path::Path, plan: &crate::bookart::compile::RenderPlan) -> Result<()> {
    println!(
        "  {} SVG for the `{}` tier is a raster trace — rebuild with `--features bookart-trace` (the PNG is the deliverable §7.5)",
        style("·").yellow(), plan.tier
    );
    Ok(())
}

/// B1: `bookart vectorize <raster> --out <svg>` — trace a raster ornament into a compact SVG. Behind
/// the `bookart-trace` feature; a clear note (not a silent no-op) when it isn't compiled in.
#[cfg(feature = "bookart-trace")]
fn run_vectorize(a: VectorizeArgs) -> Result<()> {
    let tint = crate::bookart::finish::parse_tint(&a.tint);
    let svg = crate::bookart::finish::trace::trace_file(&a.image, tint, a.dpi)
        .with_context(|| format!("tracing {}", a.image.display()))?;
    std::fs::write(&a.out, &svg).with_context(|| format!("writing {}", a.out.display()))?;
    println!(
        "{} {}  ({} · {:.1} KB)",
        style("traced").green(),
        a.out.display(),
        a.image.display(),
        svg.len() as f32 / 1024.0
    );
    Ok(())
}

#[cfg(not(feature = "bookart-trace"))]
fn run_vectorize(_a: VectorizeArgs) -> Result<()> {
    anyhow::bail!(
        "bookart vectorize needs the `bookart-trace` feature — rebuild with `--features bookart-trace` \
         (it pulls an extra image-tracing stack, so it's opt-in). The procedural tier's `--svg` is \
         born-vector and always available."
    )
}

/// A5: land a rendered ornament (+ its `.json` sidecar) in a `plakat photos` album, curated with its
/// bookart recipe. `photos` is an optional feature; when it's not compiled in, say so instead of
/// silently dropping the request.
#[cfg(feature = "photos")]
fn import_ornament(out: &std::path::Path, album: &std::path::Path) -> Result<()> {
    let n = crate::photos::import::import_outputs(album, &[out.to_path_buf()], false)
        .with_context(|| format!("importing {} into album {}", out.display(), album.display()))?;
    println!("  {} imported into album {} ({n} file(s))", style("↳").cyan(), album.display());
    Ok(())
}

#[cfg(not(feature = "photos"))]
fn import_ornament(_out: &std::path::Path, _album: &std::path::Path) -> Result<()> {
    println!("  {} --import needs the `photos` feature (not compiled in) — skipped", style("·").yellow());
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
