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
    do_render(mk("frontispiece", Some("diffusion")), &a.out.join(front), &a.model, base_seed, a.steps, a.svg, 1).await.context("frontispiece")?;

    let mut all_files = vec![front.to_string()];
    let mut tex_ch = Vec::new();
    let mut man_ch = Vec::new();
    for (i, ch) in chapters.iter().enumerate() {
        let n = i + 1;
        let (hseed, tseed) = (kit::ornament_seed(base_seed, i * 2 + 1), kit::ornament_seed(base_seed, i * 2 + 2));
        let (hfile, tfile) = (format!("ch{n:02}_headpiece.png"), format!("ch{n:02}_tailpiece.png"));
        println!("\n{} [ch {n}/{}] {}  (headpiece seed {hseed})", style("→").cyan(), chapters.len(), ch.title);
        // headpiece: a wide diffusion banner, seed-varied per chapter (a variation of the shared motif).
        do_render(mk("headpiece", Some("diffusion")), &a.out.join(&hfile), &a.model, hseed, a.steps, a.svg, 1).await.with_context(|| format!("chapter {n} headpiece"))?;
        // tailpiece: procedural (fast, no weights).
        do_render(mk("tailpiece", Some("procedural")), &a.out.join(&tfile), &a.model, tseed, a.steps, a.svg, 1).await.with_context(|| format!("chapter {n} tailpiece"))?;
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
        do_render(per, &file, &a.model, seed_i, a.steps, a.svg, 1).await.with_context(|| format!("kit ornament {i} ({kind})"))?;
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
    let spec = BookArtSpec::load(&a.spec)?;
    do_render(spec, &a.out, &a.model, a.seed, a.steps, a.svg, a.attempts).await
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
    do_render(spec, &a.out, &a.model, a.seed, a.steps, false, a.attempts).await
}

/// The shared render core (used by `render` and `illustrate`): resolve → tier → finish → symmetry →
/// page canvas → PNG (+ opt-in born-vector SVG for the procedural tier).
async fn do_render(spec: BookArtSpec, out: &std::path::Path, model: &str, seed: u64, steps: usize, svg: bool, attempts: u32) -> Result<()> {
    use crate::bookart::{finish, geometry, procedural, scorecard};
    let plan = compile::resolve(&spec);
    let tb = geometry::text_block(&plan.page, &spec);
    let layout = geometry::layout_for(&plan.ornament_kind, &tb);
    let r0 = layout.rects[0];

    let ornament = match plan.tier.as_str() {
        // B3: vector-native, no weights.
        "procedural" => {
            let gray = procedural::generate(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h);
            finish::finish_procedural(&gray, &plan)
        }
        // B4: diffusion + optional matte, with B6 scorecard rejection sampling.
        "diffusion" => {
            let (gw, gh) = gen_size(r0.w, r0.h);
            let tries = attempts.max(1);
            let (mut best, mut fewest) = (None, usize::MAX);
            for i in 0..tries {
                let (raw, tmp) = diffuse(model, &plan, gw, gh, steps, seed + i as u64).await?;
                let finished = if plan.transparency_mode == "matte" {
                    println!("  {} U2Net matte → silhouette", style("↳").cyan());
                    matte_silhouette(&tmp, &raw, &plan).await?
                } else {
                    finish::finish_ornament(&raw, &plan)
                };
                let _ = std::fs::remove_file(&tmp);
                let sc = scorecard::score(&finished, &plan);
                if sc.passes {
                    if tries > 1 {
                        println!("  {} scorecard PASS on attempt {}/{}", style("✓").green(), i + 1, tries);
                    }
                    best = Some(finished);
                    break;
                }
                if sc.notes.len() < fewest {
                    fewest = sc.notes.len();
                    best = Some(finished);
                }
                if tries > 1 {
                    println!("  {} attempt {}/{} FAIL ({} issue(s)), retrying…", style("·").yellow(), i + 1, tries, sc.notes.len());
                }
            }
            best.context("no diffusion image")?
        }
        // B5: composite — procedural frame + diffusion line-art inlay.
        "composite" => {
            let (frame_paths, (wx, wy, ww, wh)) = procedural::frame(&plan.symmetry, r0.w, r0.h);
            let width = (r0.w.min(r0.h) as f32 * 0.004).max(1.5);
            let frame_rgba = finish::finish_procedural(&procedural::rasterise(&frame_paths, r0.w, r0.h, width), &plan);
            let (gw, gh) = gen_size(ww, wh);
            let (raw, tmp) = diffuse(model, &plan, gw, gh, steps, seed).await?;
            let _ = std::fs::remove_file(&tmp);
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

    // Symmetry (no-op for `none`); skipped for `composite` (frame already symmetric; picture is a scene).
    let orn = if plan.tier == "composite" { ornament } else { geometry::symmetrize(&ornament, &plan.symmetry) };
    let page = finish::canvas::place_on_canvas(&orn, &plan.page, &layout);
    finish::canvas::save_png_dpi(&page, out, plan.page.dpi)?;
    println!(
        "{} {}  ({} × {} px @ {} DPI · {} · {} · {} · {} piece(s))",
        style("wrote").green(),
        out.display(),
        page.width(),
        page.height(),
        plan.page.dpi,
        plan.tier,
        plan.ornament_kind,
        plan.symmetry,
        layout.rects.len()
    );

    // Opt-in born-vector SVG (§7.5) — procedural only; the raster trace is a documented fast-follow.
    if svg || plan.formats.iter().any(|f| f == "svg") {
        if plan.tier == "procedural" {
            let paths = procedural::generate_paths(&plan.ornament_kind, &plan.symmetry, r0.w, r0.h);
            let stroke = (r0.w.min(r0.h) as f32 * 0.004).max(1.5);
            let all: Vec<_> = layout.rects.iter().flat_map(|r| finish::vector::transform_to_rect(&paths, r, r0.w, r0.h)).collect();
            let svg_str = finish::vector::polylines_to_svg(&all, plan.page.w_px, plan.page.h_px, plan.page.dpi, stroke, finish::parse_tint(&plan.tint));
            let svg_path = out.with_extension("svg");
            std::fs::write(&svg_path, svg_str).with_context(|| format!("writing {}", svg_path.display()))?;
            println!("  {} born-vector SVG → {}", style("↳").cyan(), svg_path.display());
        } else {
            println!("  {} SVG for the `{}` tier (raster trace) is a fast-follow — the PNG is the deliverable (§7.5)", style("·").yellow(), plan.tier);
        }
    }
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
