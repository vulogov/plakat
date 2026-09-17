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
    /// Export a set of small procedural ornaments (fleurons/dinkus) as an OpenType dingbat font (B4) —
    /// type a letter (`a`–`h`), get an ornament. For inline use in InDesign / LaTeX.
    Font(FontArgs),
    /// Wrap a rendered **border** ornament into a self-contained Typst artifact — a bordered page plus a
    /// placement API (`#framed[...]`, `#place-on-page(...)`) — and, with `--verify`, compile it to PDF.
    Typst(TypstArgs),
    /// Generate an old-style (letterpress) book / chapter **title page** from an HJSON spec — hierarchical
    /// centred type, an imprint at the foot, an optional ornamental border/emblem — as a compilable Typst
    /// artifact usable in a Typst book. `--verify` compiles it to PDF.
    #[command(name = "title-page")]
    TitlePage(TitlePageArgs),
    /// Generate a book **cover / dust jacket** from an HJSON spec — the three panels (back · spine · front)
    /// laid flat on one sheet, with the **spine width computed from the page count**, optional flaps, and
    /// fold guides. Reuses the title-page styles. A compilable Typst artifact; `--verify` compiles to PDF.
    Cover(CoverArgs),
    /// Assemble a whole typeset **book** from a Markdown manuscript — a `#include`d title page, chapter
    /// openers (headpiece · CHAPTER N · title), body prose with a raised initial, running heads + folios,
    /// tailpieces, and a colophon → one compilable Typst file. `--verify` compiles to PDF.
    Book(BookArgs),
    /// Tile a motif into a seamless **endpaper** / decorative diaper (grid · half-drop · diamond) across a
    /// page → a print-sized PNG. Weight-free (pure compositing); motifs clip at the trim.
    Endpaper(EndpaperArgs),
}

#[derive(Args, Debug)]
pub struct TitlePageArgs {
    /// The title-page HJSON spec (`style`, `border`, `lines: [{role, text|src}]`). See BOOKART docs.
    pub spec: PathBuf,
    /// Output Typst file (`.typ`). Referenced assets (border, ornaments) are copied beside it.
    #[arg(long)]
    pub out: PathBuf,
    /// Page size (`a5`/`a4`/`b5`/…). Overrides the spec's `page`.
    #[arg(long)]
    pub page: Option<String>,
    /// Typographic style: `letterpress` (default) · `engraved` · `modern` · `playbill`. Overrides the spec's `style`.
    #[arg(long)]
    pub style: Option<String>,
    /// Type margin from the page edge, in mm (when there is no border). Default 22.
    #[arg(long, default_value_t = 22.0)]
    pub margin: f32,
    /// Shrink the type (and any plates) just enough to fit ONE page — measured with `typst`. Kills the
    /// silent 2-page overflow when a spec is too tall for its page.
    #[arg(long, default_value_t = false)]
    pub fit: bool,
    /// Historical typography: old-style figures + historical ligatures (also settable as `historical` in the spec).
    #[arg(long, default_value_t = false)]
    pub historical: bool,
    /// After writing, compile to PDF with `typst` to verify it renders.
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug)]
pub struct CoverArgs {
    /// The cover HJSON spec (`style`, `page`, `pages`, `front`/`spine`/`back: [{role,text|src}]`). See BOOKART docs.
    pub spec: PathBuf,
    /// Output Typst file (`.typ`). Referenced assets (front image) are copied beside it.
    #[arg(long)]
    pub out: PathBuf,
    /// Trim (page) size (`a5`/`b5`/…). Overrides the spec's `page`.
    #[arg(long)]
    pub page: Option<String>,
    /// Typographic style: `letterpress` (default) · `engraved` · `modern` · `playbill`. Overrides the spec's `style`.
    #[arg(long)]
    pub style: Option<String>,
    /// Page count — the spine width is `pages × paper (+ board)`. Overrides the spec's `pages`.
    #[arg(long)]
    pub pages: Option<u32>,
    /// Paper caliper in mm per page (default 0.06 ≈ 80–90 gsm text). Overrides the spec's `paper`.
    #[arg(long)]
    pub paper: Option<f32>,
    /// Flap width in mm (0 = a plain paperback cover; a positive value adds jacket flaps). Overrides the spec's `flap`.
    #[arg(long)]
    pub flap: Option<f32>,
    /// Historical typography: old-style figures + historical ligatures (also settable as `historical` in the spec).
    #[arg(long, default_value_t = false)]
    pub historical: bool,
    /// After writing, compile to PDF with `typst` to verify it renders.
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug)]
pub struct BookArgs {
    /// The Markdown manuscript (a `#`/`##` line opens a chapter; blank lines separate paragraphs).
    pub manuscript: PathBuf,
    /// Output Typst file (`.typ`). Referenced assets (title page, ornaments) are copied beside it.
    #[arg(long)]
    pub out: PathBuf,
    /// Page size (`a5`/`a4`/`b5`/…). Default a5.
    #[arg(long)]
    pub page: Option<String>,
    /// A `bookart title-page` artifact (`.typ`) to render as the first leaf.
    #[arg(long = "title-page")]
    pub title_page: Option<PathBuf>,
    /// Running-head text (usually the book title). Empty = no running head.
    #[arg(long)]
    pub running_head: Option<String>,
    /// A headpiece image atop every chapter opener (a kit ornament / device PNG).
    #[arg(long)]
    pub headpiece: Option<PathBuf>,
    /// A tailpiece image centred at the end of every chapter.
    #[arg(long)]
    pub tailpiece: Option<PathBuf>,
    /// A divider image for scene breaks (`***`); absent = a typographic asterism.
    #[arg(long)]
    pub divider: Option<PathBuf>,
    /// A colophon line, small-caps, centred on the final leaf.
    #[arg(long)]
    pub colophon: Option<String>,
    /// After writing, compile to PDF with `typst` to verify it renders.
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug)]
pub struct EndpaperArgs {
    /// The motif image (a transparent B/W ornament — a `bookart render`/`kit` fleuron, dinkus, …).
    #[arg(long)]
    pub motif: PathBuf,
    /// Output PNG (the print-sized endpaper).
    #[arg(long)]
    pub out: PathBuf,
    /// Page size (`a5`/`a4`/`b5`/…). Default a5.
    #[arg(long)]
    pub page: Option<String>,
    /// Repeat lattice: `grid` (default) · `half-drop` · `diamond`.
    #[arg(long, default_value = "grid")]
    pub layout: String,
    /// Motif width in mm. Default 22.
    #[arg(long, default_value_t = 22.0)]
    pub tile: f32,
    /// Gap between motif cells in mm. Default 10.
    #[arg(long, default_value_t = 10.0)]
    pub gap: f32,
    /// Background fill as `#rrggbb` (else transparent). E.g. `#f4efe6` for a warm laid paper.
    #[arg(long)]
    pub bg: Option<String>,
}

#[derive(Args, Debug)]
pub struct TypstArgs {
    /// Full-page border ornament (a bookart-rendered `border` PNG; SVG works too). The text box is fitted to
    /// the border's measured clear window. Use this OR `--corner`.
    #[arg(long)]
    pub border: Option<PathBuf>,
    /// A small CORNER ornament (a bookart-rendered `corner` PNG). Builds a RESTRAINED frame — thin rules +
    /// this ornament mirrored at all four corners — that never dominates the page. Use this OR `--border`.
    #[arg(long)]
    pub corner: Option<PathBuf>,
    /// Size of each corner ornament, in mm (`--corner` mode).
    #[arg(long, default_value_t = 18.0)]
    pub corner_size: f32,
    /// Frame rule thickness, in pt (`--corner` mode); 0 draws corners only, no connecting rules.
    #[arg(long, default_value_t = 0.6)]
    pub rule: f32,
    /// Output Typst file (`.typ`). Referenced assets are copied beside it so it compiles anywhere.
    #[arg(long)]
    pub out: PathBuf,
    /// Page size (`a4`/`a5`/`a6`/`b5`/`letter`/…). Ignored when `--spec` is given.
    #[arg(long, default_value = "a5")]
    pub page: String,
    /// Read the page size from a bookart spec instead of `--page`.
    #[arg(long)]
    pub spec: Option<PathBuf>,
    /// Uniform page margin, in mm — the gap between the page edge and the border. The border is sized to
    /// the resulting margin box (`page − margins`). Per-side flags below override this default.
    #[arg(long, default_value_t = 12.0)]
    pub margin: f32,
    /// Top page margin, mm (overrides `--margin`).
    #[arg(long)]
    pub margin_top: Option<f32>,
    /// Bottom page margin, mm (overrides `--margin`).
    #[arg(long)]
    pub margin_bottom: Option<f32>,
    /// Left page margin, mm (overrides `--margin`).
    #[arg(long)]
    pub margin_left: Option<f32>,
    /// Right page margin, mm (overrides `--margin`).
    #[arg(long)]
    pub margin_right: Option<f32>,
    /// Extra safety clearance in mm added inside the border's MEASURED clear window (the text box is fitted
    /// to the ornament automatically; this is just breathing room on top).
    #[arg(long, default_value_t = 2.0)]
    pub safety: f32,
    /// Optional title, placed centred at the top of the safe area.
    #[arg(long)]
    pub title: Option<String>,
    /// Optional text file whose contents become the body (Typst markup allowed). Omit for a lorem stand-in.
    #[arg(long)]
    pub body: Option<PathBuf>,
    /// Optional image placed centred under the body — copied beside the artifact.
    #[arg(long)]
    pub image: Option<PathBuf>,
    /// Placed-image width, as a percentage of the content width.
    #[arg(long, default_value_t = 60)]
    pub image_width: u32,
    /// After writing, compile the artifact to a PDF with `typst` to verify it renders.
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug)]
pub struct FontArgs {
    /// Output font path (`.otf` / `.ttf`).
    #[arg(long)]
    pub out: PathBuf,
    /// The font family name embedded in the file.
    #[arg(long, default_value = "PlakatDingbats")]
    pub family: String,
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
    /// B2: a TrueType/OpenType font for a glyph-driven `initial` (any script; needs `shaped-labels`).
    #[arg(long)]
    pub font: Option<PathBuf>,
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
    /// B2: a TrueType/OpenType font for a glyph-driven `initial` (needs `shaped-labels`).
    #[arg(long)]
    pub font: Option<PathBuf>,
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
        BookartCmd::Font(a) => run_font(a),
        BookartCmd::Typst(a) => run_typst(a),
        BookartCmd::TitlePage(a) => run_title_page(a),
        BookartCmd::Cover(a) => run_cover(a),
        BookartCmd::Book(a) => run_book(a),
        BookartCmd::Endpaper(a) => run_endpaper(a),
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
    // B6: an `.epub` is parsed via its spine/TOC (feature `epub`); anything else is Markdown / a plain list.
    let is_epub = a.book.extension().is_some_and(|e| e.eq_ignore_ascii_case("epub"));
    let chapters = if is_epub {
        parse_epub_book(&a.book)?
    } else {
        let text = std::fs::read_to_string(&a.book).with_context(|| format!("reading {}", a.book.display()))?;
        manuscript::parse_chapters(&text)
    };
    if chapters.is_empty() {
        anyhow::bail!("no chapters found in {} (Markdown `#` headings, one title per line, or an EPUB TOC)", a.book.display());
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
    do_render(mk("frontispiece", Some("diffusion")), &a.out.join(front), &a.model, base_seed, a.steps, a.svg, 1, None, false, None).await.context("frontispiece")?;

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
        do_render(mk("headpiece", Some("procedural")), &a.out.join(&hfile), &a.model, hseed, a.steps, a.svg, 1, None, false, None).await.with_context(|| format!("chapter {n} headpiece"))?;
        // tailpiece: a procedural cul-de-lampe, also varied per chapter.
        do_render(mk("tailpiece", Some("procedural")), &a.out.join(&tfile), &a.model, tseed, a.steps, a.svg, 1, None, false, None).await.with_context(|| format!("chapter {n} tailpiece"))?;
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
        do_render(per, &file, &a.model, seed_i, a.steps, a.svg, 1, None, false, None).await.with_context(|| format!("kit ornament {i} ({kind})"))?;
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
    do_render(spec, &a.out, &a.model, a.seed, a.steps, a.svg, a.attempts, a.import.as_deref(), a.cache_raw, a.font.clone()).await
}

async fn run_illustrate(a: IllustrateArgs) -> Result<()> {
    use crate::bookart::spec::{BookArtSpec, Ornament, Page};
    // A single B/W plate: synthesise a diffusion-tier spec from the prompt + flags. D1: for `--type
    // initial` the prompt's first letter becomes the glyph, so `--font` renders a real historiated
    // initial (was a dead flag on `illustrate`); other types render the prompt as a pictorial plate.
    let glyph = (a.kind == "initial")
        .then(|| a.prompt.chars().find(|c| c.is_alphabetic()).map(|c| c.to_string()))
        .flatten();
    let spec = BookArtSpec {
        schema: Some(crate::bookart::SCHEMA_VERSION.into()),
        origin: Some(a.origin),
        technique: Some(a.technique),
        page: Some(Page { size: Some(a.page), ..Default::default() }),
        ornament: Some(Ornament { kind: Some(a.kind), tier: Some("diffusion".into()), prompt: Some(a.prompt), glyph, ..Default::default() }),
        ..Default::default()
    };
    do_render(spec, &a.out, &a.model, a.seed, a.steps, false, a.attempts, a.import.as_deref(), a.cache_raw, a.font.clone()).await
}

/// The shared render entry (used by `render`, `illustrate`, `kit`, `manuscript`): drive the library
/// render core ([`crate::bookart::render::render_spec`]), then write the PNG (+ opt-in SVG) to disk.
#[allow(clippy::too_many_arguments)]
async fn do_render(spec: BookArtSpec, out: &std::path::Path, model: &str, seed: u64, steps: usize, svg: bool, attempts: u32, import: Option<&std::path::Path>, cache_raw: bool, font: Option<PathBuf>) -> Result<()> {
    use crate::bookart::render::{recipe_metadata, render_spec, RenderOpts};
    let r = render_spec(&spec, &RenderOpts { model: model.into(), seed, steps, svg, attempts, font }).await?;
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

/// B6: parse an EPUB's chapters (feature `epub`); a clear note when the feature isn't compiled in.
#[cfg(feature = "epub")]
fn parse_epub_book(path: &std::path::Path) -> Result<Vec<crate::bookart::manuscript::Chapter>> {
    crate::bookart::epub::parse_epub_chapters(path)
}

#[cfg(not(feature = "epub"))]
fn parse_epub_book(_path: &std::path::Path) -> Result<Vec<crate::bookart::manuscript::Chapter>> {
    anyhow::bail!(
        "EPUB input needs the `epub` feature — rebuild with `--features epub` (it pulls a zip/deflate \
         stack). Or export the book's chapter list to Markdown / one-title-per-line."
    )
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

/// B4: `bookart font` — export the default dingbat set as an OpenType font.
fn run_font(a: FontArgs) -> Result<()> {
    let set = crate::bookart::font::default_set();
    let bytes = crate::bookart::font::build_font(&set, &a.family).context("building the dingbat font")?;
    std::fs::write(&a.out, &bytes).with_context(|| format!("writing {}", a.out.display()))?;
    let glyphs: String = set.iter().map(|d| d.ch).collect();
    println!(
        "{} {}  ({} · {} glyph(s) [{}] · {:.1} KB)",
        style("wrote").green(),
        a.out.display(),
        a.family,
        set.len(),
        glyphs,
        bytes.len() as f32 / 1024.0
    );
    Ok(())
}

/// `bookart typst` — wrap a border ornament into a self-contained, PDF-compilable Typst artifact:
/// a page whose background IS the border, plus `#framed[...]` / `#place-on-page(...)` to put content on
/// top. Copies the assets beside the `.typ` (Typst's root sandbox needs them local) and, with `--verify`,
/// compiles it to a PDF to prove it renders.
fn run_typst(a: TypstArgs) -> Result<()> {
    use crate::bookart::spec::Page;
    use crate::bookart::{geometry, typst as typ, BookArtSpec};

    // 1. Resolve the physical page size — from a spec's page block, or the `--page` name.
    let page_res = if let Some(spec_path) = &a.spec {
        let spec = BookArtSpec::load(spec_path)?;
        geometry::resolve_page(spec.page.as_ref())
    } else {
        geometry::resolve_page(Some(&Page { size: Some(a.page.clone()), ..Default::default() }))
    };

    // The artifact's directory — assets must sit beside the `.typ` so Typst's root sandbox resolves them.
    let art_dir = a
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&art_dir).with_context(|| format!("creating {}", art_dir.display()))?;

    // 2. Exactly one frame source, plus the shared content + margins.
    anyhow::ensure!(
        a.border.is_some() ^ a.corner.is_some(),
        "give exactly one of --border (full ornament, text fitted to its clear window) or --corner \
         (a small ornament framed at the four corners — a restrained frame that never dominates)",
    );
    let image_ref = match &a.image {
        Some(img) => Some(copy_beside(img, &art_dir).context("copying the placed image")?),
        None => None,
    };
    let body = match &a.body {
        Some(p) => Some(std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?),
        None => None,
    };
    let place = typ::Placement { title: a.title.clone(), body, image_ref, image_width_pct: a.image_width };
    // Per-side page margins fall back to the uniform --margin.
    let m = a.margin.max(0.0);
    let margin = typ::Margins {
        top: a.margin_top.unwrap_or(m).max(0.0),
        bottom: a.margin_bottom.unwrap_or(m).max(0.0),
        left: a.margin_left.unwrap_or(m).max(0.0),
        right: a.margin_right.unwrap_or(m).max(0.0),
    };

    // 3. Emit — a RESTRAINED corner frame, or a full border with the text fitted to its clear window.
    if let Some(corner_path) = &a.corner {
        // `bookart corner` renders all four corners on one page; crop a single tile so it can be mirrored
        // into a matching frame at the chosen size.
        let corner_ref = prepare_corner_tile(corner_path, &art_dir).context("preparing the corner tile")?;
        let cs = a.corner_size.max(1.0);
        let (tw, th) = (
            page_res.w_mm - margin.left - margin.right - 2.0 * cs,
            page_res.h_mm - margin.top - margin.bottom - 2.0 * cs,
        );
        anyhow::ensure!(
            tw >= 20.0 && th >= 20.0,
            "corners of {cs:.0} mm + margins leave only a {tw:.0}×{th:.0} mm text area — shrink --corner-size or --margin",
        );
        let frame = typ::CornerFrame {
            w_mm: page_res.w_mm,
            h_mm: page_res.h_mm,
            corner_ref,
            corner_mm: cs,
            rule_pt: a.rule.max(0.0),
            margin,
        };
        let src = typ::typst_corner_frame(&frame, &place);
        std::fs::write(&a.out, &src).with_context(|| format!("writing {}", a.out.display()))?;
        println!(
            "{} {}  ({} · page {}×{} mm · corner frame {:.0} mm · text {:.0}×{:.0} mm — text keeps the interior)",
            style("wrote").green(),
            a.out.display(),
            page_res.size_name,
            page_res.w_mm.round() as i32,
            page_res.h_mm.round() as i32,
            cs,
            tw,
            th,
        );
    } else {
        let border_path = a.border.as_ref().expect("border present (xor validated above)");
        let border_ref = copy_beside(border_path, &art_dir).context("copying the border image")?;
        let (bw, bh) = (page_res.w_mm - margin.left - margin.right, page_res.h_mm - margin.top - margin.bottom);
        anyhow::ensure!(
            bw > 0.0 && bh > 0.0,
            "margins ({:.0}/{:.0}/{:.0}/{:.0} mm) exceed the {} page ({:.0}×{:.0} mm) — the border would have no size",
            margin.top, margin.bottom, margin.left, margin.right, page_res.size_name, page_res.w_mm, page_res.h_mm,
        );
        // TEXT is the subject: MEASURE the border's clear window and fit the text box to it, so a page is
        // never generated with text over the ornament. A raster is measured; SVG/unreadable → a proportional
        // 12% window (a safe estimate) with a warning.
        let window = measure_clear_window(border_path);
        let text_m = typ::text_margins_from_window(page_res.w_mm, page_res.h_mm, &margin, window, a.safety.max(0.0));
        let (tw, th) = (page_res.w_mm - text_m.left - text_m.right, page_res.h_mm - text_m.top - text_m.bottom);
        anyhow::ensure!(
            tw >= 20.0 && th >= 20.0,
            "the border leaves only a {tw:.0}×{th:.0} mm text area — use a lighter border (or --corner), a larger page, or smaller --margin",
        );
        let page = typ::TypstPage { w_mm: page_res.w_mm, h_mm: page_res.h_mm, border_ref, border: margin, text: text_m };
        let src = typ::typst_artifact(&page, &place);
        std::fs::write(&a.out, &src).with_context(|| format!("writing {}", a.out.display()))?;
        println!(
            "{} {}  ({} · page {}×{} mm · border {:.0}×{:.0} mm · text {:.0}×{:.0} mm — fitted to the border)",
            style("wrote").green(),
            a.out.display(),
            page_res.size_name,
            page_res.w_mm.round() as i32,
            page_res.h_mm.round() as i32,
            bw,
            bh,
            tw,
            th,
        );
    }

    // 4. Verify: compile to PDF with Typst, proving the artifact renders.
    if a.verify {
        verify_typst(&a.out, &a.out.with_extension("pdf"))?;
    } else {
        println!("   {} typst compile {}", style("verify:").dim(), a.out.display());
    }
    Ok(())
}

/// Crop a single corner tile from a `bookart corner` render (which lays all four corners on one page).
/// Takes the ink bounding box of the TOP-LEFT quadrant — one matching corner ornament — trims it, and saves
/// it beside the artifact. The emitter mirrors this one tile into all four corners, so they match exactly.
fn prepare_corner_tile(src: &std::path::Path, dir: &std::path::Path) -> Result<String> {
    let rgba = image::open(src).with_context(|| format!("opening {}", src.display()))?.to_rgba8();
    let (w, h) = rgba.dimensions();
    let (qw, qh) = (w / 2, h / 2);
    let ink = |x: u32, y: u32| {
        let p = rgba.get_pixel(x, y).0;
        let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        p[3] > 32 && luma < 128.0
    };
    let (mut x0, mut y0, mut x1, mut y1, mut found) = (qw, qh, 0u32, 0u32, false);
    for y in 0..qh {
        for x in 0..qw {
            if ink(x, y) {
                found = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    anyhow::ensure!(found, "no ornament found in the top-left quadrant of {}", src.display());
    // Square-pad the tile (ornament anchored at its top-left) so the emitter can place it at exact corner
    // coordinates and mirror it in place — the fitted, no-overflow placement.
    let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
    let side = bw.max(bh);
    let cropped = image::imageops::crop_imm(&rgba, x0, y0, bw, bh).to_image();
    let mut tile = image::RgbaImage::from_pixel(side, side, image::Rgba([0, 0, 0, 0]));
    image::imageops::overlay(&mut tile, &cropped, 0, 0);
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("corner");
    let name = format!("{stem}_tile.png");
    tile.save(dir.join(&name)).with_context(|| format!("writing the corner tile {name}"))?;
    Ok(name)
}

/// `bookart title-page` — an old-style letterpress title page from an HJSON spec → a compilable Typst
/// artifact. Resolves the page, copies any border/ornament assets beside the `.typ`, fits the type inside a
/// border's measured clear window (reusing the `typst` geometry), emits, and (with `--verify`) compiles.
fn run_title_page(a: TitlePageArgs) -> Result<()> {
    use crate::bookart::spec::Page;
    use crate::bookart::titlepage::TitlePageSpec;
    use crate::bookart::typst::{self as typ, Margins};
    use crate::bookart::geometry;

    // 1. Load the spec (permissive HJSON).
    let text = std::fs::read_to_string(&a.spec).with_context(|| format!("reading {}", a.spec.display()))?;
    let mut spec: TitlePageSpec =
        deser_hjson::from_str(&text).map_err(|e| anyhow::anyhow!("parsing title-page spec {}: {e}", a.spec.display()))?;

    // 2. Resolve the page size — CLI --page wins, else the spec's `page`, else a5.
    let size_name = a.page.clone().or_else(|| spec.page.clone()).unwrap_or_else(|| "a5".into());
    let page_res = geometry::resolve_page(Some(&Page { size: Some(size_name), ..Default::default() }));

    // 3. Assets beside the artifact: the border + every line's image src (copied, referenced by basename).
    let art_dir = a
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&art_dir).with_context(|| format!("creating {}", art_dir.display()))?;
    let border_ref = match &spec.border {
        Some(b) => Some(copy_beside(std::path::Path::new(b), &art_dir).context("copying the border image")?),
        None => None,
    };
    for line in &mut spec.lines {
        if let Some(src) = &line.src {
            // `image`/`ornament` lines are CROPPED to their ink — a `bookart render` ornament comes on a
            // full page canvas, so tight-crop it to the device before placing it inline.
            let role = line.role.trim().to_lowercase();
            let rel = if role == "image" || role == "ornament" {
                crop_to_ink(std::path::Path::new(src), &art_dir).with_context(|| format!("cropping ornament {src}"))?
            } else {
                copy_beside(std::path::Path::new(src), &art_dir).with_context(|| format!("copying {src}"))?
            };
            line.src = Some(rel);
        }
    }

    // 4. Type box: fitted to the border's measured clear window, else a plain margin.
    let uniform = |v: f32| Margins { top: v, bottom: v, left: v, right: v };
    let (border, text_margin);
    let bmargin;
    if let (Some(bref), Some(bpath)) = (&border_ref, &spec.border) {
        let bm = uniform(12.0);
        let window = measure_clear_window(std::path::Path::new(bpath));
        let tm = typ::text_margins_from_window(page_res.w_mm, page_res.h_mm, &bm, window, 6.0);
        bmargin = bm;
        text_margin = tm;
        border = Some((bref.as_str(), &bmargin));
    } else {
        border = None;
        text_margin = uniform(a.margin.max(0.0));
    }

    let rule_pt = spec.rule.unwrap_or(0.0).max(0.0);
    let historical = a.historical || spec.historical.unwrap_or(false);
    let style_name = a.style.clone().or_else(|| spec.style.clone()).unwrap_or_else(|| "letterpress".into());
    let tp_style = crate::bookart::titlepage::Style::from_name(&style_name);
    let emit = |scale: f32| {
        crate::bookart::titlepage::title_page_typst(
            page_res.w_mm,
            page_res.h_mm,
            &text_margin,
            border,
            rule_pt,
            spec.font.as_deref(),
            &spec.lines,
            crate::bookart::titlepage::Emit { scale, historical, style: tp_style },
        )
    };

    // Emit — with `--fit`, shrink the type just enough to fit one page (measured by `typst`).
    let mut scale = 1.0f32;
    if a.fit {
        scale = fit_to_one_page(&a.out, &emit)?;
    } else {
        std::fs::write(&a.out, emit(1.0)).with_context(|| format!("writing {}", a.out.display()))?;
    }

    let (tw, th) = (page_res.w_mm - text_margin.left - text_margin.right, page_res.h_mm - text_margin.top - text_margin.bottom);
    let extras = format!(
        "{}{}",
        if scale < 0.999 { format!(" · fitted {:.0}%", scale * 100.0) } else { String::new() },
        if historical { " · historical" } else { "" },
    );
    println!(
        "{} {}  ({} · {} · page {}×{} mm · type {:.0}×{:.0} mm · {} line(s){})",
        style("wrote").green(),
        a.out.display(),
        page_res.size_name,
        style_name,
        page_res.w_mm.round() as i32,
        page_res.h_mm.round() as i32,
        tw,
        th,
        spec.lines.len(),
        extras,
    );

    if a.verify {
        verify_typst(&a.out, &a.out.with_extension("pdf"))?;
        // Even when it compiles, a too-tall page silently spills onto a second sheet. Warn (unless --fit
        // already guaranteed one page) so the overflow is never invisible.
        if !a.fit {
            if let Ok(n) = typst_pages(&a.out) {
                if n > 1 {
                    eprintln!(
                        "{}  {} rendered {} pages — the type overflows one sheet; re-run with --fit (auto-shrink) or reduce sizes",
                        style("⚠").yellow(),
                        a.out.display(),
                        n,
                    );
                }
            }
        }
    } else {
        println!("   {} typst compile {}", style("verify:").dim(), a.out.display());
    }
    Ok(())
}

/// `bookart cover` — a book cover / dust jacket from an HJSON spec → a compilable Typst artifact. Resolves
/// the trim size, computes the spine width from the page count, copies any front image beside the `.typ`,
/// emits the three-panel layout, and (with `--verify`) compiles it.
fn run_cover(a: CoverArgs) -> Result<()> {
    use crate::bookart::spec::Page;
    use crate::bookart::cover::{cover_typst, spine_width_mm, CoverLayout, CoverSpec};
    use crate::bookart::geometry;

    let text = std::fs::read_to_string(&a.spec).with_context(|| format!("reading {}", a.spec.display()))?;
    let mut spec: CoverSpec =
        deser_hjson::from_str(&text).map_err(|e| anyhow::anyhow!("parsing cover spec {}: {e}", a.spec.display()))?;

    // Trim size.
    let size_name = a.page.clone().or_else(|| spec.page.clone()).unwrap_or_else(|| "a5".into());
    let page_res = geometry::resolve_page(Some(&Page { size: Some(size_name), ..Default::default() }));

    // Spine width: explicit override, else pages × caliper (+ board).
    let pages = a.pages.or(spec.pages).unwrap_or(200);
    let paper = a.paper.or(spec.paper).unwrap_or(0.06);
    let board = spec.board.unwrap_or(0.0);
    let spine_w = spec.spine_mm.unwrap_or_else(|| spine_width_mm(pages, paper, board));
    let flap_w = a.flap.or(spec.flap).unwrap_or(0.0).max(0.0);

    // Assets beside the artifact: the front image + every panel line's image src (cropped/copied).
    let art_dir = a
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&art_dir).with_context(|| format!("creating {}", art_dir.display()))?;
    let front_bg = match &spec.border {
        Some(b) => Some(copy_beside(std::path::Path::new(b), &art_dir).context("copying the front image")?),
        None => None,
    };
    for panel in [&mut spec.front, &mut spec.spine, &mut spec.back] {
        for line in panel.iter_mut() {
            if let Some(src) = &line.src {
                let role = line.role.trim().to_lowercase();
                let rel = if role == "image" || role == "ornament" {
                    crop_to_ink(std::path::Path::new(src), &art_dir).with_context(|| format!("cropping ornament {src}"))?
                } else {
                    copy_beside(std::path::Path::new(src), &art_dir).with_context(|| format!("copying {src}"))?
                };
                line.src = Some(rel);
            }
        }
    }

    let style_name = a.style.clone().or_else(|| spec.style.clone()).unwrap_or_else(|| "letterpress".into());
    let tp_style = crate::bookart::titlepage::Style::from_name(&style_name);
    let historical = a.historical || spec.historical.unwrap_or(false);
    let emit = crate::bookart::titlepage::Emit { scale: 1.0, historical, style: tp_style };

    let layout = CoverLayout { trim_w: page_res.w_mm, trim_h: page_res.h_mm, spine_w, flap_w };
    let src = cover_typst(&layout, &spec.front, &spec.spine, &spec.back, front_bg.as_deref(), emit);
    std::fs::write(&a.out, &src).with_context(|| format!("writing {}", a.out.display()))?;

    println!(
        "{} {}  ({} · {} · trim {:.0}×{:.0} mm · spine {:.1} mm ({} pp) · total {:.0}×{:.0} mm{})",
        style("wrote").green(),
        a.out.display(),
        page_res.size_name,
        style_name,
        page_res.w_mm.round() as i32,
        page_res.h_mm.round() as i32,
        spine_w,
        pages,
        layout.total_w().round() as i32,
        page_res.h_mm.round() as i32,
        if flap_w > 0.01 { format!(" · flaps {:.0} mm", flap_w) } else { String::new() },
    );

    if a.verify {
        verify_typst(&a.out, &a.out.with_extension("pdf"))?;
    } else {
        println!("   {} typst compile {}", style("verify:").dim(), a.out.display());
    }
    Ok(())
}

/// `bookart book` — assemble a whole typeset book from a Markdown manuscript → a compilable Typst file.
fn run_book(a: BookArgs) -> Result<()> {
    use crate::bookart::spec::Page;
    use crate::bookart::book::{book_typ, parse_manuscript, Block, BookOpts};
    use crate::bookart::geometry;

    let md = std::fs::read_to_string(&a.manuscript).with_context(|| format!("reading {}", a.manuscript.display()))?;
    let (front, chapters) = parse_manuscript(&md);
    anyhow::ensure!(!chapters.is_empty(), "no chapters found in {} (a chapter is a line beginning `#` or `##`)", a.manuscript.display());

    let size_name = a.page.clone().unwrap_or_else(|| "a5".into());
    let page_res = geometry::resolve_page(Some(&Page { size: Some(size_name), ..Default::default() }));

    let art_dir = a
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    std::fs::create_dir_all(&art_dir).with_context(|| format!("creating {}", art_dir.display()))?;

    // The title page is `#include`d — Typst resolves ITS image paths relative to itself, so it must sit in
    // the book's directory with its assets. Same-dir → reference by basename; else copy it beside and warn.
    let same_dir = |p: &std::path::Path| -> bool {
        match (p.parent().and_then(|d| std::fs::canonicalize(d).ok()), std::fs::canonicalize(&art_dir).ok()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    };
    let title_ref = match &a.title_page {
        Some(tp) => {
            if same_dir(tp) {
                Some(tp.file_name().unwrap().to_string_lossy().into_owned())
            } else {
                eprintln!(
                    "{}  the title page {} isn't beside the book — copying it; make sure its images are alongside too",
                    style("⚠").yellow(),
                    tp.display(),
                );
                Some(copy_beside(tp, &art_dir).context("copying the title page")?)
            }
        }
        None => None,
    };
    let headpiece = match &a.headpiece {
        Some(p) => Some(crop_to_ink(p, &art_dir).context("cropping the headpiece")?),
        None => None,
    };
    let tailpiece = match &a.tailpiece {
        Some(p) => Some(crop_to_ink(p, &art_dir).context("cropping the tailpiece")?),
        None => None,
    };
    let divider = match &a.divider {
        Some(p) => Some(crop_to_ink(p, &art_dir).context("cropping the divider")?),
        None => None,
    };

    let opts = BookOpts {
        w_mm: page_res.w_mm,
        h_mm: page_res.h_mm,
        running_head: a.running_head.as_deref().unwrap_or(""),
        title_page: title_ref.as_deref(),
        headpiece: headpiece.as_deref(),
        tailpiece: tailpiece.as_deref(),
        divider: divider.as_deref(),
        colophon: a.colophon.as_deref().unwrap_or(""),
    };
    let src = book_typ(&front, &chapters, &opts);
    std::fs::write(&a.out, &src).with_context(|| format!("writing {}", a.out.display()))?;

    let words: usize = chapters
        .iter()
        .flat_map(|c| c.blocks.iter())
        .map(|b| match b {
            Block::Para(t) | Block::Section(t) => t.split_whitespace().count(),
            Block::Quote(ps) => ps.iter().map(|p| p.split_whitespace().count()).sum(),
            Block::SceneBreak => 0,
        })
        .sum();
    println!(
        "{} {}  ({} · {}×{} mm · {} chapter(s) · ~{} words{}{})",
        style("wrote").green(),
        a.out.display(),
        page_res.size_name,
        page_res.w_mm.round() as i32,
        page_res.h_mm.round() as i32,
        chapters.len(),
        words,
        if title_ref.is_some() { " · title page" } else { "" },
        if headpiece.is_some() { " · headpieces" } else { "" },
    );

    if a.verify {
        verify_typst(&a.out, &a.out.with_extension("pdf"))?;
    } else {
        println!("   {} typst compile {}", style("verify:").dim(), a.out.display());
    }
    Ok(())
}

/// `bookart endpaper` — tile a motif into a seamless decorative diaper across a page → a print-sized PNG.
fn run_endpaper(a: EndpaperArgs) -> Result<()> {
    use crate::bookart::spec::Page;
    use crate::bookart::endpaper::{render_endpaper, EndpaperOpts, Layout};
    use crate::bookart::geometry;

    const DPI: f32 = 300.0;
    let mm_to_px = |mm: f32| (mm / 25.4 * DPI).round().max(1.0) as u32;

    let raw = image::open(&a.motif).with_context(|| format!("opening motif {}", a.motif.display()))?.to_rgba8();
    // Crop the motif to its ink so the tile is the device, not a mostly-empty canvas.
    let motif = crop_rgba_to_ink(&raw);

    let size_name = a.page.clone().unwrap_or_else(|| "a5".into());
    let page_res = geometry::resolve_page(Some(&Page { size: Some(size_name), ..Default::default() }));

    let bg = match &a.bg {
        Some(hex) => Some(parse_hex_rgba(hex).with_context(|| format!("parsing --bg {hex}"))?),
        None => None,
    };
    let layout = Layout::from_name(&a.layout);
    let opts = EndpaperOpts {
        w: mm_to_px(page_res.w_mm),
        h: mm_to_px(page_res.h_mm),
        tile: mm_to_px(a.tile.max(1.0)),
        gap: mm_to_px(a.gap.max(0.0)),
        layout,
        bg,
    };
    let out = render_endpaper(&motif, &opts);
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;

    println!(
        "{} {}  ({} · {}×{} px @ {:.0} DPI · {} · tile {:.0} mm gap {:.0} mm{})",
        style("wrote").green(),
        a.out.display(),
        page_res.size_name,
        opts.w,
        opts.h,
        DPI,
        a.layout,
        a.tile,
        a.gap,
        if bg.is_some() { " · tinted" } else { " · transparent" },
    );
    Ok(())
}

/// Crop an in-memory RGBA image to its ink bounding box (non-transparent, non-near-white), with a small
/// transparent pad. A blank image is returned unchanged.
fn crop_rgba_to_ink(rgba: &image::RgbaImage) -> image::RgbaImage {
    let (w, h) = rgba.dimensions();
    let (mut x0, mut y0, mut x1, mut y1, mut found) = (w, h, 0u32, 0u32, false);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y).0;
            let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            if p[3] > 24 && luma < 240.0 {
                found = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if !found || x1 < x0 || y1 < y0 {
        return rgba.clone();
    }
    let pad = ((x1 - x0).max(y1 - y0) / 40).max(2);
    let cx0 = x0.saturating_sub(pad);
    let cy0 = y0.saturating_sub(pad);
    let cw = (x1 - cx0 + 1 + pad).min(w - cx0);
    let ch = (y1 - cy0 + 1 + pad).min(h - cy0);
    image::imageops::crop_imm(rgba, cx0, cy0, cw, ch).to_image()
}

/// Parse a `#rrggbb` (or `rrggbb`) colour into an opaque RGBA.
fn parse_hex_rgba(hex: &str) -> Result<[u8; 4]> {
    let h = hex.trim().trim_start_matches('#');
    anyhow::ensure!(h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit()), "expected #rrggbb, got `{hex}`");
    let byte = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).unwrap();
    Ok([byte(0), byte(2), byte(4), 255])
}

/// Count the pages a Typst file lays out, by rendering to a throwaway low-DPI PNG set and counting them.
/// Uses the real Typst layout (exact), reading the file's assets relative to the `.typ` as usual.
fn typst_pages(typ_path: &std::path::Path) -> Result<usize> {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("plakat-fit-{}-{}", std::process::id(), stamp));
    std::fs::create_dir_all(&dir)?;
    let tmpl = dir.join("p-{p}.png");
    let out = std::process::Command::new("typst")
        .arg("compile").arg(typ_path).arg(&tmpl).arg("--ppi").arg("10")
        .output();
    let result = (|| -> Result<usize> {
        let out = match out {
            Ok(o) => o,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                anyhow::bail!("typst not found on PATH");
            }
            Err(e) => return Err(anyhow::Error::from(e).context("running `typst compile`")),
        };
        anyhow::ensure!(out.status.success(), "typst could not compile:\n{}", String::from_utf8_lossy(&out.stderr).trim());
        let n = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .count();
        Ok(n.max(1))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Emit at successively smaller scales until the page lays out as ONE sheet (measured by `typst`), writing
/// the final artifact and returning the scale used. Falls back to 1.0 with a warning if `typst` is absent.
fn fit_to_one_page(out: &std::path::Path, emit: &dyn Fn(f32) -> String) -> Result<f32> {
    const FLOOR: f32 = 0.55;
    const STEP: f32 = 0.94;
    let mut scale = 1.0f32;
    loop {
        std::fs::write(out, emit(scale)).with_context(|| format!("writing {}", out.display()))?;
        match typst_pages(out) {
            Ok(1) => return Ok(scale),
            Ok(n) => {
                if scale > FLOOR {
                    scale *= STEP;
                    continue;
                }
                eprintln!(
                    "{}  still {} pages at the {:.0}% floor — the spec has more than one page of content; trim it",
                    style("⚠").yellow(),
                    n,
                    FLOOR * 100.0,
                );
                return Ok(scale);
            }
            Err(e) => {
                eprintln!("{}  --fit needs typst to measure pages ({e}); wrote at full size", style("⚠").yellow());
                std::fs::write(out, emit(1.0))?;
                return Ok(1.0);
            }
        }
    }
}

/// Tight-crop an ornament to its ink (its non-transparent bounding box) and save it beside the artifact,
/// returning the basename. A `bookart render` ornament arrives on a full page canvas; this reduces it to
/// the device so it places inline in a title page. A fully-opaque/blank image is copied as-is.
fn crop_to_ink(src: &std::path::Path, dir: &std::path::Path) -> Result<String> {
    let rgba = image::open(src).with_context(|| format!("opening {}", src.display()))?.to_rgba8();
    let (w, h) = rgba.dimensions();
    let (mut x0, mut y0, mut x1, mut y1, mut found) = (w, h, 0u32, 0u32, false);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y).0;
            // Content = a non-transparent pixel that isn't near-white (the ornament ink / mass).
            let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            if p[3] > 24 && luma < 240.0 {
                found = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    let name = format!("{}_crop.png", src.file_stem().and_then(|s| s.to_str()).unwrap_or("ornament"));
    let dest = dir.join(&name);
    if found && x1 >= x0 && y1 >= y0 {
        // A small transparent margin so the ink doesn't touch the box edge.
        let pad = ((x1 - x0).max(y1 - y0) / 40).max(2);
        let cx0 = x0.saturating_sub(pad);
        let cy0 = y0.saturating_sub(pad);
        let cw = (x1 - cx0 + 1 + pad).min(w - cx0);
        let ch = (y1 - cy0 + 1 + pad).min(h - cy0);
        image::imageops::crop_imm(&rgba, cx0, cy0, cw, ch).to_image().save(&dest)?;
    } else {
        rgba.save(&dest)?;
    }
    Ok(name)
}

/// Measure the border's inner clear window (fractions of its size) so the text box can be fitted to it.
/// A raster border is loaded and its ink mask derived (a pixel is ink when it is not transparent AND dark).
/// An SVG or an unreadable image can't be measured here → a proportional 12% window (a safe estimate),
/// with a one-line warning so the result is never silently wrong.
fn measure_clear_window(border: &std::path::Path) -> (f32, f32, f32, f32) {
    match image::open(border) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let ink = |x: u32, y: u32| {
                let p = rgba.get_pixel(x, y).0;
                let luma = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
                p[3] > 32 && luma < 128.0
            };
            crate::bookart::typst::clear_window(w, h, ink)
        }
        Err(e) => {
            eprintln!(
                "{}  couldn't measure {} ({e}) — fitting the text to a proportional 12% window; tune --margin/--safety if it overlaps",
                style("⚠").yellow(),
                border.display(),
            );
            (0.12, 0.12, 0.88, 0.88)
        }
    }
}

/// Copy `src` into `dir` (skipping when it is already there) and return its basename — how the artifact
/// references it. Keeping assets local is what lets the emitted `.typ` compile anywhere Typst can reach.
fn copy_beside(src: &std::path::Path, dir: &std::path::Path) -> Result<String> {
    let name = src.file_name().ok_or_else(|| anyhow::anyhow!("{} has no filename", src.display()))?;
    anyhow::ensure!(src.exists(), "{} does not exist", src.display());
    let dest = dir.join(name);
    let same = std::fs::canonicalize(src)
        .ok()
        .zip(std::fs::canonicalize(&dest).ok())
        .map(|(a, b)| a == b)
        .unwrap_or(false);
    if !same {
        std::fs::copy(src, &dest).with_context(|| format!("copying {} → {}", src.display(), dest.display()))?;
    }
    Ok(name.to_string_lossy().into_owned())
}

/// Compile the emitted artifact to a PDF with the `typst` CLI — the verification step. A missing `typst`
/// is a clear, actionable error (the `.typ` is still written); a compile failure surfaces Typst's own stderr.
fn verify_typst(typ_path: &std::path::Path, pdf: &std::path::Path) -> Result<()> {
    let out = std::process::Command::new("typst").arg("compile").arg(typ_path).arg(pdf).output();
    let out = match out {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "typst not found on PATH — install Typst to --verify (the artifact {} was written)",
                typ_path.display()
            );
        }
        Err(e) => return Err(anyhow::Error::from(e).context("running `typst compile`")),
    };
    if !out.status.success() {
        anyhow::bail!("typst could not compile the artifact:\n{}", String::from_utf8_lossy(&out.stderr).trim());
    }
    let bytes = std::fs::metadata(pdf).map(|m| m.len()).unwrap_or(0);
    anyhow::ensure!(bytes > 0, "typst reported success but wrote no PDF");
    println!(
        "{} {}  ({:.1} KB — compiled + verified)",
        style("pdf").green(),
        pdf.display(),
        bytes as f32 / 1024.0
    );
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
