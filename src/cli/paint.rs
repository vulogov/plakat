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
    /// and the foreground advances (depth "layers"). Uses the armature's depth map. 0 = flat single plane
    /// (default — opt in; a non-zero default fogged every painting that had any depth).
    #[arg(long, default_value_t = 0.0)]
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
    /// INTER-PASS DRYING (0..1): how much the canvas dries between passes. 0 = never (fully wet-into-wet, the
    /// masses smear into mud); 1 = bone dry (crisp overlays). Default 0.5 — the main dial against a muddy look.
    #[arg(long, default_value_t = 0.5)]
    pub dry: f32,
    /// BLOCK-IN COVERAGE (0..1): how gap-free the first pass lays its base. 0 = raked/dry (grainy — ground shows
    /// through); 1 = smooth opaque cover. Default 0.65 — the main lever against speckly "dry pastel" grain.
    #[arg(long, default_value_t = 0.65)]
    pub coverage: f32,
    /// OPACITY / body (0.1..1) — overrides the medium default (1 = opaque; low = transparent).
    #[arg(long)]
    pub opacity: Option<f32>,
    /// PICKUP (0..1) dirty-brush drag — overrides the medium default.
    #[arg(long)]
    pub pickup: Option<f32>,
    /// IMPASTO (0..1) textured thick-paint relief at output — overrides the medium default (oil high, flat 0).
    #[arg(long)]
    pub impasto: Option<f32>,
    /// BROKEN COLOUR (0..1) — per-stroke hue/chroma variation for optical-mix vibrancy (oil/gouache/pastel).
    #[arg(long)]
    pub broken: Option<f32>,
    /// CONTOUR (0..1) — line-drawing pass over the strongest edges (pen/pencil/charcoal).
    #[arg(long)]
    pub contour: Option<f32>,
    /// SALIENCY-GATED DENSITY (0..1, opt-in): reserve dense strokes for the focal subject and lay flat/empty
    /// regions THIN — stops a big stroke budget over-working the background into a uniform hatch. 0 = off.
    #[arg(long)]
    pub saliency: Option<f32>,
    /// RESERVE threshold (0..1, surface-white media): cells brighter than this keep the bare paper (no stroke).
    /// Raise toward 1 to CLOSE white holes in light passages; lower to keep more paper. Default 0.72 (watercolour/ink).
    #[arg(long)]
    pub reserve: Option<f32>,
    /// SELECTIVE DETAIL (0..1, opt-in): paint the masses loose but fire the crisp detail tier ONLY in the focal
    /// region (eyes/glasses) — loose-wash + sharp accents. Pair with `--style impressionist`. Small = tighter focus.
    #[arg(long)]
    pub focus_detail: Option<f32>,
    /// PRESERVE FACE (0..1, opt-in): DETECT the face(s) and fire the crisp detail tier only on the real face box
    /// — loose everywhere else. Like --focus-detail but model-targeted. Higher = more of the face preserved crisp.
    #[arg(long)]
    pub preserve_face: Option<f32>,
    /// SPLATTER (0..1, opt-in): flick fine pigment droplets across the painting — the watercolour/ink spatter mark.
    #[arg(long)]
    pub splatter: Option<f32>,
    /// EDGE POOLING (0..1, opt-in): darken pigment at wash boundaries — the watercolour edge-bloom / "cauliflower".
    #[arg(long)]
    pub edge_pool: Option<f32>,
    /// PAPER EDGE (0..1, opt-in): fade to a deckled bare-paper border — the torn-paper vignette a watercolour sits in.
    #[arg(long)]
    pub paper_edge: Option<f32>,
    /// CONTRAST (0.5..2, 1 = neutral): painting-safe finish grade, recorded for replay.
    #[arg(long)]
    pub contrast: Option<f32>,
    /// WARMTH (−1..1, 0 = neutral): finish white-balance shift (+ warm / − cool), recorded for replay.
    #[arg(long)]
    pub warmth: Option<f32>,
    /// CLARITY (0..1, opt-in): gentle LOCAL contrast (not edge sharpening), recorded for replay.
    #[arg(long)]
    pub clarity: Option<f32>,
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
    /// Analyze an image (art director) and write a PAINTING PLAN — the structural decisions the paint stage runs.
    Plan(PlanArgs),
}

#[derive(Args, Debug)]
pub struct PlanArgs {
    /// The image to analyze (a photo or a generation).
    pub input: PathBuf,
    /// Where to write the plan (HJSON). Default: alongside the image as `<name>.plan.hjson`. `-` = stdout.
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    /// The medium the plan targets (drives the reserve decision).
    #[arg(long, default_value = "watercolour")]
    pub medium: String,
    /// The palette the plan targets.
    #[arg(long, default_value = "image")]
    pub palette: String,
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
    pub dry: f32,
    pub coverage: f32,
    pub opacity: Option<f32>,
    pub pickup: Option<f32>,
    pub impasto: Option<f32>,
    pub broken: Option<f32>,
    pub contour: Option<f32>,
    pub saliency: Option<f32>,
    pub reserve: Option<f32>,
    pub focus_detail: Option<f32>,
    pub preserve_face: Option<f32>,
    pub splatter: Option<f32>,
    pub edge_pool: Option<f32>,
    pub paper_edge: Option<f32>,
    pub contrast: Option<f32>,
    pub warmth: Option<f32>,
    pub clarity: Option<f32>,
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

/// Deterministic k-means over RGB points (few iterations) — the dominant colours of an image.
fn kmeans_rgb(px: &[[f32; 3]], k: usize, iters: usize) -> Vec<[f32; 3]> {
    if px.is_empty() {
        return Vec::new();
    }
    let k = k.max(1).min(px.len());
    let mut cents: Vec<[f32; 3]> = (0..k).map(|i| px[(i * px.len() / k).min(px.len() - 1)]).collect();
    let mut assign = vec![0usize; px.len()];
    for _ in 0..iters {
        for (i, p) in px.iter().enumerate() {
            let (mut best, mut bd) = (0usize, f32::MAX);
            for (c, ct) in cents.iter().enumerate() {
                let d = (p[0] - ct[0]).powi(2) + (p[1] - ct[1]).powi(2) + (p[2] - ct[2]).powi(2);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            assign[i] = best;
        }
        let mut sum = vec![[0f32; 3]; k];
        let mut cnt = vec![0f32; k];
        for (i, p) in px.iter().enumerate() {
            let a = assign[i];
            for j in 0..3 {
                sum[a][j] += p[j];
            }
            cnt[a] += 1.0;
        }
        for c in 0..k {
            if cnt[c] > 0.0 {
                for j in 0..3 {
                    cents[c][j] = sum[c][j] / cnt[c];
                }
            }
        }
    }
    cents
}

/// Build a PALETTE FROM the reference IMAGE: cluster its dominant colours into pigments (plus a near-white and
/// near-black so the value range and ground are covered), so any photo repaints cleanly in any medium instead
/// of being forced through a fixed palette that can't represent its colours. The pigments are leaked to
/// `'static` — intentional, once per CLI run.
fn palette_from_image(img: &image::RgbImage, k: usize) -> crate::paint::palette::Palette {
    use crate::paint::pigment::Pigment;
    let small = image::imageops::resize(img, 64, 64, image::imageops::FilterType::Triangle);
    let px: Vec<[f32; 3]> = small.pixels().map(|p| [p.0[0] as f32, p.0[1] as f32, p.0[2] as f32]).collect();
    // MORE clusters → colours captured PROPORTIONALLY: a small vivid area (a red shirt) becomes its own minor
    // pigment instead of being averaged away, without over-representing it (which warms the whole picture).
    let clusters = kmeans_rgb(&px, k.saturating_sub(2).max(4), 14);
    // Always include a near-white (ground / lights) and a near-black (darks), then the scene's dominant hues.
    let mut cols: Vec<crate::paint::color::Srgb> = vec![[247, 245, 241], [24, 24, 28]];
    for c in clusters {
        cols.push([c[0].round().clamp(0.0, 255.0) as u8, c[1].round().clamp(0.0, 255.0) as u8, c[2].round().clamp(0.0, 255.0) as u8]);
    }
    // Safety net: if NO strongly-saturated pigment made it in, add the single most saturated distinct colour so a
    // vivid accent isn't lost entirely — but only one, so it can't dominate.
    let sat_of = |c: &[u8; 3]| -> f32 {
        let (mx, mn) = (*c.iter().max().unwrap() as f32, *c.iter().min().unwrap() as f32);
        if mx < 1.0 { 0.0 } else { (mx - mn) / mx }
    };
    if !cols.iter().any(|c| sat_of(c) > 0.45) {
        if let Some(c) = small.pixels().map(|p| p.0).filter(|c| sat_of(c) > 0.45).max_by(|a, b| sat_of(a).partial_cmp(&sat_of(b)).unwrap_or(std::cmp::Ordering::Equal)) {
            cols.push(c);
        }
    }
    let pigments: Vec<Pigment> = cols.iter().enumerate().map(|(i, c)| Pigment { name: Box::leak(format!("img-{i}").into_boxed_str()), masstone: *c }).collect();
    crate::paint::palette::Palette { name: "image", pigments: Box::leak(pigments.into_boxed_slice()) }
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

fn parse_edge_mode(s: &str) -> Result<crate::paint::painter::EdgeMode> {
    use crate::paint::painter::EdgeMode;
    match s.trim().to_ascii_lowercase().as_str() {
        "line" => Ok(EdgeMode::Line),
        "colour" | "color" | "temperature" => Ok(EdgeMode::Colour),
        "knife" | "scrape" | "lift" => Ok(EdgeMode::Knife),
        "lost" | "none" | "soft" => Ok(EdgeMode::Lost),
        other => anyhow::bail!("unknown --silhouette-mode {other:?} (use: line | colour | knife | lost)"),
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
    /// Palette name, or `image` / `auto` to DERIVE a palette from the reference's dominant colours.
    #[arg(long, default_value = "zorn")]
    pub palette: String,
    /// MEDIUM to repaint the image in (oil-direct / watercolour / gouache / ink-wash / pen-ink / pencil /
    /// pastel / charcoal / acrylic / …). Applies the full technique behaviour; the flags below override it.
    #[arg(long)]
    pub medium: Option<String>,
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
    /// INTER-PASS DRYING (0..1): how much the canvas dries between passes. 0 = never (masses smear into mud);
    /// 1 = bone dry (crisp overlays). Default 0.5 — the main dial against a muddy/washed look.
    #[arg(long, default_value_t = 0.5)]
    pub dry: f32,
    /// BLOCK-IN COVERAGE (0..1): how gap-free the first pass lays its base. 0 = raked/dry (grainy — ground shows
    /// through); 1 = smooth opaque cover. Default 0.65 — the main lever against speckly "dry pastel" grain.
    #[arg(long, default_value_t = 0.65)]
    pub coverage: f32,
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
    /// BROKEN COLOUR (0..1) per-stroke hue/chroma variation.
    #[arg(long)]
    pub broken: Option<f32>,
    /// CONTOUR (0..1) line-drawing pass over the strongest edges.
    #[arg(long)]
    pub contour: Option<f32>,
    /// SALIENCY-GATED DENSITY (0..1, opt-in): reserve dense strokes for the focal subject and lay flat/empty
    /// regions THIN (so a big stroke budget stops over-working the background into a uniform hatch). 0 = off.
    #[arg(long)]
    pub saliency: Option<f32>,
    /// RESERVE threshold (0..1, surface-white media): cells brighter than this keep the bare paper (no stroke).
    /// Raise toward 1 to CLOSE white holes in light passages; lower to keep more paper. Default 0.72 (watercolour/ink).
    #[arg(long)]
    pub reserve: Option<f32>,
    /// SELECTIVE DETAIL (0..1, opt-in): paint the masses loose but fire the crisp detail tier ONLY in the focal
    /// region (eyes/glasses) — loose-wash + sharp accents. Pair with `--style impressionist`. Small = tighter focus.
    #[arg(long)]
    pub focus_detail: Option<f32>,
    /// PRESERVE FACE (0..1, opt-in): DETECT the face(s) and fire the crisp detail tier only on the real face box
    /// — loose everywhere else. Like --focus-detail but model-targeted (uses the face detector). Higher = more of
    /// the face preserved crisp; lower = only the core. Pair with a loose base (`--style impressionist`).
    #[arg(long)]
    pub preserve_face: Option<f32>,
    /// SPLATTER (0..1, opt-in): flick fine pigment droplets across the painting — the watercolour/ink spatter
    /// mark (spray, snow, sparkle). Higher = denser. On surface-white media a few droplets lift to the paper.
    #[arg(long)]
    pub splatter: Option<f32>,
    /// EDGE POOLING (0..1, opt-in): darken pigment where a wash meets a hard boundary — the watercolour
    /// edge-bloom / "cauliflower" ring a drying wash leaves. Higher = stronger rings.
    #[arg(long)]
    pub edge_pool: Option<f32>,
    /// PAPER EDGE (0..1, opt-in): fade to bare paper at the borders with an irregular DECKLED edge — the
    /// torn-paper vignette a watercolour sits in. Higher = wider fade. Great for portraits on paper.
    #[arg(long)]
    pub paper_edge: Option<f32>,
    /// CONTRAST (0.5..2, 1 = neutral): a painting-safe finish grade — S-curve tonal contrast, recorded for replay.
    #[arg(long)]
    pub contrast: Option<f32>,
    /// WARMTH (−1..1, 0 = neutral): finish white-balance shift — + warm (amber), − cool (blue). Recorded for replay.
    #[arg(long)]
    pub warmth: Option<f32>,
    /// CLARITY (0..1, opt-in): gentle LOCAL contrast (midtone punch) — NOT edge sharpening. Recorded for replay.
    #[arg(long)]
    pub clarity: Option<f32>,
    /// VALUE KEY (0..1, opt-in): expand the reference's tonal range BEFORE painting so the picture reads with
    /// real DARKS and LIGHTS instead of a foggy midtone — the biggest lever for punch from a low-contrast photo.
    /// The spec path does this always; here it is a knob. 0 = off; ~0.7–1.0 for a flat photo.
    #[arg(long)]
    pub value_key: Option<f32>,
    /// ARMATURE resolution (px, RFC §1.1/§5): paint from a COARSE structural armature at this short-side size
    /// instead of the full-resolution photo, so the engine INVENTS the surface rather than TRACING detail (which
    /// is what turns a beard into scribble). ~48–96 is the paintable range (§12.1). Unset = paint from the photo
    /// (a filter — the RFC anti-pattern). This is the plan-vs-pixels switch.
    #[arg(long)]
    pub armature: Option<u32>,
    /// VALUE MASSES in the structure-preserving armature (RFC §5): how many value levels it quantises to — the
    /// block-in a painter sees. Fewer = bolder flatter masses; more = subtler (and, too high, filter-like). Default 8.
    #[arg(long)]
    pub armature_levels: Option<u32>,
    /// FOCAL armature resolution (px, RFC §5.2): with `--armature`, paint the DETECTED FACE from a finer armature
    /// (this size) than the rest — a wash beard/background AND a crisp face in one pass. ~160–220 works. Detects a
    /// face automatically (no need for --preserve-face). Must be larger than --armature.
    #[arg(long)]
    pub armature_face: Option<u32>,
    /// SUBJECT-BODY armature resolution (px, RFC §5.2 multi-region): with `--armature`, paint the matted SUBJECT
    /// body from a MID armature (this size) — background coarsest, body mid, face fine. Runs U2Net to matte the
    /// subject. Between --armature and --armature-face.
    #[arg(long)]
    pub armature_body: Option<u32>,
    /// AERIAL PERSPECTIVE via the subject matte (0..1): veil the BACKGROUND so the subject advances. Uses the
    /// U2Net matte (run for --armature-body), not the CPU depth proxy that `--haze` uses.
    #[arg(long)]
    pub recede: Option<f32>,
    /// SEMANTIC tiers (RFC §5.2): detect parts (hair/beard) with OWL-ViT and give them their own armature tier —
    /// a coarse WASH for the beard, kept softer than the body. Runs the open-vocab detector.
    #[arg(long)]
    pub semantic: bool,
    /// FAMILY SEPARATION (RFC §3.3): partition the reference into LIGHT and SHADOW families and enforce the
    /// invariant (no light value darker than the lightest shadow), so masses read SOLID rather than a washed
    /// average. The single biggest lever against the "washed photograph" look. `--plan auto` enables it.
    #[arg(long)]
    pub families: bool,
    /// COMMIT SHADOWS (0..1, RFC §3.3): paint the DARK value masses DECISIVELY — lift the reserve, deepen the
    /// darks pass over pass, and load more pigment there — so the painting has a solid value backbone instead of
    /// a pale wash. `--plan auto` enables it. The direct fix for "covering the image with a wash".
    #[arg(long)]
    pub commit_shadows: Option<f32>,
    /// SILHOUETTE (0..1, RFC §5/§7): mark the detected subject boundary so a light shirt/shoulders read by their
    /// EDGE against a light ground instead of vanishing. Needs a subject (matte). `--plan auto` enables it.
    #[arg(long)]
    pub silhouette: Option<f32>,
    /// How the silhouette EDGE is marked (RFC §7 edge craft): `line` (soft contour) · `colour` (a temperature
    /// shift, no line) · `knife` (a scraped/lifted crisp edge) · `lost` (dissolved). Default `line`.
    #[arg(long)]
    pub silhouette_mode: Option<String>,
    /// SAM precise masks (RFC §5): use MobileSAM (prompted from the detected face) for a PRECISE subject and face
    /// mask — a SHARP silhouette edge and a face-shaped focal region — instead of the soft U2Net matte / feathered
    /// box. Sharpens the silhouette and improves the face. `--plan auto` enables it.
    #[arg(long)]
    pub sam: bool,
    /// PAINTING PLAN (RFC §5): `auto` analyses the image (art director) and fills the structural decisions
    /// (armature, focal armature, value-key, reserve, budget, medium, palette) that unset flags leave open; or a
    /// path to a `plan.hjson` (from `plakat paint plan`) to paint from a saved/edited plan. Explicit flags win.
    #[arg(long)]
    pub plan: Option<String>,
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
        Some(PaintCmd::From(a)) => run_from(a).await,
        Some(PaintCmd::Replay(a)) => run_replay(a),
        Some(PaintCmd::Export(a)) => run_export(a),
        Some(PaintCmd::Timelapse(a)) => run_timelapse(a),
        Some(PaintCmd::Palette(a)) => run_palette(a),
        Some(PaintCmd::Plan(a)) => run_plan(a).await,
        None => match args.spec {
            Some(spec) => run_spec(SpecArgs { spec, out: args.out, size: args.size, report: args.report, planes: args.planes, critic: args.critic, families: args.families, crisp: args.crisp, strokes: args.strokes, style: args.style, define: args.define, haze: args.haze, stroke_length: args.stroke_length, stroke_width: args.stroke_width, bleed: args.bleed, dry: args.dry, coverage: args.coverage, opacity: args.opacity, pickup: args.pickup, impasto: args.impasto, broken: args.broken, contour: args.contour, saliency: args.saliency, reserve: args.reserve, focus_detail: args.focus_detail, preserve_face: args.preserve_face, splatter: args.splatter, edge_pool: args.edge_pool, paper_edge: args.paper_edge, contrast: args.contrast, warmth: args.warmth, clarity: args.clarity }).await,
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

/// Detect faces in `path` and build a feathered 0..1 mask (row-major, sized `w`×`h`) over the face box(es), for
/// `--preserve-face`. Runs the SCRFD detector (small — CPU is fine). `None` when no face is found. The box is
/// expanded a little (brow/chin) and the edges feathered, so the crisp detail tier fades into the loose masses
/// rather than leaving a hard rectangle. Detects at native resolution, then resizes the mask to the paint size.
async fn build_face_mask(path: &std::path::Path, w: u32, h: u32) -> Result<Option<Vec<f32>>> {
    use candle_core::DType;
    let (iw, ih) = image::image_dimensions(path).with_context(|| format!("reading dimensions of {}", path.display()))?;
    let device = crate::device::select("auto")?;
    let weights = crate::pipelines::scrfd::resolve_scrfd_weights().await.context("resolving the face-detector weights")?
        .context("face-detector weights not available — run a faceswap/restore once to fetch them")?;
    let det = crate::pipelines::scrfd::SCRFDDetector::load(&weights, crate::pipelines::scrfd::SCRFDConfig::default(), &device, DType::F32).context("loading the face detector")?;
    let faces = det.detect(path).context("detecting faces")?;
    if faces.is_empty() {
        return Ok(None);
    }
    let mut m = image::GrayImage::new(iw, ih);
    for f in &faces {
        let [x1, y1, x2, y2] = f.bbox;
        let (ex, ey) = ((x2 - x1) * 0.12, (y2 - y1) * 0.12);
        let x1 = (x1 - ex).max(0.0) as u32;
        let y1 = (y1 - ey).max(0.0) as u32;
        let x2 = ((x2 + ex).min(iw as f32) as u32).min(iw);
        let y2 = ((y2 + ey).min(ih as f32) as u32).min(ih);
        for yy in y1..y2 {
            for xx in x1..x2 {
                m.put_pixel(xx, yy, image::Luma([255]));
            }
        }
    }
    let mean_face = faces.iter().map(|f| (f.bbox[2] - f.bbox[0]).min(f.bbox[3] - f.bbox[1])).sum::<f32>() / faces.len() as f32;
    let feather = (mean_face * 0.12).clamp(2.0, 60.0);
    let blurred = image::imageops::blur(&m, feather);
    let scaled = image::imageops::resize(&blurred, w, h, image::imageops::FilterType::Triangle);
    let mask: Vec<f32> = scaled.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    println!("{}  preserve-face: {} face(s) detected → focal mask", style("·").dim(), faces.len());
    Ok(Some(mask))
}

/// Human summary of the stroke count against the budget. The budget is a CEILING, not a quota: the gates
/// (saliency / reserve / focus / preserve-face / restate) can exhaust the eligible cells before it is reached,
/// so when fewer strokes were laid we say so explicitly instead of silently reporting a number below the budget.
fn stroke_summary(performed: usize, budget: usize) -> String {
    if performed < budget {
        format!("{performed} of {budget} strokes performed (gates capped placement below budget)")
    } else {
        format!("{performed} strokes laid")
    }
}

/// Build a feathered 0..1 mask (sized `w`×`h`) covering a set of boxes in the ORIGINAL image's coordinates
/// (`iw`×`ih`). Boxes are expanded slightly and the edges feathered so a region tier fades into its neighbours.
fn boxes_to_mask(boxes: &[(f32, f32, f32, f32)], iw: u32, ih: u32, w: u32, h: u32) -> Vec<f32> {
    let mut m = image::GrayImage::new(iw, ih);
    let mut sizes = 0.0f32;
    for &(x0, y0, x1, y1) in boxes {
        let (ex, ey) = ((x1 - x0) * 0.08, (y1 - y0) * 0.08);
        let x0 = (x0 - ex).max(0.0) as u32;
        let y0 = (y0 - ey).max(0.0) as u32;
        let x1 = ((x1 + ex).min(iw as f32) as u32).min(iw);
        let y1 = ((y1 + ey).min(ih as f32) as u32).min(ih);
        sizes += (x1 - x0).min(y1 - y0) as f32;
        for yy in y0..y1 {
            for xx in x0..x1 {
                m.put_pixel(xx, yy, image::Luma([255]));
            }
        }
    }
    let feather = (sizes / boxes.len().max(1) as f32 * 0.1).clamp(2.0, 50.0);
    let blurred = image::imageops::blur(&m, feather);
    let scaled = image::imageops::resize(&blurred, w, h, image::imageops::FilterType::Triangle);
    scaled.pixels().map(|p| p.0[0] as f32 / 255.0).collect()
}

/// SEMANTIC regions (RFC §5.2) via OWL-ViT. Returns `(hair/beard wash tiers, clothing mask)`:
/// - hair/beard → a COARSE wash armature tier (kept softer than the body);
/// - clothing/shoulders → a mask to EXTEND the subject fact, so a light shirt is painted, not reserved to paper.
/// Both are open-vocab detections; a model-free empty result if nothing is found.
async fn build_semantic_regions(path: &std::path::Path, w: u32, h: u32, coarse: u32, body: u32, face: u32) -> Result<(Vec<(Vec<f32>, u32)>, Option<Vec<f32>>)> {
    let device = crate::device::select("auto")?;
    let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(&device).await.context("loading OWL-ViT")?;
    let (iw, ih) = image::image_dimensions(path).with_context(|| format!("reading dimensions of {}", path.display()))?;
    let detect = |queries: &[&str], thr: f32| -> Vec<(f32, f32, f32, f32)> {
        let mut boxes = Vec::new();
        for q in queries {
            for d in owl.detect_all(path, q, thr, 4).unwrap_or_default() {
                boxes.push((d.x0, d.y0, d.x1, d.y1));
            }
        }
        boxes
    };
    let mut tiers = Vec::new();
    // A named part → its armature-resolution ROLE, derived from the plan's own tiers (not image-tuned):
    //   hair/beard = COARSE wash · skin = MID smooth form · hands = FINE (structure, a secondary focal).
    let hair_res = (coarse + 12).clamp(coarse + 4, body.saturating_sub(8).max(coarse + 6));
    let skin_res = (body + (face.saturating_sub(body)) / 4).clamp(body, face);
    let hands_res = ((body + face) / 2).clamp(body, face);
    for (label, queries, thr, res) in [
        ("hair/beard", &["a beard", "long hair", "hair", "a moustache"][..], 0.12_f32, hair_res),
        ("skin", &["skin", "a neck", "a bald head", "a forehead"][..], 0.11, skin_res),
        ("hands", &["a hand", "hands", "fingers"][..], 0.11, hands_res),
    ] {
        let boxes = detect(queries, thr);
        if !boxes.is_empty() {
            println!("{}  semantic: {} {label} region(s) → armature tier {res}px", style("·").dim(), boxes.len());
            tiers.push((boxes_to_mask(&boxes, iw, ih, w, h), res));
        }
    }
    // CLOTHING / SHOULDERS → extend the subject so a light shirt is PAINTED, not reserved to blank paper.
    let clothing = detect(&["a shirt", "clothing", "a t-shirt", "shoulders", "a jacket"], 0.10);
    let clothing_mask = if clothing.is_empty() {
        println!("{}  semantic: no clothing detected", style("·").yellow());
        None
    } else {
        println!("{}  semantic: {} clothing region(s) → extend the subject (paint the shirt)", style("·").dim(), clothing.len());
        Some(boxes_to_mask(&clothing, iw, ih, w, h))
    };
    Ok((tiers, clothing_mask))
}

/// Detect the primary (largest, highest-score) face box `[x0,y0,x1,y1]` in original-image pixels, via SCRFD.
async fn detect_primary_face(path: &std::path::Path) -> Result<Option<[f32; 4]>> {
    use candle_core::DType;
    let device = crate::device::select("auto")?;
    let weights = match crate::pipelines::scrfd::resolve_scrfd_weights().await.ok().flatten() {
        Some(w) => w,
        None => return Ok(None),
    };
    let det = crate::pipelines::scrfd::SCRFDDetector::load(&weights, crate::pipelines::scrfd::SCRFDConfig::default(), &device, DType::F32)?;
    let mut faces = det.detect(path)?;
    faces.sort_by(|a, b| {
        let area = |f: &crate::pipelines::scrfd::Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
        area(b).partial_cmp(&area(a)).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(faces.first().map(|f| f.bbox))
}

/// Precise SUBJECT + FACE masks via MobileSAM, prompted from the detected face box (a fact). SAM segments the
/// actual silhouette, so the edge is SHARP (unlike the soft U2Net matte / feathered box). Returns `(subject,
/// face)` masks sized `w`×`h` in [0,1]. `None` if no face is found (nothing to prompt with).
async fn sam_regions(path: &std::path::Path, w: u32, h: u32) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
    use crate::pipelines::sam::{build_selection_mask, PointPrompt};
    let face = match detect_primary_face(path).await? {
        Some(b) => b,
        None => return Ok((None, None)),
    };
    let device = crate::device::select("auto")?;
    let (iw, ih) = image::image_dimensions(path)?;
    let (fx, fy) = ((face[0] + face[2]) * 0.5, (face[1] + face[3]) * 0.5);
    let fh = (face[3] - face[1]).max(1.0);
    let corners = |mut v: Vec<PointPrompt>| {
        for (cx, cy) in [(2.0, 2.0), (iw as f64 - 2.0, 2.0), (2.0, ih as f64 - 2.0), (iw as f64 - 2.0, ih as f64 - 2.0)] {
            v.push(PointPrompt { x: cx, y: cy, foreground: false });
        }
        v
    };
    let mask_to_vec = |m: image::GrayImage| -> Vec<f32> {
        let s = image::imageops::resize(&m, w, h, image::imageops::FilterType::Triangle);
        s.pixels().map(|p| p.0[0] as f32 / 255.0).collect()
    };
    // SUBJECT: foreground on the face + down the torso; background at the corners.
    let subj_pts = corners(vec![
        PointPrompt { x: fx as f64, y: fy as f64, foreground: true },
        PointPrompt { x: fx as f64, y: (fy + 1.6 * fh).min(ih as f32 - 2.0) as f64, foreground: true },
        PointPrompt { x: fx as f64, y: (fy + 2.8 * fh).min(ih as f32 - 2.0) as f64, foreground: true },
    ]);
    let subject = build_selection_mask(path, &subj_pts, &device).await.ok().map(mask_to_vec);
    // FACE/HEAD: foreground on the face + forehead; background at the corners AND the torso (exclude the body).
    let face_pts = corners(vec![
        PointPrompt { x: fx as f64, y: fy as f64, foreground: true },
        PointPrompt { x: fx as f64, y: (fy - 0.3 * fh).max(2.0) as f64, foreground: true },
        PointPrompt { x: fx as f64, y: (fy + 2.6 * fh).min(ih as f32 - 2.0) as f64, foreground: false },
    ]);
    let face_mask = build_selection_mask(path, &face_pts, &device).await.ok().map(|m| {
        // Feather the face mask a touch so its armature/detail region fades into the surroundings.
        let blurred = image::imageops::blur(&m, (fh * 0.06).clamp(2.0, 30.0));
        mask_to_vec(blurred)
    });
    Ok((subject, face_mask))
}

/// Global luma standard deviation in [0,1] — a cheap proxy for tonal contrast (low = flat/foggy reference).
fn luma_stddev(img: &image::RgbImage) -> f32 {
    let n = (img.width() * img.height()).max(1) as f32;
    let lumas = img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0);
    let mean = lumas.clone().sum::<f32>() / n;
    (lumas.map(|l| (l - mean).powi(2)).sum::<f32>() / n).sqrt()
}

/// Gather the signals the plan analyzer needs: a face count (SCRFD) and the tonal-contrast proxy.
async fn analyze_image(path: &std::path::Path, medium: &str, palette: &str) -> Result<crate::paint::plan::Analysis> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?.to_rgb8();
    let (w, h) = img.dimensions();
    // Face presence via the detector (a mask is Some when a face is found).
    let faces = match build_face_mask(path, w, h).await {
        Ok(Some(_)) => 1,
        _ => 0,
    };
    let surface_white = crate::paint::medium::MediumProfile::by_name(medium).map(|m| m.white_source == crate::paint::medium::WhiteSource::Surface).unwrap_or(false);
    Ok(crate::paint::plan::Analysis {
        faces,
        luma_stddev: luma_stddev(&img),
        medium: medium.to_string(),
        palette: palette.to_string(),
        short_side: w.min(h),
        long_side: w.max(h),
        surface_white,
    })
}

async fn run_plan(a: PlanArgs) -> Result<()> {
    let analysis = analyze_image(&a.input, &a.medium, &a.palette).await?;
    let plan = crate::paint::plan::plan_from(&analysis);
    let text = plan.to_hjson();
    match a.out.as_deref() {
        Some(p) if p.as_os_str() == "-" => print!("{text}"),
        _ => {
            let out = a.out.unwrap_or_else(|| a.input.with_extension("plan.hjson"));
            std::fs::write(&out, &text).with_context(|| format!("writing {}", out.display()))?;
            println!("{}  painting plan → {}", style("✓").green(), out.display());
            for n in &plan.notes {
                println!("{}  {n}", style("·").dim());
            }
        }
    }
    Ok(())
}

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
    params.dry = a.dry.clamp(0.0, 1.0);
    params.coverage = a.coverage.clamp(0.0, 1.0);
    params.opacity = spec.opacity.or(a.opacity).unwrap_or(plan.medium.body).clamp(0.1, 1.0);
    params.impasto = spec.impasto.or(a.impasto).unwrap_or(plan.medium.impasto).clamp(0.0, 1.0);
    params.brush.k_pickup = spec.pickup.or(a.pickup).unwrap_or(plan.medium.pickup).clamp(0.0, 1.0);
    // MATERIAL physics: the paint's own behaviour, defaulting to the medium.
    params.chroma = spec.chroma.unwrap_or(plan.medium.chroma).clamp(0.3, 2.0);
    params.dry_shift = spec.dry_shift.unwrap_or(plan.medium.dry_shift).clamp(-0.4, 0.4);
    params.granulate = spec.granulate.unwrap_or(plan.medium.granulate).clamp(0.0, 1.0);
    params.sheen = spec.sheen.unwrap_or(plan.medium.sheen).clamp(0.0, 1.0);
    params.lift = spec.lift.unwrap_or(plan.medium.lift).clamp(0.0, 1.0);
    params.broken = spec.broken.or(a.broken).unwrap_or(plan.medium.broken).clamp(0.0, 1.0);
    params.contour = spec.contour.or(a.contour).unwrap_or(plan.medium.contour).clamp(0.0, 1.0);
    params.saliency = spec.saliency.or(a.saliency).unwrap_or(0.0).clamp(0.0, 1.0);
    params.focus_detail = spec.focus_detail.or(a.focus_detail).unwrap_or(0.0).clamp(0.0, 1.0);
    params.splatter = spec.splatter.or(a.splatter).unwrap_or(0.0).clamp(0.0, 1.0);
    params.edge_pool = spec.edge_pool.or(a.edge_pool).unwrap_or(0.0).clamp(0.0, 1.0);
    params.paper_edge = spec.paper_edge.or(a.paper_edge).unwrap_or(0.0).clamp(0.0, 1.0);
    params.contrast = spec.contrast.or(a.contrast).unwrap_or(1.0).clamp(0.3, 3.0);
    params.warmth = spec.warmth.or(a.warmth).unwrap_or(0.0).clamp(-1.0, 1.0);
    params.clarity = spec.clarity.or(a.clarity).unwrap_or(0.0).clamp(0.0, 1.0);
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
    // Reserve threshold: cells brighter than this keep the bare paper (no stroke). `--reserve`/`reserve:` override
    // the medium default — raise it (→1) to CLOSE white holes in light passages, lower it to keep more paper.
    let reserve_default = (plan.medium.white_source == WhiteSource::Surface).then_some(0.72);
    params.reserve = spec.reserve.or(a.reserve).map(|r| r.clamp(0.0, 1.0)).or(reserve_default);
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

    // PALETTE-FROM-IMAGE: `palette: image`/`auto` derives a palette from the reference's dominant colours, so a
    // photo or supplied image repaints cleanly in any medium (not forced through a palette that can't hold it).
    if matches!(spec.palette.as_deref().map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("image") | Some("auto")) {
        params.palette = palette_from_image(&reference, 16);
        println!("{}  palette: derived {} pigments from the image", style("·").dim(), params.palette.pigments.len());
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

    // PRESERVE FACE (opt-in): detect the face on the (working-resolution) reference and fire the crisp detail
    // tier only there. Detect on a temp PNG since the detector reads a path.
    if let Some(pf) = spec.preserve_face.or(a.preserve_face) {
        params.preserve_face = pf.clamp(0.0, 1.0);
        let (rw, rh) = reference.dimensions();
        let tmp = tempfile::Builder::new().prefix("plakat-paint-face-").suffix(".png").tempfile().context("face-detect scratch file")?;
        reference.save(tmp.path()).context("writing the reference for face detection")?;
        params.face_mask = build_face_mask(tmp.path(), rw, rh).await?;
        if params.face_mask.is_none() {
            println!("{}  preserve-face: no face detected — painting without a face focal region", style("·").yellow());
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
        let pb = crate::ui::progress::step_bar(params.budget as u64, "painting");
        let r = painter::paint_from_image_progress(&reference, &params, &|placed| pb.set_position(placed as u64));
        pb.set_position(r.strokes as u64);
        pb.finish_and_clear();
        r
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

    println!("{}  {} → {}  ·  score → {}  ·  recipe → {}", style("✓").green(), stroke_summary(result.strokes, plan.budget), out.display(), score_path.display(), sidecar.display());
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

async fn run_from(mut a: FromArgs) -> Result<()> {
    // PAINTING PLAN (RFC §5): resolve `--plan auto|<file>` FIRST and let it fill the structural decisions that
    // unset flags leave open — the art director hands the technique a plan. Explicit flags always win.
    if let Some(spec) = a.plan.clone() {
        let plan = if spec == "auto" {
            let analysis = analyze_image(&a.input, a.medium.as_deref().unwrap_or("watercolour"), &a.palette).await?;
            let p = crate::paint::plan::plan_from(&analysis);
            println!("{}  plan (auto):", style("◆").cyan());
            for n in &p.notes {
                println!("{}  {n}", style("·").dim());
            }
            p
        } else {
            let text = std::fs::read_to_string(&spec).with_context(|| format!("reading plan {spec}"))?;
            crate::paint::plan::PaintPlan::parse(&text).with_context(|| format!("parsing plan {spec}"))?
        };
        if a.medium.is_none() {
            a.medium = Some(plan.medium.clone());
        }
        if a.palette == "zorn" {
            a.palette = plan.palette.clone();
        }
        if a.style == "legible" {
            a.style = plan.style.clone();
        }
        if a.armature.is_none() {
            a.armature = Some(plan.armature);
        }
        if a.armature_face.is_none() {
            a.armature_face = plan.armature_face;
        }
        if a.armature_body.is_none() {
            a.armature_body = plan.armature_body;
        }
        if a.recede.is_none() && plan.recede > 0.0 {
            a.recede = Some(plan.recede);
        }
        if plan.semantic {
            a.semantic = true;
        }
        if plan.families {
            a.families = true;
        }
        if a.commit_shadows.is_none() && plan.commit_shadows > 0.0 {
            a.commit_shadows = Some(plan.commit_shadows);
        }
        if a.silhouette.is_none() && plan.silhouette > 0.0 {
            a.silhouette = Some(plan.silhouette);
        }
        if a.silhouette_mode.is_none() {
            a.silhouette_mode = plan.silhouette_mode.clone();
        }
        if plan.sam {
            a.sam = true;
        }
        if a.value_key.is_none() {
            a.value_key = Some(plan.value_key);
        }
        if a.reserve.is_none() {
            a.reserve = plan.reserve;
        }
        if a.budget == 1500 {
            if let Some(b) = plan.budget {
                a.budget = b;
            }
        }
    }
    let mut img = image::open(&a.input).with_context(|| format!("opening {}", a.input.display()))?.to_rgb8();
    let (w, h) = img.dimensions();

    // Palette: `image`/`auto` derives one from the reference; otherwise a named palette (defaulting to the
    // medium's own when a medium is given).
    let palette = match a.palette.trim().to_ascii_lowercase().as_str() {
        "image" | "auto" => {
            let p = palette_from_image(&img, 16);
            println!("{}  palette: derived {} pigments from the image", style("·").dim(), p.pigments.len());
            p
        }
        _ if a.medium.is_some() && a.palette == "zorn" => {
            // A medium was chosen but no palette was named — DERIVE one from the image. A fixed medium palette
            // (e.g. a landscape palette) rarely fits an arbitrary photo (a portrait's skin, a plaid shirt …).
            let p = palette_from_image(&img, 16);
            println!("{}  palette: derived {} pigments from the image", style("·").dim(), p.pigments.len());
            p
        }
        name => Palette::by_name(name).with_context(|| format!("unknown palette {:?} — try: {}, image", name, palette::ALL.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?,
    };

    // Brush sizes: derived from the image if not given — a coarse block-in down to a FINE restatement (the
    // finest reaches near `min_brush`, so faces and detail resolve, not just masses).
    let brush_sizes = a.brush.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| {
        let coarse = (w.max(h) as f32 / 18.0).max(a.min_brush * 2.0);
        vec![coarse, coarse * 0.55, coarse * 0.3, (coarse * 0.16).max(a.min_brush)]
    });

    let mut params = PaintParams::new(palette, a.budget);
    // MEDIUM: apply the full technique behaviour (as the spec path does); the flags below still override.
    if let Some(mname) = &a.medium {
        use crate::paint::medium::{MarkModel, WhiteSource};
        let m = crate::paint::medium::MediumProfile::by_name(mname).with_context(|| format!("unknown medium {mname:?} — try: {}", crate::paint::medium::EXECUTABLE.join(" / ")))?;
        params.medium = m.name.to_string();
        params.bleed = m.bleed;
        params.opacity = m.body;
        params.impasto = m.impasto;
        params.chroma = m.chroma;
        params.dry_shift = m.dry_shift;
        params.granulate = m.granulate;
        params.sheen = m.sheen;
        params.lift = m.lift;
        params.brush.k_pickup = m.pickup;
        params.broken = m.broken;
        params.contour = m.contour;
        params.density = m.mark_model == MarkModel::Density;
        if m.white_source == WhiteSource::Surface {
            params.reserve = Some(0.72);
        } else {
            params.ground = Some([236, 230, 220]);
        }
        println!("{}  medium: {} (full technique)", style("·").dim(), m.name);
    }
    // Reserve threshold override (`--reserve`): raise → close white holes in light passages, lower → more paper.
    if let Some(rv) = a.reserve {
        params.reserve = Some(rv.clamp(0.0, 1.0));
    }
    params.brush_sizes = brush_sizes.clone();
    params.min_brush = a.min_brush;
    params.seed = a.seed;
    params.style = parse_style(&a.style)?;
    params.define = a.define.clamp(0.0, 1.0);
    params.stroke_len = a.stroke_length.clamp(0.2, 4.0);
    params.stroke_width = a.stroke_width.clamp(0.3, 3.0);
    params.dry = a.dry.clamp(0.0, 1.0);
    params.coverage = a.coverage.clamp(0.0, 1.0);
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
    if let Some(bk) = a.broken {
        params.broken = bk.clamp(0.0, 1.0);
    }
    if let Some(co) = a.contour {
        params.contour = co.clamp(0.0, 1.0);
    }
    if let Some(sl) = a.saliency {
        params.saliency = sl.clamp(0.0, 1.0);
    }
    if let Some(fd) = a.focus_detail {
        params.focus_detail = fd.clamp(0.0, 1.0);
    }
    if let Some(sp) = a.splatter {
        params.splatter = sp.clamp(0.0, 1.0);
    }
    if let Some(ep) = a.edge_pool {
        params.edge_pool = ep.clamp(0.0, 1.0);
    }
    if let Some(pe) = a.paper_edge {
        params.paper_edge = pe.clamp(0.0, 1.0);
    }
    if let Some(ct) = a.contrast {
        params.contrast = ct.clamp(0.3, 3.0);
    }
    if let Some(wm) = a.warmth {
        params.warmth = wm.clamp(-1.0, 1.0);
    }
    if let Some(cl) = a.clarity {
        params.clarity = cl.clamp(0.0, 1.0);
    }
    // Detect the face once if EITHER preserve-face (crisp detail tier) or armature-face (focal armature) needs it.
    if let Some(pf) = a.preserve_face {
        params.preserve_face = pf.clamp(0.0, 1.0);
    }
    params.armature_face_side = a.armature_face;
    if a.preserve_face.is_some() || a.armature_face.is_some() {
        params.face_mask = build_face_mask(&a.input, w, h).await?;
        if params.face_mask.is_none() {
            println!("{}  face: none detected — painting without a face focal region", style("·").yellow());
        }
    }
    // MULTI-REGION armature (RFC §5.2): matte the SUBJECT (U2Net) so the body paints from a mid armature and the
    // background from the coarsest — plus optional aerial recession of the (matted) background.
    params.armature_body_side = a.armature_body;
    if a.armature_body.is_some() || a.recede.is_some() {
        let device = crate::device::select("auto")?;
        let matter = crate::pipelines::matting::Matter::load(&device).await.context("loading U2Net for the subject matte")?;
        let alpha = matter.matte(&img).context("matting the subject")?;
        let mask: Vec<f32> = alpha.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        let covered = mask.iter().filter(|&&m| m > 0.5).count();
        let total = mask.len().max(1);
        if covered > total / 50 && covered < total * 49 / 50 {
            println!("{}  subject matte: {}% foreground → three-tier armature", style("·").dim(), covered * 100 / total);
            if let Some(r) = a.recede.filter(|&r| r > 0.0) {
                // The matte IS the depth here: subject near (advances), background far (recedes/veils).
                params.depth = Some(mask.clone());
                params.haze = r.clamp(0.0, 1.0);
            }
            params.subject_mask = Some(mask);
        } else {
            println!("{}  subject matte: no clear subject — skipping the body tier", style("·").yellow());
            params.armature_body_side = None;
        }
    }
    // SEMANTIC regions (RFC §5.2): OWL-ViT detects hair/beard → a coarse wash tier; and clothing/shoulders →
    // EXTEND the subject fact so a light shirt is painted (not reserved to paper — the fix for vanished shoulders).
    if a.semantic {
        let coarse = a.armature.unwrap_or(72);
        let body = a.armature_body.unwrap_or(coarse + 40);
        let face = a.armature_face.unwrap_or(body + 96);
        let (tiers, clothing) = build_semantic_regions(&a.input, w, h, coarse, body, face).await?;
        params.region_tiers = tiers;
        if let Some(cloth) = clothing {
            // Union the clothing into the subject mask so the reserve treats the shirt as subject, not background.
            match params.subject_mask.as_mut() {
                Some(subj) if subj.len() == cloth.len() => {
                    for (s, c) in subj.iter_mut().zip(&cloth) {
                        *s = s.max(*c);
                    }
                }
                _ => params.subject_mask = Some(cloth),
            }
        }
    }
    // SAM PRECISE MASKS (RFC §5): replace the soft subject/face masks with MobileSAM's precise silhouette (prompted
    // from the detected face) — a SHARP silhouette edge and a face-shaped focal region. Run last so it overrides.
    if a.sam {
        let (subject, face_m) = sam_regions(&a.input, w, h).await?;
        if let Some(s) = subject {
            let cov = s.iter().filter(|&&m| m > 0.5).count();
            if cov > s.len() / 50 && cov < s.len() * 49 / 50 {
                println!("{}  sam: precise subject mask ({}% foreground) — sharp silhouette", style("·").dim(), cov * 100 / s.len().max(1));
                // Union with any clothing already added, so SAM sharpens without dropping detected clothing.
                match params.subject_mask.as_mut() {
                    Some(sm) if sm.len() == s.len() => {
                        for (a, b) in sm.iter_mut().zip(&s) {
                            *a = a.max(*b);
                        }
                    }
                    _ => params.subject_mask = Some(s),
                }
            } else {
                println!("{}  sam: subject mask unusable — keeping the matte", style("·").yellow());
            }
        }
        if let Some(fm) = face_m {
            let cov = fm.iter().filter(|&&m| m > 0.4).count();
            if cov > fm.len() / 200 && cov < fm.len() / 2 {
                println!("{}  sam: precise face mask — face-shaped focal region", style("·").dim());
                params.face_mask = Some(fm);
            }
        }
    }
    if a.haze > 0.0 {
        // A CPU depth proxy (central + low = near) so aerial perspective can be exercised without a depth model.
        params.depth = Some(crate::paint::armature::depth_proxy(w, h));
        params.haze = a.haze.clamp(0.0, 1.0);
    }

    // RFC §1.1 — the plan-vs-pixels switch: paint from a COARSE structural armature (structure survives
    // downsampling, detail does not), so the engine INVENTS the surface instead of TRACING the photo. Unset keeps
    // the legacy full-resolution "filter" behaviour that the existing tuning knobs operate on.
    params.armature_side = a.armature.filter(|&s| s > 0);
    if let Some(lv) = a.armature_levels {
        params.armature_levels = lv.clamp(2, 32);
    }

    // VALUE KEY (§5.5.2): expand the reference's tonal range so the painting reads with real darks/lights instead
    // of a foggy midtone — the spec path does this always; here it is opt-in. Blend by strength so it is tunable.
    if let Some(vk) = a.value_key.filter(|&v| v > 0.0) {
        let vk = vk.clamp(0.0, 1.0);
        let colour: Vec<crate::paint::color::Srgb> = img.pixels().map(|p| p.0).collect();
        let keyed = crate::paint::armature::value_key(&colour, 0.04, 0.96);
        for (i, p) in img.pixels_mut().enumerate() {
            for c in 0..3 {
                p.0[c] = (p.0[c] as f32 * (1.0 - vk) + keyed[i][c] as f32 * vk).round().clamp(0.0, 255.0) as u8;
            }
        }
        println!("{}  value-key: tonal range expanded (strength {vk:.2})", style("·").dim());
    }

    // FAMILY SEPARATION (RFC §3.3): partition light/shadow families and enforce the invariant, so masses read
    // SOLID instead of a washed photographic average. The spec path does this via --families; here it is wired for
    // the from-photo path too. Applied AFTER value-key so it groups the re-keyed values.
    if a.families {
        let (fw, fh) = img.dimensions();
        let colour: Vec<crate::paint::color::Srgb> = img.pixels().map(|p| p.0).collect();
        let keyed = crate::paint::armature::key_families(&colour, fw, fh, 135.0, 40.0);
        for (i, p) in img.pixels_mut().enumerate() {
            p.0 = keyed[i];
        }
        println!("{}  families: light/shadow split · invariant enforced (solid masses)", style("·").dim());
    }
    // COMMIT SHADOWS (RFC §3.3): paint the dark masses decisively (see the painter). Computed from the value-keyed,
    // family-split reference so it targets the real shadow structure.
    params.commit_shadows = a.commit_shadows.unwrap_or(0.0).clamp(0.0, 1.0);
    if params.commit_shadows > 0.0 {
        println!("{}  commit-shadows: dark masses painted decisively (strength {:.2})", style("·").dim(), params.commit_shadows);
    }
    params.silhouette = a.silhouette.unwrap_or(0.0).clamp(0.0, 1.0);
    if let Some(m) = a.silhouette_mode.as_deref() {
        params.silhouette_mode = parse_edge_mode(m)?;
    }
    if params.silhouette > 0.0 {
        println!("{}  silhouette: subject edge marked ({:?}, strength {:.2})", style("·").dim(), params.silhouette_mode, params.silhouette);
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

    let pb = crate::ui::progress::step_bar(params.budget as u64, "painting");
    let result = painter::paint_from_image_progress(&img, &params, &|placed| pb.set_position(placed as u64));
    pb.set_position(result.strokes as u64);
    pb.finish_and_clear();
    let out = result.canvas.to_image_finished(&result.score.header.finish());
    if let Some(parent) = a.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;

    // The canonical artifact: the replayable stroke score, next to the image.
    let score_path = a.out.with_extension("strokes");
    std::fs::write(&score_path, result.score.to_text()).with_context(|| format!("writing {}", score_path.display()))?;

    println!("{}  {} → {}  ·  score → {}", style("✓").green(), stroke_summary(result.strokes, a.budget), a.out.display(), score_path.display());
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
