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
    /// PIGMENT DIFFUSION (-1..+1, default 0 = none): which way the wet pigment travels. +1 = into the darks (they
    /// charge up, the lights stay clean, crisp light-side edges); -1 = out into the lights (feathered halos, soft
    /// edges). Needs a wet medium (`--bleed` > 0).
    #[arg(long, allow_hyphen_values = true)]
    pub diffuse: Option<f32>,
    /// THREADS for the up-front pigment mixing of each pass (0 = every core, the default). One painter lays the
    /// strokes whatever the count — the thread count never changes the picture.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// FILL (0..1, default 0 = auto): a density floor. When the painting stops below this share of the budget,
    /// the finest pass repeats with its restate floor halved each round until the share is spent. More worked
    /// and denser on demand — not more detail than the reference holds.
    #[arg(long, default_value_t = 0.0)]
    pub fill: f32,
    /// INTER-PASS DRYING (0..1): how much the canvas dries between passes. 0 = never (fully wet-into-wet, the
    /// masses smear into mud); 1 = bone dry (crisp overlays). Default 0.5 — the main dial against a muddy look.
    #[arg(long, default_value_t = 0.5)]
    pub dry: f32,
    /// SHADOW FLOOR (0..0.45): the value re-key lifts the SHADOW family into a narrow band above this floor instead of
    /// stretching darks to black (RFC §3.3 — a shadow mass is solid, never crushed). Default 0.16 (plan-tunable).
    #[arg(long)]
    pub shadow_floor: Option<f32>,
    /// FINEST-LAYER STROKE LENGTH (0.1..2): stroke length tapers with the layers — the wide block-in keeps long
    /// covering strokes, each thinner layer on top is shorter, down to this on the finest. Short fine dabs make
    /// features read; the plan picks ~0.3. 1 = every layer keeps its profile's own length.
    #[arg(long)]
    pub detail_length: Option<f32>,
    /// BLOCK-IN COVERAGE (0..1): how gap-free the first pass lays its base. 0 = raked/dry (grainy — ground shows
    /// through); 1 = smooth opaque cover. Default 0 (opt-in): marginal on most images, and a covering footprint
    /// can bleed into reserved paper.
    #[arg(long, default_value_t = 0.0)]
    pub coverage: f32,
    /// DETAIL COHERENCE bar (0..1, default 0.14): min structure a detail stroke needs to land. Higher = cleaner
    /// (fewer stray speckle marks on flat sky/walls, softer); lower = busier. Background held ~2.6x stricter.
    #[arg(long, default_value_t = 0.14)]
    pub detail_coherence: f32,
    /// DETAIL RESTATE floor (0..1, default 0.08): a fine-layer mark lands only where the canvas still disagrees with
    /// the target by at least this much — error-driven. Lower = the fine layers restate everything (specks on
    /// smooth masses); higher = they touch only unresolved features.
    #[arg(long, default_value_t = 0.08)]
    pub detail_restate: f32,
    /// DETAIL SHARPEN (0 = off, default): unsharp-mask the reference the fine layers paint from (radius as a fraction
    /// of the brush). It puts dark overshoot at hairlines and eye sockets that dense fine layers turn into black
    /// scrawl; 0.4 was the old behaviour.
    #[arg(long, default_value_t = 0.0)]
    pub detail_sharpen: f32,
    /// DETAIL TEXTURE (0..1): how much of the subject's own fine texture the fine layers restate inside the flat
    /// masses the armature simplified (wheat, grass, bark). 0 = fine layers read the plain armature.
    #[arg(long, default_value_t = 1.0)]
    pub detail_texture: f32,
    /// GRADATION (0..1, default 0): keep slow ramps continuous in the armature — a cloud, a soft-lit wall,
    /// still water keep their turning form instead of a few flat tones with contour edges. Edges snap as before.
    #[arg(long, default_value_t = 0.0)]
    pub gradation: f32,
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
    /// Plan a NEW painting (from scratch, RFC §1.1): coarse armature tiers, `new: true` in the plan.
    #[arg(long = "new", num_args = 0..=1, default_missing_value = "true", default_value_t = false, action = clap::ArgAction::Set)]
    pub new_painting: bool,
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
    pub diffuse: Option<f32>,
    pub threads: usize,
    pub fill: f32,
    pub dry: f32,
    pub shadow_floor: Option<f32>,
    pub detail_length: Option<f32>,
    pub coverage: f32,
    pub detail_coherence: f32,
    pub detail_restate: f32,
    pub detail_sharpen: f32,
    pub detail_texture: f32,
    pub gradation: f32,
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

/// The WATERCOLOUR's palette from the picture: k-means in CIELAB seeded by FARTHEST POINT (k-means++), so a
/// small vivid passage — a stained-glass pane, a flower box — gets a pigment of its own instead of the
/// strided seeding's duplicates of the biggest mass; each cluster's pigment is its chroma extreme; near
/// duplicates (ΔE < 6) dropped. Only the brush watercolour uses it (the other media keep `palette_from_image`
/// byte for byte).
fn palette_for_watercolour(img: &image::RgbImage, k: usize) -> crate::paint::palette::Palette {
    use crate::paint::pigment::Pigment;
    let small = image::imageops::resize(img, 160, 160, image::imageops::FilterType::Nearest);
    let srgbs: Vec<crate::paint::color::Srgb> = small.pixels().map(|p| p.0).collect();
    let lab: Vec<[f32; 3]> = srgbs.iter().map(|&c| { let l = crate::paint::color::srgb_to_lab(c); [l.l, l.a, l.b] }).collect();
    let d2 = |a: &[f32; 3], b: &[f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    // Farthest-point seeding, then Lloyd iterations.
    let kk = k.saturating_sub(2).max(4);
    let mut cents: Vec<[f32; 3]> = vec![lab[lab.len() / 2]];
    let mut dist: Vec<f32> = lab.iter().map(|p| d2(p, &cents[0])).collect();
    while cents.len() < kk {
        let (i, _) = dist.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        cents.push(lab[i]);
        for (j, p) in lab.iter().enumerate() {
            dist[j] = dist[j].min(d2(p, &lab[i]));
        }
    }
    let mut assign = vec![0usize; lab.len()];
    for _ in 0..12 {
        for (i, p) in lab.iter().enumerate() {
            assign[i] = (0..cents.len()).min_by(|&a, &b| d2(p, &cents[a]).partial_cmp(&d2(p, &cents[b])).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        }
        let mut sum = vec![[0f32; 3]; cents.len()];
        let mut cnt = vec![0usize; cents.len()];
        for (i, p) in lab.iter().enumerate() {
            for j in 0..3 { sum[assign[i]][j] += p[j]; }
            cnt[assign[i]] += 1;
        }
        for (ci, cent) in cents.iter_mut().enumerate() {
            if cnt[ci] > 0 { for j in 0..3 { cent[j] = sum[ci][j] / cnt[ci] as f32; } }
        }
    }
    let chroma = |pi: usize| (lab[pi][1].powi(2) + lab[pi][2].powi(2)).sqrt();
    let mut cols: Vec<crate::paint::color::Srgb> = vec![[247, 245, 241], [24, 24, 28]];
    let mut picked: Vec<[f32; 3]> = Vec::new();
    for ci in 0..cents.len() {
        let mut m: Vec<usize> = (0..lab.len()).filter(|&i| assign[i] == ci).collect();
        if m.is_empty() { continue; }
        m.sort_by(|&a, &b| chroma(a).partial_cmp(&chroma(b)).unwrap_or(std::cmp::Ordering::Equal));
        let pi = m[(m.len() - 1) * 85 / 100];
        if picked.iter().any(|q| d2(q, &lab[pi]) < 36.0) { continue; }
        picked.push(lab[pi]);
        cols.push(srgbs[pi]);
    }
    let pigments: Vec<Pigment> = cols.iter().enumerate().map(|(i, c)| Pigment { name: Box::leak(format!("img-{i}").into_boxed_str()), masstone: *c }).collect();
    crate::paint::palette::Palette { name: "image", pigments: Box::leak(pigments.into_boxed_slice()) }
}

/// Build a PALETTE FROM the reference IMAGE: cluster its dominant colours into pigments (plus a near-white and
/// near-black so the value range and ground are covered), so any photo repaints cleanly in any medium instead
/// of being forced through a fixed palette that can't represent its colours. The pigments are leaked to
/// `'static` — intentional, once per CLI run.
fn palette_from_image(img: &image::RgbImage, k: usize) -> crate::paint::palette::Palette {
    use crate::paint::pigment::Pigment;
    // Sample ACTUAL pixels (nearest-neighbour = a strided sample of the full-resolution image), never a box
    // average: a Triangle downsample to 64² averaged each 16×16 patch, so a grass passage of green blades with
    // blue and white flower specks became one olive-grey — and the palette derived from it had no green and no
    // blue at all (measured: the grass band painted at a quarter of the source's saturation while the sunlit
    // wall was fine). A painter's tubes span the colours that are THERE, not the average of a patch.
    let small = image::imageops::resize(img, 128, 128, image::imageops::FilterType::Nearest);
    let srgbs: Vec<crate::paint::color::Srgb> = small.pixels().map(|p| p.0).collect();
    // Cluster in CIELAB, not RGB. In RGB every dark colour sits close to every other dark colour, so a dark
    // green, a dark blue and a dark grey fall into one neutral "dark" cluster whose extreme is still mud —
    // measured on a real scene as a 76% saturation collapse in a grass-and-flowers passage while the sunlit
    // wall (light, warm — well separated in RGB) kept its chroma. Lab's a*/b* carry hue at any lightness, so
    // the cool darks get their own pigments. MORE clusters → colours captured PROPORTIONALLY: a small vivid area
    // becomes its own minor pigment instead of being averaged away, without over-representing it.
    let lab: Vec<[f32; 3]> = srgbs
        .iter()
        .map(|&c| {
            let l = crate::paint::color::srgb_to_lab(c);
            [l.l, l.a, l.b]
        })
        .collect();
    let clusters = kmeans_rgb(&lab, k.saturating_sub(2).max(4), 14);
    // Always include a near-white (ground / lights) and a near-black (darks), then the scene's dominant hues.
    let mut cols: Vec<crate::paint::color::Srgb> = vec![[247, 245, 241], [24, 24, 28]];
    // Each cluster's PIGMENT is its CHROMA EXTREME, not its centroid. A centroid is an average — always duller
    // than the colours it summarises — and Kubelka-Munk can only mix DOWN from the pigments it has, so a palette
    // of averages caps the chroma of the whole painting. A painter lays out tube colours and greys them by
    // mixing; picking each cluster's purest member (by Lab chroma, √(a²+b²)) does the same while keeping the hue
    // the image actually has. The 85th percentile, not the maximum, so a noisy outlier can't hijack a hue.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
    for (pi, p) in lab.iter().enumerate() {
        let (mut best, mut bd) = (0usize, f32::MAX);
        for (i, c) in clusters.iter().enumerate() {
            let d = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
            if d < bd {
                bd = d;
                best = i;
            }
        }
        members[best].push(pi);
    }
    let chroma = |pi: &usize| -> f32 { (lab[*pi][1].powi(2) + lab[*pi][2].powi(2)).sqrt() };
    for (i, c) in clusters.iter().enumerate() {
        let m = &mut members[i];
        if m.len() >= 8 {
            m.sort_by(|a, b| chroma(a).partial_cmp(&chroma(b)).unwrap_or(std::cmp::Ordering::Equal));
            cols.push(srgbs[m[(m.len() - 1) * 85 / 100]]);
        } else {
            // A tiny cluster: fall back to its nearest actual pixel to the centroid (never invent a colour).
            let near = lab.iter().enumerate().min_by(|(_, a), (_, b)| {
                let da = (a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2) + (a[2] - c[2]).powi(2);
                let db = (b[0] - c[0]).powi(2) + (b[1] - c[1]).powi(2) + (b[2] - c[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some((pi, _)) = near {
                cols.push(srgbs[pi]);
            }
        }
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
    /// MEDIUM to repaint the image in (oil-direct / oil-indirect / acrylic / gouache / tempera / pastel /
    /// watercolour / line-and-wash / early-book-illustration / ink-wash / japanese-ink / pen-ink / durer / pencil /
    /// black-pencil / charcoal). Applies the
    /// full technique behaviour — physics, finish and the medium's own MARK; the flags below override it.
    #[arg(long)]
    pub medium: Option<String>,
    /// Total stroke budget — inviolable.
    #[arg(long, default_value_t = 1500)]
    pub budget: usize,
    /// Brush radii, coarse → fine (comma-separated). Default: derived from the image size.
    #[arg(long, value_delimiter = ',')]
    pub brush: Option<Vec<f32>>,
    /// The smallest brush allowed (keeps the finest pass off pixel detail).
    #[arg(long, default_value_t = 2.0)]
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
    /// PIGMENT DIFFUSION (-1..+1, default 0 = none): which way the wet pigment travels. +1 = into the darks (they
    /// charge up, the lights stay clean, crisp light-side edges); -1 = out into the lights (feathered halos, soft
    /// edges). Needs a wet medium (`--bleed` > 0).
    #[arg(long, allow_hyphen_values = true)]
    pub diffuse: Option<f32>,
    /// THREADS for the up-front pigment mixing of each pass (0 = every core, the default). One painter lays the
    /// strokes whatever the count — the thread count never changes the picture.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// FILL (0..1, default 0 = auto): a density floor. When the painting stops below this share of the budget,
    /// the finest pass repeats with its restate floor halved each round until the share is spent. More worked
    /// and denser on demand — not more detail than the reference holds.
    #[arg(long, default_value_t = 0.0)]
    pub fill: f32,
    /// LEAKS (0..1, 0 = none): runs of pigment that drip down out of the wet washes under gravity, tapering to
    /// a drop where they dried — the mark that says "this was liquid" more than any other. Only the big wet
    /// masses leak, from their lower edge, in their own pigment; wet-on-wet runs further, dry-on-dry never
    /// leaks, and a run is never started across a face. A watercolourist courts it or guards against it, so
    /// it is yours: nothing leaks unless asked.
    #[arg(long)]
    pub leak: Option<f32>,
    /// TECHNIQUE: how wet the paper is when each layer goes down — the decision that makes a watercolour look
    /// the way it does. `wet-on-wet` floods one wash into the next: soft blooms, colours running together, no
    /// hard edges anywhere. `wet-on-dry` lets each wash SET before the next: crisp wash boundaries with the
    /// dark pigment rim where they dried, soft modelling within — the classic watercolour, and the default
    /// for a wet medium. `dry-on-dry` drags a barely-loaded brush over dry paper: no bleeding, the paper's
    /// tooth breaking every stroke, pigment granulating in the hollows. Sets drying, bleed, edge pooling and
    /// granulation together, so it is one decision rather than four.
    #[arg(long, value_name = "wet-on-wet|wet-on-dry|dry-on-dry")]
    pub technique: Option<String>,
    /// HOTSPOT (0..1): polish flat, blown specular highlights — the shine on a bald head, a glazed pot, wet
    /// stone. The armature snaps such a highlight into ONE value mass and the brush fills it flat, so what
    /// should be a turning form reads as a hole cut in the picture: a pale plateau with a hard rim. This
    /// re-models it as a DOME, brightest at its own centre and easing to the value its rim sits against. The
    /// gradient is invented from the shape's geometry, never copied from the source, and only the value moves
    /// — the hue stays, because a highlight is a lightness event. 0 leaves the plateau alone; the default
    /// softens it while KEEPING the highlight, which is where the light is; 1 models it fully.
    /// Default: 0.5 for a new painting, 0 otherwise.
    #[arg(long)]
    pub hotspot: Option<f32>,
    /// RIGGER (0..1, 0 = off): put back the few shapes too THIN for the brush ladder to lay at all — a stem,
    /// a spoon handle, the line of a shelf. Anything narrower than the finest brush does not soften, it
    /// disappears; a painter finishes with a rigger and puts those few things back. Draws RIDGES (a thin shape
    /// is lighter or darker than BOTH its sides, so an edge detector fires beside it and never on it), in the
    /// shape's own colour, and only where the painting LOST one. Rationed hard — a wiry picture is worse than
    /// a missing stem. Default: 0.35 for a new painting, 0 otherwise.
    #[arg(long)]
    pub rigger: Option<f32>,
    /// INFILL: what a stroke follows where the picture gives it NOTHING to follow — a flat passage, which in
    /// a dark interior is most of the canvas. `flat` lays long level marks, the way a painter blends a sky;
    /// it is right for atmosphere and wrong for a dark mass, where every stroke runs horizontally and the
    /// passage tiles into a rectangular quilt. `follow` carries the direction inward from the nearest
    /// structure, so a dark mass is stroked along the shelf edge or silhouette that bounds it. A NUMBER is a
    /// fixed stroke angle in degrees from horizontal — the painter's own decision about a passage.
    /// Default: `follow` for a new painting, `flat` otherwise (the path whose renders are already accepted).
    #[arg(long, value_name = "follow|flat|DEGREES")]
    pub infill: Option<String>,
    /// HAIR MASK: a grey PNG, white where hair / beard / fur is. Those passages are painted with the STRAND
    /// tool — many raked bristle lanes, almost no pickup, long narrow marks tapering to a point, following the
    /// picture's own growth direction, and a minority of strands breaking the silhouette. Given, it REPLACES
    /// the part detector, which boxes only the hair it can name: a long beard came out strands at the top and
    /// a smooth mass at its fall, and no automatic region has yet managed the whole of one. Make one with
    /// `plakat segment` or `plakat remove --what`. Resized to the painting; white = hair.
    #[arg(long, value_name = "PNG")]
    pub hair_mask: Option<std::path::PathBuf>,
    /// NEW PAINTING (default false; `--new` or `--new true`): paint FROM SCRATCH, as RFC PAINT-1 specifies — the
    /// picture is read once into a REDUCED armature (its things, none of its texture) and painted with
    /// the medium's strokes: a minimum brush per plane (broad in the background, finer on the figure, finest
    /// on faces only), a coverage budget, a toned ground under an opaque medium. Without it `paint` tracks its source closely.
    #[arg(long = "new", num_args = 0..=1, default_missing_value = "true", default_value_t = false, action = clap::ArgAction::Set)]
    pub new_painting: bool,
    /// INTER-PASS DRYING (0..1): how much the canvas dries between passes. 0 = never (masses smear into mud);
    /// 1 = bone dry (crisp overlays). Default 0.5 — the main dial against a muddy/washed look.
    #[arg(long, default_value_t = 0.5)]
    pub dry: f32,
    /// SHADOW FLOOR (0..0.45): the value re-key lifts the SHADOW family into a narrow band above this floor instead of
    /// stretching darks to black (RFC §3.3 — a shadow mass is solid, never crushed). Default 0.16 (plan-tunable).
    #[arg(long)]
    pub shadow_floor: Option<f32>,
    /// FINEST-LAYER STROKE LENGTH (0.1..2): stroke length tapers with the layers — the wide block-in keeps long
    /// covering strokes, each thinner layer on top is shorter, down to this on the finest. Short fine dabs make
    /// features read; the plan picks ~0.3. 1 = every layer keeps its profile's own length.
    #[arg(long)]
    pub detail_length: Option<f32>,
    /// BLOCK-IN COVERAGE (0..1): how gap-free the first pass lays its base. 0 = raked/dry (grainy — ground shows
    /// through); 1 = smooth opaque cover. Default 0 (opt-in): marginal on most images, and a covering footprint
    /// can bleed into reserved paper.
    #[arg(long, default_value_t = 0.0)]
    pub coverage: f32,
    /// DETAIL COHERENCE bar (0..1, default 0.14): min structure a detail stroke needs to land. Higher = cleaner
    /// (fewer stray speckle marks on flat sky/walls, softer); lower = busier. Background held ~2.6x stricter.
    #[arg(long, default_value_t = 0.14)]
    pub detail_coherence: f32,
    /// DETAIL RESTATE floor (0..1, default 0.08): a fine-layer mark lands only where the canvas still disagrees with
    /// the target by at least this much — error-driven. Lower = the fine layers restate everything (specks on
    /// smooth masses); higher = they touch only unresolved features.
    #[arg(long, default_value_t = 0.08)]
    pub detail_restate: f32,
    /// DETAIL SHARPEN (0 = off, default): unsharp-mask the reference the fine layers paint from (radius as a fraction
    /// of the brush). It puts dark overshoot at hairlines and eye sockets that dense fine layers turn into black
    /// scrawl; 0.4 was the old behaviour.
    #[arg(long, default_value_t = 0.0)]
    pub detail_sharpen: f32,
    /// DETAIL TEXTURE (0..1): how much of the subject's own fine texture the fine layers restate inside the flat
    /// masses the armature simplified (wheat, grass, bark). 0 = fine layers read the plain armature.
    #[arg(long, default_value_t = 1.0)]
    pub detail_texture: f32,
    /// GRADATION (0..1, default 0): keep slow ramps continuous in the armature — a cloud, a soft-lit wall,
    /// still water keep their turning form instead of a few flat tones with contour edges. Edges snap as before.
    #[arg(long, default_value_t = 0.0)]
    pub gradation: f32,
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
            Some(spec) => run_spec(SpecArgs { spec, out: args.out, size: args.size, report: args.report, planes: args.planes, critic: args.critic, families: args.families, crisp: args.crisp, strokes: args.strokes, style: args.style, define: args.define, haze: args.haze, stroke_length: args.stroke_length, stroke_width: args.stroke_width, bleed: args.bleed, diffuse: args.diffuse, threads: args.threads, fill: args.fill, dry: args.dry, shadow_floor: args.shadow_floor, detail_length: args.detail_length, coverage: args.coverage,detail_coherence: args.detail_coherence, detail_restate: args.detail_restate, detail_sharpen: args.detail_sharpen, detail_texture: args.detail_texture, gradation: args.gradation, opacity: args.opacity, pickup: args.pickup, impasto: args.impasto, broken: args.broken, contour: args.contour, saliency: args.saliency, reserve: args.reserve, focus_detail: args.focus_detail, preserve_face: args.preserve_face, splatter: args.splatter, edge_pool: args.edge_pool, paper_edge: args.paper_edge, contrast: args.contrast, warmth: args.warmth, clarity: args.clarity }).await,
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

    // MEDIUM: oil-direct | oil-indirect | acrylic | gouache | tempera | pastel | watercolour | line-and-wash |
    //         early-book-illustration | ink-wash | japanese-ink | pen-ink | durer | pencil | black-pencil | charcoal
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
    Ok(build_face_mask_ext(path, w, h).await?.map(|(m, _)| m))
}

/// The EXTENT (px, at paint size) a focal tier must resolve: the smaller box side of the SMALLEST main face —
/// main = at least half the size of the largest, so a passer-by in the distance does not set the tier, and
/// every face the picture is about is read at the tier's resolution. Measured on the detector's own boxes:
/// a mask's connected regions merge when heads stand close, and the merged blob is no face's width.
fn main_face_extent(sides: &[f32]) -> Option<f32> {
    let largest = sides.iter().copied().fold(0.0f32, f32::max);
    sides.iter().copied().filter(|s| *s > 0.0 && *s >= largest * 0.5).fold(None, |m: Option<f32>, s| Some(m.map_or(s, |v| v.min(s))))
}

/// [`build_face_mask`] plus the main faces' extent at paint size (see [`main_face_extent`]).
async fn build_face_mask_ext(path: &std::path::Path, w: u32, h: u32) -> Result<Option<(Vec<f32>, f32)>> {
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
    let sides: Vec<f32> = faces.iter().map(|f| (f.bbox[2] - f.bbox[0]).min(f.bbox[3] - f.bbox[1]) * 1.24).collect();
    let scale = (w as f32 / iw.max(1) as f32).min(h as f32 / ih.max(1) as f32);
    let extent = main_face_extent(&sides).unwrap_or(mean_face) * scale;
    Ok(Some((mask, extent)))
}

/// What the run cost, per pass and in total. A pass's rate is what tells a wide block-in from a fine
/// restatement: the same budget of marks costs very differently depending on the brush laying them.
fn print_paint_stats(stats: &[crate::paint::painter::PassStat], total_strokes: usize, seconds: f64) {
    if stats.is_empty() {
        return;
    }
    let mins = (seconds / 60.0).floor() as u64;
    let secs = seconds - (mins as f64) * 60.0;
    let when = if mins > 0 { format!("{mins}m {secs:04.1}s") } else { format!("{secs:.1}s") };
    println!("{}  painted in {when} · {} strokes · {:.0}/s overall", style("·").dim(), total_strokes, total_strokes as f64 / seconds.max(1e-6));
    for st in stats {
        if st.strokes == 0 && st.seconds < 0.05 {
            continue;
        }
        println!(
            "     {:<14} {:>5.0}px {:>9} strokes {:>8.1}/s {:>7.1}s",
            st.stage,
            st.radius,
            st.strokes,
            st.strokes as f64 / st.seconds.max(1e-6),
            st.seconds
        );
    }
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
async fn build_semantic_regions(path: &std::path::Path, w: u32, h: u32, coarse: u32, body: u32, face: u32, fine_hair: bool) -> Result<(Vec<(Vec<f32>, u32)>, Option<Vec<f32>>, Option<Vec<f32>>)> {
    let device = crate::device::select("auto")?;
    let owl = crate::pipelines::owlvit::OwlViT::load_pretrained(&device).await.context("loading OWL-ViT")?;
    let (iw, ih) = image::image_dimensions(path).with_context(|| format!("reading dimensions of {}", path.display()))?;
    // `PLAKAT_PAINT_SEMANTIC=1` prints what each query actually scored. A part that is never detected is
    // silent otherwise, and a silent miss looks exactly like a part the picture does not contain.
    let loud = std::env::var("PLAKAT_PAINT_SEMANTIC").is_ok();
    let detect = |queries: &[&str], thr: f32| -> Vec<(f32, f32, f32, f32)> {
        let mut boxes = Vec::new();
        for q in queries {
            let hits = owl.detect_all(path, q, if loud { 0.0 } else { thr }, 4).unwrap_or_default();
            if loud {
                let best = hits.iter().map(|d| d.score).fold(0.0f32, f32::max);
                println!("{}  semantic probe: {:?} best score {:.3} (threshold {thr:.2}) → {} box(es)", style("·").dim(), q, best, hits.iter().filter(|d| d.score >= thr).count());
            }
            for d in hits.into_iter().filter(|d| d.score >= thr) {
                boxes.push((d.x0, d.y0, d.x1, d.y1));
            }
        }
        boxes
    };
    let mut tiers = Vec::new();
    let mut hair_mask: Option<Vec<f32>> = None;
    // A named part → its armature-resolution ROLE, derived from the plan's own tiers (not image-tuned):
    //   skin = MID smooth form · hands = FINE (structure, a secondary focal).
    // HAIR was a COARSE wash, softer than the body. That is backwards: hair is the finest structure on a figure
    // after the features, and softening it is why a mane and a beard came out as a lumpy mass. A new painting
    // reads it BETWEEN the body and the face, where a lock of hair is a thing with a shape. (The old tier is
    // kept for the path that tracks its source, whose renders are already accepted.)
    let hair_res = if fine_hair {
        ((body + face) / 2).clamp(body, face)
    } else {
        (coarse + 12).clamp(coarse + 4, body.saturating_sub(8).max(coarse + 6))
    };
    let skin_res = (body + (face.saturating_sub(body)) / 4).clamp(body, face);
    let hands_res = ((body + face) / 2).clamp(body, face);
    for (label, queries, thr, res) in [
        // FUR too: the queries were human-only, so a lion's mane matched nothing at all and the whole hair
        // path never ran on an animal.
        // Part-level words find a human's hair. They do NOT find an animal's coat: on a picture that is half
        // lion, "fur" and "a mane" both scored under 0.06 while every part query sat at noise. So name the
        // ANIMAL as well — on an animal the coat IS the body, and the region is then the animal's box narrowed
        // by the subject matte (below).
        ("hair/beard/fur", &["a beard", "long hair", "hair", "a moustache", "fur", "a mane",
                             "a lion", "a dog", "a cat", "a horse", "a bear", "a wolf", "a furry animal"][..], 0.12_f32, hair_res),
        ("skin", &["skin", "a neck", "a bald head", "a forehead"][..], 0.11, skin_res),
        ("hands", &["a hand", "hands", "fingers"][..], 0.11, hands_res),
    ] {
        let boxes = detect(queries, thr);
        if !boxes.is_empty() {
            println!("{}  semantic: {} {label} region(s) → armature tier {res}px", style("·").dim(), boxes.len());
            let m = boxes_to_mask(&boxes, iw, ih, w, h);
            if label.starts_with("hair") {
                hair_mask = Some(m.clone());
            }
            tiers.push((m, res));
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
    Ok((tiers, clothing_mask, hair_mask))
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

/// Which of SAM's candidate masks is the PART the prompt sits in, by area alone.
///
/// SAM answers one point with three masks — roughly a subpart, a part, and the whole object. Reject the whole
/// object by SIZE: a beard is not several times the region a detector named for it, nor a large share of the
/// frame. Of what remains take the LARGEST, which reaches furthest down the hair rather than catching a
/// fragment of it. `None` when nothing is believable.
fn choose_part_scale(areas: &[usize], seed_area: usize, frame: usize) -> Option<usize> {
    areas
        .iter()
        .enumerate()
        .filter(|&(_, &a)| a <= (seed_area * 3).max(frame / 20) && a * 4 >= seed_area)
        .max_by_key(|&(_, &a)| a)
        .map(|(i, _)| i)
}

/// The hair's own extent, from SAM's PART-scale mask.
///
/// A detector boxes the hair it can name — the head, the top of a beard — and stops, so the strand tool ran
/// down to where the box ended and the rest was painted as a smooth mass. SAM knows where the beard ends; the
/// trick is asking it the right question. Prompted with a point it returns three masks (roughly a subpart, a
/// part, and the whole object), and taking the highest predicted IoU takes the WHOLE OBJECT — a point inside
/// a beard came back as the entire seated man. So choose by SCALE instead: the SMALLEST candidate that still
/// covers most of the named seed. That is the part the prompt is inside of.
///
/// `None` when there is no seed, or when every candidate is implausible as hair.
async fn sam_hair_extent(path: &std::path::Path, w: u32, h: u32, seed: &[f32]) -> Result<Option<Vec<f32>>> {
    use crate::pipelines::sam::{build_selection_masks, PointPrompt};
    let (iw, ih) = image::image_dimensions(path)?;
    let seeded: Vec<usize> = (0..seed.len()).filter(|&i| seed[i] > 0.5).collect();
    if seeded.len() < 64 {
        return Ok(None);
    }
    let (mut sx, mut sy) = (0f64, 0f64);
    for &i in &seeded {
        sx += (i % w as usize) as f64;
        sy += (i / w as usize) as f64;
    }
    let n = seeded.len() as f64;
    let sc = (iw as f64 / w as f64, ih as f64 / h as f64);
    // ONE point, at the seed's centre of mass. Two points spread down the beard seemed the safer prompt and
    // is the opposite: SAM answers several points with an object CONTAINING them all, so every candidate came
    // back a torso. Asked at one place it offers the part that place is in.
    let pts = vec![PointPrompt { x: sx / n * sc.0, y: sy / n * sc.1, foreground: true }];
    let device = crate::device::select("auto")?;
    let Ok((masks, _)) = build_selection_masks(path, &pts, &device).await else { return Ok(None) };

    let dump = std::env::var("PLAKAT_PAINT_MASKS").ok();
    let mut cands: Vec<(usize, Vec<f32>)> = Vec::new();
    for (mi, m) in masks.into_iter().enumerate() {
        let scaled = image::imageops::resize(&m, w, h, image::imageops::FilterType::Triangle);
        let v: Vec<f32> = scaled.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
        let area = v.iter().filter(|&&x| x > 0.5).count();
        if let Some(d) = &dump {
            let _ = scaled.save(std::path::Path::new(d).join(format!("sam_cand{mi}.png")));
            let cov = seeded.iter().filter(|&&i| v[i] > 0.5).count();
            println!("{}  sam candidate {mi}: area {}% · covers {}% of the named seed", style("·").dim(), area * 100 / v.len().max(1), cov * 100 / seeded.len().max(1));
        }
        // Reject the WHOLE-OBJECT candidate by size: a part that is several times the hair the detector
        // named, or a large share of the frame, is the person, not their beard. Of what is left take the
        // LARGEST — the fullest reach down the beard, rather than a fragment of it. Seed coverage is NOT a
        // test: the seed also holds the hair behind an ear, which a beard-only mask rightly does not contain.
        cands.push((area, v));
    }
    let areas: Vec<usize> = cands.iter().map(|(a, _)| *a).collect();
    let frame = (w as usize) * (h as usize);
    match choose_part_scale(&areas, seeded.len(), frame).map(|i| cands.swap_remove(i)) {
        Some((area, v)) => {
            println!("{}  hair/fur: SAM's part-scale extent over {}% of the frame (the detector named {}%)", style("·").dim(), area * 100 / v.len().max(1), seeded.len() * 100 / v.len().max(1));
            Ok(Some(v))
        }
        None => {
            println!("{}  hair/fur: no SAM candidate was believable as the named hair — keeping the detector's boxes", style("·").yellow());
            Ok(None)
        }
    }
}

/// The wet technique (see `--technique`): the painter's enum, plus the canvas-level behaviour that goes with
/// it — `(technique, drying between layers, bleed multiplier, pigment diffusion)`. What the technique does to
/// each STROKE and WASH lives in the painter; this is only what happens to the sheet between them.
fn technique_of(v: &str) -> Result<(crate::paint::painter::WetTechnique, f32, f32, f32)> {
    use crate::paint::painter::WetTechnique as T;
    Ok(match v.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "wet-on-wet" | "wet" => (T::WetOnWet, 0.12, 1.6, 0.35),
        "wet-on-dry" | "classic" => (T::WetOnDry, 1.0, 1.0, 0.0),
        "dry-on-dry" | "dry" | "drybrush" => (T::DryOnDry, 1.0, 0.15, 0.0),
        other => anyhow::bail!("--technique expects wet-on-wet, wet-on-dry or dry-on-dry — got {other:?}"),
    })
}

/// `follow` / `flat` / a stroke angle in degrees.
fn parse_infill(v: &str) -> Result<crate::paint::painter::FlowInfill> {
    use crate::paint::painter::FlowInfill;
    let t = v.trim();
    Ok(match t.to_ascii_lowercase().as_str() {
        "follow" | "structure" => FlowInfill::Follow,
        "flat" | "level" => FlowInfill::Flat,
        _ => FlowInfill::Angle(t.parse::<f32>().with_context(|| format!("--infill expects follow, flat, or an angle in degrees — got {t:?}"))?),
    })
}

/// Load a HAIR MASK png (white = hair) and fit it to the painting. A hand-drawn mask has a hard edge, and a
/// hard edge in this mask is a hard edge in the TOOL — the boundary between strand-painted and mass-painted
/// passages reads as a defect, which is the whole complaint the mask exists to answer. So it is feathered a
/// little: a few pixels, enough to hide the switch, far too few to blur which side of the beard's silhouette
/// a pixel is on.
fn load_hair_mask(path: &std::path::Path, w: u32, h: u32) -> Result<Vec<f32>> {
    let m = image::open(path).with_context(|| format!("opening the hair mask {}", path.display()))?.to_luma8();
    let feather = ((w.min(h) as f32) / 400.0).clamp(2.0, 16.0);
    let blurred = image::imageops::blur(&m, feather);
    let scaled = image::imageops::resize(&blurred, w, h, image::imageops::FilterType::Triangle);
    Ok(scaled.pixels().map(|p| p.0[0] as f32 / 255.0).collect())
}

/// Global luma standard deviation in [0,1] — a cheap proxy for tonal contrast (low = flat/foggy reference).
/// Drive the painting bar from the painter's events: the position is strokes laid; the message says which
/// pass is running, its brush, and — while the bar stands still — that the pass is mixing its colours on the
/// worker threads.
fn paint_progress(pb: &indicatif::ProgressBar, ev: crate::paint::painter::PaintProgress) {
    use crate::paint::painter::PaintProgress;
    match ev {
        PaintProgress::Placed(n) => pb.set_position(n as u64),
        PaintProgress::Mixing { pass, passes, radius, colours, threads } => pb.set_message(format!("pass {pass}/{passes} · brush {radius:.0}px · mixing {colours} colours on {threads} thread(s)…")),
        PaintProgress::Painting { pass, passes, radius } => pb.set_message(format!("pass {pass}/{passes} · brush {radius:.0}px")),
    }
}

/// Under `gradation`, where the family invariant holds off: the painter's ramp field at the background
/// armature's scale, scaled by the control (None when the control is 0 — the stage runs exactly as before).
fn family_soften(img: &image::RgbImage, params: &crate::paint::painter::PaintParams) -> Option<Vec<f32>> {
    if params.gradation <= 0.0 {
        return None;
    }
    let (w, h) = img.dimensions();
    let side = params.armature_side.unwrap_or(150).max(1);
    let r = ((w.min(h) as f32 / side as f32).round() as usize).clamp(2, 16);
    let at = |m: Option<&[f32]>, i: usize| m.and_then(|m| m.get(i).copied()).unwrap_or(0.0);
    Some(crate::paint::painter::ramp_field(img, r, params.armature_levels.max(2)).into_iter().enumerate().map(|(i, v)| v * params.gradation * (1.0 - at(params.face_mask.as_deref(), i).max(at(params.subject_mask.as_deref(), i)))).collect())
}

fn luma_stddev(img: &image::RgbImage) -> f32 {
    let n = (img.width() * img.height()).max(1) as f32;
    let lumas = img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0);
    let mean = lumas.clone().sum::<f32>() / n;
    (lumas.map(|l| (l - mean).powi(2)).sum::<f32>() / n).sqrt()
}

/// Gather the signals the plan analyzer needs: a face count (SCRFD) and the tonal-contrast proxy.
async fn analyze_image(path: &std::path::Path, medium: &str, palette: &str, from_scratch: bool) -> Result<crate::paint::plan::Analysis> {
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
        luma_mean: img.pixels().map(|p| (0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32) / 255.0).sum::<f32>() / (img.width() * img.height()).max(1) as f32,
        medium: medium.to_string(),
        palette: palette.to_string(),
        short_side: w.min(h),
        long_side: w.max(h),
        surface_white,
        structure: crate::paint::plan::structure_of(&img),
        from_scratch,
    })
}

async fn run_plan(a: PlanArgs) -> Result<()> {
    let analysis = analyze_image(&a.input, &a.medium, &a.palette, a.new_painting).await?;
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
    params.diffuse = spec.diffuse.or(a.diffuse).unwrap_or(0.0).clamp(-1.0, 1.0);
    params.threads = spec.threads.unwrap_or(a.threads);
    params.fill = spec.fill.unwrap_or(a.fill).clamp(0.0, 1.0);
    params.dry = a.dry.clamp(0.0, 1.0);
    params.coverage = a.coverage.clamp(0.0, 1.0);
    params.detail_len = a.detail_length.unwrap_or(1.0).clamp(0.1, 2.0);
    params.detail_coherence = a.detail_coherence.clamp(0.0, 1.0);
    params.detail_restate = a.detail_restate.clamp(0.0, 1.0);
    params.detail_sharpen = a.detail_sharpen.clamp(0.0, 1.0);
    params.detail_texture = a.detail_texture.clamp(0.0, 1.0);
    params.gradation = spec.gradation.unwrap_or(a.gradation).clamp(0.0, 1.0);
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
        let soften = family_soften(&reference, &params);
        let keyed = crate::paint::armature::key_families_soft(&colour, w, h, 135.0, 40.0, soften.as_deref());
        for (i, p) in reference.pixels_mut().enumerate() {
            p.0 = keyed[i];
        }
        println!("{}  family split · invariant enforced (light/shadow masses)", style("·").dim());
    }
    // VALUE RE-KEY (§5.5.2): expand the reference's tonal range so the painting reads with real lights and
    // darks rather than collapsing toward the mid ground.
    {
        let colour: Vec<crate::paint::color::Srgb> = reference.pixels().map(|p| p.0).collect();
        let keyed = crate::paint::armature::value_key(&colour, 0.04, 0.96, a.shadow_floor.unwrap_or(0.16).clamp(0.0, 0.45));
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
        let r = painter::paint_from_image_progress(&reference, &params, &|ev| paint_progress(&pb, ev));
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

    print_paint_stats(&result.stats, result.strokes, result.seconds);
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
    // Whether the background armature tier came from the plan (adjustable by the matte below) or the user.
    let mut armature_from_plan = false;
    // The plan's brush-ladder cut, applied after the medium has built its ladder (see `PaintPlan::ladder_keep`).
    let mut plan_ladder_keep: Option<usize> = None;
    // Whether the stroke budget was named on the command line (a new painting otherwise counts its own).
    let mut budget_explicit = a.budget != 1500;
    if let Some(spec) = a.plan.clone() {
        let plan = if spec == "auto" {
            let analysis = analyze_image(&a.input, a.medium.as_deref().unwrap_or("watercolour"), &a.palette, a.new_painting).await?;
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
        if plan.from_scratch {
            a.new_painting = true;
        }
        if a.hair_mask.is_none() {
            a.hair_mask = plan.hair_mask.as_ref().map(std::path::PathBuf::from);
        }
        if a.infill.is_none() {
            a.infill = plan.infill.clone();
        }
        if a.rigger.is_none() {
            a.rigger = plan.rigger;
        }
        if a.hotspot.is_none() {
            a.hotspot = plan.hotspot;
        }
        if a.technique.is_none() {
            a.technique = plan.technique.clone();
        }
        if a.leak.is_none() {
            a.leak = plan.leak;
        }
        if a.armature_levels.is_none() {
            a.armature_levels = plan.levels;
        }
        if a.splatter.is_none() {
            a.splatter = plan.splatter;
        }
        if a.edge_pool.is_none() {
            a.edge_pool = plan.edge_pool;
        }
        if a.granulate.is_none() {
            a.granulate = plan.granulate;
        }
        if let Some(n) = plan.ladder_keep {
            plan_ladder_keep = Some(n);
        }
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
            armature_from_plan = true;
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
        if a.shadow_floor.is_none() {
            a.shadow_floor = Some(plan.shadow_floor);
        }
        // A luminous medium reserves its paper RELATIVE to the keyed picture (below); the plan's absolute
        // cutoff would otherwise pass for an explicit `--reserve` and silence it.
        let luminous_medium = a.medium.as_deref().and_then(crate::paint::medium::MediumProfile::by_name).map(|m| m.mark.luminous).unwrap_or(false);
        if a.reserve.is_none() && !luminous_medium {
            a.reserve = plan.reserve;
        }
        if a.budget == 1500 {
            if let Some(b) = plan.budget {
                a.budget = b;
                // A budget the plan names is as explicit as one on the command line: the coverage count a
                // new painting works out for itself must not overrule it.
                budget_explicit = true;
            }
        }
        if a.detail_length.is_none() {
            a.detail_length = Some(plan.detail_len);
        }
    }
    let mut img = image::open(&a.input).with_context(|| format!("opening {}", a.input.display()))?.to_rgb8();
    let (w, h) = img.dimensions();

    // Palette: `image`/`auto` derives one from the reference; otherwise a named palette (defaulting to the
    // medium's own when a medium is given).
    // EXPERIMENT (PLAKAT_WC_BRUSH): a brush watercolour paints the picture re-KEYED to the paper, so its
    // pigments are derived from the keyed picture — a night scene's own pigments hold no light warm colour,
    // and keyed-up skin was mixed from the lamp glow's pale blue (teal patches on every lit face).
    let wc_brush_cli = a.new_painting && a.medium.as_deref() == Some("watercolour") && (std::env::var_os("PLAKAT_WC_BRUSH").is_some() || std::env::var_os("PLAKAT_WC_WASH").is_some());
    let img_for_palette: image::RgbImage = if wc_brush_cli && std::env::var_os("PLAKAT_WCB_NOKEY").is_none() && !(std::env::var_os("PLAKAT_WC_WASH").is_some() && std::env::var_os("PLAKAT_WCB_KEY").is_none()) {
        let envf = |n: &str, d: f32| std::env::var(n).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        crate::paint::painter::key_image(&img, &img, envf("PLAKAT_WCB_DEPTH", if std::env::var_os("PLAKAT_WC_WASH").is_some() { 0.8 } else { 0.9 }), envf("PLAKAT_WCB_GAMMA", if std::env::var_os("PLAKAT_WC_WASH").is_some() { 1.6 } else { 2.6 }), envf("PLAKAT_WCB_HI", if std::env::var_os("PLAKAT_WC_WASH").is_some() { 0.985 } else { 0.95 }))
    } else {
        img.clone()
    };
    // A WATERCOLOUR PIGMENT IS DARK IN MASSTONE AND CLEAN IN TINT. A film of pigment can never be darker
    // than the pigment itself, and a pigment taken straight from the picture is as light as the passage it
    // came from — so every dark had to be mixed with black, and the tint of that mix is grey. The brush
    // watercolour takes each image pigment's CHROMATICITY (the picture's hue, nothing named) and sets its
    // masstone deep; the value of every wash then comes from concentration, as it does on paper.
    let dark_masstones = |p: crate::paint::palette::Palette| -> crate::paint::palette::Palette {
        use crate::paint::pigment::Pigment;
        let depth = std::env::var("PLAKAT_WCB_MASSTONE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.05);
        let pigs: Vec<Pigment> = p
            .pigments
            .iter()
            .map(|pg| {
                let lin = crate::paint::color::srgb_to_linear(pg.masstone);
                let l = crate::paint::color::linear_luma(lin).max(1e-4);
                let (mx, mn) = (lin[0].max(lin[1]).max(lin[2]), lin[0].min(lin[1]).min(lin[2]));
                // Near-neutral pigments (the paper white, the dark) are left as they are.
                if mx - mn < 0.02 * mx.max(0.02) || (depth < 1.0 && l < depth) {
                    return *pg;
                }
                // `PLAKAT_WCB_MASSTONE` ≥ 1: the VIVID masstone instead — the hue at its brightest saturated
                // form (a Kubelka-Munk tint of a deep masstone goes grey; of a vivid one stays clean).
                let sc = if depth >= 1.0 { 1.0 / mx.max(1e-4) } else { depth / l };
                Pigment { name: pg.name, masstone: crate::paint::color::linear_to_srgb([(lin[0] * sc).min(1.0), (lin[1] * sc).min(1.0), (lin[2] * sc).min(1.0)]) }
            })
            .collect();
        crate::paint::palette::Palette { name: p.name, pigments: Box::leak(pigs.into_boxed_slice()) }
    };
    // CHROMA GAIN (calibration, not preference): a transparent film reads at well under its pigment's
    // chroma (measured on flat patches: the film lands at roughly 60% of the chroma of the colour it was
    // mixed for), so the brush watercolour's pigments are derived with their chroma raised by the gain that
    // brings the film back to the picture's own chroma — every hue by the same factor, the picture's hues
    // and nothing else, clipped to the gamut.
    let chroma_gain = |p: crate::paint::palette::Palette, g: f32| -> crate::paint::palette::Palette {
        use crate::paint::pigment::Pigment;
        let pigs: Vec<Pigment> = p
            .pigments
            .iter()
            .map(|pg| {
                let lin = crate::paint::color::srgb_to_linear(pg.masstone);
                let l = crate::paint::color::linear_luma(lin);
                let v = [l + (lin[0] - l) * g, l + (lin[1] - l) * g, l + (lin[2] - l) * g];
                Pigment { name: pg.name, masstone: crate::paint::color::linear_to_srgb([v[0].clamp(0.0, 1.0), v[1].clamp(0.0, 1.0), v[2].clamp(0.0, 1.0)]) }
            })
            .collect();
        crate::paint::palette::Palette { name: p.name, pigments: Box::leak(pigs.into_boxed_slice()) }
    };
    let wcb_gain = std::env::var("PLAKAT_WCB_GAIN").ok().and_then(|v| v.parse::<f32>().ok()).filter(|_| wc_brush_cli);
    let palette = match a.palette.trim().to_ascii_lowercase().as_str() {
        "image" | "auto" => {
            // A NEW painting mixes from a LIMITED palette (RFC §1.2): eight pigments of this picture.
            let npig = std::env::var("PLAKAT_WCB_PIGMENTS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(if std::env::var_os("PLAKAT_WC_WASH").is_some() { 24 } else { 16 });
            let p = if std::env::var_os("PLAKAT_WC_WASH").is_some() { palette_for_watercolour(&img_for_palette, npig) } else { palette_from_image(&img_for_palette, npig) };
            let p = if wc_brush_cli && std::env::var_os("PLAKAT_WCB_MASSTONE").is_some() { dark_masstones(p) } else { p };
            let p = match wcb_gain { Some(g) => chroma_gain(p, g), None => p };
            println!("{}  palette: derived {} pigments from the image", style("·").dim(), p.pigments.len());
            p
        }
        _ if a.medium.is_some() && a.palette == "zorn" => {
            // A medium was chosen but no palette was named — DERIVE one from the image. A fixed medium palette
            // (e.g. a landscape palette) rarely fits an arbitrary photo (a portrait's skin, a plaid shirt …).
            // A NEW painting mixes from a LIMITED palette (RFC §1.2): eight pigments of this picture.
            let p = palette_from_image(&img, 16);
            println!("{}  palette: derived {} pigments from the image", style("·").dim(), p.pigments.len());
            p
        }
        name => Palette::by_name(name).with_context(|| format!("unknown palette {:?} — try: {}, image", name, palette::ALL.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")))?,
    };

    // Brush sizes: derived from the image if not given — a coarse block-in down to a FINE restatement (the
    // finest reaches near `min_brush`, so faces and detail resolve, not just masses).
    let brush_sizes = a.brush.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| {
        // A painter COVERS with a wide brush and long strokes, then resolves features with thin, short ones. So the
        // ladder keeps a WIDE coarse brush (~1/18 of the long side, 57 px at 1024²) for a hole-free block-in and
        // halves all the way down to the finest (2 px) for detail; the DETAIL tier alone gets short marks (the
        // plan's `detail_len`). Thinning/shortening everything starved the block-in of coverage — white specks
        // of ground between dabs — while a wide-only ladder smeared the features.
        let coarse = (w.max(h) as f32 / 18.0).max(a.min_brush * 8.0);
        let mut ladder: Vec<f32> = Vec::new();
        let mut r = coarse;
        while r >= a.min_brush * 1.5 {
            ladder.push(r);
            r *= 0.5;
        }
        ladder.push(a.min_brush);
        ladder
    });

    if a.new_painting && a.armature.is_none() {
        // No plan: the new painting's tiers all the same.
        a.armature = Some(painter::NEW_BACKGROUND_SIDE);
        armature_from_plan = true;
    }
    let mut params = PaintParams::new(palette, a.budget);
    params.from_scratch = a.new_painting;
    // A new painting FOLLOWS by default: it is the path being judged, and the quilt is its most visible
    // artefact. The path that tracks its source keeps laying flat passages level, so its accepted renders
    // stay exactly as they are. Either can be overridden outright.
    params.rigger = a.rigger.unwrap_or(if a.new_painting { 0.35 } else { 0.0 }).clamp(0.0, 1.0);
    params.leak = a.leak.unwrap_or(0.0).clamp(0.0, 1.0);
    params.hotspot = a.hotspot.unwrap_or(if a.new_painting { 0.5 } else { 0.0 }).clamp(0.0, 1.0);
    params.infill = match a.infill.as_deref() {
        Some(v) => parse_infill(v)?,
        None if a.new_painting => painter::FlowInfill::Follow,
        None => painter::FlowInfill::Flat,
    };
    // MEDIUM: apply the full technique behaviour (as the spec path does); the flags below still override.
    // The medium's MARK character (its brush, stroke proportions, charge, hatching, own grey) — applied AFTER
    // the flags below so it multiplies what the user asked for.
    let mut mark: Option<crate::paint::medium::MarkCharacter> = None;
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
        mark = Some(m.mark);
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
    if let Some(mk) = mark {
        if let Some(role) = mk.role {
            params.layer_brush = Some(painter::BrushProfile::named(role));
        }
        params.stroke_len *= mk.stroke_len;
        params.stroke_width *= mk.stroke_width;
        params.charge *= mk.charge;
        params.hatch_angle = mk.hatch_angle;
        params.engrave = mk.engrave;
        if (mk.budget_scale - 1.0).abs() > 1e-3 {
            params.budget = ((params.budget as f32 * mk.budget_scale) as usize).max(500);
        }
        if let Some(l) = mk.levels {
            params.armature_levels = l.clamp(2, 32);
        }
        if let Some(rv) = mk.reserve {
            params.reserve = Some(rv.clamp(0.0, 1.0));
        }
        if let Some(n) = mk.ladder_keep {
            params.brush_sizes.truncate(n.max(1));
        }
        params.draw_contours = mk.draw_contours;
        params.brush_drawing = mk.brush_drawing;
        params.sumi = mk.sumi;
        params.luminous = mk.luminous;
        params.book = mk.book;
        if mk.luminous && mk.role.is_none() {
            // A WASH is laid flat: a streak-free brush with a SQUARE edge at every pass — a dried wash has a hard
            // edge (a soft feathered one read as a blur); the raked block-in and the filbert restatements scrub
            // a wash into scumbled oil. `--brush` overrides.
            params.layer_brush = Some(painter::BrushProfile { radius_scale: 1.05, len: 1.1, streak: 0.0, round: 0.3, waver: 0.06 });
        }
        if mk.monochrome {
            // Graphite / sumi draw in their own grey whatever the picture's colours.
            params.palette = crate::paint::palette::SUMI;
            println!("{}  palette: monochrome medium → its own grey (sumi)", style("·").dim());
        }
    }
    params.dry = a.dry.clamp(0.0, 1.0);
    params.coverage = a.coverage.clamp(0.0, 1.0);
    params.detail_len = a.detail_length.unwrap_or(1.0).clamp(0.1, 2.0);
    params.detail_coherence = a.detail_coherence.clamp(0.0, 1.0);
    params.detail_restate = a.detail_restate.clamp(0.0, 1.0);
    params.detail_sharpen = a.detail_sharpen.clamp(0.0, 1.0);
    params.detail_texture = a.detail_texture.clamp(0.0, 1.0);
    params.gradation = a.gradation.clamp(0.0, 1.0);
    if let Some(b) = a.bleed {
        params.bleed = b.clamp(0.0, 1.0);
    }
    if let Some(d) = a.diffuse {
        params.diffuse = d.clamp(-1.0, 1.0);
    }
    params.threads = a.threads;
    params.fill = a.fill.clamp(0.0, 1.0);
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
    // The medium's mark on the FINISH and the block-in (after the flags, so they multiply what was asked).
    if let Some(mk) = mark {
        params.contrast = (params.contrast * mk.contrast).clamp(0.3, 3.0);
        params.coverage = params.coverage.max(mk.coverage);
    }
    // Detect the face once if EITHER preserve-face (crisp detail tier) or armature-face (focal armature) needs it.
    if let Some(pf) = a.preserve_face {
        params.preserve_face = pf.clamp(0.0, 1.0);
    }
    params.armature_face_side = a.armature_face;
    let mut face_extent: Option<f32> = None;
    // A wash medium painted from scratch always finds its faces: the shape-aware reserve must know where NOT to
    // leave paper, and the fine brushes it keeps are for the face alone. (The focal PLANE still only runs when
    // `armature_face` asks for it — a beard is not cut by this.)
    let luminous_new = a.new_painting && a.medium.as_deref().and_then(crate::paint::medium::MediumProfile::by_name).map(|m| m.mark.luminous).unwrap_or(false);
    if a.preserve_face.is_some() || a.armature_face.is_some() || luminous_new {
        if let Some((m, e)) = build_face_mask_ext(&a.input, w, h).await? {
            params.face_mask = Some(m);
            face_extent = Some(e);
        }
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
            // The background goes COARSER than the body only when the subject fills the frame (a portrait, a
            // bust). In a SCENE where the matted subject is small, the background IS the picture — a field, a
            // sky, a street — and painting it from the coarsest tier erased its structure wholesale (a landscape
            // reduced to a few flat bands with small figures in front). Give it the body's resolution; the face
            // tier stays finer. Only when the plan chose the background tier — an explicit `--armature` is kept.
            if armature_from_plan && covered * 100 / total < 30 {
                if let Some(body) = a.armature_body.filter(|&b| a.armature.map_or(true, |c| b > c)) {
                    a.armature = Some(body);
                    println!("{}  small subject ({}%) → the background carries the picture: background armature {}px (body tier)", style("·").dim(), covered * 100 / total, body);
                }
            }
            if let Some(r) = a.recede.filter(|&r| r > 0.0) {
                // The matte IS the depth here: subject near (advances), background far (recedes/veils). But a HARD
                // subject/background boundary makes the aerial veil switch abruptly across it — a visible
                // horizontal SEAM between "veiled, light" and "full, dark". FEATHER the matte into a smooth depth
                // field (blur only this copy — the subject mask below stays sharp) so the recession is gradual.
                let (mw, mh) = (img.width(), img.height());
                let gm = image::GrayImage::from_fn(mw, mh, |x, y| image::Luma([(mask[(y * mw + x) as usize] * 255.0).clamp(0.0, 255.0) as u8]));
                let feather = (mw.max(mh) as f32 * 0.05).max(2.0);
                let blurred = image::imageops::blur(&gm, feather);
                params.depth = Some(blurred.pixels().map(|p| p.0[0] as f32 / 255.0).collect());
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
        let (tiers, clothing, hair) = build_semantic_regions(&a.input, w, h, coarse, body, face, a.new_painting).await?;
        params.region_tiers = tiers;
        // Hand the painter the hair/fur region so it changes TOOL there (see `PaintParams::hair_mask`).
        // An ANIMAL query returns the whole animal, so narrow it to what is actually coat: inside the subject
        // matte (not the ground showing through the box) and OUTSIDE the face (a muzzle, an eye and a nose are
        // smooth form — painting them with the strand tool would rake the features into fur).
        if a.new_painting {
            params.hair_mask = if let Some(mut hm) = hair {
                for (i, v) in hm.iter_mut().enumerate() {
                    if let Some(sm) = params.subject_mask.as_deref() {
                        *v *= sm.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
                    }
                    if let Some(fm) = params.face_mask.as_deref() {
                        *v *= 1.0 - fm.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    }
                }
                // The detector named WHERE hair is; SAM says how far it goes (see `sam_hair_extent`). The
                // face is smooth form however the segmenter drew the object, so it comes back out either way.
                if let Ok(Some(ext)) = sam_hair_extent(&a.input, w, h, &hm).await {
                    for (v, e) in hm.iter_mut().zip(&ext) {
                        *v = v.max(*e);
                    }
                    if let Some(fm) = params.face_mask.as_deref() {
                        for (i, v) in hm.iter_mut().enumerate() {
                            *v *= 1.0 - fm.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                        }
                    }
                }
                if let Ok(dir) = std::env::var("PLAKAT_PAINT_MASKS") {
                    let g = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(hm[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8]));
                    let _ = g.save(std::path::Path::new(&dir).join("mask_hair.png"));
                }
                let cov = hm.iter().filter(|&&v| v > 0.5).count();
                println!("{}  hair/fur: the strand tool over {}% of the frame (finer floor · raked lanes · no pickup · strands break the silhouette)", style("·").dim(), cov * 100 / hm.len().max(1));
                Some(hm)
            } else {
                None
            };
        }
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
                // A NEW painting's focal plane is EVERY face found: SAM's precise mask is prompted from one
                // face, and replacing the detector's mask with it left the other faces in the figure tier
                // (a second child's face painted as a blur beside a resolved one).
                params.face_mask = Some(match params.face_mask.take() {
                    Some(prev) if params.from_scratch && prev.len() == fm.len() => prev.iter().zip(&fm).map(|(a, b)| a.max(*b)).collect(),
                    _ => fm,
                });
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
        // A luminous medium keys HIGH: the darkest 5% land at 0.30, not 0.04 — a wash's darks stay transparent.
        let out_low = if params.book { 0.30 } else if params.luminous { 0.12 } else { 0.04 };
        let keyed = crate::paint::armature::value_key(&colour, out_low, 0.96, a.shadow_floor.unwrap_or(0.16).clamp(0.0, 0.45));
        for (i, p) in img.pixels_mut().enumerate() {
            for c in 0..3 {
                p.0[c] = (p.0[c] as f32 * (1.0 - vk) + keyed[i][c] as f32 * vk).round().clamp(0.0, 255.0) as u8;
            }
        }
        println!("{}  value-key: tonal range expanded (strength {vk:.2})", style("·").dim());
    }
    if params.luminous {
        // The wash edges BLOOM (the cauliflower rim where a wash dries) unless `--edge-pool` was set; the
        // watercolour spatters a little unless `--splatter` was set.
        if a.edge_pool.is_none() {
            params.edge_pool = 0.3;
        }
        if !params.book && a.splatter.is_none() {
            params.splatter = 0.25;
        }
        // Washes SET before the next one is laid (wet-into-wet only within a wash, never a muddy stack) unless
        // `--dry` was set; and the paper is reserved RELATIVE to the KEYED picture: its lightest ~15% stays
        // paper (measured on the picture the painter reads — judged before the key, a lifted picture went
        // almost all paper). `--reserve` overrides.
        if (a.dry - 0.5).abs() < 1e-6 {
            params.dry = 1.0;
        }
        if a.reserve.is_none() {
            let mut lum: Vec<f32> = img.pixels().map(|px| (0.299 * px.0[0] as f32 + 0.587 * px.0[1] as f32 + 0.114 * px.0[2] as f32) / 255.0).collect();
            lum.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let (qn, lo) = if params.book { (85, 0.6) } else { (88, 0.5) };
            let q = lum.get(lum.len() * qn / 100).copied().unwrap_or(0.8);
            params.reserve = Some(q.clamp(lo, 0.9));
            println!("{}  luminous medium → paper reserved above luma {:.2} (the lightest ~15% of the keyed picture)", style("·").dim(), params.reserve.unwrap_or(0.0));
        }
    }

    // TECHNIQUE: one decision that sets drying, bleed, edge pooling and granulation together (see the flag).
    // Applied after the medium and its luminous defaults, so it shapes what the medium brought rather than
    // replacing it; an explicit `--dry` / `--bleed` / `--edge-pool` still wins, as the flags always do.
    // The technique says HOW pigment behaves; the plan's and medium's amounts say HOW MUCH of each effect.
    // The two compose instead of fighting — an earlier form multiplied the amounts here and was silently
    // switched off the moment a plan named one of them, so every technique rendered identically.
    if let Some(t) = a.technique.as_deref() {
        use crate::paint::painter::WetTechnique as T;
        let (tech, dry, bl, diffuse) = technique_of(t)?;
        params.technique = tech;
        if (a.dry - 0.5).abs() < 1e-6 {
            params.dry = dry;
        }
        if a.bleed.is_none() {
            params.bleed = (params.bleed * bl).clamp(0.0, 1.0);
        }
        if a.diffuse.is_none() {
            params.diffuse = diffuse;
        }
        // Behaviour, so applied whatever the amount was set to: standing water keeps pigment from settling
        // into the tooth, a dry brush drags it straight into the hollows; and water cannot pool at an edge
        // that never set.
        match tech {
            T::WetOnWet => {
                params.granulate = (params.granulate * 0.6).clamp(0.0, 1.0);
                params.edge_pool = 0.0;
            }
            T::DryOnDry => {
                params.granulate = (params.granulate * 1.8).clamp(0.0, 1.0);
                params.edge_pool = 0.0;
            }
            _ => {}
        }
        println!("{}  technique {t}: dry {:.2} · bleed {:.2} · diffuse {:.2} · granulate {:.2} · rims {} · splatter {:.2}", style("·").dim(), params.dry, params.bleed, params.diffuse, params.granulate, if params.edge_pool > 0.0 { "form" } else { "none" }, params.splatter);
    }

    // FAMILY SEPARATION (RFC §3.3): partition light/shadow families and enforce the invariant, so masses read
    // SOLID instead of a washed photographic average. The spec path does this via --families; here it is wired for
    // the from-photo path too. Applied AFTER value-key so it groups the re-keyed values.
    if a.families {
        let (fw, fh) = img.dimensions();
        let colour: Vec<crate::paint::color::Srgb> = img.pixels().map(|p| p.0).collect();
        let soften = family_soften(&img, &params);
        let keyed = crate::paint::armature::key_families_soft(&colour, fw, fh, 135.0, 40.0, soften.as_deref());
        for (i, p) in img.pixels_mut().enumerate() {
            p.0 = keyed[i];
        }
        println!("{}  families: light/shadow split · invariant enforced (solid masses)", style("·").dim());
    }
    // An EXPLICIT hair mask wins outright over anything the part detector found. It is the instrument for the
    // one thing automatic detection keeps getting wrong — a long beard's full fall — and it is deliberately
    // not gated on `--new`: the strand tool and the growth-direction field both work from the mask alone, so
    // naming a region is enough to paint hair with hair's tool on either path. (A new painting additionally
    // gives that region a finer minimum brush, which is a plane decision and stays with the planes.)
    if let Some(hp) = a.hair_mask.clone() {
        let m = load_hair_mask(&hp, w, h)?;
        let cov = m.iter().filter(|&&v| v > 0.5).count();
        if cov == 0 {
            println!("{}  hair mask {} is empty — nothing will be painted with the strand tool", style("·").yellow(), hp.display());
        } else {
            println!("{}  hair mask {} → the strand tool over {}% of the frame (replaces the part detector)", style("·").dim(), hp.display(), cov * 100 / m.len().max(1));
        }
        params.hair_mask = Some(m);
    }

    // DIAGNOSTIC (`PLAKAT_PAINT_MASKS=<dir>`): write the planes as found — the focal (face) mask and the subject
    // matte — as grey PNGs, to see WHERE a plane ends when a picture shows its edge. Never changes the painting.
    if let Ok(dir) = std::env::var("PLAKAT_PAINT_MASKS") {
        for (name, m) in [("face", params.face_mask.as_deref()), ("subject", params.subject_mask.as_deref()), ("hair", params.hair_mask.as_deref())] {
            if let Some(m) = m.filter(|m| m.len() == (w * h) as usize) {
                let g = image::GrayImage::from_fn(w, h, |x, y| image::Luma([(m[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0) as u8]));
                let _ = g.save(std::path::Path::new(&dir).join(format!("mask_{name}.png")));
            }
        }
    }
    // The PLAN's ladder cut, last, so it overrides whatever the medium chose.
    if let Some(n) = plan_ladder_keep {
        let n = n.max(1).min(params.brush_sizes.len());
        if params.face_mask.is_some() && n < params.brush_sizes.len() {
            // The cut brushes are kept, for the face alone: a face painted with nothing finer than a wash
            // is not a face, and the user's one hard line here is that faces stay recognisable.
            params.face_ladder_from = Some(n);
            println!("{}  plan: brush ladder cut to the {} coarsest for the sheet ({:?}); the finer {} kept for the face", style("·").dim(), n, params.brush_sizes[..n].iter().map(|r| r.round() as u32).collect::<Vec<_>>(), params.brush_sizes.len() - n);
        } else {
            params.brush_sizes.truncate(n);
            println!("{}  plan: brush ladder cut to the {} coarsest ({:?})", style("·").dim(), n, params.brush_sizes.iter().map(|r| r.round() as u32).collect::<Vec<_>>());
        }
    }

    if params.from_scratch {
        // Nothing restates the source's texture.
        params.detail_texture = 0.0;
        // IMPRIMATURA: an opaque painting begun from nothing is begun on a TONED ground — the picture's own
        // mean colour (in linear light) — so where a broad mark leaves a gap, the gap is the picture's tone,
        // not a fleck of white priming. A transparent medium keeps its paper: the paper is its light.
        if params.ground.is_some() {
            let lin = |v: u8| { let c = v as f64 / 255.0; if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) } };
            let mut acc = [0.0f64; 3];
            for px in img.pixels() {
                for c in 0..3 {
                    acc[c] += lin(px.0[c]);
                }
            }
            let n = (img.width() as f64 * img.height() as f64).max(1.0);
            let enc = |v: f64| { let v = v / n; let c = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }; (c * 255.0).round().clamp(0.0, 255.0) as u8 };
            let tone = [enc(acc[0]), enc(acc[1]), enc(acc[2])];
            params.ground = Some(tone);
            println!("{}  new painting: toned ground rgb({}, {}, {}) — the picture's mean colour", style("·").dim(), tone[0], tone[1], tone[2]);
        }
        // ARMATURE FIDELITY IS INDEPENDENT OF CANVAS COVERAGE (RFC §5.2): a figure and a face are read at a
        // fixed number of pixels across THEIR OWN extent, whatever share of the sheet they take — a small face
        // in a wide scene keeps its structure, a face that fills the frame is not over-resolved. `--armature`
        // is the DETAIL dial of a new painting: it names the background tier and the others follow in ratio.
        let short = w.min(h) as f32;
        let nominal = if params.face_mask.is_some() { painter::NEW_BACKGROUND_SIDE } else { painter::NEW_BACKGROUND_SIDE_PLAIN };
        let bg_side = a.armature.unwrap_or(nominal).clamp(16, short as u32);
        let xs = bg_side as f32 / nominal as f32;
        params.armature_side = Some(bg_side);
        let tier = |across: f32, extent: Option<f32>, lo: u32| extent.map(|e| ((across * xs * short / e.max(8.0)).round() as u32).clamp(lo, (short as u32).max(lo)));
        if let Some(side) = tier(painter::NEW_FIGURE_ACROSS, params.subject_mask.as_deref().and_then(|m| painter::region_extent(m, w, h)), bg_side) {
            params.armature_body_side = Some(side);
        }
        let body_side = params.armature_body_side.unwrap_or(bg_side);
        // A face's extent is the detector's own box (heads standing close merge into one region of a mask).
        let face_extent = face_extent.or_else(|| params.face_mask.as_deref().and_then(|m| painter::region_extent(m, w, h)));
        if let Some(side) = tier(painter::NEW_FACE_ACROSS, face_extent, body_side) {
            params.armature_face_side = Some(side);
        }
        // The figure's silhouette is a SEAM (RFC §7): strokes end at it, so the figure stands against its ground.
        if params.region_mask.is_none() {
            params.region_mask = params.subject_mask.as_ref().map(|m| m.iter().map(|v| *v > 0.5).collect());
        }
        let sides = (bg_side, params.armature_body_side, params.armature_face_side);
        if !budget_explicit && !params.density {
            // …and the budget is the coverage of the planes as found.
            let floor = painter::plane_floor_field(w, h, params.min_brush, sides, params.subject_mask.as_deref(), params.face_mask.as_deref(), params.hair_mask.as_deref());
            let b = painter::from_scratch_budget(w, h, &params.brush_sizes, params.min_brush, &floor);
            params.budget = b;
            a.budget = b;
        }
        let (bg, body, focal) = painter::plane_floors(w, h, params.min_brush, sides);
        println!(
            "{}  new painting: armature {}px background · {}px figure · {}px faces — minimum brush {:.0} / {:.0} / {:.0} px · budget {} strokes (coverage)",
            style("·").dim(),
            bg_side,
            sides.1.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
            sides.2.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
            bg,
            body,
            focal,
            params.budget
        );
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
    let result = painter::paint_from_image_progress(&img, &params, &|ev| paint_progress(&pb, ev));
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

    print_paint_stats(&result.stats, result.strokes, result.seconds);
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

#[cfg(test)]
mod focal_tier_tests {
    use super::main_face_extent;

    #[test]
    fn the_part_scale_choice_rejects_the_whole_object() {
        let frame = 1_000_000usize;
        let seed = 20_000; // 2% of the frame, the sort of region a part detector names for a beard
        // SAM's three: a fragment, the beard, the whole man. Take the beard — the largest that is not the man.
        assert_eq!(super::choose_part_scale(&[6_000, 40_000, 230_000], seed, frame), Some(1));
        // Nothing believable: every candidate is the person.
        assert_eq!(super::choose_part_scale(&[230_000, 260_000], seed, frame), None);
        // A fragment far smaller than what was named is not the hair either.
        assert_eq!(super::choose_part_scale(&[900], seed, frame), None);
        // On a picture that is mostly animal, a coat IS most of the subject — the frame share must not veto it.
        assert_eq!(super::choose_part_scale(&[210_000], 220_000, frame), Some(0));
    }

    #[test]
    fn a_hair_mask_loads_white_as_hair_and_softens_its_own_edge() {
        // White is hair, black is not, and the switch between the two tools is feathered — a hard edge in the
        // mask is a hard edge in the TOOL, and that boundary reads as exactly the defect the mask exists to fix.
        let dir = std::env::temp_dir().join("plakat_hair_mask_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.png");
        let img = image::GrayImage::from_fn(64, 64, |x, _| image::Luma([if x < 32 { 255 } else { 0 }]));
        img.save(&path).unwrap();

        let m = super::load_hair_mask(&path, 64, 64).unwrap();
        assert_eq!(m.len(), 64 * 64);
        assert!(m[32 * 64 + 4] > 0.9, "the white half is hair");
        assert!(m[32 * 64 + 60] < 0.1, "the black half is not");
        // Across the boundary the value must pass through the middle rather than jump.
        let mid: Vec<f32> = (24..40).map(|x| m[32 * 64 + x]).collect();
        assert!(mid.iter().any(|v| (0.2..0.8).contains(v)), "the edge is feathered, not a step: {mid:?}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_focal_tier_is_sized_by_the_smallest_main_face() {
        // Three heads close together: the tier resolves the smallest of them; a distant passer-by is ignored.
        assert_eq!(main_face_extent(&[300.0, 220.0, 240.0, 40.0]), Some(220.0));
        assert_eq!(main_face_extent(&[120.0]), Some(120.0));
        assert_eq!(main_face_extent(&[]), None);
    }
}
