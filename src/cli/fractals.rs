//! `plakat fractals` — Track-A (pure-CPU) fractal rendering CLI (RFC FRACTALS-1, Phase 1).
//!
//! Resolution order for the spec: a base (`--fractal-clone PATH` → `--fractal-spec FILE`
//! → built-in default), then any explicitly-provided `--fractal-*` flag overrides that
//! base. `--fractal-dump-spec` prints the fully-resolved spec as JSON and renders nothing.

use anyhow::{Context, Result};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

use crate::fractals::{
    self,
    spec::{Coloring, FractalKind, TrapShape},
    FractalSpec,
};

#[derive(Args, Debug, Clone)]
pub struct FractalsArgs {
    /// Shape the FRACTAL from keywords in a description (family/mood/coloring/depth); any
    /// text that names no family derives a distinctive fractal from a hash of the words.
    /// This does NOT paint a scene — for that use `--fractal-paint --fractal-prompt`.
    /// E.g. `--fractal-from "a fiery burning ship, deep zoom"`.
    #[arg(long = "fractal-from", value_name = "TEXT", help_heading = "Input & output")]
    pub from: Option<String>,

    /// LLM provider for `--fractal-from` (deepseek | gemini | local | local:<alias> | auto).
    /// When set, an LLM maps the description to a spec (falling back to the offline keyword
    /// mapper on any failure). Omit for the fast, fully-offline keyword mapper.
    #[arg(long = "fractal-provider", value_name = "PROVIDER", help_heading = "Input & output")]
    pub provider: Option<String>,

    /// Load the base spec from an HJSON/JSON file (CLI flags still override its fields).
    #[arg(long = "fractal-spec", value_name = "FILE", help_heading = "Input & output")]
    pub spec: Option<PathBuf>,

    /// Reconstruct the base spec from a PNG previously written by `plakat fractals`
    /// (reads its embedded `fractalspec` chunk). Takes precedence over `--fractal-spec`.
    #[arg(long = "fractal-clone", value_name = "PNG", help_heading = "Input & output")]
    pub clone_from: Option<PathBuf>,

    /// Fractal family: mandelbrot | julia | burning-ship.
    #[arg(long = "fractal-kind", value_name = "KIND", help_heading = "Fractal shape")]
    pub kind: Option<String>,

    /// Viewport center in the complex plane as `RE,IM`.
    #[arg(long = "fractal-center", value_name = "RE,IM", allow_hyphen_values = true, help_heading = "Fractal shape")]
    pub center: Option<String>,

    /// Zoom factor (vertical axis spans `3.0 / zoom` complex units).
    #[arg(long = "fractal-zoom", value_name = "Z", help_heading = "Fractal shape")]
    pub zoom: Option<f64>,

    /// Iteration cap (escape budget).
    #[arg(long = "fractal-iter", value_name = "N", help_heading = "Fractal shape")]
    pub iter: Option<u32>,

    /// Julia constant as `RE,IM` (only used when kind = julia).
    #[arg(long = "fractal-julia-c", value_name = "RE,IM", allow_hyphen_values = true, help_heading = "Per-family")]
    pub julia_c: Option<String>,

    /// Exponent for the `z^power` step (2 = classic; other = multibrot / Newton degree).
    #[arg(long = "fractal-power", value_name = "P", help_heading = "Per-family")]
    pub power: Option<f64>,

    /// Output size as `WxH` (e.g. `1920x1080`).
    #[arg(long = "fractal-size", value_name = "WxH", help_heading = "Fractal shape")]
    pub size: Option<String>,

    /// Coloring: smooth | histogram | distance | orbit-trap | angle | stripe.
    #[arg(long = "fractal-coloring", value_name = "MODE", help_heading = "Coloring & palette")]
    pub coloring: Option<String>,

    /// Anti-aliasing: render at NxN samples per pixel then downsample (1..=8; 1 = off).
    #[arg(long = "fractal-supersample", value_name = "N", help_heading = "Coloring & palette")]
    pub supersample: Option<u32>,

    /// Orbit-trap shape (for `--fractal-coloring orbit-trap`): point | cross | circle.
    #[arg(long = "fractal-trap-shape", value_name = "SHAPE", help_heading = "Coloring & palette")]
    pub trap_shape: Option<String>,

    /// Orbit-trap center as `RE,IM`.
    #[arg(long = "fractal-trap-point", value_name = "RE,IM", allow_hyphen_values = true, help_heading = "Coloring & palette")]
    pub trap_point: Option<String>,

    /// Stripe-average angular frequency (for `--fractal-coloring stripe`).
    #[arg(long = "fractal-stripe-freq", value_name = "F", help_heading = "Coloring & palette")]
    pub stripe_freq: Option<f64>,

    /// Distance-estimate contrast (for `--fractal-coloring distance`; larger = thinner).
    #[arg(long = "fractal-de-scale", value_name = "S", help_heading = "Coloring & palette")]
    pub de_scale: Option<f64>,

    /// Buddhabrot sample count (for `--fractal-kind buddhabrot`).
    #[arg(long = "fractal-buddha-samples", value_name = "N", help_heading = "Per-family")]
    pub buddha_samples: Option<u64>,

    /// Seed for stochastic families (buddhabrot / ifs). Same seed → identical output.
    #[arg(long = "fractal-seed", value_name = "N", help_heading = "Fractal shape")]
    pub seed: Option<u64>,

    /// IFS preset (for `--fractal-kind ifs`): barnsley-fern | sierpinski | dragon | levy |
    /// tree | spiral.
    #[arg(long = "fractal-ifs-preset", value_name = "NAME", help_heading = "Per-family")]
    pub ifs_preset: Option<String>,

    /// IFS chaos-game point count (for `--fractal-kind ifs`).
    #[arg(long = "fractal-ifs-iterations", value_name = "N", help_heading = "Per-family")]
    pub ifs_iterations: Option<u64>,

    /// L-system preset (for `--fractal-kind lsystem`): koch | koch-snowflake | sierpinski |
    /// dragon | hilbert | gosper | plant | bush.
    #[arg(long = "fractal-lsystem-preset", value_name = "NAME", help_heading = "Per-family")]
    pub lsystem_preset: Option<String>,

    /// L-system turn angle in degrees (for `--fractal-kind lsystem`).
    #[arg(long = "fractal-lsystem-angle", value_name = "DEG", help_heading = "Per-family")]
    pub lsystem_angle: Option<f64>,

    /// L-system rewrite depth (for `--fractal-kind lsystem`; grows exponentially).
    #[arg(long = "fractal-lsystem-depth", value_name = "N", help_heading = "Per-family")]
    pub lsystem_depth: Option<u32>,

    /// Flame preset (for `--fractal-kind flame`): sierpinski | spherical | swirl | spiral | flame.
    #[arg(long = "fractal-flame-preset", value_name = "NAME", help_heading = "Per-family")]
    pub flame_preset: Option<String>,

    /// Flame rotational symmetry (1 = none).
    #[arg(long = "fractal-flame-symmetry", value_name = "N", help_heading = "Per-family")]
    pub flame_symmetry: Option<u32>,

    /// Flame chaos-game iteration count.
    #[arg(long = "fractal-flame-iterations", value_name = "N", help_heading = "Per-family")]
    pub flame_iterations: Option<u64>,

    /// Strange-attractor preset (for `--fractal-kind attractor`): clifford | dejong | bedhead |
    /// duffing | ikeda | lorenz | rossler | svensson | hopalong | fractal-dream.
    #[arg(long = "fractal-attractor-preset", value_name = "NAME", help_heading = "Per-family")]
    pub attractor_preset: Option<String>,

    /// Attractor trajectory step count.
    #[arg(long = "fractal-attractor-iterations", value_name = "N", help_heading = "Per-family")]
    pub attractor_iterations: Option<u64>,

    /// 3D shape (for `--fractal-kind raymarch`): mandelbulb | mandelbox | menger |
    /// sierpinski3d | quat-julia.
    #[arg(long = "fractal-raymarch-shape", value_name = "SHAPE", help_heading = "Per-family")]
    pub raymarch_shape: Option<String>,

    /// Mandelbulb exponent (for `--fractal-kind raymarch`; 8 is classic).
    #[arg(long = "fractal-raymarch-power", value_name = "P", allow_hyphen_values = true, help_heading = "Per-family")]
    pub raymarch_power: Option<f64>,

    /// Camera yaw in degrees (orbit around the 3D fractal).
    #[arg(long = "fractal-raymarch-yaw", value_name = "DEG", allow_hyphen_values = true, help_heading = "Per-family")]
    pub raymarch_yaw: Option<f64>,

    /// Camera pitch in degrees.
    #[arg(long = "fractal-raymarch-pitch", value_name = "DEG", allow_hyphen_values = true, help_heading = "Per-family")]
    pub raymarch_pitch: Option<f64>,

    /// Camera distance from the origin.
    #[arg(long = "fractal-raymarch-dist", value_name = "D", help_heading = "Per-family")]
    pub raymarch_dist: Option<f64>,

    /// Palette preset: fire | ice | electric | neon | pastel | monochrome | midnight | earth.
    #[arg(long = "fractal-palette", value_name = "NAME", help_heading = "Coloring & palette")]
    pub palette: Option<String>,

    /// Explicit gradient stops, comma-separated `#rrggbb` (overrides the preset).
    #[arg(long = "fractal-stops", value_name = "#hex,#hex,...", help_heading = "Coloring & palette")]
    pub stops: Option<String>,

    /// Color the fractal with a photo sampled at the orbit trap (the `plakat photos`
    /// bridge). Sets `--fractal-coloring image`. Best on Julia / Mandelbrot.
    #[arg(long = "fractal-trap-image", value_name = "IMAGE", help_heading = "Coloring & palette")]
    pub trap_image: Option<PathBuf>,

    /// Compose an R×C grid instead of one fractal: julia-sweep | zoom-grid | palette-grid
    /// | variation-sweep.
    #[arg(long = "fractal-compose", value_name = "MODE", help_heading = "Composition")]
    pub compose: Option<String>,

    /// Grid shape for `--fractal-compose` as `RxC` (default 4x4).
    #[arg(long = "fractal-grid", value_name = "RxC", help_heading = "Composition")]
    pub grid: Option<String>,

    /// Aesthetic keep-best for `--fractal-compose`: score every cell with the LAION
    /// predictor (same as `plakat rank`), highlight the top-K in the grid, and also write
    /// each as its own `<out>_best-<n>.png` (with embedded spec). Loads a small model.
    #[arg(long = "fractal-keep-best", value_name = "K", help_heading = "Composition")]
    pub keep_best: Option<usize>,

    /// Render an animation to video instead of a still: zoom | julia-sweep | param-sweep.
    /// Output format follows `--fractal-out`'s extension (.mp4 needs ffmpeg; .gif is
    /// pure-Rust). `zoom` zooms from 1× to `--fractal-zoom` (deep zooms use perturbation).
    #[arg(long = "fractal-animate", value_name = "MODE", help_heading = "Animation")]
    pub animate: Option<String>,

    /// Number of animation frames (default 120).
    #[arg(long = "fractal-frames", value_name = "N", help_heading = "Animation")]
    pub frames: Option<u32>,

    /// Animation frame rate (default 30).
    #[arg(long = "fractal-fps", value_name = "F", help_heading = "Animation")]
    pub fps: Option<u32>,

    /// Output PNG path (the deterministic Track-A render).
    #[arg(long = "fractal-out", value_name = "PATH", default_value = "out/fractal.png", help_heading = "Input & output")]
    pub out: PathBuf,

    /// Print the fully-resolved spec as JSON and exit without rendering.
    #[arg(long = "fractal-dump-spec", help_heading = "Input & output")]
    pub dump_spec: bool,

    /// Open the interactive TUI explorer: pan / zoom / retune live, then `s` to save to
    /// `--fractal-out`. Needs a graphics-capable terminal + the `ui` feature.
    #[arg(long = "fractal-explore", help_heading = "Input & output")]
    pub explore: bool,

    // ── Track B: optional AI enhancement pass (RFC FRACTALS-1, Phase 4) ──
    /// Enable the AI paint pass: repaint the Track-A render via ControlNet-guided img2img.
    /// The painted image is written next to `--fractal-out` (`<name>.painted.png`) unless
    /// `--fractal-paint-out` is given. Needs a model download + GPU for real speed.
    #[arg(long = "fractal-paint", help_heading = "AI paint (Track B)")]
    pub paint: bool,

    /// Painted-output path (implies `--fractal-paint`).
    #[arg(long = "fractal-paint-out", value_name = "PATH", help_heading = "AI paint (Track B)")]
    pub paint_out: Option<PathBuf>,

    /// Paint pipeline: `txt2img` (default — a scene *shaped by* the fractal: ControlNet-only,
    /// so the model paints a real sky / horizon / lighting from the prompt) or `img2img`
    /// (a scene *made of* the fractal: keeps its colors and layout, more abstract).
    #[arg(long = "fractal-paint-mode", value_name = "MODE", help_heading = "AI paint (Track B)")]
    pub paint_mode: Option<String>,

    /// Paint model alias (default sdxl).
    #[arg(long = "fractal-sd-model", value_name = "ALIAS", help_heading = "AI paint (Track B)")]
    pub sd_model: Option<String>,

    /// Paint prompt (default: a per-family auto prompt).
    #[arg(long = "fractal-prompt", value_name = "TEXT", help_heading = "AI paint (Track B)")]
    pub prompt: Option<String>,

    /// Paint negative prompt.
    #[arg(long = "fractal-negative", value_name = "TEXT", help_heading = "AI paint (Track B)")]
    pub negative: Option<String>,

    /// img2img strength in [0,1] (how far the repaint departs from the fractal).
    #[arg(long = "fractal-sd-strength", value_name = "S", help_heading = "AI paint (Track B)")]
    pub sd_strength: Option<f32>,

    /// Paint diffusion steps.
    #[arg(long = "fractal-sd-steps", value_name = "N", help_heading = "AI paint (Track B)")]
    pub sd_steps: Option<u32>,

    /// Paint CFG guidance scale.
    #[arg(long = "fractal-sd-guidance", value_name = "G", help_heading = "AI paint (Track B)")]
    pub sd_guidance: Option<f64>,

    /// ControlNet type override (default: per-family — canny / lineart / softedge).
    #[arg(long = "fractal-sd-control", value_name = "KIND", help_heading = "AI paint (Track B)")]
    pub sd_control: Option<String>,

    /// ControlNet conditioning scale.
    #[arg(long = "fractal-sd-control-strength", value_name = "S", help_heading = "AI paint (Track B)")]
    pub sd_control_strength: Option<f32>,

    /// Paint LoRA (repeatable): HF `org/name[:scale]`, `civitai:ID`, or a local path.
    #[arg(long = "fractal-sd-lora", value_name = "SPEC", help_heading = "AI paint (Track B)")]
    pub sd_lora: Vec<String>,

    /// Paint LoRA scale.
    #[arg(long = "fractal-sd-lora-scale", value_name = "S", help_heading = "AI paint (Track B)")]
    pub sd_lora_scale: Option<f32>,
}

fn parse_pair(s: &str, what: &str) -> Result<[f64; 2]> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        anyhow::bail!("{what} must be `RE,IM` (got {s:?})");
    }
    let re = parts[0].parse::<f64>().with_context(|| format!("{what} real part {:?}", parts[0]))?;
    let im = parts[1].parse::<f64>().with_context(|| format!("{what} imaginary part {:?}", parts[1]))?;
    Ok([re, im])
}

/// Derive the painted-output path from the Track-A path: `x/y.png` → `x/y.painted.png`.
fn painted_path(out: &Path) -> PathBuf {
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("fractal");
    let ext = out.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let name = format!("{stem}.painted.{ext}");
    match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(name),
        _ => PathBuf::from(name),
    }
}

/// Default an animation output to `.mp4` when the path is still a still-image extension.
fn anim_out_path(out: &Path) -> PathBuf {
    match out.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) {
        Some(e) if e == "mp4" || e == "gif" || e == "webm" => out.to_path_buf(),
        _ => out.with_extension("mp4"),
    }
}

fn parse_grid(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X', '×']).map(str::trim).collect();
    if parts.len() != 2 {
        anyhow::bail!("grid must be `RxC` (got {s:?})");
    }
    let r = parts[0].parse::<u32>().with_context(|| format!("rows {:?}", parts[0]))?;
    let c = parts[1].parse::<u32>().with_context(|| format!("cols {:?}", parts[1]))?;
    if r == 0 || c == 0 {
        anyhow::bail!("grid dimensions must be non-zero (got {s:?})");
    }
    Ok((r, c))
}

/// `foo.png` → `foo_best-3.png` (the individually-kept top cell for rank `n`, 1-based).
fn best_out_path(out: &Path, n: usize) -> PathBuf {
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("compose");
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let name = format!("{stem}_best-{n}.{ext}");
    out.parent().map(|p| p.join(&name)).unwrap_or_else(|| PathBuf::from(name))
}

/// Draw a `t`-px `[r,g,b]` frame just inside the cell at `(row, col)` of a `cw×ch`-per-cell
/// grid, into the packed-RGB `canvas` of width `gw`. Marks the aesthetic-best cells.
#[allow(clippy::too_many_arguments)]
fn highlight_cell(canvas: &mut [u8], gw: u32, cw: u32, ch: u32, row: u32, col: u32, rgb: [u8; 3], t: u32) {
    let (gw, cw, ch) = (gw as usize, cw as usize, ch as usize);
    let (x0, y0) = (col as usize * cw, row as usize * ch);
    let t = (t as usize).min(cw / 2).min(ch / 2).max(1);
    let mut put = |x: usize, y: usize| {
        let o = (y * gw + x) * 3;
        canvas[o..o + 3].copy_from_slice(&rgb);
    };
    for dy in 0..ch {
        for dx in 0..cw {
            if dx < t || dx >= cw - t || dy < t || dy >= ch - t {
                put(x0 + dx, y0 + dy);
            }
        }
    }
}

/// Aesthetic keep-best for a composed grid: score every cell with the LAION predictor,
/// highlight the top-`k` in the grid, save the grid, and emit each top cell as its own file.
#[allow(clippy::too_many_arguments)]
async fn compose_keep_best(
    grid: &mut fractals::RenderedFractal,
    cells: &[fractals::compose::CellRender],
    k: usize,
    mode_s: &str,
    rows: u32,
    cols: u32,
    spec: &FractalSpec,
    out: &Path,
    device_spec: &str,
    started: std::time::Instant,
) -> Result<()> {
    let device = crate::device::select(device_spec)?;
    let scorer = crate::pipelines::aesthetic::AestheticScorer::load(&device)
        .await
        .context("loading the aesthetic scorer for --fractal-keep-best")?;

    // Score each cell. Each is written to a temp PNG (carrying its own spec) so the same file
    // both feeds the scorer and becomes the kept output for the top-K.
    let scratch = tempfile::tempdir().context("keep-best scratch dir")?;
    let mut scored: Vec<(usize, f32, PathBuf)> = Vec::with_capacity(cells.len());
    let spb = ProgressBar::new(cells.len() as u64);
    spb.set_style(
        ProgressStyle::with_template("  {spinner:.magenta} scoring [{bar:30.magenta/blue}] {pos}/{len} cells")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
    );
    for (i, cell) in cells.iter().enumerate() {
        let p = scratch.path().join(format!("cell-{:03}.png", cell.idx));
        fractals::image_io::save_png_with_spec(
            &cell.rendered.pixels, cell.rendered.width, cell.rendered.height, &cell.spec, &p,
        )?;
        let s = scorer.score_path(&p).with_context(|| format!("scoring cell {}", cell.idx))?;
        scored.push((i, s, p));
        spb.inc(1);
    }
    spb.finish_and_clear();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let cw = (spec.width / cols).max(16);
    let ch = (spec.height / rows).max(16);
    let keep = k.min(scored.len());
    // Highlight the winners in the grid (gold), runners-up nothing.
    for (rank, (ci, _score, _p)) in scored.iter().take(keep).enumerate() {
        let cell = &cells[*ci];
        // Brightest gold for #1, dimming slightly down the ranking, so first place reads.
        let shade = 255u8.saturating_sub((rank as u8).saturating_mul(18));
        let border = (cw.min(ch) / 24).max(3);
        highlight_cell(&mut grid.pixels, grid.width, cw, ch, cell.row, cell.col, [shade, (shade as u32 * 3 / 4) as u8, 20], border);
    }
    fractals::image_io::save_png_with_spec(&grid.pixels, grid.width, grid.height, spec, out)?;

    // Emit each kept cell as its own file (copy the already-spec-embedded temp render).
    let mut kept_paths = Vec::with_capacity(keep);
    for (rank, (_ci, _score, p)) in scored.iter().take(keep).enumerate() {
        let dest = best_out_path(out, rank + 1);
        std::fs::copy(p, &dest).with_context(|| format!("writing kept cell {}", dest.display()))?;
        kept_paths.push(dest);
    }

    println!(
        "composed {mode_s} {rows}x{cols} {}x{} → {} ({:.2}s)",
        grid.width, grid.height, out.display(), started.elapsed().as_secs_f64()
    );
    println!("--fractal-keep-best: scored {} cells, kept top {keep}:", scored.len());
    for (rank, ((_ci, score, _p), dest)) in scored.iter().take(keep).zip(&kept_paths).enumerate() {
        println!("  #{:<2} {:6.3}  {}", rank + 1, score, dest.display());
    }
    Ok(())
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split(['x', 'X', '×']).map(str::trim).collect();
    if parts.len() != 2 {
        anyhow::bail!("size must be `WxH` (got {s:?})");
    }
    let w = parts[0].parse::<u32>().with_context(|| format!("width {:?}", parts[0]))?;
    let h = parts[1].parse::<u32>().with_context(|| format!("height {:?}", parts[1]))?;
    if w == 0 || h == 0 {
        anyhow::bail!("size dimensions must be non-zero (got {s:?})");
    }
    Ok((w, h))
}

/// Build the resolved spec from the base + CLI overrides.
/// Choose the base spec before CLI overrides: clone → spec-file → prose (offline keyword
/// mapper) → default.
fn resolve_base(args: &FractalsArgs) -> Result<FractalSpec> {
    Ok(if let Some(png) = &args.clone_from {
        fractals::spec::read_spec_chunk(png)?
            .with_context(|| format!("{} has no embedded fractalspec chunk", png.display()))?
    } else if let Some(file) = &args.spec {
        FractalSpec::load(file)?
    } else if let Some(prose) = &args.from {
        fractals::prompt::spec_from_prose(prose)
    } else {
        FractalSpec::default()
    })
}

/// Sync spec resolution (offline base + CLI overrides) — the test entry point; the runtime
/// path goes through `resolve_spec_async`.
#[cfg(test)]
fn resolve_spec(args: &FractalsArgs) -> Result<FractalSpec> {
    finish_spec(resolve_base(args)?, args)
}

/// Async spec resolution: LLM prose→spec when `--fractal-from` is paired with
/// `--fractal-provider` (falls back to the keyword mapper), else the offline base.
async fn resolve_spec_async(args: &FractalsArgs) -> Result<FractalSpec> {
    let base = match (&args.from, &args.provider) {
        (Some(prose), Some(provider)) => fractals::prompt::spec_from_prose_llm(prose, provider).await,
        _ => resolve_base(args)?,
    };
    finish_spec(base, args)
}

/// Apply all `--fractal-*` overrides to a base spec, then validate.
fn finish_spec(mut spec: FractalSpec, args: &FractalsArgs) -> Result<FractalSpec> {
    if let Some(k) = &args.kind {
        spec.kind = FractalKind::parse(k)?;
    }
    if let Some(c) = &args.center {
        spec.center = parse_pair(c, "center")?;
        // Preserve the full-precision decimal strings for perturbation deep zoom (used
        // only when zoom is deep; the f64 `center` drives the normal path).
        let parts: Vec<&str> = c.split(',').map(str::trim).collect();
        if parts.len() == 2 {
            spec.center_hi = [parts[0].to_string(), parts[1].to_string()];
        }
    }
    if let Some(z) = args.zoom {
        spec.zoom = z;
    }
    if let Some(n) = args.iter {
        spec.max_iter = n;
    }
    if let Some(jc) = &args.julia_c {
        spec.julia_c = parse_pair(jc, "julia-c")?;
    }
    if let Some(p) = args.power {
        spec.power = p;
    }
    if let Some(sz) = &args.size {
        let (w, h) = parse_size(sz)?;
        spec.width = w;
        spec.height = h;
    }
    if let Some(p) = &args.palette {
        spec.palette.preset = p.clone();
        spec.palette.stops.clear(); // an explicit preset clears any inherited stops
    }
    if let Some(stops) = &args.stops {
        spec.palette.stops = stops.split(',').map(|s| s.trim().to_string()).collect();
    }
    if let Some(img) = &args.trap_image {
        spec.trap_image = img.to_string_lossy().into_owned();
        spec.coloring = Coloring::Image; // providing a trap image implies image coloring
    }
    if let Some(c) = &args.coloring {
        spec.coloring = Coloring::parse(c)?; // explicit --fractal-coloring still wins
    }
    if let Some(ss) = args.supersample {
        spec.supersample = ss;
    }
    if let Some(sh) = &args.trap_shape {
        spec.trap.shape = TrapShape::parse(sh)?;
    }
    if let Some(tp) = &args.trap_point {
        spec.trap.point = parse_pair(tp, "trap-point")?;
    }
    if let Some(f) = args.stripe_freq {
        spec.stripe_freq = f;
    }
    if let Some(s) = args.de_scale {
        spec.de_scale = s;
    }
    if let Some(n) = args.buddha_samples {
        spec.buddha_samples = n;
    }
    if let Some(s) = args.seed {
        spec.seed = s;
    }
    if let Some(p) = &args.ifs_preset {
        spec.ifs.preset = p.clone();
        spec.ifs.maps.clear();
    }
    if let Some(n) = args.ifs_iterations {
        spec.ifs.iterations = n;
    }
    if let Some(p) = &args.lsystem_preset {
        spec.lsystem.preset = p.clone();
        spec.lsystem.axiom.clear();
    }
    if let Some(a) = args.lsystem_angle {
        spec.lsystem.angle = a;
    }
    if let Some(d) = args.lsystem_depth {
        spec.lsystem.iterations = d;
    }
    if let Some(p) = &args.flame_preset {
        spec.flame.preset = p.clone();
        spec.flame.functions.clear();
    }
    if let Some(s) = args.flame_symmetry {
        spec.flame.symmetry = s;
    }
    if let Some(n) = args.flame_iterations {
        spec.flame.iterations = n;
    }
    if let Some(p) = &args.attractor_preset {
        spec.attractor.preset = p.clone();
        spec.attractor.params.clear();
    }
    if let Some(n) = args.attractor_iterations {
        spec.attractor.iterations = n;
    }
    if let Some(s) = &args.raymarch_shape {
        spec.raymarch.shape = s.clone();
    }
    if let Some(p) = args.raymarch_power {
        spec.raymarch.power = p;
    }
    if let Some(y) = args.raymarch_yaw {
        spec.raymarch.camera_yaw = y;
    }
    if let Some(p) = args.raymarch_pitch {
        spec.raymarch.camera_pitch = p;
    }
    if let Some(d) = args.raymarch_dist {
        spec.raymarch.camera_dist = d;
    }

    // Track B (AI paint).
    if args.paint || args.paint_out.is_some() {
        spec.ai.enabled = true;
    }
    if let Some(m) = &args.paint_mode {
        spec.ai.mode = m.clone();
        // The default (txt2img) uses a looser control (0.4, set in AiSpec). img2img anchors
        // on the init too, so it reads a touch stronger — bump unless the user set one.
        let img = ["img2img", "i2i"].iter().any(|k| m.eq_ignore_ascii_case(k));
        if img && args.sd_control_strength.is_none() {
            spec.ai.control_strength = 0.55;
        }
    }
    if let Some(m) = &args.sd_model {
        spec.ai.model = m.clone();
    }
    if let Some(p) = &args.prompt {
        spec.ai.prompt = p.clone();
    }
    if let Some(nprompt) = &args.negative {
        spec.ai.negative = nprompt.clone();
    }
    if let Some(s) = args.sd_strength {
        spec.ai.strength = s;
    }
    if let Some(n) = args.sd_steps {
        spec.ai.steps = n;
    }
    if let Some(g) = args.sd_guidance {
        spec.ai.guidance = g;
    }
    if let Some(c) = &args.sd_control {
        spec.ai.control = c.clone();
    }
    if let Some(s) = args.sd_control_strength {
        spec.ai.control_strength = s;
    }
    if !args.sd_lora.is_empty() {
        spec.ai.loras = args.sd_lora.clone();
    }
    if let Some(s) = args.sd_lora_scale {
        spec.ai.lora_scale = s;
    }

    spec.validate()?;
    Ok(spec)
}

pub async fn run(args: FractalsArgs, device_spec: &str) -> Result<()> {
    let spec = resolve_spec_async(&args).await?;

    // Gently steer users who put a *scene* into --fractal-from (it shapes the fractal, not
    // a painted scene). Only for the offline keyword path (no provider), when they named no
    // fractal family and aren't already painting.
    if let (Some(from), None) = (&args.from, &args.provider) {
        if !spec.ai.enabled && !fractals::prompt::names_a_family(from) {
            eprintln!(
                "note: --fractal-from shapes the FRACTAL (it found no family name, so it \
                 derived one from your words). To paint a scene from a description, add \
                 `--fractal-paint --fractal-prompt \"{from}\"`."
            );
        }
    }

    if args.dump_spec {
        println!("{}", spec.to_json()?);
        return Ok(());
    }

    if args.explore {
        #[cfg(feature = "ui")]
        {
            return fractals::explorer::run(spec, args.out.clone());
        }
        #[cfg(not(feature = "ui"))]
        {
            let _ = &spec;
            anyhow::bail!(
                "--fractal-explore needs the TUI stack — rebuild with `--features fractals,ui` \
                 (the default build includes it)"
            );
        }
    }

    let started = std::time::Instant::now();

    // A live progress bar driven by the renderer's callback (fires from worker threads).
    // Declared here so the compose branch and the single-render path can both drive it.
    let pb = ProgressBar::new(1);

    // Composition mode: an R×C grid of related sub-fractals.
    if let Some(mode_s) = &args.compose {
        let mode = fractals::compose::ComposeMode::parse(mode_s)?;
        let (rows, cols) = match &args.grid {
            Some(g) => parse_grid(g)?,
            None => (4, 4),
        };
        pb.set_style(
            ProgressStyle::with_template("  {spinner:.cyan} composing [{bar:30.cyan/blue}] {pos}/{len} cells")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        let report = |done: u64, total: u64| {
            if pb.length() != Some(total) {
                pb.set_length(total.max(1));
            }
            pb.set_position(done);
        };
        // Plain grid vs aesthetic keep-best (score + highlight + emit the top-K cells).
        match args.keep_best {
            None => {
                let r = fractals::compose::compose(&spec, mode, rows, cols, &report)?;
                pb.finish_and_clear();
                fractals::image_io::save_png_with_spec(&r.pixels, r.width, r.height, &spec, &args.out)?;
                println!(
                    "composed {mode_s} {rows}x{cols} {}x{} → {} ({:.2}s)",
                    r.width, r.height, args.out.display(), started.elapsed().as_secs_f64()
                );
            }
            Some(k) => {
                let (mut grid, cells) = fractals::compose::compose_cells(&spec, mode, rows, cols, &report)?;
                pb.finish_and_clear();
                compose_keep_best(&mut grid, &cells, k, mode_s, rows, cols, &spec, &args.out, device_spec, started).await?;
            }
        }
        return Ok(());
    }

    // Animation mode: render frames + encode to video.
    if let Some(mode_s) = &args.animate {
        let mode = fractals::animation::AnimMode::parse(mode_s)?;
        let frames = args.frames.unwrap_or(120);
        let fps = args.fps.unwrap_or(30);
        let out = anim_out_path(&args.out);
        pb.set_style(
            ProgressStyle::with_template("  {spinner:.cyan} frame [{bar:30.cyan/blue}] {pos}/{len}  {elapsed}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        let report = |done: u64, total: u64| {
            if pb.length() != Some(total) {
                pb.set_length(total.max(1));
            }
            pb.set_position(done);
        };
        fractals::animation::render_animation(&spec, mode, frames, fps, &out, &report)?;
        pb.finish_and_clear();
        println!(
            "animated {mode_s} {frames} frames @ {fps}fps → {} ({:.1}s)",
            out.display(),
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    pb.set_style(
        ProgressStyle::with_template(
            "  {spinner:.cyan} {msg} [{bar:30.cyan/blue}] {percent:>3}%  {elapsed}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb.set_message(format!("rendering {}", spec.kind.as_str()));
    let report = |done: u64, total: u64| {
        if pb.length() != Some(total) {
            pb.set_length(total.max(1));
        }
        pb.set_position(done);
    };
    fractals::render_to_file_with_progress(&spec, &args.out, &report)?;
    pb.finish_and_clear();

    let dt = started.elapsed();
    println!(
        "fractal {} {}x{} → {} ({:.2}s)",
        spec.kind.as_str(),
        spec.width,
        spec.height,
        args.out.display(),
        dt.as_secs_f64()
    );

    // Track B — optional AI paint pass. The device (GPU) is resolved lazily, only when
    // painting, so Track A stays entirely device-free.
    if spec.ai.enabled {
        let paint_out = args.paint_out.clone().unwrap_or_else(|| painted_path(&args.out));
        let mode = if fractals::ai_pass::is_txt2img(&spec) { "txt2img" } else { "img2img" };
        let control = if spec.ai.control.trim().is_empty() {
            fractals::ai_pass::default_control_for_kind(spec.kind).slug().to_string()
        } else {
            spec.ai.control.clone()
        };
        if mode == "txt2img" {
            println!("painting via {} ({mode}, control: {control} {})…", spec.ai.model, spec.ai.control_strength);
        } else {
            println!(
                "painting via {} ({mode}, control: {control}, strength {})…",
                spec.ai.model, spec.ai.strength,
            );
        }
        // Device resolution honors `--device`: the default `auto` auto-detects and uses
        // the GPU (CUDA → Metal → CPU); an explicit `--device cpu` is respected; an
        // explicit `metal` / `cuda[:N]` is honored (and errors clearly if not built in).
        let device = crate::device::select(device_spec)?;
        let label = crate::device::label(&device);
        println!("painting on {label}…");
        if device.is_cpu() && device_spec.trim().eq_ignore_ascii_case("auto") {
            eprintln!(
                "  note: no GPU backend detected — painting on CPU (slow). Rebuild with \
                 `--features metal` (macOS) or `--features cuda` (NVIDIA) for GPU."
            );
        }
        fractals::ai_pass::run_ai_pass(&spec, &args.out, &paint_out, device).await?;
        println!("painted fractal → {}", paint_out.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> FractalsArgs {
        FractalsArgs {
            from: None,
            provider: None,
            spec: None,
            clone_from: None,
            trap_image: None,
            compose: None,
            keep_best: None,
            grid: None,
            animate: None,
            frames: None,
            fps: None,
            kind: None,
            center: None,
            zoom: None,
            iter: None,
            julia_c: None,
            power: None,
            size: None,
            palette: None,
            stops: None,
            coloring: None,
            supersample: None,
            trap_shape: None,
            trap_point: None,
            stripe_freq: None,
            de_scale: None,
            buddha_samples: None,
            seed: None,
            ifs_preset: None,
            ifs_iterations: None,
            lsystem_preset: None,
            lsystem_angle: None,
            lsystem_depth: None,
            flame_preset: None,
            flame_symmetry: None,
            flame_iterations: None,
            attractor_preset: None,
            attractor_iterations: None,
            raymarch_shape: None,
            raymarch_power: None,
            raymarch_yaw: None,
            raymarch_pitch: None,
            raymarch_dist: None,
            paint: false,
            paint_out: None,
            paint_mode: None,
            sd_model: None,
            prompt: None,
            negative: None,
            sd_strength: None,
            sd_steps: None,
            sd_guidance: None,
            sd_control: None,
            sd_control_strength: None,
            sd_lora: Vec::new(),
            sd_lora_scale: None,
            out: PathBuf::from("out/fractal.png"),
            dump_spec: false,
            explore: false,
        }
    }

    #[test]
    fn parse_helpers() {
        assert_eq!(parse_pair("-0.5, 0.25", "c").unwrap(), [-0.5, 0.25]);
        assert!(parse_pair("1.0", "c").is_err());
        assert_eq!(parse_size("1920x1080").unwrap(), (1920, 1080));
        assert!(parse_size("100").is_err());
        assert!(parse_size("0x100").is_err());
    }

    #[test]
    fn cli_overrides_apply() {
        let args = FractalsArgs {
            kind: Some("julia".into()),
            center: Some("0.1,0.2".into()),
            zoom: Some(3.0),
            iter: Some(300),
            size: Some("640x480".into()),
            palette: Some("ice".into()),
            ..base_args()
        };
        let spec = resolve_spec(&args).unwrap();
        assert_eq!(spec.kind, FractalKind::Julia);
        assert_eq!(spec.center, [0.1, 0.2]);
        assert_eq!(spec.zoom, 3.0);
        assert_eq!(spec.max_iter, 300);
        assert_eq!((spec.width, spec.height), (640, 480));
        assert_eq!(spec.palette.preset, "ice");
    }

    #[test]
    fn explicit_stops_override_preset() {
        let args = FractalsArgs {
            stops: Some("#000000, #ffffff".into()),
            ..base_args()
        };
        let spec = resolve_spec(&args).unwrap();
        assert_eq!(spec.palette.stops, vec!["#000000", "#ffffff"]);
    }

    #[test]
    fn phase2_overrides_apply() {
        let args = FractalsArgs {
            kind: Some("tricorn".into()),
            coloring: Some("orbit-trap".into()),
            supersample: Some(2),
            trap_shape: Some("circle".into()),
            trap_point: Some("0.5,-0.5".into()),
            stripe_freq: Some(9.0),
            seed: Some(123),
            ..base_args()
        };
        let spec = resolve_spec(&args).unwrap();
        assert_eq!(spec.kind, FractalKind::Tricorn);
        assert_eq!(spec.coloring, Coloring::OrbitTrap);
        assert_eq!(spec.supersample, 2);
        assert_eq!(spec.trap.shape, TrapShape::Circle);
        assert_eq!(spec.trap.point, [0.5, -0.5]);
        assert_eq!(spec.stripe_freq, 9.0);
        assert_eq!(spec.seed, 123);
    }

    #[test]
    fn bad_coloring_errors() {
        let args = FractalsArgs { coloring: Some("rainbow".into()), ..base_args() };
        assert!(resolve_spec(&args).is_err());
    }

    #[test]
    fn paint_flag_enables_ai_and_applies_knobs() {
        let args = FractalsArgs {
            paint: true,
            sd_model: Some("sd15".into()),
            sd_strength: Some(0.4),
            sd_control: Some("softedge".into()),
            sd_lora: vec!["org/frac:0.7".into()],
            ..base_args()
        };
        let spec = resolve_spec(&args).unwrap();
        assert!(spec.ai.enabled);
        assert_eq!(spec.ai.model, "sd15");
        assert_eq!(spec.ai.strength, 0.4);
        assert_eq!(spec.ai.control, "softedge");
        assert_eq!(spec.ai.loras, vec!["org/frac:0.7"]);
        // Not painting by default.
        assert!(!resolve_spec(&base_args()).unwrap().ai.enabled);
    }

    #[test]
    fn paint_out_implies_enabled_and_derived_path() {
        let args = FractalsArgs { paint_out: Some(PathBuf::from("x/y.png")), ..base_args() };
        assert!(resolve_spec(&args).unwrap().ai.enabled);
        assert_eq!(painted_path(Path::new("out/frac.png")), PathBuf::from("out/frac.painted.png"));
        assert_eq!(painted_path(Path::new("frac.png")), PathBuf::from("frac.painted.png"));
    }

    #[test]
    fn default_resolves_and_validates() {
        let spec = resolve_spec(&base_args()).unwrap();
        assert_eq!(spec.kind, FractalKind::Mandelbrot);
        assert!(spec.validate().is_ok());
    }
}
