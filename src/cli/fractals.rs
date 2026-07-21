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
    /// Load the base spec from an HJSON/JSON file (CLI flags still override its fields).
    #[arg(long = "fractal-spec", value_name = "FILE")]
    pub spec: Option<PathBuf>,

    /// Reconstruct the base spec from a PNG previously written by `plakat fractals`
    /// (reads its embedded `fractalspec` chunk). Takes precedence over `--fractal-spec`.
    #[arg(long = "fractal-clone", value_name = "PNG")]
    pub clone_from: Option<PathBuf>,

    /// Fractal family: mandelbrot | julia | burning-ship.
    #[arg(long = "fractal-kind", value_name = "KIND")]
    pub kind: Option<String>,

    /// Viewport center in the complex plane as `RE,IM`.
    #[arg(long = "fractal-center", value_name = "RE,IM")]
    pub center: Option<String>,

    /// Zoom factor (vertical axis spans `3.0 / zoom` complex units).
    #[arg(long = "fractal-zoom", value_name = "Z")]
    pub zoom: Option<f64>,

    /// Iteration cap (escape budget).
    #[arg(long = "fractal-iter", value_name = "N")]
    pub iter: Option<u32>,

    /// Julia constant as `RE,IM` (only used when kind = julia).
    #[arg(long = "fractal-julia-c", value_name = "RE,IM")]
    pub julia_c: Option<String>,

    /// Exponent for the `z^power` step (2 = classic; other = multibrot / Newton degree).
    #[arg(long = "fractal-power", value_name = "P")]
    pub power: Option<f64>,

    /// Output size as `WxH` (e.g. `1920x1080`).
    #[arg(long = "fractal-size", value_name = "WxH")]
    pub size: Option<String>,

    /// Coloring: smooth | histogram | distance | orbit-trap | angle | stripe.
    #[arg(long = "fractal-coloring", value_name = "MODE")]
    pub coloring: Option<String>,

    /// Anti-aliasing: render at NxN samples per pixel then downsample (1..=8; 1 = off).
    #[arg(long = "fractal-supersample", value_name = "N")]
    pub supersample: Option<u32>,

    /// Orbit-trap shape (for `--fractal-coloring orbit-trap`): point | cross | circle.
    #[arg(long = "fractal-trap-shape", value_name = "SHAPE")]
    pub trap_shape: Option<String>,

    /// Orbit-trap center as `RE,IM`.
    #[arg(long = "fractal-trap-point", value_name = "RE,IM")]
    pub trap_point: Option<String>,

    /// Stripe-average angular frequency (for `--fractal-coloring stripe`).
    #[arg(long = "fractal-stripe-freq", value_name = "F")]
    pub stripe_freq: Option<f64>,

    /// Distance-estimate contrast (for `--fractal-coloring distance`; larger = thinner).
    #[arg(long = "fractal-de-scale", value_name = "S")]
    pub de_scale: Option<f64>,

    /// Buddhabrot sample count (for `--fractal-kind buddhabrot`).
    #[arg(long = "fractal-buddha-samples", value_name = "N")]
    pub buddha_samples: Option<u64>,

    /// Seed for stochastic families (buddhabrot / ifs). Same seed → identical output.
    #[arg(long = "fractal-seed", value_name = "N")]
    pub seed: Option<u64>,

    /// IFS preset (for `--fractal-kind ifs`): barnsley-fern | sierpinski | dragon | levy |
    /// tree | spiral.
    #[arg(long = "fractal-ifs-preset", value_name = "NAME")]
    pub ifs_preset: Option<String>,

    /// IFS chaos-game point count (for `--fractal-kind ifs`).
    #[arg(long = "fractal-ifs-iterations", value_name = "N")]
    pub ifs_iterations: Option<u64>,

    /// L-system preset (for `--fractal-kind lsystem`): koch | koch-snowflake | sierpinski |
    /// dragon | hilbert | gosper | plant | bush.
    #[arg(long = "fractal-lsystem-preset", value_name = "NAME")]
    pub lsystem_preset: Option<String>,

    /// L-system turn angle in degrees (for `--fractal-kind lsystem`).
    #[arg(long = "fractal-lsystem-angle", value_name = "DEG")]
    pub lsystem_angle: Option<f64>,

    /// L-system rewrite depth (for `--fractal-kind lsystem`; grows exponentially).
    #[arg(long = "fractal-lsystem-depth", value_name = "N")]
    pub lsystem_depth: Option<u32>,

    /// Flame preset (for `--fractal-kind flame`): sierpinski | spherical | swirl | spiral | flame.
    #[arg(long = "fractal-flame-preset", value_name = "NAME")]
    pub flame_preset: Option<String>,

    /// Flame rotational symmetry (1 = none).
    #[arg(long = "fractal-flame-symmetry", value_name = "N")]
    pub flame_symmetry: Option<u32>,

    /// Flame chaos-game iteration count.
    #[arg(long = "fractal-flame-iterations", value_name = "N")]
    pub flame_iterations: Option<u64>,

    /// Strange-attractor preset (for `--fractal-kind attractor`): clifford | dejong | bedhead |
    /// duffing | ikeda | lorenz | rossler.
    #[arg(long = "fractal-attractor-preset", value_name = "NAME")]
    pub attractor_preset: Option<String>,

    /// Attractor trajectory step count.
    #[arg(long = "fractal-attractor-iterations", value_name = "N")]
    pub attractor_iterations: Option<u64>,

    /// Palette preset: fire | ice | electric | neon | pastel | monochrome | midnight | earth.
    #[arg(long = "fractal-palette", value_name = "NAME")]
    pub palette: Option<String>,

    /// Explicit gradient stops, comma-separated `#rrggbb` (overrides the preset).
    #[arg(long = "fractal-stops", value_name = "#hex,#hex,...")]
    pub stops: Option<String>,

    /// Output PNG path (the deterministic Track-A render).
    #[arg(long = "fractal-out", value_name = "PATH", default_value = "out/fractal.png")]
    pub out: PathBuf,

    /// Print the fully-resolved spec as JSON and exit without rendering.
    #[arg(long = "fractal-dump-spec")]
    pub dump_spec: bool,

    /// Open the interactive TUI explorer: pan / zoom / retune live, then `s` to save to
    /// `--fractal-out`. Needs a graphics-capable terminal + the `ui` feature.
    #[arg(long = "fractal-explore")]
    pub explore: bool,

    // ── Track B: optional AI enhancement pass (RFC FRACTALS-1, Phase 4) ──
    /// Enable the AI paint pass: repaint the Track-A render via ControlNet-guided img2img.
    /// The painted image is written next to `--fractal-out` (`<name>.painted.png`) unless
    /// `--fractal-paint-out` is given. Needs a model download + GPU for real speed.
    #[arg(long = "fractal-paint")]
    pub paint: bool,

    /// Painted-output path (implies `--fractal-paint`).
    #[arg(long = "fractal-paint-out", value_name = "PATH")]
    pub paint_out: Option<PathBuf>,

    /// Paint model alias (default sdxl).
    #[arg(long = "fractal-sd-model", value_name = "ALIAS")]
    pub sd_model: Option<String>,

    /// Paint prompt (default: a per-family auto prompt).
    #[arg(long = "fractal-prompt", value_name = "TEXT")]
    pub prompt: Option<String>,

    /// Paint negative prompt.
    #[arg(long = "fractal-negative", value_name = "TEXT")]
    pub negative: Option<String>,

    /// img2img strength in [0,1] (how far the repaint departs from the fractal).
    #[arg(long = "fractal-sd-strength", value_name = "S")]
    pub sd_strength: Option<f32>,

    /// Paint diffusion steps.
    #[arg(long = "fractal-sd-steps", value_name = "N")]
    pub sd_steps: Option<u32>,

    /// Paint CFG guidance scale.
    #[arg(long = "fractal-sd-guidance", value_name = "G")]
    pub sd_guidance: Option<f64>,

    /// ControlNet type override (default: per-family — canny / lineart / softedge).
    #[arg(long = "fractal-sd-control", value_name = "KIND")]
    pub sd_control: Option<String>,

    /// ControlNet conditioning scale.
    #[arg(long = "fractal-sd-control-strength", value_name = "S")]
    pub sd_control_strength: Option<f32>,

    /// Paint LoRA (repeatable): HF `org/name[:scale]`, `civitai:ID`, or a local path.
    #[arg(long = "fractal-sd-lora", value_name = "SPEC")]
    pub sd_lora: Vec<String>,

    /// Paint LoRA scale.
    #[arg(long = "fractal-sd-lora-scale", value_name = "S")]
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
fn resolve_spec(args: &FractalsArgs) -> Result<FractalSpec> {
    let mut spec = if let Some(png) = &args.clone_from {
        fractals::spec::read_spec_chunk(png)?
            .with_context(|| format!("{} has no embedded fractalspec chunk", png.display()))?
    } else if let Some(file) = &args.spec {
        FractalSpec::load(file)?
    } else {
        FractalSpec::default()
    };

    if let Some(k) = &args.kind {
        spec.kind = FractalKind::parse(k)?;
    }
    if let Some(c) = &args.center {
        spec.center = parse_pair(c, "center")?;
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
    if let Some(c) = &args.coloring {
        spec.coloring = Coloring::parse(c)?;
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

    // Track B (AI paint).
    if args.paint || args.paint_out.is_some() {
        spec.ai.enabled = true;
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
    let spec = resolve_spec(&args)?;

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
    let pb = ProgressBar::new(1);
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
        println!(
            "painting via {} (control: {}, strength {})…",
            spec.ai.model,
            if spec.ai.control.trim().is_empty() {
                fractals::ai_pass::default_control_for_kind(spec.kind).slug().to_string()
            } else {
                spec.ai.control.clone()
            },
            spec.ai.strength,
        );
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
            spec: None,
            clone_from: None,
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
            paint: false,
            paint_out: None,
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
