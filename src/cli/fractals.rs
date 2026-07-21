//! `plakat fractals` — Track-A (pure-CPU) fractal rendering CLI (RFC FRACTALS-1, Phase 1).
//!
//! Resolution order for the spec: a base (`--fractal-clone PATH` → `--fractal-spec FILE`
//! → built-in default), then any explicitly-provided `--fractal-*` flag overrides that
//! base. `--fractal-dump-spec` prints the fully-resolved spec as JSON and renders nothing.

use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

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

    /// Seed for stochastic families (buddhabrot). Same seed → identical output.
    #[arg(long = "fractal-seed", value_name = "N")]
    pub seed: Option<u64>,

    /// Palette preset: fire | ice | electric | neon | pastel | monochrome | midnight | earth.
    #[arg(long = "fractal-palette", value_name = "NAME")]
    pub palette: Option<String>,

    /// Explicit gradient stops, comma-separated `#rrggbb` (overrides the preset).
    #[arg(long = "fractal-stops", value_name = "#hex,#hex,...")]
    pub stops: Option<String>,

    /// Output PNG path.
    #[arg(long = "fractal-out", value_name = "PATH", default_value = "out/fractal.png")]
    pub out: PathBuf,

    /// Print the fully-resolved spec as JSON and exit without rendering.
    #[arg(long = "fractal-dump-spec")]
    pub dump_spec: bool,
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

    spec.validate()?;
    Ok(spec)
}

pub async fn run(args: FractalsArgs) -> Result<()> {
    let spec = resolve_spec(&args)?;

    if args.dump_spec {
        println!("{}", spec.to_json()?);
        return Ok(());
    }

    let started = std::time::Instant::now();
    fractals::render_to_file(&spec, &args.out)?;
    let dt = started.elapsed();
    println!(
        "fractal {} {}x{} → {} ({:.2}s)",
        spec.kind.as_str(),
        spec.width,
        spec.height,
        args.out.display(),
        dt.as_secs_f64()
    );
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
            out: PathBuf::from("out/fractal.png"),
            dump_spec: false,
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
    fn default_resolves_and_validates() {
        let spec = resolve_spec(&base_args()).unwrap();
        assert_eq!(spec.kind, FractalKind::Mandelbrot);
        assert!(spec.validate().is_ok());
    }
}
