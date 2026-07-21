//! `FractalSpec` — the single authoritative, seed-stable, human-writable (HJSON)
//! description of a fractal render. Embedded as a `fractalspec` tEXt chunk in every
//! output PNG so `plakat fractals --fractal-clone PATH` can reconstruct the exact image.
//!
//! RFC FRACTALS-1, Phases 1–2. Escape-time only for now (11 families + buddhabrot);
//! later phases add `ifs` / `lsystem` / `flame` / `attractor` / `raymarch`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Which escape-time fractal to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FractalKind {
    /// z ← z^power + c, with c the pixel and z₀ = 0.
    Mandelbrot,
    /// z ← z^power + c, with c a fixed constant (`julia_c`) and z₀ = the pixel.
    Julia,
    /// z ← (|Re z| + i|Im z|)^power + c — the Burning Ship.
    BurningShip,
    /// z ← conj(z)^power + c — the Mandelbar / Tricorn.
    Tricorn,
    /// z ← z^power + c with power > 2 — the Multibrot (same recurrence as Mandelbrot,
    /// named separately for discoverability; set `power`).
    Multibrot,
    /// Newton's method on z^degree − 1 (degree = round(power)); colored by convergence.
    Newton,
    /// Nova (relaxed Newton with an additive c): z ← z − relax·f/f′ + c.
    Nova,
    /// Phoenix: zₙ₊₁ = zₙ² + c + p·zₙ₋₁ (uses the previous iterate; z₀ = pixel).
    Phoenix,
    /// Magnet type I: z ← ((z² + c − 1)/(2z + c − 2))².
    Magnet,
    /// z ← c · sin(z) (transcendental; z₀ = pixel).
    Sine,
    /// z ← c · exp(z) (transcendental; z₀ = pixel).
    Exp,
    /// Buddhabrot — density plot of escaping Mandelbrot orbits (stochastic, seeded).
    Buddhabrot,
    /// Iterated Function System — chaos-game point attractor (Barnsley fern, Sierpiński…).
    Ifs,
    /// L-system — Lindenmayer rewriting + turtle line drawing (Koch, dragon, plants…).
    Lsystem,
}

impl FractalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => "mandelbrot",
            FractalKind::Julia => "julia",
            FractalKind::BurningShip => "burning-ship",
            FractalKind::Tricorn => "tricorn",
            FractalKind::Multibrot => "multibrot",
            FractalKind::Newton => "newton",
            FractalKind::Nova => "nova",
            FractalKind::Phoenix => "phoenix",
            FractalKind::Magnet => "magnet",
            FractalKind::Sine => "sine",
            FractalKind::Exp => "exp",
            FractalKind::Buddhabrot => "buddhabrot",
            FractalKind::Ifs => "ifs",
            FractalKind::Lsystem => "lsystem",
        }
    }

    /// Buddhabrot renders via density accumulation, not a per-pixel escape field.
    pub fn is_buddhabrot(self) -> bool {
        self == FractalKind::Buddhabrot
    }

    /// The per-pixel complex-plane escape families (everything except buddhabrot / the
    /// line-drawing families).
    pub fn is_escape_time(self) -> bool {
        !matches!(self, FractalKind::Buddhabrot | FractalKind::Ifs | FractalKind::Lsystem)
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "mandelbrot" | "mandel" | "m" => Ok(FractalKind::Mandelbrot),
            "julia" | "j" => Ok(FractalKind::Julia),
            "burning-ship" | "burningship" | "ship" | "bs" => Ok(FractalKind::BurningShip),
            "tricorn" | "mandelbar" => Ok(FractalKind::Tricorn),
            "multibrot" | "multi" => Ok(FractalKind::Multibrot),
            "newton" => Ok(FractalKind::Newton),
            "nova" => Ok(FractalKind::Nova),
            "phoenix" => Ok(FractalKind::Phoenix),
            "magnet" => Ok(FractalKind::Magnet),
            "sine" | "sin" => Ok(FractalKind::Sine),
            "exp" | "exponential" => Ok(FractalKind::Exp),
            "buddhabrot" | "buddha" => Ok(FractalKind::Buddhabrot),
            "ifs" => Ok(FractalKind::Ifs),
            "lsystem" | "l-system" | "lsys" => Ok(FractalKind::Lsystem),
            other => anyhow::bail!(
                "unknown fractal kind {other:?} (want: mandelbrot | julia | burning-ship | \
                 tricorn | multibrot | newton | nova | phoenix | magnet | sine | exp | buddhabrot \
                 | ifs | lsystem)"
            ),
        }
    }
}

/// The palette + coloring description. A `preset` name OR an explicit list of `#rrggbb`
/// stops (stops win if both are given). `cycles`/`offset` shape the gradient sweep;
/// `cyclic` repeats it instead of clamping; `interior` colors non-escaping points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PaletteSpec {
    /// Named preset: fire · ice · electric · neon · pastel · monochrome · midnight · earth.
    pub preset: String,
    /// Explicit `#rrggbb` gradient stops (overrides `preset` when non-empty).
    pub stops: Vec<String>,
    /// Gradient sweeps per full escape range (>1 → banding).
    pub cycles: f64,
    /// Phase offset added before sampling (rotates the gradient).
    pub offset: f64,
    /// Repeat the gradient (`fract`) instead of clamping at the ends.
    pub cyclic: bool,
    /// Interior (non-escaping) color as `#rrggbb`.
    pub interior: String,
}

impl Default for PaletteSpec {
    fn default() -> Self {
        PaletteSpec {
            preset: "fire".to_string(),
            stops: Vec::new(),
            cycles: 1.0,
            offset: 0.0,
            cyclic: false,
            interior: "#000000".to_string(),
        }
    }
}

/// Coloring algorithm applied to the escape result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Coloring {
    /// Continuous (Bernstein-smoothed) escape count — no visible iteration bands.
    #[default]
    Smooth,
    /// Histogram-equalized iteration count — even color distribution across the frame.
    Histogram,
    /// Boundary distance estimate — thin, evenly-lit filaments (needs a holomorphic family).
    Distance,
    /// Orbit trap — color by the closest approach of the orbit to a shape (`trap`).
    OrbitTrap,
    /// Final-iterate argument (angle) — good for Newton-basin coloring.
    Angle,
    /// Stripe average — smooth banded "flame" texturing from the orbit's angular history.
    Stripe,
}

impl Coloring {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "smooth" => Ok(Coloring::Smooth),
            "histogram" | "hist" => Ok(Coloring::Histogram),
            "distance" | "de" => Ok(Coloring::Distance),
            "orbit-trap" | "trap" => Ok(Coloring::OrbitTrap),
            "angle" => Ok(Coloring::Angle),
            "stripe" | "stripe-average" => Ok(Coloring::Stripe),
            other => anyhow::bail!(
                "unknown coloring {other:?} (want: smooth | histogram | distance | orbit-trap | \
                 angle | stripe)"
            ),
        }
    }
}

/// Orbit-trap shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum TrapShape {
    /// Distance to a point.
    #[default]
    Point,
    /// Distance to the nearer of the two axes through the trap point (a cross).
    Cross,
    /// Distance to a circle of radius `radius` centered on the trap point.
    Circle,
}

impl TrapShape {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "point" => Ok(TrapShape::Point),
            "cross" => Ok(TrapShape::Cross),
            "circle" => Ok(TrapShape::Circle),
            other => anyhow::bail!("unknown trap shape {other:?} (want: point | cross | circle)"),
        }
    }
}

/// Orbit-trap configuration (used when `coloring = orbit-trap`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrapSpec {
    pub shape: TrapShape,
    /// Trap center `[re, im]`.
    pub point: [f64; 2],
    /// Circle radius (for `shape = circle`).
    pub radius: f64,
    /// Contrast: larger squeezes the mapped range (`t = tanh(dist · scale)`).
    pub scale: f64,
}

impl Default for TrapSpec {
    fn default() -> Self {
        TrapSpec { shape: TrapShape::Point, point: [0.0, 0.0], radius: 0.5, scale: 4.0 }
    }
}

/// One affine contraction map of an IFS: `x' = a·x + b·y + e`, `y' = c·x + d·y + f`,
/// selected with relative weight `p` in the chaos game.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IfsMap {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
    pub p: f64,
}

impl Default for IfsMap {
    fn default() -> Self {
        IfsMap { a: 0.5, b: 0.0, c: 0.0, d: 0.5, e: 0.0, f: 0.0, p: 1.0 }
    }
}

/// Iterated Function System configuration (chaos game). A `preset` name OR an explicit
/// list of affine `maps` (maps win when non-empty).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IfsSpec {
    /// Named preset: barnsley-fern · sierpinski · dragon · levy · tree · spiral.
    pub preset: String,
    /// Explicit affine maps (overrides `preset` when non-empty).
    pub maps: Vec<IfsMap>,
    /// Chaos-game point count (higher = denser / smoother).
    pub iterations: u64,
    /// Discard the first `warmup` points (settle onto the attractor).
    pub warmup: u32,
    /// Fraction of the canvas the attractor fills (0<margin≤1).
    pub margin: f64,
}

impl Default for IfsSpec {
    fn default() -> Self {
        IfsSpec {
            preset: "barnsley-fern".to_string(),
            maps: Vec::new(),
            iterations: 2_000_000,
            warmup: 20,
            margin: 0.9,
        }
    }
}

/// L-system configuration: Lindenmayer rewriting + turtle drawing. A `preset` name OR an
/// explicit `axiom` + `rules` (the explicit grammar wins when `axiom` is non-empty).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LsystemSpec {
    /// Named preset: koch · koch-snowflake · sierpinski · dragon · hilbert · gosper · plant · bush.
    pub preset: String,
    /// Starting string (overrides `preset` when non-empty).
    pub axiom: String,
    /// Rewrite rules, each `"X=..."` (LHS is a single symbol).
    pub rules: Vec<String>,
    /// Turn angle in degrees (`+` / `-`).
    pub angle: f64,
    /// Rewrite depth (each pass expands every symbol; grows fast).
    pub iterations: u32,
    /// Initial turtle heading in degrees (0 = east/right, 90 = up).
    pub start_angle: f64,
    /// Stroke width in pixels.
    pub line_width: u32,
    /// Fraction of the canvas the drawing fills (0<margin≤1).
    pub margin: f64,
}

impl Default for LsystemSpec {
    fn default() -> Self {
        LsystemSpec {
            preset: "koch-snowflake".to_string(),
            axiom: String::new(),
            rules: Vec::new(),
            angle: 60.0,
            iterations: 4,
            start_angle: 0.0,
            line_width: 1,
            margin: 0.9,
        }
    }
}

/// A complete, deterministic fractal render request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FractalSpec {
    pub kind: FractalKind,
    pub width: u32,
    pub height: u32,
    /// Viewport center in the complex plane, `[re, im]`.
    pub center: [f64; 2],
    /// Zoom factor: the shorter (vertical) axis spans `3.0 / zoom` complex units.
    pub zoom: f64,
    /// Iteration cap — the escape budget. Higher = more boundary detail (and slower).
    pub max_iter: u32,
    /// Escape radius: iteration stops once `|z| > escape_radius`. Large values (256)
    /// give smoother coloring than the mathematically-minimal 2.0.
    pub escape_radius: f64,
    /// The Julia constant `[re, im]` (used by `julia`, `phoenix`, `nova`).
    pub julia_c: [f64; 2],
    /// Exponent for the `z^power` step (2 = classic; other = multibrot / Newton degree).
    pub power: f64,
    /// Anti-aliasing: render at `supersample`× per axis then box-downsample (1 = off).
    pub supersample: u32,
    pub palette: PaletteSpec,
    pub coloring: Coloring,
    pub trap: TrapSpec,
    /// Stripe-average angular frequency (higher = finer bands).
    pub stripe_freq: f64,
    /// Distance-estimate contrast (larger = thinner filaments).
    pub de_scale: f64,
    /// Phoenix distortion constant `p` `[re, im]`.
    pub phoenix_p: [f64; 2],
    /// Nova relaxation factor `[re, im]`.
    pub nova_relax: [f64; 2],
    /// Buddhabrot: number of random sample points (higher = smoother density).
    pub buddha_samples: u64,
    /// Buddhabrot: only accumulate orbits that escape after at least this many iterations
    /// (suppresses the low-detail halo).
    pub buddha_min_iter: u32,
    /// IFS (chaos-game) configuration (used when `kind = ifs`).
    pub ifs: IfsSpec,
    /// L-system configuration (used when `kind = lsystem`).
    pub lsystem: LsystemSpec,
    /// Reserved / stochastic-family seed. Escape-time is deterministic regardless;
    /// buddhabrot uses it so its sampling is reproducible.
    pub seed: u64,
}

impl Default for FractalSpec {
    fn default() -> Self {
        FractalSpec {
            kind: FractalKind::Mandelbrot,
            width: 1024,
            height: 1024,
            center: [-0.5, 0.0],
            zoom: 1.0,
            max_iter: 500,
            escape_radius: 256.0,
            julia_c: [-0.8, 0.156],
            power: 2.0,
            supersample: 1,
            palette: PaletteSpec::default(),
            coloring: Coloring::Smooth,
            trap: TrapSpec::default(),
            stripe_freq: 6.0,
            de_scale: 1.0,
            phoenix_p: [-0.5, 0.0],
            nova_relax: [1.0, 0.0],
            buddha_samples: 5_000_000,
            buddha_min_iter: 20,
            ifs: IfsSpec::default(),
            lsystem: LsystemSpec::default(),
            seed: 0,
        }
    }
}

impl FractalSpec {
    /// Parse a spec from HJSON (or JSON — JSON is valid HJSON) text.
    pub fn from_hjson(text: &str) -> Result<Self> {
        deser_hjson::from_str(text).context("parsing FractalSpec HJSON")
    }

    /// Serialize to pretty JSON (valid HJSON) — used by `--fractal-dump-spec` and the
    /// embedded tEXt chunk.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serializing FractalSpec")
    }

    /// Load a spec from a `.hjson` / `.json` file.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading fractal spec {}", path.display()))?;
        Self::from_hjson(&text)
    }

    /// The Newton / Nova polynomial degree (from `power`, min 2).
    pub fn newton_degree(&self) -> f64 {
        self.power.round().max(2.0)
    }

    /// Validate the spec — cheap sanity checks so a bad spec fails before a long render.
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            anyhow::bail!("fractal dimensions must be non-zero (got {}x{})", self.width, self.height);
        }
        if !self.zoom.is_finite() || self.zoom <= 0.0 {
            anyhow::bail!("zoom must be a positive finite number (got {})", self.zoom);
        }
        if self.max_iter == 0 {
            anyhow::bail!("max_iter must be at least 1");
        }
        if !self.escape_radius.is_finite() || self.escape_radius <= 1.0 {
            anyhow::bail!("escape_radius must be a finite number > 1 (got {})", self.escape_radius);
        }
        if !self.power.is_finite() {
            anyhow::bail!("power must be finite (got {})", self.power);
        }
        if self.supersample == 0 || self.supersample > 8 {
            anyhow::bail!("supersample must be in 1..=8 (got {})", self.supersample);
        }
        // Guard against an accidental multi-billion-pixel supersampled allocation.
        let ss = self.supersample as u64;
        let px = self.width as u64 * self.height as u64 * ss * ss;
        if px > 500_000_000 {
            anyhow::bail!(
                "render too large: {}x{} at {}x supersample = {} samples (cap 500M)",
                self.width, self.height, self.supersample, px
            );
        }
        if self.kind == FractalKind::Ifs {
            if self.ifs.iterations == 0 || self.ifs.iterations > 200_000_000 {
                anyhow::bail!("ifs.iterations must be in 1..=200M (got {})", self.ifs.iterations);
            }
            if !(self.ifs.margin > 0.0 && self.ifs.margin <= 1.0) {
                anyhow::bail!("ifs.margin must be in (0,1] (got {})", self.ifs.margin);
            }
        }
        if self.kind == FractalKind::Lsystem {
            if self.lsystem.iterations > 20 {
                anyhow::bail!(
                    "lsystem.iterations must be ≤ 20 (grammar grows exponentially; got {})",
                    self.lsystem.iterations
                );
            }
            if !self.lsystem.angle.is_finite() {
                anyhow::bail!("lsystem.angle must be finite");
            }
            if !(self.lsystem.margin > 0.0 && self.lsystem.margin <= 1.0) {
                anyhow::bail!("lsystem.margin must be in (0,1] (got {})", self.lsystem.margin);
            }
        }
        Ok(())
    }
}

/// The PNG tEXt keyword under which the spec travels.
pub const SPEC_CHUNK_KEYWORD: &str = "fractalspec";

/// Read the embedded `fractalspec` tEXt chunk from a PNG (`--fractal-clone`).
/// Returns `Ok(None)` when the PNG has no such chunk (e.g. not a plakat fractal).
pub fn read_spec_chunk(path: &Path) -> Result<Option<FractalSpec>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let reader = decoder
        .read_info()
        .with_context(|| format!("decoding {}", path.display()))?;
    for chunk in &reader.info().uncompressed_latin1_text {
        if chunk.keyword == SPEC_CHUNK_KEYWORD {
            let spec = FractalSpec::from_hjson(&chunk.text)
                .with_context(|| format!("parsing embedded fractalspec in {}", path.display()))?;
            return Ok(Some(spec));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_json_round_trips() {
        let spec = FractalSpec {
            kind: FractalKind::Phoenix,
            center: [0.123, -0.456],
            zoom: 4.0,
            julia_c: [-0.8, 0.156],
            coloring: Coloring::OrbitTrap,
            supersample: 3,
            ..FractalSpec::default()
        };
        let json = spec.to_json().unwrap();
        let back = FractalSpec::from_hjson(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn hjson_defaults_fill_missing_fields() {
        // A minimal HJSON spec — every omitted field takes its default.
        let spec = FractalSpec::from_hjson("{ kind: tricorn, max_iter: 200 }").unwrap();
        assert_eq!(spec.kind, FractalKind::Tricorn);
        assert_eq!(spec.max_iter, 200);
        assert_eq!(spec.width, 1024); // default
        assert_eq!(spec.supersample, 1);
        assert_eq!(spec.coloring, Coloring::Smooth);
    }

    #[test]
    fn kind_parse_accepts_aliases() {
        assert_eq!(FractalKind::parse("Mandelbrot").unwrap(), FractalKind::Mandelbrot);
        assert_eq!(FractalKind::parse("mandelbar").unwrap(), FractalKind::Tricorn);
        assert_eq!(FractalKind::parse("buddha").unwrap(), FractalKind::Buddhabrot);
        assert!(FractalKind::parse("koch").is_err());
    }

    #[test]
    fn coloring_parses_kebab() {
        let spec = FractalSpec::from_hjson("{ coloring: orbit-trap }").unwrap();
        assert_eq!(spec.coloring, Coloring::OrbitTrap);
    }

    #[test]
    fn validate_rejects_degenerate_specs() {
        let mut spec = FractalSpec::default();
        assert!(spec.validate().is_ok());
        spec.width = 0;
        assert!(spec.validate().is_err());
        spec.width = 512;
        spec.zoom = 0.0;
        assert!(spec.validate().is_err());
        spec.zoom = 1.0;
        spec.supersample = 9;
        assert!(spec.validate().is_err());
    }
}
