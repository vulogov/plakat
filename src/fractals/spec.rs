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
    /// Fractal flame — IFS + non-linear variations + log-density color (Draves).
    Flame,
    /// Strange attractor — density plot of a chaotic map / ODE trajectory.
    Attractor,
    /// 3D distance-estimated fractal, sphere-traced (Mandelbulb, Mandelbox, Menger…).
    Raymarch,
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
            FractalKind::Flame => "flame",
            FractalKind::Attractor => "attractor",
            FractalKind::Raymarch => "raymarch",
        }
    }

    /// Buddhabrot renders via density accumulation, not a per-pixel escape field.
    pub fn is_buddhabrot(self) -> bool {
        self == FractalKind::Buddhabrot
    }

    /// The per-pixel complex-plane escape families (everything except buddhabrot / the
    /// line-drawing / density / raymarched families).
    pub fn is_escape_time(self) -> bool {
        !matches!(
            self,
            FractalKind::Buddhabrot
                | FractalKind::Ifs
                | FractalKind::Lsystem
                | FractalKind::Flame
                | FractalKind::Attractor
                | FractalKind::Raymarch
        )
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
            "flame" => Ok(FractalKind::Flame),
            "attractor" | "strange-attractor" => Ok(FractalKind::Attractor),
            "raymarch" | "3d" | "mandelbulb" => Ok(FractalKind::Raymarch),
            other => anyhow::bail!(
                "unknown fractal kind {other:?} (want: mandelbrot | julia | burning-ship | \
                 tricorn | multibrot | newton | nova | phoenix | magnet | sine | exp | buddhabrot \
                 | ifs | lsystem | flame | attractor | raymarch)"
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
    /// Orbit-trap by an image — sample a photo at the orbit's closest approach (the bridge
    /// to `plakat photos`: a Julia set textured by a photograph). Needs `trap_image`.
    Image,
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
            "image" | "image-trap" => Ok(Coloring::Image),
            other => anyhow::bail!(
                "unknown coloring {other:?} (want: smooth | histogram | distance | orbit-trap | \
                 angle | stripe | image)"
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

/// One weighted non-linear variation in a flame function (`name` → the variation, `weight`
/// → its blend coefficient). See `flame::VARIATIONS` for the supported names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarWeight {
    pub name: String,
    pub weight: f64,
}

/// One function ("transform") of a fractal flame: an affine pre-transform, a weighted sum
/// of non-linear variations, a color coordinate, and a selection weight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlameFunction {
    /// Affine pre-transform `[a, b, c, d, e, f]`: `x' = a·x + b·y + c`, `y' = d·x + e·y + f`.
    pub affine: [f64; 6],
    /// Weighted non-linear variations applied after the affine (summed).
    pub variations: Vec<VarWeight>,
    /// Color coordinate in `[0,1]` (looked up in the palette).
    pub color: f64,
    /// Relative selection probability in the chaos game.
    pub weight: f64,
}

impl Default for FlameFunction {
    fn default() -> Self {
        FlameFunction {
            affine: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            variations: vec![VarWeight { name: "linear".to_string(), weight: 1.0 }],
            color: 0.0,
            weight: 1.0,
        }
    }
}

/// Fractal flame configuration. A `preset` name OR explicit `functions` (functions win).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlameSpec {
    /// Named preset: sierpinski · spherical · swirl · spiral · flame.
    pub preset: String,
    /// Explicit flame functions (overrides `preset` when non-empty).
    pub functions: Vec<FlameFunction>,
    /// Chaos-game iteration count (higher = smoother density).
    pub iterations: u64,
    /// Discard the first `warmup` points.
    pub warmup: u32,
    /// Tone-mapping gamma (2.2 is standard).
    pub gamma: f64,
    /// Overall brightness multiplier.
    pub brightness: f64,
    /// Rotational symmetry count (1 = none; N replicates each plotted point N-fold).
    pub symmetry: u32,
    /// Fraction of the canvas the flame fills.
    pub margin: f64,
}

impl Default for FlameSpec {
    fn default() -> Self {
        FlameSpec {
            preset: "flame".to_string(),
            functions: Vec::new(),
            iterations: 4_000_000,
            warmup: 20,
            gamma: 2.2,
            brightness: 1.3,
            symmetry: 1,
            margin: 0.9,
        }
    }
}

/// Strange-attractor configuration: a named chaotic map / ODE, with optional parameter
/// override. The trajectory's visited points are accumulated into a density image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AttractorSpec {
    /// Named attractor: clifford · dejong · bedhead · duffing · ikeda · lorenz · rossler.
    pub preset: String,
    /// Parameter override (empty = the preset's classic parameters).
    pub params: Vec<f64>,
    /// Number of trajectory steps to accumulate.
    pub iterations: u64,
    /// Discard the first `warmup` steps (settle onto the attractor).
    pub warmup: u32,
    /// Fraction of the canvas the attractor fills.
    pub margin: f64,
}

impl Default for AttractorSpec {
    fn default() -> Self {
        AttractorSpec {
            preset: "clifford".to_string(),
            params: Vec::new(),
            iterations: 4_000_000,
            warmup: 100,
            margin: 0.9,
        }
    }
}

/// 3D distance-estimated (raymarched) fractal configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RaymarchSpec {
    /// Shape: mandelbulb · mandelbox · menger · sierpinski3d · quat-julia.
    pub shape: String,
    /// Mandelbulb exponent (8 is the classic).
    pub power: f64,
    /// Fractal iteration count (detail of the distance estimator).
    pub iterations: u32,
    /// Maximum sphere-tracing steps per ray.
    pub max_steps: u32,
    /// Far clip distance.
    pub max_dist: f64,
    /// Surface-hit threshold.
    pub epsilon: f64,
    /// Orbit camera yaw / pitch (degrees) and distance from the origin.
    pub camera_yaw: f64,
    pub camera_pitch: f64,
    pub camera_dist: f64,
    /// Vertical field of view (degrees).
    pub fov: f64,
    /// Light direction `[x, y, z]`.
    pub light: [f64; 3],
    /// Ambient-occlusion shading.
    pub ao: bool,
    /// Mandelbox scale factor.
    pub box_scale: f64,
    /// Quaternion-Julia constant `[a, b, c, d]`.
    pub quat_c: [f64; 4],
}

impl Default for RaymarchSpec {
    fn default() -> Self {
        RaymarchSpec {
            shape: "mandelbulb".to_string(),
            power: 8.0,
            iterations: 12,
            max_steps: 160,
            max_dist: 12.0,
            epsilon: 0.0008,
            camera_yaw: 40.0,
            camera_pitch: 22.0,
            camera_dist: 2.6,
            fov: 55.0,
            light: [0.6, 0.7, -0.5],
            ao: true,
            box_scale: 2.5,
            quat_c: [-0.2, 0.6, 0.2, 0.0],
        }
    }
}

/// Track-B (AI enhancement) configuration. When `enabled`, the deterministic Track-A
/// render feeds a ControlNet-conditioned img2img pass through the generation stack.
/// Empty string fields mean "auto": `prompt`/`negative` fall back to a per-kind default,
/// `control` to [`FractalKind`]'s default control type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiSpec {
    pub enabled: bool,
    /// Paint pipeline: `img2img` (the fractal is the init image *and* ControlNet — a scene
    /// **made of** the fractal, keeps its colors/layout) or `txt2img` (the fractal is
    /// ControlNet **only** — a scene **shaped by** the fractal, free composition: real sky,
    /// horizon, lighting from the prompt).
    pub mode: String,
    /// Model alias (sdxl / sd15 / …).
    pub model: String,
    /// Positive prompt ("" = per-kind auto).
    pub prompt: String,
    /// Negative prompt ("" = auto).
    pub negative: String,
    /// img2img strength in [0,1] (how far from the Track-A base).
    pub strength: f32,
    pub steps: u32,
    pub guidance: f64,
    /// ControlNet type ("" = per-kind auto: canny / lineart / softedge).
    pub control: String,
    /// ControlNet conditioning scale.
    pub control_strength: f32,
    /// LoRA specs (HF `org/name[:scale]`, `civitai:ID`, or a local path).
    pub loras: Vec<String>,
    pub lora_scale: f32,
}

impl Default for AiSpec {
    fn default() -> Self {
        AiSpec {
            enabled: false,
            mode: "img2img".to_string(),
            model: "sdxl".to_string(),
            prompt: String::new(),
            negative: String::new(),
            // Fractals are abstract (often near-black) img2img bases, so the paint pass
            // needs more freedom than map's already-painted base: a high strength lets the
            // prompt's scene actually emerge, while a moderate control keeps the fractal's
            // composition as guidance rather than a hard lock. Turn strength DOWN
            // (`--fractal-sd-strength 0.5`) for subtle "enhance the fractal" instead.
            strength: 0.78,
            steps: 28,
            guidance: 6.5,
            control: String::new(),
            control_strength: 0.55,
            loras: Vec::new(),
            lora_scale: 0.9,
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
    /// Path to the image sampled by `coloring = image` (orbit-trap-image / photo bridge).
    pub trap_image: String,
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
    /// Fractal-flame configuration (used when `kind = flame`).
    pub flame: FlameSpec,
    /// Strange-attractor configuration (used when `kind = attractor`).
    pub attractor: AttractorSpec,
    /// Raymarched-3D configuration (used when `kind = raymarch`).
    pub raymarch: RaymarchSpec,
    /// Track-B AI enhancement configuration.
    pub ai: AiSpec,
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
            trap_image: String::new(),
            de_scale: 1.0,
            phoenix_p: [-0.5, 0.0],
            nova_relax: [1.0, 0.0],
            buddha_samples: 5_000_000,
            buddha_min_iter: 20,
            ifs: IfsSpec::default(),
            lsystem: LsystemSpec::default(),
            flame: FlameSpec::default(),
            attractor: AttractorSpec::default(),
            raymarch: RaymarchSpec::default(),
            ai: AiSpec::default(),
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
        if self.kind == FractalKind::Flame {
            if self.flame.iterations == 0 || self.flame.iterations > 200_000_000 {
                anyhow::bail!("flame.iterations must be in 1..=200M (got {})", self.flame.iterations);
            }
            if !self.flame.gamma.is_finite() || self.flame.gamma <= 0.0 {
                anyhow::bail!("flame.gamma must be a positive finite number (got {})", self.flame.gamma);
            }
            if self.flame.symmetry == 0 || self.flame.symmetry > 24 {
                anyhow::bail!("flame.symmetry must be in 1..=24 (got {})", self.flame.symmetry);
            }
        }
        if self.kind == FractalKind::Attractor
            && (self.attractor.iterations == 0 || self.attractor.iterations > 500_000_000)
        {
            anyhow::bail!(
                "attractor.iterations must be in 1..=500M (got {})",
                self.attractor.iterations
            );
        }
        if self.kind == FractalKind::Raymarch {
            let r = &self.raymarch;
            if r.iterations == 0 || r.iterations > 200 {
                anyhow::bail!("raymarch.iterations must be in 1..=200 (got {})", r.iterations);
            }
            if r.max_steps == 0 || r.max_steps > 4000 {
                anyhow::bail!("raymarch.max_steps must be in 1..=4000 (got {})", r.max_steps);
            }
            if !r.epsilon.is_finite() || r.epsilon <= 0.0 {
                anyhow::bail!("raymarch.epsilon must be positive (got {})", r.epsilon);
            }
            if !r.camera_dist.is_finite() || r.camera_dist <= 0.0 {
                anyhow::bail!("raymarch.camera_dist must be positive (got {})", r.camera_dist);
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
