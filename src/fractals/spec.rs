//! `FractalSpec` — the single authoritative, seed-stable, human-writable (HJSON)
//! description of a fractal render. Embedded as a `fractalspec` tEXt chunk in every
//! output PNG so `plakat fractals --fractal-clone PATH` can reconstruct the exact image.
//!
//! RFC FRACTALS-1, Phase 1. Escape-time only for now; later phases add `ifs` / `lsystem`
//! / `flame` / `attractor` / `raymarch` variants alongside `EscapeParams`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Which escape-time fractal to render. Phase 1 ships the three canonical families;
/// Phase 2 extends this list (tricorn, multibrot, newton, phoenix, …).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FractalKind {
    /// z ← z^power + c, with c the pixel and z₀ = 0.
    Mandelbrot,
    /// z ← z^power + c, with c a fixed constant (`julia_c`) and z₀ = the pixel.
    Julia,
    /// z ← (|Re z| + i|Im z|)^power + c — the Burning Ship.
    BurningShip,
}

impl FractalKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => "mandelbrot",
            FractalKind::Julia => "julia",
            FractalKind::BurningShip => "burning-ship",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "mandelbrot" | "mandel" | "m" => Ok(FractalKind::Mandelbrot),
            "julia" | "j" => Ok(FractalKind::Julia),
            "burning-ship" | "burningship" | "ship" | "bs" => Ok(FractalKind::BurningShip),
            other => anyhow::bail!(
                "unknown fractal kind {other:?} (want: mandelbrot | julia | burning-ship)"
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

/// Coloring algorithm applied to the escape result. Phase 1 = smooth (continuous)
/// iteration count; Phase 2 adds histogram / distance-estimate / orbit-trap / angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Coloring {
    /// Continuous (Bernstein-smoothed) escape count — no visible iteration bands.
    #[default]
    Smooth,
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
    /// The Julia constant `[re, im]` (only used when `kind = julia`).
    pub julia_c: [f64; 2],
    /// Exponent for the `z^power` step (2 = classic; other values = multibrot).
    pub power: f64,
    pub palette: PaletteSpec,
    pub coloring: Coloring,
    /// Reserved for stochastic families (flame/attractor). Escape-time is fully
    /// deterministic, but carrying the seed keeps every spec reproducible.
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
            palette: PaletteSpec::default(),
            coloring: Coloring::Smooth,
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
            kind: FractalKind::Julia,
            center: [0.123, -0.456],
            zoom: 4.0,
            julia_c: [-0.8, 0.156],
            ..FractalSpec::default()
        };
        let json = spec.to_json().unwrap();
        let back = FractalSpec::from_hjson(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn hjson_defaults_fill_missing_fields() {
        // A minimal HJSON spec — every omitted field takes its default.
        let spec = FractalSpec::from_hjson("{ kind: burning-ship, max_iter: 200 }").unwrap();
        assert_eq!(spec.kind, FractalKind::BurningShip);
        assert_eq!(spec.max_iter, 200);
        assert_eq!(spec.width, 1024); // default
        assert_eq!(spec.zoom, 1.0);
    }

    #[test]
    fn kind_parse_accepts_aliases() {
        assert_eq!(FractalKind::parse("Mandelbrot").unwrap(), FractalKind::Mandelbrot);
        assert_eq!(FractalKind::parse("j").unwrap(), FractalKind::Julia);
        assert_eq!(FractalKind::parse("burning_ship").unwrap(), FractalKind::BurningShip);
        assert!(FractalKind::parse("koch").is_err());
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
    }
}
