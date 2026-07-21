//! Composition — "random but regular" fractal canvases (RFC FRACTALS-1, Phase 8).
//!
//! Renders an R×C grid of related sub-fractals and tiles them into one image: a Julia
//! parameter sweep, a progressive zoom sequence, a palette contact-sheet, or a variation
//! (seed/parameter) sweep. Each cell is a full Track-A render, so all coloring modes and
//! families compose. Deterministic; pure CPU.

use anyhow::Result;
use std::f64::consts::PI;

use super::progress::ProgressFn;
use super::spec::{FractalKind, FractalSpec};
use super::RenderedFractal;

/// A grid composition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeMode {
    /// Julia sets with `c` swept around the classic circle `0.7885·e^{iθ}`.
    JuliaSweep,
    /// The base fractal at progressively deeper zoom (self-similarity contact sheet).
    ZoomGrid,
    /// The base fractal cycled through the built-in palettes.
    PaletteGrid,
    /// Per-cell variation: seed sweep (stochastic families) / parameter nudge (escape).
    VariationSweep,
}

impl ComposeMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "julia-sweep" | "julia" => ComposeMode::JuliaSweep,
            "zoom-grid" | "zoom" => ComposeMode::ZoomGrid,
            "palette-grid" | "palette" => ComposeMode::PaletteGrid,
            "variation-sweep" | "variation" | "sweep" => ComposeMode::VariationSweep,
            other => anyhow::bail!(
                "unknown compose mode {other:?} (want: julia-sweep | zoom-grid | palette-grid | \
                 variation-sweep)"
            ),
        })
    }
}

const PALETTES: &[&str] =
    &["fire", "ice", "electric", "neon", "pastel", "monochrome", "midnight", "earth"];

/// Build the sub-spec for cell `idx` of `n`, at `cw × ch` pixels.
fn cell_spec(base: &FractalSpec, mode: ComposeMode, idx: usize, n: usize, cw: u32, ch: u32) -> FractalSpec {
    let mut s = base.clone();
    s.width = cw;
    s.height = ch;
    s.supersample = 1;
    // Keep stochastic cells snappy.
    s.buddha_samples = s.buddha_samples.min(2_000_000);
    s.ifs.iterations = s.ifs.iterations.min(1_500_000);
    s.flame.iterations = s.flame.iterations.min(1_500_000);
    s.attractor.iterations = s.attractor.iterations.min(1_500_000);
    let f = idx as f64 / (n.max(1) as f64);
    match mode {
        ComposeMode::JuliaSweep => {
            s.kind = FractalKind::Julia;
            let theta = 2.0 * PI * f;
            s.julia_c = [0.7885 * theta.cos(), 0.7885 * theta.sin()];
            s.center = [0.0, 0.0];
            s.zoom = 1.15;
        }
        ComposeMode::ZoomGrid => {
            s.zoom = base.zoom * 2.5f64.powi(idx as i32);
            // More iterations as we zoom in, so detail survives.
            s.max_iter = (base.max_iter as f64 * (1.0 + idx as f64 * 0.4)) as u32;
        }
        ComposeMode::PaletteGrid => {
            s.palette.preset = PALETTES[idx % PALETTES.len()].to_string();
            s.palette.stops.clear();
        }
        ComposeMode::VariationSweep => {
            s.seed = base.seed.wrapping_add(idx as u64);
            match base.kind {
                FractalKind::Julia | FractalKind::Phoenix => {
                    let theta = 2.0 * PI * f;
                    s.julia_c = [
                        base.julia_c[0] + 0.25 * theta.cos(),
                        base.julia_c[1] + 0.25 * theta.sin(),
                    ];
                }
                FractalKind::Flame | FractalKind::Attractor | FractalKind::Buddhabrot
                | FractalKind::Ifs => { /* seed sweep only */ }
                _ => s.power = base.power + idx as f64 * 0.15,
            }
        }
    }
    s
}

/// Render an `rows × cols` grid composition into a single image.
pub fn compose(
    base: &FractalSpec,
    mode: ComposeMode,
    rows: u32,
    cols: u32,
    prog: ProgressFn,
) -> Result<RenderedFractal> {
    if rows == 0 || cols == 0 {
        anyhow::bail!("compose grid must be at least 1x1");
    }
    let cw = (base.width / cols).max(16);
    let ch = (base.height / rows).max(16);
    let (out_w, out_h) = (cw * cols, ch * rows);
    let n = (rows * cols) as usize;
    let mut canvas = vec![0u8; out_w as usize * out_h as usize * 3];

    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            let cell = cell_spec(base, mode, idx, n, cw, ch);
            let rendered = super::render_spec(&cell)?;
            // Blit the cell into the canvas.
            for y in 0..ch as usize {
                let src = &rendered.pixels[y * cw as usize * 3..(y + 1) * cw as usize * 3];
                let gx = c as usize * cw as usize;
                let gy = r as usize * ch as usize + y;
                let dst = (gy * out_w as usize + gx) * 3;
                canvas[dst..dst + cw as usize * 3].copy_from_slice(src);
            }
            prog((idx + 1) as u64, n as u64);
        }
    }
    Ok(RenderedFractal { width: out_w, height: out_h, pixels: canvas })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> FractalSpec {
        FractalSpec { width: 160, height: 160, ..FractalSpec::default() }
    }

    #[test]
    fn mode_parse() {
        assert_eq!(ComposeMode::parse("julia-sweep").unwrap(), ComposeMode::JuliaSweep);
        assert_eq!(ComposeMode::parse("zoom").unwrap(), ComposeMode::ZoomGrid);
        assert!(ComposeMode::parse("nope").is_err());
    }

    #[test]
    fn julia_sweep_grid_renders_and_tiles() {
        let r = compose(&base(), ComposeMode::JuliaSweep, 2, 2, &|_, _| {}).unwrap();
        // 160/2 = 80 per cell → 160x160 canvas.
        assert_eq!((r.width, r.height), (160, 160));
        assert_eq!(r.pixels.len(), 160 * 160 * 3);
        assert!(r.pixels.chunks(3).any(|p| p != &r.pixels[0..3]));
    }

    #[test]
    fn cells_differ_across_the_sweep() {
        // Two different Julia c values must produce different cells.
        let a = cell_spec(&base(), ComposeMode::JuliaSweep, 0, 4, 80, 80);
        let b = cell_spec(&base(), ComposeMode::JuliaSweep, 2, 4, 80, 80);
        assert_ne!(a.julia_c, b.julia_c);
    }

    #[test]
    fn zoom_grid_deepens() {
        let a = cell_spec(&base(), ComposeMode::ZoomGrid, 0, 4, 80, 80);
        let b = cell_spec(&base(), ComposeMode::ZoomGrid, 3, 4, 80, 80);
        assert!(b.zoom > a.zoom);
    }

    #[test]
    fn palette_grid_varies_palette() {
        let a = cell_spec(&base(), ComposeMode::PaletteGrid, 0, 8, 80, 80);
        let b = cell_spec(&base(), ComposeMode::PaletteGrid, 1, 8, 80, 80);
        assert_ne!(a.palette.preset, b.palette.preset);
    }
}
