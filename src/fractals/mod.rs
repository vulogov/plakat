//! `plakat fractals` — non-AI + AI-assisted fractal generation (RFC FRACTALS-1).
//!
//! Two tracks, mirroring `plakat map`:
//!   * **Track A** (this module): a pure-CPU, deterministic, rayon-parallel render engine.
//!     Same [`FractalSpec`] → byte-identical pixels; no GPU, no model, fully offline.
//!   * **Track B** (later phase): an optional AI enhancement pass that feeds the Track-A
//!     render through ControlNet-conditioned img2img.
//!
//! Phase 1 ships the escape-time core: Mandelbrot / Julia / Burning Ship, smooth coloring,
//! Lab-space palettes, and PNG output with the spec embedded for `--fractal-clone`.

pub mod coloring;
pub mod image_io;
pub mod palette;
pub mod render;
pub mod spec;

use anyhow::Result;
use std::path::Path;

pub use palette::Palette;
pub use spec::{Coloring, FractalKind, FractalSpec, PaletteSpec};

/// A finished Track-A render held in memory (packed `RGB8`, row-major).
#[derive(Debug, Clone)]
pub struct RenderedFractal {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Render a spec to an in-memory RGB buffer (Track A). Validates first, so a bad spec
/// fails fast rather than after a long iteration loop.
pub fn render_spec(spec: &FractalSpec) -> Result<RenderedFractal> {
    spec.validate()?;
    let palette = Palette::from_spec(&spec.palette)?;
    let field = render::render_escape(spec);
    let pixels = coloring::colorize(spec, &field, &palette);
    Ok(RenderedFractal { width: spec.width, height: spec.height, pixels })
}

/// Render a spec straight to a PNG file, embedding the spec for `--fractal-clone`.
pub fn render_to_file(spec: &FractalSpec, path: &Path) -> Result<()> {
    let r = render_spec(spec)?;
    image_io::save_png_with_spec(&r.pixels, r.width, r.height, spec, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_spec_produces_full_buffer() {
        let spec = FractalSpec { width: 24, height: 16, ..FractalSpec::default() };
        let r = render_spec(&spec).unwrap();
        assert_eq!(r.width, 24);
        assert_eq!(r.height, 16);
        assert_eq!(r.pixels.len(), 24 * 16 * 3);
    }

    #[test]
    fn end_to_end_file_carries_spec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let spec = FractalSpec { width: 32, height: 32, zoom: 2.0, ..FractalSpec::default() };
        render_to_file(&spec, &path).unwrap();
        let back = spec::read_spec_chunk(&path).unwrap().unwrap();
        assert_eq!(back, spec);
    }

    #[test]
    fn invalid_spec_is_rejected_before_render() {
        let spec = FractalSpec { width: 0, ..FractalSpec::default() };
        assert!(render_spec(&spec).is_err());
    }
}
