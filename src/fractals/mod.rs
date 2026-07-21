//! `plakat fractals` — non-AI + AI-assisted fractal generation (RFC FRACTALS-1).
//!
//! Two tracks, mirroring `plakat map`:
//!   * **Track A** (this module): a pure-CPU, deterministic, rayon-parallel render engine.
//!     Same [`FractalSpec`] → byte-identical pixels; no GPU, no model, fully offline.
//!   * **Track B** (later phase): an optional AI enhancement pass that feeds the Track-A
//!     render through ControlNet-conditioned img2img.
//!
//! Phase 1 shipped the escape-time core; Phase 2 adds the full family (tricorn, multibrot,
//! newton, nova, phoenix, magnet, sine, exp), five extra coloring modes (histogram /
//! distance-estimate / orbit-trap / angle / stripe), supersampling AA, and buddhabrot.

pub mod buddhabrot;
pub mod coloring;
pub mod image_io;
pub mod palette;
pub mod render;
pub mod spec;

use anyhow::Result;
use rayon::prelude::*;
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

/// A high-resolution copy of `spec` for supersampling: dimensions ×`ss`, `supersample`
/// reset to 1 (so nothing downstream re-supersamples). Same framing, more samples.
fn supersampled(spec: &FractalSpec) -> (FractalSpec, u32) {
    let ss = spec.supersample.clamp(1, 8);
    if ss == 1 {
        return (spec.clone(), 1);
    }
    let hi = FractalSpec {
        width: spec.width * ss,
        height: spec.height * ss,
        supersample: 1,
        ..spec.clone()
    };
    (hi, ss)
}

/// Box-downsample a packed `RGB8` buffer by `ss` per axis (`ss²` samples averaged).
fn downsample(hi: &[u8], hw: u32, hh: u32, ss: u32) -> Vec<u8> {
    let (ow, oh) = (hw / ss, hh / ss);
    let (hw, ss) = (hw as usize, ss as usize);
    let mut out = vec![0u8; ow as usize * oh as usize * 3];
    out.par_chunks_mut(ow as usize * 3)
        .enumerate()
        .for_each(|(oy, row)| {
            for ox in 0..ow as usize {
                let mut sum = [0u32; 3];
                for dy in 0..ss {
                    let sy = oy * ss + dy;
                    for dx in 0..ss {
                        let sx = ox * ss + dx;
                        let base = (sy * hw + sx) * 3;
                        sum[0] += hi[base] as u32;
                        sum[1] += hi[base + 1] as u32;
                        sum[2] += hi[base + 2] as u32;
                    }
                }
                let cnt = (ss * ss) as u32;
                let o = ox * 3;
                row[o] = (sum[0] / cnt) as u8;
                row[o + 1] = (sum[1] / cnt) as u8;
                row[o + 2] = (sum[2] / cnt) as u8;
            }
        });
    out
}

/// Render a spec to an in-memory RGB buffer (Track A). Validates first, so a bad spec
/// fails fast rather than after a long iteration loop.
pub fn render_spec(spec: &FractalSpec) -> Result<RenderedFractal> {
    spec.validate()?;
    let palette = Palette::from_spec(&spec.palette)?;
    let (hi, ss) = supersampled(spec);

    let pixels_hi = if hi.kind.is_buddhabrot() {
        let (hist, max) = buddhabrot::render_density(&hi);
        coloring::colorize_density(&hist, max, &palette)
    } else {
        let field = render::render_escape(&hi);
        coloring::colorize(&hi, &field, &palette)
    };

    let pixels = if ss > 1 {
        downsample(&pixels_hi, hi.width, hi.height, ss)
    } else {
        pixels_hi
    };
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
        assert_eq!((r.width, r.height), (24, 16));
        assert_eq!(r.pixels.len(), 24 * 16 * 3);
    }

    #[test]
    fn supersample_keeps_output_dims_and_changes_pixels() {
        let base = FractalSpec { width: 32, height: 32, ..FractalSpec::default() };
        let aa = FractalSpec { supersample: 3, ..base.clone() };
        let r1 = render_spec(&base).unwrap();
        let r3 = render_spec(&aa).unwrap();
        assert_eq!((r3.width, r3.height), (32, 32)); // output size unchanged
        assert_eq!(r3.pixels.len(), r1.pixels.len());
        assert_ne!(r1.pixels, r3.pixels); // AA smooths edges → different bytes
    }

    #[test]
    fn buddhabrot_renders_through_the_pipeline() {
        let spec = FractalSpec {
            kind: FractalKind::Buddhabrot, width: 32, height: 32, center: [-0.5, 0.0],
            zoom: 0.7, max_iter: 150, buddha_samples: 100_000, buddha_min_iter: 5,
            seed: 1, ..FractalSpec::default()
        };
        let r = render_spec(&spec).unwrap();
        assert_eq!(r.pixels.len(), 32 * 32 * 3);
        assert!(r.pixels.chunks(3).any(|p| p != &r.pixels[0..3]));
    }

    #[test]
    fn end_to_end_file_carries_spec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        let spec = FractalSpec {
            kind: FractalKind::Julia, width: 32, height: 32, zoom: 2.0,
            coloring: Coloring::Stripe, supersample: 2, ..FractalSpec::default()
        };
        render_to_file(&spec, &path).unwrap();
        assert_eq!(spec::read_spec_chunk(&path).unwrap().unwrap(), spec);
    }

    #[test]
    fn invalid_spec_is_rejected_before_render() {
        assert!(render_spec(&FractalSpec { width: 0, ..FractalSpec::default() }).is_err());
    }
}
