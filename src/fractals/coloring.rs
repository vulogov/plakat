//! Turns an escape field into an RGB pixel buffer via a [`Palette`].
//!
//! Phase 1 implements the `Smooth` coloring: the continuous escape count normalized to
//! `[0,1]` and looked up in the Lab-space gradient; interior points take the palette's
//! solid interior color. RFC FRACTALS-1, Phase 1.

use rayon::prelude::*;

use super::palette::Palette;
use super::render::Escape;
use super::spec::{Coloring, FractalSpec};

/// Map the escape field to a packed `RGB8` buffer (`width*height*3` bytes), row-major.
pub fn colorize(spec: &FractalSpec, field: &[Escape], palette: &Palette) -> Vec<u8> {
    let n = (spec.width as usize) * (spec.height as usize);
    debug_assert_eq!(field.len(), n);
    let interior = palette.interior();
    let inv_max = 1.0 / spec.max_iter as f64;

    let mut buf = vec![0u8; n * 3];
    buf.par_chunks_mut(3)
        .zip(field.par_iter())
        .for_each(|(px, e)| {
            let rgb = if e.inside {
                interior
            } else {
                match spec.coloring {
                    Coloring::Smooth => palette.sample(e.smooth * inv_max),
                }
            };
            px.copy_from_slice(&rgb);
        });
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::render::render_escape;
    use crate::fractals::spec::PaletteSpec;

    #[test]
    fn interior_takes_interior_color() {
        let spec = FractalSpec {
            width: 32,
            height: 32,
            palette: PaletteSpec { interior: "#123456".into(), ..PaletteSpec::default() },
            ..FractalSpec::default()
        };
        let field = render_escape(&spec);
        let pal = Palette::from_spec(&spec.palette).unwrap();
        let buf = colorize(&spec, &field, &pal);
        assert_eq!(buf.len(), 32 * 32 * 3);
        // Some interior pixel must carry the exact interior color.
        let found = buf.chunks(3).zip(field.iter())
            .any(|(px, e)| e.inside && px == [0x12, 0x34, 0x56]);
        assert!(found, "an interior pixel should use the interior color");
    }

    #[test]
    fn colorize_is_deterministic() {
        let spec = FractalSpec { width: 40, height: 40, ..FractalSpec::default() };
        let field = render_escape(&spec);
        let pal = Palette::from_spec(&spec.palette).unwrap();
        assert_eq!(colorize(&spec, &field, &pal), colorize(&spec, &field, &pal));
    }
}
