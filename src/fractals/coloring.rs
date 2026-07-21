//! Turns an escape field (or a buddhabrot density) into an RGB pixel buffer via a
//! [`Palette`]. Six coloring modes (RFC FRACTALS-1, Phase 2):
//!
//! * `Smooth` — continuous escape count.
//! * `Histogram` — iteration-count histogram equalization (even color spread).
//! * `Distance` — boundary distance estimate (thin filaments).
//! * `OrbitTrap` — closest orbit approach to a shape.
//! * `Angle` — final-iterate argument (Newton basins).
//! * `Stripe` — angular stripe-average texturing.

use rayon::prelude::*;
use std::f64::consts::PI;

use super::palette::Palette;
use super::render::Escape;
use super::spec::{Coloring, FractalSpec};

/// Complex-plane units per pixel for this spec (matches `Viewport::scale`).
fn pixel_scale(spec: &FractalSpec) -> f64 {
    (3.0 / spec.zoom) / spec.height.max(1) as f64
}

/// Build the histogram-equalization lookup: for each integer iteration count, the
/// cumulative fraction of *exterior* pixels with ≤ that count. Index by `iters`.
fn histogram_cdf(spec: &FractalSpec, field: &[Escape]) -> Vec<f64> {
    let m = spec.max_iter as usize;
    let mut counts = vec![0u64; m + 2];
    let mut total = 0u64;
    for e in field {
        if !e.inside {
            counts[(e.iters as usize).min(m + 1)] += 1;
            total += 1;
        }
    }
    let mut cdf = vec![0.0f64; m + 2];
    if total == 0 {
        return cdf;
    }
    let mut acc = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        acc += c;
        cdf[i] = acc as f64 / total as f64;
    }
    cdf
}

/// Map a single escape result to a normalized gradient position `t ∈ [0,1]`.
#[inline]
fn escape_t(spec: &FractalSpec, e: &Escape, scale: f64, cdf: &[f64], inv_max: f64) -> f64 {
    match spec.coloring {
        Coloring::Smooth => e.smooth * inv_max,
        Coloring::Histogram => cdf.get(e.iters as usize).copied().unwrap_or(0.0),
        Coloring::Distance => {
            if e.distance < 0.0 {
                e.smooth * inv_max // family without a valid estimate → fall back
            } else {
                (e.distance / scale * spec.de_scale).tanh()
            }
        }
        Coloring::OrbitTrap => {
            if e.trap.is_finite() {
                (e.trap * spec.trap.scale).tanh()
            } else {
                0.0
            }
        }
        Coloring::Angle => (e.final_z.arg() + PI) / (2.0 * PI),
        Coloring::Stripe => e.stripe,
        // Image trap falls back to the orbit-trap gradient when no image is supplied.
        Coloring::Image => {
            if e.trap.is_finite() { (e.trap * spec.trap.scale).tanh() } else { 0.0 }
        }
    }
}

/// Sample an image at normalized `(u, v) ∈ [0,1]²` (nearest neighbor, clamped).
fn sample_image(img: &image::RgbImage, u: f64, v: f64) -> [u8; 3] {
    let (w, h) = (img.width(), img.height());
    let x = ((u.clamp(0.0, 1.0)) * (w.saturating_sub(1)) as f64).round() as u32;
    let y = ((v.clamp(0.0, 1.0)) * (h.saturating_sub(1)) as f64).round() as u32;
    img.get_pixel(x.min(w - 1), y.min(h - 1)).0
}

/// Map the escape field to a packed `RGB8` buffer (`width*height*3` bytes), row-major.
/// `trap_img` (when the coloring is `Image`) is the photo sampled at each orbit's closest
/// approach — the `plakat photos` bridge.
pub fn colorize(
    spec: &FractalSpec,
    field: &[Escape],
    palette: &Palette,
    trap_img: Option<&image::RgbImage>,
) -> Vec<u8> {
    let n = (spec.width as usize) * (spec.height as usize);
    debug_assert_eq!(field.len(), n);
    let interior = palette.interior();
    let inv_max = 1.0 / spec.max_iter as f64;
    let scale = pixel_scale(spec);
    // Histogram equalization needs a whole-frame pre-pass; other modes leave it empty.
    let cdf = if spec.coloring == Coloring::Histogram {
        histogram_cdf(spec, field)
    } else {
        Vec::new()
    };
    let img = if spec.coloring == Coloring::Image { trap_img } else { None };
    let (tp, ts) = (spec.trap.point, spec.trap.scale);

    let mut buf = vec![0u8; n * 3];
    buf.par_chunks_mut(3).zip(field.par_iter()).for_each(|(px, e)| {
        let rgb = if e.inside {
            interior
        } else if let Some(im) = img {
            // Map the closest-approach orbit point into the photo's UV space.
            let u = 0.5 + 0.5 * ((e.trap_z.re - tp[0]) * ts).tanh();
            let v = 0.5 - 0.5 * ((e.trap_z.im - tp[1]) * ts).tanh();
            sample_image(im, u, v)
        } else {
            palette.sample(escape_t(spec, e, scale, &cdf, inv_max))
        };
        px.copy_from_slice(&rgb);
    });
    buf
}

/// Map a buddhabrot density histogram to RGB via log scaling (reveals faint structure).
pub fn colorize_density(hist: &[u32], max: u32, palette: &Palette) -> Vec<u8> {
    let inv_log = if max > 0 { 1.0 / ((max as f64) + 1.0).ln() } else { 0.0 };
    let mut buf = vec![0u8; hist.len() * 3];
    buf.par_chunks_mut(3).zip(hist.par_iter()).for_each(|(px, &v)| {
        let t = if v == 0 { 0.0 } else { ((v as f64) + 1.0).ln() * inv_log };
        px.copy_from_slice(&palette.sample(t));
    });
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::render::render_escape;
    use crate::fractals::spec::{FractalKind, PaletteSpec};

    fn spec_with(coloring: Coloring) -> FractalSpec {
        FractalSpec { width: 40, height: 40, coloring, ..FractalSpec::default() }
    }

    #[test]
    fn interior_takes_interior_color() {
        let spec = FractalSpec {
            width: 32,
            height: 32,
            palette: PaletteSpec { interior: "#123456".into(), ..PaletteSpec::default() },
            ..FractalSpec::default()
        };
        let field = render_escape(&spec, &|_, _| {});
        let pal = Palette::from_spec(&spec.palette).unwrap();
        let buf = colorize(&spec, &field, &pal, None);
        assert_eq!(buf.len(), 32 * 32 * 3);
        assert!(buf.chunks(3).zip(field.iter())
            .any(|(px, e)| e.inside && px == [0x12, 0x34, 0x56]));
    }

    #[test]
    fn every_coloring_mode_produces_a_buffer() {
        for c in [
            Coloring::Smooth, Coloring::Histogram, Coloring::Distance,
            Coloring::OrbitTrap, Coloring::Angle, Coloring::Stripe,
        ] {
            let spec = spec_with(c);
            let field = render_escape(&spec, &|_, _| {});
            let pal = Palette::from_spec(&spec.palette).unwrap();
            let buf = colorize(&spec, &field, &pal, None);
            assert_eq!(buf.len(), 40 * 40 * 3, "{c:?}");
            // Not a single flat color (some variation exists).
            assert!(buf.chunks(3).any(|p| p != &buf[0..3]), "{c:?} was flat");
        }
    }

    #[test]
    fn histogram_and_smooth_differ() {
        let field = render_escape(&spec_with(Coloring::Smooth), &|_, _| {});
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        let smooth = colorize(&spec_with(Coloring::Smooth), &field, &pal, None);
        // Same field, histogram equalization → a different mapping.
        let hist_spec = spec_with(Coloring::Histogram);
        let hist_field = render_escape(&hist_spec, &|_, _| {});
        let hist = colorize(&hist_spec, &hist_field, &pal, None);
        assert_ne!(smooth, hist);
    }

    #[test]
    fn angle_mode_colors_newton_basins() {
        let spec = FractalSpec {
            kind: FractalKind::Newton, width: 40, height: 40, power: 3.0,
            center: [0.0, 0.0], zoom: 0.5, coloring: Coloring::Angle,
            ..FractalSpec::default()
        };
        let field = render_escape(&spec, &|_, _| {});
        let pal = Palette::from_spec(&spec.palette).unwrap();
        let buf = colorize(&spec, &field, &pal, None);
        assert!(buf.chunks(3).any(|p| p != &buf[0..3]));
    }

    #[test]
    fn density_colorize_maps_zero_to_start() {
        let pal = Palette::from_spec(&PaletteSpec {
            stops: vec!["#000000".into(), "#ffffff".into()], ..PaletteSpec::default()
        }).unwrap();
        let hist = vec![0u32, 10, 100];
        let buf = colorize_density(&hist, 100, &pal);
        assert_eq!(&buf[0..3], &[0, 0, 0]); // empty cell → gradient start
        assert!(buf[6] > buf[3]); // denser cell → brighter
    }
}
