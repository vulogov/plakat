//! Escape-time rendering: viewport→complex-plane mapping and the per-pixel iteration.
//!
//! Deterministic and rayon-parallel: `render_escape` fills one `Escape` per pixel with
//! rows split across the thread pool (`par_chunks_mut`). Same spec → byte-identical
//! escape field (no RNG, no atomics, no accumulation order dependence).
//! RFC FRACTALS-1, Phase 1.

use num_complex::Complex;
use rayon::prelude::*;

use super::spec::{FractalKind, FractalSpec};

/// Maps pixel coordinates to points in the complex plane. Square pixels: the vertical
/// axis spans `3.0 / zoom` units about `center`, the horizontal axis scales by aspect.
#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    width: u32,
    height: u32,
    center: [f64; 2],
    /// Complex-plane units per pixel (isotropic).
    scale: f64,
}

impl Viewport {
    pub fn new(spec: &FractalSpec) -> Self {
        let scale = (3.0 / spec.zoom) / spec.height.max(1) as f64;
        Viewport {
            width: spec.width,
            height: spec.height,
            center: spec.center,
            scale,
        }
    }

    /// Complex coordinate of the *center* of pixel `(px, py)`. `+im` points up
    /// (screen `y` grows downward, so the imaginary axis is flipped).
    pub fn pixel_to_complex(&self, px: u32, py: u32) -> Complex<f64> {
        let re = self.center[0] + (px as f64 + 0.5 - self.width as f64 / 2.0) * self.scale;
        let im = self.center[1] - (py as f64 + 0.5 - self.height as f64 / 2.0) * self.scale;
        Complex::new(re, im)
    }
}

/// The result of iterating one pixel to escape (or to the iteration cap).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Escape {
    /// Never escaped within `max_iter` — an interior point.
    pub inside: bool,
    /// Integer iteration count at escape (`max_iter` for interior points).
    pub iters: u32,
    /// Continuous (smoothed) escape count — feeds banding-free coloring.
    pub smooth: f64,
}

/// Iterate a single pixel. Splits on `kind`/`power` so the common `power == 2` path
/// avoids the transcendental `powf`.
#[inline]
fn escape_at(spec: &FractalSpec, pixel: Complex<f64>) -> Escape {
    let (mut z, c) = match spec.kind {
        // z₀ = 0, c = the pixel.
        FractalKind::Mandelbrot | FractalKind::BurningShip => (Complex::new(0.0, 0.0), pixel),
        // z₀ = the pixel, c = the fixed Julia constant.
        FractalKind::Julia => (pixel, Complex::new(spec.julia_c[0], spec.julia_c[1])),
    };
    let burning = spec.kind == FractalKind::BurningShip;
    let quadratic = (spec.power - 2.0).abs() < f64::EPSILON;
    let escape2 = spec.escape_radius * spec.escape_radius;

    let mut n = 0u32;
    while n < spec.max_iter {
        if burning {
            z = Complex::new(z.re.abs(), z.im.abs());
        }
        z = if quadratic { z * z } else { z.powf(spec.power) } + c;
        n += 1;
        let mag2 = z.norm_sqr();
        if mag2 > escape2 {
            // Smooth (continuous) iteration count via the escape-potential estimate:
            //   ν = ln(ln|z|) / ln(power);   smooth = (n) - ν
            // Large escape radius makes ν small and the bands vanish.
            let log_zn = 0.5 * mag2.ln(); // = ln|z|
            let nu = log_zn.ln() / spec.power.ln();
            let smooth = (n as f64 - nu).max(0.0);
            return Escape { inside: false, iters: n, smooth };
        }
        if !mag2.is_finite() {
            // Overflow (large powers / far points) — treat as escaped this step.
            return Escape { inside: false, iters: n, smooth: n as f64 };
        }
    }
    Escape { inside: true, iters: spec.max_iter, smooth: spec.max_iter as f64 }
}

/// Render the full escape field, one `Escape` per pixel, row-parallel via rayon.
pub fn render_escape(spec: &FractalSpec) -> Vec<Escape> {
    let vp = Viewport::new(spec);
    let (w, h) = (spec.width as usize, spec.height as usize);
    let mut field = vec![Escape::default(); w * h];
    field
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out)| {
            let py = row as u32;
            for (px, cell) in out.iter_mut().enumerate() {
                *cell = escape_at(spec, vp.pixel_to_complex(px as u32, py));
            }
        });
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mandel_default() -> FractalSpec {
        FractalSpec { width: 64, height: 64, ..FractalSpec::default() }
    }

    #[test]
    fn viewport_centers_and_scales() {
        let spec = FractalSpec {
            width: 100,
            height: 100,
            center: [0.0, 0.0],
            zoom: 1.0,
            ..FractalSpec::default()
        };
        let vp = Viewport::new(&spec);
        // Pixel (50,50) is the ~center → within a pixel of the origin (each axis carries
        // a half-pixel offset of 0.5 * scale = 0.015 → norm ≈ 0.021).
        let c = vp.pixel_to_complex(50, 50);
        assert!(c.norm() < 0.03, "center pixel maps near origin, got {c}");
        // Vertical span is 3.0 units at zoom 1 → top row near +1.5 im.
        let top = vp.pixel_to_complex(50, 0);
        assert!((top.im - 1.485).abs() < 0.05, "top im ~1.5, got {}", top.im);
    }

    #[test]
    fn origin_is_inside_mandelbrot() {
        let spec = mandel_default();
        let e = escape_at(&spec, Complex::new(0.0, 0.0));
        assert!(e.inside);
        // A point well outside the set escapes fast.
        let out = escape_at(&spec, Complex::new(2.0, 2.0));
        assert!(!out.inside && out.iters < 10);
    }

    #[test]
    fn escape_field_is_deterministic() {
        // Byte-stability: two independent renders of the same spec are identical
        // (no accumulation-order or thread-scheduling dependence).
        let spec = mandel_default();
        let a = render_escape(&spec);
        let b = render_escape(&spec);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64 * 64);
        // The set has both interior and exterior pixels at this framing.
        assert!(a.iter().any(|e| e.inside));
        assert!(a.iter().any(|e| !e.inside));
    }

    #[test]
    fn julia_and_burning_ship_render() {
        for kind in [FractalKind::Julia, FractalKind::BurningShip] {
            let spec = FractalSpec { kind, width: 48, height: 48, ..FractalSpec::default() };
            let field = render_escape(&spec);
            assert_eq!(field.len(), 48 * 48);
            assert!(field.iter().any(|e| !e.inside));
        }
    }
}
