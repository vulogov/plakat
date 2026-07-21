//! Escape-time rendering: viewport↔complex-plane mapping and the per-pixel iteration.
//!
//! Deterministic and rayon-parallel: `render_escape` fills one [`Escape`] per pixel with
//! rows split across the thread pool (`par_chunks_mut`). Same spec → byte-identical
//! escape field. The loop accumulates only the extra channels the chosen coloring needs
//! (derivative for distance-estimate, closest-approach for orbit-trap, angular history
//! for stripe-average) so the default `Smooth` path stays cheap. RFC FRACTALS-1, Phase 2.

use num_complex::Complex;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use super::progress::ProgressFn;
use super::spec::{Coloring, FractalKind, FractalSpec, TrapShape, TrapSpec};

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
        Viewport { width: spec.width, height: spec.height, center: spec.center, scale }
    }

    /// Complex-plane units per pixel.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Complex coordinate of the *center* of pixel `(px, py)`. `+im` points up.
    pub fn pixel_to_complex(&self, px: u32, py: u32) -> Complex<f64> {
        let re = self.center[0] + (px as f64 + 0.5 - self.width as f64 / 2.0) * self.scale;
        let im = self.center[1] - (py as f64 + 0.5 - self.height as f64 / 2.0) * self.scale;
        Complex::new(re, im)
    }

    /// Inverse map: pixel containing complex point `z`, or `None` if outside the frame.
    /// (Used by the buddhabrot orbit splatter.)
    pub fn complex_to_pixel(&self, z: Complex<f64>) -> Option<(u32, u32)> {
        let fx = (z.re - self.center[0]) / self.scale + self.width as f64 / 2.0 - 0.5;
        let fy = self.height as f64 / 2.0 - (z.im - self.center[1]) / self.scale - 0.5;
        let px = fx.round();
        let py = fy.round();
        if px >= 0.0 && px < self.width as f64 && py >= 0.0 && py < self.height as f64 {
            Some((px as u32, py as u32))
        } else {
            None
        }
    }
}

/// The result of iterating one pixel to escape (or to the iteration cap).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Escape {
    /// Never escaped within `max_iter` — an interior point.
    pub inside: bool,
    /// Integer iteration count at escape (`max_iter` for interior points).
    pub iters: u32,
    /// Continuous (smoothed) escape count — feeds banding-free coloring.
    pub smooth: f64,
    /// Final iterate (angle coloring; buddhabrot unused).
    pub final_z: Complex<f64>,
    /// Closest approach of the orbit to the trap shape (orbit-trap coloring).
    pub trap: f64,
    /// Boundary distance estimate in complex units (distance-estimate coloring); `-1` = n/a.
    pub distance: f64,
    /// Mean angular-stripe value over the orbit, in `[0,1]` (stripe coloring).
    pub stripe: f64,
    /// The orbit point at the closest trap approach (image-trap coloring).
    pub trap_z: Complex<f64>,
}

impl Default for Escape {
    fn default() -> Self {
        Escape {
            inside: true,
            iters: 0,
            smooth: 0.0,
            final_z: Complex::new(0.0, 0.0),
            trap: f64::INFINITY,
            distance: -1.0,
            stripe: 0.0,
            trap_z: Complex::new(0.0, 0.0),
        }
    }
}

/// Which extra channels the iteration must accumulate for the chosen coloring.
#[derive(Debug, Clone, Copy)]
struct Feats {
    deriv: bool,
    trap: bool,
    stripe: bool,
}

impl Feats {
    fn for_coloring(c: Coloring) -> Self {
        Feats {
            deriv: matches!(c, Coloring::Distance),
            trap: matches!(c, Coloring::OrbitTrap | Coloring::Image),
            stripe: matches!(c, Coloring::Stripe),
        }
    }
}

#[inline]
fn trap_distance(z: Complex<f64>, t: &TrapSpec) -> f64 {
    let p = Complex::new(t.point[0], t.point[1]);
    match t.shape {
        TrapShape::Point => (z - p).norm(),
        TrapShape::Cross => (z.re - p.re).abs().min((z.im - p.im).abs()),
        TrapShape::Circle => ((z - p).norm() - t.radius).abs(),
    }
}

/// The `z^power` step's exponent for smooth coloring, or `NaN` if the family isn't
/// polynomial (transcendental / convergence families fall back to integer iteration).
#[inline]
fn smoothing_exp(kind: FractalKind, power: f64) -> f64 {
    match kind {
        FractalKind::Mandelbrot | FractalKind::Julia | FractalKind::Multibrot
        | FractalKind::Tricorn => power,
        FractalKind::BurningShip | FractalKind::Phoenix | FractalKind::Magnet => 2.0,
        _ => f64::NAN,
    }
}

/// Iterate a single escape-family pixel.
fn escape_at(spec: &FractalSpec, pixel: Complex<f64>, feats: Feats) -> Escape {
    match spec.kind {
        FractalKind::Newton | FractalKind::Nova => return newton_at(spec, pixel),
        _ => {}
    }

    let julia_c = Complex::new(spec.julia_c[0], spec.julia_c[1]);
    let phoenix_p = Complex::new(spec.phoenix_p[0], spec.phoenix_p[1]);
    // z₀ and the parameter c per family. Julia / Phoenix and the transcendental Sine /
    // Exp maps are dynamical-plane: z₀ = pixel, c = the fixed `julia_c` constant (starting
    // Sine/Exp at 0 would be a fixed point — c·sin(0) = 0 — and render nothing).
    let (mut z, c) = match spec.kind {
        FractalKind::Julia | FractalKind::Phoenix | FractalKind::Sine | FractalKind::Exp => {
            (pixel, julia_c)
        }
        _ => (Complex::new(0.0, 0.0), pixel),
    };
    let power = spec.power;
    let quadratic = (power - 2.0).abs() < f64::EPSILON;
    let escape2 = spec.escape_radius * spec.escape_radius;
    let holomorphic = matches!(
        spec.kind,
        FractalKind::Mandelbrot | FractalKind::Julia | FractalKind::Multibrot
    );

    let mut dz = Complex::new(1.0, 0.0);
    let mut z_prev = Complex::new(0.0, 0.0);
    let mut trap = f64::INFINITY;
    let mut trap_z = Complex::new(0.0, 0.0);
    let mut stripe_sum = 0.0f64;
    let mut stripe_prev = 0.0f64;
    let mut stripe_count = 0u32;

    let mut n = 0u32;
    while n < spec.max_iter {
        // Derivative recurrence (before z updates), only for the holomorphic families
        // where distance estimation is well-defined.
        if feats.deriv && holomorphic {
            let zpm1 = if quadratic { z } else { z.powf(power - 1.0) };
            let add = if spec.kind == FractalKind::Julia { 0.0 } else { 1.0 };
            dz = Complex::new(power, 0.0) * zpm1 * dz + add;
        }

        // The family step.
        z = match spec.kind {
            FractalKind::Mandelbrot | FractalKind::Julia | FractalKind::Multibrot => {
                (if quadratic { z * z } else { z.powf(power) }) + c
            }
            FractalKind::BurningShip => {
                let a = Complex::new(z.re.abs(), z.im.abs());
                (if quadratic { a * a } else { a.powf(power) }) + c
            }
            FractalKind::Tricorn => {
                let cz = z.conj();
                (if quadratic { cz * cz } else { cz.powf(power) }) + c
            }
            FractalKind::Phoenix => {
                let zn = z * z + c + phoenix_p * z_prev;
                z_prev = z;
                zn
            }
            FractalKind::Magnet => {
                let num = z * z + c - Complex::new(1.0, 0.0);
                let den = Complex::new(2.0, 0.0) * z + c - Complex::new(2.0, 0.0);
                let r = num / den;
                r * r
            }
            FractalKind::Sine => c * z.sin(),
            FractalKind::Exp => c * z.exp(),
            FractalKind::Newton | FractalKind::Nova | FractalKind::Buddhabrot
            | FractalKind::Ifs | FractalKind::Lsystem | FractalKind::Flame
            | FractalKind::Attractor | FractalKind::Raymarch => unreachable!(),
        };
        n += 1;

        if feats.trap {
            let d = trap_distance(z, &spec.trap);
            if d < trap {
                trap = d;
                trap_z = z;
            }
        }
        if feats.stripe {
            let s = 0.5 + 0.5 * (spec.stripe_freq * z.arg()).sin();
            stripe_prev = stripe_sum / stripe_count.max(1) as f64;
            stripe_sum += s;
            stripe_count += 1;
        }

        // Magnet type I converges to the fixed point z = 1 in the interior.
        if spec.kind == FractalKind::Magnet && (z - Complex::new(1.0, 0.0)).norm_sqr() < 1e-12 {
            break;
        }

        let mag2 = z.norm_sqr();
        if mag2 > escape2 || !mag2.is_finite() {
            let sexp = smoothing_exp(spec.kind, power);
            let smooth = if mag2.is_finite() && sexp > 1.0 {
                let log_zn = 0.5 * mag2.ln();
                (n as f64 - log_zn.ln() / sexp.ln()).max(0.0)
            } else {
                n as f64
            };
            // Distance estimate: d = |z|·ln|z| / |dz|  (exterior, holomorphic only).
            let distance = if feats.deriv && holomorphic && mag2.is_finite() {
                let mag = mag2.sqrt();
                let dzn = dz.norm();
                if dzn > 0.0 { mag * mag.ln() / dzn } else { -1.0 }
            } else {
                -1.0
            };
            // Stripe average, smoothed between the last two partial means by the escape frac.
            let stripe = if feats.stripe && stripe_count > 0 {
                let avg = stripe_sum / stripe_count as f64;
                let frac = (smooth - smooth.floor()).clamp(0.0, 1.0);
                (frac * avg + (1.0 - frac) * stripe_prev).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return Escape {
                inside: false, iters: n, smooth, final_z: z, trap, distance, stripe, trap_z,
            };
        }
    }
    Escape {
        inside: true,
        iters: spec.max_iter,
        smooth: spec.max_iter as f64,
        final_z: z,
        trap,
        distance: -1.0,
        stripe: 0.0,
        trap_z,
    }
}

/// Newton / Nova: iterate to convergence on z^degree − 1; color by the step count and
/// the final root (its argument), so the Angle coloring paints the basins.
fn newton_at(spec: &FractalSpec, pixel: Complex<f64>) -> Escape {
    let deg = spec.newton_degree();
    let relax = if spec.kind == FractalKind::Nova {
        Complex::new(spec.nova_relax[0], spec.nova_relax[1])
    } else {
        Complex::new(1.0, 0.0)
    };
    let c = if spec.kind == FractalKind::Nova {
        Complex::new(spec.julia_c[0] + pixel.re, spec.julia_c[1] + pixel.im)
    } else {
        Complex::new(0.0, 0.0)
    };
    let mut z = if spec.kind == FractalKind::Nova { Complex::new(1.0, 0.0) } else { pixel };
    let eps = 1e-6;

    let mut n = 0u32;
    while n < spec.max_iter {
        let zpm1 = z.powf(deg - 1.0); // z^{deg-1}
        let f = z * zpm1 - Complex::new(1.0, 0.0); // z^deg - 1
        let fp = Complex::new(deg, 0.0) * zpm1; // deg·z^{deg-1}
        if fp.norm_sqr() < 1e-30 {
            break; // derivative vanished — treat as non-convergent
        }
        let delta = relax * (f / fp);
        z = z - delta + c;
        n += 1;
        if delta.norm() < eps {
            // Converged: continuous count via the log-ratio to the threshold.
            let smooth = n as f64 + (eps.ln() - delta.norm().max(1e-300).ln())
                / (eps.ln().abs().max(1.0));
            return Escape {
                inside: false,
                iters: n,
                smooth: smooth.max(0.0),
                final_z: z,
                trap: f64::INFINITY,
                distance: -1.0,
                stripe: 0.0,
                trap_z: Complex::new(0.0, 0.0),
            };
        }
        if !z.norm_sqr().is_finite() {
            break;
        }
    }
    Escape { inside: true, iters: spec.max_iter, smooth: spec.max_iter as f64, ..Escape::default() }
}

/// Render the full escape field, one [`Escape`] per pixel, row-parallel via rayon.
/// `prog(done_rows, total_rows)` is called as rows complete (from worker threads).
pub fn render_escape(spec: &FractalSpec, prog: ProgressFn) -> Vec<Escape> {
    let vp = Viewport::new(spec);
    let feats = Feats::for_coloring(spec.coloring);
    let (w, h) = (spec.width as usize, spec.height as usize);
    let done = AtomicU64::new(0);
    let mut field = vec![Escape::default(); w * h];
    field.par_chunks_mut(w).enumerate().for_each(|(row, out)| {
        let py = row as u32;
        for (px, cell) in out.iter_mut().enumerate() {
            *cell = escape_at(spec, vp.pixel_to_complex(px as u32, py), feats);
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        prog(d, h as u64);
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
    fn viewport_round_trips_through_pixel() {
        let spec = FractalSpec { width: 200, height: 150, zoom: 3.0, ..FractalSpec::default() };
        let vp = Viewport::new(&spec);
        for &(px, py) in &[(0u32, 0u32), (100, 75), (199, 149)] {
            let z = vp.pixel_to_complex(px, py);
            assert_eq!(vp.complex_to_pixel(z), Some((px, py)));
        }
        // A point far outside the frame maps to None.
        assert_eq!(vp.complex_to_pixel(Complex::new(100.0, 100.0)), None);
    }

    #[test]
    fn origin_is_inside_mandelbrot() {
        let spec = mandel_default();
        let f = Feats::for_coloring(spec.coloring);
        assert!(escape_at(&spec, Complex::new(0.0, 0.0), f).inside);
        let out = escape_at(&spec, Complex::new(2.0, 2.0), f);
        assert!(!out.inside && out.iters < 10);
    }

    #[test]
    fn escape_field_is_deterministic() {
        let spec = mandel_default();
        let a = render_escape(&spec, &|_, _| {});
        let b = render_escape(&spec, &|_, _| {});
        assert_eq!(a, b);
        assert_eq!(a.len(), 64 * 64);
        assert!(a.iter().any(|e| e.inside));
        assert!(a.iter().any(|e| !e.inside));
    }

    #[test]
    fn all_escape_families_render_something() {
        for kind in [
            FractalKind::Mandelbrot, FractalKind::Julia, FractalKind::BurningShip,
            FractalKind::Tricorn, FractalKind::Multibrot, FractalKind::Newton,
            FractalKind::Nova, FractalKind::Phoenix, FractalKind::Magnet,
            FractalKind::Sine, FractalKind::Exp,
        ] {
            let spec = FractalSpec {
                kind, width: 48, height: 48, power: 3.0, max_iter: 100,
                center: [0.0, 0.0], zoom: 0.6, ..FractalSpec::default()
            };
            let field = render_escape(&spec, &|_, _| {});
            assert_eq!(field.len(), 48 * 48, "{kind:?}");
            // Every family produces a mix (some escaped / converged pixel exists).
            assert!(field.iter().any(|e| !e.inside), "{kind:?} produced no exterior");
        }
    }

    #[test]
    fn distance_estimate_populated_for_mandelbrot() {
        let spec = FractalSpec {
            width: 32, height: 32, coloring: Coloring::Distance, ..FractalSpec::default()
        };
        let field = render_escape(&spec, &|_, _| {});
        assert!(field.iter().any(|e| !e.inside && e.distance >= 0.0));
    }

    #[test]
    fn orbit_trap_and_stripe_populate() {
        let trap = FractalSpec {
            width: 32, height: 32, coloring: Coloring::OrbitTrap, ..FractalSpec::default()
        };
        assert!(render_escape(&trap, &|_, _| {}).iter().any(|e| !e.inside && e.trap.is_finite()));
        let stripe = FractalSpec { coloring: Coloring::Stripe, ..trap };
        assert!(render_escape(&stripe, &|_, _| {}).iter().any(|e| !e.inside && e.stripe > 0.0));
    }
}
