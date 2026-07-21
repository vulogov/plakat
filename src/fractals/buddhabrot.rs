//! Buddhabrot — the density plot of *escaping* Mandelbrot orbits (RFC FRACTALS-1, Phase 2).
//!
//! Unlike the per-pixel escape families, buddhabrot is a Monte-Carlo accumulation: sample
//! points `c`, and for each that escapes, splat every point its orbit visited into a
//! histogram. We keep it deterministic by seeding one RNG per work-chunk from `spec.seed`
//! and the chunk index (integer histogram sums are order-independent), so `same spec →
//! same density` regardless of thread scheduling.

use num_complex::Complex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use super::progress::ProgressFn;
use super::render::Viewport;
use super::spec::FractalSpec;

/// Sampling region in the c-plane — the classic Mandelbrot bounds, so orbits that pass
/// *through* a zoomed viewport but originate outside it still contribute.
const SAMPLE_RE: (f64, f64) = (-2.2, 1.0);
const SAMPLE_IM: (f64, f64) = (-1.5, 1.5);
/// Escape test for the sampling pass (radius 2 is exact for Mandelbrot).
const ESCAPE2: f64 = 4.0;

/// Accumulate the buddhabrot density into a `width*height` histogram (row-major),
/// returning `(histogram, max_count)`. `prog(done_chunks, total_chunks)` fires as
/// sampling chunks complete.
pub fn render_density(spec: &FractalSpec, prog: ProgressFn) -> (Vec<u32>, u32) {
    let vp = Viewport::new(spec);
    let (w, h) = (spec.width as usize, spec.height as usize);
    let n = w * h;
    let max_iter = spec.max_iter;
    let min_iter = spec.buddha_min_iter;

    // Split the sample budget across a fixed number of chunks so the workload is
    // deterministic (chunk `i` always draws the same samples from seed ⊕ i).
    let chunks: u64 = 64;
    let per = spec.buddha_samples / chunks;
    let done = AtomicU64::new(0);

    let partials: Vec<Vec<u32>> = (0..chunks)
        .into_par_iter()
        .map(|ci| {
            let mut hist = vec![0u32; n];
            let mut rng = StdRng::seed_from_u64(spec.seed ^ (ci.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let mut orbit: Vec<Complex<f64>> = Vec::with_capacity(max_iter as usize);
            for _ in 0..per {
                let c = Complex::new(
                    rng.gen_range(SAMPLE_RE.0..SAMPLE_RE.1),
                    rng.gen_range(SAMPLE_IM.0..SAMPLE_IM.1),
                );
                orbit.clear();
                let mut z = Complex::new(0.0, 0.0);
                let mut escaped = false;
                let mut iters = 0u32;
                while iters < max_iter {
                    z = z * z + c;
                    iters += 1;
                    orbit.push(z);
                    if z.norm_sqr() > ESCAPE2 {
                        escaped = true;
                        break;
                    }
                }
                // Only escaping orbits contribute; the min-iter gate drops the low-detail halo.
                if escaped && iters >= min_iter {
                    for &p in &orbit {
                        if let Some((px, py)) = vp.complex_to_pixel(p) {
                            hist[py as usize * w + px as usize] += 1;
                        }
                    }
                }
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            prog(d, chunks);
            hist
        })
        .collect();

    // Sum the partial histograms (commutative → deterministic).
    let mut hist = vec![0u32; n];
    for part in &partials {
        for (acc, &v) in hist.iter_mut().zip(part.iter()) {
            *acc = acc.saturating_add(v);
        }
    }
    let max = hist.iter().copied().max().unwrap_or(0);
    (hist, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::FractalKind;

    fn small() -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Buddhabrot,
            width: 48,
            height: 48,
            center: [-0.5, 0.0],
            zoom: 0.7,
            max_iter: 200,
            buddha_samples: 200_000,
            buddha_min_iter: 5,
            seed: 42,
            ..FractalSpec::default()
        }
    }

    #[test]
    fn density_is_deterministic_and_nonempty() {
        let (a, amax) = render_density(&small(), &|_, _| {});
        let (b, bmax) = render_density(&small(), &|_, _| {});
        assert_eq!(a, b, "same spec → same density");
        assert_eq!(amax, bmax);
        assert!(amax > 0, "some pixels accumulated");
        assert_eq!(a.len(), 48 * 48);
        assert!(a.iter().any(|&v| v > 0));
    }

    #[test]
    fn different_seed_changes_density() {
        let (a, _) = render_density(&small(), &|_, _| {});
        let (b, _) = render_density(&FractalSpec { seed: 7, ..small() }, &|_, _| {});
        assert_ne!(a, b);
    }
}
