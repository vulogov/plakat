//! Perturbation-theory deep zoom for the Mandelbrot set (RFC FRACTALS-1 → 4.2 "depth").
//!
//! `f64` runs out of mantissa around zoom ≈ 1e13 — the center coordinate can't be resolved
//! finer, so deeper zooms pixelate. Perturbation theory fixes this: iterate one
//! **high-precision reference orbit** `Zₙ` (arbitrary precision, via `astro-float` — pure
//! Rust, no GMP), then render every pixel as a cheap `f64` **delta** `δₙ` relative to it:
//!
//! ```text
//!   z = Zₙ + δₙ,   δₙ₊₁ = 2·Zₙ·δₙ + δₙ² + δc
//! ```
//!
//! Only the reference (and the center) need arbitrary precision — the per-pixel math is
//! `f64`, so deep zoom stays fast and parallel. Pixels whose orbit strays too far from the
//! reference **glitch**; we detect them (Pauldelbrot's criterion) and re-render them against
//! secondary references until the frame is clean.

use anyhow::{Context, Result};
use astro_float::{BigFloat, Consts, Radix, RoundingMode};
use num_complex::Complex;
use rayon::prelude::*;

use super::progress::ProgressFn;
use super::render::Escape;
use super::spec::FractalSpec;

/// Skip per-op rounding on the reference orbit — the working precision already carries
/// plenty of guard digits, so this is both faster and accurate enough.
const RM: RoundingMode = RoundingMode::None;
/// Pauldelbrot glitch criterion: a pixel is glitched at iteration n when |z|² dips below
/// this fraction of the reference |Zₙ|² (the pixel's orbit has diverged from the reference).
const GLITCH_TOL: f64 = 1e-6;
/// Cap on secondary-reference rounds (each cleans up the previous round's glitches).
const MAX_ROUNDS: usize = 16;

/// Zoom past which `f64` centers lose precision → perturbation is worth it.
pub const DEEP_ZOOM_THRESHOLD: f64 = 1e12;

/// The iteration budget a deep render actually needs — the per-pixel delta grows slowly
/// from a tiny `dc`, so a shallow `max_iter` renders a black frame. Scales ~`2500·log10(zoom)`
/// and never lowers the caller's request. Callers must apply this to the spec so the
/// **colorizer normalizes by the same value** (or deep frames wash out to 1–2 colors).
pub fn effective_max_iter(zoom: f64, requested: u32) -> u32 {
    let floor = (2500.0 * zoom.max(1.0).log10()) as u32;
    requested.max(floor).min(1_000_000)
}

/// Working precision (bits) for a given zoom: the decimal digits needed to resolve the
/// center (`log10(zoom)`) plus a generous guard, converted to bits.
fn precision_bits(zoom: f64) -> usize {
    let digits = zoom.max(1.0).log10() + 30.0;
    ((digits * std::f64::consts::LOG2_10).ceil() as usize).clamp(64, 200_000)
}

/// Downcast a `BigFloat` to `f64` via its decimal formatting (exact enough; reference-orbit
/// values are O(1)). Runs only during reference-orbit construction, never per pixel.
fn to_f64(b: &BigFloat, cc: &mut Consts) -> f64 {
    b.format(Radix::Dec, RoundingMode::ToEven, cc)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Parse a decimal-string coordinate to `BigFloat` at precision `p`.
fn parse_bf(s: &str, p: usize, cc: &mut Consts) -> Result<BigFloat> {
    let s = s.trim();
    let v = BigFloat::parse(s, Radix::Dec, p, RM, cc);
    if v.is_nan() {
        anyhow::bail!("could not parse high-precision coordinate {s:?}");
    }
    Ok(v)
}

/// A reference orbit: `Zₙ` downcast to `f64`, and the iteration at which the reference
/// itself escaped (or `max_iter`).
struct Reference {
    zn: Vec<Complex<f64>>,
    escaped_at: u32,
}

/// Iterate `z ← z² + c` at high precision from the origin, storing each `Zₙ` as `f64`.
fn reference_orbit(cx: &BigFloat, cy: &BigFloat, max_iter: u32, p: usize, cc: &mut Consts) -> Reference {
    let two = BigFloat::from_f64(2.0, p);
    let mut zx = BigFloat::from_f64(0.0, p);
    let mut zy = BigFloat::from_f64(0.0, p);
    let mut zn = Vec::with_capacity(max_iter as usize + 1);
    zn.push(Complex::new(0.0, 0.0));
    for n in 1..=max_iter {
        // zx' = zx² − zy² + cx ; zy' = 2·zx·zy + cy
        let zx2 = zx.mul(&zx, p, RM);
        let zy2 = zy.mul(&zy, p, RM);
        let xy = zx.mul(&zy, p, RM);
        let nzx = zx2.sub(&zy2, p, RM).add(cx, p, RM);
        let nzy = xy.mul(&two, p, RM).add(cy, p, RM);
        zx = nzx;
        zy = nzy;
        let (fx, fy) = (to_f64(&zx, cc), to_f64(&zy, cc));
        zn.push(Complex::new(fx, fy));
        if fx * fx + fy * fy > 4.0 {
            return Reference { zn, escaped_at: n };
        }
    }
    Reference { zn, escaped_at: max_iter }
}

/// The perturbation iteration for one pixel. Returns the `Escape` result, or — when the
/// pixel glitches — `Err(glitch_iteration)` so the driver can pick a good secondary
/// reference (the pixel that survived longest before glitching).
fn perturb(
    r: &Reference,
    dc: Complex<f64>,
    max_iter: u32,
    escape2: f64,
) -> std::result::Result<Escape, u32> {
    let mut d = Complex::new(0.0, 0.0);
    let mut n = 0u32;
    while n < r.escaped_at {
        let zref = r.zn[n as usize];
        let z = zref + d;
        let mag2 = z.norm_sqr();
        if mag2 > escape2 {
            let log_zn = 0.5 * mag2.ln();
            let smooth = (n as f64 - log_zn.ln() / std::f64::consts::LN_2).max(0.0);
            return Ok(Escape { inside: false, iters: n, smooth, final_z: z, ..Escape::default() });
        }
        // Pauldelbrot: |z|² ≪ |Zₙ|² → the orbit strayed from the reference (glitch).
        let refmag2 = zref.norm_sqr();
        if refmag2 > 0.0 && mag2 < GLITCH_TOL * refmag2 {
            return Err(n);
        }
        d = Complex::new(2.0, 0.0) * zref * d + d * d + dc;
        n += 1;
    }
    if r.escaped_at >= max_iter {
        // Reference never escaped → this pixel is interior (to the iteration cap).
        let z = r.zn[r.escaped_at as usize] + d;
        Ok(Escape { inside: true, iters: max_iter, smooth: max_iter as f64, final_z: z, ..Escape::default() })
    } else {
        // Reference escaped before this pixel did — needs a longer/closer reference.
        Err(r.escaped_at)
    }
}

/// Pixel `(px, py)` → its `f64` offset from the image center, in complex-plane units.
#[inline]
fn pixel_offset(px: usize, py: usize, w: usize, h: usize, scale: f64) -> Complex<f64> {
    let re = (px as f64 + 0.5 - w as f64 / 2.0) * scale;
    let im = -(py as f64 + 0.5 - h as f64 / 2.0) * scale; // +im points up
    Complex::new(re, im)
}

/// Render a deep-zoom Mandelbrot escape field via perturbation. Output matches
/// `render::render_escape` so the normal colorizers apply.
pub fn render_mandelbrot(spec: &FractalSpec, prog: ProgressFn) -> Result<Vec<Escape>> {
    let (w, h) = (spec.width as usize, spec.height as usize);
    let n = w * h;
    // Depth-scaled iteration budget (idempotent — the caller has usually already applied it
    // to the spec so the colorizer normalizes by the same value).
    let max_iter = effective_max_iter(spec.zoom, spec.max_iter);
    let escape2 = spec.escape_radius * spec.escape_radius;
    let scale = (3.0 / spec.zoom) / h as f64;
    let p = precision_bits(spec.zoom);

    let mut cc = Consts::new().context("astro-float constants")?;
    // Center: high-precision strings when supplied, else format the f64 center.
    let (cx, cy) = if !spec.center_hi[0].trim().is_empty() {
        (parse_bf(&spec.center_hi[0], p, &mut cc)?, parse_bf(&spec.center_hi[1], p, &mut cc)?)
    } else {
        (BigFloat::from_f64(spec.center[0], p), BigFloat::from_f64(spec.center[1], p))
    };

    let mut field = vec![Escape::default(); n]; // default = interior
    let mut pending: Vec<usize> = (0..n).collect();
    // First reference = the image center (offset 0).
    let (mut ref_cx, mut ref_cy) = (cx.clone(), cy.clone());
    let mut ref_off = Complex::new(0.0, 0.0);

    for round in 0..MAX_ROUNDS {
        if pending.is_empty() {
            break;
        }
        let reference = reference_orbit(&ref_cx, &ref_cy, max_iter, p, &mut cc);

        // Render all pending pixels against this reference, in parallel (pure f64).
        let results: Vec<(usize, std::result::Result<Escape, u32>)> = pending
            .par_iter()
            .map(|&idx| {
                let dc = pixel_offset(idx % w, idx / w, w, h, scale) - ref_off;
                (idx, perturb(&reference, dc, max_iter, escape2))
            })
            .collect();

        let mut glitched: Vec<(usize, u32)> = Vec::new(); // (pixel, iters survived before glitch)
        let mut n_esc = 0u64;
        for (idx, res) in results {
            match res {
                Ok(e) => {
                    if !e.inside {
                        n_esc += 1;
                    }
                    field[idx] = e;
                }
                Err(giter) => glitched.push((idx, giter)),
            }
        }
        if std::env::var("PLAKAT_DZ_DEBUG").is_ok() {
            eprintln!(
                "  dz round {round}: ref.escaped_at={} p={p}bits  escaped={n_esc} glitched={} pending_in={}",
                reference.escaped_at, glitched.len(), pending.len()
            );
        }
        prog((round + 1) as u64, MAX_ROUNDS as u64);
        if glitched.is_empty() {
            break;
        }
        // SEAMS-1 P8: re-reference at a pixel that is BOTH well-behaved (survived longest before
        // glitching → a representative orbit) AND central to the glitched region. A central reference
        // minimises the per-pixel delta magnitude across the cluster, so it cleans more of it per round —
        // fewer secondary rounds and fewer residual glitch blobs at extreme depth than picking the single
        // longest-surviving pixel (which can sit at a cluster edge).
        let maxg = glitched.iter().map(|&(_, g)| g).max().unwrap_or(0);
        let thresh = maxg.saturating_sub(maxg / 10); // the top ~10% of survival = the deepest orbits
        let (mut sx, mut sy) = (0f64, 0f64);
        for &(i, _) in &glitched {
            sx += (i % w) as f64;
            sy += (i / w) as f64;
        }
        let (cen_x, cen_y) = (sx / glitched.len() as f64, sy / glitched.len() as f64);
        let dist2 = |i: usize| ((i % w) as f64 - cen_x).powi(2) + ((i / w) as f64 - cen_y).powi(2);
        let pick = glitched
            .iter()
            .filter(|&&(_, g)| g >= thresh)
            .min_by(|&&(a, _), &&(b, _)| dist2(a).partial_cmp(&dist2(b)).unwrap())
            .map(|&(i, _)| i)
            .unwrap_or(glitched[glitched.len() / 2].0);
        let off = pixel_offset(pick % w, pick / w, w, h, scale);
        ref_cx = cx.add(&BigFloat::from_f64(off.re, p), p, RM);
        ref_cy = cy.add(&BigFloat::from_f64(off.im, p), p, RM);
        ref_off = off;
        pending = glitched.into_iter().map(|(i, _)| i).collect();
    }
    prog(MAX_ROUNDS as u64, MAX_ROUNDS as u64);
    Ok(field)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::FractalKind;

    fn base(zoom: f64) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Mandelbrot,
            width: 64,
            height: 64,
            center: [-0.5, 0.0],
            zoom,
            max_iter: 300,
            ..FractalSpec::default()
        }
    }

    #[test]
    fn precision_scales_with_zoom() {
        assert!(precision_bits(1e12) < precision_bits(1e60));
        assert!(precision_bits(1e100) > 400);
    }

    #[test]
    fn downcast_roundtrips() {
        let mut cc = Consts::new().unwrap();
        let b = parse_bf("-0.743643887037158704752191506", 256, &mut cc).unwrap();
        assert!((to_f64(&b, &mut cc) - -0.7436438870371587).abs() < 1e-12);
    }

    #[test]
    fn matches_f64_renderer_at_moderate_zoom() {
        // Where both are valid (zoom well within f64), perturbation must agree with the
        // direct f64 escape renderer on the inside/outside classification.
        let spec = base(500.0);
        let deep = render_mandelbrot(&spec, &|_, _| {}).unwrap();
        let direct = crate::fractals::render::render_escape(&spec, &|_, _| {});
        assert_eq!(deep.len(), direct.len());
        let mismatches = deep
            .iter()
            .zip(direct.iter())
            .filter(|(a, b)| a.inside != b.inside)
            .count();
        // Allow a tiny boundary-pixel tolerance (smooth vs perturbation rounding).
        assert!(mismatches <= 4, "inside/outside mismatch on {mismatches} pixels");
    }

    #[test]
    fn deep_zoom_produces_structure() {
        // A famous deep location, zoomed far past the f64 limit.
        let spec = FractalSpec {
            kind: FractalKind::Mandelbrot,
            width: 64,
            height: 64,
            center: [0.0, 0.0],
            center_hi: [
                "-0.743643887037158704752191506114774".into(),
                "0.131825904205311970493132056385139".into(),
            ],
            zoom: 1e15,
            // max_iter left at the default — the deep-zoom auto-floor raises it.
            ..FractalSpec::default()
        };
        let field = render_mandelbrot(&spec, &|_, _| {}).unwrap();
        assert_eq!(field.len(), 64 * 64);
        // The frame has both interior and exterior (real structure, not a flat blob).
        assert!(field.iter().any(|e| e.inside), "no interior pixels");
        assert!(field.iter().any(|e| !e.inside), "no exterior pixels");
        // P8 guard: unrendered glitch pixels default to `inside=true`, so a glitch WIPEOUT would leave
        // the frame almost entirely "interior". A substantial exterior fraction proves the secondary
        // references actually cleaned the glitched region.
        let exterior = field.iter().filter(|e| !e.inside).count();
        assert!(exterior > field.len() / 10, "exterior only {exterior}/{} — glitches not cleaned", field.len());
    }
}
