//! Iterated Function System — the *chaos game* (RFC FRACTALS-1, Phase 3).
//!
//! An IFS is a set of affine contraction maps with relative weights. Starting from a
//! point, we repeatedly pick a map (weighted) and apply it, plotting each visited point
//! into a density histogram — the attractor emerges (Barnsley fern, Sierpiński, dragon…).
//!
//! Deterministic (seeded `StdRng`) and memory-light: two passes over the same seeded
//! stream — pass 1 finds the attractor bounds, pass 2 splats into the histogram — so no
//! multi-million-point buffer is held. The density is colored via [`colorize_density`].

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::coloring::colorize_density;
use super::palette::Palette;
use super::plot::{Bounds, Fit};
use super::progress::ProgressFn;
use super::spec::{FractalSpec, IfsMap};

/// Resolve the affine maps for a spec: explicit `maps` win, else the named preset.
pub fn resolve_maps(spec: &FractalSpec) -> Result<Vec<IfsMap>> {
    if !spec.ifs.maps.is_empty() {
        return Ok(spec.ifs.maps.clone());
    }
    preset_maps(&spec.ifs.preset).with_context(|| {
        format!(
            "unknown IFS preset {:?} (want: barnsley-fern | sierpinski | dragon | levy | \
             tree | spiral — or supply explicit maps)",
            spec.ifs.preset
        )
    })
}

fn m(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, p: f64) -> IfsMap {
    IfsMap { a, b, c, d, e, f, p }
}

/// The built-in IFS presets.
pub fn preset_maps(name: &str) -> Option<Vec<IfsMap>> {
    let maps = match name.trim().to_ascii_lowercase().as_str() {
        "barnsley-fern" | "fern" => vec![
            m(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
            m(0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
            m(0.2, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
            m(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
        ],
        "sierpinski" | "sierpinski-triangle" => vec![
            m(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0),
            m(0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0),
            m(0.5, 0.0, 0.0, 0.5, 0.25, 0.433_012_7, 1.0),
        ],
        "dragon" | "heighway" => vec![
            m(0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 1.0),
            m(-0.5, -0.5, 0.5, -0.5, 1.0, 0.0, 1.0),
        ],
        "levy" | "levy-c" => vec![
            m(0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 1.0),
            m(0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 1.0),
        ],
        "tree" => vec![
            m(0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.05),
            m(0.42, -0.42, 0.42, 0.42, 0.0, 0.2, 0.4),
            m(0.42, 0.42, -0.42, 0.42, 0.0, 0.2, 0.4),
            m(0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 0.15),
        ],
        "spiral" => vec![
            m(0.787_879, -0.424_242, 0.242_424, 0.859_848, 1.758_647, 1.408_065, 0.9),
            m(-0.121_212, 0.257_576, 0.151_515, 0.053_030, -6.721_654, 1.377_236, 0.1),
        ],
        _ => return None,
    };
    Some(maps)
}

/// Cumulative-probability table for weighted map selection (normalized).
fn cumulative(maps: &[IfsMap]) -> Vec<f64> {
    let total: f64 = maps.iter().map(|m| m.p.max(0.0)).sum::<f64>().max(1e-12);
    let mut acc = 0.0;
    maps.iter()
        .map(|m| {
            acc += m.p.max(0.0) / total;
            acc
        })
        .collect()
}

#[inline]
fn pick(cum: &[f64], r: f64) -> usize {
    cum.iter().position(|&c| r <= c).unwrap_or(cum.len() - 1)
}

/// Run the chaos game over one seeded stream, invoking `visit(x, y)` for each post-warmup
/// point. Reproducible: same seed → same sequence.
fn chaos_game<F: FnMut(f64, f64)>(
    maps: &[IfsMap],
    cum: &[f64],
    seed: u64,
    iterations: u64,
    warmup: u32,
    mut visit: F,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut x, mut y) = (0.0f64, 0.0f64);
    for i in 0..iterations {
        let mp = &maps[pick(cum, rng.gen_range(0.0..1.0))];
        let nx = mp.a * x + mp.b * y + mp.e;
        let ny = mp.c * x + mp.d * y + mp.f;
        x = nx;
        y = ny;
        if i as u32 >= warmup && x.is_finite() && y.is_finite() {
            visit(x, y);
        }
    }
}

/// Render the IFS attractor to a packed `RGB8` buffer.
pub fn render(spec: &FractalSpec, palette: &Palette, prog: ProgressFn) -> Result<Vec<u8>> {
    let maps = resolve_maps(spec)?;
    let cum = cumulative(&maps);
    let (w, h) = (spec.width as usize, spec.height as usize);
    let iters = spec.ifs.iterations;
    let warmup = spec.ifs.warmup;
    let total = iters * 2; // two passes

    // Pass 1 — attractor bounds.
    let mut bounds = Bounds::empty();
    let mut counter = 0u64;
    chaos_game(&maps, &cum, spec.seed, iters, warmup, |x, y| {
        bounds.include(x, y);
        counter += 1;
        if counter % 262_144 == 0 {
            prog(counter, total);
        }
    });
    if !bounds.is_valid() {
        anyhow::bail!("IFS produced no finite points (check the maps)");
    }
    let fit = Fit::new(&bounds, spec.width, spec.height, spec.ifs.margin, spec.zoom);

    // Pass 2 — density accumulation (same seed → same stream).
    let mut hist = vec![0u32; w * h];
    chaos_game(&maps, &cum, spec.seed, iters, warmup, |x, y| {
        if let Some((px, py)) = fit.map_px(x, y) {
            hist[py as usize * w + px as usize] = hist[py as usize * w + px as usize].saturating_add(1);
        }
        counter += 1;
        if counter % 262_144 == 0 {
            prog(counter, total);
        }
    });
    prog(total, total);

    let max = hist.iter().copied().max().unwrap_or(0);
    Ok(colorize_density(&hist, max, palette))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::{FractalKind, IfsSpec, PaletteSpec};

    fn spec_for(preset: &str) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Ifs,
            width: 64,
            height: 64,
            ifs: IfsSpec { preset: preset.into(), iterations: 200_000, ..IfsSpec::default() },
            ..FractalSpec::default()
        }
    }

    #[test]
    fn presets_resolve() {
        for p in ["barnsley-fern", "sierpinski", "dragon", "levy", "tree", "spiral"] {
            assert!(resolve_maps(&spec_for(p)).is_ok(), "{p} failed");
        }
        assert!(resolve_maps(&spec_for("nope")).is_err());
    }

    #[test]
    fn fern_renders_and_is_deterministic() {
        let spec = spec_for("barnsley-fern");
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        let a = render(&spec, &pal, &|_, _| {}).unwrap();
        let b = render(&spec, &pal, &|_, _| {}).unwrap();
        assert_eq!(a, b, "same spec → same pixels");
        assert_eq!(a.len(), 64 * 64 * 3);
        assert!(a.chunks(3).any(|p| p != &a[0..3]), "attractor drew something");
    }

    #[test]
    fn custom_maps_override_preset() {
        let spec = FractalSpec {
            kind: FractalKind::Ifs,
            width: 48,
            height: 48,
            ifs: IfsSpec {
                preset: "barnsley-fern".into(), // ignored because maps is non-empty
                maps: vec![
                    m(0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0),
                    m(0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 1.0),
                    m(0.5, 0.0, 0.0, 0.5, 0.25, 0.5, 1.0),
                ],
                iterations: 100_000,
                ..IfsSpec::default()
            },
            ..FractalSpec::default()
        };
        assert_eq!(resolve_maps(&spec).unwrap().len(), 3);
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        assert!(render(&spec, &pal, &|_, _| {}).is_ok());
    }
}
