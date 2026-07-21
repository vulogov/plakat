//! Fractal flame — IFS + non-linear variations + log-density color (RFC FRACTALS-1,
//! Phase 5), after Scott Draves' algorithm.
//!
//! Like the chaos game, but each function applies an affine pre-transform followed by a
//! weighted sum of non-linear *variations*, and carries a color coordinate that blends
//! along the orbit. Points accumulate into per-pixel hit-count + color sums; the final
//! image is log-density tone-mapped with gamma. Deterministic (seeded `StdRng`).

use anyhow::{Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

use super::palette::Palette;
use super::plot::{Bounds, Fit};
use super::progress::ProgressFn;
use super::spec::{FlameFunction, FractalSpec, VarWeight};

/// The supported variation names, indexed by id.
pub const VARIATIONS: &[&str] = &[
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar", "handkerchief",
    "heart", "disc", "spiral", "hyperbolic", "diamond", "ex", "fisheye", "exponential",
    "power", "cosine", "bubble",
];

fn variation_id(name: &str) -> Result<usize> {
    VARIATIONS
        .iter()
        .position(|v| v.eq_ignore_ascii_case(name.trim()))
        .with_context(|| {
            format!("unknown flame variation {name:?} (one of: {})", VARIATIONS.join(", "))
        })
}

const EPS: f64 = 1e-9;

/// Apply variation `id` to `(x, y)`.
fn apply_variation(id: usize, x: f64, y: f64) -> (f64, f64) {
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let theta = y.atan2(x);
    match id {
        0 => (x, y),                                    // linear
        1 => (x.sin(), y.sin()),                        // sinusoidal
        2 => (x / (r2 + EPS), y / (r2 + EPS)),          // spherical
        3 => (x * r2.sin() - y * r2.cos(), x * r2.cos() + y * r2.sin()), // swirl
        4 => ((x - y) * (x + y) / (r + EPS), 2.0 * x * y / (r + EPS)),   // horseshoe
        5 => (theta / PI, r - 1.0),                     // polar
        6 => (r * (theta + r).sin(), r * (theta - r).cos()), // handkerchief
        7 => (r * (theta * r).sin(), -r * (theta * r).cos()), // heart
        8 => {
            let t = theta / PI;
            (t * (PI * r).sin(), t * (PI * r).cos()) // disc
        }
        9 => ((theta.cos() + r.sin()) / (r + EPS), (theta.sin() - r.cos()) / (r + EPS)), // spiral
        10 => (theta.sin() / (r + EPS), r * theta.cos()), // hyperbolic
        11 => (theta.sin() * r.cos(), theta.cos() * r.sin()), // diamond
        12 => {
            let p0 = (theta + r).sin();
            let p1 = (theta - r).cos();
            (r * (p0.powi(3) + p1.powi(3)), r * (p0.powi(3) - p1.powi(3))) // ex
        }
        13 => (2.0 / (r + 1.0) * y, 2.0 / (r + 1.0) * x), // fisheye
        14 => ((x - 1.0).exp() * (PI * y).cos(), (x - 1.0).exp() * (PI * y).sin()), // exponential
        15 => {
            let rp = r.powf(theta.sin());
            (rp * theta.cos(), rp * theta.sin()) // power
        }
        16 => ((PI * x).cos() * y.cosh(), -(PI * x).sin() * y.sinh()), // cosine
        17 => {
            let f = 4.0 / (r2 + 4.0);
            (f * x, f * y) // bubble
        }
        _ => (x, y),
    }
}

/// A flame function with its variations resolved to ids.
struct ResolvedFn {
    affine: [f64; 6],
    vars: Vec<(usize, f64)>,
    color: f64,
    weight: f64,
}

fn resolve_functions(spec: &FractalSpec) -> Result<Vec<ResolvedFn>> {
    let raw = if !spec.flame.functions.is_empty() {
        spec.flame.functions.clone()
    } else {
        preset_functions(&spec.flame.preset).with_context(|| {
            format!(
                "unknown flame preset {:?} (want: sierpinski | spherical | swirl | spiral | flame \
                 — or supply explicit functions)",
                spec.flame.preset
            )
        })?
    };
    raw.iter()
        .map(|f| {
            let vars = f
                .variations
                .iter()
                .map(|v| Ok((variation_id(&v.name)?, v.weight)))
                .collect::<Result<Vec<_>>>()?;
            Ok(ResolvedFn { affine: f.affine, vars, color: f.color, weight: f.weight.max(0.0) })
        })
        .collect()
}

fn f(affine: [f64; 6], vars: &[(&str, f64)], color: f64, weight: f64) -> FlameFunction {
    FlameFunction {
        affine,
        variations: vars.iter().map(|(n, w)| VarWeight { name: (*n).into(), weight: *w }).collect(),
        color,
        weight,
    }
}

/// Built-in flame presets.
pub fn preset_functions(name: &str) -> Option<Vec<FlameFunction>> {
    let fns = match name.trim().to_ascii_lowercase().as_str() {
        "sierpinski" => vec![
            f([0.5, 0.0, 0.0, 0.0, 0.5, 0.0], &[("linear", 1.0)], 0.0, 1.0),
            f([0.5, 0.0, 0.5, 0.0, 0.5, 0.0], &[("linear", 1.0)], 0.5, 1.0),
            f([0.5, 0.0, 0.25, 0.0, 0.5, 0.5], &[("linear", 1.0)], 1.0, 1.0),
        ],
        "spherical" => vec![
            f([0.9, 0.0, 0.0, 0.0, 0.9, 0.0], &[("spherical", 1.0)], 0.0, 1.0),
            f([0.5, 0.0, 0.4, 0.0, 0.5, 0.0], &[("linear", 1.0)], 0.7, 1.0),
            f([0.5, 0.0, -0.4, 0.0, 0.5, 0.3], &[("sinusoidal", 1.0)], 1.0, 1.0),
        ],
        "swirl" => vec![
            f([0.7, 0.3, 0.0, -0.3, 0.7, 0.0], &[("swirl", 1.0)], 0.0, 1.0),
            f([0.4, 0.0, 0.4, 0.0, 0.4, 0.0], &[("spherical", 1.0)], 0.9, 1.0),
        ],
        "spiral" => vec![
            f([0.8, 0.0, 0.0, 0.0, 0.8, 0.0], &[("spiral", 1.0)], 0.0, 1.0),
            f([0.5, 0.0, 0.3, 0.0, 0.5, 0.2], &[("linear", 0.5), ("sinusoidal", 0.5)], 1.0, 1.0),
        ],
        "flame" => vec![
            f([0.85, 0.0, 0.0, 0.0, 0.85, 0.0], &[("spherical", 1.0)], 0.0, 1.0),
            f([0.5, 0.0, 0.45, 0.0, 0.5, 0.0], &[("sinusoidal", 1.0)], 0.5, 1.0),
            f([0.5, 0.0, -0.45, 0.0, 0.5, 0.3], &[("swirl", 1.0)], 1.0, 1.0),
        ],
        _ => return None,
    };
    Some(fns)
}

fn cumulative(fns: &[ResolvedFn]) -> Vec<f64> {
    let total: f64 = fns.iter().map(|f| f.weight).sum::<f64>().max(1e-12);
    let mut acc = 0.0;
    fns.iter()
        .map(|f| {
            acc += f.weight / total;
            acc
        })
        .collect()
}

/// Run the flame chaos game, invoking `visit(x, y, color)` for each post-warmup point.
/// Reproducible for a fixed `seed`.
fn chaos<F: FnMut(f64, f64, f64)>(
    fns: &[ResolvedFn],
    cum: &[f64],
    seed: u64,
    iterations: u64,
    warmup: u32,
    mut visit: F,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut x, mut y) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
    let mut col: f64 = rng.gen_range(0.0..1.0);
    for i in 0..iterations {
        let r = rng.gen_range(0.0..1.0);
        let fi = cum.iter().position(|&c| r <= c).unwrap_or(cum.len() - 1);
        let f = &fns[fi];
        let a = &f.affine;
        let (ax, ay) = (a[0] * x + a[1] * y + a[2], a[3] * x + a[4] * y + a[5]);
        let (mut vx, mut vy) = (0.0, 0.0);
        for &(vid, w) in &f.vars {
            let (px, py) = apply_variation(vid, ax, ay);
            vx += w * px;
            vy += w * py;
        }
        x = vx;
        y = vy;
        col = (col + f.color) * 0.5;
        if i as u32 >= warmup && x.is_finite() && y.is_finite() {
            visit(x, y, col);
        }
    }
}

/// Render the fractal flame to a packed `RGB8` buffer.
pub fn render(spec: &FractalSpec, palette: &Palette, prog: ProgressFn) -> Result<Vec<u8>> {
    let fns = resolve_functions(spec)?;
    if fns.is_empty() {
        anyhow::bail!("flame has no functions");
    }
    let cum = cumulative(&fns);
    let (w, h) = (spec.width as usize, spec.height as usize);
    let iters = spec.flame.iterations;
    let warmup = spec.flame.warmup;
    let sym = spec.flame.symmetry.max(1);
    let total = iters * 2;

    // Pass 1 — bounds.
    let mut bounds = Bounds::empty();
    let mut counter = 0u64;
    chaos(&fns, &cum, spec.seed, iters, warmup, |x, y, _| {
        bounds.include(x, y);
        counter += 1;
        if counter % 262_144 == 0 {
            prog(counter, total);
        }
    });
    if !bounds.is_valid() {
        anyhow::bail!("flame produced no finite points (check the functions)");
    }
    let fit = Fit::new(&bounds, spec.width, spec.height, spec.flame.margin, spec.zoom);

    // Pass 2 — hit count + accumulated color.
    let n = w * h;
    let mut count = vec![0u32; n];
    let mut rgb = vec![[0f32; 3]; n];
    let rot: Vec<(f64, f64)> = (0..sym)
        .map(|s| {
            let a = 2.0 * PI * s as f64 / sym as f64;
            (a.cos(), a.sin())
        })
        .collect();
    chaos(&fns, &cum, spec.seed, iters, warmup, |x, y, col| {
        let [cr, cg, cb] = palette.sample(col);
        for &(ca, sa) in &rot {
            let (rx, ry) = (x * ca - y * sa, x * sa + y * ca);
            if let Some((px, py)) = fit.map_px(rx, ry) {
                let idx = py as usize * w + px as usize;
                count[idx] = count[idx].saturating_add(1);
                rgb[idx][0] += cr as f32;
                rgb[idx][1] += cg as f32;
                rgb[idx][2] += cb as f32;
            }
        }
        counter += 1;
        if counter % 262_144 == 0 {
            prog(counter, total);
        }
    });
    prog(total, total);

    // Log-density tone mapping with gamma + brightness.
    let max = count.iter().copied().max().unwrap_or(0);
    let lmax = ((max as f64) + 1.0).ln().max(1e-9);
    let inv_gamma = 1.0 / spec.flame.gamma;
    let bright = spec.flame.brightness;
    let interior = palette.interior();
    let mut buf = vec![0u8; n * 3];
    for i in 0..n {
        let c = count[i];
        if c == 0 {
            buf[i * 3..i * 3 + 3].copy_from_slice(&interior);
            continue;
        }
        let cf = c as f64;
        let alpha = (((cf + 1.0).ln() / lmax).powf(inv_gamma) * bright).clamp(0.0, 1.0);
        for k in 0..3 {
            let avg = rgb[i][k] as f64 / cf; // mean color 0..255
            buf[i * 3 + k] = (avg * alpha).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::{FlameSpec, FractalKind, PaletteSpec};

    fn spec_for(preset: &str) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Flame,
            width: 64,
            height: 64,
            flame: FlameSpec { preset: preset.into(), iterations: 300_000, ..FlameSpec::default() },
            ..FractalSpec::default()
        }
    }

    #[test]
    fn variation_ids_resolve() {
        assert_eq!(variation_id("linear").unwrap(), 0);
        assert_eq!(variation_id("Bubble").unwrap(), VARIATIONS.len() - 1);
        assert!(variation_id("nope").is_err());
    }

    #[test]
    fn all_presets_render_and_are_deterministic() {
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        for p in ["sierpinski", "spherical", "swirl", "spiral", "flame"] {
            let spec = spec_for(p);
            let a = render(&spec, &pal, &|_, _| {}).unwrap();
            let b = render(&spec, &pal, &|_, _| {}).unwrap();
            assert_eq!(a, b, "{p} not deterministic");
            assert_eq!(a.len(), 64 * 64 * 3);
            assert!(a.chunks(3).any(|px| px != &a[0..3]), "{p} was flat");
        }
    }

    #[test]
    fn symmetry_changes_output() {
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        let base = spec_for("swirl");
        let mut sym = base.clone();
        sym.flame.symmetry = 6;
        assert_ne!(render(&base, &pal, &|_, _| {}).unwrap(), render(&sym, &pal, &|_, _| {}).unwrap());
    }

    #[test]
    fn bad_variation_name_errors() {
        let spec = FractalSpec {
            flame: FlameSpec {
                functions: vec![f([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[("nope", 1.0)], 0.0, 1.0)],
                ..FlameSpec::default()
            },
            ..spec_for("flame")
        };
        assert!(render(&spec, &Palette::from_spec(&PaletteSpec::default()).unwrap(), &|_, _| {}).is_err());
    }
}
