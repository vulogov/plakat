//! Strange attractors — density plots of chaotic trajectories (RFC FRACTALS-1, Phase 5).
//!
//! A single deterministic trajectory of a chaotic 2D map (Clifford, De Jong, Bedhead,
//! Duffing, Ikeda) or a 3D ODE (Lorenz, Rössler, integrated with RK4 and projected to 2D)
//! is followed for millions of steps; each visited point is splatted into a density
//! histogram, revealing the attractor. Deterministic (fixed start, no RNG) and
//! memory-light (two-pass: bounds → density), colored via [`colorize_density`].

use anyhow::{Context, Result};

use super::coloring::colorize_density;
use super::palette::Palette;
use super::plot::{Bounds, Fit};
use super::progress::ProgressFn;
use super::spec::FractalSpec;

/// Which attractor equation to iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttractorKind {
    Clifford,
    DeJong,
    Bedhead,
    Duffing,
    Ikeda,
    Lorenz,
    Rossler,
    Svensson,
    Hopalong,
    FractalDream,
}

impl AttractorKind {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "clifford" => AttractorKind::Clifford,
            "dejong" | "de-jong" | "peter-de-jong" => AttractorKind::DeJong,
            "bedhead" => AttractorKind::Bedhead,
            "duffing" => AttractorKind::Duffing,
            "ikeda" => AttractorKind::Ikeda,
            "lorenz" => AttractorKind::Lorenz,
            "rossler" | "rössler" => AttractorKind::Rossler,
            "svensson" => AttractorKind::Svensson,
            "hopalong" => AttractorKind::Hopalong,
            "fractal-dream" | "fractaldream" | "dream" => AttractorKind::FractalDream,
            other => anyhow::bail!(
                "unknown attractor {other:?} (want: clifford | dejong | bedhead | duffing | \
                 ikeda | lorenz | rossler | svensson | hopalong | fractal-dream)"
            ),
        })
    }

    /// Whether this is a continuous ODE (integrated) vs a discrete map (iterated).
    fn is_ode(self) -> bool {
        matches!(self, AttractorKind::Lorenz | AttractorKind::Rossler)
    }

    /// The classic parameters for this attractor.
    pub fn default_params(self) -> Vec<f64> {
        match self {
            AttractorKind::Clifford => vec![-1.4, 1.6, 1.0, 0.7],
            AttractorKind::DeJong => vec![1.641, 1.902, 0.316, 1.525],
            AttractorKind::Bedhead => vec![-0.81, -0.92],
            AttractorKind::Duffing => vec![2.75, 0.2],
            AttractorKind::Ikeda => vec![0.918],
            AttractorKind::Lorenz => vec![10.0, 28.0, 8.0 / 3.0],
            AttractorKind::Rossler => vec![0.2, 0.2, 5.7],
            AttractorKind::Svensson => vec![1.40, 1.56, 1.40, -6.56],
            AttractorKind::Hopalong => vec![2.0, 1.0, 0.0],
            AttractorKind::FractalDream => vec![-0.966, 2.879, 0.765, 0.744],
        }
    }
}

/// One step of a 2D map: `(x, y) → (x', y')`.
fn map_step(kind: AttractorKind, x: f64, y: f64, p: &[f64]) -> (f64, f64) {
    match kind {
        AttractorKind::Clifford => {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            ((a * y).sin() + c * (a * x).cos(), (b * x).sin() + d * (b * y).cos())
        }
        AttractorKind::DeJong => {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            ((a * y).sin() - (b * x).cos(), (c * x).sin() - (d * y).cos())
        }
        AttractorKind::Bedhead => {
            let (a, b) = (p[0], p[1]);
            ((x * y / b).sin() * y + (a * x - y).cos(), x + y.sin() / b)
        }
        AttractorKind::Duffing => {
            // Discrete Duffing map.
            let (a, b) = (p[0], p[1]);
            (y, -b * x + a * y - y * y * y)
        }
        AttractorKind::Ikeda => {
            let u = p[0];
            let t = 0.4 - 6.0 / (1.0 + x * x + y * y);
            (1.0 + u * (x * t.cos() - y * t.sin()), u * (x * t.sin() + y * t.cos()))
        }
        AttractorKind::Svensson => {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            (d * (a * x).sin() - (b * y).sin(), c * (a * x).cos() + (b * y).cos())
        }
        AttractorKind::Hopalong => {
            let (a, b, c) = (p[0], p[1], p[2]);
            (y - x.signum() * (b * x - c).abs().sqrt(), a - x)
        }
        AttractorKind::FractalDream => {
            let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
            ((b * y).sin() + c * (b * x).sin(), (a * x).sin() + d * (a * y).sin())
        }
        _ => unreachable!("map_step called for an ODE"),
    }
}

/// One RK4 step of a 3D ODE `(x,y,z) → (x',y',z')` with timestep `dt`.
fn ode_step(kind: AttractorKind, x: f64, y: f64, z: f64, p: &[f64], dt: f64) -> (f64, f64, f64) {
    let deriv = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        match kind {
            AttractorKind::Lorenz => {
                let (s, r, b) = (p[0], p[1], p[2]);
                (s * (y - x), x * (r - z) - y, x * y - b * z)
            }
            AttractorKind::Rossler => {
                let (a, b, c) = (p[0], p[1], p[2]);
                (-y - z, x + a * y, b + z * (x - c))
            }
            _ => unreachable!("ode_step called for a map"),
        }
    };
    let k1 = deriv(x, y, z);
    let k2 = deriv(x + 0.5 * dt * k1.0, y + 0.5 * dt * k1.1, z + 0.5 * dt * k1.2);
    let k3 = deriv(x + 0.5 * dt * k2.0, y + 0.5 * dt * k2.1, z + 0.5 * dt * k2.2);
    let k4 = deriv(x + dt * k3.0, y + dt * k3.1, z + dt * k3.2);
    (
        x + dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
        y + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1),
        z + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2),
    )
}

/// Project a (possibly 3D) state to the 2D plotting plane.
fn project(kind: AttractorKind, x: f64, y: f64, z: f64) -> (f64, f64) {
    match kind {
        AttractorKind::Lorenz => (x, z), // the classic butterfly view
        AttractorKind::Rossler => (x, y),
        _ => (x, y),
    }
}

/// Walk the trajectory, invoking `visit(px, py)` with each post-warmup projected point.
fn trajectory<F: FnMut(f64, f64)>(
    kind: AttractorKind,
    p: &[f64],
    iterations: u64,
    warmup: u32,
    mut visit: F,
) {
    let (mut x, mut y, mut z) = (0.1, 0.0, 0.0);
    let dt = 0.008;
    for i in 0..iterations {
        if kind.is_ode() {
            let (nx, ny, nz) = ode_step(kind, x, y, z, p, dt);
            x = nx;
            y = ny;
            z = nz;
        } else {
            let (nx, ny) = map_step(kind, x, y, p);
            x = nx;
            y = ny;
        }
        if i as u32 >= warmup && x.is_finite() && y.is_finite() && z.is_finite() {
            let (px, py) = project(kind, x, y, z);
            visit(px, py);
        }
    }
}

/// Render the strange attractor to a packed `RGB8` buffer.
pub fn render(spec: &FractalSpec, palette: &Palette, prog: ProgressFn) -> Result<Vec<u8>> {
    let kind = AttractorKind::parse(&spec.attractor.preset)
        .with_context(|| format!("resolving attractor {:?}", spec.attractor.preset))?;
    let params = if spec.attractor.params.is_empty() {
        kind.default_params()
    } else {
        spec.attractor.params.clone()
    };
    let needed = match kind {
        AttractorKind::Ikeda => 1,
        AttractorKind::Bedhead | AttractorKind::Duffing => 2,
        AttractorKind::Hopalong | AttractorKind::Lorenz | AttractorKind::Rossler => 3,
        // Clifford / DeJong / Svensson / FractalDream index p[3].
        _ => 4,
    };
    if params.len() < needed {
        anyhow::bail!("attractor {:?} needs {needed} parameters, got {}", kind, params.len());
    }

    let (w, h) = (spec.width as usize, spec.height as usize);
    let iters = spec.attractor.iterations;
    let warmup = spec.attractor.warmup;
    let total = iters * 2;

    // Pass 1 — bounds.
    let mut bounds = Bounds::empty();
    let mut counter = 0u64;
    trajectory(kind, &params, iters, warmup, |x, y| {
        bounds.include(x, y);
        counter += 1;
        if counter % 262_144 == 0 {
            prog(counter, total);
        }
    });
    if !bounds.is_valid() {
        anyhow::bail!("attractor produced no finite points");
    }
    let fit = Fit::new(&bounds, spec.width, spec.height, spec.attractor.margin, spec.zoom);

    // Pass 2 — density.
    let mut hist = vec![0u32; w * h];
    trajectory(kind, &params, iters, warmup, |x, y| {
        if let Some((px, py)) = fit.map_px(x, y) {
            let idx = py as usize * w + px as usize;
            hist[idx] = hist[idx].saturating_add(1);
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
    use crate::fractals::spec::{AttractorSpec, FractalKind, PaletteSpec};

    fn spec_for(preset: &str) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Attractor,
            width: 64,
            height: 64,
            attractor: AttractorSpec { preset: preset.into(), iterations: 200_000, ..AttractorSpec::default() },
            ..FractalSpec::default()
        }
    }

    #[test]
    fn all_attractors_render_and_are_deterministic() {
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        for p in [
            "clifford", "dejong", "bedhead", "duffing", "ikeda", "lorenz", "rossler",
            "svensson", "hopalong", "fractal-dream",
        ] {
            let spec = spec_for(p);
            let a = render(&spec, &pal, &|_, _| {}).unwrap();
            let b = render(&spec, &pal, &|_, _| {}).unwrap();
            assert_eq!(a, b, "{p} not deterministic");
            assert_eq!(a.len(), 64 * 64 * 3);
            assert!(a.chunks(3).any(|px| px != &a[0..3]), "{p} was flat");
        }
    }

    #[test]
    fn unknown_preset_errors() {
        assert!(AttractorKind::parse("nope").is_err());
    }

    #[test]
    fn custom_params_apply() {
        let spec = FractalSpec {
            attractor: AttractorSpec {
                preset: "clifford".into(),
                params: vec![-1.7, 1.3, -0.1, -1.2],
                iterations: 100_000,
                ..AttractorSpec::default()
            },
            ..spec_for("clifford")
        };
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        assert!(render(&spec, &pal, &|_, _| {}).is_ok());
    }
}
