//! 3D distance-estimated fractals, sphere-traced (RFC FRACTALS-1, Phase 7).
//!
//! Each shape is a signed distance estimator (DE) in ℝ³; a per-pixel ray is marched by
//! the DE (sphere tracing) until it hits the surface, then shaded with Phong + ambient
//! occlusion. The distance field *is* the depth map — which is why Track B conditions
//! these on Depth. Pure CPU, rayon-parallel per row, deterministic. Uses `nalgebra` for
//! the vector math (the dep added back in Phase 1 for exactly this).

use anyhow::Result;
use nalgebra::Vector3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use super::palette::Palette;
use super::progress::ProgressFn;
use super::spec::{FractalSpec, RaymarchSpec};

type V3 = Vector3<f64>;

/// The supported 3D fractal shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Mandelbulb,
    Mandelbox,
    Menger,
    Sierpinski3d,
    QuatJulia,
}

impl Shape {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "mandelbulb" | "bulb" => Shape::Mandelbulb,
            "mandelbox" | "box" => Shape::Mandelbox,
            "menger" | "menger-sponge" | "sponge" => Shape::Menger,
            "sierpinski3d" | "sierpinski" | "tetra" | "tetrahedron" => Shape::Sierpinski3d,
            "quat-julia" | "quatjulia" | "julia3d" | "quaternion" => Shape::QuatJulia,
            other => anyhow::bail!(
                "unknown 3D shape {other:?} (want: mandelbulb | mandelbox | menger | \
                 sierpinski3d | quat-julia)"
            ),
        })
    }
}

// ── Distance estimators ──────────────────────────────────────────────────────
// Each returns (signed_distance, orbit_trap) where the trap ∈ [0,1] drives coloring.

fn de_mandelbulb(p: V3, power: f64, iters: u32) -> (f64, f64) {
    let mut z = p;
    let mut dr = 1.0;
    let mut r = 0.0;
    let mut trap = f64::INFINITY;
    for _ in 0..iters {
        r = z.norm();
        if r > 2.0 {
            break;
        }
        trap = trap.min(r);
        let theta = (z.z / r).acos();
        let phi = z.y.atan2(z.x);
        dr = r.powf(power - 1.0) * power * dr + 1.0;
        let zr = r.powf(power);
        let (t, ph) = (theta * power, phi * power);
        z = zr * V3::new(t.sin() * ph.cos(), t.sin() * ph.sin(), t.cos()) + p;
    }
    (0.5 * r.max(1e-9).ln() * r / dr, normalize_trap(trap, 2.0))
}

fn de_mandelbox(p: V3, scale: f64, iters: u32) -> (f64, f64) {
    let (min_r2, fixed_r2) = (0.25, 1.0);
    let mut z = p;
    let mut dr = 1.0;
    let mut trap = f64::INFINITY;
    for _ in 0..iters {
        // Box fold.
        z = z.map(|c| c.clamp(-1.0, 1.0) * 2.0 - c);
        // Sphere fold.
        let r2 = z.dot(&z);
        if r2 < min_r2 {
            let t = fixed_r2 / min_r2;
            z *= t;
            dr *= t;
        } else if r2 < fixed_r2 {
            let t = fixed_r2 / r2;
            z *= t;
            dr *= t;
        }
        z = z * scale + p;
        dr = dr * scale.abs() + 1.0;
        trap = trap.min(z.norm());
    }
    (z.norm() / dr.abs(), normalize_trap(trap, 4.0))
}

fn de_menger(p: V3, iters: u32) -> (f64, f64) {
    // IQ's cross-subtraction Menger sponge.
    let bd = box_de(p, V3::new(1.0, 1.0, 1.0));
    let mut d = bd;
    let mut s = 1.0;
    let mut trap = f64::INFINITY;
    for _ in 0..iters {
        let a = (p * s).map(|c| c.rem_euclid(2.0) - 1.0);
        s *= 3.0;
        let r = a.map(|c| (1.0 - 3.0 * c.abs()).abs());
        let da = r.x.max(r.y);
        let db = r.y.max(r.z);
        let dc = r.z.max(r.x);
        let c = (da.min(db.min(dc)) - 1.0) / s;
        d = d.max(c);
        trap = trap.min(a.norm());
    }
    (d, normalize_trap(trap, 1.7))
}

fn de_sierpinski(p: V3, iters: u32) -> (f64, f64) {
    let scale = 2.0;
    let offset = V3::new(1.0, 1.0, 1.0);
    let mut z = p;
    let mut trap = f64::INFINITY;
    for _ in 0..iters {
        if z.x + z.y < 0.0 {
            z = V3::new(-z.y, -z.x, z.z);
        }
        if z.x + z.z < 0.0 {
            z = V3::new(-z.z, z.y, -z.x);
        }
        if z.y + z.z < 0.0 {
            z = V3::new(z.x, -z.z, -z.y);
        }
        z = z * scale - offset * (scale - 1.0);
        trap = trap.min(z.norm());
    }
    (z.norm() * scale.powi(-(iters as i32)), normalize_trap(trap, 3.0))
}

fn de_quat_julia(p: V3, c: [f64; 4], iters: u32) -> (f64, f64) {
    let mut z = [p.x, p.y, p.z, 0.0];
    let mut dz = [1.0, 0.0, 0.0, 0.0];
    let mut trap = f64::INFINITY;
    let mut r = 0.0;
    for _ in 0..iters {
        r = qnorm(z);
        if r > 4.0 {
            break;
        }
        trap = trap.min(r);
        dz = qscale(qmul(z, dz), 2.0); // dz = 2·z·dz
        z = qadd(qmul(z, z), c); // z = z² + c
    }
    let dzn = qnorm(dz).max(1e-9);
    (0.5 * r.max(1e-9).ln() * r / dzn, normalize_trap(trap, 4.0))
}

/// Signed distance to an axis-aligned box of half-extent `b`.
fn box_de(p: V3, b: V3) -> f64 {
    let q = p.map(|c| c.abs()) - b;
    q.map(|c| c.max(0.0)).norm() + q.x.max(q.y.max(q.z)).min(0.0)
}

fn normalize_trap(trap: f64, span: f64) -> f64 {
    if trap.is_finite() {
        (trap / span).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ── Quaternion helpers (a, b, c, d) ──────────────────────────────────────────
fn qmul(x: [f64; 4], y: [f64; 4]) -> [f64; 4] {
    [
        x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
        x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
        x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
        x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
    ]
}
fn qadd(x: [f64; 4], y: [f64; 4]) -> [f64; 4] {
    [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]]
}
fn qscale(x: [f64; 4], s: f64) -> [f64; 4] {
    [x[0] * s, x[1] * s, x[2] * s, x[3] * s]
}
fn qnorm(x: [f64; 4]) -> f64 {
    (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt()
}

/// Dispatch to the shape's DE, returning `(distance, orbit_trap)`.
fn de(shape: Shape, p: V3, spec: &RaymarchSpec) -> (f64, f64) {
    match shape {
        Shape::Mandelbulb => de_mandelbulb(p, spec.power, spec.iterations),
        Shape::Mandelbox => de_mandelbox(p, spec.box_scale, spec.iterations),
        Shape::Menger => de_menger(p, spec.iterations),
        Shape::Sierpinski3d => de_sierpinski(p, spec.iterations),
        Shape::QuatJulia => de_quat_julia(p, spec.quat_c, spec.iterations),
    }
}

/// Surface normal by central differences of the DE.
fn normal(shape: Shape, p: V3, spec: &RaymarchSpec) -> V3 {
    let e = spec.epsilon.max(1e-5);
    let dx = V3::new(e, 0.0, 0.0);
    let dy = V3::new(0.0, e, 0.0);
    let dz = V3::new(0.0, 0.0, e);
    let n = V3::new(
        de(shape, p + dx, spec).0 - de(shape, p - dx, spec).0,
        de(shape, p + dy, spec).0 - de(shape, p - dy, spec).0,
        de(shape, p + dz, spec).0 - de(shape, p - dz, spec).0,
    );
    let len = n.norm();
    if len > 1e-12 { n / len } else { V3::new(0.0, 1.0, 0.0) }
}

/// Cheap ambient occlusion by sampling the DE along the normal.
fn ambient_occlusion(shape: Shape, p: V3, n: V3, spec: &RaymarchSpec) -> f64 {
    let mut occ = 0.0;
    let mut w = 1.0;
    for i in 1..=5 {
        let dist = 0.012 * i as f64;
        let d = de(shape, p + n * dist, spec).0;
        occ += (dist - d).max(0.0) * w;
        w *= 0.7;
    }
    (1.0 - occ * 1.5).clamp(0.0, 1.0)
}

/// Render the raymarched fractal to a packed `RGB8` buffer.
pub fn render(spec: &FractalSpec, palette: &Palette, prog: ProgressFn) -> Result<Vec<u8>> {
    let rm = &spec.raymarch;
    let shape = Shape::parse(&rm.shape)?;
    let (w, h) = (spec.width as usize, spec.height as usize);

    // Orbit camera looking at the origin.
    let (yaw, pitch) = (rm.camera_yaw.to_radians(), rm.camera_pitch.to_radians());
    let ro = V3::new(
        rm.camera_dist * pitch.cos() * yaw.cos(),
        rm.camera_dist * pitch.sin(),
        rm.camera_dist * pitch.cos() * yaw.sin(),
    );
    let forward = (-ro).normalize();
    let world_up = V3::new(0.0, 1.0, 0.0);
    let right = forward.cross(&world_up).normalize();
    let up = right.cross(&forward);
    let tan_fov = (rm.fov.to_radians() * 0.5).tan();
    let aspect = w as f64 / h as f64;
    let light = V3::new(rm.light[0], rm.light[1], rm.light[2]).normalize();
    let interior = palette.interior();

    let done = AtomicU64::new(0);
    let mut buf = vec![0u8; w * h * 3];
    buf.par_chunks_mut(w * 3).enumerate().for_each(|(py, row)| {
        for px in 0..w {
            let ndc_x = (2.0 * (px as f64 + 0.5) / w as f64 - 1.0) * aspect * tan_fov;
            let ndc_y = (1.0 - 2.0 * (py as f64 + 0.5) / h as f64) * tan_fov;
            let rd = (forward + right * ndc_x + up * ndc_y).normalize();

            // Sphere trace.
            let mut t = 0.0;
            let mut hit = false;
            let mut trap = 0.0;
            for _ in 0..rm.max_steps {
                let p = ro + rd * t;
                let (d, tr) = de(shape, p, rm);
                if d < rm.epsilon {
                    hit = true;
                    trap = tr;
                    break;
                }
                t += d;
                if t > rm.max_dist {
                    break;
                }
            }

            let rgb = if hit {
                let p = ro + rd * t;
                let n = normal(shape, p, rm);
                let diff = n.dot(&light).max(0.0);
                let view = (ro - p).normalize();
                let refl = light - n * 2.0 * light.dot(&n); // reflect(-light) about n
                let spec_term = view.dot(&refl).max(0.0).powf(28.0);
                let ao = if rm.ao { ambient_occlusion(shape, p, n, rm) } else { 1.0 };
                // Fog by depth for cueing.
                let fog = (1.0 - (t / rm.max_dist)).clamp(0.0, 1.0);
                let base = palette.sample(trap);
                let shade = (0.12 + 0.88 * diff * ao) * fog;
                let mut out = [0u8; 3];
                for k in 0..3 {
                    let v = base[k] as f64 * shade + 255.0 * spec_term * 0.35 * ao;
                    out[k] = v.round().clamp(0.0, 255.0) as u8;
                }
                out
            } else {
                interior
            };
            row[px * 3..px * 3 + 3].copy_from_slice(&rgb);
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        prog(d, h as u64);
    });
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::{FractalKind, PaletteSpec, RaymarchSpec};

    fn spec_for(shape: &str) -> FractalSpec {
        FractalSpec {
            kind: FractalKind::Raymarch,
            width: 48,
            height: 48,
            raymarch: RaymarchSpec { shape: shape.into(), max_steps: 80, ..RaymarchSpec::default() },
            ..FractalSpec::default()
        }
    }

    #[test]
    fn all_shapes_render_and_hit_the_surface() {
        let pal = Palette::from_spec(&PaletteSpec::default()).unwrap();
        for s in ["mandelbulb", "mandelbox", "menger", "sierpinski3d", "quat-julia"] {
            let spec = spec_for(s);
            let a = render(&spec, &pal, &|_, _| {}).unwrap();
            let b = render(&spec, &pal, &|_, _| {}).unwrap();
            assert_eq!(a, b, "{s} not deterministic");
            assert_eq!(a.len(), 48 * 48 * 3);
            // The fractal is on-screen: some non-background pixels exist.
            assert!(a.chunks(3).any(|px| px != &a[0..3]), "{s} rendered empty");
        }
    }

    #[test]
    fn unknown_shape_errors() {
        assert!(Shape::parse("teapot").is_err());
    }

    #[test]
    fn box_de_is_signed() {
        assert!(box_de(V3::new(0.0, 0.0, 0.0), V3::new(1.0, 1.0, 1.0)) < 0.0); // inside
        assert!(box_de(V3::new(2.0, 0.0, 0.0), V3::new(1.0, 1.0, 1.0)) > 0.0); // outside
    }

    #[test]
    fn quaternion_mul_identity() {
        let q = [0.5, -0.3, 0.2, 0.1];
        let id = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(qmul(q, id), q);
    }
}
