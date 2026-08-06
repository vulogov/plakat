//! Pure-Rust PBR preview (RFC TEXTURE-1 §11). Shades a lit sphere (or plane) that samples the **tiled**
//! material through its derived normal with a Cook-Torrance-lite BRDF (one directional + ambient) — so
//! you *see* the material lit, deterministically, without a GPU or a 3D tool. Not a renderer; a sanity
//! view.

use crate::texture::derive::Material;
use image::{GrayImage, Rgb, RgbImage};
use std::f32::consts::{PI, TAU};

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn mul(a: V3, s: f32) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
fn mulv(a: V3, b: V3) -> V3 { [a[0] * b[0], a[1] * b[1], a[2] * b[2]] }
fn dot(a: V3, b: V3) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: V3, b: V3) -> V3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn norm(a: V3) -> V3 { let m = dot(a, a).sqrt().max(1e-9); mul(a, 1.0 / m) }
fn mix(a: V3, b: V3, t: f32) -> V3 { [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t] }

/// The preview geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Sphere,
    Plane,
}

impl Shape {
    pub fn parse(s: &str) -> Shape {
        if s.eq_ignore_ascii_case("plane") { Shape::Plane } else { Shape::Sphere }
    }
}

/// Wrapped bilinear sample of an RGB map at UV in `[0,1)` (tiling).
fn sample_rgb(img: &RgbImage, u: f32, v: f32) -> V3 {
    let (w, h) = img.dimensions();
    let fx = (u.rem_euclid(1.0)) * w as f32 - 0.5;
    let fy = (v.rem_euclid(1.0)) * h as f32 - 0.5;
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let px = |x: i32, y: i32| {
        let xx = ((x % w as i32 + w as i32) % w as i32) as u32;
        let yy = ((y % h as i32 + h as i32) % h as i32) as u32;
        let p = img.get_pixel(xx, yy).0;
        [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]
    };
    let a = mix(px(x0, y0), px(x0 + 1, y0), tx);
    let b = mix(px(x0, y0 + 1), px(x0 + 1, y0 + 1), tx);
    mix(a, b, ty)
}

fn sample_gray(img: &GrayImage, u: f32, v: f32) -> f32 {
    let (w, h) = img.dimensions();
    let x = ((u.rem_euclid(1.0)) * w as f32) as u32 % w;
    let y = ((v.rem_euclid(1.0)) * h as f32) as u32 % h;
    img.get_pixel(x, y).0[0] as f32 / 255.0
}

/// Shade one surface point. `ng` = geometric normal, `(t,b)` its tangent frame, `uv` the material coord.
fn shade(m: &Material, ng: V3, t: V3, bt: V3, u: f32, v: f32) -> V3 {
    let albedo = {
        let a = sample_rgb(&m.albedo, u, v);
        [a[0].powf(2.2), a[1].powf(2.2), a[2].powf(2.2)] // sRGB → linear
    };
    let nts = sample_rgb(&m.normal, u, v);
    let nts = [nts[0] * 2.0 - 1.0, nts[1] * 2.0 - 1.0, nts[2] * 2.0 - 1.0];
    let n = norm(add(add(mul(t, nts[0]), mul(bt, nts[1])), mul(ng, nts[2].max(0.05))));
    let rough = sample_gray(&m.roughness, u, v).clamp(0.05, 1.0);
    let metal = sample_gray(&m.metallic, u, v);
    let ao = sample_gray(&m.ao, u, v);

    let l = norm([-0.35, 0.55, 0.75]);
    let view = [0.0, 0.0, 1.0];
    let h = norm(add(l, view));
    let ndl = dot(n, l).max(0.0);
    let ndv = dot(n, view).max(1e-3);
    let ndh = dot(n, h).max(0.0);
    let hdv = dot(h, view).max(0.0);

    // C1: anisotropy — a directional roughness so the highlight ELONGATES along the grain (a brushed
    // metal is sharp along its grooves, blurred across). Approximation: split roughness by the half-
    // vector's alignment with the grain in the tangent plane. Guarded → isotropic path is untouched.
    let rough = if let Some(aniso) = &m.anisotropy {
        let a = sample_rgb(aniso, u, v);
        let strength = a[2];
        if strength > 0.004 {
            let gd = norm([a[0] * 2.0 - 1.0, a[1] * 2.0 - 1.0, 0.0]); // grain dir in the (t,bt) basis
            let ht = norm([dot(h, t), dot(h, bt), 0.0]); // half-vector projected to the tangent plane
            let along = dot(gd, ht).abs().clamp(0.0, 1.0); // 1 = along grain, 0 = across
            let sharp = rough * (1.0 - 0.6 * strength);
            let blur = rough * (1.0 + 0.9 * strength);
            (sharp + (blur - sharp) * (1.0 - along * along)).clamp(0.05, 1.0)
        } else {
            rough
        }
    } else {
        rough
    };

    // GGX-lite Cook-Torrance.
    let a2 = (rough * rough).powi(2);
    let d = a2 / (PI * ((ndh * ndh * (a2 - 1.0) + 1.0)).powi(2)).max(1e-6);
    let k = (rough + 1.0).powi(2) / 8.0;
    let g = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    let f0 = mix([0.04, 0.04, 0.04], albedo, metal);
    let f = add(f0, mul(add([1.0, 1.0, 1.0], mul(f0, -1.0)), (1.0 - hdv).powi(5)));
    let spec = mul(f, d * g / (4.0 * ndl * ndv + 1e-4));
    let kd = mul(add([1.0, 1.0, 1.0], mul(f, -1.0)), 1.0 - metal);
    let diffuse = mulv(kd, mul(albedo, 1.0 / PI));

    let light = [1.15, 1.12, 1.05];
    let lo = mulv(add(diffuse, spec), mul(light, ndl));
    let ambient = mulv(mul(albedo, 0.28 * ao), [0.9, 0.95, 1.0]);
    add(ambient, lo)
}

fn tonemap(c: V3) -> Rgb<u8> {
    // Reinhard + gamma 2.2.
    let f = |x: f32| ((x / (x + 1.0)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
    Rgb([f(c[0]), f(c[1]), f(c[2])])
}

/// Render a `size`×`size` lit preview of the material.
pub fn render(m: &Material, shape: Shape, size: u32) -> RgbImage {
    let mut out = RgbImage::new(size, size);
    let tiles = 3.0; // how many times the material repeats across the preview
    for j in 0..size {
        for i in 0..size {
            // background: a soft vertical gradient.
            let bgv = 0.16 + 0.10 * (j as f32 / size as f32);
            let mut color = [bgv, bgv, bgv * 1.05];

            let px = (i as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let py = 1.0 - (j as f32 + 0.5) / size as f32 * 2.0;
            match shape {
                Shape::Sphere => {
                    let d2 = px * px + py * py;
                    if d2 <= 1.0 {
                        let ng = [px, py, (1.0 - d2).sqrt()];
                        // spherical UV (a few tiles) + a tangent frame.
                        let u = (0.5 + ng[0].atan2(ng[2]) / TAU) * tiles;
                        let v = (0.5 - (ng[1]).asin() / PI) * tiles;
                        let t = norm(cross([0.0, 1.0, 0.0], ng));
                        let bt = cross(ng, t);
                        color = shade(m, ng, t, bt, u, v);
                    }
                }
                Shape::Plane => {
                    // a plane tilted back, filling the frame.
                    let ng = norm([0.0, 0.45, 0.9]);
                    let u = (px * 0.5 + 0.5) * tiles;
                    let v = (py * 0.5 + 0.5) * tiles / 0.6; // foreshorten
                    let t = [1.0, 0.0, 0.0];
                    let bt = cross(ng, t);
                    color = shade(m, ng, t, bt, u, v);
                }
            }
            out.put_pixel(i, j, tonemap(color));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::compile::ChannelSource;

    #[test]
    fn preview_is_deterministic_and_lit() {
        let albedo = RgbImage::from_fn(32, 32, |x, y| Rgb([(x * 8) as u8, (y * 8) as u8, 120]));
        let m = Material::derive(albedo, None, 1.0, true, 1.0, &ChannelSource::FromAlbedo, &ChannelSource::Scalar(0.0));
        let a = render(&m, Shape::Sphere, 96);
        let b = render(&m, Shape::Sphere, 96);
        assert_eq!(a.as_raw(), b.as_raw(), "deterministic");
        // The sphere region is lit (has bright + dark pixels — not a flat fill).
        let lumas: Vec<u8> = a.pixels().map(|p| p.0[0]).collect();
        let (mn, mx) = (*lumas.iter().min().unwrap(), *lumas.iter().max().unwrap());
        assert!(mx - mn > 40, "preview should have lit shading range ({mn}..{mx})");
    }
}
