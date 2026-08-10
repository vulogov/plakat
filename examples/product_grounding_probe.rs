//! G0.1 (ROADMAP PRODUCT-1) — the one novel weight-free algorithm: **grounding** a subject with a
//! contact shadow + floor reflection derived purely from its **alpha matte**, so a product-shot subject
//! sits on the ground instead of floating. No model, no GPU — this is what P1 builds on.
//!
//! Given a subject cutout (RGBA), a ground line, and a key-light direction it must:
//!   (a) cast a **contact shadow** on the floor — the alpha projected to the ground, offset away from the
//!       light, blurred into a soft penumbra, densest at the contact line and fading away;
//!   (b) drop a **floor reflection** — the subject flipped about the ground line, foreshortened by the
//!       camera, fading with distance;
//!   (c) composite sweep ← reflection ← shadow ← subject.
//! It renders BOTH shadow models — a **soft contact** pool and a **perspective cast** rake — side by side
//! so G0 can settle RFC Q2. Pure/weight-free:
//!
//!   cargo run --release --example product_grounding_probe
//!
//! Exit PASS: both shadows anchored at the base (darkest at contact, softer away), the reflection aligned
//! to the foot-line, no bright/dark halo around the cutout in the sweep, and the subject left untouched.

use image::{Rgb, RgbImage, Rgba, RgbaImage};

const W: usize = 700;
const H: usize = 800;

// -------- synthetic subject: a rounded "bottle" cutout, alpha = 255 inside --------
fn make_subject(ground_y: usize) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(W as u32, H as u32, Rgba([0, 0, 0, 0]));
    let cx = W as f32 / 2.0;
    let body_w = 150.0;
    let body_top = ground_y as f32 - 380.0;
    let neck_w = 54.0;
    let neck_top = body_top - 90.0;
    let cap_top = neck_top - 34.0;
    let corner = 42.0;
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32, y as f32);
            let mut inside = false;
            // body: rounded rectangle
            if fy >= body_top && fy <= ground_y as f32 && (fx - cx).abs() <= body_w / 2.0 {
                let dx = (fx - cx).abs() - (body_w / 2.0 - corner);
                let dy_top = body_top + corner - fy;
                if dx <= 0.0 || dy_top <= 0.0 || (dx * dx + dy_top * dy_top).sqrt() <= corner {
                    inside = true;
                }
            }
            // neck
            if fy >= neck_top && fy < body_top && (fx - cx).abs() <= neck_w / 2.0 {
                inside = true;
            }
            // cap
            if fy >= cap_top && fy < neck_top && (fx - cx).abs() <= neck_w / 2.0 + 6.0 {
                inside = true;
            }
            if inside {
                // a two-tone product so "subject intact" is checkable: teal body, dark cap.
                let col = if fy < neck_top { [40, 40, 46] } else { [26, 140, 150] };
                img.put_pixel(x as u32, y as u32, Rgba([col[0], col[1], col[2], 255]));
            }
        }
    }
    img
}

fn alpha_f32(img: &RgbaImage) -> Vec<f32> {
    img.pixels().map(|p| p.0[3] as f32 / 255.0).collect()
}

// -------- separable box blur on an f32 buffer (repeat for a gaussian-ish penumbra) --------
fn box_blur(src: &[f32], r: usize, passes: usize) -> Vec<f32> {
    let mut buf = src.to_vec();
    for _ in 0..passes {
        // horizontal
        let mut tmp = vec![0.0f32; W * H];
        for y in 0..H {
            for x in 0..W {
                let (mut acc, mut n) = (0.0, 0.0);
                for dx in x.saturating_sub(r)..=(x + r).min(W - 1) {
                    acc += buf[y * W + dx];
                    n += 1.0;
                }
                tmp[y * W + x] = acc / n;
            }
        }
        // vertical
        for y in 0..H {
            for x in 0..W {
                let (mut acc, mut n) = (0.0, 0.0);
                for dy in y.saturating_sub(r)..=(y + r).min(H - 1) {
                    acc += tmp[dy * W + x];
                    n += 1.0;
                }
                buf[y * W + x] = acc / n;
            }
        }
    }
    buf
}

/// Project the subject alpha onto the ground plane to build a shadow buffer. `kx`/`ky` push higher parts
/// of the subject away from the light + down the floor (perspective); higher parts fade. Anchored at the
/// contact line (height-above-ground = 0 → lands right at the base). Then blurred + scaled to `opacity`.
fn project_shadow(alpha: &[f32], ground_y: usize, kx: f32, ky: f32, blur_r: usize, passes: usize, opacity: f32) -> Vec<f32> {
    let mut acc = vec![0.0f32; W * H];
    for y in 0..ground_y.min(H) {
        for x in 0..W {
            let a = alpha[y * W + x];
            if a <= 0.01 {
                continue;
            }
            let d = (ground_y as f32 - y as f32).max(0.0); // height above ground
            let gx = x as f32 + kx * d;
            let gy = ground_y as f32 + ky * d;
            let fade = (1.0 - d / 460.0 * 0.6).clamp(0.0, 1.0); // higher parts cast fainter
            let (ix, iy) = (gx.round() as isize, gy.round() as isize);
            if ix >= 0 && (ix as usize) < W && iy >= 0 && (iy as usize) < H {
                acc[iy as usize * W + ix as usize] += a * fade;
            }
        }
    }
    let blurred = box_blur(&acc, blur_r, passes);
    // A floor shadow lives ON the floor: clamp to the ground plane so the blur can't bleed a dark halo
    // up into the sweep beside the subject (above the contact line the subject occludes it anyway).
    blurred
        .iter()
        .enumerate()
        .map(|(i, v)| if i / W < ground_y { 0.0 } else { (v * opacity).clamp(0.0, 1.0) })
        .collect()
}

/// Floor reflection: flip the subject about `ground_y`, foreshorten by `squash` (camera), fade over
/// `falloff` of the subject height. Returns an RGBA layer to composite under the subject.
fn reflection(subject: &RgbaImage, ground_y: usize, squash: f32, falloff: f32) -> RgbaImage {
    let mut r = RgbaImage::from_pixel(W as u32, H as u32, Rgba([0, 0, 0, 0]));
    for y in ground_y..H {
        let below = (y - ground_y) as f32; // distance below the foot-line
        let src_y = ground_y as f32 - below / squash; // mirror + foreshorten
        if src_y < 0.0 {
            continue;
        }
        let sy = src_y.round() as u32;
        if sy >= H as u32 {
            continue;
        }
        let fade = (1.0 - below / (falloff * 380.0)).clamp(0.0, 1.0);
        for x in 0..W as u32 {
            let p = subject.get_pixel(x, sy);
            if p.0[3] > 0 {
                let a = (p.0[3] as f32 / 255.0 * fade * 0.5 * 255.0) as u8; // reflections are dim
                r.put_pixel(x, y as u32, Rgba([p.0[0], p.0[1], p.0[2], a]));
            }
        }
    }
    r
}

// -------- compositing --------
fn over(dst: &mut RgbImage, src: &RgbaImage) {
    for (x, y, p) in src.enumerate_pixels() {
        let a = p.0[3] as f32 / 255.0;
        if a <= 0.0 {
            continue;
        }
        let d = dst.get_pixel(x, y).0;
        let blend = |i: usize| (p.0[i] as f32 * a + d[i] as f32 * (1.0 - a)).round() as u8;
        dst.put_pixel(x, y, Rgb([blend(0), blend(1), blend(2)]));
    }
}

fn darken(dst: &mut RgbImage, shadow: &[f32], max_dark: f32) {
    for y in 0..H {
        for x in 0..W {
            let s = shadow[y * W + x];
            if s <= 0.0 {
                continue;
            }
            let k = 1.0 - s * max_dark;
            let d = dst.get_pixel(x as u32, y as u32).0;
            dst.put_pixel(x as u32, y as u32, Rgb([(d[0] as f32 * k) as u8, (d[1] as f32 * k) as u8, (d[2] as f32 * k) as u8]));
        }
    }
}

/// A neutral studio sweep: white at the top easing to a light grey toward the floor.
fn sweep() -> RgbImage {
    let mut img = RgbImage::new(W as u32, H as u32);
    for y in 0..H {
        let t = y as f32 / H as f32;
        let v = (255.0 - t * 28.0) as u8;
        for x in 0..W {
            img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
        }
    }
    img
}

struct Composite {
    out: RgbImage,
    shadow: Vec<f32>,
    sweep: RgbImage,
}

fn build(subject: &RgbaImage, alpha: &[f32], ground_y: usize, kx: f32, ky: f32, blur_r: usize, opacity: f32) -> Composite {
    let base = sweep();
    let mut out = base.clone();
    let refl = reflection(subject, ground_y, 2.0, 0.55); // eye-level squash
    over(&mut out, &refl);
    let shadow = project_shadow(alpha, ground_y, kx, ky, blur_r, 3, opacity);
    darken(&mut out, &shadow, 0.55);
    over(&mut out, subject);
    Composite { out, shadow, sweep: base }
}

fn band_mean(buf: &[f32], y0: usize, y1: usize) -> f32 {
    let (mut acc, mut n) = (0.0, 0.0);
    for y in y0..y1.min(H) {
        for x in 0..W {
            acc += buf[y * W + x];
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        0.0
    }
}

fn main() {
    let ground_y = (H as f32 * 0.72) as usize;
    let subject = make_subject(ground_y);
    let alpha = alpha_f32(&subject);

    // Variant A — soft contact pool (light from top-left → gentle offset right, wide blur).
    let a = build(&subject, &alpha, ground_y, 0.10, 0.14, 22, 0.9);
    // Variant B — perspective cast rake (long horizontal skew, tighter blur).
    let b = build(&subject, &alpha, ground_y, 0.55, 0.10, 10, 0.85);

    // ---- measurements ----
    let near = (ground_y, ground_y + 10);
    let far = (ground_y + 90, ground_y + 110);
    let anchored_a = band_mean(&a.shadow, near.0, near.1) > band_mean(&a.shadow, far.0, far.1) * 1.5;
    let anchored_b = band_mean(&b.shadow, near.0, near.1) > band_mean(&b.shadow, far.0, far.1) * 1.2;
    // B rakes to the side: its shadow mass sits right of the subject centre more than A's.
    let mass_x = |buf: &[f32]| -> f32 {
        let (mut sx, mut s) = (0.0f32, 0.0f32);
        for y in ground_y..H {
            for x in 0..W {
                let v = buf[y * W + x];
                sx += v * x as f32;
                s += v;
            }
        }
        if s > 0.0 {
            sx / s
        } else {
            W as f32 / 2.0
        }
    };
    let rakes_side = mass_x(&b.shadow) > mass_x(&a.shadow) + 8.0;

    // reflection aligned: first reflected row below the foot-line is right at ground_y.
    let refl = reflection(&subject, ground_y, 2.0, 0.55);
    let refl_top = (ground_y..H).find(|&y| (0..W as u32).any(|x| refl.get_pixel(x, y as u32).0[3] > 0));
    let refl_aligned = refl_top.map(|y| y <= ground_y + 3).unwrap_or(false);

    // no halo: a ring just outside the subject silhouette, ABOVE the ground (pure sweep), is untouched.
    let mut halo_max = 0.0f32;
    for y in 40..(ground_y - 20) {
        for x in 4..(W - 4) {
            let i = y * W + x;
            let edge = alpha[i] < 0.5 && (alpha[i - 1] > 0.5 || alpha[i + 1] > 0.5 || alpha[i - W] > 0.5 || alpha[i + W] > 0.5);
            if edge {
                let o = a.out.get_pixel(x as u32, y as u32).0[0] as f32;
                let s = a.sweep.get_pixel(x as u32, y as u32).0[0] as f32;
                halo_max = halo_max.max((o - s).abs());
            }
        }
    }
    let no_halo = halo_max <= 4.0;

    // subject intact: every fully-opaque subject pixel survives the composite unchanged.
    let mut intact = true;
    for (x, y, p) in subject.enumerate_pixels() {
        if p.0[3] == 255 {
            let o = a.out.get_pixel(x, y).0;
            if o != [p.0[0], p.0[1], p.0[2]] {
                intact = false;
                break;
            }
        }
    }

    // side-by-side preview.
    let mut sheet = RgbImage::new((W * 2 + 20) as u32, H as u32);
    for (x, y, p) in a.out.enumerate_pixels() {
        sheet.put_pixel(x, y, *p);
    }
    for (x, y, p) in b.out.enumerate_pixels() {
        sheet.put_pixel(x + W as u32 + 20, y, *p);
    }
    let _ = sheet.save("/tmp/product_grounding_probe.png");

    println!("G0.1 — product grounding (contact shadow + floor reflection from the alpha)");
    println!("  canvas {W}x{H}, ground_y {ground_y}");
    println!("  variant A (soft contact): anchored {anchored_a}");
    println!("  variant B (perspective cast): anchored {anchored_b}, rakes to side {rakes_side}");
    println!("  reflection aligned to foot-line: {refl_aligned} (top row {:?}, ground {ground_y})", refl_top);
    println!("  no halo around cutout: {no_halo} (max Δ {halo_max:.1})");
    println!("  subject intact: {intact}");
    println!("  (saved /tmp/product_grounding_probe.png — left: soft contact, right: perspective cast)");

    let pass = anchored_a && anchored_b && rakes_side && refl_aligned && no_halo && intact;
    println!("\n{}", if pass {
        "PASS — grounding (shadow + reflection) holds from the alpha alone → P1 uses it. Q2: both shadow models work; soft-contact is the default, perspective-cast is a `camera`/`shadow: hard` option."
    } else {
        "FAIL — revisit the grounding algorithm before P1."
    });
}
