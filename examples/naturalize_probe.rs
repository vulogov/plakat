//! G0.1 (ROADMAP QUALITY-1) — the one novel weight-free algorithm: **analog naturalize**. Turn a
//! too-clean, over-saturated digital image into one that reads as physical media — film grain + chromatic
//! aberration + vignette + bloom + a desaturating film grade — *without destroying the picture*. No model,
//! no GPU; this is what P1 builds on.
//!
//! It runs the pass on a synthetic "AI-clean" image (flat gradients + a hard edge + a uniform high-freq
//! "AI-painterly" texture tile, all over-saturated) and measures an **AI-tell delta**:
//!   - mean saturation DROPS (kills the #1 tell),
//!   - high-frequency variance RISES (grain breaks the uniform texture),
//!   - a radial channel offset APPEARS at off-centre edges (chromatic aberration),
//!   - the corners DARKEN (vignette),
//!   - while STRUCTURE is preserved (luminance correlation to the input stays high — we degraded the
//!     fingerprint, not the picture).
//!
//!   cargo run --release --example naturalize_probe
//!
//! Exit PASS when all five hold.

use image::{Rgb, RgbImage};

const W: usize = 512;
const H: usize = 512;

// ---------- a synthetic "AI-clean, over-saturated" input ----------
fn make_input() -> RgbImage {
    let mut img = RgbImage::new(W as u32, H as u32);
    for y in 0..H {
        for x in 0..W {
            // an over-saturated smooth gradient (the "dissolving background").
            let t = x as f32 / W as f32;
            let mut r = (20.0 + t * 40.0) as i32;
            let mut g = (40.0 + (1.0 - t) * 120.0) as i32;
            let mut b = (200.0 - t * 60.0) as i32;
            // a uniform, too-regular high-freq tile (stands in for "AI-painterly repeating texture").
            if x < W / 2 && y < H / 2 {
                let checker = ((x / 3 + y / 3) % 2) as i32 * 26 - 13;
                g += checker;
                b += checker;
            }
            // a hard vertical edge near the TOP-RIGHT corner (large radius) so chromatic aberration is
            // measurable there. A UNIFORM grey surround (not the gradient) → a single clean R/B crossing
            // at x=460, so any post-pass R/B separation is the aberration alone.
            if (60..160).contains(&y) && x >= 400 {
                let v = if x >= 460 { 235 } else { 40 };
                r = v;
                g = v;
                b = v;
            }
            img.put_pixel(x as u32, y as u32, Rgb([r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8]));
        }
    }
    img
}

// ---------- helpers ----------
fn lum(p: &Rgb<u8>) -> f32 {
    0.299 * p.0[0] as f32 + 0.587 * p.0[1] as f32 + 0.114 * p.0[2] as f32
}

fn saturation(p: &Rgb<u8>) -> f32 {
    let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    if mx <= 0.0 {
        0.0
    } else {
        (mx - mn) / mx
    }
}

fn mean_saturation(img: &RgbImage) -> f32 {
    let s: f32 = img.pixels().map(saturation).sum();
    s / (W * H) as f32
}

/// A cheap high-frequency energy over a **flat** region (smooth gradient, no edge/texture): mean squared
/// (lum − 3×3-box-blurred lum). Grain raises it; the region is chosen so nothing else does.
fn highpass_energy(img: &RgbImage) -> f32 {
    let l: Vec<f32> = img.pixels().map(lum).collect();
    let mut e = 0.0f32;
    let mut n = 0.0f32;
    for y in 350..480 {
        for x in 300..500 {
            let mut acc = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    acc += l[((y as i32 + dy) as usize) * W + (x as i32 + dx) as usize];
                }
            }
            let blur = acc / 9.0;
            let d = l[y * W + x] - blur;
            e += d * d;
            n += 1.0;
        }
    }
    e / n
}

fn bilinear(img: &RgbImage, ch: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (W - 1) as f32);
    let y = y.clamp(0.0, (H - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(W as u32 - 1), (y0 + 1).min(H as u32 - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: u32, yy: u32| img.get_pixel(xx, yy).0[ch] as f32;
    let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
    let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Deterministic value noise in [-1,1] from integer coords (no RNG → reproducible).
fn noise(x: usize, y: usize) -> f32 {
    let mut h = (x as u32).wrapping_mul(374761393).wrapping_add((y as u32).wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

// ---------- the naturalize pass (the algorithm P1 will build on) ----------
struct Params {
    grain: f32,
    aberration: f32,
    vignette: f32,
    bloom: f32,
    desaturate: f32,
    warm: f32,
}

fn naturalize(src: &RgbImage, p: &Params) -> RgbImage {
    let (cx, cy) = (W as f32 / 2.0, H as f32 / 2.0);
    let maxr = (cx * cx + cy * cy).sqrt();
    // 1. chromatic aberration — sample R outward / B inward along the radius, growing with r².
    let mut out = RgbImage::new(W as u32, H as u32);
    for y in 0..H {
        for x in 0..W {
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr; // 0..1
            let shift = p.aberration * r * r * 10.0; // px at the corner
            let (ux, uy) = if r > 1e-4 { (dx / (r * maxr), dy / (r * maxr)) } else { (0.0, 0.0) };
            let rr = bilinear(src, 0, x as f32 + ux * shift, y as f32 + uy * shift);
            let gg = src.get_pixel(x as u32, y as u32).0[1] as f32;
            let bb = bilinear(src, 2, x as f32 - ux * shift, y as f32 - uy * shift);
            out.put_pixel(x as u32, y as u32, Rgb([rr as u8, gg as u8, bb as u8]));
        }
    }
    // 2. bloom — screen a blurred highlight mask back in.
    if p.bloom > 0.0 {
        let mut hi = vec![0.0f32; W * H];
        for y in 0..H {
            for x in 0..W {
                let l = lum(out.get_pixel(x as u32, y as u32));
                hi[y * W + x] = ((l - 200.0).max(0.0)) / 55.0; // 0..1 for the brightest
            }
        }
        // small box blur of the mask
        let mut blur = vec![0.0f32; W * H];
        let r = 4i32;
        for y in 0..H {
            for x in 0..W {
                let (mut acc, mut n) = (0.0, 0.0);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                        if xx >= 0 && yy >= 0 && (xx as usize) < W && (yy as usize) < H {
                            acc += hi[yy as usize * W + xx as usize];
                            n += 1.0;
                        }
                    }
                }
                blur[y * W + x] = acc / n;
            }
        }
        for y in 0..H {
            for x in 0..W {
                let b = (blur[y * W + x] * p.bloom * 60.0).min(60.0);
                let px = out.get_pixel_mut(x as u32, y as u32);
                for c in 0..3 {
                    px.0[c] = (px.0[c] as f32 + b).min(255.0) as u8;
                }
            }
        }
    }
    // 3. per-pixel grade (desaturate + warm lift) + vignette + grain.
    for y in 0..H {
        for x in 0..W {
            let px = out.get_pixel_mut(x as u32, y as u32);
            let mut c = [px.0[0] as f32, px.0[1] as f32, px.0[2] as f32];
            // desaturate toward luminance
            let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
            for v in c.iter_mut() {
                *v = *v * (1.0 - p.desaturate) + l * p.desaturate;
            }
            // warm film lift (adds to R, small to G, subtracts B in shadows)
            let shadow = (1.0 - l / 255.0).clamp(0.0, 1.0);
            c[0] += p.warm * 14.0 * shadow;
            c[2] -= p.warm * 10.0 * shadow;
            // vignette
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let r = (dx * dx + dy * dy).sqrt() / maxr;
            let vig = 1.0 - p.vignette * r * r;
            for v in c.iter_mut() {
                *v *= vig;
            }
            // luminance-weighted film grain (noisier in the mids/shadows)
            let g_amp = p.grain * 22.0 * (0.4 + 0.6 * shadow);
            let n = noise(x, y);
            c[0] += n * g_amp;
            c[1] += noise(x + 7, y + 3) * g_amp * 0.9;
            c[2] += noise(x + 13, y + 11) * g_amp * 0.9;
            px.0 = [c[0].clamp(0.0, 255.0) as u8, c[1].clamp(0.0, 255.0) as u8, c[2].clamp(0.0, 255.0) as u8];
        }
    }
    out
}

/// The x of the channel's 50% crossing along a row, over [x0,x1] (dark→bright edge).
fn edge_x(img: &RgbImage, ch: usize, row: usize, x0: usize, x1: usize) -> f32 {
    let mut lo = 255.0f32;
    let mut hi = 0.0f32;
    for x in x0..x1 {
        let v = img.get_pixel(x as u32, row as u32).0[ch] as f32;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let mid = (lo + hi) / 2.0;
    for x in x0..x1 - 1 {
        let a = img.get_pixel(x as u32, row as u32).0[ch] as f32;
        let b = img.get_pixel((x + 1) as u32, row as u32).0[ch] as f32;
        if (a - mid) * (b - mid) <= 0.0 && (b - a).abs() > 1.0 {
            return x as f32 + (mid - a) / (b - a);
        }
    }
    (x0 + x1) as f32 / 2.0
}

fn corr_lum(a: &RgbImage, b: &RgbImage) -> f32 {
    let (la, lb): (Vec<f32>, Vec<f32>) = (a.pixels().map(lum).collect(), b.pixels().map(lum).collect());
    let (ma, mb) = (la.iter().sum::<f32>() / la.len() as f32, lb.iter().sum::<f32>() / lb.len() as f32);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..la.len() {
        let (x, y) = (la[i] - ma, lb[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-6)
}

fn corner_lum(img: &RgbImage) -> f32 {
    let mut s = 0.0;
    let mut n = 0.0;
    for y in 0..40 {
        for x in 0..40 {
            s += lum(img.get_pixel(x, y));
            n += 1.0;
        }
    }
    s / n
}
fn center_lum(img: &RgbImage) -> f32 {
    let mut s = 0.0;
    let mut n = 0.0;
    for y in H / 2 - 20..H / 2 + 20 {
        for x in W / 2 - 20..W / 2 + 20 {
            s += lum(img.get_pixel(x as u32, y as u32));
            n += 1.0;
        }
    }
    s / n
}

fn main() {
    let input = make_input();
    let p = Params { grain: 0.4, aberration: 0.6, vignette: 0.35, bloom: 0.15, desaturate: 0.15, warm: 0.4 };
    let out = naturalize(&input, &p);

    // side-by-side preview
    let mut sheet = RgbImage::new((W * 2 + 16) as u32, H as u32);
    for (x, y, px) in input.enumerate_pixels() {
        sheet.put_pixel(x, y, *px);
    }
    for (x, y, px) in out.enumerate_pixels() {
        sheet.put_pixel(x + W as u32 + 16, y, *px);
    }
    let _ = sheet.save("/tmp/naturalize_probe.png");

    // --- measurements ---
    let sat_before = mean_saturation(&input);
    let sat_after = mean_saturation(&out);
    let sat_drops = sat_after < sat_before - 0.01;

    let hp_before = highpass_energy(&input);
    let hp_after = highpass_energy(&out);
    let grain_rises = hp_after > hp_before * 1.2;

    // aberration: R-edge vs B-edge separation on a row through the near-corner hard edge (~x450, y110).
    let sep_before = (edge_x(&input, 0, 110, 420, 500) - edge_x(&input, 2, 110, 420, 500)).abs();
    let sep_after = (edge_x(&out, 0, 110, 420, 500) - edge_x(&out, 2, 110, 420, 500)).abs();
    let aberration_present = sep_after > sep_before + 0.8;

    let corner_before = corner_lum(&input);
    let corner_after = corner_lum(&out);
    let vignette_present = corner_after < corner_before - 3.0 && corner_after < center_lum(&out);

    let structure = corr_lum(&input, &out);
    let structure_ok = structure > 0.9;

    println!("G0.1 — analog naturalize (film grain + aberration + vignette + bloom + film grade)");
    println!("  saturation:  {sat_before:.3} → {sat_after:.3}   drops {sat_drops}");
    println!("  hi-freq var: {hp_before:.2} → {hp_after:.2}   grain rises {grain_rises}");
    println!("  aberration:  R/B edge sep {sep_before:.2}px → {sep_after:.2}px   present {aberration_present}");
    println!("  vignette:    corner lum {corner_before:.1} → {corner_after:.1}   present {vignette_present}");
    println!("  structure:   lum-corr {structure:.4}   preserved {structure_ok}");
    println!("  (saved /tmp/naturalize_probe.png — left: AI-clean input, right: naturalized)");

    let pass = sat_drops && grain_rises && aberration_present && vignette_present && structure_ok;
    println!("\n{}", if pass {
        "PASS — the analog naturalize pass degrades the AI fingerprint (saturation↓, grain↑, aberration + vignette) while preserving the picture → P1 uses it."
    } else {
        "FAIL — revisit the naturalize algorithm before P1."
    });
}
