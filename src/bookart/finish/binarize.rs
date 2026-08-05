//! Technique binarisers (RFC BOOKART-1 §7.1). The finisher's first stage: a grayscale render → the
//! ornament's ink idiom (clean line / bold woodcut / stipple / engraving / silhouette / halftone). Pure,
//! deterministic, self-contained (a small separable Gaussian — no feature-gated deps, so it builds under
//! `--no-default-features`). Each binariser is `GrayImage → GrayImage` (white paper 255, black ink 0).

use image::{GrayImage, Luma};

/// Otsu global threshold on a luma histogram (also used by the scorecard).
pub fn otsu(g: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for p in g.pixels() {
        hist[p[0] as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 128;
    }
    let sum_all: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let (mut wb, mut sumb, mut best_t, mut best_var) = (0u64, 0.0f64, 128u8, -1.0f64);
    for t in 0..256 {
        wb += hist[t];
        if wb == 0 {
            continue;
        }
        let wf = total - wb;
        if wf == 0 {
            break;
        }
        sumb += t as f64 * hist[t] as f64;
        let mb = sumb / wb as f64;
        let mf = (sum_all - sumb) / wf as f64;
        let var = wb as f64 * wf as f64 * (mb - mf) * (mb - mf);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// A separable Gaussian blur → f32 buffer (edge-clamped). Deterministic.
fn blur(g: &GrayImage, sigma: f32) -> Vec<f32> {
    let (w, h) = (g.width() as i32, g.height() as i32);
    let r = (sigma * 3.0).ceil().max(1.0) as i32;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    for v in &mut k {
        *v /= s;
    }
    let src: Vec<f32> = g.pixels().map(|p| p.0[0] as f32).collect();
    let idx = |x: i32, y: i32| (y * w + x) as usize;
    let mut tmp = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let xx = (x + ki as i32 - r).clamp(0, w - 1);
                acc += kv * src[idx(xx, y)];
            }
            tmp[idx(x, y)] = acc;
        }
    }
    let mut out = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let yy = (y + ki as i32 - r).clamp(0, h - 1);
                acc += kv * tmp[idx(x, yy)];
            }
            out[idx(x, y)] = acc;
        }
    }
    out
}

/// XDoG line extraction — clean dark lines on white, the default for pen/line + cross-hatch.
/// Stretch the tonal range to a 1st/99th-percentile window so a **low-contrast** (faint) render still
/// yields solid edges — the root cause of thin `line` output on soft LoRA renders. A full-range input
/// is ~unchanged, so dense renders don't over-darken.
fn autocontrast(g: &GrayImage) -> GrayImage {
    let mut hist = [0u32; 256];
    for p in g.pixels() {
        hist[p.0[0] as usize] += 1;
    }
    let n = (g.width() * g.height()) as f32;
    let clip = (n * 0.01) as u32;
    let (mut lo, mut acc) = (0usize, 0u32);
    while lo < 255 {
        acc += hist[lo];
        if acc >= clip {
            break;
        }
        lo += 1;
    }
    let (mut hi, mut acc2) = (255usize, 0u32);
    while hi > 0 {
        acc2 += hist[hi];
        if acc2 >= clip {
            break;
        }
        hi -= 1;
    }
    if hi <= lo + 4 {
        return g.clone(); // already flat or full-range — leave it
    }
    let (lo, hi) = (lo as f32, hi as f32);
    let mut out = g.clone();
    for p in out.pixels_mut() {
        p.0[0] = (((p.0[0] as f32 - lo) / (hi - lo)) * 255.0).clamp(0.0, 255.0).round() as u8;
    }
    out
}

/// XDoG line extraction, **contrast-adaptive** and **ink-weight-responsive** (B3). `ink_weight`
/// (default 0.6) biases the DoG threshold `tau` and a darkening gamma: higher weight captures fainter
/// edges and darkens them (bolder line), lower weight thins. At 0.6 the result matches the historical
/// tuning. Autocontrast first, so a faint render reads as clean line instead of sparse specks.
pub fn xdog(g: &GrayImage, ink_weight: f32) -> GrayImage {
    let g = autocontrast(g);
    let (w, h) = (g.width(), g.height());
    let bias = ink_weight - 0.6;
    let (sigma, k) = (0.8f32, 1.6f32);
    let tau = (0.985 + bias * 0.03).clamp(0.95, 0.999);
    let phi = 14.0f32;
    let gamma = (1.0 + bias * 2.0).clamp(0.4, 3.0); // >1 darkens (bolder), <1 lightens
    let b1 = blur(&g, sigma);
    let b2 = blur(&g, sigma * k);
    let mut out = GrayImage::new(w, h);
    for (i, p) in out.pixels_mut().enumerate() {
        let d = (b1[i] - tau * b2[i]) / 255.0; // normalised DoG; edges → negative
        let e = if d >= 0.0 { 1.0 } else { 1.0 + (phi * d).tanh() };
        let e = e.clamp(0.0, 1.0).powf(gamma);
        p.0[0] = (e * 255.0).round() as u8;
    }
    out
}

/// Hard threshold at `t` → 1-bit (≤ t = ink).
fn threshold_at(g: &GrayImage, t: u8) -> GrayImage {
    let mut out = g.clone();
    for p in out.pixels_mut() {
        p.0[0] = if p.0[0] <= t { 0 } else { 255 };
    }
    out
}

fn invert(g: &GrayImage) -> GrayImage {
    let mut out = g.clone();
    for p in out.pixels_mut() {
        p.0[0] = 255 - p.0[0];
    }
    out
}

/// Grow the black (ink) regions by `r` px — a bolder woodcut mass.
fn dilate_ink(g: &GrayImage, r: i32) -> GrayImage {
    let (w, h) = (g.width() as i32, g.height() as i32);
    let mut out = g.clone();
    for y in 0..h {
        for x in 0..w {
            if g.get_pixel(x as u32, y as u32).0[0] != 0 {
                let mut ink = false;
                'scan: for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && xx < w && yy < h && g.get_pixel(xx as u32, yy as u32).0[0] == 0 {
                            ink = true;
                            break 'scan;
                        }
                    }
                }
                if ink {
                    out.put_pixel(x as u32, y as u32, Luma([0]));
                }
            }
        }
    }
    out
}

/// Floyd–Steinberg error diffusion → 1-bit stipple.
fn floyd_steinberg(g: &GrayImage) -> GrayImage {
    let (w, h) = (g.width() as i32, g.height() as i32);
    let mut buf: Vec<f32> = g.pixels().map(|p| p.0[0] as f32).collect();
    let at = |x: i32, y: i32| (y * w + x) as usize;
    let mut out = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let old = buf[at(x, y)];
            let new = if old < 128.0 { 0.0 } else { 255.0 };
            let err = old - new;
            out.put_pixel(x as u32, y as u32, Luma([new as u8]));
            let mut spread = |xx: i32, yy: i32, f: f32| {
                if xx >= 0 && yy >= 0 && xx < w && yy < h {
                    buf[at(xx, yy)] += err * f;
                }
            };
            spread(x + 1, y, 7.0 / 16.0);
            spread(x - 1, y + 1, 3.0 / 16.0);
            spread(x, y + 1, 5.0 / 16.0);
            spread(x + 1, y + 1, 1.0 / 16.0);
        }
    }
    out
}

/// 4×4 Bayer ordered dither → halftone tone (ink-wash).
fn bayer(g: &GrayImage) -> GrayImage {
    const M: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    let mut out = g.clone();
    for (x, y, p) in out.enumerate_pixels_mut() {
        let thr = (M[(y % 4) as usize][(x % 4) as usize] as u32 * 16 + 8) as u8;
        p.0[0] = if p.0[0] <= thr { 0 } else { 255 };
    }
    out
}

/// Dispatch by the lexicon binariser name (`RenderPlan.binariser`). `ink_weight` biases the threshold
/// on threshold-based idioms (heavier weight → more ink).
pub fn binarise(g: &GrayImage, name: &str, ink_weight: f32) -> GrayImage {
    let t = (otsu(g) as f32 + (ink_weight - 0.6) * 60.0).clamp(1.0, 254.0) as u8;
    match name {
        // B3: woodcut dilation scales with ink_weight — 0px at the 0.6 default (threshold alone, not a
        // slab), up to 2px when the user asks for a heavier hand.
        "threshold-bold" => dilate_ink(&threshold_at(g, t), (((ink_weight - 0.6) * 5.0).round() as i32).clamp(0, 2)),
        "engrave-invert" => invert(&xdog(g, ink_weight)),
        "dither" => floyd_steinberg(g),
        "matte-solid" => threshold_at(g, t),
        "threshold-invert" => invert(&threshold_at(g, t)),
        "halftone" => bayer(g),
        _ => xdog(g, ink_weight), // "xdog" / line / default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64×64 test card: left half white, right half a mid-grey wedge, a black vertical line at x=20.
    fn card() -> GrayImage {
        let mut g = GrayImage::from_pixel(64, 64, Luma([255]));
        for y in 0..64 {
            for x in 0..64 {
                if x >= 32 {
                    g.put_pixel(x, y, Luma([128]));
                }
                if x == 20 {
                    g.put_pixel(x, y, Luma([0]));
                }
            }
        }
        g
    }

    #[test]
    fn otsu_splits_the_wedge() {
        let t = otsu(&card());
        assert!(t > 0 && t < 255);
    }

    #[test]
    fn xdog_marks_the_edge_and_is_deterministic() {
        let g = card();
        let a = xdog(&g, 0.6);
        let b = xdog(&g, 0.6);
        assert_eq!(a.as_raw(), b.as_raw(), "deterministic");
        // some dark line pixels exist (the black line / the 32-boundary edge).
        assert!(a.pixels().any(|p| p.0[0] < 100), "xdog produced no dark line");
    }

    #[test]
    fn ink_weight_dials_line_boldness() {
        // B3: a heavier ink weight must produce at least as much ink as a lighter one (monotone dial).
        let g = card();
        let ink = |w: f32| xdog(&g, w).pixels().filter(|p| p.0[0] < 128).count();
        assert!(ink(0.85) >= ink(0.6), "ink-weight 0.85 should be ≥ 0.6");
        assert!(ink(0.6) >= ink(0.35), "ink-weight 0.6 should be ≥ 0.35");
    }

    #[test]
    fn threshold_bold_is_1bit_and_has_ink() {
        let b = binarise(&card(), "threshold-bold", 0.6);
        assert!(b.pixels().all(|p| p.0[0] == 0 || p.0[0] == 255), "not 1-bit");
        assert!(b.pixels().any(|p| p.0[0] == 0), "no ink");
    }

    #[test]
    fn every_binariser_runs_and_is_deterministic() {
        let g = card();
        for name in ["xdog", "threshold-bold", "engrave-invert", "dither", "matte-solid", "threshold-invert", "halftone"] {
            let a = binarise(&g, name, 0.6);
            let b = binarise(&g, name, 0.6);
            assert_eq!(a.as_raw(), b.as_raw(), "{name} not deterministic");
            assert_eq!((a.width(), a.height()), (64, 64));
        }
    }
}
