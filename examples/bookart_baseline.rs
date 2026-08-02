//! G0.2 baseline harness (RFC BOOKART-1 §2.3 / ROADMAP_BOOKART_1 G0.2): quantify the
//! **naive** "generate grey → key out the background" control against the bookart
//! approach (**binarise → luminance-alpha**), so every later phase has numbers to beat.
//!
//! Metrics on a raw generated ornament PNG:
//!   chroma      — mean/max HSV saturation + fraction of "coloured" px (truly B/W?)
//!   page-haze   — mean alpha on near-white px after transparency (page stays clean?)
//!   alpha-halo  — fraction of partial-alpha px, 8<α<247 (soft-edge fringe / halo?)
//!   symmetry    — bilateral fold RMS, 0=perfect (diffusion can't hold it → motivates
//!                 the symmetry engine; independent of the transparency treatment)
//!
//! run:  cargo run --release --example bookart_baseline -- <raw.png> [out_dir]
//! also writes <name>_naive.png / <name>_ours.png composited over a cream page.

use image::{GrayImage, Luma, Rgb, RgbImage};

fn to_luma(img: &RgbImage) -> GrayImage {
    let mut g = GrayImage::new(img.width(), img.height());
    for (x, y, p) in img.enumerate_pixels() {
        let l = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        g.put_pixel(x, y, Luma([l.round() as u8]));
    }
    g
}

/// HSV-ish saturation stats: how far from grey is the "black and white" render really?
fn chroma_stats(img: &RgbImage) -> (f32, f32, f32) {
    let (mut sum, mut max, mut coloured, mut n) = (0.0f32, 0.0f32, 0u64, 0u64);
    for p in img.pixels() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let sat = if mx <= 0.0 { 0.0 } else { (mx - mn) / mx };
        sum += sat;
        max = max.max(sat);
        if sat > 0.10 {
            coloured += 1;
        }
        n += 1;
    }
    let n = n.max(1) as f32;
    (sum / n, max, coloured as f32 / n)
}

/// Bilateral (mirror about the vertical mid-axis) fold RMS on luma, normalised 0..1.
fn bilateral_rms(g: &GrayImage) -> f32 {
    let (w, h) = (g.width(), g.height());
    let (mut acc, mut n) = (0.0f64, 0u64);
    for y in 0..h {
        for x in 0..w / 2 {
            let a = g.get_pixel(x, y).0[0] as f64;
            let b = g.get_pixel(w - 1 - x, y).0[0] as f64;
            let d = (a - b) / 255.0;
            acc += d * d;
            n += 1;
        }
    }
    ((acc / n.max(1) as f64).sqrt()) as f32
}

/// Otsu global threshold on a luma histogram.
fn otsu(g: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for p in g.pixels() {
        hist[p[0] as usize] += 1;
    }
    let total: u64 = hist.iter().sum();
    let sum_all: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let (mut wb, mut sumb, mut best_t, mut best_var) = (0u64, 0.0f64, 0u8, -1.0f64);
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

fn alpha_curve(l: u8, white_cut: f32, gamma: f32) -> u8 {
    let cov = (255.0 - l as f32) / 255.0;
    let cov = ((cov - white_cut) / (1.0 - white_cut)).clamp(0.0, 1.0);
    (cov.powf(gamma) * 255.0).round() as u8
}

/// page-haze (mean α on near-white px) + alpha-halo (partial-α fraction) for an α field.
fn alpha_stats(g: &GrayImage, alpha: &dyn Fn(u8) -> u8) -> (f32, f32) {
    let (mut haze, mut nh, mut partial, mut n) = (0u64, 0u64, 0u64, 0u64);
    for p in g.pixels() {
        let l = p[0];
        let a = alpha(l);
        if l > 238 {
            haze += a as u64;
            nh += 1;
        }
        if a > 8 && a < 247 {
            partial += 1;
        }
        n += 1;
    }
    (haze as f32 / nh.max(1) as f32, partial as f32 / n.max(1) as f32)
}

fn over_cream(g: &GrayImage, alpha: &dyn Fn(u8) -> u8) -> RgbImage {
    let cream = [232.0f32, 220.0, 192.0];
    let mut out = RgbImage::new(g.width(), g.height());
    for (x, y, p) in g.enumerate_pixels() {
        let a = alpha(p[0]) as f32 / 255.0; // ink black over cream
        out.put_pixel(x, y, Rgb([(cream[0] * (1.0 - a)) as u8, (cream[1] * (1.0 - a)) as u8, (cream[2] * (1.0 - a)) as u8]));
    }
    out
}

fn main() {
    let raw = std::env::args().nth(1).expect("usage: bookart_baseline <raw.png> [out_dir]");
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| ".".into());
    let stem = std::path::Path::new(&raw).file_stem().unwrap().to_string_lossy().to_string();
    let img = image::open(&raw).expect("open raw").to_rgb8();
    let g = to_luma(&img);

    let (mean_sat, max_sat, frac_col) = chroma_stats(&img);
    let sym = bilateral_rms(&g);
    let t = otsu(&g);

    // NAIVE: linear luminance→alpha on the raw grey (no binarise, no white-cut).
    let naive = move |l: u8| (255 - l);
    let (haze_n, halo_n) = alpha_stats(&g, &naive);

    // OURS: Otsu-binarise (kills chroma + soft grey), then the G0.5 curve (white_cut .07, γ .70).
    let gbin: GrayImage = {
        let mut b = g.clone();
        for p in b.pixels_mut() {
            p.0[0] = if p.0[0] <= t { 0 } else { 255 };
        }
        b
    };
    let ours = move |l: u8| alpha_curve(l, 0.07, 0.70);
    let (haze_o, halo_o) = alpha_stats(&gbin, &ours);
    let (_ms2, _mx2, frac_col_bin) = (0.0, 0.0, 0.0); // binarised → chroma is 0 by construction

    println!("== {stem} ({}x{}) ==", img.width(), img.height());
    println!("raw chroma      : mean-sat {mean_sat:.3}  max-sat {max_sat:.3}  coloured-frac {frac_col:.3}   (0 = truly B/W)");
    println!("bilateral symm  : RMS {sym:.3}   (0 = perfect; diffusion won't hold it → symmetry engine)");
    println!("otsu threshold  : {t}");
    println!("NAIVE (linear α on raw grey):  page-haze {haze_n:6.1}   alpha-halo {halo_n:.3}   chroma retained {frac_col:.3}");
    println!("OURS  (binarise + G0.5 curve): page-haze {haze_o:6.1}   alpha-halo {halo_o:.3}   chroma {frac_col_bin:.3}");

    let np = format!("{out_dir}/{stem}_naive.png");
    let op = format!("{out_dir}/{stem}_ours.png");
    over_cream(&g, &naive).save(&np).unwrap();
    over_cream(&gbin, &ours).save(&op).unwrap();
    println!("wrote {np}  {op}");
}
