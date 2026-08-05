//! G0.A (ROADMAP 6.4.0 "Deepen texture") — the LOAD-BEARING probe for **spatially-varying metallic
//! masks** on composite materials. 6.3's `derive::metallic_from_albedo` is a *per-pixel* rule
//! (`sat<0.12 && l>0.5`); on a rusted-iron plate (bare steel ⇄ orange rust) that leaves a noisy,
//! incoherent mask. This probe decides HOW Track A should derive a structured mask, measure-first, by
//! scoring three candidates' IoU against a known ground-truth metal mask on a synthetic fixture:
//!
//!   1. baseline    — the current per-pixel heuristic (control, no spatial coherence)
//!   2. region-vote — per-pixel score → large circular box (majority vote) → threshold (isotropic coherence)
//!   3. bilateral   — per-pixel score → edge-preserving (circular) smoothing → threshold (edge-aware coherence)
//!
//! The three are increasing levels of spatial coherence. The fixture is built with the exact failure
//! modes per-pixel classification suffers: DARK SCRATCHES on the steel (low luma → per-pixel misses),
//! and PALE DESATURATED RUST patches (low sat → per-pixel false-fires as metal). A region-coherent
//! approach should fill the scratches and out-vote the pale patches.
//!
//! Pure/weight-free — no GPU, no model. Run:
//!   cargo run --release --example texture_metallic_probe
//!
//! Exit criterion: pick the approach with the best IoU that stays region-coherent (and, being a
//! circular-window op, tiles). A generated ControlNet mask stays the documented escalation.

use image::{Rgb, RgbImage};

// ---------- deterministic hash noise (no rand dep, reproducible) ----------
fn hash2(x: u32, y: u32, salt: u32) -> f32 {
    let mut h = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263)).wrapping_add(salt.wrapping_mul(2246822519));
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32) // [0,1)
}

// ---------- colour helpers ----------
fn luma01(p: &Rgb<u8>) -> f32 {
    (0.2126 * p.0[0] as f32 + 0.7152 * p.0[1] as f32 + 0.0722 * p.0[2] as f32) / 255.0
}
fn sat01(p: &Rgb<u8>) -> f32 {
    let (r, g, b) = (p.0[0] as f32, p.0[1] as f32, p.0[2] as f32);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    if mx > 0.0 { (mx - mn) / mx } else { 0.0 }
}
// ---------- the synthetic rusted-iron fixture + ground-truth metal mask ----------
// Bare steel = low-sat grey, bright; rust = saturated orange-brown, darker. Metal regions are a few
// blobs (a plate with corrosion eating in from edges + patches). Ground truth is the blob membership,
// BEFORE colour noise, so IoU measures how well each approach recovers the true material regions.
const N: u32 = 256;

fn is_metal_truth(x: u32, y: u32) -> bool {
    let fx = x as f32 / N as f32;
    let fy = y as f32 / N as f32;
    // rust creeps in from the border and in a couple of blobs
    let border = fx.min(1.0 - fx).min(fy).min(1.0 - fy); // 0 at edge → 0.5 center
    let mut rust = border < 0.18; // corroded rim
    // two rust blobs
    let d1 = ((fx - 0.35).powi(2) + (fy - 0.4).powi(2)).sqrt();
    let d2 = ((fx - 0.7).powi(2) + (fy - 0.68).powi(2)).sqrt();
    if d1 < 0.16 || d2 < 0.12 {
        rust = true;
    }
    !rust
}

fn build_fixture() -> (RgbImage, Vec<bool>) {
    let mut img = RgbImage::new(N, N);
    let mut truth = vec![false; (N * N) as usize];
    for y in 0..N {
        for x in 0..N {
            let metal = is_metal_truth(x, y);
            truth[(y * N + x) as usize] = metal;
            let n = hash2(x, y, 1) - 0.5; // [-0.5,0.5]
            let n2 = hash2(x, y, 2);
            let fx = x as f32 / N as f32;
            let fy = y as f32 / N as f32;
            let (r, g, b) = if metal {
                // bright bare steel ~0.62 luma, near-zero sat. FAILURE MODE: dark diagonal scratches
                // (a streak field) drive ~15% of metal pixels below the per-pixel luma gate.
                let streak = ((fx * 22.0 + fy * 37.0).sin() * (fy * 15.0).cos()).abs();
                let scratch = if streak > 0.86 || n2 > 0.97 { -0.34 } else { 0.0 };
                let base = 0.62 + 0.08 * n + scratch;
                let v = (base.clamp(0.06, 0.97) * 255.0) as u8;
                (v.saturating_sub(2), v, v.saturating_add(3)) // faint cool tint, still low-sat
            } else {
                // rust: mostly saturated orange-brown. FAILURE MODE: ~18% PALE efflorescence patches —
                // low saturation, mid-bright, faint brown — which fool the per-pixel sat<0.12 gate.
                let pale = hash2(x, y, 3) > 0.82;
                if pale {
                    let l = (0.55 + 0.12 * n).clamp(0.2, 0.85);
                    let v = (l * 255.0) as u8; // near-grey → low sat
                    (v, v.saturating_sub(6), v.saturating_sub(12)) // barely-brown, sat ~0.05
                } else {
                    let l = (0.34 + 0.16 * n).clamp(0.10, 0.62);
                    (((l * 1.55).clamp(0.0, 1.0) * 255.0) as u8, ((l * 0.72) * 255.0) as u8, ((l * 0.30) * 255.0) as u8)
                }
            };
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    (img, truth)
}

// ---------- candidate 1: current per-pixel heuristic (binary, no coherence) ----------
fn cand_baseline(img: &RgbImage) -> Vec<f32> {
    img.pixels()
        .map(|p| {
            let l = luma01(p);
            let s = sat01(p);
            if s < 0.12 && l > 0.5 { 1.0 } else { 0.0 }
        })
        .collect()
}

/// soft per-pixel metal-ness in [0,1] (the coherence candidates smooth THIS, not the binary): low sat
/// and bright → metal. Smooth ramps so the vote is graded, not a cliff.
fn metal_soft(img: &RgbImage) -> Vec<f32> {
    img.pixels()
        .map(|p| {
            let l = luma01(p);
            let s = sat01(p);
            let sat_term = ((0.16 - s) / 0.10).clamp(0.0, 1.0); // 1 when very grey, 0 by sat 0.16
            let lum_term = ((l - 0.42) / 0.18).clamp(0.0, 1.0); // 1 when bright
            sat_term * lum_term
        })
        .collect()
}

// ---------- candidate 2: region-vote (large circular box = isotropic majority) ----------
fn cand_region_vote(img: &RgbImage) -> Vec<f32> {
    let soft = metal_soft(img);
    let n = N as i32;
    let idx = |x: i32, y: i32| -> usize { ((y.rem_euclid(n)) * n + x.rem_euclid(n)) as usize };
    let radius = 8i32; // must exceed the scratch width / pale-patch size to out-vote them
    let mut out = vec![0f32; soft.len()];
    for y in 0..n {
        for x in 0..n {
            let mut acc = 0f32;
            let mut cnt = 0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    acc += soft[idx(x + dx, y + dy)];
                    cnt += 1.0;
                }
            }
            out[idx(x, y)] = if acc / cnt >= 0.5 { 1.0 } else { 0.0 };
        }
    }
    out
}

// ---------- candidate 3: bilateral (edge-preserving, circular → tileable) ----------
fn cand_bilateral(img: &RgbImage) -> Vec<f32> {
    let soft = metal_soft(img);
    let n = N as i32;
    let idx = |x: i32, y: i32| -> usize { ((y.rem_euclid(n)) * n + x.rem_euclid(n)) as usize };
    let radius = 8i32;
    let sigma_c = 0.22f32; // colour range
    let mut out = vec![0f32; soft.len()];
    for y in 0..n {
        for x in 0..n {
            let cp = img.get_pixel(x as u32, y as u32);
            let (cl, cs) = (luma01(cp), sat01(cp));
            let mut wsum = 0f32;
            let mut vsum = 0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let np = img.get_pixel((x + dx).rem_euclid(n) as u32, (y + dy).rem_euclid(n) as u32);
                    let dl = luma01(np) - cl;
                    let ds = sat01(np) - cs;
                    let w = (-(dl * dl + ds * ds) / (2.0 * sigma_c * sigma_c)).exp();
                    wsum += w;
                    vsum += w * soft[idx(x + dx, y + dy)];
                }
            }
            out[idx(x, y)] = if wsum > 0.0 && vsum / wsum >= 0.5 { 1.0 } else { 0.0 };
        }
    }
    out
}

// ---------- scoring ----------
struct Score {
    iou: f32,
    precision: f32,
    recall: f32,
}
fn score(mask: &[f32], truth: &[bool]) -> Score {
    let (mut inter, mut uni, mut tp, mut pp, mut ap) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for (m, &t) in mask.iter().zip(truth) {
        let pred = *m >= 0.5;
        if pred {
            pp += 1;
        }
        if t {
            ap += 1;
        }
        if pred && t {
            inter += 1;
            tp += 1;
        }
        if pred || t {
            uni += 1;
        }
    }
    Score {
        iou: if uni > 0 { inter as f32 / uni as f32 } else { 0.0 },
        precision: if pp > 0 { tp as f32 / pp as f32 } else { 0.0 },
        recall: if ap > 0 { tp as f32 / ap as f32 } else { 0.0 },
    }
}

fn main() {
    let (img, truth) = build_fixture();
    let metal_frac = truth.iter().filter(|&&t| t).count() as f32 / truth.len() as f32;
    let _ = img.save("/tmp/texture_metallic_fixture.png");
    println!("G0.A — spatial metallic-mask approach probe");
    println!("fixture: {N}×{N} rusted-iron, ground-truth metal fraction = {:.3}", metal_frac);
    println!("(saved /tmp/texture_metallic_fixture.png)\n");

    let cands: Vec<(&str, Vec<f32>)> = vec![
        ("baseline (per-pixel)", cand_baseline(&img)),
        ("region-vote (box r=8)", cand_region_vote(&img)),
        ("bilateral (edge-aware r=8)", cand_bilateral(&img)),
    ];
    println!("{:<28} {:>7} {:>7} {:>7}", "approach", "IoU", "prec", "recall");
    println!("{}", "-".repeat(52));
    let mut best = ("", -1.0f32);
    for (name, mask) in &cands {
        let s = score(mask, &truth);
        println!("{:<28} {:>7.3} {:>7.3} {:>7.3}", name, s.iou, s.precision, s.recall);
        if s.iou > best.1 {
            best = (name, s.iou);
        }
    }
    println!("\nBEST: {}  (IoU {:.3})", best.0, best.1);
    println!("EXIT: promote the winner into Track A `metallic:\"auto\"`; keep scalar + from-albedo.");
}
