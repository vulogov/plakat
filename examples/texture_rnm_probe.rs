//! G0.1 (ROADMAP 6.5.0 "Trim sheets & decals") — the one non-trivial algorithm: blending a **decal's**
//! tangent-space normal over a **base** normal. A naive lerp FLATTENS both (halves the base tilt and the
//! detail amplitude). Reoriented Normal Mapping (RNM, Barré-Brisebois & Hill, "Blending in Detail")
//! reorients the detail so it rides the base slope. This probe validates RNM against a lerp on two
//! falsifiable fixtures + a general case, and confirms the result stays a unit normal. Pure/weight-free:
//!
//!   cargo run --release --example texture_rnm_probe
//!
//! Exit: RNM preserves base tilt (flat detail → base unchanged), preserves detail amplitude (detail over
//! flat base → detail unchanged), and stays unit-length — where the lerp flattens both. → Track B uses RNM.

use image::Rgb;

fn enc(v: [f32; 3]) -> Rgb<u8> {
    Rgb([0, 1, 2].map(|i| ((v[i] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8))
}
fn dec(p: Rgb<u8>) -> [f32; 3] {
    [0, 1, 2].map(|i| p.0[i] as f32 / 255.0 * 2.0 - 1.0)
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Reoriented Normal Mapping blend of a detail normal over a base normal (both `[0,1]` RGB → `[0,1]` RGB).
fn rnm(base: Rgb<u8>, detail: Rgb<u8>) -> Rgb<u8> {
    let b = base.0.map(|c| c as f32 / 255.0);
    let d = detail.0.map(|c| c as f32 / 255.0);
    // t = base*(2,2,2)+(-1,-1,0) ; u = detail*(-2,-2,2)+(1,1,-1)
    let t = [b[0] * 2.0 - 1.0, b[1] * 2.0 - 1.0, b[2] * 2.0];
    let u = [d[0] * -2.0 + 1.0, d[1] * -2.0 + 1.0, d[2] * 2.0 - 1.0];
    let dt = t[0] * u[0] + t[1] * u[1] + t[2] * u[2];
    let r = [t[0] * dt / t[2] - u[0], t[1] * dt / t[2] - u[1], t[2] * dt / t[2] - u[2]];
    enc(normalize(r))
}

/// Naive per-channel lerp (the wrong way) at t=0.5, renormalised so it's a fair unit-length comparison.
fn lerp(base: Rgb<u8>, detail: Rgb<u8>) -> Rgb<u8> {
    let (b, d) = (dec(base), dec(detail));
    enc(normalize([(b[0] + d[0]) * 0.5, (b[1] + d[1]) * 0.5, (b[2] + d[2]) * 0.5]))
}

fn main() {
    let flat = enc(normalize([0.0, 0.0, 1.0])); // straight-up normal (no detail / flat base)
    let tilted = enc(normalize([-0.6, 0.0, 1.0])); // a base that faces partly -x

    println!("G0.1 — reoriented normal mapping (RNM) vs naive lerp\n");

    // TEST 1 — flat DETAIL over a tilted BASE. RNM must return the base tilt unchanged; lerp flattens it.
    let base_nx = dec(tilted)[0];
    let rnm_nx = dec(rnm(tilted, flat))[0];
    let lerp_nx = dec(lerp(tilted, flat))[0];
    println!("test 1 — flat detail over a tilted base (base nx = {base_nx:.3}):");
    println!("   RNM  nx {rnm_nx:.3}   (err {:.3})   <- should match the base", (rnm_nx - base_nx).abs());
    println!("   lerp nx {lerp_nx:.3}   (err {:.3})   <- flattened toward 0", (lerp_nx - base_nx).abs());
    let t1 = (rnm_nx - base_nx).abs() < 0.02 && lerp_nx.abs() < base_nx.abs() * 0.7;

    // TEST 2 — DETAIL bumps over a flat BASE. RNM must return the detail at full amplitude; lerp halves it.
    let bump = enc(normalize([0.5, -0.3, 1.0])); // a detail bump
    let det_nx = dec(bump)[0];
    let rnm2 = dec(rnm(flat, bump))[0];
    let lerp2 = dec(lerp(flat, bump))[0];
    println!("\ntest 2 — detail bump over a flat base (detail nx = {det_nx:.3}):");
    println!("   RNM  nx {rnm2:.3}   (amplitude kept {:.0}%)", rnm2 / det_nx * 100.0);
    println!("   lerp nx {lerp2:.3}   (amplitude kept {:.0}%)", lerp2 / det_nx * 100.0);
    let t2 = (rnm2 - det_nx).abs() < 0.02 && lerp2 < det_nx * 0.7;

    // TEST 3 — general: bumps over a tilted base. RNM keeps the base MEAN and the detail amplitude; and
    // every blended texel is unit-length.
    let mut base_sum = 0.0;
    let mut rnm_sum = 0.0;
    let mut rnm_var = 0.0;
    let mut det_var = 0.0;
    let mut unit_ok = 0u32;
    let n = 64;
    for i in 0..n {
        let phase = (i as f32 / n as f32 * std::f32::consts::TAU).sin();
        let d = enc(normalize([0.4 * phase, 0.2 * phase, 1.0])); // varying detail bump
        let blended = rnm(tilted, d);
        let (bx, rx, dx) = (dec(tilted)[0], dec(blended)[0], dec(d)[0]);
        base_sum += bx;
        rnm_sum += rx;
        rnm_var += (rx - bx).powi(2); // detail deviation around the base
        det_var += dx * dx;
        let v = dec(blended);
        if ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() - 1.0).abs() < 0.02 {
            unit_ok += 1;
        }
    }
    let mean_err = (rnm_sum / n as f32 - base_sum / n as f32).abs();
    let amp_ratio = (rnm_var / det_var).sqrt(); // detail amplitude carried onto the base
    let unit_frac = unit_ok as f32 / n as f32;
    println!("\ntest 3 — bumps over a tilted base:");
    println!("   base-mean preserved (err {mean_err:.3})  ·  detail amplitude carried {:.0}%  ·  unit-length {:.0}%", amp_ratio * 100.0, unit_frac * 100.0);
    // mean_err tolerance 0.03 absorbs 8-bit quantization across the several encode/decode round-trips
    // (base→u8, detail→u8, blended→u8); RNM's algebra is exact (tests 1 & 2 hit err 0.000).
    let t3 = mean_err < 0.03 && amp_ratio > 0.6 && unit_frac > 0.98;

    println!("\n{}", if t1 && t2 && t3 {
        "PASS — RNM preserves base tilt + detail amplitude + unit-length; lerp flattens. Track B uses RNM."
    } else {
        "FAIL — RNM did not behave as expected; revisit the blend before Track B."
    });
}
