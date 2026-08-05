//! G0.B (ROADMAP 6.4.0 "Deepen texture") — probe for **per-step latent-roll** as the high-frequency
//! seamless path (Track B1), before wiring it into the real SD sampler. 6.3 tiles a generated albedo
//! with a flat prompt + a post-hoc boundary feather; on HIGH-FREQUENCY materials (gravel, grain,
//! chainmail) feather *softens* a residual seam rather than removing it. The candidate fix rolls the
//! latent by a random offset each denoise step so the model's (zero-padded, non-circular) convs see the
//! boundary at every phase — distributing the discontinuity instead of accumulating it at one join.
//!
//! What is cleanly provable WEIGHT-FREE is the MECHANISM: **averaging a zero-padded (non-circular) conv
//! over all circular shifts `roll(-s)∘F_zero∘roll(s)` recovers the circular-padded conv** — i.e.
//! rolling distributes the fixed boundary bias uniformly, turning a seam-making operator into an
//! (approximately) seamless one. This probe measures the tile-join seam of a non-tileable high-freq
//! field under one mild low-pass, three ways:
//!   (a) zero-pad conv                     — the real UNet boundary behaviour (a seam)
//!   (c) circular-pad conv                 — the ideal / B2 escalation reference (proven in G0.1)
//!   (avg) shift-averaged zero-pad conv    — the roll MECHANISM: ≈ (c) if rolling distributes the bias
//!
//!   cargo run --release --features metal --example texture_roll_probe
//!
//! Exit: if (avg) closes most of the (a)→(c) gap, the roll mechanism is validated → B1 ships the
//! per-step (single-shift) approximation, with its definitive high-frequency number measured on a real
//! material during the B1 build (contained, default-off), and B2 (vendored circular ResNet, already
//! proven feasible in G0.1) as the guaranteed fallback. Keep candle-Metal ops ≤4-D.

use candle_core::{DType, Device, Result, Tensor, D};

/// Circular ("wrap") pad of the last two dims by `p` (the G0.1 primitive).
fn circular_pad2d(t: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(t.clone());
    }
    let w = t.dim(D::Minus1)?;
    let left = t.narrow(D::Minus1, w - p, p)?;
    let right = t.narrow(D::Minus1, 0, p)?;
    let t = Tensor::cat(&[&left, &t, &right], D::Minus1)?;
    let h = t.dim(D::Minus2)?;
    let top = t.narrow(D::Minus2, h - p, p)?;
    let bot = t.narrow(D::Minus2, 0, p)?;
    Tensor::cat(&[&top, &t, &bot], D::Minus2)
}

/// Circular shift of the last two dims by (dx, dy) — the per-step "roll". Invertible via (-dx,-dy).
fn roll2d(t: &Tensor, dx: usize, dy: usize) -> Result<Tensor> {
    let w = t.dim(D::Minus1)?;
    let h = t.dim(D::Minus2)?;
    let dx = dx % w;
    let dy = dy % h;
    let t = if dx == 0 {
        t.clone()
    } else {
        let a = t.narrow(D::Minus1, w - dx, dx)?;
        let b = t.narrow(D::Minus1, 0, w - dx)?;
        Tensor::cat(&[&a, &b], D::Minus1)?
    };
    if dy == 0 {
        Ok(t)
    } else {
        let a = t.narrow(D::Minus2, h - dy, dy)?;
        let b = t.narrow(D::Minus2, 0, h - dy)?;
        Tensor::cat(&[&a, &b], D::Minus2)
    }
}

/// zero-padded 3×3 conv (candle's default pad = zero) — the non-circular UNet behaviour.
fn conv_zero(t: &Tensor, w: &Tensor) -> Result<Tensor> {
    t.conv2d(w, 1, 1, 1, 1) // padding=1 (zero), stride 1
}
/// circular-padded 3×3 conv — wrap-pad then padding=0.
fn conv_circular(t: &Tensor, w: &Tensor) -> Result<Tensor> {
    circular_pad2d(t, 1)?.conv2d(w, 0, 1, 1, 1)
}

/// tile-join seam ratio: RMS of the wrap discontinuity (col0↔colN-1, row0↔rowN-1) over the mean RMS of
/// interior adjacent differences. ≈1 → the join is as smooth as the interior (seamless); >1 → a seam.
fn seam_ratio(t: &Tensor) -> Result<f32> {
    let w = t.dim(D::Minus1)?;
    let h = t.dim(D::Minus2)?;
    let rms = |a: &Tensor, b: &Tensor| -> Result<f32> {
        Ok((a - b)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    };
    // wrap seam (x): last col vs first col; (y): last row vs first row
    let colL = t.narrow(D::Minus1, w - 1, 1)?;
    let colR = t.narrow(D::Minus1, 0, 1)?;
    let rowT = t.narrow(D::Minus2, h - 1, 1)?;
    let rowB = t.narrow(D::Minus2, 0, 1)?;
    let seam = 0.5 * (rms(&colL, &colR)? + rms(&rowT, &rowB)?);
    // interior adjacent-diff RMS (x and y), a couple of samples mid-field
    let ix = rms(&t.narrow(D::Minus1, w / 2, 1)?, &t.narrow(D::Minus1, w / 2 + 1, 1)?)?;
    let iy = rms(&t.narrow(D::Minus2, h / 2, 1)?, &t.narrow(D::Minus2, h / 2 + 1, 1)?)?;
    let interior = 0.5 * (ix + iy);
    Ok(if interior > 1e-8 { seam / interior } else { 0.0 })
}

fn main() -> Result<()> {
    let device = if candle_core::utils::metal_is_available() {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("G0.B — per-step latent-roll seamless probe   (device: {:?})", device);

    let (c, n) = (4usize, 48usize); // channels, spatial — small so all-shift averaging is cheap
    // NON-tileable high-frequency field: NON-integer, zero-mean gratings (no DC ramp) so opposite edges
    // are unrelated but bounded — a real join seam a boundary method can distribute. seam > 1.
    let mut data = vec![0f32; c * n * n];
    for ch in 0..c {
        for y in 0..n {
            for x in 0..n {
                let fx = x as f32 / n as f32;
                let fy = y as f32 / n as f32;
                let k1 = 5.3 + ch as f32 * 1.1; // non-integer → does NOT wrap
                let k2 = 7.7 + ch as f32 * 1.3;
                data[ch * n * n + y * n + x] = (std::f32::consts::TAU * (k1 * fx)).sin()
                    * (std::f32::consts::TAU * (k2 * fy)).cos()
                    + 0.5 * (std::f32::consts::TAU * (k2 * fx + k1 * fy)).sin();
            }
        }
    }
    let x0 = Tensor::from_vec(data, (1, c, n, n), &device)?.to_dtype(DType::F32)?;
    println!("input NON-tileable field seam ratio: {:.3} (>1 = a real join seam)\n", seam_ratio(&x0)?);

    // a mild low-pass conv (so it HAS a boundary effect to distribute): centre + a small blur ring.
    let mut wdata = vec![0f32; c * c * 3 * 3];
    for o in 0..c {
        // depthwise 3×3 box-ish low-pass on the matching channel
        for k in 0..9 {
            wdata[(o * c + o) * 9 + k] = if k == 4 { 0.5 } else { 0.5 / 8.0 };
        }
    }
    let wconv = Tensor::from_vec(wdata, (c, c, 3, 3), &device)?;

    // (a) zero-pad single conv
    let xa = conv_zero(&x0, &wconv)?;
    // (c) circular-pad single conv (ideal)
    let xc = conv_circular(&x0, &wconv)?;
    // (avg) shift-average of the zero-pad conv over ALL circular shifts: roll(-s)∘F_zero∘roll(s).
    // (subsample every 2 px for speed — still 24×24 = 576 shifts.)
    let step = 2usize;
    let mut acc: Option<Tensor> = None;
    let mut cnt = 0f32;
    for sy in (0..n).step_by(step) {
        for sx in (0..n).step_by(step) {
            let r = roll2d(&x0, sx, sy)?;
            let f = conv_zero(&r, &wconv)?;
            let u = roll2d(&f, n - sx, n - sy)?;
            acc = Some(match acc {
                None => u,
                Some(a) => (a + u)?,
            });
            cnt += 1.0;
        }
    }
    let xavg = (acc.unwrap() / cnt as f64)?;

    let (sa, sc, savg) = (seam_ratio(&xa)?, seam_ratio(&xc)?, seam_ratio(&xavg)?);
    println!("single mild low-pass, tile-join seam ratio:");
    println!("  (a)  zero-pad conv             {:.3}   <- the seam the UNet leaves", sa);
    println!("  (c)  circular-pad conv         {:.3}   <- ideal / B2 reference", sc);
    println!("  (avg) shift-averaged zero-pad  {:.3}   <- the roll MECHANISM", savg);
    let gap = (sa - sc).abs().max(1e-6);
    let closed = ((sa - savg) / (sa - sc)).clamp(0.0, 2.0);
    println!("\nzero-pad→circular seam gap = {:.3}; shift-averaging closes {:.0}% of it.", gap, closed * 100.0);
    if closed >= 0.8 {
        println!("EXIT: roll MECHANISM validated (shift-averaging ≈ circular). Ship B1 (per-step roll),");
        println!("      measure its hi-freq number on a real material in the B1 build; B2 conv is the fallback.");
    } else if closed >= 0.4 {
        println!("EXIT: roll helps partially — ship B1 roll + feather; keep B2 vendored-conv ready.");
    } else {
        println!("EXIT: roll mechanism weak here — lean on B2 vendored circular ResNet (proven in G0.1).");
    }
    Ok(())
}
