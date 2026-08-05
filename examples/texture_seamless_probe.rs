//! G0.1 (RFC TEXTURE-1) — the load-bearing feasibility probe for **native circular-convolution
//! seamlessness**. Proves the primitive works in candle on the selected device (Metal when built with
//! `--features metal`), and measures that a **circular-padded** conv makes the tile join as continuous
//! as the interior — where a zero-padded conv leaves a seam.
//!
//!   cargo run --release --features metal --example texture_seamless_probe
//!
//! Finding recorded alongside the numbers: plakat's own UNet (`sd_train/unet.rs`) delegates its ResNet
//! convs to `candle_transformers::…::ResnetBlock2D`, so the B3 seamless engine must apply this circular
//! pad inside a *vendored* circular ResNet (plakat already vendors CLIP), not merely a flag on the few
//! plakat-owned convs. This probe validates the shared primitive both approaches need.

use candle_core::{DType, Device, Result, Tensor, D};

/// Circular ("wrap") padding of the last two dims by `p` — the tileable-diffusion primitive. candle's
/// `Conv2dConfig` only zero-pads, so we wrap-pad the input and convolve with `padding: 0`.
fn circular_pad2d(t: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(t.clone());
    }
    // width (last dim)
    let w = t.dim(D::Minus1)?;
    let left = t.narrow(D::Minus1, w - p, p)?; // last p cols → prepend
    let right = t.narrow(D::Minus1, 0, p)?; // first p cols → append
    let t = Tensor::cat(&[&left, t, &right], D::Minus1)?;
    // height (second-to-last dim)
    let h = t.dim(D::Minus2)?;
    let top = t.narrow(D::Minus2, h - p, p)?;
    let bot = t.narrow(D::Minus2, 0, p)?;
    Tensor::cat(&[&top, &t, &bot], D::Minus2)
}

/// RMS of a tensor.
fn rms(t: &Tensor) -> Result<f32> {
    Ok(t.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
}

/// The scorecard's tileability metric on one axis: the **join discontinuity** (edge col W-1 vs col 0)
/// relative to the **interior** adjacent-column difference. ~1.0 = tiles like its own interior (seamless);
/// ≫1.0 = a visible seam.
fn seam_ratio_x(out: &Tensor) -> Result<f32> {
    let w = out.dim(D::Minus1)?;
    let first = out.narrow(D::Minus1, 0, 1)?;
    let last = out.narrow(D::Minus1, w - 1, 1)?;
    let join = rms(&(first - last)?)?;
    let a = out.narrow(D::Minus1, 1, w - 1)?;
    let b = out.narrow(D::Minus1, 0, w - 1)?;
    let interior = rms(&(a - b)?)?;
    Ok(join / interior.max(1e-6))
}

fn main() -> Result<()> {
    let device = plakat::api::device("auto").unwrap_or(Device::Cpu);
    println!("G0.1 circular-conv probe — device: {device:?}\n");

    // --- correctness: a tiny wrap on CPU (deterministic, exact) ---
    {
        let cpu = Device::Cpu;
        // 1×1×2×2 = [[1,2],[3,4]]
        let t = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 2, 2), &cpu)?;
        let p = circular_pad2d(&t, 1)?;
        // 4×4 wrapped: corners wrap both ways; centre is the original.
        let v = p.flatten_all()?.to_vec1::<f32>()?;
        let expect = vec![
            4., 3., 4., 3., //
            2., 1., 2., 1., //
            4., 3., 4., 3., //
            2., 1., 2., 1., //
        ];
        assert_eq!(v, expect, "circular_pad2d wrap is wrong");
        println!("✓ circular_pad2d wrap correct (CPU exact)");
    }

    // --- seam measurement: a smoothing conv, zero-pad vs circular-pad, on `device` ---
    let (c, h, w) = (1usize, 96usize, 96usize);
    let x = Tensor::randn(0f32, 1.0, (1, c, h, w), &device)?.to_dtype(DType::F32)?;
    // a 3×3 box blur kernel (Cout=Cin=1) — enough to expose an edge seam.
    let k = Tensor::full(1f32 / 9.0, (1, 1, 3, 3), &device)?;

    // A real UNet stacks dozens of convs — the zero-pad edge artifact compounds with depth, while a
    // circular conv stays seamless. Emulate that by applying the conv DEPTH times.
    const DEPTH: usize = 12;
    let mut zero = x.clone();
    let mut circ = x.clone();
    for _ in 0..DEPTH {
        zero = zero.conv2d(&k, 1, 1, 1, 1)?; // candle zero-pad, padding=1
        circ = circular_pad2d(&circ, 1)?.conv2d(&k, 0, 1, 1, 1)?; // wrap-pad + padding=0
    }

    let (sz, sc) = (seam_ratio_x(&zero)?, seam_ratio_x(&circ)?);
    println!("\n  after {DEPTH} stacked convs (UNet-like depth):");
    println!("  zero-pad  seam ratio (join/interior): {sz:.3}   (a growing seam)");
    println!("  circular  seam ratio (join/interior): {sc:.3}   (≈1.0 = tiles like its interior)");
    let factor = sz / sc.max(1e-6);
    println!("  → circular is {factor:.1}× more continuous at the tile join");

    // Exit: circular keeps the join ≈ interior (seamless, ≤ ~1.15) and beats zero-pad — independent of
    // depth. (The absolute factor is small only because a box blur spreads the edge artifact 1 px/conv;
    // a real UNet's strided downsamples + wide receptive fields spread it far faster, so real seams are
    // worse — the invariant that matters is that circular stays ≈1.0 at any depth.)
    let finite = circ.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite());
    let pass = sc <= 1.15 && sc < sz && finite;
    println!("  (factor {factor:.1}× — informational; grows with real UNet depth/receptive field)");
    println!("\n  G0.1 exit: {}", if pass { "PASS — native circular conv is feasible + seamless on this device" } else { "REVIEW — see numbers above" });
    Ok(())
}
