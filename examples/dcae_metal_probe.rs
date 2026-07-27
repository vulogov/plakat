//! Probe: which DC-AE encoder-unique op diverges on Metal vs CPU?
//! The Sana DC-AE *decode* works on Metal (txt2img renders) but *encode* returns black,
//! so img2img/inpaint can't preserve the init. Isolate the broken op with random inputs
//! (no model weights) by running each candidate on CPU and Metal and comparing.
//! Run: `cargo run --release --features metal --example dcae_metal_probe`
//! (throwaway diagnostic.)

use candle_core::{DType, Device, Result, Tensor};

/// `F.pixel_unshuffle(x, r)` — the encoder downsample shortcut (permute+reshape).
fn pixel_unshuffle(x: &Tensor, r: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    x.reshape((b, c, h / r, r, w / r, r))?
        .permute((0, 1, 3, 5, 2, 4))?
        .contiguous()?
        .reshape((b, c * r * r, h / r, w / r))
}

/// The group-average after pixel_unshuffle: reshape to 5-D + mean over the middle axis.
fn group_avg_5d(y: &Tensor, group: usize) -> Result<Tensor> {
    let (b, cc, hh, ww) = y.dims4()?;
    y.reshape((b, cc / group, group, hh, ww))?.mean(2)
}

/// Compare a CPU tensor and a Metal tensor by max-abs diff (both moved to CPU F32).
fn cmp(name: &str, cpu: &Tensor, metal: &Tensor) -> Result<()> {
    let a = cpu.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = metal
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "{name}: shape mismatch");
    let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mag = a.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
    let flag = if max > 1e-3 { "  <<< DIVERGES" } else { "" };
    println!("  {name:22} max|Δ|={max:.6}  (magnitude {mag:.4}){flag}");
    Ok(())
}

fn main() -> Result<()> {
    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    // Encoder-ish shapes: a mid-stage feature map.
    let (b, c, h, w) = (1usize, 64usize, 64usize, 64usize);

    // Same random data on both devices (seed CPU, copy to Metal).
    let x_cpu = Tensor::randn(0f32, 1.0, (b, c, h, w), &cpu)?;
    let x_metal = x_cpu.to_device(&metal)?;

    println!("DC-AE encoder op probe (Metal vs CPU), input ({b},{c},{h},{w}):");

    // 1) pixel_unshuffle
    cmp("pixel_unshuffle", &pixel_unshuffle(&x_cpu, 2)?, &pixel_unshuffle(&x_metal, 2)?)?;

    // 2) group-average (5-D reshape + mean over middle axis) — prime suspect
    let yu_cpu = pixel_unshuffle(&x_cpu, 2)?; // (1, 256, 32, 32)
    let yu_metal = pixel_unshuffle(&x_metal, 2)?;
    for &g in &[2usize, 4] {
        cmp(
            &format!("group_avg_5d(g={g})"),
            &group_avg_5d(&yu_cpu, g)?,
            &group_avg_5d(&yu_metal, g)?,
        )?;
    }

    // 3) plain mean over a middle axis of a 4-D tensor (control)
    cmp("mean4d_axis1", &x_cpu.mean(1)?, &x_metal.mean(1)?)?;

    // 4) mean over the middle axis of a 5-D tensor directly (isolate rank-5 reduce)
    let r5_cpu = x_cpu.reshape((b, 16, 4, h, w))?;
    let r5_metal = x_metal.reshape((b, 16, 4, h, w))?;
    cmp("mean5d_axis2", &r5_cpu.mean(2)?, &r5_metal.mean(2)?)?;

    // ---- candidate Metal-safe replacements for group_avg_5d ----
    // A) rank-3 reduce: (b*cout, group, hh*ww) mean over axis 1.
    let group_avg_r3 = |y: &Tensor, g: usize| -> Result<Tensor> {
        let (b, cc, hh, ww) = y.dims4()?;
        let cout = cc / g;
        y.reshape((b * cout, g, hh * ww))?
            .mean(1)?
            .reshape((b, cout, hh, ww))
    };
    cmp("fix_r3(g=2)", &group_avg_5d(&yu_cpu, 2)?, &group_avg_r3(&yu_metal, 2)?)?;
    cmp("fix_r3(g=4)", &group_avg_5d(&yu_cpu, 4)?, &group_avg_r3(&yu_metal, 4)?)?;

    // B) narrow-add at rank-5 (elementwise add only, no reduce).
    let group_avg_add = |y: &Tensor, g: usize| -> Result<Tensor> {
        let (b, cc, hh, ww) = y.dims4()?;
        let cout = cc / g;
        let y5 = y.reshape((b, cout, g, hh, ww))?;
        let mut acc = y5.narrow(2, 0, 1)?;
        for j in 1..g {
            acc = (acc + y5.narrow(2, j, 1)?)?;
        }
        (acc.squeeze(2)? * (1.0 / g as f64))
    };
    cmp("fix_add(g=2)", &group_avg_5d(&yu_cpu, 2)?, &group_avg_add(&yu_metal, 2)?)?;
    cmp("fix_add(g=4)", &group_avg_5d(&yu_cpu, 4)?, &group_avg_add(&yu_metal, 4)?)?;

    // 5) stride-2 conv (encoder downsample conv)
    let wt_cpu = Tensor::randn(0f32, 0.05, (c, c, 3, 3), &cpu)?;
    let wt_metal = wt_cpu.to_device(&metal)?;
    let conv_cpu = x_cpu.conv2d(&wt_cpu, 1, 2, 1, 1)?;
    let conv_metal = x_metal.conv2d(&wt_metal, 1, 2, 1, 1)?;
    cmp("conv2d_stride2", &conv_cpu, &conv_metal)?;

    Ok(())
}
