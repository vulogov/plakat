//! SPIKE (throwaway): can candle's Metal backend run the *backward* pass an
//! SDXL-UNet LoRA training step needs? Tests AdamW + backprop through the
//! ops a real step hits. If these fail on Metal, in-plakat training is
//! CPU-only (impractical for SDXL) and external-train-then-pin wins.
//!
//!   cargo run --release --features metal --bin train_spike
use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

// F32 random Var — `Var::randn` defaults to F64, which Metal's rand_uniform
// doesn't implement.
fn vr(s: impl Into<Shape>, d: &Device) -> candle_core::Result<Var> {
    Var::from_tensor(&Tensor::randn(0f32, 0.02f32, s, d)?)
}

fn check(name: &str, r: candle_core::Result<()>) {
    match r {
        Ok(()) => println!("  ✓ {name}"),
        Err(e) => println!("  ✗ {name} — {e}"),
    }
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    println!("== candle Metal training spike ==\ndevice: {dev:?}\n");

    // T1: a real LoRA-shaped training step — down·up matmul + AdamW. Does the
    // loss actually go down (gradients flow + optimizer updates) on Metal?
    let a = vr((64, 8), &dev)?; // LoRA down (rank 8)
    let b = vr((8, 64), &dev)?; // LoRA up
    let mut opt = AdamW::new(
        vec![a.clone(), b.clone()],
        ParamsAdamW { lr: 1e-2, ..Default::default() },
    )?;
    let x = Tensor::randn(0f32, 1f32, (4, 64), &dev)?;
    let target = Tensor::randn(0f32, 1f32, (4, 64), &dev)?;
    let (mut first, mut last) = (0f32, 0f32);
    for i in 0..40 {
        let y = x.matmul(a.as_tensor())?.matmul(b.as_tensor())?;
        let loss = (&y - &target)?.sqr()?.mean_all()?;
        last = loss.to_scalar::<f32>()?;
        if i == 0 {
            first = last;
        }
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }
    println!(
        "T1 LoRA matmul train: loss {first:.4} → {last:.4}  [{}]\n",
        if last < first && last.is_finite() { "LEARNS ✓" } else { "NO LEARNING ✗" }
    );

    // T2: backward-kernel coverage for the SDXL UNet's heavy ops.
    println!("backward-kernel coverage on Metal:");
    check("conv2d", (|| {
        let w = vr((8, 4, 3, 3), &dev)?;
        let img = Tensor::randn(0f32, 1f32, (1, 4, 16, 16), &dev)?;
        img.conv2d(w.as_tensor(), 1, 1, 1, 1)?.sqr()?.mean_all()?.backward()?;
        Ok(())
    })());
    check("softmax (attention)", (|| {
        let w = vr((16, 16), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (8, 16), &dev)?;
        candle_nn::ops::softmax_last_dim(&x.matmul(w.as_tensor())?)?
            .sqr()?
            .mean_all()?
            .backward()?;
        Ok(())
    })());
    check("layernorm ops (mean/var/broadcast)", (|| {
        let g = Var::ones((16,), DType::F32, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (8, 16), &dev)?;
        let mean = x.mean_keepdim(1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(1)?;
        let norm = xc.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        norm.broadcast_mul(g.as_tensor())?.sqr()?.mean_all()?.backward()?;
        Ok(())
    })());
    check("silu", (|| {
        let w = vr((16, 16), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (8, 16), &dev)?;
        x.matmul(w.as_tensor())?.silu()?.sqr()?.mean_all()?.backward()?;
        Ok(())
    })());

    // T5: the real thing — a LoraLinear's trainable adapter learns through
    // set_train_adapter + the optimizer (grad must route to the Vars).
    {
        use candle_core::Module;
        use plakat::pipelines::lora_linear::LoraLinear;
        let (in_d, out_d, rank) = (32usize, 32usize, 4usize);
        let base = candle_nn::Linear::new(Tensor::randn(0f32, 0.1f32, (out_d, in_d), &dev)?, None);
        let ll = LoraLinear::from_linear(base)?;
        let a = vr((rank, in_d), &dev)?;
        let b = vr((out_d, rank), &dev)?;
        ll.set_train_adapter(a.clone(), b.clone(), 1.0);
        let mut opt = AdamW::new(
            vec![a.clone(), b.clone()],
            ParamsAdamW { lr: 5e-2, ..Default::default() },
        )?;
        let x = Tensor::randn(0f32, 1f32, (16, in_d), &dev)?;
        let target = Tensor::randn(0f32, 1f32, (16, out_d), &dev)?;
        let (mut f, mut l) = (0f32, 0f32);
        for i in 0..80 {
            let y = ll.forward(&x)?;
            let loss = (&y - &target)?.sqr()?.mean_all()?;
            l = loss.to_scalar::<f32>()?;
            if i == 0 {
                f = l;
            }
            opt.step(&loss.backward()?)?;
        }
        println!(
            "\nT5 LoraLinear adapter train: loss {f:.4} → {l:.4}  [{}]",
            if l < f && l.is_finite() { "LEARNS ✓" } else { "✗" }
        );
    }
    Ok(())
}
