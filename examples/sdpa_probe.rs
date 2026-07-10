//! Phase-2 probe: is candle's Metal SDPA correct + faster than plakat's eager attention?
//! Run: `cargo run --release --features metal --example sdpa_probe`
//! (throwaway — decides whether the attention-optimization item is worth pursuing.)

use candle_core::{DType, Device, Result, Tensor, D};

fn eager(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
    let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
    attn.matmul(&v.contiguous()?)
}

fn main() -> Result<()> {
    let dev = Device::new_metal(0)?;
    // (heads, seq, head_dim): SD3-MMDiT-ish self-attention at a few sequence lengths.
    for &(h, s, d) in &[(24usize, 1024usize, 64usize), (24, 1536usize, 64), (10, 4096usize, 64)] {
        let (b, scale) = (1usize, 1.0 / (d as f64).sqrt());
        let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?;

        // correctness: max abs diff eager vs sdpa
        let e = eager(&q, &k, &v, scale)?;
        let sd = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
        let diff = (e.to_dtype(DType::F32)? - sd.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let mean = e.abs()?.mean_all()?.to_scalar::<f32>()?;

        // timing
        let n = 30;
        for _ in 0..5 {
            eager(&q, &k, &v, scale)?;
        }
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            eager(&q, &k, &v, scale)?;
        }
        dev.synchronize()?;
        let eager_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

        for _ in 0..5 {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
        }
        dev.synchronize()?;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
        }
        dev.synchronize()?;
        let sdpa_ms = t1.elapsed().as_secs_f64() * 1e3 / n as f64;

        println!(
            "h={h} s={s} d={d}: max|eager-sdpa|={diff:.3e} (mean|e|={mean:.3}) | eager {eager_ms:.2}ms  sdpa {sdpa_ms:.2}ms  => {:.2}x",
            eager_ms / sdpa_ms
        );
    }
    Ok(())
}
