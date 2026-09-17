//! Family-agnostic **blended latent diffusion** for LAYERED-1 repair (S4) and lift (S5), built on
//! [`NoiseSpace`](crate::pipelines::noise_space::NoiseSpace).
//!
//! Repair re-denoises the finish latent with the finish model, but only INSIDE a layer's mask: every step,
//! the region OUTSIDE the mask is replaced with the forward-noised original latent (so it re-settles to
//! what was already there), while the inside is left to the model. This is the RePaint / blended-latent
//! idea made family-agnostic — the only per-family piece is `NoiseSpace::noise_to`, so SD-family, Flux,
//! PixArt and the flow families all use one helper instead of each pipeline's dedicated inpaint model.
//!
//! `mask` is a soft latent mask in `[0,1]`: 1 = inside (regenerate), 0 = outside (keep). Releasing the mask
//! for the last few steps (RFC: `repair.release`) is just calling with an all-ones mask, so the whole
//! latent settles together — expressed here by [`blend`] with `mask = 1`.

use candle_core::{Result, Tensor};

use crate::pipelines::noise_space::NoiseSpace;

/// `mask ⊙ inside + (1 − mask) ⊙ outside`. With `mask = 1` the result is `inside` unchanged (the released
/// phase); with `mask = 0` it is `outside`. Shapes must broadcast (`mask` may be `(1,1,H,W)`).
pub fn blend(inside: &Tensor, mask: &Tensor, outside: &Tensor) -> Result<Tensor> {
    let keep = inside.broadcast_mul(mask)?;
    let one = Tensor::ones_like(mask)?;
    let inv = one.broadcast_sub(mask)?;
    let drop = outside.broadcast_mul(&inv)?;
    keep + drop
}

/// One masked-denoise blend step: replace everything OUTSIDE `mask` with the original latent forward-noised
/// to the level after `step`, keeping the model's `running` latent inside. `clean` is the pre-repair finish
/// latent; `noise` is a fixed tensor (reproducible). Inside the mask the model's own prediction survives.
pub fn step_blend(space: &dyn NoiseSpace, running: &Tensor, mask: &Tensor, clean: &Tensor, noise: &Tensor, step: usize) -> Result<Tensor> {
    let outside = space.noise_to(clean, noise, step)?;
    blend(running, mask, &outside)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::noise_space::{FlowSpace, LatentGeometry};
    use candle_core::{Device, DType};

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    #[test]
    fn blend_endpoints() {
        let d = Device::Cpu;
        let inside = Tensor::full(3.0f32, (1, 1, 4, 4), &d).unwrap();
        let outside = Tensor::full(9.0f32, (1, 1, 4, 4), &d).unwrap();
        let ones = Tensor::ones((1, 1, 4, 4), DType::F32, &d).unwrap();
        let zeros = Tensor::zeros((1, 1, 4, 4), DType::F32, &d).unwrap();
        let half = Tensor::full(0.5f32, (1, 1, 4, 4), &d).unwrap();
        let m1 = blend(&inside, &ones, &outside).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let m0 = blend(&inside, &zeros, &outside).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let mh = blend(&inside, &half, &outside).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((m1 - 3.0).abs() < 1e-6, "mask=1 keeps inside");
        assert!((m0 - 9.0).abs() < 1e-6, "mask=0 takes outside");
        assert!((mh - 6.0).abs() < 1e-6, "mask=0.5 is the midpoint");
    }

    #[test]
    fn broadcast_single_channel_mask() {
        // A (1,1,H,W) mask blends a multi-channel latent.
        let d = Device::Cpu;
        let inside = Tensor::full(1.0f32, (1, 4, 4, 4), &d).unwrap();
        let outside = Tensor::full(5.0f32, (1, 4, 4, 4), &d).unwrap();
        let mut mv = vec![0f32; 16];
        for (i, x) in mv.iter_mut().enumerate() {
            *x = if i < 8 { 1.0 } else { 0.0 };
        }
        let mask = Tensor::from_vec(mv, (1, 1, 4, 4), &d).unwrap();
        let out = blend(&inside, &mask, &outside).unwrap();
        assert_eq!(out.dims4().unwrap(), (1, 4, 4, 4));
        // Half kept (1), half dropped (5) → mean 3.
        assert!((out.mean_all().unwrap().to_scalar::<f32>().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn step_blend_noises_only_outside() {
        // Released mask (all 1) → running is untouched regardless of the noise level.
        let d = Device::Cpu;
        let running = Tensor::full(2.0f32, (1, 4, 4, 4), &d).unwrap();
        let clean = Tensor::full(7.0f32, (1, 4, 4, 4), &d).unwrap();
        let noise = Tensor::full(0.0f32, (1, 4, 4, 4), &d).unwrap();
        let ones = Tensor::ones((1, 1, 4, 4), DType::F32, &d).unwrap();
        let space = FlowSpace::new(vec![0.5], geom());
        let out = step_blend(&space, &running, &ones, &clean, &noise, 0).unwrap();
        assert!((out.mean_all().unwrap().to_scalar::<f32>().unwrap() - 2.0).abs() < 1e-6, "released mask keeps running");
        // Fully masked-out (all 0) at σ=0 → the clean original (7).
        let zeros = Tensor::zeros((1, 1, 4, 4), DType::F32, &d).unwrap();
        let space0 = FlowSpace::new(vec![0.0], geom());
        let out0 = step_blend(&space0, &running, &zeros, &clean, &noise, 0).unwrap();
        assert!((out0.mean_all().unwrap().to_scalar::<f32>().unwrap() - 7.0).abs() < 1e-6, "outside → clean original at σ=0");
    }
}
