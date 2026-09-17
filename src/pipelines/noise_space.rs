//! `NoiseSpace` — the per-family integration seam for LAYERED-1 (RFC §Making the finish model-agnostic).
//!
//! The layered anchor needs three family-specific operations on the finish model's latent:
//!   * **`noise_to`** — forward-noise a clean latent (the guide `G`) to the exact level the sampler has
//!     reached after a given step, so it can be compared with the running latent. ε-prediction families use
//!     their scheduler's `add_noise`; flow-matching families interpolate `(1−σ)·clean + σ·noise`.
//!   * **`to_spatial` / `from_spatial`** — map the sampler's latent to a `(B, C, H, W)` spatial latent the
//!     low-pass filter works on, and back. Identity for the UNet/DiT families; Flux packs 2×2 patches into
//!     a token sequence, so its mapping is a real unpack/pack.
//!   * **`geometry`** — the latent/unit/pool sizes the size classifier and filter need.
//!
//! Only the trait + the flow-matching impl live here; the ε-prediction impls wrap each family's scheduler
//! and are constructed inside that family's denoise loop (where the scheduler exists).

use candle_core::{Result, Tensor};

/// Latent geometry of a finish family (RFC §Size classes): `v` = VAE/latent pixel size (8 for the 8× VAE
/// families, 32 for Sana), `u` = denoised-unit size (8 UNet cell, 16 for a 2×2-patchified DiT token, 32 for
/// Sana), `pool_levels` = the low-pass pyramid depth `L`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LatentGeometry {
    pub v: usize,
    pub u: usize,
    pub pool_levels: usize,
}

/// The family-specific latent operations the layered anchor and repair passes need.
pub trait NoiseSpace {
    /// Forward-noise `clean` to the level the sampler has reached AFTER `step`, using `noise` (a fixed
    /// tensor for reproducibility). At the final step this returns ≈ `clean`.
    fn noise_to(&self, clean: &Tensor, noise: &Tensor, step: usize) -> Result<Tensor>;

    /// Sampler latent → spatial `(B, C, H, W)` latent the low-pass filter works on. Identity for most
    /// families; Flux unpacks its 2×2-patch token sequence.
    fn to_spatial(&self, latent: &Tensor) -> Result<Tensor>;

    /// Inverse of [`to_spatial`](Self::to_spatial).
    fn from_spatial(&self, spatial: &Tensor) -> Result<Tensor>;

    /// The family descriptor for the size classifier and the filter.
    fn geometry(&self) -> LatentGeometry;
}

/// Flow-matching noise space (SD 3/3.5, Flux, Sana): the level after step `k` is the sigma `sigmas[k]`, and
/// forward-noising is the flow interpolation `x = (1 − σ)·clean + σ·noise`. `sigmas` is the per-step level
/// the sampler moves TO (decreasing to ~0 at the last step); `to_spatial`/`from_spatial` are supplied by the
/// caller so Flux can plug in its unpack/pack while the DiT families use the identity.
pub struct FlowSpace {
    sigmas: Vec<f32>,
    geom: LatentGeometry,
}

impl FlowSpace {
    pub fn new(sigmas: Vec<f32>, geom: LatentGeometry) -> Self {
        Self { sigmas, geom }
    }

    /// The sigma level after `step` (clamped to `[0,1]`; past the schedule end = 0 = clean).
    fn sigma_after(&self, step: usize) -> f32 {
        self.sigmas.get(step).copied().unwrap_or(0.0).clamp(0.0, 1.0)
    }
}

impl NoiseSpace for FlowSpace {
    fn noise_to(&self, clean: &Tensor, noise: &Tensor, step: usize) -> Result<Tensor> {
        let s = self.sigma_after(step) as f64;
        // (1 − s)·clean + s·noise
        (clean * (1.0 - s))? + (noise * s)?
    }

    fn to_spatial(&self, latent: &Tensor) -> Result<Tensor> {
        Ok(latent.clone())
    }

    fn from_spatial(&self, spatial: &Tensor) -> Result<Tensor> {
        Ok(spatial.clone())
    }

    fn geometry(&self) -> LatentGeometry {
        self.geom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    #[test]
    fn flow_noise_endpoints_and_midpoint() {
        let d = Device::Cpu;
        let clean = Tensor::full(2.0f32, (1, 4, 4, 4), &d).unwrap();
        let noise = Tensor::full(10.0f32, (1, 4, 4, 4), &d).unwrap();
        let fs = FlowSpace::new(vec![1.0, 0.5, 0.0], geom());
        // step 0 → σ=1 → pure noise; step 2 → σ=0 → clean; step 1 → σ=0.5 → midpoint (6.0).
        let at0 = fs.noise_to(&clean, &noise, 0).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let at2 = fs.noise_to(&clean, &noise, 2).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let at1 = fs.noise_to(&clean, &noise, 1).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((at0 - 10.0).abs() < 1e-5, "σ=1 → noise (got {at0})");
        assert!((at2 - 2.0).abs() < 1e-5, "σ=0 → clean (got {at2})");
        assert!((at1 - 6.0).abs() < 1e-5, "σ=0.5 → midpoint (got {at1})");
        // Past the schedule end = clean.
        let past = fs.noise_to(&clean, &noise, 9).unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((past - 2.0).abs() < 1e-5, "past-end → clean");
    }

    #[test]
    fn flow_spatial_is_identity() {
        let d = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &d).unwrap();
        let fs = FlowSpace::new(vec![0.0], geom());
        let round = fs.from_spatial(&fs.to_spatial(&x).unwrap()).unwrap();
        let diff = (&round - &x).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert_eq!(diff, 0.0);
        assert_eq!(fs.geometry(), geom());
    }
}
