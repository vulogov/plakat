//! `LayeredHook` — the S3 guide anchor (RFC §S3 / §Finish). It is the `StepHook` the finish pipeline runs;
//! its `refine_latent` (the seam wired across every family in P0) steers the running latent toward the
//! low-frequency guide during the early, per-layer windows:
//!
//! ```text
//! x ← x + m(f) ⊙ ( LP(ŷ) − LP(x) )
//! ŷ = NoiseSpace::noise_to(G, n_guide, step)          // the guide, forward-noised to the current level
//! m(f) = W ⊙ ramp(f; E),  ramp(f;E)=clamp((E−f)/r,0,1) // per-pixel weight × a window that closes at E
//! ```
//!
//! Only LOW frequencies are substituted (`LP`), and only inside each pixel's window (`f < E`), so layout
//! and binding are anchored while texture/detail (high frequencies) remain the model's own — premise D1/D3.
//! Everything is done in SPATIAL latent space via `to_spatial`/`from_spatial`, so the same hook drives the
//! UNet, DiT, Flux (packed tokens) and Cascade families unchanged. Chains to an inner hook so TUI
//! progress/cancel keep working.

use candle_core::{Result, Tensor};

use crate::pipelines::lowpass::lowpass;
use crate::pipelines::noise_space::NoiseSpace;
use crate::pipelines::step_hook::{StepControl, StepHook};

/// The guide-anchor hook. Holds the guide latent + fixed guide noise (SPATIAL, `(1,C,H,W)`) and the
/// per-pixel weight / window-end maps (`(1,1,H,W)`, latent resolution). `ramp` is the window-close ramp `r`.
pub struct LayeredHook<'a> {
    guide: Tensor,
    noise: Tensor,
    weight: Tensor,
    window_end: Tensor,
    ramp: f32,
    /// The largest window end across the maps — once `f` passes it, anchoring is finished (early-out).
    max_window: f32,
    inner: Option<&'a mut dyn StepHook>,
}

impl<'a> LayeredHook<'a> {
    /// Build the hook. `guide`/`noise` are `(1,C,H,W)` spatial latents; `weight`/`window_end` are `(1,1,H,W)`
    /// in `[0,1]`; `ramp` (`r`, default 0.1) is the per-pixel window-close ramp. `inner` chains progress/cancel.
    pub fn new(
        guide: Tensor,
        noise: Tensor,
        weight: Tensor,
        window_end: Tensor,
        ramp: f32,
        inner: Option<&'a mut dyn StepHook>,
    ) -> Result<Self> {
        // The anchor math runs in F32 (see `refine_latent`); normalise the inputs here so `noise_to` and the
        // mask operate on matching dtypes regardless of the sampler's latent dtype.
        let f32 = candle_core::DType::F32;
        let (guide, noise) = (guide.to_dtype(f32)?, noise.to_dtype(f32)?);
        let (weight, window_end) = (weight.to_dtype(f32)?, window_end.to_dtype(f32)?);
        let max_window = window_end.max_all()?.to_scalar::<f32>()?;
        Ok(Self { guide, noise, weight, window_end, ramp: ramp.max(1e-4), max_window, inner })
    }

    /// `m(f) = W ⊙ clamp((E − f)/r, 0, 1)` — the per-pixel anchor strength at step fraction `f`.
    fn mask(&self, f: f32) -> Result<Tensor> {
        // ramp = clamp((E − f) / r, 0, 1)
        let ramp = self
            .window_end
            .affine(1.0, -(f as f64))? // E − f
            .affine(1.0 / self.ramp as f64, 0.0)? // / r
            .clamp(0.0, 1.0)?;
        self.weight.mul(&ramp)
    }
}

impl StepHook for LayeredHook<'_> {
    fn on_step(&mut self, step: usize, total: usize) -> StepControl {
        match &mut self.inner {
            Some(h) => h.on_step(step, total),
            None => StepControl::Continue,
        }
    }

    fn wants_preview(&self, step: usize, total: usize) -> bool {
        self.inner.as_ref().map(|h| h.wants_preview(step, total)).unwrap_or(false)
    }

    fn on_preview(&mut self, step: usize, image: image::RgbImage) {
        if let Some(h) = &mut self.inner {
            h.on_preview(step, image);
        }
    }

    fn is_cancelled(&self) -> bool {
        self.inner.as_ref().map(|h| h.is_cancelled()).unwrap_or(false)
    }

    fn refine_latent(&mut self, step: usize, total: usize, space: &dyn NoiseSpace, latent: &Tensor) -> Result<Option<Tensor>> {
        use candle_core::DType;
        let f = step as f32 / total.max(1) as f32;
        // Past every window → the model runs free (no anchoring).
        if f >= self.max_window {
            return Ok(None);
        }
        let levels = space.geometry().pool_levels;
        let ld = latent.dtype();

        // All anchor math runs in F32: the running latent is often F16 (Metal), and the low-pass pyramid
        // (avg-pool + bilinear) is only reliable in F32 there. `guide`/`noise`/`weight`/`window_end` are
        // already F32 (built by the guide stage). The result is cast back to the sampler's own dtype.
        // Work in spatial latent space (identity for most families; Flux unpacks its token sequence).
        let xs = space.to_spatial(latent)?.to_dtype(DType::F32)?;
        // ŷ = the guide forward-noised to the current level (linear/`add_noise`, so spatial G/noise are fine).
        let ys = space.noise_to(&self.guide, &self.noise, step)?.to_dtype(DType::F32)?;

        // Substitute LOW frequencies only: correction = LP(ŷ) − LP(x).
        let corr = (lowpass(&ys, levels)? - lowpass(&xs, levels)?)?;
        // Apply masked by W ⊙ ramp(f;E), broadcasting the (1,1,H,W) mask over the channels.
        let m = self.mask(f)?.to_dtype(DType::F32)?;
        let xs2 = (xs + corr.broadcast_mul(&m)?)?;

        Ok(Some(space.from_spatial(&xs2.to_dtype(ld)?)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::noise_space::{FlowSpace, LatentGeometry};
    use candle_core::{DType, Device};

    fn geom() -> LatentGeometry {
        LatentGeometry { v: 8, u: 8, pool_levels: 2 }
    }

    /// A `(1,1,H,W)` map filled with a constant.
    fn map(v: f32, h: usize, w: usize, d: &Device) -> Tensor {
        Tensor::full(v, (1, 1, h, w), d).unwrap()
    }

    /// LP energy — mean square of the low-passed spatial tensor.
    fn lp_of(t: &Tensor, l: usize) -> Tensor {
        lowpass(t, l).unwrap()
    }

    #[test]
    fn anchors_low_freq_to_guide_preserving_high_freq() {
        let d = Device::Cpu;
        let (h, w) = (16usize, 16usize);
        // Guide G: a smooth horizontal gradient (all low-frequency).
        let mut gv = vec![0f32; 4 * h * w];
        for c in 0..4 {
            for y in 0..h {
                for x in 0..w {
                    gv[c * h * w + y * w + x] = x as f32 / w as f32;
                }
            }
        }
        let guide = Tensor::from_vec(gv, (1, 4, h, w), &d).unwrap();
        // Running latent x: a DIFFERENT smooth field + a high-frequency checkerboard.
        let mut xv = vec![0f32; 4 * h * w];
        for c in 0..4 {
            for y in 0..h {
                for x in 0..w {
                    let low = 1.0 - y as f32 / h as f32; // different low structure
                    let hi = if (x + y) % 2 == 0 { 0.3 } else { -0.3 }; // high-freq detail
                    xv[c * h * w + y * w + x] = low + hi;
                }
            }
        }
        let x = Tensor::from_vec(xv, (1, 4, h, w), &d).unwrap();

        // FlowSpace with σ=0 → noise_to(G, noise, 0) = G exactly (identity to_spatial). Full window (E=1),
        // weight 1, so m = 1 at f = 0.
        let space = FlowSpace::new(vec![0.0], geom());
        let noise = Tensor::zeros((1, 4, h, w), DType::F32, &d).unwrap();
        let mut hook = LayeredHook::new(guide.clone(), noise, map(1.0, h, w, &d), map(1.0, h, w, &d), 0.1, None).unwrap();

        let out = hook.refine_latent(0, 10, &space, &x).unwrap().expect("anchored");
        let l = geom().pool_levels;

        // EXACT: with m = 1 and ŷ = G, the change is precisely the low-frequency correction LP(G) − LP(x)
        // — a purely low-pass tensor, so the model's high-frequency detail is left untouched.
        let delta = (&out - &x).unwrap();
        let expected = (&lp_of(&guide, l) - &lp_of(&x, l)).unwrap();
        let dd = (&delta - &expected).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(dd < 1e-5, "change == LP(G) − LP(x) (max diff {dd})");

        // DIRECTION: the result's low band is closer to the guide's than x's was.
        let before = (&lp_of(&x, l) - &lp_of(&guide, l)).unwrap().abs().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let after = (&lp_of(&out, l) - &lp_of(&guide, l)).unwrap().abs().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(after < before, "anchor moved the low band toward the guide (after {after} < before {before})");
    }

    #[test]
    fn no_change_past_the_window() {
        let d = Device::Cpu;
        let (h, w) = (8usize, 8usize);
        let guide = Tensor::randn(0f32, 1f32, (1, 4, h, w), &d).unwrap();
        let noise = Tensor::zeros((1, 4, h, w), DType::F32, &d).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 4, h, w), &d).unwrap();
        let space = FlowSpace::new(vec![0.0], geom());
        // Window ends at 0.3 everywhere → at f = 0.5 the hook returns None (model runs free).
        let mut hook = LayeredHook::new(guide, noise, map(1.0, h, w, &d), map(0.3, h, w, &d), 0.1, None).unwrap();
        assert!(hook.refine_latent(5, 10, &space, &x).unwrap().is_none(), "past the window → no anchoring");
        // Inside the window (f = 0.0) it does act.
        assert!(hook.refine_latent(0, 10, &space, &x).unwrap().is_some(), "inside the window → anchors");
    }

    #[test]
    fn zero_weight_is_a_noop() {
        let d = Device::Cpu;
        let (h, w) = (8usize, 8usize);
        let guide = Tensor::randn(0f32, 1f32, (1, 4, h, w), &d).unwrap();
        let noise = Tensor::zeros((1, 4, h, w), DType::F32, &d).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 4, h, w), &d).unwrap();
        let space = FlowSpace::new(vec![0.0], geom());
        // Weight 0 everywhere → the mask is 0 → the latent is unchanged even inside the window.
        let mut hook = LayeredHook::new(guide, noise, map(0.0, h, w, &d), map(1.0, h, w, &d), 0.1, None).unwrap();
        let out = hook.refine_latent(0, 10, &space, &x).unwrap().unwrap();
        let diff = (&out - &x).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-6, "zero weight leaves the latent unchanged (max diff {diff})");
    }
}
