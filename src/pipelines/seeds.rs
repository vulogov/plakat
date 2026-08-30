//! v0.34 phase 1: device-aware seed preparation.
//!
//! Background: candle's Metal backend truncates `device.set_seed(u64)`
//! to 32 bits internally. Two seeds that differ only above bit 32
//! alias to the same RNG state — collisions for `--seed >= 2^32`.
//!
//! v0.33 phase 3's reproducibility audit flagged this as 8 ⚠ rows
//! ("GUARANTEED-Metal-u32"). This module is the v0.34 phase 1 fix.
//!
//! Strategy: a SplitMix64 hash mixes all u64 bits down to u32, but
//! only for **Metal seeds that overflow u32**. Seeds < 2^32 pass
//! through unchanged on every backend — existing users with
//! `--seed 12345 --device metal` get byte-identical output.
//!
//! Migration impact:
//! - `--seed N < 2^32`: byte-identical (all backends).
//! - `--seed N >= 2^32` on Metal: previously collided to
//!   `N mod 2^32`; now distinct via SplitMix64.
//! - `--seed N >= 2^32` on CPU / CUDA: unchanged (full u64).

use candle_core::Device;

/// Prepare a seed for `device.set_seed(...)`. On Metal, when the
/// seed exceeds u32, mixes via SplitMix64 + reduces to u32 so the
/// downstream truncation no longer collides. Otherwise returns the
/// seed unchanged.
pub fn prepare_seed(seed: u64, device: &Device) -> u64 {
    let is_metal = matches!(device, Device::Metal(_));
    if is_metal && seed > u32::MAX as u64 {
        splitmix64_to_u32(seed) as u64
    } else {
        seed
    }
}

/// SplitMix64 → reduce to u32. Standard high-quality mixer, used
/// here to spread all 64 input bits across the output before the
/// Metal backend's internal truncation.
fn splitmix64_to_u32(seed: u64) -> u32 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z = z ^ (z >> 31);
    (z & 0xFFFF_FFFF) as u32
}

/// Spherically interpolate (**slerp**) between two init-noise latents for the
/// **subseed / variation-seed** feature (6.25.0 P1). `strength` in `[0,1]`: `0`
/// returns `base` unchanged, `1` returns `sub`. Slerp (not lerp) keeps the blended
/// tensor on the ~N(0,1) hypersphere the sampler expects, so a small strength nudges
/// the composition without washing out contrast — the A1111 `--subseed` behaviour.
///
/// The math runs on the flattened f32 vectors (init noise is tiny — e.g. 4·64·64 =
/// 16 K floats — so a CPU round-trip is free and dodges Metal reduction quirks). Falls
/// back to a plain lerp when the two vectors are nearly (anti)parallel (`sin(omega)→0`),
/// which is where slerp is numerically unstable.
pub fn slerp_latents(
    strength: f32,
    base: &candle_core::Tensor,
    sub: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let t = strength.clamp(0.0, 1.0);
    if t <= f32::EPSILON {
        return Ok(base.clone());
    }
    let shape = base.shape().clone();
    let dtype = base.dtype();
    let device = base.device().clone();
    let a = base.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = sub.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let out = slerp_vecs(t, &a, &b);
    candle_core::Tensor::from_vec(out, shape, &device)?.to_dtype(dtype)
}

/// Element-shared slerp over two equal-length vectors (the numeric core of
/// [`slerp_latents`], split out so it's unit-testable without a device).
fn slerp_vecs(t: f32, a: &[f32], b: &[f32]) -> Vec<f32> {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Cosine of the angle between the two noise vectors.
    let cos = if na > 0.0 && nb > 0.0 { (dot / (na * nb)).clamp(-1.0, 1.0) } else { 0.0 };
    let omega = cos.acos();
    let sin = omega.sin();
    // Near-(anti)parallel → slerp is unstable; lerp is the correct limit.
    if sin.abs() < 1e-4 {
        return a.iter().zip(b).map(|(x, y)| x * (1.0 - t) + y * t).collect();
    }
    let wa = ((1.0 - t) * omega).sin() / sin;
    let wb = (t * omega).sin() / sin;
    a.iter().zip(b).map(|(x, y)| x * wa + y * wb).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slerp_endpoints_and_symmetry() {
        let a = vec![1.0f32, 0.0, 0.5, -0.3, 0.8];
        let b = vec![0.0f32, 1.0, -0.2, 0.4, 0.1];
        // strength 0 → base; strength 1 → sub (within fp tolerance).
        let z = slerp_vecs(0.0, &a, &b);
        assert!(z.iter().zip(&a).all(|(x, y)| (x - y).abs() < 1e-5));
        let one = slerp_vecs(1.0, &a, &b);
        assert!(one.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-4));
        // midpoint is between the two, not equal to either endpoint.
        let mid = slerp_vecs(0.5, &a, &b);
        assert!(mid.iter().zip(&a).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn slerp_parallel_falls_back_to_lerp() {
        // b = 2·a → exactly parallel; slerp would divide by sin(0). Lerp fallback.
        let a = vec![0.2f32, -0.4, 0.6];
        let b: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
        let mid = slerp_vecs(0.5, &a, &b);
        let expect: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x + y) * 0.5).collect();
        assert!(mid.iter().zip(&expect).all(|(x, y)| (x - y).abs() < 1e-5));
    }

    #[test]
    fn cpu_passthrough_low_seed() {
        let d = Device::Cpu;
        assert_eq!(prepare_seed(12345, &d), 12345);
    }

    #[test]
    fn cpu_passthrough_high_seed() {
        // CPU's set_seed accepts the full u64 — no fixup.
        let d = Device::Cpu;
        let high = 0x1_0000_0000;
        assert_eq!(prepare_seed(high, &d), high);
    }

    #[test]
    fn cpu_passthrough_max_seed() {
        let d = Device::Cpu;
        assert_eq!(prepare_seed(u64::MAX, &d), u64::MAX);
    }

    // The Metal arm can't be tested without a Metal device. The
    // SplitMix64 mixer can be tested independently to confirm:
    // - distinct high-bit inputs produce distinct outputs
    // - low-bit input differences propagate through the hash

    #[test]
    fn mixer_distinguishes_high_bit_collisions() {
        // These two seeds alias under simple u32 truncation
        // (both have low-32 = 0).
        let a = 0x1_0000_0000_u64;
        let b = 0x2_0000_0000_u64;
        assert_ne!(splitmix64_to_u32(a), splitmix64_to_u32(b));
    }

    #[test]
    fn mixer_distinguishes_adjacent_high_seeds() {
        let a = 0x1_0000_0000_u64;
        let b = 0x1_0000_0001_u64;
        assert_ne!(splitmix64_to_u32(a), splitmix64_to_u32(b));
    }

    #[test]
    fn mixer_zero_seed_stable() {
        // Zero gets a well-mixed non-zero hash — important because
        // many "uninitialized" RNG paths zero out.
        let h = splitmix64_to_u32(0);
        assert_ne!(h, 0);
    }

    #[test]
    fn mixer_distinguishes_low_seeds_when_hashed() {
        // When the mixer IS applied, adjacent low seeds remain
        // distinct (collision probability ~2^-32 by birthday).
        let a = splitmix64_to_u32(1);
        let b = splitmix64_to_u32(2);
        assert_ne!(a, b);
    }
}
