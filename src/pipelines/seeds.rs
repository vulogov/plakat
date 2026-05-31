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

#[cfg(test)]
mod tests {
    use super::*;

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
