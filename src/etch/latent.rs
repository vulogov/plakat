//! L2 — **latent etch** (RFC ETCH-1 §L2), Tree-Ring style. Because plakat owns the sampler, the mark
//! goes into the **initial latent `z_T`** rather than onto pixels: a key-derived pattern written into
//! concentric **Fourier rings** of one latent channel, so the sampling trajectory *amplifies* it into
//! the image's global structure (semantic, not a residual). Detection needs DDIM inversion back to `z_T`
//! then a ring correlation — so it lives behind `doctor --if-plakat --verify` (a model load).
//!
//! This module is the pure **ring codec** (embed/correlate on a latent tensor) — testable without a
//! model. The write-wiring (into the sampler's `z_T`) and DDIM-inversion detection are wired in the
//! pipeline; capacity is a presence bit + a ~16-bit `EtchId` prefix, not the full 64 (RFC §L2 "Capacity").

use super::EtchId;
use candle_core::{IndexOp, Result, Tensor};
use sha2::{Digest, Sha256};

/// Rings carrying id bits (the low-frequency-adjacent bands). The rest of the marked bands form a fixed
/// key presence pattern.
pub const ID_RINGS: usize = 16;
/// Total marked rings (id rings + presence rings).
pub const RINGS: usize = 24;
/// Ring amplitude on the latent spectrum (latents ~N(0,1); |FFT| ~ sqrt(HW) ≈ 64 at 64², so this
/// dominates the marked bands).
const AMP: f32 = 96.0;

/// L2's publisher tag: a key-derived 64-bit value whose top `ID_RINGS` bits ride the id-rings. L2 carries
/// presence + this key tag (not the render-specific `EtchId`, which L0/L1/L3 hold) so the sampler needs
/// no recipe — verification confirms "a plakat latent made with this key".
pub fn key_tag(key: &str) -> EtchId {
    let mut h = Sha256::new();
    h.update(b"plakat-etch-l2-tag");
    h.update(key.as_bytes());
    let d = h.finalize();
    EtchId(u64::from_be_bytes(d[..8].try_into().unwrap()))
}

/// The frequency-radius band of DFT bin `(k,l)` in an `H×W` **unshifted** spectrum (low freq wraps to the
/// corners, so use the min-distance-to-edge per axis).
fn ring_of(k: usize, l: usize, h: usize, w: usize) -> usize {
    let fk = k.min(h - k) as f32;
    let fl = l.min(w - l) as f32;
    (fk * fk + fl * fl).sqrt().round() as usize
}

/// The target real coefficient for ring `r` given the payload: id rings carry a bit of `id`, presence
/// rings carry a fixed key-derived sign. `±AMP`.
fn ring_target(key: &str, id: EtchId, r: usize) -> f32 {
    let bit = if r < ID_RINGS {
        (id.0 >> (63 - r)) & 1 == 1 // the r-th (MSB-first) id bit → this ring
    } else {
        // presence ring: a key-derived pseudo-random sign.
        let mut h = Sha256::new();
        h.update(key.as_bytes());
        h.update((r as u64).to_be_bytes());
        h.finalize()[0] & 1 == 1
    };
    if bit { AMP } else { -AMP }
}

/// Extract channel 0's `H×W` plane from `(re, im)` as flat `Vec<f32>`.
fn plane(t: &Tensor) -> Result<Vec<f32>> {
    t.i((0, 0))?.flatten_all()?.to_vec1::<f32>()
}

/// Embed the ring pattern into `latent` (a `(1,C,H,W)` initial noise). Marks channel 0's spectrum, then
/// inverse-FFTs → the marked latent. Other channels untouched.
pub fn embed_rings(latent: &Tensor, key: &str, id: EtchId) -> Result<Tensor> {
    let (b, c, h, w) = latent.dims4()?;
    let latent_f = latent.to_dtype(candle_core::DType::F32)?;
    let (re, im) = crate::pipelines::fft::fft2_real(&latent_f)?;
    let mut re_v = plane(&re)?;
    let mut im_v = plane(&im)?;
    for k in 0..h {
        for l in 0..w {
            let r = ring_of(k, l, h, w);
            if r < RINGS {
                re_v[k * w + l] = ring_target(key, id, r);
                im_v[k * w + l] = 0.0;
            }
        }
    }
    // rebuild the modified channel-0 planes, keep the other channels' spectra unchanged.
    let dev = latent.device();
    let re_new = splice_channel0(&re, &re_v, b, c, h, w, dev)?;
    let im_new = splice_channel0(&im, &im_v, b, c, h, w, dev)?;
    let marked = crate::pipelines::fft::ifft2_real(&re_new, &im_new)?;
    // replace channel 0 of the latent with the marked plane; keep channels 1.. as-is.
    let marked0 = marked.i((0, 0))?; // (H,W)
    splice_channel0_spatial(&latent_f, &marked0, b, c, h, w, dev)?.to_dtype(latent.dtype())
}

/// Replace channel 0's `H×W` plane of a `(B,C,H,W)` spectrum tensor with `plane_v`.
fn splice_channel0(full: &Tensor, plane_v: &[f32], b: usize, c: usize, h: usize, w: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let ch0 = Tensor::from_vec(plane_v.to_vec(), (1, 1, h, w), dev)?;
    if c == 1 {
        return Ok(ch0);
    }
    let rest = full.narrow(1, 1, c - 1)?; // channels 1..c
    Tensor::cat(&[&ch0, &rest], 1)?.reshape((b, c, h, w))
}
/// As above but for a spatial-domain `(B,C,H,W)` and a `(H,W)` replacement plane.
fn splice_channel0_spatial(full: &Tensor, plane0: &Tensor, b: usize, c: usize, h: usize, w: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let ch0 = plane0.reshape((1, 1, h, w))?;
    let _ = dev;
    if c == 1 {
        return Ok(ch0);
    }
    let rest = full.narrow(1, 1, c - 1)?;
    Tensor::cat(&[&ch0, &rest], 1)?.reshape((b, c, h, w))
}

/// Correlate a latent's channel-0 rings against the key pattern → (presence `[0,1]`, recovered `EtchId`).
/// `presence` is the fraction of marked rings whose sign matches the expected pattern (0.5 = chance).
pub fn correlate_rings(latent: &Tensor, key: &str) -> Result<(f32, EtchId)> {
    let (_b, _c, h, w) = latent.dims4()?;
    let latent_f = latent.to_dtype(candle_core::DType::F32)?;
    let (re, _im) = crate::pipelines::fft::fft2_real(&latent_f)?;
    let re_v = plane(&re)?;
    // average the real coefficient per ring.
    let mut sum = vec![0f64; RINGS];
    let mut cnt = vec![0u32; RINGS];
    for k in 0..h {
        for l in 0..w {
            let r = ring_of(k, l, h, w);
            if r < RINGS {
                sum[r] += re_v[k * w + l] as f64;
                cnt[r] += 1;
            }
        }
    }
    let mut id: u64 = 0;
    let mut matches = 0usize;
    let mut total = 0usize;
    for r in 0..RINGS {
        if cnt[r] == 0 {
            continue;
        }
        let mean = sum[r] / cnt[r] as f64;
        let bit = mean > 0.0;
        if r < ID_RINGS {
            id = (id << 1) | bit as u64;
        }
        // expected sign for presence-consistency: compare to the key pattern (id unknown at read time for
        // presence rings; for id rings we can't check without the id, so presence uses the presence rings).
        if r >= ID_RINGS {
            let expected = ring_target(key, EtchId(0), r) > 0.0; // id irrelevant for presence rings
            if bit == expected {
                matches += 1;
            }
            total += 1;
        }
    }
    // the id occupies the top ID_RINGS bits; shift into the high 64-bit positions (prefix).
    let id = id << (64 - ID_RINGS);
    let presence = if total > 0 { matches as f32 / total as f32 } else { 0.0 };
    Ok((presence, EtchId(id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn noise(h: usize, w: usize, c: usize, seed: u64) -> Tensor {
        // deterministic pseudo-Gaussian-ish latent.
        let n = c * h * w;
        let v: Vec<f32> = (0..n).map(|i| (((seed.wrapping_mul(2654435761).wrapping_add(i as u64)) % 2000) as f32 / 1000.0) - 1.0).collect();
        Tensor::from_vec(v, (1, c, h, w), &Device::Cpu).unwrap().to_dtype(DType::F32).unwrap()
    }

    #[test]
    fn embed_then_correlate_recovers_presence_and_id_prefix() {
        let z = noise(64, 64, 4, 1);
        let id = EtchId(0xabcd_0000_0000_0000); // top 16 bits carry through ID_RINGS
        let marked = embed_rings(&z, "k", id).unwrap();
        let (presence, rid) = correlate_rings(&marked, "k").unwrap();
        assert!(presence > 0.95, "presence {presence} should be ~1 on the just-marked latent");
        // the top ID_RINGS bits of the recovered id match the embedded prefix.
        let mask = !0u64 << (64 - ID_RINGS);
        assert_eq!(rid.0 & mask, id.0 & mask, "id prefix recovered: {} vs {}", rid.hex(), id.hex());
    }

    #[test]
    fn unmarked_latent_reads_low_presence() {
        let z = noise(64, 64, 4, 7);
        let (presence, _) = correlate_rings(&z, "k").unwrap();
        assert!(presence < 0.85, "an unmarked latent should be near chance, got {presence}");
    }
}
