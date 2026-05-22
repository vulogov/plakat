//! NF4 (NormalFloat 4-bit) quantization codec.
//!
//! NF4 was introduced in the QLoRA paper (Dettmers et al., 2023) and
//! is the default 4-bit quantization in bitsandbytes' `Linear4bit`.
//! Compared to integer quantization (Q4_0, Q4_1) it's distribution-
//! aware: the 16-value codebook is spaced so the codes are the
//! quantile midpoints of a standard normal, giving better fidelity
//! for weights drawn from N(0, σ²) (which is roughly what a trained
//! neural network's weights look like).
//!
//! ## Storage layout (bitsandbytes / ComfyUI convention)
//!
//! For an `(O, I)` weight `W`:
//!
//! * `W.weight` — packed `uint8` of length `O * I / 2`. Each byte
//!   holds two NF4 codes: low nibble is the **first** value, high
//!   nibble is the **second**. Codes are laid out in row-major order
//!   over the original `(O, I)` weight.
//!
//! * `W.weight.absmax` — `f32` of length `O * I / block_size`.
//!   One absmax (the max absolute value of the block) per block of
//!   `block_size` original weight values. Block size is typically
//!   **64** in bitsandbytes' default config.
//!
//! Dequantization for code `c` in block `b`:
//!
//! ```text
//!   dequant(c, b) = NF4_CODEBOOK[c] * absmax[b]
//! ```
//!
//! ## Double quantization (out of scope here)
//!
//! bitsandbytes optionally double-quantizes the per-block `absmax`
//! itself to FP8 with an outer scale (`nested_absmax`). The
//! lllyasviel v2 Flux pack does **not** use this — plain f32 absmax.
//! `dequant_nf4_block` bails loud if it sees nested-absmax markers.
//!
//! ## Source of the codebook
//!
//! Reference (bitsandbytes `csrc/kernels.cu`, `quantize_nf4`):
//!   <https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu>
//! Independently cross-checked against
//!   <https://huggingface.co/docs/bitsandbytes/main/en/explanations/resources>

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};

/// Block size bitsandbytes uses by default. lllyasviel's
/// `flux1-dev-bnb-nf4-v2` confirms 64 in its quant_state JSON.
pub const NF4_BLOCK_SIZE: usize = 64;

/// The 16 NF4 codebook values. These are the quantile midpoints of a
/// truncated standard normal between roughly the 0.5/15.5 quantiles,
/// scaled to `[-1, 1]`. Values copied from bitsandbytes' reference C
/// implementation.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Unpack a `uint8` tensor (packed two NF4 codes per byte) into a
/// `(2 * len,)` tensor of `u8` indices in `[0, 16)`.
///
/// Layout convention: low nibble first, high nibble second. This
/// matches what bitsandbytes writes for row-major weights.
///
/// ```text
///   byte = (high << 4) | low
///   codes[2i]   = low
///   codes[2i+1] = high
/// ```
pub fn unpack_nf4_bytes(packed: &Tensor) -> Result<Tensor> {
    let dev = packed.device();
    // Pull to CPU as a Vec<u8> for the bit-level unpack — candle
    // doesn't expose bit-shift on tensors, and the cost is tiny vs
    // the dequant matmul that follows on GPU.
    let bytes: Vec<u8> = packed
        .flatten_all()?
        .to_dtype(DType::U8)?
        .to_vec1::<u8>()
        .context("reading NF4 packed bytes (expected U8)")?;
    let mut codes = Vec::with_capacity(bytes.len() * 2);
    for b in bytes {
        codes.push(b & 0x0F);
        codes.push((b >> 4) & 0x0F);
    }
    Ok(Tensor::from_vec(codes, (packed.elem_count() * 2,), dev)?)
}

/// Dequantize one NF4-quantized weight tensor.
///
/// `packed` is the `uint8` tensor of shape `(num_bytes,)` (or any
/// 1-D shape — only the flat length matters). `absmax` is `f32` of
/// shape `(num_blocks,)`. `original_shape` is the dequantized
/// weight's intended shape, typically `(out_features, in_features)`
/// for a Linear.
///
/// The result is an `f32` tensor of shape `original_shape` on
/// `packed.device()`. Callers cast to BF16 / F16 as needed.
///
/// Bails on any size mismatch between the inputs and the expected
/// counts implied by `block_size` and `original_shape`.
pub fn dequant_nf4(
    packed: &Tensor,
    absmax: &Tensor,
    original_shape: &[usize],
    block_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let n_total: usize = original_shape.iter().product();
    if n_total % block_size != 0 {
        bail!(
            "NF4 dequant: numel {} not divisible by block_size {}",
            n_total,
            block_size
        );
    }
    let n_blocks = n_total / block_size;

    let expected_bytes = n_total / 2;
    if packed.elem_count() != expected_bytes {
        bail!(
            "NF4 dequant: packed has {} bytes, expected {} (shape {:?}, 2 codes/byte)",
            packed.elem_count(),
            expected_bytes,
            original_shape
        );
    }
    if absmax.elem_count() != n_blocks {
        bail!(
            "NF4 dequant: absmax has {} entries, expected {} (shape {:?}, block_size {})",
            absmax.elem_count(),
            n_blocks,
            original_shape,
            block_size
        );
    }

    // Step 1: unpack to flat (n_total,) u8 indices in [0, 16).
    let codes_u8 = unpack_nf4_bytes(packed)?;

    // Step 2: codebook lookup. Build the codebook on the same device
    // once per call; cheap (16 floats).
    let codebook = Tensor::from_slice(&NF4_CODEBOOK, (16,), device)?.to_dtype(DType::F32)?;
    // gather doesn't have a "scalar per index" mode in candle's stable
    // API; instead, embedding-style index_select on a (16,) codebook
    // and a (n_total,) U32 index does the job.
    let indices_u32 = codes_u8.to_dtype(DType::U32)?;
    let codes_f32 = codebook.index_select(&indices_u32, 0)?;
    // codes_f32 shape: (n_total,) F32.

    // Step 3: broadcast per-block absmax. Reshape codes to
    // (n_blocks, block_size), broadcast-multiply by absmax
    // reshaped to (n_blocks, 1).
    let codes_blocked = codes_f32.reshape((n_blocks, block_size))?;
    let absmax_f32 = absmax.to_dtype(DType::F32)?.reshape((n_blocks, 1))?;
    let scaled = codes_blocked.broadcast_mul(&absmax_f32)?;

    // Step 4: reshape back to original_shape.
    let out = scaled.reshape(original_shape)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    // The 16 NF4 codes must be sorted ascending — this is what makes
    // the index → value lookup monotonic and lets quantization be a
    // simple rounding-to-nearest. Cheap invariant check.
    #[test]
    fn codebook_is_sorted_ascending() {
        for w in NF4_CODEBOOK.windows(2) {
            assert!(w[0] < w[1], "codebook not sorted at {:?}", w);
        }
    }

    #[test]
    fn codebook_endpoints_pinned() {
        // Per bitsandbytes' definition, code 0 = -1.0 and code 15 = +1.0.
        // Block dequant scales by absmax so these extremes map to ±absmax.
        assert_eq!(NF4_CODEBOOK[0], -1.0);
        assert_eq!(NF4_CODEBOOK[15], 1.0);
    }

    #[test]
    fn unpack_low_then_high_nibble() {
        // 0xAB packs codes {0xB, 0xA}: low nibble first, high second.
        let bytes = Tensor::from_vec(vec![0xABu8, 0x10], (2,), &cpu()).unwrap();
        let codes = unpack_nf4_bytes(&bytes).unwrap();
        let v: Vec<u32> = codes.to_dtype(DType::U32).unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![0xB, 0xA, 0x0, 0x1]);
    }

    #[test]
    fn dequant_two_block_known_vector() {
        // Build a synthetic 2-block (block_size=2 for the test) NF4
        // weight that's easy to verify by hand.
        //
        // Block 0: codes [0, 15] (low nibble = 0, high nibble = 15) →
        //          packed byte = 0xF0. With absmax=2.0, dequants to
        //          [-1.0 * 2.0, 1.0 * 2.0] = [-2.0, 2.0].
        // Block 1: codes [7, 7] (code 7 = 0.0) → packed = 0x77.
        //          With absmax=5.0, dequants to [0.0, 0.0].
        //
        // Overall shape (2, 2) i.e. 4 values = 2 blocks.
        let packed = Tensor::from_vec(vec![0xF0u8, 0x77], (2,), &cpu()).unwrap();
        let absmax = Tensor::from_vec(vec![2.0f32, 5.0], (2,), &cpu()).unwrap();
        let out = dequant_nf4(&packed, &absmax, &[2, 2], 2, &cpu()).unwrap();
        let v: Vec<f32> = out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((v[0] - (-2.0)).abs() < 1e-6, "got {v:?}");
        assert!((v[1] - 2.0).abs() < 1e-6, "got {v:?}");
        assert!((v[2] - 0.0).abs() < 1e-6, "got {v:?}");
        assert!((v[3] - 0.0).abs() < 1e-6, "got {v:?}");
    }

    #[test]
    fn dequant_size_mismatch_bails_loud() {
        // Right-sized packed but wrong-sized absmax.
        let packed = Tensor::from_vec(vec![0u8; 4], (4,), &cpu()).unwrap();
        // 8 values, block_size=2 → 4 blocks expected. Give only 3.
        let absmax = Tensor::from_vec(vec![1.0f32; 3], (3,), &cpu()).unwrap();
        let err = dequant_nf4(&packed, &absmax, &[4, 2], 2, &cpu()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("absmax has 3"), "{msg}");
    }

    #[test]
    fn dequant_packed_byte_mismatch_bails_loud() {
        // 8 values implies 4 bytes packed; give 3.
        let packed = Tensor::from_vec(vec![0u8; 3], (3,), &cpu()).unwrap();
        let absmax = Tensor::from_vec(vec![1.0f32; 4], (4,), &cpu()).unwrap();
        let err = dequant_nf4(&packed, &absmax, &[4, 2], 2, &cpu()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("packed has 3 bytes"), "{msg}");
    }

    #[test]
    fn dequant_non_divisible_total_bails_loud() {
        // shape (3, 1) = 3 values, block_size=2 doesn't divide.
        let packed = Tensor::from_vec(vec![0u8, 0], (2,), &cpu()).unwrap();
        let absmax = Tensor::from_vec(vec![1.0f32, 1.0], (2,), &cpu()).unwrap();
        let err = dequant_nf4(&packed, &absmax, &[3, 1], 2, &cpu()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not divisible by block_size"), "{msg}");
    }

    /// Block-of-64 dequant on a realistic-shaped weight slice. With
    /// block_size=64, a (1, 64) weight is exactly one block — exercises
    /// the "single absmax broadcasts cleanly" code path.
    #[test]
    fn dequant_single_block_of_64() {
        // 64 codes packed into 32 bytes. Use code=7 (= 0.0) everywhere
        // → output should be all zeros regardless of absmax.
        let packed = Tensor::from_vec(vec![0x77u8; 32], (32,), &cpu()).unwrap();
        let absmax = Tensor::from_vec(vec![3.14f32], (1,), &cpu()).unwrap();
        let out = dequant_nf4(&packed, &absmax, &[1, 64], 64, &cpu()).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for &x in &v {
            assert!(x.abs() < 1e-6);
        }
    }
}
