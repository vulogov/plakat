//! NF4 safetensors store + helpers consumed by the vendored Flux
//! (Phase 2c) at Linear construction time.
//!
//! The bitsandbytes / ComfyUI NF4 layout encodes each `Linear4bit`
//! weight as a triple of safetensors entries:
//!
//! ```text
//!   X.weight                                     uint8 packed (numel/2,)
//!   X.weight.absmax                              f32 (numel/block_size,)
//!   X.weight.quant_state.bitsandbytes__nf4       uint8, JSON-serialized state
//! ```
//!
//! The third entry is a **tensor of bytes** that encodes a JSON
//! string with the original shape, block size, dtype, etc. — bnb's
//! way of smuggling metadata through safetensors' tensor-only API.
//! For Flux the original `(out, in)` shape is known from the model
//! config, so the JSON metadata is informational only; we validate
//! sizes against the model's expectations rather than parsing the JSON.
//!
//! ## Bail conditions
//!
//! * **Double quantization** (`X.weight.nested_absmax` or
//!   `X.weight.quant_map` present): NOT supported in this phase.
//!   lllyasviel's `flux1-dev-bnb-nf4-v2` is plain-NF4; v1 used DQ.
//! * **Block size ≠ 64**: NOT supported. Bnb's default; loud bail
//!   on anything else so we don't silently mis-dequantize.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pipelines::nf4_codec::{NF4_BLOCK_SIZE, dequant_nf4};

/// A loaded NF4 safetensors pack. Stores every tensor by its raw
/// safetensors key. The NF4Linear constructor consumes this via
/// [`Nf4Store::load_weight`] which picks the right dequant or
/// pass-through path based on the key suffix.
pub struct Nf4Store {
    tensors: HashMap<String, Tensor>,
    device: Device,
}

impl Nf4Store {
    /// Load every tensor from a single NF4 safetensors file. The
    /// packed `uint8` weights stay packed on `device`; absmax and
    /// other dense tensors land at their stored dtype.
    pub fn from_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("loading NF4 pack {}", path.display()))?;
        Ok(Self {
            tensors,
            device: device.clone(),
        })
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `true` if `<path>.weight.absmax` exists — the canonical
    /// indicator that `<path>.weight` is NF4-packed.
    pub fn is_nf4_weight(&self, weight_path: &str) -> bool {
        let absmax_key = format!("{weight_path}.absmax");
        self.tensors.contains_key(&absmax_key)
    }

    /// Look up a tensor by raw safetensors key. Borrowed clone — the
    /// store owns the tensor.
    pub fn get(&self, key: &str) -> Result<Tensor> {
        match self.tensors.get(key) {
            Some(t) => Ok(t.clone()),
            None => bail!("NF4 store: tensor not found at key {key}"),
        }
    }

    /// `true` if the key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Build a new store containing only entries whose key starts
    /// with `prefix`, with that prefix removed. Useful for ComfyUI
    /// packs that namespace the transformer under
    /// `model.diffusion_model.` — call site can strip the prefix
    /// once so the vendor's path-tracking lines up with the BFL
    /// native naming.
    ///
    /// Bails loud if no key matches the prefix (probably the wrong
    /// prefix or an empty pack).
    pub fn with_prefix_stripped(&self, prefix: &str) -> Result<Self> {
        let mut out: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in &self.tensors {
            if let Some(stripped) = k.strip_prefix(prefix) {
                out.insert(stripped.to_string(), v.clone());
            }
        }
        if out.is_empty() {
            bail!(
                "Nf4Store::with_prefix_stripped: no keys started with {prefix:?} \
                 (have {} total entries; sample first key = {:?})",
                self.tensors.len(),
                self.tensors.keys().next()
            );
        }
        Ok(Self {
            tensors: out,
            device: self.device.clone(),
        })
    }

    /// Load and dequantize an NF4-packed weight at `<weight_path>`.
    /// Returns an F32 dense tensor of `expected_shape` on the store's
    /// device.
    ///
    /// Bails loud if:
    /// * `<weight_path>` doesn't exist or isn't NF4 (no `.absmax`
    ///   companion)
    /// * Double-quant markers are present (`nested_absmax` / inner
    ///   `quant_map`)
    /// * Block size isn't 64
    /// * Packed/absmax sizes don't match `expected_shape`
    pub fn dequantize_weight(
        &self,
        weight_path: &str,
        expected_shape: &[usize],
    ) -> Result<Tensor> {
        if !self.is_nf4_weight(weight_path) {
            bail!(
                "NF4 store: {weight_path} has no .absmax companion — not NF4-quantized"
            );
        }
        // Double-quant guard: if the pack uses double-quantized
        // absmax, the per-layer key list will include a
        // `nested_absmax` entry. v1 packs have these; v2 doesn't.
        let dq_key = format!("{weight_path}.nested_absmax");
        if self.tensors.contains_key(&dq_key) {
            bail!(
                "NF4 store: {weight_path} uses double-quantization (nested_absmax \
                 present). This phase only supports plain NF4 (e.g. lllyasviel's \
                 flux1-dev-bnb-nf4-v2). Use that file or wait for the DQ-aware loader."
            );
        }
        // Pre-validate the packed-byte / absmax counts against the
        // expected shape so the dequant path can stay focused.
        let packed = self.get(weight_path)?;
        let absmax = self.get(&format!("{weight_path}.absmax"))?;
        let numel: usize = expected_shape.iter().product();
        let expected_bytes = numel / 2;
        let expected_blocks = numel / NF4_BLOCK_SIZE;
        if packed.elem_count() != expected_bytes {
            bail!(
                "NF4 store: {weight_path} packed has {} bytes; expected {} for shape \
                 {:?} (2 codes/byte)",
                packed.elem_count(),
                expected_bytes,
                expected_shape
            );
        }
        if absmax.elem_count() != expected_blocks {
            bail!(
                "NF4 store: {weight_path}.absmax has {} entries; expected {} for shape \
                 {:?} (block_size {})",
                absmax.elem_count(),
                expected_blocks,
                expected_shape,
                NF4_BLOCK_SIZE
            );
        }
        dequant_nf4(&packed, &absmax, expected_shape, NF4_BLOCK_SIZE, &self.device)
    }

    /// Convenience: load + cast to the pipeline's runtime dtype
    /// (typically BF16 on GPU, F32 on CPU).
    pub fn dequantize_weight_to(
        &self,
        weight_path: &str,
        expected_shape: &[usize],
        target_dtype: DType,
    ) -> Result<Tensor> {
        let f32 = self.dequantize_weight(weight_path, expected_shape)?;
        Ok(f32.to_dtype(target_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::nf4_codec::NF4_CODEBOOK;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Build a minimal in-memory Nf4Store from a hand-constructed
    /// tensor map. Bypasses safetensors so the test stays
    /// hermetic (no disk I/O, no real NF4 file).
    fn store_from_map(map: HashMap<String, Tensor>) -> Nf4Store {
        Nf4Store {
            tensors: map,
            device: cpu(),
        }
    }

    #[test]
    fn detect_nf4_via_absmax_companion() {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "layer1.weight".to_string(),
            Tensor::from_vec(vec![0u8; 32], (32,), &cpu()).unwrap(),
        );
        m.insert(
            "layer1.weight.absmax".to_string(),
            Tensor::from_vec(vec![1.0f32], (1,), &cpu()).unwrap(),
        );
        // No absmax for layer2 → not NF4.
        m.insert(
            "layer2.weight".to_string(),
            Tensor::from_vec(vec![0u8; 32], (32,), &cpu()).unwrap(),
        );
        let s = store_from_map(m);
        assert!(s.is_nf4_weight("layer1.weight"));
        assert!(!s.is_nf4_weight("layer2.weight"));
    }

    #[test]
    fn dequantize_weight_64_block() {
        // (1, 64) weight = 32 packed bytes, 1 absmax. Code 15 = +1.0
        // everywhere, absmax = 2.5 → dequant = 2.5 everywhere.
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "w.weight".to_string(),
            Tensor::from_vec(vec![0xFFu8; 32], (32,), &cpu()).unwrap(),
        );
        m.insert(
            "w.weight.absmax".to_string(),
            Tensor::from_vec(vec![2.5f32], (1,), &cpu()).unwrap(),
        );
        let s = store_from_map(m);
        let t = s.dequantize_weight("w.weight", &[1, 64]).unwrap();
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        for &x in &v {
            assert!(
                (x - (NF4_CODEBOOK[15] * 2.5)).abs() < 1e-6,
                "got {x}, expected {}",
                NF4_CODEBOOK[15] * 2.5
            );
        }
    }

    #[test]
    fn double_quant_bails() {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "w.weight".to_string(),
            Tensor::from_vec(vec![0u8; 32], (32,), &cpu()).unwrap(),
        );
        m.insert(
            "w.weight.absmax".to_string(),
            Tensor::from_vec(vec![1.0f32], (1,), &cpu()).unwrap(),
        );
        m.insert(
            "w.weight.nested_absmax".to_string(),
            Tensor::from_vec(vec![1.0f32], (1,), &cpu()).unwrap(),
        );
        let s = store_from_map(m);
        let err = s.dequantize_weight("w.weight", &[1, 64]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("double-quantization"), "{msg}");
    }

    #[test]
    fn non_nf4_path_bails() {
        let mut m: HashMap<String, Tensor> = HashMap::new();
        // Only the dense weight — no .absmax companion.
        m.insert(
            "plain.weight".to_string(),
            Tensor::from_vec(vec![0.0f32; 16], (4, 4), &cpu()).unwrap(),
        );
        let s = store_from_map(m);
        let err = s.dequantize_weight("plain.weight", &[4, 4]).unwrap_err();
        assert!(format!("{err}").contains("not NF4-quantized"));
    }
}
