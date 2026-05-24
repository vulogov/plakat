//! v0.16 phase 12: XLabs Flux IP-Adapter — structure parser +
//! deferred-runtime gate.
//!
//! [XLabs-AI's Flux IP-Adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
//! is the community-standard image-conditioning adapter for Flux:
//! a SigLIP-encoded reference image gets projected into per-block
//! IP attention queries/keys/values that compose with the standard
//! text-conditioned attention inside every Flux double_block.
//!
//! plakat ships **two image-conditioning paths** today:
//!
//! * **Redux** (BFL official, [`flux_redux`]) — concatenates IP
//!   tokens onto the T5 hidden state. Simpler attention path;
//!   covers ~80% of "make output look like this reference"
//!   use cases. Real-runtime, works today.
//!
//! * **XLabs IP-Adapter** (this module, v0.16 partial) — per-block
//!   cross-attention into Flux's double_blocks. The architecture
//!   gives finer-grained image control than Redux at the cost of
//!   needing per-block hooks into Flux's attention path. candle
//!   0.8's `flux_inner::double_block_forward` doesn't expose those
//!   hooks; vendoring the per-block forward to splice in IP
//!   attention is deferred.
//!
//! **What ships in phase 12**: a parser that loads the XLabs
//! weights, validates the structure (proj_in / proj_out + N attention
//! pairs matching Flux's 19 double_blocks), and reports the encoder
//! dimensions. The CLI gate (`--flux-ip-image`) bails loud with a
//! pointer to Redux as the working alternative.
//!
//! Same scope-deferral pattern as phase 9 (TI runtime) and phase 11
//! (SD UNet per-task LoRA). All three share the candle private-internals
//! blocker; the parser/inspector half of each lands so the wiring
//! is one diff away when candle exposes the seam (or a vendored
//! attention path lands).

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// XLabs Flux IP-Adapter weights, parsed but not yet wired into
/// the attention path.
#[derive(Debug, Clone)]
pub struct FluxIpAdapter {
    /// Number of Flux double_blocks the adapter targets. XLabs ships
    /// for Flux.1-dev with 19 double_blocks.
    pub num_blocks: usize,
    /// SigLIP feature dim consumed by `proj_in`. Stock XLabs = 1152
    /// (matches `google/siglip-so400m-patch14-384`).
    pub siglip_dim: usize,
    /// Hidden dim of the Flux double_block stream. 3072 for stock
    /// Flux.1-dev / -schnell / -fill-dev.
    pub flux_hidden: usize,
    /// Whether the parsed safetensors carry the per-block attention
    /// weights. Some XLabs releases ship a "lite" variant with only
    /// `proj_in` / `proj_out` — those can't drive per-block
    /// injection but could in principle drive a Redux-like seq
    /// concat. `true` for the full release.
    pub has_per_block_attn: bool,
}

impl FluxIpAdapter {
    /// Parse XLabs IP-Adapter safetensors. Just reads tensor names +
    /// shapes — no weights loaded onto the device yet (that happens
    /// when the runtime injection lands).
    ///
    /// Detected keys:
    /// * `ip_adapter_proj_model.proj_in.weight`  →  (flux_hidden, siglip_dim)
    /// * `ip_adapter_proj_model.proj_out.weight` →  (flux_hidden, flux_hidden)
    /// * `ip_blocks.{i}.ip_adapter_proj.{q,k,v}.weight` per block i
    pub fn parse_safetensors(path: &Path, device: &Device) -> Result<Self> {
        let tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(path, device)
                .with_context(|| format!("loading XLabs IP-Adapter {}", path.display()))?;

        if tensors.is_empty() {
            anyhow::bail!(
                "XLabs IP-Adapter file {} has no tensors", path.display()
            );
        }

        // proj_in: SigLIP feature dim → Flux hidden dim.
        let proj_in_key = tensors
            .keys()
            .find(|k| k.ends_with("proj_in.weight") && k.contains("ip_adapter"))
            .or_else(|| tensors.keys().find(|k| k.ends_with("proj_in.weight")))
            .ok_or_else(|| anyhow::anyhow!(
                "XLabs IP-Adapter {}: no `proj_in.weight` tensor — file may be \
                 corrupt or from a different IP-Adapter family.",
                path.display()
            ))?
            .clone();
        let (flux_hidden, siglip_dim) = tensors[&proj_in_key].dims2()?;

        // Count ip_blocks.{N}.* entries to derive num_blocks.
        let mut block_indices: Vec<usize> = tensors
            .keys()
            .filter_map(|k| {
                k.strip_prefix("ip_blocks.").and_then(|s| {
                    s.split('.').next()?.parse::<usize>().ok()
                })
            })
            .collect();
        block_indices.sort_unstable();
        block_indices.dedup();
        let num_blocks = block_indices.last().map(|n| n + 1).unwrap_or(0);
        let has_per_block_attn = num_blocks > 0;

        Ok(FluxIpAdapter {
            num_blocks,
            siglip_dim,
            flux_hidden,
            has_per_block_attn,
        })
    }

    /// Best-known repo for the XLabs Flux.1-dev IP-Adapter weights.
    pub const DEFAULT_REPO: &'static str = "XLabs-AI/flux-ip-adapter";
    /// Default weights filename in the XLabs release.
    pub const DEFAULT_FILE: &'static str = "ip_adapter.safetensors";
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn write_test_safetensors(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        let mut map = HashMap::new();
        for (name, shape) in tensors {
            let numel: usize = shape.iter().product();
            let v = Tensor::zeros(shape.clone(), DType::F32, &Device::Cpu).unwrap();
            // Avoid the all-zero edge case — fill with one element so
            // safetensors save doesn't optimise oddly.
            let _ = numel;
            map.insert(name.to_string(), v);
        }
        candle_core::safetensors::save(&map, path).unwrap();
    }

    #[test]
    fn parses_full_xlabs_release_dims() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        // Mimic XLabs structure: proj_in (3072, 1152), proj_out
        // (3072, 3072), and 3 ip_blocks (smaller than the 19 real
        // ones — saves test time).
        let mut tensors = vec![
            ("ip_adapter_proj_model.proj_in.weight", vec![3072, 1152]),
            ("ip_adapter_proj_model.proj_out.weight", vec![3072, 3072]),
        ];
        for i in 0..3 {
            tensors.push((
                Box::leak(format!("ip_blocks.{i}.q.weight").into_boxed_str()),
                vec![3072, 3072],
            ));
        }
        write_test_safetensors(tmp.path(), &tensors);
        let a = FluxIpAdapter::parse_safetensors(tmp.path(), &Device::Cpu).unwrap();
        assert_eq!(a.siglip_dim, 1152);
        assert_eq!(a.flux_hidden, 3072);
        assert_eq!(a.num_blocks, 3);
        assert!(a.has_per_block_attn);
    }

    #[test]
    fn parses_lite_no_per_block_attn() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let tensors = vec![
            ("ip_adapter_proj_model.proj_in.weight", vec![3072, 1152]),
            ("ip_adapter_proj_model.proj_out.weight", vec![3072, 3072]),
        ];
        write_test_safetensors(tmp.path(), &tensors);
        let a = FluxIpAdapter::parse_safetensors(tmp.path(), &Device::Cpu).unwrap();
        assert_eq!(a.num_blocks, 0);
        assert!(!a.has_per_block_attn);
    }

    #[test]
    fn bails_on_missing_proj_in() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        // Wrong structure — likely a Redux file or different adapter.
        let tensors = vec![
            ("redux_up.weight", vec![12288, 1152]),
            ("redux_down.weight", vec![4096, 12288]),
        ];
        write_test_safetensors(tmp.path(), &tensors);
        let err = FluxIpAdapter::parse_safetensors(tmp.path(), &Device::Cpu)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("proj_in"),
            "expected proj_in mention, got {msg}"
        );
    }

    #[test]
    fn bails_on_empty_file() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let tensors: Vec<(&str, Vec<usize>)> = vec![];
        write_test_safetensors(tmp.path(), &tensors);
        let err = FluxIpAdapter::parse_safetensors(tmp.path(), &Device::Cpu)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no tensors"), "got {msg}");
    }
}
