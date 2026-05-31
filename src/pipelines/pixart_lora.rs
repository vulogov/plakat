//! PixArt-Σ LoRA merge — v0.35 phase 4.
//!
//! Diffusers-format PEFT LoRAs for the PixArt-Σ DiT-XL/2 backbone.
//! Mirrors `flux_lora.rs` / `sd3_lora.rs` in shape (reusing the
//! shared `LoraGroup` + `group_peft_keys` + `compute_delta` +
//! `apply_delta` helpers — backbone-agnostic) but with PixArt's
//! own key resolver.
//!
//! ## What's specific to PixArt
//!
//! The v0.35 phase 1 DiT lays out per-block weights with the same
//! diffusers names a PEFT LoRA targets:
//!
//! ```text
//!     transformer_blocks.{i}.attn1.to_q.weight       (H, H)
//!     transformer_blocks.{i}.attn1.to_k.weight       (H, H)
//!     transformer_blocks.{i}.attn1.to_v.weight       (H, H)
//!     transformer_blocks.{i}.attn1.to_out.0.weight   (H, H)
//!     transformer_blocks.{i}.attn2.to_q.weight       (H, H)
//!     transformer_blocks.{i}.attn2.to_k.weight       (H, H)
//!     transformer_blocks.{i}.attn2.to_v.weight       (H, H)
//!     transformer_blocks.{i}.attn2.to_out.0.weight   (H, H)
//!     transformer_blocks.{i}.ff.net.0.proj.weight    (4H, H)
//!     transformer_blocks.{i}.ff.net.2.weight         (H, 4H)
//! ```
//!
//! Where `H = hidden_size = 1152` for XL/2. Unlike Flux + SD3
//! MMDiT, PixArt has NO fused QKV — `to_q` / `to_k` / `to_v` are
//! three distinct Linears, so every LoRA target is
//! `RowSlice::Full` (no partial-row math).
//!
//! ## Diffusers PEFT key mapping
//!
//! ```text
//!     transformer.transformer_blocks.{i}.attn1.to_q.lora_A.weight   →
//!         transformer_blocks.{i}.attn1.to_q.weight        [full rows]
//!     transformer.transformer_blocks.{i}.ff.net.0.proj.lora_A.weight →
//!         transformer_blocks.{i}.ff.net.0.proj.weight     [full rows]
//!     ...
//! ```
//!
//! The optional `transformer.` prefix is stripped by
//! `flux_lora::group_peft_keys` (a single PEFT convention shared
//! by every diffusers DiT LoRA). After grouping, the logical name
//! IS the base key — `pixart_lora` just appends `.weight` and
//! checks the merged map.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pipelines::flux_lora::{
    LoraTarget, RowSlice, apply_delta, compute_delta, group_peft_keys,
};
use crate::pipelines::lora::ResolvedLora;

/// Public entry. Merges PEFT-format PixArt LoRAs into the base
/// transformer safetensors and writes the result to `out_path`.
/// Returns `(modified_targets, total_lora_groups)`.
pub fn merge_pixart_loras_into_weights(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    let mut merged: HashMap<String, Tensor> =
        candle_core::safetensors::load(base_path, device).with_context(|| {
            format!(
                "loading base PixArt transformer weights {}",
                base_path.display()
            )
        })?;
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading PixArt LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let (n_mod, n_targets) =
            apply_one_pixart_lora(&mut merged, &lora_tensors, effective_scale, device)?;
        modified += n_mod;
        total_groups += n_targets;
        if n_targets > 0 {
            tracing::info!(
                target: "plakat",
                "PixArt LoRA {} → {n_mod}/{n_targets} targets merged (scale {:.2})",
                lora.display,
                effective_scale
            );
        }
    }

    candle_core::safetensors::save(&merged, out_path).with_context(|| {
        format!("writing merged PixArt transformer to {}", out_path.display())
    })?;
    Ok((modified, total_groups))
}

/// Group + resolve + apply one PixArt LoRA file into `merged`.
fn apply_one_pixart_lora(
    merged: &mut HashMap<String, Tensor>,
    lora_tensors: &HashMap<String, Tensor>,
    effective_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    let groups = group_peft_keys(lora_tensors);
    let n_total = groups.len();
    let mut n_modified = 0usize;
    for (logical, group) in &groups {
        let Some(target) = resolve_target(logical) else {
            tracing::debug!(
                target: "plakat",
                "PixArt LoRA: unknown logical target `{logical}` — skipping"
            );
            continue;
        };
        if !merged.contains_key(&target.base_key) {
            tracing::debug!(
                target: "plakat",
                "PixArt LoRA target `{logical}` resolves to `{}` which is not \
                 present in the base transformer weights — skipping",
                target.base_key
            );
            continue;
        }
        let delta = compute_delta(group, effective_scale, device)?;
        let base = merged
            .get(&target.base_key)
            .expect("checked just above")
            .clone();
        let updated = apply_delta(&base, &delta, target.slice)?;
        merged.insert(target.base_key.clone(), updated);
        n_modified += 1;
    }
    Ok((n_modified, n_total))
}

/// PixArt-Σ has no fused QKV — every LoRA target is `RowSlice::Full`
/// and the base key is just `{logical}.weight`. Accepted logical
/// names cover the 10 per-block targets diffusers PEFT LoRAs ship:
/// attn1.{to_q,to_k,to_v,to_out.0}, attn2.{same four},
/// ff.net.{0.proj,2}. Returns `None` for any other key (silently
/// skipped — same convention Flux + SD3 resolvers use).
fn resolve_target(logical: &str) -> Option<LoraTarget> {
    // Must live under a transformer_blocks.{i}.* path.
    let rest = logical.strip_prefix("transformer_blocks.")?;
    let (block_idx_str, tail) = rest.split_once('.')?;
    let _block_idx: usize = block_idx_str.parse().ok()?;
    // Accepted leaf paths — exactly the linears the PixArt DiT
    // exposes for cross-attn + self-attn + FF.
    let accepted = [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ];
    if !accepted.contains(&tail) {
        return None;
    }
    Some(LoraTarget {
        base_key: format!("{logical}.weight"),
        slice: RowSlice::Full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::collections::HashMap;

    fn cpu_zero(rank: usize, in_dim: usize) -> Tensor {
        Tensor::zeros((rank, in_dim), DType::F32, &Device::Cpu).unwrap()
    }
    fn cpu_up(out_dim: usize, rank: usize) -> Tensor {
        Tensor::zeros((out_dim, rank), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn resolve_attn1_to_q_returns_full_row_slice() {
        let t = resolve_target("transformer_blocks.5.attn1.to_q")
            .expect("attn1.to_q must resolve");
        assert_eq!(t.base_key, "transformer_blocks.5.attn1.to_q.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolve_attn2_cross_to_v_resolves() {
        let t = resolve_target("transformer_blocks.0.attn2.to_v")
            .expect("attn2.to_v must resolve");
        assert_eq!(t.base_key, "transformer_blocks.0.attn2.to_v.weight");
    }

    #[test]
    fn resolve_ff_net_layers_resolve() {
        assert!(resolve_target("transformer_blocks.27.ff.net.0.proj").is_some());
        assert!(resolve_target("transformer_blocks.0.ff.net.2").is_some());
    }

    #[test]
    fn resolve_unknown_leaf_returns_none() {
        assert!(resolve_target("transformer_blocks.5.norm1.linear").is_none());
        assert!(resolve_target("transformer_blocks.5.attn1.to_out.1").is_none());
        assert!(resolve_target("pos_embed.proj").is_none());
        assert!(resolve_target("adaln_single.linear").is_none());
    }

    #[test]
    fn resolve_non_block_path_returns_none() {
        assert!(resolve_target("caption_projection.linear_1").is_none());
        assert!(resolve_target("proj_out").is_none());
        assert!(resolve_target("transformer_blocks").is_none()); // no trailing dot
    }

    #[test]
    fn resolve_non_numeric_block_index_returns_none() {
        assert!(resolve_target("transformer_blocks.foo.attn1.to_q").is_none());
    }

    #[test]
    fn group_peft_keys_strips_transformer_prefix_for_pixart() {
        // Verify the shared grouping helper plays nice with PixArt's
        // diffusers PEFT key convention.
        let mut t = HashMap::new();
        let rank = 4;
        let in_dim = 1152;
        let out_dim = 1152;
        t.insert(
            "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight".to_string(),
            cpu_zero(rank, in_dim),
        );
        t.insert(
            "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight".to_string(),
            cpu_up(out_dim, rank),
        );
        let g = group_peft_keys(&t);
        // After prefix strip, the logical key is the bare layer path.
        let entry = g.get("transformer_blocks.0.attn1.to_q").unwrap_or_else(|| {
            panic!(
                "expected logical `transformer_blocks.0.attn1.to_q`, got {:?}",
                g.keys().collect::<Vec<_>>()
            )
        });
        assert!(entry.a.is_some());
        assert!(entry.b.is_some());
    }
}
