//! SD3 / SD3.5 LoRA merge — v0.15 phase 3.
//!
//! Mirrors `flux_lora.rs` in shape but resolves PEFT logical names
//! against candle's MMDiT safetensors layout. The PEFT-format grouping
//! (`lora_A` / `lora_B` / `alpha`), delta math (`B @ A * alpha/rank`),
//! and row-slice apply are all reused from `flux_lora` — those are
//! backbone-agnostic.
//!
//! ## What's specific to MMDiT
//!
//! Candle's MMDiT (`candle_transformers::models::mmdit`) lays out
//! per-block weights as:
//!
//! ```text
//!     joint_blocks.{i}.x_block.attn.qkv.weight       (3*H, H)   — fused QKV
//!     joint_blocks.{i}.x_block.attn.proj.weight      (H, H)
//!     joint_blocks.{i}.x_block.mlp.fc1.weight        (4H, H)
//!     joint_blocks.{i}.x_block.mlp.fc2.weight        (H, 4H)
//!     joint_blocks.{i}.x_block.adaLN_modulation.1.weight (6H, H)
//!     joint_blocks.{i}.context_block.attn.qkv.weight (3*H, H)   — fused QKV
//!     joint_blocks.{i}.context_block.attn.proj.weight (H, H)    ← absent on last block
//!     joint_blocks.{i}.context_block.mlp.fc1.weight  (4H, H)
//!     joint_blocks.{i}.context_block.mlp.fc2.weight  (H, 4H)
//!     joint_blocks.{i}.context_block.adaLN_modulation.1.weight (6H, H)
//! ```
//!
//! Where `H = head_size * depth` (SD3 / SD3.5-Medium: H=1536;
//! SD3.5-Large: H=2432). SD3.5-Medium-Large also gains `attn2.qkv` per
//! x_block — not yet a LoRA target in shipping diffusers SD3 LoRAs, so
//! we don't resolve those keys.
//!
//! ## Diffusers PEFT key mapping
//!
//! Diffusers SD3 LoRAs name their targets like:
//!
//! ```text
//!     transformer.transformer_blocks.{i}.attn.to_q       → x_block.attn.qkv  [0..H)
//!     transformer.transformer_blocks.{i}.attn.to_k       → x_block.attn.qkv  [H..2H)
//!     transformer.transformer_blocks.{i}.attn.to_v       → x_block.attn.qkv  [2H..3H)
//!     transformer.transformer_blocks.{i}.attn.to_out.0   → x_block.attn.proj
//!     transformer.transformer_blocks.{i}.attn.add_q_proj → context_block.attn.qkv  [0..H)
//!     transformer.transformer_blocks.{i}.attn.add_k_proj → context_block.attn.qkv  [H..2H)
//!     transformer.transformer_blocks.{i}.attn.add_v_proj → context_block.attn.qkv  [2H..3H)
//!     transformer.transformer_blocks.{i}.attn.to_add_out → context_block.attn.proj  (skipped on last block)
//!     transformer.transformer_blocks.{i}.ff.net.0.proj   → x_block.mlp.fc1
//!     transformer.transformer_blocks.{i}.ff.net.2        → x_block.mlp.fc2
//!     transformer.transformer_blocks.{i}.ff_context.net.0.proj → context_block.mlp.fc1
//!     transformer.transformer_blocks.{i}.ff_context.net.2 → context_block.mlp.fc2
//!     transformer.transformer_blocks.{i}.norm1.linear    → x_block.adaLN_modulation.1
//!     transformer.transformer_blocks.{i}.norm1_context.linear → context_block.adaLN_modulation.1
//! ```
//!
//! Same `transformer.` prefix stripping + `.lora_A/B/alpha` grouping
//! the Flux PEFT resolver uses.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pipelines::flux_lora::{
    LoraGroup, LoraTarget, RowSlice, apply_delta, compute_delta, group_peft_keys,
};
use crate::pipelines::lora::ResolvedLora;

/// Public entry. Merges PEFT-format SD3 LoRAs into the base MMDiT
/// safetensors and writes the result to `out_path`. Returns
/// `(modified_groups, total_lora_groups_in_files)`.
///
/// `hidden_size` must match the variant being loaded
/// (`head_size * depth`): 1536 for SD3 / SD3.5-Medium, 2432 for
/// SD3.5-Large. The resolver uses it for QKV-fusion slicing math.
pub fn merge_sd3_loras_into_weights(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    hidden_size: usize,
    device: &Device,
) -> Result<(usize, usize)> {
    let mut merged: HashMap<String, Tensor> =
        candle_core::safetensors::load(base_path, device).with_context(|| {
            format!(
                "loading base SD3 MMDiT weights {}",
                base_path.display()
            )
        })?;
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading SD3 LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let (n_mod, n_targets) = apply_one_sd3_lora(
            &mut merged,
            &lora_tensors,
            effective_scale,
            hidden_size,
            device,
        )?;
        modified += n_mod;
        total_groups += n_targets;
        if n_targets > 0 {
            tracing::info!(
                target: "plakat",
                "SD3 LoRA {} → {n_mod}/{n_targets} targets merged (scale {:.2})",
                lora.display,
                effective_scale
            );
        }
    }

    candle_core::safetensors::save(&merged, out_path).with_context(|| {
        format!("writing merged SD3 MMDiT to {}", out_path.display())
    })?;
    Ok((modified, total_groups))
}

fn apply_one_sd3_lora(
    merged: &mut HashMap<String, Tensor>,
    lora_tensors: &HashMap<String, Tensor>,
    effective_scale: f32,
    hidden_size: usize,
    device: &Device,
) -> Result<(usize, usize)> {
    let groups = group_peft_keys(lora_tensors);
    let mut n_mod = 0usize;
    let total = groups.len();
    for (logical, group) in groups {
        if group.a.is_none() || group.b.is_none() {
            continue;
        }
        let target = match resolve_target(&logical, hidden_size) {
            Some(t) => t,
            None => {
                tracing::debug!(
                    target: "plakat",
                    "SD3 LoRA: skipping unresolvable target {logical}"
                );
                continue;
            }
        };
        let base = match merged.get(&target.base_key) {
            Some(b) => b.clone(),
            None => {
                // Last block's context branch is QkvOnly (no .proj or
                // to_add_out). PEFT files include those keys for
                // architectural uniformity; we silently skip them.
                tracing::debug!(
                    target: "plakat",
                    "SD3 LoRA: target {} not in base — skipping",
                    target.base_key
                );
                continue;
            }
        };
        let delta = compute_delta(&group, effective_scale, device)
            .with_context(|| format!("computing SD3 LoRA delta for {logical}"))?;
        let updated = apply_delta(&base, &delta, target.slice).with_context(|| {
            format!("applying SD3 LoRA delta for {logical} into {}", target.base_key)
        })?;
        merged.insert(target.base_key.clone(), updated);
        n_mod += 1;
    }
    Ok((n_mod, total))
}

/// Resolve a SD3 LoRA logical name (PEFT) to the MMDiT base tensor
/// key + row slice. Strips an optional `transformer.` prefix to
/// match diffusers' SD3 LoRA convention.
pub(crate) fn resolve_target(logical: &str, hidden_size: usize) -> Option<LoraTarget> {
    let logical = logical.strip_prefix("transformer.").unwrap_or(logical);
    let rest = logical.strip_prefix("transformer_blocks.")?;
    resolve_block(rest, hidden_size)
}

fn resolve_block(rest: &str, h: usize) -> Option<LoraTarget> {
    let (idx_str, tail) = rest.split_once('.')?;
    let i: usize = idx_str.parse().ok()?;
    let x_key = |suffix: &str| format!("joint_blocks.{i}.x_block.{suffix}.weight");
    let c_key = |suffix: &str| format!("joint_blocks.{i}.context_block.{suffix}.weight");
    let qkv_q = RowSlice::Partial { start: 0, end: h };
    let qkv_k = RowSlice::Partial { start: h, end: 2 * h };
    let qkv_v = RowSlice::Partial { start: 2 * h, end: 3 * h };
    match tail {
        // ---------- x_block (image stream) ----------
        "attn.to_q" => Some(LoraTarget { base_key: x_key("attn.qkv"), slice: qkv_q }),
        "attn.to_k" => Some(LoraTarget { base_key: x_key("attn.qkv"), slice: qkv_k }),
        "attn.to_v" => Some(LoraTarget { base_key: x_key("attn.qkv"), slice: qkv_v }),
        "attn.to_out.0" => Some(LoraTarget {
            base_key: x_key("attn.proj"),
            slice: RowSlice::Full,
        }),
        "ff.net.0.proj" => Some(LoraTarget {
            base_key: x_key("mlp.fc1"),
            slice: RowSlice::Full,
        }),
        "ff.net.2" => Some(LoraTarget {
            base_key: x_key("mlp.fc2"),
            slice: RowSlice::Full,
        }),
        "norm1.linear" => Some(LoraTarget {
            base_key: x_key("adaLN_modulation.1"),
            slice: RowSlice::Full,
        }),
        // ---------- context_block (text stream) ----------
        // Last block uses QkvOnly: `.qkv` exists but `.proj` /
        // `to_add_out` do not. apply_one_sd3_lora handles the missing
        // base tensor by skipping; we still resolve the path here so
        // the skip is a "weight not present" rather than "unknown key".
        "attn.add_q_proj" => Some(LoraTarget { base_key: c_key("attn.qkv"), slice: qkv_q }),
        "attn.add_k_proj" => Some(LoraTarget { base_key: c_key("attn.qkv"), slice: qkv_k }),
        "attn.add_v_proj" => Some(LoraTarget { base_key: c_key("attn.qkv"), slice: qkv_v }),
        "attn.to_add_out" => Some(LoraTarget {
            base_key: c_key("attn.proj"),
            slice: RowSlice::Full,
        }),
        "ff_context.net.0.proj" => Some(LoraTarget {
            base_key: c_key("mlp.fc1"),
            slice: RowSlice::Full,
        }),
        "ff_context.net.2" => Some(LoraTarget {
            base_key: c_key("mlp.fc2"),
            slice: RowSlice::Full,
        }),
        "norm1_context.linear" => Some(LoraTarget {
            base_key: c_key("adaLN_modulation.1"),
            slice: RowSlice::Full,
        }),
        _ => None,
    }
}

// Suppress unused-import warning when nothing in this module ends up
// touching `LoraGroup` directly. The tests below use it for shape
// assertions.
#[allow(dead_code)]
fn _suppress_unused() -> Option<LoraGroup> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- v0.15 phase 3 — SD3 LoRA path resolution ----------

    const SD3_MEDIUM_HIDDEN: usize = 1536; // head=64, depth=24
    const SD3_LARGE_HIDDEN: usize = 2432; // head=64, depth=38

    #[test]
    fn resolves_x_block_qkv_slices() {
        let h = SD3_MEDIUM_HIDDEN;
        let q = resolve_target("transformer_blocks.0.attn.to_q", h).unwrap();
        assert_eq!(q.base_key, "joint_blocks.0.x_block.attn.qkv.weight");
        assert!(matches!(q.slice, RowSlice::Partial { start: 0, end } if end == h));
        let k = resolve_target("transformer_blocks.7.attn.to_k", h).unwrap();
        assert_eq!(k.base_key, "joint_blocks.7.x_block.attn.qkv.weight");
        assert!(matches!(
            k.slice,
            RowSlice::Partial { start, end }
                if start == h && end == 2 * h
        ));
        let v = resolve_target("transformer_blocks.23.attn.to_v", h).unwrap();
        assert_eq!(v.base_key, "joint_blocks.23.x_block.attn.qkv.weight");
        assert!(matches!(
            v.slice,
            RowSlice::Partial { start, end }
                if start == 2 * h && end == 3 * h
        ));
    }

    #[test]
    fn resolves_context_block_qkv_slices() {
        let h = SD3_MEDIUM_HIDDEN;
        let aq = resolve_target("transformer_blocks.5.attn.add_q_proj", h).unwrap();
        assert_eq!(
            aq.base_key,
            "joint_blocks.5.context_block.attn.qkv.weight"
        );
        let ak = resolve_target("transformer_blocks.5.attn.add_k_proj", h).unwrap();
        assert!(matches!(
            ak.slice,
            RowSlice::Partial { start, end } if start == h && end == 2 * h
        ));
        let av = resolve_target("transformer_blocks.5.attn.add_v_proj", h).unwrap();
        assert!(matches!(
            av.slice,
            RowSlice::Partial { start, end } if start == 2 * h && end == 3 * h
        ));
    }

    #[test]
    fn resolves_attn_projection_outputs() {
        let h = SD3_MEDIUM_HIDDEN;
        let p = resolve_target("transformer_blocks.0.attn.to_out.0", h).unwrap();
        assert_eq!(p.base_key, "joint_blocks.0.x_block.attn.proj.weight");
        assert!(matches!(p.slice, RowSlice::Full));
        let c = resolve_target("transformer_blocks.0.attn.to_add_out", h).unwrap();
        assert_eq!(c.base_key, "joint_blocks.0.context_block.attn.proj.weight");
    }

    #[test]
    fn resolves_mlp_branches() {
        let h = SD3_MEDIUM_HIDDEN;
        let x0 = resolve_target("transformer_blocks.0.ff.net.0.proj", h).unwrap();
        assert_eq!(x0.base_key, "joint_blocks.0.x_block.mlp.fc1.weight");
        let x2 = resolve_target("transformer_blocks.0.ff.net.2", h).unwrap();
        assert_eq!(x2.base_key, "joint_blocks.0.x_block.mlp.fc2.weight");
        let c0 = resolve_target("transformer_blocks.0.ff_context.net.0.proj", h).unwrap();
        assert_eq!(c0.base_key, "joint_blocks.0.context_block.mlp.fc1.weight");
        let c2 = resolve_target("transformer_blocks.0.ff_context.net.2", h).unwrap();
        assert_eq!(c2.base_key, "joint_blocks.0.context_block.mlp.fc2.weight");
    }

    #[test]
    fn resolves_adaln_modulation() {
        let h = SD3_MEDIUM_HIDDEN;
        let n1 = resolve_target("transformer_blocks.4.norm1.linear", h).unwrap();
        assert_eq!(
            n1.base_key,
            "joint_blocks.4.x_block.adaLN_modulation.1.weight"
        );
        let n1c = resolve_target("transformer_blocks.4.norm1_context.linear", h).unwrap();
        assert_eq!(
            n1c.base_key,
            "joint_blocks.4.context_block.adaLN_modulation.1.weight"
        );
    }

    #[test]
    fn strips_transformer_prefix() {
        // Diffusers SD3 LoRAs commonly carry the `transformer.` prefix
        // (the SD3 PEFT root). resolve_target must accept both.
        let h = SD3_MEDIUM_HIDDEN;
        let with = resolve_target(
            "transformer.transformer_blocks.0.attn.to_q",
            h,
        )
        .unwrap();
        let without = resolve_target("transformer_blocks.0.attn.to_q", h).unwrap();
        assert_eq!(with.base_key, without.base_key);
    }

    #[test]
    fn slices_scale_with_hidden_size() {
        // SD3.5-Large doubles hidden_size from 1536 to 2432, so the
        // QKV row offsets shift accordingly.
        let q_m = resolve_target("transformer_blocks.0.attn.to_q", SD3_MEDIUM_HIDDEN).unwrap();
        let q_l = resolve_target("transformer_blocks.0.attn.to_q", SD3_LARGE_HIDDEN).unwrap();
        assert!(matches!(
            q_m.slice,
            RowSlice::Partial { start: 0, end } if end == SD3_MEDIUM_HIDDEN
        ));
        assert!(matches!(
            q_l.slice,
            RowSlice::Partial { start: 0, end } if end == SD3_LARGE_HIDDEN
        ));
    }

    #[test]
    fn unknown_paths_return_none() {
        let h = SD3_MEDIUM_HIDDEN;
        assert!(resolve_target("transformer_blocks.0.unknown.path", h).is_none());
        // Flux-style paths must not accidentally resolve here.
        assert!(resolve_target("double_blocks.0.img_attn.qkv", h).is_none());
        assert!(resolve_target("single_blocks.0.linear1", h).is_none());
    }
}
