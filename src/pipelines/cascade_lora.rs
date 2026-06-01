//! Stable Cascade LoRA merge — v0.38 phase 3.
//!
//! Two-stage diffusers-format PEFT LoRA support for Stable Cascade.
//! A single LoRA file usually targets ONE of the prior UNets
//! (community Cascade LoRAs almost always target Stage C — the
//! heavy semantic stage — since that's where prompt-following lives);
//! the same file may also carry Stage B deltas. The merger walks
//! each user LoRA twice: once for Stage C targets, once for Stage B
//! targets. Keys that match neither are skipped with a debug log.
//!
//! Mirrors `pixart_lora.rs` in shape (reuses `LoraGroup` +
//! `group_peft_keys` + `compute_delta` + `apply_delta` — backbone-
//! agnostic) but with two per-stage resolvers + two merge entry
//! points so `cascade::Pipeline::load` can mmap each merged tempfile
//! into its own VarBuilder.
//!
//! ## Upstream tensor naming
//!
//! Diffusers Cascade LoRAs use the upstream stage names as a
//! key prefix:
//!
//! ```text
//!     // Stage B (called "decoder" in diffusers — Würstchen v3 lineage)
//!     decoder.up_blocks.{i}.{j}.attn.to_q.lora_{A,B}.weight
//!     decoder.up_blocks.{i}.{j}.attn.to_k.lora_{A,B}.weight
//!     ...
//!     // Stage C (called "prior" in diffusers)
//!     prior.up_blocks.{i}.{j}.attn.to_q.lora_{A,B}.weight
//!     ...
//! ```
//!
//! Plakat's internal tensor layout uses different prefixes
//! (`encoder_levels.*` / `decoder_levels.*` per `cascade_unet.rs`).
//! The resolver translates each upstream logical name to the
//! matching internal `*.weight` key — when the translation hits
//! nothing in the base safetensors, the LoRA target is silently
//! skipped (same convention `pixart_lora::apply_one_pixart_lora`
//! uses). This means a real-weight Cascade LoRA + a non-aligned
//! plakat checkpoint will do nothing; real-weight verification
//! at user smoke time is the gating step — same caveat the rest
//! of v0.37/v0.38 phase 0/1 ship with.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pipelines::flux_lora::{
    LoraTarget, RowSlice, apply_delta, compute_delta, group_peft_keys,
};
use crate::pipelines::lora::ResolvedLora;

/// Stable Cascade prior stage discriminator. The merger dispatches
/// each LoRA key against the matching stage's resolver — keys that
/// don't carry either prefix are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Stage B — the latent prior. Diffusers prefix: `decoder.`.
    B,
    /// Stage C — the high-res prior. Diffusers prefix: `prior.`.
    C,
}

impl Stage {
    pub fn upstream_prefix(self) -> &'static str {
        match self {
            Self::B => "decoder.",
            Self::C => "prior.",
        }
    }
}

/// Merge a Stable Cascade Stage B (decoder) LoRA stack into the
/// base safetensors and write the result to `out_path`. Only LoRA
/// keys with the `decoder.` upstream prefix are considered; other
/// keys are silently passed over (the matching Stage C call picks
/// them up).
pub fn merge_cascade_b_loras_into_weights(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    merge_for_stage(base_path, out_path, loras, default_scale, device, Stage::B)
}

/// Stage C (prior) sibling of [`merge_cascade_b_loras_into_weights`].
pub fn merge_cascade_c_loras_into_weights(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    merge_for_stage(base_path, out_path, loras, default_scale, device, Stage::C)
}

fn merge_for_stage(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
    stage: Stage,
) -> Result<(usize, usize)> {
    let mut merged: HashMap<String, Tensor> =
        candle_core::safetensors::load(base_path, device).with_context(|| {
            format!(
                "loading base Stable Cascade Stage {:?} weights {}",
                stage,
                base_path.display()
            )
        })?;
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading Cascade LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let (n_mod, n_targets) = apply_one_lora(
            &mut merged,
            &lora_tensors,
            effective_scale,
            device,
            stage,
        )?;
        modified += n_mod;
        total_groups += n_targets;
        if n_targets > 0 {
            tracing::info!(
                target: "plakat",
                "Cascade Stage {:?} LoRA {} → {n_mod}/{n_targets} targets merged (scale {:.2})",
                stage, lora.display, effective_scale
            );
        }
    }

    candle_core::safetensors::save(&merged, out_path).with_context(|| {
        format!(
            "writing merged Cascade Stage {:?} weights to {}",
            stage,
            out_path.display()
        )
    })?;
    Ok((modified, total_groups))
}

fn apply_one_lora(
    merged: &mut HashMap<String, Tensor>,
    lora_tensors: &HashMap<String, Tensor>,
    effective_scale: f32,
    device: &Device,
    stage: Stage,
) -> Result<(usize, usize)> {
    let groups = group_peft_keys(lora_tensors);
    let prefix = stage.upstream_prefix();
    let mut n_modified = 0usize;
    let mut n_total = 0usize;
    for (logical, group) in &groups {
        // Stage filter: only consider LoRA keys carrying this stage's
        // upstream prefix. Counts toward `n_total` only when the
        // prefix matches AND the resolver recognises the leaf path.
        let Some(stage_logical) = logical.strip_prefix(prefix) else {
            continue;
        };
        let Some(target) = resolve_target(stage_logical) else {
            tracing::debug!(
                target: "plakat",
                "Cascade Stage {:?} LoRA: unknown logical target `{logical}` — skipping",
                stage
            );
            continue;
        };
        n_total += 1;
        if !merged.contains_key(&target.base_key) {
            tracing::debug!(
                target: "plakat",
                "Cascade Stage {:?} LoRA target `{logical}` resolves to `{}` which \
                 is not present in the base weights — skipping (likely tensor-naming \
                 misalignment with upstream).",
                stage, target.base_key
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

/// Translate an upstream logical leaf path (with the `decoder.` /
/// `prior.` prefix already stripped) to plakat's internal base
/// safetensors key.
///
/// Upstream convention (community Cascade LoRAs):
///
/// ```text
///     up_blocks.{level}.{pos}.attention.{to_q,to_k,to_v,to_out.0}
///     down_blocks.{level}.{pos}.attention.{to_q,to_k,to_v,to_out.0}
/// ```
///
/// v0.39 phase 0g plakat internal convention (cascade_prior.rs) — now
/// matches upstream exactly since the prior architecture was rewritten:
///
/// ```text
///     down_blocks.{level}.{pos}.attention.{to_q,to_k,to_v,to_out.0}.weight
///     encoder_levels.{i}.attentions.{j}.{self_attn,cross_attn}.to_q.weight
/// ```
///
/// Returns the SELF-ATTN target (Stable Cascade LoRAs almost always
/// target self-attention — cross-attn to text is fixed by training).
/// Returns `None` for non-attention leaves (FF layers are not
/// currently exposed as LoRA targets — community LoRAs rarely
/// touch them on Cascade).
fn resolve_target(stage_logical: &str) -> Option<LoraTarget> {
    // v0.39 phase 0g: tensor names now match upstream after the
    // cascade_prior rewrite. Identity mapping for the level + position
    // segments; we just append `.weight` to the resolved attention
    // leaf and route via the upstream block kind.
    let rest = stage_logical
        .strip_prefix("up_blocks.")
        .or_else(|| stage_logical.strip_prefix("down_blocks."))?;
    // up_blocks vs down_blocks both share the same internal block path
    // (`down_blocks.{level}.{pos}.attention.*` for down,
    // `up_blocks.{level}.{pos}.attention.*` for up). Preserve the original
    // prefix.
    let prefix = if stage_logical.starts_with("up_blocks.") {
        "up_blocks"
    } else {
        "down_blocks"
    };
    let (level_idx_str, tail) = rest.split_once('.')?;
    let _level_idx: usize = level_idx_str.parse().ok()?;
    let (pos_idx_str, leaf) = tail.split_once('.')?;
    let _pos_idx: usize = pos_idx_str.parse().ok()?;
    // Accept upstream's `attention.to_*` paths (the new cascade_prior
    // AttnBlock uses `attention.{to_q,to_k,to_v,to_out.0}` exactly).
    let leaf_after_attn = leaf
        .strip_prefix("attention.")
        .or_else(|| leaf.strip_prefix("attn."))?;
    let accepted = ["to_q", "to_k", "to_v", "to_out.0"];
    if !accepted.contains(&leaf_after_attn) {
        return None;
    }
    let base_key =
        format!("{prefix}.{level_idx_str}.{pos_idx_str}.attention.{leaf_after_attn}.weight");
    Some(LoraTarget {
        base_key,
        slice: RowSlice::Full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_upstream_prefixes_match_diffusers_convention() {
        assert_eq!(Stage::B.upstream_prefix(), "decoder.");
        assert_eq!(Stage::C.upstream_prefix(), "prior.");
    }

    #[test]
    fn resolve_up_block_attention_to_q_routes_to_upstream_layout() {
        // v0.39 phase 0g: identity mapping for level/pos; appends `.weight`.
        let t = resolve_target("up_blocks.0.2.attention.to_q")
            .expect("up_blocks attention.to_q must resolve");
        assert_eq!(t.base_key, "up_blocks.0.2.attention.to_q.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolve_down_block_routes_to_upstream_layout() {
        let t = resolve_target("down_blocks.1.11.attention.to_v")
            .expect("down_blocks attention.to_v must resolve");
        assert_eq!(t.base_key, "down_blocks.1.11.attention.to_v.weight");
    }

    #[test]
    fn resolve_to_out_with_index_zero_resolves() {
        let t = resolve_target("up_blocks.0.2.attention.to_out.0")
            .expect("attention.to_out.0 must resolve");
        assert_eq!(
            t.base_key,
            "up_blocks.0.2.attention.to_out.0.weight"
        );
    }

    #[test]
    fn resolve_accepts_legacy_attn_prefix() {
        // Some community LoRAs use `attn.*` instead of `attention.*`;
        // accept both for compatibility.
        let t = resolve_target("up_blocks.0.0.attn.to_q")
            .expect("attn.to_q must resolve");
        assert_eq!(t.base_key, "up_blocks.0.0.attention.to_q.weight");
    }

    #[test]
    fn resolve_non_attention_leaf_returns_none() {
        assert!(resolve_target("up_blocks.0.0.ff.net.0.proj").is_none());
        assert!(resolve_target("up_blocks.0.0.norm.linear").is_none());
        assert!(resolve_target("up_blocks.0.0.attention.to_out.1").is_none());
    }

    #[test]
    fn resolve_top_level_path_returns_none() {
        assert!(resolve_target("conv_in.weight").is_none());
        assert!(resolve_target("time_embedding.linear_1").is_none());
        // The stage prefix is stripped at the merger level, so a
        // logical path that doesn't start with up_blocks/down_blocks
        // is correctly rejected.
        assert!(resolve_target("clip_g_projection.linear").is_none());
    }
}
