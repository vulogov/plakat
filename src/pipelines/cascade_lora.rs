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
use candle_core::{DType, Device, Tensor};
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
    /// Diffusers/PEFT dotted prefix (`prior.` / `decoder.`).
    pub fn upstream_prefix(self) -> &'static str {
        match self {
            Self::B => "decoder.",
            Self::C => "prior.",
        }
    }

    /// v0.42 phase 1: kohya / sd-scripts underscored prefix. This is
    /// the format real community Cascade LoRAs actually ship in (e.g.
    /// `lora_prior_unet_down_blocks_0_11_attention_attn_to_q`), NOT the
    /// dotted PEFT form — which is why pre-v0.42 LoRA merging silently
    /// no-op'd on every kohya-trained Cascade LoRA.
    pub fn kohya_prefix(self) -> &'static str {
        match self {
            Self::B => "lora_decoder_unet_",
            Self::C => "lora_prior_unet_",
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
    let dotted = stage.upstream_prefix();
    let kohya = stage.kohya_prefix();
    // v0.42 phase 1: DoRA magnitudes. Community Cascade LoRAs are
    // frequently DoRAs (Weight-Decomposed LoRA) — they carry a
    // per-output `.dora_scale` magnitude vector and the merged weight
    // is `m·(W+ΔW)/‖W+ΔW‖`, NOT just `W+ΔW`. Applying the LoRA delta
    // without the renorm corrupts the weights (the ΔW direction is
    // large precisely because the magnitude normalises it away). Map
    // each target's dora_scale here; `apply_dora` does the renorm.
    let dora_scales: HashMap<&str, &Tensor> = lora_tensors
        .iter()
        .filter_map(|(k, v)| k.strip_suffix(".dora_scale").map(|l| (l, v)))
        .collect();
    let mut n_modified = 0usize;
    let mut n_total = 0usize;
    for (logical, group) in &groups {
        // Stage filter: accept either the diffusers dotted prefix
        // (`prior.`/`decoder.`) or the kohya underscored prefix
        // (`lora_prior_unet_`/`lora_decoder_unet_`). Keys for the other
        // stage / neither format are skipped (the sibling call handles
        // its own stage).
        let target = if let Some(sl) = logical.strip_prefix(dotted) {
            resolve_target(sl)
        } else if let Some(sl) = logical.strip_prefix(kohya) {
            resolve_kohya_target(sl)
        } else {
            continue;
        };
        n_total += 1;
        let Some(target) = target else {
            tracing::debug!(
                target: "plakat",
                "Cascade Stage {:?} LoRA: unknown logical target `{logical}` — skipping",
                stage
            );
            continue;
        };
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
        let updated = match dora_scales.get(logical.as_str()) {
            Some(dora) => apply_dora(&base, &delta, dora)?,
            None => apply_delta(&base, &delta, target.slice)?,
        };
        merged.insert(target.base_key.clone(), updated);
        n_modified += 1;
    }
    Ok((n_modified, n_total))
}

/// v0.42 phase 1: DoRA (Weight-Decomposed LoRA) weight fuse. The
/// merged weight renorms `V = W + ΔW` so each vector along the
/// magnitude axis is scaled to its stored `dora_scale` target:
///
/// ```text
///   V  = W + ΔW                            (ΔW = the already-scaled delta)
///   W' = dora_scale · V / ‖V‖_axis         (norm over the OTHER axis)
/// ```
///
/// IMPORTANT — the renorm axis differs by trainer convention, and the
/// wrong axis is catastrophic. PEFT's reference DoRA
/// (`peft/tuners/lora/dora.py`) normalises over `dim=1` (per output
/// row); kohya / sd-scripts — the format real community Cascade LoRAs
/// ship in — computes the magnitude over the transposed axis, so
/// `dora_scale` is a per-**input-column** vector renormed over `dim=0`.
/// Picking wrong multiplies a per-column magnitude across the rows (or
/// vice-versa), scrambling every weight at full force regardless of
/// LoRA strength — the v0.42-phase-1 magenta/cream corruption.
///
/// [`pick_dora_axis`] auto-detects the correct axis: for non-square
/// weights the `dora_scale` length disambiguates; for square weights it
/// picks the axis where `dora_scale / ‖W‖` is most uniform (the matching
/// axis clusters tightly — CoV ~0.08 on the real `cascade_anime` DoRA —
/// while the wrong axis is wildly variable — CoV ~0.39). So a kohya DoRA
/// and a genuine PEFT-format DoRA both merge correctly.
fn apply_dora(base: &Tensor, delta: &Tensor, dora_scale: &Tensor) -> Result<Tensor> {
    let dtype = base.dtype();
    let base_f = base.to_dtype(DType::F32)?;
    let delta_f = delta.to_dtype(DType::F32)?;
    let v = (&base_f + &delta_f)?; // [out, in]
    let (out, in_dim) = v.dims2()?;
    let m_flat = dora_scale.to_dtype(DType::F32)?.flatten_all()?; // [n]

    let updated = if pick_dora_axis(&base_f, &m_flat, out, in_dim)? == DoraAxis::Column {
        // per input column: ‖V‖ over the output dim → [1, in].
        let v_norm = v.sqr()?.sum_keepdim(0)?.sqrt()?;
        let m = m_flat.reshape((1, in_dim))?;
        v.broadcast_mul(&m.broadcast_div(&v_norm)?)?
    } else {
        // per output row: ‖V‖ over the input dim → [out, 1].
        let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
        let m = m_flat.reshape((out, 1))?;
        v.broadcast_mul(&m.broadcast_div(&v_norm)?)?
    };
    updated.to_dtype(dtype).map_err(|e| e.into())
}

/// Which axis a DoRA `dora_scale` vector indexes. `Column` = per
/// input column (kohya / sd-scripts, renorm over `dim=0`); `Row` = per
/// output row (PEFT, renorm over `dim=1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DoraAxis {
    Column,
    Row,
}

/// Auto-detect the DoRA magnitude axis. Length disambiguates a
/// non-square weight outright; for a square weight, the matching axis
/// is the one where `dora_scale / ‖W‖_axis` has the lower coefficient
/// of variation (the trainer initialised `dora_scale` to that axis's
/// norm, so the ratio is near-constant; the transposed axis is noise).
fn pick_dora_axis(
    base_f: &Tensor,
    m_flat: &Tensor,
    out: usize,
    in_dim: usize,
) -> Result<DoraAxis> {
    let n = m_flat.elem_count();
    if n == in_dim && n != out {
        return Ok(DoraAxis::Column);
    }
    if n == out && n != in_dim {
        return Ok(DoraAxis::Row);
    }
    // Square (or ambiguous): compare dispersion of dora/‖W‖ per axis.
    let col_norm = base_f.sqr()?.sum(0)?.sqrt()?; // [in]
    let row_norm = base_f.sqr()?.sum(1)?.sqrt()?; // [out]
    let col_cov = coeff_of_variation(&m_flat.broadcast_div(&col_norm)?)?;
    let row_cov = coeff_of_variation(&m_flat.broadcast_div(&row_norm)?)?;
    Ok(if col_cov <= row_cov {
        DoraAxis::Column
    } else {
        DoraAxis::Row
    })
}

/// Coefficient of variation (std / |mean|) of a 1-D tensor. Used to
/// score how uniform a `dora_scale / ‖W‖` ratio is — the matching axis
/// minimises it.
fn coeff_of_variation(x: &Tensor) -> Result<f64> {
    let mean = x.mean_all()?.to_scalar::<f32>()? as f64;
    let mean_sq = x.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let var = (mean_sq - mean * mean).max(0.0);
    Ok(if mean.abs() < 1e-12 {
        f64::INFINITY
    } else {
        var.sqrt() / mean.abs()
    })
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

/// v0.42 phase 1: resolve a kohya / sd-scripts underscored logical
/// path (the stage prefix `lora_prior_unet_` / `lora_decoder_unet_`
/// already stripped) to plakat's internal base key.
///
/// Real community Cascade LoRAs name targets like
/// `down_blocks_0_11_attention_attn_to_q` (dots flattened to
/// underscores, attention leaf prefixed `attn_`, output proj named
/// `out_proj`). This maps to `down_blocks.0.11.attention.to_q.weight`:
///
/// - `attn_to_q` → `to_q`, `attn_to_k` → `to_k`, `attn_to_v` → `to_v`
/// - `attn_out_proj` → `to_out.0`
fn resolve_kohya_target(stage_logical: &str) -> Option<LoraTarget> {
    let (block, rest) = if let Some(r) = stage_logical.strip_prefix("down_blocks_") {
        ("down_blocks", r)
    } else if let Some(r) = stage_logical.strip_prefix("up_blocks_") {
        ("up_blocks", r)
    } else {
        return None;
    };
    // rest = "{level}_{pos}_attention_{leaf}"
    let (level, rest) = rest.split_once('_')?;
    level.parse::<usize>().ok()?;
    let (pos, leaf) = rest.split_once("_attention_")?;
    pos.parse::<usize>().ok()?;
    let mapped = match leaf {
        "attn_to_q" => "to_q",
        "attn_to_k" => "to_k",
        "attn_to_v" => "to_v",
        "attn_out_proj" => "to_out.0",
        _ => return None,
    };
    Some(LoraTarget {
        base_key: format!("{block}.{level}.{pos}.attention.{mapped}.weight"),
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
    fn resolve_kohya_real_lora_keys_route_to_base() {
        // v0.42 phase 1: the exact key shapes from the real-world
        // `cascade_3.6B_anime_DoRa.safetensors` (after group_peft_keys
        // strips `.lora_down.weight` and the stage merger strips the
        // `lora_prior_unet_` prefix).
        let q = resolve_kohya_target("down_blocks_0_11_attention_attn_to_q")
            .expect("kohya attn_to_q must resolve");
        assert_eq!(q.base_key, "down_blocks.0.11.attention.to_q.weight");

        let out = resolve_kohya_target("up_blocks_0_2_attention_attn_out_proj")
            .expect("kohya attn_out_proj must resolve");
        assert_eq!(out.base_key, "up_blocks.0.2.attention.to_out.0.weight");

        let k = resolve_kohya_target("down_blocks_1_23_attention_attn_to_k")
            .expect("kohya attn_to_k must resolve");
        assert_eq!(k.base_key, "down_blocks.1.23.attention.to_k.weight");
    }

    #[test]
    fn resolve_kohya_rejects_non_attention_and_malformed() {
        assert!(resolve_kohya_target("down_blocks_0_0_ff_net_0_proj").is_none());
        assert!(resolve_kohya_target("conv_in").is_none());
        assert!(resolve_kohya_target("down_blocks_0_11_attention_attn_to_out_1").is_none());
    }

    #[test]
    fn kohya_prefixes_distinguish_stages() {
        assert_eq!(Stage::C.kohya_prefix(), "lora_prior_unet_");
        assert_eq!(Stage::B.kohya_prefix(), "lora_decoder_unet_");
    }

    #[test]
    fn apply_one_lora_merges_kohya_format_end_to_end() {
        // v0.42 phase 1 regression guard: a kohya-format Cascade LoRA
        // (the real-world shape) must actually modify the base weight.
        // Pre-v0.42 this silently no-op'd (n_modified == 0) because the
        // merger only accepted the dotted `prior.` prefix.
        use candle_core::{Device, Tensor};
        use std::collections::HashMap;
        let device = Device::Cpu;
        let mut merged: HashMap<String, Tensor> = HashMap::new();
        merged.insert(
            "down_blocks.0.11.attention.to_q.weight".to_string(),
            Tensor::zeros((8, 8), candle_core::DType::F32, &device).unwrap(),
        );
        let mut lora: HashMap<String, Tensor> = HashMap::new();
        let base = "lora_prior_unet_down_blocks_0_11_attention_attn_to_q";
        // rank-2 LoRA: down (r×in) = (2,8), up (out×r) = (8,2), alpha=2.
        lora.insert(
            format!("{base}.lora_down.weight"),
            Tensor::ones((2, 8), candle_core::DType::F32, &device).unwrap(),
        );
        lora.insert(
            format!("{base}.lora_up.weight"),
            Tensor::ones((8, 2), candle_core::DType::F32, &device).unwrap(),
        );
        lora.insert(
            format!("{base}.alpha"),
            Tensor::new(2.0f32, &device).unwrap(),
        );
        let (n_mod, n_total) =
            apply_one_lora(&mut merged, &lora, 1.0, &device, Stage::C).unwrap();
        assert_eq!(n_mod, 1, "kohya target must merge (was 0 pre-v0.42)");
        assert_eq!(n_total, 1);
        // up@down = ones(8,2)@ones(2,8) = 2 everywhere; alpha/rank=1,
        // scale=1 → delta = 2. Base was 0 → merged = 2.
        let w = merged.get("down_blocks.0.11.attention.to_q.weight").unwrap();
        let v = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 2.0).abs() < 1e-5), "got {:?}", &v[..4]);
    }

    #[test]
    fn apply_dora_column_axis_pins_column_magnitudes() {
        // v0.42 phase 1: a kohya DoRA renorms per INPUT COLUMN. Use a
        // NON-square weight (out=2, in=3) so the dora_scale length (3)
        // forces the column axis unambiguously. col2=[1,0] (norm 1) is
        // pinned to dora[2]=2; col0/col1 already at norm 5 → unchanged.
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;
        let base = Tensor::new(&[[3.0f32, 0.0, 1.0], [4.0, 5.0, 0.0]], &device).unwrap();
        let delta = Tensor::zeros((2, 3), candle_core::DType::F32, &device).unwrap();
        let dora = Tensor::new(&[[5.0f32, 5.0, 2.0]], &device).unwrap(); // len = in
        let out = apply_dora(&base, &delta, &dora).unwrap();
        let v = out.to_vec2::<f32>().unwrap();
        let col = |j: usize| (v[0][j].powi(2) + v[1][j].powi(2)).sqrt();
        assert!((col(0) - 5.0).abs() < 1e-3 && (col(1) - 5.0).abs() < 1e-3);
        assert!((col(2) - 2.0).abs() < 1e-3, "col2 pinned to 2, got {}", col(2));
    }

    #[test]
    fn apply_dora_row_axis_pins_row_magnitudes() {
        // The PEFT convention: a dora_scale whose length matches the
        // OUTPUT dim forces the row axis. Non-square (out=3, in=2);
        // row2=[1,0] (norm 1) pinned to dora[2]=2.
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;
        let base = Tensor::new(&[[3.0f32, 4.0], [0.0, 5.0], [1.0, 0.0]], &device).unwrap();
        let delta = Tensor::zeros((3, 2), candle_core::DType::F32, &device).unwrap();
        let dora = Tensor::new(&[[5.0f32, 5.0, 2.0]], &device).unwrap(); // len = out
        let out = apply_dora(&base, &delta, &dora).unwrap();
        let v = out.to_vec2::<f32>().unwrap();
        let row = |i: usize| (v[i][0].powi(2) + v[i][1].powi(2)).sqrt();
        assert!((row(0) - 5.0).abs() < 1e-3 && (row(1) - 5.0).abs() < 1e-3);
        assert!((row(2) - 2.0).abs() < 1e-3, "row2 pinned to 2, got {}", row(2));
    }

    #[test]
    fn pick_dora_axis_square_uses_lower_dispersion() {
        // Square weights can't be disambiguated by length, so the axis
        // is chosen by whichever dora/‖W‖ ratio is most uniform.
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;
        // base: col0=[3,4] norm 5, col1=[0,5] norm 5; row0=[3,0] norm 3,
        // row1=[4,5] norm ~6.40.
        let base = Tensor::new(&[[3.0f32, 0.0], [4.0, 5.0]], &device).unwrap();
        // dora == column norms → column ratio is constant (CoV 0) → Column.
        let col_aligned = Tensor::new(&[5.0f32, 5.0], &device).unwrap();
        assert_eq!(
            pick_dora_axis(&base, &col_aligned, 2, 2).unwrap(),
            DoraAxis::Column
        );
        // dora == row norms → row ratio constant → Row.
        let row_aligned = Tensor::new(&[3.0f32, (41.0f32).sqrt()], &device).unwrap();
        assert_eq!(
            pick_dora_axis(&base, &row_aligned, 2, 2).unwrap(),
            DoraAxis::Row
        );
    }

    #[test]
    fn pick_dora_axis_nonsquare_uses_length() {
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;
        let base = Tensor::zeros((2, 3), candle_core::DType::F32, &device).unwrap();
        let len3 = Tensor::zeros(3, candle_core::DType::F32, &device).unwrap();
        let len2 = Tensor::zeros(2, candle_core::DType::F32, &device).unwrap();
        // out=2, in=3: length-3 → Column, length-2 → Row.
        assert_eq!(pick_dora_axis(&base, &len3, 2, 3).unwrap(), DoraAxis::Column);
        assert_eq!(pick_dora_axis(&base, &len2, 2, 3).unwrap(), DoraAxis::Row);
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
