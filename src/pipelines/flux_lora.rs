//! Flux LoRA merge — v0.12 phase 1 of the Flux modernization.
//!
//! Flux LoRAs ship in two main conventions on HuggingFace today:
//!
//! 1. **Diffusers PEFT format** (this module's scope). Keys look like
//!    `transformer.transformer_blocks.0.attn.to_q.lora_A.weight` and the
//!    matching `.lora_B.weight`. Sometimes the `transformer.` prefix is
//!    elided. The base diffusers naming addresses the model's
//!    *logical* Q/K/V/MLP/etc. layers separately.
//!
//! 2. **AI-Toolkit / kohya-style** (out of scope for this commit).
//!    Keys like `lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight`
//!    — already in candle-Flux's flattened underscore-naming, so a
//!    follow-up can add this format with much smaller plumbing.
//!
//! ## The fused-Linear problem
//!
//! Flux's safetensors fuse multiple logical projections into single
//! Linear layers:
//!
//! * Each DoubleStream block stores Q/K/V as one `(3·hidden, hidden)`
//!   `img_attn.qkv.weight` (and the parallel `txt_attn.qkv.weight`).
//! * Each SingleStream block stores `[Q; K; V; MLP_up]` as one
//!   `(3·hidden + mlp_dim, hidden)` `linear1.weight`.
//!
//! Diffusers PEFT LoRAs target each logical projection separately
//! (`attn.to_q`, `attn.to_k`, etc.), so merging requires writing a
//! LoRA delta into a *row-slice* of the fused base tensor — not the
//! whole thing. The math:
//!
//! ```text
//!   delta  = up @ down * (alpha / rank) * scale
//!   base[row_start..row_end, :] += delta
//! ```
//!
//! See [`apply_delta`] for the partial-row implementation.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::pipelines::lora::ResolvedLora;

/// Hidden width of stock Flux.1-dev/schnell. Used for slice math on
/// fused QKV / MLP Linears. SD-family LoRAs that get fed here by
/// accident will silently fail to match these dimensions and skip.
const FLUX_HIDDEN: usize = 3072;
const FLUX_MLP_DIM: usize = 12288;
/// Total output rows of a DoubleStream block's `*_attn.qkv.weight`.
const QKV_OUT: usize = 3 * FLUX_HIDDEN;

/// Where a LoRA delta lands inside the base tensor.
#[derive(Debug, Clone, Copy)]
enum RowSlice {
    /// Add the delta to the entire base tensor (shapes must match).
    Full,
    /// Add the delta only to rows `[start, end)` of dim 0. The base
    /// tensor's dim-0 may be larger than the delta's; partial overlap
    /// is the whole point.
    Partial { start: usize, end: usize },
}

/// One LoRA target — base tensor name and the slice of it the delta
/// applies to.
#[derive(Debug, Clone)]
struct LoraTarget {
    base_key: String,
    slice: RowSlice,
}

/// Public entry. Merges PEFT-format Flux LoRAs into the base Flux
/// transformer safetensors and writes the result to `out_path`.
/// Returns `(modified_groups, total_lora_groups_in_files)`.
pub fn merge_flux_loras_into_weights(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    let mut merged: HashMap<String, Tensor> =
        candle_core::safetensors::load(base_path, device).with_context(|| {
            format!(
                "loading base Flux transformer weights {}",
                base_path.display()
            )
        })?;
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading Flux LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let (n_mod, n_targets) = apply_one_flux_lora(
            &mut merged,
            &lora_tensors,
            effective_scale,
            device,
        )?;
        modified += n_mod;
        total_groups += n_targets;
        if n_targets > 0 {
            tracing::info!(
                target: "plakat",
                "Flux LoRA {} → {n_mod}/{n_targets} targets merged (scale {:.2})",
                lora.display,
                effective_scale
            );
        }
    }

    candle_core::safetensors::save(&merged, out_path).with_context(|| {
        format!("writing merged Flux transformer to {}", out_path.display())
    })?;
    Ok((modified, total_groups))
}

/// Group a PEFT-format Flux LoRA's tensors into (logical_target_name → group).
/// Strips an optional `transformer.` prefix. Unknown keys are dropped.
#[derive(Default, Debug)]
struct LoraGroup {
    /// LoRA "down" matrix, shape `(rank, in)`.
    a: Option<Tensor>,
    /// LoRA "up" matrix, shape `(out, rank)`.
    b: Option<Tensor>,
    /// Optional learnable alpha — if missing, `alpha = rank`.
    alpha: Option<Tensor>,
}

fn group_peft_keys(
    lora_tensors: &HashMap<String, Tensor>,
) -> HashMap<String, LoraGroup> {
    let mut groups: HashMap<String, LoraGroup> = HashMap::new();
    for (k, v) in lora_tensors.iter() {
        // Strip optional "transformer." prefix.
        let k_norm = k.strip_prefix("transformer.").unwrap_or(k.as_str());
        // PEFT keys look like `...path/to/layer.lora_A.weight` (or
        // `.lora_B.weight`, or `.alpha`). Strip the trailing
        // .weight first to make the suffix match simpler.
        let key_no_weight = k_norm.strip_suffix(".weight").unwrap_or(k_norm);
        let (logical, role) = if let Some(stem) = key_no_weight.strip_suffix(".lora_A") {
            (stem, "a")
        } else if let Some(stem) = key_no_weight.strip_suffix(".lora_B") {
            (stem, "b")
        } else if let Some(stem) = key_no_weight.strip_suffix(".lora_down") {
            // Some Flux LoRAs (mixed-format files in the wild) use
            // the kohya-style lora_down/lora_up suffixes inside a
            // PEFT-looking parent prefix. Accept both.
            (stem, "a")
        } else if let Some(stem) = key_no_weight.strip_suffix(".lora_up") {
            (stem, "b")
        } else if let Some(stem) = key_no_weight.strip_suffix(".alpha") {
            (stem, "alpha")
        } else {
            continue;
        };

        let entry = groups.entry(logical.to_string()).or_default();
        match role {
            "a" => entry.a = Some(v.clone()),
            "b" => entry.b = Some(v.clone()),
            "alpha" => entry.alpha = Some(v.clone()),
            _ => {}
        }
    }
    groups
}

/// Resolve a PEFT logical name (e.g. `transformer_blocks.0.attn.to_q`)
/// to the candle-Flux base tensor key + the row slice the LoRA delta
/// applies to. Returns `None` for unrecognised paths (silently skipped).
fn resolve_target(logical: &str) -> Option<LoraTarget> {
    // ---------- DoubleStream blocks ----------
    if let Some(rest) = logical.strip_prefix("transformer_blocks.") {
        return resolve_double_block(rest);
    }
    // ---------- SingleStream blocks ----------
    if let Some(rest) = logical.strip_prefix("single_transformer_blocks.") {
        return resolve_single_block(rest);
    }
    None
}

fn resolve_double_block(rest: &str) -> Option<LoraTarget> {
    // rest looks like `{i}.{path}`. Split off the block index.
    let (block_idx_str, tail) = rest.split_once('.')?;
    let i: usize = block_idx_str.parse().ok()?;
    let block_key = |suffix: &str| format!("double_blocks.{i}.{suffix}.weight");
    // Image stream.
    match tail {
        "attn.to_q" => Some(LoraTarget {
            base_key: block_key("img_attn.qkv"),
            slice: RowSlice::Partial { start: 0, end: FLUX_HIDDEN },
        }),
        "attn.to_k" => Some(LoraTarget {
            base_key: block_key("img_attn.qkv"),
            slice: RowSlice::Partial {
                start: FLUX_HIDDEN,
                end: 2 * FLUX_HIDDEN,
            },
        }),
        "attn.to_v" => Some(LoraTarget {
            base_key: block_key("img_attn.qkv"),
            slice: RowSlice::Partial {
                start: 2 * FLUX_HIDDEN,
                end: QKV_OUT,
            },
        }),
        "attn.to_out.0" => Some(LoraTarget {
            base_key: block_key("img_attn.proj"),
            slice: RowSlice::Full,
        }),
        "ff.net.0.proj" => Some(LoraTarget {
            base_key: block_key("img_mlp.0"),
            slice: RowSlice::Full,
        }),
        "ff.net.2" => Some(LoraTarget {
            base_key: block_key("img_mlp.2"),
            slice: RowSlice::Full,
        }),
        "norm1.linear" => Some(LoraTarget {
            base_key: block_key("img_mod.lin"),
            slice: RowSlice::Full,
        }),
        // Text stream.
        "attn.add_q_proj" => Some(LoraTarget {
            base_key: block_key("txt_attn.qkv"),
            slice: RowSlice::Partial { start: 0, end: FLUX_HIDDEN },
        }),
        "attn.add_k_proj" => Some(LoraTarget {
            base_key: block_key("txt_attn.qkv"),
            slice: RowSlice::Partial {
                start: FLUX_HIDDEN,
                end: 2 * FLUX_HIDDEN,
            },
        }),
        "attn.add_v_proj" => Some(LoraTarget {
            base_key: block_key("txt_attn.qkv"),
            slice: RowSlice::Partial {
                start: 2 * FLUX_HIDDEN,
                end: QKV_OUT,
            },
        }),
        "attn.to_add_out" => Some(LoraTarget {
            base_key: block_key("txt_attn.proj"),
            slice: RowSlice::Full,
        }),
        "ff_context.net.0.proj" => Some(LoraTarget {
            base_key: block_key("txt_mlp.0"),
            slice: RowSlice::Full,
        }),
        "ff_context.net.2" => Some(LoraTarget {
            base_key: block_key("txt_mlp.2"),
            slice: RowSlice::Full,
        }),
        "norm1_context.linear" => Some(LoraTarget {
            base_key: block_key("txt_mod.lin"),
            slice: RowSlice::Full,
        }),
        _ => None,
    }
}

fn resolve_single_block(rest: &str) -> Option<LoraTarget> {
    let (block_idx_str, tail) = rest.split_once('.')?;
    let i: usize = block_idx_str.parse().ok()?;
    let block_key = |suffix: &str| format!("single_blocks.{i}.{suffix}.weight");
    // SingleStream's `linear1` fuses [Q; K; V; MLP_up] along its
    // output dim: rows 0..hidden = Q, hidden..2h = K, 2h..3h = V,
    // 3h..3h+mlp_dim = MLP_up. `linear2` is the final out projection.
    let mlp_start = QKV_OUT;
    let mlp_end = QKV_OUT + FLUX_MLP_DIM;
    match tail {
        "attn.to_q" => Some(LoraTarget {
            base_key: block_key("linear1"),
            slice: RowSlice::Partial { start: 0, end: FLUX_HIDDEN },
        }),
        "attn.to_k" => Some(LoraTarget {
            base_key: block_key("linear1"),
            slice: RowSlice::Partial {
                start: FLUX_HIDDEN,
                end: 2 * FLUX_HIDDEN,
            },
        }),
        "attn.to_v" => Some(LoraTarget {
            base_key: block_key("linear1"),
            slice: RowSlice::Partial {
                start: 2 * FLUX_HIDDEN,
                end: QKV_OUT,
            },
        }),
        "proj_mlp" => Some(LoraTarget {
            base_key: block_key("linear1"),
            slice: RowSlice::Partial {
                start: mlp_start,
                end: mlp_end,
            },
        }),
        "proj_out" => Some(LoraTarget {
            base_key: block_key("linear2"),
            slice: RowSlice::Full,
        }),
        "norm.linear" => Some(LoraTarget {
            base_key: block_key("modulation.lin"),
            slice: RowSlice::Full,
        }),
        _ => None,
    }
}

/// Compute the LoRA delta `B @ A * (alpha / rank)`.
fn compute_delta(
    group: &LoraGroup,
    effective_scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let a = group.a.as_ref().ok_or_else(|| anyhow!("missing lora_A"))?;
    let b = group.b.as_ref().ok_or_else(|| anyhow!("missing lora_B"))?;
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let rank = a_f32.dim(0)? as f32;
    let alpha_f32 = match group.alpha.as_ref() {
        Some(t) => {
            let t = t.to_dtype(DType::F32)?;
            // `alpha` may be stored as a scalar tensor or a 1-element vector.
            match t.dims().len() {
                0 => t.to_scalar::<f32>()?,
                _ => t
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(rank),
            }
        }
        None => rank,
    };
    let scaling = (alpha_f32 / rank.max(1.0)) * effective_scale;
    // delta = B @ A — shape (out, in).
    let delta = b_f32.matmul(&a_f32)?;
    let _ = device;
    Ok((delta * scaling as f64)?)
}

/// Add `delta` into `base` at `slice`. If `slice` is `Partial`, only
/// the rows `[start, end)` of dim 0 are updated. Preserves base dtype.
fn apply_delta(base: &Tensor, delta: &Tensor, slice: RowSlice) -> Result<Tensor> {
    let base_dtype = base.dtype();
    // Math in F32 — Flux's BF16 has the range but not the resolution
    // for accurate accumulation of small LoRA deltas, especially
    // across stacked LoRAs.
    let base_f32 = base.to_dtype(DType::F32)?;
    let delta_f32 = delta.to_dtype(DType::F32)?;
    let updated = match slice {
        RowSlice::Full => {
            // Dimensions must match exactly.
            if base.shape() != delta.shape() {
                anyhow::bail!(
                    "Flux LoRA shape mismatch (full): base {:?} vs delta {:?}",
                    base.dims(),
                    delta.dims()
                );
            }
            (base_f32 + delta_f32)?
        }
        RowSlice::Partial { start, end } => {
            let base_dims = base_f32.dims();
            let n = base_dims.first().copied().unwrap_or(0);
            if end > n || start >= end {
                anyhow::bail!(
                    "Flux LoRA partial-row out of bounds: rows {start}..{end} \
                     into base dim-0 {n}"
                );
            }
            let slab = base_f32.narrow(0, start, end - start)?;
            if slab.shape() != delta_f32.shape() {
                anyhow::bail!(
                    "Flux LoRA partial-row shape mismatch: rows {start}..{end} \
                     of base = {:?}, delta = {:?}",
                    slab.dims(),
                    delta_f32.dims()
                );
            }
            let new_slab = (slab + delta_f32)?;
            // Build the slice_assign ranges. Dim 0 is the row-slice;
            // the trailing dims are full.
            let mut ranges: Vec<std::ops::Range<usize>> =
                Vec::with_capacity(base_dims.len());
            ranges.push(start..end);
            for &d in &base_dims[1..] {
                ranges.push(0..d);
            }
            base_f32.slice_assign(&ranges, &new_slab)?
        }
    };
    Ok(updated.to_dtype(base_dtype)?)
}

fn apply_one_flux_lora(
    merged: &mut HashMap<String, Tensor>,
    lora_tensors: &HashMap<String, Tensor>,
    effective_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    let groups = group_peft_keys(lora_tensors);
    let mut modified = 0usize;
    let total = groups.len();
    for (logical, group) in groups {
        if group.a.is_none() || group.b.is_none() {
            tracing::debug!(
                target: "plakat",
                "Flux LoRA skip {logical}: incomplete group (missing A/B)"
            );
            continue;
        }
        let target = match resolve_target(&logical) {
            Some(t) => t,
            None => {
                tracing::debug!(
                    target: "plakat",
                    "Flux LoRA skip {logical}: no candle-Flux target mapping"
                );
                continue;
            }
        };
        let base = match merged.get(&target.base_key) {
            Some(t) => t.clone(),
            None => {
                tracing::debug!(
                    target: "plakat",
                    "Flux LoRA skip {logical}: base key {} not in transformer",
                    target.base_key
                );
                continue;
            }
        };
        let delta = compute_delta(&group, effective_scale, device).with_context(|| {
            format!("computing Flux LoRA delta for {logical}")
        })?;
        let updated = apply_delta(&base, &delta, target.slice).with_context(|| {
            format!("applying Flux LoRA delta for {logical} into {}", target.base_key)
        })?;
        merged.insert(target.base_key, updated);
        modified += 1;
    }
    Ok((modified, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_double_block_q() {
        let t = resolve_target("transformer_blocks.0.attn.to_q").unwrap();
        assert_eq!(t.base_key, "double_blocks.0.img_attn.qkv.weight");
        match t.slice {
            RowSlice::Partial { start, end } => {
                assert_eq!(start, 0);
                assert_eq!(end, FLUX_HIDDEN);
            }
            _ => panic!("expected Partial slice for to_q"),
        }
    }

    #[test]
    fn resolves_double_block_text_stream() {
        // Text-stream Q goes to txt_attn.qkv rows 0..hidden.
        let t = resolve_target("transformer_blocks.5.attn.add_q_proj").unwrap();
        assert_eq!(t.base_key, "double_blocks.5.txt_attn.qkv.weight");
        match t.slice {
            RowSlice::Partial { start, end } => {
                assert_eq!(start, 0);
                assert_eq!(end, FLUX_HIDDEN);
            }
            _ => panic!("expected Partial slice"),
        }
    }

    #[test]
    fn resolves_single_block_proj_mlp() {
        let t = resolve_target("single_transformer_blocks.10.proj_mlp").unwrap();
        assert_eq!(t.base_key, "single_blocks.10.linear1.weight");
        match t.slice {
            RowSlice::Partial { start, end } => {
                assert_eq!(start, QKV_OUT);
                assert_eq!(end, QKV_OUT + FLUX_MLP_DIM);
            }
            _ => panic!("expected Partial slice for proj_mlp"),
        }
    }

    #[test]
    fn resolves_full_proj_out() {
        let t = resolve_target("single_transformer_blocks.0.proj_out").unwrap();
        assert_eq!(t.base_key, "single_blocks.0.linear2.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn unknown_targets_return_none() {
        assert!(resolve_target("vae.encoder.something").is_none());
        assert!(resolve_target("transformer_blocks.0.unknown_field").is_none());
    }

    #[test]
    fn peft_grouping_handles_transformer_prefix() {
        let mut t = HashMap::new();
        let a = Tensor::zeros((4, 3072), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::zeros((3072, 4), DType::F32, &Device::Cpu).unwrap();
        t.insert(
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight".to_string(),
            a,
        );
        t.insert(
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight".to_string(),
            b,
        );
        let groups = group_peft_keys(&t);
        let g = groups
            .get("transformer_blocks.0.attn.to_q")
            .expect("transformer. prefix should be stripped");
        assert!(g.a.is_some() && g.b.is_some());
    }
}
