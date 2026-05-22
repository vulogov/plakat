//! Flux LoRA merge — v0.12 phase 1 + v0.13 phase 1d.
//!
//! Flux LoRAs ship in two main conventions on HuggingFace today, and
//! this module handles both:
//!
//! 1. **Diffusers PEFT format**. Keys look like
//!    `transformer.transformer_blocks.0.attn.to_q.lora_A.weight` and
//!    the matching `.lora_B.weight`. Sometimes the `transformer.`
//!    prefix is elided. Targets each *logical* Q/K/V/MLP projection
//!    separately — needs row-slice math against Flux's fused Linears.
//!
//! 2. **AI-Toolkit / kohya-style** (v0.13 addition). Keys like
//!    `lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight` —
//!    already in candle-Flux's flattened underscore-naming and trained
//!    against the *fused* tensor directly. No slice math needed; the
//!    delta is the full base shape.
//!
//! Both formats share the grouping code below (a/b/alpha lookups
//! handle both `lora_A`/`lora_B` and `lora_down`/`lora_up`), so the
//! only per-format work is the `resolve_target` dispatch.
//!
//! ## The fused-Linear problem (PEFT only)
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
//! See [`apply_delta`] for the partial-row implementation. AI-Toolkit
//! LoRAs targeting fused tensors skip the slice math (`RowSlice::Full`).

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
///
/// v0.15 phase 3: promoted to `pub(crate)` so `sd3_lora.rs` can
/// reuse the same row-slice math for MMDiT's fused QKV targets.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RowSlice {
    /// Add the delta to the entire base tensor (shapes must match).
    Full,
    /// Add the delta only to rows `[start, end)` of dim 0. The base
    /// tensor's dim-0 may be larger than the delta's; partial overlap
    /// is the whole point.
    Partial { start: usize, end: usize },
}

/// One LoRA target — base tensor name and the slice of it the delta
/// applies to. `pub(crate)` so the SD3 LoRA resolver can produce
/// these alongside Flux's.
#[derive(Debug, Clone)]
pub(crate) struct LoraTarget {
    pub(crate) base_key: String,
    pub(crate) slice: RowSlice,
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
///
/// v0.15 phase 3: `pub(crate)` so `sd3_lora.rs` reuses the same
/// grouping for diffusers PEFT-format SD3 LoRAs (which use identical
/// `.lora_A` / `.lora_B` / `.alpha` suffix conventions).
#[derive(Default, Debug)]
pub(crate) struct LoraGroup {
    /// LoRA "down" matrix, shape `(rank, in)`.
    pub(crate) a: Option<Tensor>,
    /// LoRA "up" matrix, shape `(out, rank)`.
    pub(crate) b: Option<Tensor>,
    /// Optional learnable alpha — if missing, `alpha = rank`.
    pub(crate) alpha: Option<Tensor>,
}

pub(crate) fn group_peft_keys(
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

/// Resolve a LoRA logical name to the candle-Flux base tensor key +
/// the row slice the LoRA delta applies to. Tries PEFT/diffusers
/// naming first (dot-separated, fused Linears addressed by logical
/// sub-projection), then falls back to AI-Toolkit / kohya naming
/// (underscore-flattened, fused Linears addressed directly).
/// Returns `None` for unrecognised paths (silently skipped).
fn resolve_target(logical: &str) -> Option<LoraTarget> {
    // ---------- PEFT DoubleStream blocks ----------
    if let Some(rest) = logical.strip_prefix("transformer_blocks.") {
        return resolve_double_block(rest);
    }
    // ---------- PEFT SingleStream blocks ----------
    if let Some(rest) = logical.strip_prefix("single_transformer_blocks.") {
        return resolve_single_block(rest);
    }
    // ---------- AI-Toolkit / kohya ----------
    // `lora_unet_` prefix marks the underscore-flattened naming.
    // Some exports drop the prefix and emit `double_blocks_…` /
    // `single_blocks_…` directly, so we also try the bare form.
    let aitoolkit_rest = logical
        .strip_prefix("lora_unet_")
        .unwrap_or(logical);
    if let Some(rest) = aitoolkit_rest.strip_prefix("double_blocks_") {
        return resolve_aitoolkit_double(rest);
    }
    if let Some(rest) = aitoolkit_rest.strip_prefix("single_blocks_") {
        return resolve_aitoolkit_single(rest);
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

/// AI-Toolkit DoubleStream resolver. Input `rest` is the underscore-
/// flattened sub-path after `[lora_unet_]double_blocks_`, e.g.
/// `0_img_attn_qkv` or `12_txt_mod_lin`. AI-Toolkit trains against the
/// fused Linear, so the delta covers the full base tensor (no slice).
fn resolve_aitoolkit_double(rest: &str) -> Option<LoraTarget> {
    let (idx_str, tail) = rest.split_once('_')?;
    let i: usize = idx_str.parse().ok()?;
    let block_key = |suffix: &str| format!("double_blocks.{i}.{suffix}.weight");
    let target = |suffix: &str| Some(LoraTarget {
        base_key: block_key(suffix),
        slice: RowSlice::Full,
    });
    match tail {
        // Image stream.
        "img_attn_qkv" => target("img_attn.qkv"),
        "img_attn_proj" => target("img_attn.proj"),
        "img_mlp_0" => target("img_mlp.0"),
        "img_mlp_2" => target("img_mlp.2"),
        "img_mod_lin" => target("img_mod.lin"),
        // Text stream.
        "txt_attn_qkv" => target("txt_attn.qkv"),
        "txt_attn_proj" => target("txt_attn.proj"),
        "txt_mlp_0" => target("txt_mlp.0"),
        "txt_mlp_2" => target("txt_mlp.2"),
        "txt_mod_lin" => target("txt_mod.lin"),
        _ => None,
    }
}

/// AI-Toolkit SingleStream resolver. Input `rest` is the underscore-
/// flattened sub-path after `[lora_unet_]single_blocks_`. `linear1`
/// fuses Q+K+V+MLP_up, so AI-Toolkit trains a 4-way merged delta that
/// the merger writes as a single full-shape update.
fn resolve_aitoolkit_single(rest: &str) -> Option<LoraTarget> {
    let (idx_str, tail) = rest.split_once('_')?;
    let i: usize = idx_str.parse().ok()?;
    let block_key = |suffix: &str| format!("single_blocks.{i}.{suffix}.weight");
    let target = |suffix: &str| Some(LoraTarget {
        base_key: block_key(suffix),
        slice: RowSlice::Full,
    });
    match tail {
        "linear1" => target("linear1"),
        "linear2" => target("linear2"),
        "modulation_lin" => target("modulation.lin"),
        _ => None,
    }
}

/// Compute the LoRA delta `B @ A * (alpha / rank)`.
///
/// v0.15 phase 3: `pub(crate)` for SD3 LoRA reuse — the formula is
/// identical across diffusers PEFT LoRAs regardless of the backbone.
pub(crate) fn compute_delta(
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
///
/// v0.15 phase 3: `pub(crate)` for SD3 LoRA reuse. MMDiT's fused
/// QKV per joint block uses the same Partial-slice math (e.g.
/// `to_q` → rows `[0, hidden)`, `to_k` → `[hidden, 2*hidden)`).
pub(crate) fn apply_delta(base: &Tensor, delta: &Tensor, slice: RowSlice) -> Result<Tensor> {
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

/// v0.13 phase 1e: precompute LoRA-merged dense overrides for a
/// quantized (GGUF) Flux model.
///
/// Returns a map keyed by **full base tensor path including
/// `.weight`** (e.g. `"double_blocks.0.img_attn.qkv.weight"`) mapping
/// to the BF16 merged dense tensor. Only Linears whose path appears in
/// this map need to be loaded as dense; everything else stays 4-bit.
///
/// The function dequantizes each targeted base tensor exactly once
/// (subsequent LoRAs that hit the same tensor accumulate into the
/// already-dense version), applies `apply_delta` with the resolved row
/// slice (full or partial), and casts to BF16 — the runtime dtype.
/// Multiple LoRAs touching the same target compose additively, exactly
/// like the BF16 safetensors merge path.
/// v0.14 phase 8b: derive the canonical Flux base-tensor shape
/// `(out_dim, in_dim)` from its safetensors path + the Flux config.
/// Used by `precompute_nf4_overrides` to know how to size each NF4
/// dequant — the NF4 codec's `dequant_nf4` requires an explicit shape
/// (unlike the GGUF path, which stores it in the QTensor metadata).
///
/// Returns `None` for paths that aren't standard Flux Linear weight
/// keys (e.g. LayerNorm scales, RmsNorm scales) — those don't carry
/// NF4 packed data anyway.
pub fn flux_target_shape(
    base_key: &str,
    cfg: &crate::pipelines::flux_inner::Config,
) -> Option<(usize, usize)> {
    let stem = base_key.strip_suffix(".weight").unwrap_or(base_key);
    let h = cfg.hidden_size;
    let mlp = (h as f64 * cfg.mlp_ratio) as usize;
    // DoubleStream blocks: `double_blocks.{i}.{img|txt}_*`.
    if let Some(rest) = stem.strip_prefix("double_blocks.") {
        let (_idx, tail) = rest.split_once('.')?;
        return match tail {
            "img_attn.qkv" | "txt_attn.qkv" => Some((3 * h, h)),
            "img_attn.proj" | "txt_attn.proj" => Some((h, h)),
            "img_mlp.0" | "txt_mlp.0" => Some((mlp, h)),
            "img_mlp.2" | "txt_mlp.2" => Some((h, mlp)),
            "img_mod.lin" | "txt_mod.lin" => Some((6 * h, h)),
            _ => None,
        };
    }
    if let Some(rest) = stem.strip_prefix("single_blocks.") {
        let (_idx, tail) = rest.split_once('.')?;
        return match tail {
            "linear1" => Some((3 * h + mlp, h)),
            "linear2" => Some((h, h + mlp)),
            "modulation.lin" => Some((3 * h, h)),
            _ => None,
        };
    }
    None
}

/// v0.14 phase 8b: NF4-equivalent of `precompute_quantized_overrides`.
/// Walks `loras`, resolves each target via the standard PEFT /
/// AI-Toolkit resolvers, dequantizes only the affected NF4-packed
/// weights via the codec, applies deltas with the right row-slice
/// math, and returns a HashMap of dense BF16 tensors keyed by full
/// base path (`<path>.weight`). Non-targeted Linears stay packed in
/// the store — the same selective-dequant memory profile the GGUF
/// path delivers (v0.13 phase 1e).
pub fn precompute_nf4_overrides(
    store: &crate::pipelines::nf4_loader::Nf4Store,
    loras: &[ResolvedLora],
    default_scale: f32,
    cfg: &crate::pipelines::flux_inner::Config,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, usize, usize)> {
    let mut overrides: HashMap<String, Tensor> = HashMap::new();
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading Flux LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let groups = group_peft_keys(&lora_tensors);
        total_groups += groups.len();
        for (logical, group) in groups {
            if group.a.is_none() || group.b.is_none() {
                continue;
            }
            let target = match resolve_target(&logical) {
                Some(t) => t,
                None => continue,
            };
            // Resolve the (out, in) shape from the Flux config so the
            // NF4 codec knows how to lay out the dequantized tensor.
            let shape = match flux_target_shape(&target.base_key, cfg) {
                Some(s) => s,
                None => {
                    tracing::debug!(
                        target: "plakat",
                        "Flux NF4 LoRA: skipping unresolvable target {}",
                        target.base_key
                    );
                    continue;
                }
            };
            // Dequantize base once per target — subsequent LoRAs reuse
            // the already-dense version from the map.
            let base_dense = if let Some(existing) = overrides.get(&target.base_key) {
                existing.to_dtype(DType::F32)?
            } else {
                store
                    .dequantize_weight(&target.base_key, &[shape.0, shape.1])
                    .with_context(|| {
                        format!(
                            "dequantizing NF4 base for Flux LoRA target {}",
                            target.base_key
                        )
                    })?
            };
            let delta = compute_delta(&group, effective_scale, device)
                .with_context(|| format!("computing Flux LoRA delta for {logical}"))?;
            let updated = apply_delta(&base_dense, &delta, target.slice).with_context(
                || format!("applying Flux LoRA delta for {logical} into {}", target.base_key),
            )?;
            // BF16 storage matches the runtime dtype on GPU — keeps
            // override memory comparable to the equivalent BF16-Flux
            // Linear (~56 MB for a fused QKV vs the 4-bit packed
            // ~7 MB it replaces).
            let bf16 = updated.to_dtype(DType::BF16)?;
            overrides.insert(target.base_key.clone(), bf16);
            modified += 1;
        }
        if total_groups > 0 {
            tracing::info!(
                target: "plakat",
                "Flux LoRA (NF4) {} → {modified}/{total_groups} targets merged (scale {:.2})",
                lora.display,
                effective_scale
            );
        }
    }

    Ok((overrides, modified, total_groups))
}

pub fn precompute_quantized_overrides(
    qvb: &candle_transformers::quantized_var_builder::VarBuilder,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(HashMap<String, Tensor>, usize, usize)> {
    let mut overrides: HashMap<String, Tensor> = HashMap::new();
    let mut modified = 0usize;
    let mut total_groups = 0usize;

    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading Flux LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let groups = group_peft_keys(&lora_tensors);
        total_groups += groups.len();
        for (logical, group) in groups {
            if group.a.is_none() || group.b.is_none() {
                continue;
            }
            let target = match resolve_target(&logical) {
                Some(t) => t,
                None => continue,
            };
            // Dequantize base once per target — subsequent LoRAs reuse
            // the already-dense version from the map.
            let base_dense = if let Some(existing) = overrides.get(&target.base_key) {
                existing.clone()
            } else {
                let qt = qvb.get_no_shape(&target.base_key).with_context(|| {
                    format!(
                        "Flux LoRA target {} not found in GGUF — skipping",
                        target.base_key
                    )
                })?;
                qt.dequantize(device)?
            };
            let delta = compute_delta(&group, effective_scale, device)
                .with_context(|| format!("computing Flux LoRA delta for {logical}"))?;
            let updated = apply_delta(&base_dense, &delta, target.slice).with_context(
                || format!("applying Flux LoRA delta for {logical} into {}", target.base_key),
            )?;
            // Cast to BF16 — runtime dtype on GPU. F32 storage would
            // double memory per merged Linear with no quality win, and
            // candle's `QMatMul::Tensor` forward is dtype-agnostic.
            let bf16 = updated.to_dtype(DType::BF16)?;
            overrides.insert(target.base_key.clone(), bf16);
            modified += 1;
        }
        if total_groups > 0 {
            tracing::info!(
                target: "plakat",
                "Flux LoRA (GGUF) {} → {modified}/{total_groups} targets merged (scale {:.2})",
                lora.display,
                effective_scale
            );
        }
    }

    Ok((overrides, modified, total_groups))
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

    // ---------- AI-Toolkit / kohya format (v0.13 phase 1d) ----------

    #[test]
    fn resolves_aitoolkit_double_qkv_full() {
        // AI-Toolkit trains a single (3*hidden, hidden) delta against
        // the fused QKV Linear — so the slice is Full, not a partial
        // row slab like PEFT.
        let t = resolve_target("lora_unet_double_blocks_0_img_attn_qkv").unwrap();
        assert_eq!(t.base_key, "double_blocks.0.img_attn.qkv.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolves_aitoolkit_double_text_stream() {
        let t = resolve_target("lora_unet_double_blocks_7_txt_mlp_0").unwrap();
        assert_eq!(t.base_key, "double_blocks.7.txt_mlp.0.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolves_aitoolkit_single_linear1_full() {
        // single-block linear1 is the 4-way fused [Q;K;V;MLP_up].
        // AI-Toolkit deltas hit the whole fused matrix.
        let t = resolve_target("lora_unet_single_blocks_3_linear1").unwrap();
        assert_eq!(t.base_key, "single_blocks.3.linear1.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolves_aitoolkit_single_modulation() {
        let t = resolve_target("lora_unet_single_blocks_37_modulation_lin").unwrap();
        assert_eq!(t.base_key, "single_blocks.37.modulation.lin.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn resolves_aitoolkit_without_lora_unet_prefix() {
        // Some exports drop the `lora_unet_` prefix and write the
        // sub-path directly. Should still resolve.
        let t = resolve_target("double_blocks_5_img_attn_proj").unwrap();
        assert_eq!(t.base_key, "double_blocks.5.img_attn.proj.weight");
        assert!(matches!(t.slice, RowSlice::Full));
    }

    #[test]
    fn aitoolkit_unknown_subpaths_return_none() {
        // AI-Toolkit naming with a sub-path candle-Flux doesn't have.
        assert!(resolve_target("lora_unet_double_blocks_0_img_attn_qkv_extra").is_none());
        assert!(resolve_target("lora_unet_single_blocks_0_unknown").is_none());
        // Missing block index.
        assert!(resolve_target("lora_unet_double_blocks_img_attn_qkv").is_none());
    }

    #[test]
    fn aitoolkit_grouping_handles_kohya_suffixes() {
        // Real AI-Toolkit files use `lora_down`/`lora_up` (kohya
        // convention) — these are already accepted by group_peft_keys
        // since it falls back when `lora_A`/`lora_B` don't match.
        let mut t = HashMap::new();
        let a = Tensor::zeros((4, 3072), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::zeros((9216, 4), DType::F32, &Device::Cpu).unwrap();
        t.insert(
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight".to_string(),
            a,
        );
        t.insert(
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight".to_string(),
            b,
        );
        let groups = group_peft_keys(&t);
        let g = groups
            .get("lora_unet_double_blocks_0_img_attn_qkv")
            .expect("AI-Toolkit logical name should be the full underscore path");
        assert!(g.a.is_some() && g.b.is_some());
    }

    // ---------- v0.14 phase 8b — flux_target_shape ----------
    // Path → (out, in) derivation used by the NF4 LoRA precompute to
    // size dequant calls. The codec needs the explicit shape (unlike
    // GGUF QTensor, which carries its own metadata).

    fn dev_cfg() -> crate::pipelines::flux_inner::Config {
        crate::pipelines::flux_inner::Config::dev()
    }

    #[test]
    fn target_shape_double_block_qkv() {
        let cfg = dev_cfg();
        let h = cfg.hidden_size;
        let s = flux_target_shape("double_blocks.0.img_attn.qkv.weight", &cfg).unwrap();
        assert_eq!(s, (3 * h, h));
        // Same for the txt branch.
        let s = flux_target_shape("double_blocks.12.txt_attn.qkv.weight", &cfg).unwrap();
        assert_eq!(s, (3 * h, h));
    }

    #[test]
    fn target_shape_double_block_mlp() {
        let cfg = dev_cfg();
        let h = cfg.hidden_size;
        let mlp = (h as f64 * cfg.mlp_ratio) as usize;
        assert_eq!(
            flux_target_shape("double_blocks.0.img_mlp.0.weight", &cfg),
            Some((mlp, h))
        );
        assert_eq!(
            flux_target_shape("double_blocks.0.img_mlp.2.weight", &cfg),
            Some((h, mlp))
        );
    }

    #[test]
    fn target_shape_double_block_mod() {
        let cfg = dev_cfg();
        let h = cfg.hidden_size;
        assert_eq!(
            flux_target_shape("double_blocks.5.img_mod.lin.weight", &cfg),
            Some((6 * h, h))
        );
    }

    #[test]
    fn target_shape_single_block_fused() {
        let cfg = dev_cfg();
        let h = cfg.hidden_size;
        let mlp = (h as f64 * cfg.mlp_ratio) as usize;
        // linear1 fuses [Q; K; V; MLP_up] along dim 0.
        assert_eq!(
            flux_target_shape("single_blocks.10.linear1.weight", &cfg),
            Some((3 * h + mlp, h))
        );
        // linear2 is the final projection — input is concat(attn_out, mlp).
        assert_eq!(
            flux_target_shape("single_blocks.10.linear2.weight", &cfg),
            Some((h, h + mlp))
        );
    }

    #[test]
    fn target_shape_unknown_returns_none() {
        let cfg = dev_cfg();
        // Path that doesn't match any Flux Linear (e.g. a QkNorm
        // scale or some custom adapter key).
        assert_eq!(
            flux_target_shape("double_blocks.0.img_attn.norm.query_norm.scale", &cfg),
            None
        );
        assert_eq!(flux_target_shape("some.unrelated.path", &cfg), None);
    }
}
