#![allow(dead_code)]

//! FaceID UNet LoRA conversion — currently unused for the standard
//! `_sd15` / `_sdxl` variants.
//!
//! h94 ships the UNet LoRA for the standard FaceID variants as a
//! *separate* kohya-format safetensors (`ip-adapter-faceid_sd15_lora.safetensors`
//! etc.) which our existing LoRA merger consumes directly with no
//! conversion. The converter below stays around because some FaceID
//! variants (`*_plus_*`, `*_portrait_*`, …) may bundle the LoRA
//! inline in `ip-adapter-faceid_*.bin` alongside `image_proj.*`. If we
//! ever wire those, this module is the conversion entry-point.
//!
//! ## Inline-bundled `.bin` layout (when applicable)
//!
//! Such a `.bin` would ship two things:
//!
//!   1. `image_proj.*` — the small MLP that maps the 512-d ArcFace
//!      embedding to 4 cross-attention tokens. Consumed by `FaceIdEncoder`.
//!   2. `ip_adapter.<idx>.*` — a UNet cross-attention LoRA.
//!
//! h94 stores the LoRA in a non-standard layout: `ip_adapter.<idx>` for
//! `idx ∈ [0, 16)` indexes diffusers' `unet.attn_processors.values()`
//! iteration over SD 1.5's 16 cross-attention layers. The LoRA is on
//! the `attn2` (cross-attention) `to_q` / `to_k` / `to_v` / `to_out[0]`
//! projections.
//!
//! This module:
//!   * Hardcodes the SD 1.5 iteration order (matches diffusers).
//!   * Reads the `ip_adapter.<idx>.*` keys from the .bin via
//!     `candle::pickle::PthTensors`.
//!   * Re-emits as kohya-format safetensors that our existing LoRA
//!     merger consumes without modification.
//!
//! ## What this does NOT do
//!
//! * **No SDXL FaceID LoRA conversion.** SDXL UNet has ~70 cross-attn
//!   layers (more transformer blocks per attention), in a different
//!   order. A separate path table + conversion entry-point is needed
//!   if an inline-bundled SDXL variant ever needs to be supported.
//! * **No verification of the diffusers iteration order against h94's
//!   training**. The order below is taken from diffusers' source for
//!   `UNet2DConditionModel.attn_processors`. If h94 used a different
//!   iteration, idx → layer is misaligned and the LoRA does subtly
//!   wrong things. Empirical testing required.
//!
//! ## Shared cross-attention caveat
//!
//! Reference IP-Adapter-FaceID applies this LoRA to the *decoupled*
//! IP cross-attention projections (`to_k_ip` / `to_v_ip`), separate
//! from the text-token K/V. candle 0.8's UNet has no attention hook
//! for that decoupling, so the LoRA here lands on the shared K/V
//! projections that handle BOTH text and image tokens. Net effect:
//! identity preservation improves, text-prompt fidelity may shift
//! slightly. Documented quality ceiling for the foreseeable future.

use anyhow::{Context, Result, bail};
use candle_core::{Device, pickle::PthTensors, safetensors};
use std::collections::HashMap;
use std::path::Path;

/// Diffusers' iteration order for SD 1.5 `UNet2DConditionModel.attn_processors`,
/// filtered to cross-attention layers only (`attn2`). The 16-element list
/// maps `ip_adapter.<idx>` from h94's .bin → the UNet cross-attention
/// module path.
///
/// Structure:
///   * `down_blocks.0` cross-attn (×2) → idx 0..1
///   * `down_blocks.1` cross-attn (×2) → idx 2..3
///   * `down_blocks.2` cross-attn (×2) → idx 4..5
///   * `mid_block` cross-attn (×1)    → idx 6
///   * `up_blocks.1` cross-attn (×3)   → idx 7..9
///   * `up_blocks.2` cross-attn (×3)   → idx 10..12
///   * `up_blocks.3` cross-attn (×3)   → idx 13..15
///
/// (`down_blocks.3` is a non-cross-attn `DownBlock2D`; `up_blocks.0` is a
/// non-cross-attn `UpBlock2D`; both are skipped by `attn_processors`.)
///
/// Paths use `_` separators (kohya convention). The LoRA merger
/// translates them back to diffusers' dot-form internally.
const SD15_CROSS_ATTN_PATHS: &[&str] = &[
    "down_blocks_0_attentions_0_transformer_blocks_0_attn2",
    "down_blocks_0_attentions_1_transformer_blocks_0_attn2",
    "down_blocks_1_attentions_0_transformer_blocks_0_attn2",
    "down_blocks_1_attentions_1_transformer_blocks_0_attn2",
    "down_blocks_2_attentions_0_transformer_blocks_0_attn2",
    "down_blocks_2_attentions_1_transformer_blocks_0_attn2",
    "mid_block_attentions_0_transformer_blocks_0_attn2",
    "up_blocks_1_attentions_0_transformer_blocks_0_attn2",
    "up_blocks_1_attentions_1_transformer_blocks_0_attn2",
    "up_blocks_1_attentions_2_transformer_blocks_0_attn2",
    "up_blocks_2_attentions_0_transformer_blocks_0_attn2",
    "up_blocks_2_attentions_1_transformer_blocks_0_attn2",
    "up_blocks_2_attentions_2_transformer_blocks_0_attn2",
    "up_blocks_3_attentions_0_transformer_blocks_0_attn2",
    "up_blocks_3_attentions_1_transformer_blocks_0_attn2",
    "up_blocks_3_attentions_2_transformer_blocks_0_attn2",
];

/// Diffusers' iteration order for SDXL's `UNet2DConditionModel.attn_processors`,
/// filtered to cross-attention layers (`attn2`). 70 entries total —
/// matches the SDXL FaceID `.bin`'s `ip_adapter.<idx>` indices 0..69.
///
/// SDXL UNet structure (cross-attention sites only):
///
/// | Block            | Attentions | Transformer blocks each | Sites |
/// |------------------|-----------:|------------------------:|------:|
/// | down_blocks.0    |          0 |                       — |     0 |
/// | down_blocks.1    |          2 |                       2 |     4 |
/// | down_blocks.2    |          2 |                      10 |    20 |
/// | mid_block        |          1 |                      10 |    10 |
/// | up_blocks.0      |          3 |                      10 |    30 |
/// | up_blocks.1      |          3 |                       2 |     6 |
/// | up_blocks.2      |          0 |                       — |     0 |
/// | **total**        |            |                         | **70** |
///
/// (`down_blocks.0` and `up_blocks.2` are non-cross-attn blocks, skipped
/// by `attn_processors`.)
fn sdxl_cross_attn_paths() -> Vec<String> {
    let mut paths = Vec::with_capacity(70);

    // down_blocks.1: 2 attentions × 2 transformer_blocks
    for attn in 0..2 {
        for tb in 0..2 {
            paths.push(format!(
                "down_blocks_1_attentions_{attn}_transformer_blocks_{tb}_attn2"
            ));
        }
    }
    // down_blocks.2: 2 attentions × 10 transformer_blocks
    for attn in 0..2 {
        for tb in 0..10 {
            paths.push(format!(
                "down_blocks_2_attentions_{attn}_transformer_blocks_{tb}_attn2"
            ));
        }
    }
    // mid_block: 1 attention × 10 transformer_blocks
    for tb in 0..10 {
        paths.push(format!(
            "mid_block_attentions_0_transformer_blocks_{tb}_attn2"
        ));
    }
    // up_blocks.0: 3 attentions × 10 transformer_blocks
    for attn in 0..3 {
        for tb in 0..10 {
            paths.push(format!(
                "up_blocks_0_attentions_{attn}_transformer_blocks_{tb}_attn2"
            ));
        }
    }
    // up_blocks.1: 3 attentions × 2 transformer_blocks
    for attn in 0..3 {
        for tb in 0..2 {
            paths.push(format!(
                "up_blocks_1_attentions_{attn}_transformer_blocks_{tb}_attn2"
            ));
        }
    }

    debug_assert_eq!(paths.len(), 70, "SDXL cross-attn count mismatch");
    paths
}

/// Source-side projection name → kohya target-side leaf.
/// `to_out_lora` lands on `to_out_0` because `to_out` is a PyTorch
/// `nn.Sequential([Linear, Dropout])` and the LoRA targets `[0]`.
const PROJ_MAP: &[(&str, &str)] = &[
    ("to_q_lora", "to_q"),
    ("to_k_lora", "to_k"),
    ("to_v_lora", "to_v"),
    ("to_out_lora", "to_out_0"),
];

/// Try multiple naming conventions for the lora sub-key. h94's files
/// across versions use either `<proj>.up.weight` or `<proj>.lora.up.weight`
/// inside `ip_adapter.<idx>`. We probe both.
fn lora_sub_key_candidates(side: &str) -> [&'static str; 2] {
    match side {
        "up" => ["up.weight", "lora.up.weight"],
        "down" => ["down.weight", "lora.down.weight"],
        _ => unreachable!("side must be up or down"),
    }
}

/// Convert h94's IP-Adapter-FaceID SD 1.5 .bin to a kohya-format
/// safetensors at `output_path`. The output is consumable by the
/// existing `merge_loras_into_weights` UNet pass without modification.
///
/// Returns the number of LoRA target layers populated. A return of 0
/// means the .bin had no recognisable `ip_adapter.*` subtree (probably
/// the wrong file).
pub fn convert_faceid_sd15_to_kohya(
    faceid_bin: &Path,
    output_path: &Path,
    device: &Device,
) -> Result<usize> {
    let paths: Vec<String> =
        SD15_CROSS_ATTN_PATHS.iter().map(|s| (*s).to_string()).collect();
    convert_with_paths(faceid_bin, output_path, &paths, "SD 1.5", device)
}

/// Convert h94's IP-Adapter-FaceID SDXL .bin to kohya safetensors.
/// SDXL UNet has ~70 cross-attention sites (vs SD 1.5's 16); the path
/// table is generated programmatically by `sdxl_cross_attn_paths`.
pub fn convert_faceid_sdxl_to_kohya(
    faceid_bin: &Path,
    output_path: &Path,
    device: &Device,
) -> Result<usize> {
    let paths = sdxl_cross_attn_paths();
    convert_with_paths(faceid_bin, output_path, &paths, "SDXL", device)
}

/// Shared core of the SD 1.5 + SDXL conversions. `paths` is the ordered
/// list mapping `ip_adapter.<idx>` → UNet cross-attn path (kohya
/// underscore form). `label` is a user-facing string for error messages.
fn convert_with_paths(
    faceid_bin: &Path,
    output_path: &Path,
    paths: &[String],
    label: &str,
    device: &Device,
) -> Result<usize> {
    let pth = PthTensors::new(faceid_bin, None).with_context(|| {
        format!("opening FaceID .bin {}", faceid_bin.display())
    })?;
    let info = pth.tensor_infos();

    // Quick sanity: there must be at least one `ip_adapter.0.*` key. If
    // not, this is probably a basic IP-Adapter file (no UNet LoRA) or
    // the wrong FaceID variant — bail out instead of silently writing
    // an empty safetensors.
    let has_ip_adapter = info
        .keys()
        .any(|k| k.starts_with("ip_adapter.0."));
    if !has_ip_adapter {
        bail!(
            "no `ip_adapter.*` keys in {} — wrong file or unsupported \
             FaceID variant for {label}.",
            faceid_bin.display()
        );
    }

    let mut out: HashMap<String, candle_core::Tensor> = HashMap::new();
    let mut targets_populated = 0usize;

    for (idx, path) in paths.iter().enumerate() {
        let mut any_proj_for_this_idx = false;
        for (proj_src, proj_dst) in PROJ_MAP {
            // Try each sub-key naming pattern; first one that resolves wins.
            // Both up + down must use the same pattern within a single proj.
            let up_tensor = lora_sub_key_candidates("up")
                .iter()
                .find_map(|sub| {
                    let key = format!("ip_adapter.{idx}.{proj_src}.{sub}");
                    pth.get(&key).ok().flatten()
                });
            let down_tensor = lora_sub_key_candidates("down")
                .iter()
                .find_map(|sub| {
                    let key = format!("ip_adapter.{idx}.{proj_src}.{sub}");
                    pth.get(&key).ok().flatten()
                });

            if let (Some(up), Some(down)) = (up_tensor, down_tensor) {
                let up = up.to_device(device)?;
                let down = down.to_device(device)?;
                let kohya_up =
                    format!("lora_unet_{path}_{proj_dst}.lora_up.weight");
                let kohya_down =
                    format!("lora_unet_{path}_{proj_dst}.lora_down.weight");
                out.insert(kohya_up, up);
                out.insert(kohya_down, down);
                any_proj_for_this_idx = true;
            }
            // Silently skip projections that aren't present — some FaceID
            // variants only LoRA a subset (e.g. K + V but not Q + Out).
        }
        if any_proj_for_this_idx {
            targets_populated += 1;
        }
    }

    if targets_populated == 0 {
        bail!(
            "FaceID .bin {} had ip_adapter.* keys but no LoRA pairs matched \
             expected (proj_lora.up/down.weight or proj_lora.lora.up/down.weight) \
             for {label} ({} expected paths). File format may have changed; \
             please open an issue with the file's key listing.",
            faceid_bin.display(),
            paths.len()
        );
    }

    safetensors::save(&out, output_path)
        .with_context(|| format!("writing kohya LoRA to {}", output_path.display()))?;
    Ok(targets_populated)
}
