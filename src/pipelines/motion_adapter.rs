//! v0.26 phase 1: AnimateDiff V3 motion-adapter loader.
//!
//! Downloads the motion-adapter weights from HF (
//! `guoyww/animatediff-motion-adapter-v1-5-3` — ~1.4 GB), parses
//! its `config.json`, and exposes the per-block tensor layout so
//! phase 2's UNet integration has the target known.
//!
//! ## Architecture (V3)
//!
//! Per `config.json`:
//!
//! ```jsonc
//! {
//!   "block_out_channels":              [320, 640, 1280, 1280], // SD 1.5 block dims
//!   "motion_layers_per_block":         2,    // 2 temporal transformer blocks
//!                                            // per UNet down-/up- block
//!   "motion_max_seq_length":           32,   // max frame count the temporal
//!                                            // position-encoding supports
//!   "motion_mid_block_layers_per_block": 1,
//!   "motion_norm_num_groups":          32,
//!   "motion_num_attention_heads":      8,
//!   "use_motion_mid_block":            false // V3 SKIPS the mid block.
//!                                            // V1/V2 have it.
//! }
//! ```
//!
//! V3 has **4 down-blocks + 4 up-blocks × 2 motion layers each =
//! 16 motion modules**. Mid-block is skipped. Total motion-module
//! parameters: ~1.4 GB.
//!
//! Tensor naming (diffusers convention) — for each motion module:
//!
//! ```text
//! down_blocks.{0..=3}.motion_modules.{0,1}.
//!   temporal_transformer.norm.{weight,bias}
//!   temporal_transformer.proj_in.{weight,bias}
//!   temporal_transformer.transformer_blocks.0.
//!     attention_blocks.{0,1}.{to_q,to_k,to_v,to_out.0}.weight
//!     ff.net.{0.proj,2}.{weight,bias}
//!     norms.{0,1,2}.{weight,bias}
//!   temporal_transformer.proj_out.{weight,bias}
//!
//! up_blocks.{0..=3}.motion_modules.{0,1}.
//!   ... (same shape as down_blocks)
//! ```
//!
//! ## What this module ships in phase 1
//!
//! - [`MotionAdapterConfig`]: serde-deserialized config struct.
//! - [`MotionAdapter`]: holds the loaded safetensors path + config
//!   + an enumerated tensor key list. Phase 2 consumes this to
//!   build the actual `TemporalAttention` modules and splice them
//!   into the SD 1.5 UNet forward pass.
//! - [`MotionAdapter::load`]: async loader that downloads via
//!   `crate::hf::download::get_first_of` (one canonical repo, no
//!   fallback mirror).
//! - [`MotionAdapter::summary`]: human-readable layout dump for
//!   the phase 2 dev loop.
//!
//! Phase 1 does NOT yet:
//! - Build per-block temporal-attention modules (phase 2)
//! - Splice into the SD 1.5 UNet (phase 2)
//! - Sample N-frame latents (phase 3)
//!
//! Loader runs on every plakat startup that uses
//! `--animatediff`; cache-hits after first download.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

/// HF repo + filename pair. Single source — no fallback mirrors
/// for v0.26 (the repo is canonical + stable). If the upstream
/// goes down, users can override via `--motion-adapter-from PATH`
/// in phase 4.
const REPO_V3: &str = "guoyww/animatediff-motion-adapter-v1-5-3";
/// v0.27 phase 1: SDXL motion adapter (beta). Same `MotionAdapter`
/// config schema as V3 — only `block_out_channels` differs
/// (`[320, 640, 1280]` for SDXL vs `[320, 640, 1280, 1280]` for V3
/// SD 1.5).
const REPO_SDXL_BETA: &str = "guoyww/animatediff-motion-adapter-sdxl-beta";
/// v0.28 phase 1: AnimateLCM motion adapter for 4-step generation.
/// Same V3-compatible block channels for SD 1.5 (`[320, 640, 1280,
/// 1280]`) but flips `use_motion_mid_block` to true (V1/V2 style)
/// so the adapter has 17 modules instead of 16. Pairs with the LCM
/// scheduler + 4-step guidance-distilled inference for a ~5x
/// speedup vs V3 + DDIM at 20 steps. Upstream:
/// <https://huggingface.co/wangfuyun/AnimateLCM>.
const REPO_ANIMATELCM: &str = "wangfuyun/AnimateLCM";
const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "diffusion_pytorch_model.safetensors";

/// V3 motion-adapter config — exactly matches the upstream JSON.
/// Each field is documented inline so phase 2's integration can
/// reference one source of truth.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MotionAdapterConfig {
    /// Always `"MotionAdapter"` for valid V3 weights.
    #[serde(default, rename = "_class_name")]
    pub class_name: String,

    /// Diffusers version the config was authored for. Informational.
    #[serde(default, rename = "_diffusers_version")]
    pub diffusers_version: String,

    /// SD 1.5 UNet block channel dimensions: `[320, 640, 1280, 1280]`.
    /// Each motion module attaches to a block of the matching
    /// dimensionality. SDXL motion adapters use different channels
    /// (`[320, 640, 1280]`) — phase 1 ships SD 1.5 only (RFC Q2).
    pub block_out_channels: Vec<usize>,

    /// Number of temporal transformer blocks per UNet down-/up-
    /// block. V3 ships 2. Each block carries the
    /// `attention_blocks.{0,1}` pair (self-attention + cross-
    /// attention, though cross-attn is identity for motion).
    pub motion_layers_per_block: usize,

    /// Maximum frame count the temporal positional embedding
    /// supports. V3 is 32, which is double V1/V2's 24. Generation
    /// beyond this rolls back to the start (or fails — phase 3
    /// decides the policy).
    pub motion_max_seq_length: usize,

    /// V1/V2 have a mid-block motion module; V3 doesn't (see
    /// `use_motion_mid_block`). Field is present for forward
    /// compat.
    pub motion_mid_block_layers_per_block: usize,

    /// GroupNorm group count for the temporal layers. Matches
    /// SD 1.5's GroupNorm convention.
    pub motion_norm_num_groups: usize,

    /// Multi-head attention head count. V3 is 8.
    pub motion_num_attention_heads: usize,

    /// V3 = false (no mid-block motion module). V1/V2 = true.
    /// Phase 2 gates the mid-block splice on this flag.
    #[serde(default)]
    pub use_motion_mid_block: bool,
}

impl MotionAdapterConfig {
    /// Parse a `config.json` payload. Fails loudly on missing
    /// required fields (block_out_channels, motion_layers_per_block,
    /// motion_num_attention_heads).
    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Self =
            serde_json::from_str(s).context("parsing motion-adapter config.json")?;
        cfg.validate()
            .context("validating motion-adapter config")?;
        Ok(cfg)
    }

    /// Sanity-check the deserialized config against the
    /// invariants phase 2 will rely on. Cheap; runs on every load.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.block_out_channels.is_empty(),
            "motion-adapter: block_out_channels must be non-empty"
        );
        anyhow::ensure!(
            self.motion_layers_per_block > 0,
            "motion-adapter: motion_layers_per_block must be > 0"
        );
        anyhow::ensure!(
            self.motion_num_attention_heads > 0,
            "motion-adapter: motion_num_attention_heads must be > 0"
        );
        anyhow::ensure!(
            self.motion_max_seq_length > 0,
            "motion-adapter: motion_max_seq_length must be > 0"
        );
        Ok(())
    }

    /// Number of down-/up-blocks the adapter splices into.
    /// SD 1.5 = 4 (this is `block_out_channels.len()`).
    pub fn num_blocks(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Total count of motion modules across down + up + (optional) mid.
    /// V3 SD 1.5: 4 down × 2 + 4 up × 2 + 0 mid = 16.
    pub fn total_motion_modules(&self) -> usize {
        let nb = self.num_blocks();
        let down_up = 2 * nb * self.motion_layers_per_block;
        let mid = if self.use_motion_mid_block {
            self.motion_mid_block_layers_per_block
        } else {
            0
        };
        down_up + mid
    }
}

/// Loaded V3 motion adapter. Holds the path to the on-disk
/// safetensors, the parsed config, and the enumerated tensor key
/// list (for phase 2's per-block module construction). Doesn't
/// hold the actual tensor data — phase 2's UNet splice builds
/// per-block modules from the safetensors directly.
pub struct MotionAdapter {
    /// Parsed `config.json`.
    pub config: MotionAdapterConfig,
    /// Resolved path to the safetensors file on disk
    /// (post-download, post-HF-cache).
    pub weights_path: PathBuf,
    /// Sorted list of tensor keys + their shapes. Surfaced for
    /// phase 2's dev loop and for [`Self::summary`].
    tensor_layout: Vec<(String, Vec<usize>)>,
}

impl MotionAdapter {
    /// Download (if needed) and load the V3 motion adapter.
    /// Network-required on first run; cache-hits subsequently.
    pub async fn load_v3() -> Result<Self> {
        Self::load_from_repo(REPO_V3).await
    }

    /// v0.27 phase 1: download (if needed) and load the SDXL beta
    /// motion adapter from `guoyww/animatediff-motion-adapter-sdxl-beta`.
    /// Same config schema as V3 — see [`MotionAdapterConfig`]. Differs
    /// only in `block_out_channels` ([320, 640, 1280] for SDXL's
    /// 3-block UNet vs V3's [320, 640, 1280, 1280] for SD 1.5's 4-block
    /// UNet). Network-required on first run; cache-hits subsequently.
    pub async fn load_sdxl_beta() -> Result<Self> {
        Self::load_from_repo(REPO_SDXL_BETA).await
    }

    /// v0.28 phase 1: download (if needed) and load the AnimateLCM
    /// motion adapter from `wangfuyun/AnimateLCM`. SD 1.5 base
    /// architecture (`block_out_channels = [320, 640, 1280, 1280]`)
    /// but uses the V1/V2-style mid-block motion module
    /// (`use_motion_mid_block = true`) so the adapter has 17
    /// modules instead of V3's 16.
    ///
    /// Pairs with the LCM scheduler at 4 denoise steps for ~5×
    /// speedup vs V3 + DDIM at 20 steps. Caller is responsible
    /// for setting the scheduler + step count appropriately.
    /// Network-required on first run; cache-hits subsequently.
    pub async fn load_animatelcm() -> Result<Self> {
        Self::load_from_repo(REPO_ANIMATELCM).await
    }

    /// Inner loader shared by `load_v3` + `load_sdxl_beta`. Downloads
    /// `config.json` + `diffusion_pytorch_model.safetensors` from the
    /// given repo and returns the parsed pair.
    async fn load_from_repo(repo: &str) -> Result<Self> {
        // Config first — small file; failure here is a clean
        // "wrong repo / network down" diagnostic before the >1 GB
        // safetensors fetch.
        let config_path = crate::hf::download::get_file(repo, CONFIG_FILE)
            .await
            .with_context(|| {
                format!("downloading motion-adapter config from {repo}/{CONFIG_FILE}")
            })?;
        let config_bytes = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config = MotionAdapterConfig::from_json(&config_bytes)?;

        let weights_path = crate::hf::download::get_file(repo, WEIGHTS_FILE)
            .await
            .with_context(|| {
                format!("downloading motion-adapter weights from {repo}/{WEIGHTS_FILE}")
            })?;

        let tensor_layout = read_tensor_layout(&weights_path)?;

        Ok(Self {
            config,
            weights_path,
            tensor_layout,
        })
    }

    /// v0.26 phase 4: load V3 with motion LoRAs merged in. Equivalent
    /// to `load_v3()` when `motion_loras` is empty; otherwise
    /// downloads each LoRA via [`crate::pipelines::lora::LoraSpec::resolve`],
    /// merges into a tempfile via [`crate::pipelines::lora::merge_loras_into_weights`]
    /// with `MergeTarget::MOTION_ADAPTER`, and loads the merged
    /// tempfile. The tempfile lives as long as the returned
    /// [`MotionAdapter`] — drop the adapter to release it.
    pub async fn load_v3_with_motion_loras(
        motion_loras: &[crate::pipelines::lora::LoraSpec],
        default_scale: f32,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Self::load_with_motion_loras(
            Self::load_v3().await?,
            motion_loras,
            default_scale,
            device,
        )
        .await
    }

    /// v0.27 phase 1: load SDXL beta with motion LoRAs merged in.
    /// Parallels [`Self::load_v3_with_motion_loras`]. Motion-LoRA
    /// tensor-key naming is the same across V3 and SDXL beta
    /// (PEFT-bare keys; the `MergeTarget::MOTION_ADAPTER` lookup
    /// table is shared).
    pub async fn load_sdxl_beta_with_motion_loras(
        motion_loras: &[crate::pipelines::lora::LoraSpec],
        default_scale: f32,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Self::load_with_motion_loras(
            Self::load_sdxl_beta().await?,
            motion_loras,
            default_scale,
            device,
        )
        .await
    }

    /// v0.28 phase 3: synthetic constructor for tests that need a
    /// `MotionAdapter` without touching the network. Builds an
    /// in-memory instance with the provided config + an empty
    /// safetensors layout. Not for production use — the
    /// `weights_path` points at a non-existent dummy path.
    #[doc(hidden)]
    pub fn synthetic_for_test(config: MotionAdapterConfig) -> Self {
        Self {
            config,
            weights_path: std::path::PathBuf::from("/dev/null/synthetic-motion-adapter.safetensors"),
            tensor_layout: Vec::new(),
        }
    }

    /// v0.28 phase 1: load AnimateLCM with motion LoRAs merged in.
    /// Same tensor-key convention as V3 / SDXL beta — the
    /// `MergeTarget::MOTION_ADAPTER` lookup table is shared.
    pub async fn load_animatelcm_with_motion_loras(
        motion_loras: &[crate::pipelines::lora::LoraSpec],
        default_scale: f32,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Self::load_with_motion_loras(
            Self::load_animatelcm().await?,
            motion_loras,
            default_scale,
            device,
        )
        .await
    }

    /// Shared LoRA-merge step. Takes a freshly-loaded base adapter
    /// and either returns it unchanged (empty LoRA list) or merges
    /// the LoRAs into a tempfile and rebuilds the adapter from the
    /// merged weights.
    async fn load_with_motion_loras(
        base: Self,
        motion_loras: &[crate::pipelines::lora::LoraSpec],
        default_scale: f32,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if motion_loras.is_empty() {
            return Ok(base);
        }

        let mut resolved = Vec::with_capacity(motion_loras.len());
        for (i, spec) in motion_loras.iter().enumerate() {
            resolved.push(
                spec.resolve()
                    .await
                    .with_context(|| format!("resolving motion LoRA #{i}"))?,
            );
        }

        let tmp = tempfile::Builder::new()
            .prefix("plakat-motion-lora-merged-")
            .suffix(".safetensors")
            .tempfile()
            .context("creating tempfile for merged motion-adapter")?;
        let (modified, total) = crate::pipelines::lora::merge_loras_into_weights(
            &base.weights_path,
            tmp.path(),
            &resolved,
            default_scale,
            device,
            crate::pipelines::lora::MergeTarget::MOTION_ADAPTER,
        )
        .context("merging motion LoRAs into motion-adapter weights")?;
        tracing::info!(
            target: "plakat",
            "motion LoRA merge: {modified}/{total} targets across {} LoRA(s)",
            resolved.len(),
        );

        // Detach the tempfile so the merged safetensors outlives
        // this call. The returned PathBuf lives on `MotionAdapter`
        // until drop — OS reclaims at process exit.
        let (_file, merged_path) = tmp
            .keep()
            .context("detaching merged motion-adapter tempfile")?;
        let tensor_layout = read_tensor_layout(&merged_path)?;
        Ok(Self {
            config: base.config,
            weights_path: merged_path,
            tensor_layout,
        })
    }

    /// Total number of tensors in the adapter. Sanity check after
    /// load.
    pub fn tensor_count(&self) -> usize {
        self.tensor_layout.len()
    }

    /// Iterator over `(key, shape)` for every loaded tensor.
    /// Sorted by key. Used by phase 2 to enumerate per-block
    /// modules at construction time.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &[usize])> {
        self.tensor_layout
            .iter()
            .map(|(k, s)| (k.as_str(), s.as_slice()))
    }

    /// Open a [`VarBuilder`] rooted at the adapter's safetensors.
    /// Phase 2 will use this to materialize per-block weights.
    pub fn varbuilder(&self, dtype: DType, device: &Device) -> Result<VarBuilder<'_>> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&self.weights_path], dtype, device)?
        };
        Ok(vb)
    }

    /// Human-readable layout summary. Useful for the phase 2 dev
    /// loop + for `--verbose` logging. Lists the per-block module
    /// breakdown derived from the config.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "AnimateDiff V3 motion adapter\n\
             ──────────────────────────────\n\
             weights:        {}\n\
             tensor count:   {}\n\
             block channels: {:?}\n\
             layers/block:   {}\n\
             attn heads:     {}\n\
             max frames:     {}\n\
             mid-block:      {}\n\
             total modules:  {}\n",
            self.weights_path.display(),
            self.tensor_count(),
            self.config.block_out_channels,
            self.config.motion_layers_per_block,
            self.config.motion_num_attention_heads,
            self.config.motion_max_seq_length,
            if self.config.use_motion_mid_block {
                format!("yes ({} layers)", self.config.motion_mid_block_layers_per_block)
            } else {
                "no (V3)".to_string()
            },
            self.config.total_motion_modules(),
        ));

        // Per-block prefix tensor counts — a quick sanity scan.
        // Phase 2 needs this to ensure every block's motion module
        // has the right tensor pack before constructing it.
        let mut down_counts = vec![0usize; self.config.num_blocks()];
        let mut up_counts = vec![0usize; self.config.num_blocks()];
        let mut mid_count = 0usize;
        let mut other_count = 0usize;
        for (key, _shape) in &self.tensor_layout {
            if let Some(rest) = key.strip_prefix("down_blocks.") {
                if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
                {
                    if idx < down_counts.len() {
                        down_counts[idx] += 1;
                        continue;
                    }
                }
            }
            if let Some(rest) = key.strip_prefix("up_blocks.") {
                if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
                {
                    if idx < up_counts.len() {
                        up_counts[idx] += 1;
                        continue;
                    }
                }
            }
            if key.starts_with("mid_block.") {
                mid_count += 1;
                continue;
            }
            other_count += 1;
        }
        out.push_str(&format!(
            "\nPer-block tensor counts:\n  down_blocks: {:?}\n  up_blocks:   {:?}\n  mid_block:   {}\n  other:       {}\n",
            down_counts, up_counts, mid_count, other_count,
        ));
        out
    }
}

/// Read the safetensors header without mapping the data into
/// memory. Returns `(key, shape)` pairs sorted by key.
///
/// We need only the layout for the phase 2 dev loop and the
/// summary — the actual tensor bytes get mmap'd later when phase
/// 2 builds the modules.
fn read_tensor_layout(path: &std::path::Path) -> Result<Vec<(String, Vec<usize>)>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading {} for header inspection", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing safetensors header at {}", path.display()))?;
    let mut out: Vec<(String, Vec<usize>)> = st
        .names()
        .into_iter()
        .map(|name| {
            let view = st.tensor(name).expect("name from names() must resolve");
            (name.to_string(), view.shape().to_vec())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real V3 config.json, captured 2026-05-27 from
    /// `guoyww/animatediff-motion-adapter-v1-5-3/raw/main/config.json`.
    /// Pinned here so the config-shape test works offline.
    const V3_CONFIG_JSON: &str = r#"{
        "_class_name": "MotionAdapter",
        "_diffusers_version": "0.25.0.dev0",
        "block_out_channels": [320, 640, 1280, 1280],
        "motion_layers_per_block": 2,
        "motion_max_seq_length": 32,
        "motion_mid_block_layers_per_block": 1,
        "motion_norm_num_groups": 32,
        "motion_num_attention_heads": 8,
        "use_motion_mid_block": false
    }"#;

    #[test]
    fn config_parses_v3() {
        let cfg = MotionAdapterConfig::from_json(V3_CONFIG_JSON).unwrap();
        assert_eq!(cfg.class_name, "MotionAdapter");
        assert_eq!(cfg.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(cfg.motion_layers_per_block, 2);
        assert_eq!(cfg.motion_max_seq_length, 32);
        assert_eq!(cfg.motion_num_attention_heads, 8);
        assert!(!cfg.use_motion_mid_block);
    }

    #[test]
    fn config_validate_rejects_empty_blocks() {
        let bad = MotionAdapterConfig {
            class_name: "MotionAdapter".into(),
            diffusers_version: "test".into(),
            block_out_channels: vec![],
            motion_layers_per_block: 2,
            motion_max_seq_length: 32,
            motion_mid_block_layers_per_block: 1,
            motion_norm_num_groups: 32,
            motion_num_attention_heads: 8,
            use_motion_mid_block: false,
        };
        let err = bad.validate().unwrap_err();
        assert!(err.to_string().contains("block_out_channels"));
    }

    #[test]
    fn config_validate_rejects_zero_layers() {
        let bad = MotionAdapterConfig {
            class_name: "MotionAdapter".into(),
            diffusers_version: "test".into(),
            block_out_channels: vec![320],
            motion_layers_per_block: 0,
            motion_max_seq_length: 32,
            motion_mid_block_layers_per_block: 1,
            motion_norm_num_groups: 32,
            motion_num_attention_heads: 8,
            use_motion_mid_block: false,
        };
        let err = bad.validate().unwrap_err();
        assert!(err.to_string().contains("motion_layers_per_block"));
    }

    #[test]
    fn config_validate_rejects_zero_heads() {
        let bad = MotionAdapterConfig {
            class_name: "MotionAdapter".into(),
            diffusers_version: "test".into(),
            block_out_channels: vec![320],
            motion_layers_per_block: 2,
            motion_max_seq_length: 32,
            motion_mid_block_layers_per_block: 1,
            motion_norm_num_groups: 32,
            motion_num_attention_heads: 0,
            use_motion_mid_block: false,
        };
        let err = bad.validate().unwrap_err();
        assert!(err.to_string().contains("attention_heads"));
    }

    #[test]
    fn config_validate_rejects_zero_seq_length() {
        let bad = MotionAdapterConfig {
            class_name: "MotionAdapter".into(),
            diffusers_version: "test".into(),
            block_out_channels: vec![320],
            motion_layers_per_block: 2,
            motion_max_seq_length: 0,
            motion_mid_block_layers_per_block: 1,
            motion_norm_num_groups: 32,
            motion_num_attention_heads: 8,
            use_motion_mid_block: false,
        };
        let err = bad.validate().unwrap_err();
        assert!(err.to_string().contains("max_seq_length"));
    }

    /// V3 SD 1.5: 4 down × 2 + 4 up × 2 + 0 mid = 16 motion modules.
    #[test]
    fn v3_total_motion_modules_count() {
        let cfg = MotionAdapterConfig::from_json(V3_CONFIG_JSON).unwrap();
        assert_eq!(cfg.num_blocks(), 4);
        assert_eq!(cfg.total_motion_modules(), 16);
    }

    /// v0.27 phase 1: the SDXL beta motion-adapter config from
    /// `guoyww/animatediff-motion-adapter-sdxl-beta/config.json`,
    /// captured 2026-05-28 via WebFetch. Pinned offline so the
    /// schema test doesn't need network.
    const SDXL_BETA_CONFIG_JSON: &str = r#"{
        "_class_name": "MotionAdapter",
        "_diffusers_version": "0.26.0.dev0",
        "block_out_channels": [320, 640, 1280],
        "motion_layers_per_block": 2,
        "motion_max_seq_length": 32,
        "motion_mid_block_layers_per_block": 1,
        "motion_norm_num_groups": 32,
        "motion_num_attention_heads": 8,
        "use_motion_mid_block": false
    }"#;

    /// v0.28 phase 1: AnimateLCM config from `wangfuyun/AnimateLCM`,
    /// captured 2026-05-28 via WebFetch. Same SD 1.5 block channels
    /// as V3, but flips `use_motion_mid_block` to true and adds a
    /// `conv_in_channels: null` field that serde ignores. Total
    /// motion modules = 17 (4 down × 2 + 4 up × 2 + 1 mid).
    const ANIMATELCM_CONFIG_JSON: &str = r#"{
        "_class_name": "MotionAdapter",
        "_diffusers_version": "0.27.0.dev0",
        "block_out_channels": [320, 640, 1280, 1280],
        "conv_in_channels": null,
        "motion_layers_per_block": 2,
        "motion_max_seq_length": 32,
        "motion_mid_block_layers_per_block": 1,
        "motion_norm_num_groups": 32,
        "motion_num_attention_heads": 8,
        "use_motion_mid_block": true
    }"#;

    #[test]
    fn config_parses_animatelcm() {
        let cfg = MotionAdapterConfig::from_json(ANIMATELCM_CONFIG_JSON).unwrap();
        assert_eq!(cfg.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(cfg.motion_layers_per_block, 2);
        assert_eq!(cfg.motion_max_seq_length, 32);
        assert!(cfg.use_motion_mid_block, "AnimateLCM uses mid-block motion");
        // 4 down × 2 + 4 up × 2 + 1 mid = 17.
        assert_eq!(cfg.num_blocks(), 4);
        assert_eq!(cfg.total_motion_modules(), 17);
    }

    /// SDXL beta config parses with the same schema as V3; only
    /// `block_out_channels` differs (3 blocks for SDXL vs 4 for V3).
    #[test]
    fn config_parses_sdxl_beta() {
        let cfg = MotionAdapterConfig::from_json(SDXL_BETA_CONFIG_JSON).unwrap();
        assert_eq!(cfg.class_name, "MotionAdapter");
        assert_eq!(cfg.block_out_channels, vec![320, 640, 1280]);
        assert_eq!(cfg.motion_layers_per_block, 2);
        assert_eq!(cfg.motion_max_seq_length, 32);
        assert_eq!(cfg.motion_num_attention_heads, 8);
        assert!(!cfg.use_motion_mid_block);
        // 3 down × 2 + 3 up × 2 = 12 modules. No mid in SDXL beta.
        assert_eq!(cfg.num_blocks(), 3);
        assert_eq!(cfg.total_motion_modules(), 12);
    }

    /// V2-style config (mid block present) shifts the count.
    #[test]
    fn v2_style_total_motion_modules_count() {
        let v2_json = r#"{
            "_class_name": "MotionAdapter",
            "_diffusers_version": "0.21.0",
            "block_out_channels": [320, 640, 1280, 1280],
            "motion_layers_per_block": 2,
            "motion_max_seq_length": 24,
            "motion_mid_block_layers_per_block": 1,
            "motion_norm_num_groups": 32,
            "motion_num_attention_heads": 8,
            "use_motion_mid_block": true
        }"#;
        let cfg = MotionAdapterConfig::from_json(v2_json).unwrap();
        // 4 down × 2 + 4 up × 2 + 1 mid = 17.
        assert_eq!(cfg.total_motion_modules(), 17);
        assert!(cfg.use_motion_mid_block);
    }

    #[test]
    fn config_rejects_malformed_json() {
        let err = MotionAdapterConfig::from_json("{ not valid").unwrap_err();
        assert!(err.to_string().contains("parsing"));
    }

    #[test]
    fn config_rejects_missing_required_field() {
        // No block_out_channels — required.
        let bad = r#"{
            "_class_name": "MotionAdapter",
            "motion_layers_per_block": 2,
            "motion_max_seq_length": 32,
            "motion_mid_block_layers_per_block": 1,
            "motion_norm_num_groups": 32,
            "motion_num_attention_heads": 8,
            "use_motion_mid_block": false
        }"#;
        let err = MotionAdapterConfig::from_json(bad).unwrap_err();
        assert!(err.to_string().contains("parsing"));
    }

    /// Motion LoRA merge: writing a synthetic adapter +
    /// synthetic LoRA, calling `merge_loras_into_weights` with
    /// `MergeTarget::MOTION_ADAPTER`, asserts the merged file
    /// has the same key set and modified at least one target.
    /// No network required.
    #[test]
    fn motion_lora_merges_into_synthetic_adapter() {
        use crate::pipelines::lora::{
            MergeTarget, ResolvedLora, merge_loras_into_weights,
        };
        use candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Build a minimal synthetic motion-adapter safetensors
        // with ONE attention block's worth of tensors. Real V3
        // has 16 modules × ~20 tensors each; one's enough for the
        // merge round-trip test.
        let tmp_dir = tempfile::tempdir().unwrap();
        let adapter_path = tmp_dir.path().join("adapter.safetensors");
        let lora_path = tmp_dir.path().join("motion-lora.safetensors");
        let out_path = tmp_dir.path().join("merged.safetensors");

        let dim = 16;
        let rank = 4;

        let base_key =
            "down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_q.weight".to_string();
        let base_weight = Tensor::randn(0.0f32, 1.0, (dim, dim), &device).unwrap();
        let mut adapter: HashMap<String, Tensor> = HashMap::new();
        adapter.insert(base_key.clone(), base_weight.clone());
        candle_core::safetensors::save(&adapter, &adapter_path).unwrap();

        // Build the motion LoRA: down + up matrices at the same
        // key path, suffixed `.lora.down.weight` and `.lora.up.weight`.
        // The merge logic strips these suffixes to find the base.
        let stem = base_key.strip_suffix(".weight").unwrap();
        let lora_down = Tensor::randn(0.0f32, 0.01, (rank, dim), &device).unwrap();
        let lora_up = Tensor::randn(0.0f32, 0.01, (dim, rank), &device).unwrap();
        let mut lora: HashMap<String, Tensor> = HashMap::new();
        lora.insert(format!("{stem}.lora.down.weight"), lora_down);
        lora.insert(format!("{stem}.lora.up.weight"), lora_up);
        candle_core::safetensors::save(&lora, &lora_path).unwrap();

        let resolved = ResolvedLora {
            path: lora_path,
            scale: 1.0,
            display: "test-motion-lora".to_string(),
        };
        let (modified, total) = merge_loras_into_weights(
            &adapter_path,
            &out_path,
            &[resolved],
            1.0,
            &device,
            MergeTarget::MOTION_ADAPTER,
        )
        .expect("merge succeeds");
        assert_eq!(total, 1, "should see exactly 1 LoRA target group");
        assert_eq!(modified, 1, "should modify the single matching target");

        // Verify the merged file has the same base key and that
        // the value differs from the original (i.e. the delta
        // was actually applied).
        let merged: HashMap<String, Tensor> =
            candle_core::safetensors::load(&out_path, &device).unwrap();
        assert!(merged.contains_key(&base_key));
        let merged_w = merged.get(&base_key).unwrap();
        let diff = (merged_w - &base_weight)
            .unwrap()
            .abs()
            .unwrap()
            .mean_all()
            .unwrap();
        let v: f32 = diff.to_vec0().unwrap();
        // With random small init, the delta should be visibly
        // non-zero. Loose bound — the actual value depends on
        // the random init.
        let _ = dtype;
        assert!(v > 0.0, "merged tensor identical to base — delta not applied");
    }

    /// Empty motion-LoRA list is a no-op: load_v3_with_motion_loras
    /// returns the same MotionAdapter as load_v3 (no tempfile, no
    /// merge).
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    #[ignore] // network — downloads V3 adapter
    async fn load_v3_with_empty_motion_loras_is_noop() {
        use candle_core::Device;
        let adapter =
            MotionAdapter::load_v3_with_motion_loras(&[], 1.0, &Device::Cpu)
                .await
                .expect("load");
        // Same tensor count as direct load.
        let direct = MotionAdapter::load_v3().await.unwrap();
        assert_eq!(adapter.tensor_count(), direct.tensor_count());
        assert_eq!(adapter.weights_path, direct.weights_path);
    }

    /// Network-required: downloads ~1.4 GB on first run. Run
    /// explicitly via `cargo test --test motion_adapter_smoke
    /// -- --ignored` once that file lands. For now skip by
    /// default — same pattern as `style_detect_smoke`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    #[ignore]
    async fn load_v3_end_to_end_smoke() {
        let adapter = MotionAdapter::load_v3().await.expect("download + load V3");
        assert_eq!(adapter.config.block_out_channels, vec![320, 640, 1280, 1280]);
        assert_eq!(adapter.config.motion_layers_per_block, 2);
        assert!(!adapter.config.use_motion_mid_block);
        assert!(adapter.tensor_count() > 0);
        // V3 has 16 motion modules; each carries ~20 tensors
        // (LayerNorms, proj_in/out, transformer blocks). Loose
        // sanity bound: at least 16 × 10 tensors.
        assert!(
            adapter.tensor_count() >= 160,
            "unexpectedly few tensors: {}",
            adapter.tensor_count()
        );

        // Every tensor should be under one of the expected prefixes.
        for (key, _shape) in adapter.tensors() {
            let ok = key.starts_with("down_blocks.")
                || key.starts_with("up_blocks.")
                || key.starts_with("mid_block.");
            assert!(ok, "unexpected tensor key: {key}");
        }

        // Surface the layout for human inspection when running
        // with `--nocapture`.
        eprintln!("{}", adapter.summary());
    }
}
