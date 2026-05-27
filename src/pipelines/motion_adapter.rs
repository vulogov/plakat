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
        // Config first — small file (~200 bytes); failure here is
        // a clean "wrong repo / network down" diagnostic before
        // the 1.4 GB safetensors fetch.
        let config_path = crate::hf::download::get_file(REPO_V3, CONFIG_FILE)
            .await
            .with_context(|| {
                format!("downloading motion-adapter config from {REPO_V3}/{CONFIG_FILE}")
            })?;
        let config_bytes = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config = MotionAdapterConfig::from_json(&config_bytes)?;

        let weights_path =
            crate::hf::download::get_file(REPO_V3, WEIGHTS_FILE).await.with_context(|| {
                format!("downloading motion-adapter weights from {REPO_V3}/{WEIGHTS_FILE}")
            })?;

        let tensor_layout = read_tensor_layout(&weights_path)?;

        Ok(Self {
            config,
            weights_path,
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
