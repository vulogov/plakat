//! Flux Redux — BFL's official image-conditioning adapter.
//!
//! Redux is the simplest "image prompt → Flux" recipe: it encodes a
//! reference image through Google's SigLIP-so400m-patch14-384, then
//! projects the resulting 729 patch embeddings (1152-d each) into
//! T5's 4096-d hidden space via a tiny 2-layer MLP. Those 729
//! "image tokens" get **sequence-concatenated** to T5's text tokens
//! before the standard Flux transformer forward.
//!
//! This is fundamentally different from SD's IP-Adapter (which adds
//! cross-attention layers to the UNet) and from XLabs / InstantX
//! Flux-IPA forks (which inject extra K/V projections inside each
//! Flux block). Redux requires **zero changes** to Flux's
//! transformer — the only delta is the text conditioning growing
//! from `(B, t5_seq, 4096)` to `(B, t5_seq + 729, 4096)`.
//!
//! ## Adapter shape (per BFL's reference)
//!
//! ```text
//!   ReduxAdapter(x: (B, 729, 1152)) ->
//!     redux_up   (Linear 1152 -> 3 * 4096 = 12288)
//!     silu
//!     redux_down (Linear 12288 -> 4096)
//!   -> (B, 729, 4096)
//! ```
//!
//! Weights ship as `black-forest-labs/FLUX.1-Redux-dev/flux1-redux-dev.safetensors`
//! (~140 MB). Keys: `redux_up.{weight,bias}`, `redux_down.{weight,bias}`.
//!
//! ## SigLIP weights
//!
//! `google/siglip-so400m-patch14-384` ships text + vision in one
//! safetensors. Only the vision tower is needed (with `use_head =
//! false` so we get the raw post-LayerNorm patch sequence, not the
//! pooled head output).
//!
//! ## Image preprocessing
//!
//! SigLIP-so400m expects 384×384 RGB normalised to `[-1, 1]`
//! (mean = std = 0.5). Resize via the triangle filter — same default
//! plakat uses for the SD AE preprocessing.

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::siglip;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// One Redux reference image with an associated weight. Parsed from
/// CLI specs of the form `path` (weight = 1.0) or `path:weight=F`.
///
/// `weight` scales the 729 image tokens before they get concatenated
/// onto the T5 hidden state — `0.0` makes the image contribute
/// nothing (effectively turning it off), `1.0` is full strength
/// (BFL's recipe), values up to ~2.0 amplify. Negative or non-finite
/// weights bail at parse time.
#[derive(Debug, Clone, PartialEq)]
pub struct ReduxSpec {
    pub path: PathBuf,
    pub weight: f32,
}

impl ReduxSpec {
    /// Build a spec for a single image at full strength.
    pub fn at_default_weight(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            weight: 1.0,
        }
    }
}

impl FromStr for ReduxSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        // Grammar:
        //   `path`                  → weight = 1.0
        //   `path:weight=F.F`       → weight = F.F (named, to disambiguate
        //                              from filenames containing colons)
        let (raw_path, weight) = if let Some((p, opts)) = s.rsplit_once(':') {
            if let Some(w) = opts.strip_prefix("weight=") {
                let w: f32 = w.parse().with_context(|| {
                    format!("--redux-image '{s}': can't parse weight={w:?}")
                })?;
                (p, w)
            } else {
                // Treat the whole string as the path — `:opts` without
                // `weight=` prefix is probably part of the filename.
                (s, 1.0)
            }
        } else {
            (s, 1.0)
        };
        if !weight.is_finite() {
            bail!("--redux-image '{s}': weight must be finite (got {weight})");
        }
        if weight < 0.0 {
            bail!("--redux-image '{s}': weight must be ≥ 0 (got {weight})");
        }
        Ok(Self {
            path: PathBuf::from(raw_path),
            weight,
        })
    }
}

/// Small 2-layer MLP that projects SigLIP-so400m patch embeddings
/// (1152-d) into Flux's text hidden dimension (4096-d). One MLP
/// instance per pipeline; weights are tiny (~140 MB BF16) compared
/// to SigLIP (~860 MB) or the Flux transformer (~24 GB).
#[derive(Debug, Clone)]
pub struct ReduxAdapter {
    redux_up: Linear,
    redux_down: Linear,
}

impl ReduxAdapter {
    /// Standard Redux config: SigLIP-so400m → Flux T5 hidden dim.
    pub const SIGLIP_DIM: usize = 1152;
    pub const T5_DIM: usize = 4096;
    /// `redux_up` widens to 3 × T5_DIM = 12288 before the silu
    /// activation, then `redux_down` projects back to T5_DIM.
    pub const HIDDEN_DIM: usize = 3 * Self::T5_DIM;

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let redux_up = candle_nn::linear(Self::SIGLIP_DIM, Self::HIDDEN_DIM, vb.pp("redux_up"))?;
        let redux_down = candle_nn::linear(Self::HIDDEN_DIM, Self::T5_DIM, vb.pp("redux_down"))?;
        Ok(Self {
            redux_up,
            redux_down,
        })
    }

    /// Forward: `(B, N, 1152) -> (B, N, 4096)`. For SigLIP-so400m
    /// patch-14 at 384², N = 27² = 729.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.redux_up.forward(xs)?.silu()?;
        Ok(self.redux_down.forward(&h)?)
    }
}

/// End-to-end image → Redux tokens encoder. Bundles the SigLIP
/// vision tower with the Redux adapter MLP.
pub struct ReduxEncoder {
    siglip: siglip::VisionModel,
    adapter: ReduxAdapter,
    dtype: DType,
    device: Device,
}

impl ReduxEncoder {
    /// `siglip_repo`: HF repo with the SigLIP vision tower (e.g.
    /// `google/siglip-so400m-patch14-384`).
    /// `redux_repo`: HF repo with the Redux adapter weights (e.g.
    /// `black-forest-labs/FLUX.1-Redux-dev`).
    /// `dtype`: runtime dtype the SigLIP + adapter forwards run in.
    /// BF16 on GPU; F32 on CPU (matches the Flux pipeline's choice).
    pub async fn load(
        siglip_repo: &str,
        redux_repo: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // SigLIP: load config to size the model, then weights.
        let siglip_cfg_path =
            crate::hf::download::get_file(siglip_repo, "config.json").await?;
        let siglip_cfg_str = std::fs::read_to_string(&siglip_cfg_path)
            .with_context(|| format!("read SigLIP config {}", siglip_cfg_path.display()))?;
        let cfg: siglip::Config = serde_json::from_str(&siglip_cfg_str)
            .context("parse SigLIP config")?;
        let vision_cfg = cfg.vision_config;
        // SigLIP-so400m-patch14-384 hidden_size must be 1152 — bail
        // loud rather than silently mis-projecting through the Redux
        // adapter (which is hard-wired to 1152 → 12288).
        if vision_cfg.hidden_size != ReduxAdapter::SIGLIP_DIM {
            return Err(anyhow!(
                "Redux: SigLIP vision encoder hidden_size = {} but adapter expects {} \
                 (try `google/siglip-so400m-patch14-384`).",
                vision_cfg.hidden_size,
                ReduxAdapter::SIGLIP_DIM
            ));
        }
        let siglip_weights = crate::hf::download::get_first_of(&[
            (siglip_repo, "model.fp16.safetensors"),
            (siglip_repo, "model.safetensors"),
        ])
        .await?;
        let siglip_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&siglip_weights], dtype, device)?
        };
        // SigLIP saves text + vision side-by-side under `text_model.*`
        // and `vision_model.*`. Use the vision subtree.
        let siglip = siglip::VisionModel::new(&vision_cfg, false, siglip_vb.pp("vision_model"))?;

        // Redux adapter.
        let redux_weights = crate::hf::download::get_first_of(&[
            (redux_repo, "flux1-redux-dev.safetensors"),
            (redux_repo, "diffusion_pytorch_model.safetensors"),
        ])
        .await
        .context("downloading Redux adapter weights")?;
        let redux_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&redux_weights], dtype, device)?
        };
        let adapter = ReduxAdapter::new(redux_vb)?;

        Ok(Self {
            siglip,
            adapter,
            dtype,
            device: device.clone(),
        })
    }

    /// Encode a reference image into Redux tokens `(1, 729, 4096)`
    /// ready to seq-concat onto a T5 hidden state.
    pub fn encode_image(&self, path: &Path) -> Result<Tensor> {
        self.encode_image_scaled(path, 1.0)
    }

    /// Encode + scale: same as `encode_image` but multiplies the
    /// resulting tokens by `weight` before returning. A weight of
    /// `0.0` produces an all-zero tensor that contributes nothing to
    /// the attention; `1.0` is BFL's default; larger values amplify.
    /// Skips the SigLIP + adapter forward entirely when weight = 0
    /// to save the work.
    pub fn encode_image_scaled(&self, path: &Path, weight: f32) -> Result<Tensor> {
        if !weight.is_finite() || weight < 0.0 {
            anyhow::bail!(
                "Redux weight must be finite ≥ 0 (got {weight})",
            );
        }
        let pixels = preprocess_image_for_siglip(path, &self.device, self.dtype)?;
        let siglip_out = self.siglip.forward(&pixels)?;
        let mut tokens = self.adapter.forward(&siglip_out)?;
        if (weight - 1.0).abs() > f32::EPSILON {
            tokens = (tokens * weight as f64)?;
        }
        Ok(tokens)
    }
}

/// Resize an image to 384×384 RGB and normalise to `[-1, 1]` (SigLIP
/// convention). Returns `(1, 3, 384, 384)` at `dtype` on `device`.
pub fn preprocess_image_for_siglip(
    path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // Standard SigLIP normalisation: mean = std = 0.5 across all
    // channels, i.e. pixel/255 then * 2 - 1.
    let img = image::open(path)
        .with_context(|| format!("opening Redux reference image {}", path.display()))?;
    let resized = img.resize_exact(
        SIGLIP_INPUT_SIZE as u32,
        SIGLIP_INPUT_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let mut buf: Vec<f32> = Vec::with_capacity(3 * SIGLIP_INPUT_SIZE * SIGLIP_INPUT_SIZE);
    // Channels-first: emit R plane, then G, then B (NCHW layout).
    for c in 0..3 {
        for y in 0..SIGLIP_INPUT_SIZE {
            for x in 0..SIGLIP_INPUT_SIZE {
                let p = rgb.get_pixel(x as u32, y as u32).0[c];
                // pixel/255 → [0, 1] → 2x - 1 → [-1, 1].
                buf.push((p as f32) / 127.5 - 1.0);
            }
        }
    }
    let t = Tensor::from_vec(
        buf,
        (1, 3, SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE),
        device,
    )?
    .to_dtype(dtype)?;
    Ok(t)
}

pub const SIGLIP_INPUT_SIZE: usize = 384;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Build a fake ReduxAdapter from a VarMap so we can exercise the
    /// forward without downloading the real ~140 MB weights. Confirms
    /// the dimension chain (1152 -> 12288 -> 4096) and the silu
    /// activation slot.
    #[test]
    fn adapter_forward_shape() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &cpu());
        let adapter = ReduxAdapter::new(vb).unwrap();
        // (B=2, N=729, 1152) input — same shape SigLIP-so400m emits
        // at 384² with patch 14.
        let x = Tensor::zeros((2, 729, ReduxAdapter::SIGLIP_DIM), DType::F32, &cpu()).unwrap();
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 729, ReduxAdapter::T5_DIM]);
    }

    // v0.14 phase 3c — `ReduxSpec` parser.

    #[test]
    fn spec_parses_bare_path() {
        let s: ReduxSpec = "ref.png".parse().unwrap();
        assert_eq!(s.path, PathBuf::from("ref.png"));
        assert_eq!(s.weight, 1.0);
    }

    #[test]
    fn spec_parses_path_with_weight() {
        let s: ReduxSpec = "./images/ref.png:weight=0.7".parse().unwrap();
        assert_eq!(s.path, PathBuf::from("./images/ref.png"));
        assert!((s.weight - 0.7).abs() < 1e-6);
    }

    #[test]
    fn spec_filename_with_colon_no_weight() {
        // A `:` not followed by `weight=` is treated as part of the
        // path (e.g. Windows-style filenames are rare here but the
        // safeguard means existing files don't get reinterpreted).
        let s: ReduxSpec = "weird:name.png".parse().unwrap();
        assert_eq!(s.path, PathBuf::from("weird:name.png"));
        assert_eq!(s.weight, 1.0);
    }

    #[test]
    fn spec_rejects_negative_weight() {
        let err = "ref.png:weight=-0.1".parse::<ReduxSpec>().unwrap_err();
        assert!(format!("{err}").contains("weight must be"), "{err}");
    }

    #[test]
    fn spec_rejects_nan_weight() {
        let err = "ref.png:weight=NaN".parse::<ReduxSpec>().unwrap_err();
        assert!(format!("{err}").contains("must be finite"), "{err}");
    }

    /// Zero input → linear layers (without bias zeroed in VarMap
    /// default-init) shouldn't NaN. Mostly a smoke test for the silu
    /// + matmul plumbing.
    #[test]
    fn adapter_forward_finite() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &cpu());
        let adapter = ReduxAdapter::new(vb).unwrap();
        let x = Tensor::ones((1, 729, ReduxAdapter::SIGLIP_DIM), DType::F32, &cpu()).unwrap();
        let y = adapter.forward(&x).unwrap();
        let mins: f32 = y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let maxs: f32 = y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(mins.is_finite(), "min not finite: {mins}");
        assert!(maxs.is_finite(), "max not finite: {maxs}");
    }
}
