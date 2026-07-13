//! LAION aesthetic predictor v2 — a 0–10 aesthetic score from a CLIP ViT-L/14 image embedding.
//!
//! The predictor is a tiny MLP (`768 → 1024 → 128 → 64 → 16 → 1`, no activations — a linear-MSE
//! head) over the **L2-normalised** CLIP ViT-L/14 *image* embedding. We reuse plakat's existing
//! `ImageEncoder` (the ViT-L vision tower Stable Cascade already loads) for the embedding, and load
//! the MLP straight from LAION's `.pth` via candle's pickle reader — no conversion/upload needed.
//!
//! Used by `plakat rank` (score + sort a directory) and `generate --keep-best K` (auto-keep the
//! top-K of a batch). The score is also the first sort/filter key the 3.0 collection manager needs.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::pipelines::ip_adapter::{clip_l_vision_config, ImageEncoder};

/// CLIP ViT-L/14 image encoder (openai) + the LAION aesthetic MLP.
const CLIP_REPO: &str = "openai/clip-vit-large-patch14";
const CLIP_FILE: &str = "model.safetensors";
const MLP_REPO: &str = "camenduru/improved-aesthetic-predictor";
const MLP_FILE: &str = "sac+logos+ava1-l14-linearMSE.pth";

pub struct AestheticScorer {
    encoder: ImageEncoder,
    /// The 5 Linear layers at `nn.Sequential` indices 0/2/4/6/7 (the others are inference-noop
    /// Dropouts). Applied in order with no activations between.
    layers: Vec<Linear>,
    device: Device,
}

impl AestheticScorer {
    /// Load the CLIP ViT-L vision tower + the aesthetic MLP. F32 throughout — the model is tiny and
    /// runs once per image, so accuracy (stable ranking) beats speed.
    pub async fn load(device: &Device) -> Result<Self> {
        let clip = crate::hf::download::get_file(CLIP_REPO, CLIP_FILE)
            .await
            .map_err(|e| anyhow::anyhow!("fetching CLIP ViT-L/14 for aesthetic scoring: {e:#}"))?;
        let encoder =
            ImageEncoder::load_with_config(&clip, &clip_l_vision_config(), device, DType::F32)?;

        let pth = crate::hf::download::get_file(MLP_REPO, MLP_FILE)
            .await
            .map_err(|e| anyhow::anyhow!("fetching LAION aesthetic predictor MLP: {e:#}"))?;
        let vb = VarBuilder::from_pth(&pth, DType::F32, device)?;
        let shapes = [(768usize, 1024usize), (1024, 128), (128, 64), (64, 16), (16, 1)];
        let idx = [0usize, 2, 4, 6, 7];
        let mut layers = Vec::with_capacity(5);
        for (i, (inp, out)) in idx.iter().zip(shapes.iter()) {
            layers.push(candle_nn::linear(*inp, *out, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self { encoder, layers, device: device.clone() })
    }

    /// Score an image file. Returns the raw predictor output (roughly 0–10; higher = more
    /// aesthetic per the LAION training distribution).
    pub fn score_path(&self, path: &Path) -> Result<f32> {
        let pixels = crate::imaging::preprocess::clip_image_tensor(path, 224, &self.device, DType::F32)?;
        self.score_pixels(&pixels)
    }

    fn score_pixels(&self, pixels: &Tensor) -> Result<f32> {
        let emb = self.encoder.encode(pixels)?; // (1, 768)
        // L2-normalise — the predictor was trained on normalised CLIP embeddings.
        let norm = emb.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let mut x = emb.broadcast_div(&norm)?.to_dtype(DType::F32)?;
        for l in &self.layers {
            x = l.forward(&x)?; // linear-MSE head: no activations between layers
        }
        Ok(x.flatten_all()?.to_vec1::<f32>()?[0])
    }
}
