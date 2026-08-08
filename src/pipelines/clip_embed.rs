//! CLIP joint image/text embeddings for visual search (`plakat photos`, RFC PHOTOS-1 Phase 7).
//!
//! Loads the full `openai/clip-vit-large-patch14` model (both towers + both projection heads — the
//! same `model.safetensors` the aesthetic scorer already caches) and produces L2-normalized 768-d
//! embeddings in the shared joint space, so `cosine(embed_text(query), embed_image(photo))` is the
//! standard CLIP similarity. Feeds a text→image "find images that look like this" search.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::{Activation, ClipTextConfig};
use candle_transformers::models::clip::{div_l2_norm, ClipConfig, ClipModel};
use tokenizers::Tokenizer;

use crate::pipelines::ip_adapter::clip_l_vision_config;

const CLIP_REPO: &str = "openai/clip-vit-large-patch14";
const MAX_TOKENS: usize = 77;

/// CLIP ViT-L/14 text-tower config (matches the `text_config` of the openai repo).
fn clip_l_text_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 49408,
        embed_dim: 768,
        intermediate_size: 3072,
        max_position_embeddings: MAX_TOKENS,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        projection_dim: 768,
        activation: Activation::QuickGelu,
    }
}

/// A loaded CLIP model + tokenizer that embeds images and text into the same 768-d space.
pub struct ClipEmbedder {
    model: ClipModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl ClipEmbedder {
    /// Load the model (from the shared openai CLIP `model.safetensors`) + tokenizer. ~1.7 GB F32.
    pub async fn load(device: &Device) -> Result<Self> {
        let weights = crate::hf::download::get_file(CLIP_REPO, "model.safetensors")
            .await
            .context("downloading CLIP weights")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)?
        };
        let cfg = ClipConfig {
            text_config: clip_l_text_config(),
            vision_config: clip_l_vision_config(),
            logit_scale_init_value: 2.6592,
            image_size: 224,
        };
        let model = ClipModel::new(vb, &cfg).context("building CLIP model")?;
        let tok_path = crate::hf::download::get_file(CLIP_REPO, "tokenizer.json")
            .await
            .context("downloading CLIP tokenizer")?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("loading CLIP tokenizer: {e}"))?;
        Ok(Self { model, tokenizer, device: device.clone() })
    }

    /// Whether the CLIP weights + tokenizer are already cached (no download) — for ETCH-1 L3's offline
    /// charter.
    pub fn is_cached() -> bool {
        crate::hf::download::file_is_cached(CLIP_REPO, "model.safetensors")
            && crate::hf::download::file_is_cached(CLIP_REPO, "tokenizer.json")
    }

    /// L2-normalized image embedding (768 floats) for the file at `path`.
    pub fn embed_image(&self, path: &Path) -> Result<Vec<f32>> {
        let pixels = crate::imaging::preprocess::clip_image_tensor(path, 224, &self.device, DType::F32)?;
        let feats = self.model.get_image_features(&pixels)?;
        to_unit_vec(&feats)
    }

    /// L2-normalized text embedding (768 floats) for `text`.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenize(text)?;
        let feats = self.model.get_text_features(&ids)?;
        to_unit_vec(&feats)
    }

    /// Tokenize `text` → a `(1, 77)` id tensor: the CLIP BPE tokens (BOS/EOS added), zero-padded to
    /// 77 (padding sits after EOS, harmless under CLIP's causal text attention + EOS pooling).
    fn tokenize(&self, text: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizing query: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS);
            *ids.last_mut().unwrap() = 49407; // keep EOS at the end for pooling
        } else {
            ids.resize(MAX_TOKENS, 0);
        }
        Ok(Tensor::new(ids, &self.device)?.unsqueeze(0)?)
    }
}

/// Flatten a (1, 768) feature tensor to an L2-normalized `Vec<f32>`.
fn to_unit_vec(feats: &Tensor) -> Result<Vec<f32>> {
    let normed = div_l2_norm(feats)?;
    Ok(normed.flatten_all()?.to_vec1::<f32>()?)
}

/// Cosine similarity of two already-unit-normalized embeddings (a plain dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::cosine;

    #[test]
    fn cosine_of_unit_vectors() {
        let a = [1.0, 0.0, 0.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &[0.0, 1.0, 0.0]).abs() < 1e-6);
        // Opposite direction → -1.
        assert!((cosine(&a, &[-1.0, 0.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    // Real load + forward. Downloads ~1.7 GB the first time; run explicitly:
    //   cargo test -p plakat --features photos -- --ignored clip_loads
    #[tokio::test]
    #[ignore]
    async fn clip_loads_and_embeds_into_joint_space() {
        use super::ClipEmbedder;
        let emb = ClipEmbedder::load(&candle_core::Device::Cpu).await.unwrap();
        let t = emb.embed_text("a red fox in the snow").unwrap();
        assert_eq!(t.len(), 768);
        assert!((t.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-3, "text embed is unit-norm");

        let dir = std::env::temp_dir().join("plakat-clip-smoke");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.png");
        image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(64, 64, image::Rgb([200, 30, 30])))
            .save(&p)
            .unwrap();
        let i = emb.embed_image(&p).unwrap();
        assert_eq!(i.len(), 768);
        let c = cosine(&t, &i);
        assert!(c.is_finite() && c.abs() <= 1.01, "cosine in range: {c}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
