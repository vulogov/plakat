//! CLIP text↔image **adherence**: how well a render matches its prompt, in the CLIP joint space. Loads the
//! CLIP ViT-L/14 TEXT tower + `text_projection` from the same `openai/clip-vit-large-patch14` weights the
//! aesthetic scorer already uses for images, so `image · text` (cosine of the L2-normalised joint
//! embeddings) is the contrastive adherence in ~[0.15, 0.35]. Used as a second objective in `--improve`
//! so a winning edit must stay FAITHFUL to the prompt, not just look prettier.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use tokenizers::Tokenizer;

use crate::pipelines::vendored_clip::{self, ClipTextTransformer};

const CLIP_REPO: &str = "openai/clip-vit-large-patch14";

pub struct ClipAdherence {
    text_encoder: ClipTextTransformer,
    text_projection: Linear,
    tokenizer: Tokenizer,
    device: Device,
    max_pos: usize,
    eos_id: u32,
}

impl ClipAdherence {
    /// Load the CLIP-L text tower + projection + tokenizer (F32). Co-locate on `device` with the aesthetic
    /// scorer's image tower so `image · text` is a single dot product.
    pub async fn load(device: &Device) -> Result<Self> {
        let clip = crate::hf::download::get_file(CLIP_REPO, "model.safetensors")
            .await
            .map_err(|e| anyhow::anyhow!("fetching CLIP ViT-L/14 for adherence: {e:#}"))?;
        let cfg = vendored_clip::Config::v1_5();
        let text_encoder = vendored_clip::build_clip_transformer(&cfg, &clip, device, DType::F32)?;
        // `text_projection` (768→768, no bias) maps the pooled EOS hidden into the joint space.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&clip], DType::F32, device)? };
        let text_projection = candle_nn::linear_no_bias(cfg.embed_dim, cfg.projection_dim, vb.pp("text_projection"))?;
        let tok_path = crate::hf::download::get_file(CLIP_REPO, "tokenizer.json")
            .await
            .map_err(|e| anyhow::anyhow!("fetching CLIP tokenizer for adherence: {e:#}"))?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("CLIP tokenizer: {e}"))?;
        let eos_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407);
        Ok(Self { text_encoder, text_projection, tokenizer, device: device.clone(), max_pos: cfg.max_position_embeddings, eos_id })
    }

    /// The L2-normalised CLIP joint TEXT embedding `(1, 768)` for a prompt.
    pub fn text_embedding(&self, text: &str) -> Result<Tensor> {
        let mut ids = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("CLIP tokenize: {e}"))?
            .get_ids()
            .to_vec();
        ids.truncate(self.max_pos);
        ids.resize(self.max_pos, self.eos_id);
        // HF CLIP pools at the EOS token = the highest id in the sequence; its first occurrence is the real
        // end-of-text (content ids and BOS are all < EOS).
        let eos_pos = ids.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
        let t = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden = self.text_encoder.forward(&t)?; // (1, 77, 768), final-LN applied
        let pooled = hidden.narrow(1, eos_pos, 1)?.squeeze(1)?; // (1, 768)
        let emb = self.text_projection.forward(&pooled)?; // (1, 768) joint space
        let norm = emb.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        Ok(emb.broadcast_div(&norm)?)
    }

    /// Contrastive adherence = cosine of the (already-normalised) image embedding and the text embedding,
    /// in ~[-1, 1] (typically 0.15–0.35 for a faithful render).
    pub fn adherence(&self, image_norm: &Tensor, text: &str) -> Result<f32> {
        let txt = self.text_embedding(text)?;
        Ok((image_norm * &txt)?.sum_all()?.to_scalar::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Manual validation (downloads CLIP): a matching prompt must out-score a mismatched one on the same
    /// image. Run with: `cargo test --features metal -p plakat clip_adherence -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn matching_prompt_outscores_mismatch() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let device = crate::device::select("auto").unwrap();
            let aes = crate::pipelines::aesthetic::AestheticScorer::load(&device).await.unwrap();
            let clip = ClipAdherence::load(&device).await.unwrap();
            let img = std::path::Path::new("corpus/images/bookart-titlepage/frontispiece_crop.png");
            let emb = aes.image_embedding(img).unwrap();
            let ship = clip.adherence(&emb, "a tall sailing ship on the sea, engraving").unwrap();
            let food = clip.adherence(&emb, "a plate of spaghetti and meatballs").unwrap();
            println!("adherence: ship={ship:.4}  food={food:.4}");
            assert!(ship > food, "matching prompt ({ship:.4}) should beat mismatch ({food:.4})");
            assert!(ship > 0.15, "a faithful match should score > 0.15 (got {ship:.4})");
            // Text-tower sanity (independent of the image): a prompt matches itself ≈ 1.0, a near-synonym
            // high, an unrelated topic low — confirms EOS pooling + projection are correct.
            let cat = clip.text_embedding("a photograph of a cat").unwrap();
            let same = (&cat * &clip.text_embedding("a photograph of a cat").unwrap()).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let kitten = (&cat * &clip.text_embedding("a photo of a kitten").unwrap()).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let physics = (&cat * &clip.text_embedding("a diagram of quantum field theory").unwrap()).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            println!("text-text: same={same:.4}  kitten={kitten:.4}  physics={physics:.4}");
            assert!(same > 0.99, "identical text ≈ 1.0 (got {same:.4})");
            assert!(kitten > physics + 0.1, "kitten ({kitten:.4}) ≫ physics ({physics:.4})");
        });
    }
}
