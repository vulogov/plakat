//! OWL-ViT open-vocabulary object detector (`google/owlvit-base-patch32`) in candle
//! (ROADMAP_4.10.0). Powers `plakat remove --what "<text>"`: given an image + a text query, it
//! predicts a box per patch and a text-similarity logit, and we take the best-scoring box.
//!
//! OWL-ViT ≈ CLIP ViT-B/32 (at **768px**) vision + CLIP text + two small heads. The encoders are
//! reused from candle (`ClipVisionTransformer` / `ClipTextTransformer`); this module adds the
//! OWL-ViT-specific pieces:
//!   * the **class-token merge** — `patch_embeds * broadcast(class_token)` then an extra LayerNorm,
//!   * the **box head** (MLP → cxcywh, biased by the patch grid position), and
//!   * the **class head** (project image feats into the text space, cosine-sim vs the query, then a
//!     learned per-box shift/scale).
//!
//! Weight quirk: OWL-ViT spells the CLIP pre-layernorm `pre_layernorm`, candle expects the original
//! typo `pre_layrnorm` — we remap that one key when building the vision VarBuilder.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::clip::text_model::{ClipTextConfig, ClipTextTransformer};
use candle_transformers::models::clip::vision_model::{ClipVisionConfig, ClipVisionTransformer};
use std::collections::HashMap;

const IMAGE_SIZE: usize = 768;
const PATCH: usize = 32;
const GRID: usize = IMAGE_SIZE / PATCH; // 24
const VDIM: usize = 768; // vision hidden
const TDIM: usize = 512; // text hidden
const MAX_QUERY_LEN: usize = 16;

/// CLIP mean/std (OpenAI), for the `detect` preprocessing path.
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// A detected box in pixel coordinates `[x0, y0, x1, y1]` with its score.
#[derive(Clone, Copy, Debug)]
pub struct Detection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

pub struct OwlViT {
    vision: ClipVisionTransformer,
    post_layernorm: LayerNorm, // owlvit.vision_model.post_layernorm (applied to the full sequence)
    merge_ln: LayerNorm,       // owlvit.layer_norm (after the class-token merge)
    text: ClipTextTransformer,
    text_projection: Linear,
    box_dense0: Linear,
    box_dense1: Linear,
    box_dense2: Linear,
    class_dense0: Linear,
    logit_shift: Linear,
    logit_scale: Linear,
    box_bias: Tensor, // (GRID*GRID, 4), logit space
    tokenizer: Option<tokenizers::Tokenizer>,
    device: Device,
}

impl OwlViT {
    /// Load from a single `model.safetensors` (F32). Vision + text use candle's CLIP encoders (with
    /// the `pre_layernorm`→`pre_layrnorm` key remap); the heads/projections load directly.
    pub fn load(weights: &std::path::Path, device: &Device) -> Result<Self> {
        let all = candle_core::safetensors::load(weights, device).context("loading OWL-ViT weights")?;

        // Vision sub-VarBuilder: strip prefix + remap the pre-layernorm key.
        let mut vmap: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in &all {
            if let Some(rest) = k.strip_prefix("owlvit.vision_model.") {
                let rest = if rest.starts_with("pre_layernorm") {
                    rest.replacen("pre_layernorm", "pre_layrnorm", 1)
                } else {
                    rest.to_string()
                };
                vmap.insert(rest, v.clone());
            }
        }
        let vvb = VarBuilder::from_tensors(vmap, DType::F32, device);
        let mut vcfg = ClipVisionConfig::vit_base_patch32();
        vcfg.image_size = IMAGE_SIZE;
        let vision = ClipVisionTransformer::new(vvb, &vcfg).context("OWL-ViT vision tower")?;

        // Text sub-VarBuilder.
        let mut tmap: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in &all {
            if let Some(rest) = k.strip_prefix("owlvit.text_model.") {
                tmap.insert(rest.to_string(), v.clone());
            }
        }
        let tvb = VarBuilder::from_tensors(tmap, DType::F32, device);
        let mut tcfg = ClipTextConfig::vit_base_patch32();
        tcfg.max_position_embeddings = MAX_QUERY_LEN;
        let text = ClipTextTransformer::new(tvb, &tcfg).context("OWL-ViT text tower")?;

        // Heads + projections (full keys).
        let hvb = VarBuilder::from_tensors(all, DType::F32, device);
        let post_layernorm = layer_norm(VDIM, 1e-5, hvb.pp("owlvit.vision_model.post_layernorm"))?;
        let merge_ln = layer_norm(VDIM, 1e-5, hvb.pp("layer_norm"))?;
        let text_projection = linear_no_bias(TDIM, TDIM, hvb.pp("owlvit.text_projection"))?;
        let box_dense0 = linear(VDIM, VDIM, hvb.pp("box_head.dense0"))?;
        let box_dense1 = linear(VDIM, VDIM, hvb.pp("box_head.dense1"))?;
        let box_dense2 = linear(VDIM, 4, hvb.pp("box_head.dense2"))?;
        let class_dense0 = linear(VDIM, TDIM, hvb.pp("class_head.dense0"))?;
        let logit_shift = linear(VDIM, 1, hvb.pp("class_head.logit_shift"))?;
        let logit_scale = linear(VDIM, 1, hvb.pp("class_head.logit_scale"))?;
        let box_bias = compute_box_bias(GRID, GRID, device)?;

        // Tokenizer sits alongside the weights in the snapshot dir (needed for `detect`, not verify).
        let tokenizer = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists())
            .and_then(|p| tokenizers::Tokenizer::from_file(&p).ok());

        Ok(Self {
            vision,
            post_layernorm,
            merge_ln,
            text,
            text_projection,
            box_dense0,
            box_dense1,
            box_dense2,
            class_dense0,
            logit_shift,
            logit_scale,
            box_bias,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Image features `(B, GRID², VDIM)` — the post-LN'd patch tokens merged with the class token.
    pub fn image_feats(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hs = self.vision.output_hidden_states(pixel_values)?;
        // candle pushes the pooled+post-LN token last; the full 577-token sequence is the one before.
        let seq = hs
            .iter()
            .rev()
            .find(|t| t.dims().len() == 3 && t.dim(1).unwrap_or(0) == GRID * GRID + 1)
            .context("OWL-ViT: no full-sequence vision hidden state")?;
        let x = self.post_layernorm.forward(seq)?; // (B, 577, VDIM)
        let (b, n, d) = x.dims3()?;
        let class_tok = x.i((.., 0..1, ..))?.broadcast_as((b, n - 1, d))?; // (B,576,D)
        let patches = x.i((.., 1.., ..))?; // (B,576,D)
        let merged = (patches * class_tok)?;
        Ok(self.merge_ln.forward(&merged)?)
    }

    /// Query embeddings `(num_queries, TDIM)` — EOS-pooled CLIP text (candle pools at the argmax of
    /// input_ids) → text_projection.
    pub fn query_embeds(&self, input_ids: &Tensor) -> Result<Tensor> {
        let pooled = self.text.forward(input_ids)?; // (num_q, TDIM)
        Ok(self.text_projection.forward(&pooled)?)
    }

    /// Full forward → `(pred_boxes (B, GRID², 4) cxcywh in [0,1], logits (B, GRID², num_queries))`.
    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let feats = self.image_feats(pixel_values)?;
        let q = self.query_embeds(input_ids)?;

        // Box head → cxcywh (bias in logit space, then sigmoid).
        let bx = self.box_dense0.forward(&feats)?.gelu_erf()?;
        let bx = self.box_dense1.forward(&bx)?.gelu_erf()?;
        let bx = self.box_dense2.forward(&bx)?; // (B,576,4)
        let boxes = candle_nn::ops::sigmoid(&bx.broadcast_add(&self.box_bias)?)?;

        // Class head → cosine-sim logits with a learned per-box shift/scale.
        let ice = self.class_dense0.forward(&feats)?; // (B,576,TDIM)
        let ice = l2_normalize(&ice)?;
        let qn = l2_normalize(&q)?; // (num_q, TDIM)
        // (B,576,TDIM) @ (TDIM,num_q) → (B,576,num_q)
        let (b, p, _) = ice.dims3()?;
        let logits = ice.broadcast_matmul(&qn.t()?)?;
        let shift = self.logit_shift.forward(&feats)?; // (B,576,1)
        let scale = (self.logit_scale.forward(&feats)?.elu(1.0)? + 1.0)?; // (B,576,1)
        let logits = logits.broadcast_add(&shift)?.broadcast_mul(&scale)?;
        let _ = (b, p);
        Ok((boxes, logits))
    }

    /// Detect the single best-scoring box for `query` in `image` (any size). Returns pixel `[x0,y0,x1,y1]`
    /// in the ORIGINAL image coordinates, or `None` if the top score (sigmoid) is below `threshold`.
    pub fn detect(&self, image_path: &std::path::Path, query: &str, threshold: f32) -> Result<Option<Detection>> {
        let (orig_w, orig_h) = image::image_dimensions(image_path)?;
        let pixel_values = self.preprocess_image(image_path)?;
        let input_ids = self.tokenize(query)?;
        let (boxes, logits) = self.forward(&pixel_values, &input_ids)?;
        // best patch by logit for query 0.
        let logit = logits.i((0, .., 0))?; // (576,)
        let scores = candle_nn::ops::sigmoid(&logit)?.to_vec1::<f32>()?;
        let (best, &best_score) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .context("no patches")?;
        if best_score < threshold {
            return Ok(None);
        }
        let cxcywh = boxes.i((0, best, ..))?.to_vec1::<f32>()?; // in [0,1] of the 768 square
        // OWL-ViT boxes are relative to the padded square (longest side). The processor pads to a
        // square, so map [0,1] → pixels on the max(orig_w,orig_h) square, then clip to the image.
        let side = orig_w.max(orig_h) as f32;
        let (cx, cy, bw, bh) = (cxcywh[0] * side, cxcywh[1] * side, cxcywh[2] * side, cxcywh[3] * side);
        let x0 = (cx - bw / 2.0).clamp(0.0, orig_w as f32);
        let y0 = (cy - bh / 2.0).clamp(0.0, orig_h as f32);
        let x1 = (cx + bw / 2.0).clamp(0.0, orig_w as f32);
        let y1 = (cy + bh / 2.0).clamp(0.0, orig_h as f32);
        Ok(Some(Detection { x0, y0, x1, y1, score: best_score }))
    }

    /// Preprocess an image the OWL-ViT way: pad to a square (longest side), resize to 768, CLIP
    /// mean/std normalize → `(1, 3, 768, 768)` F32.
    fn preprocess_image(&self, path: &std::path::Path) -> Result<Tensor> {
        let img = image::open(path)?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let side = w.max(h);
        // Pad to a top-left square (OWL-ViT pads with 0.5 grey after normalization ≈ pad raw then norm).
        let mut square = image::RgbImage::from_pixel(side, side, image::Rgb([0, 0, 0]));
        image::imageops::overlay(&mut square, &img, 0, 0);
        let resized = image::imageops::resize(&square, IMAGE_SIZE as u32, IMAGE_SIZE as u32, image::imageops::FilterType::Triangle);
        let mut data = vec![0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
        for (x, y, p) in resized.enumerate_pixels() {
            for c in 0..3 {
                let v = p.0[c] as f32 / 255.0;
                data[c * IMAGE_SIZE * IMAGE_SIZE + (y as usize) * IMAGE_SIZE + x as usize] =
                    (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
        Ok(Tensor::from_vec(data, (1, 3, IMAGE_SIZE, IMAGE_SIZE), &self.device)?)
    }

    /// Tokenize a single query with the OWL-ViT (CLIP BPE) tokenizer, padded to MAX_QUERY_LEN.
    fn tokenize(&self, query: &str) -> Result<Tensor> {
        let tok = self.tokenizer.as_ref().context("OWL-ViT tokenizer not loaded (no tokenizer.json in snapshot)")?;
        let enc = tok.encode(query, true).map_err(|e| anyhow::anyhow!("tokenizing query: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        ids.truncate(MAX_QUERY_LEN);
        while ids.len() < MAX_QUERY_LEN {
            ids.push(0);
        }
        Ok(Tensor::from_vec(ids, (1, MAX_QUERY_LEN), &self.device)?)
    }
}

/// L2-normalize the last dim (+1e-6, OWL-ViT's epsilon).
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + 1e-6)?;
    Ok(x.broadcast_div(&norm)?)
}

/// OWL-ViT box bias: each patch's box center is biased to its grid position, size to the patch size.
/// `(h*w, 4)` in logit space (added before the box sigmoid).
fn compute_box_bias(h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let logit = |c: f32| (c + 1e-4).ln() - (1.0 - c + 1e-4).ln();
    let size = 1.0 / w as f32; // == 1/h (square grid)
    let size_bias = logit(size.clamp(0.0, 1.0));
    let mut v = Vec::with_capacity(h * w * 4);
    for i in 0..h {
        for j in 0..w {
            let cx = ((j as f32 + 1.0) / w as f32).clamp(0.0, 1.0);
            let cy = ((i as f32 + 1.0) / h as f32).clamp(0.0, 1.0);
            v.push(logit(cx));
            v.push(logit(cy));
            v.push(size_bias);
            v.push(size_bias);
        }
    }
    Ok(Tensor::from_vec(v, (h * w, 4), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(&b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da.sqrt() * db.sqrt() + 1e-12)
    }

    /// Verify the OWL-ViT port against a diffusers/transformers dump. Opt-in (`PLAKAT_OWLVIT_VERIFY=1`);
    /// needs the checkpoint cached + `owlvit` goldens.
    #[test]
    fn owlvit_matches_transformers() {
        if std::env::var("PLAKAT_OWLVIT_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let home = std::env::var("HOME").unwrap();
        let base = format!("{home}/.cache/huggingface/hub/models--google--owlvit-base-patch32/snapshots");
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
        let weights = snap.join("model.safetensors");
        let model = OwlViT::load(&weights, &dev).unwrap();

        let g = candle_core::safetensors::load("tools/reference/out/owlvit/goldens.safetensors", &dev).unwrap();
        let feats = model.image_feats(&g["pixel_values"]).unwrap();
        let cf = corr(&feats, &g["image_feats"]);
        eprintln!("owlvit image_feats corr = {cf:.6} shape={:?}", feats.dims());

        let q = model.query_embeds(&g["input_ids"]).unwrap();
        let cq = corr(&q, &g["query_embeds"]);
        eprintln!("owlvit query_embeds corr = {cq:.6} shape={:?}", q.dims());

        let (boxes, logits) = model.forward(&g["pixel_values"], &g["input_ids"]).unwrap();
        let cb = corr(&boxes, &g["pred_boxes"]);
        let cl = corr(&logits, &g["logits"]);
        eprintln!("owlvit boxes corr = {cb:.6}  logits corr = {cl:.6}");
        assert!(cf > 0.999, "image_feats corr {cf}");
        assert!(cq > 0.999, "query_embeds corr {cq}");
        assert!(cb > 0.999, "boxes corr {cb}");
        assert!(cl > 0.999, "logits corr {cl}");
    }
}
