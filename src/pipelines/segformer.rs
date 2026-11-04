//! Semantic segmentation via SegFormer-B0 fine-tuned on ADE20k (150 classes) — the CLASS MAP of the watercolour
//! SIGHT (7.17.0 set 3, Cargo feature `aqua-sight`; `paint::sight` folds the 150 labels into the painter's eight).
//!
//! The smallest SegFormer (3.7 M parameters, ~15 MB of safetensors, the official `nvidia/segformer-b0-finetuned-
//! ade-512-512` checkpoint — no community conversion needed, the repo ships `model.safetensors` and `config.json`).
//! Pure candle: `candle_transformers::models::segformer`.
//!
//! Inference flow:
//!
//! 1. Resize the picture to 512 × 512, normalise with the ImageNet stats.
//! 2. Forward → logits `(1, 150, 128, 128)` (the decode head works at a quarter of the input).
//! 3. Softmax over the classes — the caller gets the PROBABILITIES on the model's grid, not an argmax: the painter's
//!    classes are sums of ADE classes, and a sum of probabilities is a confidence an argmax cannot give.
//!
//! The labels come from the checkpoint's own `config.json` (`id2label`), so a differently fine-tuned SegFormer
//! works unchanged as long as its labels are names the painter's table knows.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segformer::{Config, SemanticSegmentationModel};
use image::RgbImage;

/// The checkpoint: the official SegFormer-B0 fine-tuned on ADE20k at 512².
pub const REPO: &str = "nvidia/segformer-b0-finetuned-ade-512-512";
/// Model input resolution (square).
const INPUT_SIZE: u32 = 512;

/// Lazy-loaded SegFormer pipeline. Reuse one instance across many images — the load is the dominant cost.
pub struct Segmenter {
    model: SemanticSegmentationModel,
    labels: Vec<String>,
    device: Device,
}

/// The class probabilities on the model's grid: `probs[c * gh * gw + y * gw + x]`, a softmax over `c` at every cell.
#[derive(Debug, Clone)]
pub struct Segmentation {
    pub probs: Vec<f32>,
    pub classes: usize,
    pub gw: usize,
    pub gh: usize,
}

impl Segmenter {
    /// Download (or hit the cache for) the checkpoint's config and safetensors and instantiate the model on `device`.
    pub async fn load(device: &Device) -> Result<Self> {
        let cfg_path = crate::hf::download::get_file(REPO, "config.json")
            .await
            .context("downloading SegFormer-B0 ADE20k config.json")?;
        let weights = crate::hf::download::get_file(REPO, "model.safetensors")
            .await
            .context("downloading SegFormer-B0 ADE20k model.safetensors")?;
        let cfg_text = std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?;
        let cfg: Config = serde_json::from_str(&cfg_text).context("parsing the SegFormer config")?;
        let mut labels: Vec<(usize, String)> = cfg
            .id2label
            .iter()
            .filter_map(|(k, v)| k.trim().parse::<usize>().ok().map(|i| (i, v.clone())))
            .collect();
        labels.sort_by_key(|(i, _)| *i);
        anyhow::ensure!(
            !labels.is_empty() && labels.iter().enumerate().all(|(i, (j, _))| i == *j),
            "the SegFormer config's id2label is not a dense 0..n"
        );
        let n = labels.len();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)? };
        let model = SemanticSegmentationModel::new(&cfg, n, vb).context("building SegFormer")?;
        Ok(Self { model, labels: labels.into_iter().map(|(_, l)| l).collect(), device: device.clone() })
    }

    /// The checkpoint's class names, in class order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Segment the picture: the class probabilities on the model's grid (a quarter of 512²).
    pub fn segment(&self, img: &RgbImage) -> Result<Segmentation> {
        let x = preprocess(img, INPUT_SIZE, &self.device)?;
        let logits = self.model.forward(&x).context("SegFormer forward")?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let (_, classes, gh, gw) = probs.dims4()?;
        let probs: Vec<f32> = probs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(Segmentation { probs, classes, gw, gh })
    }
}

/// Resize + ImageNet-normalise one picture to `(1, 3, size, size)`.
fn preprocess(img: &RgbImage, size: u32, device: &Device) -> Result<Tensor> {
    let resized = image::imageops::resize(img, size, size, image::imageops::FilterType::Triangle);
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let n = (size as usize) * (size as usize);
    let mut buf: Vec<f32> = vec![0.0; 3 * n];
    let raw = resized.as_raw();
    let (r_dst, rest) = buf.split_at_mut(n);
    let (g_dst, b_dst) = rest.split_at_mut(n);
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        r_dst[i] = (chunk[0] as f32 / 255.0 - mean[0]) / std[0];
        g_dst[i] = (chunk[1] as f32 / 255.0 - mean[1]) / std[1];
        b_dst[i] = (chunk[2] as f32 / 255.0 - mean[2]) / std[2];
    }
    Ok(Tensor::from_vec(buf, (1, 3, size as usize, size as usize), device)?)
}
