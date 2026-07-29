//! PIPNet-98 (WFLW) facial-landmark aligner in candle (ROADMAP_5.0.0, topology v1 = WFLW-98).
//!
//! A ResNet-18 backbone + a pixel-in-pixel head: a 256×256 face crop → 98 landmarks. Powers the
//! geometry engine (Layer 2) and every `landmark` / `local_anomaly` / `region_*` scorecard probe.
//! Chosen over the canonical InsightFace 2d106det because that model's weights are non-commercial;
//! PIPNet (MIT) is license-clean (see `Documentation/PERSONA_GATING.md`).
//!
//! Weights are the converted `pipnet_r18_wflw_98.safetensors` (`tools/reference/pipnet_dump.py` renames
//! the ONNX initializers to this ResNet-18 tree, BN folded into each conv's weight+bias).
//! Verified vs an onnxruntime dump (`PLAKAT_PIPNET_VERIFY`).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};

const INPUT: usize = 256;
const GRID: usize = 8; // 256 / 32 stride
pub const NUM_LANDMARKS: usize = 98;
const NUM_NB: usize = 10; // neighbours per landmark (nb head = 98*10)

/// ImageNet normalisation (PIPNet-WFLW preprocessing).
#[allow(clippy::excessive_precision)]
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
#[allow(clippy::excessive_precision)]
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn cfg(padding: usize, stride: usize) -> Conv2dConfig {
    Conv2dConfig { padding, stride, dilation: 1, groups: 1, cudnn_fwd_algo: None }
}

/// A ResNet-18 basic block (folded-BN convs, so each conv carries a bias).
struct BasicBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    downsample: Option<Conv2d>,
}

impl BasicBlock {
    fn load(in_c: usize, out_c: usize, stride: usize, downsample: bool, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d(in_c, out_c, 3, cfg(1, stride), vb.pp("conv1"))?;
        let conv2 = conv2d(out_c, out_c, 3, cfg(1, 1), vb.pp("conv2"))?;
        let downsample = if downsample {
            Some(conv2d(in_c, out_c, 1, cfg(0, stride), vb.pp("downsample"))?)
        } else {
            None
        };
        Ok(Self { conv1, conv2, downsample })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = match &self.downsample {
            Some(d) => d.forward(x)?,
            None => x.clone(),
        };
        let out = self.conv1.forward(x)?.relu()?;
        let out = self.conv2.forward(&out)?;
        Ok((out + identity)?.relu()?)
    }
}

/// The raw network outputs (before landmark decode), each `(1, C, 8, 8)`.
pub struct Heads {
    pub cls: Tensor,      // (1, 98, 8, 8) — per-landmark grid-cell logits
    pub offset_x: Tensor, // (1, 98, 8, 8)
    pub offset_y: Tensor, // (1, 98, 8, 8)
    pub nb_x: Tensor,     // (1, 980, 8, 8)
    pub nb_y: Tensor,     // (1, 980, 8, 8)
}

pub struct PipNet {
    conv1: Conv2d,
    layer1: [BasicBlock; 2],
    layer2: [BasicBlock; 2],
    layer3: [BasicBlock; 2],
    layer4: [BasicBlock; 2],
    cls_layer: Conv2d,
    x_layer: Conv2d,
    y_layer: Conv2d,
    nb_x_layer: Conv2d,
    nb_y_layer: Conv2d,
    device: Device,
}

impl PipNet {
    /// The hosted converted weights (see `Documentation/PERSONA_GATING.md`).
    pub const REPO: &'static str = "vulogov98/plakat-persona";
    pub const WEIGHTS: &'static str = "pipnet_r18_wflw_98.safetensors";

    /// Download + load the hosted PIPNet-98 weights (~49 MB, once).
    pub async fn load_pretrained(device: &Device) -> Result<Self> {
        let weights = crate::hf::download::get_file(Self::REPO, Self::WEIGHTS)
            .await
            .context("downloading PIPNet-98 aligner weights")?;
        Self::load(&weights, device)
    }

    pub fn load(weights: &std::path::Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let conv1 = conv2d(3, 64, 7, cfg(3, 2), vb.pp("conv1")).context("PIPNet conv1")?;
        let l = |i: usize, o: usize, s: usize, ds: bool, name: &str, blk: usize| {
            BasicBlock::load(i, o, s, ds, vb.pp(format!("{name}.{blk}")))
        };
        let layer1 = [l(64, 64, 1, false, "layer1", 0)?, l(64, 64, 1, false, "layer1", 1)?];
        let layer2 = [l(64, 128, 2, true, "layer2", 0)?, l(128, 128, 1, false, "layer2", 1)?];
        let layer3 = [l(128, 256, 2, true, "layer3", 0)?, l(256, 256, 1, false, "layer3", 1)?];
        let layer4 = [l(256, 512, 2, true, "layer4", 0)?, l(512, 512, 1, false, "layer4", 1)?];
        let head = |out: usize, name: &str| conv2d(512, out, 1, cfg(0, 1), vb.pp(name));
        Ok(Self {
            conv1,
            layer1,
            layer2,
            layer3,
            layer4,
            cls_layer: head(NUM_LANDMARKS, "cls_layer")?,
            x_layer: head(NUM_LANDMARKS, "x_layer")?,
            y_layer: head(NUM_LANDMARKS, "y_layer")?,
            nb_x_layer: head(NUM_LANDMARKS * NUM_NB, "nb_x_layer")?,
            nb_y_layer: head(NUM_LANDMARKS * NUM_NB, "nb_y_layer")?,
            device: device.clone(),
        })
    }

    /// Forward a normalised `(1,3,256,256)` input → the raw heads.
    pub fn forward(&self, x: &Tensor) -> Result<Heads> {
        // stem: conv1 (7×7 s2) → relu → maxpool (3×3 s2, pad 1). Post-relu ≥0, so 0-pad == correct.
        let mut h = self.conv1.forward(x)?.relu()?;
        h = h.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
        h = h.max_pool2d_with_stride(3, 2)?;
        for b in &self.layer1 {
            h = b.forward(&h)?;
        }
        for b in &self.layer2 {
            h = b.forward(&h)?;
        }
        for b in &self.layer3 {
            h = b.forward(&h)?;
        }
        for b in &self.layer4 {
            h = b.forward(&h)?; // (1,512,8,8)
        }
        Ok(Heads {
            cls: self.cls_layer.forward(&h)?,
            offset_x: self.x_layer.forward(&h)?,
            offset_y: self.y_layer.forward(&h)?,
            nb_x: self.nb_x_layer.forward(&h)?,
            nb_y: self.nb_y_layer.forward(&h)?,
        })
    }

    /// Decode the heads into 98 landmarks in `[0,1]` face-crop coordinates (PIP: argmax the grid cell
    /// per landmark, add the within-cell offset, normalise by the grid). Neighbour refinement (nb_*) is
    /// omitted — the argmax+offset decode is the standard inference path and is enough for anchoring.
    pub fn decode(heads: &Heads) -> Result<Vec<(f32, f32)>> {
        let cls: Vec<f32> = heads.cls.flatten_all()?.to_vec1()?; // 98*8*8
        let ox: Vec<f32> = heads.offset_x.flatten_all()?.to_vec1()?;
        let oy: Vec<f32> = heads.offset_y.flatten_all()?.to_vec1()?;
        let cells = GRID * GRID;
        let mut pts = Vec::with_capacity(NUM_LANDMARKS);
        for k in 0..NUM_LANDMARKS {
            let base = k * cells;
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for c in 0..cells {
                if cls[base + c] > bestv {
                    bestv = cls[base + c];
                    best = c;
                }
            }
            let (gy, gx) = (best / GRID, best % GRID);
            let x = (gx as f32 + ox[base + best]) / GRID as f32;
            let y = (gy as f32 + oy[base + best]) / GRID as f32;
            pts.push((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)));
        }
        Ok(pts)
    }

    /// Preprocess an already-cropped face image → normalised `(1,3,256,256)`.
    pub fn preprocess(&self, img: &image::RgbImage) -> Result<Tensor> {
        let r = image::imageops::resize(img, INPUT as u32, INPUT as u32, image::imageops::FilterType::Triangle);
        let mut data = vec![0f32; 3 * INPUT * INPUT];
        for (x, y, p) in r.enumerate_pixels() {
            for c in 0..3 {
                let v = p.0[c] as f32 / 255.0;
                data[c * INPUT * INPUT + y as usize * INPUT + x as usize] = (v - MEAN[c]) / STD[c];
            }
        }
        Ok(Tensor::from_vec(data, (1, 3, INPUT, INPUT), &self.device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let (mut num, mut da, mut db) = (0.0f32, 0.0, 0.0);
        for (x, y) in a.iter().zip(&b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da.sqrt() * db.sqrt() + 1e-12)
    }

    /// Verify the PIPNet-98 forward against an onnxruntime dump. Opt-in (`PLAKAT_PIPNET_VERIFY=1`);
    /// needs `tools/reference/out/pipnet-wflw98/{pipnet_r18_wflw_98,goldens}.safetensors`.
    #[test]
    fn pipnet_matches_onnxruntime() {
        if std::env::var("PLAKAT_PIPNET_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let dir = std::path::Path::new("tools/reference/out/pipnet-wflw98");
        let net = PipNet::load(&dir.join("pipnet_r18_wflw_98.safetensors"), &dev).unwrap();
        let g = candle_core::safetensors::load(dir.join("goldens.safetensors"), &dev).unwrap();
        let heads = net.forward(&g["input"]).unwrap();
        for (name, got, want) in [
            ("cls_map", &heads.cls, &g["cls_map"]),
            ("offset_x", &heads.offset_x, &g["offset_x"]),
            ("offset_y", &heads.offset_y, &g["offset_y"]),
            ("nb_x", &heads.nb_x, &g["nb_x"]),
            ("nb_y", &heads.nb_y, &g["nb_y"]),
        ] {
            let c = corr(got, want);
            eprintln!("pipnet {name}: corr={c:.6} shape={:?}", got.dims());
            assert!(c > 0.999, "{name} corr {c} < 0.999");
        }
        // decode sanity: 98 points, all in [0,1].
        let pts = PipNet::decode(&heads).unwrap();
        assert_eq!(pts.len(), NUM_LANDMARKS);
        assert!(pts.iter().all(|&(x, y)| (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)));
    }
}
