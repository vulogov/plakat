//! Face-swap engine — the identity mechanism behind `multiperson`.
//!
//! Pipeline per target face: SCRFD 5-point landmarks → `norm_crop` to the
//! ArcFace template (112 for the source embedding, 128 for the swap target) →
//! ArcFace identity of the **source** → `inswapper_128` → inverse-warp + feather
//! the swapped crop back into the scene. All three models are numerically
//! verified against InsightFace (see `pipelines::{scrfd,inswapper,face_models}`).
//!
//! This is not a CLI command of its own — it's a reusable engine. `multiperson`
//! generates a coherent scene, then swaps each persona's face with their source.

#![allow(dead_code)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use image::RgbImage;
use std::path::Path;

use crate::pipelines::face_models::{self, IResnet50};
use crate::pipelines::inswapper::Inswapper;
use crate::pipelines::scrfd::{Face, SCRFDConfig, SCRFDDetector};

/// Edge feather (in 128² crop pixels) for blending the swapped face back.
const FEATHER: f32 = 16.0;

/// Default plakat-hosted converted weights (`plakat convert-onnx` of InsightFace
/// `w600k_r50.onnx` / `inswapper_128.onnx`). Override with
/// `PLAKAT_ARCFACE_WEIGHTS`/`_HF` and `PLAKAT_INSWAPPER_WEIGHTS`/`_HF`.
pub const DEFAULT_ARCFACE_REPO: &str = "vulogov98/plakat-arcface-w600k";
pub const DEFAULT_ARCFACE_FILE: &str = "arcface_w600k.safetensors";
pub const DEFAULT_INSWAPPER_REPO: &str = "vulogov98/plakat-inswapper-128";
pub const DEFAULT_INSWAPPER_FILE: &str = "inswapper_128.safetensors";

/// Resolve a weight file from `PLAKAT_<KEY>_WEIGHTS` (local) /
/// `PLAKAT_<KEY>_HF` (`repo#file`) / a bundled default repo.
async fn resolve_weight(
    key: &str,
    default_repo: &str,
    default_file: &str,
) -> Result<std::path::PathBuf> {
    if let Ok(p) = std::env::var(format!("PLAKAT_{key}_WEIGHTS")) {
        let path = std::path::PathBuf::from(&p);
        anyhow::ensure!(path.exists(), "PLAKAT_{key}_WEIGHTS {p} does not exist");
        return Ok(path);
    }
    let (repo, file) = if let Ok(spec) = std::env::var(format!("PLAKAT_{key}_HF")) {
        crate::pipelines::ip_adapter::parse_hf_spec(&spec, &format!("PLAKAT_{key}_HF"))?
    } else {
        (default_repo.to_string(), default_file.to_string())
    };
    let s = crate::ui::progress::spinner(&format!("Downloading {key} weights ({repo}/{file})"));
    let path = crate::hf::download::get_file(&repo, &file)
        .await
        .with_context(|| format!("downloading {key} weights from {repo}/{file}"))?;
    s.finish_with_message(format!("✓ {key} weights cached"));
    Ok(path)
}

pub struct FaceSwapper {
    detector: SCRFDDetector,
    arcface: IResnet50,
    emap: Tensor, // (512, 512)
    inswapper: Inswapper,
    device: Device,
    dtype: DType,
}

impl FaceSwapper {
    /// Load from explicit converted-safetensors paths (SCRFD, ArcFace, inswapper).
    /// The inswapper file also carries `emap` (emitted by `convert-onnx`).
    pub fn load_from(
        scrfd: &Path,
        arcface: &Path,
        inswapper: &Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut detector = SCRFDDetector::load(scrfd, SCRFDConfig::default(), device, DType::F32)
            .context("loading SCRFD for face-swap")?;
        // Lower the score threshold: generated scene faces are often small and
        // painterly (watercolor) and score below the 0.5 default; region-matching
        // tolerates the occasional spurious detection, but a missed face means no
        // swap at all. 0.35 catches them.
        detector.score_threshold = 0.35;
        // ArcFace + inswapper run in F32 for numerical fidelity (they're small).
        let arc_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[arcface], DType::F32, device)?
        };
        let arcface = IResnet50::new(arc_vb).context("loading ArcFace for face-swap")?;
        let insw_tensors = candle_core::safetensors::load(inswapper, device)?;
        let emap = insw_tensors
            .get("emap")
            .context("inswapper weights missing `emap` — reconvert with a current convert-onnx")?
            .to_dtype(DType::F32)?;
        let inswapper = Inswapper::load(inswapper, device, DType::F32)
            .context("loading inswapper for face-swap")?;
        Ok(Self { detector, arcface, emap, inswapper, device: device.clone(), dtype })
    }

    /// Load with weights auto-resolved: SCRFD (its own default), ArcFace, and
    /// inswapper (env overrides or the bundled default repos).
    pub async fn load_resolved(device: &Device, dtype: DType) -> Result<Self> {
        let scrfd = crate::pipelines::scrfd::resolve_scrfd_weights()
            .await?
            .context("face-swap needs SCRFD weights (none resolved)")?;
        let arcface = resolve_weight("ARCFACE", DEFAULT_ARCFACE_REPO, DEFAULT_ARCFACE_FILE).await?;
        let inswapper =
            resolve_weight("INSWAPPER", DEFAULT_INSWAPPER_REPO, DEFAULT_INSWAPPER_FILE).await?;
        Self::load_from(&scrfd, &arcface, &inswapper, device, dtype)
    }

    /// Detect faces in an image file (largest first), with 5-point landmarks.
    pub fn detect(&self, path: &Path) -> Result<Vec<Face>> {
        let mut faces = self.detector.detect(path)?;
        faces.sort_by(|a, b| {
            let area = |f: &Face| (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
            area(b).partial_cmp(&area(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(faces)
    }

    /// Compute the source identity latent (`normalize(arcface_emb @ emap)`) — the
    /// `source` input to the swapper — from a source face photo (largest face).
    pub fn source_latent(&self, source_face: &Path) -> Result<Tensor> {
        let orig = image::open(source_face)
            .with_context(|| format!("opening source face {}", source_face.display()))?
            .to_rgb8();

        // SCRFD is tuned for faces that are a fraction of the frame; a tightly
        // cropped portrait (face filling >50% of the image) scores below
        // threshold or is missed. If the raw photo yields no face, pad it with a
        // white margin so the face becomes a detectable size, then detect there.
        let (img, face) = match self.detect(source_face)?.into_iter().next() {
            Some(f) => (orig, f),
            None => {
                let padded = pad_white(&orig, 0.6);
                let tmp = tempfile::Builder::new()
                    .prefix("plakat-src-pad-")
                    .suffix(".png")
                    .tempfile()?;
                padded.save(tmp.path())?;
                let f = self
                    .detect(tmp.path())?
                    .into_iter()
                    .next()
                    .with_context(|| {
                        format!(
                            "no face detected in {} (even after padding — try a less \
                             tightly-cropped photo with headroom around the face)",
                            source_face.display()
                        )
                    })?;
                (padded, f)
            }
        };
        let (aligned, _) = face_models::norm_crop(&img, face.landmarks, 112);
        let t = img_to_tensor(&aligned, 127.5, 127.5, &self.device)?;
        let emb = self.arcface.forward(&t)?; // (1,512) unit-norm
        let latent = emb.matmul(&self.emap)?; // (1,512)
        l2_normalize(&latent)
    }

    /// Swap the face at `target_landmarks` (pixel coords in `scene`) with the
    /// source identity `latent`, returning the modified scene.
    pub fn swap_into(
        &self,
        scene: &RgbImage,
        target_landmarks: [[f32; 2]; 5],
        latent: &Tensor,
    ) -> Result<RgbImage> {
        let (result, _, _) = self.swap_into_debug(scene, target_landmarks, latent)?;
        Ok(result)
    }

    /// Like `swap_into` but also returns the 128² aligned **target** crop and the
    /// 128² **swapped** crop — for diagnosing whether the swap transferred at the
    /// crop level (vs. a paste-back / scale issue).
    pub fn swap_into_debug(
        &self,
        scene: &RgbImage,
        target_landmarks: [[f32; 2]; 5],
        latent: &Tensor,
    ) -> Result<(RgbImage, RgbImage, RgbImage)> {
        let (target128, forward) = face_models::norm_crop(scene, target_landmarks, 128);
        let t = img_to_tensor(&target128, 0.0, 255.0, &self.device)?; // inswapper: /255, no mean
        let swapped = self.inswapper.forward(&t, latent)?; // (1,3,128,128) in [0,1]
        let swapped_img = tensor_to_img(&swapped)?;
        let result = paste_back(scene, &swapped_img, forward);
        Ok((result, target128, swapped_img))
    }
}

/// `RgbImage` → `(1, 3, H, W)` f32 with `(v - mean) / std`, RGB order.
fn img_to_tensor(img: &RgbImage, mean: f32, std: f32, device: &Device) -> Result<Tensor> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                data[c * h * w + y * w + x] = (p[c] as f32 - mean) / std;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, h, w), device)?)
}

/// `(1, 3, H, W)` f32 in `[0,1]` → `RgbImage`.
fn tensor_to_img(t: &Tensor) -> Result<RgbImage> {
    let (_b, _c, h, w) = t.dims4()?;
    let t = t.i(0)?.clamp(0.0, 1.0)?; // (3,H,W)
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut img = RgbImage::new(w as u32, h as u32);
    let plane = h * w;
    for y in 0..h {
        for x in 0..w {
            let px = [
                (v[y * w + x] * 255.0).round() as u8,
                (v[plane + y * w + x] * 255.0).round() as u8,
                (v[2 * plane + y * w + x] * 255.0).round() as u8,
            ];
            img.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    Ok(img)
}

/// Composite the 128² swapped crop back into the scene. `forward` maps scene
/// pixels → crop coords; a feathered mask (distance to the crop border) hides the
/// seam. Matches InsightFace's eroded+blurred paste-back, simplified.
fn paste_back(scene: &RgbImage, swapped: &RgbImage, forward: [f32; 6]) -> RgbImage {
    let (w, h) = (scene.width(), scene.height());
    let (cw, ch) = (swapped.width() as f32, swapped.height() as f32);
    let mut out = scene.clone();
    for dy in 0..h {
        for dx in 0..w {
            let cx = forward[0] * dx as f32 + forward[1] * dy as f32 + forward[2];
            let cy = forward[3] * dx as f32 + forward[4] * dy as f32 + forward[5];
            if cx < 0.0 || cy < 0.0 || cx >= cw || cy >= ch {
                continue;
            }
            let border = cx.min(cy).min(cw - 1.0 - cx).min(ch - 1.0 - cy);
            let alpha = (border / FEATHER).clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }
            let s = sample_bilinear(swapped, cx, cy);
            let base = scene.get_pixel(dx, dy).0;
            let mut px = [0u8; 3];
            for c in 0..3 {
                px[c] = (s[c] as f32 * alpha + base[c] as f32 * (1.0 - alpha))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(dx, dy, image::Rgb(px));
        }
    }
    out
}

fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let x0 = x.floor().clamp(0.0, w - 1.0);
    let y0 = y.floor().clamp(0.0, h - 1.0);
    let x1 = (x0 + 1.0).min(w - 1.0);
    let y1 = (y0 + 1.0).min(h - 1.0);
    let ax = (x - x0).clamp(0.0, 1.0);
    let ay = (y - y0).clamp(0.0, 1.0);
    let p = |xx: f32, yy: f32| img.get_pixel(xx as u32, yy as u32).0;
    let (p00, p10, p01, p11) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] as f32 * (1.0 - ax) + p10[c] as f32 * ax;
        let bot = p01[c] as f32 * (1.0 - ax) + p11[c] as f32 * ax;
        out[c] = (top * (1.0 - ay) + bot * ay).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Pad an image with a white border of `frac × max(w,h)` on each side — gives a
/// tightly-cropped face room so SCRFD (trained on smaller faces) can detect it.
fn pad_white(img: &RgbImage, frac: f32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let m = ((w.max(h) as f32) * frac).round() as u32;
    let mut out = RgbImage::from_pixel(w + 2 * m, h + 2 * m, image::Rgb([255, 255, 255]));
    image::imageops::overlay(&mut out, img, m as i64, m as i64);
    out
}

fn l2_normalize(t: &Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(t.broadcast_div(&(norm + 1e-12)?)?)
}
