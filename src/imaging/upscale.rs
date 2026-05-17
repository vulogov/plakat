//! Image upscaling. Two families:
//!   * Classical (Nearest/Bilinear/Bicubic/Lanczos3) — pure CPU resize via
//!     the `image` crate. Fast, predictable, no model weights.
//!   * ML (Real-ESRGAN variants) — runs an RRDBNet on the device. Recovers
//!     fine detail that classical filters can't synthesise. ~16 MB
//!     download per variant on first use.

use anyhow::{Result, anyhow};
use candle_core::Device;
use image::imageops::FilterType;
use std::path::Path;
use std::str::FromStr;

use crate::pipelines::real_esrgan::{self, Variant as EsrganVariant};

#[derive(Debug, Clone, Copy)]
pub enum Method {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
    RealEsrganX2,
    RealEsrganX4,
    RealEsrganAnimeX4,
}

impl Method {
    /// Whether this method uses a neural model (requires Device + download).
    pub fn is_ml(self) -> bool {
        matches!(
            self,
            Self::RealEsrganX2 | Self::RealEsrganX4 | Self::RealEsrganAnimeX4
        )
    }

    /// Native scale factor for ML methods (the model dictates it; --scale is
    /// ignored). `None` for classical methods, which use `--scale` directly.
    pub fn native_scale(self) -> Option<f32> {
        match self {
            Self::RealEsrganX2 => Some(2.0),
            Self::RealEsrganX4 | Self::RealEsrganAnimeX4 => Some(4.0),
            _ => None,
        }
    }
}

impl FromStr for Method {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "nearest" => Self::Nearest,
            "bilinear" | "triangle" => Self::Bilinear,
            "bicubic" | "catmull-rom" | "catmullrom" => Self::Bicubic,
            "lanczos" | "lanczos3" => Self::Lanczos3,
            "real-esrgan-x2" | "realesrgan-x2" | "real-esrgan-x2plus" => Self::RealEsrganX2,
            "real-esrgan-x4" | "realesrgan-x4" | "real-esrgan-x4plus" => Self::RealEsrganX4,
            "real-esrgan-anime-x4"
            | "real-esrgan-x4plus-anime"
            | "real-esrgan-anime"
            | "real-esrgan-x4plus-anime-6b" => Self::RealEsrganAnimeX4,
            other => return Err(anyhow!("unknown filter {other:?}")),
        })
    }
}

impl From<Method> for FilterType {
    fn from(m: Method) -> Self {
        match m {
            Method::Nearest => FilterType::Nearest,
            Method::Bilinear => FilterType::Triangle,
            Method::Bicubic => FilterType::CatmullRom,
            Method::Lanczos3 => FilterType::Lanczos3,
            _ => FilterType::Lanczos3, // unreachable in practice — caller routes ML separately
        }
    }
}

fn esrgan_variant(m: Method) -> Option<EsrganVariant> {
    match m {
        Method::RealEsrganX2 => Some(EsrganVariant::X2Plus),
        Method::RealEsrganX4 => Some(EsrganVariant::X4Plus),
        Method::RealEsrganAnimeX4 => Some(EsrganVariant::X4AnimeB6),
        _ => None,
    }
}

/// ML upscaler — downloads weights if needed, builds the model, runs inference.
/// Returns (in_w, in_h, out_w, out_h). Native scale is fixed by the model
/// (`--scale` from CLI is ignored when method is ML).
///
/// One-shot path: each call re-downloads + re-builds. For multi-call use
/// (scenarios with N tasks all upscaling), construct an `EsrganPipeline` once
/// and call its `upscale_file` repeatedly.
pub async fn ml_upscale(
    in_path: &Path,
    out_path: &Path,
    method: Method,
    device: &Device,
) -> Result<(u32, u32, u32, u32)> {
    let p = EsrganPipeline::load(method, device).await?;
    p.upscale_file(in_path, out_path)
}

/// Reusable Real-ESRGAN model handle. Build once with `load`, call
/// `upscale_file` any number of times — model stays resident on the device.
pub struct EsrganPipeline {
    model: real_esrgan::Model,
    variant: real_esrgan::Variant,
    device: Device,
}

impl EsrganPipeline {
    pub async fn load(method: Method, device: &Device) -> Result<Self> {
        let variant = esrgan_variant(method)
            .ok_or_else(|| anyhow!("EsrganPipeline::load needs an ML method, got {method:?}"))?;
        let (repo, cfg) = real_esrgan::variant_repo_and_config(variant);

        let dl = crate::ui::progress::spinner(&format!("Downloading Real-ESRGAN ({repo})"));
        let weights = crate::hf::download::get_first_of(&[
            (repo, "diffusion_pytorch_model.fp16.safetensors"),
            (repo, "diffusion_pytorch_model.safetensors"),
        ])
        .await?;
        dl.finish_with_message(format!("✓ weights ready ({repo})"));

        let build = crate::ui::progress::spinner(&format!("Loading Real-ESRGAN ({variant:?})"));
        let model = real_esrgan::Model::load(&weights, &cfg, device)?;
        build.finish_with_message(format!("✓ model loaded ({variant:?})"));

        Ok(Self {
            model,
            variant,
            device: device.clone(),
        })
    }

    #[allow(dead_code)]
    pub fn variant(&self) -> real_esrgan::Variant {
        self.variant
    }

    pub fn upscale_file(
        &self,
        in_path: &Path,
        out_path: &Path,
    ) -> Result<(u32, u32, u32, u32)> {
        self.model.upscale_file(in_path, out_path, &self.device)
    }
}

pub fn upscale(
    in_path: &Path,
    out_path: &Path,
    scale: f32,
    method: Method,
) -> Result<(u32, u32, u32, u32)> {
    if method.is_ml() {
        anyhow::bail!(
            "method {method:?} is ML; call `ml_upscale` instead (needs Device + async)"
        );
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(anyhow!("scale must be > 0 (got {scale})"));
    }
    let img = image::open(in_path)?;
    let (w, h) = (img.width(), img.height());
    let nw = ((w as f32) * scale).round().max(1.0) as u32;
    let nh = ((h as f32) * scale).round().max(1.0) as u32;
    let resized = img.resize_exact(nw, nh, method.into());
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    resized.save(out_path)?;
    Ok((w, h, nw, nh))
}
