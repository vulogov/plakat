//! Classical image upscaling via the `image` crate. No AI model — produces
//! crisp predictable output without extra weights. For ML upscaling
//! (Real-ESRGAN / SwinIR) we'd need to port that model separately.

use anyhow::{Result, anyhow};
use image::imageops::FilterType;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum Method {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl FromStr for Method {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "nearest" => Self::Nearest,
            "bilinear" | "triangle" => Self::Bilinear,
            "bicubic" | "catmull-rom" | "catmullrom" => Self::Bicubic,
            "lanczos" | "lanczos3" => Self::Lanczos3,
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
        }
    }
}

pub fn upscale(
    in_path: &Path,
    out_path: &Path,
    scale: f32,
    method: Method,
) -> Result<(u32, u32, u32, u32)> {
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
