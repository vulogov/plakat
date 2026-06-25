use anyhow::{Result, anyhow};
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub struct Size {
    pub w: u32,
    pub h: u32,
}

impl FromStr for Size {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<_> = s.split(|c| c == 'x' || c == 'X').collect();
        if parts.len() != 2 {
            return Err(anyhow!("expected WxH, got {s:?}"));
        }
        Ok(Size {
            w: parts[0].parse()?,
            h: parts[1].parse()?,
        })
    }
}

/// Diffusion U-Nets need spatial dims divisible by 8.
const ROUND: u32 = 8;

pub fn resolve(size: Option<Size>, aspect: Option<&str>, base: u32) -> Result<(u32, u32)> {
    if let Some(s) = size {
        return Ok((round_to(s.w, ROUND), round_to(s.h, ROUND)));
    }
    if let Some(a) = aspect {
        let (an, ad) = parse_aspect(a)?;
        let (w, h) = if an >= ad {
            (
                ((base as f32) * (an as f32 / ad as f32)).round() as u32,
                base,
            )
        } else {
            (
                base,
                ((base as f32) * (ad as f32 / an as f32)).round() as u32,
            )
        };
        return Ok((round_to(w, ROUND), round_to(h, ROUND)));
    }
    Ok((round_to(base, ROUND), round_to(base, ROUND)))
}

fn parse_aspect(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<_> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(anyhow!("aspect must be N:M, got {s:?}"));
    }
    let n: u32 = parts[0].parse()?;
    let m: u32 = parts[1].parse()?;
    if n == 0 || m == 0 {
        return Err(anyhow!("aspect components must be non-zero"));
    }
    Ok((n, m))
}

fn round_to(n: u32, m: u32) -> u32 {
    let r = (n / m) * m;
    if r == 0 { m } else { r }
}

/// Warn (once, to the user) when a requested render size is large enough to risk
/// Metal's single-buffer allocation limit on a 24 GB unified-memory box — the
/// most common avoidable OOM. Non-destructive: it never changes the size, only
/// suggests a smaller `--size` if the run aborts. No-op off Metal.
pub fn warn_large_for_metal(width: u32, height: u32, device: &candle_core::Device) {
    const METAL_SOFT_PIXELS: u64 = 1024 * 1024; // ~1MP; SDXL 1024² is borderline-OK
    if device.is_metal() && (width as u64) * (height as u64) > METAL_SOFT_PIXELS {
        crate::ui::progress::println(&format!(
            "  {} {width}×{height} is large for Metal — if it OOMs (Metal single-buffer \
             limit), retry with a smaller --size (e.g. 768×768 / 1024×768).",
            console::style("!").yellow().bold()
        ));
    }
}
