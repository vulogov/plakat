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
