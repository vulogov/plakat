//! `plakat naturalize` (RFC QUALITY-1) — the weight-free analog post-pass: film grain + chromatic
//! aberration + vignette + bloom + a desaturating film grade over any image, to break the digital-clean,
//! over-saturated "AI-generated" fingerprint. No GPU.
//!
//! Etch bar (RFC §Etch preservation): plakat's own provenance is carried forward — the L0 JSON sidecar +
//! the PNG text chunks are copied onto the output so `doctor --if-plakat` still finds it. (A proper
//! re-etch — re-embedding L1 into the new pixels with a `parent` chain — lands in P2; `--no-reetch` writes
//! a clean, un-etched output.)

use anyhow::{Context, Result};
use clap::Args;
use console::style;
use std::path::{Path, PathBuf};

use crate::naturalize::{self, Params, Preset};

#[derive(Args, Debug)]
pub struct NaturalizeArgs {
    /// Input image.
    pub input: PathBuf,
    /// Output image.
    #[arg(long)]
    pub out: PathBuf,
    /// Strength bundle: `subtle` (default) | `photo` | `painting`. All aim at contemporary realism (no
    /// retro/vintage look).
    #[arg(long)]
    pub preset: Option<String>,
    /// Override film-grain amount (0..1).
    #[arg(long)]
    pub grain: Option<f32>,
    /// Override chromatic-aberration amount (0..1).
    #[arg(long)]
    pub aberration: Option<f32>,
    /// Override vignette amount (0..1).
    #[arg(long)]
    pub vignette: Option<f32>,
    /// Override highlight bloom (0..1).
    #[arg(long)]
    pub bloom: Option<f32>,
    /// Override desaturation toward luminance (0..1).
    #[arg(long)]
    pub desaturate: Option<f32>,
    /// Override warm film lift in the shadows (0..1).
    #[arg(long)]
    pub warm: Option<f32>,
    /// Override radial defocus (0..1).
    #[arg(long)]
    pub defocus: Option<f32>,
    /// Write a clean, un-etched output — do NOT carry plakat provenance forward.
    #[arg(long)]
    pub no_reetch: bool,
}

pub async fn run(a: NaturalizeArgs) -> Result<()> {
    let base = match a.preset.as_deref() {
        Some(s) => Preset::parse(s).with_context(|| format!("unknown preset `{s}` (subtle|photo|painting)"))?,
        None => Preset::Subtle,
    };
    let mut p: Params = base.params();
    if let Some(v) = a.grain {
        p.grain = v;
    }
    if let Some(v) = a.aberration {
        p.aberration = v;
    }
    if let Some(v) = a.vignette {
        p.vignette = v;
    }
    if let Some(v) = a.bloom {
        p.bloom = v;
    }
    if let Some(v) = a.desaturate {
        p.desaturate = v;
    }
    if let Some(v) = a.warm {
        p.warm = v;
    }
    if let Some(v) = a.defocus {
        p.defocus = v;
    }

    let img = image::open(&a.input).with_context(|| format!("reading {}", a.input.display()))?.to_rgb8();
    let out = naturalize::apply(&img, &p);
    out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;

    // Etch bar: carry plakat provenance forward (unless opted out).
    let mut carried = false;
    if !a.no_reetch {
        carried = carry_provenance(&a.input, &a.out).unwrap_or(false);
    }

    let preset_label = a.preset.as_deref().unwrap_or("subtle");
    println!("{} {}  (naturalize · {preset_label})", style("wrote").green(), a.out.display());
    if carried {
        println!("  {} plakat provenance carried forward (L0). Note: pixels changed — a full re-etch (L1) lands in P2; `--no-reetch` for a clean output.", style("etch").cyan());
    }
    Ok(())
}

/// Carry plakat's L0 provenance from `src` to `dst`: copy the JSON sidecar and splice the PNG text chunks
/// (`parameters` / `etch`). Returns true if anything was carried. Best-effort.
fn carry_provenance(src: &Path, dst: &Path) -> Result<bool> {
    let mut any = false;
    // 1. JSON sidecar (the carrier `doctor --if-plakat` reads).
    let src_side = crate::imaging::io::sidecar_path(src);
    if src_side.exists() {
        let dst_side = crate::imaging::io::sidecar_path(dst);
        if src_side != dst_side {
            std::fs::copy(&src_side, &dst_side).with_context(|| format!("copying sidecar to {}", dst_side.display()))?;
        }
        any = true;
    }
    // 2. PNG text chunks (verbatim splice — CRCs stay valid). Only for PNG→PNG.
    if src.extension().and_then(|s| s.to_str()) == Some("png") && dst.extension().and_then(|s| s.to_str()) == Some("png") {
        if splice_png_text_chunks(src, dst).unwrap_or(false) {
            any = true;
        }
    }
    Ok(any)
}

/// Copy every `tEXt`/`zTXt`/`iTXt` chunk from `src` into `dst`, inserted right after `IHDR`. Verbatim
/// (length+type+data+CRC), so no re-CRC is needed. Best-effort; returns whether any chunk was carried.
fn splice_png_text_chunks(src: &Path, dst: &Path) -> Result<bool> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    let src_bytes = std::fs::read(src)?;
    let dst_bytes = std::fs::read(dst)?;
    if src_bytes.len() < 8 || src_bytes[..8] != SIG || dst_bytes.len() < 8 || dst_bytes[..8] != SIG {
        return Ok(false);
    }
    // collect text chunks (raw, including length+type+data+crc) from src.
    let mut text: Vec<&[u8]> = Vec::new();
    let mut i = 8usize;
    while i + 8 <= src_bytes.len() {
        let len = u32::from_be_bytes([src_bytes[i], src_bytes[i + 1], src_bytes[i + 2], src_bytes[i + 3]]) as usize;
        let ty = &src_bytes[i + 4..i + 8];
        let end = i + 12 + len; // 4 len + 4 type + len data + 4 crc
        if end > src_bytes.len() {
            break;
        }
        if matches!(ty, b"tEXt" | b"zTXt" | b"iTXt") {
            text.push(&src_bytes[i..end]);
        }
        if ty == b"IEND" {
            break;
        }
        i = end;
    }
    if text.is_empty() {
        return Ok(false);
    }
    // find the end of IHDR in dst (first chunk after the signature).
    let ihdr_len = u32::from_be_bytes([dst_bytes[8], dst_bytes[9], dst_bytes[10], dst_bytes[11]]) as usize;
    let ihdr_end = 8 + 12 + ihdr_len;
    if ihdr_end > dst_bytes.len() {
        return Ok(false);
    }
    let mut out = Vec::with_capacity(dst_bytes.len() + text.iter().map(|c| c.len()).sum::<usize>());
    out.extend_from_slice(&dst_bytes[..ihdr_end]);
    for c in &text {
        out.extend_from_slice(c);
    }
    out.extend_from_slice(&dst_bytes[ihdr_end..]);
    std::fs::write(dst, out).with_context(|| format!("re-writing {} with carried text chunks", dst.display()))?;
    Ok(true)
}
