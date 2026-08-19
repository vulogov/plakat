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

    // ---- content focus qualifiers (RFC QUALITY-1): pre-tune the pass to a subject's AI tell. `N` is the
    // blend weight (0 = off, 1 = midpoint of the preset and that subject's de-AI profile, >1 = stronger).
    /// Focus for **people / portraits** (plastic skin) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub people: Option<f32>,
    /// Focus for **skies** (banding / too-smooth) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub sky: Option<f32>,
    /// Focus for **vegetation / foliage** (cloud-like repeating mush) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub vegetation: Option<f32>,
    /// Focus for **cityscapes** (razor-clean geometry) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub cityscape: Option<f32>,
    /// Focus for **landscape / scenery** (atmosphere) — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub landscape: Option<f32>,
    /// Focus for **seascape** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub sea: Option<f32>,
    /// Focus for **riverscape** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub river: Option<f32>,
    /// Focus for **mechanical apparatus / transports** surface — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub mechanics: Option<f32>,
    /// Focus for **household / indoor** scenes — weight N.
    #[arg(long, value_name = "N", help_heading = "Content focus")]
    pub household: Option<f32>,

    // ---- corrective focuses (model-backed: img2img / inpaint, NOT the analog pass) ----
    /// Fix **geometry** (incoherent structure / joinery) via img2img — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub geometry: Option<f32>,
    /// Fix **anatomy** (proportions / hands) via img2img — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub anatomy: Option<f32>,
    /// Make **lookalike faces distinct** (detect + inpaint each duplicate) — weight N.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub no_twins: Option<f32>,
    /// Model for the corrective img2img / inpaint passes.
    #[arg(long, default_value = "sdxl", help_heading = "Corrective (needs a model)")]
    pub model: String,
    /// Steps for the corrective passes.
    #[arg(long, default_value_t = 24, help_heading = "Corrective (needs a model)")]
    pub refine_steps: usize,
    /// Device for the corrective passes.
    #[arg(long, default_value = "auto", help_heading = "Corrective (needs a model)")]
    pub device: String,

    /// Remove a **ghost signature** smudge from a corner (`br`/`bl`/`tr`/`tl`) — a weight-free
    /// content-aware dissolve. Foreign-artifact only; never touches plakat's own etch.
    #[arg(long, value_name = "CORNER")]
    pub designature: Option<String>,
    /// Strength of the ghost-signature dissolve (0..1).
    #[arg(long, default_value_t = 0.9)]
    pub designature_strength: f32,

    /// Write a clean, un-etched output — do NOT carry plakat provenance forward.
    #[arg(long)]
    pub no_reetch: bool,
}

pub async fn run(a: NaturalizeArgs) -> Result<()> {
    let base = match a.preset.as_deref() {
        Some(s) => Preset::parse(s).with_context(|| format!("unknown preset `{s}` (subtle|photo|painting)"))?,
        None => Preset::Subtle,
    };
    // content focus: blend the preset toward each active subject's de-AI profile, THEN apply explicit
    // per-param overrides (which always win).
    let focuses: Vec<(naturalize::Focus, f32)> = [
        (naturalize::Focus::People, a.people),
        (naturalize::Focus::Sky, a.sky),
        (naturalize::Focus::Vegetation, a.vegetation),
        (naturalize::Focus::Cityscape, a.cityscape),
        (naturalize::Focus::Landscape, a.landscape),
        (naturalize::Focus::Sea, a.sea),
        (naturalize::Focus::River, a.river),
        (naturalize::Focus::Mechanics, a.mechanics),
        (naturalize::Focus::Household, a.household),
    ]
    .into_iter()
    .filter_map(|(f, n)| n.filter(|v| *v > 0.0).map(|v| (f, v)))
    .collect();
    let mut p: Params = naturalize::blend_focus(base.params(), &focuses);
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

    // Corrective refine (model-backed) runs FIRST, so the analog pass grains the fixed structure.
    let corrective = naturalize::refine::Corrective {
        geometry: a.geometry.unwrap_or(0.0),
        anatomy: a.anatomy.unwrap_or(0.0),
        no_twins: a.no_twins.unwrap_or(0.0),
    };
    let tmp = tempfile::tempdir().context("temp dir for naturalize refine")?;
    let src_for_analog = if corrective.any() {
        let refined = tmp.path().join("refined.png");
        naturalize::refine::refine(&a.input, &refined, &corrective, &a.model, Some(&a.device), a.refine_steps, tmp.path()).await?;
        refined
    } else {
        a.input.clone()
    };

    let mut img = image::open(&src_for_analog).with_context(|| format!("reading {}", src_for_analog.display()))?.to_rgb8();
    let ai_before = naturalize::ai_tell_score(&img);
    // ghost-signature removal (weight-free) before the analog pass.
    if let Some(cs) = a.designature.as_deref() {
        let corner = naturalize::Corner::parse(cs).with_context(|| format!("unknown corner `{cs}` (br|bl|tr|tl)"))?;
        img = naturalize::designature(&img, corner, a.designature_strength);
    }
    let out = naturalize::apply(&img, &p);
    let ai_after = naturalize::ai_tell_score(&out);
    out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;

    // Etch bar: carry plakat provenance forward (unless opted out).
    let mut carried = false;
    if !a.no_reetch {
        carried = carry_provenance(&a.input, &a.out).unwrap_or(false);
    }

    let preset_label = a.preset.as_deref().unwrap_or("subtle");
    let focus_note = if focuses.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = focuses.iter().map(|(f, n)| format!("{}:{n}", format!("{f:?}").to_ascii_lowercase())).collect();
        format!(" · focus {}", names.join(","))
    };
    println!("{} {}  (naturalize · {preset_label}{focus_note})", style("wrote").green(), a.out.display());
    let _ = ai_before;
    println!("  {} AI-tell {:.3} (0=human … 1=AI; a batch-ranking heuristic)", style("score").cyan(), ai_after);
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

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

/// Extract every `tEXt`/`zTXt`/`iTXt` chunk (raw: length+type+data+CRC) from PNG `bytes`.
fn extract_text_chunks(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if bytes.len() < 8 || bytes[..8] != PNG_SIG {
        return out;
    }
    let mut i = 8usize;
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let ty = &bytes[i + 4..i + 8];
        let end = i + 12 + len;
        if end > bytes.len() {
            break;
        }
        if matches!(ty, b"tEXt" | b"zTXt" | b"iTXt") {
            out.push(bytes[i..end].to_vec());
        }
        if ty == b"IEND" {
            break;
        }
        i = end;
    }
    out
}

/// Insert raw text `chunks` into the PNG at `path`, right after `IHDR` (verbatim, CRCs already valid).
fn inject_text_chunks(path: &Path, chunks: &[Vec<u8>]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 || bytes[..8] != PNG_SIG {
        return Ok(());
    }
    let ihdr_len = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let ihdr_end = 8 + 12 + ihdr_len;
    if ihdr_end > bytes.len() {
        return Ok(());
    }
    let mut out = Vec::with_capacity(bytes.len() + chunks.iter().map(|c| c.len()).sum::<usize>());
    out.extend_from_slice(&bytes[..ihdr_end]);
    for c in chunks {
        out.extend_from_slice(c);
    }
    out.extend_from_slice(&bytes[ihdr_end..]);
    std::fs::write(path, out).with_context(|| format!("re-writing {} with text chunks", path.display()))?;
    Ok(())
}

/// Copy the `tEXt`/`zTXt`/`iTXt` chunks from `src` into `dst`. Returns whether any were carried.
fn splice_png_text_chunks(src: &Path, dst: &Path) -> Result<bool> {
    let chunks = extract_text_chunks(&std::fs::read(src)?);
    if chunks.is_empty() {
        return Ok(false);
    }
    inject_text_chunks(dst, &chunks)?;
    Ok(true)
}

/// Apply the analog naturalize pass to `path` IN PLACE from a compact spec, preserving the PNG text
/// chunks (the L0 etch carrier; the JSON sidecar is a separate file and is left untouched). Used by
/// `generate --naturalize` and the scenario `naturalize:` field. Weight-free.
pub fn apply_inplace(path: &Path, spec: &str) -> Result<()> {
    let params = crate::naturalize::from_spec(spec);
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let chunks = extract_text_chunks(&bytes);
    let img = image::load_from_memory(&bytes).with_context(|| format!("decoding {}", path.display()))?.to_rgb8();
    let out = crate::naturalize::apply(&img, &params);
    out.save(path).with_context(|| format!("writing {}", path.display()))?;
    inject_text_chunks(path, &chunks)?;
    Ok(())
}
