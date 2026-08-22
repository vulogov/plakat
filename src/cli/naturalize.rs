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
    /// Override the **quality-improvement** ("de-slop") strength (0..1) — gray-world white balance +
    /// robust auto-levels + vibrance + unsharp, run FIRST to make the colours & detail genuinely better
    /// before any analog look. `0` disables it. Defaults come from the preset (subtle 0.55 … photo 0.70).
    #[arg(long)]
    pub polish: Option<f32>,
    /// Override the **micro-texture** strength (0..1+) — fine pore / micro-wrinkle detail added only to the
    /// unnaturally-smooth regions (variance-gated, mid-tones), the fix for plastic AI skin. High on
    /// `--people`; set explicitly for other smooth surfaces.
    #[arg(long)]
    pub micro: Option<f32>,

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
    /// **Face-protected repair** (0..1) — the art-safe structural fix. Detects faces and PROTECTS them
    /// (never regenerated → soft artistic faces survive, no uncanny valley), then gently repaints the rest
    /// IN-STYLE to attempt broken hands/feet/limbs. Preserves character where whole-image `--geometry`
    /// would wreck it. Pair with `--style`/`--medium` to hold the medium. Needs a model + faces.
    #[arg(long, value_name = "N", help_heading = "Corrective (needs a model)")]
    pub repair: Option<f32>,
    /// What `--repair` may touch: `figures` (default — only the figures' bodies, faces AND background
    /// preserved) · `non-face` (all non-face pixels, background regenerates) · `full` (whole image).
    #[arg(long = "repair-scope", value_name = "SCOPE", default_value = "figures", help_heading = "Corrective (needs a model)")]
    pub repair_scope: String,
    /// Art **style/medium** to preserve during model corrections (`--repair`/`--geometry`/`--anatomy`),
    /// e.g. `--style "vintage watercolor storybook illustration"`. Anchors the re-paint to the source
    /// medium instead of drifting to photoreal (the cause of art regressions).
    #[arg(long, value_name = "TEXT", help_heading = "Corrective (needs a model)")]
    pub style: Option<String>,
    /// Shorthand for common art media (expands to a `--style` string): `watercolor` | `oil` | `ink` |
    /// `gouache` | `pencil` | `acrylic` | `pastel` | `comic`. `--style` overrides this.
    #[arg(long, value_name = "MEDIUM", help_heading = "Corrective (needs a model)")]
    pub medium: Option<String>,
    /// **De-clutter** — remove named nonsensical slop objects (OWL-ViT + inpaint) BEFORE the geometry
    /// fix. Comma-separated, e.g. `--declutter "overhead wires,cables"`. The only thing that kills a
    /// *compositional* hallucination (floating catenary wires, phantom rails) that img2img can't fix in
    /// place. Best-effort: an undetected target is skipped. Needs a model.
    #[arg(long, value_name = "OBJECTS", help_heading = "Corrective (needs a model)")]
    pub declutter: Option<String>,

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

/// Whether a `--declutter` target names thin-line clutter (wires/cables/lines) — routed to the weight-free
/// [`naturalize::wire_mask`] detector instead of OWL-ViT, which can't see thin lines.
fn is_wire_query(q: &str) -> bool {
    let q = q.to_ascii_lowercase();
    ["wire", "cable", "power line", "overhead line", "catenary", "telephone line", "electric line", "tram line", "wiring"]
        .iter()
        .any(|k| q.contains(k))
}

/// Resolve the art style to preserve during model corrections: explicit `--style` wins; else expand a
/// `--medium` preset to a descriptive style string; else `None`.
fn resolve_style(style: Option<&str>, medium: Option<&str>) -> Option<String> {
    if let Some(s) = style {
        if !s.trim().is_empty() {
            return Some(s.to_string());
        }
    }
    let m = medium?.trim().to_ascii_lowercase();
    let s = match m.as_str() {
        "watercolor" | "watercolour" => "soft wet-on-wet watercolor illustration, natural pigment granulation, paper texture",
        "oil" => "oil painting, visible brush strokes, impasto texture, canvas",
        "ink" => "ink drawing, pen and ink linework, cross-hatching",
        "gouache" => "gouache painting, matte opaque pigment, flat washes",
        "pencil" => "graphite pencil sketch, soft shading, paper tooth",
        "acrylic" => "acrylic painting, bold brushwork",
        "pastel" => "soft pastel drawing, chalky pigment, blended tones",
        "comic" => "comic book illustration, clean ink lines, cel shading",
        other => return Some(other.to_string()), // pass unknown media through verbatim
    };
    Some(s.to_string())
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
    if let Some(v) = a.polish {
        p.polish = v;
    }
    if let Some(v) = a.micro {
        p.micro = v;
    }

    let tmp = tempfile::tempdir().context("temp dir for naturalize refine")?;
    let art_style = resolve_style(a.style.as_deref(), a.medium.as_deref());

    // 0. Face-protected repair (model-backed, art-safe) runs FIRST when requested: protect the faces,
    //    gently repaint the rest IN-STYLE to attempt broken limbs — the character-preserving alternative
    //    to whole-image --geometry on figure art.
    let mut current_input = a.input.clone();
    if let Some(n) = a.repair.filter(|v| *v > 0.0) {
        let strength = (0.3 * n).clamp(0.12, 0.6);
        let scope = naturalize::refine::RepairScope::parse(&a.repair_scope)
            .with_context(|| format!("unknown --repair-scope `{}` (figures|non-face|full)", a.repair_scope))?;
        let repaired = tmp.path().join("repaired.png");
        match naturalize::refine::repair_protected(&a.input, &repaired, strength, art_style.as_deref(), scope, &a.model, Some(&a.device), a.refine_steps, tmp.path()).await {
            Ok(true) => {
                println!("  {} face-protected repair (scope {:?}, strength {strength:.2}{})", style("de-slop").green(), scope, art_style.as_deref().map(|s| format!(", style: {s}")).unwrap_or_default());
                current_input = repaired;
            }
            Ok(false) => println!("  {} repair skipped (no faces / no detector) — try --geometry", style("de-slop").yellow()),
            Err(e) => tracing::warn!(target: "plakat", "naturalize --repair: {e}"),
        }
    }

    // 1. Corrective refine (model-backed): fix structure (geometry / anatomy) via whole-image img2img.
    //    (Character-destructive on cohesive art — prefer --repair there; fine for photoreal / non-figure.)
    let corrective = naturalize::refine::Corrective {
        geometry: a.geometry.unwrap_or(0.0),
        anatomy: a.anatomy.unwrap_or(0.0),
        no_twins: a.no_twins.unwrap_or(0.0),
    };
    current_input = if corrective.any() {
        let refined = tmp.path().join("refined.png");
        naturalize::refine::refine(&current_input, &refined, &corrective, &a.model, Some(&a.device), a.refine_steps, tmp.path()).await?;
        refined
    } else {
        current_input
    };

    // 2. De-clutter runs AFTER the geometry fix — the geometry img2img REGENERATES the scene (and would
    //    re-hallucinate removed wires), so clutter removal must be the LAST model step. Remove named
    //    compositional slop (floating wires, phantom rails) that img2img can't fix in place.
    if let Some(spec) = a.declutter.as_deref() {
        let targets: Vec<&str> = spec.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
        if !targets.is_empty() {
            let device = crate::api::device(&a.device).context("resolving device for --declutter")?;
            for (i, q) in targets.iter().enumerate() {
                let out = tmp.path().join(format!("declutter_{i}.png"));
                // Wire-like queries → weight-free sky-gated wire detector + inpaint (OWL-ViT is blind to
                // thin lines). Everything else → OWL-ViT open-vocab detection + inpaint.
                if is_wire_query(q) {
                    let img = image::open(&current_input).with_context(|| format!("reading {}", current_input.display()))?.to_rgb8();
                    let mask = naturalize::wire_mask(&img, 0.6);
                    let coverage = mask.pixels().filter(|p| p.0[0] > 127).count();
                    if coverage == 0 {
                        println!("  {} no thin-line structure detected for '{q}' — skipped", style("de-slop").yellow());
                        continue;
                    }
                    match crate::cli::remove::inpaint_masked(&current_input, &mask, &out, "sdxl-inpaint", &device, a.refine_steps).await {
                        Ok(()) => {
                            println!("  {} removed thin-line clutter ('{q}', {coverage}px)", style("de-slop").green());
                            current_input = out;
                        }
                        Err(e) => tracing::warn!(target: "plakat", "naturalize --declutter '{q}' (wire): {e}"),
                    }
                } else {
                    match crate::cli::remove::declutter_one(&current_input, q, &out, "sdxl-inpaint", &device, a.refine_steps).await {
                        Ok(true) => {
                            println!("  {} decluttered '{q}'", style("de-slop").green());
                            current_input = out;
                        }
                        Ok(false) => println!("  {} '{q}' not found — skipped", style("de-slop").yellow()),
                        Err(e) => tracing::warn!(target: "plakat", "naturalize --declutter '{q}': {e}"),
                    }
                }
            }
        }
    }

    let src_for_analog = current_input;

    let mut img = image::open(&src_for_analog).with_context(|| format!("reading {}", src_for_analog.display()))?.to_rgb8();
    let ai_before = naturalize::ai_tell_score(&img);
    // ghost-signature removal (weight-free) before the analog pass.
    if let Some(cs) = a.designature.as_deref() {
        let corner = naturalize::Corner::parse(cs).with_context(|| format!("unknown corner `{cs}` (br|bl|tr|tl)"))?;
        img = naturalize::designature(&img, corner, a.designature_strength);
    }
    let out = naturalize::apply(&img, &p);
    let ai_after = naturalize::ai_tell_score(&out);

    // Etch bar (QUALITY-2 P2): if the input was plakat-etched, re-etch the output — re-embed L1 into the
    // new pixels + chain the original as `parent` — so `doctor --if-plakat` resolves it as a derivative.
    // Otherwise plain-save and (for non-etched images) carry any metadata sidecar/chunks forward.
    let mut reetched: Option<crate::etch::EtchId> = None;
    let mut fresh: Option<crate::etch::EtchId> = None;
    let mut carried = false;
    if !a.no_reetch {
        reetched = crate::etch::reetch(&a.input, out.as_raw(), out.width(), out.height(), &a.out).unwrap_or(None);
    }
    if reetched.is_none() {
        // No plakat parent to chain. If the user explicitly asked for `--etch`, freshly etch this output
        // (plakat produced *this* naturalized image) — same claim `generate --etch` makes. Otherwise
        // plain-save and carry any existing provenance forward.
        if !a.no_reetch && crate::etch::active().is_some() {
            fresh = crate::etch::fresh_etch(out.as_raw(), out.width(), out.height(), &a.out, None).ok();
        }
        if fresh.is_none() {
            out.save(&a.out).with_context(|| format!("writing {}", a.out.display()))?;
            if !a.no_reetch {
                carried = carry_provenance(&a.input, &a.out).unwrap_or(false);
            }
        }
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
    if let Some(id) = reetched {
        println!("  {} re-etched (fresh L1 in the new pixels, id {:016x}, source chained as parent) — `doctor --if-plakat` verifies it, provenance preserved.", style("etch").green(), id.0);
    } else if let Some(id) = fresh {
        println!("  {} etched (fresh L0+L1, id {:016x}, no parent — plakat produced this naturalized image) — `doctor --if-plakat` verifies it.", style("etch").green(), id.0);
    } else if carried {
        println!("  {} metadata carried forward (input not plakat-etched).", style("etch").cyan());
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
pub(crate) fn extract_text_chunks(bytes: &[u8]) -> Vec<Vec<u8>> {
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
pub(crate) fn inject_text_chunks(path: &Path, chunks: &[Vec<u8>]) -> Result<()> {
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
