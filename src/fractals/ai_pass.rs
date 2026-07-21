//! Track B — the optional AI enhancement pass (RFC FRACTALS-1, Phase 4).
//!
//! The deterministic Track-A render is used as both the img2img init image *and* the
//! ControlNet conditioning source: the fractal's structure guides a diffusion repaint
//! that adds texture, lighting, and material. Reuses the existing generation stack
//! (`pipelines::img2img` + `pipelines::controlnet`), mirroring `map::render_sd`.
//!
//! Opt-in only (`ai.enabled` / `--fractal-paint`); Track A is always saved first and is
//! never touched by this pass.

use anyhow::{Context, Result};
use candle_core::Device;
use std::path::Path;
use std::str::FromStr;

use crate::pipelines::controlnet::{ControlKind, ControlSpec};
use crate::pipelines::img2img::{self, Request};
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;

use super::spec::{FractalKind, FractalSpec};

/// An SDXL LoRA that leans a repaint toward fractal/psychedelic detail. Not applied by
/// default (the repo isn't a hard dependency) — pass it via `--fractal-sd-lora` or set
/// `ai.loras` if you have it cached. Kept here as the canonical suggestion.
pub const SUGGESTED_FRACTAL_LORA: &str = "artificialguybr/fractalredmond";

/// The default ControlNet type per family: escape-time boundaries read best as **Canny**
/// edges; the crisp line families as **Lineart**; buddhabrot's soft density as **SoftEdge**.
pub fn default_control_for_kind(kind: FractalKind) -> ControlKind {
    match kind {
        FractalKind::Ifs | FractalKind::Lsystem => ControlKind::Lineart,
        FractalKind::Buddhabrot => ControlKind::SoftEdge,
        _ => ControlKind::Canny,
    }
}

/// A deterministic per-family `(positive, negative)` prompt for the paint pass.
pub fn fractal_prompt(spec: &FractalSpec) -> (String, String) {
    let subject = match spec.kind {
        FractalKind::BurningShip => {
            "an ominous burning-ship fractal, dark molten structure, embers and smoke"
        }
        FractalKind::Newton | FractalKind::Nova => {
            "a newton fractal, smooth glossy basins, liquid metal, iridescent boundaries"
        }
        FractalKind::Ifs => {
            "a delicate fractal fern, organic botanical structure, dewy leaves, elegant"
        }
        FractalKind::Lsystem => {
            "an intricate fractal plant, botanical ink line-art, elegant recursive branching"
        }
        FractalKind::Buddhabrot => {
            "an ethereal cosmic nebula, glowing stardust, deep-space astrophotography"
        }
        _ => "an intricate fractal, ornate recursive filigree, iridescent, mesmerizing deep zoom",
    };
    let positive = format!(
        "{subject}, highly detailed, sharp focus, rich color, volumetric light, 8k, masterpiece"
    );
    let negative =
        "blurry, low quality, jpeg artifacts, watermark, text, signature, flat, washed out, deformed"
            .to_string();
    (positive, negative)
}

/// Parse the LoRA CLI/spec strings into `LoraSpec`s (skipping blanks and `none`).
fn parse_loras(specs: &[String]) -> Result<Vec<LoraSpec>> {
    specs
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("none"))
        .map(|s| LoraSpec::from_str(s).with_context(|| format!("parsing LoRA spec {s:?}")))
        .collect()
}

/// The newest PNG file in `dir` (img2img writes `plakat-img2img-<seed>.png`).
fn newest_png(dir: &Path) -> Result<std::path::PathBuf> {
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("png"))
            == Some(true)
        {
            let mtime = path.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().map(|(t, _)| mtime >= *t).unwrap_or(true) {
                best = Some((mtime, path));
            }
        }
    }
    best.map(|(_, p)| p)
        .with_context(|| format!("AI paint pass produced no PNG in {}", dir.display()))
}

/// Run the Track-B paint pass: `base_png` (the Track-A render) → ControlNet-guided
/// img2img → `out`. `base_png` is left untouched.
pub async fn run_ai_pass(
    spec: &FractalSpec,
    base_png: &Path,
    out: &Path,
    device: Device,
) -> Result<()> {
    let ai = &spec.ai;
    let (mut positive, mut negative) = fractal_prompt(spec);
    if !ai.prompt.trim().is_empty() {
        positive = ai.prompt.clone();
    }
    if !ai.negative.trim().is_empty() {
        negative = ai.negative.clone();
    }

    let control = if ai.control.trim().is_empty() {
        default_control_for_kind(spec.kind)
    } else {
        ControlKind::from_str(ai.control.trim())
            .map_err(|e| anyhow::anyhow!("bad control type {:?}: {e}", ai.control))?
    };

    let loras = parse_loras(&ai.loras)?;
    let scratch = tempfile::tempdir().context("creating AI-pass scratch dir")?;

    let req = Request {
        prompt: positive,
        negative,
        model: ai.model.clone(),
        device,
        loras,
        lora_scale: ai.lora_scale,
        input: base_png.to_path_buf(),
        mask: None,
        mask_feather: 0,
        mask_invert: false,
        width: spec.width,
        height: spec.height,
        count: 1,
        steps: ai.steps as usize,
        guidance: ai.guidance,
        scheduler: SchedulerKind::Default,
        strength: ai.strength,
        seed: Some(spec.seed),
        out_dir: scratch.path().to_path_buf(),
        controls: vec![ControlSpec {
            kind: control,
            image: None,
            from: Some(base_png.to_path_buf()), // auto-annotate the Track-A render
            video: None,
            strength: ai.control_strength,
            start: 0.0,
            end: 1.0,
        }],
    };

    img2img::run(req).await.context("fractal AI paint pass (img2img + ControlNet)")?;

    let produced = newest_png(scratch.path())?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    std::fs::copy(&produced, out)
        .with_context(|| format!("writing painted fractal to {}", out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::spec::AiSpec;

    #[test]
    fn control_defaults_per_family() {
        assert_eq!(default_control_for_kind(FractalKind::Mandelbrot), ControlKind::Canny);
        assert_eq!(default_control_for_kind(FractalKind::Ifs), ControlKind::Lineart);
        assert_eq!(default_control_for_kind(FractalKind::Lsystem), ControlKind::Lineart);
        assert_eq!(default_control_for_kind(FractalKind::Buddhabrot), ControlKind::SoftEdge);
    }

    #[test]
    fn prompt_is_family_specific() {
        let m = fractal_prompt(&FractalSpec::default());
        assert!(m.0.contains("fractal"));
        let plant = fractal_prompt(&FractalSpec { kind: FractalKind::Lsystem, ..FractalSpec::default() });
        assert!(plant.0.contains("plant") || plant.0.contains("botanical"));
        assert!(!m.1.is_empty()); // negative prompt present
    }

    #[test]
    fn lora_parsing_skips_blanks_and_none() {
        let out = parse_loras(&["".into(), "none".into(), "org/name:0.8".into()]).unwrap();
        assert_eq!(out.len(), 1);
        assert!(parse_loras(&["  ".into()]).unwrap().is_empty());
    }

    #[test]
    fn explicit_control_overrides_default() {
        // A spec with an explicit control type is honored (parse path).
        let ai = AiSpec { control: "softedge".into(), ..AiSpec::default() };
        assert_eq!(ControlKind::from_str(ai.control.trim()).unwrap(), ControlKind::SoftEdge);
    }
}
