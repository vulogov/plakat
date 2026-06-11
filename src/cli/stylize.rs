use anyhow::{Result, bail};
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

/// Curated `--strength` values for common stylize use cases. Each preset
/// is what we'd actually recommend after experience with the
/// shared-cross-attention IP-Adapter implementation we ship — see
/// [Documentation/GENERATE.md] for the full strength → behaviour map.
#[derive(Clone, Copy, Debug)]
pub enum StylizePreset {
    /// Face-preserving restyling. `0.35` — keeps identity intact while
    /// picking up REF's palette/brushwork.
    Portrait,
    /// Balanced restyling for non-face scenes (landscapes, objects,
    /// architecture). `0.55` — clear style shift; coarse structure preserved.
    Scene,
    /// Tonal/colour grading. `0.25` — adds REF's character without
    /// redrawing anything; safest preset for photos you want to keep
    /// recognisable.
    Grading,
}

impl StylizePreset {
    pub fn strength(self) -> f32 {
        match self {
            Self::Portrait => 0.35,
            Self::Scene => 0.55,
            Self::Grading => 0.25,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Portrait => "portrait (0.35, face-preserving)",
            Self::Scene => "scene (0.55, balanced)",
            Self::Grading => "grading (0.25, tonal only)",
        }
    }
}

impl std::str::FromStr for StylizePreset {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "portrait" | "face" | "person" => Self::Portrait,
            "scene" | "landscape" | "balanced" => Self::Scene,
            "grading" | "grade" | "tonal" | "color" | "colour" => Self::Grading,
            other => bail!(
                "unknown stylize preset {other:?} \
                 (valid: portrait, scene, grading)"
            ),
        })
    }
}

/// Transfer a reference image's *look* onto a subject via IP-Adapter — no
/// prompt, no training. Runs on SD 1.5 (`--model sd15`) or SDXL
/// (`--model sdxl` — sharper, native 1024²; SD 1.5 kept as a fallback).
///
/// The DEFAULT path is a ref-guided *variation* tool: the IP-Adapter transfers a
/// reference's CONTENT / appearance / palette, NOT painterly *texture*, so output
/// stays photoreal even on SDXL. For true painterly STYLE transfer use
/// **`--instantstyle`** (SDXL only — InstantStyle injects the reference only into
/// the style block via a decoupled IP cross-attention, decoupling style from
/// content), or a trained style LoRA (`plakat style train`) / `--look`.
/// `--ref-blur` suppresses the ref's content on the default path.
#[derive(ClapArgs, Debug)]
pub struct StylizeArgs {
    /// Input image to transform.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Style reference image.
    #[arg(long = "ref", value_name = "REF")]
    pub reference: PathBuf,

    /// Output image path.
    #[arg(long, value_name = "OUT")]
    pub out: PathBuf,

    /// Strength of style transfer in [0.0, 1.0]. Higher = closer to REF.
    /// Default 0.7 is "heavy restyle" — drop to 0.35 for face inputs,
    /// or use `--for portrait` (which does this automatically).
    /// Explicit `--strength` always wins over `--for`.
    #[arg(long, default_value_t = 0.7)]
    pub strength: f32,

    /// Strength preset shortcut: pick a documented strength for a use case.
    /// Overridden by an explicit `--strength`.
    ///
    ///   portrait → 0.35 (face-preserving)
    ///   scene    → 0.55 (balanced; landscapes/objects)
    ///   grading  → 0.25 (tonal/colour only)
    #[arg(long = "preset", value_name = "PRESET")]
    pub preset: Option<StylizePreset>,

    /// Base diffusion model (alias or HF repo id). Currently SD 1.5 only.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Denoising steps.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// **v0.46**: Gaussian-blur the reference before encoding it (sigma; 0 =
    /// off). Wipes the ref's fine content (subject/face) while keeping its
    /// broad style — palette, texture, composition — so stylize transfers a
    /// LOOK, not a subject. The cheap "style not content" knob; try ~8-14.
    #[arg(long = "ref-blur", default_value_t = 0.0)]
    pub ref_blur: f32,

    /// **v0.46**: scale the reference's influence (1.0 = full). Lower lets the
    /// prompt own the subject while the ref owns the look.
    #[arg(long = "ref-weight", default_value_t = 1.0)]
    pub ref_weight: f32,

    /// **InstantStyle** (SD 1.5 + SDXL): true painterly STYLE transfer — inject the
    /// reference only into the style block (SDXL `up_blocks.0.attentions.1`, SD 1.5
    /// `up_blocks.1.attentions.1`) via a decoupled IP cross-attention, not the
    /// content/palette concat path. The real style machine vs the ref-variation default.
    #[arg(long, default_value_t = false)]
    pub instantstyle: bool,

    /// InstantStyle injection strength (the style-block IP scale). 1.0 is the
    /// t2i-canonical value, but stylize is **img2img**, which dilutes a single
    /// block — the style reads clearly only at higher scale (~3-5) AND higher
    /// `--strength` (~0.8, so the denoise has room to repaint). 3.0 is a visible
    /// default; raise for heavier paint, lower to keep more of the photo.
    #[arg(long = "style-scale", default_value_t = 3.0)]
    pub style_scale: f32,
}

/// Sentinel matching `StylizeArgs::strength`'s `default_value_t`. Used to
/// detect "user didn't pass --strength" so a preset can take effect.
const DEFAULT_STRENGTH: f32 = 0.7;

pub async fn run(args: StylizeArgs, device: Device) -> Result<()> {
    // Resolve effective strength:
    //   * explicit --strength (anything ≠ default) wins;
    //   * else, preset value if given;
    //   * else, default.
    let user_set_strength = (args.strength - DEFAULT_STRENGTH).abs() > f32::EPSILON;
    let effective_strength = if user_set_strength {
        args.strength
    } else if let Some(preset) = args.preset {
        let s = preset.strength();
        crate::ui::progress::println(&format!(
            "stylize preset: {} → --strength {:.2}",
            preset.label(),
            s
        ));
        s
    } else {
        args.strength
    };

    // Warn if both were given and they disagree (explicit wins, but the
    // user probably wanted only one).
    if user_set_strength && args.preset.is_some() {
        let preset = args.preset.unwrap();
        crate::ui::progress::println(&format!(
            "note: both --strength {:.2} and --for {} were given; \
             using --strength (explicit override).",
            args.strength,
            preset.label(),
        ));
    }

    crate::pipelines::stylize::run(crate::pipelines::stylize::Request {
        input: args.input,
        reference: args.reference,
        out: args.out,
        strength: effective_strength,
        model: args.model,
        steps: args.steps,
        seed: args.seed,
        ref_blur: args.ref_blur,
        ref_weight: args.ref_weight,
        instantstyle: args.instantstyle,
        style_scale: args.style_scale,
        device,
    })
    .await
}
