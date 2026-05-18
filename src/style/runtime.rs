//! Runtime helper: load catalog + encoder, detect or pick by name,
//! resolve to LoRAs + trigger.
//!
//! One call site per subcommand. Generate/portrait wrap a single call
//! here; scenario calls it once at scenario load time (or per task once
//! per-task style-ref ships).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use console::style;

use crate::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::t2i::Variant;

use super::catalog::{BaseModel, DetectionResult, ResolvedStyle, StyleCatalog};
use super::detect::detect_style;
use super::encode::encode_reference_photo;

/// Input to [`prepare_style`].
pub struct StylePrepRequest<'a> {
    /// Reference photo to detect style from. Either this or
    /// `style_override` (or both) must be set.
    pub style_ref: Option<&'a Path>,
    /// Bypass detection; pick a style by id. Overrides detection when
    /// both are set.
    pub style_override: Option<&'a str>,
    /// Multiplier on each catalog LoRA's `:scale`. `1.0` uses authored
    /// scales verbatim.
    pub style_strength: f32,
    /// Override the bundled catalog directory.
    pub style_catalog: Option<&'a Path>,
    /// Raw `--model` string. Mapped through `Variant::detect` then
    /// `BaseModel::from_variant` to pick the right per-base LoRA slot.
    pub model: &'a str,
    /// Used only for the warning flag — `true` when the caller has
    /// user-supplied LoRAs that the catalog is about to override.
    pub user_loras_nonempty: bool,
    pub device: &'a Device,
}

/// Output of [`prepare_style`]. Plumbed into the existing generation flow.
pub struct StylePrep {
    /// LoRA spec strings in plakat's existing `LoraSpec::from_str`
    /// grammar. Caller `.parse::<LoraSpec>()`s them.
    pub lora_specs: Vec<String>,
    /// Prepended ahead of the bare prompt (generate/portrait) or
    /// `lora-header` (scenario).
    pub trigger: String,
    /// Appended to the user's negative prompt.
    pub negative_extras: String,
    /// `true` when the user had `--lora` set alongside `--style-ref`;
    /// caller should print a one-line warning.
    pub warn_user_loras_overridden: bool,
    /// `true` when the resolved style has no LoRA / trigger / negative
    /// configured; caller should print a one-line "detection-only" note.
    pub warn_detection_only: bool,
    /// What detection produced (or what `--style` forced). Surfaced for
    /// logging.
    pub picked_style_id: String,
    /// Full detection result, if detection ran. `None` when
    /// `style_override` was used without `style_ref`.
    pub detection: Option<DetectionResult>,
}

const CATALOG_DEFAULT: &str = "assets/style_catalog";
const RUNTIME_ENCODER_ID: &str = "clip-h-laion2b";

/// End-to-end: load catalog, encode photo if needed, detect or pick,
/// resolve, return everything the generation pipeline needs to inject.
pub async fn prepare_style(req: StylePrepRequest<'_>) -> Result<StylePrep> {
    if req.style_ref.is_none() && req.style_override.is_none() {
        return Err(anyhow!(
            "prepare_style called with neither style_ref nor style_override"
        ));
    }

    // 1. Locate + load the catalog.
    let catalog_dir: PathBuf = req
        .style_catalog
        .map(|p| p.to_owned())
        .unwrap_or_else(|| PathBuf::from(CATALOG_DEFAULT));
    let catalog = StyleCatalog::load(&catalog_dir, req.device)?;
    catalog.assert_encoder(RUNTIME_ENCODER_ID)?;

    // 2. Detection (only if --style-ref is set).
    let detection = if let Some(photo) = req.style_ref {
        let weights =
            crate::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors")
                .await?;
        let encoder = ImageEncoder::load(&weights, req.device, DType::F32)?;
        let emb = encode_reference_photo(&encoder, photo, req.device)?;
        Some(detect_style(&catalog, &emb, 5)?)
    } else {
        None
    };

    // 3. Pick a style. --style overrides detection when both are set.
    let picked_style_id: String = match (req.style_override, detection.as_ref()) {
        (Some(id), _) => id.to_owned(),
        (None, Some(det)) => det.picked.clone().ok_or_else(|| {
            let closest = det
                .top
                .first()
                .map(|m| format!("{} ({:.4})", m.style_id, m.score))
                .unwrap_or_else(|| "<empty>".to_string());
            anyhow!(
                "no style above min_confidence={}; closest: {}. \
                 Pass --style <id> to force a specific style, or lower the \
                 threshold in catalog.json.",
                catalog.policy.min_confidence,
                closest
            )
        })?,
        (None, None) => unreachable!("guarded above"),
    };

    // 4. Resolve against the active base model.
    let base = BaseModel::from_variant(Variant::detect(req.model));
    let resolved: ResolvedStyle = catalog.resolve(&picked_style_id, base, req.style_strength)?;

    let warn_detection_only = resolved.is_detection_only();

    Ok(StylePrep {
        lora_specs: resolved.loras.into_iter().map(|l| l.spec).collect(),
        trigger: resolved.trigger,
        negative_extras: resolved.negative_extras,
        warn_user_loras_overridden: req.user_loras_nonempty,
        warn_detection_only,
        picked_style_id,
        detection,
    })
}

/// Parse the resolved LoRA spec strings into plakat's `LoraSpec` type.
/// Splits parsing out from `prepare_style` so the style module doesn't
/// have to surface a LoRA type, and callers can plug the result
/// directly into their existing `Vec<LoraSpec>` field.
pub fn parse_resolved_loras(prep: &StylePrep) -> Result<Vec<LoraSpec>> {
    prep.lora_specs
        .iter()
        .map(|s| {
            LoraSpec::from_str(s).with_context(|| format!("parsing catalog LoRA spec '{}'", s))
        })
        .collect()
}

/// Print the standard log lines for a [`StylePrep`]: picked style,
/// ambiguity note, user-LoRA-override warning, detection-only warning.
/// Shared by every subcommand that applies a style — single source of
/// truth for the log surface.
///
/// `n_user_loras_dropped` is only consulted when
/// `prep.warn_user_loras_overridden` is set, and is used to print the
/// exact count in the override warning.
pub fn log_style_prep(prep: &StylePrep, n_user_loras_dropped: usize) {
    crate::ui::progress::println(&format!(
        "  {} style: {}",
        style("→").cyan().bold(),
        style(&prep.picked_style_id).bold()
    ));

    if let Some(det) = &prep.detection {
        if det.ambiguous {
            if let Some(runner_up) = det.top.get(1) {
                crate::ui::progress::println(&format!(
                    "  {} ambiguous — runner-up: {} ({:.4})",
                    style("⚠").yellow(),
                    runner_up.style_id,
                    runner_up.score,
                ));
            }
        }
    }

    if prep.warn_user_loras_overridden {
        crate::ui::progress::println(&format!(
            "  {} --style-ref overrides {} user-specified LoRA(s); using catalog LoRAs only",
            style("⚠").yellow(),
            n_user_loras_dropped
        ));
    }

    if prep.warn_detection_only {
        crate::ui::progress::println(&format!(
            "  {} style '{}' is detection-only in the catalog (no LoRAs configured); \
             running with base model only",
            style("⚠").yellow(),
            prep.picked_style_id
        ));
    }
}

/// Combine a user-supplied negative prompt with the style's
/// `negative_extras`. Empty inputs are handled cleanly.
pub fn combine_negative(user: &str, extras: &str) -> String {
    match (user.is_empty(), extras.is_empty()) {
        (true, true) => String::new(),
        (true, false) => extras.to_owned(),
        (false, true) => user.to_owned(),
        (false, false) => format!("{}, {}", user, extras),
    }
}

/// Prepend `trigger` to `prompt`, with an exact-substring dedup guard
/// so already-present trigger text isn't duplicated.
pub fn prepend_trigger(trigger: &str, prompt: &str) -> String {
    if trigger.is_empty() {
        return prompt.to_owned();
    }
    if prompt.contains(trigger) {
        return prompt.to_owned();
    }
    if prompt.is_empty() {
        return trigger.to_owned();
    }
    format!("{}, {}", trigger, prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_negative_handles_empty_inputs() {
        assert_eq!(combine_negative("", ""), "");
        assert_eq!(combine_negative("low quality", ""), "low quality");
        assert_eq!(combine_negative("", "photo, glossy"), "photo, glossy");
        assert_eq!(
            combine_negative("low quality", "photo, glossy"),
            "low quality, photo, glossy"
        );
    }

    #[test]
    fn prepend_trigger_dedups_exact_substring() {
        assert_eq!(prepend_trigger("", "a fox"), "a fox");
        assert_eq!(prepend_trigger("watercolor", ""), "watercolor");
        assert_eq!(
            prepend_trigger("watercolor", "a fox in a forest"),
            "watercolor, a fox in a forest"
        );
        // Already present — no duplication.
        assert_eq!(
            prepend_trigger("watercolor", "a fox in watercolor"),
            "a fox in watercolor"
        );
    }
}
