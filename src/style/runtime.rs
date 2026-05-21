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

use super::catalog::{BaseModel, DetectionResult, ResolvedLoraRef, ResolvedStyle, StyleCatalog};
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
    /// LoRAs the catalog resolved for this style. Each carries the spec
    /// string (in plakat's `LoraSpec::from_str` grammar) plus an optional
    /// pinned revision SHA. Use [`parse_resolved_loras`] to turn this
    /// into a `Vec<LoraSpec>` for the existing pipeline.
    pub loras: Vec<ResolvedLoraRef>,
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

/// Shared state for style-resolution work that may happen multiple times
/// in one process — e.g., a scenario with per-task `style-ref` overrides
/// where the same CLIP-H encoder serves many photos.
///
/// Loads the catalog eagerly (cheap), the CLIP-H encoder lazily on first
/// `prepare()` that needs it. Subsequent calls reuse the same encoder
/// rather than re-loading 2.5 GB of weights.
pub struct StyleSession {
    catalog: StyleCatalog,
    /// Phase 7f: stored as `Arc` so the encoder can be shared with
    /// `portrait::Pipeline`'s identity encoder when both run in one
    /// process (scenarios, portrait + style-ref). Lazy-loaded on
    /// first `prepare()` that needs it.
    encoder: Option<std::sync::Arc<ImageEncoder>>,
    device: Device,
}

impl StyleSession {
    /// Construct a session — catalog loaded immediately, encoder
    /// deferred until needed.
    pub fn load(catalog_dir: Option<&Path>, device: Device) -> Result<Self> {
        let catalog_dir: PathBuf = catalog_dir
            .map(|p| p.to_owned())
            .unwrap_or_else(|| PathBuf::from(CATALOG_DEFAULT));
        let catalog = StyleCatalog::load(&catalog_dir, &device)?;
        catalog.assert_encoder(RUNTIME_ENCODER_ID)?;
        Ok(Self {
            catalog,
            encoder: None,
            device,
        })
    }

    /// Side-effecting: ensures `self.encoder` is `Some` after returning.
    /// Returns `()` rather than `&ImageEncoder` so the caller can hold
    /// both `&self.encoder` and `&self.device` simultaneously without
    /// the borrow checker objecting (an `&mut self -> &T` chain locks
    /// `self` for the lifetime of the returned reference).
    async fn ensure_encoder_loaded(&mut self) -> Result<()> {
        if self.encoder.is_none() {
            let weights = crate::hf::download::get_file(
                IPA_REPO,
                "models/image_encoder/model.safetensors",
            )
            .await?;
            self.encoder = Some(std::sync::Arc::new(ImageEncoder::load(
                &weights,
                &self.device,
                DType::F32,
            )?));
        }
        Ok(())
    }

    /// Phase 7f. Inject a pre-loaded CLIP-H encoder so the session
    /// won't load one itself. Caller-supplied dtype/device need to
    /// match the rest of the session's expectations (F32 is the
    /// standard for stylize). No-op if the session already has one.
    pub fn set_shared_encoder(
        &mut self,
        encoder: std::sync::Arc<ImageEncoder>,
    ) {
        if self.encoder.is_none() {
            self.encoder = Some(encoder);
        }
    }

    /// Phase 7f. Hand out the encoder if it's been loaded — useful
    /// when one CLI flow lazy-loads CLIP-H here and wants to feed it
    /// into a later pipeline build instead of paying a second load.
    /// Returns `None` if `prepare()` hasn't run yet.
    pub fn shared_encoder(&self) -> Option<std::sync::Arc<ImageEncoder>> {
        self.encoder.clone()
    }

    /// Detect (if photo set) and resolve a style against the catalog.
    /// `req.style_catalog` is ignored — the catalog was locked in at
    /// session-load time.
    pub async fn prepare(&mut self, req: StylePrepRequest<'_>) -> Result<StylePrep> {
        if req.style_ref.is_none() && req.style_override.is_none() {
            return Err(anyhow!(
                "StyleSession::prepare called with neither style_ref nor style_override"
            ));
        }

        let detection = if let Some(photo) = req.style_ref {
            self.ensure_encoder_loaded().await?;
            let encoder = self.encoder.as_ref().expect("ensure_encoder_loaded sets this");
            let emb = encode_reference_photo(encoder, photo, &self.device)?;
            Some(detect_style(&self.catalog, &emb, 5)?)
        } else {
            None
        };

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
                    self.catalog.policy.min_confidence,
                    closest
                )
            })?,
            (None, None) => unreachable!("guarded above"),
        };

        let base = BaseModel::from_variant(Variant::detect(req.model));
        let resolved: ResolvedStyle =
            self.catalog.resolve(&picked_style_id, base, req.style_strength)?;
        let warn_detection_only = resolved.is_detection_only();

        Ok(StylePrep {
            loras: resolved.loras,
            trigger: resolved.trigger,
            negative_extras: resolved.negative_extras,
            warn_user_loras_overridden: req.user_loras_nonempty,
            warn_detection_only,
            picked_style_id,
            detection,
        })
    }
}

/// Convenience: load a fresh session, run one prepare call, return the
/// result. Equivalent to `StyleSession::load(...)?.prepare(req).await`
/// for callers that only need a single resolve.
///
/// Generate / portrait use this — they only call prepare once per
/// invocation, so amortizing the catalog/encoder across calls isn't
/// needed. Scenarios with per-task style-ref use the session API
/// directly to share the encoder across tasks.
pub async fn prepare_style(req: StylePrepRequest<'_>) -> Result<StylePrep> {
    let (prep, _) = prepare_style_with_session(req).await?;
    Ok(prep)
}

/// Phase 7f variant of [`prepare_style`] that also hands back any
/// CLIP-H image encoder the session lazy-loaded during the prepare
/// call. Lets the CLI feed that same encoder into the downstream
/// portrait pipeline so PlusFace identity doesn't pay for a second
/// load of the same ~2.5 GB weight set. Returns `None` for the encoder
/// when the prep didn't actually need to encode (e.g. user passed
/// `--style ID` directly without a `--style-ref` photo, or the prep
/// failed before the lazy load fired).
pub async fn prepare_style_with_session(
    req: StylePrepRequest<'_>,
) -> Result<(StylePrep, Option<std::sync::Arc<ImageEncoder>>)> {
    let mut session = StyleSession::load(req.style_catalog, req.device.clone())?;
    let prep = session.prepare(req).await?;
    let shared = session.shared_encoder();
    Ok((prep, shared))
}

/// Parse resolved LoRA refs into plakat's `LoraSpec` type. When the
/// catalog pinned a `revision` for a hub-sourced LoRA, swap in a
/// pinned-revision spec so the downloader hits the exact commit SHA
/// rather than the repo's current `main`.
///
/// Splits parsing out from `prepare_style` so the style module
/// doesn't have to surface a `LoraSpec` type from this function,
/// and callers can plug the result directly into their existing
/// `Vec<LoraSpec>` field.
pub fn parse_resolved_loras(prep: &StylePrep) -> Result<Vec<LoraSpec>> {
    prep.loras
        .iter()
        .map(|r| {
            let parsed = LoraSpec::from_str(&r.spec)
                .with_context(|| format!("parsing catalog LoRA spec '{}'", r.spec))?;
            // Only hub specs honor revision pinning. Local-path specs
            // keep their path as-is — revision is meaningless there.
            match (parsed.source, r.revision.as_ref()) {
                (crate::pipelines::lora::LoraSource::Hub { repo, file, .. }, Some(rev)) => {
                    Ok(LoraSpec::hub_pinned(repo, file, Some(rev.clone()), parsed.scale))
                }
                (source, _) => Ok(LoraSpec {
                    source,
                    scale: parsed.scale,
                }),
            }
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
