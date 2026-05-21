//! Style catalog: schema types + loader.
//!
//! The catalog is a directory containing:
//!
//!   * `catalog.json` — routing metadata: style id → description, exemplar
//!     keys, per-base-model LoRA specs + trigger phrases.
//!   * `exemplars.safetensors` — CLIP-H pooled embeddings of exemplar
//!     images, L2-normalized, keyed `<style_id>/<idx>`.
//!
//! The schema is deserialize-only — catalogs are produced by the build
//! tool, never written by plakat at runtime.
//!
//! ## Versioning
//!
//! Two independent fields:
//!
//!   * `schema_version` — bumps on breaking JSON-schema changes.
//!   * `encoder.id` — fingerprint of the encoder that produced the
//!     exemplars (e.g. `clip-h-laion2b`). Cosines against incompatible
//!     embedding spaces are meaningless, so the runtime asserts this
//!     matches the encoder it's about to use.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::pipelines::t2i::Variant;

// ---------------------------------------------------------------------------
// Wire types — mirror catalog.json 1:1. Deserialize only.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct RawCatalog {
    pub schema_version: u32,
    pub encoder: EncoderMeta,
    #[serde(default)]
    pub detection: DetectionPolicy,
    pub styles: Vec<RawStyle>,
}

#[derive(Debug, Deserialize)]
pub struct EncoderMeta {
    pub id: String,
    pub embed_dim: usize,
    pub exemplars_file: String,
    #[serde(default = "default_preprocess")]
    pub preprocess: String,
}

fn default_preprocess() -> String {
    "clip-standard-224".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct DetectionPolicy {
    #[serde(default)]
    pub aggregation: Aggregation,
    #[serde(default = "default_min_conf")]
    pub min_confidence: f32,
    #[serde(default = "default_margin")]
    pub margin_over_runner_up: f32,
}

impl Default for DetectionPolicy {
    fn default() -> Self {
        Self {
            aggregation: Aggregation::default(),
            min_confidence: default_min_conf(),
            margin_over_runner_up: default_margin(),
        }
    }
}

fn default_min_conf() -> f32 {
    0.22
}
fn default_margin() -> f32 {
    0.02
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum Aggregation {
    Max,
    Mean,
    #[default]
    Top3Mean,
}

#[derive(Debug, Deserialize, Hash, Eq, PartialEq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum BaseModel {
    Sd15,
    Sdxl,
    Flux,
}

impl BaseModel {
    /// Map plakat's fine-grained model `Variant` into the catalog's 3-slot
    /// base-model family.
    ///
    /// SD 1.5 and SD 2.1 share the catalog's `sd15` slot because both use
    /// CLIP-L text encoders with 768-d cross-attention; SD 1.5 LoRAs
    /// generally work on 2.1 (and vice versa). The SDXL and SDXL-Turbo
    /// UNets share architecture too — same `sdxl` slot.
    pub fn from_variant(v: Variant) -> Self {
        match v {
            Variant::Sd15 | Variant::Sd21 => Self::Sd15,
            Variant::Sdxl | Variant::SdxlTurbo => Self::Sdxl,
            Variant::FluxSchnell | Variant::FluxDev | Variant::FluxFillDev => Self::Flux,
        }
    }

    /// Human-readable slug for log/error messages. Matches `serde(rename_all = "lowercase")`.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Sd15 => "sd15",
            Self::Sdxl => "sdxl",
            Self::Flux => "flux",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RawStyle {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    pub exemplar_keys: Vec<String>,
    #[serde(default)]
    pub models: HashMap<BaseModel, ModelEntry>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelEntry {
    #[serde(default)]
    pub loras: Vec<LoraEntry>,
    #[serde(default)]
    pub trigger: String,
    #[serde(default)]
    pub negative_extras: String,
}

/// String-or-object union — both `"org/repo:0.8"` and
/// `{ spec: "org/repo:0.8", revision: "..." }` parse.
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum LoraEntry {
    Shorthand(String),
    Full {
        spec: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        license: Option<String>,
        #[serde(default)]
        license_url: Option<String>,
    },
}

impl LoraEntry {
    pub fn spec(&self) -> &str {
        match self {
            Self::Shorthand(s) => s,
            Self::Full { spec, .. } => spec,
        }
    }

    pub fn revision(&self) -> Option<&str> {
        match self {
            Self::Shorthand(_) => None,
            Self::Full { revision, .. } => revision.as_deref(),
        }
    }
}

// ---------------------------------------------------------------------------
// Loaded types — after the JSON + safetensors are joined.
// ---------------------------------------------------------------------------

pub struct StyleCatalog {
    pub encoder_id: String,
    pub embed_dim: usize,
    pub policy: DetectionPolicy,
    /// Indexed by style id for O(1) lookup.
    pub styles: HashMap<String, LoadedStyle>,
    /// Preserves the catalog's JSON order — used for stable iteration in
    /// detect output and `plakat style list`.
    pub order: Vec<String>,
}

pub struct LoadedStyle {
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// Shape `(N_exemplars, embed_dim)`, f32, L2-normalized along dim 1.
    pub exemplars: Tensor,
    pub models: HashMap<BaseModel, ModelEntry>,
}

pub struct StyleMatch {
    pub style_id: String,
    pub display_name: String,
    pub score: f32,
}

pub struct DetectionResult {
    /// Sorted descending by score; length ≤ `top_k`.
    pub top: Vec<StyleMatch>,
    /// `None` when no style cleared `policy.min_confidence`.
    pub picked: Option<String>,
    /// `top[0].score - top[1].score < policy.margin_over_runner_up`.
    pub ambiguous: bool,
}

/// A LoRA reference, post-resolution, with the strength multiplier already
/// folded into the spec's `:scale` suffix. Pass `spec` directly into
/// plakat's existing `LoraSpec::from_str` grammar.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLoraRef {
    pub spec: String,
    pub revision: Option<String>,
}

/// Result of resolving a detected (or named) style against the active
/// base model. The consumer feeds `loras` into the standard LoRA-stacking
/// path, prepends `trigger` ahead of the existing `lora-header`, and
/// appends `negative_extras` to the user's negative prompt.
#[derive(Debug, Clone)]
pub struct ResolvedStyle {
    pub style_id: String,
    pub loras: Vec<ResolvedLoraRef>,
    pub trigger: String,
    pub negative_extras: String,
}

// ---------------------------------------------------------------------------
// Loader.
// ---------------------------------------------------------------------------

impl StyleCatalog {
    /// Load and validate a catalog from a directory containing
    /// `catalog.json` + the encoder's referenced exemplars file.
    pub fn load(catalog_dir: &Path, device: &Device) -> Result<Self> {
        let json_path = catalog_dir.join("catalog.json");
        let raw: RawCatalog = serde_json::from_str(
            &std::fs::read_to_string(&json_path)
                .with_context(|| format!("reading {}", json_path.display()))?,
        )
        .with_context(|| format!("parsing {}", json_path.display()))?;

        if raw.schema_version != 1 {
            bail!(
                "style catalog at {} uses schema_version={}, plakat supports 1",
                json_path.display(),
                raw.schema_version
            );
        }

        let exemplars_path = catalog_dir.join(&raw.encoder.exemplars_file);
        let tensors = candle_core::safetensors::load(&exemplars_path, device)
            .with_context(|| format!("reading {}", exemplars_path.display()))?;

        let mut styles = HashMap::with_capacity(raw.styles.len());
        let mut order = Vec::with_capacity(raw.styles.len());

        for s in raw.styles {
            if s.exemplar_keys.is_empty() {
                bail!("style '{}' has zero exemplar keys", s.id);
            }

            let rows: Vec<Tensor> = s
                .exemplar_keys
                .iter()
                .map(|k| {
                    tensors
                        .get(k)
                        .ok_or_else(|| {
                            anyhow!(
                                "style '{}' references missing exemplar key '{}'",
                                s.id,
                                k
                            )
                        })
                        .and_then(|t| Ok(t.to_dtype(DType::F32)?))
                })
                .collect::<Result<_>>()?;

            let exemplars = Tensor::stack(&rows, 0)?;
            let (n, d) = exemplars.dims2()?;
            if d != raw.encoder.embed_dim {
                bail!(
                    "style '{}' exemplars have dim {}, catalog declares embed_dim={}",
                    s.id,
                    d,
                    raw.encoder.embed_dim
                );
            }
            let _ = n; // used only for the shape assertion above

            order.push(s.id.clone());
            styles.insert(
                s.id.clone(),
                LoadedStyle {
                    id: s.id,
                    display_name: s.display_name,
                    description: s.description,
                    exemplars,
                    models: s.models,
                },
            );
        }

        if styles.is_empty() {
            bail!("style catalog has zero styles");
        }

        Ok(Self {
            encoder_id: raw.encoder.id,
            embed_dim: raw.encoder.embed_dim,
            policy: raw.detection,
            styles,
            order,
        })
    }

    /// Hard fail if the runtime encoder doesn't match the catalog's.
    /// Cosines across mismatched embedding spaces are meaningless.
    pub fn assert_encoder(&self, runtime_id: &str) -> Result<()> {
        if self.encoder_id != runtime_id {
            bail!(
                "style catalog was built with encoder '{}' but runtime is using '{}'",
                self.encoder_id,
                runtime_id
            );
        }
        Ok(())
    }

    /// Resolve a detected (or named) style into the LoRAs + trigger phrase
    /// the generation pipeline will apply.
    ///
    /// `strength_multiplier` scales every LoRA's `:scale` suffix. `1.0`
    /// uses the catalog's authored scales verbatim. The resulting `spec`
    /// strings are in plakat's existing `LoraSpec::from_str` grammar —
    /// pass them straight into the LoRA-stacking path.
    ///
    /// Returns an empty `ResolvedStyle` when the style has no `models`
    /// section at all (detection-only style; the caller should warn but
    /// continue). Returns an error when the style has models for *other*
    /// bases but not the requested one — that's a real cross-base
    /// coverage gap, not an authoring choice.
    pub fn resolve(
        &self,
        style_id: &str,
        base: BaseModel,
        strength_multiplier: f32,
    ) -> Result<ResolvedStyle> {
        let style = self
            .styles
            .get(style_id)
            .ok_or_else(|| anyhow!("unknown style id '{}'", style_id))?;

        if style.models.is_empty() {
            return Ok(ResolvedStyle {
                style_id: style_id.to_owned(),
                loras: Vec::new(),
                trigger: String::new(),
                negative_extras: String::new(),
            });
        }

        let model_entry = style.models.get(&base).ok_or_else(|| {
            let mut supported: Vec<&'static str> = style.models.keys().map(|b| b.slug()).collect();
            supported.sort();
            anyhow!(
                "style '{}' has no LoRA mapping for {}; supported bases for this style: [{}]",
                style_id,
                base.slug(),
                supported.join(", ")
            )
        })?;

        let loras = model_entry
            .loras
            .iter()
            .map(|entry| ResolvedLoraRef {
                spec: rewrite_scale(entry.spec(), strength_multiplier),
                revision: entry.revision().map(str::to_owned),
            })
            .collect();

        Ok(ResolvedStyle {
            style_id: style_id.to_owned(),
            loras,
            trigger: model_entry.trigger.clone(),
            negative_extras: model_entry.negative_extras.clone(),
        })
    }
}

impl ResolvedStyle {
    /// True when this style has no LoRAs and no trigger phrase configured
    /// — i.e. the catalog can detect this style but has no transfer
    /// instructions. Callers should warn and proceed without injecting.
    pub fn is_detection_only(&self) -> bool {
        self.loras.is_empty() && self.trigger.is_empty() && self.negative_extras.is_empty()
    }
}

/// Multiply the trailing `:scale` on a LoRA spec by `mult`. Re-emits the
/// spec in the same grammar `LoraSpec::from_str` accepts.
///
/// Cases:
/// - `"org/repo:0.8"`, `mult=0.5` → `"org/repo:0.4"`
/// - `"org/repo"`, `mult=0.7` → `"org/repo:0.7"` (no prior scale; assume 1.0)
/// - `"org/repo#f.safetensors:0.8"`, `mult=2.0` → `"org/repo#f.safetensors:1.6"`
/// - `"/abs/path:0.6"`, `mult=1.0` → `"/abs/path:0.6"`
fn rewrite_scale(spec: &str, mult: f32) -> String {
    if let Some((head, tail)) = spec.rsplit_once(':') {
        if let Ok(scale) = tail.parse::<f32>() {
            return format!("{}:{}", head, scale * mult);
        }
    }
    // No trailing `:float` — treat as implicit 1.0 and emit explicit scale.
    format!("{}:{}", spec, mult)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_scale_preserves_grammar() {
        // Standard HF spec with explicit scale.
        assert_eq!(rewrite_scale("org/repo:0.8", 0.5), "org/repo:0.4");
        // No prior scale — implicit 1.0.
        assert_eq!(rewrite_scale("org/repo", 0.7), "org/repo:0.7");
        // HF spec with explicit file path inside the repo.
        assert_eq!(
            rewrite_scale("org/repo#f.safetensors:0.8", 2.0),
            "org/repo#f.safetensors:1.6"
        );
        // Local path.
        assert_eq!(rewrite_scale("/abs/path:0.6", 1.0), "/abs/path:0.6");
        // Identity: mult=1.0 leaves scaled specs unchanged.
        assert_eq!(rewrite_scale("repo:0.8", 1.0), "repo:0.8");
    }

    #[test]
    fn base_model_from_variant_collapses_to_3_slots() {
        assert_eq!(BaseModel::from_variant(Variant::Sd15), BaseModel::Sd15);
        assert_eq!(BaseModel::from_variant(Variant::Sd21), BaseModel::Sd15);
        assert_eq!(BaseModel::from_variant(Variant::Sdxl), BaseModel::Sdxl);
        assert_eq!(BaseModel::from_variant(Variant::SdxlTurbo), BaseModel::Sdxl);
        assert_eq!(BaseModel::from_variant(Variant::FluxSchnell), BaseModel::Flux);
        assert_eq!(BaseModel::from_variant(Variant::FluxDev), BaseModel::Flux);
    }
}
