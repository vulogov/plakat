//! v0.25: art-style presets — `--look` (medium axis) + `--genre`
//! (subject-domain axis).
//!
//! Both axes share the same [`PresetSpec`] shape. Looks live in
//! `assets/looks/catalog.json` (8 bundled mediums); genres live in
//! `assets/genres/catalog.json` (1 bundled: anime). User extensions
//! land under `$CONFIG_DIR/{looks,genres}/*.json` (wired in phase 9).
//!
//! See `Documentation/RFC_v0.25_LOOKS_AND_GENRES.md` for the design.

pub mod discovery;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

/// One preset entry — shared shape for [`Kind::Look`] and
/// [`Kind::Genre`]. JSON shape matches `assets/{looks,genres}/catalog.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresetSpec {
    /// Kebab-case identifier; the value passed to `--look NAME` /
    /// `--genre NAME`.
    pub name: String,
    /// Human-readable label shown in `--list` output and logs.
    pub display_name: String,
    /// One-line summary.
    pub description: String,

    /// Prepended to the user's prompt when the preset is applied.
    /// Compositional (always applies when preset is loaded).
    pub prompt_prefix: Option<String>,
    /// Appended to the user's prompt.
    pub prompt_suffix: Option<String>,
    /// Appended (comma-joined) to the user's `--negative` string.
    pub negative_extras: Option<String>,

    /// Recommended sampler. Override-only: applied only if user
    /// didn't pass `--scheduler` (detected via [`GenerationParams::scheduler`] being `None`).
    pub scheduler_hint: Option<String>,
    /// Recommended step count. Override-only.
    pub steps: Option<usize>,
    /// Recommended CFG scale. Override-only.
    pub guidance: Option<f64>,

    /// Drives automatic LoRA discovery (phases 4–5). Discovery fires
    /// only when the caller's LoRA stack is empty AND this field is
    /// set.
    pub lora_query: Option<LoraQuery>,

    /// Compatible base-model families. `None` = compatible with all
    /// (sd15 / sdxl / flux / sd3). Otherwise an allow-list.
    pub base_compat: Option<Vec<String>>,
}

/// Search terms that drive Civitai/HF/local LoRA discovery (phases
/// 4–5). Both fields are optional but the catalog convention is to
/// populate both for best discovery recall.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraQuery {
    /// Exact-match tags (Civitai-style taxonomy).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Fuzzy-search keywords (free text).
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// Which catalog a [`Catalog`] reads from. Affects only the default
/// asset path + the JSON outer key (`looks` vs `genres`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Look,
    Genre,
}

impl Kind {
    fn outer_key(self) -> &'static str {
        match self {
            Kind::Look => "looks",
            Kind::Genre => "genres",
        }
    }
    fn default_asset_dir(self) -> &'static str {
        match self {
            Kind::Look => "assets/looks",
            Kind::Genre => "assets/genres",
        }
    }
}

/// Loaded catalog — schema_version + entries. Cheap to clone.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub schema_version: u32,
    pub entries: Vec<PresetSpec>,
    pub kind: Kind,
}

impl Catalog {
    /// Load the bundled catalog for `kind` from
    /// `assets/{looks,genres}/catalog.json` relative to the current
    /// working directory. Phase 9 layers user-extension lookup on
    /// top via [`Self::load_with_user_dir`].
    pub fn load_default(kind: Kind) -> Result<Self> {
        let path = PathBuf::from(kind.default_asset_dir()).join("catalog.json");
        Self::load_from(&path, kind)
    }

    /// Load from an explicit `catalog.json` path. Surfaced for tests
    /// + scripting (`plakat.config.set "looks_catalog" "..."` lands
    /// in phase 9).
    pub fn load_from(path: &Path, kind: Kind) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("reading {} catalog at {}", kind.outer_key(), path.display()))?;
        Self::parse(&bytes, kind)
    }

    /// Parse a serialized catalog from bytes. Validates that:
    /// - `schema_version` is present
    /// - the outer `looks` / `genres` array exists
    /// - every entry has a `name` field
    /// - names are unique within the catalog
    pub fn parse(bytes: &[u8], kind: Kind) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_slice(bytes)
            .with_context(|| format!("parsing {} catalog JSON", kind.outer_key()))?;

        let schema_version = raw
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("{} catalog missing schema_version", kind.outer_key()))?
            as u32;

        let entries_val = raw
            .get(kind.outer_key())
            .ok_or_else(|| anyhow!("{} catalog missing outer `{}` array", kind.outer_key(), kind.outer_key()))?;

        let entries: Vec<PresetSpec> = serde_json::from_value(entries_val.clone())
            .with_context(|| format!("decoding {} catalog entries", kind.outer_key()))?;

        let mut seen = std::collections::HashSet::new();
        for entry in &entries {
            if entry.name.is_empty() {
                return Err(anyhow!("{} catalog entry has empty `name`", kind.outer_key()));
            }
            if !seen.insert(&entry.name) {
                return Err(anyhow!(
                    "{} catalog has duplicate `name`: {}",
                    kind.outer_key(),
                    entry.name
                ));
            }
        }

        Ok(Self {
            schema_version,
            entries,
            kind,
        })
    }

    /// Look up an entry by `name`. Case-sensitive (kebab-case is the
    /// convention).
    pub fn find(&self, name: &str) -> Option<&PresetSpec> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// All entry names in catalog order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }
}

/// Generation parameters subject to preset application. The
/// `Option<T>` shape lets [`PresetSpec::apply`] distinguish
/// "user-passed" (`Some`) from "user-didn't-pass" (`None`). The
/// caller is responsible for building this from CLI/scenario/bund
/// input.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationParams {
    pub prompt: String,
    pub negative: String,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub scheduler: Option<String>,
}

impl PresetSpec {
    /// Apply this preset to `params`. Three buckets:
    ///
    /// 1. **Override fields** (`steps`, `guidance`, `scheduler`):
    ///    fill in only when the user left them `None`. Never
    ///    overwrite a `Some(_)`.
    /// 2. **Compositional fields** (`prompt_prefix`,
    ///    `prompt_suffix`, `negative_extras`): always append/prepend
    ///    when set on the preset. The user can opt out by not
    ///    passing the preset.
    /// 3. **Discovery-gating fields** (`lora_query`, `base_compat`):
    ///    not applied here; consumed by the discovery client
    ///    (phase 4) and the base-compat filter.
    pub fn apply(&self, params: &mut GenerationParams) {
        // Bucket 1: override-only
        if params.steps.is_none() {
            params.steps = self.steps;
        }
        if params.guidance.is_none() {
            params.guidance = self.guidance;
        }
        if params.scheduler.is_none() {
            params.scheduler = self.scheduler_hint.clone();
        }

        // Bucket 2: compositional
        if let Some(prefix) = &self.prompt_prefix {
            params.prompt = if params.prompt.is_empty() {
                prefix.clone()
            } else {
                format!("{prefix}, {}", params.prompt)
            };
        }
        if let Some(suffix) = &self.prompt_suffix {
            // Suffix is authored with a leading ", " in the bundled
            // catalog so a direct concat works without double
            // commas. If it lacks the leading separator, we add one.
            let needs_sep = !suffix.starts_with(',') && !suffix.starts_with(' ');
            if needs_sep && !params.prompt.is_empty() {
                params.prompt.push_str(", ");
            }
            params.prompt.push_str(suffix);
        }
        if let Some(extras) = &self.negative_extras {
            params.negative = if params.negative.is_empty() {
                extras.clone()
            } else {
                format!("{}, {extras}", params.negative)
            };
        }
    }

    /// True if this preset's `base_compat` allows `base` (or if
    /// `base_compat` is `None`, meaning compatible with everything).
    pub fn is_compatible_with(&self, base: &str) -> bool {
        match &self.base_compat {
            None => true,
            Some(list) => list.iter().any(|b| b == base),
        }
    }
}

/// Apply look + genre presets to `params`. Loads the bundled
/// catalogs from `assets/{looks,genres}/catalog.json`. Order is
/// **look first, genre second** — under the override-only rule
/// the first applier fills `Option::None` fields; the second
/// only contributes its compositional pieces (prompt
/// prefix/suffix, negative_extras). This matches the natural
/// reading "a watercolor anime" where the medium (look) governs
/// sampler/steps and the genre (anime) adds subject framing.
///
/// Returns the resolved `(look_spec, genre_spec)` so the caller
/// can log what was applied and feed `lora_query` into the
/// discovery step (phases 4–5).
pub fn apply_presets(
    look_name: Option<&str>,
    genre_name: Option<&str>,
    params: &mut GenerationParams,
) -> Result<(Option<PresetSpec>, Option<PresetSpec>)> {
    let look_spec = match look_name {
        Some(name) => {
            let cat = Catalog::load_default(Kind::Look)?;
            let spec = cat
                .find(name)
                .ok_or_else(|| {
                    anyhow!(
                        "unknown --look {name:?} (try one of: {})",
                        cat.names().join(", ")
                    )
                })?
                .clone();
            spec.apply(params);
            Some(spec)
        }
        None => None,
    };
    let genre_spec = match genre_name {
        Some(name) => {
            let cat = Catalog::load_default(Kind::Genre)?;
            let spec = cat
                .find(name)
                .ok_or_else(|| {
                    anyhow!(
                        "unknown --genre {name:?} (try one of: {})",
                        cat.names().join(", ")
                    )
                })?
                .clone();
            spec.apply(params);
            Some(spec)
        }
        None => None,
    };
    Ok((look_spec, genre_spec))
}

/// End-to-end helper for CLI subcommands: applies look + genre
/// presets, runs LoRA discovery when appropriate, and prepends
/// trigger words. Mutates both `params` (sampler/prompt/negative
/// fields) and `loras` (the LoRA stack — discovery push).
///
/// Designed so that wiring a new CLI subcommand is just:
/// 1. Build a `GenerationParams` from the args, with `Option<>`
///    fields set per the "user passed?" detection trick.
/// 2. Call this helper.
/// 3. Write `params` back to the args fields.
///
/// Phase 3 wired `generate`; phase 6 reuses this for `portrait` /
/// `img2img` / `outpaint`. Network/cache logic lives in
/// [`discovery::discover_lora`]; this helper is the orchestration
/// layer above it.
pub async fn apply_presets_with_discovery(
    look_name: Option<&str>,
    genre_name: Option<&str>,
    offline: bool,
    base: discovery::BaseFamily,
    params: &mut GenerationParams,
    loras: &mut Vec<crate::pipelines::lora::LoraSpec>,
) -> Result<()> {
    let (look_spec, genre_spec) = apply_presets(look_name, genre_name, params)?;

    if let Some(l) = &look_spec {
        crate::ui::progress::println(&format!(
            "  look '{}': prompt/negative composed{}{}{}",
            l.name,
            l.steps
                .map(|s| format!(", steps={s}"))
                .unwrap_or_default(),
            l.guidance
                .map(|g| format!(", guidance={g}"))
                .unwrap_or_default(),
            l.lora_query
                .as_ref()
                .filter(|_| loras.is_empty())
                .map(|_| ", lora-discovery=pending")
                .unwrap_or_default(),
        ));
    }
    if let Some(g) = &genre_spec {
        crate::ui::progress::println(&format!(
            "  genre '{}': prompt/negative composed",
            g.name
        ));
    }

    // Discovery: only when the user hasn't supplied LoRAs.
    if !loras.is_empty() {
        return Ok(());
    }
    let query_source = look_spec
        .as_ref()
        .filter(|s| s.lora_query.is_some())
        .or(genre_spec.as_ref().filter(|s| s.lora_query.is_some()));
    let Some(spec) = query_source else {
        return Ok(());
    };
    let query = spec.lora_query.as_ref().expect("filter guarantees Some");
    let opts = discovery::DiscoveryOptions::with_defaults(offline, base);
    match discovery::discover_lora(query, &spec.name, &opts).await {
        Ok(Some(d)) => {
            crate::ui::progress::println(&format!(
                "  discovered LoRA '{}' (scale={}) for '{}'{}",
                d.model_name,
                d.spec.scale,
                spec.name,
                d.source_url
                    .as_deref()
                    .map(|u| format!(" — {u}"))
                    .unwrap_or_default(),
            ));
            if !d.trigger_words.is_empty() {
                let trigger = d.trigger_words.join(", ");
                params.prompt = crate::style::prepend_trigger(&trigger, &params.prompt);
                crate::ui::progress::println(&format!(
                    "  trigger words prepended: {trigger}"
                ));
            }
            loras.push(d.spec);
        }
        Ok(None) => {
            crate::ui::progress::println(&format!(
                "  no compatible LoRA found for '{}'{}",
                spec.name,
                if offline { " (offline)" } else { "" },
            ));
        }
        Err(e) => {
            tracing::warn!(
                target: "plakat",
                "look-discovery failed for {}: {e:#}",
                spec.name
            );
            crate::ui::progress::println(&format!(
                "  ⚠ discovery failed for '{}': {e}",
                spec.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_preset() -> PresetSpec {
        PresetSpec {
            name: "watercolor".into(),
            display_name: "Watercolor".into(),
            description: "soft washes".into(),
            prompt_prefix: Some("watercolor painting, soft washes".into()),
            prompt_suffix: Some(", on cold-pressed paper".into()),
            negative_extras: Some("photographic, oil painting".into()),
            scheduler_hint: Some("dpmpp-2m".into()),
            steps: Some(32),
            guidance: Some(6.0),
            lora_query: Some(LoraQuery {
                tags: vec!["watercolor".into()],
                keywords: vec!["watercolor".into(), "aquarelle".into()],
            }),
            base_compat: None,
        }
    }

    #[test]
    fn bundled_looks_catalog_parses() {
        let cat = Catalog::load_default(Kind::Look).expect("load bundled looks");
        assert_eq!(cat.schema_version, 1);
        assert_eq!(cat.entries.len(), 8);
        let names = cat.names();
        for must_have in [
            "ink-wash",
            "watercolor",
            "oil-painting",
            "charcoal",
            "pencil",
            "chalk-pastel",
            "linocut",
            "gouache",
        ] {
            assert!(names.contains(&must_have), "missing {must_have}");
        }
    }

    #[test]
    fn bundled_genres_catalog_parses() {
        let cat = Catalog::load_default(Kind::Genre).expect("load bundled genres");
        assert_eq!(cat.schema_version, 1);
        assert_eq!(cat.entries.len(), 1);
        assert_eq!(cat.find("anime").map(|e| e.name.as_str()), Some("anime"));
    }

    #[test]
    fn find_returns_none_for_unknown() {
        let cat = Catalog::load_default(Kind::Look).unwrap();
        assert!(cat.find("not-a-real-look").is_none());
    }

    #[test]
    fn parse_rejects_duplicate_names() {
        let json = br#"{
            "schema_version": 1,
            "looks": [
                {"name":"x","display_name":"X","description":"d",
                 "prompt_prefix":null,"prompt_suffix":null,"negative_extras":null,
                 "scheduler_hint":null,"steps":null,"guidance":null,
                 "lora_query":null,"base_compat":null},
                {"name":"x","display_name":"X2","description":"d2",
                 "prompt_prefix":null,"prompt_suffix":null,"negative_extras":null,
                 "scheduler_hint":null,"steps":null,"guidance":null,
                 "lora_query":null,"base_compat":null}
            ]
        }"#;
        let err = Catalog::parse(json, Kind::Look).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "{err}");
    }

    #[test]
    fn parse_rejects_missing_schema_version() {
        let json = br#"{"looks": []}"#;
        let err = Catalog::parse(json, Kind::Look).unwrap_err();
        assert!(err.to_string().contains("schema_version"), "{err}");
    }

    #[test]
    fn parse_rejects_missing_outer_array() {
        let json = br#"{"schema_version": 1}"#;
        let err = Catalog::parse(json, Kind::Look).unwrap_err();
        assert!(err.to_string().contains("looks"), "{err}");
    }

    /// Bucket 1: empty user fields take the preset's values.
    #[test]
    fn merge_empty_user_takes_preset() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "a cottage".into(),
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(p.steps, Some(32));
        assert_eq!(p.guidance, Some(6.0));
        assert_eq!(p.scheduler.as_deref(), Some("dpmpp-2m"));
    }

    /// Bucket 1: populated user fields are preserved verbatim.
    #[test]
    fn merge_populated_user_wins() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "a cottage".into(),
            steps: Some(50),
            guidance: Some(9.0),
            scheduler: Some("euler".into()),
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(p.steps, Some(50));
        assert_eq!(p.guidance, Some(9.0));
        assert_eq!(p.scheduler.as_deref(), Some("euler"));
    }

    /// Bucket 1: partial user — only the None fields fill.
    #[test]
    fn merge_partial_user_fills_only_unset() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "a cottage".into(),
            steps: Some(50), // user set this
            // guidance + scheduler left None
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(p.steps, Some(50));
        assert_eq!(p.guidance, Some(6.0));
        assert_eq!(p.scheduler.as_deref(), Some("dpmpp-2m"));
    }

    /// Bucket 2: prompt prefix prepends; suffix appends.
    #[test]
    fn merge_prompt_prefix_suffix() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "a cottage".into(),
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(
            p.prompt,
            "watercolor painting, soft washes, a cottage, on cold-pressed paper"
        );
    }

    /// Bucket 2: negative extras append (comma-joined).
    #[test]
    fn merge_negative_appends() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "x".into(),
            negative: "blurry, low quality".into(),
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(p.negative, "blurry, low quality, photographic, oil painting");
    }

    /// Bucket 2: empty negative — preset's extras become the value.
    #[test]
    fn merge_negative_empty_takes_extras() {
        let preset = make_preset();
        let mut p = GenerationParams {
            prompt: "x".into(),
            negative: String::new(),
            ..Default::default()
        };
        preset.apply(&mut p);
        assert_eq!(p.negative, "photographic, oil painting");
    }

    /// Bucket 2: empty prompt — prefix becomes the prompt; suffix appends.
    #[test]
    fn merge_empty_prompt_uses_prefix_then_suffix() {
        let preset = make_preset();
        let mut p = GenerationParams::default();
        preset.apply(&mut p);
        assert_eq!(
            p.prompt,
            "watercolor painting, soft washes, on cold-pressed paper"
        );
    }

    /// base_compat: None compatible with everything.
    #[test]
    fn base_compat_none_matches_all() {
        let p = make_preset();
        assert!(p.is_compatible_with("sd15"));
        assert!(p.is_compatible_with("sdxl"));
        assert!(p.is_compatible_with("flux"));
        assert!(p.is_compatible_with("sd3"));
    }

    /// base_compat: explicit allow-list.
    #[test]
    fn base_compat_restricts() {
        let mut p = make_preset();
        p.base_compat = Some(vec!["sdxl".into(), "flux".into()]);
        assert!(!p.is_compatible_with("sd15"));
        assert!(p.is_compatible_with("sdxl"));
        assert!(p.is_compatible_with("flux"));
        assert!(!p.is_compatible_with("sd3"));
    }

    /// Preset with no override fields set leaves params unchanged
    /// for those fields — the override buckets are truly optional.
    #[test]
    fn merge_preset_with_no_overrides_is_noop_on_scalar_fields() {
        let mut preset = make_preset();
        preset.scheduler_hint = None;
        preset.steps = None;
        preset.guidance = None;
        preset.prompt_prefix = None;
        preset.prompt_suffix = None;
        preset.negative_extras = None;

        let before = GenerationParams {
            prompt: "x".into(),
            negative: "y".into(),
            steps: Some(11),
            guidance: Some(2.5),
            scheduler: Some("custom".into()),
        };
        let mut after = before.clone();
        preset.apply(&mut after);
        assert_eq!(after, before);
    }

    /// Genre catalog applies through the same `apply()` path.
    #[test]
    fn genre_preset_applies_same_as_look() {
        let cat = Catalog::load_default(Kind::Genre).unwrap();
        let anime = cat.find("anime").expect("anime present");
        let mut p = GenerationParams {
            prompt: "a knight".into(),
            ..Default::default()
        };
        anime.apply(&mut p);
        assert_eq!(p.steps, Some(24));
        assert!(p.prompt.contains("anime"));
    }

    /// `apply_presets` end-to-end: looks up the bundled catalogs,
    /// applies both presets in order, returns the resolved specs.
    #[test]
    fn apply_presets_resolves_both() {
        let mut p = GenerationParams {
            prompt: "a knight".into(),
            ..Default::default()
        };
        let (l, g) = apply_presets(Some("watercolor"), Some("anime"), &mut p).unwrap();
        assert_eq!(l.map(|s| s.name), Some("watercolor".into()));
        assert_eq!(g.map(|s| s.name), Some("anime".into()));
        // Look (applied first) sets the override fields.
        assert_eq!(p.steps, Some(32));
        // Compositional fields from both stack.
        assert!(p.prompt.contains("watercolor"));
        assert!(p.prompt.contains("anime"));
        assert!(p.prompt.contains("a knight"));
    }

    /// `apply_presets` with neither name returns (None, None) and
    /// leaves params untouched.
    #[test]
    fn apply_presets_no_names_is_noop() {
        let before = GenerationParams {
            prompt: "x".into(),
            steps: Some(10),
            ..Default::default()
        };
        let mut after = before.clone();
        let (l, g) = apply_presets(None, None, &mut after).unwrap();
        assert!(l.is_none() && g.is_none());
        assert_eq!(after, before);
    }

    /// `apply_presets` errors on unknown look name with a helpful
    /// message listing valid names.
    #[test]
    fn apply_presets_unknown_look_errors_with_choices() {
        let mut p = GenerationParams::default();
        let err = apply_presets(Some("not-a-real-look"), None, &mut p).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown --look"), "{msg}");
        assert!(msg.contains("watercolor"), "{msg}");
    }

    /// `apply_presets` errors on unknown genre name.
    #[test]
    fn apply_presets_unknown_genre_errors() {
        let mut p = GenerationParams::default();
        let err = apply_presets(None, Some("not-a-real-genre"), &mut p).unwrap_err();
        assert!(err.to_string().contains("unknown --genre"));
    }

    /// Composing a look + a genre on the same params: the second
    /// applied wins on override fields (caller decides order); both
    /// contribute compositional fields additively.
    #[test]
    fn composing_look_and_genre() {
        let looks = Catalog::load_default(Kind::Look).unwrap();
        let genres = Catalog::load_default(Kind::Genre).unwrap();
        let watercolor = looks.find("watercolor").unwrap();
        let anime = genres.find("anime").unwrap();

        let mut p = GenerationParams {
            prompt: "a knight".into(),
            ..Default::default()
        };
        // Apply look first, then genre.
        watercolor.apply(&mut p);
        anime.apply(&mut p);

        // Override fields took look's value (anime found them Some,
        // so it skipped — matches the override-only rule).
        assert_eq!(p.steps, Some(32)); // from watercolor
        assert_eq!(p.guidance, Some(6.0)); // from watercolor
        assert_eq!(p.scheduler.as_deref(), Some("dpmpp-2m")); // from watercolor

        // Compositional fields from BOTH stack.
        assert!(p.prompt.contains("watercolor"));
        assert!(p.prompt.contains("anime"));
        assert!(p.negative.contains("photographic"));
    }
}
