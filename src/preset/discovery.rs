//! v0.25 phase 4: automatic LoRA discovery for `--look` / `--genre`.
//!
//! Pipeline:
//!
//! 1. **Cache check** — read
//!    `$PLAKAT_CACHE_DIR/look-discovery/<name>__<base>.json`. Hit →
//!    reconstruct [`DiscoveredLora`] from the saved
//!    `(model_id, version_id, trigger_words)` tuple. The actual
//!    LoRA file caching is handled by `civitai::download` further
//!    downstream.
//!
//! 2. **Offline short-circuit** — if `options.offline` is set, cache
//!    miss returns `Ok(None)` rather than hitting the network. Phase
//!    5 layers HF Hub + local-cache scan behind this gate.
//!
//! 3. **Civitai search** — query the public REST API
//!    ([`crate::civitai::api::search`]), filter results by the LoRA's
//!    `baseModel` field against the user's pipeline family, pick the
//!    first compatible version, write the cache entry.
//!
//! 4. **HF Hub fallback / local scan** — deferred to phase 5.
//!
//! The actual file download happens later via the existing
//! `LoraSource::Civitai` resolution path
//! (`crate::pipelines::lora::LoraSpec::resolve`).
//!
//! See `Documentation/RFC_v0.25_LOOKS_AND_GENRES.md` §4.3.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::civitai::api;
use crate::pipelines::lora::{CivitaiIdKind, LoraSource, LoraSpec};
use crate::pipelines::t2i::Variant;

use super::LoraQuery;

/// Coarse pipeline family for LoRA-compatibility matching. Civitai
/// LoRAs publish their `baseModel` at this granularity (or coarser),
/// so finer plakat variants (e.g. FluxFillDev vs FluxDev) all map to
/// the same family for discovery purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseFamily {
    Sd15,
    Sd21,
    Sdxl,
    Flux,
    Sd3,
}

impl BaseFamily {
    /// Map a `--model` alias / HF repo / variant string to its family.
    /// Delegates to [`Variant::detect`] for the heavy lifting.
    pub fn from_model_arg(model: &str) -> Self {
        match Variant::detect(model) {
            Variant::Sd15 => Self::Sd15,
            Variant::Sd21 => Self::Sd21,
            Variant::Sdxl | Variant::SdxlTurbo => Self::Sdxl,
            Variant::FluxSchnell
            | Variant::FluxDev
            | Variant::FluxFillDev
            | Variant::FluxCannyDev
            | Variant::FluxDepthDev
            | Variant::FluxKontextDev => Self::Flux,
            Variant::Sd35Medium
            | Variant::Sd35Large
            | Variant::Sd35LargeTurbo
            | Variant::Sd3Medium => Self::Sd3,
        }
    }

    /// Slug used in cache filenames. Stable across releases — changing
    /// these breaks every user's cache.
    pub fn cache_slug(&self) -> &'static str {
        match self {
            Self::Sd15 => "sd15",
            Self::Sd21 => "sd21",
            Self::Sdxl => "sdxl",
            Self::Flux => "flux",
            Self::Sd3 => "sd3",
        }
    }

    /// True when the Civitai `baseModel` string indicates a LoRA is
    /// compatible with this family. Civitai's strings vary —
    /// "SD 1.5" / "SDXL 1.0" / "Flux.1 D" / "Pony" / "Illustrious" /
    /// "SD 3.5 Medium" etc. SDXL-derivative finetunes (Pony,
    /// Illustrious, NoobAI) count as SDXL-compatible since they share
    /// the LoRA weight layout.
    pub fn civitai_matches(&self, civitai_base: &str) -> bool {
        let b = civitai_base.to_lowercase();
        match self {
            Self::Sd15 => b.contains("sd 1.5") || b.contains("sd1.5") || b.contains("sd 1.4"),
            Self::Sd21 => b.contains("sd 2.0") || b.contains("sd 2.1") || b.contains("sd2.1"),
            Self::Sdxl => {
                b.contains("sdxl")
                    || b.contains("sd xl")
                    || b == "pony"
                    || b.starts_with("pony")
                    || b == "illustrious"
                    || b.starts_with("illustrious")
                    || b.starts_with("noobai")
            }
            Self::Flux => b.contains("flux"),
            Self::Sd3 => b.contains("sd 3") || b.contains("sd3") || b.contains("stable-diffusion-3"),
        }
    }
}

/// Where a discovered LoRA came from. v0.25 phase 4 ships Civitai;
/// phase 5 adds HuggingFace + LocalCache.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "source", rename_all = "lowercase")]
pub enum Source {
    Civitai { model_id: u64, version_id: u64 },
    /// Phase 5 — placeholder so the enum is forward-compatible with
    /// older cache files written before that phase lands.
    #[serde(rename = "huggingface")]
    HuggingFace { repo: String },
    /// Phase 5.
    #[serde(rename = "local")]
    LocalCache { path: PathBuf },
}

/// Result of a successful discovery call. The caller pushes
/// `spec` onto `args.loras` and (when non-empty) prepends
/// `trigger_words` onto the prompt.
#[derive(Debug, Clone)]
pub struct DiscoveredLora {
    pub spec: LoraSpec,
    pub trigger_words: Vec<String>,
    pub source: Source,
    pub model_name: String,
    /// Source URL — used for license attribution in the log line.
    pub source_url: Option<String>,
}

/// Knobs the caller passes in: offline switch, pipeline family,
/// and the cache root.
#[derive(Debug, Clone)]
pub struct DiscoveryOptions {
    pub offline: bool,
    pub base: BaseFamily,
    /// Discovery cache directory. Defaults via
    /// [`default_cache_root`] when constructed via
    /// [`DiscoveryOptions::with_defaults`].
    pub cache_root: PathBuf,
    /// LoRA scale used when constructing the [`LoraSpec`]. Curated
    /// default `0.8` — same as the v0.23 style catalog's typical
    /// LoRA scale.
    pub scale: f32,
}

impl DiscoveryOptions {
    pub fn with_defaults(offline: bool, base: BaseFamily) -> Self {
        Self {
            offline,
            base,
            cache_root: default_cache_root(),
            scale: 0.8,
        }
    }
}

/// Cache directory under the shared plakat cache root. Sibling to
/// `civitai/` and the HF hub dir so `--cache-dir` controls all
/// three.
pub fn default_cache_root() -> PathBuf {
    let hf = crate::hf::cache::hf_cache_root();
    let parent = hf.parent().unwrap_or(&hf);
    parent.join("look-discovery")
}

/// Serialized cache entry — written after a successful discovery,
/// read on subsequent calls to skip the network round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedDiscovery {
    schema_version: u32,
    source: Source,
    model_name: String,
    trigger_words: Vec<String>,
    source_url: Option<String>,
    /// UNIX epoch seconds — for future cache invalidation policy
    /// (e.g. "refresh if older than 30 days"). v0.25 phase 4 doesn't
    /// expire entries; the user can clear the cache manually.
    discovered_at: u64,
}

const CACHE_SCHEMA_VERSION: u32 = 1;

impl CachedDiscovery {
    fn from_discovered(d: &DiscoveredLora) -> Self {
        Self {
            schema_version: CACHE_SCHEMA_VERSION,
            source: d.source.clone(),
            model_name: d.model_name.clone(),
            trigger_words: d.trigger_words.clone(),
            source_url: d.source_url.clone(),
            discovered_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Rebuild a [`DiscoveredLora`] from the cached tuple. Uses
    /// `scale` from the call's options (we don't pin scale into the
    /// cache so the user can adjust it without invalidating).
    fn to_discovered(self, scale: f32) -> DiscoveredLora {
        let spec = match &self.source {
            Source::Civitai { model_id: _, version_id } => LoraSpec {
                source: LoraSource::Civitai {
                    id_kind: CivitaiIdKind::Version(*version_id),
                    file: None,
                },
                scale,
            },
            Source::HuggingFace { repo } => LoraSpec {
                source: LoraSource::Hub {
                    repo: repo.clone(),
                    file: None,
                    revision: None,
                },
                scale,
            },
            Source::LocalCache { path } => LoraSpec {
                source: LoraSource::Hub {
                    // Local paths use Hub::repo as the path
                    // (resolve() detects file:// or absolute paths).
                    repo: path.display().to_string(),
                    file: None,
                    revision: None,
                },
                scale,
            },
        };
        DiscoveredLora {
            spec,
            trigger_words: self.trigger_words,
            source: self.source,
            model_name: self.model_name,
            source_url: self.source_url,
        }
    }
}

/// Compute the cache path for a `(preset_name, base_family)` pair.
fn cache_path(opts: &DiscoveryOptions, preset_name: &str) -> PathBuf {
    // Sanitize the preset name — catalog names are kebab-case but
    // user-provided names might contain path separators or other
    // unfriendly bytes.
    let safe: String = preset_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    opts.cache_root
        .join(format!("{safe}__{}.json", opts.base.cache_slug()))
}

/// Read a cache entry if present + parseable. Errors are
/// downgraded to `None` so a corrupt cache doesn't break discovery
/// — the next call refreshes it.
fn read_cache(opts: &DiscoveryOptions, preset_name: &str) -> Option<CachedDiscovery> {
    let path = cache_path(opts, preset_name);
    let bytes = fs::read(&path).ok()?;
    let cached: CachedDiscovery = serde_json::from_slice(&bytes).ok()?;
    if cached.schema_version != CACHE_SCHEMA_VERSION {
        return None;
    }
    Some(cached)
}

/// Write a cache entry. Failure is a non-fatal warning — discovery
/// already succeeded, so a missing cache just means next run pays
/// the network round-trip again.
fn write_cache(opts: &DiscoveryOptions, preset_name: &str, d: &DiscoveredLora) {
    let path = cache_path(opts, preset_name);
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            tracing::warn!(
                target: "plakat",
                "look-discovery: failed to create cache dir {}: {e}",
                parent.display()
            );
            return;
        }
    }
    match serde_json::to_vec_pretty(&CachedDiscovery::from_discovered(d)) {
        Ok(bytes) => {
            if let Err(e) = fs::write(&path, bytes) {
                tracing::warn!(
                    target: "plakat",
                    "look-discovery: failed to write cache to {}: {e}",
                    path.display()
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                target: "plakat",
                "look-discovery: failed to serialize cache entry: {e}"
            );
        }
    }
}

/// Build the free-text query string passed to Civitai's `query=`.
/// Joins the [`LoraQuery::keywords`] with spaces — Civitai's search
/// is fuzzy + token-matched.
fn build_query_string(query: &LoraQuery) -> String {
    if !query.keywords.is_empty() {
        query.keywords.join(" ")
    } else {
        query.tags.join(" ")
    }
}

/// Pick the first compatible version across all returned models.
/// Scoring is simple — Civitai's search already ranks by relevance,
/// so we walk in order and take the first hit. Skips NSFW models
/// (per RFC §11 risk register) since v0.25 doesn't auto-download
/// NSFW.
fn pick_best_version(
    models: &[api::Model],
    base: BaseFamily,
) -> Option<(&api::Model, &api::ModelVersion)> {
    for model in models {
        if model.nsfw {
            continue;
        }
        for version in &model.model_versions {
            if let Some(bm) = &version.base_model {
                if base.civitai_matches(bm) {
                    return Some((model, version));
                }
            }
        }
    }
    None
}

/// Hit Civitai for a LoRA matching `query` on `base`. Returns
/// `Ok(None)` when no compatible LoRA shows up in the first page
/// (we don't paginate — the top-20 results from Civitai's
/// relevance-ranked search are the practical universe).
async fn try_civitai(
    query: &LoraQuery,
    base: BaseFamily,
    scale: f32,
) -> Result<Option<DiscoveredLora>> {
    let q = build_query_string(query);
    if q.is_empty() {
        return Ok(None);
    }
    let resp = api::search(&q, Some(api::AssetType::Lora), 20, 1)
        .await
        .with_context(|| format!("Civitai search for {q:?}"))?;
    let Some((model, version)) = pick_best_version(&resp.items, base) else {
        return Ok(None);
    };
    let spec = LoraSpec {
        source: LoraSource::Civitai {
            id_kind: CivitaiIdKind::Version(version.id),
            file: None,
        },
        scale,
    };
    Ok(Some(DiscoveredLora {
        spec,
        trigger_words: version.trained_words.clone(),
        source: Source::Civitai {
            model_id: model.id,
            version_id: version.id,
        },
        model_name: model.name.clone(),
        source_url: Some(format!("https://civitai.com/models/{}", model.id)),
    }))
}

/// Public entry point.
///
/// Returns `Ok(Some(_))` when a compatible LoRA was found (cache or
/// network), `Ok(None)` when no match found, `Err` on a hard
/// network / API error. Soft failures (transient HTTP, malformed
/// response) bubble up as `Err`.
pub async fn discover_lora(
    query: &LoraQuery,
    preset_name: &str,
    options: &DiscoveryOptions,
) -> Result<Option<DiscoveredLora>> {
    // 1. Cache check.
    if let Some(cached) = read_cache(options, preset_name) {
        tracing::debug!(
            target: "plakat",
            "look-discovery cache hit for {preset_name}/{}",
            options.base.cache_slug()
        );
        return Ok(Some(cached.to_discovered(options.scale)));
    }

    // 2. Offline short-circuit (phase 5 will layer local-scan here).
    if options.offline {
        tracing::debug!(
            target: "plakat",
            "look-discovery offline + no cache for {preset_name}/{} — skipping",
            options.base.cache_slug()
        );
        return Ok(None);
    }

    // 3. Civitai (phase 5 will fall through to HF Hub on None).
    let discovered = try_civitai(query, options.base, options.scale).await?;

    // 4. Cache successful results.
    if let Some(d) = &discovered {
        write_cache(options, preset_name, d);
    }

    Ok(discovered)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BaseFamily ---

    #[test]
    fn base_family_from_model_arg() {
        assert_eq!(BaseFamily::from_model_arg("sd15"), BaseFamily::Sd15);
        assert_eq!(BaseFamily::from_model_arg("sdxl"), BaseFamily::Sdxl);
        assert_eq!(BaseFamily::from_model_arg("sdxl-turbo"), BaseFamily::Sdxl);
        assert_eq!(BaseFamily::from_model_arg("flux-dev"), BaseFamily::Flux);
        assert_eq!(BaseFamily::from_model_arg("flux-schnell"), BaseFamily::Flux);
        assert_eq!(BaseFamily::from_model_arg("flux-fill-dev"), BaseFamily::Flux);
        assert_eq!(BaseFamily::from_model_arg("sd35-medium"), BaseFamily::Sd3);
        assert_eq!(BaseFamily::from_model_arg("sd35-large"), BaseFamily::Sd3);
    }

    #[test]
    fn civitai_matches_sd15() {
        assert!(BaseFamily::Sd15.civitai_matches("SD 1.5"));
        assert!(BaseFamily::Sd15.civitai_matches("sd 1.5"));
        assert!(BaseFamily::Sd15.civitai_matches("SD 1.4"));
        assert!(!BaseFamily::Sd15.civitai_matches("SDXL 1.0"));
        assert!(!BaseFamily::Sd15.civitai_matches("Flux.1 D"));
    }

    #[test]
    fn civitai_matches_sdxl() {
        assert!(BaseFamily::Sdxl.civitai_matches("SDXL 1.0"));
        assert!(BaseFamily::Sdxl.civitai_matches("SDXL Turbo"));
        assert!(BaseFamily::Sdxl.civitai_matches("SDXL Lightning"));
        // SDXL-derivatives — share the LoRA layout.
        assert!(BaseFamily::Sdxl.civitai_matches("Pony"));
        assert!(BaseFamily::Sdxl.civitai_matches("Illustrious"));
        assert!(BaseFamily::Sdxl.civitai_matches("NoobAI"));
        assert!(!BaseFamily::Sdxl.civitai_matches("SD 1.5"));
        assert!(!BaseFamily::Sdxl.civitai_matches("Flux.1 D"));
    }

    #[test]
    fn civitai_matches_flux() {
        assert!(BaseFamily::Flux.civitai_matches("Flux.1 D"));
        assert!(BaseFamily::Flux.civitai_matches("Flux.1 S"));
        assert!(BaseFamily::Flux.civitai_matches("FLUX"));
        assert!(!BaseFamily::Flux.civitai_matches("SDXL 1.0"));
        assert!(!BaseFamily::Flux.civitai_matches("SD 1.5"));
    }

    #[test]
    fn civitai_matches_sd3() {
        assert!(BaseFamily::Sd3.civitai_matches("SD 3"));
        assert!(BaseFamily::Sd3.civitai_matches("SD 3.5 Medium"));
        assert!(BaseFamily::Sd3.civitai_matches("SD 3.5 Large"));
        assert!(!BaseFamily::Sd3.civitai_matches("SDXL 1.0"));
    }

    #[test]
    fn base_family_cache_slug_stable() {
        // Don't change these without considering existing user caches.
        assert_eq!(BaseFamily::Sd15.cache_slug(), "sd15");
        assert_eq!(BaseFamily::Sd21.cache_slug(), "sd21");
        assert_eq!(BaseFamily::Sdxl.cache_slug(), "sdxl");
        assert_eq!(BaseFamily::Flux.cache_slug(), "flux");
        assert_eq!(BaseFamily::Sd3.cache_slug(), "sd3");
    }

    // --- Query string building ---

    #[test]
    fn build_query_prefers_keywords() {
        let q = LoraQuery {
            tags: vec!["watercolor".into()],
            keywords: vec!["watercolor".into(), "aquarelle".into()],
        };
        assert_eq!(build_query_string(&q), "watercolor aquarelle");
    }

    #[test]
    fn build_query_falls_back_to_tags() {
        let q = LoraQuery {
            tags: vec!["watercolor".into(), "traditional".into()],
            keywords: vec![],
        };
        assert_eq!(build_query_string(&q), "watercolor traditional");
    }

    #[test]
    fn build_query_empty() {
        let q = LoraQuery {
            tags: vec![],
            keywords: vec![],
        };
        assert_eq!(build_query_string(&q), "");
    }

    // --- Best-version picking ---

    fn mk_model(id: u64, nsfw: bool, base: &str, version_id: u64, triggers: Vec<&str>) -> api::Model {
        api::Model {
            id,
            name: format!("model-{id}"),
            asset_type: "LORA".into(),
            nsfw,
            description: None,
            tags: vec![],
            stats: api::ModelStats::default(),
            creator: None,
            model_versions: vec![api::ModelVersion {
                id: version_id,
                name: "v1".into(),
                base_model: Some(base.into()),
                trained_words: triggers.iter().map(|s| s.to_string()).collect(),
                download_url: None,
                files: vec![],
            }],
        }
    }

    #[test]
    fn pick_best_version_first_compatible() {
        let models = vec![
            mk_model(1, false, "SD 1.5", 11, vec!["sd15-trigger"]),
            mk_model(2, false, "SDXL 1.0", 22, vec!["sdxl-trigger"]),
            mk_model(3, false, "Flux.1 D", 33, vec!["flux-trigger"]),
        ];
        let pick = pick_best_version(&models, BaseFamily::Sdxl).unwrap();
        assert_eq!(pick.0.id, 2);
        assert_eq!(pick.1.id, 22);
    }

    #[test]
    fn pick_best_version_skips_nsfw() {
        let models = vec![
            mk_model(1, true, "SDXL 1.0", 11, vec![]), // nsfw — skip
            mk_model(2, false, "SDXL 1.0", 22, vec![]), // pick this
        ];
        let pick = pick_best_version(&models, BaseFamily::Sdxl).unwrap();
        assert_eq!(pick.0.id, 2);
    }

    #[test]
    fn pick_best_version_returns_none_when_no_compat() {
        let models = vec![mk_model(1, false, "SD 1.5", 11, vec![])];
        assert!(pick_best_version(&models, BaseFamily::Flux).is_none());
    }

    // --- Cache round-trip (no network) ---

    fn tmp_options(base: BaseFamily) -> (tempfile::TempDir, DiscoveryOptions) {
        let dir = tempfile::tempdir().unwrap();
        let opts = DiscoveryOptions {
            offline: false,
            base,
            cache_root: dir.path().to_path_buf(),
            scale: 0.8,
        };
        (dir, opts)
    }

    fn fake_discovered() -> DiscoveredLora {
        DiscoveredLora {
            spec: LoraSpec {
                source: LoraSource::Civitai {
                    id_kind: CivitaiIdKind::Version(789),
                    file: None,
                },
                scale: 0.8,
            },
            trigger_words: vec!["watercolor".into(), "wash".into()],
            source: Source::Civitai {
                model_id: 12345,
                version_id: 789,
            },
            model_name: "Watercolor LoRA".into(),
            source_url: Some("https://civitai.com/models/12345".into()),
        }
    }

    #[test]
    fn cache_round_trip() {
        let (_dir, opts) = tmp_options(BaseFamily::Sd15);
        let d = fake_discovered();
        write_cache(&opts, "watercolor", &d);

        let read = read_cache(&opts, "watercolor").expect("cache hit");
        assert_eq!(read.schema_version, CACHE_SCHEMA_VERSION);
        assert_eq!(read.model_name, "Watercolor LoRA");
        assert_eq!(read.trigger_words, vec!["watercolor", "wash"]);
        match read.source {
            Source::Civitai { model_id, version_id } => {
                assert_eq!(model_id, 12345);
                assert_eq!(version_id, 789);
            }
            other => panic!("expected Civitai, got {other:?}"),
        }
    }

    #[test]
    fn cache_miss_returns_none() {
        let (_dir, opts) = tmp_options(BaseFamily::Sd15);
        assert!(read_cache(&opts, "not-cached").is_none());
    }

    #[test]
    fn cache_corrupt_returns_none() {
        let (_dir, opts) = tmp_options(BaseFamily::Sd15);
        let path = cache_path(&opts, "broken");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ not valid json").unwrap();
        assert!(read_cache(&opts, "broken").is_none());
    }

    #[test]
    fn cache_wrong_schema_version_returns_none() {
        let (_dir, opts) = tmp_options(BaseFamily::Sd15);
        let path = cache_path(&opts, "old");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            br#"{"schema_version":999,"source":{"source":"civitai","model_id":1,"version_id":2},"model_name":"x","trigger_words":[],"source_url":null,"discovered_at":0}"#,
        ).unwrap();
        assert!(read_cache(&opts, "old").is_none());
    }

    #[test]
    fn cache_path_sanitizes_name() {
        let (_dir, opts) = tmp_options(BaseFamily::Sdxl);
        // Path separators + special chars get replaced with `_`.
        let p = cache_path(&opts, "evil/../name");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains('/'), "got {name}");
        assert!(!name.contains(".."), "got {name}");
        assert!(name.ends_with("__sdxl.json"));
    }

    #[test]
    fn cache_path_differs_by_base() {
        let dir = tempfile::tempdir().unwrap();
        let make = |base| DiscoveryOptions {
            offline: false,
            base,
            cache_root: dir.path().to_path_buf(),
            scale: 0.8,
        };
        let sd15 = cache_path(&make(BaseFamily::Sd15), "watercolor");
        let sdxl = cache_path(&make(BaseFamily::Sdxl), "watercolor");
        let flux = cache_path(&make(BaseFamily::Flux), "watercolor");
        assert_ne!(sd15, sdxl);
        assert_ne!(sdxl, flux);
        assert_ne!(sd15, flux);
    }

    // --- Offline behaviour (no network, no cache → None) ---

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn offline_with_no_cache_returns_none() {
        let (_dir, mut opts) = tmp_options(BaseFamily::Sdxl);
        opts.offline = true;
        let q = LoraQuery {
            tags: vec!["watercolor".into()],
            keywords: vec!["watercolor".into()],
        };
        let result = discover_lora(&q, "watercolor", &opts).await.unwrap();
        assert!(result.is_none(), "offline + no cache must return None");
    }

    /// Cache hit short-circuits even when `offline: true`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn offline_with_cache_hits() {
        let (_dir, mut opts) = tmp_options(BaseFamily::Sd15);
        opts.offline = true;
        write_cache(&opts, "watercolor", &fake_discovered());

        let q = LoraQuery {
            tags: vec!["watercolor".into()],
            keywords: vec!["watercolor".into()],
        };
        let result = discover_lora(&q, "watercolor", &opts).await.unwrap();
        let d = result.expect("cache should have hit");
        assert_eq!(d.model_name, "Watercolor LoRA");
        match d.source {
            Source::Civitai { version_id, .. } => assert_eq!(version_id, 789),
            other => panic!("expected Civitai, got {other:?}"),
        }
    }

    /// Cache hit reconstructs a usable LoraSpec pointing at the
    /// pinned Civitai version. Subsequent generate flow will
    /// download via the existing civitai::download path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn cache_hit_reconstructs_pinned_version() {
        let (_dir, opts) = tmp_options(BaseFamily::Sd15);
        write_cache(&opts, "watercolor", &fake_discovered());

        let q = LoraQuery {
            tags: vec!["x".into()],
            keywords: vec!["x".into()],
        };
        let d = discover_lora(&q, "watercolor", &opts)
            .await
            .unwrap()
            .unwrap();

        match d.spec.source {
            LoraSource::Civitai {
                id_kind: CivitaiIdKind::Version(v),
                ..
            } => assert_eq!(v, 789),
            other => panic!("expected Civitai version, got {other:?}"),
        }
        assert!((d.spec.scale - 0.8).abs() < f32::EPSILON);
    }
}
