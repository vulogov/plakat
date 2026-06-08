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
    /// v0.35 phase 0: PixArt-Σ family. Routes through
    /// `pipelines::pixart`. Preset discovery (look / genre /
    /// LoRA per-family) gets PixArt coverage in v0.35 phase 3.
    PixArt,
    /// v0.37 phase 0: Stable Cascade family. Routes through
    /// `pipelines::cascade`. LoRA discovery + preset coverage
    /// land in v0.38+ (LoRA is the v0.37 cycle's explicit
    /// deferral).
    StableCascade,
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
            Variant::PixArt => Self::PixArt,
            Variant::StableCascade => Self::StableCascade,
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
            Self::PixArt => "pixart",
            Self::StableCascade => "cascade",
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
            // v0.35 phase 0: Civitai PixArt LoRA discovery comes in
            // v0.35 phase 4. Conservative match for now — Civitai
            // tags PixArt LoRAs as "PixArt", "PixArt Sigma", or
            // similar; capture all of those.
            Self::PixArt => b.contains("pixart"),
            // v0.37 phase 0: Civitai Stable Cascade LoRA discovery
            // lands in v0.38+ (LoRA support is the v0.37 cycle's
            // explicit deferral). Conservative substring match.
            Self::StableCascade => b.contains("cascade"),
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
/// and the cache roots.
#[derive(Debug, Clone)]
pub struct DiscoveryOptions {
    pub offline: bool,
    pub base: BaseFamily,
    /// Discovery cache directory. Defaults via
    /// [`default_cache_root`] when constructed via
    /// [`DiscoveryOptions::with_defaults`].
    pub cache_root: PathBuf,
    /// Civitai download cache root — scanned by [`try_local_scan`]
    /// for already-downloaded LoRAs. Defaults via
    /// [`crate::civitai::download::cache_root`]; tests override.
    pub civitai_cache_root: PathBuf,
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
            civitai_cache_root: crate::civitai::download::cache_root(),
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
                source: LoraSource::Local(path.clone()),
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

/// HF Hub's model-search response shape (we only deserialize what
/// we use — `full=true` adds `tags`).
#[derive(Debug, Deserialize)]
struct HfSearchEntry {
    id: String,
    #[serde(default)]
    #[allow(dead_code)] // exposed for future ranking work
    likes: u64,
    #[serde(default)]
    #[allow(dead_code)]
    downloads: u64,
    #[serde(default)]
    tags: Vec<String>,
}

/// Pattern-match a HuggingFace repo's `id` and tag list against a
/// [`BaseFamily`]. HF doesn't standardize a per-LoRA `base_model`
/// field accessible at search time, so we rely on naming
/// conventions + the model card tags.
fn hf_repo_matches_base(repo_id: &str, tags: &[String], base: BaseFamily) -> bool {
    let id_l = repo_id.to_lowercase();
    let id_check = match base {
        BaseFamily::Sd15 => {
            id_l.contains("sd15")
                || id_l.contains("sd-1.5")
                || id_l.contains("sd_1.5")
                || id_l.contains("v1-5")
                || id_l.contains("v1.5")
        }
        BaseFamily::Sd21 => {
            id_l.contains("sd21")
                || id_l.contains("sd-2.1")
                || id_l.contains("v2-1")
                || id_l.contains("v2.1")
        }
        BaseFamily::Sdxl => {
            id_l.contains("sdxl")
                || id_l.contains("sd-xl")
                || id_l.contains("xl-base")
                || id_l.contains("pony")
        }
        BaseFamily::Flux => id_l.contains("flux"),
        BaseFamily::Sd3 => id_l.contains("sd3") || id_l.contains("sd-3"),
        BaseFamily::PixArt => id_l.contains("pixart"),
        BaseFamily::StableCascade => id_l.contains("cascade"),
    };
    if id_check {
        return true;
    }
    let tag_patterns: &[&str] = match base {
        BaseFamily::Sd15 => &["sd15", "stable-diffusion-1.5", "stable-diffusion-v1-5", "sd-1.5"],
        BaseFamily::Sd21 => &["sd21", "stable-diffusion-2.1", "stable-diffusion-2", "sd-2.1"],
        BaseFamily::Sdxl => &["sdxl", "stable-diffusion-xl"],
        BaseFamily::Flux => &["flux", "flux-dev", "flux-schnell", "flux.1"],
        BaseFamily::Sd3 => &["sd3", "stable-diffusion-3"],
        BaseFamily::PixArt => &["pixart", "pixart-sigma", "pixart-alpha"],
        BaseFamily::StableCascade => &["stable-cascade", "cascade"],
    };
    tags.iter().any(|t| {
        let tl = t.to_lowercase();
        tag_patterns.iter().any(|p| tl.contains(p))
    })
}

/// HF Hub search fallback. Lower-quality metadata than Civitai
/// (no trigger words, no reliable per-LoRA base-model field), but
/// fills the gap when Civitai has no match — e.g., for art mediums
/// where HF hosts the canonical LoRA.
async fn try_hf_hub(
    query: &LoraQuery,
    base: BaseFamily,
    scale: f32,
) -> Result<Option<DiscoveredLora>> {
    let q = build_query_string(query);
    if q.is_empty() {
        return Ok(None);
    }
    // Append "lora" to the search string so generic medium words
    // ("watercolor") don't return base models / pipelines.
    let q_with_lora = format!("{q} lora");
    let limit = "20";
    let url = reqwest::Url::parse_with_params(
        "https://huggingface.co/api/models",
        &[
            ("search", q_with_lora.as_str()),
            ("filter", "lora"),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", limit),
            ("full", "true"),
        ],
    )
    .context("building HF Hub search URL")?;

    let client = reqwest::Client::builder()
        .user_agent(concat!("plakat/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("building HF Hub client")?;
    let resp = client
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("HF Hub search GET {url}"))?;
    if !resp.status().is_success() {
        // HF returning 429 / 5xx isn't fatal — discovery chains to
        // the next source. Log + soft-fail.
        tracing::debug!(
            target: "plakat",
            "HF Hub search returned {} for {q_with_lora:?}",
            resp.status()
        );
        return Ok(None);
    }
    let items: Vec<HfSearchEntry> = resp
        .json()
        .await
        .context("parsing HF Hub search response")?;
    for item in items {
        if !hf_repo_matches_base(&item.id, &item.tags, base) {
            continue;
        }
        let spec = LoraSpec {
            source: LoraSource::Hub {
                repo: item.id.clone(),
                file: None,
                revision: None,
            },
            scale,
        };
        return Ok(Some(DiscoveredLora {
            spec,
            // HF doesn't expose trigger words at search time. The
            // model card's README may have them, but parsing it for
            // every search is too chatty for v0.25. Caller skips
            // the trigger-prepend step when this vec is empty.
            trigger_words: vec![],
            source: Source::HuggingFace {
                repo: item.id.clone(),
            },
            model_name: item.id.clone(),
            source_url: Some(format!("https://huggingface.co/{}", item.id)),
        }));
    }
    Ok(None)
}

/// Local-cache scan. Walks `civitai_cache_root` (plakat's own
/// download cache) and reads each `metadata.json` to find an
/// already-downloaded LoRA whose base + trigger words match.
///
/// Useful for two cases:
/// 1. **Offline mode**: when network is unavailable, we still want
///    a previously discovered LoRA to be usable.
/// 2. **User pre-pulled**: if a user did `plakat civitai download`
///    for a watercolor LoRA, a later `--look watercolor` invocation
///    can pick that up automatically.
fn try_local_scan(
    query: &LoraQuery,
    base: BaseFamily,
    scale: f32,
    civitai_cache_root: &std::path::Path,
) -> Result<Option<DiscoveredLora>> {
    if !civitai_cache_root.exists() {
        return Ok(None);
    }
    let keywords: Vec<String> = query
        .keywords
        .iter()
        .chain(query.tags.iter())
        .filter(|k| !k.is_empty())
        .map(|s| s.to_lowercase())
        .collect();
    if keywords.is_empty() {
        return Ok(None);
    }

    let read = match fs::read_dir(civitai_cache_root) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    for model_entry in read.flatten() {
        let model_path = model_entry.path();
        if !model_path.is_dir() {
            continue;
        }
        let Some(model_name) = model_path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(model_id_str) = model_name.strip_prefix("model-") else {
            continue;
        };
        let Ok(model_id) = model_id_str.parse::<u64>() else {
            continue;
        };

        let Ok(version_read) = fs::read_dir(&model_path) else {
            continue;
        };
        for version_entry in version_read.flatten() {
            let version_path = version_entry.path();
            if !version_path.is_dir() {
                continue;
            }
            let Some(version_name) = version_path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(version_id_str) = version_name.strip_prefix("version-") else {
                continue;
            };
            let Ok(version_id) = version_id_str.parse::<u64>() else {
                continue;
            };

            let meta_path = version_path.join("metadata.json");
            let Ok(bytes) = fs::read(&meta_path) else {
                continue;
            };
            let version: api::ModelVersion = match serde_json::from_slice(&bytes) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Base compatibility.
            let Some(bm) = version.base_model.as_deref() else {
                continue;
            };
            if !base.civitai_matches(bm) {
                continue;
            }

            // Keyword match: trigger words OR safetensors filename.
            let triggers_lower: Vec<String> = version
                .trained_words
                .iter()
                .map(|s| s.to_lowercase())
                .collect();
            let safetensors_lower: Option<String> = fs::read_dir(&version_path)
                .ok()
                .and_then(|r| {
                    r.flatten()
                        .filter_map(|e| {
                            let n = e.file_name().to_string_lossy().to_lowercase();
                            n.ends_with(".safetensors").then_some(n)
                        })
                        .next()
                });
            let model_name_lower = version.name.to_lowercase();
            let matches_keyword = keywords.iter().any(|k| {
                triggers_lower.iter().any(|t| t.contains(k))
                    || safetensors_lower
                        .as_deref()
                        .map(|f| f.contains(k))
                        .unwrap_or(false)
                    || model_name_lower.contains(k)
            });
            if !matches_keyword {
                continue;
            }

            // Hit — attribute to Civitai (LoraSpec resolution will
            // see the cached file and short-circuit the network).
            let spec = LoraSpec {
                source: LoraSource::Civitai {
                    id_kind: CivitaiIdKind::Version(version_id),
                    file: None,
                },
                scale,
            };
            return Ok(Some(DiscoveredLora {
                spec,
                trigger_words: version.trained_words.clone(),
                source: Source::Civitai {
                    model_id,
                    version_id,
                },
                model_name: format!("(local) {}", version.name),
                source_url: Some(format!("https://civitai.com/models/{model_id}")),
            }));
        }
    }
    Ok(None)
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
/// Source order:
/// 1. Discovery cache (on-disk JSON keyed by preset+base).
/// 2. **If `offline`**: local-cache scan only (civitai download cache).
/// 3. **Else**: Civitai → HuggingFace Hub → local-cache scan.
///
/// Returns `Ok(Some(_))` when a compatible LoRA was found,
/// `Ok(None)` when every source missed, `Err` on a hard error
/// (cache I/O failure, malformed API response). Network 4xx/5xx
/// from any single source are downgraded to "no match" and chain
/// to the next source.
pub async fn discover_lora(
    query: &LoraQuery,
    preset_name: &str,
    options: &DiscoveryOptions,
) -> Result<Option<DiscoveredLora>> {
    // 1. Cache check (cheap, always tried).
    if let Some(cached) = read_cache(options, preset_name) {
        // Offline can't download: a cached *remote* (Civitai/HF) spec is
        // only usable if its file is already on disk. A prior run can cache
        // the discovery *spec* without ever completing the file download —
        // returning it here would make the offline path hit the network and
        // time out. So when offline + remote, skip the cached spec and fall
        // through to the file-verified local scan (§2); a true miss returns
        // None and the look falls back to its prompt preset.
        let remote = matches!(
            cached.source,
            Source::Civitai { .. } | Source::HuggingFace { .. }
        );
        if !(options.offline && remote) {
            tracing::debug!(
                target: "plakat",
                "look-discovery cache hit for {preset_name}/{}",
                options.base.cache_slug()
            );
            return Ok(Some(cached.to_discovered(options.scale)));
        }
    }

    // 2. Offline short-circuit: local-cache scan only, no network.
    if options.offline {
        tracing::debug!(
            target: "plakat",
            "look-discovery offline path for {preset_name}/{}",
            options.base.cache_slug()
        );
        let result = try_local_scan(
            query,
            options.base,
            options.scale,
            &options.civitai_cache_root,
        )?;
        if let Some(d) = &result {
            write_cache(options, preset_name, d);
        }
        return Ok(result);
    }

    // 3. Online chain: Civitai → HF Hub → local-cache scan.
    if let Some(d) = try_civitai(query, options.base, options.scale).await? {
        write_cache(options, preset_name, &d);
        return Ok(Some(d));
    }
    tracing::debug!(
        target: "plakat",
        "look-discovery Civitai miss for {preset_name} — falling through to HF Hub"
    );

    if let Some(d) = try_hf_hub(query, options.base, options.scale).await? {
        write_cache(options, preset_name, &d);
        return Ok(Some(d));
    }
    tracing::debug!(
        target: "plakat",
        "look-discovery HF Hub miss for {preset_name} — falling through to local scan"
    );

    if let Some(d) = try_local_scan(
        query,
        options.base,
        options.scale,
        &options.civitai_cache_root,
    )? {
        write_cache(options, preset_name, &d);
        return Ok(Some(d));
    }

    Ok(None)
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
            cache_root: dir.path().join("discovery"),
            // Point the civitai-cache-scan root inside the same
            // tempdir so tests don't accidentally touch the user's
            // real download cache. Default to a non-existent path
            // (try_local_scan early-returns None).
            civitai_cache_root: dir.path().join("civitai-empty"),
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
            civitai_cache_root: dir.path().to_path_buf(),
            scale: 0.8,
        };
        let sd15 = cache_path(&make(BaseFamily::Sd15), "watercolor");
        let sdxl = cache_path(&make(BaseFamily::Sdxl), "watercolor");
        let flux = cache_path(&make(BaseFamily::Flux), "watercolor");
        assert_ne!(sd15, sdxl);
        assert_ne!(sdxl, flux);
        assert_ne!(sd15, flux);
    }

    // --- HF Hub repo matching ---

    #[test]
    fn hf_repo_matches_sd15_by_id() {
        assert!(hf_repo_matches_base(
            "ostris/watercolor-sd-1.5",
            &[],
            BaseFamily::Sd15
        ));
        assert!(hf_repo_matches_base("user/awesome-sd15-style", &[], BaseFamily::Sd15));
        assert!(!hf_repo_matches_base("user/awesome-sdxl-style", &[], BaseFamily::Sd15));
    }

    #[test]
    fn hf_repo_matches_sdxl_by_id() {
        assert!(hf_repo_matches_base("user/watercolor-sdxl", &[], BaseFamily::Sdxl));
        assert!(hf_repo_matches_base("user/sd-xl-style", &[], BaseFamily::Sdxl));
        assert!(hf_repo_matches_base("user/pony-style", &[], BaseFamily::Sdxl));
        assert!(!hf_repo_matches_base("user/watercolor-flux", &[], BaseFamily::Sdxl));
    }

    #[test]
    fn hf_repo_matches_flux_by_id() {
        assert!(hf_repo_matches_base("strangerzonehf/flux-style", &[], BaseFamily::Flux));
        assert!(!hf_repo_matches_base("user/watercolor-sdxl", &[], BaseFamily::Flux));
    }

    #[test]
    fn hf_repo_matches_by_tag_when_id_silent() {
        // Repo id doesn't say "sdxl"; tags do.
        let tags = vec!["stable-diffusion-xl".to_string(), "lora".to_string()];
        assert!(hf_repo_matches_base("user/cool-painting", &tags, BaseFamily::Sdxl));
        // Same repo id, wrong tag set → no match.
        let tags_sd15 = vec!["stable-diffusion-v1-5".to_string()];
        assert!(hf_repo_matches_base("user/cool-painting", &tags_sd15, BaseFamily::Sd15));
        assert!(!hf_repo_matches_base("user/cool-painting", &tags_sd15, BaseFamily::Sdxl));
    }

    // --- Local-cache scan ---

    /// Helper: drop a fake Civitai cache entry at
    /// `root/model-M/version-V/{file.safetensors, metadata.json}`.
    fn make_fake_civitai_entry(
        root: &std::path::Path,
        model_id: u64,
        version_id: u64,
        base_model: &str,
        trained_words: &[&str],
        filename: &str,
    ) {
        let dir = root
            .join(format!("model-{model_id}"))
            .join(format!("version-{version_id}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(filename), b"fake-safetensors-bytes").unwrap();
        let version = api::ModelVersion {
            id: version_id,
            name: format!("v{version_id}"),
            base_model: Some(base_model.into()),
            trained_words: trained_words.iter().map(|s| s.to_string()).collect(),
            download_url: None,
            files: vec![],
        };
        std::fs::write(
            dir.join("metadata.json"),
            serde_json::to_vec_pretty(&version).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn local_scan_finds_by_trigger_word() {
        let dir = tempfile::tempdir().unwrap();
        make_fake_civitai_entry(
            dir.path(),
            42,
            7,
            "SD 1.5",
            &["watercolor wash", "soft"],
            "wc.safetensors",
        );
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["watercolor".into()],
        };
        let result = try_local_scan(&q, BaseFamily::Sd15, 0.8, dir.path())
            .unwrap()
            .expect("local-scan hit");
        match result.source {
            Source::Civitai { model_id, version_id } => {
                assert_eq!(model_id, 42);
                assert_eq!(version_id, 7);
            }
            other => panic!("expected Civitai, got {other:?}"),
        }
        assert_eq!(result.trigger_words, vec!["watercolor wash", "soft"]);
    }

    #[test]
    fn local_scan_finds_by_filename() {
        let dir = tempfile::tempdir().unwrap();
        make_fake_civitai_entry(
            dir.path(),
            42,
            7,
            "SDXL 1.0",
            &[], // no trigger words
            "oil-painting-style.safetensors",
        );
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["oil".into()],
        };
        let result = try_local_scan(&q, BaseFamily::Sdxl, 0.8, dir.path())
            .unwrap()
            .expect("local-scan hit");
        match result.source {
            Source::Civitai { version_id, .. } => assert_eq!(version_id, 7),
            other => panic!("expected Civitai, got {other:?}"),
        }
    }

    #[test]
    fn local_scan_filters_by_base() {
        let dir = tempfile::tempdir().unwrap();
        // Two LoRAs match the keyword but only one matches the base.
        make_fake_civitai_entry(
            dir.path(),
            1,
            100,
            "SD 1.5",
            &["watercolor"],
            "a.safetensors",
        );
        make_fake_civitai_entry(
            dir.path(),
            2,
            200,
            "Flux.1 D",
            &["watercolor"],
            "b.safetensors",
        );
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["watercolor".into()],
        };
        let flux = try_local_scan(&q, BaseFamily::Flux, 0.8, dir.path())
            .unwrap()
            .expect("flux hit");
        match flux.source {
            Source::Civitai { model_id, .. } => assert_eq!(model_id, 2),
            other => panic!("got {other:?}"),
        }
        let sd15 = try_local_scan(&q, BaseFamily::Sd15, 0.8, dir.path())
            .unwrap()
            .expect("sd15 hit");
        match sd15.source {
            Source::Civitai { model_id, .. } => assert_eq!(model_id, 1),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn local_scan_misses_on_nonexistent_root() {
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["watercolor".into()],
        };
        let result = try_local_scan(
            &q,
            BaseFamily::Sd15,
            0.8,
            std::path::Path::new("/nonexistent-discovery-root-xyz123"),
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn local_scan_misses_when_no_keyword_match() {
        let dir = tempfile::tempdir().unwrap();
        make_fake_civitai_entry(
            dir.path(),
            1,
            100,
            "SD 1.5",
            &["cyberpunk"],
            "x.safetensors",
        );
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["watercolor".into()],
        };
        assert!(try_local_scan(&q, BaseFamily::Sd15, 0.8, dir.path()).unwrap().is_none());
    }

    #[test]
    fn local_scan_misses_when_no_keywords() {
        let dir = tempfile::tempdir().unwrap();
        make_fake_civitai_entry(
            dir.path(),
            1,
            100,
            "SD 1.5",
            &["watercolor"],
            "x.safetensors",
        );
        let q = LoraQuery {
            tags: vec![],
            keywords: vec![],
        };
        // Empty query → no scan.
        assert!(try_local_scan(&q, BaseFamily::Sd15, 0.8, dir.path()).unwrap().is_none());
    }

    /// Offline + local-scan hit: full chain end-to-end with no
    /// network. Also writes a cache entry so the next call is even
    /// faster.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn offline_local_scan_promotes_to_cache() {
        let dir = tempfile::tempdir().unwrap();
        let civitai_dir = dir.path().join("civitai");
        make_fake_civitai_entry(
            &civitai_dir,
            42,
            7,
            "SDXL 1.0",
            &["oil painting"],
            "oil.safetensors",
        );
        let opts = DiscoveryOptions {
            offline: true,
            base: BaseFamily::Sdxl,
            cache_root: dir.path().join("discovery"),
            civitai_cache_root: civitai_dir,
            scale: 0.8,
        };
        let q = LoraQuery {
            tags: vec![],
            keywords: vec!["oil".into()],
        };
        let result = discover_lora(&q, "oil-painting", &opts)
            .await
            .unwrap()
            .expect("local-scan hit");
        match result.source {
            Source::Civitai { model_id, .. } => assert_eq!(model_id, 42),
            other => panic!("got {other:?}"),
        }
        // Verify the cache got written.
        assert!(read_cache(&opts, "oil-painting").is_some());
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

    /// Offline + a cached *remote* (Civitai/HF) spec whose file isn't on
    /// disk must NOT be returned — resolving it would hit the network and
    /// time out. It falls through to the local-cache scan (empty here) →
    /// None, so the look uses its prompt preset instead of crashing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn offline_remote_cache_without_file_returns_none() {
        let (_dir, mut opts) = tmp_options(BaseFamily::Sd15);
        opts.offline = true;
        write_cache(&opts, "watercolor", &fake_discovered());

        let q = LoraQuery {
            tags: vec!["watercolor".into()],
            keywords: vec!["watercolor".into()],
        };
        let result = discover_lora(&q, "watercolor", &opts).await.unwrap();
        assert!(
            result.is_none(),
            "offline + remote cached spec without a local file must return None"
        );
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
