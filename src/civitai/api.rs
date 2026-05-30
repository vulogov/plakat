//! Typed wrappers over Civitai's public REST API.
//!
//! Reference: <https://github.com/civitai/civitai/wiki/REST-API-Reference>
//!
//! Endpoints used:
//!
//! * `GET /api/v1/models?query=Q&types=T&limit=N` — search.
//! * `GET /api/v1/models/:id`                   — full model + versions.
//! * `GET /api/v1/model-versions/:id`           — single version's files.
//! * `GET /api/v1/model-versions/by-hash/:hash` — find a version by
//!   the safetensors SHA256 — useful for "what is this random file
//!   I downloaded years ago" lookups.
//!
//! All types are pragmatic — we only deserialize what we actually
//! display + use to drive the downloader. Future fields the Civitai
//! team adds (recently `nsfwLevel`, `availability`, ...) get
//! silently ignored by serde's default behaviour.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://civitai.com/api/v1";
const USER_AGENT: &str = concat!("plakat/", env!("CARGO_PKG_VERSION"));

/// Civitai asset type. Lowercase strings match the API's `types=`
/// query parameter; the API itself returns uppercase
/// (`"LORA"` / `"Checkpoint"` / ...) — we normalise both forms in
/// `Display` and `FromStr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetType {
    Checkpoint,
    Lora,
    LoCon,
    TextualInversion,
    Hypernetwork,
    AestheticGradient,
    Controlnet,
    Poses,
    VAE,
    Other,
}

impl AssetType {
    /// Civitai's canonical type token in `types=` queries.
    pub fn as_query(&self) -> &'static str {
        match self {
            Self::Checkpoint => "Checkpoint",
            Self::Lora => "LORA",
            Self::LoCon => "LoCon",
            Self::TextualInversion => "TextualInversion",
            Self::Hypernetwork => "Hypernetwork",
            Self::AestheticGradient => "AestheticGradient",
            Self::Controlnet => "Controlnet",
            Self::Poses => "Poses",
            Self::VAE => "VAE",
            Self::Other => "Other",
        }
    }
}

impl std::str::FromStr for AssetType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_lowercase().as_str() {
            "checkpoint" | "ckpt" | "model" => Self::Checkpoint,
            "lora" => Self::Lora,
            "locon" | "lycoris" => Self::LoCon,
            "textualinversion" | "ti" | "embedding" => Self::TextualInversion,
            "hypernetwork" => Self::Hypernetwork,
            "aestheticgradient" => Self::AestheticGradient,
            "controlnet" | "cn" => Self::Controlnet,
            "poses" => Self::Poses,
            "vae" => Self::VAE,
            "other" => Self::Other,
            other => bail!(
                "unknown Civitai type {other:?} (try: checkpoint | lora | locon | ti | controlnet | vae)"
            ),
        })
    }
}

impl std::fmt::Display for AssetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_query())
    }
}

/// Top-level search response.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub items: Vec<Model>,
    #[serde(default)]
    pub metadata: SearchMetadata,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchMetadata {
    #[serde(rename = "currentPage")]
    pub current_page: Option<u32>,
    #[serde(rename = "pageSize")]
    pub page_size: Option<u32>,
    #[serde(rename = "totalItems")]
    pub total_items: Option<u32>,
    #[serde(rename = "totalPages")]
    pub total_pages: Option<u32>,
    /// Cursor for the next page when the API is in cursor-paged
    /// mode (active when `query=` is set — Civitai rejects the
    /// `page=` param in that case). Empty when there are no more
    /// results.
    #[serde(rename = "nextCursor", default)]
    pub next_cursor: Option<String>,
    /// The API also surfaces `nextPage` as a fully-formed URL
    /// (cursor already embedded). We deserialize it for symmetry
    /// but the search helper drives off `next_cursor` directly.
    #[serde(rename = "nextPage", default)]
    pub next_page: Option<String>,
}

/// One model (top-level container; one model can ship many
/// versions). The fields we keep map directly to what we display in
/// the search-result table + use to drive a download.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    pub id: u64,
    pub name: String,
    /// Civitai's category token. Common: `"LORA"`, `"Checkpoint"`,
    /// `"TextualInversion"`, `"Controlnet"`, `"VAE"`.
    #[serde(rename = "type")]
    pub asset_type: String,
    #[serde(default)]
    pub nsfw: bool,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub stats: ModelStats,
    #[serde(default)]
    pub creator: Option<Creator>,
    #[serde(rename = "modelVersions", default)]
    pub model_versions: Vec<ModelVersion>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ModelStats {
    #[serde(rename = "downloadCount", default)]
    pub download_count: u64,
    #[serde(rename = "favoriteCount", default)]
    pub favorite_count: u64,
    #[serde(rename = "thumbsUpCount", default)]
    pub thumbs_up_count: u64,
    #[serde(rename = "rating", default)]
    pub rating: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Creator {
    pub username: String,
}

/// One version of a model. Each version pins specific weights, a
/// base model (e.g. "SD 1.5" / "SDXL 1.0" / "Flux.1 D"), trigger
/// words for LoRAs, and one or more downloadable files.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelVersion {
    pub id: u64,
    /// Version name as the creator labelled it (`"v1.0"`, `"epoch-12"`).
    pub name: String,
    /// Base model the version was trained against — drives the
    /// `--model` flag the user needs to pair this LoRA with.
    /// Values vary: "SD 1.5" / "SDXL 1.0" / "Flux.1 D" / "Pony" / ...
    #[serde(rename = "baseModel", default)]
    pub base_model: Option<String>,
    #[serde(rename = "trainedWords", default)]
    pub trained_words: Vec<String>,
    #[serde(rename = "downloadUrl", default)]
    pub download_url: Option<String>,
    #[serde(default)]
    pub files: Vec<VersionFile>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VersionFile {
    pub id: u64,
    pub name: String,
    #[serde(rename = "sizeKB", default)]
    pub size_kb: f64,
    /// Civitai's per-file download URL. The query string ends with
    /// `?token=...` when accessed authenticated; we use the
    /// version-level `/api/download/models/:versionId` endpoint
    /// instead, which redirects to the right file for primary
    /// downloads.
    #[serde(rename = "downloadUrl", default)]
    pub download_url: Option<String>,
    #[serde(default)]
    pub hashes: VersionFileHashes,
    /// Civitai marks one file per version as primary (the
    /// safetensors most users want). Defaults to `false` for
    /// secondary files (e.g. config.json, .yaml metadata).
    #[serde(default)]
    pub primary: bool,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct VersionFileHashes {
    #[serde(rename = "SHA256", default)]
    pub sha256: Option<String>,
    #[serde(rename = "BLAKE3", default)]
    pub blake3: Option<String>,
}

fn client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        // Civitai's CDN sometimes takes a moment to negotiate the
        // 302 to the actual file. 30s overall is plenty for the API
        // calls (downloads use a separate client with no timeout).
        .timeout(std::time::Duration::from_secs(30));
    if let Ok(token) = std::env::var("CIVITAI_API_KEY") {
        if !token.is_empty() {
            let mut headers = reqwest::header::HeaderMap::new();
            let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .context("formatting CIVITAI_API_KEY into Authorization header")?;
            auth.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, auth);
            builder = builder.default_headers(headers);
        }
    }
    Ok(builder.build()?)
}

/// Search Civitai. `query` is the search string (matches name +
/// tags + description). `asset_type` filters to one category
/// (e.g. LoRA only). `limit` caps the page size — Civitai's
/// max is 100; we clamp at 100 here too.
///
/// Pagination shape depends on whether `query` is set:
///
/// * **Browse mode** (`query` empty) — Civitai accepts the standard
///   `page=N` query parameter. `page > 1` issues one request.
/// * **Search mode** (`query` non-empty) — Civitai rejects `page=`
///   with a 400 ("Cannot use page param with query search. Use
///   cursor-based pagination."). To preserve the `--page N` UX, we
///   walk cursors `page - 1` times from page 1, then return the
///   final page's response. Each cursor walk is one HTTP round-trip
///   — fine for typical `--page 2`/`--page 3` browsing; brittle for
///   deep paging where it'd be cheaper to refine the query.
/// v0.31 phase 1 (swap): list a Civitai creator's full library by
/// `username`, walking the API's cursor chain until no more pages
/// remain. Caller passes an optional `asset_type` filter (`lora`,
/// `checkpoint`, etc.) and a per-page batch size.
///
/// Returns every matched model across all pages flattened into a
/// single `Vec`. Civitai's per-page limit caps at 100; pages of 100
/// minimise API round-trips during a sync. Honours `CIVITAI_API_KEY`
/// for higher rate limits (same Authorization header the other
/// helpers use).
pub async fn list_by_username(
    username: &str,
    asset_type: Option<AssetType>,
    page_size: u32,
) -> Result<Vec<Model>> {
    let page_size = page_size.clamp(1, 100);
    let client = client()?;
    let mut all: Vec<Model> = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut url = reqwest::Url::parse(&format!("{BASE_URL}/models"))?;
        {
            let mut q = url.query_pairs_mut();
            q.append_pair("username", username);
            q.append_pair("limit", &page_size.to_string());
            if let Some(c) = cursor.as_deref() {
                q.append_pair("cursor", c);
            }
            if let Some(t) = asset_type {
                q.append_pair("types", t.as_query());
            }
        }
        let resp = fetch_search_page(&client, &url).await?;
        let next = resp.metadata.next_cursor.clone();
        all.extend(resp.items);
        match next {
            Some(c) if !c.is_empty() => cursor = Some(c),
            _ => break,
        }
    }
    Ok(all)
}

pub async fn search(
    query: &str,
    asset_type: Option<AssetType>,
    limit: u32,
    page: u32,
) -> Result<SearchResponse> {
    let limit = limit.clamp(1, 100);
    let page = page.max(1);
    let client = client()?;

    if query.is_empty() {
        // Browse mode — page-based pagination still works.
        let url = build_search_url(query, asset_type, limit, None, Some(page))?;
        return fetch_search_page(&client, &url).await;
    }

    // Search mode — first page is page=N=1, subsequent pages walk
    // the `metadata.nextCursor` chain. Civitai's response includes
    // the cursor for whatever the *next* request needs.
    let mut url = build_search_url(query, asset_type, limit, None, None)?;
    let mut resp = fetch_search_page(&client, &url).await?;
    for _ in 1..page {
        let cursor = match resp.metadata.next_cursor.as_deref() {
            Some(c) if !c.is_empty() => c.to_string(),
            _ => {
                // Ran out of pages before reaching the requested
                // `--page`. Return the last successful page so the
                // CLI can show an empty / partial result rather than
                // erroring — matches the "no more results" UX.
                return Ok(resp);
            }
        };
        url = build_search_url(query, asset_type, limit, Some(&cursor), None)?;
        resp = fetch_search_page(&client, &url).await?;
    }
    Ok(resp)
}

/// Build the search URL with the right pagination knobs. Exactly
/// one of `cursor` / `page` should be `Some` (matches Civitai's
/// either-or constraint); both `None` means "first page, default
/// shape".
fn build_search_url(
    query: &str,
    asset_type: Option<AssetType>,
    limit: u32,
    cursor: Option<&str>,
    page: Option<u32>,
) -> Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(&format!("{BASE_URL}/models"))?;
    {
        let mut q = url.query_pairs_mut();
        if !query.is_empty() {
            q.append_pair("query", query);
        }
        q.append_pair("limit", &limit.to_string());
        if let Some(c) = cursor {
            q.append_pair("cursor", c);
        } else if let Some(p) = page {
            q.append_pair("page", &p.to_string());
        }
        if let Some(t) = asset_type {
            q.append_pair("types", t.as_query());
        }
    }
    Ok(url)
}

async fn fetch_search_page(
    client: &reqwest::Client,
    url: &reqwest::Url,
) -> Result<SearchResponse> {
    let resp = client
        .get(url.clone())
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "(no body)".to_string());
        bail!("Civitai search failed: {status}: {body}");
    }
    resp.json()
        .await
        .context("parsing Civitai search response")
}

/// Fetch one model by its top-level ID. Returns the full record
/// including every published version + every version's files.
pub async fn get_model(model_id: u64) -> Result<Model> {
    let url = format!("{BASE_URL}/models/{model_id}");
    let client = client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "(no body)".to_string());
        bail!("Civitai model fetch failed: {status}: {body}");
    }
    resp.json().await.context("parsing Civitai model response")
}

/// Fetch one model version by its ID. Use this when the user
/// paste-passes a `?modelVersionId=...` URL from the browser.
pub async fn get_version(version_id: u64) -> Result<ModelVersion> {
    let url = format!("{BASE_URL}/model-versions/{version_id}");
    let client = client()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "(no body)".to_string());
        bail!("Civitai version fetch failed: {status}: {body}");
    }
    resp.json()
        .await
        .context("parsing Civitai model-version response")
}

/// Parse a Civitai URL or bare ID into a `(model_id, version_id)`
/// pair. Accepts:
///
/// * `123456` — bare integer = model ID, no specific version.
/// * `civitai:123456` — same.
/// * `https://civitai.com/models/123456` — model URL.
/// * `https://civitai.com/models/123456?modelVersionId=789` — pinned
///   to a specific version.
/// * `https://civitai.com/api/download/models/789` — direct
///   download URL → version_id only (model unknown until we look
///   it up).
///
/// Returns `(Some(model_id), version_id)` for model URLs;
/// `(None, Some(version_id))` for the api/download form.
pub fn parse_ref(s: &str) -> Result<(Option<u64>, Option<u64>)> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        bail!("empty Civitai reference");
    }
    // Bare integer → model ID.
    if let Ok(n) = trimmed.parse::<u64>() {
        return Ok((Some(n), None));
    }
    if let Some(rest) = trimmed.strip_prefix("civitai:") {
        if let Ok(n) = rest.parse::<u64>() {
            return Ok((Some(n), None));
        }
    }
    // URL.
    if let Ok(url) = reqwest::Url::parse(trimmed) {
        let host = url.host_str().unwrap_or("");
        if !host.ends_with("civitai.com") {
            bail!(
                "expected a civitai.com URL, got host {host:?} in {trimmed:?}"
            );
        }
        let path: Vec<&str> = url.path_segments().map(|s| s.collect()).unwrap_or_default();
        // /models/<id>(/anything)?modelVersionId=<id>
        if path.first() == Some(&"models")
            && let Some(id_str) = path.get(1)
            && let Ok(model_id) = id_str.parse::<u64>()
        {
            let version_id = url
                .query_pairs()
                .find(|(k, _)| k == "modelVersionId")
                .and_then(|(_, v)| v.parse::<u64>().ok());
            return Ok((Some(model_id), version_id));
        }
        // /api/download/models/<version_id>
        if path.first() == Some(&"api")
            && path.get(1) == Some(&"download")
            && path.get(2) == Some(&"models")
            && let Some(id_str) = path.get(3)
            && let Ok(version_id) = id_str.parse::<u64>()
        {
            return Ok((None, Some(version_id)));
        }
        bail!("couldn't extract model/version IDs from URL {trimmed:?}");
    }
    bail!("not a number or civitai.com URL: {trimmed:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ref_bare_integer() {
        assert_eq!(parse_ref("12345").unwrap(), (Some(12345), None));
    }

    #[test]
    fn parse_ref_civitai_prefix() {
        assert_eq!(parse_ref("civitai:12345").unwrap(), (Some(12345), None));
    }

    #[test]
    fn parse_ref_model_url_without_version() {
        let r = parse_ref("https://civitai.com/models/12345").unwrap();
        assert_eq!(r, (Some(12345), None));
    }

    #[test]
    fn parse_ref_model_url_with_slug() {
        // Civitai URLs commonly include a slug after the ID:
        //   https://civitai.com/models/12345/my-cool-lora
        let r = parse_ref("https://civitai.com/models/12345/my-cool-lora").unwrap();
        assert_eq!(r, (Some(12345), None));
    }

    #[test]
    fn parse_ref_model_url_with_version() {
        let r = parse_ref(
            "https://civitai.com/models/12345?modelVersionId=789"
        )
        .unwrap();
        assert_eq!(r, (Some(12345), Some(789)));
    }

    #[test]
    fn parse_ref_download_url() {
        let r = parse_ref("https://civitai.com/api/download/models/789").unwrap();
        assert_eq!(r, (None, Some(789)));
    }

    #[test]
    fn parse_ref_rejects_non_civitai_host() {
        let err = parse_ref("https://huggingface.co/models/12345").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("civitai.com"), "got {msg}");
    }

    #[test]
    fn parse_ref_rejects_garbage() {
        assert!(parse_ref("not a number").is_err());
        assert!(parse_ref("").is_err());
    }

    #[test]
    fn asset_type_roundtrips_lowercase_aliases() {
        use std::str::FromStr;
        assert_eq!(AssetType::from_str("lora").unwrap(), AssetType::Lora);
        assert_eq!(AssetType::from_str("LoRA").unwrap(), AssetType::Lora);
        assert_eq!(AssetType::from_str("ti").unwrap(), AssetType::TextualInversion);
        assert_eq!(AssetType::from_str("ckpt").unwrap(), AssetType::Checkpoint);
        assert_eq!(AssetType::from_str("vae").unwrap(), AssetType::VAE);
    }

    #[test]
    fn asset_type_query_string_is_civitai_canonical() {
        assert_eq!(AssetType::Lora.as_query(), "LORA");
        assert_eq!(AssetType::Checkpoint.as_query(), "Checkpoint");
        assert_eq!(AssetType::TextualInversion.as_query(), "TextualInversion");
    }

    #[test]
    fn asset_type_unknown_bails() {
        use std::str::FromStr;
        let err = AssetType::from_str("crystal-ball").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("crystal-ball"), "got {msg}");
    }

    // Search URL builder — Civitai rejects `page=` with `query=`
    // (cursor-based pagination instead); make sure we never send
    // both, and that browse mode (no query) keeps using `page=`.

    fn query_pairs(url: &reqwest::Url) -> std::collections::HashMap<String, String> {
        url.query_pairs()
            .into_owned()
            .collect()
    }

    #[test]
    fn search_url_browse_mode_uses_page() {
        // No query → page-based pagination works.
        let url = build_search_url("", Some(AssetType::Lora), 10, None, Some(2)).unwrap();
        let q = query_pairs(&url);
        assert_eq!(q.get("page").map(String::as_str), Some("2"));
        assert!(q.get("query").is_none(), "browse mode shouldn't send query=");
        assert!(q.get("cursor").is_none());
        assert_eq!(q.get("limit").map(String::as_str), Some("10"));
        assert_eq!(q.get("types").map(String::as_str), Some("LORA"));
    }

    #[test]
    fn search_url_query_first_page_omits_page() {
        // Query + page 1 → no `page=` (Civitai rejects `page=` with
        // `query=`). The CLI default `--page 1` must not trigger the
        // 400-bad-request the old code path produced.
        let url = build_search_url("watercolor", Some(AssetType::Lora), 10, None, None).unwrap();
        let q = query_pairs(&url);
        assert!(q.get("page").is_none(), "query + first page must not send page=");
        assert!(q.get("cursor").is_none());
        assert_eq!(q.get("query").map(String::as_str), Some("watercolor"));
    }

    #[test]
    fn search_url_cursor_paging() {
        // Query + cursor walk → no `page=`, only `cursor=`.
        let url = build_search_url(
            "watercolor",
            Some(AssetType::Lora),
            10,
            Some("abc123"),
            None,
        )
        .unwrap();
        let q = query_pairs(&url);
        assert!(q.get("page").is_none());
        assert_eq!(q.get("cursor").map(String::as_str), Some("abc123"));
        assert_eq!(q.get("query").map(String::as_str), Some("watercolor"));
    }
}
