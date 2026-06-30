//! File downloader for Civitai assets. Streams the response body
//! into a temp file alongside the target, then atomic-renames on
//! success so a Ctrl-C mid-download doesn't leave a half-written
//! `.safetensors` in the cache that the model loader would mmap into
//! the void.
//!
//! Cache layout (under `<plakat-cache>/civitai/`):
//!
//! ```text
//! <plakat-cache>/civitai/
//! ├── model-<modelId>/
//! │   └── version-<versionId>/
//! │       ├── <filename>.safetensors
//! │       └── metadata.json    ← serialized ModelVersion
//! ```
//!
//! Re-downloads short-circuit when the file already exists at the
//! expected path AND the on-disk size matches `sizeKB`. Mismatched
//! sizes re-download (corruption recovery).

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::api::{Model, ModelVersion, VersionFile, get_model, get_version};

const USER_AGENT: &str = concat!("plakat/", env!("CARGO_PKG_VERSION"));

/// Root of the Civitai cache. Sibling to the HF cache so the
/// existing `--cache-dir` flag controls both.
pub fn cache_root() -> PathBuf {
    let hf = crate::hf::cache::hf_cache_root();
    // hf_cache_root() returns the *hub* dir for HF — go up one to
    // get the shared root, then drop into `civitai/`. When the user
    // sets `--cache-dir`, that becomes the root and we land inside.
    let parent = hf.parent().unwrap_or(&hf);
    parent.join("civitai")
}

/// Compute the on-disk path for one version's file. Doesn't create
/// any directories — call [`ensure_dir`] before writing.
pub fn version_file_path(model_id: u64, version_id: u64, filename: &str) -> PathBuf {
    cache_root()
        .join(format!("model-{model_id}"))
        .join(format!("version-{version_id}"))
        .join(filename)
}

fn ensure_dir(p: &Path) -> Result<()> {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating Civitai cache dir {}", parent.display()))?;
    }
    Ok(())
}

/// v0.20: surface the LoRA's trigger words at resolve time. Logged
/// to stdout via the `ui::progress` channel (same surface as
/// download progress messages) so users see it inline with the
/// rest of the generate-flow output. Logged for both cache hits
/// and fresh downloads — Civitai LoRAs almost always need
/// trigger phrases in the prompt to activate properly, and silent
/// LoRAs (no apparent effect) are a top user-friction signal.
///
/// Empty `trained_words` (LoRA has no triggers; uncommon but
/// happens with style LoRAs that activate purely from scale) is a
/// silent no-op rather than a warning.
fn log_trigger_words(model_id: u64, version_id: u64, trained_words: &[String]) {
    for line in format_trigger_lines(model_id, version_id, trained_words) {
        crate::ui::progress::println(&line);
    }
}

/// Pure helper for [`log_trigger_words`] — returns the lines that
/// would be printed. Empty when the LoRA has no trigger words, so
/// the caller naturally emits nothing. Split out so the
/// formatting logic is unit-testable without capturing stdout.
fn format_trigger_lines(
    model_id: u64,
    version_id: u64,
    trained_words: &[String],
) -> Vec<String> {
    // Filter blanks defensively: Civitai sometimes stores empty
    // strings in `trainedWords` when an author leaves placeholders.
    // We don't want to show "trigger words: , , ,".
    let words: Vec<&str> = trained_words
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if words.is_empty() {
        return Vec::new();
    }
    let formatted = words.join(", ");
    vec![
        format!(
            "  ✦ Civitai LoRA {model_id} (v{version_id}) trigger words: {formatted}"
        ),
        "    → consider adding these to your prompt for the LoRA to activate"
            .to_string(),
    ]
}

/// Result of one download.
pub struct DownloadResult {
    /// Local path the file ended up at.
    pub path: PathBuf,
    /// Total bytes written. Zero on cache hit.
    pub bytes_written: u64,
    /// `true` when the file was already cached at the expected size
    /// and we skipped the network entirely.
    pub cache_hit: bool,
}

/// Download a Civitai asset by `(model_id, version_id)`. Either or
/// both can be `None`:
///
/// * `(Some(m), None)` — fetch model, use first version.
/// * `(None, Some(v))` — fetch version directly; model_id is
///   discovered by re-querying.
/// * `(Some(m), Some(v))` — use both; cheapest path (one API call).
///
/// `file_name` is optional — when absent, picks the primary file
/// (Civitai marks one per version) or the first if none are primary.
pub async fn download_version(
    model_id: Option<u64>,
    version_id: Option<u64>,
    file_name: Option<&str>,
) -> Result<DownloadResult> {
    let (version, resolved_model_id) = resolve_version(model_id, version_id).await?;
    // v0.20: surface the LoRA's trained trigger words so users
    // know what to put in their prompt. Civitai LoRA cards almost
    // always list one or more "trigger phrases" (the tokens the
    // LoRA was trained to respond to); without them in the prompt
    // the LoRA's effect is muted at best. Logged unconditionally
    // (cache hit + fresh download) so the info surfaces every run.
    log_trigger_words(resolved_model_id, version.id, &version.trained_words);
    let file = pick_file(&version, file_name)?;
    let target = version_file_path(resolved_model_id, version.id, &file.name);

    // Cache hit check: file present AND on-disk size matches the
    // version's reported `sizeKB`. Civitai stores it as a float
    // (kilobytes), so we compare against the byte form with a small
    // tolerance for rounding.
    if target.exists() {
        if let Ok(meta) = std::fs::metadata(&target) {
            let expected_bytes = (file.size_kb * 1024.0) as u64;
            let actual = meta.len();
            // Allow ±2 KB tolerance for the rounding on Civitai's
            // side. Mismatches beyond that re-download.
            let diff = expected_bytes.abs_diff(actual);
            if diff < 2048 {
                tracing::debug!(
                    target: "plakat",
                    "Civitai cache hit: {} ({} bytes)",
                    target.display(),
                    actual
                );
                return Ok(DownloadResult {
                    path: target,
                    bytes_written: 0,
                    cache_hit: true,
                });
            }
            tracing::warn!(
                target: "plakat",
                "Civitai cache size mismatch for {}: on-disk {} vs expected {} \
                 — re-downloading.",
                target.display(),
                actual,
                expected_bytes
            );
        }
    }

    ensure_dir(&target)?;
    // Persist metadata next to the file for later inspection /
    // `--lora` resolution. Best-effort: a failure here shouldn't
    // block the download.
    if let Some(parent) = target.parent() {
        let meta_path = parent.join("metadata.json");
        if let Ok(json) = serde_json::to_string_pretty(&version) {
            let _ = std::fs::write(&meta_path, json);
        }
    }

    // Download via the version-level endpoint — redirects to the
    // primary file by default; for non-primary files we send the
    // per-file URL directly.
    let url = if file.primary || file_name.is_none() {
        format!("https://civitai.com/api/download/models/{}", version.id)
    } else {
        file.download_url
            .clone()
            .ok_or_else(|| anyhow::anyhow!(
                "Civitai file {:?} has no downloadUrl — try primary file instead",
                file.name
            ))?
    };

    let bytes_written = stream_to_file(&url, &target).await
        .with_context(|| format!("downloading Civitai file from {url}"))?;

    // SHA-256 integrity check: Civitai publishes the expected hash per file. A mismatch
    // means a corrupt / truncated / tampered download — delete it so a retry re-fetches
    // rather than leaving a bad .safetensors the loader would mmap. No published hash →
    // skip (size was already checked).
    if let Some(expected) = file.hashes.sha256.as_deref().filter(|h| !h.is_empty()) {
        verify_sha256(&target, expected).inspect_err(|_| {
            let _ = std::fs::remove_file(&target);
        })?;
    }

    Ok(DownloadResult {
        path: target,
        bytes_written,
        cache_hit: false,
    })
}

/// Verify a file's SHA-256 against the (hex, case-insensitive) `expected` digest. Reads
/// the file in chunks so a multi-GB checkpoint doesn't load into memory.
pub fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {} to verify", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| format!("hashing {}", path.display()))?;
    let got = hasher.finalize();
    let got_hex = got.iter().map(|b| format!("{b:02x}")).collect::<String>();
    if !got_hex.eq_ignore_ascii_case(expected.trim()) {
        bail!(
            "SHA-256 mismatch for {}: expected {}, got {} — download corrupt, deleted",
            path.display(),
            expected.trim().to_lowercase(),
            got_hex
        );
    }
    tracing::debug!(target: "plakat", "SHA-256 verified for {}", path.display());
    Ok(())
}

/// Resolve `(model_id, version_id)` into a concrete `ModelVersion`
/// + the model_id it belongs to. Handles all three (Some/Some,
/// Some/None, None/Some) input shapes.
async fn resolve_version(
    model_id: Option<u64>,
    version_id: Option<u64>,
) -> Result<(ModelVersion, u64)> {
    match (model_id, version_id) {
        (_, Some(v)) => {
            // Version ID lookup is cheapest — one API call. We don't
            // know the parent model ID from the version response
            // alone, so fall back to caller-supplied model_id or "0"
            // (unknown — used only as a cache-bucket; we prefer the
            // real ID for clarity).
            let ver = get_version(v).await?;
            // Walk back from version → model to get the parent ID
            // if the caller didn't supply it. This costs one extra
            // API call but only on `--ref civitai-version-id` paths
            // where we'd otherwise bucket the cache under model-0.
            let mid = if let Some(m) = model_id {
                m
            } else {
                find_model_for_version(v).await.unwrap_or(0)
            };
            Ok((ver, mid))
        }
        (Some(m), None) => {
            let model = get_model(m).await?;
            let ver = pick_version(&model)?.clone();
            Ok((ver, m))
        }
        (None, None) => bail!("download needs at least one of model_id or version_id"),
    }
}

/// Best-effort: scan the model search results for one that mentions
/// `version_id` in its versions list. Used to bucket a `--ref`
/// version-only download under its parent model_id. Returns
/// `Ok(model_id)` when found.
async fn find_model_for_version(version_id: u64) -> Result<u64> {
    // The Civitai API doesn't expose a reverse lookup. As a
    // pragmatic alternative we fetch the version → its first file
    // → its downloadUrl, which embeds the version's parent model
    // ID in the redirected URL. But that's expensive + brittle.
    // Simpler: skip the lookup; the cache bucket stays "model-0"
    // which is functional. The user only sees this in the local
    // cache path and the model ID is recorded inside metadata.json.
    let _ = version_id;
    bail!("reverse model lookup not implemented — falling through to model-0 bucket")
}

/// Pick the version to download from a Model. Strategy: use the
/// first (most recent) version Civitai returns. The API sorts
/// newest-first within `modelVersions`.
fn pick_version(model: &Model) -> Result<&ModelVersion> {
    model.model_versions.first().ok_or_else(|| {
        anyhow::anyhow!(
            "Civitai model {} ({}) has no published versions",
            model.id,
            model.name
        )
    })
}

/// Pick the file to download from a Version. When `name` is set,
/// match by case-insensitive filename equality. Otherwise prefer
/// `primary: true`; fall back to the first file.
fn pick_file<'a>(
    version: &'a ModelVersion,
    name: Option<&str>,
) -> Result<&'a VersionFile> {
    if let Some(n) = name {
        let lc = n.to_lowercase();
        if let Some(f) = version.files.iter().find(|f| f.name.to_lowercase() == lc) {
            return Ok(f);
        }
        bail!(
            "Civitai version {} ({}) has no file named {n:?}. Available: [{}]",
            version.id,
            version.name,
            version.files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
    if let Some(p) = version.files.iter().find(|f| f.primary) {
        return Ok(p);
    }
    version.files.first().ok_or_else(|| {
        anyhow::anyhow!(
            "Civitai version {} ({}) has no files at all",
            version.id,
            version.name
        )
    })
}

/// Hint appended when a Civitai download *request* fails outright (connect
/// / timeout — distinct from a successful response that returns 401). The
/// download endpoint is auth-gated, and without a key the GET frequently
/// hangs instead of returning a clean 401, so a bare "operation timed out"
/// is misleading — point at the API key.
fn civitai_download_hint(has_token: bool) -> &'static str {
    if has_token {
        "the CIVITAI_API_KEY may be invalid/expired, or the network/CDN is unreachable"
    } else {
        "Civitai gates downloads behind authentication — set CIVITAI_API_KEY \
         (https://civitai.com/user/account → API Keys); without it the download \
         endpoint 401s or hangs"
    }
}

async fn stream_to_file(url: &str, target: &Path) -> Result<u64> {
    let token = std::env::var("CIVITAI_API_KEY").ok().filter(|t| !t.is_empty());
    let mut builder = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        // Downloads can be large; we want NO overall request timeout, but a
        // per-read timeout to detect a hung CDN. NOTE: `.timeout(ZERO)` does
        // NOT mean "disabled" — reqwest treats it as a 0-second total
        // timeout, so every download aborted instantly (the long-standing
        // "operation timed out"). Use `read_timeout` (resets each chunk) and
        // leave the overall timeout at its default (none).
        .pool_idle_timeout(std::time::Duration::from_secs(60))
        .read_timeout(std::time::Duration::from_secs(60));
    if let Some(t) = token.as_ref() {
        let mut headers = reqwest::header::HeaderMap::new();
        let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {t}"))
            .context("formatting CIVITAI_API_KEY into Authorization header")?;
        auth.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, auth);
        builder = builder.default_headers(headers);
    }
    let client = builder.build()?;

    // Write to a sibling `<target>.partial` and rename on success so an interrupted
    // download doesn't poison the cache slot. A leftover `.partial` from a previous run
    // is RESUMED via an HTTP Range request rather than re-downloaded from scratch.
    let tmp = target.with_extension(format!(
        "{}partial",
        target.extension().and_then(|e| e.to_str()).map(|e| format!("{e}.")).unwrap_or_default()
    ));

    // Resume loop: request from the partial's current size; the server's status decides
    // whether we append, restart, or (range unsatisfiable) wipe + retry from zero.
    let mut resume_from = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    let (resp, mut written) = loop {
        let mut req = client.get(url);
        if resume_from > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
        }
        let resp = req.send().await.map_err(|e| {
            anyhow::anyhow!(
                "Civitai download request to {url} failed: {e}. {}",
                civitai_download_hint(token.is_some())
            )
        })?;
        let code = resp.status().as_u16();
        match resume_action(code, resume_from) {
            ResumeAction::Append => break (resp, resume_from),
            ResumeAction::Restart => break (resp, 0),
            ResumeAction::WipeAndRetry => {
                // The partial is >= the served size (416) — start over cleanly.
                let _ = std::fs::remove_file(&tmp);
                resume_from = 0;
                continue;
            }
            ResumeAction::Gated => bail!(
                "Civitai download returned 401 — this asset is gated. Set CIVITAI_API_KEY \
                 from https://civitai.com/user/account → API Keys."
            ),
            ResumeAction::Fail => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_else(|_| "(no body)".to_string());
                bail!("Civitai download failed: {status}: {body}");
            }
        }
    };

    // `written == resume_from` (the bytes already on disk). Total is what's left plus
    // what we have, so the bar reflects the whole file even on a resume.
    let remaining = resp.content_length();
    let total = remaining.map(|r| r + written);
    let bar = match total {
        Some(n) => crate::ui::progress::bytes_bar(n, &format!("⤓ {}", short_name(target))),
        None => crate::ui::progress::spinner(&format!("⤓ {}", short_name(target))),
    };
    bar.set_position(written);

    // Append when resuming (written > 0), else create/truncate.
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(written > 0)
        .truncate(written == 0)
        .open(&tmp)
        .with_context(|| format!("opening temp file {}", tmp.display()))?;

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading Civitai response chunk")?;
        file.write_all(&chunk)
            .with_context(|| format!("writing to {}", tmp.display()))?;
        written += chunk.len() as u64;
        bar.set_position(written);
    }
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, target)
        .with_context(|| format!("renaming {} → {}", tmp.display(), target.display()))?;
    bar.finish_with_message(format!("✓ {}", short_name(target)));
    Ok(written)
}

/// What to do with a download response given the partial-file size we requested from.
#[derive(Debug, PartialEq, Eq)]
enum ResumeAction {
    /// 206 Partial Content — append to the partial (resume).
    Append,
    /// 200 OK — the server sent the whole file (ignored our range) — restart from zero.
    Restart,
    /// 416 Range Not Satisfiable — the partial is already ≥ the file; wipe + retry fresh.
    WipeAndRetry,
    /// 401 — gated asset.
    Gated,
    /// Any other non-success status.
    Fail,
}

/// Map an HTTP status + the byte offset we asked to resume from to a [`ResumeAction`].
/// Pure — unit-tested.
fn resume_action(status: u16, resume_from: u64) -> ResumeAction {
    match status {
        206 => ResumeAction::Append,
        200 => ResumeAction::Restart,
        416 if resume_from > 0 => ResumeAction::WipeAndRetry,
        401 => ResumeAction::Gated,
        s if (200..300).contains(&s) => ResumeAction::Restart, // other 2xx → full body
        _ => ResumeAction::Fail,
    }
}

fn short_name(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_file_path_is_under_cache_root() {
        let p = version_file_path(123, 456, "model.safetensors");
        assert!(p.ends_with("civitai/model-123/version-456/model.safetensors"));
    }

    #[test]
    fn download_hint_points_at_api_key_without_token() {
        // No token → the failure is most likely the missing API key.
        let no_token = civitai_download_hint(false);
        assert!(no_token.contains("CIVITAI_API_KEY"));
        // With a token → don't blame the (present) key's absence.
        let with_token = civitai_download_hint(true);
        assert!(!with_token.contains("set CIVITAI_API_KEY"));
    }

    #[test]
    fn resume_action_maps_statuses() {
        // Fresh download (resume_from = 0): a 200 is a full body.
        assert_eq!(resume_action(200, 0), ResumeAction::Restart);
        // Partial content → append to the partial.
        assert_eq!(resume_action(206, 1024), ResumeAction::Append);
        // Server ignored the range and sent everything → restart from zero.
        assert_eq!(resume_action(200, 1024), ResumeAction::Restart);
        // Range not satisfiable on a non-empty partial → wipe + retry.
        assert_eq!(resume_action(416, 1024), ResumeAction::WipeAndRetry);
        // 416 with nothing to resume is just a failure.
        assert_eq!(resume_action(416, 0), ResumeAction::Fail);
        assert_eq!(resume_action(401, 0), ResumeAction::Gated);
        assert_eq!(resume_action(404, 0), ResumeAction::Fail);
        assert_eq!(resume_action(500, 512), ResumeAction::Fail);
    }

    #[test]
    fn verify_sha256_accepts_the_right_hash_and_rejects_a_wrong_one() {
        let d = std::env::temp_dir().join("plakat-civitai-sha-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("blob.bin");
        std::fs::write(&f, b"hello plakat").unwrap();
        // Known SHA-256 of "hello plakat".
        let expected = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"hello plakat");
            h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        // Correct hash (and case-insensitive) passes.
        assert!(verify_sha256(&f, &expected).is_ok());
        assert!(verify_sha256(&f, &expected.to_uppercase()).is_ok());
        // A wrong hash fails with a mismatch message.
        let err = verify_sha256(&f, &"00".repeat(32)).unwrap_err().to_string();
        assert!(err.contains("SHA-256 mismatch"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn pick_file_prefers_primary_when_name_absent() {
        let v = ModelVersion {
            id: 1,
            name: "v1".into(),
            base_model: None,
            trained_words: vec![],
            download_url: None,
            files: vec![
                VersionFile {
                    id: 1,
                    name: "config.yaml".into(),
                    size_kb: 1.0,
                    download_url: None,
                    hashes: Default::default(),
                    primary: false,
                },
                VersionFile {
                    id: 2,
                    name: "model.safetensors".into(),
                    size_kb: 100.0,
                    download_url: None,
                    hashes: Default::default(),
                    primary: true,
                },
            ],
        };
        let picked = pick_file(&v, None).unwrap();
        assert_eq!(picked.name, "model.safetensors");
    }

    #[test]
    fn pick_file_falls_back_to_first_when_no_primary() {
        let v = ModelVersion {
            id: 1,
            name: "v1".into(),
            base_model: None,
            trained_words: vec![],
            download_url: None,
            files: vec![
                VersionFile {
                    id: 1,
                    name: "a.safetensors".into(),
                    size_kb: 1.0,
                    download_url: None,
                    hashes: Default::default(),
                    primary: false,
                },
                VersionFile {
                    id: 2,
                    name: "b.safetensors".into(),
                    size_kb: 2.0,
                    download_url: None,
                    hashes: Default::default(),
                    primary: false,
                },
            ],
        };
        let picked = pick_file(&v, None).unwrap();
        assert_eq!(picked.name, "a.safetensors");
    }

    #[test]
    fn pick_file_by_name_is_case_insensitive() {
        let v = ModelVersion {
            id: 1,
            name: "v1".into(),
            base_model: None,
            trained_words: vec![],
            download_url: None,
            files: vec![VersionFile {
                id: 1,
                name: "MyLora.safetensors".into(),
                size_kb: 1.0,
                download_url: None,
                hashes: Default::default(),
                primary: true,
            }],
        };
        let picked = pick_file(&v, Some("mylora.safetensors")).unwrap();
        assert_eq!(picked.name, "MyLora.safetensors");
    }

    #[test]
    fn pick_file_missing_name_lists_available() {
        let v = ModelVersion {
            id: 1,
            name: "v1".into(),
            base_model: None,
            trained_words: vec![],
            download_url: None,
            files: vec![VersionFile {
                id: 1,
                name: "the-only.safetensors".into(),
                size_kb: 1.0,
                download_url: None,
                hashes: Default::default(),
                primary: true,
            }],
        };
        let err = pick_file(&v, Some("nothere.safetensors")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("the-only.safetensors"), "got {msg}");
    }

    #[test]
    fn trigger_lines_empty_when_no_words() {
        assert!(format_trigger_lines(1, 2, &[]).is_empty());
    }

    #[test]
    fn trigger_lines_filter_blank_entries() {
        let words = vec!["".into(), "  ".into()];
        assert!(format_trigger_lines(1, 2, &words).is_empty());
    }

    #[test]
    fn trigger_lines_single_word() {
        let words = vec!["watercolor".into()];
        let lines = format_trigger_lines(123, 456, &words);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("123"), "got {}", lines[0]);
        assert!(lines[0].contains("v456"), "got {}", lines[0]);
        assert!(lines[0].contains("watercolor"), "got {}", lines[0]);
        assert!(lines[1].contains("consider adding"), "got {}", lines[1]);
    }

    #[test]
    fn trigger_lines_multiple_words_joined() {
        let words = vec!["soft pastels".into(), "watercolor".into()];
        let lines = format_trigger_lines(7, 8, &words);
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("soft pastels, watercolor"),
            "got {}",
            lines[0]
        );
    }

    #[test]
    fn trigger_lines_trim_whitespace() {
        let words = vec!["  pad  ".into(), "  ".into(), "ok".into()];
        let lines = format_trigger_lines(1, 2, &words);
        assert_eq!(lines.len(), 2);
        // " " entry filtered, "  pad  " trimmed to "pad"
        assert!(lines[0].contains("pad, ok"), "got {}", lines[0]);
    }
}
