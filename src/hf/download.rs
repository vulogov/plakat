use anyhow::{Context, Result, anyhow};
use hf_hub::api::tokio::{Api, ApiBuilder, Progress};
use hf_hub::{Cache, Repo, RepoType};
use indicatif::ProgressBar;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn api() -> Result<Api> {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }
    // Respect the user's --cache-dir flag (or PLAKAT_CACHE_DIR / HF_HOME).
    builder = builder.with_cache_dir(crate::hf::cache::hf_cache_root());
    Ok(builder.build()?)
}

fn friendly_error(repo: &str, file: &str, e: impl std::fmt::Display) -> anyhow::Error {
    let raw = e.to_string();
    let has_org = repo.contains('/');
    let token_set = std::env::var("HF_TOKEN").is_ok()
        || std::env::var("HUGGING_FACE_HUB_TOKEN").is_ok();

    let mut hints: Vec<String> = Vec::new();

    if raw.contains("401") || raw.contains("Unauthorized") {
        if !has_org {
            hints.push(format!(
                "the repo id looks incomplete ({repo:?}). HuggingFace repos are \
                 \"<org>/<name>\", e.g. black-forest-labs/FLUX.1-dev. Try the alias \
                 flux-dev, or pass the full repo id."
            ));
        }
        hints.push(format!(
            "401 usually means the repo is gated. Open https://huggingface.co/{repo} \
             and click \"Agree and access repository\", then {action}.",
            action = if token_set {
                "retry — your HF_TOKEN should now work"
            } else {
                "set HF_TOKEN to a token from https://huggingface.co/settings/tokens"
            }
        ));
    } else if raw.contains("403") || raw.contains("Forbidden") {
        hints.push(format!(
            "403: your HF_TOKEN exists but lacks read access to {repo}. Accept the \
             repo's license at https://huggingface.co/{repo}, then retry."
        ));
    } else if raw.contains("404") || raw.contains("Not Found") {
        if !has_org {
            hints.push(format!(
                "the repo id looks incomplete ({repo:?}). HuggingFace repos are \
                 \"<org>/<name>\". Maybe you meant `black-forest-labs/{repo}`?"
            ));
        } else {
            hints.push(format!(
                "404: file or repo doesn't exist. Check https://huggingface.co/{repo}."
            ));
        }
    } else if raw.contains("etag") || raw.contains("Etag") {
        hints.push(format!(
            "the HF response had no ETag header. This usually means the repo is \
             gone, gated, or private. Check https://huggingface.co/{repo}; if gated, \
             set HF_TOKEN."
        ));
    }

    if hints.is_empty() {
        anyhow!("downloading {repo}/{file}: {raw}")
    } else {
        anyhow!(
            "downloading {repo}/{file}: {raw}\nhint: {}",
            hints.join("\nhint: ")
        )
    }
}

/// Try each (repo, file) pair in order; return the first that downloads.
/// Useful for cross-repo fallbacks (e.g. tokenizer.json missing in legacy
/// SD repos can be fetched from openai/clip-vit-large-patch14 instead).
pub async fn get_first_of(candidates: &[(&str, &str)]) -> Result<PathBuf> {
    let mut last_err = None;
    for (repo, file) in candidates {
        match get_file(repo, file).await {
            Ok(p) => return Ok(p),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("no candidates given")))
}

/// Fetch a single file from `repo` at `revision` (commit SHA, tag, or
/// branch name), returning its local cache path. Cached separately
/// from other revisions — the cache key is `(repo, revision, file)`.
///
/// Use [`get_file`] when no specific revision is required (defaults
/// to `main`).
pub async fn get_file_at(repo: &str, file: &str, revision: &str) -> Result<PathBuf> {
    let repo = crate::hf::resolve_alias(repo).to_string();
    let api = api()?.repo(Repo::with_revision(
        repo.clone(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Show the revision short-prefix in the spinner only when it's not
    // `main` — keeps the common case unchanged, and makes pinned-revision
    // fetches visibly distinct in scenario logs.
    let rev_note = if revision == "main" {
        String::new()
    } else {
        // char-based, not byte-slice: a revision (branch/tag) with a multibyte char
        // crossing byte 8 would panic `[..8]` ("not a char boundary").
        format!(" @ {}", revision.chars().take(8).collect::<String>())
    };

    // Fast path: already in the local cache → no network, no bar (instant).
    if let Some(path) = cached_path(&repo, file, revision) {
        return Ok(path);
    }

    // Download with a REAL byte-progress bar (plakat's `bytes_bar`, which is
    // rerouted into the TUI Output pane). hf-hub's own `get()` uses its own bar on
    // stderr, which never reaches the TUI — so we drive ours via the Progress trait.
    let bar = crate::ui::progress::bytes_bar(0, &format!("⤓ {repo}  {file}{rev_note}"));
    let progress = BarProgress { bar: bar.clone(), done: Arc::new(AtomicU64::new(0)) };
    match api.download_with_progress(file, progress).await {
        Ok(path) => {
            bar.finish_and_clear();
            Ok(path)
        }
        Err(e) => {
            bar.finish_with_message(format!("✗ {repo}  {file}{rev_note}"));
            Err(friendly_error(&repo, file, e))
        }
    }
}

/// Look up a file in the local HF cache without downloading. `Some(path)` → cached
/// (return instantly, no progress bar); `None` → must download.
fn cached_path(repo: &str, file: &str, revision: &str) -> Option<PathBuf> {
    Cache::new(crate::hf::cache::hf_cache_root())
        .repo(Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string()))
        .get(file)
}

/// Whether a model alias's weights are already in the local cache (no download
/// needed). Checks `model_index.json` — the diffusers manifest present in every
/// SD-family repo — so the TUI can auto-load a cached default model on startup
/// without triggering a multi-GB download.
pub fn is_cached(alias: &str) -> bool {
    let repo = crate::hf::resolve_alias(alias);
    cached_path(repo, "model_index.json", "main").is_some()
}

/// hf-hub's cache directory for `repo` (`<root>/models--<org>--<name>`).
fn repo_cache_dir(repo: &str) -> PathBuf {
    crate::hf::cache::hf_cache_root().join(format!("models--{}", repo.replace('/', "--")))
}

/// A lock untouched for at least this long is treated as stale (left by a crashed/OOM'd
/// process). Younger locks may belong to a download in flight RIGHT NOW (e.g. the TUI
/// download manager fetching into the same repo cache concurrently) — removing one would
/// let two writers race the same blob and corrupt it. Generous enough that even a slow
/// large fetch's lock is younger.
const STALE_LOCK_AGE: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Remove stale hf-hub `*.lock` files under a repo's cache dir. An interrupted download
/// (e.g. a Ctrl-C or the OOM-guard hard-exit mid-fetch) can leave a `.lock` behind that
/// makes a later load of an already-cached model hang or refuse. Only locks older than
/// [`STALE_LOCK_AGE`] are swept, so a concurrent in-flight download's fresh lock survives.
/// Returns how many were removed.
pub fn clean_stale_locks(alias: &str) -> usize {
    let repo = crate::hf::resolve_alias(alias);
    clean_locks_older_than(&repo_cache_dir(repo), STALE_LOCK_AGE)
}

/// Remove `*.lock` files under `dir` and `dir/blobs` whose mtime is at least `min_age`
/// old. Returns the count. Locks whose mtime can't be read are left alone (conservative).
fn clean_locks_older_than(dir: &Path, min_age: std::time::Duration) -> usize {
    let mut removed = 0;
    for sub in [dir.to_path_buf(), dir.join("blobs")] {
        if let Ok(rd) = std::fs::read_dir(&sub) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("lock") {
                    continue;
                }
                let stale = std::fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .map(|t| t.elapsed().map(|age| age >= min_age).unwrap_or(false))
                    .unwrap_or(false);
                if stale && std::fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
        }
    }
    removed
}

/// Drives a plakat `bytes_bar` from hf-hub's download callbacks → a real `%` /
/// bytes progress bar (in the CLI, and rerouted into the TUI Output pane).
#[derive(Clone)]
struct BarProgress {
    bar: ProgressBar,
    done: Arc<AtomicU64>,
}

impl Progress for BarProgress {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.bar.set_length(size as u64);
        self.done.store(0, Ordering::Relaxed);
    }
    async fn update(&mut self, size: usize) {
        let done = self.done.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
        self.bar.set_position(done);
    }
    async fn finish(&mut self) {
        self.bar.finish_and_clear();
    }
}

/// Fetch a single file from `repo`'s `main` revision. Thin wrapper over
/// [`get_file_at`] for callers that don't pin a specific commit.
pub async fn get_file(repo: &str, file: &str) -> Result<PathBuf> {
    get_file_at(repo, file, "main").await
}

/// Pull every model file in a repo, reporting how many landed. Enumerates the
/// repo's ACTUAL files (any layout) rather than guessing a fixed SD-diffusers
/// filename list — which 404s on PixArt / SD3 (`transformer/`, sharded T5),
/// Cascade, and single-file checkpoints. Gated / missing repos surface one
/// clear message instead of a wall of per-file 404s.
pub async fn pull_all(repo: &str) -> Result<()> {
    let resolved = crate::hf::resolve_alias(repo).to_string();
    let files = crate::hf::info::repo_files(&resolved)
        .await
        .with_context(|| format!("pull {repo}"))?;
    let wanted = select_pull_files(&files);
    if wanted.is_empty() {
        return Err(anyhow!("repo {resolved} lists no model files to pull"));
    }
    let mut ok = 0usize;
    let mut failed = 0usize;
    for f in &wanted {
        match get_file(&resolved, f).await {
            Ok(_) => ok += 1,
            Err(e) => {
                tracing::warn!(target: "plakat", "pull {resolved}: skip {f}: {e}");
                failed += 1;
            }
        }
    }
    if ok == 0 {
        return Err(anyhow!(
            "no files fetched from {resolved} ({} attempted)",
            wanted.len()
        ))
        .with_context(|| format!("pull {repo}"));
    }
    tracing::info!(
        target: "plakat",
        "pulled {ok}/{} files from {resolved}{}",
        wanted.len(),
        if failed == 0 { String::new() } else { format!(" ({failed} skipped)") }
    );
    Ok(())
}

/// From the repo's full file list, pick what's worth pulling: drop preview
/// images + docs, and when both `X.fp16.safetensors` and `X.safetensors` exist
/// for one component keep only the fp16 (what plakat loads on GPU).
pub(crate) fn select_pull_files(files: &[String]) -> Vec<String> {
    use std::collections::HashSet;
    let set: HashSet<&str> = files.iter().map(String::as_str).collect();
    files
        .iter()
        .filter(|f| {
            let l = f.to_ascii_lowercase();
            let is_doc = l.ends_with(".png")
                || l.ends_with(".jpg")
                || l.ends_with(".jpeg")
                || l.ends_with(".webp")
                || l.ends_with(".gif")
                || l.ends_with(".md")
                || l.ends_with(".gitattributes");
            if is_doc {
                return false;
            }
            // Drop the full-precision twin when an fp16 sibling exists.
            if let Some(stem) = f.strip_suffix(".safetensors") {
                if !stem.ends_with(".fp16") {
                    let fp16 = format!("{stem}.fp16.safetensors");
                    if set.contains(fp16.as_str()) {
                        return false;
                    }
                }
            }
            true
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_cached_false_for_unknown_model() {
        // A bogus alias resolves to itself and is never in the cache.
        assert!(!is_cached("definitely-not-a-real-model-xyz-123"));
    }

    #[test]
    fn clean_locks_removes_stale_lock_files_only() {
        let d = std::env::temp_dir().join("plakat-lock-sweep-test");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("blobs")).unwrap();
        std::fs::create_dir_all(d.join("snapshots")).unwrap();
        // Two stale locks (one top-level, one under blobs) + real files that must survive.
        std::fs::write(d.join("a.lock"), b"").unwrap();
        std::fs::write(d.join("blobs/deadbeef.lock"), b"").unwrap();
        std::fs::write(d.join("blobs/deadbeef"), b"weights").unwrap(); // the real blob
        std::fs::write(d.join("refs"), b"main").unwrap();
        // A fresh lock (age ~0) is NOT swept with the real threshold — it may belong to a
        // concurrent in-flight download (removing it would race the blob).
        assert_eq!(clean_locks_older_than(&d, std::time::Duration::from_secs(300)), 0, "fresh locks preserved");
        assert!(d.join("a.lock").exists(), "fresh lock kept");
        // With a zero threshold every lock is stale → both swept; real files survive.
        let removed = clean_locks_older_than(&d, std::time::Duration::ZERO);
        assert_eq!(removed, 2, "both .lock files swept");
        assert!(!d.join("a.lock").exists() && !d.join("blobs/deadbeef.lock").exists());
        assert!(d.join("blobs/deadbeef").exists(), "the real blob is untouched");
        assert!(d.join("refs").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn select_pull_files_prefers_fp16_and_drops_previews() {
        let files: Vec<String> = [
            "model_index.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors", // dropped: fp16 twin exists
            "vae/diffusion_pytorch_model.safetensors",   // kept: no fp16 twin
            "transformer/diffusion_pytorch_model.safetensors", // PixArt/SD3 layout
            "preview.png",                               // dropped: image
            "README.md",                                 // dropped: doc
            ".gitattributes",                            // dropped
            "tokenizer/merges.txt",                      // kept
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let got = select_pull_files(&files);
        assert!(got.contains(&"unet/diffusion_pytorch_model.fp16.safetensors".to_string()));
        assert!(!got.contains(&"unet/diffusion_pytorch_model.safetensors".to_string()));
        assert!(got.contains(&"vae/diffusion_pytorch_model.safetensors".to_string()));
        assert!(got.contains(&"transformer/diffusion_pytorch_model.safetensors".to_string()));
        assert!(got.contains(&"tokenizer/merges.txt".to_string()));
        assert!(!got.iter().any(|f| f.ends_with(".png") || f.ends_with(".md") || f == ".gitattributes"));
    }
}
