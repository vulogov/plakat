use anyhow::{Context, Result, anyhow};
use hf_hub::api::tokio::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;

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
        format!(" @ {}", &revision[..revision.len().min(8)])
    };

    let pb = crate::ui::progress::spinner(&format!("⤓ {repo}  {file}{rev_note}"));
    match api.get(file).await {
        Ok(path) => {
            pb.finish_and_clear();
            Ok(path)
        }
        Err(e) => {
            pb.finish_with_message(format!("✗ {repo}  {file}{rev_note}"));
            Err(friendly_error(&repo, file, e))
        }
    }
}

/// Fetch a single file from `repo`'s `main` revision. Thin wrapper over
/// [`get_file_at`] for callers that don't pin a specific commit.
pub async fn get_file(repo: &str, file: &str) -> Result<PathBuf> {
    get_file_at(repo, file, "main").await
}

/// Fetch the canonical SD layout for a repo, reporting how many files landed.
/// Files that 404 are skipped silently; other errors are reported.
pub async fn pull_all(repo: &str) -> Result<()> {
    let candidates: &[&str] = &[
        "model_index.json",
        "tokenizer/tokenizer.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer_2/tokenizer.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.fp16.safetensors",
        "text_encoder_2/model.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ];
    let mut ok = 0usize;
    let mut first_err: Option<anyhow::Error> = None;
    let mut other_files: Vec<String> = Vec::new();
    for f in candidates {
        match get_file(repo, f).await {
            Ok(_) => ok += 1,
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                } else {
                    other_files.push(f.to_string());
                }
            }
        }
    }
    if ok == 0 {
        let e = first_err.unwrap();
        let total_failed = 1 + other_files.len();
        return Err(anyhow!(
            "no files fetched from {repo} ({total_failed} attempts, same kind of error).\n\
             First failure:\n{e}"
        ))
        .with_context(|| format!("pull {repo}"));
    }
    tracing::info!(target: "plakat", "pulled {ok}/{total} files from {repo}", total = candidates.len());
    if let Some(e) = first_err {
        tracing::debug!(target: "plakat", "{} files skipped/missing; first: {e}", 1 + other_files.len());
    }
    Ok(())
}
