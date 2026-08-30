use anyhow::{Result, anyhow};
use std::path::PathBuf;

pub mod a1111;
pub mod break_chunks;
pub mod deepseek;
pub mod gemini;
pub mod vision;
pub mod lora_tags;
pub mod negative_presets;
pub mod scheduling;
pub mod weighted_encoding;
pub mod wildcards;

/// v0.19: CLI-tunable knobs for the prompt enhancer. The API-keyed
/// providers (DeepSeek / Gemini) honour the system-prompt override
/// only; temperature / max_new_tokens / cache are local-only
/// concerns (the API endpoints have their own defaults).
#[derive(Debug, Clone, Default)]
pub struct EnhanceArgs {
    /// `--enhance-system PATH` — load a custom system prompt from
    /// disk. `None` falls back to the built-in [`SYSTEM`] default.
    pub system_path: Option<PathBuf>,
    /// `--enhance-temp F` — local-LLM sampling temperature.
    /// `None` → greedy (the default; reproducible). Ignored on
    /// DeepSeek / Gemini.
    pub temperature: Option<f64>,
    /// `--enhance-max-tokens N` — local-LLM generation cap.
    /// `None` → 96 (the default). Ignored on DeepSeek / Gemini.
    pub max_new_tokens: Option<usize>,
    /// `--enhance-cache` — opt-in disk cache for the local
    /// enhancer. SHA-256 of (alias, system, user, temp, max_tokens)
    /// keys the on-disk lookup. Hits skip the LLM forward.
    pub cache: bool,
}

pub async fn enhance(provider: &str, prompt: &str) -> Result<String> {
    enhance_with_args(provider, prompt, &EnhanceArgs::default()).await
}

/// v0.19: full enhance dispatch with CLI-supplied opts. The
/// signature additions are backwards-compatible: callers that
/// don't care pass `&EnhanceArgs::default()` (or use the legacy
/// [`enhance`] one-line wrapper).
pub async fn enhance_with_args(
    provider: &str,
    prompt: &str,
    args: &EnhanceArgs,
) -> Result<String> {
    let system = resolve_system(args)?;
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::enhance(prompt).await,
        "gemini" => gemini::enhance(prompt).await,
        // v0.18: local LLM-based enhance (Qwen2.5-1.5B by default).
        // Forms accepted:
        //   "local"             — default alias (qwen2.5-1.5b)
        //   "local:smollm2-360m" — explicit fallback alias
        //   "local:qwen2.5-1.5b" — explicit default alias
        "local" => {
            enhance_local(crate::llm::DEFAULT_ALIAS, prompt, &system, args).await
        }
        other if other.starts_with("local:") => {
            let alias = &other["local:".len()..];
            enhance_local(alias, prompt, &system, args).await
        }
        other if other == "auto" => enhance_auto(prompt, &system, args).await,
        other => Err(anyhow!(
            "unknown prompt enhancer: {other} (supported: deepseek, gemini, local, \
             local:<alias>, auto)"
        )),
    }
}

/// Like [`enhance_with_args`] but with an explicit, caller-built `system`
/// prompt (not the built-in / `--enhance-system`). `plakat compile` uses this to
/// pass a family-aware system prompt per scene through the same provider stack.
pub async fn complete(
    provider: &str,
    system: &str,
    user: &str,
    args: &EnhanceArgs,
) -> Result<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::enhance_with_system(system, user).await,
        "gemini" => gemini::enhance_with_system(system, user).await,
        "local" => enhance_local(crate::llm::DEFAULT_ALIAS, user, system, args).await,
        other if other.starts_with("local:") => {
            let alias = other["local:".len()..].to_string();
            enhance_local(&alias, user, system, args).await
        }
        "auto" => enhance_auto(user, system, args).await,
        other => Err(anyhow!(
            "unknown provider: {other} (supported: deepseek, gemini, local, local:<alias>, auto)"
        )),
    }
}

/// Load `--enhance-system PATH` from disk when set; fall back to
/// the built-in [`SYSTEM`] default otherwise. Returns an owned
/// string so the lifetime stays unambiguous across the async dispatch.
fn resolve_system(args: &EnhanceArgs) -> Result<String> {
    match &args.system_path {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| anyhow!("reading --enhance-system {}: {e}", p.display()))
            .map(|s| s.trim().to_string()),
        None => Ok(SYSTEM.to_string()),
    }
}

/// v0.18: dispatch to the local LLM enhancer. Refusals + empty
/// output fall back to the original prompt with a warn log — same
/// graceful-degrade semantics as the API-based enhancers when they
/// hit a network error.
async fn enhance_local(
    alias: &str,
    prompt: &str,
    system: &str,
    args: &EnhanceArgs,
) -> Result<String> {
    let device = crate::device::select("auto")
        .map_err(|e| anyhow!("device selection for enhance: {e}"))?;

    // v0.19 opts: assemble from CLI overrides + EnhanceOpts defaults.
    let mut opts = crate::llm::EnhanceOpts::default();
    if let Some(t) = args.temperature {
        opts.temperature = t;
    }
    if let Some(n) = args.max_new_tokens {
        opts.max_new_tokens = n;
    }

    // v0.19 cache: SHA-256 (alias, system, user, temp, max_tokens)
    // → enhanced text on disk. Cache hits skip the LLM forward
    // entirely. Opt-in via --enhance-cache to avoid stale-hit
    // surprises during system-prompt iteration.
    if args.cache {
        let key = crate::llm::cache::CacheKey {
            alias,
            system,
            user: prompt,
            temperature: opts.temperature,
            max_new_tokens: opts.max_new_tokens,
        };
        if let Some(cached) = crate::llm::cache::lookup(&key) {
            tracing::info!(
                target: "plakat",
                "enhance cache hit ({alias})"
            );
            return Ok(cached);
        }
    }

    let result = match crate::llm::enhance(alias, device, system, prompt, opts).await {
        Ok(enhanced) => enhanced,
        Err(crate::llm::EnhanceError::Refused) => {
            tracing::warn!(
                target: "plakat",
                "local enhancer ({alias}) returned a refusal or empty output \
                 — falling back to the un-enhanced prompt"
            );
            return Ok(prompt.to_string());
        }
        Err(crate::llm::EnhanceError::Other(e)) => return Err(e),
    };

    // Write to cache on success — never cache the refusal fallback.
    if args.cache {
        let key = crate::llm::cache::CacheKey {
            alias,
            system,
            user: prompt,
            temperature: opts.temperature,
            max_new_tokens: opts.max_new_tokens,
        };
        if let Err(e) = crate::llm::cache::store(&key, &result) {
            tracing::warn!(
                target: "plakat",
                "enhance cache write failed: {e} — result still returned"
            );
        }
    }
    Ok(result)
}

/// v0.18: `--enhance auto` — pick a provider based on what's
/// available. Priority: DeepSeek if `DEEPSEEK_API_KEY` is set →
/// Gemini if `GEMINI_API_KEY` is set → local. The local arm always
/// works (no API key required), so this never errors at the
/// provider-selection layer.
/// The concrete provider `auto` resolves to right now — `"deepseek"`, `"gemini"`,
/// or `"local"` — for display (the TUI shows it before enhancing). Mirrors
/// [`enhance_auto`]'s key detection; a non-`auto` provider is returned as-is.
pub fn resolve_provider_label(provider: &str) -> String {
    if provider.eq_ignore_ascii_case("auto") {
        let cfg = crate::config::Config::load().ok();
        if cfg.as_ref().is_some_and(|c| c.deepseek_api_key.is_some()) {
            return "deepseek".into();
        }
        if cfg.as_ref().is_some_and(|c| c.gemini_api_key.is_some()) {
            return "gemini".into();
        }
        return "local".into();
    }
    provider.to_string()
}

async fn enhance_auto(prompt: &str, system: &str, args: &EnhanceArgs) -> Result<String> {
    // Detect API keys the SAME way the providers do — `Config::load()` folds in
    // `~/.config/plakat/config.toml` AND the env var. The old check only looked at
    // the env var, so a key living in config.toml made `auto` silently fall back to
    // the local LLM even though `--enhance deepseek` worked. Now `auto` routes to
    // DeepSeek/Gemini whenever the CLI's explicit providers would.
    let cfg = crate::config::Config::load().ok();
    let has_deepseek = cfg.as_ref().is_some_and(|c| c.deepseek_api_key.is_some());
    let has_gemini = cfg.as_ref().is_some_and(|c| c.gemini_api_key.is_some());

    if has_deepseek {
        tracing::info!(
            target: "plakat",
            "enhance auto: routing to DeepSeek (key in config.toml or env)"
        );
        return deepseek::enhance(prompt).await;
    }
    if has_gemini {
        tracing::info!(
            target: "plakat",
            "enhance auto: routing to Gemini (key in config.toml or env)"
        );
        return gemini::enhance(prompt).await;
    }
    tracing::info!(
        target: "plakat",
        "enhance auto: no API key set — falling back to local enhancer ({})",
        crate::llm::DEFAULT_ALIAS
    );
    enhance_local(crate::llm::DEFAULT_ALIAS, prompt, system, args).await
}

pub const SYSTEM: &str = "You rewrite text-to-image prompts. \
Add concrete visual detail (subject, composition, lighting, medium, mood, style). \
Keep it under 70 tokens. Output ONLY the rewritten prompt, no preamble, no quotes.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_provider_label_passes_explicit_providers_through() {
        assert_eq!(resolve_provider_label("deepseek"), "deepseek");
        assert_eq!(resolve_provider_label("gemini"), "gemini");
        assert_eq!(resolve_provider_label("local"), "local");
        assert_eq!(resolve_provider_label("local:qwen2.5-1.5b"), "local:qwen2.5-1.5b");
        // `auto` resolves to one of the known providers (depends on config/env keys).
        assert!(["deepseek", "gemini", "local"].contains(&resolve_provider_label("auto").as_str()));
    }
}
