use anyhow::{Result, anyhow};

pub mod a1111;
pub mod break_chunks;
pub mod deepseek;
pub mod gemini;
pub mod lora_tags;
pub mod weighted_encoding;
pub mod wildcards;

pub async fn enhance(provider: &str, prompt: &str) -> Result<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::enhance(prompt).await,
        "gemini" => gemini::enhance(prompt).await,
        // v0.18: local LLM-based enhance (Qwen2.5-1.5B by default).
        // Forms accepted:
        //   "local"             — default alias (qwen2.5-1.5b)
        //   "local:smollm2-360m" — explicit fallback alias
        //   "local:qwen2.5-1.5b" — explicit default alias
        "local" => enhance_local(crate::llm::DEFAULT_ALIAS, prompt).await,
        other if other.starts_with("local:") => {
            let alias = &other["local:".len()..];
            enhance_local(alias, prompt).await
        }
        other if other == "auto" => enhance_auto(prompt).await,
        other => Err(anyhow!(
            "unknown prompt enhancer: {other} (supported: deepseek, gemini, local, \
             local:<alias>, auto)"
        )),
    }
}

/// v0.18: dispatch to the local LLM enhancer. Refusals + empty
/// output fall back to the original prompt with a warn log — same
/// graceful-degrade semantics as the API-based enhancers when they
/// hit a network error.
async fn enhance_local(alias: &str, prompt: &str) -> Result<String> {
    let device = crate::device::select("auto")
        .map_err(|e| anyhow!("device selection for enhance: {e}"))?;
    match crate::llm::enhance(
        alias,
        device,
        SYSTEM,
        prompt,
        crate::llm::EnhanceOpts::default(),
    )
    .await
    {
        Ok(enhanced) => Ok(enhanced),
        Err(crate::llm::EnhanceError::Refused) => {
            tracing::warn!(
                target: "plakat",
                "local enhancer ({alias}) returned a refusal or empty output \
                 — falling back to the un-enhanced prompt"
            );
            Ok(prompt.to_string())
        }
        Err(crate::llm::EnhanceError::Other(e)) => Err(e),
    }
}

/// v0.18: `--enhance auto` — pick a provider based on what's
/// available. Priority: DeepSeek if `DEEPSEEK_API_KEY` is set →
/// Gemini if `GEMINI_API_KEY` is set → local. The local arm always
/// works (no API key required), so this never errors at the
/// provider-selection layer.
async fn enhance_auto(prompt: &str) -> Result<String> {
    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        tracing::info!(
            target: "plakat",
            "enhance auto: routing to DeepSeek (DEEPSEEK_API_KEY set)"
        );
        return deepseek::enhance(prompt).await;
    }
    if std::env::var("GEMINI_API_KEY").is_ok() {
        tracing::info!(
            target: "plakat",
            "enhance auto: routing to Gemini (GEMINI_API_KEY set)"
        );
        return gemini::enhance(prompt).await;
    }
    tracing::info!(
        target: "plakat",
        "enhance auto: no API key set — falling back to local enhancer ({})",
        crate::llm::DEFAULT_ALIAS
    );
    enhance_local(crate::llm::DEFAULT_ALIAS, prompt).await
}

pub const SYSTEM: &str = "You rewrite text-to-image prompts. \
Add concrete visual detail (subject, composition, lighting, medium, mood, style). \
Keep it under 70 tokens. Output ONLY the rewritten prompt, no preamble, no quotes.";
