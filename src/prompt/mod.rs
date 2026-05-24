use anyhow::{Result, anyhow};

pub mod a1111;
pub mod deepseek;
pub mod gemini;
pub mod weighted_encoding;
pub mod wildcards;

pub async fn enhance(provider: &str, prompt: &str) -> Result<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::enhance(prompt).await,
        "gemini" => gemini::enhance(prompt).await,
        other => Err(anyhow!(
            "unknown prompt enhancer: {other} (supported: deepseek, gemini)"
        )),
    }
}

pub const SYSTEM: &str = "You rewrite text-to-image prompts. \
Add concrete visual detail (subject, composition, lighting, medium, mood, style). \
Keep it under 70 tokens. Output ONLY the rewritten prompt, no preamble, no quotes.";
