//! Ollama provider for the prompt/enhance stack. Talks to a local (or `OLLAMA_HOST`) Ollama server's
//! `/api/chat` endpoint, so any model the user has pulled (`ollama pull qwen2.5-coder:14b`) is usable as a
//! `--provider ollama` / `ollama:<model>` enhancer — a bigger local model than the in-process GGUF aliases,
//! without an API key. Mirrors [`super::deepseek`]'s shape.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// The model used for a bare `ollama` provider (override with `ollama:<model>`).
pub const DEFAULT_MODEL: &str = "qwen2.5-coder:14b";

/// Base URL of the Ollama server (`OLLAMA_HOST`, else localhost).
fn host() -> String {
    let h = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    // OLLAMA_HOST is sometimes set bare (`host:port`); normalise to a URL.
    if h.starts_with("http://") || h.starts_with("https://") {
        h
    } else {
        format!("http://{h}")
    }
}

#[derive(Serialize)]
struct Req<'a> {
    model: &'a str,
    messages: Vec<Msg<'a>>,
    stream: bool,
    options: Opts,
}
#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}
#[derive(Serialize)]
struct Opts {
    temperature: f32,
    num_predict: i32,
    seed: u64,
}
#[derive(Deserialize)]
struct Resp {
    message: RespMsg,
}
#[derive(Deserialize)]
struct RespMsg {
    content: String,
}

/// Enhance with the built-in system prompt on the default model.
pub async fn enhance(prompt: &str) -> Result<String> {
    enhance_with_system_model(DEFAULT_MODEL, super::SYSTEM, prompt).await
}

/// Enhance with a caller-supplied system prompt on the default model — the shape the router calls.
pub async fn enhance_with_system(system: &str, prompt: &str) -> Result<String> {
    enhance_with_system_model(DEFAULT_MODEL, system, prompt).await
}

/// Run one chat completion against Ollama's `/api/chat` (non-streaming). `num_predict = -1` = no cap; a
/// generous budget is used so a full structured reply (e.g. a layer plan) is never truncated.
pub async fn enhance_with_system_model(model: &str, system: &str, prompt: &str) -> Result<String> {
    let body = Req {
        model,
        messages: vec![Msg { role: "system", content: system }, Msg { role: "user", content: prompt }],
        stream: false,
        options: Opts { temperature: 0.0, num_predict: 2048, seed: 0 },
    };
    let url = format!("{}/api/chat", host());
    let resp: Resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow!("contacting Ollama at {url} (is `ollama serve` running, and `{model}` pulled?): {e}"))?
        .error_for_status()
        .map_err(|e| anyhow!("Ollama returned an error for model {model:?}: {e}"))?
        .json()
        .await
        .map_err(|e| anyhow!("parsing the Ollama response: {e}"))?;
    let out = resp.message.content.trim().to_string();
    if out.is_empty() {
        return Err(anyhow!("Ollama returned an empty response for model {model:?}"));
    }
    Ok(out)
}
