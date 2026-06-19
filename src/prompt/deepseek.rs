use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Req<'a> {
    model: &'a str,
    messages: Vec<Msg<'a>>,
    temperature: f32,
}
#[derive(Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}
#[derive(Deserialize)]
struct Resp {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: RespMsg,
}
#[derive(Deserialize)]
struct RespMsg {
    content: String,
}

pub async fn enhance(prompt: &str) -> Result<String> {
    enhance_with_system(super::SYSTEM, prompt).await
}

/// Like [`enhance`] but with a caller-supplied system prompt — used by
/// `plakat compile`, which builds a family-aware system prompt per scene.
pub async fn enhance_with_system(system: &str, prompt: &str) -> Result<String> {
    let key = crate::config::Config::load()?
        .deepseek_api_key
        .ok_or_else(|| anyhow!("DEEPSEEK_API_KEY not set"))?;

    let body = Req {
        model: "deepseek-chat",
        messages: vec![
            Msg {
                role: "system",
                content: system,
            },
            Msg {
                role: "user",
                content: prompt,
            },
        ],
        temperature: 0.6,
    };

    let resp: Resp = reqwest::Client::new()
        .post("https://api.deepseek.com/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    resp.choices
        .into_iter()
        .next()
        .map(|c| c.message.content.trim().to_string())
        .ok_or_else(|| anyhow!("no choices in DeepSeek response"))
}
