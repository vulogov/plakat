use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Req {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction")]
    system: Content,
}
#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}
#[derive(Serialize)]
struct Part {
    text: String,
}
#[derive(Deserialize)]
struct Resp {
    candidates: Vec<Candidate>,
}
#[derive(Deserialize)]
struct Candidate {
    content: RespContent,
}
#[derive(Deserialize)]
struct RespContent {
    parts: Vec<RespPart>,
}
#[derive(Deserialize)]
struct RespPart {
    text: String,
}

pub async fn enhance(prompt: &str) -> Result<String> {
    enhance_with_system(super::SYSTEM, prompt).await
}

/// Like [`enhance`] but with a caller-supplied system prompt — used by
/// `plakat compile`, which builds a family-aware system prompt per scene.
pub async fn enhance_with_system(system: &str, prompt: &str) -> Result<String> {
    let key = crate::config::Config::load()?
        .gemini_api_key
        .ok_or_else(|| anyhow!("GEMINI_API_KEY not set"))?;

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    );
    let body = Req {
        contents: vec![Content {
            parts: vec![Part {
                text: prompt.into(),
            }],
        }],
        system: Content {
            parts: vec![Part {
                text: system.into(),
            }],
        },
    };

    let resp: Resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    resp.candidates
        .into_iter()
        .next()
        .and_then(|c| c.content.parts.into_iter().next())
        .map(|p| p.text.trim().to_string())
        .ok_or_else(|| anyhow!("no candidates in Gemini response"))
}
