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

// --- Vision (OpenAI-compatible image_url), used by the provider-agnostic `super::vision`. ---

#[derive(Serialize)]
struct VReq<'a> {
    model: &'a str,
    messages: Vec<VMsg<'a>>,
    temperature: f32,
}
#[derive(Serialize)]
struct VMsg<'a> {
    role: &'a str,
    content: Vec<VPart<'a>>,
}
#[derive(Serialize)]
#[serde(tag = "type")]
enum VPart<'a> {
    #[serde(rename = "text")]
    Text { text: &'a str },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}
#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

/// Vision via the OpenAI-compatible chat API (image as a `data:` URL). Works with any vision-capable
/// model on the DeepSeek-compatible endpoint; a text-only model will reject the image with its own
/// error. Requires `DEEPSEEK_API_KEY`.
pub async fn describe_image_jpeg(instruction: &str, jpeg_b64: &str) -> Result<String> {
    let key = crate::config::Config::load()?
        .deepseek_api_key
        .ok_or_else(|| anyhow!("DEEPSEEK_API_KEY not set"))?;
    let body = VReq {
        model: "deepseek-chat",
        messages: vec![VMsg {
            role: "user",
            content: vec![
                VPart::Text { text: instruction },
                VPart::ImageUrl {
                    image_url: ImageUrl { url: format!("data:image/jpeg;base64,{jpeg_b64}") },
                },
            ],
        }],
        temperature: 0.2,
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
