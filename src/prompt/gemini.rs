use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Google's maintained "latest flash" alias. Unlike a pinned `gemini-1.5-flash` (which was **retired in
/// September 2025** and now 404s — silently breaking every compile enhance/translate call), Google repoints
/// this alias as models are rotated, so plakat keeps working across retirements with no recompile.
const DEFAULT_MODEL: &str = "gemini-flash-latest";

/// The Gemini model id for enhance + vision, resolved WITHOUT recompiling: `config.toml`
/// (`gemini_model = "…"`) → `GEMINI_MODEL` env → the maintained `gemini-flash-latest` alias. Change it in
/// one place — the config file — to follow (or pin) Google's models whenever they rotate.
fn model() -> String {
    crate::config::Config::load()
        .ok()
        .and_then(|c| c.gemini_model)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string())
}

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
#[derive(Deserialize, Default)]
struct Resp {
    #[serde(default)]
    candidates: Vec<Candidate>,
    // When the *prompt* itself is blocked, Gemini returns no candidates and puts the reason here.
    #[serde(rename = "promptFeedback", default)]
    prompt_feedback: Option<PromptFeedback>,
}
#[derive(Deserialize)]
struct Candidate {
    // Absent when the candidate finished with e.g. SAFETY / MAX_TOKENS before emitting any text.
    #[serde(default)]
    content: Option<RespContent>,
    #[serde(rename = "finishReason", default)]
    finish_reason: Option<String>,
}
#[derive(Deserialize, Default)]
struct RespContent {
    #[serde(default)]
    parts: Vec<RespPart>,
}
#[derive(Deserialize)]
struct RespPart {
    #[serde(default)]
    text: String,
}
#[derive(Deserialize, Default)]
struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    block_reason: Option<String>,
}

/// Pull the first non-empty text part out of a Gemini response, or a *diagnostic* error that names the
/// real reason (SAFETY / MAX_TOKENS / prompt-block) instead of a bare "no candidates" — so a failed
/// enhance/vision call is legible in the logs rather than a silent verbatim fallback.
fn extract_text(resp: Resp) -> Result<String> {
    let text: Option<String> = resp
        .candidates
        .iter()
        .flat_map(|c| c.content.iter())
        .flat_map(|c| c.parts.iter())
        .map(|p| p.text.trim())
        .find(|t| !t.is_empty())
        .map(|t| t.to_string());
    if let Some(t) = text {
        return Ok(t);
    }
    if let Some(reason) = resp.prompt_feedback.and_then(|f| f.block_reason) {
        return Err(anyhow!("Gemini blocked the prompt (blockReason: {reason})"));
    }
    match resp.candidates.first().and_then(|c| c.finish_reason.clone()) {
        Some(r) if r != "STOP" => Err(anyhow!(
            "Gemini returned no text (finishReason: {r}) — try a shorter prompt or a different --compile-provider"
        )),
        _ => Err(anyhow!("Gemini returned an empty response (no text parts)")),
    }
}

/// POST a JSON body to a Gemini `generateContent` endpoint and parse the response. Reads the raw body
/// first so an HTTP error or an unexpected shape surfaces the actual server text (truncated) instead of
/// an opaque serde error.
async fn post_generate<B: Serialize>(url: &str, body: &B) -> Result<Resp> {
    let http = reqwest::Client::new().post(url).json(body).send().await?;
    let status = http.status();
    let raw = http.text().await?;
    if !status.is_success() {
        let snippet: String = raw.chars().take(300).collect();
        return Err(anyhow!("Gemini HTTP {status}: {snippet}"));
    }
    serde_json::from_str::<Resp>(&raw).map_err(|e| {
        let snippet: String = raw.chars().take(300).collect();
        anyhow!("Gemini response did not parse ({e}): {snippet}")
    })
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
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={key}",
        model()
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

    extract_text(post_generate(&url, &body).await?)
}

// --- Vision (image → text): used by `plakat photos` autotag / describe. ---

#[derive(Serialize)]
struct VReq {
    contents: Vec<VContent>,
}
#[derive(Serialize)]
struct VContent {
    parts: Vec<VPart>,
}
#[derive(Serialize)]
#[serde(untagged)]
enum VPart {
    Text { text: String },
    Inline {
        #[serde(rename = "inline_data")]
        inline_data: InlineData,
    },
}
#[derive(Serialize)]
struct InlineData {
    #[serde(rename = "mime_type")]
    mime_type: String,
    data: String,
}

/// Gemini vision: send a JPEG (base64) + an instruction, return the text answer. The dispatcher in
/// [`super::vision`] handles image encoding + provider selection. Requires `GEMINI_API_KEY`.
pub async fn describe_image_jpeg(instruction: &str, jpeg_b64: &str) -> Result<String> {
    let key = crate::config::Config::load()?
        .gemini_api_key
        .ok_or_else(|| anyhow!("GEMINI_API_KEY not set"))?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={key}",
        model()
    );
    let body = VReq {
        contents: vec![VContent {
            parts: vec![
                VPart::Text { text: instruction.into() },
                VPart::Inline {
                    inline_data: InlineData {
                        mime_type: "image/jpeg".into(),
                        data: jpeg_b64.to_string(),
                    },
                },
            ],
        }],
    };

    extract_text(post_generate(&url, &body).await?)
}
