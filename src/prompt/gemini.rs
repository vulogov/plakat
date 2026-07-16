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

/// Send an image + an instruction to Gemini vision and return the text answer. The image is
/// re-encoded to JPEG at ≤1024 px before upload (smaller payload, uniform mime). Requires
/// `GEMINI_API_KEY`.
pub async fn describe_image(image_path: &std::path::Path, instruction: &str) -> Result<String> {
    let key = crate::config::Config::load()?
        .gemini_api_key
        .ok_or_else(|| anyhow!("GEMINI_API_KEY not set"))?;

    // Downscale + re-encode to JPEG in memory.
    let img = image::open(image_path)?;
    let img = if img.width().max(img.height()) > 1024 {
        img.resize(1024, 1024, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    );
    let body = VReq {
        contents: vec![VContent {
            parts: vec![
                VPart::Text { text: instruction.into() },
                VPart::Inline {
                    inline_data: InlineData {
                        mime_type: "image/jpeg".into(),
                        data: base64_encode(&buf),
                    },
                },
            ],
        }],
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

/// Standard base64 (no line breaks) — small enough to inline rather than add a crate.
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64_encode;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
