//! Provider-agnostic image→text (vision). Routes a "look at this image and answer" request to any
//! configured LLM provider that supports vision — Gemini (native) or an OpenAI-compatible endpoint
//! like DeepSeek (image_url data URI). The image is re-encoded to JPEG ≤1024 px once here, then the
//! same base64 is handed to whichever provider is selected. Text-only providers (the local LLM)
//! return a clear "no vision model" error.

use std::path::Path;

use anyhow::{anyhow, Result};

/// Describe/analyze `image` with `instruction` using `provider` (`gemini` / `deepseek` / `auto` /
/// `local`). `auto` resolves to a vision-capable configured provider.
pub async fn describe_image(provider: &str, image: &Path, instruction: &str) -> Result<String> {
    let b64 = image_jpeg_base64(image)?;
    match resolve_vision_provider(provider).as_str() {
        "gemini" => super::gemini::describe_image_jpeg(instruction, &b64).await,
        "deepseek" => super::deepseek::describe_image_jpeg(instruction, &b64).await,
        other if other.starts_with("local") => Err(anyhow!(
            "the local LLM has no vision model — configure a vision provider \
             (set GEMINI_API_KEY, or DEEPSEEK_API_KEY for an OpenAI-compatible vision endpoint)"
        )),
        other => Err(anyhow!("provider {other:?} does not support image analysis")),
    }
}

/// Resolve the vision provider. Explicit names pass through; `auto` prefers a **vision-capable**
/// configured provider — Gemini first (reliable vision), then DeepSeek, else `local` (which errors
/// with a helpful message).
pub fn resolve_vision_provider(provider: &str) -> String {
    if !provider.eq_ignore_ascii_case("auto") {
        return provider.to_string();
    }
    let cfg = crate::config::Config::load().ok();
    if cfg.as_ref().is_some_and(|c| c.gemini_api_key.is_some()) {
        return "gemini".into();
    }
    if cfg.as_ref().is_some_and(|c| c.deepseek_api_key.is_some()) {
        return "deepseek".into();
    }
    "local".into()
}

/// Load `path`, downscale so the longer side is ≤ 1024 px, and re-encode to JPEG as base64 (uniform
/// payload for every provider).
pub fn image_jpeg_base64(path: &Path) -> Result<String> {
    let img = image::open(path)?;
    let img = if img.width().max(img.height()) > 1024 {
        img.resize(1024, 1024, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
    Ok(base64_encode(&buf))
}

/// Standard base64 (no line breaks) — small enough to inline rather than add a crate.
pub fn base64_encode(data: &[u8]) -> String {
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
    use super::*;

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

    #[test]
    fn explicit_providers_pass_through() {
        assert_eq!(resolve_vision_provider("gemini"), "gemini");
        assert_eq!(resolve_vision_provider("deepseek"), "deepseek");
        assert_eq!(resolve_vision_provider("local"), "local");
        // `auto` resolves to one of the known providers (depends on configured keys).
        assert!(["gemini", "deepseek", "local"].contains(&resolve_vision_provider("auto").as_str()));
    }
}
