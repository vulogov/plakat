//! Vision + AI metadata (RFC PHOTOS-1 Phase 7) — describe/tag an image with Gemini vision and write
//! the result into its `album.hjson` record, so the whole library becomes searchable (feeds the
//! metadata search in [`crate::textsearch`]). Requires `GEMINI_API_KEY`. Network op, run off the UI
//! thread by the parent module.

use std::path::Path;

use anyhow::Result;

/// A vision request against one image.
#[derive(Clone, Copy, Debug)]
pub enum VisionOp {
    /// Produce a set of content/style tags.
    Autotag,
    /// Produce a one-sentence caption.
    Describe,
}

impl VisionOp {
    pub fn label(self) -> &'static str {
        match self {
            VisionOp::Autotag => "autotag",
            VisionOp::Describe => "describe",
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            VisionOp::Autotag => {
                "Look at this image and produce 5 to 12 short lowercase keyword tags describing its \
                 content, subjects, setting, colours, and style. Respond with ONLY a comma-separated \
                 list of tags — no numbering, no sentences, no extra text."
            }
            VisionOp::Describe => {
                "Write a single vivid caption of at most 20 words describing this image. Respond with \
                 ONLY the sentence — no quotes, no prefix."
            }
        }
    }
}

/// The parsed outcome of a vision call, ready to merge into an [`crate::photos::hjson::ImageRecord`].
pub enum VisionOutcome {
    Tags(Vec<String>),
    Caption(String),
}

/// Run `op` on `image` via Gemini vision, returning the parsed outcome.
pub async fn run(op: VisionOp, image: &Path) -> Result<VisionOutcome> {
    let text = crate::prompt::gemini::describe_image(image, op.instruction()).await?;
    Ok(match op {
        VisionOp::Autotag => VisionOutcome::Tags(parse_tag_reply(&text)),
        VisionOp::Describe => VisionOutcome::Caption(clean_caption(&text)),
    })
}

/// Parse a comma-separated tag reply into trimmed, lowercased, de-duplicated tags (≤ 12).
pub fn parse_tag_reply(reply: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in reply.replace(['\n', ';'], ",").split(',') {
        let t = raw.trim().trim_matches(['.', '"', '#', '-', '*', ' ']).to_lowercase();
        if !t.is_empty() && t.len() <= 40 && !out.contains(&t) {
            out.push(t);
        }
        if out.len() >= 12 {
            break;
        }
    }
    out
}

/// Trim a caption reply to a single clean line.
fn clean_caption(reply: &str) -> String {
    reply.lines().next().unwrap_or(reply).trim().trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_dedups_tag_replies() {
        let r = parse_tag_reply("Sunset, beach, \"palm trees\", sunset, ocean.\nwarm colors");
        assert_eq!(r, ["sunset", "beach", "palm trees", "ocean", "warm colors"]);
        // Bulleted / numbered noise is stripped.
        assert_eq!(parse_tag_reply("- cat\n- dog"), ["cat", "dog"]);
    }

    #[test]
    fn caption_is_single_clean_line() {
        assert_eq!(clean_caption("\"A red fox in snow.\"\nextra"), "A red fox in snow.");
    }
}
