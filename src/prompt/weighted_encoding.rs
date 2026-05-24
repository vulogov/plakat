//! Generic per-token weight broadcast for A1111-style attention
//! syntax. Used by both CLIP (BPE, BOS+EOT) and T5 (sentencepiece,
//! no BOS, EOS only) tokenizer call sites — each provides its own
//! [`WeightedTokenConfig`] and the helper produces the same
//! `(ids, weights)` tensor pair the encoder integration multiplies
//! against per-token hidden states.
//!
//! Why a generic core: CLIP uses `<|startoftext|>` + `<|endoftext|>`
//! and a 77-token budget; T5 has no BOS, uses `</s>` as EOS, and
//! Flux/SD3 run it at 256 or 512 tokens. Same tokenize-each-segment
//! + per-token-row broadcast technique applies to both; only the
//! special-token IDs and budget differ.
//!
//! Sentencepiece alignment caveat: encoding a segment in isolation
//! may produce a slightly different subtoken split than encoding the
//! same text inside a longer string (the leading `▁` boundary marker
//! depends on the previous character). The weight-per-resulting-
//! subtoken contract is preserved either way — the visual effect of
//! `(token:1.5)` matches A1111's behaviour even when the subtoken
//! count drifts by 1.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use crate::prompt::a1111::WeightedSegment;

/// Per-call config that captures the encoder's special-token
/// conventions. CLIP: `bos_id: Some(<|startoftext|>)`,
/// `eos_id: <|endoftext|>`, `pad_id: <|endoftext|>` (or `!` for
/// SDXL CLIP-G). T5: `bos_id: None`, `eos_id: </s>` (id 1),
/// `pad_id: <pad>` (id 0).
pub struct WeightedTokenConfig<'a> {
    pub tokenizer: &'a Tokenizer,
    /// Sequence length to pad/truncate to.
    pub max_len: usize,
    /// BOS token. `None` for T5 and any other encoder that doesn't
    /// prepend one.
    pub bos_id: Option<u32>,
    /// EOS/EOT token, appended once at end of body.
    pub eos_id: u32,
    /// Padding token ID for filling out to `max_len`.
    pub pad_id: u32,
}

/// Tokenize pre-parsed `segments` into `(ids, weights)`:
/// - `ids`: `(1, max_len)` u32 tensor — fed straight to the encoder.
/// - `weights`: `(1, max_len, 1)` tensor in `dtype` — broadcast-
///   multiplies element-wise against the encoder's per-token hidden
///   states.
///
/// BOS / EOS / pad tokens all carry weight 1.0; only body tokens
/// carry their segment's weight.
pub fn tokenize_weighted(
    cfg: &WeightedTokenConfig,
    segments: &[WeightedSegment],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let max_len = cfg.max_len;
    let bos_reserved = if cfg.bos_id.is_some() { 1 } else { 0 };
    let body_budget = max_len.saturating_sub(bos_reserved + 1); // -1 for EOS

    let mut ids: Vec<u32> = Vec::with_capacity(max_len);
    let mut weights: Vec<f32> = Vec::with_capacity(max_len);

    if let Some(bos) = cfg.bos_id {
        ids.push(bos);
        weights.push(1.0);
    }

    let body_start = ids.len();
    'outer: for seg in segments {
        let seg_ids = cfg
            .tokenizer
            .encode(seg.text.as_str(), false)
            .map_err(|e| anyhow!("encode segment {:?}: {e}", seg.text))?
            .get_ids()
            .to_vec();
        for id in seg_ids {
            if ids.len() - body_start >= body_budget {
                break 'outer;
            }
            ids.push(id);
            weights.push(seg.weight);
        }
    }

    ids.push(cfg.eos_id);
    weights.push(1.0);

    while ids.len() < max_len {
        ids.push(cfg.pad_id);
        weights.push(1.0);
    }

    let ids_t = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
    let weights_t =
        Tensor::from_vec(weights, (1, max_len, 1), device)?.to_dtype(dtype)?;
    Ok((ids_t, weights_t))
}

/// Convenience: parse `text` for attention syntax, then tokenize.
pub fn tokenize_with_attention(
    cfg: &WeightedTokenConfig,
    text: &str,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let segments = crate::prompt::a1111::parse(text);
    tokenize_weighted(cfg, &segments, device, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use tokenizers::{models::wordpiece::WordPiece, Tokenizer};

    /// Build a toy whitespace WordPiece tokenizer with a known vocab.
    /// Lets us assert on exact token IDs without depending on HF
    /// network access.
    fn toy_tokenizer() -> Tokenizer {
        // Vocab maps known words to IDs; everything else falls back
        // to UNK.
        let vocab: std::collections::HashMap<String, u32> = [
            ("[UNK]", 0u32),
            ("<bos>", 1),
            ("<eos>", 2),
            ("<pad>", 3),
            ("a", 10),
            ("b", 11),
            ("c", 12),
            ("d", 13),
            ("hello", 20),
            ("world", 21),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let model = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
        t
    }

    #[test]
    fn t5_style_no_bos_uses_pad_id_zero() {
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 8,
            bos_id: None,
            eos_id: 2,
            pad_id: 3,
        };
        let segs = vec![WeightedSegment { text: "a b c".into(), weight: 1.5 }];
        let (ids, weights) = tokenize_weighted(&cfg, &segs, &Device::Cpu, DType::F32).unwrap();

        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        // No BOS at position 0 — body starts immediately. EOS at end
        // of body, then pad-3 fills the tail.
        assert_eq!(ids_v, vec![10, 11, 12, 2, 3, 3, 3, 3]);

        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        // a/b/c carry weight 1.5; EOS + pads carry 1.0.
        assert_eq!(w_v, vec![1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn clip_style_bos_eos_with_default_weight_pads() {
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 6,
            bos_id: Some(1),
            eos_id: 2,
            pad_id: 2, // CLIP often pads with <|endoftext|>
        };
        let segs = vec![WeightedSegment { text: "hello world".into(), weight: 1.0 }];
        let (ids, weights) = tokenize_weighted(&cfg, &segs, &Device::Cpu, DType::F32).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        // bos=1, hello=20, world=21, eos=2, pad=2, pad=2.
        assert_eq!(ids_v, vec![1, 20, 21, 2, 2, 2]);
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        // Everything weight 1.0 in this unweighted case.
        assert_eq!(w_v, vec![1.0; 6]);
    }

    #[test]
    fn multi_segment_weights_track_per_segment() {
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 8,
            bos_id: None,
            eos_id: 2,
            pad_id: 3,
        };
        let segs = vec![
            WeightedSegment { text: "a".into(), weight: 1.5 },
            WeightedSegment { text: "b c".into(), weight: 0.5 },
        ];
        let (_ids, weights) =
            tokenize_weighted(&cfg, &segs, &Device::Cpu, DType::F32).unwrap();
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        // a:1.5, b:0.5, c:0.5, EOS:1.0, pads:1.0.
        assert_eq!(w_v[0..3], [1.5, 0.5, 0.5]);
        assert_eq!(w_v[3], 1.0);
    }

    #[test]
    fn body_is_truncated_to_fit_budget() {
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 4, // budget = 4 - 1(EOS) = 3 body tokens
            bos_id: None,
            eos_id: 2,
            pad_id: 3,
        };
        let segs = vec![WeightedSegment { text: "a b c d".into(), weight: 1.0 }];
        let (ids, _w) =
            tokenize_weighted(&cfg, &segs, &Device::Cpu, DType::F32).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        // Only a/b/c fit; d is truncated; EOS appended.
        assert_eq!(ids_v, vec![10, 11, 12, 2]);
    }

    #[test]
    fn tokenize_with_attention_parses_then_encodes() {
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 8,
            bos_id: None,
            eos_id: 2,
            pad_id: 3,
        };
        let (_ids, weights) =
            tokenize_with_attention(&cfg, "(a:1.5) b", &Device::Cpu, DType::F32).unwrap();
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        // a weighted 1.5, b weighted 1.0 (no syntax around it), EOS 1.0.
        assert_eq!(w_v[0], 1.5);
        assert_eq!(w_v[1], 1.0);
    }

    // v0.18 phase 4 — negative-prompt regression suite. The negative
    // branch of CFG (SD 1.5 / 2.1, SDXL, SD3 / 3.5) calls the same
    // encode_with_attention / embed_xl / encode_prompt entry points
    // as the conditional branch, so the same WeightedTokenConfig
    // produces the same weight broadcast. These tests verify the
    // helper itself doesn't have any positive-branch-specific
    // assumption that would silently no-op the negative.

    #[test]
    fn negative_style_prompt_weights_multiple_segments() {
        // Typical anti-pattern negative — multiple weighted bad-quality
        // tokens. Same parse + tokenize path as positive prompts.
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 12,
            bos_id: Some(1),
            eos_id: 2,
            pad_id: 2,
        };
        // (a:1.5), [b:0.5], c → three segments, distinct weights.
        let (_ids, weights) =
            tokenize_with_attention(&cfg, "(a:1.5), [b:0.5], c", &Device::Cpu, DType::F32)
                .unwrap();
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        // BOS=1.0; "a"=1.5; comma+space coalesced into next seg, etc.
        // Easier assertion: every position in w_v that decodes to id 10
        // (token "a") must be 1.5; every position decoding to id 11
        // ("b") must be 0.5; every position to id 12 ("c") must be 1.0.
        let ids_v: Vec<u32> =
            tokenize_with_attention(&cfg, "(a:1.5), [b:0.5], c", &Device::Cpu, DType::F32)
                .unwrap()
                .0
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
        for (i, id) in ids_v.iter().enumerate() {
            match id {
                10 => assert!((w_v[i] - 1.5).abs() < 1e-6, "a at {i} should be 1.5"),
                11 => assert!((w_v[i] - 0.5).abs() < 1e-6, "b at {i} should be 0.5"),
                12 => assert!((w_v[i] - 1.0).abs() < 1e-6, "c at {i} should be 1.0"),
                _ => {} // BOS / EOS / pad — weight 1.0 by helper contract
            }
        }
    }

    #[test]
    fn empty_negative_yields_no_body_only_bos_eos_pads() {
        // The most common production case: --negative defaulted to "".
        // Must produce a valid (1, max_len) sequence with body length 0
        // — every position carries weight 1.0 — without panicking.
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 6,
            bos_id: Some(1),
            eos_id: 2,
            pad_id: 2,
        };
        let (ids, weights) =
            tokenize_with_attention(&cfg, "", &Device::Cpu, DType::F32).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        // BOS(1), EOS(2), pad(2), pad(2), pad(2), pad(2).
        assert_eq!(ids_v, vec![1, 2, 2, 2, 2, 2]);
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w_v, vec![1.0; 6]);
    }

    #[test]
    fn empty_negative_t5_style_no_bos() {
        // Same coverage on the T5 side — no BOS, pad=0.
        let tok = toy_tokenizer();
        let cfg = WeightedTokenConfig {
            tokenizer: &tok,
            max_len: 5,
            bos_id: None,
            eos_id: 2,
            pad_id: 3,
        };
        let (ids, weights) =
            tokenize_with_attention(&cfg, "", &Device::Cpu, DType::F32).unwrap();
        let ids_v: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
        // EOS at position 0 (no BOS), pads to max_len.
        assert_eq!(ids_v, vec![2, 3, 3, 3, 3]);
        let w_v: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(w_v, vec![1.0; 5]);
    }
}
