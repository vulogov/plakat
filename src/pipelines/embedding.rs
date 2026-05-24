//! v0.16 phase 9: Textual Inversion (embedding) support.
//!
//! Textual Inversion learns new "words" by training one or more
//! embedding vectors against a small image set. The output is a
//! tiny `.safetensors` file (typically 5-50 KB) holding either:
//!
//! * **A1111-style** — a single tensor (`emb_params`,
//!   `string_to_param`, or `*`) shaped `(N, embed_dim)` where
//!   `embed_dim` is 768 (SD 1.5), 1024 (SD 2.1), or 1280 (SDXL
//!   CLIP-G). `N` is the number of "vectors per token" — usually
//!   1-8.
//!
//! * **Diffusers-style** — top-level keys are the trigger token
//!   strings (e.g. `<my-concept>`), each carrying the vector(s).
//!
//! * **SDXL dual-encoder** — two top-level keys `clip_l` (768d) and
//!   `clip_g` (1280d). Both must be applied for SDXL to recognise
//!   the trigger.
//!
//! plakat's first cut targets SD 1.5 / SD 2.1 (single CLIP-L) and
//! bails loud for SDXL — its dual-encoder TI lands in a follow-up.
//! The runtime path mirrors LoRA's tempfile-merge approach: at
//! load time, append the TI rows to the text encoder's
//! `token_embedding.weight` and write a new safetensors that
//! VarBuilder mmaps. Tokenizer mutation adds the trigger tokens
//! at the matching new IDs.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// CLIP-L embedding dim for SD 1.5.
pub const SD15_EMBED_DIM: usize = 768;
/// CLIP-L embedding dim for SD 2.1 (different from SD 1.5).
pub const SD21_EMBED_DIM: usize = 1024;
/// CLIP-G embedding dim for SDXL.
pub const SDXL_G_EMBED_DIM: usize = 1280;

/// One Textual Inversion spec from the CLI:
/// `path_or_repo[:trigger][:scale]`.
#[derive(Debug, Clone)]
pub struct EmbeddingSpec {
    /// File path or HF repo spec. Resolved by [`resolve`] into a
    /// concrete local path.
    pub source: String,
    /// Trigger word the user wants to type in prompts. When unset,
    /// derived from the source filename (e.g.
    /// `embeddings/anime-girl.safetensors` → `anime-girl`).
    pub trigger: Option<String>,
    /// Scale multiplier on the TI vectors. `1.0` = standard; lower
    /// values weaken the concept's effect. Defaults to `1.0`.
    pub scale: f32,
}

impl std::str::FromStr for EmbeddingSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        // Format: PATH_OR_REPO[:trigger][:scale]
        //
        // The path itself may contain colons (Windows-style — unlikely
        // here but harmless) so we parse from the right: try the last
        // segment as scale, the second-to-last as trigger, the rest
        // as path. Common shapes:
        //   `path/to/foo.safetensors`           → (path, None, 1.0)
        //   `path/to/foo.safetensors:mytrigger` → (path, Some, 1.0)
        //   `path/to/foo.safetensors:mytrigger:0.7` → (path, Some, 0.7)
        //   `path/to/foo.safetensors:0.7`       → ambiguous; treat as scale
        let parts: Vec<&str> = s.split(':').collect();
        match parts.as_slice() {
            [single] => Ok(EmbeddingSpec {
                source: single.to_string(),
                trigger: None,
                scale: 1.0,
            }),
            [head @ .., tail] if tail.parse::<f32>().is_ok() => {
                let scale: f32 = tail.parse()?;
                if head.len() == 1 {
                    Ok(EmbeddingSpec {
                        source: head[0].to_string(),
                        trigger: None,
                        scale,
                    })
                } else {
                    let trigger = head[head.len() - 1].to_string();
                    let source = head[..head.len() - 1].join(":");
                    Ok(EmbeddingSpec {
                        source,
                        trigger: Some(trigger),
                        scale,
                    })
                }
            }
            head @ [_, _, ..] => Ok(EmbeddingSpec {
                source: head[..head.len() - 1].join(":"),
                trigger: Some(head[head.len() - 1].to_string()),
                scale: 1.0,
            }),
            _ => bail!("empty embedding spec"),
        }
    }
}

/// Resolved Textual Inversion ready to merge: trigger + the actual
/// `(N, embed_dim)` tensor. `N` is the number of "vectors per
/// token" — most TIs use 1, some use 2-8 for richer concepts.
#[derive(Debug, Clone)]
pub struct ResolvedEmbedding {
    pub trigger: String,
    pub vectors: Tensor,
    pub scale: f32,
}

impl ResolvedEmbedding {
    /// Number of vectors per token. Each becomes one new token in
    /// the vocab (e.g. `<trigger>_0`, `<trigger>_1`, ...).
    pub fn num_tokens(&self) -> Result<usize> {
        Ok(self.vectors.dim(0)?)
    }

    /// Embedding dimension — must match the model's CLIP embed_dim.
    pub fn embed_dim(&self) -> Result<usize> {
        Ok(self.vectors.dim(1)?)
    }
}

/// Parse a TI safetensors file into a single resolved embedding.
/// `spec.trigger` overrides the filename-derived default.
///
/// Returns the embedding ready to merge. The vectors are NOT yet
/// scaled — that happens during merge so the caller can preview the
/// raw tensor first.
pub fn parse_safetensors(
    path: &Path,
    spec: &EmbeddingSpec,
    device: &Device,
) -> Result<ResolvedEmbedding> {
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading embedding safetensors {}", path.display()))?;

    if tensors.is_empty() {
        bail!(
            "embedding file {} has no tensors",
            path.display()
        );
    }

    // Reject SDXL dual-encoder format for now — needs a CLIP-G
    // counterpart merge too, which the v0.16 phase 9 scope skips.
    if tensors.contains_key("clip_g") {
        bail!(
            "embedding {} has a `clip_g` tensor — SDXL dual-encoder TIs aren't \
             wired yet (v0.16 phase 9 is SD 1.5 / SD 2.1 only). Use the CLIP-L \
             half via a LoRA-format SDXL embedding, or check if a LoRA \
             replacement exists.",
            path.display()
        );
    }

    // Pick the vector tensor. Prefer well-known A1111 keys; fall
    // back to the only/single 2D tensor in the file.
    let preferred_keys = ["emb_params", "string_to_param", "clip_l", "*"];
    let chosen_key = preferred_keys
        .iter()
        .find(|k| tensors.contains_key(**k))
        .copied();

    let (key, vectors) = if let Some(k) = chosen_key {
        (k.to_string(), tensors[k].clone())
    } else {
        // Find the single 2D tensor.
        let two_d: Vec<&String> = tensors
            .iter()
            .filter(|(_, t)| t.rank() == 2)
            .map(|(k, _)| k)
            .collect();
        match two_d.as_slice() {
            [k] => ((*k).clone(), tensors[*k].clone()),
            [] => bail!(
                "embedding {} has no 2D tensor (need shape `(N, embed_dim)`)",
                path.display()
            ),
            many => bail!(
                "embedding {} has {} candidate 2D tensors ({}); not sure which to \
                 use. Try renaming the safetensors with a single tensor under \
                 the key `emb_params`.",
                path.display(),
                many.len(),
                many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    };

    if vectors.rank() != 2 {
        bail!(
            "embedding {} tensor `{}` has rank {} — expected rank 2 (N, embed_dim)",
            path.display(),
            key,
            vectors.rank()
        );
    }

    let trigger = spec
        .trigger
        .clone()
        .unwrap_or_else(|| derive_trigger_from_path(path));

    Ok(ResolvedEmbedding {
        trigger,
        vectors,
        scale: spec.scale,
    })
}

/// Default trigger word from the file path. Strips the extension
/// and replaces non-alphanumeric chars with `-` so the resulting
/// token tokenises consistently:
///   `embeddings/Anime Girl v2.safetensors` → `Anime-Girl-v2`
pub fn derive_trigger_from_path(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("ti");
    stem.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' })
        .collect()
}

/// v0.16 phase 9: merge a stack of TIs into the SD CLIP-L text
/// encoder safetensors. Mirrors `lora::merge_loras_into_weights` —
/// reads the base safetensors, extends
/// `text_model.embeddings.token_embedding.weight` by appending the
/// TI rows, and writes to `out_path`.
///
/// Returns `MergeReport { new_vocab_size, registered: Vec<(trigger, base_token_id, num_tokens)> }`
/// — the caller bumps `Config.vocab_size` to `new_vocab_size` and
/// passes the registration list to the tokenizer mutator.
pub fn merge_embeddings_into_te_weights(
    base_path: &Path,
    out_path: &Path,
    embeddings: &[ResolvedEmbedding],
    expected_embed_dim: usize,
    device: &Device,
) -> Result<MergeReport> {
    let mut weights: HashMap<String, Tensor> = candle_core::safetensors::load(base_path, device)
        .with_context(|| format!("loading text encoder weights {}", base_path.display()))?;

    let token_emb_key = "text_model.embeddings.token_embedding.weight";
    let base_token_emb = weights
        .get(token_emb_key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!(
            "text encoder safetensors {} has no `{token_emb_key}` — wrong file?",
            base_path.display()
        ))?;
    let (orig_vocab_size, base_dim) = base_token_emb.dims2()?;

    if base_dim != expected_embed_dim {
        bail!(
            "TI merge: base token embedding has dim {} but model expects {} \
             (mismatched SD variant — SD 1.5 = 768, SD 2.1 = 1024)",
            base_dim,
            expected_embed_dim
        );
    }

    let mut all_new_rows: Vec<Tensor> = Vec::new();
    let mut registered = Vec::with_capacity(embeddings.len());
    let mut running_offset = 0usize;

    for emb in embeddings {
        let emb_dim = emb.embed_dim()?;
        if emb_dim != expected_embed_dim {
            bail!(
                "embedding `{}` has dim {} but model expects {} — mismatched \
                 SD variant (use an SD 1.5 TI on --model sd15, SD 2.1 on sd21, \
                 etc.)",
                emb.trigger,
                emb_dim,
                expected_embed_dim
            );
        }
        // Coerce the TI rows to the base token_embedding dtype +
        // device so the cat works cleanly.
        let rows = emb
            .vectors
            .to_dtype(base_token_emb.dtype())?
            .to_device(base_token_emb.device())?;
        // Apply user-supplied scale (default 1.0).
        let scaled = if (emb.scale - 1.0).abs() > f32::EPSILON {
            (rows * emb.scale as f64)?
        } else {
            rows
        };
        let n_tokens = scaled.dim(0)?;
        all_new_rows.push(scaled);
        registered.push(EmbeddingRegistration {
            trigger: emb.trigger.clone(),
            base_token_id: (orig_vocab_size + running_offset) as u32,
            num_tokens: n_tokens,
        });
        running_offset += n_tokens;
    }

    // Concatenate base + all new rows along dim 0 (vocab).
    let mut cat_inputs: Vec<&Tensor> = Vec::with_capacity(all_new_rows.len() + 1);
    cat_inputs.push(&base_token_emb);
    for r in &all_new_rows {
        cat_inputs.push(r);
    }
    let extended = Tensor::cat(&cat_inputs, 0)?;
    let new_vocab_size = orig_vocab_size + running_offset;
    debug_assert_eq!(extended.dim(0)?, new_vocab_size);

    weights.insert(token_emb_key.to_string(), extended);
    candle_core::safetensors::save(&weights, out_path)
        .with_context(|| format!("writing extended text encoder to {}", out_path.display()))?;
    tracing::info!(
        target: "plakat",
        "TI merged {} embedding(s) into text encoder: vocab {} → {}",
        registered.len(),
        orig_vocab_size,
        new_vocab_size
    );
    Ok(MergeReport {
        new_vocab_size,
        registered,
    })
}

/// Result of [`merge_embeddings_into_te_weights`].
#[derive(Debug, Clone)]
pub struct MergeReport {
    /// New `vocab_size` for the CLIP Config. The caller must
    /// update its `Config.vocab_size` to this value before
    /// constructing the text encoder, otherwise candle's
    /// `embedding(vocab_size, embed_dim, ...)` build will reject
    /// the larger weight matrix.
    pub new_vocab_size: usize,
    /// One entry per merged embedding. The `base_token_id` is the
    /// vocab ID of the FIRST new token from this embedding; the
    /// remaining `num_tokens - 1` follow sequentially.
    pub registered: Vec<EmbeddingRegistration>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingRegistration {
    pub trigger: String,
    pub base_token_id: u32,
    pub num_tokens: usize,
}

impl EmbeddingRegistration {
    /// The token strings the tokenizer should map to the new IDs.
    /// For an N-token embedding with `trigger="my-style"`:
    ///   N == 1 → ["my-style"]
    ///   N >  1 → ["my-style", "my-style_1", ..., "my-style_{N-1}"]
    pub fn token_strings(&self) -> Vec<String> {
        if self.num_tokens == 1 {
            vec![self.trigger.clone()]
        } else {
            std::iter::once(self.trigger.clone())
                .chain(
                    (1..self.num_tokens).map(|i| format!("{}_{i}", self.trigger))
                )
                .collect()
        }
    }
}

/// v0.16 phase 9: resolve an `EmbeddingSpec` into a local path.
/// Mirrors `lora::resolve` — supports local paths, HF repos (with
/// optional `#path/to/file.safetensors`), and Civitai cache paths
/// (treated as local since the user already downloaded them).
pub async fn resolve(spec: &EmbeddingSpec) -> Result<PathBuf> {
    let src = &spec.source;
    // Local path? Check first; HF repos are typically `org/name`
    // with no slash count > 1 (and no `.` in the org name), but
    // accept any existing path as local for simplicity.
    let as_path = PathBuf::from(src);
    if as_path.exists() {
        return Ok(as_path);
    }
    // Treat as HF repo. Supports `repo#path/to/file.safetensors`
    // for explicit file selection.
    let (repo, file) = if let Some((r, f)) = src.split_once('#') {
        (r.to_string(), f.to_string())
    } else {
        // Default to common filenames in TI repos.
        (src.to_string(), "learned_embeds.safetensors".to_string())
    };
    crate::hf::download::get_file(&repo, &file)
        .await
        .with_context(|| format!("resolving embedding {src:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn spec_path_only() {
        let s: EmbeddingSpec = "foo.safetensors".parse().unwrap();
        assert_eq!(s.source, "foo.safetensors");
        assert!(s.trigger.is_none());
        assert!((s.scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn spec_path_with_trigger() {
        let s: EmbeddingSpec = "foo.safetensors:mytrigger".parse().unwrap();
        assert_eq!(s.source, "foo.safetensors");
        assert_eq!(s.trigger.as_deref(), Some("mytrigger"));
        assert!((s.scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn spec_path_with_trigger_and_scale() {
        let s: EmbeddingSpec = "foo.safetensors:mytrigger:0.7".parse().unwrap();
        assert_eq!(s.source, "foo.safetensors");
        assert_eq!(s.trigger.as_deref(), Some("mytrigger"));
        assert!((s.scale - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn spec_path_with_scale_only_treats_trailing_number_as_scale() {
        let s: EmbeddingSpec = "foo.safetensors:0.5".parse().unwrap();
        assert_eq!(s.source, "foo.safetensors");
        assert!(s.trigger.is_none(), "trailing number should be scale, not trigger");
        assert!((s.scale - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn spec_hf_repo() {
        let s: EmbeddingSpec = "username/my-embedding:mytrigger".parse().unwrap();
        assert_eq!(s.source, "username/my-embedding");
        assert_eq!(s.trigger.as_deref(), Some("mytrigger"));
    }

    #[test]
    fn derive_trigger_handles_spaces_and_punct() {
        let p = PathBuf::from("embeddings/Anime Girl v2.safetensors");
        assert_eq!(derive_trigger_from_path(&p), "Anime-Girl-v2");
        let p2 = PathBuf::from("embeddings/clean_name.safetensors");
        assert_eq!(derive_trigger_from_path(&p2), "clean_name");
    }

    #[test]
    fn registration_token_strings_single_token() {
        let r = EmbeddingRegistration {
            trigger: "my-style".to_string(),
            base_token_id: 100,
            num_tokens: 1,
        };
        assert_eq!(r.token_strings(), vec!["my-style".to_string()]);
    }

    #[test]
    fn registration_token_strings_multi_token() {
        let r = EmbeddingRegistration {
            trigger: "my-style".to_string(),
            base_token_id: 100,
            num_tokens: 4,
        };
        assert_eq!(
            r.token_strings(),
            vec![
                "my-style".to_string(),
                "my-style_1".to_string(),
                "my-style_2".to_string(),
                "my-style_3".to_string(),
            ]
        );
    }

    #[test]
    fn parse_safetensors_rejects_sdxl_dual_format() {
        // Build a minimal safetensors with a `clip_g` key →
        // should bail with the SDXL-not-supported message.
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let clip_l = Tensor::zeros((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap();
        let clip_g = Tensor::zeros((1, 1280), candle_core::DType::F32, &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("clip_l".to_string(), clip_l);
        map.insert("clip_g".to_string(), clip_g);
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec = EmbeddingSpec::from_str("ignored").unwrap();
        let err = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("SDXL"), "got {msg}");
    }

    #[test]
    fn parse_safetensors_picks_emb_params_key() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        // A1111 canonical key.
        let mut map = HashMap::new();
        let v = Tensor::ones((2, 768), candle_core::DType::F32, &Device::Cpu).unwrap();
        map.insert("emb_params".to_string(), v);
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec: EmbeddingSpec = "ignored:my-style".parse().unwrap();
        let r = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap();
        assert_eq!(r.trigger, "my-style");
        assert_eq!(r.num_tokens().unwrap(), 2);
        assert_eq!(r.embed_dim().unwrap(), 768);
    }

    #[test]
    fn parse_safetensors_picks_single_2d_tensor_fallback() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        // Some random key — should fall through to "the only 2D tensor".
        let mut map = HashMap::new();
        let v = Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap();
        map.insert("MyConcept".to_string(), v);
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let r = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap();
        // Trigger defaults to the file stem (random tempfile name).
        assert_eq!(r.num_tokens().unwrap(), 1);
        assert_eq!(r.embed_dim().unwrap(), 768);
    }

    #[test]
    fn parse_safetensors_ambiguous_multi_2d_bails() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let v1 = Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("a".to_string(), v1);
        map.insert("b".to_string(), v2);
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("candidate 2D tensors"), "got {msg}");
    }

    #[test]
    fn merge_extends_vocab_and_records_registrations() {
        let tmp_base = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let tmp_out = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();

        // Build a minimal base TE safetensors with the canonical
        // token_embedding key (vocab=10 for testing brevity; real
        // CLIP is 49408).
        let mut base = HashMap::new();
        base.insert(
            "text_model.embeddings.token_embedding.weight".to_string(),
            Tensor::zeros((10, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&base, tmp_base.path()).unwrap();

        // Two TIs: one 2-token, one 1-token.
        let ti_a = ResolvedEmbedding {
            trigger: "style-a".to_string(),
            vectors: Tensor::ones((2, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            scale: 1.0,
        };
        let ti_b = ResolvedEmbedding {
            trigger: "style-b".to_string(),
            vectors: Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            scale: 0.5,
        };

        let report = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[ti_a, ti_b],
            768,
            &Device::Cpu,
        )
        .unwrap();

        // Original 10 + 2 + 1 = 13.
        assert_eq!(report.new_vocab_size, 13);
        assert_eq!(report.registered.len(), 2);
        assert_eq!(report.registered[0].trigger, "style-a");
        assert_eq!(report.registered[0].base_token_id, 10);
        assert_eq!(report.registered[0].num_tokens, 2);
        assert_eq!(report.registered[1].trigger, "style-b");
        assert_eq!(report.registered[1].base_token_id, 12);
        assert_eq!(report.registered[1].num_tokens, 1);

        // Verify the output file has the extended embedding.
        let out: HashMap<String, Tensor> =
            candle_core::safetensors::load(tmp_out.path(), &Device::Cpu).unwrap();
        let ext = &out["text_model.embeddings.token_embedding.weight"];
        assert_eq!(ext.dims2().unwrap(), (13, 768));
    }

    #[test]
    fn merge_bails_on_dim_mismatch() {
        let tmp_base = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let tmp_out = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let mut base = HashMap::new();
        base.insert(
            "text_model.embeddings.token_embedding.weight".to_string(),
            Tensor::zeros((10, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&base, tmp_base.path()).unwrap();

        // SD 2.1-dim TI (1024) against an SD 1.5-dim base (768).
        let mismatched = ResolvedEmbedding {
            trigger: "x".into(),
            vectors: Tensor::ones((1, 1024), candle_core::DType::F32, &Device::Cpu).unwrap(),
            scale: 1.0,
        };
        let err = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[mismatched],
            768,
            &Device::Cpu,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dim 1024"), "got {msg}");
        assert!(msg.contains("768"), "got {msg}");
    }
}
