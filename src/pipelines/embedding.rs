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
///
/// v0.31 phase 0: SDXL dual-encoder TIs ship two tensors in the
/// same file — a 768d `clip_l` and a 1280d `clip_g`. When the
/// parser sees both, `vectors_g` is populated; otherwise it's
/// `None` (SD 1.5 / SD 2.1 TIs, and CLIP-L-only SDXL TIs).
#[derive(Debug, Clone)]
pub struct ResolvedEmbedding {
    pub trigger: String,
    /// CLIP-L vectors. Always present. 768d for SD 1.5 + SDXL
    /// CLIP-L; 1024d for SD 2.1.
    pub vectors: Tensor,
    /// CLIP-G vectors. Only present for SDXL dual-encoder TIs
    /// (1280d). `None` for single-encoder TIs.
    pub vectors_g: Option<Tensor>,
    pub scale: f32,
}

/// Which half of a dual-encoder TI a merge call operates on.
/// SDXL CLIP-L and CLIP-G have independent tokenizers and
/// `token_embedding.weight` matrices; each gets its own pass
/// through the merger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingHalf {
    /// CLIP-L weights — `emb.vectors`. Used for every TI (single
    /// and dual). Single-encoder TIs only extend this half.
    ClipL,
    /// CLIP-G weights — `emb.vectors_g`. Used only on SDXL with
    /// dual-encoder TIs. Skips TIs whose `vectors_g` is `None`.
    ClipG,
}

impl ResolvedEmbedding {
    /// Number of vectors per token. Each becomes one new token in
    /// the vocab (e.g. `<trigger>_0`, `<trigger>_1`, ...).
    pub fn num_tokens(&self) -> Result<usize> {
        Ok(self.vectors.dim(0)?)
    }

    /// CLIP-L embedding dimension. Always available.
    pub fn embed_dim(&self) -> Result<usize> {
        Ok(self.vectors.dim(1)?)
    }

    /// `true` when this TI carries a CLIP-G half (SDXL dual format).
    pub fn has_clip_g(&self) -> bool {
        self.vectors_g.is_some()
    }

    /// CLIP-G embedding dimension. Errors when `vectors_g` is `None`.
    pub fn embed_dim_g(&self) -> Result<usize> {
        let g = self
            .vectors_g
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "embedding `{}` has no CLIP-G half — single-encoder TI",
                self.trigger
            ))?;
        Ok(g.dim(1)?)
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
    // v0.34 phase 3: Auto1111 two-files SDXL TI convention. Some
    // exports ship the CLIP-L and CLIP-G halves as SEPARATE files,
    // typically with `_clip_l.safetensors` + `_clip_g.safetensors`
    // suffixes. v0.31 phase 0's parser handled only the single-file
    // dual-key format (Civitai standard). Detection:
    //   - Path ends `_clip_l.safetensors` → try `_clip_g` companion.
    //     If found, stitch both halves into a dual ResolvedEmbedding.
    //   - Path ends `_clip_g.safetensors` → bail with helpful hint
    //     (we want the user to pass the `_clip_l` file as primary).
    //   - Otherwise → existing single-file logic.
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with("_clip_g.safetensors") {
        bail!(
            "embedding {} looks like the CLIP-G half of an Auto1111 \
             two-files SDXL TI. Pass the CLIP-L half (the `_clip_l.\
             safetensors` companion in the same directory) — the \
             parser will auto-discover the CLIP-G half from there.",
            path.display()
        );
    }
    if let Some(stem) = name.strip_suffix("_clip_l.safetensors") {
        let companion = path.with_file_name(format!("{stem}_clip_g.safetensors"));
        if companion.exists() {
            return parse_two_files_dual(path, &companion, stem, spec, device);
        }
        // Fall through — `_clip_l.safetensors` without a companion
        // is just a single-encoder TI with an unusual filename.
    }

    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading embedding safetensors {}", path.display()))?;

    if tensors.is_empty() {
        bail!(
            "embedding file {} has no tensors",
            path.display()
        );
    }

    // v0.31 phase 0: SDXL dual-encoder format. Files carry both
    // `clip_l` (768d, SDXL CLIP-L) and `clip_g` (1280d, SDXL
    // CLIP-G) top-level keys. Both halves apply during SDXL load.
    let has_clip_l = tensors.contains_key("clip_l");
    let has_clip_g = tensors.contains_key("clip_g");
    if has_clip_g && !has_clip_l {
        bail!(
            "embedding {} has a `clip_g` tensor without a `clip_l` companion. \
             SDXL dual-encoder TIs ship both halves in the same file. If \
             you have a CLIP-G-only TI, rename the tensor under `clip_l` \
             after confirming the embed_dim matches your SDXL CLIP-L (768).",
            path.display()
        );
    }
    if has_clip_l && has_clip_g {
        let vectors = tensors["clip_l"].clone();
        let vectors_g = tensors["clip_g"].clone();
        if vectors.rank() != 2 {
            bail!(
                "embedding {} CLIP-L tensor has rank {} — expected rank 2",
                path.display(),
                vectors.rank()
            );
        }
        if vectors_g.rank() != 2 {
            bail!(
                "embedding {} CLIP-G tensor has rank {} — expected rank 2",
                path.display(),
                vectors_g.rank()
            );
        }
        let n_l = vectors.dim(0)?;
        let n_g = vectors_g.dim(0)?;
        if n_l != n_g {
            bail!(
                "embedding {} dual-encoder TI mismatch: clip_l has {} vectors \
                 but clip_g has {}. Both halves must agree on token count \
                 (same trigger maps to the same N IDs in both encoders).",
                path.display(),
                n_l,
                n_g
            );
        }
        let trigger = spec
            .trigger
            .clone()
            .unwrap_or_else(|| derive_trigger_from_path(path));
        return Ok(ResolvedEmbedding {
            trigger,
            vectors,
            vectors_g: Some(vectors_g),
            scale: spec.scale,
        });
    }

    // Single-encoder TI (SD 1.5, SD 2.1, or CLIP-L-only SDXL).
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
        vectors_g: None,
        scale: spec.scale,
    })
}

/// v0.34 phase 3: Auto1111 two-files SDXL TI loader. Reads the
/// CLIP-L half from `clip_l_path` and the CLIP-G half from
/// `clip_g_path`, stitches them into a dual ResolvedEmbedding
/// matching the v0.31 phase 0 single-file dual format. `name_stem`
/// is the shared filename stem (without the `_clip_l` suffix);
/// used to derive the trigger when the spec didn't override it.
fn parse_two_files_dual(
    clip_l_path: &Path,
    clip_g_path: &Path,
    name_stem: &str,
    spec: &EmbeddingSpec,
    device: &Device,
) -> Result<ResolvedEmbedding> {
    // Each file should carry a single 2D tensor at the conventional
    // key (`clip_l` / `clip_g`), `emb_params`, or any 2D tensor.
    let l_tensors: HashMap<String, Tensor> =
        candle_core::safetensors::load(clip_l_path, device).with_context(|| {
            format!(
                "loading CLIP-L half of two-files SDXL TI: {}",
                clip_l_path.display()
            )
        })?;
    let g_tensors: HashMap<String, Tensor> =
        candle_core::safetensors::load(clip_g_path, device).with_context(|| {
            format!(
                "loading CLIP-G half of two-files SDXL TI: {}",
                clip_g_path.display()
            )
        })?;

    let l_vec = pick_single_2d_tensor(&l_tensors, "CLIP-L", clip_l_path)?;
    let g_vec = pick_single_2d_tensor(&g_tensors, "CLIP-G", clip_g_path)?;

    let n_l = l_vec.dim(0)?;
    let n_g = g_vec.dim(0)?;
    if n_l != n_g {
        bail!(
            "two-files SDXL TI mismatch: {} has {} vectors but {} has {}. \
             Both halves must agree on token count.",
            clip_l_path.display(),
            n_l,
            clip_g_path.display(),
            n_g,
        );
    }
    let l_dim = l_vec.dim(1)?;
    let g_dim = g_vec.dim(1)?;
    if l_dim != SD15_EMBED_DIM {
        bail!(
            "two-files SDXL TI: CLIP-L file {} has embed_dim {} — expected {} \
             (SDXL CLIP-L is the same 768d as SD 1.5).",
            clip_l_path.display(),
            l_dim,
            SD15_EMBED_DIM,
        );
    }
    if g_dim != SDXL_G_EMBED_DIM {
        bail!(
            "two-files SDXL TI: CLIP-G file {} has embed_dim {} — expected {} \
             (SDXL CLIP-G).",
            clip_g_path.display(),
            g_dim,
            SDXL_G_EMBED_DIM,
        );
    }

    let trigger = spec
        .trigger
        .clone()
        .unwrap_or_else(|| derive_trigger_stem(name_stem));

    Ok(ResolvedEmbedding {
        trigger,
        vectors: l_vec,
        vectors_g: Some(g_vec),
        scale: spec.scale,
    })
}

/// v0.34 phase 3: helper for the two-files loader. Each half's
/// safetensors should carry one 2D tensor — try the conventional
/// keys first, then fall back to the single 2D tensor in the file.
fn pick_single_2d_tensor(
    tensors: &HashMap<String, Tensor>,
    half_label: &str,
    path: &Path,
) -> Result<Tensor> {
    if tensors.is_empty() {
        bail!(
            "two-files SDXL TI: {} half {} has no tensors",
            half_label,
            path.display(),
        );
    }
    let preferred = ["clip_l", "clip_g", "emb_params", "string_to_param", "*"];
    if let Some(k) = preferred.iter().find(|k| tensors.contains_key(**k)) {
        let v = tensors[*k].clone();
        if v.rank() == 2 {
            return Ok(v);
        }
    }
    let two_d: Vec<&Tensor> = tensors.values().filter(|t| t.rank() == 2).collect();
    match two_d.as_slice() {
        [v] => Ok((*v).clone()),
        [] => bail!(
            "two-files SDXL TI: {} half {} has no 2D tensor",
            half_label,
            path.display(),
        ),
        many => bail!(
            "two-files SDXL TI: {} half {} has {} candidate 2D tensors. \
             Rename the file so its single 2D tensor lives under `emb_params`, \
             `clip_l`, or `clip_g`.",
            half_label,
            path.display(),
            many.len(),
        ),
    }
}

/// v0.34 phase 3: derive a trigger token from a filename stem
/// (without the `_clip_l` suffix). Same character-normalisation as
/// `derive_trigger_from_path` so the rendered token tokenises
/// consistently.
fn derive_trigger_stem(stem: &str) -> String {
    stem.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' })
        .collect()
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
    half: EmbeddingHalf,
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
             (mismatched SD variant — SD 1.5 / SDXL CLIP-L = 768, SD 2.1 = 1024, \
             SDXL CLIP-G = 1280)",
            base_dim,
            expected_embed_dim
        );
    }

    let mut all_new_rows: Vec<Tensor> = Vec::new();
    let mut registered = Vec::with_capacity(embeddings.len());
    let mut running_offset = 0usize;

    for emb in embeddings {
        // v0.31 phase 0: pick the half this pass operates on. CLIP-G
        // skips embeddings without a clip_g tensor (single-encoder
        // TIs only extend CLIP-L).
        let (vectors_src, emb_dim) = match half {
            EmbeddingHalf::ClipL => (Some(&emb.vectors), emb.embed_dim()?),
            EmbeddingHalf::ClipG => match emb.vectors_g.as_ref() {
                None => continue, // single-encoder TI — skip CLIP-G pass
                Some(v) => (Some(v), emb.embed_dim_g()?),
            },
        };
        let Some(vec_src) = vectors_src else { continue };

        if emb_dim != expected_embed_dim {
            bail!(
                "embedding `{}` ({:?} half) has dim {} but model expects {} — \
                 mismatched SD variant",
                emb.trigger,
                half,
                emb_dim,
                expected_embed_dim
            );
        }
        // Coerce the TI rows to the base token_embedding dtype +
        // device so the cat works cleanly.
        let rows = vec_src
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
    crate::pipelines::atomic_safetensors_save(&weights, out_path)
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
    // Explicit `repo#path/to/file.safetensors` → that exact file.
    if let Some((repo, file)) = src.split_once('#') {
        return crate::hf::download::get_file(repo, file)
            .await
            .with_context(|| format!("resolving embedding {src:?}"));
    }
    // Bare repo: TI repos don't share a filename convention — sd-concepts use
    // `learned_embeds.safetensors`, others name the file after the repo (e.g.
    // `gsdf/EasyNegative` → `EasyNegative.safetensors`). Try both before failing.
    let tail = src.rsplit('/').next().unwrap_or(src.as_str());
    let tail_file = format!("{tail}.safetensors");
    crate::hf::download::get_first_of(&[
        (src.as_str(), "learned_embeds.safetensors"),
        (src.as_str(), tail_file.as_str()),
    ])
    .await
    .with_context(|| {
        format!(
            "resolving embedding {src:?} (tried learned_embeds.safetensors + {tail_file}; \
             use repo#file.safetensors to name another)"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // -------------------------------------------------------------
    // v0.31 phase 0: SDXL dual-encoder TI parser.
    // -------------------------------------------------------------

    fn save_dual_ti(path: &Path, n: usize) {
        let mut map = HashMap::new();
        map.insert(
            "clip_l".to_string(),
            Tensor::ones((n, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "clip_g".to_string(),
            Tensor::ones((n, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, path).unwrap();
    }

    #[test]
    fn parse_sdxl_dual_ti_populates_both_halves() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        save_dual_ti(tmp.path(), 2);
        let spec: EmbeddingSpec = "ignored:dual-style".parse().unwrap();
        let r = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap();
        assert_eq!(r.trigger, "dual-style");
        assert!(r.has_clip_g(), "dual-format TI must populate vectors_g");
        assert_eq!(r.embed_dim().unwrap(), 768);
        assert_eq!(r.embed_dim_g().unwrap(), 1280);
        assert_eq!(r.num_tokens().unwrap(), 2);
    }

    #[test]
    fn parse_clip_g_only_bails_with_pointer() {
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let mut map = HashMap::new();
        map.insert(
            "clip_g".to_string(),
            Tensor::ones((1, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("clip_l` companion"), "got {msg}");
    }

    #[test]
    fn parse_dual_ti_token_count_mismatch_bails() {
        // clip_l with 2 vectors, clip_g with 3 — bail (both halves
        // must agree on N so the trigger maps to the same number
        // of new IDs in both encoders).
        let tmp = tempfile::Builder::new()
            .suffix(".safetensors")
            .tempfile()
            .unwrap();
        let mut map = HashMap::new();
        map.insert(
            "clip_l".to_string(),
            Tensor::ones((2, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        map.insert(
            "clip_g".to_string(),
            Tensor::ones((3, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&map, tmp.path()).unwrap();
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(tmp.path(), &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dual-encoder TI mismatch"), "got {msg}");
    }

    #[test]
    fn merge_clip_g_half_skips_single_encoder_tis() {
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
            Tensor::zeros((10, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&base, tmp_base.path()).unwrap();

        let single = ResolvedEmbedding {
            trigger: "single".into(),
            vectors: Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            vectors_g: None,
            scale: 1.0,
        };
        let dual = ResolvedEmbedding {
            trigger: "dual".into(),
            vectors: Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            vectors_g: Some(
                Tensor::ones((1, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            scale: 1.0,
        };

        let report = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[single, dual],
            1280,
            &Device::Cpu,
            EmbeddingHalf::ClipG,
        )
        .unwrap();

        // CLIP-G pass: only the dual TI extends. Original 10 + 1 = 11.
        assert_eq!(report.new_vocab_size, 11);
        assert_eq!(report.registered.len(), 1);
        assert_eq!(report.registered[0].trigger, "dual");
        assert_eq!(report.registered[0].base_token_id, 10);
    }

    #[test]
    fn merge_clip_g_bails_on_dim_mismatch() {
        // CLIP-G expects 1280; the dual TI's clip_g half is 768 — bail.
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
            Tensor::zeros((10, 1280), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        candle_core::safetensors::save(&base, tmp_base.path()).unwrap();

        let bad_dual = ResolvedEmbedding {
            trigger: "bad".into(),
            vectors: Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            vectors_g: Some(
                Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            ),
            scale: 1.0,
        };
        let err = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[bad_dual],
            1280,
            &Device::Cpu,
            EmbeddingHalf::ClipG,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dim 768"), "got {msg}");
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
            vectors_g: None,
            scale: 1.0,
        };
        let ti_b = ResolvedEmbedding {
            trigger: "style-b".to_string(),
            vectors: Tensor::ones((1, 768), candle_core::DType::F32, &Device::Cpu).unwrap(),
            vectors_g: None,
            scale: 0.5,
        };

        let report = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[ti_a, ti_b],
            768,
            &Device::Cpu,
            EmbeddingHalf::ClipL,
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
            vectors_g: None,
            scale: 1.0,
        };
        let err = merge_embeddings_into_te_weights(
            tmp_base.path(),
            tmp_out.path(),
            &[mismatched],
            768,
            &Device::Cpu,
            EmbeddingHalf::ClipL,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("dim 1024"), "got {msg}");
        assert!(msg.contains("768"), "got {msg}");
    }

    // ---------------------------------------------------------------
    // v0.34 phase 3: Auto1111 two-files SDXL TI loader.
    // ---------------------------------------------------------------

    /// Helper: write a safetensors file with a single 2D tensor at
    /// the given key. Returns the (path, tempdir) so the tempdir
    /// stays alive while tests use the file.
    fn write_ti_half(
        dir: &std::path::Path,
        filename: &str,
        key: &str,
        n: usize,
        dim: usize,
    ) -> std::path::PathBuf {
        use candle_core::DType;
        let path = dir.join(filename);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let v = Tensor::ones((n, dim), DType::F32, &Device::Cpu).unwrap();
        map.insert(key.to_string(), v);
        candle_core::safetensors::save(&map, &path).unwrap();
        path
    }

    #[test]
    fn two_files_dual_assembles_into_dual_resolved() {
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "mystyle_clip_l.safetensors", "clip_l", 2, 768);
        let _g_path = write_ti_half(dir.path(), "mystyle_clip_g.safetensors", "clip_g", 2, 1280);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let r = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap();
        assert_eq!(r.trigger, "mystyle");
        assert!(r.has_clip_g(), "dual half must populate vectors_g");
        assert_eq!(r.embed_dim().unwrap(), 768);
        assert_eq!(r.embed_dim_g().unwrap(), 1280);
        assert_eq!(r.num_tokens().unwrap(), 2);
    }

    #[test]
    fn two_files_dual_falls_back_when_no_companion() {
        // `_clip_l.safetensors` without a `_clip_g.safetensors`
        // companion is treated as a single-encoder TI with an
        // unusual filename — no error, no dual half.
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "lonely_clip_l.safetensors", "clip_l", 1, 768);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let r = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap();
        assert!(!r.has_clip_g(), "no companion → single-encoder TI");
        assert_eq!(r.embed_dim().unwrap(), 768);
    }

    #[test]
    fn two_files_dual_rejects_bare_clip_g_path() {
        let dir = tempfile::tempdir().unwrap();
        let g_path = write_ti_half(dir.path(), "concept_clip_g.safetensors", "clip_g", 1, 1280);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(&g_path, &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("two-files SDXL TI"), "got {msg}");
        assert!(msg.contains("CLIP-L"), "got {msg}");
        assert!(msg.contains("_clip_l"), "got {msg}");
    }

    #[test]
    fn two_files_dual_rejects_token_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "x_clip_l.safetensors", "clip_l", 2, 768);
        let _g_path = write_ti_half(dir.path(), "x_clip_g.safetensors", "clip_g", 3, 1280);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("mismatch"), "got {msg}");
        assert!(msg.contains("2"), "got {msg}");
        assert!(msg.contains("3"), "got {msg}");
    }

    #[test]
    fn two_files_dual_rejects_wrong_clip_l_dim() {
        let dir = tempfile::tempdir().unwrap();
        // CLIP-L stuffed with wrong dim (1024 instead of 768).
        let l_path = write_ti_half(dir.path(), "y_clip_l.safetensors", "clip_l", 1, 1024);
        let _g_path = write_ti_half(dir.path(), "y_clip_g.safetensors", "clip_g", 1, 1280);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CLIP-L"), "got {msg}");
        assert!(msg.contains("768"), "got {msg}");
        assert!(msg.contains("1024"), "got {msg}");
    }

    #[test]
    fn two_files_dual_rejects_wrong_clip_g_dim() {
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "z_clip_l.safetensors", "clip_l", 1, 768);
        // CLIP-G stuffed with wrong dim (768 instead of 1280).
        let _g_path = write_ti_half(dir.path(), "z_clip_g.safetensors", "clip_g", 1, 768);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let err = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CLIP-G"), "got {msg}");
        assert!(msg.contains("1280"), "got {msg}");
        assert!(msg.contains("768"), "got {msg}");
    }

    #[test]
    fn two_files_dual_derives_trigger_from_stem() {
        // Trigger derives from the shared filename stem, NOT from
        // the full filename (which would include `_clip_l`).
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "anime girl_clip_l.safetensors", "clip_l", 1, 768);
        let _g_path = write_ti_half(dir.path(), "anime girl_clip_g.safetensors", "clip_g", 1, 1280);
        let spec: EmbeddingSpec = "ignored".parse().unwrap();
        let r = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap();
        assert_eq!(r.trigger, "anime-girl");
    }

    #[test]
    fn two_files_dual_spec_trigger_overrides_stem() {
        let dir = tempfile::tempdir().unwrap();
        let l_path = write_ti_half(dir.path(), "stem_clip_l.safetensors", "clip_l", 1, 768);
        let _g_path = write_ti_half(dir.path(), "stem_clip_g.safetensors", "clip_g", 1, 1280);
        let spec: EmbeddingSpec = "ignored:my-override".parse().unwrap();
        let r = parse_safetensors(&l_path, &spec, &Device::Cpu).unwrap();
        assert_eq!(r.trigger, "my-override");
    }
}
