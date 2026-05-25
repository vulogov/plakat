//! Local LLM runtime — loads a quantized GGUF model + tokenizer
//! once and runs prompt-enhancement decode loops on demand.
//!
//! Lifecycle:
//!
//! 1. First `enhance` call resolves the alias, downloads the
//!    GGUF + tokenizer (HF cache aware), and constructs an
//!    [`Enhancer`].
//! 2. The instance is stashed in a process-wide
//!    `tokio::sync::Mutex<Option<…>>` keyed by `(alias, device)`
//!    so subsequent calls reuse the loaded weights — important
//!    for `plakat scenario` runs that enhance dozens of prompts
//!    back-to-back.
//! 3. `enhance` formats the chat template, decodes greedily up to
//!    `max_new_tokens` (default 96), and runs the output through
//!    `templates::sanitize`. A refusal / empty response surfaces
//!    as `Err(EnhanceError::Refused)` so the caller can fall back
//!    to the user's original prompt rather than feed a refusal
//!    into the diffusion encoder.
//!
//! Decoding defaults to greedy (`temperature = 0.0`). The
//! reproducibility-by-default stance matches the rest of plakat —
//! a fixed prompt + fixed seed produces a fixed image, with or
//! without `--enhance local`.

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{quantized_llama, quantized_qwen2};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::llm::aliases::{self, Family, ModelDescriptor};
use crate::llm::templates;

/// Decode-time options. `seed` lets callers reproduce a temperature
/// > 0 sample; greedy decoding (the default) ignores it.
#[derive(Debug, Clone, Copy)]
pub struct EnhanceOpts {
    pub seed: u64,
    pub temperature: f64,
    pub max_new_tokens: usize,
}

impl Default for EnhanceOpts {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 0.0, // greedy
            max_new_tokens: 96,
        }
    }
}

/// Errors specific to the enhance path. Carries the original
/// prompt so the caller can fall back without re-passing it.
#[derive(Debug)]
pub enum EnhanceError {
    /// Model produced a refusal / empty response. Caller should
    /// log a warning and use the un-enhanced prompt.
    Refused,
    /// Anything else — model load failure, tokenizer mismatch,
    /// I/O error during GGUF read.
    Other(anyhow::Error),
}

impl std::fmt::Display for EnhanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused => write!(f, "model refused / produced empty output"),
            Self::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EnhanceError {}

impl From<anyhow::Error> for EnhanceError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

/// One loaded LLM. Held inside the global cache so scenarios reuse
/// the same weights across all their tasks.
pub struct Enhancer {
    pub descriptor: &'static ModelDescriptor,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

enum ModelWeights {
    Qwen2(quantized_qwen2::ModelWeights),
    Llama(quantized_llama::ModelWeights),
}

impl ModelWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        Ok(match self {
            Self::Qwen2(m) => m.forward(x, index_pos)?,
            Self::Llama(m) => m.forward(x, index_pos)?,
        })
    }
}

impl Enhancer {
    /// Download (cache-aware) + load the GGUF + tokenizer for the
    /// given alias on the given device.
    pub async fn load(alias: &str, device: Device) -> Result<Self> {
        let descriptor = aliases::resolve(alias).ok_or_else(|| {
            anyhow!(
                "unknown enhance model {alias:?}; supported: {}",
                aliases::supported_aliases()
            )
        })?;

        let gguf_path =
            crate::hf::download::get_file(descriptor.gguf_repo, descriptor.gguf_file)
                .await
                .with_context(|| {
                    format!(
                        "downloading {} from {}",
                        descriptor.gguf_file, descriptor.gguf_repo
                    )
                })?;
        let tokenizer_path =
            crate::hf::download::get_file(descriptor.tokenizer_repo, "tokenizer.json")
                .await
                .with_context(|| {
                    format!(
                        "downloading tokenizer.json from {}",
                        descriptor.tokenizer_repo
                    )
                })?;

        let weights = load_weights(descriptor.family, &gguf_path, &device)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer from {}: {e}", tokenizer_path.display()))?;

        // EOS varies per family. Look up by the canonical token
        // string the tokenizer's vocab uses.
        let eos_token_id = match descriptor.family {
            Family::Qwen2 => tokenizer
                .token_to_id("<|im_end|>")
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .ok_or_else(|| anyhow!("Qwen tokenizer missing <|im_end|>"))?,
            Family::Llama => tokenizer
                .token_to_id("<|im_end|>")
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .or_else(|| tokenizer.token_to_id("</s>"))
                .ok_or_else(|| anyhow!("Llama tokenizer missing EOS token"))?,
        };

        Ok(Self {
            descriptor,
            weights,
            tokenizer,
            device,
            eos_token_id,
        })
    }

    /// Run one prompt through the model and return the sanitized
    /// rewritten text. Greedy decoding by default; bumping
    /// `opts.temperature` enables sampling (seeded for
    /// reproducibility).
    pub fn enhance(
        &mut self,
        system: &str,
        user: &str,
        opts: EnhanceOpts,
    ) -> Result<String, EnhanceError> {
        let formatted = templates::format(self.descriptor.family, system, user);
        let prompt_tokens = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| EnhanceError::Other(anyhow!("tokenize prompt: {e}")))?
            .get_ids()
            .to_vec();

        let temperature = if opts.temperature <= 0.0 {
            None
        } else {
            Some(opts.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(opts.seed, temperature, None);

        let mut all_tokens = prompt_tokens.clone();
        let mut generated: Vec<u32> = Vec::with_capacity(opts.max_new_tokens);
        let mut index_pos = 0usize;
        for index in 0..opts.max_new_tokens {
            let context_size = if index > 0 { 1 } else { all_tokens.len() };
            let ctx_start = all_tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&all_tokens[ctx_start..], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| EnhanceError::Other(e.into()))?;
            let logits = self
                .weights
                .forward(&input, index_pos)
                .map_err(EnhanceError::Other)?;
            let logits = logits
                .squeeze(0)
                .and_then(|t| {
                    if t.dims().len() == 2 {
                        let last = t.dim(0).map(|n| n.saturating_sub(1))?;
                        t.i(last)
                    } else {
                        Ok(t)
                    }
                })
                .map_err(|e| EnhanceError::Other(e.into()))?;
            let next_token = logits_processor
                .sample(&logits)
                .map_err(|e| EnhanceError::Other(e.into()))?;
            index_pos += context_size;
            if next_token == self.eos_token_id {
                break;
            }
            all_tokens.push(next_token);
            generated.push(next_token);
        }

        let raw = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| EnhanceError::Other(anyhow!("decode output: {e}")))?;
        templates::sanitize(&raw).ok_or(EnhanceError::Refused)
    }
}

fn load_weights(
    family: Family,
    gguf_path: &PathBuf,
    device: &Device,
) -> Result<ModelWeights> {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::open(gguf_path)
        .with_context(|| format!("opening {}", gguf_path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("reading GGUF header from {}: {e}", gguf_path.display()))?;
    match family {
        Family::Qwen2 => Ok(ModelWeights::Qwen2(
            quantized_qwen2::ModelWeights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow!("loading Qwen2 GGUF: {e}"))?,
        )),
        Family::Llama => Ok(ModelWeights::Llama(
            quantized_llama::ModelWeights::from_gguf(content, &mut file, device)
                .map_err(|e| anyhow!("loading Llama GGUF: {e}"))?,
        )),
    }
}

// -----------------------------------------------------------------
// Process-wide cache so scenarios reuse loaded weights.
// -----------------------------------------------------------------

use tokio::sync::Mutex;

#[derive(Default)]
struct CacheEntry {
    /// `(alias_lowercase, device_label)` so a process running both
    /// CPU and Metal-resident enhancers (rare; usually one) keeps
    /// them distinct.
    key: String,
    enhancer: Option<Arc<Mutex<Enhancer>>>,
}

static CACHE: std::sync::OnceLock<Mutex<CacheEntry>> = std::sync::OnceLock::new();

fn cache() -> &'static Mutex<CacheEntry> {
    CACHE.get_or_init(|| Mutex::new(CacheEntry::default()))
}

fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

/// Top-level entry point. Loads + caches the model on first call,
/// reuses on subsequent calls. Concurrent callers serialise on
/// the inner mutex — the decode loop isn't thread-safe (KV cache
/// is `&mut self`), so this matches the model's contract.
pub async fn enhance(
    alias: &str,
    device: Device,
    system: &str,
    user: &str,
    opts: EnhanceOpts,
) -> Result<String, EnhanceError> {
    let key = format!("{}@{}", alias.to_lowercase(), device_label(&device));

    let enhancer_arc = {
        let mut cache_lock = cache().lock().await;
        if cache_lock.key != key || cache_lock.enhancer.is_none() {
            tracing::info!(
                target: "plakat",
                "Loading local enhancer {alias} on {} (first use; downloads on cache miss)",
                device_label(&device)
            );
            let loaded = Enhancer::load(alias, device.clone())
                .await
                .map_err(EnhanceError::Other)?;
            cache_lock.key = key.clone();
            cache_lock.enhancer = Some(Arc::new(Mutex::new(loaded)));
        }
        cache_lock
            .enhancer
            .as_ref()
            .expect("just inserted")
            .clone()
    };
    let mut enhancer = enhancer_arc.lock().await;
    enhancer.enhance(system, user, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhance_opts_defaults_are_greedy_with_96_tokens() {
        let o = EnhanceOpts::default();
        assert_eq!(o.temperature, 0.0);
        assert_eq!(o.max_new_tokens, 96);
        assert_eq!(o.seed, 0);
    }

    #[test]
    fn enhance_error_display_marks_refusal() {
        let e = EnhanceError::Refused;
        assert!(format!("{e}").contains("refused"));
    }

    #[test]
    fn device_label_covers_each_variant() {
        assert_eq!(device_label(&Device::Cpu), "cpu");
        // CUDA / Metal devices can't be instantiated in unit tests
        // without the feature flags; covered manually.
    }
}
