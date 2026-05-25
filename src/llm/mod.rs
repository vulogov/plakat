//! Local LLM-based prompt enhancer. Runs a small quantized
//! instruction-tuned model (Qwen2.5-1.5B-Instruct by default,
//! SmolLM2-360M-Instruct as a CPU-budget fallback) in-process via
//! candle's `quantized_qwen2` / `quantized_llama` backends, so
//! users get the v0.13 "enhance my prompt" workflow without an
//! API key.
//!
//! Public entry point: [`enhance`]. The `prompt::enhance` dispatch
//! routes the `"local"` provider arm here; users invoke it via
//! `plakat generate --enhance local "..."`.
//!
//! Caching: the loaded weights live in a process-wide
//! `OnceLock<Mutex<…>>` keyed by `(alias, device)`. Scenarios that
//! enhance dozens of prompts back-to-back pay the GGUF load cost
//! once.

pub mod aliases;
pub mod enhancer;
pub mod templates;

pub use aliases::{DEFAULT_ALIAS, Family, ModelDescriptor};
pub use enhancer::{EnhanceError, EnhanceOpts};

use anyhow::Result;
use candle_core::Device;

/// Run a single enhance pass. `alias` is one of the registered
/// model short names ([`aliases::REGISTRY`]); pass
/// [`DEFAULT_ALIAS`] when the user didn't specify one. `device` is
/// the candle device to run the LLM forward on — CPU by default;
/// Metal / CUDA work too if the binary was built with the
/// matching feature.
///
/// On refusal / empty output the inner error variant is
/// [`EnhanceError::Refused`] so the caller can fall back to the
/// un-enhanced prompt without surfacing a panic.
pub async fn enhance(
    alias: &str,
    device: Device,
    system: &str,
    user: &str,
    opts: EnhanceOpts,
) -> Result<String, EnhanceError> {
    enhancer::enhance(alias, device, system, user, opts).await
}
