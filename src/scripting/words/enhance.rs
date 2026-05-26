//! v0.22 phase 10: `plakat.enhance` host word.
//!
//! ```bund
//! "auto" "enhance_provider" plakat.config.set
//! "true" "enhance_keep_original" plakat.config.set
//! "sd15" plakat.load
//! "a knight" plakat.enhance
//! plakat.generate
//! ```
//!
//! Stack effect: `( prompt -- enhanced )`. Pops one string (the
//! raw prompt), runs the configured provider, pushes one string
//! (the enhanced rewrite, possibly with the original
//! `BREAK`-appended when `enhance_keep_original` is set + an
//! SD-family model is loaded). Provider selection + temperature
//! / max-tokens / cache / system-prompt-path all come from
//! `config.enhance_*` keys.
//!
//! No script-side state mutation: enhance is pure
//! (prompt-in, prompt-out) modulo the LLM cache, which lives in
//! `crate::llm::enhancer`'s process-wide singleton. Calling
//! `plakat.enhance` twice on the same prompt with greedy
//! decoding returns the same output both times (same as the
//! CLI's `--enhance local`).
//!
//! Family-aware `enhance_keep_original`: Flux + SD3 short-circuit
//! to "enhanced only" because BREAK is CLIP-specific. The check
//! reuses `cli::generate::maybe_keep_original` so the behaviour
//! is byte-identical to the CLI.

use rust_dynamic::value::Value;
use rust_multistackvm::multistackvm::VM;
use std::path::PathBuf;

use crate::scripting::ctx::with_ctx;
use crate::scripting::helpers::{
    BundResult, pull, push, require_depth, to_bund_err, value_to_string,
};

const TAG: &str = "plakat.enhance";

pub fn plakat_enhance(vm: &mut VM) -> BundResult<'_> {
    do_plakat_enhance(vm).map_err(to_bund_err)
}

fn do_plakat_enhance(vm: &mut VM) -> anyhow::Result<&mut VM> {
    require_depth(vm, 1, TAG)?;
    let prompt_v = pull(vm, TAG)?;
    let prompt = value_to_string(prompt_v, "prompt", TAG)?;

    // Snapshot the relevant config into owned values *before*
    // crossing the async boundary. We hold the read lock only
    // for the snapshot, not for the LLM forward (which can run
    // for many seconds on CPU).
    let snapshot = with_ctx(|ctx| EnhanceSnapshot {
        provider: ctx.config.enhance_provider.clone(),
        temperature: ctx.config.enhance_temp,
        max_new_tokens: ctx.config.enhance_max_tokens,
        cache: ctx.config.enhance_cache,
        system_path: if ctx.config.enhance_system.is_empty() {
            None
        } else {
            Some(PathBuf::from(&ctx.config.enhance_system))
        },
        keep_original: ctx.config.enhance_keep_original,
        loaded_model: ctx.loaded_model().map(|s| s.to_string()),
    })?;

    if snapshot.provider.is_empty() {
        anyhow::bail!(
            "{TAG}: no provider configured. Set one via \
             plakat.config.set (e.g. \"auto\" \"enhance_provider\" \
             plakat.config.set). Accepted: auto, deepseek, gemini, \
             local, local:<alias>."
        );
    }

    let args = crate::prompt::EnhanceArgs {
        system_path: snapshot.system_path,
        temperature: snapshot.temperature,
        max_new_tokens: snapshot.max_new_tokens,
        cache: snapshot.cache,
    };

    let handle = tokio::runtime::Handle::try_current().map_err(|e| {
        anyhow::anyhow!(
            "{TAG}: no tokio runtime in scope (eval must run on a \
             multi-threaded runtime). Underlying error: {e}"
        )
    })?;

    let enhanced = tokio::task::block_in_place(|| {
        handle.block_on(crate::prompt::enhance_with_args(
            &snapshot.provider,
            &prompt,
            &args,
        ))
    })?;

    // v0.20-flavoured keep-original: BREAK-join when the SD-family
    // path applies. Family detection needs a loaded model alias.
    // When no model is loaded the keep_original flag is a no-op
    // with a warn — the user can call plakat.enhance before
    // plakat.load, which is fine; we just can't decide on BREAK
    // application until family is known.
    let final_prompt = if snapshot.keep_original {
        match snapshot.loaded_model {
            Some(alias) => {
                let resolved = if alias.contains('/') {
                    alias
                } else {
                    crate::hf::resolve_alias(&alias).to_string()
                };
                crate::cli::generate::maybe_keep_original(
                    &resolved, enhanced, &prompt, true,
                )
            }
            None => {
                tracing::warn!(
                    target: "plakat",
                    "{TAG}: enhance_keep_original is true but no model is \
                     loaded — can't decide on BREAK (CLIP) vs no-BREAK \
                     (Flux/SD3). Returning enhanced-only. Load the model \
                     first to apply keep-original semantics."
                );
                enhanced
            }
        }
    } else {
        enhanced
    };

    let preview = if final_prompt.chars().count() > 80 {
        format!("{}...", final_prompt.chars().take(80).collect::<String>())
    } else {
        final_prompt.clone()
    };
    tracing::info!(
        target: "plakat",
        "{TAG}: ({}): {preview:?}",
        snapshot.provider
    );
    push(vm, Value::from_string(final_prompt));
    Ok(vm)
}

struct EnhanceSnapshot {
    provider: String,
    temperature: Option<f64>,
    max_new_tokens: Option<usize>,
    cache: bool,
    system_path: Option<PathBuf>,
    keep_original: bool,
    loaded_model: Option<String>,
}
