//! v0.21: `plakat.*` host words registered into the bundcore VM.
//!
//! Each word lives in its own file (`echo.rs`, `load.rs`,
//! `generate.rs`, …). [`register_plakat_words`] wires every word
//! into the VM via [`VM::register_inline`]. The host fns are
//! plain `fn` pointers; see `super::ctx` for the state-sharing
//! singleton that gets around the no-closures constraint.

use anyhow::{Result, anyhow};
use rust_multistackvm::multistackvm::VM;

pub mod config;
pub mod echo;
pub mod generate;
pub mod img2img;
pub mod load;
pub mod lora;
pub mod portrait;
pub mod save;
pub mod upscale;

/// Register every `plakat.*` word into `vm`. v0.21 shipped 7
/// MVP words. v0.22 phase 4 adds the `plakat.lora.*` namespace.
pub fn register_plakat_words(vm: &mut VM) -> Result<()> {
    vm.register_inline("plakat.echo".to_string(), echo::plakat_echo)
        .map_err(|e| anyhow!("registering plakat.echo: {e}"))?;
    vm.register_inline("plakat.load".to_string(), load::plakat_load)
        .map_err(|e| anyhow!("registering plakat.load: {e}"))?;
    vm.register_inline("plakat.generate".to_string(), generate::plakat_generate)
        .map_err(|e| anyhow!("registering plakat.generate: {e}"))?;
    vm.register_inline("plakat.img2img".to_string(), img2img::plakat_img2img)
        .map_err(|e| anyhow!("registering plakat.img2img: {e}"))?;
    vm.register_inline("plakat.portrait".to_string(), portrait::plakat_portrait)
        .map_err(|e| anyhow!("registering plakat.portrait: {e}"))?;
    vm.register_inline("plakat.save".to_string(), save::plakat_save)
        .map_err(|e| anyhow!("registering plakat.save: {e}"))?;
    vm.register_inline("plakat.upscale".to_string(), upscale::plakat_upscale)
        .map_err(|e| anyhow!("registering plakat.upscale: {e}"))?;
    vm.register_inline(
        "plakat.config.set".to_string(),
        config::plakat_config_set,
    )
    .map_err(|e| anyhow!("registering plakat.config.set: {e}"))?;
    // v0.22 phase 4: plakat.lora.* namespace.
    vm.register_inline("plakat.lora.add".to_string(), lora::plakat_lora_add)
        .map_err(|e| anyhow!("registering plakat.lora.add: {e}"))?;
    vm.register_inline("plakat.lora.clear".to_string(), lora::plakat_lora_clear)
        .map_err(|e| anyhow!("registering plakat.lora.clear: {e}"))?;
    vm.register_inline("plakat.lora.list".to_string(), lora::plakat_lora_list)
        .map_err(|e| anyhow!("registering plakat.lora.list: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plakat_words_is_idempotent_on_fresh_vm() {
        let mut vm = VM::new();
        register_plakat_words(&mut vm).unwrap();
        // A second call would re-register (bundcore allows upsert).
        // We don't enforce idempotency yet; this test just pins
        // that "fresh VM + register" doesn't fail with the v0.21
        // word set.
    }
}
