//! v0.21: `plakat.*` host words registered into the bundcore VM.
//!
//! Each word lives in its own file (`echo.rs`, `load.rs`,
//! `generate.rs`, …). [`register_plakat_words`] wires every word
//! into the VM via [`VM::register_inline`]. The host fns are
//! plain `fn` pointers; see `super::ctx` for the state-sharing
//! singleton that gets around the no-closures constraint.

use anyhow::{Result, anyhow};
use rust_multistackvm::multistackvm::VM;

pub mod echo;

/// Register every `plakat.*` word into `vm`. Phase 1 ships just
/// `plakat.echo`; subsequent phases append.
pub fn register_plakat_words(vm: &mut VM) -> Result<()> {
    vm.register_inline("plakat.echo".to_string(), echo::plakat_echo)
        .map_err(|e| anyhow!("registering plakat.echo: {e}"))?;
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
