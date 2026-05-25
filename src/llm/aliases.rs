//! Short-name → GGUF descriptor lookup for the local prompt
//! enhancer. The two shipped defaults match the v0.18 design
//! brainstorm: Qwen2.5-1.5B-Instruct as the recommended choice
//! (best instruction-following at small size, Apache-2.0) and
//! SmolLM2-360M-Instruct as the CPU-budget fallback.

/// Which model architecture the GGUF embeds. Drives the loader
/// (`quantized_qwen2` vs `quantized_llama`) and chat template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Qwen2 / Qwen2.5 — ChatML-style template with
    /// `<|im_start|>` / `<|im_end|>` role markers.
    Qwen2,
    /// Llama-arch GGUF (SmolLM2 is Llama-style at the model level).
    /// Uses the Llama 3-style chat template.
    Llama,
}

/// A registered model alias and the HF repo / GGUF file it points
/// at. `tokenizer_repo` is where to fetch the matching
/// `tokenizer.json` — usually the base model's HF repo, since the
/// GGUF mirrors often don't carry a HF-format tokenizer.
#[derive(Debug, Clone, Copy)]
pub struct ModelDescriptor {
    /// Canonical short name accepted by `--enhance-model`.
    pub alias: &'static str,
    /// HF repo hosting the GGUF.
    pub gguf_repo: &'static str,
    /// Filename inside the GGUF repo.
    pub gguf_file: &'static str,
    /// HF repo for the matching `tokenizer.json`. Usually the base
    /// (non-GGUF) model's repo.
    pub tokenizer_repo: &'static str,
    /// Architecture family.
    pub family: Family,
    /// Approximate disk footprint at this quantization, in MiB.
    /// User-facing only — shown in download progress.
    pub size_mib: u32,
}

/// The shipped registry. Keep tight — every entry triggers a
/// multi-hundred-megabyte download on first use.
pub const REGISTRY: &[ModelDescriptor] = &[
    // Recommended default. Qwen2.5-1.5B-Instruct at Q4_K_M is the
    // sweet spot for instruction-following + small footprint.
    // Apache-2.0 license — no HF gate.
    ModelDescriptor {
        alias: "qwen2.5-1.5b",
        gguf_repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        gguf_file: "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        tokenizer_repo: "Qwen/Qwen2.5-1.5B-Instruct",
        family: Family::Qwen2,
        size_mib: 1_020,
    },
    // CPU-budget fallback. SmolLM2-360M at Q4_K_M lands in ~230 MiB
    // and is plenty for "add some adjectives" prompt-enhancement.
    ModelDescriptor {
        alias: "smollm2-360m",
        gguf_repo: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        gguf_file: "smollm2-360m-instruct-q4_k_m.gguf",
        tokenizer_repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
        family: Family::Llama,
        size_mib: 230,
    },
];

/// The default model alias when `--enhance local` is used without
/// an explicit `--enhance-model`. Picks the larger Qwen by default
/// because the quality jump over SmolLM2 is meaningful for the
/// "rewrite this prompt with more detail" workload.
pub const DEFAULT_ALIAS: &str = "qwen2.5-1.5b";

/// Look up a descriptor by alias. Returns `None` for unregistered
/// aliases — callers should pair with the supported-list message
/// in `error_supported_aliases` for a friendly diagnostic.
pub fn resolve(alias: &str) -> Option<&'static ModelDescriptor> {
    REGISTRY.iter().find(|d| d.alias.eq_ignore_ascii_case(alias))
}

/// Comma-joined list of every registered alias — used in error
/// messages so a fat-fingered `--enhance-model` doesn't leave the
/// user grepping the source for valid names.
pub fn supported_aliases() -> String {
    REGISTRY
        .iter()
        .map(|d| d.alias)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_aliases_are_unique() {
        let mut seen: Vec<&str> = REGISTRY.iter().map(|d| d.alias).collect();
        seen.sort();
        let mut dedup = seen.clone();
        dedup.dedup();
        assert_eq!(seen.len(), dedup.len(), "alias collision in REGISTRY");
    }

    #[test]
    fn default_alias_is_registered() {
        assert!(
            resolve(DEFAULT_ALIAS).is_some(),
            "DEFAULT_ALIAS {DEFAULT_ALIAS:?} not in REGISTRY"
        );
    }

    #[test]
    fn resolve_case_insensitive() {
        assert!(resolve("Qwen2.5-1.5B").is_some());
        assert!(resolve("smollm2-360m").is_some());
    }

    #[test]
    fn resolve_unknown_returns_none() {
        assert!(resolve("not-a-real-model").is_none());
    }

    #[test]
    fn every_descriptor_has_required_fields() {
        for d in REGISTRY {
            assert!(!d.alias.is_empty());
            assert!(!d.gguf_repo.is_empty());
            assert!(d.gguf_file.ends_with(".gguf"));
            assert!(!d.tokenizer_repo.is_empty());
            assert!(d.size_mib > 0);
        }
    }
}
