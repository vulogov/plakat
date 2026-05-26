//! v0.22 phase 1: the cached pipeline behind [`ScriptCtx`].
//!
//! v0.21 reloaded the model on every `plakat.generate` /
//! `plakat.img2img` / `plakat.portrait`. v0.22 caches the loaded
//! pipeline by alias so subsequent calls reuse it — a single SDXL
//! load (~10s on a 4090) amortises across a script that produces
//! dozens of images.
//!
//! Phase 1 ships only the SD-family variant. Phases 2-3 will add
//! `Flux(flux::Pipeline)` and `Sd3(sd3::Pipeline)` and lift the
//! [`super::script_entry::validate_supported_for_phase_2`] gate.

use crate::pipelines::{flux, portrait, sd3};

/// The active pipeline, cached by `(alias, pipeline)` in
/// [`super::ctx::ScriptCtx::loaded`].
///
/// SD-family case uses [`portrait::Pipeline`] because it
/// generalises across the three image-producing host words
/// (text-to-image with empty photos, img2img via
/// `run_with_pipeline`, portrait with identity).
///
/// Flux case (v0.22 phase 2) uses [`flux::Pipeline`] which holds
/// the BFL DiT transformer + T5 + CLIP encoders + autoencoder.
/// One Flux pipeline serves both `plakat.generate` (text-to-image)
/// and `plakat.img2img` (init image + strength fields on
/// `flux::GenRequest`). `plakat.portrait` bails on Flux — Flux
/// has no IP-Adapter-Plus-Face checkpoint; future portrait-on-
/// Flux work would need a separate adapter strategy.
///
/// SD3 / SD3.5 (phase 3) will add a third variant.
pub enum LoadedPipeline {
    SdFamily(portrait::Pipeline),
    Flux(flux::Pipeline),
    Sd3(sd3::Pipeline),
}

/// Three families plakat recognises at the script layer. Used to
/// pick the right load+generate path before paying the model
/// load cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineFamily {
    /// SD 1.5 / 2.1 / SDXL / SDXL-Turbo.
    SdFamily,
    /// Any Flux variant (Dev / Schnell / Fill / Canny / Depth /
    /// Kontext, BF16 or GGUF or NF4).
    Flux,
    /// SD3 / SD3.5 — gated for phase 3.
    Sd3,
}

impl PipelineFamily {
    /// Resolve an alias (or canonical HF repo path) to a family.
    /// Resolves the alias to its repo id first so detection works
    /// against the canonical name (`sd21` only carries the SD-2.1
    /// substrings after alias resolution).
    pub fn detect(alias: &str) -> Self {
        let resolved = if alias.contains('/') {
            alias.to_string()
        } else {
            crate::hf::resolve_alias(alias).to_string()
        };
        let variant = crate::pipelines::t2i::Variant::detect(&resolved);
        if variant.is_flux() {
            Self::Flux
        } else if variant.is_sd3() {
            Self::Sd3
        } else {
            Self::SdFamily
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_sd_family_aliases() {
        for alias in &["sd15", "sd21", "sdxl", "sdxl-turbo"] {
            assert_eq!(
                PipelineFamily::detect(alias),
                PipelineFamily::SdFamily,
                "alias {alias:?}"
            );
        }
    }

    #[test]
    fn detect_flux_aliases() {
        // Every Flux variant + GGUF / NF4 should classify as Flux.
        for alias in &[
            "flux-dev",
            "flux-schnell",
            "flux-fill-dev",
            "flux-canny-dev",
            "flux-depth-dev",
            "flux-kontext-dev",
            "flux-dev-gguf",
            "flux-dev-nf4",
            "flux-schnell-gguf",
        ] {
            assert_eq!(
                PipelineFamily::detect(alias),
                PipelineFamily::Flux,
                "alias {alias:?}"
            );
        }
    }

    #[test]
    fn detect_sd3_aliases() {
        for alias in &[
            "sd3-medium",
            "sd35-medium",
            "sd35-large",
            "sd35-large-turbo",
        ] {
            assert_eq!(
                PipelineFamily::detect(alias),
                PipelineFamily::Sd3,
                "alias {alias:?}"
            );
        }
    }

    #[test]
    fn detect_resolves_canonical_hf_repos() {
        assert_eq!(
            PipelineFamily::detect("black-forest-labs/FLUX.1-dev"),
            PipelineFamily::Flux,
        );
        assert_eq!(
            PipelineFamily::detect("stabilityai/stable-diffusion-3.5-medium"),
            PipelineFamily::Sd3,
        );
        assert_eq!(
            PipelineFamily::detect("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            PipelineFamily::SdFamily,
        );
    }
}
