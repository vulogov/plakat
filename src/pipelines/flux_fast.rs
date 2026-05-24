//! Distillation-LoRA presets for fast Flux inference.
//!
//! Hyper-FLUX (ByteDance) and FLUX.1-Turbo-Alpha (alimama-creative)
//! are distillations of Flux.1-dev to 4–16 steps. They ship as **LoRAs**
//! applied on top of the base model — every parameter that drives a
//! preset is already expressible via `--loras` + `--steps` + `--guidance`.
//! The `--fast` flag is a thin sugar layer over those three:
//!
//! 1. Resolve the preset name (e.g. `hyper-8`) into a `LoraSpec` plus
//!    recommended step / guidance defaults.
//! 2. Prepend the preset LoRA onto the user's existing `--loras`
//!    stack so any task-specific LoRA still applies on top.
//! 3. Override the CLI's default `--steps` / `--guidance` **only if**
//!    the user didn't pass them explicitly (defaults stay sticky;
//!    explicit overrides win).
//!
//! Why a curated registry: the LoRA paths inside the published packs
//! are stable (HF guarantees the repo / filename), the scales /
//! step counts are not obvious without reading model cards, and the
//! same preset applies broadly across users. Bundling the well-known
//! configs is a one-line UX win.
//!
//! ## Composition guarantees
//!
//! * `--fast` requires a non-Fill Flux variant. Fill's mask-driven
//!   denoise interacts oddly with the distillation schedule —
//!   handle separately if it ever lands.
//! * Composes with the user's `--loras`: preset LoRA loads first,
//!   then user LoRAs (later entries override matching keys at
//!   merge time).
//! * Composes with GGUF + NF4 (LoRA on quantized is wired for GGUF
//!   only in v0.13 phase 1e; NF4 + LoRA is deferred — caller bails
//!   loud if `--fast` is combined with `--model flux-*-nf4`).

use anyhow::{Context, Result};
use std::str::FromStr;

use crate::pipelines::lora::LoraSpec;

/// Model family a preset targets. Drives the dispatch's
/// model-compatibility check in `cli::generate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastTarget {
    /// Flux (BF16 or GGUF) — Hyper-FLUX, Turbo-Alpha.
    Flux,
    /// SDXL / SDXL-Turbo — LCM-LoRA-SDXL.
    Sdxl,
    /// Stable Diffusion 1.5 — LCM-LoRA-SDv1.5.
    Sd15,
}

/// One curated distillation preset. `lora_repo` + `lora_file` point at
/// an HF safetensors; `lora_scale` is the LoRA-merge weight; `steps`
/// + `guidance` are the recommended sampling settings for this
/// preset's distillation schedule.
#[derive(Debug, Clone)]
pub struct FastPreset {
    /// Canonical kebab-case name (what users type after `--fast`).
    pub name: &'static str,
    /// One-line summary shown in `--help` and error messages.
    pub description: &'static str,
    /// Model family this preset is wired for.
    pub target: FastTarget,
    /// HF repo id hosting the LoRA.
    pub lora_repo: &'static str,
    /// File inside the repo. `None` = the loader picks the default.
    pub lora_file: Option<&'static str>,
    /// LoRA merge scale recommended by the model card.
    pub lora_scale: f32,
    /// Step count the distillation was trained for.
    pub steps: usize,
    /// Guidance recommendation. Distillations targeting CFG-free
    /// inference (e.g. Hyper-FLUX) use `1.0`; those that keep CFG
    /// (e.g. Turbo-Alpha) use a normal Flux guidance.
    pub guidance: f64,
    /// Optional scheduler name to force. The string matches
    /// `SchedulerKind::FromStr` accepted tokens — e.g. `"lcm"` for
    /// LCM-LoRA distillations. `None` means "leave the user's
    /// `--scheduler` alone" (Flux defaults are fine for
    /// Hyper / Turbo).
    pub scheduler_hint: Option<&'static str>,
}

impl FastPreset {
    /// Construct a `LoraSpec` ready to feed the standard LoRA loader.
    pub fn to_lora_spec(&self) -> LoraSpec {
        LoraSpec::hub_pinned(
            self.lora_repo.to_string(),
            self.lora_file.map(|s| s.to_string()),
            None, // no revision pinning — HF main is fine for curated picks
            self.lora_scale,
        )
    }
}

/// The curated preset list. Adding a preset = adding a row here.
///
/// Sources:
/// * Hyper-FLUX: ByteDance/Hyper-SD model card
///   <https://huggingface.co/ByteDance/Hyper-SD>
/// * Turbo-Alpha: alimama-creative/FLUX.1-Turbo-Alpha model card
///   <https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha>
pub const PRESETS: &[FastPreset] = &[
    FastPreset {
        name: "hyper-8",
        description: "ByteDance Hyper-FLUX 8-step distillation (CFG-free)",
        target: FastTarget::Flux,
        lora_repo: "ByteDance/Hyper-SD",
        lora_file: Some("Hyper-FLUX.1-dev-8steps-lora.safetensors"),
        lora_scale: 0.125,
        steps: 8,
        guidance: 1.0,
        scheduler_hint: None,
    },
    FastPreset {
        name: "hyper-16",
        description: "ByteDance Hyper-FLUX 16-step distillation (CFG-free)",
        target: FastTarget::Flux,
        lora_repo: "ByteDance/Hyper-SD",
        lora_file: Some("Hyper-FLUX.1-dev-16steps-lora.safetensors"),
        lora_scale: 0.125,
        steps: 16,
        guidance: 1.0,
        scheduler_hint: None,
    },
    FastPreset {
        name: "turbo-alpha",
        description: "alimama-creative FLUX.1-Turbo-Alpha 8-step distillation",
        target: FastTarget::Flux,
        lora_repo: "alimama-creative/FLUX.1-Turbo-Alpha",
        // Repo ships a single safetensors at the root; let the loader
        // pick `diffusion_pytorch_model.safetensors` as the default.
        lora_file: Some("diffusion_pytorch_model.safetensors"),
        lora_scale: 1.0,
        steps: 8,
        guidance: 3.5,
        scheduler_hint: None,
    },
    // v0.17 phase I: LCM-LoRA for SDXL. Pair the LoRA with the
    // `lcm` scheduler at 4 steps / guidance 1.5 to get the full
    // 4-step distillation behaviour. The model card
    // (`latent-consistency/lcm-lora-sdxl`) recommends 4-8 steps
    // with CFG in [1.0, 2.0]. The preset picks the midpoint of
    // that band so output stays prompt-adherent without burning.
    FastPreset {
        name: "lcm-sdxl",
        description: "Latent Consistency LoRA for SDXL — 4-step inference at CFG 1.5",
        target: FastTarget::Sdxl,
        lora_repo: "latent-consistency/lcm-lora-sdxl",
        // Repo ships `pytorch_lora_weights.safetensors` as the
        // canonical filename; let the auto-discover pick.
        lora_file: None,
        lora_scale: 1.0,
        steps: 4,
        guidance: 1.5,
        scheduler_hint: Some("lcm"),
    },
    // v0.18 phase 1: LCM-LoRA for SD 1.5. Same distillation recipe
    // as the SDXL preset, against the v1.5 base. The model card
    // (`latent-consistency/lcm-lora-sdv1-5`) recommends 4-8 steps
    // with CFG in [1.0, 2.0]; we pick the same midpoint as lcm-sdxl
    // for muscle-memory consistency.
    FastPreset {
        name: "lcm-sd15",
        description: "Latent Consistency LoRA for SD 1.5 — 4-step inference at CFG 1.5",
        target: FastTarget::Sd15,
        lora_repo: "latent-consistency/lcm-lora-sdv1-5",
        lora_file: None,
        lora_scale: 1.0,
        steps: 4,
        guidance: 1.5,
        scheduler_hint: Some("lcm"),
    },
];

/// Lookup by name. Returns `None` for unknown names.
pub fn lookup(name: &str) -> Option<&'static FastPreset> {
    PRESETS.iter().find(|p| p.name == name)
}

/// Lookup with a helpful error message that lists the supported
/// presets when the user fat-fingers a name.
pub fn resolve(name: &str) -> Result<&'static FastPreset> {
    lookup(name).with_context(|| {
        let supported = PRESETS
            .iter()
            .map(|p| p.name)
            .collect::<Vec<_>>()
            .join(", ");
        format!("--fast preset '{name}' isn't recognised. Supported: {supported}")
    })
}

/// Parse a `--fast` arg via clap. Wraps `resolve` so clap can produce
/// the proper "value parser" diagnostics if the name is unknown.
#[derive(Debug, Clone)]
pub struct FastPresetArg(pub &'static FastPreset);

impl FromStr for FastPresetArg {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let preset = resolve(s)?;
        Ok(Self(preset))
    }
}

/// Result of applying a preset on top of CLI args. Used by the
/// generate dispatch — `lora_extra` gets PREPENDED to the user's
/// `--loras` stack; `steps_override` / `guidance_override` overwrite
/// the corresponding fields **only if the user didn't pass them
/// explicitly** (the caller does the "explicit?" check, since clap
/// doesn't surface whether a default fired or a user typed the
/// default).
#[derive(Debug, Clone)]
pub struct PresetApplication {
    pub lora_extra: LoraSpec,
    pub steps_override: usize,
    pub guidance_override: f64,
}

impl FastPreset {
    /// Pre-baked application bundle. Convenience for callers.
    pub fn apply(&self) -> PresetApplication {
        PresetApplication {
            lora_extra: self.to_lora_spec(),
            steps_override: self.steps,
            guidance_override: self.guidance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_unique() {
        let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        names.sort();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(
            names.len(),
            deduped.len(),
            "preset names must be unique (sorted = {names:?})"
        );
    }

    #[test]
    fn registry_lookup_known_preset() {
        let p = lookup("hyper-8").expect("hyper-8 is in the registry");
        assert_eq!(p.steps, 8);
        assert_eq!(p.guidance, 1.0);
        assert!((p.lora_scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn registry_lookup_unknown() {
        assert!(lookup("does-not-exist").is_none());
        let err = resolve("does-not-exist").unwrap_err();
        let msg = format!("{err}");
        // Error message must list at least one valid preset so users
        // can self-correct without grepping source.
        assert!(msg.contains("hyper-8"), "{msg}");
    }

    #[test]
    fn registry_scales_are_finite_and_positive() {
        for p in PRESETS {
            assert!(p.lora_scale.is_finite(), "{} scale not finite", p.name);
            assert!(
                p.lora_scale > 0.0 && p.lora_scale <= 2.0,
                "{} scale {} outside (0, 2]",
                p.name,
                p.lora_scale,
            );
            assert!(p.steps > 0 && p.steps <= 64, "{} steps out of range", p.name);
            assert!(
                p.guidance >= 0.0 && p.guidance <= 30.0,
                "{} guidance out of range",
                p.name
            );
        }
    }

    #[test]
    fn from_str_parses_via_clap_path() {
        let arg: FastPresetArg = "turbo-alpha".parse().unwrap();
        assert_eq!(arg.0.name, "turbo-alpha");
        assert_eq!(arg.0.steps, 8);
    }

    // v0.17 phase I — LCM-LoRA SDXL preset.

    #[test]
    fn lcm_sdxl_preset_registered() {
        let p = lookup("lcm-sdxl").expect("lcm-sdxl preset registered");
        assert_eq!(p.target, FastTarget::Sdxl);
        assert_eq!(p.lora_repo, "latent-consistency/lcm-lora-sdxl");
        assert_eq!(p.steps, 4);
        assert!((p.guidance - 1.5).abs() < f64::EPSILON);
        assert_eq!(p.scheduler_hint, Some("lcm"));
    }

    #[test]
    fn flux_presets_carry_flux_target() {
        for name in ["hyper-8", "hyper-16", "turbo-alpha"] {
            let p = lookup(name).unwrap();
            assert_eq!(p.target, FastTarget::Flux, "{name} should be Flux-targeted");
        }
    }

    #[test]
    fn scheduler_hint_is_only_set_for_lcm() {
        // The Flux distillations are scheduler-agnostic (rectified
        // flow works with any sampler); only LCM-LoRA needs to pin
        // the scheduler at preset-apply time.
        for p in PRESETS {
            if p.name.starts_with("lcm-") {
                assert!(
                    p.scheduler_hint == Some("lcm"),
                    "{} must pin scheduler to lcm",
                    p.name
                );
            } else {
                assert!(
                    p.scheduler_hint.is_none(),
                    "{} unexpectedly carries a scheduler hint",
                    p.name
                );
            }
        }
    }

    // v0.18 phase 1 — LCM-LoRA SD 1.5 preset.

    #[test]
    fn lcm_sd15_preset_registered() {
        let p = lookup("lcm-sd15").expect("lcm-sd15 preset registered");
        assert_eq!(p.target, FastTarget::Sd15);
        assert_eq!(p.lora_repo, "latent-consistency/lcm-lora-sdv1-5");
        assert_eq!(p.steps, 4);
        assert!((p.guidance - 1.5).abs() < f64::EPSILON);
        assert_eq!(p.scheduler_hint, Some("lcm"));
    }
}
