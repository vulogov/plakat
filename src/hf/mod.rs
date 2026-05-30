pub mod cache;
pub mod download;
pub mod info;
pub mod search;

/// A single alias group: one canonical HuggingFace repo id and the
/// short aliases plakat accepts for it. v0.20 #4 promoted this from
/// a hand-written `match` to a static table so `plakat models
/// aliases` can enumerate it (single source of truth — adding an
/// alias here updates both resolution and the listing).
pub struct AliasEntry {
    /// Short aliases users can pass via `--model`. The first entry
    /// is treated as the "canonical short name" for display.
    pub aliases: &'static [&'static str],
    /// HuggingFace repo id the aliases resolve to.
    pub repo: &'static str,
    /// Model family — used to group rows under headings in the
    /// `plakat models aliases` table.
    pub family: &'static str,
    /// Sub-kind within the family ("base", "inpaint", "GGUF", …).
    pub kind: &'static str,
    /// True when the HF repo is gated (HF_TOKEN required).
    pub gated: bool,
    /// One-line description shown in the `models aliases` table.
    pub note: &'static str,
}

/// Compatibility redirects: full HF repo ids that no longer host
/// the weights but should still resolve to the live mirror. Kept
/// separate from [`ALIAS_TABLE`] so the alias listing only shows
/// the short aliases users would actually type.
const REPO_REDIRECTS: &[(&str, &str)] = &[
    (
        "runwayml/stable-diffusion-v1-5",
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ),
    (
        "runwayml/stable-diffusion-inpainting",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
    ),
];

/// All `--model` aliases plakat understands. Grouped by family so
/// `plakat models aliases` can print headings in the order
/// declared here.
pub const ALIAS_TABLE: &[AliasEntry] = &[
    // ── SD 1.5 ───────────────────────────────────────────────────
    AliasEntry {
        aliases: &["sd15", "sd-1.5"],
        repo: "stable-diffusion-v1-5/stable-diffusion-v1-5",
        family: "SD 1.5",
        kind: "base",
        gated: false,
        note: "Stable Diffusion 1.5 base (community mirror — Runway pulled the original in 2024)",
    },
    AliasEntry {
        aliases: &["sd15-inpaint", "sd15-inpainting", "sd-1.5-inpaint", "sd-1.5-inpainting"],
        repo: "stable-diffusion-v1-5/stable-diffusion-inpainting",
        family: "SD 1.5",
        kind: "inpaint",
        gated: false,
        note: "9-channel UNet inpainting variant (4 latent + 1 mask + 4 masked-image latents)",
    },
    // ── SD 2.1 ───────────────────────────────────────────────────
    AliasEntry {
        aliases: &["sd21", "sd-2.1"],
        repo: "stabilityai/stable-diffusion-2-1",
        family: "SD 2.1",
        kind: "base",
        gated: false,
        note: "Stable Diffusion 2.1 base — 768×768 native, OpenCLIP-H text encoder",
    },
    // ── SDXL ─────────────────────────────────────────────────────
    AliasEntry {
        aliases: &["sdxl"],
        repo: "stabilityai/stable-diffusion-xl-base-1.0",
        family: "SDXL",
        kind: "base",
        gated: false,
        note: "Stable Diffusion XL base — 1024² native, dual text encoders",
    },
    AliasEntry {
        aliases: &["sdxl-turbo"],
        repo: "stabilityai/sdxl-turbo",
        family: "SDXL",
        kind: "turbo",
        gated: false,
        note: "Adversarial-distilled SDXL — 1-4 steps, no CFG",
    },
    AliasEntry {
        aliases: &["sdxl-inpaint", "sdxl-inpainting"],
        repo: "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        family: "SDXL",
        kind: "inpaint",
        gated: false,
        note: "SDXL inpainting — 9-channel UNet matching the SD 1.5 inpaint contract",
    },
    AliasEntry {
        aliases: &["pony", "pony-v6", "pony-diffusion-v6"],
        repo: "AstraliteHeart/pony-diffusion-v6-xl",
        family: "SDXL",
        kind: "base",
        gated: false,
        note: "Pony Diffusion v6 XL — popular SDXL fine-tune. Pair with `--look pony` for score-based prompt conventions.",
    },
    // ── SD 3.x ───────────────────────────────────────────────────
    AliasEntry {
        aliases: &["sd35-medium", "sd3.5-medium", "stable-diffusion-3.5-medium"],
        repo: "stabilityai/stable-diffusion-3.5-medium",
        family: "SD 3.x",
        kind: "base",
        gated: true,
        note: "Stable Diffusion 3.5 Medium — MMDiT, 2.5B params (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["sd35-large", "sd3.5-large", "stable-diffusion-3.5-large"],
        repo: "stabilityai/stable-diffusion-3.5-large",
        family: "SD 3.x",
        kind: "base",
        gated: true,
        note: "Stable Diffusion 3.5 Large — MMDiT, 8B params (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &[
            "sd35-large-turbo",
            "sd3.5-large-turbo",
            "stable-diffusion-3.5-large-turbo",
        ],
        repo: "stabilityai/stable-diffusion-3.5-large-turbo",
        family: "SD 3.x",
        kind: "turbo",
        gated: true,
        note: "SD 3.5 Large distilled to 4-8 steps (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["sd3-medium", "stable-diffusion-3-medium"],
        repo: "stabilityai/stable-diffusion-3-medium",
        family: "SD 3.x",
        kind: "base",
        gated: true,
        note: "Original SD3 Medium (June 2024 release — superseded by 3.5)",
    },
    // ── Flux ─────────────────────────────────────────────────────
    AliasEntry {
        aliases: &["flux-dev", "FLUX.1-dev", "flux1-dev"],
        repo: "black-forest-labs/FLUX.1-dev",
        family: "Flux",
        kind: "base",
        gated: true,
        note: "Flux.1-dev — 12B DiT, guidance-distilled (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-schnell", "FLUX.1-schnell", "flux1-schnell"],
        repo: "black-forest-labs/FLUX.1-schnell",
        family: "Flux",
        kind: "turbo",
        gated: false,
        note: "Flux.1-schnell — 4-step distillation, Apache-2.0",
    },
    AliasEntry {
        aliases: &["flux-fill-dev", "flux-fill", "flux1-fill-dev", "FLUX.1-Fill-dev"],
        repo: "black-forest-labs/FLUX.1-Fill-dev",
        family: "Flux",
        kind: "Fill",
        gated: true,
        note: "Flux.1-Fill-dev — 384-channel img_in for inpaint/outpaint (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-canny-dev", "flux1-canny-dev", "FLUX.1-Canny-dev"],
        repo: "black-forest-labs/FLUX.1-Canny-dev",
        family: "Flux",
        kind: "Canny",
        gated: true,
        note: "Flux concept checkpoint with baked-in Canny conditioning (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-depth-dev", "flux1-depth-dev", "FLUX.1-Depth-dev"],
        repo: "black-forest-labs/FLUX.1-Depth-dev",
        family: "Flux",
        kind: "Depth",
        gated: true,
        note: "Flux concept checkpoint with baked-in depth conditioning (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-kontext-dev", "flux1-kontext-dev", "FLUX.1-Kontext-dev"],
        repo: "black-forest-labs/FLUX.1-Kontext-dev",
        family: "Flux",
        kind: "Kontext",
        gated: true,
        note: "Flux.1-Kontext-dev — reference-image editing via sequence concat (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-kontext-dev-gguf", "flux-kontext-gguf", "flux-kontext-q4"],
        repo: "unsloth/FLUX.1-Kontext-dev-GGUF",
        family: "Flux",
        kind: "Kontext-GGUF",
        gated: false,
        note: "GGUF (Q4/Q6/Q8) Kontext mirror — practical on 16 GB GPUs",
    },
    AliasEntry {
        aliases: &["flux-redux", "flux-redux-dev", "FLUX.1-Redux-dev"],
        repo: "black-forest-labs/FLUX.1-Redux-dev",
        family: "Flux",
        kind: "Redux",
        gated: true,
        note: "Flux Redux adapter — SigLIP-conditioned variation (HF_TOKEN required)",
    },
    AliasEntry {
        aliases: &["flux-dev-gguf", "flux-dev-q4"],
        repo: "city96/FLUX.1-dev-gguf",
        family: "Flux",
        kind: "GGUF",
        gated: false,
        note: "city96 GGUF mirror of Flux.1-dev — transformer drops ~24 GB → ~7 GB at Q4_K_S",
    },
    AliasEntry {
        aliases: &["flux-schnell-gguf", "flux-schnell-q4"],
        repo: "city96/FLUX.1-schnell-gguf",
        family: "Flux",
        kind: "GGUF",
        gated: false,
        note: "city96 GGUF mirror of Flux.1-schnell",
    },
    AliasEntry {
        aliases: &["flux-fill-dev-gguf", "flux-fill-gguf", "flux-fill-q4"],
        repo: "city96/FLUX.1-Fill-dev-gguf",
        family: "Flux",
        kind: "Fill-GGUF",
        gated: false,
        note: "city96 GGUF mirror of Flux.1-Fill-dev",
    },
    AliasEntry {
        aliases: &["flux-dev-nf4", "flux-nf4", "flux1-dev-nf4"],
        repo: "lllyasviel/flux1-dev-bnb-nf4-v2",
        family: "Flux",
        kind: "NF4",
        gated: false,
        note: "bitsandbytes NF4 Flux — weights stay 4-bit at inference (~6 GB transformer)",
    },
];

/// Map short aliases to canonical HuggingFace repo ids.
///
/// Note: Runway removed `runwayml/stable-diffusion-v1-5` from HF in mid-2024;
/// the community mirror at `stable-diffusion-v1-5/stable-diffusion-v1-5` is
/// what's used now (see [`REPO_REDIRECTS`]).
pub fn resolve_alias(name: &str) -> &str {
    for entry in ALIAS_TABLE {
        if entry.aliases.iter().any(|a| *a == name) {
            return entry.repo;
        }
    }
    for (from, to) in REPO_REDIRECTS {
        if *from == name {
            return to;
        }
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    // v0.18 Kontext phase 3 — alias resolution for the GGUF mirror.

    #[test]
    fn resolves_flux_kontext_dev_bf16() {
        assert_eq!(
            resolve_alias("flux-kontext-dev"),
            "black-forest-labs/FLUX.1-Kontext-dev"
        );
        assert_eq!(
            resolve_alias("flux1-kontext-dev"),
            "black-forest-labs/FLUX.1-Kontext-dev"
        );
    }

    #[test]
    fn resolves_flux_kontext_dev_gguf() {
        assert_eq!(
            resolve_alias("flux-kontext-dev-gguf"),
            "unsloth/FLUX.1-Kontext-dev-GGUF"
        );
        assert_eq!(
            resolve_alias("flux-kontext-gguf"),
            "unsloth/FLUX.1-Kontext-dev-GGUF"
        );
        assert_eq!(
            resolve_alias("flux-kontext-q4"),
            "unsloth/FLUX.1-Kontext-dev-GGUF"
        );
    }

    #[test]
    fn unknown_alias_passes_through() {
        // The fallthrough arm hands the input back verbatim so
        // explicit HF repo paths flow without modification.
        assert_eq!(
            resolve_alias("acme-research/some-custom-flux-lora"),
            "acme-research/some-custom-flux-lora"
        );
    }

    // v0.20 #4: alias-table refactor regression tests. These pin
    // resolutions that pre-existed the data-driven rewrite so
    // future edits to ALIAS_TABLE can't silently break callers.

    #[test]
    fn resolves_classic_sd_aliases() {
        assert_eq!(
            resolve_alias("sd15"),
            "stable-diffusion-v1-5/stable-diffusion-v1-5"
        );
        assert_eq!(
            resolve_alias("sd-1.5"),
            "stable-diffusion-v1-5/stable-diffusion-v1-5"
        );
        assert_eq!(resolve_alias("sd21"), "stabilityai/stable-diffusion-2-1");
        assert_eq!(
            resolve_alias("sdxl"),
            "stabilityai/stable-diffusion-xl-base-1.0"
        );
    }

    #[test]
    fn resolves_repo_redirects() {
        assert_eq!(
            resolve_alias("runwayml/stable-diffusion-v1-5"),
            "stable-diffusion-v1-5/stable-diffusion-v1-5"
        );
        assert_eq!(
            resolve_alias("runwayml/stable-diffusion-inpainting"),
            "stable-diffusion-v1-5/stable-diffusion-inpainting"
        );
    }

    #[test]
    fn alias_table_has_no_duplicate_aliases() {
        // A typo that gives two entries the same alias would
        // cause the first match to win silently. Catch that here.
        let mut seen = std::collections::HashSet::new();
        for entry in ALIAS_TABLE {
            for alias in entry.aliases {
                assert!(
                    seen.insert(*alias),
                    "duplicate alias '{alias}' in ALIAS_TABLE"
                );
            }
        }
    }

    #[test]
    fn alias_table_canonical_aliases_are_short() {
        // First alias is treated as the "canonical" short name in
        // display contexts — should never be a slash-bearing HF
        // path or longer than the others.
        for entry in ALIAS_TABLE {
            let first = entry.aliases.first().expect("at least one alias");
            assert!(
                !first.contains('/'),
                "canonical alias '{first}' for {} looks like a repo id",
                entry.repo
            );
        }
    }

    /// v0.31 phase 1 swap: Pony Diffusion v6 XL alias resolves.
    #[test]
    fn resolves_pony_diffusion_v6_xl() {
        for alias in ["pony", "pony-v6", "pony-diffusion-v6"] {
            assert_eq!(
                resolve_alias(alias),
                "AstraliteHeart/pony-diffusion-v6-xl",
                "alias {alias} should resolve to Pony Diffusion v6 XL",
            );
        }
    }
}
