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
    // Stability gated `stabilityai/stable-diffusion-2-1` (404 for tokens
    // that haven't accepted the licence), so `sd21` resolved to an
    // unreachable repo. Point at an ungated 768 v-prediction mirror so
    // `--model sd21` works out of the box — the same move plakat already
    // made for `sd15` after Runway deleted the original SD 1.5 repo. The
    // diffusers-safetensors layout + v-pred scheduler are identical to
    // the canonical 2-1, verified end to end.
    AliasEntry {
        aliases: &["sd21", "sd-2.1"],
        repo: "nlightcho/stable-diffusion-2-1",
        family: "SD 2.1",
        kind: "base",
        gated: false,
        note: "Stable Diffusion 2.1 — 768×768 native, OpenCLIP-H text encoder (ungated mirror)",
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
        // The original `AstraliteHeart/pony-diffusion-v6-xl` was removed from HF
        // (404). Repointed to a complete, non-gated diffusers-layout mirror.
        repo: "stablediffusionapi/pony-diffusion-v6-xl",
        family: "SDXL",
        kind: "base",
        gated: false,
        note: "Pony Diffusion v6 XL — popular SDXL fine-tune. Pair with `--look pony` for score-based prompt conventions.",
    },
    // ── SD 3.x ───────────────────────────────────────────────────
    AliasEntry {
        aliases: &["sd35", "sd35-medium", "sd3.5-medium", "stable-diffusion-3.5-medium"],
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
    // v0.35 phase 0: PixArt Sigma — fourth model family (DiT
    // backbone + T5-XXL text encoder). Canonical 1024-MS variant.
    // v0.36 phase 2: 512-MS variant added; 2K-MS lands in v0.36
    // phase 3 alongside KV-compression.
    AliasEntry {
        aliases: &["pixart", "pixart-sigma", "pixart-1024"],
        repo: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        family: "PixArt",
        kind: "base",
        gated: false,
        note: "PixArt-Σ XL 2 1024-MS — DiT + T5-XXL (~0.6B DiT + 4.7B T5)",
    },
    // v4.5 phase 0: Sana 1.6B 1024px — sixth model family (Linear-DiT +
    // DC-AE 32× autoencoder + Gemma-2-2B text encoder). BF16 diffusers repo.
    AliasEntry {
        aliases: &["sana", "sana-1600m", "sana-1024"],
        repo: "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        family: "Sana",
        kind: "base",
        gated: false,
        note: "Sana 1.6B 1024px — Linear-DiT + DC-AE 32× + Gemma-2-2B (~1.6B DiT)",
    },
    // v4.6: Sana variants. Same DC-AE + Gemma-2; the DiT config is read from transformer/config.json.
    AliasEntry {
        aliases: &["sana-600m"],
        repo: "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        family: "Sana",
        kind: "base",
        gated: false,
        note: "Sana 0.6B 1024px — smaller/faster DiT (28 layers, 1152 hidden)",
    },
    AliasEntry {
        aliases: &["sana-512"],
        repo: "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        family: "Sana",
        kind: "base",
        gated: false,
        note: "Sana 1.6B 512px — same DiT, trained at 512² (faster, lower-res). Use --size 512x512",
    },
    AliasEntry {
        aliases: &["sana-2k"],
        repo: "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        family: "Sana",
        kind: "base",
        gated: false,
        note: "Sana 1.6B 2K — same DiT, trained at 2048². Use --size 2048x2048 (memory-heavy)",
    },
    // v4.7: Sana-1.5 1.6B — adds qk_norm (rms_norm_across_heads); otherwise the base 1024px arch.
    AliasEntry {
        aliases: &["sana-1.5", "sana1.5", "sana-15"],
        repo: "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        family: "Sana",
        kind: "base",
        gated: false,
        note: "Sana-1.5 1.6B 1024px — improved checkpoint (qk_norm)",
    },
    AliasEntry {
        aliases: &["pixart-512", "pixart-sigma-512"],
        repo: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        family: "PixArt",
        kind: "base",
        gated: false,
        note: "PixArt-Σ XL 2 512-MS — same DiT-XL/2 as 1024-MS, trained at 512² (faster, smaller VRAM)",
    },
    // v0.36 phase 3: 2K-MS variant with KV-compression.
    AliasEntry {
        aliases: &["pixart-2k", "pixart-sigma-2k"],
        repo: "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
        family: "PixArt",
        kind: "base",
        gated: false,
        note: "PixArt-Σ XL 2 2K-MS — 2x KV-compression in self-attn (Σ paper §3.2) for 2048² output viability",
    },
    // v0.37 phase 0: Stable Cascade — fifth model family (3-stage
    // architecture: Stage A VAE + Stage B latent prior + Stage C
    // high-res prior). Phase 0 wires the Full repo only; Lite
    // variant routing lands alongside the v0.37 phase 2/3 stage
    // implementations.
    AliasEntry {
        aliases: &["stable-cascade", "cascade"],
        repo: "stabilityai/stable-cascade",
        family: "StableCascade",
        kind: "base",
        gated: false,
        note: "Stable Cascade — 3-stage architecture with CLIP-G text encoder (~3.6B Stage C + 1.5B Stage B + 3.6M Stage A VAE)",
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

/// Look up the full alias-table entry for a model name (canonical or any
/// accepted synonym). Returns `None` for unknown names. Used by `doctor
/// --capability` to read each model's repo + `gated` flag together.
pub fn entry_for_alias(name: &str) -> Option<&'static AliasEntry> {
    ALIAS_TABLE
        .iter()
        .find(|e| e.aliases.iter().any(|a| *a == name))
}

/// v0.33 phase 1: enumerate every known alias (first canonical
/// + every accepted synonym) as a flat list. Used by
/// `error_hints::hint_unknown_alias` to suggest the closest
/// match for typos like `sd1.5` → `sd15`.
pub fn all_known_aliases() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for entry in ALIAS_TABLE {
        for alias in entry.aliases {
            out.push(*alias);
        }
    }
    out
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
        assert_eq!(resolve_alias("sd21"), "nlightcho/stable-diffusion-2-1");
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
                "stablediffusionapi/pony-diffusion-v6-xl",
                "alias {alias} should resolve to a working Pony Diffusion v6 XL mirror",
            );
        }
    }
}
