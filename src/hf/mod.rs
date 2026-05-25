pub mod cache;
pub mod download;
pub mod info;
pub mod search;

/// Map short aliases to canonical HuggingFace repo ids.
///
/// Note: Runway removed `runwayml/stable-diffusion-v1-5` from HF in mid-2024;
/// the community mirror at `stable-diffusion-v1-5/stable-diffusion-v1-5` is
/// what's used now.
pub fn resolve_alias(name: &str) -> &str {
    match name {
        "sd15" | "sd-1.5" => "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "runwayml/stable-diffusion-v1-5" => "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "sd21" | "sd-2.1" => "stabilityai/stable-diffusion-2-1",
        "sdxl" => "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-turbo" => "stabilityai/sdxl-turbo",
        // v0.12: SDXL Inpainting. 9-channel UNet (4 latent + 1 mask + 4
        // masked_image_latents). Loaded by SdCore with in_channels=9.
        "sdxl-inpaint" | "sdxl-inpainting" => "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        // v0.12: SD 1.5 Inpainting. Same 9-channel UNet contract as
        // SDXL-Inpaint, just on the SD 1.5 architecture (cross_attn_dim
        // 768, 4 down blocks). Runway's original repo
        // (`runwayml/stable-diffusion-inpainting`) was pulled in 2024;
        // the community mirror parallels the SD 1.5 base mirror.
        "sd15-inpaint" | "sd15-inpainting" | "sd-1.5-inpaint" | "sd-1.5-inpainting" => {
            "stable-diffusion-v1-5/stable-diffusion-inpainting"
        }
        // Back-compat for the removed Runway repo id.
        "runwayml/stable-diffusion-inpainting" => {
            "stable-diffusion-v1-5/stable-diffusion-inpainting"
        }
        "flux-schnell" | "FLUX.1-schnell" | "flux1-schnell" => {
            "black-forest-labs/FLUX.1-schnell"
        }
        "flux-dev" | "FLUX.1-dev" | "flux1-dev" => "black-forest-labs/FLUX.1-dev",
        // v0.13 phase 2: Flux.1-Fill-dev — BFL's dedicated inpainting
        // checkpoint. Same DiT architecture as Flux.1-dev except
        // `img_in` takes 384 input channels (64 noise + 64 masked
        // latent + 256 image-space mask). Gated repo; HF_TOKEN
        // required.
        "flux-fill-dev" | "flux-fill" | "flux1-fill-dev" | "FLUX.1-Fill-dev" => {
            "black-forest-labs/FLUX.1-Fill-dev"
        }
        // v0.15 phase 4: BFL "concept" Flux checkpoints. Canny / Depth
        // conditioning is baked into a 128-channel `img_in` (64 noise
        // + 64 VAE-encoded conditioning latent). Caller supplies the
        // canny edge map / depth map; the model card recommends
        // guidance ~30. Both are gated repos.
        "flux-canny-dev" | "flux1-canny-dev" | "FLUX.1-Canny-dev" => {
            "black-forest-labs/FLUX.1-Canny-dev"
        }
        "flux-depth-dev" | "flux1-depth-dev" | "FLUX.1-Depth-dev" => {
            "black-forest-labs/FLUX.1-Depth-dev"
        }
        // v0.18: FLUX.1-Kontext-dev — BFL's image-editing checkpoint.
        // Same architecture as Flux.1-dev (`img_in` stays at 64); the
        // difference is at the DiT input level — a reference image is
        // VAE-encoded and sequence-concatenated onto the noise tokens.
        // Gated repo; HF_TOKEN required.
        "flux-kontext-dev" | "flux1-kontext-dev" | "FLUX.1-Kontext-dev" => {
            "black-forest-labs/FLUX.1-Kontext-dev"
        }
        // v0.18 Kontext phase 3: GGUF (4-bit/6-bit/8-bit) Kontext
        // via the unsloth mirror. Filename convention matches the
        // city96 Flux GGUF packs (`flux1-kontext-dev-Q4_K_M.gguf`).
        // Not gated.
        "flux-kontext-dev-gguf" | "flux-kontext-gguf" | "flux-kontext-q4" => {
            "unsloth/FLUX.1-Kontext-dev-GGUF"
        }
        // v0.13: 4-bit quantized Flux via GGUF. city96's mirrors are
        // the canonical community source. The transformer drops from
        // ~24 GB BF16 to ~7 GB Q4_K_S — Flux becomes practical on
        // 16 GB GPUs. Text encoders (T5-XXL, CLIP-L) stay at full
        // precision in Phase 1; quantized T5 lands in 1b.
        "flux-dev-gguf" | "flux-dev-q4" => "city96/FLUX.1-dev-gguf",
        "flux-schnell-gguf" | "flux-schnell-q4" => "city96/FLUX.1-schnell-gguf",
        // GGUF mirror for Fill — city96 ships this in the same
        // family as the BF16 model.
        // v0.14 phase 1a / 8a: Stable Diffusion 3 / 3.5 (Stability AI's
        // MMDiT). All gated repos — HF_TOKEN required.
        "sd35-medium" | "sd3.5-medium" | "stable-diffusion-3.5-medium" => {
            "stabilityai/stable-diffusion-3.5-medium"
        }
        "sd35-large" | "sd3.5-large" | "stable-diffusion-3.5-large" => {
            "stabilityai/stable-diffusion-3.5-large"
        }
        "sd35-large-turbo"
        | "sd3.5-large-turbo"
        | "stable-diffusion-3.5-large-turbo" => {
            "stabilityai/stable-diffusion-3.5-large-turbo"
        }
        // Original SD3 (June 2024 release) — superseded by 3.5 but
        // some workflows still target it.
        "sd3-medium" | "stable-diffusion-3-medium" => {
            "stabilityai/stable-diffusion-3-medium"
        }
        // v0.14 phase 2d: NF4 (bitsandbytes 4-bit) Flux. lllyasviel's
        // pack is the canonical community NF4 source. Weights stay
        // 4-bit at inference (~6 GB transformer); per-call dequant is
        // slower than GGUF Q4 but works on any candle device.
        "flux-dev-nf4" | "flux-nf4" | "flux1-dev-nf4" => {
            "lllyasviel/flux1-dev-bnb-nf4-v2"
        }
        // v0.14 phase 3: Flux Redux adapter (image-conditioned Flux
        // via SigLIP-so400m). Plakat's pipeline pulls the adapter
        // from this repo on demand when `--redux-image` is set;
        // resolving the alias is mostly for `plakat models` UX.
        "flux-redux" | "flux-redux-dev" | "FLUX.1-Redux-dev" => {
            "black-forest-labs/FLUX.1-Redux-dev"
        }
        "flux-fill-dev-gguf" | "flux-fill-gguf" | "flux-fill-q4" => {
            "city96/FLUX.1-Fill-dev-gguf"
        }
        other => other,
    }
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
}
