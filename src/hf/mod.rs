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
        // v0.13: 4-bit quantized Flux via GGUF. city96's mirrors are
        // the canonical community source. The transformer drops from
        // ~24 GB BF16 to ~7 GB Q4_K_S — Flux becomes practical on
        // 16 GB GPUs. Text encoders (T5-XXL, CLIP-L) stay at full
        // precision in Phase 1; quantized T5 lands in 1b.
        "flux-dev-gguf" | "flux-dev-q4" => "city96/FLUX.1-dev-gguf",
        "flux-schnell-gguf" | "flux-schnell-q4" => "city96/FLUX.1-schnell-gguf",
        // GGUF mirror for Fill — city96 ships this in the same
        // family as the BF16 model.
        "flux-fill-dev-gguf" | "flux-fill-gguf" | "flux-fill-q4" => {
            "city96/FLUX.1-Fill-dev-gguf"
        }
        other => other,
    }
}
