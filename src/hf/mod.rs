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
        "flux-schnell" | "FLUX.1-schnell" | "flux1-schnell" => {
            "black-forest-labs/FLUX.1-schnell"
        }
        "flux-dev" | "FLUX.1-dev" | "flux1-dev" => "black-forest-labs/FLUX.1-dev",
        other => other,
    }
}
