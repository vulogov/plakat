"""SDXL golden dumper.

Wired capture points (matching t2i::capture_intermediates):
  clip.encoded   — diffusers `prompt_embeds` = the concatenated dual-encoder penultimate
                   hidden states (1, 77, 2048) fed to the UNet cross-attention.
  clip_g.pooled  — diffusers `pooled_prompt_embeds` = the CLIP-G pooled add_embedding
                   (1, 1280). The EOS-pooling bug lived here (argmax picked a higher-id TI
                   trigger instead of the EOS row).

Still to wire (see ../correspondence.md): add_time_ids (order (h,w)), unet.mid, vae.decoded.
"""

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = ""
PLAKAT_ARCH = "sdxl_unet@1"
DEFAULT_THRESHOLDS = {
    "clip.encoded": (0.9995, 0.04),
    "clip_g.pooled": (0.999, 0.05),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusionXLPipeline

    dtype = torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(REPO, torch_dtype=dtype).to(device)

    # diffusers SDXL encode_prompt → (prompt_embeds, neg_embeds, pooled, neg_pooled). No CFG,
    # so we take the cond prompt_embeds + pooled — exactly what plakat captures without CFG.
    with torch.no_grad():
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=fx.prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return {
        "clip.encoded": prompt_embeds,   # (1, 77, 2048)
        "clip_g.pooled": pooled,         # (1, 1280)
    }
