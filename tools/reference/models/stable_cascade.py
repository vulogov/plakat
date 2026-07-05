"""Stable Cascade golden dumper.

Wired capture point (matching cascade::capture_intermediates):
  clip_g.pooled  — the CLIP-G pooled text vector (Stage C's clip_txt_pooled_mapper + Stage B's
                   only conditioning). diffusers' prior text pooling.

Still to wire (see ../correspondence.md): effnet (Stage-C image cond), stage_c.block0.
"""

REPO = "stabilityai/stable-cascade-prior"  # the prior; the combined pipeline is stabilityai/stable-cascade
REVISION = ""
PLAKAT_ARCH = "cascade_prior@1"
DEFAULT_THRESHOLDS = {
    "clip_g.pooled": (0.999, 0.05),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableCascadePriorPipeline

    prior = StableCascadePriorPipeline.from_pretrained(REPO, torch_dtype=torch.float32).to(device)
    # No CFG → the cond pooled. diffusers cascade prior encode_prompt returns
    # (prompt_embeds, prompt_embeds_pooled, negative_embeds, negative_embeds_pooled).
    with torch.no_grad():
        _, pooled, _, _ = prior.encode_prompt(
            device=device, batch_size=1, num_images_per_prompt=1,
            do_classifier_free_guidance=False, prompt=fx.prompt,
        )
    # diffusers returns (1, 1, 1280); plakat's cascade pooled is (1, 1280) → squeeze the seq dim.
    if pooled.dim() == 3:
        pooled = pooled.squeeze(1)
    return {"clip_g.pooled": pooled}
