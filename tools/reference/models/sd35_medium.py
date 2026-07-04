"""SD 3.5-medium (MMDiT) golden dumper.

Wired capture point (matching sd3::capture_intermediates):
  pooled_y  — diffusers `pooled_prompt_embeds` = the concat of the two CLIP pooled vectors
              fed to the MMDiT. **The concat ORDER was the killer SD3 bug** — this golden is
              diffusers' authoritative order; the comparison proves plakat's `pooled_y` matches.

Still to wire (see ../correspondence.md): t5.hidden (BF16), mmdit.block0.
"""

REPO = "stabilityai/stable-diffusion-3.5-medium"
REVISION = ""
PLAKAT_ARCH = "mmdit_inner@1"
DEFAULT_THRESHOLDS = {
    "pooled_y": (0.999, 0.05),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype = torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(REPO, torch_dtype=dtype).to(device)

    # No CFG → take the cond pooled. diffusers SD3 encode_prompt returns
    # (prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds).
    with torch.no_grad():
        _, _, pooled, _ = pipe.encode_prompt(
            prompt=fx.prompt, prompt_2=None, prompt_3=None,
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
    # `pooled` is the concat of the two CLIP pooled vectors — the ORDER is what we verify.
    return {"pooled_y": pooled}
