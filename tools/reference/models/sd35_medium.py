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
    # corr is the correctness signal for the concat ORDER (0.99998 → order is right).
    # max_abs 0.15 headroom matches SDXL/Cascade clip_g.pooled: the CLIP-G pooled half
    # carries large-magnitude elements where candle-vs-torch differ by ~0.085 abs (~0.1%).
    "pooled_y": (0.999, 0.15),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype = torch.float32
    # `pooled_y` is the concat of the two CLIP pooled vectors — it does NOT use T5.
    # Skip T5-XXL (F32 ~19GB → OOMs 24GB) and the VAE. Keep the transformer: diffusers'
    # encode_prompt still runs the T5 branch (with text_encoder_3=None it builds a zero
    # pad sized from `transformer.config.joint_attention_dim`), so the MMDiT must stay
    # loaded — but that + the two CLIP encoders fits 24GB in F32.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        REPO, torch_dtype=dtype,
        text_encoder_3=None, tokenizer_3=None,  # no T5-XXL
        vae=None,                                # not needed for text pooling
    ).to(device)

    # No CFG → take the cond pooled. diffusers SD3 encode_prompt returns
    # (prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds).
    with torch.no_grad():
        _, _, pooled, _ = pipe.encode_prompt(
            prompt=fx.prompt, prompt_2=None, prompt_3=None,
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
    # `pooled` is the concat of the two CLIP pooled vectors — the ORDER is what we verify.
    return {"pooled_y": pooled}
