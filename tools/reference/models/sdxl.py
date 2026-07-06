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
    # corr is the correctness signal (the CLIP-L pad-token fix took this to 1.00000).
    # max_abs headroom absorbs the CLIP-G half's pre-final-LN attention-sink activations
    # (magnitude ~100+ over 32 layers), where candle-vs-torch accumulation order differs by
    # ~0.3 abs / ~0.3% relative — not a correctness gap.
    "clip.encoded": (0.9999, 0.5),
    "clip_g.pooled": (0.999, 0.15),  # pooled vectors match in direction (corr ~1.0); larger magnitude → looser abs bound
    "unet.out": (0.999, 0.05),       # full noise prediction ε (down+mid+up); O(1) scale
    "unet.mid": (0.999, 0.2),        # mid-block activation (pre-up); larger magnitude → looser abs bound
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusionXLPipeline
    import fixtures

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
    captured = {
        "clip.encoded": prompt_embeds,   # (1, 77, 2048)
        "clip_g.pooled": pooled,         # (1, 1280)
    }

    # --- unet.out (full ε) + unet.mid (mid-block) on the SHARED latent at a FIXED timestep ---
    # add_time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w] — matches plakat's
    # `build_add_time_ids_base(h, w)` = [h, w, 0, 0, h, w].
    latent = fixtures.deterministic_latent(4, fx.height // 8, fx.width // 8).to(device)
    add_time_ids = torch.tensor(
        [[fx.height, fx.width, 0, 0, fx.height, fx.width]], dtype=dtype, device=device
    )
    added_cond = {"text_embeds": pooled, "time_ids": add_time_ids}

    # Capture the mid-block output via a forward hook (diffusers UNet returns only ε).
    mid_holder = {}
    h = pipe.unet.mid_block.register_forward_hook(
        lambda m, i, o: mid_holder.__setitem__("mid", o)
    )
    with torch.no_grad():
        eps = pipe.unet(
            latent, 500, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond
        ).sample  # (1, 4, 64, 64)
    h.remove()
    captured["unet.out"] = eps
    captured["unet.mid"] = mid_holder["mid"]  # (1, 1280, 8, 8)
    return captured
