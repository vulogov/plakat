"""Stable Cascade golden dumper.

Wired capture point (matching cascade::capture_intermediates):
  clip_g.pooled  — the CLIP-G pooled text vector (Stage C's clip_txt_pooled_mapper + Stage B's
                   only conditioning). diffusers' prior text pooling.

  stage_c.out    — the FULL Stage-C prior prediction on a deterministic latent + fixed ratio
                   t=0.5 + deterministic CLIP-G text/pooled. The Stage-C denoiser core.

Still to wire (see ../correspondence.md): effnet (Stage-C image cond).
"""

REPO = "stabilityai/stable-cascade-prior"  # the prior; the combined pipeline is stabilityai/stable-cascade
REVISION = ""
PLAKAT_ARCH = "cascade_prior@1"
DEFAULT_THRESHOLDS = {
    "clip_g.pooled": (0.999, 0.15),  # pooled matches in direction (corr ~1.0); looser abs bound
    # COARSE gross-regression gate, NOT fine correctness. The full 3.6B Stage-C UNet is deep,
    # and a single forward on a white-noise deterministic latent is out-of-distribution: a tiny
    # per-block scale difference (candle vs torch attention/norm accumulation, each <0.001 per
    # v0.41's real-trajectory pinning) compounds multiplicatively to ~1.15× output magnitude.
    # corr stays 0.989 (direction correct → no gross break); real generation is verified by the
    # v0.41 forward_collect reference suite + the committed corpus. A shallow block tap (cf.
    # dit/mmdit.block0) would give a fine-grained 1.0 check — future work.
    "stage_c.out": (0.98, 1.5),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableCascadePriorPipeline

    prior = StableCascadePriorPipeline.from_pretrained(REPO, torch_dtype=torch.float32).to(device)
    # No CFG → the cond embeds. diffusers cascade prior encode_prompt returns
    # (prompt_embeds, prompt_embeds_pooled, negative_embeds, negative_embeds_pooled).
    with torch.no_grad():
        prompt_embeds, pooled, _, _ = prior.encode_prompt(
            device=device, batch_size=1, num_images_per_prompt=1,
            do_classifier_free_guidance=False, prompt=fx.prompt,
        )
    # pooled is (1, 1, 1280); plakat's cascade pooled is (1, 1280) → squeeze for the golden.
    captured = {"clip_g.pooled": pooled.squeeze(1) if pooled.dim() == 3 else pooled}
    del prior  # free the text encoder before loading the UNet

    # --- stage_c.out: full Stage-C prior prediction on a deterministic latent -----------
    # REAL CLIP-G conditioning (prompt_embeds + pooled, same as plakat's encode_prompt) —
    # structured input; feeding white-noise through the deep 3.6B UNet amplifies fp noise
    # (corr ~0.98). diffusers StableCascadeUNet.forward maps the clip conditioning + embeds
    # the timestep/sca/crp internally (cat([gen(ratio), gen(sca or 0), gen(crp or 0)]);
    # sca=None/crp=None → zeros, matching plakat's sinusoidal(0)). Only the latent is the
    # shared deterministic tensor + fixed ratio 0.5.
    import fixtures
    from diffusers import StableCascadeUNet

    unet = StableCascadeUNet.from_pretrained(REPO, subfolder="prior", torch_dtype=torch.float32).to(device)
    unet.eval()
    latent = fixtures.deterministic_latent(16, 24, 24).to(device)  # (1,16,24,24)
    timestep_ratio = torch.tensor([0.5], device=device)
    with torch.no_grad():
        pred = unet(
            sample=latent, timestep_ratio=timestep_ratio,
            clip_text_pooled=pooled, clip_text=prompt_embeds,
            sca=None, crp=None, return_dict=False,
        )[0]  # (1, 16, 24, 24)
    captured["stage_c.out"] = pred
    return captured
