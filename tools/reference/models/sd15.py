"""SD 1.5 golden dumper — the WORKED REFERENCE EXAMPLE other families copy.

Captures three load-bearing intermediates (the historically bug-prone surfaces):
  clip_l.penultimate  — CLIP text-encoder penultimate hidden state (the clip-skip layer;
                        plakat's SD clip-skip bug lived exactly here).
  unet.mid            — UNet mid-block output for one forward at a fixed timestep.
  vae.decoded         — VAE decode of the fixture's initial latent (the F16-VAE class).

⚠ SCAFFOLD: the hook points + the seed→latent construction below must be validated against
plakat's own capture points when authoring (see ../correspondence.md). A mismatch in EITHER
side means the golden doesn't correspond to what plakat taps — chase it before trusting the
numbers. This file shows the *pattern*; treat the specifics as a starting point.
"""

REPO = "runwayml/stable-diffusion-v1-5"
REVISION = ""  # pin the resolved commit sha when authoring
PLAKAT_ARCH = "sd_core@1"

# Correlation must be near-perfect for correctness; max_abs loose enough for BF16 rounding.
DEFAULT_THRESHOLDS = {
    "clip_l.penultimate": (0.9995, 0.03),
    "unet.mid": (0.999, 0.05),
    "vae.decoded": (0.999, 0.02),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = torch.float32  # author goldens in F32; plakat's capture is cast to F32 to compare
    pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=dtype)
    pipe = pipe.to(device)
    captured = {}

    # --- clip_l.penultimate: hidden_states[-2] of the CLIP text encoder ---
    # plakat's clip-skip returns the PENULTIMATE hidden state (before the final layer norm);
    # matching that here is the whole point (the SD clip-skip noise bug was this row).
    tok = pipe.tokenizer(
        fx.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc = pipe.text_encoder(tok.input_ids, output_hidden_states=True)
    captured["clip_l.penultimate"] = enc.hidden_states[-2]
    text_embeds = enc.hidden_states[-1]  # the encoder-hidden-states fed to the UNet's cross-attn

    # --- unet.mid: one UNet forward at a fixed timestep, tap the mid block ---
    g = torch.Generator(device=device).manual_seed(fx.seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, fx.height // 8, fx.width // 8),
        generator=g, device=device, dtype=dtype,
    )
    mid_out = {}
    h = pipe.unet.mid_block.register_forward_hook(lambda m, i, o: mid_out.__setitem__("t", o))
    t = torch.tensor([500], device=device)  # a mid-schedule timestep; must match plakat's tap
    with torch.no_grad():
        pipe.unet(latents, t, encoder_hidden_states=text_embeds)
    h.remove()
    captured["unet.mid"] = mid_out["t"]

    # --- vae.decoded: decode the initial latent ---
    with torch.no_grad():
        dec = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    captured["vae.decoded"] = dec

    return captured
