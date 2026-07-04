"""SD 1.5 golden dumper — the WORKED REFERENCE EXAMPLE other families copy.

Captures the load-bearing intermediates (the historically bug-prone surfaces):
  clip.encoded        — the text conditioning fed to the UNet cross-attention = diffusers'
                        `prompt_embeds` (last_hidden_state, POST final-layernorm). This is
                        what plakat's `encode_prompt` returns at the default clip-skip, and
                        the home of the SD clip-skip noise bug. (NOTE: this is the FINAL
                        hidden state, not the penultimate — the clip-skip FIX made plakat's
                        default return the final, matching diffusers.)
  unet.mid            — UNet mid-block output for one forward at a fixed timestep. (Not yet
                        wired on the plakat side — needs a UNet-internal tap.)
  vae.decoded         — VAE decode of the fixture's initial latent (the F16-VAE class). (Not
                        yet wired — needs a matching seed→latent.)

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
    "clip.encoded": (0.9995, 0.03),
    "vae.decoded": (0.999, 0.03),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusionPipeline
    import fixtures

    dtype = torch.float32  # author goldens in F32; plakat's capture is cast to F32 to compare
    pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=dtype).to(device)
    captured = {}

    # --- clip.encoded: diffusers `prompt_embeds` = last_hidden_state (POST final-layernorm) ---
    # This is exactly what plakat's `encode_prompt` returns at the default clip-skip and feeds
    # the UNet cross-attention. `text_encoder(ids)[0]` is the post-layernorm last hidden state.
    tok = pipe.tokenizer(
        fx.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(tok.input_ids)[0]  # (1, 77, 768), post-final-layernorm
    captured["clip.encoded"] = prompt_embeds

    # --- vae.decoded: decode the SHARED deterministic latent (no seeded RNG; matches plakat) ---
    latent = fixtures.deterministic_latent(4, fx.height // 8, fx.width // 8).to(device)
    with torch.no_grad():
        captured["vae.decoded"] = pipe.vae.decode(latent).sample  # decode the RAW latent (both sides)

    # unet.mid: author once the plakat side wires a UNet-internal tap. See ../correspondence.md.
    return captured
