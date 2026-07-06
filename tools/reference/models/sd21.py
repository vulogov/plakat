"""SD 2.1 golden dumper — validates the OpenCLIP ("!"-pad) branch of the CLIP
padding rule that the SDXL finding surfaced.

SD 2.1's tokenizer pads with `"!"` (id 0), NOT `<|endoftext|>` — the OpenCLIP
ViT-H convention (same as SDXL's CLIP-G / laion bigG). plakat's `Config::v2_1`
already pads with `"!"`, so this golden confirms that branch is correct (contrast
SDXL CLIP-L, which wrongly used `"!"` and was fixed to EOS). See ../correspondence.md.

Captures (matching t2i::capture_intermediates at clip_skip=1):
  clip.encoded  — diffusers `prompt_embeds` = last_hidden_state (POST final-layernorm),
                  the (1, 77, 1024) OpenCLIP conditioning fed to the UNet cross-attention.
  vae.decoded   — VAE decode of the shared deterministic latent.
"""

REPO = "nlightcho/stable-diffusion-2-1"  # matches plakat's `sd21` alias (stabilityai repo is gated)
REVISION = ""
PLAKAT_ARCH = "sd_core@1"
DEFAULT_THRESHOLDS = {
    "clip.encoded": (0.9995, 0.05),
    "vae.decoded": (0.999, 0.03),
}


def dump(fx, device: str):
    import torch
    from diffusers import StableDiffusionPipeline
    import fixtures

    dtype = torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(REPO, torch_dtype=dtype).to(device)
    captured = {}

    # clip.encoded: diffusers `prompt_embeds` = last_hidden_state (POST final-layernorm).
    # Padding is `"!"` (id 0) — the OpenCLIP pad token — exactly what plakat's v2_1 uses.
    tok = pipe.tokenizer(
        fx.prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(tok.input_ids)[0]  # (1, 77, 1024)
    captured["clip.encoded"] = prompt_embeds

    # vae.decoded: decode the SHARED deterministic latent (matches plakat's capture).
    latent = fixtures.deterministic_latent(4, fx.height // 8, fx.width // 8).to(device)
    with torch.no_grad():
        captured["vae.decoded"] = pipe.vae.decode(latent).sample
    return captured
