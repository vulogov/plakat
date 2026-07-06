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
    # first joint block x-stream: corr 1.0 (block math correct); max_abs headroom for the
    # 1536-d activations where candle-vs-torch accumulation differs by ~0.29 (~sub-%).
    "mmdit.block0": (0.999, 0.5),
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
    captured = {"pooled_y": pooled}
    del pipe  # free the CLIP encoders before loading the MMDiT

    # --- mmdit.block0: first joint block on DETERMINISTIC inputs -----------------------
    # Load ONLY the MMDiT (no CLIP/T5/VAE — y/context are synthetic). Feed a deterministic
    # 16-ch latent + fixed t=500 + deterministic pooled `y` (seed 3) + deterministic
    # `context` (seed 2). Hook transformer_blocks[0]; take its x-stream (image tokens,
    # 1024 = 32×32). Exercises patch/pos + timestep+vector embed + context-embed + block.
    import fixtures
    from diffusers import SD3Transformer2DModel

    tf = SD3Transformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.float32)
    tf.eval()
    CONTEXT_SEQ = 154  # arbitrary but must match plakat's capture (the shared contract)
    in_ch = tf.config.in_channels
    pooled_dim = tf.config.pooled_projection_dim
    ctx_dim = tf.config.joint_attention_dim
    latent = fixtures.deterministic_latent(in_ch, fx.height // 8, fx.width // 8)  # (1,16,64,64)
    y = fixtures.deterministic_tensor((1, pooled_dim), seed=3)
    context = fixtures.deterministic_tensor((1, CONTEXT_SEQ, ctx_dim), seed=2)
    timestep = torch.tensor([500.0])

    holder = {}
    h = tf.transformer_blocks[0].register_forward_hook(lambda m, i, o: holder.__setitem__("b0", o))
    with torch.no_grad():
        tf(hidden_states=latent, encoder_hidden_states=context, pooled_projections=y,
           timestep=timestep, return_dict=False)
    h.remove()
    o = holder["b0"]
    # diffusers JointTransformerBlock returns (encoder_hidden_states, hidden_states); pick the
    # image (x) stream by token count (1024 = 32×32) rather than relying on tuple order.
    parts = list(o) if isinstance(o, (tuple, list)) else [o]
    x_stream = max(parts, key=lambda t: t.shape[1])
    captured["mmdit.block0"] = x_stream  # (1, 1024, hidden)
    return captured
