"""PixArt-Σ (DiT) golden dumper.

Wired capture point (matching pixart::capture_intermediates):
  dit.pos_embed  — the 2D sin-cos patch positional embedding for the fixture resolution.
                   **The pos-embed scaling (H/W half-swap + base_size/interpolation) was a
                   real DiT bug.** Prompt-independent — a pure function of (config, resolution).

Still to wire (see ../correspondence.md): t5.hidden (BF16), adaln.embedded_timestep, dit.block0.

⚠ The pos-embed layout MUST match plakat's `build_2d_sincos_pos_embed` exactly (grid order,
interpolation_scale, shape `(1, tokens, hidden)`). diffusers computes this inside
`PatchEmbed`; below we call diffusers' own `get_2d_sincos_pos_embed` to author the golden.
Validate the shape + a couple of values against a plakat capture on first run.
"""

REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
REVISION = ""
PLAKAT_ARCH = "pixart_dit@1"
DEFAULT_THRESHOLDS = {
    "dit.pos_embed": (0.9999, 0.01),
    "dit.block0": (0.999, 0.1),   # first transformer block output (patch+adaLN+caption+block)
    "t5.hidden": (0.999, 0.5),    # T5 caption embedding WITH pad attention mask (large activations)
    "adaln.embedded_timestep": (0.999, 0.05),  # final-adaLN embedded timestep (1, hidden)
}


def dump(fx, device: str):
    import torch
    from diffusers import PixArtTransformer2DModel
    from diffusers.models.embeddings import get_2d_sincos_pos_embed

    # pos_embed is a pure function of (config, resolution) — no weights, no tokenizer/T5
    # (which needs tiktoken under transformers 5). Load ONLY the transformer config.
    cfg = PixArtTransformer2DModel.load_config(REPO, subfolder="transformer")
    hidden = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    grid = fx.height // 8 // cfg["patch_size"]  # square fixture → grid_h == grid_w
    interp = max(1.0, (cfg["sample_size"] * cfg["patch_size"]) / 64.0)  # mirror plakat's interp

    pe = get_2d_sincos_pos_embed(  # diffusers >=0.33: torch tensor, output_type='pt'
        embed_dim=hidden, grid_size=grid, base_size=cfg["sample_size"],
        interpolation_scale=interp, output_type="pt",
    ).float()
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)  # → (1, tokens, hidden)
    captured = {"dit.pos_embed": pe}

    # --- dit.block0: first transformer block on DETERMINISTIC inputs -------------------
    # Load ONLY the transformer (no T5/VAE). Feed a deterministic latent + deterministic
    # caption (LCG, same as plakat) + fixed t=500 + the fixture's resolution/aspect. Hook
    # transformer_blocks[0] to grab its output. This exercises patch-embed + adaLN +
    # caption-projection + the first block — the DiT block math, isolated from T5.
    import fixtures
    tf = PixArtTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.float32)
    tf.eval()
    max_tokens = cfg.get("max_caption_tokens") or 300
    caption_channels = cfg.get("caption_channels", 4096)
    latent = fixtures.deterministic_latent(cfg["in_channels"], fx.height // 8, fx.width // 8)
    caption = fixtures.deterministic_tensor((1, max_tokens, caption_channels), seed=2)
    timestep = torch.tensor([500.0])
    # Deterministic caption mask (encoder_attention_mask): first half real (1), second half
    # pad (0). Exercises the v2.1 cross-attention pad masking — must match plakat's synthetic mask.
    enc_mask = torch.cat([torch.ones(1, max_tokens // 2),
                          torch.zeros(1, max_tokens - max_tokens // 2)], dim=1)
    added_cond = {
        "resolution": torch.tensor([[float(fx.height), float(fx.width)]]),
        "aspect_ratio": torch.tensor([[1.0]]),
    }
    holder = {}
    h = tf.transformer_blocks[0].register_forward_hook(lambda m, i, o: holder.__setitem__("b0", o))
    with torch.no_grad():
        tf(hidden_states=latent, encoder_hidden_states=caption, timestep=timestep,
           encoder_attention_mask=enc_mask, added_cond_kwargs=added_cond, return_dict=False)
    h.remove()
    b0 = holder["b0"]
    captured["dit.block0"] = b0[0] if isinstance(b0, tuple) else b0  # (1, tokens, hidden)

    # --- adaln.embedded_timestep: the (1, hidden) vector the FINAL adaLN consumes -------
    # diffusers AdaLayerNormSingle returns (t_block, embedded_timestep); we want the 2nd.
    with torch.no_grad():
        _t_block, embedded = tf.adaln_single(
            timestep, added_cond, batch_size=1, hidden_dtype=torch.float32
        )
    captured["adaln.embedded_timestep"] = embedded  # (1, hidden)
    del tf  # free the DiT (~2.4GB) before loading T5-XXL (F32 ~19GB)

    # --- t5.hidden: caption embedding WITH the padding attention mask ------------------
    # diffusers ALWAYS passes attention_mask to T5; plakat previously didn't (the bug).
    # Use the flan-t5-base tokenizer (matches plakat's drop-in; the PixArt spiece tokenizer
    # needs tiktoken/sentencepiece which aren't installed). F32 to match plakat's CPU T5.
    from transformers import T5EncoderModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5 = T5EncoderModel.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    ti = tok(fx.prompt, padding="max_length", max_length=max_tokens, truncation=True, return_tensors="pt")
    with torch.no_grad():
        t5_hidden = t5(ti.input_ids, attention_mask=ti.attention_mask)[0].float()  # (1, max_tokens, 4096)
    captured["t5.hidden"] = t5_hidden
    return captured
