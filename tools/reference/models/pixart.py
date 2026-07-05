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
    return {"dit.pos_embed": pe}
