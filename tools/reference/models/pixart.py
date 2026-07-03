"""PixArt-Σ (DiT) golden dumper — STUB. Follow models/sd15.py.

Capture points (the DiT bug surfaces — all real past fixes):
  t5.hidden               — T5 caption, BF16 (F16 overflow → inf).
  dit.pos_embed           — patch pos-embed (H/W half-swap + base_size/interp scaling bug).
  adaln.embedded_timestep — final-adaLN uses the embedded timestep.
  dit.block0              — first transformer block (+ 2K KV-compression when present —
                            detected from the kv_proj_conv2d tensor, not the repo name).
  vae.decoded.
"""

REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
REVISION = ""
PLAKAT_ARCH = "pixart_dit@1"
DEFAULT_THRESHOLDS = {
    "t5.hidden": (0.999, 0.05),
    "dit.pos_embed": (0.9999, 0.01),
    "dit.block0": (0.998, 0.08),
    "vae.decoded": (0.999, 0.02),
}


def dump(fx, device: str):
    raise NotImplementedError(
        "author PixArt-Σ goldens following models/sd15.py — verify pos-embed scaling + "
        "embedded-timestep adaLN + BF16 T5 (see ../correspondence.md)"
    )
