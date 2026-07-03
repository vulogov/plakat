"""SD 3.5-medium (MMDiT) golden dumper — STUB. Follow models/sd15.py.

Capture points (the MMDiT bug surfaces — all real past fixes):
  pooled_y            — the pooled conditioning, order [CLIP-L, CLIP-G] (the killer bug was
                        the swapped order).
  t5.hidden           — T5 caption embedding, encoded in BF16 (F16 overflowed → inf captions).
  timestep_embed      — timestep × 1000 scaling.
  mmdit.block0        — first joint block output (AdaLayerNormContinuous scale/shift order,
                        QK-norm).
  vae.decoded.
"""

REPO = "stabilityai/stable-diffusion-3.5-medium"
REVISION = ""
PLAKAT_ARCH = "mmdit_inner@1"
DEFAULT_THRESHOLDS = {
    "pooled_y": (0.999, 0.05),
    "t5.hidden": (0.999, 0.05),
    "mmdit.block0": (0.998, 0.08),
    "vae.decoded": (0.999, 0.02),
}


def dump(fx, device: str):
    raise NotImplementedError(
        "author SD3.5 goldens following models/sd15.py — verify pooled_y is [CLIP-L, CLIP-G] "
        "and T5 runs in BF16 (see ../correspondence.md)"
    )
