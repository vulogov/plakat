"""Stable Cascade golden dumper — STUB. Follow models/sd15.py.

Capture points (the Wuerstchen/Cascade surfaces):
  clip_g.pooled     — CLIP-G text pooling.
  effnet            — the EfficientNetV2-S image-conditioning embedding (Stage C).
  stage_c.block0    — first Stage-C prior block (FiLM time injection, sca/crp).
  stage_b.decoded / vae.decoded — Stage A/B decode.

NOTE: Cascade is a 3-stage model; author only the conditioning + one prior block for Tier 1.
"""

REPO = "stabilityai/stable-cascade"
REVISION = ""
PLAKAT_ARCH = "cascade_prior@1"
DEFAULT_THRESHOLDS = {
    "clip_g.pooled": (0.999, 0.05),
    "effnet": (0.999, 0.05),
    "stage_c.block0": (0.998, 0.08),
}


def dump(fx, device: str):
    raise NotImplementedError(
        "author Stable Cascade goldens following models/sd15.py — Wuerstchen scheduler, "
        "FiLM time injection, effnet conditioning (see ../correspondence.md)"
    )
