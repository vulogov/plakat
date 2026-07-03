"""AnimateDiff (SD1.5 + SDXL motion) golden dumper — STUB. Follow models/sd15.py.

Capture points (the motion + CFG surfaces):
  motion.block0       — first motion-module output (pos-embed placement was the dominant
                        AnimateDiff bug; corr here catches a regression).
  cfg_batch.layout    — a tiny tensor asserting the CFG conditioning is BLOCKED
                        [uncond×F, cond×F] for the SDXL path (the frame ≥ 2 scramble bug —
                        already guarded structurally in verify Tier 0; a golden confirms the
                        real motion forward uses it).
  unet.mid, vae.decoded.

Use an AESTHETIC base (DreamShaper), not vanilla SD1.5 — vanilla mosaics in diffusers too.
"""

REPO = "guoyww/animatediff-motion-adapter-v1-5-3"  # motion adapter; pair with a DreamShaper base
REVISION = ""
PLAKAT_ARCH = "animatediff@1"
DEFAULT_THRESHOLDS = {
    "motion.block0": (0.998, 0.08),
    "unet.mid": (0.999, 0.06),
    "vae.decoded": (0.999, 0.02),
}


def dump(fx, device: str):
    raise NotImplementedError(
        "author AnimateDiff goldens following models/sd15.py — DreamShaper base, verify the "
        "BLOCKED CFG layout on the SDXL path (see ../correspondence.md)"
    )
