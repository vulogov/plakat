"""SDXL golden dumper — STUB. Follow models/sd15.py.

Capture points (the SDXL-specific bug surfaces):
  clip_l.penultimate / clip_g.penultimate — dual encoders.
  clip_g.pooled                            — the pooled add_embedding; note the EOT-pooling
                                             row must be the EOS id, NOT argmax (TI-vocab bug).
  add_time_ids                             — micro-conditioning (order is (h, w); regional swap bug).
  unet.mid, vae.decoded.
"""

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = ""
PLAKAT_ARCH = "sdxl_unet@1"
DEFAULT_THRESHOLDS = {
    "clip_l.penultimate": (0.9995, 0.03),
    "clip_g.penultimate": (0.9995, 0.03),
    "clip_g.pooled": (0.999, 0.05),
    "unet.mid": (0.999, 0.06),
    "vae.decoded": (0.999, 0.02),
}


def dump(fx, device: str):
    raise NotImplementedError(
        "author SDXL goldens following models/sd15.py — see ../correspondence.md for the "
        "dual-encoder + EOS-pooling + (h,w) time-id details"
    )
