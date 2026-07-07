"""AnimateDiff (SD1.5 V3 motion) golden dumper.

Capture point (matching the AnimateDiff branch in verify::tier1::run_model):
  motion.block0  — the first temporal-transformer motion module
                   (`down_blocks[0].motion_modules[0]`) on a DETERMINISTIC per-frame input.
                   The pos-embed placement was the dominant AnimateDiff bug (v0.43); a corr
                   check here catches a regression. The module weights come from the adapter
                   (base-independent), so no base model is needed to author.

The CFG BLOCKED-batch-layout bug is separately guarded in verify Tier 0 (no weights).
"""

REPO = "guoyww/animatediff-motion-adapter-v1-5-3"  # the V3 motion adapter
REVISION = ""
PLAKAT_ARCH = "animatediff@1"
DEFAULT_THRESHOLDS = {
    "motion.block0": (0.999, 0.05),  # one temporal transformer; shallow → tight bound
}


def dump(fx, device: str):
    import torch
    from diffusers import MotionAdapter
    import fixtures

    adapter = MotionAdapter.from_pretrained(REPO, torch_dtype=torch.float32).to(device)
    adapter.eval()
    # First motion module: down_blocks[0].motion_modules[0], 320-channel (SD 1.5 block 0).
    mm0 = adapter.down_blocks[0].motion_modules[0]
    channels = mm0.proj_in.in_features  # 320

    # Deterministic per-frame input (B*F, C, H, W): B=1, F=16 (V3 window), 8×8 spatial. seed 4
    # matches plakat's `verify::deterministic_tensor(&[16, C, 8, 8], 4, ...)`.
    num_frames = 16
    inp = fixtures.deterministic_tensor((num_frames, channels, 8, 8), seed=4).to(device)
    with torch.no_grad():
        out = mm0(inp, num_frames=num_frames)
    # AnimateDiffTransformer3D.forward returns the residual-added tensor (matches plakat).
    out = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
    return {"motion.block0": out}  # (16, 320, 8, 8)
