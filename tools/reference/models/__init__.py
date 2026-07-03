# Per-family golden dumpers. Each module exposes:
#   REPO / REVISION      — the HF repo (+ revision) diffusers loads.
#   PLAKAT_ARCH          — arch tag recorded in the manifest (bump on module changes).
#   DEFAULT_THRESHOLDS   — optional dict[name -> (corr_min, max_abs)] per capture point.
#   dump(fx, device)     — run the fixture, return dict[name -> torch.Tensor] (any dtype/device;
#                          dump.py normalizes to F32/CPU). Names MUST match plakat's TensorTap
#                          capture-point names — see ../correspondence.md.
