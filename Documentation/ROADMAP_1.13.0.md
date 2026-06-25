# plakat 1.13.0 — roadmap

1.12.0 went all-in on **multiperson + a verified face-swap stack** (SCRFD / ArcFace /
inswapper ported and checked against onnxruntime, `convert-onnx`, and three fixed
never-verified components). That pushed the **memory & stability** work — which 1.11.0
already flagged and 1.12.0 deferred — out a second time. 1.13.0 pays that down, and
matures the multiperson identity path with the follow-ups the 1.12.0 effort surfaced.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — memory & stability (headline, twice-deferred)

The real ceiling on this 24 GB box is unified memory; these are the concrete bites:

- [x] **Gradient checkpointing** — INVESTIGATED, dead end on candle 0.10.2 (no recompute API; CustomOp bwd gives no parameter grads). See Documentation/GRADIENT_CHECKPOINTING.md. Original goal: verify the PixArt / SD3.5 / Cascade
      trainers on 24 GB (they OOM at the first backward). candle has no native support;
      prototype manual detach + recompute on one denoiser and measure. If infeasible,
      document the dead end so we stop re-attempting it.
- [x] **Cap render size on Metal** — DONE (warn on >1MP renders for Metal; `imaging::sizes::warn_large_for_metal`). Original: — the default `generate` size spikes Metal's
      single-buffer allocation on SD 1.5 / large SDXL and can OOM the host. Auto-cap or
      warn, or transparently tile large single-pass renders.
- [x] **OOM-guard tuning** — DONE (sustained window 3→5 inference / 12 training; `PLAKAT_OOM_GUARD_SUSTAINED`). Original: — the watchdog can fire on a transient first-backward / decode
      spike the OS would otherwise absorb via swap. Longer sustained window, or a
      "training vs inference" sensitivity.
- [ ] **Memory-bound SD3.5 DreamBooth** / `regional.sh sdxl|sd35` renders — unblocked once
      the above land.

## B — multiperson maturity (1.12.0 follow-ups)

The honest 1.12.0 ceiling: face-swap identity is strong only on **few, prominent,
roughly-frontal faces from photos**; crowds read faintly; hair/build come from the
generated figure. These items push that ceiling:

- [ ] **Face-restore / detail pass after swap** — small scene faces lose detail when the
      128² swap is composited back down. An ADetailer-style high-res face refine (the
      pieces exist) would make crowd faces read more clearly.
- [x] **Child / body-scale skeletons for `--pose`** — DONE (`--scale LABEL:0.7`). Original: — the synthetic skeleton is
      adult-proportioned, so a child persona renders adult-sized. Scale per a
      persona age/build hint.
- [ ] **FaceID in the inpaint identity path** — research puts FaceID at ~75–85% likeness vs
      plus-face's ~50–70%; wire it for the (non-swap) inpaint route.
- [ ] **Regional eps-blend (RFC M2)** — seam-free single coherent pass: per step, denoise
      each region with its identity and blend the noise predictions by soft masks. Reuses
      `region_mask` + portrait IP; low risk, ~N× wall time. (`RFC_MULTIPERSON_REVIEW.md`.)
- [ ] *(stretch)* **Masked decoupled IP-attention (RFC M3)** — true single-pass spatial
      identity routing on the vendored UNet. Medium-hard; the real fix for one-pass
      multi-identity.

## C — per-person LoRA (parked decision — revisit only on request)

The one lever that beats the face-swap ceiling for *whole-person* fidelity (hair + build +
any pose) is a **per-person LoRA** — but the research is clear that combining several in one
image bleeds identities, and it needs several images per person. Single-person
**LoRA-into-scene** is feasible and would be a strong "this exact person, any pose" path;
multi-person-in-one-image stays frontier. Not scheduled; here so the option isn't lost.

## D — carried map / debt (off-track, opt-in)

- [ ] **Multi-tile world maps** — stitch adjacent tiles into a seamless world map.
- [ ] River **deltas** at navigable mouths; **marsh hatching** for Wetland regions.
- [ ] Seasonal palette on the **painted** (`--map-render-sd`) path; political layer in
      GeoJSON / SVG export.

## Notes

- `--features metal` (Apple Silicon) / `--features cuda` (NVIDIA) for GPU; default build is
  CPU-only.
- Training output is non-deterministic → each trainer lands with a `corpus/*_train.sh`
  driver + a committed showcase, verified on-box where memory permits.
- ~~Flux training / regional~~ — still **skipped** while Flux is broken on Metal. Park until
  a CUDA / CI path exists.
