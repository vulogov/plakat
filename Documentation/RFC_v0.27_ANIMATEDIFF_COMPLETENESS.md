# RFC v0.27 — AnimateDiff completeness

**Status:** decisions locked 2026-05-28 — ready for phase 0.

**Predecessor:** [`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md) — shipped AnimateDiff *infrastructure* with inference dispatch deferred to v0.26.1.

## 1. TL;DR

v0.27 is the **AnimateDiff completeness** push. It closes the v0.26.1 inference-dispatch deferral and lands the three carry themes the user named when scoping this cycle:

1. **SDXL motion adapter** — AnimateDiff on SDXL via `guoyww/animatediff-motion-adapter-sdxl-beta`.
2. **AnimateDiff + ControlNet** — per-frame coherent control conditioning (single control image applied to every frame in v0.27; per-frame video-control deferred).
3. **Long-form AnimateDiff** — sliding-window inference over the existing V3 motion adapter; overlap-blend in latent space to produce 64+ frame outputs from V3's 32-frame native cap.

After v0.27, AnimateDiff is **feature-complete** across the two SD families (SD 1.5 + SDXL) on the two main composition axes (ControlNet + long-form). HotShot-XL, per-frame ControlNet inputs (video-to-video), and `plakat.animate` Bund integration are deferred — each is a deliberate v0.28+ surface, not a v0.27 carry.

Estimated **9 phases / ~13–17 sessions**. Sized comparably to v0.26.

## 2. Why this is the v0.27 cycle

1. **Inference dispatch is non-negotiable.** v0.26 shipped working infrastructure but `plakat animate --animatediff` bails today. Folding the v0.26.1 closure into phase 0 means a single release where SDXL / CN / long-form can each be quality-validated against a working baseline.

2. **All three themes share `forward_with_motion`.** SDXL motion adapter, AnimateDiff + ControlNet, and sliding-window long-form all extend (or call into) the same `Sd15MotionUNet::forward_with_motion` path that phase 0 brings online. Splitting them across cycles would mean re-validating the same core path three times.

3. **SD 1.5 → SDXL parity is overdue.** The motion-adapter ecosystem split a year ago into SD 1.5 V3 (highest quality, ubiquitous community LoRA support) and SDXL beta (newer base, larger generation surface). Shipping both in one cycle means we don't have to design two animate documentation arcs.

4. **Long-form is community-table-stakes.** Pure V3 caps at 16-frame outputs (the official V3 motion-module training distribution); 32 is the empirical extension limit before motion drift dominates. Real-world AnimateDiff usage on Civitai / Reddit / Discord runs sliding-window for the 4-second-plus clips. Without it, plakat's animate surface looks toy-grade next to AUTOMATIC1111's AnimateDiff extension.

## 3. Decisions locked (4)

The user answered these via AskUserQuestion on 2026-05-28:

### 3.1 Release shape — **Fold v0.26.1 into v0.27 phase 0**

One release covering inference dispatch + SDXL + CN + long-form. Quality validation happens alongside the new themes. The alternative (ship v0.26.1 standalone first, then v0.27) was rejected on operational-simplicity grounds: same total work, two releases instead of one, and v0.27 themes would have to gate on a tag that exists primarily as a status update.

### 3.2 Block-vendoring depth — **Keep v0.26's block-boundary splice**

The v0.26 phase 3 splice applies motion modules at down/up block *outputs* rather than per-(resnet+attn) layer (the latter requires vendoring ~800 LOC of CrossAttnDownBlock2D + CrossAttnUpBlock2D from diffusers). Phase 0 ships inference with the existing splice. If quality is empirically poor, the per-layer upgrade lands inside the cycle (escalation budget ~2 sessions). The conditional-upgrade option (option C from the question set) was rejected as functionally identical to this default — every cycle has implicit "we'll fix it if it breaks" scope.

### 3.3 SDXL adapter — **`guoyww/animatediff-motion-adapter-sdxl-beta`**

Official sibling to the V3 SD 1.5 adapter. Marked "beta" upstream but stable enough for production. Same diffusers-rs loading path. The "multiple variants" option (HotShot-XL + animate-x-XL) was rejected: HotShot-XL is a different architecture, not a drop-in adapter swap; ships in v0.28+ if at all.

### 3.4 Long-form approach — **Sliding window over V3**

Generate overlapping 16-frame windows; cross-fade in latent space with a linear ramp over the overlap region; stitch. Output up to 64-frame clips reliably, 128–256-frame clips with quality degradation. Works on top of the existing V3 path (no new architecture). HotShot-XL was rejected as too-big-for-one-cycle. The "both" option was rejected because shipping HotShot-XL needs its own design pass (different scheduler, different motion module shape, ~14 GB weights vs V3's ~1.4 GB).

## 4. Phase plan

| # | Phase | Sessions |
|---|---|---|
| 0 | V3 inference dispatch (SD 1.5) — close v0.26 deferral | 2–3 |
| 1 | SDXL motion adapter load — config + UNet block splice | 3 |
| 2 | SDXL inference dispatch — `plakat animate --animatediff --model sdxl-*` | 1–2 |
| 3 | AnimateDiff + ControlNet (SD 1.5) — single control per N frames | 2 |
| 4 | AnimateDiff + ControlNet (SDXL) — extend phase 3 | 1–2 |
| 5 | Long-form sliding window (SD 1.5) — latent cross-fade stitch | 2–3 |
| 6 | Long-form sliding window (SDXL) — extend phase 5 | 1 |
| 7 | Tutorials + integration tests | 1 |
| 8 | Cycle close-out (RFC retrospective + 7-step release) | 0.5 |

## 5. Phase 0 — V3 inference dispatch (the v0.26.1 work)

The v0.26 phase 5 `AnimateDiffPipeline::generate()` currently bails with:

```
not yet implemented in v0.26.0 — folded into v0.27 phase 0
```

Phase 0 replaces that bail with a working N-frame scheduler loop:

1. **Latent setup.** Allocate a batch of shape `(N, 4, H/8, W/8)` for N frames. Single seed → deterministic per-frame seeds: `seed`, `seed+1`, ..., `seed+N-1`.

2. **Scheduler loop.** Standard SD 1.5 sampler steps. Each step calls `Sd15MotionUNet::forward_with_motion(latents, t, prompt_embeds, motion_modules=Some(&self.motion), num_frames=N)`. Motion modules consume the full N-frame batch on the frame axis; the outer UNet treats each frame independently.

3. **VAE decode.** Per-frame decode loop over the N latents → N `DynamicImage`s.

4. **Output dispatch.** Based on `Format` enum (v0.26 phase 4 — `Frames | Gif | Mp4 | Webm | All`): write PNG frames, encode GIF in-process, shell to ffmpeg for MP4 / WebM.

5. **Quality validation.** Run at least one 16-frame test (16 fps, 512×512, deterministic seed) against a reference prompt. Compare against the block-boundary splice expectation. If motion is visibly broken (per-frame jitter, no temporal coherence), escalate to per-layer block vendoring inside phase 0 (~2 extra sessions) — do not defer.

## 6. Phase 5 — long-form details

The sliding-window stitcher is the technical risk in this cycle. Design:

```
total_frames = 64, window_size = 16, overlap = 4
↓
window_0: frames 0..16  (frames 12..16 overlap with window_1)
window_1: frames 12..28 (frames 24..28 overlap with window_2)
window_2: frames 24..40 (frames 36..40 overlap with window_3)
window_3: frames 36..52 (frames 48..52 overlap with window_4)
window_4: frames 48..64
```

Each window runs an independent N=16 AnimateDiff inference. In the overlap region, the two windows' **latents** (not RGB) blend with a linear ramp `[1.0, 0.75, 0.5, 0.25] → [0.0, 0.25, 0.5, 0.75]` per overlapping frame. Linear ramp keeps the math simple; users can extend to cosine / SmoothStep later. Latent-space blend dominates RGB-space blend because the VAE decode is non-linear — RGB blending introduces colour-banding artefacts in the overlap, latent blending stays smooth.

Per-window seed handling: each window gets `seed + window_index * window_size`. The first frame of window_{i+1} sees the same noise schedule as the (overlap-th-from-end) frame of window_i because we initialise the overlap region from the blended latents.

CLI surface:

```
plakat animate "prompt" \
  --animatediff \
  --frames 64 \
  --window-size 16 \
  --window-overlap 4 \
  --fps 16 \
  --format mp4
```

Defaults: `--window-size 16` (V3 native), `--window-overlap 4` (25% — empirical sweet spot from the AnimateDiff community).

## 7. Risk register

| Risk | Mitigation |
|---|---|
| **Block-boundary splice produces visibly broken motion** | Escalate to per-layer vendoring inside phase 0 (~2 sessions). Don't ship phase 0 with poor quality. |
| **SDXL motion adapter is incompatible with our SDXL UNet config** | `guoyww/animatediff-motion-adapter-sdxl-beta` expects a specific UNet input shape; verify with a tensor-shape dry run before vendoring sdxl_motion_unet.rs (~30min check). |
| **Sliding-window seams visible despite latent blend** | Tune overlap (4 → 6 → 8); switch from linear to cosine ramp; document hard upper bound (e.g. 128 frames quality OK, 256 frames degraded). |
| **Memory budget for SDXL+CN+animate** | Document the tier: SDXL animate at 16 frames needs ~24 GB VRAM. Document the fallback (drop frame count, drop CN, drop resolution). |
| **ffmpeg subprocess unavailability on Windows runners** | Already handled by v0.26 `imaging::video::ffmpeg_version()` check; reuse without modification. |

## 8. What's NOT in v0.27

- **Per-layer block vendoring** — deferred unless phase 0 quality forces it.
- **HotShot-XL** — different architecture; own cycle if pursued.
- **Per-frame ControlNet inputs (video-to-video)** — v0.27 ships single-control-applied-to-every-frame. Per-frame would need a control source format (video / annotation directory) + per-frame annotator pipeline.
- **`plakat.animate` Bund host word** — CLI is the v0.27 primary surface. Scripting integration can land in v0.28.
- **Audio on MP4** — video-only; users mux post-hoc with ffmpeg.
- **AnimateDiff for SD3 / Flux** — neither has an upstream-supported motion adapter equivalent today.

## 9. Acceptance criteria

v0.27 ships when:

- [ ] `plakat animate --animatediff` works end-to-end on at least one SD 1.5 model + at least one SDXL model.
- [ ] `plakat animate --animatediff --control-image PATH` produces visibly-controlled N frames on both SD 1.5 and SDXL.
- [ ] `plakat animate --animatediff --frames 64` produces a smoothly-stitched 64-frame clip with no visible window-seam artefacts.
- [ ] All four output formats (frames / GIF / MP4 / WebM) work for the above three modes.
- [ ] Tutorial + integration tests covering all of the above land in phase 7.
- [ ] No new compile warnings; all existing tests pass.
- [ ] No new Civitai / HF tokens hardcoded; offline-mode flags continue to work.

## 10. Migration / compatibility

- **No breaking CLI changes.** `plakat animate` flags from v0.20 + v0.26 all continue to work.
- **No breaking Bund changes.** No new host words this cycle.
- **No breaking config keys.** New CLI flags only.
- **Output PNG metadata.** Existing `parameters` tEXt + JSON sidecar formats unchanged; `Mode: animate` continues to mark frames as before.

## 11. Out-of-scope decisions for this RFC

The following will be resolved inside phases, not pre-locked:

- Exact CrossAttnDownBlock2D vendoring strategy IF phase 0 quality requires per-layer splice.
- Exact ramp shape for long-form blend (linear vs cosine vs SmoothStep).
- ControlNet weight schedule across frames — if `--control-strength` should ramp across frames or stay flat (flat is the v0.27 default).
- Specific SDXL CN model choice for phase 4 (canny + depth at minimum, but the v0.23 SDXL CN surface offers more).
