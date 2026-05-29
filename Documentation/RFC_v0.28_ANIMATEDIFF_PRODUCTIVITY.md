# RFC v0.28 — AnimateDiff productivity

**Status:** decisions locked 2026-05-28 — ready for phase 0.

**Predecessors:**
- [`RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md`](RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md) — feature completeness across SD 1.5 + SDXL × inference / CN / long-form.
- [`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md) — infrastructure cycle.

## 1. TL;DR

v0.27 made AnimateDiff **feature-complete** in scope. v0.28 makes
it **pleasant to use in practice** — closing the loudest v0.27
deferrals where the underlying machinery already exists, plus a
single transformative UX improvement (4-step generation via
AnimateLCM).

Four phases, all low-risk:

1. **Multi-CN through `--animatediff`** — wire `controls.first()` →
   full slice via the existing `sum_controlnet_residuals` helper.
2. **AnimateLCM support** — 4-step animate via the
   `wangfuyun/AnimateLCM` motion adapter + LCM scheduler. ~5×
   speedup.
3. **`plakat.animate` Bund host word** — last major CLI surface
   not yet in scripting. Enables scenarios to drive animate.
4. **`plakat motion-adapter info / list`** — inspection commands
   parallel to `plakat civitai info`.

Plus tutorials/tests + close-out: **~6 phases, ~7 sessions total.**
Sized smaller than v0.27 by design — a tight polish cycle, not a
new architectural push.

## 2. Why this is the v0.28 cycle

1. **v0.27's deferral list called out these by name.** Multi-CN
   and `plakat.animate` are both listed in the v0.27 "Deferred to
   v0.28+" README section. Closing them while the v0.27 work is
   freshest in everyone's head is the lowest-risk timing.

2. **AnimateLCM is the single biggest UX win available.** Animate
   runs at 20 steps × 16 frames currently take ~4 minutes on a
   24 GB GPU. With LCM at 4 steps that's ~50 seconds. Same
   quality bar (LCM is what every distillation paper since 2024
   targets). The community model is stable, the scheduler is
   already shipped (`SchedulerKind::Lcm`), and the underlying
   pipeline path doesn't change.

3. **The bigger themes can wait.** Per-frame video CN
   (`--control-video PATH`), per-layer motion splice (RFC §3.2
   escalation), FreeNoise/FreeInit, HotShot-XL — all real
   carries. None blocked by v0.27 or v0.28; all can land in v0.29+
   when there's appetite for a bigger release. v0.28 is the
   "harvest the easy wins" cycle.

## 3. Decisions locked (2)

The user answered via AskUserQuestion on 2026-05-28:

### 3.1 Cycle shape — **Lean productivity (items 1–4 only)**

The recommendation. Tight, ships fast, closes the loudest v0.27
deferrals. Leaves quality + new-architecture themes (per-layer
splice, FreeNoise, video-CN, HotShot-XL) for v0.29 when there's
explicit appetite for a bigger swing.

### 3.2 Must-have items — **All four (1, 2, 3, 4)**

Multi-CN, AnimateLCM, `plakat.animate` Bund word, motion-adapter
inspection. The cycle is sized so all four land; phase 3
(inspection) is the only one with "OK to drop if cycle wants to
stay smaller" framing, but it's the smallest phase and lands.

## 4. Phase plan

| # | Phase | Sessions |
|---|---|---|
| 0 | Multi-CN through `--animatediff` (both variants) | 1 |
| 1 | AnimateLCM (4-step animate) | 2 |
| 2 | `plakat.animate` Bund host word | 2 |
| 3 | `plakat motion-adapter info` / `list` | 1 |
| 4 | Tutorials + integration tests | 0.5 |
| 5 | Cycle close-out | 0.5 |

Total: ~7 sessions.

## 5. Phase 0 — multi-CN through `--animatediff`

Today `AnimateDiffPipeline::generate` and
`AnimateDiffSdxlPipeline::generate` look at `controls.first()` and
silently warn if there are more conditioners. v0.10 already ships
`pipelines::controlnet::sum_controlnet_residuals` for the multi-CN
sum on the t2i path; this phase wires it through animate.

Changes:

- Remove the "ignoring N extra conditioners" warn.
- Pre-tile each conditioner's `(1, 3, H, W)` to the per-step batch
  outside the loop (cached per-conditioner).
- Per step: for each conditioner, run `cr.net.forward(...)` →
  residuals. Sum across conditioners per residual slot.
- Pass summed `(down, mid)` to `forward_with_motion` as today.

CLI: no new flags. `--control` becomes repeatable (the existing
flag attr already supports multi-occurrence; just need to plumb a
`Vec<String>` instead of `Option<String>`).

Verified path: the helper's contract is "all CNs target the same
UNet variant". The variant gate from v0.27 phase 3 ensures this
already.

## 6. Phase 1 — AnimateLCM

Two pieces:

1. **`MotionAdapter::load_animatelcm()`** — downloads
   `wangfuyun/AnimateLCM` (the SD 1.5 variant first; SDXL variant
   is `wangfuyun/AnimateLCM-SDXL` and lands in a follow-up if
   user-validated). Same `MotionAdapterConfig` schema as V3 (the
   tensor key naming uses the same upstream convention that v0.27
   phase 2 fixed for V3).

2. **`--lcm` CLI flag** on `plakat animate --animatediff`. When
   set:
   - Picks AnimateLCM instead of V3 / SDXL beta.
   - Switches scheduler to `SchedulerKind::Lcm` (existing).
   - Defaults `--steps` to 4 (LCM training distribution).
   - Defaults `--guidance` to 1.5 (LCM-trained models are
     guidance-distilled; 1.5 is the diffusers recommended value).
   - All defaults are overrideable.

Quality validation: LCM has its own quality knobs (4 steps is
the sweet spot; 6-8 steps work too). User-machine validation as
ever; the wiring is straightforward.

## 7. Phase 2 — `plakat.animate` Bund host word

Today: 49 host words. Every major CLI verb except `animate` has a
Bund equivalent. Adding `plakat.animate` closes the symmetry and
lets scenarios drive batch animate runs.

Stack effect (proposal):
```
( prompt out_pattern -- )
```

Where:
- `prompt` is the AnimateDiff prompt string.
- `out_pattern` is a `format!`-style path with `{:04}` for the
  frame index. Example: `"./out/anim-{:04}.png"`.

The pipeline reads frame count + size + steps + guidance + scheduler
+ controls from `ctx.config` (the existing scripting context). Same
pattern as `plakat.generate`.

Returns nothing (frames written directly to disk). Single-frame
designs don't work for animate — the caller cares about the file
sequence, not handles in the registry. Future expansion could add
`plakat.animate.handles` if a user surfaces a need.

Composes naturally:
```bund
"sd15" plakat.load
20 "steps" plakat.config.set
"watercolor" plakat.look.apply              // v0.25 preset
"depth" "/path/depth.png" plakat.controlnet.add
"a fox in a meadow" "fox-{:04}.png" plakat.animate
```

Scope cap: single-prompt AnimateDiff only (the lerp-mode animate is
v0.20 territory and isn't exposed via Bund either; could be a
follow-up `plakat.animate.lerp` if needed).

## 8. Phase 3 — `plakat motion-adapter info / list`

Two subcommands under a new `plakat motion-adapter` namespace,
parallel to `plakat civitai info`:

**`plakat motion-adapter info REPO`** — downloads the adapter's
`config.json` + safetensors header, then prints:
- Class name + diffusers version
- `block_out_channels`, `motion_layers_per_block`,
  `motion_max_seq_length`, `motion_num_attention_heads`
- Total motion modules (computed via `cfg.total_motion_modules()`)
- Tensor count + per-block tensor breakdown (from
  `MotionAdapter::summary()`)
- Detected base family (SD 1.5 vs SDXL vs unknown) via heuristic
  on `block_out_channels.len()`.

**`plakat motion-adapter list`** — prints known plakat-supported
adapter repos:
- `guoyww/animatediff-motion-adapter-v1-5-3` (SD 1.5 V3)
- `guoyww/animatediff-motion-adapter-sdxl-beta` (SDXL beta)
- `wangfuyun/AnimateLCM` (LCM, v0.28 phase 1)
- Plus a section listing community repos that should work via the
  same loader but aren't officially tested (V1 SD 1.5, V2 SD 1.5,
  community SDXL fine-tunes).

Useful for users debugging "why isn't this third-party adapter
loading" or "what does V3 actually contain". Mirrors `plakat
civitai info` for LoRA inspection.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| **AnimateLCM tensor naming differs from V3/SDXL beta** | The v0.27 phase 2 fix established the canonical schema (`motion_modules.{j}.norm`, `transformer_blocks.0.attn{1,2}`, etc.). AnimateLCM follows the same upstream diffusers convention; verified via WebFetch of its config.json before phase 1 starts. |
| **Multi-CN OOMs at production resolution** | Same memory math as single CN, multiplied. Document a clear OOM-fallback list in the tutorial (drop frames → drop res → drop one CN). |
| **`plakat.animate` Bund integration competes with scenarios** | Scenarios already support animate via HJSON. Bund is for tighter programmatic control + the existing Bund-driven workflows where the user wants animate alongside `plakat.generate` calls. Both surfaces can coexist. |
| **AnimateLCM at 4 steps produces low-quality output on some prompts** | Document that `--steps 6` or `--steps 8` work too; let the user dial. The 4-step default matches diffusers' recommended use. |

## 10. What's NOT in v0.28

The bigger themes called out in the v0.27 deferral list, deferred
again to v0.29+ unless the user signals appetite:

- **Per-frame video ControlNet (`--control-video PATH`)** — read a
  video, per-frame annotate, per-frame CN residuals. The biggest
  unaddressed v0.27 deferral by far. ~4 sessions.
- **Per-layer motion splice** (RFC §3.2 escalation) — vendor
  CrossAttnDownBlock2D/UpBlock2D, apply motion per (resnet+attn)
  layer. ~3-4 sessions.
- **FreeNoise / FreeInit long-form** — shared noise across
  overlapping frames during denoising. Better than v0.27's
  post-hoc latent blend. ~3 sessions.
- **HotShot-XL** — different architecture, ~14 GB weights. ~5-6
  sessions.
- **SDXL AnimateLCM** — `wangfuyun/AnimateLCM-SDXL`. Will land if
  v0.28 phase 1's SD 1.5 path is user-validated and the SDXL
  variant has matching upstream stability.

## 11. Acceptance criteria

v0.28 ships when:

- [ ] `plakat animate --animatediff --control depth --control-image A --control canny --control-image B` runs both CNs and the result shows visible compositing of both signals (smoke test, manual quality check on a real GPU).
- [ ] `plakat animate --animatediff --lcm --from "..." --frames 16` produces output in ~5× less wall-clock time than the same command without `--lcm`.
- [ ] `plakat run script.bund` containing `plakat.animate` writes the expected per-frame PNGs.
- [ ] `plakat motion-adapter info guoyww/animatediff-motion-adapter-v1-5-3` prints the V3 config + tensor summary without network errors after a cold cache.
- [ ] 948 → ~960 lib tests; new CLI smoke + integration tests for each phase.
- [ ] No new compile warnings.

## 12. Out-of-scope decisions for this RFC

Resolved inside phases, not pre-locked:

- Exact path-pattern syntax for `plakat.animate` (`{:04}` vs
  `%04d` vs explicit `--start --pad` — pick during phase 2 based
  on what feels natural in Bund context).
- AnimateLCM SDXL inclusion in this cycle vs v0.29 (resolved at
  phase 1 closeout based on stability).
- Whether `plakat motion-adapter list` is hardcoded or pulled from
  a config file (likely hardcoded for v0.28; config-driven if
  community contributions warrant it).
