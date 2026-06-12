# plakat — 1.0.0 feature cycle

The 1.0-**readiness** roadmap ([`ROADMAP_1.0.md`](ROADMAP_1.0.md)) is closed —
0.47.0 proved + froze the surface. **1.0.0 is a feature cycle.** Three tracks,
chosen to (a) clear the explicit deferrals and (b) deepen two areas where plakat
already owns the hard parts: **scene editing** and **training**.

Status legend: `[ ]` open · `[x]` done · `[~]` in progress · `[?]` decision needed.
Each item notes **what exists**, **candle-feasibility**, and **effort** (S/M/L).

---

## Part 0 — Close the deferrals (warm-up; quick wins)

- [x] **Cascade Stage-B + ControlNet — FALSIFIED (no work needed).** Investigation
      found this is a *non-feature*: Stable Cascade's decoupled design applies
      ControlNet (and LoRA) to **Stage C alone** — "the stages B and A models do not
      need to be updated … we only need to finetune the Stage C model to achieve a
      custom style or ControlNet" (Stability-AI, *Introducing Stable Cascade*). Stage B
      is a fixed semantic-compressor/super-resolver that preserves Stage C's structure
      through the decode, so a Stage-B CN is redundant — and no upstream Stage-B CN
      weights exist to align against (the diffusers-reference comparison every plakat
      model relies on would be impossible). Cascade + CN already works end-to-end
      (`cascade.hjson` canny, rendered). Action taken: reframed the `forward_with_cn`
      Stage-C-only guard from "follow-up" to a documented design invariant; added the
      Stage-C-only note to the Cascade tutorial. Dropped like CannyFilter-224 / INT8.
- [x] **SDXL tiled scripting — DONE.** Added `plakat.tiled.enable` / `disable`
      words (`scripting/words/tiled.rs`) + dispatched the SD-family `plakat.generate`
      to the pipeline's `generate_tiled` when set (bails on tiled+ControlNet).
      `tile_size`/`tile_stride` were already config keys. Frozen in STABILITY.md;
      driver `corpus/tiled_script.{bund,sh}` (1536² SDXL, pending render).

---

## Part 1 — "Compose & edit scenes" (Theme A) 🎯 the distinctive bet

The matte → integral-artefacts → InstantStyle trio is already a *scene composition*
tool. This turns it into an **editing** capability. The enabler is precise
selection; everything else composes pieces plakat owns.

- [ ] **Selection: SAM + depth** — *(M, enabler)* port candle's **Segment-Anything**
      (`mobile-sam`/SAM ViT) for click/box → mask. `depth.rs` already exists, so
      "select by depth band" is nearly free. Output: a reusable mask any edit op
      consumes. **Everything below depends on this.**
- [ ] **Object removal / replacement** — *(M)* SAM/matte mask → inpaint
      (`img2img --mask`, which exists). "remove the person" (inpaint background),
      "replace the car" (masked img2img with a prompt). New surface: a `plakat edit`
      verb wrapping select → inpaint.
- [ ] **Regional prompting** — *(L)* different prompts in different canvas regions
      (reuse the artefact **zones**). Needs masked/region-routed cross-attention in
      the SD-core UNet — the hardest candle piece here; prototype on SD 1.5/SDXL.
- [ ] **Layered scenes** — *(M)* a scenario construct that stacks
      `generate → matte → composite → blend` as named **layers** with z-order +
      per-layer ops. Builds directly on artefacts + scenarios; mostly schema + a
      compositor loop. Freeze the new scenario keys.
- [ ] **Relighting composited artefacts** (IC-Light) — *(L, stretch)* the honest weak
      spot of artefact compositing — a relight pass would make composites *real*, not
      just grounded. IC-Light is an SD 1.5-based relighting model; porting the model +
      the lighting condition is the work. Defer if Part 1 runs long.

**Proof:** a `corpus/edit.sh` driver (select → remove → replace → relight) — the
verification corpus is the thesis.

---

## Part 2 — "Train your own everything" (Theme C; can run parallel to Part 1)

plakat trains **style** LoRAs (`sd_train/trainer.rs`, SD 1.5/SDXL/SD 3.5). Same
machinery, deeper:

- [ ] **DreamBooth / subject LoRAs** — *(M)* learn a *subject* ("my dog `sks`"), not
      a style. Same trainer + objective; add **class prior-preservation** (a few
      class images to stop overfitting/language-drift) and the subject data path.
      `plakat subject train` alongside `style train`.
- [ ] **Textual-Inversion training** — *(M)* *create* an embedding (you already load +
      inject TIs via the vendored CLIP, `embedding.rs`). A small loop optimizing the
      trigger vector against a few images — cheaper than a LoRA, composable with one.
      `plakat embedding train`.
- [ ] **Resumable training + validation previews** — *(S/M)* numbered checkpoints
      already exist; add **resume-from-checkpoint** and a **validation render** every
      N steps (a fixed prompt+seed) so you can watch the style/subject converge and
      stop at the sweet spot (the watercolour LoRA needed exactly this).
- [ ] **Trainer ergonomics** — *(S)* better schedulers, LR warmup/decay, and an
      auto-stop heuristic (the imprint-phase signal we added to the progress line).

**Proof:** extend `style_train.sh` into a `train.sh` covering style + subject + TI,
each → a committed render.

---

## Sequencing

1. **Part 0** (deferrals) — small, ships confidence early.
2. **Part 1 §SAM** (the enabler) — unblocks all of scene editing.
3. **Part 2** in parallel (independent of Part 1; different files).
4. Then Part 1's edit ops → layers → (relight stretch).

Critical dependency: **SAM gates scene editing.** Training has no external deps —
it can start immediately.

## Explicitly out of scope for 1.0.0
- Theme B (depth/pose ControlNet, InstantID) — the annotators exist, but parked
  this cycle.
- `plakat serve` (HTTP daemon) — high value, but a standalone cycle.
- Flux-on-Metal (blocked upstream in candle), additional model families.
