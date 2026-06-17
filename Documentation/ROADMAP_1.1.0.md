# plakat 1.1.0 — roadmap

1.0.0 shipped "compose & edit scenes" + "train your own everything" + the Metal
OOM guard. 1.1.0 finishes the threads left open, all SemVer-additive on the frozen
1.0 contracts.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Carryovers from the 1.0.0 cycle

- [ ] **Textual-Inversion training** — *(M)* `plakat embedding train`: optimize a
      trigger token vector against a few images (we already load/inject TIs via the
      vendored CLIP). Cheaper than a LoRA, composes with one. The last Part 2 item.
- [ ] **Compose `generate:` / inline `matte:` layers** — *(M)* render a layer
      inline, or U2Net-matte a layer on the fly, inside `plakat compose` (today's
      layers are `load`-only; pre-render / pre-matte for now). GPU.
- [ ] **Flux regional prompting** — *(M)* `--region` for Flux (today it bails).
      Flux's flow-matching transformer needs its own per-region velocity blend.
- [ ] **sd35 DreamBooth** — *(M)* prior preservation for the SD3.5 MMDiT trainer
      (the `--class-dir` path is sd15/sdxl only; sd35 bails).
- [ ] **IC-Light relighting** — *(L, stretch)* relight composited artefacts so they
      sit in the scene's light, not just on it. SD 1.5-based; porting the model is
      the work.
- [ ] **SAM depth-band selection** — *(S)* `depth.rs` exists; "select by depth band"
      is a nearly-free extra mask source for `segment`.

## Verification debt (GPU)

- [ ] `regional.sh sdxl` / `sd35` (sd15 verified; sd35 likely OOMs 24 GB).
- [ ] `resume_train.sh` final render (resume verified; the render OOM'd — re-run with
      freed RAM to commit the proof image).

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon
  (its own cycle); additional model families.
