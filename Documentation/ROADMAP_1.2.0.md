# plakat 1.2.0 — roadmap

1.1.0 shipped "train your own words" (Textual Inversion incl. SDXL dual-encoder),
live compose (`generate:`/`matte:` layers), depth-band selection, and SD3.5
DreamBooth. 1.2.0 carries the open threads forward, all SemVer-additive on the
frozen 1.0 contracts.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Carryovers from the 1.1.0 cycle

- [ ] **Flux regional prompting** — *(M)* `--region` for Flux (today it bails).
      Flux's flow-matching transformer needs its own per-region velocity blend.
      Note: Flux is broken on Metal (candle GGUF kernel bug), so this lands
      **code-only / unverifiable** on the dev box — verify on CPU/CUDA.
- [ ] **IC-Light relighting** — *(L, stretch)* relight composited artefacts so they
      sit in the scene's light, not just on it. SD 1.5-based; porting the model is
      the work. Pairs naturally with the new `compose` `matte:`/`generate:` layers.

## Verification debt (memory-bound on 24 GB — needs more RAM or a bigger box)

- [ ] **SD3.5 DreamBooth render** — training + LoRA-merge verified; the render OOMs
      at the merge step even at 512² / `--device cpu`. Code-complete; proof awaits RAM.
- [ ] `regional.sh sdxl` / `sd35` (sd15 verified; sd35 likely OOMs 24 GB).
- [ ] `resume_train.sh` final render (resume verified; the render OOM'd).

## Ideas (unscheduled)

- Multi-vector Textual Inversion (`--vectors N`) — more capacity than the single
  vector for subjects/styles that one vector under-captures.
- `compose` `segment:` layer source (point/depth mask → cut-out inline), closing
  the segment → compose loop without a temp file.

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon
  (its own cycle); additional model families.
