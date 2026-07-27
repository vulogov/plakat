# plakat 4.9.0 — roadmap: one-shot edit commands

Productize the existing segment + inpaint + matte stack into two ergonomic verbs — `plakat remove`
(erase an object, fill seamlessly) and `plakat replace-bg` (swap the background) — then add
open-vocabulary **text targeting** (`--what`) via an OWL-ViT detector port.

Ground rules: additive; each phase lands with a coherence/verify check; `Cargo.lock` in sync; no
Anthropic/Claude attribution anywhere. Frozen commands stay byte-identical.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — `plakat remove` (SAM select → inpaint fill) — DONE

Mirror the `outpaint` pattern (build a mask, hand off to `img2img --mask`). Object selection reuses the
existing SAM interface: `--point X,Y` (repeatable, `:bg` to carve), `--depth-band LO,HI`, and a new
`--box X0,Y0,X1,Y1` (rectangular mask). Grow + feather the mask, then inpaint the region.

- [x] `src/cli/remove.rs` + wired into `cli/mod.rs` (module, `Remove` variant, is_heavy, dispatch with
      import). Reuses `sam::build_selection_mask`/`depth_band_to_mask`/`intersect_masks`/`finish_mask`;
      `box_to_mask` helper (normalised-or-pixel, unit-tested); `segment::parse_points`/`parse_band` made
      `pub(crate)`. `--grow` (default 8) + `--mask-feather` (8).
- [x] Mask → tempdir PNG → `img2img::run` with `--mask`, strength 1.0, default model `sdxl-inpaint`,
      `--prompt` (empty = background continuation). Tempdir held until run returns. `--what` present but
      bails until Phase 3 (OWL-ViT).
- [x] Verify (Metal): box-removed the townsquare crate → **preserved region mean|Δ| 9.05** (SDXL-inpaint
      VAE floor), masked region regenerated (58.5). Mechanism correct. (Fill *quality* is model-dependent —
      SD-inpaint hallucinates context, the documented LaMa gap; a mis-placed SAM point over-selects, e.g.
      a ground point → 32% mask, expected SAM behaviour not a bug.)

## Phase 2 — `plakat replace-bg` (matte → new bg → composite)

- [ ] `src/cli/replace_bg.rs`: U2Net matte (`matting::cutout` → RGBA foreground), generate a new
      background from `--prompt` (txt2img at the image dims, `--model`), then alpha-composite the
      foreground over it (feather the matte edge; optional `--bg-image PATH` to composite over a supplied
      background instead of generating). Preserve the subject exactly.
- [ ] Verify: subject pixels preserved under the matte; new background follows the prompt; edges clean.

## Phase 3 — text targeting (`--what "…"`) via an OWL-ViT port

Open-vocabulary detection so `plakat remove img --what "the trash can"` (and `replace-bg --keep "the
person"`) work. OWL-ViT ≈ CLIP ViT-B/32 (image + text, which plakat already has in `vendored_clip.rs`)
+ a box-regression MLP head + a class head (patch embeds projected into the CLIP text space, dotted with
the query text embeds → per-box logits). Feed the top box → SAM → the Phase-1/2 mask.

- [ ] `src/pipelines/owlvit.rs`: port `OwlViTForObjectDetection` (reuse the CLIP ViT/text encoder;
      add the objectness + box heads + the query-text class head). Config/weights from
      `google/owlvit-base-patch32`.
- [ ] `tools/reference/owlvit_dump.py` + an env-gated corr test: box predictions / logits vs
      `transformers` `OwlViTForObjectDetection` at corr > 0.999 on a fixed image + query.
- [ ] Wire `--what` into `remove` (+ `--keep` into `replace-bg`): detect → best box → SAM refine → mask.
      RISK: if the port stalls, Phase 3 ships as a follow-up and 4.9.0 releases with Phases 1–2 (point/box
      selection) — do NOT block the release on the detector.

## Phase 4 — docs + release

- [ ] Tutorials for `remove` / `replace-bg` (+ `--what` if Phase 3 lands); README what's-new; capability
      notes. New memory reference for the edit-verbs. Cut the 4.9.0 release.

## Notes / risks

- No text→region detector exists today and candle 0.10.2 ships none — Phase 3 is a from-scratch port
  (OWL-ViT is the most portable: CLIP-based, reuses `vendored_clip`). Treat it as the at-risk phase.
- `remove` fill quality depends on the inpaint model; default `sdxl-inpaint`. LaMa-style dedicated
  erase is out of scope (SD inpaint is good enough for the one-shot verb).
- Composition in `replace-bg` is alpha over a flat generated plate — no relighting (that's `relight`).
