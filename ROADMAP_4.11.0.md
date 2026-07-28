# plakat 4.11.0 — roadmap: finish the edit verbs

Two follow-ups deferred from the 4.9/4.10 edit-verbs work: **SAM box-refine** (tight object masks for
`remove --what`, instead of the raw detection rectangle) and **`replace-bg --keep "<subject>"`** (protect
an OWL-ViT-detected subject instead of the U2Net matte). Both compose OWL-ViT + SAM + the existing paths.

Ground rules: additive; each phase lands with a coherence/verify check; `Cargo.lock` in sync; no
Anthropic/Claude attribution anywhere. Frozen commands stay byte-identical.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — SAM box-refine for `remove --what` — DONE

- [x] Shared `remove::detect_object_mask(image, query, device, w, h, refine)`: OWL-ViT detect → SAM
      prompted with a **foreground point at the box center + background points just outside the four box
      edges** (MobileSAM over-selects from a lone point; the bg hints bound it) → `∩ rect_mask(box)` →
      fall back to the rectangle only if SAM collapses (< 0.5% of the image).
- [x] `--box-only` escape hatch on `remove` (skip SAM, use the raw rectangle).
- [x] Verify (Metal): `remove townsquare --what "a person"` cleanly removes a detected figure (left
      figure + rest preserved); refined mask slightly tighter than box-only (the inpaint VAE drift
      dominates the output metric, so tightness shows best in `--keep`).

## Phase 2 — `replace-bg --keep "<subject>"` — DONE

- [x] `replace_bg.rs` `--keep "<text>"`: uses `detect_object_mask` as the composite alpha instead of
      the U2Net salient matte; bg + composite path unchanged.
- [x] Verify (Metal): `replace-bg portrait --keep "a man" --prompt "beach"` — clean subject cutout (no
      background halo → confirms the SAM refine tightens the mask), background swapped.

## Phase 3 — docs + release

- [ ] EDIT_TUTORIAL: note the SAM-refined `--what` mask + `replace-bg --keep`; README what's-new; update
      [[reference_edit_verbs]] + [[reference_owlvit]]. Cut the 4.11.0 release.

## Notes / risks

- SAM (MobileSAM) takes only point prompts in plakat; the center-point + box-∩ is the pragmatic refine
  (a true SAM box-prompt path is a bigger change, not worth it here).
- If the box center lands on a hole/background inside the box, SAM may select wrong → the box-∩ + the
  <2% fallback keep it safe (worst case = today's rectangle).
- `--keep` shares the OWL-ViT load + detect + SAM-refine with `remove --what`; factor a small shared
  helper (`owlvit_refined_mask(image, query, device)`) so both call it.
