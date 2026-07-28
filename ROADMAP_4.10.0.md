# plakat 4.10.0 — roadmap: OWL-ViT open-vocabulary text targeting

The 4.9.0 deferral: port OWL-ViT (`google/owlvit-base-patch32`) so `plakat remove --what "the trash
can"` detects the object from text. OWL-ViT ≈ CLIP ViT-B/32 vision + CLIP text + two small detection
heads — the encoders are largely reusable (candle `ClipVisionTransformer`, plakat `vendored_clip`
text), so the new code is the OWL-ViT-specific merge + heads + box decode + the `--what` wiring.

Ground rules: additive; each phase lands with a reference-corr or coherence check; `Cargo.lock` in sync;
no Anthropic/Claude attribution anywhere. Frozen commands stay byte-identical.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Architecture (from the transformers source)

- **Vision**: CLIP ViT-B/32 at **768px** (patch 32 → 24×24 = 576 patches + 1 class = 577 tokens).
  `last_hidden_state` → `post_layernorm` → **class-token merge**: `emb = patches * broadcast(class_tok)`
  → OWL-ViT `layer_norm` → image_feats `(B, 576, 768)`.
- **Box head**: `dense0→gelu→dense1→gelu→dense2(4)` on image_feats → `+ box_bias` (grid-corner logit +
  patch-size bias) → sigmoid → cxcywh in [0,1].
- **Class head**: `dense0(768→512)` → L2-normalize; `pred_logits = image_embeds · query_embedsᵀ`;
  `logit_shift = shift(image_feats)`, `logit_scale = elu(scale(image_feats))+1`; `(logits+shift)*scale`.
- **Text**: CLIP text (512d) → EOS-pooled → `text_projection` → query_embeds `(num_queries, 512)`.

## Phase 1 — port the OWL-ViT model (`src/pipelines/owlvit.rs`) — DONE

- [x] Vision backbone = candle `ClipVisionTransformer` (`vit_base_patch32`, image_size=768); take the
      full 577-token sequence from `output_hidden_states` (the entry before the pushed pooled token),
      apply our own `post_layernorm`, do the class-token merge + OWL-ViT `layer_norm`.
- [x] Text tower = candle `ClipTextTransformer` (max_position 16); `.forward(input_ids)` EOS-pools (causal
      mask makes the padding mask moot for the EOS token) → `text_projection`.
- [x] Box head (3 linears, `gelu_erf`) + `compute_box_bias` + sigmoid; class head (dense0 → L2-norm →
      cosine-sim vs queries → `(logits+shift)*(elu(scale)+1)`). All rank ≤ 3.
- [x] Loading from one `model.safetensors` via `VarBuilder::from_tensors` sub-maps; the `pre_layernorm`
      → `pre_layrnorm` key remap for candle's CLIP vision.

## Phase 2 — verify against transformers — DONE

- [x] `tools/reference/owlvit_dump.py` (fixed synthetic image + 2 queries) → pixel_values, input_ids,
      image_feats, query_embeds, pred_boxes, logits.
- [x] Env-gated corr test (`PLAKAT_OWLVIT_VERIFY`): **image_feats 1.000000, query_embeds 0.999904,
      boxes 1.000000, logits 1.000000** vs the transformers dump (CPU canonical).

## Phase 3 — wire `--what` into the edit verbs — DONE (remove)

- [x] `owlvit::detect(image, query, threshold) → Option<Detection>` (best box by score; cxcywh→pixel
      xyxy, mapped through the pad-to-square). `preprocess_image` (pad→768, CLIP mean/std) + `tokenize`
      (CLIP BPE from `openai/clip-vit-large-patch14` — OWL-ViT ships only the legacy vocab/merges).
- [x] `owlvit::OwlViT::load_pretrained` (downloads `google/owlvit-base-patch32` + the CLIP tokenizer).
      `plakat remove --what "…"`: detect → a rectangular box mask (`rect_mask`) → the Phase-1 grow/feather
      + inpaint path. Bail dropped.
- [x] Verify (Metal): `remove portrait --what "a cowboy hat"` — the hat is detected + removed; change
      concentrated in the TOP (hat) third (mean|Δ| 85 vs bottom 7). Detection targeting correct.
- [ ] Deferred: `--keep "…"` on `replace-bg` (protect a detected subject) — optional, follow-up. SAM
      box-refine (tighter mask than the rectangle) — optional, follow-up.

## Phase 4 — docs + release

- [x] EDIT_TUTORIAL `--what` section (now the 4th selection mode); README banner + "what's new in 4.10.0";
      new `reference_owlvit` memory + updated `reference_edit_verbs`.
- [x] Cut the 4.10.0 release — v4.10.0 @ 016f2e6: tag pushed → Release CI green (6 assets + SHA256SUMS),
      `cargo publish --locked` (on crates.io), main fast-forwarded, notes set via `gh release edit`.

## Notes / risks

- candle `ClipVisionTransformer` pools `[:,0]` in `forward`; use `output_hidden_states().last()` for the
  full sequence, and apply `post_layernorm` ourselves (candle's is private). Watch the CLIP `pre_layrnorm`
  key spelling.
- OWL-ViT image size is **768** (not CLIP's 224) → position embeddings are 577; the config must say so.
- SAM refine from a box: if awkward, fall back to using the detected box directly as the `remove` mask
  (coarser but functional) — don't block on perfect SAM box-prompting.
