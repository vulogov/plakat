# Embeddings — Textual Inversion at generation time

A **Textual Inversion (TI) embedding** is a tiny learned vector that stands in
for a bundle of concepts or quality cues behind a single trigger word. `plakat
generate --embedding` injects one at runtime — no training, no model swap.

```bash
# Baseline vs +EasyNegative (a popular SD 1.5 *negative-quality* embedding):
plakat generate "full-body photo of a woman in a sunlit garden, detailed hands" \
  --model sd15 --negative "EasyNegative" \
  --embedding "embed/EasyNegative#EasyNegative.safetensors:EasyNegative" \
  --out out/
```

EasyNegative is a *negative* embedding: you put its trigger in `--negative` so
the model steers **away** from the bad-anatomy / low-quality cluster it encodes.

## The `--embedding` spec

```
PATH_OR_REPO[#file.safetensors][:trigger][:scale]
```

- **`PATH_OR_REPO`** — a local `.safetensors` file, or an HF repo id. A bare repo
  looks for `learned_embeds.safetensors`; use `repo#file.safetensors` to name a
  different file.
- **`:trigger`** — the word that activates it in your prompt/negative. Defaults to
  the file stem.
- **`:scale`** — strength multiplier (default `1.0`).

Repeatable — pass `--embedding` more than once to stack several.

## Inspect a TI file

```bash
plakat embedding info EasyNegative.safetensors   # trigger word + dims + variant
```

## Notes

- TI embeddings are **model-family specific** — an SD 1.5 embedding won't load on
  SDXL (different CLIP dims). plakat reports a clear error on a mismatch.
- Runtime injection works on **SD 1.5 and SDXL** (the vendored-CLIP path, v0.30).
- A *positive* embedding works the same way — put its trigger in the **prompt**
  instead of the negative.
