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

## Train your own word (`embedding train`)

The inverse of loading a TI: learn a **new token embedding** from a few images
with the whole model **frozen** — only the placeholder vector(s) are optimized.
Point it at a folder of images that share a look (or a subject), give the new
token a name, and seed it from a coarse class word:

```bash
plakat embedding train --base sdxl --from-dir ./stained-glass \
  --token sgwin --init-word glass --steps 1000 --out glass.safetensors

plakat generate "a sgwin cat" --model sdxl --embedding glass.safetensors:sgwin:0.6
```

- **Bases:** `sd15` / `sd21` learn one CLIP-L vector; **`sdxl`** learns a CLIP-L +
  CLIP-G pair (a dual-encoder TI, saved as one file with both halves).
- **`--init-word`** — a simple class word to start from (e.g. `glass`, `toy`,
  `art`); TI converges far faster from a sensible seed than from noise.
- **It's subject-dependent** — one vector lands harder on some subjects than
  others. Lower the load-time scale (`…:trigger:0.6`) if a strong token overpowers
  the composition; raise `--steps` if it's too faint.
- **SDXL tuning:** use `--lr 5e-4` (the sd15 default `5e-3` over-cooks the
  dual-encoder TI to a blur), and render around `…:0.6` scale.

Proof: `corpus/embedding_train.sh <sd15|sd21|sdxl>` learns a stained-glass token
and applies it to new subjects (`corpus/images/embedding-train/<base>/`).

## Notes

- TI embeddings are **model-family specific** — an SD 1.5 embedding won't load on
  SDXL (different CLIP dims). plakat reports a clear error on a mismatch.
- Runtime injection works on **SD 1.5 and SDXL** (the vendored-CLIP path, v0.30).
- A *positive* embedding works the same way — put its trigger in the **prompt**
  instead of the negative.
