# `plakat verify` — operator guide

The practical how-to for the model-correctness harness. For the design rationale and the
self-containment invariant, see [`RFC_VERIFY.md`](RFC_VERIFY.md).

`plakat verify` turns the ad-hoc "diffusers reference-comparison" method — the thing that
has repeatedly caught silently-wrong models — into a committed, repeatable subcommand. The
shipped binary is **pure Rust** (no python/torch/diffusers); it fetches golden reference
tensors from a Hugging Face **dataset** exactly like it fetches model weights.

It already earned its keep: it found the **SDXL CLIP-L pad-token bug** (`clip.encoded`
corr 0.991 — CLIP-L was padded with `"!"`/id 0 instead of EOS). See
[`tools/reference/correspondence.md`](../tools/reference/correspondence.md).

## Running

```bash
plakat verify                         # all applicable tiers, pilot model set
plakat verify --tier 0                # structural/determinism only (no downloads, ~0.2s)
plakat verify --tier 1 --model sdxl   # per-module correctness for one model (fetches goldens)
plakat verify --tier 2 --model sd15   # end-to-end perceptual regression
plakat verify --json                  # machine-readable report (CI gating)
plakat verify --golden-dir ./out      # use locally-authored goldens instead of the HF dataset
plakat verify --device cpu            # force CPU (goldens are authored on CPU/F32)
```

Exit code is non-zero if any check fails — safe to gate CI on.

### The three tiers

| tier | what | needs |
|---|---|---|
| **0** structural | CFG batch layout, map-render determinism — invariants that need no data | nothing (fast, always-on CI gate) |
| **1** per-module | load each model, capture named intermediates, compare vs golden tensors (corr + max-abs) | model weights + goldens |
| **2** end-to-end | render a fixture via the REAL generate path, compare to a frozen golden PNG (SSIM + mean-abs) | model weights + golden PNG |

Tier 1/2 fetch goldens from the HF dataset **`vulogov98/plakat-verify`** (override with
`PLAKAT_VERIFY_DATASET`). Tier 2 forces determinism with `PLAKAT_VERIFY_DET_INIT` (candle's
CPU RNG isn't seed-reproducible) + a non-ancestral scheduler.

### Coverage (all vs diffusers 0.38, all hosted)

| family | conditioning | denoiser / core | tier 2 |
|---|---|---|---|
| SD 1.5 | `clip.encoded`, `vae.decoded` | `unet.out` (1.0) | SSIM 1.0 |
| SD 2.1 | `clip.encoded`, `vae.decoded` | — | — |
| SDXL | `clip.encoded`, `clip_g.pooled` | `unet.out`, `unet.mid` (1.0) | — |
| PixArt-Σ | `dit.pos_embed` | `dit.block0` (1.0) | — |
| Stable Cascade | `clip_g.pooled` | `stage_c.block0` (1.0¹) | — |
| SD 3.5-medium | `pooled_y` | `mmdit.block0` (1.0) | — |
| AnimateDiff | — | `motion.block0` (1.0) | — |

¹ Cascade's `stage_c.block0` taps the conditioned-conv core (embedding + first Res + Time)
**before** the first Attn block — self-attention over the 576 white-noise verify tokens is
OOD-ill-conditioned (a *deep* full-forward gave only a coarse 0.989). Res + Time are
well-conditioned → corr 1.0. Attention coverage stays with the v0.41 reference suite + corpus.

## Authoring + hosting goldens (maintainer, offline)

Goldens live **only** in the offline authoring harness `tools/reference/` (excluded from the
crate). Authoring needs diffusers; verifying never does.

```bash
# 1. Author one model's goldens (runs the diffusers reference, writes safetensors + manifest)
python tools/reference/dump.py --model sdxl --device cpu --out ./out
#    → ./out/sdxl/portrait_v1/{goldens.safetensors, manifest.json}

# 2. Validate plakat against the local goldens (chase any finding, or confirm the match)
plakat verify --tier 1 --model sdxl --golden-dir ./out

# 3. Host: upload the whole tree to the HF dataset (maintainer's write token)
hf upload vulogov98/plakat-verify ./out . --repo-type=dataset
```

The dataset layout is the contract shared with `dump.py`: `<model>/<fixture>/{manifest.json,
goldens.safetensors}` (+ `golden.png` for Tier 2). `manifest.json` records each tensor's
shape + `corr_min`/`max_abs` thresholds + `provenance` (the diffusers version).

Localization aid: set `PLAKAT_VERIFY_DUMP_DIR=<dir>` to dump plakat's captured tensors, then
diff element-wise against the golden (how the pad-token bug was pinned).

## Adding a capture point

The rule ([`correspondence.md`](../tools/reference/correspondence.md) is the contract): a
capture-point **name** must denote the *same* intermediate on both sides.

1. **plakat side** — add the name to a pipeline's `capture_intermediates` (or the AnimateDiff
   branch in `verify::tier1::run_model`). It must be **additive**: reuse the real forward
   internals (`encode_prompt`, `unet.forward`, `capture_mid`, `capture_block0`, …) so the
   captured tensor is exactly what generation produces. No hot-path change.
2. **golden side** — add it to `tools/reference/models/<model>.py`, tapping the corresponding
   diffusers module (a forward hook, or the model's own function). Set a threshold in that
   file's `DEFAULT_THRESHOLDS`.
3. **isolate confounds** — feed deterministic inputs where a real encoder would conflate two
   things: `verify::deterministic_latent` (init latent) and `verify::deterministic_tensor`
   (caption/context/pooled) are seeded LCGs mirrored byte-for-byte by
   `fixtures.deterministic_tensor` in Python. Structured conditioning matters for deep
   full-forward taps (white noise through a deep net is OOD — see Cascade).
4. **document** the correspondence row and **re-author + re-host** the golden.

### Calibrating thresholds

`corr` carries correctness; `max_abs` needs headroom for candle-vs-torch fp accumulation.
Pooled vectors (large CLIP-G sink magnitudes) use `max_abs` 0.15. Pre-final-LN penultimate
states have ~100–850-magnitude attention sinks → looser abs bounds. corr ≥ 0.9999 is the norm
for well-conditioned taps.

## CI

- **Every push/PR** runs Tier 0 (`.github/workflows/ci.yml`, the `test` job) — zero downloads,
  ~0.2s, a reliable structural gate.
- **On demand** (Actions → "Run workflow" → CI): the `verify-models` job builds release +
  runs Tier 1/2 for a chosen `model` input (default `sd15`), fetching weights + goldens from
  HF. Opt-in because it downloads multi-GB weights and runs CPU inference.
