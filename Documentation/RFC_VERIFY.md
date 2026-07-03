# RFC: `plakat verify` — a first-class model-correctness harness

Status: **draft** (2.0.0 candidate theme). Owner: TBD.

## Goal

Turn the ad-hoc "diffusers reference-comparison" method — the thing that has repeatedly
rescued this project from silently-wrong models (SD clip-skip, SDXL VAE-F16, MMDiT pooled
order, PixArt pos-embed, SD3.5, and the SDXL-AnimateDiff CFG scramble the 1.22.0 audit
found) — into a **durable, repeatable, committed** subcommand: `plakat verify`.

## Hard invariant: the shipped tool stays self-contained

**Once shipped, plakat depends on nothing external except models.** Therefore:

- The **`plakat verify` binary is pure Rust.** No Python, no diffusers, no torch at
  runtime or build time. It only ever touches Hugging Face (for models *and* for the
  reference artifacts) — which is the one allowed external, exactly like model weights.
- **diffusers is used only OFFLINE, once**, by a maintainer, to author the golden
  reference tensors. That authoring code lives in `tools/reference/` and is **excluded
  from the published crate** (same treatment `corpus/`, `gallery` assets already get).
  Its *output* — frozen golden tensors — is the only thing that reaches users.
- The golden artifacts are **hosted on HF** (a dataset repo, e.g.
  `vulogov/plakat-verify-references`) and fetched on demand through the existing hf-hub
  cache, exactly like a model. No new network surface, no new runtime dependency.

So the dependency graph of the *shipped* thing is unchanged: `plakat` + `hf-hub` + models.
The reference data is just more HF-hosted bytes.

## The problem this closes (that today's tools don't)

plakat already has three verification pieces — all of which miss silent-wrong-output:

| Existing | Proves | Blind to |
|---|---|---|
| `corpus/` (`*.hjson` + `*.sh`) | the pipeline *runs* end-to-end | a deterministically-wrong image still "passes"; a human must eyeball `GALLERY.md` |
| `plakat gallery` | regenerate + index a visual gallery | same — a review artifact, not a gate |
| `plakat doctor --reproducibility-check` | same seed → same bytes (**determinism**) | **correctness** — the bytes can be deterministically wrong |

The gap is **per-module numerical correctness against a reference implementation.** That is
what `plakat verify` adds, and what would have caught every silent-noise bug at module
load-and-forward time, before it ever produced a plausible-but-wrong image.

## Architecture — three planes

```
  OFFLINE (maintainer, once per model+revision)     ON HF (frozen)            SHIPPED (pure Rust)
  ┌───────────────────────────────────────┐        ┌──────────────────┐      ┌──────────────────────┐
  │ tools/reference/  (python + diffusers) │  push  │ dataset repo:    │ pull │ plakat verify        │
  │  • reproduce plakat's fixture input    │───────▶│  <alias>/<fix>/  │─────▶│  • load model (HF)   │
  │  • dump named intermediate tensors     │        │   goldens.safeten│      │  • run plakat module │
  │  • write manifest.json (+ thresholds)  │        │   manifest.json  │      │  • compare vs golden │
  └───────────────────────────────────────┘        │  corpus/<n>.png   │      │  • pass / fail       │
                                                    └──────────────────┘      └──────────────────────┘
        NOT in the crate                              allowed external            no python/diffusers
```

- **Fixtures are defined by plakat**, in Rust (`src/verify/fixtures.rs`): a deterministic
  `(prompt, negative, seed, size, steps)` plus a list of **named checkpoints** (e.g.
  `clip_l.penultimate`, `unet.mid`, `mmdit.pooled_y`, `vae.decoded`,
  `animatediff.cfg_batch_layout`). The authoring harness must reproduce the *same* input
  and dump at the *matching* points — the correspondence table (diffusers module ↔ plakat
  module) is documented per family in `tools/reference/README.md`.
- **Capture in plakat** reuses the existing `StepHook` mechanism, generalized to a
  `TensorTap` that records a named intermediate on demand (most pipelines already thread a
  hook; this extends it to arbitrary named captures behind a `--verify-capture` flag).

## Golden artifact format

Per `(model-alias, fixture)`:

- `goldens.safetensors` — the named intermediate tensors (small: encoder outputs, one
  transformer block's output, pooled vectors, the VAE-decoded latent; NOT full images).
- `manifest.json`:
  ```json
  {
    "model": "sd15", "model_revision": "…", "fixture": "portrait_v1",
    "plakat_arch": "sd_core@3",           // bump when a module's shape/semantics change
    "provenance": "diffusers==0.27.2",    // or "plakat@<sha>" for a regression baseline
    "tensors": {
      "clip_l.penultimate": { "shape": [1,77,768], "corr_min": 0.999, "max_abs": 0.02 },
      "unet.mid":           { "shape": [1,1280,8,8], "corr_min": 0.99, "max_abs": 0.05 }
    }
  }
  ```

Two provenances, both just "golden tensors on HF":
- **Correctness oracle** (`provenance: diffusers…`) — captured once offline; proves plakat
  matches the reference implementation. The load-bearing check.
- **Regression baseline** (`provenance: plakat@<sha>`) — after a model passes the oracle,
  snapshot plakat's own output at that commit as a cheap, self-contained drift guard.

## Verification tiers (cheapest / broadest first)

- **Tier 0 — structural & determinism (zero external data).** Pure self-checks: same seed
  → same bytes (fold in `doctor --reproducibility-check`), shape/dtype contracts, CFG
  batch-layout invariants (the AnimateDiff bug was a pure layout mismatch — assertable with
  synthetic tensors, no weights), map byte-stability. **Fully self-contained, CI-runnable.**
- **Tier 1 — per-module correctness (golden tensors from HF).** Load the model, feed the
  fixture, capture the named intermediates, compare against the goldens (correlation ≥
  `corr_min`, max-abs ≤ `max_abs`). This is the tier that catches the silent-noise class.
- **Tier 2 — end-to-end perceptual (golden PNGs from HF).** Regenerate a corpus image,
  compare against a committed golden PNG with a **backend-tolerant** metric (not bit-exact —
  GPU math differs across backends; use a coarse perceptual distance). Promotes
  corpus/gallery from human-eyeball to pass/fail.

## CLI surface

```
plakat verify                     # Tier 0 (no downloads) + Tier 1 for cached models
plakat verify --model sd15        # one model, all applicable tiers
plakat verify --tier 0            # structural/determinism only (CI default)
plakat verify --full              # all tiers, all models (fetches goldens + weights)
plakat verify --json              # machine-readable report (for CI gating)
plakat verify --update-baseline   # snapshot current plakat output as the regression golden
```

Exit non-zero on any failure; `--json` emits per-tensor correlations for triage.

## Phase plan

- **Phase 0 — RFC + fixtures + Tier 0.** Land `src/verify/`, the fixture format, the CLI
  skeleton, and Tier 0 (reuse `--reproducibility-check`, add shape/CFG-layout invariants +
  map determinism). Deliverable: `plakat verify` runs green with **zero downloads**. The
  AnimateDiff CFG-layout invariant ships here as a regression guard for the 1.22.0 fix.
- **Phase 1 — comparison core + capture + HF fetch.** Golden safetensors + manifest
  loader; the pure-Rust comparators (corr / cosine / max-abs); the `TensorTap` capture in
  the SD-family pipeline; HF-dataset fetch via the existing cache. Deliverable:
  `plakat verify --tier 1 --model sd15` compares against hand-placed goldens.
- **Phase 2 — authoring harness + pilot goldens on HF.** `tools/reference/` (excluded from
  the crate): per-family diffusers dump scripts + the module-correspondence docs. Generate
  and publish goldens for a pilot set — **sd15, sdxl, sd3.5-medium, pixart-Σ, cascade,
  animatediff** — to the HF dataset repo. Deliverable: `plakat verify --tier 1` passes for
  the pilot models against the **diffusers oracle**.
- **Phase 3 — Tier 2 perceptual gate.** Golden corpus PNGs on HF + a backend-tolerant
  metric. Deliverable: `plakat verify --tier 2` gates the corpus.
- **Phase 4 — CI, baselines, docs.** CI runs Tier 0 always + Tier 1 for ungated/fitting
  models (self-hosted runner for the big ones); `--update-baseline` snapshots regression
  goldens; `VERIFY.md` + `doctor` integration. Deliverable: verification is a documented,
  repeatable gate — and the deferred **T5 pad-mask fix (BUGFIX 1.2)** and the **map
  cross-machine determinism decision** become cheap to verify instead of scary.

## Self-containment guarantee (restated, explicit)

- Published crate deps: unchanged (`plakat` + `hf-hub` + existing). **No** python / torch /
  diffusers anywhere in `Cargo.toml`, build scripts, or runtime.
- `tools/reference/` (the diffusers authoring code) is `exclude`d from the crate, like
  `corpus/`. It is maintainer tooling, run offline, never by end users.
- All reference data is HF-hosted and fetched through the same client as models. A user with
  only `plakat` + network to HF can run every tier.

## Non-goals (initially)

- Not a *training* verifier (loss curves / LoRA quality) — that's a separate axis.
- Not bit-exact cross-backend equality (GPU math varies) — Tier 2 is perceptual by design.
- Not a replacement for `corpus`/`gallery` — it *promotes* them to gates.

## Risks / open questions

- **Golden staleness:** a pipeline refactor that changes a module's shape invalidates its
  goldens → gate on `plakat_arch` in the manifest and re-author when it bumps.
- **Threshold calibration:** `corr_min` / `max_abs` per tensor need empirical tuning (BF16
  vs F32 tolerances differ) — start loose, tighten as data accrues.
- **Gated / large models in CI:** Tier 1 for gated weights (sd3.5, flux) can't run on a
  public runner → those are local/self-hosted-runner gates; Tier 0 always runs.
- **Authoring reproducibility:** matching plakat's exact input in diffusers (tokenization,
  seed → latent, scheduler) is the fiddly part — document the correspondence rigorously; a
  mismatch there is itself a finding.
