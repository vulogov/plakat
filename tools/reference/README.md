# `tools/reference/` — golden-tensor authoring harness for `plakat verify`

Offline, maintainer-only tooling that produces the **golden reference tensors** Tier 1 of
`plakat verify` compares against. See [`Documentation/RFC_VERIFY.md`](../../Documentation/RFC_VERIFY.md).

## The self-containment contract (read this first)

**plakat never runs this.** This directory is `exclude`d from the published crate (see
`Cargo.toml`), is not a build or runtime dependency, and is not shipped to users. It is a
maintainer's *authoring* step: run it once (per model + revision), with diffusers and the
real weights, to emit golden artifacts. Those artifacts get **frozen on Hugging Face**; the
pure-Rust `plakat verify` then fetches them like model weights. The dependency graph of the
shipped tool is unchanged — `plakat` + `hf-hub` + models. diffusers lives here and only here.

```
tools/reference/  (python + diffusers, THIS dir, offline)
        │  run once per (model, revision)
        ▼
   goldens.safetensors + manifest.json
        │  push
        ▼
   HF dataset repo  (frozen, the allowed external)
        │  fetch (Phase 1b/2)
        ▼
   plakat verify --tier 1   (pure Rust — no python, no diffusers)
```

## Layout

- `fixtures.py` — the canonical, deterministic inputs (`prompt`, `negative`, `seed`, `size`,
  `steps`). **These MUST match plakat's fixtures** (Rust side), or the goldens won't
  correspond to what plakat captures. Keep the two in sync; the fixture id is the contract.
- `dump.py` — the driver: `python dump.py --model sd15 --fixture portrait_v1 --out out/`.
- `models/` — one dumper per family. `sd15.py` is the worked reference example; the others
  follow the same pattern (`# TODO` stubs to fill when authoring that family).
- `correspondence.md` — **the crux**: for every capture-point name, which diffusers module
  it comes from and which plakat module it must match (clip-skip semantics, pooled order,
  CFG batch layout, …). A wrong mapping here silently authors a wrong golden.

## Output format (must match plakat's Rust side)

Per `(model, fixture)`, `dump.py` writes:

- `goldens.safetensors` — the named intermediate tensors (F32), keyed by the capture-point
  names plakat's `TensorTap` emits (`clip_l.penultimate`, `unet.mid`, `vae.decoded`, …).
- `manifest.json` — the schema `src/verify/manifest.rs` parses:
  ```json
  {
    "model": "sd15", "model_revision": "<sha>", "fixture": "portrait_v1",
    "plakat_arch": "sd_core@1", "provenance": "diffusers==0.27.2",
    "tensors": {
      "clip_l.penultimate": { "shape": [1,77,768], "corr_min": 0.999, "max_abs": 0.03 }
    }
  }
  ```

## Setup + run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tools/reference/requirements.txt
# authors goldens under tools/reference/out/sd15/portrait_v1/
python tools/reference/dump.py --model sd15 --fixture portrait_v1 --out tools/reference/out
```

Then verify locally against the fresh goldens (Phase 1b, once the pipeline capture points
are wired):

```bash
plakat verify --tier 1 --model sd15 --golden-dir tools/reference/out
```

and, when it passes, publish the artifacts to the HF dataset repo.

## Authoring discipline

- **Reproduce plakat's exact input.** Same seed → same latent, same tokenizer, same
  scheduler, same size. A mismatch there is itself a finding — chase it before trusting the
  golden.
- **Correctness oracle vs regression baseline.** `provenance: diffusers==X` is the oracle
  (proves plakat matches the reference). `provenance: plakat@<sha>` is a cheap drift
  baseline authored by running plakat itself — set it only after the oracle passes.
- **Bump `plakat_arch`** when a module's shape/semantics change so stale goldens are
  rejected, not silently mis-compared.
