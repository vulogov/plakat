# The road to 3.0 — brainstorm (2.5 → 3.0)

**North star (3.x flagship): a TUI photo/image collection manager.** Not just *generating* images
— **organizing, searching, curating, and re-working a growing collection** of them, from inside
the terminal. Everything in 2.5–2.9 should either (a) make the images worth managing better
(quality), (b) make browsing/generating in the manager fast (performance), (c) make a long-running,
file-heavy tool trustworthy (stability), or (d) *be* a building block of the manager itself.

This is a brainstorm — nothing here is committed. Scope each cycle with the user.

---

## What the 3.0 flagship actually needs (so we build toward it)

A collection manager is more than a grid of thumbnails. It needs:

- **A catalog** — a crash-safe index of every image (generated + imported) with its recipe,
  prompt, model, seed, LoRAs, tags, rating, timestamp, dimensions. *plakat already embeds recipes
  in PNGs and has History semantic search (UI cycle) — the seed exists.*
- **Search** — by metadata *and* by content (CLIP-embedding similarity; plakat has CLIP). Dedup /
  near-dup detection.
- **Thumbnails + preview** — fast decode + a thumbnail cache; inline preview (ratatui-image exists).
- **Act on an item** — re-generate from its embedded recipe, make variations, upscale, edit,
  export, delete — the whole engine, one keystroke from any image.
- **Curate** — rate, favorite, cull, batch-op; ideally *auto-rank* by an aesthetic score.
- **Ingest** — import external images, extract metadata (incl. A1111/ComfyUI), CLIP-embed for search.

⇒ The building blocks the pre-3.0 cycles must deliver: **a catalog/index (Track C + feature),
CLIP-embedding search, thumbnailing/caching (Track B), aesthetic scoring (Track A), import/interop
(feature), and a rock-solid long-running TUI (Track C).**

---

## Track A — Image-quality generation features

Ranked by ROI (grounded in the capability inventory — many are cheap, no new weights):

1. **PAG (Perturbed-Attention Guidance)** — now *tractable* because 2.4 put attention on SDPA;
   PAG perturbs attention to sharpen structure/coherence. Big quality lift, no weights. + **SAG**.
2. **FreeU** — backbone/skip-connection rescaling. Free quality, a few lines.
3. **CFG-rescale + dynamic thresholding** — tame high-guidance burn / oversaturation.
4. **ControlNet-Tile + diffusion tiled-upscale (SUPIR-lite)** — the missing control kind; coherent
   512→2K/4K with *hallucinated* detail. The feature people reach for most.
5. **Face/detail restoration** — GFPGAN / CodeFormer pass; better ADetailer.
6. **Better/more upscalers** — 4x-UltraSharp & friends, SUPIR-style.
7. **New efficient models** — **Sana** (linear-attention DiT: fast, high-res, small — fits the 24 GB
   Metal box); **Flux fully working** *if* candle 0.11 fixes GGUF-on-Metal (see Track B).
8. **Regional prompting** — multi-subject control (also closes the last Bund gap).
9. **Aesthetic scoring / auto-curation** — a CLIP aesthetic predictor that ranks generations.
   *Directly feeds the 3.0 manager's curation.*
10. **Prompt travel / smoother interpolation** — better `animate`.

## Track B — Performance (analysis #2, building on 2.4)

2.4 found: disk-bound load (~130–250 MB/s), step-caching + SDPA are the clean compute wins, and
several items are candle-Metal-kernel-blocked. Next:

1. **candle 0.11 spike** *(asked)* — 0.11 is out (candle is pre-1.0, so not "stable"). The prize:
   *if* it fixes the Metal quantized-matmul kernel, it **unblocks GGUF Flux on Metal + int8 T5** —
   exactly what 2.4 couldn't reach. Bump on a branch, fix breaks, `plakat verify` all 7 families
   (any regression = corr < 1.0), test the 3 blockers. Decide from data.
2. **SD UNet SDPA** — the workhorses (SD1.5/2.1/SDXL). Parked in 2.4 because the SD generation UNet
   uses candle-registry attention → needs vendoring candle's SD attention (SDXL gets full head-dim-64
   coverage). ~1.2–1.5×/step like the DiTs.
3. **Finish the SDPA rollout** — the masked cross-attention paths (validate the SDPA additive-mask).
4. **`plakat serve` daemon** *(user's own RFC pending)* — amortize the 60–175 s cold load (the #1
   cost) by keeping the model resident. *Also the manager's generation backend.*
5. **VAE decode** — the 17.8 s SDXL tail is a slow candle/Metal kernel; needs kernel-level work
   (tiling lost — measured).
6. **Batch/scenario perf** — cache the T5 encode across a batch; model-resident multi-gen.
7. **Browse/thumbnail perf** *(feeds 3.0)* — fast image decode + a thumbnail cache for scrolling
   hundreds of images.

## Track C — Stability analysis

The session repeatedly hit the OOM guard (sd35 T5-XXL under memory pressure) — a real theme. A
collection manager is long-running and file-heavy, so stability is load-bearing for 3.0.

1. **Proactive memory management** — today's OOM guard is *reactive* (aborts mid-gen). Add a
   **preflight model-fit predictor** (will this model+size fit? suggest alternatives), **model
   eviction under pressure**, and a **memory budget planner**. Streaming/sharded weight load.
2. **Verify-harness extensions** — the parked items: `sd21 unet.out`, more fixtures, a **weight-backed
   verify CI gate**, and a **perf-CI gate** (`plakat bench` thresholds so speed regressions fail a PR).
3. **Graceful degradation** — on OOM, save the partial + fall back (CPU / smaller size) instead of
   aborting; clearer, actionable errors.
4. **Determinism** — cross-device reproducibility (candle RNG diverges Metal↔CPU; det-init exists —
   extend + document).
5. **Long-running robustness** *(for serve + the TUI manager)* — no leaks, model-cache eviction,
   file-handle hygiene, **crash-safe catalog writes** (extend the atomic-save pattern to the index/DB).
6. **Coverage** — property tests + fuzzing for scripting/compile and the metadata/index code.
7. **Dependency policy** — a candle pinning + upgrade strategy (informed by the 0.11 spike).

## Other features to ship before 3.* (beyond the three tracks)

Many of these are *directly* the collection manager's plumbing:

- **Prompt/recipe library** — save, name, reuse, share prompts + full recipes. (Manager building block.)
- **Import / interop** — read **A1111 / ComfyUI** PNG metadata; ingest external images with CLIP
  embedding for search. (Manager ingest.)
- **Generation queue** — a background queue: enqueue prompts → results flow into the collection.
- **Fuller provenance / metadata** — richer recipe capture; optional **C2PA-style provenance** /
  invisible watermark (trust + the manager's authenticity view).
- **Named presets / config profiles** — recipes as first-class named objects.
- **Export polish** — contact sheets, richer galleries (builds on `gallery`), video export.
- **Wildcards / scenario authoring** — richer templating for batch generation.
- **Model & LoRA management UX** — polish `models` / the LoRA hub for the manager's asset side.

---

## Suggested release arc (themed — numbers illustrative, not committed)

| cycle | theme | headline items |
|---|---|---|
| **2.5** | *finish 2.4's threads + cheap quality* | candle-0.11 spike · SD UNet SDPA · PAG / FreeU / CFG-rescale |
| **2.6** | *high-res & control quality* | ControlNet-Tile + tiled-upscale · face restoration · aesthetic scoring |
| **2.7** | *new models + serve backend* | Sana (+ Flux if 0.11 unblocks it) · `plakat serve` daemon · proactive memory mgmt |
| **2.8** | *collection foundations* | crash-safe catalog/index · CLIP-embedding search · thumbnail cache · A1111/ComfyUI import |
| **2.9** | *manager alpha + hardening* | TUI collection-manager preview (browse/search/curate/re-gen) · generation queue · long-running stability |
| **3.0** | **flagship** | **TUI photo/image collection manager** on the accumulated quality/perf/stability |

The through-line: each cycle drops something the manager needs, so 3.0 is an *assembly* of proven
parts — not a from-scratch leap. And plakat's verify harness keeps every step honest along the way.
