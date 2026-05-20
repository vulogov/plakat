# Apple hardware requirements

plakat runs natively on Apple Silicon Macs via candle's Metal
backend, and on Intel Macs as a CPU-only build. This document
covers what to expect on each tier — what works, what's painful,
and what won't run at all.

## TL;DR

- **Apple Silicon (M1/M2/M3/M4) with ≥ 16 GB unified memory, macOS
  13 or later.** Everything works. SD 1.5 portraits in seconds,
  SDXL in tens of seconds. This is the target platform.
- **Apple Silicon with 8 GB.** SD 1.5 works at 512² with a few
  caveats; SDXL is unreliable. Workable for hobbyists.
- **Intel Macs.** CPU-only. Each SD 1.5 image takes 5–15 minutes.
  Functional, not enjoyable.

## What plakat does on Apple

plakat compiles with one of three backends:

| Build feature | Backend | Targets |
|---|---|---|
| `--features metal` | Metal | Apple Silicon GPU |
| (default) | CPU | Intel + Apple Silicon CPU |
| `--features cuda` | CUDA | Not applicable on Mac |

On Apple Silicon, install with:

```bash
cargo install plakat --features metal
```

The Metal backend uses unified memory: the GPU shares the
machine's RAM, so "VRAM" and "system RAM" are the same pool. A
model that fits in N gigabytes of GPU memory needs roughly N
gigabytes of total free RAM.

## Minimum: SD 1.5 at small resolution

| Component | Requirement |
|---|---|
| Chip | Apple M1 (any variant) or newer |
| Memory | 8 GB unified |
| macOS | 13 Ventura |
| Free disk | ~10 GB for the model cache |
| Network | Required for the first run (HuggingFace download) |

**What runs:** SD 1.5 text-to-image at 512×512, 20–30 steps.
Portraits with IP-Adapter-Plus-Face. The bundled style catalog.
Real-ESRGAN upscaling at 2× or 4×.

**Caveats on 8 GB:**

- Don't run other GPU-heavy apps in parallel — Safari with many
  tabs can be enough to push SD 1.5 into swap.
- SDXL (~7 GB weights + intermediate tensors) is on the edge; it
  may run but is prone to OOM. Stick to SD 1.5.
- FaceID identity strategy adds another ~500 MB on top of base
  SD 1.5 — possible but tight.
- `--smart-zones` adds Depth-Anything-V2 (~99 MB) once loaded.
  Fine even on 8 GB; the model is small.

**Expected speed (M1, 8 GB):**

| Task | Time per image |
|---|---|
| SD 1.5 @ 512², 28 steps | 8–12 s |
| SD 1.5 portrait + IP-Adapter | 12–18 s |
| Real-ESRGAN 4× on a 768×768 input | 6–10 s |

## Recommended: SD 1.5 and SDXL, comfortable

| Component | Requirement |
|---|---|
| Chip | Apple M2 / M3 / M4 (Pro or higher) |
| Memory | 16 GB unified, ideally 24 GB+ |
| macOS | 14 Sonoma or 15 Sequoia |
| Free disk | ~30 GB for multiple model caches |

**What runs comfortably:**

- SD 1.5 at any reasonable resolution (512–1024).
- SDXL at 1024×1024.
- SDXL portraits (Plus-Face-SDXL or FaceID-SDXL).
- Multi-photo weighted portrait merges (`--photo x.jpg:0.6
  --photo y.jpg:0.4`).
- `--smart-zones` + `--artefact-blend` stacked.
- Scenarios with many tasks running sequentially.

**Expected speed (M2 Pro, 16 GB):**

| Task | Time per image |
|---|---|
| SD 1.5 @ 768², 28 steps | 4–6 s |
| SDXL @ 1024², 28 steps | 18–25 s |
| SDXL portrait + FaceID + ArcFace | 25–35 s |
| `--artefact-blend` adds | +2–4 s |
| `--smart-zones` adds | +0.5–1.5 s (depth model) |

## Ideal: Flux + SDXL + heavy scenarios

| Component | Requirement |
|---|---|
| Chip | Apple M2 Max / M3 Max / M4 Max / Ultra |
| Memory | 32–64 GB unified |
| macOS | 14 Sonoma or later |
| Free disk | ~60 GB for the full model cache |

**What runs:**

- Flux (`--model flux-schnell` and `--model flux-dev`, ~30 GB on
  disk, ~17 GB resident). Flux requires the most memory of any
  model plakat supports.
- SDXL with a refiner pass.
- Stacked LoRAs (5–10 simultaneous).
- Long scenarios (hundreds of tasks) without restarting between
  runs — model caches stay warm.

**Expected speed (M3 Max, 64 GB):**

| Task | Time per image |
|---|---|
| Flux-schnell @ 1024², 4 steps | 6–10 s |
| Flux-dev @ 1024², 28 steps | 35–55 s |
| SDXL + refiner | 40–60 s |

## Intel Macs (CPU only)

plakat builds and runs on Intel Macs but **no GPU acceleration is
available** — candle's Metal backend targets Apple Silicon Metal 3,
which Intel Macs don't expose at the feature level plakat needs.

| Component | Requirement |
|---|---|
| CPU | 4-core Intel or better |
| Memory | 16 GB |
| macOS | 12 Monterey or later |

Build / install without the metal feature:

```bash
cargo install plakat
```

**Expected speed (Intel Core i7, 16 GB, CPU only):**

| Task | Time per image |
|---|---|
| SD 1.5 @ 512², 28 steps | 5–10 min |
| SDXL | 20–40 min (not recommended) |
| Real-ESRGAN 4× | 1–3 min |

SD 1.5 at small resolution is the only practical workflow. SDXL
and Flux are technically possible but each image takes long enough
that you should consider a remote GPU instead.

## Model download sizes

The first run for a given model pulls weights from HuggingFace
into `~/.cache/huggingface/`. Cached after; subsequent runs are
disk-only.

| Model family | Download | Notes |
|---|---|---|
| SD 1.5 | ~4 GB | f16 UNet + VAE + CLIP-L |
| SD 2.1 | ~5 GB | Slightly larger UNet |
| SDXL | ~7 GB | Includes both CLIP-L and CLIP-G |
| Flux-schnell | ~30 GB | Transformer + T5-XXL + dual CLIP. ~17 GB resident after load. |
| Flux-dev | ~30 GB | Same architecture, different fine-tune. ~17 GB resident. |
| IP-Adapter-Plus-Face (SD 1.5) | ~2.5 GB | CLIP-H image encoder shared across IP-Adapter paths; ~50 MB adapter on top. |
| FaceID / ArcFace (SD 1.5) | ~250 MB | Converted ArcFace IR-ResNet50 safetensors. SCRFD detector adds ~10 MB. |
| Plus-Face-SDXL | ~2.5 GB | Same CLIP-H, different ~50 MB adapter for SDXL. |
| Depth-Anything-V2-small | ~99 MB | For `--smart-zones` (v3) |
| Real-ESRGAN x4 | ~64 MB | For ML upscaling |
| Bundled style catalog | ~5 MB | Catalog file only — LoRAs separate |

For a "comfortable" install that covers most workflows: budget
**~15 GB** of disk for the model cache.

## macOS version requirements

| macOS version | Status |
|---|---|
| 13 Ventura | Minimum for Metal backend |
| 14 Sonoma | Recommended — Metal 3 improvements |
| 15 Sequoia | Recommended |
| 12 Monterey | CPU only (Metal backend not supported by candle 0.8) |
| 11 Big Sur and earlier | Not supported |

candle 0.8 depends on Metal 3 features that landed in macOS 13.
plakat will detect missing GPU support and fall back to CPU
automatically (with a warning), so an old macOS doesn't break the
install — it just kills performance.

## Choosing `--device`

By default, plakat auto-detects the best available device:

```bash
plakat generate "..."                  # auto: tries Metal, then CPU
plakat --device metal generate "..."   # force Metal
plakat --device cpu generate "..."     # force CPU (testing / fallback)
```

Force CPU when you're profiling a model that misbehaves on Metal
(rare — usually a scheduler issue, not a model issue), or when
running on a machine with limited GPU memory and a large CPU
RAM pool. See the scheduler compatibility table in
[`GENERATE.md`](GENERATE.md#schedulers) for cases where some
schedulers are CPU-only.

## Diagnosing memory pressure

macOS reports unified memory pressure under
**Activity Monitor → Memory**. plakat works comfortably when the
pressure indicator stays green during a generation. Yellow
("compressed memory growing") is OK but slows things down; red
("swapping to disk") means you need a smaller model, lower
resolution, fewer LoRAs, or more RAM.

Common triggers for red pressure on 8 GB:

- SDXL on 8 GB — expected.
- SD 1.5 + many simultaneous LoRAs.
- A scenario where every task loads a different model (avoid
  switching `model:` per task; keep tasks on one model).
- Other GPU-heavy apps running in parallel (browsers, video
  editors, design tools).

## Apple-specific caveats

- **No CUDA, ever.** Don't pass `--features cuda` on Mac — it
  won't compile.
- **`unipc` scheduler is CUDA / CPU only.** Metal builds reject
  it; use `dpm++` or `euler-a` instead. See
  [`GENERATE.md` § Schedulers](GENERATE.md#schedulers).
- **Metal RNG masks seeds to `u32`.** If you pin a seed > 2³²
  expecting bit-exact reproducibility, you'll get the masked
  variant on Metal but the full `u64` on CUDA / CPU. Practical
  impact: zero, unless you're cross-comparing pixel-exact
  outputs between platforms.
- **First-run download.** The HuggingFace hub library caches per
  model; the first generation with each new model triggers a
  download. Subsequent runs are offline-safe.
- **Cold-start cost.** Loading the SD 1.5 UNet from disk to GPU
  takes 3–8 seconds on M-series chips. Scenarios amortize this
  across all tasks; one-off `plakat generate` calls pay it every
  time.

## Build prerequisites

To build from source on an Apple Silicon Mac:

```bash
# Xcode command-line tools (for the linker and SDK).
xcode-select --install

# Rust 1.85 or newer.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Build with Metal.
cd plakat
cargo build --release --features metal
```

The binary lands at `target/release/plakat`.

## See also

- [`GENERATE.md`](GENERATE.md) — full `plakat generate` reference
  (schedulers, devices, refiner).
- [`PERSONA.md`](PERSONA.md) — portrait generation; FaceID needs
  extra weights.
- [`STYLES.md`](STYLES.md) — style catalog system.
- [`ARTEFACTS.md`](ARTEFACTS.md) — `--smart-zones` and the depth
  model cost.
- Top-level [README](../README.md) for install instructions.
