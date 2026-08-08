# `plakat --etch` / `doctor --if-plakat` — provenance etching & graded attribution

`--etch` writes a **64-bit provenance id** (an `EtchId`) into images plakat produces, by four
independent evidence layers of decreasing fragility and decreasing precision. `plakat doctor
--if-plakat <IMAGE>` reads back whatever survived and reports a **graded verdict with a p-value —
not a boolean**. It is the plakat 6.7 provenance feature (RFC [`RFC_ETCH_1.md`](RFC_ETCH_1.md)), and
it is **fully additive**: `--etch` is **off by default**, so generation stays byte-identical for
everyone who does not opt in, and the flag is **silently ignored by every command that does not write
an image** (`models`, `doctor`, `verify`, `bench`, `inspect`) — so `alias plakat='plakat --etch'`
is safe.

## Read this first — the honesty premise

**No invisible watermark survives a determined removal or a high-strength diffusion regeneration.**
This is not an engineering gap that a later release closes; it is a property of the problem. A
watermark is by construction a low-amplitude, off-manifold perturbation, and a denoiser's *job* is to
project an image back onto the natural-image manifold — so it removes the mark as a side effect of
working correctly. Increasing the img2img noise strength monotonically destroys the payload.

So ETCH does **not** promise bit recovery through heavy edits. It promises **graded attribution** that
degrades gracefully instead of falling off a cliff:

```
exact id  →  generated  →  probable-derivative  →  no-evidence
```

The defensible claim, and the only one this document makes:

> **Verifiable through incidental editing, format churn, rescaling, and moderate generative edits;
> unenforceable against a determined remover.**

Two corollaries you must carry through every use of the feature:

- **`no-evidence` is NOT proof of non-plakat origin.** It means nothing above threshold survived —
  the mark may never have been written, or it may have been stripped, regenerated, or removed. Absence
  of evidence is not evidence of absence, and the verdict vocabulary is worded to keep that honest.
- **This is not C2PA.** There is no certificate chain and no signing authority (see *Non-goals*).

## The four evidence layers

The layers are independent: each writes on its own, each reads on its own, and the detector fuses
whatever it finds. Losing any three still yields a usable verdict. They trade robustness against
precision in opposite directions — L0 is exact but the most fragile; L3 is coarse but survives the
most.

| Layer | Carrier | Payload | Survives | Status in 6.7.0 |
|---|---|---|---|---|
| **L0** manifest | PNG `tEXt` chunk + JSON sidecar | full recipe + `EtchId` + `parent` | until metadata is stripped | **fully working** |
| **L1** pixel etch | spread-spectrum QIM on mid-band DCT of a 512² luma grid | 64-bit id + CRC-16, repetition-ECC | transcode / rescale / alpha; *partial* through a light img2img | **fully working** |
| **L2** latent etch | Tree-Ring Fourier mark in the initial latent `z_T` | presence + short id prefix | generative edits (in principle) | **write-only** — reading is a follow-up |
| **L3** fingerprint | CLIP image embedding in a local store | lookup key (not bits) | img2img, format churn, rescale | **fully working** |

### L0 — manifest (fully working)

An `etch` object written into a PNG `tEXt` chunk (beside plakat's existing `parameters` chunk) **and**
into the `<image>.png.json` sidecar. It carries the full recipe, the `EtchId`, the list of layers
written, and a `parent` field:

```json
{
  "etch": {
    "v": 1,
    "id": "9f2c4a17b3e08d5c",
    "tool": "plakat",
    "tool_version": "6.7.0",
    "layers": ["L0", "L1", "L2", "L3"],
    "parent": null
  }
}
```

`parent` carries the `EtchId` of the source image whenever **plakat itself** performed the derivation
(`img2img`, `outpaint`, `relight`, `remove`, `replace-bg`, `restore-faces`, `upscale`). This makes
plakat-internal edit chains fully traceable — the one case where we own both ends and should not be
guessing.

L0 is **free and exact when it survives**, and **dies to any metadata strip** (a screenshot, a
re-export that drops text chunks, most social-media uploads). It is included because it costs nothing
to try and it is unambiguous when present.

### L1 — pixel etch (fully working)

A spread-spectrum **QIM** (quantization index modulation) mark on a **mid-band DCT coefficient** of a
canonical **512×512 luma grid**, tiled **4×4** across the frame, protected by **repetition-ECC +
CRC-16**, with the coefficient positions **key-permuted** so the carrier is not a fixed pattern that
averages out across a corpus. **Fully-transparent pixels (`alpha == 0`) are excluded** from both
embedding and correlation.

- **Near-invisible.** At the default `--etch-strength 0.35` the embed targets **≳42 dB PSNR** against
  the un-etched render. Above `0.6` the mark starts to show in flat gradients — and poster art
  (plakat's core case) is the worst case, since it has large flat regions. The default is deliberately
  conservative; raise it only if you value robustness over fidelity.
- **Robust to the everyday.** Because it works on a *canonical resampled grid*, rescaling (Lanczos,
  browser resize, AI upscale) is inverted by the decoder's own normalization; transcoding and alpha
  changes are tolerated; the 4×4 tiling gives a quorum from any surviving region ≳25% of the frame.
- **Two honest limitations.** It is **DCT-domain**, so it needs the image to be roughly **≳512 px** to
  carry the grid. And **true crop survival needs an alignment search** that this release does not do —
  a re-cropped image can shift the grid out of registration. Both are documented follow-ups (see
  *Non-goals & deferrals*).
- Through a **light** img2img L1 typically decodes **partially** (a subset of tiles / bits); through a
  **high-strength** img2img it is gone.

### L2 — latent etch (write-only in 6.7.0)

A **Tree-Ring** Fourier mark written into concentric rings of the initial latent `z_T` **before the
first denoising step**, for **SD 1.5 / SDXL** generations (the 4-channel latent geometry — SD3 / Flux
/ Cascade differ and are skipped). Because the sampler *amplifies* `z_T` into the image's global
structure, the mark becomes part of the generated content rather than a residual on top, which is why
semantic latent marks degrade more gracefully under regeneration than pixel marks.

**Be clear about the status: in 6.7.0 the mark is WRITTEN but not READ.** Recovering it needs **DDIM
inversion** back to `z_T` (which means loading a UNet) plus a ring-correlation test — a model-inversion
pipeline that is the documented follow-up. So `doctor` currently reports L2 as `skipped`, and it does
so **whether or not** you pass `--verify` (with `--verify` it notes that inversion detection is the
pending work; without it, that a model would be needed). Treat L2 today as *provenance you are laying
down for a future verifier*, not evidence you can read back now.

### L3 — fingerprint (fully working)

A **CLIP image embedding** stored as `embedding → EtchId` in a **local, append-only store**.
Verification is a **nearest-cosine query**, not an extraction — you match on the one thing a
high-strength img2img cannot destroy: the semantics. This is the layer that covers img2img (a *soft
binding*, in C2PA's sense: the perceptual hash is the lookup key, not the proof).

- **Store location.** `$PLAKAT_HOME/etchdb` (falling back to `~/.plakat/etchdb`). It is a **plain
  local directory, never a network service.** `--etch-db none` disables L3; the other three layers
  keep working.
- **Offline and download-free at verify time.** In `doctor`, L3 runs **only when the store exists AND
  the CLIP encoder is already cached.** If either is missing it reports **`unavailable`** and never
  downloads anything — this respects the doctor charter (health-check without loading or fetching).
- **Write is deferred and batched.** Images are queued at save time and fingerprinted in one batch at
  the end of the run, so CLIP loads at most once; if the encoder isn't cached, the images simply aren't
  fingerprinted (L0/L1 are still written) and a notice explains why.
- **Thresholds are placeholders.** Cosine **≥ 0.92 → strong match**; **0.85–0.92 → probable
  derivative**; **< 0.85 → no L3 evidence.** These cut-points are **not yet calibrated** (see
  *Non-goals & deferrals*).

## The verdicts

`doctor --if-plakat` fuses the surviving layers into one of five graded verdicts. It is never a
boolean.

| Verdict | Meaning | Roughly, what triggers it |
|---|---|---|
| `generated` | plakat produced this image | L0 present-and-consistent, **or** a confident L1 decode (p < 1e-6) |
| `derived` | plakat produced an ancestor | partial L1 **+** an L3 semantic match (the shape of a light generative edit) |
| `probable-derivative` | semantically matches a known plakat output | an L3 match only, no surviving bits |
| `inconclusive` | weak / conflicting evidence — do not rely on this either way | a weak partial L1 with nothing to corroborate it |
| `no-evidence` | nothing above threshold (**absence of evidence, not evidence of absence**) | none of the above |

### Sample output

```
$ plakat doctor --if-plakat suspect.png

  Etch verification — suspect.png (1024x1024, PNG)

  L0  manifest     absent     stripped or never written
  L1  pixel etch   partial    12/16 tiles, p = 3.1e-05
  L2  latent etch  skipped    --verify not given (L2 read needs a model)
  L3  fingerprint  match      cosine 0.943 → 9f2c4a17b3e08d5c

  Verdict: derived  (plakat produced an ancestor)
  EtchId:  9f2c4a17b3e08d5c
  Note:    partial pixel etch + a semantic match — consistent with a light generative edit
```

The per-layer lines each carry a **state** (`present` / `absent` / `partial` / `match` /
`weak-match` / `no-match` / `skipped` / `unavailable`) and a detail; the **Note** interprets the
*shape* of the evidence — a partial L1 plus a strong L3 looks like a low-strength img2img, and
distinguishing that from, say, JPEG damage is information worth surfacing.

- **`--json`** emits the same report machine-readably (`file`, `width`/`height`/`format`, `verdict`,
  `meaning`, `id`, `parent`, per-layer `state`/`detail`, `note`) for CI and scripting.
- **`--verify`** is the escape hatch that would load a model — it is L2's mechanism. In 6.7.0 it
  changes L2's note but not the outcome, since inversion detection is still pending.

## The `EtchId`

The `EtchId` is a 64-bit value rendered as 16 lowercase hex nibbles. It is derived **reproducibly**
from the generation recipe and a key:

```
EtchId = SHA-256( key ‖ "plakat-etch-v1" ‖ canonical_manifest )[0..8]
```

where `canonical_manifest` is the deterministic serialization of the recipe (prompt, negative, seed,
model, sampler, steps, guidance, size, LoRA stack). Consequences:

- **Reproducible** — an identical recipe under an identical key yields an identical id (consistent with
  plakat's determinism contract).
- **Opaque** — the id reveals nothing without the manifest, so it is safe to leave in a published
  image.
- **Overridable** — `--etch-id <HEX16>` substitutes your own 64-bit value (e.g. to namespace a batch).

### Keying — public (default) vs private

- **Public key (default).** A published constant (`plakat-etch-public-v1`). Anyone with a stock
  plakat build can verify a plakat image — the ecosystem-interop mode, and the right default for an
  open-source tool. **Be honest about the trade: with the public key the carrier is public, and a
  public carrier can be subtracted.** Public-key mode is a *provenance signal against incidental
  editing*, not a defence against a motivated remover.
- **Private key.** `--etch-key <KEY>` (or `PLAKAT_ETCH_KEY`). Only holders of the key can derive the
  id or read the L1/L3 carrier. This raises the bar but does not clear it — regeneration still removes
  a mark it cannot see.

## CLI reference

### Global `--etch*` flags (verified against `plakat --help`, heading "Provenance (etch)")

| Flag | Default | Purpose |
|---|---|---|
| `--etch` | off | Write provenance etching into images plakat produces. Ignored by non-image commands. `[env: PLAKAT_ETCH]` |
| `--etch-key <KEY>` | public constant | Key for `EtchId` derivation and carrier PRNG (private mode). `[env: PLAKAT_ETCH_KEY]` |
| `--etch-id <HEX16>` | derived | Override the derived `EtchId` with an explicit 64-bit hex value. |
| `--etch-layers <LIST>` | all applicable | Comma-list of layers to write: `l0,l1,l2,l3`. |
| `--etch-strength <F32>` | `0.35` | L1 embedding strength, `0.0..=1.0`. |
| `--etch-db <PATH\|none>` | `$PLAKAT_HOME/etchdb` | L3 fingerprint store; `none` disables L3. |

### `doctor --if-plakat` flags (verified against `plakat doctor --help`)

| Flag | Purpose |
|---|---|
| `--if-plakat <PATH>` | Verify whether IMAGE originated from plakat — read the surviving layers into a graded verdict. Offline by default. Mutually exclusive with `--benchmark`. |
| `--verify` | Additionally run L2 (loads a model). The escape hatch for the model-loading path. |
| `--json` | Emit a structured JSON report instead of the human blocks. |

### Examples

```
# Etch every image this run produces (public key, all applicable layers)
plakat --etch generate "a minimalist concert poster" -o poster.png

# Private-namespace etch, L1 turned up, no fingerprint store
plakat --etch --etch-key "$MY_KEY" --etch-strength 0.5 --etch-db none \
       generate "a red propaganda poster" -o red.png

# Only the free manifest layer
plakat --etch --etch-layers l0 generate "a study" -o study.png

# Verify an image — offline, human-readable
plakat doctor --if-plakat poster.png

# Machine-readable report for CI
plakat doctor --if-plakat poster.png --json
```

## Non-goals & deferrals

Stated plainly so nothing is discovered later.

**What ETCH is not:**

1. **Not removal-proof.** Invisible pixel watermarks are provably removable by generative
   regeneration. Anyone who wants the mark gone can remove it.
2. **Not bit-recovery through high-strength img2img.** Above roughly 0.6 denoise strength expect L1 to
   be gone; L3 (semantic) is what remains, and it is a *lookup*, not an extraction.
3. **Not C2PA.** There is no certificate chain and no signing authority. A C2PA manifest alongside L0
   is a reasonable future RFC — it is not this feature.
4. **`no-evidence` ≠ proof of non-plakat origin.** Restated because it is the easiest claim to walk
   back by accident.

**Deferred in the 6.7.0 release (documented follow-ups, not silent gaps):**

- **L2 detection.** The Fourier-ring mark is written into `z_T`, but reading it back needs DDIM
  inversion (a UNet load). `doctor` reports L2 `skipped` until that pipeline ships.
- **L1 transform.** L1 is **DCT-domain** in this release (the RFC's DWT–DCT pairing is not fully in
  place); it needs the image ≳512 px.
- **L1 crop alignment-search.** True survival through a crop needs a registration/alignment search that
  is not yet implemented — a re-cropped image can shift the canonical grid out of registration.
- **L3 threshold calibration.** The 0.92 / 0.85 cosine cut-points are **placeholders pending an ROC
  study** over plakat / plakat-derived / unrelated corpora. A false `probable-derivative` on an
  unrelated image is the expensive error, so treat L3-only verdicts with corresponding caution.
