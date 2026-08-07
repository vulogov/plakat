# RFC ETCH-1: `--etch` / `plakat doctor --if-plakat`
## Provenance Etching and Derivative Attribution

**Status:** Proposed (6.x line) · **Author:** Vladimir Ulogov ·
**Surface:** global `--etch` option + `plakat doctor --if-plakat <IMAGE>` ·
**Feature:** `--features etch` (default on) · **Architecture:** four independent evidence
layers (L0–L3), fused into a single graded verdict.

---

## Overview

`--etch` writes a **64-bit `EtchId`** into every image plakat produces, by four independent
mechanisms of decreasing fragility and decreasing precision. `plakat doctor --if-plakat`
reads whatever survived and reports a graded verdict with a p-value — not a boolean.

The design premise is stated up front because it constrains everything below: **no invisible
watermark survives an adversarial or high-strength diffusion regeneration.** This is not an
engineering gap. A watermark is by construction a low-amplitude off-manifold perturbation;
a denoiser's job is to project onto the natural-image manifold, so it removes the mark as a
side effect of working correctly. The published result is that increasing diffusion noising
strength monotonically reduces the payload's mutual information, and pixel-domain marks are
provably removable by regeneration.

ETCH-1 therefore does **not** promise bit recovery through img2img. It promises *graded
attribution*: the deeper an editor goes, the coarser the surviving claim, degrading
`exact id → plakat-generated → probable derivative → no evidence` rather than falling off a
cliff.

---

## Evidence layers

| Layer | Carrier | Payload | Survives transcode / rescale / alpha | img2img ≤0.3 | img2img ≥0.6 | Cost |
|---|---|---|---|---|---|---|
| **L0** Manifest | PNG `tEXt` + JSON sidecar | full recipe + `EtchId` | until stripped | no | no | ~0 |
| **L1** Pixel etch | DWT–DCT spread spectrum in luma | 64 bit + CRC, ECC-coded | yes | partial | no | ~50 ms CPU |
| **L2** Latent etch | Fourier ring in the initial latent | ~16 bit + presence | yes | yes | partial | 0 (init only) |
| **L3** Fingerprint | CLIP embedding in a local store | lookup key, not bits | yes | yes | yes | one embed |

The layers are independent. Each writes on its own, each reads on its own, and the detector
fuses whatever it finds. Losing any three still yields a usable verdict.

### L0 — Manifest

An `etch` tEXt chunk beside the existing `parameters` chunk in `src/imaging/metadata.rs`,
plus the same structure in the `<base>.json` sidecar:

```json
{
  "etch": {
    "v": 1,
    "id": "9f2c4a17b3e08d5c",
    "tool": "plakat",
    "tool_version": "6.1.0",
    "layers": ["L0", "L1", "L2", "L3"],
    "parent": null
  }
}
```

`parent` carries the `EtchId` of the source image when plakat itself performed the
derivation (`img2img`, `outpaint`, `stylize`, `upscale`, `relight`, `remove`, `replace_bg`,
`restore_faces`). This makes plakat-internal edit chains fully traceable — the case where
we control both ends and should not be guessing.

L0 is free and dies to any metadata strip. It is included because when it survives it is
*exact*, and because it costs nothing to try.

### L1 — Pixel etch

Spread-spectrum embedding in the DWT–DCT domain of the luma plane. Specifics that matter:

**Canonical grid.** Before embedding or extraction, resample luma to a fixed 512×512
working grid. Rescaling (Lanczos, ESRGAN, browser resize, AI upscale) is then inverted by
the decoder's own normalization rather than needing scale-invariant carriers.

**Tiling for crop survival.** The ECC-coded block is tiled across a 4×4 grid of the
canonical image. Any surviving region ≳ 25% of the frame yields a quorum. Tiles are
independently decodable and voted.

**Alpha as validity mask.** Fully-transparent pixels carry no etch and are excluded from
both embedding and correlation. This is a hard requirement, not a nicety — see *Ordering
against `transparent`* below.

**Coefficient selection.** Mid-band DCT coefficients of the LL/LH subbands, chosen by a
key-derived permutation so the carrier is not a fixed pattern that averages out across a
corpus.

**Strength.** `--etch-strength` in `0.0..=1.0`, default `0.35`, targeting ≥ 42 dB PSNR
against the un-etched render. Above `0.6` the mark starts to be visible in flat gradients —
poster art (plakat's core case) has large flat regions and is the worst case for this. The
default is deliberately conservative; users who value robustness over fidelity can raise it.

**Payload framing.** 64-bit `EtchId` ‖ 16-bit CRC → ECC-expanded to 256 bits per tile.

### L2 — Latent etch

This is the layer plakat can offer and a post-hoc watermarking tool cannot: **plakat owns
the sampler**, so the mark can be placed in the initial noise rather than painted onto
pixels afterwards.

A key-derived pattern is written into concentric Fourier rings of `z_T` before the first
denoising step, in the manner of Tree-Ring / ZoDiac. Because the sampling trajectory
*amplifies* the initial latent into the image's global structure, the mark becomes part of
the generated content rather than a residual layered on top — which is exactly why semantic
watermarks degrade more gracefully under regeneration than pixel-domain ones.

Detection requires DDIM inversion back to `z_T`, then a ring-correlation test. Note the two
real limits:

- **Model coupling.** Inversion needs a compatible model. If a *different* pipeline did the
  img2img, L2 detection weakens sharply. It is strong for "someone ran plakat's own img2img
  over this," moderate for "someone ran SD 1.5 img2img," weak across families.
- **Capacity.** Ring encoding realistically carries ~16 bits plus a strong presence signal.
  It cannot carry the full 64. It carries a short `EtchId` prefix, and the presence bit.

Implementation is small: `src/pipelines/fft.rs` already supplies a candle 2D DFT (added for
FreeU's Fourier skip-filter, DFT-as-matmul, no power-of-two constraint). The latent-init
sites are already centralized enough to intercept — the same seam AnimateDiff's
`Option<Tensor>` noise override uses.

**L2 is generation-time only.** It cannot be applied to an image plakat did not generate.

### L3 — Fingerprint

The layer that actually answers the original question, by giving up on bit recovery.

At generation time, compute a CLIP image embedding and store `embedding → EtchId` in a
local, append-only store at `$PLAKAT_HOME/etchdb/`. Verification is a nearest-neighbour
query, not an extraction.

This works precisely where the other layers fail. At denoise strength 0.8 the only thing
left of the original *is* its semantics — so match on the thing that survives. This is the
same idea as C2PA's *soft binding*, where the perceptual hash is not the proof but the
lookup key into a manifest store.

`src/pipelines/clip_embed.rs` already provides `ClipEmbedder::embed_image` and `cosine`;
L3 needs a store and an index, not a new model.

**The trade.** L3 requires a store, and a store is state. Two mitigations keep it inside
plakat's local-first line: (a) the store is a plain local directory, never a network
service; (b) it is optional — `--etch-db none` disables L3 and the other three layers keep
working. A published-manifest mode (share your `etchdb` as a static file others can query
offline) is possible later; it is out of scope for ETCH-1.

**Thresholds.** Cosine ≥ 0.92 → strong match; 0.85–0.92 → probable derivative;
< 0.85 → no L3 evidence. These are placeholders pending calibration (see *Open questions*).

---

## `EtchId` derivation

```
EtchId = BLAKE3(key ‖ "plakat-etch-v1" ‖ canonical_manifest)[0..8]
```

where `canonical_manifest` is the deterministic serialization of the generation recipe
(prompt, negative, seed, model, sampler, steps, guidance, size, LoRA stack). Consequences:

- Reproducible: identical recipe under identical key → identical `EtchId`. Consistent with
  plakat's existing determinism contract (`doctor --reproducibility-check`).
- Opaque: the id reveals nothing without the manifest, so it is safe to leave in a
  published image.
- Overridable: `--etch-id <HEX16>` for users who want their own namespace.

### Keying

Two modes:

- **Public key (default).** A published constant. Anyone can verify a plakat image with a
  stock plakat build. This is the ecosystem-interop mode and the right default for an
  open-source tool.
- **Private key.** `--etch-key` / `PLAKAT_ETCH_KEY`. Only holders can verify.

The honest note: **with the public key the carrier is public, so it can be subtracted.**
Public-key mode is a provenance signal against incidental editing, not a defence against a
motivated remover. Private-key mode raises the bar but does not clear it — regeneration
removes marks it cannot see.

---

## CLI surface

### `--etch` (global)

```
--etch                    Enable provenance etching on written images.
                          [env: PLAKAT_ETCH]
--etch-key <KEY>          Key for EtchId derivation and carrier PRNG.
                          [env: PLAKAT_ETCH_KEY]
--etch-id <HEX16>         Override the derived EtchId with an explicit 64-bit value.
--etch-layers <LIST>      Comma-list of layers: l0,l1,l2,l3 (default: all applicable).
--etch-strength <F32>     L1 embedding strength, 0.0..=1.0 (default: 0.35).
--etch-db <PATH|none>     L3 fingerprint store (default: $PLAKAT_HOME/etchdb).
```

Placed in `Cli` in `src/cli/mod.rs` under `help_heading = "Global options"`, alongside
`--device` / `--cache-dir`. Naming follows `Documentation/CLI_CONVENTIONS.md`: kebab-case,
shared `etch-*` family prefix, singular flags.

**Global-option contract.** `--etch` is honoured by every subcommand that *writes an image*
and silently ignored by every subcommand that does not (`models`, `doctor`, `verify`,
`bench`, `inspect`). It is not an error to pass it to a non-image command; making it an
error would break `alias plakat='plakat --etch'`, which is the intended ergonomics.

**Layer applicability by command.** L2 requires plakat to own the sampling loop.

| Command class | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| `generate` `portrait` `multiperson` `persona` `bookart` `scenario` | ✓ | ✓ | ✓ | ✓ |
| `img2img` `outpaint` `relight` `remove` `replace_bg` `restore_faces` | ✓ | ✓ | ✓¹ | ✓ |
| `upscale` `transparent` `compose` `stylize` (classical paths) | ✓ | ✓ | — | ✓ |
| `fractals` (Track A, no diffusion) | ✓ | ✓ | — | ✓ |

¹ Only when the operation reinitializes noise; a low-strength img2img has too little noise
budget to carry rings and L2 is skipped with a `-v` notice rather than embedded weakly.

### Ordering against `transparent`

`plakat transparent` keys pixels by exact match against the upper-left corner colour. L1
perturbs luma by roughly ±1 LSB — which would break the exact match and leave a halo of
un-keyed pixels.

**Therefore the pipeline order is fixed: transparency keying runs first, etching second,
and the L1 embedder treats `alpha == 0` as an exclusion mask.** Any command that produces
an alpha channel (`transparent`, `remove`, `replace_bg`, `bookart`) must respect this.
Getting this backwards produces images that look correct and fail verification, which is
the worst failure mode available.

### `plakat doctor --if-plakat <IMAGE>`

```
--if-plakat <PATH>        Verify whether IMAGE originated from plakat.
--if-plakat-db <PATH>     Fingerprint store to query (default: $PLAKAT_HOME/etchdb).
--verify                  Also run L2 (loads a model; see charter note).
--json                    Structured report.
```

`--if-plakat` conflicts with `--benchmark`, matching the existing pattern for `--json`,
`--reproducibility-check`, and `--capability`.

**Charter note.** `doctor`'s documented contract is that it health-checks *without
downloading or loading anything*, with `--verify` as the existing escape hatch for work
that touches the network or heavy resources. ETCH-1 respects this exactly: **L0, L1, and L3
run fully offline with no model load** (L3's CLIP encoder is small and local, but it is a
model — so L3 runs only when a store is present and the encoder is already cached,
otherwise it reports `unavailable` rather than downloading). **L2 requires `--verify`**,
because DDIM inversion means loading a UNet. No new flag semantics are invented.

### Verdicts

| Verdict | Trigger | Meaning |
|---|---|---|
| `generated` | L1 or L2 decodes with p < 1e-6, or L0 present and consistent | plakat produced this image |
| `derived` | L1 partial + L3 match, or L2 presence without payload | plakat produced an ancestor |
| `probable-derivative` | L3 match only, cosine ≥ 0.92 | semantically matches a known plakat output |
| `inconclusive` | weak/conflicting evidence | do not rely on this either way |
| `no-evidence` | nothing above threshold | absence of evidence, not evidence of absence |

Human output:

```
$ plakat doctor --if-plakat suspect.png

  Etch verification — suspect.png (1024x1024, PNG, alpha: no)

  L0  manifest      absent          (stripped or never written)
  L1  pixel etch    partial         12/16 tiles, 58/64 bits, p = 3.1e-05
  L2  latent etch   skipped         (--verify not given)
  L3  fingerprint   match           cosine 0.943 → 9f2c4a17b3e08d5c

  Verdict: derived  (confidence: high)
  EtchId:  9f2c4a17b3e08d5c
  Note:    L1 bit loss is consistent with a light generative edit.
```

The last line matters. A partial L1 decode plus a strong L3 match has a *shape* — it is
what a low-strength img2img looks like, and distinguishing that from JPEG damage is
information the user wants.

`--json` emits the same structure machine-readably, consistent with doctor's existing
`--json` reports.

---

## Module layout

```
src/etch/
  mod.rs          EtchId, EtchVerdict, EtchConfig; orchestration
  payload.rs      id derivation, CRC-16, ECC encode/decode, tile framing
  manifest.rs     L0 — tEXt chunk + sidecar (extends imaging/metadata.rs)
  pixel.rs        L1 — canonical grid, DWT, DCT, embed/extract, alpha mask
  latent.rs       L2 — Fourier-ring init, DDIM inversion, ring correlation
  fingerprint.rs  L3 — embedding store, index, query
  detect.rs       evidence fusion → verdict + p-value
```

Touch points in existing code:

- `src/cli/mod.rs` — global `etch-*` flags on `Cli`
- `src/cli/doctor.rs` — `--if-plakat` arm
- `src/imaging/io.rs` — etch hook in `save_rgb_u8_with_metadata`
- `src/imaging/metadata.rs` — `etch` chunk
- `src/pipelines/*` — latent-init interception for L2
- `src/pipelines/fft.rs` — reused for ring embedding/correlation
- `src/pipelines/clip_embed.rs` — reused for L3

---

## Cargo

Already present and reusable: `sha2`, `rand`, `png 0.18`, `image 0.25`, `imageproc`,
`serde`/`serde_json`, `deser-hjson`, `rayon` (optional), and the in-tree DFT.

New, all pure-Rust:

| Crate | Purpose | Note |
|---|---|---|
| `blake3` | `EtchId` derivation | or reuse `sha2` and skip the dependency |
| `rand_chacha` | reproducible carrier PRNG from key | `rand`'s ChaCha backend; deterministic across platforms, which `SmallRng` is not |
| — | wavelet transform | **no pure-Rust DWT crate is worth the dependency.** A CDF 9/7 lifting implementation is ~150 lines and belongs in-tree |
| — | ECC | see open questions |

DCT: reuse the DFT-as-matmul in `pipelines/fft.rs`, or hand-roll a separable DCT-II on the
8×8 blocks. Adding `rustdct` is possible but a 512×512 canonical grid does not need it.

---

## Implementation phases (each independently shippable)

**Phase 1 — L0 + surface.** Global `etch-*` flags, `EtchId` derivation, tEXt chunk +
sidecar, `parent` chaining across plakat's own derivation commands,
`doctor --if-plakat` reading L0 only. Ships a working provenance story for the
metadata-preserved path in one release.

**Phase 2 — L1.** Canonical grid, CDF 9/7 lifting, DCT block selection, ECC, tiling, alpha
masking, `--etch-strength`. Robustness suite over transcode / rescale / crop / alpha /
rotate. Verdict fusion for L0+L1. This is the phase that delivers the stated requirement
minus img2img.

**Phase 3 — L3.** Fingerprint store, index, query, `--etch-db`. Adds `derived` and
`probable-derivative` verdicts. Deliberately before L2: it is cheaper, it covers strictly
more of the img2img range, and it does not touch the sampler.

**Phase 4 — L2.** Ring pattern in `z_T`, per-family latent-init interception, DDIM
inversion detect behind `doctor --if-plakat --verify`. The highest-risk phase and the one
most likely to be scoped down.

**Phase 5 — Calibration + docs.** Threshold calibration on a real corpus, ROC curves,
`Documentation/ETCH.md`, README section, honest limits stated in user-facing help text.

---

## What this does not do

Stated plainly so it is not discovered later:

1. **It does not survive determined removal.** Invisible pixel watermarks are provably
   removable by generative regeneration, and purpose-built generative-edit-robust schemes
   have been shown to collapse under guided diffusion attacks. Anyone who wants the mark
   gone can remove it.
2. **It does not survive high-strength img2img at the bit level.** Above roughly 0.6
   denoise strength, expect L1 to be gone and L2 to be unreliable. L3 is what remains.
3. **"Derivative" stops being well-defined at high strength.** At 0.9 with a changed
   prompt, the model has regenerated from its own prior with a hint. There is genuinely
   nothing left to detect, and arguably nothing left to claim.
4. **It is not C2PA.** ETCH-1 is a self-contained, offline, tool-specific mechanism with no
   certificate chain and no signing authority. Emitting a C2PA manifest alongside L0 is a
   reasonable follow-on RFC; it is not this one. Note that C2PA's own security spec
   concedes it offers no protection against complete manifest removal — which is why
   ETCH-1 leads with soft bindings rather than metadata.
5. **`no-evidence` is not proof of non-plakat origin.** The verdict vocabulary is
   deliberately worded to avoid implying otherwise, and the docs must not walk that back.

The defensible claim, and the one user-facing text should make: *verifiable through
incidental editing, format churn, rescaling, and moderate generative edits; unenforceable
against a determined remover.*

---

## Open questions

1. **ECC scheme.** BCH over GF(2^m) is the classic fit for a fixed 64-bit payload but needs
   an in-tree implementation. Reed–Solomon over GF(256) has pure-Rust crates but is
   byte-oriented and wastes capacity on a 64-bit payload. A repetition + soft-decision
   majority scheme is trivial and surprisingly competitive when tiles already vote.
   **Recommendation:** start with repetition + majority in Phase 2, revisit after the
   robustness suite produces real bit-error rates.

2. **L3 thresholds.** The 0.92/0.85 cosine cut points are unvalidated. They need an ROC
   curve over a corpus of plakat outputs, plakat-derived edits, and unrelated images. A
   false `probable-derivative` on an unrelated image is the expensive error.

3. **CLIP vs DINOv2 for L3.** CLIP is already in-tree, which is a strong argument. DINOv2
   embeddings are generally better for near-duplicate retrieval and worse for semantic
   drift. Which failure mode matters more depends on where the img2img strength distribution
   actually sits — needs measurement, not argument.

4. **`--etch` default state.** Off by default (proposed) respects least surprise and keeps
   generation byte-identical for existing users. On by default would make the ecosystem
   claim meaningful at scale. Turning it on by default is a breaking change to
   reproducibility unless L1 is excluded from the determinism contract.

5. **L2 across model families.** Ring embedding is well-characterized for SD 1.5 / SDXL
   latents. Flux, SD3, and Cascade have different latent geometries and channel counts.
   Phase 4 may reasonably ship L2 for a subset and report `unsupported` elsewhere.

6. **Store portability.** Should `etchdb` be shareable — a static, queryable artifact a
   third party can use to verify without contacting the author? This is the difference
   between personal provenance tracking and an ecosystem claim, and it is the largest
   open scope question in this RFC.

7. **Interaction with `--fractal-clone` and `fractalspec`.** Fractals already embed a spec
   chunk. Two provenance mechanisms in one PNG need a defined precedence, or a decision to
   fold `fractalspec` under the `etch` umbrella.
