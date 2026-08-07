# ROADMAP — plakat 6.7.x · ETCH-1 (provenance etching)

Implements `Documentation/RFC_ETCH_1.md` — a 64-bit `EtchId` written by four independent evidence
layers (L0–L3) + `plakat doctor --if-plakat` reading whatever survived into a graded verdict. New
`--features etch`. Branch `6.7.0` (off `main` @ `b94f685`, v6.6.0).

**RFC grounding verified:** every cited seam exists — `imaging::io::save_rgb_u8_with_metadata` (io.rs:110),
the `parameters` tEXt chunk (`imaging::metadata`), `ClipEmbedder::embed_image`+`cosine`
(clip_embed.rs:69/107), the DFT-as-matmul (`pipelines::fft`), the global-option pattern (cli/mod.rs
`Cli`, `help_heading="Global options"`, `global=true`), doctor's `conflicts_with="benchmark"` flags.

---

## Decisions adopted from the RFC's open questions (the recommendations, made concrete)

- **ECC (Q1):** repetition + soft-decision majority in Phase 2 (tiles already vote); revisit after the
  robustness suite yields real BER. **BER-driven, not chosen up front.**
- **`--etch` default (Q4):** **OFF by default** — keeps generation byte-identical for existing users and
  preserves the determinism contract; on-by-default is a later, separate decision.
- **L3 model (Q3):** **CLIP** (already in-tree via `clip_embed.rs`); DINOv2 is a later measurement.
- **`EtchId` hash:** use in-tree **`sha2`** (SHA-256 of `key ‖ "plakat-etch-v1" ‖ canonical_manifest`,
  first 8 bytes) rather than adding `blake3` — the RFC explicitly allows this ("or reuse `sha2` and skip
  the dependency"). Opaque + reproducible identically.
- **`rand_chacha`:** added only when Phase 2 needs the deterministic carrier PRNG (cross-platform stable,
  unlike `SmallRng`).
- **DWT / DCT / ECC:** in-tree (CDF 9/7 lifting ~150 lines; separable DCT-II or the fft.rs DFT; repetition
  ECC) — no new crates beyond `rand_chacha`.
- **Feature gating (deviation from RFC, gate-safety):** the RFC proposes `--features etch` (default on),
  but the CI gate runs `cargo test --no-default-features --lib`, so a default-on *feature* would be a
  **blind spot** — the exact shape of the 6.5.0 Windows regression (code the gate never compiles). So
  `etch` is **always compiled (no cargo feature)**; the runtime **`--etch` flag (default OFF)** is the
  opt-in. Pure-Rust and cheap, so always-present costs nothing and the gate fully covers it. (L2/L3
  model-loading paths gate on runtime state, not a cargo feature.)

---

## Phases (per RFC §"Implementation phases" — each independently shippable)

### Phase 1 — L0 + surface  *(the proposed 6.7.0 core)*
Global `etch-*` flags on `Cli` (`--etch`/`--etch-key`/`--etch-id`/`--etch-layers`/`--etch-strength`/
`--etch-db`, `help_heading="Provenance"`, `global=true`, env-backed); `src/etch/` module skeleton;
`EtchId` derivation (`payload.rs`, sha2-keyed, `--etch-id` override); L0 `etch` tEXt chunk + `<base>.json`
sidecar (`manifest.rs`, extends `imaging::metadata`); the etch hook in `save_rgb_u8_with_metadata`;
`parent` chaining across plakat's own derivation commands (img2img/outpaint/relight/remove/replace_bg/
restore_faces/upscale); `doctor --if-plakat <IMAGE>` reading **L0 only** (offline, no model) with the
graded verdict scaffold (`generated` on consistent L0) + `--json`. **Ships a complete provenance story
for the metadata-preserved path.** Feature `etch` (default on). G-gate: `--etch` OFF = byte-identical
output (hash-compare a render with/without the flag).

### Phase 2 — L1 pixel etch
Canonical 512×512 luma grid; CDF 9/7 lifting DWT; key-permuted mid-band DCT coefficient selection;
repetition-ECC + 16-bit CRC + tile framing (`payload.rs`); 4×4 tiling with per-tile vote; **`alpha==0`
exclusion mask**; `--etch-strength` (default 0.35, target ≥42 dB PSNR). **`transparent` ORDERING is
load-bearing: keying first, etch second** (the RFC's worst-failure-mode warning). Robustness suite:
transcode / rescale / crop / alpha / rotate → BER table drives the ECC revisit. L0+L1 verdict fusion with
a p-value. **Delivers the stated requirement minus img2img.**

### Phase 3 — L3 fingerprint
CLIP-embedding store + index + query at `$PLAKAT_HOME/etchdb/` (`fingerprint.rs`); `--etch-db PATH|none`;
adds `derived` / `probable-derivative` verdicts; L3 in `doctor --if-plakat` (offline, runs only if the
store + a cached CLIP encoder exist, else `unavailable` — respects the doctor charter). Before L2 per the
RFC (cheaper, covers more of the img2img range, no sampler touch).

### Phase 4 — L2 latent etch  *(highest risk; most likely scoped down)*
Fourier-ring pattern in `z_T` before step 0 (reusing `pipelines::fft`, at the AnimateDiff `Option<Tensor>`
noise-override seam); per-family latent-init interception (SD1.5/SDXL first, `unsupported` elsewhere —
Q5); DDIM inversion + ring correlation behind `doctor --if-plakat --verify` (the model-loading escape
hatch). Presence bit + ~16-bit `EtchId` prefix.

### Phase 5 — calibration + docs
L3 threshold ROC over plakat / plakat-derived / unrelated corpora (Q2); `Documentation/ETCH.md`; README;
**honest limits in user-facing help** (the RFC's "what this does not do" — verifiable through incidental
editing/format-churn/rescale/moderate edits, unenforceable against a determined remover).

---

## Proposed 6.7.0 scope
The RFC makes each phase independently shippable. **Recommended 6.7.0 = Phase 1 + Phase 2** — a genuinely
useful, honest provenance release (etch + verify through transcode / rescale / crop / alpha, the full
CLI/doctor surface), with **Phase 3 (L3), Phase 4 (L2), Phase 5 (calibration)** as follow-ons (6.7.x or a
later slot). *Owner to confirm the cut line.* Alternatives: Phase 1 only (tightest first release) · 1–3
(adds the fingerprint that covers img2img) · all (large).

## Non-goals (RFC §"What this does not do")
Not removal-proof; not bit-recovery through high-strength img2img; not C2PA; `no-evidence` ≠ proof of
non-plakat origin. User-facing text must not overclaim.

## Cut checklist (per the release lessons)
Cargo bump+lock in sync; gate `cargo test --no-default-features --lib` (**note: `etch` default-on, but the
gate runs `--no-default-features` — ensure the etch module compiles/tests under both**); **pin turbofish
on every new `.parse()`** (6.5.0 Windows lesson); FF `git push 6.7.0:main`; tag → CI 6-asset; `cargo
publish --locked --allow-dirty --no-default-features`; `gh release edit` + bg waiter; **verify the Windows
Release leg**; NO Claude/Anthropic coauthor.
