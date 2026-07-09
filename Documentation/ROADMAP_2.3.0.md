# plakat 2.3.0 — roadmap (planning)

The 2.x arc has been about **confidence**: 2.0 built `plakat verify`, 2.1 used it to fix a real
caption bug, 2.2 took it to full end-to-end coverage across all 6 rendering models. 2.3.0
turns outward: make plakat usable as a **library**, not just a CLI.

Anchor theme: **library / API stabilization**. Scope is refined with the user, not pre-committed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Anchor: library / API stabilization

Today `src/lib.rs` re-exports **22 internal modules as `pub`** (`pipelines`, `imaging`,
`scripting`, `hf`, …). Everything is public by accident, nothing is a *designed* surface: no
stability promise, no facade, sparse rustdoc, no library examples. A downstream crate can
technically `use plakat::pipelines::t2i::…`, but against the full churny internals.

The work is to carve out a small, documented, **semver-stable public API** and hide the rest.

Decision (2026-07-08, with the user): cover **all CLI features except the UI**, via a
**simple builder facade** (`plakat::api`) that returns images in-memory. Every non-UI command
is `async` + a `(device, request) → result` call, so the facade wraps those request structs
ergonomically. Design confirmed by a full CLI→lib-entry-point map (see below).

- [x] **Facade module `plakat::api`** — **14 builders** covering every non-UI feature area,
      returning `Image`s in memory (render-to-temp + read-back hidden inside): `Generate`,
      `Img2img` (+inpaint), `Upscale`, `Relight`, `Stylize`, `Transparent`, `Segment`,
      `Portrait`, `Multiperson`, `Map`, `Animate`, `StyleTrain`, `EmbeddingTrain`, `Verify` +
      `Image`/`device()` + re-exported value types. Purely additive (wraps existing async
      entries; no internal refactor). *Follow-up knobs — controlnet/embeddings/refiner/tiled/
      regions/flux-quant on the builders — deferred; the CLI still exposes them.*
- [x] **Hide internals** — all 22 internal `pub mod`s marked `#[doc(hidden)]` with a crate-doc
      pointing at `plakat::api` as the sole stable surface (kept `pub` so the bin/tests share
      them; no semver promise). `#[doc(hidden)]` over a feature gate because the in-crate bin
      needs them unconditionally.
- [x] **Docs + examples** — crate + module rustdoc, a runnable doctest, and
      `examples/library.rs` (generate → img2img → upscale).
- [x] **Semver hygiene** — `tests/api_surface.rs` locks the surface at compile time (rename/
      remove a public item ⇒ red test). `cargo public-api` can layer on later for a full diff.

## Bund scripting — close the coverage gaps (asked 2026-07-08)

Audit found CLI features with no Bund host-word equivalent. Closed:

- [x] **4 image-producing words** — `plakat.relight`, `plakat.transparent`, `plakat.segment`,
      `plakat.compose`. Each mirrors its CLI subcommand (block_in_place + block_on the async
      pipeline). Shared `helpers::resolve_image_arg`.
- [x] **Training words** — `plakat.style.train` (all families, dispatched via the shared
      `api::style_train`) + `plakat.embedding.train` (SDXL/SD3.5). Take a folder of images, use
      the loaded model as the base, sensible training defaults.
- [x] **Cascade `decoder_guidance`** — new `plakat.config.set` key (was hardcoded 1.1 in
      `plakat.cascade`); threads Stage-B CFG through.
- [x] **Flux quant split** — *already covered* (`quant_level` + `t5_quant_level` config keys
      thread into the Flux request via `ctx.rs`); the audit was stale. No work needed.
- [ ] **Regional prompting** (`plakat.region.*`) — deferred: NOT a partial-knob fix. The script
      path's `t2i::GenRequest` has no `regions` field (unlike `sd3::GenRequest`), so it needs
      pipeline plumbing (thread regions → the SD regional-denoise wiring) + a new namespace, not
      just a word. A focused follow-up.

Host words: 51 → **57**. SCRIPTING.md updated.

## Verify track — status (asked 2026-07-08)

**Essentially complete.** 6 families × 3 tiers × 3 fixtures, all hosted + green; CI-gated
(Tier 0 every push, weight-backed job cached + tier-selectable). Three real bugs found + fixed
across 2.0–2.2. Honestly-marginal leftovers, none worth a cycle:

- [ ] **sd21 `unet.out`** — the only *cheap* gap: SD 2.1 uses the same candle UNet as SD 1.5
      (which has `unet.out` at corr 1.0), so it's just authoring the golden. Nice-to-have.
- [ ] **AnimateDiff Tier-2** — a frozen-frame/GIF end-to-end regression. Marginal (its
      `motion.block0` Tier-1 tap already covers the motion core).
- [⏸] **Flux verify** — a genuine coverage gap (Flux isn't in the pilot set), but heavy: Flux
      is huge, GGUF-on-Metal is broken (candle kernel bug), CPU-F32 is >24 GB. Blocked on
      hardware / the candle Metal fix.
- Documented can't-dos (not reopening): full-DiT `dit.out` + SD 1.5 `unet.mid` + Cascade Attn
      (all OOD-on-synthetic-input or disproportionate-vendoring — see ROADMAP_2.1/2.2).

Recommendation: don't invest more in verify; the arc is done. `sd21 unet.out` is a 10-minute
add if we want the symmetry.

## Other candidate themes (deferred / secondary)

- [ ] **Performance pass** — profile + optimize hot paths (VAE decode, attention, weight load).
- [ ] **Flux in the UI** — carried since 1.16; unblock when capable hardware is available.

## House-keeping

- [x] **Open 2.3.0** — branch off `main` (2.2.0 release), version bump `2.2.0 → 2.3.0`.
