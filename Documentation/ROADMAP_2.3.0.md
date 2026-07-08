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

- [?] **Decide the surface** — what does "use plakat as a library" mean? Likely a thin facade:
      load a model → generate an image → get pixels/PNG (the `generate` / `img2img` / `verify`
      cores), plus the value types (`GenRequest`-like, device, scheduler, output). NOT the 22
      internal modules. Needs a decision on scope + the semver promise.
- [ ] **Facade module** — a curated `plakat::api` (or top-level re-exports) with a stable,
      ergonomic entry point independent of the CLI plumbing. Internals move behind
      `#[doc(hidden)]` / a `internals` feature so they can still churn.
- [ ] **Docs + examples** — rustdoc on the public surface, a docs.rs landing page, and
      `examples/` that compile in CI (generate an image from a few lines of Rust).
- [ ] **Semver hygiene** — a public-API test (e.g. `cargo public-api` / a doc-test snapshot) so
      breaking changes are caught; document the stability tier.

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
