# plakat 2.0.0 — roadmap (major cycle, planning)

1.0.0 was "the cut" (confidence / stability / polish). Since then 1.1–1.21 added the map
generator, the `plakat ui` terminal UI, and the power-user loop; **1.22.0** landed a
source-wide stability pass (40 bugs fixed — see `BUGFIX_PLAN.md`). 2.0.0 is the next major
milestone. This file opens as a **planning landing pad** — scope is decided with the user,
not pre-committed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried-over known work (from the 1.22.0 stability pass)

Concrete, already-scoped — the honest backlog to clear early in 2.0.

- [ ] **T5 pad attention mask** (BUGFIX 1.2) — PixArt/SD3 encode captions without masking
      pad tokens; a correct fix means the vendored T5 + threading a mask through every
      DiT/MMDiT cross-attention block. **Needs the diffusers reference-comparison harness**
      to verify it doesn't regress the (currently correct) models.
- [ ] **Transformer-trainer resume warm-up** (BUGFIX 4.1 follow-up) — apply the SD/SDXL
      resume LR warm-up to the sd3 / pixart / cascade training loops too.
- [ ] **Scenario style+persona lazy-load** (BUGFIX 3.2 full fix) — evict stylize/portrait
      pipelines per task-kind instead of holding all co-resident (currently: preflight
      warning only).
- [ ] **compvis output-block LoRA mapping** (BUGFIX 5.6) — map attention-less up-block
      upsamplers correctly (currently fail-safe under-apply).

## Decisions needed

- [?] **Map cross-machine determinism** — the docs claim byte-stable across machines, but
      terrain/coast math uses non-correctly-rounded libm transcendentals (last-ULP varies by
      platform). **Either** narrow the wording to "byte-stable on a fixed platform" **or**
      vendor fixed-point implementations of the determinism-critical transcendentals (as was
      done for the bitmap font). Decide before spending effort.
- [?] **What makes it 2.0** — the theme. Candidates below; pick with the user.

## Chosen theme — `plakat verify` (RFC_VERIFY.md)

Promote the diffusers reference-comparison method (which fixed every silent-noise bug)
into a first-class, **self-contained** subcommand. Shipped tool stays pure-Rust/HF-only;
diffusers lives only in an offline authoring step whose golden tensors are frozen on HF.
Full design + impact analysis: [`RFC_VERIFY.md`](RFC_VERIFY.md).

- [x] **Phase 0 — framework + Tier 0 (zero downloads).** `src/verify/` + `plakat verify`
      `[--tier N] [--json]`; Tier 0 structural/determinism checks (CFG batch-layout invariant
      guarding BUGFIX 1.1, map byte-stability); higher tiers report as skipped. Green offline
      + in CI, no external data.
- [x] **Phase 1 — comparison engine + capture abstraction (self-contained, tested).**
      `compare` (corr/cosine/max-abs + thresholds), the `Manifest` format + loader, golden
      safetensors loading, the `tier1` runner (`compare_against_goldens`), and the
      `TensorTap`/`CaptureBag` capture abstraction — 13 unit/integration tests, no model
      needed. `verify --tier 1` reports the pilot set as *ready, awaiting goldens*.
- [ ] **Phase 1b/2 — wire `TensorTap` capture points into the pipelines + author goldens.**
      Naturally coupled: each capture point is wired and its golden authored together
      (offline diffusers harness → HF), so the capture is verified against the reference the
      moment it lands. Pilot: sd15/sdxl/sd35/pixart/cascade/animatediff.
- [ ] **Phase 3 — Tier 2 end-to-end perceptual gate** (golden corpus PNGs).
- [ ] **Phase 4 — CI, regression baselines, docs**; makes BUGFIX 1.2 + the map-determinism
      decision cheap to verify.

## Other candidate themes (deferred / secondary)

- [ ] **Library / API stabilization** — a documented, semver-stable Rust crate API.
- [ ] **Performance pass** — profile hot paths (VAE decode, attention, weight load).
- [ ] **Flux in the UI** — unblock when capable hardware is available.
