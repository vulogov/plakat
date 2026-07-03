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

## Candidate major themes (pick 1–2)

- [ ] **Library / API stabilization** — a documented, semver-stable Rust crate API (not just
      the CLI/UI), so plakat can be embedded. A real 2.0 signal.
- [ ] **Performance pass** — profile the hot paths (VAE decode, attention, weight load),
      reduce peak memory, faster first-token-to-image.
- [ ] **Flux in the UI** — unblock when capable hardware is available (currently the only
      standing UI carry; GGUF-Flux-Metal still broken upstream).
- [ ] **Verification harness as a first-class tool** — promote the diffusers
      reference-comparison method (used to fix the silent-noise bugs) into a repeatable
      `plakat verify` that guards every model against regressions.

> Nothing here is committed. This is the menu for the 2.0 conversation.
