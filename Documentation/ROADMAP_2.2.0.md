# plakat 2.2.0 — roadmap (planning)

2.0.0 shipped **`plakat verify`** (the self-contained correctness harness). 2.1.0 was that
harness **paying off**: it found + fixed a real caption bug (PixArt/SD3 encoded T5 captions
without masking pad tokens — corr 0.70→1.0, self + cross attention), added a second prompt
fixture, the PixArt `adaln.embedded_timestep` tap, and end-to-end Tier-2 gates for SDXL +
PixArt. It also learned its limits (`dit.out`, SD 1.5 `unet.mid` — evaluated + dropped,
documented). See `RFC_VERIFY.md` / `VERIFY.md`.

2.2.0 opens as a **planning landing pad** — scope is decided with the user, not pre-committed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried-over verify follow-ups (cheap, incremental)

- [x] **Tier-2 breadth (finish)** — end-to-end gates now cover **all six** rendering models
      (sd15/sd21/sdxl/pixart/cascade/sd35), 4 pipeline families, each SSIM 1.0. Finishing SD 3.5
      + Cascade exposed two real determinism gaps the gate fixed: SD3 was gating the wrong
      noise site (plain-t2i `eps` was still random), and Cascade's Wuerstchen scheduler is
      **stochastic** (fresh N(0,1) every step) — both now env-gated deterministic under verify.
- [x] **Third fixture** — `emblem_v1`, a tokenization-edge-case prompt (numbers/hyphens/
      ampersand/punctuation). A1111 weighting was avoided (no diffusers reference). Verified
      corr 1.0 across all six models × the conditioning taps. Broader tokenization coverage.
- [ ] **Phase-4 hardening** — the opt-in `verify-models` CI job downloads multi-GB weights; a
      smaller cached/quantized fixture model would make a full-correctness gate cheaper to run.

## Candidate themes (bigger, standalone — pick one to anchor 2.2)

- [?] **Library / API stabilization** — a documented, semver-stable Rust crate API so plakat is
      usable as a library, not just a CLI. A fitting continuation of the 2.x "confidence" arc.
- [?] **Performance pass** — profile + optimize hot paths (VAE decode, attention, weight load);
      user-visible generation speedups on the 24 GB Metal box.
- [?] **Flux in the UI** — carried since 1.16; unblock when capable hardware is available.

## House-keeping

- [x] **Open 2.2.0** — branch off `main` (2.1.0 release), version bump `2.1.0 → 2.2.0`.
