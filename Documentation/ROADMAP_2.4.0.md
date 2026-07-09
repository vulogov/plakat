# plakat 2.4.0 — roadmap (planning)

2.3.0 made plakat a first-class **library** (`plakat::api`, 14 builders) and brought Bund
scripting up to CLI parity (bar one plumbing item). 2.4.0 opens as a **planning landing pad** —
scope is decided with the user, not pre-committed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried-over follow-ups (bounded, incremental)

- [ ] **Regional prompting in Bund** (`plakat.region.*`) — the one Bund gap left. NOT just a
      word: needs `t2i::GenRequest` to gain a `regions` field + the SD regional-denoise wiring
      (sd3::GenRequest already has it), then a `region.{add,clear,list}` namespace threaded like
      `plakat.lora.*`. A focused pipeline + scripting change.
- [ ] **Per-builder API knobs** — the `plakat::api` builders cover every *feature* but expose the
      common knobs only. Add the finer ones the CLI has: ControlNet, embeddings, refiner, tiled,
      regions, flux-quant. Purely additive to the (locked) surface.
- [ ] **`cargo public-api` in CI** — richer than the compile-time `tests/api_surface.rs` floor:
      a full public-surface diff so any semver-affecting change to `plakat::api` is flagged in a
      PR. (The internals are `#[doc(hidden)]`; scope the check to the api module.)
- [ ] **`sd21 unet.out` verify symmetry** — trivial: SD 2.1 uses the same candle UNet as SD 1.5
      (which has `unet.out` at corr 1.0); just author the golden. Closes the last cheap verify
      gap.

## Candidate themes (bigger, standalone — pick one to anchor 2.4)

- [?] **Performance pass** — profile + optimize hot paths (VAE decode, attention, weight load);
      user-visible generation speedups on the 24 GB Metal box. A natural next theme now that
      correctness (verify) and reach (library API) are solid.
- [?] **Flux in the UI** — carried since 1.16; unblock when capable hardware is available.
- [?] **Library polish** — a `plakat::api` prelude, streaming/progress callbacks on `run()`, and
      builder ergonomics (accept `Device` as well as `&str`), if the library gets real users.

## House-keeping

- [x] **Open 2.4.0** — branch off `main` (2.3.0 release), version bump `2.3.0 → 2.4.0`.
