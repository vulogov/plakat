# plakat 1.18.0 — roadmap

1.17.0 closed most of the `plakat ui` depth backlog (command palette, `@mention`, Chat
sessions + filmstrip, People sub-tabs / re-encode / mixed-family, Tera mode, face-aware
Canvas + finer masks, History thumbnail grid + background decode, LoRA caching / HF
pre-filter / SHA-verify) and fixed the scenario-in-TUI bugs. 1.18.0 carries the few
remaining items — all of them genuinely heavy (pipeline / infra), plus the one model
family that needs hardware.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## A — generation engine

- [⏸] **Flux in the UI** — `flux::run` dispatch (load-per-call ~25-field `Request` + the
      GGUF-Metal guard). **Postponed**: can't be verified end-to-end on the current box
      (Flux too large; GGUF-Flux-on-Metal is a known-broken kernel path). Resume when
      verifiable hardware is available.
- [ ] **In-process scenario / portrait runner** — Scenario runs and People quick-gen
      load their *own* model alongside any Chat model (double load, memory pressure).
      **Large refactor**: extract a runner that accepts an already-loaded pipeline +
      reconcile the scenario's model field. (RFC §0-R0-2)

## B — People depth (remaining)

- [ ] **Identity-preserving Chat continuation** — an IP-Adapter-aware refine path so a
      continued portrait keeps face identity (Chat refine is plain img2img today). Builds
      on the encoding-quality work (`E`).
- [ ] **Auto-encode + invalidation** — compute the encoding-quality score automatically
      on first strategy+model use, and invalidate it on a ref / strategy / model change
      (today it's explicit via `E`).

## C — History (remaining)

- [ ] **Embedding-based semantic search** — rank the `/` filter by meaning, not just
      substring (the substring filter + the thumbnail grid + background decode shipped in
      1.17.0). Needs a small text-embedding model.

## D — LoRA Hub (remaining)

- [ ] **Download manager depth** — ≤2 concurrent, range-resume, version-update detection
      (SHA-256 verify + the `●` indicator + 1h search caching shipped in 1.17.0).
