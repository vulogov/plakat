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
- [x] **In-process scenario / portrait runner** — scenario runs + People quick-gen
      (portrait / multiperson) now execute **on the ModelService thread**, which **drops
      the loaded Chat pipeline first** (deterministic — same thread) before the run loads
      its own model. Only one model is ever resident, so no double-load OOM on unified
      memory. New `ModelCommand::{RunScenario,RunPortrait,RunMultiperson}` + matching
      `ModelService` methods; the App routes through them instead of spawning competing
      threads and notes "freeing <alias> — reload with L after". (A true *shared* pipeline
      — reuse the loaded weights when the scenario's model+LoRAs match — is a further
      optimization; freeing avoids the OOM either way.)

## B — People depth (remaining)

- [ ] **Identity-preserving Chat continuation** — an IP-Adapter-aware refine path so a
      continued portrait keeps face identity (Chat refine is plain img2img today). Builds
      on the encoding-quality work (`E`).
- [x] **Auto-encode + invalidation** — the encoding-quality score now computes
      **automatically** the first time the ENCODING tab is viewed on an unscored identity
      (once per identity per session, off-thread), and is **invalidated** when the refs or
      strategy change: the sidecar is now `quality.json` carrying a `fingerprint` (FNV of
      the sorted ref path+size set + the identity strategy); on load the score is restored
      only if the fingerprint still matches, else it's dropped (→ recomputed). `E` still
      forces a recompute.

## C — History (remaining)

- [ ] **Embedding-based semantic search** — rank the `/` filter by meaning, not just
      substring (the substring filter + the thumbnail grid + background decode shipped in
      1.17.0). Needs a small text-embedding model.

## D — LoRA Hub (remaining)

- [x] **Download manager depth** — the LoRA-Hub download manager is now complete:
      **version-update detection** (`U` on a Civitai LOCAL LoRA → newer version → download),
      **≤2 concurrent downloads** with a FIFO queue (the `●` marker covers in-flight +
      queued; the status line shows the queue depth), and **range-resume** (a leftover
      `.partial` is continued via an HTTP Range request — `resume_action` maps the
      response: 206 append · 200 restart · 416 wipe+retry). With the 1.17.0 SHA-256 verify
      + 1h search caching, the manager covers the full RFC §10 surface.
