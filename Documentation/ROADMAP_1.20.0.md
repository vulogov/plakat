# plakat 1.20.0 — roadmap

1.19.0 added History semantic search, hardened `plakat ui` against OOM host crashes, and
fixed the Canvas "where am I masking / mask the latest image" gaps. With that, the **RFC
TUI-1 surface is complete** — every screen and its depth features are implemented and
shipped.

Only one UI item remains, and it's hardware-blocked. 1.20.0 is otherwise an **open
cycle** — the next direction (new pipelines, CLI work, training, or polish) is TBD.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## A — generation engine

- [⏸] **Flux in the UI** — `flux::run` dispatch (load-per-call ~25-field `Request` + the
      GGUF-Metal guard). **Postponed**: can't be verified end-to-end on the current box
      (Flux too large to run here; GGUF-Flux-on-Metal is a known-broken kernel path).
      Resume when verifiable hardware is available — the `ModelService::Loaded` enum +
      family dispatch already have the seam for it.

## Possible directions (unprioritised)

The `plakat ui` backlog is drained. Candidate next areas, pending the user's call:

- **Neural semantic search** — upgrade History's TF-IDF ranker to a real text-embedding
  model behind the existing `services::semantic::rank` seam (weight download + load cost).
- **Memory** — a TUI "hard reset" that re-execs the process to fully return candle's
  Metal buffer pool (which has no in-process force-clear), for very long sessions.
- New model families, CLI ergonomics, or training-surface work — TBD.
