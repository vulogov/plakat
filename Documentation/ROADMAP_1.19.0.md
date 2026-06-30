# plakat 1.19.0 — roadmap

1.18.0 closed almost the entire `plakat ui` depth backlog: the in-process runner (no
double-load OOM), identity-preserving Chat continuation, People auto-encode +
invalidation, and a complete download manager (version-update detection + ≤2 concurrent +
range-resume). Two items remain — one heavy, one hardware-blocked.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## A — generation engine

- [⏸] **Flux in the UI** — `flux::run` dispatch (load-per-call ~25-field `Request` + the
      GGUF-Metal guard). **Postponed**: can't be verified end-to-end on the current box
      (Flux too large; GGUF-Flux-on-Metal is a known-broken kernel path). Resume when
      verifiable hardware is available.

## B — History

- [x] **Semantic search** — `?` ranks History by *relevance* (most-related first) rather
      than substring-filtering. Implemented with the classic vector-space model: each
      image's searchable text (filename + tags + recipe) → a **TF-IDF embedding**, ranked
      by **cosine** to the query (`services/semantic.rs`, pure + tested). Chosen over a
      neural text-embedding model deliberately — a ~250MB model load would wreck History's
      lightweight feel; the TF-IDF embedder is zero-dep, instant, and offline, and gives
      real meaning-aware ranking ("snowy peak" → "a mountain in winter, fresh snow"). A
      neural-embedding upgrade remains possible behind the same `semantic::rank` seam.

---

The `plakat ui` RFC TUI-1 surface is otherwise **complete**. Beyond these, 1.19.0 is an
open cycle — the next direction (new pipelines, CLI work, or polish) is TBD.
