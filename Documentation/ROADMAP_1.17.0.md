# plakat 1.17.0 — roadmap: `plakat ui` depth, continued

1.16.0 closed the bulk of the RFC TUI-1 deferrals: SD3/PixArt/Cascade in the UI,
StepHook-wired refine + `/auto`, the LoRA Hub completeness set + Chat→Scenario, People
import/forget, History filter/tag/export/compare, Canvas outpaint, Prompt buffer
cycling, and Chat session save/load. 1.17.0 carries the heavier / more design-y items
that were deliberately left — plus the one model family that needs hardware.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

Reference: [`Documentation/RFC_TUI_1.md`](RFC_TUI_1.md).

## A — generation engine

- [⏸] **Flux in the UI** — `flux::run` dispatch (load-per-call ~25-field `Request` + the
      GGUF-Metal guard). **Postponed**: can't be verified end-to-end on the current box
      (Flux too large to run here; GGUF-Flux-on-Metal is a known-broken kernel path).
      Resume when verifiable hardware is available.
- [ ] **In-process scenario / portrait runner** — Scenario runs and People quick-gen
      load their *own* model alongside any Chat model (double load, memory pressure).
      **Large refactor**: extract a runner that accepts an already-loaded pipeline +
      reconcile the scenario's model field. (RFC §0-R0-2)

## B — People depth (remaining)

- [x] **Detail sub-tabs** (RFC §11) — six `←/→`-cycled tabs in the DETAIL pane: **REFS**
      (weighted photos + angle-coverage guidance), **ENCODING** (strategy/mode/quality +
      on-disk encoding count), **PORTFOLIO** (portfolio images + consistency score),
      **TEST** (the 4 fixed identity tests + on-disk results), **KNOWN-GOOD** (the
      `known_good[]` param-combo table), **SETTINGS** (consent + privacy audit). Added
      `encoding_quality` + `known_good[]` + `KnownGood` to the schema. The *active*
      bits — re-encode (`E`), side-by-side strategy compare, apply-a-specific-combo,
      lazy thumbnail grid — remain (see the dedicated items below).
- [ ] **Re-encode** (`E`) — explicit identity encoding with a quality score; auto on
      first strategy+model use; invalidated by ref/strategy/model change.
- [ ] **Mixed-family multiperson** — route a marked set by each persona's strategy
      instead of forcing plus-face/sd15 for the whole scene.
- [ ] **Identity-preserving Chat continuation** — an IP-Adapter-aware refine path so a
      continued portrait keeps face identity (Chat refine is plain img2img today).

## C — History (remaining)

- [ ] **True thumbnail grid** (lazy, LRU cache) instead of list + single preview.
- [ ] **Background image decode** — move the selected-image decode off the event-loop
      tick to a worker so large (upscaled) PNGs never hitch navigation.
- [ ] **Embedding-based semantic search** — rank the `/` filter by meaning, not just
      substring (the substring filter over filename/tags/recipe shipped in 1.16.0).

## D — Prompt Workspace + Canvas (remaining)

- [x] **Tera mode** (`Ctrl-T`) — toggles the compile Tera pre-pass in the Prompt
      Workspace: the buffer renders through Tera (with the panel's variable values)
      before the structural / LLM compile. A live **variable panel** (`Ctrl-V` to focus)
      lists every `{{ variable }}` the template reads (heuristic extractor that skips
      loop vars, `set` targets, `.attr`/`|filter` tails, keywords) and lets you set
      values inline; the compiled output re-renders live. Needs `--features templates`
      (else the pane shows the recompile hint). Also in the command palette.
- [x] **Canvas face-aware `B`** — the background preset now runs SCRFD on the base
      (once per base, background thread), normalizes the boxes to 0..1, and punches the
      (slightly padded) face cells out of the background fill so the inpaint preserves
      the people. Graceful no-op when no detector is configured / no faces found.
- [ ] **Finer masks** — document the external-editor + `--mask-path` path more
      prominently; optionally a finer grid toggle.

## E — polish (remaining)

- [x] **Command palette** (RFC §5) — `Ctrl-K` opens a fuzzy launcher overlay
      (`palette.rs`): context commands for the active screen + jump-to-any-screen +
      quit. Subsequence fuzzy filter; most entries replay a key into the active screen
      so existing handlers run (also fixed the global `Tab` arm shadowing Ctrl-Tab).
- [x] **Chat session filmstrip + explicit rollback / variations** — a one-line scrubber
      of every generated frame this session (`Ctrl-←/→` select → shows in the image
      pane), **rollback/branch** (`Ctrl-B` makes the selected frame the live base,
      recovering its prompt+seed) and **variation** (`Ctrl-Y` re-renders its prompt at a
      new seed). Also in the command palette. (A real decoded-thumbnail row is a future
      nicety — the terminal-graphics cost of N inline images made a text scrubber the
      right call.)
- [x] **`@mention`** people / LoRAs inline in the Chat prompt — `@` opens a live
      completion popup (people `◆` + local LoRAs `★`, fuzzy-filtered); accepting a
      person leaves a readable `@name` token expanded to its prompt fragment at submit,
      accepting a LoRA applies it + strips the token. (Styles deferred — needs a styles
      registry the UI doesn't surface yet.)

## F — LoRA Hub (smaller follow-ups)

- [ ] **Download manager depth** — ≤2 concurrent, range-resume, explicit SHA-256 verify,
      version-update detection (progress + a `●` indicator + 1h search caching shipped).
- [ ] **LLM-assessment caching** (24h sidecar) + the two-stage HF pre-filter.
