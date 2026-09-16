# ROADMAP — 6.32.0: finish the smysl improve loop

6.31.0 shipped `compile --improve` — the automatic **enhance → improve → emit** loop with the smysl
corpus as active tabu memory, writing the winning prompt back into the HJSON. Three threads were left
open at the seams. 6.32.0 closes them so the loop optimises the *real* artifact, never re-marches ground
a recompile re-opens, and feeds what it learns back into the prose-authoring loop.

Same discipline as the smysl-optimize arc: **fully additive**, the pure/deterministic half gated offline
with a stub, plain `compile` / `--no-enhance` stays **byte-identical**, no scene-specific defaults baked
into any code or LLM prompt (derive from the corpus). Gate = `cargo test --no-default-features --lib --
--test-threads=1 --skip worker_runs_jobs_in_order`. Version confirmed at cut, not now.

The three threads are independent; recommended order is **A → B → C** (each is a shippable increment).

---

## Thread A — LoRA-aware improve renders (score the REAL finish)

**The gap.** `improve_cmd` loads the resident render pipeline with `loras: Vec::new()`
(`src/cli/compile.rs`), so `--improve` scores a **plain t2i** render of the emitted prompt — not the
LoRA finish `plakat scenario` will actually produce. A prompt tuned against bare SDXL can be the wrong
prompt once the watercolour/persona LoRA is on. The aesthetic verdict must be measured on the finish the
user ships.

**Design.** The compile-level LoRA stack is scenario-**global** (`resolver::ResolvedGlobals.loras`,
emitted at the HJSON top level) and the activation token is already prepended into the compiled prompt
(`ResolvedScene.lora_trigger`, reserved from the budget). So there is one stack for the whole doc:

- In `improve_cmd`, read `doc.globals.loras` (+ `lora_scale`), parse with the existing
  `scenario::parse_resolved_loras`, and load the resident `t2i::Pipeline` with that stack instead of
  `Vec::new()`. The improve prompt already carries the trigger (compiled in), so nothing else changes in
  the render path.
- New flag **`--improve-plain`** to opt *out* (score bare t2i — faster, isolates prompt effects from the
  LoRA look). Default = use the scene's LoRAs, because that is what "improve this scene" means.
- Report the loaded stack in the up-front line so the cost/finish is never a surprise.

**Touchpoints.** `src/cli/compile.rs` (`improve_cmd` LoadRequest; reuse `scenario::parse_resolved_loras`
+ `prepend_trigger`). No change to the loop controller or the corpus.

**Risk / honesty.** (1) *Differing per-scene stacks* — compile LoRAs are uniform today, so one resident
load covers every scene; if per-scene LoRA overrides ever land in compile, reload per **distinct** stack
(cache by stack hash) or warn + fall back to globals. (2) *VRAM* — LoRAs add resident memory on top of
the pipeline + CLIP scorer; the existing OOM guard (`PLAKAT_OOM_GUARD_GB`) already covers the render, but
document the headroom. (3) The non-SD-family fallback (`sd35`/Flux, per-render `api::Generate`) must also
pass the LoRA stack.

**Verify.** Improve a scene with a real LoRA (`lora-trigger:` + a hosted style LoRA); confirm (a) the
kept candidates *visually* carry the LoRA look via `--keep-compiled-images`, (b) the score moves with the
LoRA on vs `--improve-plain`. Gate stays green (additive).

---

## Thread B — the enhancer consults the tabu corpus

**The gap.** The improve loop's proposer + controller refuse rejected moves, but the **enhance pass**
(`compile_one_scene`, prose → baseline prompt) does *not* read the corpus. So a fresh recompile can
re-introduce a phrasing the loop already proved worse, handing the loop a known-bad baseline to re-reject
(cheap, but wasteful, and it muddies "the corpus IS the process").

**Design.** Consult the corpus at enhance time and steer the enhancer away from spent phrasings:

- `smysl::rejected_phrasings(corpus)` → the `new` sides of moves recorded **reverted** (the phrasings that
  lost), distinct from the kept ones. (Reader over the `c/fix-*` bodies + their `reverted` findings —
  a sibling of `prior_fixes`.)
- An enhancer-facing hint (`smysl::enhance_avoid_hint`) appended to the enhancer *user* prompt: "avoid
  these phrasings — a prior aesthetic pass rejected them: …". Belt-and-suspenders like the fixer: the
  hint steers the LLM; a **deterministic post-enhance check** flags (and logs) any rejected phrase that
  still slipped through, so the result is never silently wrong. (No deterministic rewrite — the honest
  limit; a single re-enhance with a firmer hint is an optional retry.)
- Thread the corpus in via `CompileOpts` (new optional `corpus_text: Option<String>`, populated from
  `<stem>.smysl` in `compile_doc`/`analyze`). When absent (stdin, no corpus) → today's behaviour exactly.

**Touchpoints.** `src/smysl.rs` (`rejected_phrasings` + `enhance_avoid_hint`, gate-tested pure),
`src/compile/mod.rs` (`compile_one_scene` enhance step + `CompileOpts`).

**Risk / honesty.** (1) *LLM may ignore the hint* — same as the fixer; the deterministic flag is the
backstop. (2) **Cache correctness** — consulting the corpus makes the enhance output corpus-dependent, so
`--compile-cache` must fold the corpus hash into its key (or bypass cache when a corpus is consulted),
else a stale cached prompt hides the avoidance. This is the load-bearing correctness item for Thread B.
(3) Keep `--no-enhance` and corpus-absent compiles byte-identical.

**Verify.** Reject a move in `--improve`; recompile the same prose; confirm the enhancer's baseline no
longer contains the rejected phrase (or is flagged), and that a corpus-free compile is byte-identical to
6.31.0.

---

## Thread C — the critic reads the improve trajectory (close the loop back to prose)

**The gap.** `--analyze`/`--fix` reads the corpus's resolved/open **findings** (Phase B critic-consult)
but not the **aesthetic wins** the improve loop measured. The machine's search discovers what raises the
score; the human authoring loop never sees it. Close the loop: surface the kept wins as prose
suggestions.

**Design.**

- `smysl::improve_wins(corpus)` → the **kept** moves (`old → new` that beat the baseline by > min-gain),
  with their rank delta, read from the improve corpus.
- Extend `analyze_prose`'s PRIOR POLISH HISTORY with a critic-facing note: "these edits *measurably*
  improved the render (aesthetic +Δ) — consider folding them into the prose." Advisory by default.
- `--fix` may offer to apply a kept win **to the prose** when the phrase maps cleanly back (a new fix
  class grounded in `SourceKind::Metric` aesthetic evidence, not a heuristic). When it doesn't map (the
  win lives on the enhanced prompt, not the authored sentence), it stays a suggestion — stated honestly.

**Touchpoints.** `src/smysl.rs` (`improve_wins` reader + hint), `src/compile/mod.rs` (`analyze_prose`
hint; `apply_fixes` optional win-application), reusing the Phase B findings-hint plumbing.

**Risk / honesty.** A prompt-space win doesn't always translate to prose (enhancer rewrites). Keep it
**advisory** unless the mapping is exact; never auto-apply a win the prose can't express. This thread is
the "reach" — land A and B first; C is only worth it once the corpus reliably carries wins (it does, from
6.31.0).

---

## Build order & exit criteria

1. **A (LoRA-aware improve)** — smallest, immediate value; the score finally matches the shipped finish.
   Exit: a LoRA scene improves against its real look; `--improve-plain` opts out; gate green.
2. **B (enhancer tabu)** — closes the recompile re-introduction gap; the cache-key fix is the crux.
   Exit: a rejected phrase does not reappear on recompile; corpus-free compile byte-identical; gate green.
3. **C (critic reads wins)** — the loop closes back to prose. Exit: `--analyze` surfaces a measured win as
   a suggestion; `--fix` applies one only when it maps to prose; gate green.

Each thread is its own commit(s), gate-green and live-proven on Metal before moving on — same cadence as
the 6.31.0 arc. `plakat serve` / JSON-RPC remains **deferred** to the user's own RFC.
