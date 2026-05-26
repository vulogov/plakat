# RFC: Bund scripting language for plakat (v0.21 big swing)

**Status:** Decisions locked (2026-05-25). Ready to convert to phases.
**Author:** v0.21 cycle research.
**Date:** 2026-05-25.
**Reference repos:** `vulogov/Bund`, `vulogov/bundcore`,
`vulogov/rust_multistackvm`, `vulogov/blackInkhaven` (canonical
embedding example).

---

## 1. TL;DR

Bund is a **synchronous, concatenative, stack-based** scripting
language (Forth/Factor lineage) with a Rust embedding API. It is
viable for plakat scripting but the integration imposes three
non-trivial constraints we have to design around:

- **Host functions must be plain `fn` pointers, not closures.** All
  state-sharing happens through process-global `OnceLock` /
  `lazy_static!` singletons. Plakat already runs as a one-shot CLI so
  this is acceptable, but it shapes the API.
- **The VM is fully synchronous.** Every plakat pipeline call is
  `async fn`. Each Bund host word that touches a pipeline has to
  `block_in_place` / `block_on` internally — workable but requires
  the script eval to run on a tokio worker thread (we control this).
- **No sandboxing at the embed layer.** Bund's `--noio` flag is
  bound to its CLI; the `bundcore::Bund::new()` path registers the
  full stdlib (incl. `sudo`, `duct_sh`, `reqwest`, `zenoh`). The
  fix: skip `Bund::new()`, build our own `VM`, register only what
  we want. blackInkhaven does this via a `policy.rs` upsert pattern.

**Recommended v0.21 scope:** ship a `plakat run SCRIPT.bund`
subcommand exposing a tight `plakat.*` word namespace covering
generate / img2img / upscale / save (four primitives, see §6). Defer
portrait, stylize, outpaint, ControlNet, LoRA, scenarios to v0.22.

**Recommended integration crate:** depend on `bundcore = "=0.7.0"` +
`bund_language_parser = "=0.14.0"` (the embedder path — the `bund`
crate itself has no `[lib]` and is a binary-only package). Pin
exact versions; the author uses wildcard requirements internally
which makes the dep tree float.

---

## 2. What Bund is

**Concatenative, stack-based, Forth-like.** Postfix syntax, a
"stack-of-stacks" ring + a named Workbench stack for intermediates,
`:name { ... }` lambda definition with `register` to name them,
first-class symbols (`:foo`), lists (`[ a b c ]`), and a curry/partial-
application mechanism. Not a Lisp; not Pythonic. Users will be
writing RPN.

Minimal example:
```bund
"Hello World!" println
```

A representative real script (from blackInkhaven's docs):
```bund
:hook.summarize ( -- )
  ink.paragraph.text                                ( -- body )
  "Summarize the following paragraph in 50 words:\n" swap concat
  ink.ai.send_blocking                              ( -- summary )
  "summary.md" swap ink.fs.write                    ( -- )
;
"Summarize open paragraph?" "summarize" ink.input
```

Forth users will be at home. Everyone else faces a learning curve.
This is the single largest user-facing decision and it is **not
reversible**: once shipped, the syntax is the contract.

## 3. Embedding constraints (the parts that shape the design)

### 3.1. The `bund` crate has no `[lib]` target

`crates.io/crates/bund` is binary-only. Embedding goes through
**`bundcore`** (`Bund` struct, `eval()`, `run()`) +
**`rust_multistackvm`** (the `VM`, `register_inline`). The `bund`
binary's stdlib (filesystem, network, AI words) is not exposed
through `bundcore` — embedders register their own.

### 3.2. Host-function signature is a bare `fn` pointer

```rust
pub type VMInlineFn = fn(&mut VM) -> Result<&mut VM, easy_error::Error>;
vm.register_inline("plakat.generate".into(), plakat_generate_fn)?;
```

Not `Box<dyn Fn(...)>`. Closures that capture context **do not
compile**. Every host fn that needs plakat state reaches into a
process global. blackInkhaven uses `OnceLock<RwLock<Bund>>` for the
VM itself and `OnceLock<Store>` for the project handle; we'll do the
analogous thing with `OnceLock<RwLock<ScriptCtx>>`.

This is acceptable for plakat (one-shot CLI, single eval per process
invocation) but means **we cannot run two scripts in parallel within
one process**. That's a fine trade.

### 3.3. Bund is synchronous; plakat pipelines are async

`bund.eval()` is sync. `VMInlineFn` is sync. Plakat's
`pipelines::flux::Pipeline::generate(&mut self, &Request)` is
`async`. Every Rust→async bridge inside a host fn looks like:

```rust
fn plakat_generate(vm: &mut VM) -> Result<&mut VM, easy_error::Error> {
    let prompt = pull_string(vm)?;
    let handle = tokio::runtime::Handle::current();
    let result = tokio::task::block_in_place(|| {
        handle.block_on(async {
            with_ctx_async(|ctx| ctx.generate(&prompt)).await
        })
    });
    push_string(vm, result.map_err(to_bund_err)?);
    Ok(vm)
}
```

This requires `bund.eval()` to be called from a tokio worker thread
(multi-threaded runtime). Easy: `plakat run` is dispatched from
`cli::dispatch` which is already async, and we wrap the eval in
`spawn_blocking` (or just call it directly from the async fn body
and rely on `block_in_place`).

### 3.4. No sandboxing in the embed path

`Bund::new()` calls `init_stdlib` which registers the full word set
including filesystem, network, shell, sudo. The `--noio` flag is
parsed at the **CLI layer** of the bund binary, not in the embedder
API. Two options for plakat:

1. **Build our own VM.** `VM::new()` registers only the
   multistackvm primitives (arithmetic, stack ops, lambdas, control
   flow). Then `vm.register_inline(...)` only the plakat words.
   No filesystem, no network, no shell — by construction.
2. **Wrap `Bund` + apply a policy.** blackInkhaven's `policy.rs`
   re-registers denied stdlib words as `_disabled` stubs. More
   features available to scripts (string manipulation, math libs),
   but every new Bund version potentially adds new dangerous words
   we'd need to track.

**Recommendation:** Option 1 for v0.21 (small, auditable surface).
Revisit if scripts need stdlib features that aren't dangerous.

### 3.5. Dependency footprint

`bund` (the binary) pulls ~110 direct deps including `duckdb`,
`polars`, `arrow`, `augurs`, `lingua`, `rustface` (git fork),
`zenoh`. **`bundcore` is much lighter** — that's the crate we
actually want. Verify the transitive dep count once we pin
`bundcore = "=0.7.0"`; if it's still heavy, that informs whether we
ship Bund as a feature flag (`--features scripting`) rather than
default-on.

---

## 4. State + lifecycle plan

### 4.1. `ScriptCtx` singleton

```rust
// src/scripting/ctx.rs (new module)
struct ScriptCtx {
    device: candle_core::Device,
    out_dir: PathBuf,
    /// Lazily-loaded pipelines; built on first use, reused after.
    /// Different `--model` strings can request different pipelines;
    /// the script can hold one of each family loaded.
    loaded: HashMap<String, LoadedPipeline>,
    /// What the most recent host fn produced; consumed by `plakat.save`.
    last_image: Option<DynamicImage>,
    /// Output path history (for `plakat.last_path`).
    last_paths: Vec<PathBuf>,
}

enum LoadedPipeline {
    T2I(pipelines::t2i::Pipeline),
    Flux(pipelines::flux::Pipeline),
    Sd3(pipelines::sd3::Pipeline),
}

static CTX: OnceLock<RwLock<ScriptCtx>> = OnceLock::new();
```

`ScriptCtx::init(device, out_dir)` is called once at the top of
`plakat run`. Host fns reach state via `with_ctx(|ctx| ...)` and
`with_ctx_mut(|ctx| ...)`.

### 4.2. Script lifecycle: load-evaluate-exit

```rust
pub async fn run(args: RunArgs, device: Device) -> Result<()> {
    let source = std::fs::read_to_string(&args.script)?;
    ScriptCtx::init(device, args.out.clone())?;
    let mut vm = build_plakat_vm()?;          // VM + plakat host words
    let mut bund = bundcore::Bund::from_vm(vm); // or equivalent
    tokio::task::block_in_place(|| bund.eval(source))
        .map_err(|e| anyhow!("script error: {e}"))?;
    Ok(())
}
```

One process invocation = one script eval. No REPL in v0.21 (defer).
No script-from-CLI-arg in v0.21 (file only, defer).

### 4.3. Result handling

The script's `top-of-stack` after eval is **discarded** by default —
plakat scripts produce side effects (image files on disk), not
return values. Consider an `-x EXPR` flag in a future cycle for "eval
and print top" workflows.

---

## 5. Proposed `plakat.*` word namespace

Mirror blackInkhaven's `ink.*` convention. All host words prefixed
`plakat.` to keep the global word table organized. **First arg
listed is bottom-of-stack** (last pushed = top, so `pull` order is
reversed).

### 5.1. v0.21 MVP (seven words — decision #4)

```
plakat.load        ( model-alias -- )
   Lazily load a pipeline by alias (sd15, sdxl, flux-dev, etc).
   Idempotent: second call with the same alias is a no-op.
   Without an explicit load, plakat.generate auto-loads on first use.

plakat.generate    ( prompt -- image-handle )
   Run text→image with current config + the loaded model. Pushes a
   handle representing the rendered image (kept in ScriptCtx; the
   handle is the integer index into `last_paths` once saved, or a
   sentinel before save).

plakat.img2img     ( prompt input-path -- image-handle )
   Re-imagine an existing image. Pops path first (top), then prompt.
   Same handle contract.

plakat.portrait    ( prompt photo-path -- image-handle )
   Identity-preserving portrait via IP-Adapter / FaceID. Pops photo
   path first, then prompt. Reuses ScriptCtx's loaded model when
   compatible (SD 1.5 / SDXL). FaceID + multi-photo blends deferred
   to v0.22 (single reference photo for the MVP).

plakat.upscale     ( image-handle scale-int -- image-handle )
   Apply Lanczos upscale. scale-int = 2 or 4. ML upscaling
   (Real-ESRGAN) deferred to v0.22.

plakat.save        ( image-handle path-string -- )
   Write the handle's image to path. The image stays in
   ScriptCtx.last_image so it can be re-used (saved twice, upscaled,
   etc).

plakat.config.set  ( value key-string -- )
   Set a config knob. Knobs: "steps", "guidance", "seed", "width",
   "height", "negative", "scheduler". Mirrors `plakat generate`
   flags. Persistent across calls within one script.
```

### 5.2. Example v0.21 script

```bund
# Set up
:sdxl plakat.load
:steps 40 plakat.config.set
:guidance 7.5 plakat.config.set
:seed 42 plakat.config.set
:size "1024x1024" plakat.config.set

# Generate three images at the same seed family
"a fox in a forest, painterly"  plakat.generate  "out/fox.png"  plakat.save
"a cat in a forest, painterly"  plakat.generate  "out/cat.png"  plakat.save
"a deer in a forest, painterly" plakat.generate  "out/deer.png" plakat.save

# Re-render the fox as a 4K poster via img2img
"a fox in a forest, painterly, detailed fur, intricate lighting"
"out/fox.png"
plakat.img2img
  2 plakat.upscale
  "out/fox-2k.png" plakat.save
```

### 5.3. Deferred words (v0.22+)

- `plakat.stylize` (separate IP-Adapter wiring)
- `plakat.outpaint` (thin wrapper over img2img anyway; trivial once
  img2img + mask exists)
- `plakat.upscale.ml` (Real-ESRGAN — adds model-download surface)
- `plakat.portrait.multi-photo` + FaceID (deferred from the v0.21
  portrait MVP — single reference photo only this cycle)
- `plakat.lora.add / remove / scale` (LoRA stacking surface is large)
- `plakat.controlnet.add` (each variant has its own conditioning)
- `plakat.metadata.read / write` (JSON sidecar I/O)
- `plakat.scenario.*` (HJSON scenarios already work; don't duplicate)

---

## 6. Recommended directory layout

```
src/
├── cli/
│   └── run.rs                  # `plakat run SCRIPT` subcommand
├── scripting/
│   ├── mod.rs                  # ScriptCtx singleton + entry point
│   ├── ctx.rs                  # ScriptCtx struct + with_ctx helpers
│   ├── helpers.rs              # pull_string, push_string, require_depth, to_bund_err
│   └── words/
│       ├── mod.rs              # register_plakat_words(&mut VM)
│       ├── load.rs             # plakat.load
│       ├── generate.rs         # plakat.generate
│       ├── img2img.rs          # plakat.img2img
│       ├── upscale.rs          # plakat.upscale
│       ├── save.rs             # plakat.save
│       └── config.rs           # plakat.config.set
└── lib.rs                      # pub mod scripting;
```

Pattern lifted from blackInkhaven's `src/scripting/` layout. Helpers
shape (`pull`, `push`, `require_depth`, `value_to_*`,
`anyhow inside / easy_error at boundary` with `to_bund_err`) is
generic Bund-integration boilerplate worth copying verbatim.

---

## 7. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Bund's solo-maintainer model (9 GH stars, no Rust unit tests, wildcard version pins) | High — engine bugs surface as plakat bugs with limited upstream response | Pin exact versions (`=0.7.0`); cache the Cargo.lock in CI; consider a fork if blockers hit |
| Dependency explosion if `bundcore` transitively pulls bund's stdlib deps | Medium — could double plakat's compile time / binary size | Measure first. If bad, gate behind `--features scripting`. |
| Forth/RPN syntax is a steep learning curve for users coming from Python/JS | Medium — adoption risk | Publish a `Documentation/Tutorials/SCRIPTING_TUTORIAL.md` with 10+ runnable examples; lean on the existing scenario HJSON as the "you don't have to use this" alternative |
| Async/sync bridge fragility (`block_in_place` requires multi-threaded runtime) | Low — manageable, but a test must pin it | Smoke-test `plakat run` in CI on a script that generates one image |
| Bund has no built-in script-level type checking — every word does runtime pop-and-check | Low — Bund's idiom; users learn fast | `require_depth` + `value_to_*` helpers raise meaningful errors |
| `ScriptCtx` singleton prevents in-process parallelism | Low — plakat CLI is one-shot anyway | Document as an explicit non-goal; scenarios remain the parallelism story |

---

## 8. Decisions (locked 2026-05-25)

| # | Question | Decision | Notes |
|---|---|---|---|
| 1 | Embed crate | **`bundcore = "=0.7.0"`** | Pinned exact; carries the parser + `eval()` glue. Avoids rebuilding what bundcore already wraps. |
| 2 | Stdlib strategy | **Build our own VM, no Bund stdlib** | Skip `Bund::new()`; use `VM::new()` (multistackvm primitives only) + register only `plakat.*`. Filesystem / network / shell / sudo can't reach user scripts by construction. |
| 3 | Subcommand | **`plakat run SCRIPT.bund`** | Reserves `plakat script` for a future inspect/list subcommand. |
| 4 | MVP word set | **Seven words** | The six from §5.1 plus `plakat.portrait`. Portrait is plakat's identity-preservation showcase + a common script use case; absorb the IP-Adapter / FaceID complexity now rather than tacking it on later. |
| 5 | Build gating | **Always default-on** | Every plakat install gets scripting. Dep-tree cost is the price; revisit if measurement (phase 1) shows it's catastrophic. |
| 6 | REPL | **Ship `plakat run --repl` in v0.21** | One flag on the same subcommand; doubles the exploration value. Adds ~1 session of readline / prompt / history work. |
| 7 | Extension | **`.bund`** | Matches the engine + matches blackInkhaven; benefits from any editor tooling that emerges around Bund as a language. |

These seven decisions are the **contract for the implementation
phases** in §9 — changing one after a phase ships is a breaking
change to user scripts.

---

## 9. Phase plan (revised post-decisions)

| Phase | Deliverable | Estimate |
|---|---|---|
| **1** | `bundcore = "=0.7.0"` integration smoke. One host word: `plakat.echo`. Verifies the async bridge (`block_in_place` + `Handle::current().block_on`), the `OnceLock<RwLock<ScriptCtx>>` singleton, the `plakat run SCRIPT.bund` subcommand wiring, and the `VM::new()`-not-`Bund::new()` stdlib avoidance. **Measure transitive dep + compile-time delta here** to validate the default-on decision (revisit if catastrophic). | ~1 session |
| **2** | `plakat.load` + `plakat.generate` + `plakat.save`. End-to-end: load SD 1.5 → generate → write PNG. The minimum publishable image-producing script. | ~1 session |
| **3** | `plakat.config.set` (steps, guidance, seed, width, height, negative, scheduler). Persistent across calls within one script. | ~0.5 session |
| **4** | `plakat.img2img` (incl. pulling input from disk + the `ScriptCtx.last_image` handle-reuse contract so `generate → img2img` chains compose). | ~0.5 session |
| **5** | `plakat.portrait` (single-photo IP-Adapter; FaceID + multi-photo deferred per §5.3). Largest single-phase risk — IP-Adapter wiring is finickier than img2img. | ~1 session |
| **6** | `plakat.upscale` (Lanczos x2/x4 only; ML upscaler deferred). | ~0.5 session |
| **7** | `plakat run --repl`: prompt + history (`rustyline`?) + per-line eval against the same `ScriptCtx`. Shares 100% of the host-word surface with file mode. | ~1 session |
| **8** | `Documentation/Tutorials/SCRIPTING_TUTORIAL.md` + composition tests (real script that loads SD 1.5, generates 3 images, upscales the best one). Tutorial gets indexed in the Tutorials/README.md. | ~1 session |

**Total: ~6-6.5 sessions** for the v0.21 MVP. Up from the original
~4-5 estimate because (a) portrait adds a phase (decision #4 traded
upscale-shrink for portrait-add) and (b) REPL adds a phase
(decision #6). Still firmly in "big-swing" territory but smaller
than v0.20's Kontext+tiled or Flux animate.

### Phase ordering rationale

- Phase 1 first: validates the entire integration shape with a
  single host word before we commit to any pipeline plumbing.
- Phases 2-4 build the t2i / img2img happy path because they share
  the `pipelines::t2i` backbone — portrait gets piggybacked at
  phase 5 (also SD-family on the same backbone).
- REPL (phase 7) deliberately late so it inherits a stable host-word
  surface; readline plumbing is the easiest thing to add last.
- Tutorial (phase 8) is dead last so it reflects what actually
  shipped, not what we planned in this RFC.

---

## Appendix A: Integration boilerplate worth copying from blackInkhaven

- `OnceLock<RwLock<Bund>>` singleton + lazy init pattern.
- The `pull`/`push`/`require_depth`/`value_to_*` helper layer.
- `anyhow inside / easy_error at boundary` error pattern with
  `to_bund_err` adapter.
- Print buffer override: register `print` / `println` words that
  drain into a thread-local `RefCell<String>` rather than directly
  writing to stdout, so output is captured cleanly.
- Policy mechanism (`policy.rs`) for re-registering stdlib words
  with disabled stubs — useful even with Option 1 (build-our-own-VM)
  if Bund's primitives later add anything we want to block.

## Appendix B: Things to NOT copy from blackInkhaven

- The `ink.*` word namespace — domain-specific, irrelevant.
- The `ACTIVE_APP` raw-pointer trick — only safe because their TUI
  is single-threaded. plakat's async dispatch breaks this; use
  `OnceLock<RwLock<...>>` instead.
- The "load every Script node from DuckDB" auto-eval — they have
  store-resident scripts, plakat doesn't.
- The hook recursion-guard machinery (`hooks.rs`) — only relevant if
  plakat exposes lifecycle events (it doesn't, in v0.21).
