# RFC FACESWAP-4 — ecosystem parity: `api::FaceSwap` + scenario `type: faceswap` (6.22.0)

**Status:** SHIPPED (6.22.0). The deferred tail of FACESWAP-3 S3. **All shipped:** P1 `api::FaceSwap`
builder · P2 scenario `type: faceswap` (dry-run-verified). Face-swap is now reachable from CLI + Bund
word + api + scenario — full ecosystem parity. Face-swap already has the CLI verb
(`plakat faceswap`) and the Bund word (`plakat.faceswap`); this finishes parity so it is reachable from
the **library** and from **scenario pipelines**, like every other verb.

## What ships

### P1 — `api::FaceSwap` builder
A library facade mirroring [`api::Naturalize`] / [`api::Upscale`]:
```rust
use plakat::api::FaceSwap;
let img = FaceSwap::new("scene.png", "alice.png").face(0).run().await?;
img.save("out.png")?;
```
`new(scene, source)` → `.face(n)` / `.device(spec)` → `run() -> Image`. Loads the engine, detects
(largest-first), embeds the source, colour-matched `swap_into` on the selected face.

### P2 — scenario `type: faceswap`
A `type: faceswap` task in a scenario: `scene` + `source` (+ optional `face`) → writes the swapped image
as the task output. Reuses the CLI's per-image swap. Slots into the existing task-dispatch enum + the
per-task body block, so a swap can be one step of a batch pipeline alongside `generate` / `naturalize` /
`texture` tasks.

### P3 — docs + cut 6.22.0
README + the faceswap RFCs' status; corpus. Cut 6.22.0 (bump Cargo+lock, gate `--test-threads=1`,
turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
Same engine limits as FACESWAP-2/3 (128² identity transfer; small/occluded/profile faces may miss; weights
non-commercial). The scenario/api paths swap a single face (`face` index) — the CLI keeps the richer
multi-source / recognition / video surface.

## Sequencing
**P1** api builder → **P2** scenario task → **P3** cut. Independent.

## P4 (folded in) — `plakat compile` improvements
- **C1 faceswap task in compile** — `declares_faceswap_task` (parser) + Scene fields (resolver) + emit
  block (emitter): a `type: faceswap` prose block (with `faceswap-scene`/`faceswap-source`/`faceswap-face`)
  compiles to a scenario faceswap task. Completes faceswap parity into compile.
- **C2 validate compiled output** — `scenario::validate_hjson` (deserialise + known task types) runs on the
  emitted scenario before writing → guaranteed loadable, not just well-formed.
- **C3 decompile spec-tasks** — `--decompile` preserves the task `type:` for spec-tasks and fully
  round-trips a faceswap block (`type` + `faceswap-*`), no bogus prompt placeholder.
- **C4 `@include`** — `parser::expand_includes` inlines `@include <path>` lines (relative, recursive,
  depth-guarded) before parse, so prose sets split across files. Round-trip live-proven.

## P5 (folded in) — more `plakat compile` improvements (D1–D4)
- **D1 remaining spec-tasks** — `fractal` now compiles (`fractal-spec`/`fractal-kind`/`fractal-palette` →
  `fractal: {…}`); `animate`/`animatediff` already worked via the generic `type:` emit (confirmed).
  **multiperson deferred** — its nested `people` list doesn't map to flat `key: value` prose (needs a
  spec-file mechanism `MultipersonTaskSpec` lacks).
- **D2 richer `--lint`** — flags duplicate task names (they collide as ids) and repeated non-repeatable
  commands (`seed:` twice), on top of the existing unknown-command check.
- **D3 `compile --watch`** — re-compile on the input file's mtime change (poll-based, no deps; Ctrl-C to
  stop; file input only).
- **D4 lossless decompile** — `--decompile` now round-trips the compiler-authored spec directives for
  `texture` (from/seamless), `product`/`comic` (spec-file), and `fractal` (spec/kind/palette).
