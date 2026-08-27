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
