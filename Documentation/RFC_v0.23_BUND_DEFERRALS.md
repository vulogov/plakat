# RFC v0.23 — Bund scripting deferrals (research draft)

**Status:** research draft — open questions in §6 need user
sign-off before phase 1 can land.

**Predecessors:**
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) —
  the 7-word MVP (foundations).
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) —
  the full sweep (28 words; pipeline cache; all three families).

## 1. TL;DR

v0.22 shipped 28 host words but left seven items annotated
"deferred to v0.23" inside the source. Six of those seven are
blocked on the same root cause: the script cache holds a
`portrait::Pipeline` (good for IP-Adapter, bad for the SDXL
refiner UNet slot and the `clip_skip`/encode plumbing). v0.23
unblocks them with a cache architecture change, then fills in
the deferred wirings.

By the end of the cycle:

- The SDXL refiner UNet actually loads under
  `plakat.refiner.enable` (today it's a stateful flag that bails).
- `clip_skip` is no longer a declared-but-no-op knob.
- `plakat.style.*` ships (`apply` / `detect` / `clear` / `list`).
- `plakat.inpaint` ships (the mask path arg that `mask_feather` +
  `mask_invert` have been waiting for).
- Flux + SD3 ControlNet work from scripts (today they bail with
  a "v0.23" pointer when the stack is non-empty).

## 2. Why this is the v0.23 cycle

Three reasons:

1. **It's the natural follow-up.** v0.22 shipped seven deferral
   stubs *intentionally* — toggles + config keys whose surface
   we wanted to be stable for v0.23 so user scripts written
   against the v0.22 surface keep working. Honouring those
   stubs in v0.23 closes a debt we explicitly took on.

2. **One refactor unblocks four items.** The cache switch
   (portrait → t2i for the SD-family slot, or some variant — see
   §5) is the load-bearing change. Once it lands:
   refiner UNet + `clip_skip` + `plakat.style.*` + the t2i-side
   of `plakat.inpaint` all wire up as routine integration work.

3. **Smaller cycle than v0.22.** v0.22 was the biggest scripting
   cycle ever (12 phases). v0.23 is ~7 phases. Lets us land
   AnimateDiff / FaceID / SD3 animate as the v0.24 big swing
   without v0.23 also being a marathon.

## 3. Architectural constraints we keep

Same five from the v0.22 RFC §3:

1. **Built-our-own VM.** No filesystem / network / shell access
   reachable from scripts.
2. **Singleton context.** One script per process; one process
   per invocation.
3. **Async bridge via `block_in_place`.** Same as v0.21–v0.22.
4. **No backwards-compat hacks.** v0.22's relaxed-compat
   decision (RFC §8 #7) carries forward. If v0.22 scripts need
   to migrate, document.
5. **Standard library stays excluded.** Bund's stdlib remains
   un-imported; only `plakat.*` words sit on top of the VM
   primitives.

## 4. The deferred deliverables

Seven items, surveyed from `v0.23` markers in the source
(2026-05-26):

| # | Item | Today's state | What unblocks it |
|---|---|---|---|
| 1 | SDXL refiner UNet load | `plakat.refiner.enable` toggles a flag; `plakat.generate` bails | Cache switch (t2i::Pipeline has the refiner slot) |
| 2 | `clip_skip` wiring | Config knob declared, no-op at generate time | Cache switch (t2i::Pipeline owns `encode_*`) |
| 3 | `plakat.style.*` namespace | Only `style_strength` config key ships | New namespace + style-catalog integration |
| 4 | `plakat.inpaint` (mask path arg) | `mask_feather` + `mask_invert` declared, no mask path | New host word |
| 5 | Flux ControlNet from scripts | `plakat.generate` bails when `ctx.controlnets` non-empty | Load-time wiring on Flux pipeline |
| 6 | SD3 ControlNet from scripts | Same bail as Flux | Load-time wiring on SD3 pipeline |
| 7 | `wildcard_dir` plumbing through `generate_one` | Knob is declared + read at generate time; the *expansion* is wired (phase 11) | **Already done in v0.22 — strike from list** |

Item 7 is already wired (v0.22 phase 11 added the
`expand_prompt` call). The v0.23 deferral comment in
`config.rs:242` is stale; phase 0 will sweep these comments.

That leaves **six items** in v0.23 scope.

## 5. The cache architecture decision (the big one)

### Today

```rust
enum LoadedPipeline {
    SdFamily(portrait::Pipeline),  // ← IP-Adapter slot lives here
    Flux(flux::Pipeline),
    Sd3(sd3::Pipeline),
}
```

`portrait::Pipeline` has:
- `core()` returning `Arc<SdCore>` (same as t2i::Pipeline)
- Identity encoder (IP-Adapter-Plus-Face) slot
- No refiner UNet slot
- No `encode_*` methods that honour `clip_skip`

`t2i::Pipeline` has:
- `core()` returning `Arc<SdCore>` (parallel)
- Refiner UNet slot (`use_refiner: bool` on `LoadRequest`)
- `encode_*` methods that honour `clip_skip`
- No identity encoder slot

### Options

**A. Two slots per SD-family invocation.**

```rust
enum LoadedPipeline {
    SdT2i(t2i::Pipeline),          // generate / img2img / refiner / clip_skip
    SdPortrait(portrait::Pipeline), // portrait (IP-Adapter)
    Flux(flux::Pipeline),
    Sd3(sd3::Pipeline),
}
```

`plakat.generate` / `plakat.img2img` / `plakat.refiner.*` use the
`SdT2i` slot. `plakat.portrait` uses `SdPortrait`. They can both
be loaded for the same alias; family-changes drop both. Either
slot's `core()` is shared with `SdCore`-reusing post-process
words (adetailer / hires / artefact-blend) — those work
identically against either pipeline type.

- Pro: minimal refactor. Each pipeline keeps its own
  specialisation.
- Con: SDXL gets loaded twice in scripts that mix `plakat.portrait`
  and `plakat.generate` on the same alias. Mitigation: both
  pipelines share the same `Arc<SdCore>` (weights + tokenizers are
  refcounted), so only the slot-specific extras (refiner UNet
  for t2i, IP-Adapter encoder for portrait) duplicate.

**B. Conditional slot.**

```rust
enum LoadedPipeline {
    SdFamily(SdFamilyVariant),
    // ...
}
enum SdFamilyVariant {
    T2i(t2i::Pipeline),
    Portrait(portrait::Pipeline),
}
```

One slot at a time. `plakat.portrait` after `plakat.generate`
rebuilds the slot. Worst-case ergonomics — scripts that interleave
generate + portrait pay the SdCore reload each time.

- Pro: keeps the cache as a single slot.
- Con: degrades the v0.22 cache behaviour for portrait-interleaving
  scripts.

**C. Unify into a "super-pipeline".**

Extend `t2i::Pipeline` to hold the IP-Adapter slot, retire
`portrait::Pipeline` from the script cache (the type might
still exist for the CLI's `plakat portrait` subcommand).

- Pro: long-term cleanest.
- Con: largest refactor. Risk of regressing `cli::portrait`'s
  existing behaviour.

### Recommendation

**Option A.** Minimal refactor, no regression risk, accepts a
modest weight-duplication cost (refiner UNet + IP-Adapter encoder
are small relative to the shared SdCore). Documented as the
trade-off.

Option A also keeps the door open for a later v0.24+ Option C
refactor if we accumulate more reasons to unify.

## 6. Open questions (lock these before phase 1)

### Q1: Cache architecture (§5)

**Lock:** A / B / C. Recommend A.

### Q2: `plakat.style.*` surface

The CLI ships `--style ID` and `--style-ref PATH`. Proposed
script surface:

```bund
"poster-bold" plakat.style.apply         // by id
"./ref.jpg"   plakat.style.detect        // detect from photo
plakat.style.list                        // ( -- s_1 ... s_n n )
plakat.style.clear                       // forget the active style
0.7 "style_strength" plakat.config.set   // already declared in v0.22
```

Cache invalidation: mutations drop the cached pipeline (same as
LoRA), since the style catalog applies LoRAs at load time.

**Lock:** confirm the four-word shape vs. a thinner version
(e.g. just `apply` + `clear`). Recommend shipping all four for
REPL ergonomics.

### Q3: `plakat.inpaint` stack effect

Two viable shapes:

- **Separate word.** `plakat.inpaint ( prompt input mask -- handle )`.
  Mirrors the CLI's `--mask`. Reads `mask_feather` + `mask_invert`
  from config (already declared).
- **Extend `plakat.img2img`.** Add a config key `mask` (string
  path); when set, `plakat.img2img` runs inpaint instead of
  whole-image img2img.

**Lock:** separate word vs. config-key extension. Recommend
**separate word** — matches the explicit-is-better Bund style.

### Q4: Flux + SD3 ControlNet — one phase or two?

Both need load-time wiring; both are independent. We could:

- Ship them in one combined phase (~1 session).
- Ship Flux first, SD3 second (~0.5 session each).

**Lock:** combined vs. split. Recommend **split** — keeps each
phase small + reviewable.

### Q5: v0.22 → v0.23 compat

v0.22 shipped surface stubs (`plakat.refiner.enable`,
`adetailer_*` config keys before the post-process was wired).
Some v0.22 scripts may already use the toggle-only refiner.
After v0.23, `plakat.refiner.enable` *actually loads the refiner
UNet*. Implication: v0.22 scripts that did `plakat.refiner.enable`
"to be safe" suddenly pay the 6 GB SDXL-refiner download +
~30s extra load time.

Two options:
- **Quiet upgrade.** v0.22 scripts get the refiner UNet on next
  generate. No warning. Documented in release notes.
- **Opt-in second toggle.** Add `plakat.refiner.auto_load` (or
  a config key). Default off in v0.23 to preserve v0.22
  behaviour; users opt in.

**Lock:** quiet vs. opt-in. Recommend **quiet upgrade** — the
v0.22 docs already said the toggle "is shipped today so the
surface is stable for when the cache switches to t2i::Pipeline,"
which is exactly what happens.

### Q6: Phase ordering — depth-first or breadth-first?

Two viable orderings:

- **Depth-first.** Cache switch → refiner UNet → clip_skip →
  plakat.style.* → plakat.inpaint → Flux CN → SD3 CN. Each
  phase only unblocks the next.
- **Breadth-first.** Cache switch → all SD-family unblocks
  (refiner, clip_skip, style, inpaint) in one bigger phase →
  Flux CN → SD3 CN. Fewer phases, more per-phase scope.

**Lock:** depth-first vs. breadth-first. Recommend **depth-first**
(matches v0.22's per-namespace cadence). 7 phases each at ~0.5–1
session.

## 7. Phase plan (locked post-decision)

Assuming the recommendations above are accepted:

| # | Deliverable | Est. |
|---|---|---|
| 0 | Sweep stale v0.23 deferral comments; phase-0 hygiene + RFC commit | ~0.25 session |
| 1 | Cache architecture: add `SdT2i(t2i::Pipeline)` variant; route `plakat.generate` / `plakat.img2img` / `plakat.refiner.*` through it; `plakat.portrait` keeps `SdPortrait` | ~1 session |
| 2 | SDXL refiner UNet: wire `use_refiner` + `refiner_frac` through `get_or_load_sd_t2i`; drop the v0.23 bail | ~0.5 session |
| 3 | `clip_skip` wiring: thread through `t2i::Pipeline.encode_*` at generate-request time | ~0.5 session |
| 4 | `plakat.style.*` namespace: `apply` / `detect` / `clear` / `list` + cache invalidation | ~1 session |
| 5 | `plakat.inpaint`: new host word + `mask` arg + family bails for Flux/SD3 (or wire through Flux-fill if scope allows) | ~1 session |
| 6 | Flux ControlNet: load-time wiring; `ctx.controlnets` invalidates Flux cache; drop the v0.23 bail | ~1 session |
| 7 | SD3 ControlNet: same as phase 6 for SD3 pipeline | ~1 session |
| 8 | Docs + composition tests (mirrors v0.22 phase 12) | ~0.5 session |

**Total estimate:** 6.5–7.5 sessions. Smaller cycle than v0.22.

### Phase-ordering rationale

- **Phase 1 first.** Foundation; every later phase reads on it.
- **Phase 2 + 3 right after.** Lowest-friction unblocks; both are
  ~30-line wirings once the cache lands.
- **Phase 4 mid-cycle.** Style catalog is bigger than a single
  wire-up (needs the catalog runtime + LoRA-merge invalidation
  pattern). Lands when phase-1 confidence is high.
- **Phase 5 mid-late.** `plakat.inpaint` is genuinely new
  surface, not a wire-up.
- **Phase 6 + 7 paired but separate.** Same shape on two pipelines.

## 8. What's NOT in v0.23 (explicitly deferred to v0.24+)

Reaffirming the v0.22 deferral list, minus what v0.23 ships:

- **AnimateDiff** — still a v0.24+ multi-cycle effort. v0.20 +
  v0.21 + v0.22 + v0.23 deferrals carry forward.
- **SD3 animate** — Flux animate was v0.20; SD3 animate's
  3-encoder lerp + MMDiT rectified-flow integrator are still
  out of cycle scope.
- **FaceID + multi-photo portrait + manual landmarks/bbox** —
  meaningful cycle on its own. Carried to v0.24.
- **Real-ESRGAN ML upscaling for `plakat.upscale`** — the script
  word is still Lanczos-only. `plakat.hires` already exposes ML
  upscalers via `hires_upscaler`; the standalone `plakat.upscale`
  word's ML path lands when convenient.
- **`plakat.embedding.*`** — Textual Inversion collection word.
  Carried.
- **`plakat.stylize`** — separate from `plakat.style.*`. Carried.
- **`plakat.outpaint`** — workflow word. Carried.
- **`plakat.metadata.*`** — JSON sidecar I/O. Carried.

## 9. Appendix: source-of-truth v0.23 deferral markers

The `grep` that drove §4:

```
src/scripting/config.rs:105 — refiner UNet load
src/scripting/config.rs:110 — plakat.style.* namespace
src/scripting/config.rs:223 — plakat.inpaint
src/scripting/config.rs:235 — clip_skip wiring
src/scripting/config.rs:242 — wildcard_dir (STALE — already wired phase 11)
src/scripting/script_entry.rs:266 — plakat.inpaint mask arg
src/scripting/script_entry.rs:694 — refiner UNet bail
src/scripting/script_entry.rs:747 — Flux CN bail
src/scripting/script_entry.rs:783 — SD3 CN bail
src/scripting/script_entry.rs:884 — mask_feather + mask_invert (await inpaint)
src/scripting/script_entry.rs:943 — img2img Flux CN bail
src/scripting/script_entry.rs:988 — img2img SD3 CN bail
src/scripting/ctx.rs:71 — Flux/SD3 CN v0.23 pointer
src/scripting/ctx.rs:80 — refiner UNet v0.23 pointer
src/scripting/mod.rs:259 — refiner enable bail test
src/scripting/words/refiner.rs:28 — refiner refactor pointer
```

Phase 0 sweeps these comments + retires the stale one.
