# RFC GENERATE-CORE-1 — generate-core levers (6.25.0)

**Status:** SHIPPED (6.25.0). An **improvement** cycle on the text-to-image core. The survey
found generate-core already mature — **prompt attention weighting** (`(word:1.2)`), **BREAK**
chunking, **clip-skip**, **10 schedulers** (DDIM/EulerA/UniPC/DPM++Karras/UniPcExp/LCM/Euler/
EulerTrailing/Heun/DDPM), and **box-based regional prompting** all exist. So 6.25.0 adds the
three genuinely-missing levers, all cross-family or improving an existing feature:

## P1 — Subseed / variation-seed
`--subseed N --subseed-strength S`. The init noise for `--seed` is **slerp-blended** toward a
second seed's noise by `S ∈ [0,1]`, giving *controlled* variation — "the same image, nudged" —
instead of a fully fresh seed. `S=0` (default) is a no-op (byte-identical to before); `~0.05–0.2`
nudges composition; `1` is the subseed's noise. Follows `--count` in lockstep with `--seed` so a
batch stays a coherent family.

- Slerp (not lerp) keeps the blend on the ~N(0,1) hypersphere the sampler expects (a small
  strength nudges without washing out contrast). Runs on the flattened f32 init noise (tiny —
  a CPU round-trip is free and dodges Metal reduction quirks); lerp fallback when the two noise
  vectors are near-(anti)parallel. `pipelines::seeds::slerp_latents`.
- Wired in the primary txt2img init (SD 1.5 / SDXL). Tested: `slerp_endpoints_and_symmetry`,
  `slerp_parallel_falls_back_to_lerp`.

## P2 — Prompt scheduling + alternation
`[from:to:when]` swaps the conditioning at `when` (an integer step or a `(0,1]` fraction);
`[to:when]` inserts, `[from::when]` removes; `[a|b|c]` alternates every step. A **bare** `[x]`
stays de-emphasis (resolved *after* scheduling, by the existing weighted encoder) — the two
`[...]` meanings don't collide.

- `prompt::scheduling` resolves the per-step effective prompt, honouring `\[`/`\(` escapes and
  `[]`/`()` nesting when it splits a group's fields, and recursing into the chosen branch.
  `schedule()` returns the **distinct** per-step prompts + a `step→index` map, so `generate`
  encodes each unique prompt **once** and selects the conditioning per step in the denoise loop.
- Applies to the standard txt2img path (base UNet); the negative is shared across steps. No-op
  (single encode, unchanged) when the prompt has no scheduling syntax. Tested: 8 parser cases.
- **Limitation:** not wired into `--tiled` / `--region` / refiner-step conditioning (a
  fast-follow); scheduling covers the common txt2img case.

## P3 — Regional prompting v2 (per-region weight + feather)
The existing `--region "X0,Y0,X1,Y1:prompt"` gains optional per-region modifiers in the coord
section: **`w=`** (strength — a higher-weight region dominates where masks overlap; default 1.0)
and **`feather=`** (soft-edge width as a canvas fraction; default 0.05, max 0.5). Modifiers live
before the first `:` so they never collide with colons the prompt itself carries (`(word:1.2)`).

- `RegionSpec` carries `weight`/`feather`; `region_mask` takes a per-region feather; the blend
  scales each region's mask by its weight (`covered`/`base_mask` still use the raw geometric
  mask, so the base fills exactly the uncovered area regardless of weight). SD 1.5 / SDXL (and
  the sd3 regional path picks up per-region feather for free). Tested:
  `region_parses_weight_and_feather_modifiers`, `bigger_feather_widens_the_soft_band`.

## Honest limits
- Subseed/scheduling are the txt2img-path levers; the flow-matching families (Flux/SD3/Sana) keep
  a single conditioning (scheduling there is a larger change). Subseed's slerp is family-agnostic
  but wired at the SD/SDXL init.
- No metadata field yet records `--subseed`/`--subseed-strength` in the recipe sidecar (a
  reproducibility follow-up); the CLI flags are the surface.

## Sequencing
**P1** subseed (self-contained init blend) → **P2** scheduling (parser → base-loop select) →
**P3** regional v2 (RegionSpec + weighted blend) → cut 6.25.0 (bump Cargo+lock, gate
`--test-threads=1`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).
