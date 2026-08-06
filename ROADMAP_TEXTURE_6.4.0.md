# ROADMAP — plakat 6.4.0 · "Deepen texture"

A **deepening** cycle on the 6.3.0 `plakat texture` flagship (RFC TEXTURE-1) — no new domain, no
new RFC. It closes the two named gaps the 6.3 corpus exposed, plus usability fast-follows. Fully
additive; `plakat texture`'s command surface and output contract are unchanged unless noted.

Branch `6.4.0` (off `main` @ `0772101`). Base version 6.3.0. Reference: `Documentation/RFC_TEXTURE_1.md`.

---

## Motivation — what 6.3 left on the table

1. **The metallic/roughness channels carry no spatial structure.** `derive::metallic_from_albedo` and
   `roughness_from_albedo` (`src/texture/derive.rs`) are *per-pixel* heuristics. For a single-class
   material that's correct (stone → flat black, steel → flat white — both physically right). But a
   **composite** material — rusted iron, a gilded frame, chipped paint, coins in mud — needs a
   *spatially-varying* mask (bare-metal regions white, rust/paint black), and today it can't produce
   one worth shipping. The `from-albedo` path thresholds colour with no region coherence.
2. **The native-seamless escalation path is built but dormant.** `seamless::roll2d` +
   `SeamlessConv2d` exist and are unit-tested, but only post-hoc `make_tileable`/`feather_seam` run in
   `render.rs`. For **high-frequency** materials (fine gravel, chainmail, woven cloth, brushed grain —
   steel's x-tileability 0.35 was the corpus worst) feather *softens* a residual seam rather than
   removing it. The measured-escalation promise from 6.3 (§B3) is unredeemed.
3. **Brushed/anisotropic materials are misrepresented.** Steel's grain is directional; we only emit
   isotropic roughness + no anisotropy direction. A real gap the steel demo surfaced.
4. **Discoverability:** users reasonably read a flat metallic map as a bug (this cycle was *requested*
   after exactly that confusion). The docs and `verify` scorecard should explain, not just emit.

---

## G0 — de-risk before building (measure-first, per RFC §16 house style)

Two novel decisions need a probe before committing the build, exactly as 6.3's G0 did.

- **G0.A — spatial-channel approach (LOAD-BEARING).** Decide *how* a composite material gets a
  structured metallic/roughness mask. Candidates, cheapest-first:
  1. **Region-coherent segmentation of the albedo** — k-means / connected-components on (luma, sat,
     hue) → per-region PBR assignment from a tiny lookup. Pure, weight-free, deterministic.
  2. **A dedicated generated mask** — re-condition the diffusion to emit "white=metal / black=else"
     (ControlNet-guided by the albedo). Powerful but reliability-dubious from a T2I backbone.
  3. **Bilateral-smoothed heuristic** — keep the per-pixel rule but add spatial coherence + edge-stop.
  Probe: build a synthetic rusted-iron fixture (known metal/rust mask) + score each candidate's mask
  IoU vs ground truth. **Exit:** pick the approach whose mask IoU clears a bar on the fixture AND
  survives the tileability wrap. Default expectation: (1) primary, (3) fallback, (2) documented
  escalation — mirroring 6.3's "circular-conv is the escalation" call.
- **G0.B — per-step latent-roll efficacy.** Wire `roll2d` into a throwaway probe around a real SD
  denoise loop; render a high-frequency material (fine gravel) with (a) feather-only vs (b) latent-roll
  + feather; measure seam ratio. **Exit:** if roll+feather gets a hi-freq material under the scorecard
  seam bar, Track B ships roll (no conv surgery); if not, escalate to the vendored `SeamlessConv2d`
  ResNet. Keep candle-Metal ≤4-D rule in mind (see the candle-metal memory).

Both probes are `examples/texture_*_probe.rs`, committed. Do G0 FIRST.

### G0 RESULTS — both PASS with clear decisions (commit `8e87daa`)

- **G0.A DONE → `examples/texture_metallic_probe.rs` (weight-free).** On a synthetic rusted-iron
  fixture built with the exact per-pixel failure modes (dark scratches on steel → per-pixel *misses*;
  pale desaturated rust patches → per-pixel *false-fires*), scored vs ground-truth metal mask by IoU:
  baseline per-pixel **0.656** (prec 0.72 / rec 0.88), **region-vote (circular box r=8 majority) 0.904**
  (prec 0.99 / rec 0.91), bilateral edge-aware 0.839. **Decision: Track A `metallic:"auto"` = soft
  per-pixel metal-ness → circular region-vote → threshold** — region-coherent AND tileable (circular
  window). Notable: bilateral *loses* to region-vote because its edge-stopping wrongly *preserves* the
  pale-rust patches it should out-vote — so plain isotropic voting is the right call, not edge-aware.
- **G0.B DONE → `examples/texture_roll_probe.rs` (Metal).** A linear conv proxy can't faithfully show
  the *generative* per-step-roll benefit (lateral propagation ⇒ attenuation confounds the seam ratio),
  so the probe proves the cleanly-provable MECHANISM instead: shift-averaging a zero-pad conv
  (`roll(-s)∘F_zero∘roll(s)`) reaches seam **1.231** vs the circular-pad ideal **1.200** (zero-pad
  1.516) — **closing 90%** of the zero-pad→circular gap. **Decision: ship B1 (per-step latent-roll)**,
  and measure its definitive high-frequency number on a *real* material during the B1 build (the hook is
  contained + default-off + byte-identical when off, so build-then-measure is safe); **B2 (vendored
  circular ResNet) stays the guaranteed fallback — its feasibility is already proven by 6.3's G0.1**
  (circular conv → seam ≈ 1.0 at any depth). No feasibility risk is left un-gated.

---

## Track A — spatially-varying data channels (the headline)

- **A1 — region-coherent metallic mask.** Per G0.A: a `metallic: "auto"` (and `--metallic auto`) mode
  that segments the albedo into material regions and assigns metal vs dielectric per region, so a
  composite material gets a *structured* metallic.png. Keep the scalar and `from-albedo` modes. Circular
  ops so the mask tiles. Weight-free.
- **A2 — spatial roughness.** The same region machinery drives a `roughness: "auto"` that varies by
  region (wet/dry, polished/scratched) instead of the pure per-pixel curve. Keep `from-albedo`/scalar.
- **A3 — `--metallic-ref` / `--roughness-ref`.** Let the user supply a hand-painted mask PNG for either
  channel (ultimate control; bypasses derivation). Loads + resizes + tiles-checks.
- **A4 — scorecard: channel-structure note.** `verify` reports whether metallic/roughness are flat
  (single-class) vs structured, and states *flat-is-correct-for-single-class* in the notes — so a flat
  map reads as a decision, not a defect. Closes the discoverability gap (motivation #4).

## Track B — seamless for high-frequency materials

### B RESULT — DONE (commit `3e8f0f2`): adaptive feather shipped; tileable-generation deferred (owner-confirmed)
Measured fine gravel (worst case): feather tiles it cleanly (seam **0.05**, PASS), leaving only a mild
smear (~24% seam-band detail loss). **Shipped the safe half — adaptive feather** (`raw_seam` sizes the
band to the measured raw seam; halved gravel's smear-band width 43→21px, never worse than fixed).
**B1-full (per-step roll) / B2 (vendored circular ResNet) NOT shipped** — invasive to the shared
corr-1.0 sampler across 7 families, and the mild hi-freq-only residual doesn't justify the risk (owner
chose "ship adaptive, move on"). `roll2d`/`SeamlessConv2d` stay the dormant, proven-feasible escalation.


- **B1 — per-step latent-roll hook.** Per G0.B: a contained, default-off `tileable-diffusion` hook in
  the SD denoise loop (`roll2d` the latent per step, unroll the prediction — byte-identical when off).
  On for `texture render` when `seamless.mode == circular`. Measure hi-freq corpus materials.
- **B2 — vendored circular ResNet (escalation, gated on B1).** ONLY if B1's residual fails the bar:
  vendor a `SeamlessConv2d`-based ResNet block (candle_transformers `ResnetBlock2D` is Apache-2.0;
  G0.1/6.3 finding — the own UNet delegates resnet convs, so this must be vendored, not a flag).
  Scope strictly to what the seam demands; do not regress the corr-1.0 generation for other families.

## Track C — richness & fidelity fast-follows

- **C1 — anisotropy.** A directional roughness for brushed/grained metals: detect dominant grain
  direction (or take `anisotropy: {angle, strength}` from spec) → emit an anisotropy/direction map +
  grain-aware roughness. Preview shows the streak highlight. Scope-limited; steel is the test.
- **C2 — variations.** `texture render --variations N` → N seed variants written side-by-side (distinct
  from `--attempts`, which rejection-samples to one). Optional `--keep-best` picks by scorecard.
- **C3 — material blend.** `texture blend A.hjson B.hjson --mask grad|<png>` → a blended material set
  (stone→moss). Blend each channel with the mask; re-score.

## Track D — corpus, docs, discoverability, CUT

- **D1 — composite corpus.** Add specs that *exercise* the new channels: a rusted-iron plate (structured
  metallic), a wet-and-dry stone (spatial roughness), a fine-gravel (hi-freq seamless). Wire into
  `texture_run.sh`; update `TEXTURE_CORPUS.md`.
- **D2 — docs.** `Documentation/TEXTURE.md`: a "reading the channels" section (dielectric-black vs
  conductor-white, when a flat map is *correct* vs a bug, `auto` vs `from-albedo` vs scalar); document
  `--variations`/`blend`/anisotropy/`-ref` flags. Verify every flag vs `--help`.
- **D3 — integration parity.** Thread any new spec fields (`metallic/roughness: auto`, anisotropy,
  variations) through compile/scenario/Bund/`api::Texture`/doctor (bookart A1–A5 template).
- **D4 — CUT 6.4.0.** Bump Cargo.toml+lock 6.3.0→6.4.0, gate `cargo test --no-default-features --lib`,
  README what's-new + docs-index, FF `git push 6.4.0:main`, tag `v6.4.0` → CI 6-asset, `cargo publish
  --locked --allow-dirty --no-default-features`, `gh release edit` (GH_TOKEN=vulogov owner, no env -u;
  background waiter for the CI-created release). NO Claude/Anthropic coauthor.

---

## Sequencing

G0.A + G0.B (probes) → **A1–A4** (the headline, weight-free-heavy, ships value even alone) → **B1**
(measure) → B2 only if B1 fails → **C1–C3** (as budget allows; C2 is cheapest) → **D1–D4** (cut).
Front-load Track A: like 6.3, most of it is derivation/scorecard that needs no generation and can be
proven on supplied albedos before any GPU work.

## Non-goals (unchanged from RFC TEXTURE-1 §"non-goals")

Not a substance-graph editor; metal/rough workflow only (no spec/gloss); tangent-space normals only;
tileability MEASURED not shader-proven; preview stays an approximation. Anisotropy (C1) is a *map* +
preview approximation, not a full anisotropic BRDF solve.
