# Feasibility review — `plakat multiperson` RFC vs the 1.12.0 codebase

Reviewer's note: this evaluates the proposed **`plakat multiperson`** RFC against the
**actual** state of the repo (1.12.0, candle 0.10.2), file-by-file. The RFC was authored
against a stale baseline ("v0.44.0, candle 0.8, PERSONA.md Limits 5/7/8/9"), so part of
this review is reconciliation and part is a structural critique of the core mechanism.

**Bottom line.** The *user-facing design* (prose-in/photos-in, the three prompt layers,
the LLM scene analyser, the extended scenario schema, routing, validation, output sidecars)
is sound and ~70% of the supporting infrastructure already exists. **But the RFC's central
generation mechanism (§9, "Spatial Conditioning") does not actually route identities by
location** on this stack — as written it blends all faces uniformly across the whole image.
Delivering the RFC's real promise (Alice left, Bob right, co-present, one pass) requires
choosing a *different* conditioning mechanism than the one written. Two viable ones exist;
both are bounded work. This doc lays out the choice.

> **Clarified goal (from the author): "the goal is really placement of the specific persona
> into the generated scene."** This reframes everything below and *de-risks* it. Placement —
> Alice in the left region, Bob in the right — is **already achievable today** via Form-2
> (`personas: [{name, bbox}, …]`): generate the scene, inpaint each persona into their bbox.
> So the genuinely-new, high-value capability the RFC adds is **not the renderer — it is the
> LLM that decides *where* each persona goes from prose** (no hand-authored bboxes). That
> layer is mechanism-agnostic and can drive the **existing** Form-2 inpaint directly, which
> makes the §9 scalar-mask flaw **moot** (you don't use it at all) and means the goal is
> reachable with **zero new attention plumbing**. The single-pass / seam-free / higher-fidelity
> renderer (Option A) becomes an *upgrade*, not a prerequisite. See the revised plan in §7.

---

## 1. Baseline reconciliation (what's stale)

| RFC claim | Reality (1.12.0) | Impact |
|---|---|---|
| Target **v0.50**, baseline **v0.44.0** | Repo is **1.12.0** | Renumber; "v0.50" is meaningless now. A multiperson feature would land as a 1.x minor. |
| **candle 0.8** | **candle 0.10.2** | The "no UNet attention hooks" claim still holds for candle's *built-in* UNet, but plakat now has a **vendored** UNet with decoupled IP hooks (see §4). |
| **PERSONA.md "Limit 5/7/8/9"** (quality ceiling / seam / wall time / InstantID) | **These limits are NOT in the current `Documentation/PERSONA.md`.** | The RFC's stated motivation cites limits that don't exist as numbered items. The *underlying* facts (shared-attention quality ~50–70%, sequential inpaint) are real and documented inline in `portrait.rs:15-21`; the framing needs rewriting. |
| Keyword **`people`** (renamed from `personas`) | Today's keyword is **`personas`** (`scenario.rs` `PersonaDef`, `TaskDef.personas`) | Fine to add `people` as an alias; the deprecation plan (§11 of the RFC) is reasonable. |
| **InstantID "not implemented"** (Limit 9) | True — not ported. | Unchanged; the strategy enum is already `PlusFace | PlusFaceSdxl | FaceId | FaceIdSdxl` (`ip_adapter.rs`), extensible. |

None of this is fatal — it's a re-baselining pass. The substantive issue is §2.

---

## 2. The core-mechanism flaw (the thing to fix before building)

The RFC's value proposition is **single-pass, simultaneous, spatially-placed** identities —
"all people co-present from the first denoising step … relative positions, sizes, gaze …
emerge from the same diffusion process" (§3–§4), via §9:

```rust
// RFC §9
let weight = mask_i.mean_all()?;          // ← collapses the H/8×W/8 mask to ONE scalar
let scaled = tokens_i.broadcast_mul(weight)?;
combined   = Σ scaled;                     // then concatenated onto the text tokens
```

**How identity tokens actually reach the UNet today** (`portrait.rs:548-621`,
`build_encoder_hidden_states`):

```rust
let cond_full = Tensor::cat(&[&text_cond, image_tokens], 1)?;  // shared cross-attention
// cond:   [text(77) | image_tokens(T)]
// uncond: [text(77) | zeros(T)]
```

Identity tokens are **concatenated onto the text sequence** and consumed by the UNet's
*shared* cross-attention (same `to_k`/`to_v` as text). There is **no spatial address** on a
cross-attention token — a token influences every latent pixel equally.

So the RFC's `w_i = mask_i.mean()` throws the spatial information away *before* it could
matter, and concatenation has nowhere to put it. The result of `Σ tokens_i·mean(mask_i)`
fed to shared attention is: **every face's identity, blended by a scalar, smeared across the
entire image.** You'd get one merged/averaged identity everywhere, not person-per-region.
Spatial placement would still come *entirely from the text prompt + model priors* — i.e.
no better than typing "three people" today.

This is the single load-bearing defect. Everything else in the RFC can stand; this stage
must be replaced with a mechanism that gets the mask **into the attention**.

---

## 3. Infrastructure inventory — what already exists (reusable as-is)

| RFC needs | Exists? | Where |
|---|---|---|
| IP-Adapter identity encoders (plus-face, faceid) | ✅ | `ip_adapter.rs` — `PlusFaceEncoder` (CLIP-H→Perceiver→`(1,16,768)`/`(1,16,2048)`), faceid ArcFace→4 tok; multi-photo weighted merge built in |
| Portrait base/inpaint/blend primitives | ✅ | `portrait.rs` — `generate_latents_one`, `inpaint_latents_one`, `blend_latents_one`, `save_image` (the current Form-2 multi-persona path) |
| `personas` scenario schema (Form 1/2, bbox, multi-photo, face-bbox/landmarks) | ✅ | `scenario.rs` — `PersonaDef`, `TaskDef.personas`, `PersonaRef::Bbox` |
| Soft latent masks (feathered, latent res) | ✅ | `tiled.rs::region_mask` (5% feather signed-distance ramp) — directly usable for the RFC's soft masks |
| LLM provider stack for the scene analyser | ✅ | `llm/enhancer.rs` — `Enhancer::enhance(system, user, opts) -> String`; generic, reusable for a JSON layout call; process-wide cache pattern present |
| SCRFD face detection (for face-refine) | ✅ | `scrfd.rs` — `SCRFDDetector::detect(path) -> Vec<Face>` (bbox + 5 landmarks) |
| Gaussian blur for masks | ✅ | `imageproc` already a dep (`controlnet_annotator.rs`); `imageproc::filter::gaussian_blur_f32` available |
| Decoupled IP cross-attention (`to_k_ip`/`to_v_ip` per block) | ✅ (vendored only) | `instantstyle.rs::install_instantstyle` + `sd_train::unet`'s `install_style_ip` / `IpInjection` — **NOT** on candle's built-in UNet |

**Genuinely new code** the RFC needs regardless of mechanism: the scene-analyser
(`enhance` + a `SceneLayout` JSON schema + parse/fallback, ~250 lines), the
`MultipersonPrompt` three-layer type (~120 lines), figure↔person assignment + override
resolution, the scenario routing for `pipeline: multiperson`, and the dry-run summary.
All of that is accurate and worth keeping from the RFC. The disagreement is only §9.

---

## 4. The decoupled path plakat actually has

`instantstyle.rs` installs, per up/attn block of the **vendored** SD UNet
(`sd_train::unet::UNet2DConditionModel`), a separate IP K/V projection:

```rust
let to_k_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_k_ip"))?;
let to_v_ip = candle_nn::linear_no_bias(ctx, inner, lvb.pp("to_v_ip"))?;
ips.push(IpInjection::new(to_k_ip, to_v_ip, scale, tokens.clone()));
unet.install_style_ip(up_idx, attn_idx, ips)?;
```

This is the real lever. The IP path computes its own attention
`softmax(q_latent · k_ip) · v_ip` and adds it (scaled) to the text attention. Crucially,
**`q_latent` carries the spatial dimension** — so a per-region mask *can* be applied to the
IP attention output (zeroing person A's contribution outside the left region). That is the
diffusers "regional IP-Adapter / attention masking" technique, and it's the only way to get
true spatial routing in a single forward pass on this codebase.

Caveats: it's the **vendored** UNet (different load/forward than the portrait pipeline's
candle UNet), it's wired today for **one** token set (style), and IP-Adapter face weights
must be loadable into the `to_k_ip/to_v_ip` layout. Extending it to **N masked token sets**
is the core of Option A below.

---

## 5. The two viable mechanisms

### Option A — Regional IP-attention masking (vendored UNet, single pass)

Extend `instantstyle.rs`'s decoupled injection to hold **N** identity token sets, each with a
soft region mask; inside each block's IP attention, multiply person *i*'s attention output by
its (downsampled) `region_mask_i` before summing. One UNet forward per step.

- **Delivers the RFC's actual promise**: Alice-left/Bob-right, co-present, one denoise pass,
  no inpaint seams.
- **Wall time**: ~1× a normal denoise (the IP K/V adds are cheap) — genuinely fixes the
  "N+1 loops" cost the RFC wanted to fix.
- **New work**: generalize `IpInjection` to a `Vec<(tokens, mask)>`; thread per-block mask
  downsampling; load face IP weights into the vendored UNet (today it loads style IP). Medium;
  the plumbing exists, the multi-set + masking is new. Highest-value, moderate risk.
- **Quality**: decoupled path is closer to diffusers reference (~80%+) than shared-attention
  (~50–70%) — a side benefit. faceid still strongest for likeness.
- **Risk**: vendored UNet ≠ portrait's candle UNet (model-loading divergence); needs on-box
  verification (SD1.5/SDXL fit 24 GB, so verifiable — unlike the trainers).

### Option B — MultiDiffusion regional blend (no new attention plumbing)

Each denoise step, run the UNet **once per region** with that region's text+identity
conditioning, then blend the noise predictions by the normalized soft masks (the `tiled.rs`
regional / MultiDiffusion pattern + the existing portrait IP conditioning per region).

- **Delivers spatial + simultaneous**: each region denoises every step; blending the
  *latents/eps* (not inpainting a committed base) avoids Form-2's seams.
- **Wall time**: **N forward passes per step** → ~N× a single denoise. Does **not** fix the
  cost the RFC complains about (it's arguably worse than Form-2's N+1 *total* loops for large
  N), though it's seam-free and co-present.
- **New work**: a regional denoise loop reusing `region_mask` + `build_encoder_hidden_states`
  per region + eps blending. Lowest risk (no attention surgery, candle built-in UNet), reuses
  the most existing code.
- **Quality**: bounded by shared-attention IP (~50–70%) since it's the same injection per
  region; overlaps blend naturally.

### Quick comparison

| | A — Regional IP-attention | B — MultiDiffusion blend |
|---|---|---|
| Spatial routing | ✅ true (mask in attention) | ✅ true (mask on eps) |
| Single pass / wall time | ✅ ~1× | ❌ ~N× per step |
| Seam-free | ✅ | ✅ |
| Identity fidelity | ✅ decoupled (~80%+) | ~ shared (~50–70%) |
| New plumbing | vendored-UNet multi-set masked IP | regional eps-blend loop |
| Risk | medium (UNet divergence) | low |
| On-box verifiable (24 GB) | ✅ (SD1.5/SDXL) | ✅ |

---

## 6. What's good in the RFC (keep, independent of mechanism)

- **UX**: prose-in/photos-in, no coordinates required; bbox/hint as escape hatch. Excellent.
- **Three prompt layers** (scene / style / per-person) + the `//` separator + the
  `MultipersonPrompt` type that enforces "analyser sees scene-only". Clean; keep verbatim.
- **LLM scene analyser** returning a `SceneLayout` JSON (arrangement/figures/mood) with
  three-stage parse fallback + geometric default. Reuses `Enhancer`; sound.
- **Scenario schema** (`pipeline:`, `identity:`, `style:`, `people[*].prompt`, task overrides)
  and **routing table** (§14) — well-specified; `route_task` is correct.
- **Validation** (§18), **recipe sidecar** (§19), **dry-run** (§20), **honest limits** (§21):
  all good engineering, all reusable.

These are the bulk of the ~1,500 LOC estimate and they're accurate. The estimate's
"conditioning.rs is a drop-in" assumption is the only wrong part.

---

## 6a. Placement input model (clarified: the user supplies a relative location per persona)

The author needs to **provide a relative location for each persona** — i.e. placement is a
*per-persona input*, expressed in words, not pixel coordinates and not (only) LLM inference.
This is the RFC's §8 named-position vocabulary, **promoted to the primary interface**:

```hjson
people: [ { name: alice, at: left }
          { name: bob,   at: center }
          { name: carol, at: right } ]
```
```sh
plakat multiperson "three friends having tea" \
  --person "alice:./alice.jpg" --at "alice:left" \
  --person "bob:./bob.jpg"     --at "bob:center" \
  --person "carol:./carol.jpg" --at "carol:right"
```

Two grades of "relative", both already supported by existing infra:

- **Frame-relative zones (primary)** — `left | center_left | center | center_right | right |
  back_left | back_center | back_right | foreground | foreground_left | foreground_right`.
  Each maps to a centroid + spread (the RFC §8 tables) → a soft `region_mask` / bbox. Direct,
  deterministic, no LLM. This is the RFC's `hint`, used as the main input.
- **Persona-relative (relational, optional extension)** — `left of bob`, `behind alice`,
  `between alice and carol`. Resolve in dependency order: place anchored personas first, then
  offset. A small grammar on top of the zone table; defer to a follow-up if not needed now.

Consequence: with explicit `at:` per persona, **the LLM scene analyser is optional** — it only
fills in personas left unpinned (or is skipped entirely). The placement decision is the user's;
plakat converts each relative location → region → renders the persona there. This is the whole
goal, with no LLM dependency and no new attention plumbing.

## 7. Recommendation (revised for the clarified goal: *placement*)

The goal is placing specific personas into the scene. The existing Form-2 inpaint already
*places*; what's missing is **deciding the placement from prose** and integrating it cleanly.
So lead with the placement layer over the existing renderer, and treat the fancy renderer as
an optional upgrade — not a blocker.

1. **Re-baseline** the RFC to 1.12.0 / candle 0.10.2; drop the "Limit 5/7/8/9" framing (cite
   the real `portrait.rs:15-21` shared-attention note); add `people` as a `personas` **alias**
   (no rename churn). **Delete §9 as written** (the scalar-mask-on-shared-attention mechanism
   is unsound and unnecessary for placement).

2. **Milestones** (each on-box verifiable; SD1.5/SDXL fit 24 GB):
   - **M1 — the goal, delivered (low risk, mostly reuse).** Per-persona **relative location**
     (`at: left|center|right|foreground|back_left|…`, §6a) → centroid + spread (§8 tables) →
     soft `region_mask` / bbox → drive the **existing Form-2 inpaint** (`generate_latents_one`
     + `inpaint_latents_one` per persona region). This *is* "provide a relative location and
     place that persona there." The **LLM scene analyser is optional** here — it only auto-places
     personas with no `at:`. Ships the whole UX (prose-in/photos-in, three prompt layers,
     scenario `pipeline: multiperson`, dry-run, sidecar, validation). **No new attention
     plumbing.** Explicit `bbox` overrides `at:` for pixel control.
   - **M2 — integration quality (optional, low risk).** Replace the *sequential* inpaint with
     a **MultiDiffusion regional eps-blend** (Option B): per step, denoise each region with its
     persona's identity, blend by the soft masks. Removes Form-2's "committed base" seams +
     co-presence problems. Same conditioning, same fidelity; ~N× wall time.
   - **M3 — fidelity + single-pass (ambitious, medium risk).** Option A: masked decoupled IP
     on the vendored UNet — one forward/step, ~80%+ likeness. The payoff once M1/M2 prove the
     surface. Swap in behind the same `pipeline: multiperson`.

   Sequencing: **M1 hits the goal with near-zero risk by reusing Form-2**; M2/M3 are quality
   upgrades you can take or leave. This inverts the RFC, which front-loaded the riskiest part.

3. **Keep verbatim** from the RFC: the UX, the three-prompt-layer `MultipersonPrompt` type, the
   `SceneLayout` schema + vocabulary + fallbacks, the routing table (§14), validation (§18),
   recipe sidecar (§19), dry-run (§20), honest limits (§21). These are the bulk of the value
   and they're accurate.

## 8. Open questions for you

- **OQ-A**: Confirm the **M1-first** plan (LLM placement → existing Form-2 inpaint) as the way
  to deliver the placement goal, with M2/M3 as later upgrades — vs jumping straight to a new
  renderer.
- **OQ-B**: Is the vendored-UNet divergence (M3) worth pursuing later, given it's already how
  InstantStyle ships? Or is Form-2 / regional-blend placement quality sufficient?
- **OQ-C**: `style:` keyword collides with stylize's `style` (an IP **reference image** path);
  pick a disambiguation rule or rename (`aesthetic:`?).
- **OQ-D**: Land this as its own `multiperson` subcommand + scenario `pipeline:`, or extend the
  existing `scenario`/`personas` path (which already does Form-2) with the LLM placement layer?
