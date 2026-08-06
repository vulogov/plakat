# ROADMAP — plakat 6.5.0 · "Native seamless generation"

The headline deepening of `plakat texture` (RFC TEXTURE-1): make generation **genuinely seamless** so a
material tiles because it was *generated* tiling — not because a post-hoc feather blended the edges. This
redeems the escalation path documented since 6.3's G0.1 and deferred (measure-first) through 6.4's Track
B. It eliminates the residual high-frequency **feather smear** (6.4 finding: ~24% seam-band detail loss
on gravel). Fully additive; `texture`'s output contract is unchanged.

Branch `6.5.0` (off `main` @ `b039514`, v6.4.0). Reference: `Documentation/RFC_TEXTURE_1.md`,
`ROADMAP_TEXTURE_6.4.0.md` (Track B), the G0.1/G0.B probe findings.

---

## The problem, precisely

`texture render` generates an albedo with a flat/tileable prompt, then **feathers** the boundary. That
tiles (measured seam 0.05 even on worst-case gravel) but *softens a band* at the seam — because the
generation itself is **not** tileable (a hi-freq albedo lands at raw-seam ~1.3–2.5). The only way to
remove the smear is to make the **generation** wrap. Two mechanisms, both proven-feasible:

- **Per-step latent-roll** (G0.B, 6.4): roll the latent by a random offset each denoise step around the
  UNet forward, unroll the prediction — the zero-padded convs then see the boundary at every phase.
  Shift-averaging recovers ~**90%** of the circular-conv seam gap. **No conv surgery.**
- **Native circular convolution** (G0.1, 6.3): the UNet convs wrap (`circular_pad2d` → `padding:0`). Seam
  ≈ 1.0 at any depth. But the own UNet **delegates its resnet convs to `candle_transformers::
  ResnetBlock2D`** (`sd_train/blocks.rs:6`; SDXL in `sdxl_unet.rs`), so this means **vendoring** a
  circular resnet stack (as plakat vendors CLIP) — a bigger, higher-risk lift.

And a third surface that *also* touches the boundary: the **VAE decoder**. candle's `AutoEncoderKL`
(`self.core.vae.decode()`) is zero-padded, so even a perfectly tileable latent can pick up a seam on
decode. There's a **light** fix analogous to `upscale_tileable`: **circular-pad the tileable latent →
decode → crop** the margin (the decoder's boundary artifact lands in the cropped-off pad). "Wrap it,
don't rewrite it" — the same philosophy as `circular_pad2d` and the tiling-preserving upscale.

**Scope:** `sd15` + `sdxl` (the own-UNet default; `texture` defaults to `sdxl`). Other backbones
(Sana/Flux/SD3) are out of scope this cycle. Everything is **default-off** and byte-identical when off,
so no other caller or family changes.

---

## G0 — de-risk on the REAL generation stack (do FIRST; this cycle is the riskiest since the flagship)

Unlike 6.3/6.4's synthetic probes, G0 here must run against the actual sampler + VAE, because the risk is
regressing the corr-1.0 generation path shared by 7 families.

- **G0.1 — per-step latent-roll in the real sampler (LOAD-BEARING).** Add a contained, default-off
  `seamless_roll` flag threaded `api::Generate` → `Request` → the **primary** SD denoise loop; roll the
  latent per step around `unet.forward`, unroll the noise prediction. Measure: (a) **byte-identical when
  off** across sd15/sdxl (hard regression gate — hash the latents), (b) hi-freq **gravel** seam with
  roll-**on** / feather-**off** vs the 6.4 feather baseline + the smear-band metric, (c) Metal. **Exit:**
  roll-on/feather-off gets seam ≤ the feather baseline AND kills the smear (band detail ≈ interior) →
  Track A ships roll. If roll leaves a visible residual → Track B (vendored conv).
- **G0.2 — VAE decode seam.** Decode a *synthetic exactly-tileable latent* through `AutoEncoderKL` and
  measure the decoded image's edge-wrap seam; then measure the **pad-decode-crop** variant. **Exit:** if
  the plain decode seam is negligible → no VAE work; else Track C ships pad-decode-crop (light) — and
  only if *that* fails does the decoder get vendored.
- **G0.3 — regression surface map.** Enumerate every denoise loop the flag could reach (base / tiled /
  PAG / img2img) and confirm the flag is inert (untouched) in all of them when off. No silent coupling.

Probes are `examples/texture_seamlessgen_probe.rs` (+ reuse of the G0.1/G0.B primitives), committed.

---

## Tracks

### Track A — per-step latent-roll integration (the contained win; primary path)
- **A1** — thread `seamless_roll: Option<Roll>` from `api::Generate` → the request/config → the primary
  SD denoise loop. Roll the latent by a per-step deterministic offset (seeded, reproducible) around
  `unet.forward`; unroll the prediction so the latent stays in canonical frame between steps. Default
  off → byte-identical.
- **A2** — apply in `texture::render` when `seamless.mode == circular`: generate with roll ON, then only
  a *minimal* boundary cleanup (or none) instead of the adaptive feather. Keep the feather as the
  fallback for `mode != circular` / non-roll models.
- **A3** — the `raw_seam` measurement (6.4) now verifies the roll worked; scorecard unchanged.

### Track B — vendored circular ResNet (the deep fix; GATED on G0.1 residual)
- Only if A's residual is visible: vendor a circular-padded `ResnetBlock2D` (candle_transformers is
  Apache-2.0) into `src/pipelines/vendored_seamless.rs`; wire into the own UNet + `sdxl_unet` resnet
  stacks behind the seamless flag; the owned `conv_in`/`conv_out`/down/up convs get `circular_pad2d`
  directly. **Verify corr-1.0 is preserved when the flag is off** (dump-compare vs the current path).

### Track C — circular VAE decode (GATED on G0.2)
- Only if the VAE reintroduces a seam: **pad-decode-crop** — circular-pad the tileable latent by a few
  latent px, `vae.decode`, crop the corresponding image margin. Light, no vendoring. (Escalate to a
  vendored circular decoder only if pad-decode-crop is insufficient.)

### Track D — corpus, docs, cut
- **D1** — regenerate the hi-freq corpus materials (gravel + others) **genuinely seam-free** (roll on,
  feather off); show the before/after smear numbers.
- **D2** — docs: `Documentation/TEXTURE.md` Seamless section — native seamless is now the default for
  `circular` on sd15/sdxl; feather is the fallback. Note the roll/vendored-conv/VAE decisions the cycle
  actually made.
- **D3** — integration parity: expose the seamless-generation control where sensible (`texture` spec
  already implies it via `seamless.mode: circular`; `api::Generate` gets a `.seamless()` toggle for
  general use). doctor refresh.
- **D4** — CUT 6.5.0 (bump Cargo.toml+lock, gate `cargo test --no-default-features --lib`, FF `git push
  6.5.0:main`, tag → CI 6-asset, `cargo publish --locked --allow-dirty --no-default-features`, `gh
  release edit` GH_TOKEN=vulogov + bg waiter, NO Claude/Anthropic coauthor).

---

## Sequencing & risk posture

G0.1 (+G0.2, G0.3) → **A** (roll integration — likely sufficient, per G0.B's 90%) → measure → **B**
only if A's residual is visible → **C** only if G0.2 shows a VAE seam → **D** (cut). The bias is
**measure-first and wrap-don't-rewrite**: prefer per-step-roll + pad-decode-crop (contained, reversible,
default-off) over vendoring; vendor the resnet stack only if the measured residual demands it. Every
change is byte-identical when the flag is off, so the corr-1.0 path for all 7 families is protected by
construction.

## Non-goals
- Not extending seamless generation to Sana/Flux/SD3 this cycle (own-UNet sd15/sdxl only).
- Not a new output contract; `texture`'s maps/scorecard/export are unchanged.
- Feather is not removed — it stays the fallback for non-circular modes and non-roll models.
