# plakat — Roadmap to 1.0.0

Tracked checklist of what must land before cutting **1.0.0**. Opened on the
0.47.0 dev branch (0.46.0 shipped 2026-06-09).

Status legend: `[ ]` open · `[x]` done · `[~]` in progress · `[?]` decision needed.

## Thesis: 1.0 is confidence, not code

An audit of the tree found **2 `TODO` markers** and no half-built features —
the bails are input validation, not feature gaps. plakat is already
feature-complete and mature (23 subcommands; 7 model families; LoRA / DoRA /
TI / IP-Adapter / ControlNet; img2img / inpaint / outpaint / upscale;
portrait / identity; AnimateDiff; style training; scenarios; Bund scripting).

So 1.0 is **not** "finish the code." It is three things:

1. **Verification** — every claim is proven by a committed render (the proof
   corpus is plakat's own thesis).
2. **API stability** — freeze the public contract; breaking changes are free
   now and expensive after 1.0.
3. **Polish** — accurate docs, graceful failures, clean examples.

**Critical path:** §1 (verify) → §2 (freeze) → ship. §3–§5 run in parallel.
The single highest-value, irreversible item is **§2's API freeze**.

---

## 1. Verification — prove every claim (hard gate)

- [~] **Smart transparent + integral artefacts (RELEASE BLOCKERS).** Chroma-key
      only works on a clean studio backdrop; real-world cut-outs are photoreal /
      painted subjects on arbitrary backgrounds. Added **`transparent --matte`**:
      content-aware U2Net salient-object matting (`src/pipelines/matting.rs`;
      `jamino30/u2net-saliency`, MIT, ungated). This is the prerequisite for BOTH
      a smart cut-out AND clean artefact-library cutouts that the
      `--artefact-blend` pass integrates into a scene (not paste). Model
      **VERIFIED correct** via a PyTorch reference comparison — the candle forward
      matches the reference bit-for-bit (side1_raw, fused logits, d0 all match).
      The first weights (`jamino30/u2net-saliency`) turned out to be a DEAD
      checkpoint (d0 max 0.04 even in PyTorch); good weights (`Carve/u2net-
      universal`) fire at d0 max 1.0 through the same code. Remaining: a good
      u2net checkpoint in a **candle-loadable format** — Carve's is legacy-pickle
      `.pth` (candle only reads safetensors / zip-`.pth`), so sourcing/converting
      one. Then re-point `transparent.sh` / `artefact.sh` off chroma.
- [ ] **Render the 4 pending corpus drivers** (written, unrendered): `transparent.sh`,
      `embedding.sh`, `variation.sh`, `artefact.sh` + `artefact.hjson`. Judge
      each at full size; commit the proofs. *(GPU-bound.)*
- [?] **Flux-on-Metal** — broken (candle's Metal quantized matmul kernel; not
      plakat-fixable). **Decide:** scope Flux as **CPU/CUDA-only** and correct
      every "supported" claim, or drop it from the 1.0 model list. A 1.0 must
      not silently list a model broken on the primary platform.
- [ ] **Final model-list audit** — confirm **zero** "listed but never rendered"
      models remain across `doctor models` (the corpus caught 7; verify the
      count is 0). Every supported model has a committed proof.
- [x] **Cascade LoRA/DoRA** — engine complete (v0.38/v0.42); no ungated asset
      exists to demo → marked BLOCKED in COVERAGE (ecosystem, not code).

## 2. API freeze — the 1.0 contract

Breaking changes are cheap before 1.0 and a major-version event after. Do them now.

- [~] **CLI surface audit — DONE; renames pending.** Enumerated every flag across
      the 23 subcommands. GOOD: common flags (`steps`/`seed`/`model`/`out`/
      `negative`/`guidance`/`size`) are uniform; flag families (`control-*` /
      `hires-*` / `enhance-*` / `adetailer-*` / `artefact-*` / `grid-*` / `tile-*` /
      `window-*` / `motion-lora*`) are clean; **no legacy flag aliases** → renames
      are a clean slate. Fix before the freeze (all breaking — do now, update
      corpus + docs):
  - [x] **`--loras` → `--lora`** (repeatable) in `img2img` + `outpaint` — done
        (source + docs); now uniform with `generate`/`portrait`, repeatable style.
  - [x] **`--for` (stylize) → `--preset`** — done (source + docs).
  - [ ] **Quant flags** `--quantize-t5` / `--quant-level` / `--t5-quant-level` —
        inconsistent (verb vs noun-level). Unify (semantics decision needed).
  - [x] **`--type` (civitai) → `--asset-type`** — done (source + docs).
  - [x] **Input convention + env-var public/internal split documented** →
        `Documentation/CLI_CONVENTIONS.md` (the 1.0 CLI contract).
  - [?] **Deviation surfaced:** `stylize` / `transparent` / `upscale` take the
        primary image via `--in`, vs `img2img` / `outpaint` positional. DECIDE:
        unify to positional, or freeze the `--in` exception (postponed, like #3).
- [ ] **Freeze the scenario HJSON schema** — public contract; the whole corpus
      depends on it.
- [ ] **Freeze the Bund scripting word-set** — public DSL.
- [?] **Define the library API** — decide what is `pub` for crate consumers vs
      CLI-only; SemVer binds it after 1.0.

## 3. Resolve or document known issues

- [ ] **Metal single-buffer OOM** (Real-ESRGAN ×4, large SD 1.5 portraits) —
      catch it and emit a clear "retry with `--device cpu`" hint instead of a
      cryptic crash. 1.0 must not fail opaquely.
- [ ] **SDXL VAE-F16 black image** — memory flags it "unverified"; confirm the
      `madebyollin/sdxl-vae-fp16-fix` VAE (in `SdCore`) closes it, then mark resolved.
- [?] **Stylize / InstantStyle** — the IP-Adapter transfers content/palette, not
      painterly texture (confirmed on SD 1.5 **and** SDXL). **Decide:** ship 1.0
      with stylize documented as a *ref-variation* tool (recommended), or build
      InstantStyle per-layer injection (a full cycle — defer).
- [?] **Minor deferrals** — Cascade Stage-B + ControlNet combo; SDXL tiled
      scripting. Almost certainly post-1.0; label them explicitly.

## 4. Polish

- [ ] **Stale-docs sweep** — e.g. `embedding --help` still says runtime injection
      is "gated … lands in a follow-up" (v0.30 fixed it). Audit every `--help` /
      Status note for accuracy. 1.0 docs must not lie.
- [ ] **`examples/` cleanup** — the `spike_*` / `train_spike` examples are dev
      scaffolding ("weak"). Promote to real examples or delete.
- [ ] **Error-message coverage** — extend the v0.33 actionable-hint pass so
      every common failure carries a fix suggestion.
- [ ] **Reproducibility note** — Metal renders are not bit-reproducible; state
      it plainly (acceptable for 1.0 if documented).

## 5. Infrastructure

- [ ] **CI green + reliable** — the apt-mirrors flakiness has bitten before;
      make it rock-solid.
- [ ] **Test coverage for recent features** — smart-discovery judge, SDXL
      stylize, numbered checkpoints, the new corpus drivers.
- [ ] **docs.rs build** — confirm the Linux / no-Metal docs build is clean
      (the `cargo publish` dry-run passes; verify docs.rs specifically).
- [ ] **Crate metadata** — license, keywords, description, repo links 1.0-ready.
- [ ] **Tutorial completeness** — every subcommand has a tutorial (STYLIZE just
      added; ensure transparent / embedding / variation / artefact are covered).

## 6. The corpus is the 1.0 proof

- [ ] **Fully-rendered corpus + complete GALLERY** = the release evidence.
      Gated on §1's pending renders.

---

## Decisions needed (blockers on scope, flagged `[?]` above)

1. **Flux**: CPU/CUDA-only, or out of 1.0?
2. **Library API**: what's `pub` and SemVer-bound?
3. **Stylize**: document-as-ref-variation, or build InstantStyle?
4. **Minor deferrals**: confirm post-1.0.

## Explicitly post-1.0 (non-goals for the cut)

- InstantStyle (per-layer IP-style transfer).
- Additional ControlNet types beyond canny (depth / pose).
- Cascade Stage-B + ControlNet; SDXL tiled scripting.
- Flux on Metal (blocked upstream in candle).
