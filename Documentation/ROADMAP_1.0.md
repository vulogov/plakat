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
      The first weights (`jamino30/u2net-saliency`) were a DEAD checkpoint (d0 max
      0.04 even in PyTorch); good weights (`Carve/u2net-universal`, Apache-2.0)
      fire at d0 max 1.0 through the same code. **DONE:** candle can't read
      Carve's legacy-pickle `.pth`, so it's re-serialised to safetensors
      (`scripts/convert_u2net_to_safetensors.py`), hosted at
      `vulogov98/u2net-universal`, and auto-downloaded on first use. The matte
      fires end-to-end through candle (d0 max 1.0, identical to the PyTorch
      reference; clean salient-object silhouettes). Both `transparent.sh` /
      `artefact.sh` re-pointed to `--matte` (off chroma). **DONE: both rendered +
      committed — transparent apple cut-out + integral anime artefact scenes.**
- [x] **Render the pending corpus drivers — DONE.** `transparent.sh`,
      `artefact.sh`/`.hjson`, `embedding.sh` (EasyNegative repointed to `embed/`),
      `variation.sh` (Cascade pure+steered), `stylize.sh` (concat baseline + 2
      verified SDXL watercolours; SD 1.5 InstantStyle dropped from the demo as
      experimental/soft). All committed + in the gallery (74 images, 37 with
      metadata). **§1 render gate CLOSED.**
- [x] **Flux-on-Metal — DECIDED: CPU/CUDA-only, untested on Metal.** candle's
      Metal quantized matmul kernel corrupts GGUF (upstream, not plakat-fixable);
      BF16 is ~33 GB. Not dropped — it works off-Metal. Claims corrected:
      COVERAGE + `FEATURE_TO_MODEL.md` mark Flux ⚠️ CPU/CUDA-only / untested on Metal.
- [x] **Final model-list audit — count is 0.** The 7 once-never-rendered models
      are all fixed + committed: SD 1.5 / 2.1 / SDXL / SD3.5 / Cascade / PixArt-Σ
      have proofs; Flux is the only un-rendered family and is explicitly
      CPU/CUDA-only-scoped (COVERAGE + FEATURE_TO_MODEL). No silent gaps.
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
  - [x] **Quant flags** — `--quant-level` (ambiguous; it's the *Flux* level) →
        `--flux-quant-level`, parallel to `--t5-quant-level`; scenario key renamed
        to match. `--quantize-t5` toggle kept. Clean rename, no alias.
  - [x] **`--type` (civitai) → `--asset-type`** — done (source + docs).
  - [x] **Input convention + env-var public/internal split documented** →
        `Documentation/CLI_CONVENTIONS.md` (the 1.0 CLI contract).
  - [x] **Deviation resolved:** the `--in` exception is FROZEN (STABILITY.md) —
        positional for `generate` (prompt) + `img2img`/`outpaint` (image you're
        continuing); `--in` for the post-process operands `stylize`/`transparent`/
        `upscale`. Deliberate, documented.
- [x] **Freeze the scenario HJSON schema** — declared in STABILITY.md (the
      `Scenario`/`Task` key set is the frozen 1.0 contract).
- [x] **Freeze the Bund scripting word-set** — declared in STABILITY.md (the
      ~50 `plakat.*` host words).
- [x] **Library API** — DECIDED (STABILITY.md): plakat is a CLI; the `plakat::*`
      Rust API is **not** a stability contract. Consume via CLI / scenario / Bund.

## 3. Resolve or document known issues

- [x] **Metal single-buffer OOM** — the upscale path now wraps its result in
      `decorate_oom(OomContext::Upscale)`, so a Real-ESRGAN ×4 OOM emits a
      "retry with `--device cpu`" hint (+ "use ×2") instead of a raw crash.
      Generate already had the OOM decorator; both covered + tested.
- [x] **SDXL VAE-F16 black image — RESOLVED.** `SdCore` swaps in
      `madebyollin/sdxl-vae-fp16-fix` for F16 SDXL (`sd_core.rs:307-316`), and the
      committed SDXL corpus renders (sdxl.hjson, stylize, artefact) are clean — no
      black image.
- [x] **Stylize / InstantStyle** — BUILT + verified (0.47.0). `stylize
      --instantstyle` does true painterly style transfer via a decoupled IP
      cross-attention on the style block (SDXL `up_blocks.0.attentions.1` / SD 1.5
      `up_blocks.1.attentions.1`); confirmed clear watercolour transfer at
      `--style-scale` ~3-5 + `--strength` ~0.8. The default concat path stays as
      ref-variation. Decision resolved by building it.
- [x] **Minor deferrals — RESOLVED in the 1.0.0 cycle, not deferred:** SDXL tiled
      scripting was *built* (`plakat.tiled.*` words + driver). Cascade Stage-B +
      ControlNet was *falsified* — Stable Cascade applies CN to Stage C alone by
      design (Stages B/A are fixed), so there is nothing to build. See
      [`ROADMAP_1.0.0.md`](ROADMAP_1.0.0.md) Part 0.

## 4. Polish

- [~] **Stale-docs sweep** — fixed `embedding --help` (was "gated"), the Flux
      OOM-hint flag, and the quant-flag docs (→`--flux-quant-level`). A full
      `--help` audit remains.
- [x] **`examples/` cleanup** — removed the spike/TEMP scaffolding (`train_spike`,
      `sd_train_run`, `style_train_run`, `spike_catalog`).
- [~] **Error-message coverage** — OOM hints now offer `--device cpu` (Metal
      single-buffer). The broader audit continues.
- [x] **Reproducibility note** — stated plainly: a `## Reproducibility` section in
      the README (Metal is non-deterministic; `--seed` is repeatable same-machine;
      every output embeds its recipe + `--reproducibility-check` measures drift),
      and in STABILITY.md's "not frozen" list.

## 5. Infrastructure

- [~] **CI green + reliable** — added `ci.yml` (CPU build + lib test on every
      push/PR; no apt-mirror cross-compile → reliable). Confirm green on the runner.
- [x] **Test coverage for recent features** — +9 lib tests this cycle: InstantStyle
      block selection (the debugged bug), OOM `--device cpu` hint, colour-harmony
      math. (Matte verified via the diffusers reference comparison instead.)
- [x] **docs.rs build** — verified the CPU/no-Metal doc build is clean
      (`cargo doc --no-default-features` exit 0); pinned `[package.metadata.docs.rs]`.
- [x] **Crate metadata — 1.0-ready.** license (Unlicense), keywords, categories,
      repository + homepage all present; description refreshed (dropped the stale
      "color-key" → the current feature surface); `docs.rs` metadata pinned.
- [x] **Tutorial completeness** — TRANSPARENT / EMBEDDING / VARIATION tutorials
      added + InstantStyle section in STYLIZE; ARTEFACTS already covered.

## 6. The corpus is the 1.0 proof

- [x] **Fully-rendered corpus + complete GALLERY** = the release evidence. §1
      render gate CLOSED; 74 gallery images (37 with embedded metadata).
      Gated on §1's pending renders.

---

## Decisions needed (blockers on scope, flagged `[?]` above)

1. ~~Flux~~ — RESOLVED: CPU/CUDA-only, untested on Metal.
2. ~~Library API~~ — RESOLVED: not a stability contract (STABILITY.md).
3. ~~Stylize~~ — RESOLVED: InstantStyle built + verified (0.47.0).
4. ~~Minor deferrals~~ — CONFIRMED post-1.0.

**All four scope decisions resolved.**

## Explicitly post-1.0 (non-goals for the cut)

- ~~InstantStyle~~ — DONE in 0.47.0 (`stylize --instantstyle`); no longer post-1.0.
- Additional ControlNet types beyond canny (depth / pose).
- Cascade Stage-B + ControlNet; SDXL tiled scripting.
- Flux on Metal (blocked upstream in candle).
