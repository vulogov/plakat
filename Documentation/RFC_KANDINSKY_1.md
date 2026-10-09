# RFC KANDINSKY-1 — Kandinsky 5.0 Image Lite as plakat's eighth model family

| | |
|---|---|
| **RFC** | KANDINSKY-1 |
| **Status** | Accepted — Phases 0–6 built (7.2.0), gates measured on Metal; the TUI path is not run and CUDA is unmeasured |
| **Target** | 7.2.0 (diversify slot after 7.1.0) |
| **Author** | Vladimir Ulogov |
| **Model** | `kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` (MIT, ungated) |
| **Depends on** | Flux AE path (`pipelines::flux`), vendored CLIP-L (`pipelines::vendored_clip`), flow-matching sigma helpers (`pipelines::sana`, `pipelines::sd3`), `noise_space::FlowSpace` + `masked_denoise`, `nf4_codec`, candle-transformers 0.10.2 `models::qwen2` |
| **Compatibility** | Fully additive. No existing flag, alias, config key, scenario field, or sidecar changes shape. |
| **Lineage** | Reverses an exclusion: `Documentation/ROADMAP_2.7.0.md` lists "New base models (Kandinsky/Sana)" under *Explicitly excluded*. Sana shipped anyway in 4.5; this RFC does the same for Kandinsky. |

---

## 1. Summary

Add Kandinsky 5.0 T2I Lite as the eighth model family. It consists of:

- a 6B flow-matching DiT;
- **Qwen2.5-VL-7B** as the primary text encoder, used text-only through its last hidden state;
- **CLIP-L** pooled output as a global conditioning vector;
- the **Flux VAE** (16 channels).

The DiT is the only component that needs new model code (~700 lines). Every other piece already exists in plakat or in candle 0.10.2. The engineering weight of this RFC is in three places, not the architecture:

1. **Staged residency.** Encode, release the encoders, then denoise. This keeps the peak near ~15 GB BF16 instead of ~27 GB.
2. **A quantized tier.** Q4 Qwen plus an NF4 DiT bring the peak to about 6 GB.
3. **A parity harness** against diffusers tensor dumps, because several details of the reference implementation silently diverge from what plakat's existing Flux and Sana code does (§12, Appendix A).

## 2. Motivation

### 2.1 An LLM-grade text encoder

Every current family conditions on CLIP and/or T5, or (for Sana) on Gemma-2-2B. Kandinsky 5 conditions on a full 7B instruction-tuned LLM's hidden states, with a 512-token budget after the template. Three in-flight plakat features produce long, structured prompts that today's encoders truncate or flatten:

- `plakat compile`, with its family-aware prompt optimisation;
- layered generation (LAYERED-1), which derives per-layer prompts;
- `plakat persona`, whose HJSON person definitions compile into dense feature lists.

### 2.2 A distinct prior

LAYERED-1 and PAINT-1 both commit to working "with whatever model the user chooses". A family trained on a different data distribution from SD, Flux and Sana widens the space those features are tested against. It also gives users a different aesthetic default.

### 2.3 Licensing and friction

The model is MIT licensed, ungated, and needs no `HF_TOKEN`. Compare `sd35-*` and `flux-dev`, which are gated and need a token.

### 2.4 Why not wait

The reference implementation is now upstream in diffusers (`transformer_kandinsky.py`, `pipeline_kandinsky_t2i.py`), so there is a stable, inspectable target for parity tests. The model card's "requires a diffusers fork" note is obsolete.

## 3. Goals and non-goals

### 3.1 Goals

- G1. `plakat generate --model kandinsky5` produces images matching diffusers within the tolerances in §12, at all 7 native resolution buckets.
- G2. Peak resident memory at BF16 is **≤ 16 GB plus activations**, via staged residency.
- G3. A quantized tier (`--quantize-qwen`, `--dit-nf4`) runs on 16 GB unified-memory machines.
- G4. txt2img, img2img (SDEdit-style), and inpaint (RePaint-style), all through the existing `FlowSpace` and `masked_denoise` seams.
- G5. Full participation in the existing surfaces:
  - `doctor --capability`
  - `bench`
  - verify tier 2
  - `compile`'s family-aware prompt assembly
  - scenario batching
  - presets and TUI model picker
  - PNG `parameters` sidecar
  - `--etch`

### 3.2 Non-goals

- N1. **I2I-Lite (editing).**
  - Its pipeline passes the source image through Qwen2.5-VL's **vision tower** (`<|vision_start|><|image_pad|><|vision_end|>` in the template).
  - That requires a Qwen2.5-VL vision encoder and true 3D M-RoPE positions, neither of which candle 0.10 ships (candle has `qwen3_vl`, not `qwen2_5_vl`).
  - It is deferred to KANDINSKY-2. See §15 Q4.
- N2. NABLA sparse attention. It is only used by the video models; the T2I config is `attention_type: "regular"`.
- N3. Video (T2V/I2V Lite or Pro).
- N4. LoRA and ControlNet. No public adapters exist yet. The LoRA loader seam is noted in §11.4 but not implemented. *(LoRA lifted after the RFC: see Phase 6.)*
- N5. The `pretrain` checkpoint as a generation target. It is accepted as an alias only for future LoRA training.
- N6. A distilled or few-step mode. No distilled T2I checkpoint exists, and LCM/Lightning-style schedulers do not transfer.

## 4. The model

### 4.1 Components

| Component | HF class | Size (BF16) | plakat source |
|---|---|---|---|
| `transformer/` | `Kandinsky5Transformer3DModel` | ≈12 GB (6.0B params) | **new**: `pipelines/kandinsky_dit.rs` |
| `text_encoder/` | `Qwen2_5_VLForConditionalGeneration` | 16.6 GB on disk; **≈14.1 GB loaded** (text tower only, no `lm_head`) | candle `models::qwen2::Model` |
| `text_encoder_2/` | `CLIPTextModel` (ViT-L/14) | ≈0.25 GB loaded (the file is the whole CLIP, 1.71 GB) | `vendored_clip::ClipTextTransformer` |
| `vae/` | `AutoencoderKL` (Flux.1-dev VAE, diffusers key layout) | ≈0.17 GB | candle `stable_diffusion::vae::AutoEncoderKL`, as the SD3 path (not `fae`: wrong key layout) |
| `scheduler/` | `FlowMatchEulerDiscreteScheduler`, `shift: 5.0` | — | `sana::flow_sigmas` |
| `tokenizer/`, `tokenizer_2/` | Qwen2VLProcessor / CLIPTokenizer | — | `tokenizers` 0.20 |

Sizes are indicative only. `capability.rs` derives real figures from the cache or the HF API, never from this table.

### 4.2 Transformer config (verbatim from `transformer/config.json`)

```
in_visual_dim 16   out_visual_dim 16   time_dim 512   patch_size [1,2,2]
model_dim 2560     ff_dim 10240        num_text_blocks 2   num_visual_blocks 50
axes_dims [32,48,48]  (→ head_dim 128, 20 heads)   in_text_dim 3584   in_text_dim2 768
visual_cond false  attention_type "regular"
```

Parameter check: each visual block has

- self-attention: 4 × 2560² ≈ 26.2M
- cross-attention: ≈ 26.2M
- feed-forward: 2 × 2560 × 10240 ≈ 52.4M
- modulation: 512 × 9 × 2560 ≈ 11.8M

That is ≈117M per block, so ≈5.83B across 50 blocks. The text blocks and embeddings add ≈0.18B, giving the advertised 6B.

### 4.3 Pipeline defaults

- Steps: 50
- CFG: 3.5, with the empty string as the default negative prompt
- `max_sequence_length`: 512 (must be < 1024)
- Resolution buckets (W×H): `1024×1024, 640×1408, 1408×640, 768×1280, 1280×768, 896×1152, 1152×896`

## 5. Architecture of the port

```
src/pipelines/
  kandinsky.rs        — RunRequest, Pipeline, staged load/encode/denoise/decode, buckets   (~500)
  kandinsky_dit.rs    — Kandinsky5 DiT forward                                              (~700)
  kandinsky_text.rs   — Qwen2.5-VL text-tower adapter (config translation, template,
                        offset slice) + CLIP-L pooled                                       (~250)
  vendored_qwen2q.rs  — P4 only: quantized_qwen2 fork returning final hidden states         (~350)
```

### 5.1 Reuse map (grounded in the 7.1.0 tree at `a9a7faf`)

| Need | Existing seam |
|---|---|
| Alias registration | `hf::ALIAS_TABLE` (`src/hf/mod.rs:46`), Sana entries at `:264–298` as the pattern |
| Family variant + dispatch | `t2i::Variant` (`src/pipelines/t2i.rs:424`); Sana detection `:480`, predicate `:613`, guard `:957`, dispatch `:2944` |
| Capability row | `capability::MODELS`, `gen_base_gb`, `rough_weight_gb` (`src/capability.rs`) |
| Bench / verify family | `cli/bench.rs::Family` (`:87`), `verify/tier2.rs::Family` (`:33`) |
| Flow sigmas | `sana::flow_sigmas(steps, shift)` and `shift_t`, called with `shift = 5.0`. Both are private `fn` today (`src/pipelines/sana.rs:70`, `:78`); P1 makes them `pub(crate)` |
| CLIP-L pooled (1, 768) | `flux.rs::encode_prompt` (`:1786`) already computes exactly this tensor via `vendored_clip` |
| Flux AE | `fae::AutoEncoder` + `fae::Config`, with **`shift_factor` overridden to 0** (§9) |
| img2img / inpaint | `noise_space::FlowSpace`, `masked_denoise::step_blend` |
| Qwen tokenizer + ChatML | `tokenizers` 0.20; template idiom from `llm/templates.rs` |
| NF4 weights | `nf4_codec`, `nf4_loader` |
| Seed plumbing | `pipelines::seeds::prepare_seed` (mandatory chokepoint for every new dispatch site) |

## 6. The DiT

All names below are the diffusers module paths, which are the safetensors key prefixes in the `-Diffusers` repo.

### 6.1 Embeddings

- **Time.** `time_embeddings.{in_layer,out_layer}`.
  - `freqs = exp(−ln(10000)·arange(1280)/1280)`, then `args = t ⊗ freqs`.
  - The embedding is **`cat([cos(args), sin(args)])`, with cos first**, followed by `Linear(2560→512) → SiLU → Linear(512→512)`.
  - Computed in F32. Here `t` is the raw scheduler timestep in [0, 1000] (§8.2).
- **Text.** `text_embeddings.{in_layer,norm}`: `Linear(3584→2560)` then affine `LayerNorm`.
- **Pooled.** `pooled_text_embeddings.{in_layer,norm}`: `Linear(768→512)` then affine `LayerNorm`. The result is **added to the time embedding**, and that sum is the only global conditioning vector.
- **Visual.** `visual_embeddings.in_layer`: `Linear(1·2·2·16 = 64 → 2560)` over 2×2 spatial patches.
  - Latents are **channels-last** `(B, T=1, H/8, W/8, 16)`.
  - The patchify permute is `(0,1,3,5,2,4,6,7)` followed by `flatten(4..7)`, which makes the patch-internal order `(pt, ph, pw, C)`.
  - The adapter converts from plakat's NCHW once at entry and once at exit.

### 6.2 Positional encoding

- RoPE is applied as **2×2 rotation matrices on adjacent channel pairs** (interleaved), not as the half-split `rotate_half` form. `apply_rotary` reshapes `(…, D) → (…, D/2, 1, 2)`, multiplies by `[[cos, −sin], [sin, cos]]`, and sums the last axis.
- The rotation is computed in **F32**. The reference then casts the result to **BF16 unconditionally** (hard-coded `.to(torch.bfloat16)`), before `type_as`. On the F32/CPU path plakat must decide whether to mirror that rounding; see §12.3.
- **Text RoPE (1D).** Uses `head_dim = 128`, `max_pos = 1024`, and positions `0..len(prompt tokens)`.
  - Positive and negative prompts each get their own positions.
- **Visual RoPE (3D).** `axes_dims = [32, 48, 48]` with `max_pos = 128` per axis.
  - Positions are `t ∈ {0}`, `h ∈ 0..H/16`, `w ∈ 0..W/16`, with `scale_factor = (1, 1, 1)`.
  - The largest bucket (1408 px) gives 88 positions, which is within the 128-position table.
  - The concatenation order of the per-axis args is `[t | h | w]`.
- **Cross-attention has no RoPE.**

### 6.3 Attention

- `to_query/to_key/to_value/out_layer` are all biased, with 20 heads.
- `query_norm` and `key_norm` are `RMSNorm(128)` with learned weights, **computed in F32**, applied before RoPE.
- Attention itself is plain SDPA with no mask. The NABLA block mask path is unused (N2).

### 6.4 Blocks

**Text encoder blocks** (`text_transformer_blocks.{0,1}`) take 6 modulation params from `text_modulation.out_layer` (`SiLU → Linear(512 → 6·2560)`), chunked as `[shift, scale, gate] × {attn, ff}`:

```
x ← x + gate_a · SelfAttn_rope( LN(x)·(1+scale_a) + shift_a )
x ← x + gate_f · FF( LN(x)·(1+scale_f) + shift_f )
```

**Visual decoder blocks** (`visual_transformer_blocks.{0..49}`) take 9 params from `visual_modulation.out_layer`, chunked `[shift, scale, gate] × {self, cross, ff}`:

```
v ← v + gate_s · SelfAttn_rope3d( LN(v)·(1+scale_s) + shift_s )
v ← v + gate_c · CrossAttn( LN(v)·(1+scale_c) + shift_c , text )
v ← v + gate_f · FF( LN(v)·(1+scale_f) + shift_f )
```

Common details:

- LayerNorms are non-affine.
- FF is `Linear(2560→10240, no bias) → GELU (exact, not tanh) → Linear(10240→2560, no bias)`.
- **Every modulate and residual add runs in F32**, then is cast back to the activation dtype. This is the same F32-island discipline as Sana's linear-attention reduction.

**The text stream is time-modulated.** The text blocks consume `time_embed`, so they must be **re-run every step** for both the conditional and unconditional branches. This is cheap: 2 blocks over ≤512 tokens. Do not cache their output across steps.

### 6.5 Output layer

`out_layer.modulation.out_layer` produces 2 params (shift, scale). The layer computes non-affine `LN·(1+scale)+shift` in F32, then `out_layer.out_layer: Linear(2560 → 64)`. The unpatchify permute is `(0,1,5,2,6,3,7,4)` followed by flatten. The output is **velocity**.

### 6.6 Precision

`_keep_in_fp32_modules = [time_embeddings, modulation, visual_modulation, text_modulation]`. In the VarBuilder, load those prefixes as F32 regardless of the target dtype. Everything else follows the device default (BF16 on GPU, F32 on CPU), as Sana does today.

## 7. Text encoding

### 7.1 Qwen2.5-VL as a text-only Qwen2

- The checkpoint keys are `model.embed_tokens`, `model.layers.N.*`, `model.norm`, `lm_head`, and `visual.*`.
- candle's `qwen2::Model::new` does `vb.pp("model")`, so the keys map 1:1.
- Load **`qwen2::Model`, not `ModelForCausalLM`**. This skips `lm_head` (≈1.1 GB).
- The `visual.*` tensors are never touched, because they are never read from the mmapped safetensors.

**Config translation.** `text_encoder/config.json` is a `Qwen2_5_VLConfig`. Depending on the transformers version that wrote it, the LM fields are either top-level or nested under `text_config`. Add a small serde shim that accepts both and fills `qwen2::Config`:

- hidden 3584, intermediate 18944, 28 layers, 28 heads, 4 KV heads
- `rope_theta` 1e6, `rms_norm_eps` 1e-6, vocab 152064
- `tie_word_embeddings` false, `use_sliding_window` false, `hidden_act` silu

**M-RoPE.** Qwen2.5-VL uses multimodal RoPE (`mrope_section`). For text-only input, the temporal, height and width position ids are identical, which collapses to standard 1D RoPE. candle's qwen2 is therefore exact here. This is an **invariant to assert in a test**, not an assumption to rely on silently (§12.2).

**Mask.** Passing `attn_mask` to `qwen2::Model::forward` builds a *bidirectional* padding mask. The HF model is causal. Encode each prompt as **batch-of-1 with `attn_mask = None`**, which takes the causal path. This is equivalent to diffusers' `cu_seqlens` variable-length packing, since no padding exists.

**Template.** Use this string verbatim, including the upstream typo "promt", because the model was trained with it:

```
<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>
```

Encoding steps:

1. Tokenize the full string.
2. Truncate to `41 + max_seq` tokens.
3. Run the model and take the last hidden state, which is post-final-RMSNorm (`hidden_states[-1]` in HF equals candle's `forward` return).
4. **Slice from token 41**. The slice keeps the trailing `<|im_end|>`.

Offset 41 is the token length of the system prefix *under the reference tokenizer*. A P1 test asserts plakat's `tokenizers` 0.20 produces the same 41 (Appendix A, T7).

**Truncation.** When the user's prompt exceeds 512 tokens, warn with the dropped tail, as diffusers does. This hooks into `compile`'s existing token-budget lint.

### 7.2 CLIP-L pooled

- Load `text_encoder_2/` into `vendored_clip::ClipTextTransformer`.
- Tokenize with `max_length = 77`, `padding = "max_length"`, truncate.
- Use `pooler_output`, i.e. the final-LN hidden state at the EOT position. This is the same computation as `flux.rs::encode_prompt`'s `clip_pooled`. Factor it into a shared helper rather than duplicating it.

### 7.3 Negative prompt

- When guidance > 1, the negative prompt (default `""`) goes through **both** encoders with the same template.
- The resulting embedding has a nonzero length: the template tail after offset 41.

### 7.4 Embedding batch

Staged residency (§10) makes "encode, then swap models" the expensive step. Therefore:

- For `--count N`, encode once.
- For `scenario` and `compile` batches, **encode all prompts and negatives first**, release the encoders, then denoise all.
- The `scenario` runner gets a family hook, `pre_encode(&[Prompt]) → Vec<Embeds>`. Other families implement it as a no-op.

## 8. Sampling

### 8.1 Schedule

```
sigmas = sana::flow_sigmas(steps, shift = 5.0)     // double-shift semantics already match diffusers
```

`flow_sigmas` reproduces diffusers' init-time floor shift followed by `set_timesteps` shift. KANDINSKY-1 adds a unit test pinning the first and last three sigmas at 50 steps against values dumped from diffusers with `shift = 5.0`.

### 8.2 Step

```
t_i      = sigma_i · 1000                // raw, NOT /1000
v_cond   = DiT(x, txt_pos, pooled_pos, t_i)
v_uncond = DiT(x, txt_neg, pooled_neg, t_i)       // only when guidance > 1
v        = v_uncond + g · (v_cond − v_uncond)
x        = x + (sigma_{i+1} − sigma_i) · v
```

The conditional and unconditional branches have different text lengths, so they run as two forwards, not one padded batch. Padding would require masking cross-attention, which the reference does not do.

### 8.3 Noise

- Initial latents are `N(0, I)` with shape `(1, 1, H/8, W/8, 16)`, drawn through `seeds::prepare_seed`.
- Bit-identical seeds against diffusers are **not** a goal, since torch and candle RNGs differ. Parity testing injects the diffusers noise tensor (§12).

### 8.4 Resolution buckets

`--size` snaps to the nearest bucket by aspect ratio, then by area, and prints one line, for example:

```
kandinsky5: --size 1200x800 → 1280x768 (native bucket)
```

`--size-exact` disables snapping, warns that it is off-distribution, and requires dimensions divisible by 16.

## 9. VAE

- **Trap.** The reference decodes with `latents / scaling_factor` and does **not** add `shift_factor`. Encoding (img2img init) is the inverse: `z · scaling_factor`, with no shift subtraction.
  - candle's `fae::AutoEncoder::{encode, decode}` always applies `shift_factor`.
  - Construct the Kandinsky AE with `fae::Config { scale_factor: 0.3611, shift_factor: 0.0, ..fae::Config::dev() }`.
  - Without this, every image gets a consistent offset and a visible color cast.
  - **Verify against the official `kandinskylab/kandinsky-5` repo in P1.** If upstream does use the shift and diffusers is wrong, follow upstream and file a diffusers issue.
- **Dtype.** The VAE runs in F32 (`force_upcast: true`).
- **Memory at large sizes.** The 1408-px buckets put the Metal single-buffer decode near the known OOM edge. Use the existing `tiled` decode path when `generation_estimate_gb` exceeds budget.

## 10. Memory plan

### 10.1 Staged residency

| Stage | Resident | BF16 GB |
|---|---|---|
| Encode | Qwen text tower + CLIP-L | ≈14.4 |
| *release encoders* | | |
| Denoise | DiT + VAE | ≈12.2 + activations |
| Decode | VAE (DiT released if `--low-mem`) | ≈0.2 + decode buffer |

The peak is ≈15 GB plus activations, versus ≈27 GB with everything resident.

This is a **new pattern for plakat**: Flux, Sana and SD3 keep their encoders resident. It is implemented locally in `kandinsky.rs`:

- encoders are owned in an `Option<…>` and `take()`-dropped after §7.4;
- no global offload framework is introduced.

`--keep-encoders` opts back into full residency for interactive TUI sessions, where re-prompting is frequent and memory allows.

### 10.2 Quantized tier (P4)

| Piece | Format | GB |
|---|---|---|
| Qwen text tower | Q4_K_M GGUF of Qwen2.5-VL-7B-Instruct (LM part) | ≈4.7 |
| DiT | NF4 via `nf4_codec`, F32 islands (§6.6) left unquantized | ≈3.4 + ≈0.6 |
| CLIP-L, VAE | unchanged | ≈0.4 |

The peak is ≈6 GB, which targets 16 GB unified-memory Macs.

candle's `quantized_qwen2::ModelWeights::forward` returns last-token logits only. Add `vendored_qwen2q.rs`, a fork exposing the post-norm hidden states. This is the same move as `vendored_gemma2` and `vendored_t5`.

Acceptance for the quantized encoder, against the BF16 encoder on the 64-prompt verify corpus:

- per-token cosine ≥ 0.98 (mean);
- the `clip_adherence` score drops by ≤ 2%.

The GGUF must come from a mirror that carries the matching `tokenizer.json`, or `tokenizer_repo` points back at the Kandinsky repo. Use the same `ModelDescriptor` split as `llm/aliases.rs`.

### 10.3 Compute

At 1024² the DiT sees 4,096 image tokens.

- One forward ≈ 5.6 × 10¹³ FLOP, so 50 steps with CFG ≈ **5.6 × 10¹⁵ FLOP per image**.
- That is ≈1.6× flux-dev at 28 steps (≈3.5 × 10¹⁵).
- `doctor --capability` must communicate this: the tuning column reads `50 steps × CFG — slowest family; --steps 30 is the quality knee (P5 measures)`.

## 11. CLI and surface

### 11.1 Aliases (`hf::ALIAS_TABLE`)

| Alias | Repo |
|---|---|
| `kandinsky5`, `kandinsky`, `k5`, `kandinsky5-lite` | `kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers` |
| `kandinsky5-pretrain` | `kandinskylab/Kandinsky-5.0-T2I-Lite-pretrain-Diffusers` (N5: listed, generation warns) |
| `kandinsky5-q4` | same DiT + quantized encoder descriptor (P4) |

Detection uses the substring `kandinsky`, checked **before** any generic fallback in `Variant::from_model`.

### 11.2 Flags

These are new, family-scoped, and rejected with a hint on other families:

- `--quantize-qwen` (P4)
- `--dit-nf4` (P4)
- `--keep-encoders`
- `--size-exact`
- `--max-seq <N>` (default 512, hard cap 1023)

Existing flags, honored as-is:

- `--steps` (default 50), `--guidance` (3.5), `--negative`, `--seed`, `--count`
- `--init-image`/`--strength`, `--mask`/`--mask-feather`/`--mask-invert`
- `--etch`, `--out`

These are rejected with a clear message pointing at N4 or N1:

- `--loras`, `--control-spec`, `--ip-adapter`, `--refiner`, `--fast`

### 11.3 Capability row

```rust
ModelMeta { alias: "kandinsky5", native_res: 1024, dtype: "BF16",
            tuning: "staged load (~15 GB peak); → --quantize-qwen --dit-nf4 (~6 GB); 50 steps × CFG", metal_blocked: false },
```

Add `a if a.starts_with("kandinsky") => 4.0` to `gen_base_gb`, and `=> 27.0` (resident) to `rough_weight_gb`.

The staged peak is the honest figure. Add `ResidentEstimate::staged_peak_gb` so `doctor` reports both numbers rather than the sum.

### 11.4 Seams left open

- **LoRA.** The DiT's linear layers go through the same `lora_linear` wrapper constructor Sana uses, with 0 adapters. This keeps KANDINSKY-2 a loader-only change.
- **`compile`.** Add a `ModelFamily::Kandinsky5` prompt profile:
  - natural-language, long-form, no weight syntax (`(word:1.2)` is stripped with a lint);
  - a 512-token budget;
  - Cyrillic passthrough (no transliteration).

## 12. Validation

### 12.1 Parity harness

`tools/kandinsky_dump.py` is a dev-only script and is never shipped in the binary. At a fixed prompt, negative, and size, with injected noise, it writes `.safetensors` dumps of:

1. Qwen token ids, then the sliced hidden states (pos and neg)
2. CLIP pooled (pos and neg)
3. Time embedding at `t ∈ {1000, 500, 1}`
4. Text-stream output after 2 blocks at `t = 500`
5. Visual stream after blocks 0, 24, 49 at `t = 500`
6. Velocity at steps 0, 25, 49
7. Final latent
8. Decoded image

`tests/kandinsky_parity.rs` (ignored by default, run with `--ignored` plus `PLAKAT_PARITY_DIR`) compares stage by stage.

### 12.2 Tolerances

| Stage | F32 (CPU) | BF16 (GPU) |
|---|---|---|
| Token ids | exact | exact |
| Qwen hidden | max-abs ≤ 1e-4 | cosine ≥ 0.999 |
| CLIP pooled | ≤ 1e-5 | cosine ≥ 0.9999 |
| Block outputs | ≤ 1e-3 rel | cosine ≥ 0.998 |
| Velocity (step 0) | ≤ 1e-3 rel | cosine ≥ 0.995 |
| Decoded image | PSNR ≥ 40 dB | PSNR ≥ 32 dB, SSIM ≥ 0.97 |

### 12.3 The BF16 RoPE cast

The reference rounds rotated Q and K to BF16 even in F32 mode.

- The F32 parity run must mirror this, via a `rope_bf16_round` flag that defaults to on for parity runs.
- Measure whether disabling it changes the image (PSNR between the two). If not, drop the rounding on CPU and leave a comment.

### 12.4 CI

- Default CI (`--no-default-features --lib`) compiles all new code paths. There is no cargo feature gate, following the ETCH-1 lesson about default-on features being CI blind spots.
- Unit tests that need no weights:
  - sigma table
  - bucket snapping
  - config shim (both JSON shapes)
  - patchify/unpatchify round-trip
  - RoPE pair-rotation against a hand-computed 4-dim case
  - the M-RoPE collapse invariant: build the 3-section position ids for a text-only sequence and assert all sections are equal

## 13. Phases

Each phase is independently mergeable.

### Phase 0 — Surface

- Aliases, `Variant::Kandinsky5`, the detection order, and a dispatch stub that bails with "Kandinsky 5 pipeline lands in P1".
- Capability row, bench and verify-tier2 family arms, `error_hints` entries.
- Bucket snapping, plus `--size-exact` and the other family-scoped flags with rejection hints.
- **Gate:** `doctor --capability` lists the family; every other family's verify-tier2 output is byte-identical.

**Phase 0 as built** (`src/pipelines/kandinsky.rs` + the seams above):

- `doctor --capability` lists `kandinsky5` with the verdict **`pending`** (a new verdict value: registered, pipeline not landed) and `staged_peak_gb` in the JSON, rather than promising a fit for something that cannot run yet.
- Detection resolves short aliases: `k5` does not carry the substring, so `is_kandinsky` falls back to the resolved repo.
- `generate --model kandinsky5` validates the family's flags, snaps `--size` to a bucket (printing the line in §8.4), then stops with "Kandinsky 5 pipeline lands in P1". `bench` and `t2i::run` stop at the same message.
- Deferred, with reasons: the **verify-tier2 `Family` arm** waits for P3 (an arm with no runnable spec is dead code); the **`kandinsky5-q4` alias** waits for P4; **`--ip-adapter`** is not a `generate` flag today, so there was nothing to reject.
- The tier-2 half of the gate was not run: Phase 0 adds one early family check ahead of the existing dispatch and touches no other family's numerics. The lib test gate (2185 tests) is green.

### Phase 1 — Text encoders and VAE

- `kandinsky_text.rs`: the config shim, template and offset slice, causal batch-of-1 encode, and the CLIP-L pooled helper (factored out of `flux.rs`).
- AE construction with `shift_factor = 0`, and the upstream-repo check from §9.
- **Gate:** parity stages 1–2; T7 (offset 41); the VAE round-trip on a dumped latent at PSNR ≥ 40 dB.

**P1 as built** (`src/pipelines/kandinsky_text.rs`, `kandinsky::run_p1`, `tools/kandinsky_dump.py`), with what checking the references changed:

- **Two templates, not one.** The upstream training code (`kandinskylab/kandinsky-5`, `text_embedders.py`) uses "promt" with `crop_start` 41, as §7.1 says. But diffusers `main` has since *corrected the typo* and uses `prompt_template_encode_start_idx = 40`. plakat defaults to the **upstream** template (what the weights were trained with) and carries the diffusers one for parity runs (`PLAKAT_K5_TEMPLATE=diffusers`); the dump tool records which was used and the parity test follows it. T7 was run against the repo's real `tokenizer.json`: 41 and 40 tokens respectively.
- **The VAE is in the diffusers layout.** `vae/diffusion_pytorch_model.safetensors` is an `AutoencoderKL`, which candle's BFL-layout `flux::autoencoder` cannot read. It loads with `stable_diffusion::vae::AutoEncoderKL` exactly as plakat's SD3 path does (same shape: 4 blocks, 16 latent channels, no quant convs), and the scale is applied by hand: `decode(z / 0.3611)`, `encode(x) · 0.3611`. So §9's "construct `fae::Config` with `shift_factor = 0`" is not what was built; the effect is the same.
- **§9's check is done:** upstream `generation_utils.py` also uses `scaling_factor` alone on both encode and decode. No shift; T1 stands.
- **CLIP-L is a 1.71 GB file**, not ≈0.25 GB: `text_encoder_2/model.safetensors` is the whole CLIP ViT-L/14. Only `text_model.*` is read. A prompt over 77 CLIP tokens is cut with the EOT as its last token, as the reference tokenizer does — its own helper (`clip_ids_77`), so `flux.rs` was left untouched rather than refactored (§7.2's shared helper is not done).
- **The Qwen download skips a shard.** The index puts every `model.*` tensor in shards 1–4; shard 5 (1.09 GB) holds only `lm_head`. The loader reads the index and fetches 15.5 GB, not 16.6 GB. The checkpoint's key names match candle's `qwen2::Model` one for one (checked against the index).
- **`generate --model kandinsky5` now runs P1:** loads the encoders, encodes prompt and negative, prints token counts and statistics, releases the encoders, then stops. `PLAKAT_K5_DUMP_DIR` writes the tensors; `PLAKAT_K5_VAE_IMAGE` round-trips an image through the VAE; `PLAKAT_K5_STAGE=vae` does that alone.
- **Measured on the 24 GB Mac:** the VAE round-trip of a photograph is 34.3 dB at 512² (Metal) and 37.2 dB at 1024² (CPU). At **1024² on Metal the F32 VAE fails** with "Failed to create metal resource: Buffer" — the single-buffer cap bites at the family's base size, not only at the 1408-px buckets §9 names. P3's tiled decode fallback is needed from 1024² up on this class of machine.
- **The parity gate is run and passes** (36 GB M5 Max, the default prompt and empty negative at 1024², `upstream` template; the ignored test `kandinsky_parity_p1` with `PLAKAT_PARITY_DIR` set to a `tools/kandinsky_dump.py --out DIR` dump). Token ids match on both prompts.

  | Stage | CPU, F32 | Metal, BF16 weights |
  |---|---|---|
  | Qwen hidden, prompt | cosine 1.000000, max-abs 1.22e-3 | cosine 0.999999, max-abs 2.45e-1 |
  | Qwen hidden, negative | cosine 1.000000, max-abs 5.84e-4 | cosine 0.999999, max-abs 1.17e-1 |
  | CLIP-L pooled | cosine 1.000000, max-abs ≤ 1.1e-5 | cosine 0.99993 / 0.99994 |
  | VAE decode of a dumped latent | 108.1 dB | 120.0 dB |

- **The Qwen tower is vendored, and computes in F32** (`src/pipelines/vendored_qwen2.rs`). With candle's `qwen2::Model` run entirely in BF16 on Metal the gate FAILED: cosine 0.945 on the prompt and 0.901 on the negative against the F32 reference, where transformers' own BF16 run (MPS) holds 0.99993 and 0.99964. The port's logic was right — the same code in F32 on CPU was exact — so it is precision: candle rounds every intermediate to BF16. (candle also builds the RoPE angles in the model dtype; fixing that alone changed nothing, 0.939 / 0.901.) The vendored tower keeps the weights in BF16 (what the checkpoint stores, ≈14 GB) and widens each layer's weights to F32 for that layer's forward only, so activations are F32 and the result is the F32 reference's. It is encoder-only: no KV cache, no mask path (T3 cannot be reached). A small random-tower test (`qwen2_gpu_dtypes_track_cpu_f32`, ignored, needs a GPU) pins it without the checkpoint.
- **The F32 tolerances are relative.** §12's absolute max-abs bounds were not reachable: Qwen's hidden states peak near 117, and F32 after 28 layers lands at ≈1e-3 absolute (1e-5 relative). The test holds F32 to max-abs / reference peak ≤ 1e-4 (Qwen) and ≤ 1e-5 (CLIP); the GPU bars are unchanged (cosine ≥ 0.999 and ≥ 0.9999).
- **The 1024² F32 VAE decode runs on Metal on the 36 GB host** — the single-buffer failure above is the 24 GB machine's.
- **One thing for P3:** upstream's own sampler defaults to `scheduler_scale = 3.0` in `t2i_pipeline.py`, while the diffusers repo's scheduler config says `shift: 5.0`. The port follows the diffusers config (§8.1); worth a look when the first images exist.

### Phase 2 — DiT

- `kandinsky_dit.rs` per §6, with F32-island loading.
- **Gate:** parity stages 3–6 at both precisions, and on Metal as well as CPU (§14, the rank > 4 risk).

**P2 as built** (`src/pipelines/kandinsky_dit.rs`, `tools/kandinsky_dump.py --stage p2`), with what the measurements changed:

- **The DiT is §6 as written**, and it matched the reference on its first run, on the CPU and on Metal. The rank > 4 risk (§14) did not bite: attention is flattened to `B·heads` (3-D matmuls, a few heads at a time so one score tensor stays under 1 GiB), the RoPE rotation is done at rank ≤ 4, and the rank-6 patchify / unpatchify permutes run on the CPU once per forward. One thing §6.1 does not say: a patch goes IN as `(ph, pw, C)` but comes OUT of the output layer as `(C, ph, pw)`, so the two are not inverses of each other.
- **The weights rest in BF16 on every device** (the checkpoint is BF16 throughout, 12.0 GB), and a layer widens its weights to the activation dtype for the call. §6.6's "load the F32 islands as F32" became "widen them at use" — the same numbers, without holding 2.4 GB of F32 modulation weights. The CPU path is therefore 12 GB resident, not 24.
- **The compute dtype defaults to F32 on a GPU too** (`PLAKAT_K5_DIT_COMPUTE=bf16` opts out). It is nearly free — 8.1 s against 7.4 s for a 1024² forward on the M5 Max — and it is the difference between matching the reference and approximating it (table below). Both pass the GPU bars; this is P1's Qwen finding again in a milder form.
- **§12.3 is measured: the BF16 rounding of the rotated Q/K stays on.** With it, the text stream and visual block 0 are within 7e-5 of the reference (relative max-abs); with `PLAKAT_K5_ROPE_ROUND=0` they are 2.4e-3 and 2.5e-4 off. So the reference's rounding is mirrored on every path, F32 included. Whether it changes the *image* is P3's question.
- **§12.2's F32 bound for blocks and velocity is restated.** "≤ 1e-3 relative" holds for the time embedding, the text stream and block 0. Deeper it is not reachable against a reference computed on another backend: that same rounding turns 1e-6 differences into 4e-3 ones in a few Q/K elements, and fifty blocks carry them (velocity at step 25: 1.8e-3 on Metal). The deep stages are held to cosine ≥ 0.9999 in F32; the BF16 bars are unchanged (0.998 blocks, 0.995 velocity).
- **The reference dump is its own loop**, not the pipeline's `__call__`: it reads the embeddings back from `p1.safetensors` (no text tower), calls the transformer and the scheduler directly, and taps what §12.1 lists, plus the final latent and the decoded image for P3 (stages 7–8). 100 forwards at 1024² in F32 on MPS take 14 minutes. The decoded reference is the expected image.
- **The gate, run and passing** (36 GB M5 Max; default prompt, empty negative, seed 42, 1024², 50 steps, guidance 3.5; the reference is F32 on MPS). Cosine to the reference, with relative max-abs in brackets:

  | Stage | Metal, F32 compute | Metal, BF16 compute |
  |---|---|---|
  | Time embedding, t ∈ {1000, 500, 1} | 1.000000 (≤ 1.3e-6) | the same (an F32 island) |
  | Text stream after its 2 blocks, t = 500 | 1.000000 (6.6e-5) | 0.999965 |
  | Visual block 0 / 24 / 49, t = 500 | 1.000000 (6.3e-5 / 5.1e-5 / 3.1e-4) | 0.999991 / 0.999924 / 0.999966 |
  | Velocity, t = 500 | 1.000000 (4.0e-4) | 0.999930 |
  | Velocity at step 0, cond / uncond | 1.000000 (2.8e-4 / 8.4e-5) | 0.999934 / 0.999950 |
  | Velocity at step 25 | 1.000000 (1.8e-3 / 1.1e-3) | 0.999967 / 0.999972 |
  | Velocity at step 49 | 1.000000 (7.0e-4 / 7.7e-4) | 0.999544 / 0.999550 |

  On the **CPU** (F32, 69 s a forward) every stage is at cosine 1.000000: the time embedding within 2.3e-7, the text stream 4.9e-5, blocks 0 / 24 / 49 at 6.3e-5 / 4.9e-5 / 3.5e-4, and the velocities between 9.0e-5 and 1.6e-3 (step 25).
- **For P3:** the reference passes the timestep through the transformer's dtype, so a BF16 pipeline hands the DiT a BF16-rounded `t`; plakat passes it in F32. And the DiT takes one image and one prompt per forward — the two CFG branches differ in length (§8.2), so they are two forwards.

### Phase 3 — End-to-end and staged residency

- `kandinsky.rs` orchestration, §7.4 batching and the scenario `pre_encode` hook, `--keep-encoders`, and tiled decode fallback.
- `parameters` sidecar fields: `family=kandinsky5`, `max_seq`, `bucket`.
- **Gate:** parity stages 7–8. Peak RSS measured on 1024² and 1408×640 at ≤ 16 GB plus activations on CUDA and Metal. G1 and G2 met.

**P3 as built** (`src/pipelines/kandinsky.rs`; `t2i::run`, `cli/generate.rs` and `cli/bench.rs` dispatch to it). Parity and memory both pass on Metal; CUDA is unmeasured.

- **It generates.** `kandinsky::run` is the CLI's entry and `run_jobs(settings, jobs)` the batching one (§7.4): every distinct prompt and negative is encoded once, the encoders are dropped (`--keep-encoders` holds them), the DiT denoises every job and is dropped, the VAE decodes. The sampler is `sana::flow_sigmas(steps, 5.0)`, `t = σ·1000`, CFG as two forwards when guidance > 1, Euler. The clap defaults `--steps 28` / `--guidance 7.5` read as "unset" and become 50 / 3.5. Files are `plakat-kandinsky5-{seed}.png`; the sidecar carries `family`, `max_seq` and `bucket`. P1's developer hooks (`run_p1`, `PLAKAT_K5_DUMP_DIR`, `PLAKAT_K5_VAE_IMAGE`, `PLAKAT_K5_STAGE`) are gone.
- **Parity stages 7–8 pass on Metal, F32 compute** (`kandinsky_parity_p3`; 1024², 50 steps, guidance 3.5, seed 42, against the F32 MPS reference): the latent into steps 0 / 25 / 49 is at cosine 1.000000 / 1.000000 / 0.999981, the final latent 0.999979, the image **55.4 dB** PSNR (the VAE alone, on the reference's latent: 120 dB). 771 s for the 100 forwards. The bars are cosine ≥ 0.999 and ≥ 40 dB in F32, ≥ 0.99 and ≥ 32 dB in BF16. A 256², 6-step reference does *not* pass (final latent 0.963 with every velocity matching): off the trained resolutions the trajectory is chaotic, so the gate is run at a real bucket.
- **G2 is met on Metal: the peak footprint is 16.1 GB at every bucket.** Measured with `/usr/bin/time -l` on the 36 GB M5 Max, release build, F32 compute, OOM guard on:

  | Run | Denoise | Peak footprint |
  |---|---|---|
  | 1024², 50 steps | 767 s (15.3 s a step) | 16.1 GB |
  | 1408×640, 10 steps | 131 s | 16.1 GB |
  | 640×1408, 10 steps | 131 s | 16.1 GB |
  | 768×1280, 10 steps | 146 s | 16.1 GB |
  | 1280×768, 10 steps | 146 s | 16.1 GB |
  | 896×1152, 10 steps | 155 s | 16.1 GB |
  | 1152×896, 10 steps | 155 s | 16.1 GB |

  The peak does not move with the image size: it is the text-encoder stage (15.0 GB by the probe), not the DiT (11.3 GB loaded, 14.6 GB denoising at 1024²) and not the decode. "Maximum resident set size" reads 25–28 GB on the same runs and is not the number to watch — it counts the memory-mapped checkpoint pages, which the footprint does not. The first build of P3 did **not** meet this (31–45 GB, and four of seven buckets killed by the guard at the decode); two things were wrong, both found with `kandinsky_memory_probe` (ignored test; it prints `memwatch::footprint_gb()` stage by stage):
  - **Dropping a model on Metal does not return its memory.** candle keeps freed buffers in two pools. The one weights sit in is swept only when a command buffer fills, so the encoders were still resident while the DiT loaded (26.1 GB) and the DiT while the VAE decoded. The one intermediates sit in is never swept. The fix is `stage_device`: each stage runs on its own `Device::new_metal(0)`, and dropping the stage drops its pools — the footprint is back to 0.1 GB a second after the encoders go. What crosses a stage boundary (embeddings, latents) goes through the CPU, because **one tensor left on a stage's device pins that device's whole pool**; the probe itself did this once and held 14.6 GB.
  - **candle's whole-image VAE decode on Metal takes 38 GB at 1024²** (footprint 14.6 → 52.6 GB). In tiles it takes 10.8 GB and is faster (11.7 s against 19.5 s).
- **On Metal the VAE therefore decodes on the CPU, whole** (`decode_device`). Measured on the reference's final latent at 1024², against the reference's image:

  | Decode | Time | Memory | PSNR |
  |---|---|---|---|
  | CPU, whole (the default on Metal) | 29.5 s | +10.0 GB | 101.8 dB |
  | GPU, tiles 64 / stride 48 | 11.7 s | +10.9 GB | 35.7 dB |
  | GPU, tiles 64 / stride 32 | 11.7 s | +10.9 GB | 36.3 dB |
  | GPU, tiles 96 / stride 64 | 12.4 s | +27.5 GB | 40.9 dB |
  | CPU, tiles 64 / stride 48 | 64.1 s | +2.9 GB | 35.7 dB |

  Tiles were the first fix and they cost the image: 35.7 dB is under §12.2's 40 dB bar on their own, before the loop adds anything (an earlier 44.5 dB figure was taken on a 3-step latent and flattered them). They stay as the out-of-memory fallback of the whole decode and as an opt-in: `PLAKAT_K5_VAE_TILED=1` decodes in 64 / 48 tiles (on the GPU when this was measured; on the CPU since P4, where they take +2.9 GB). With the CPU decode a 1024² run through the CLI still peaks at 16.1 GB.
- **Stage 7–8, re-run with the CPU decode, passes in both compute dtypes** (Metal, 1024², 50 steps): F32 — final latent 0.999979, image 55.4 dB, 766 s; BF16 (`PLAKAT_K5_DIT_COMPUTE=bf16`) — latent into steps 25 / 49 at 0.999938 / 0.997630, final latent 0.997511, image 32.8 dB, 715 s. The VAE alone is 101.8 dB. BF16 clears its 32 dB bar by 0.8 dB and saves 7 % of the time, which is why F32 is the default. With the tiles as the default the same test failed on the image alone: 35.7 dB in F32 and 31.3 dB in BF16.
- **Unexplained: the time outside the denoise.** The runs above took 514–1151 s of wall time for 131–767 s of denoising, and a later 10-step 1024² run took 3123 s for a 154 s denoise. The weights are on an external USB volume that read at 22 MB/s when measured, which would put the 26 GB of checkpoints at 20 minutes; that is the likely cause, not a confirmed one, and it has not been measured on an internal disk.
- **Not done, with the reason.** The scenario `pre_encode` hook: `scenario` has its own per-family dispatch (it does not dispatch Sana either), so `run_jobs` is ready for it but nothing calls it. CUDA peak memory cannot be measured on this host. CPU stage 7–8 was not run (≈ 2 hours). §12.3's image-level question (does the Q/K rounding change the picture) is unmeasured. Upstream's own config says `scheduler_scale 3.0` where diffusers ships shift 5.0; plakat follows diffusers, the reference it is checked against.
- `bench` has a `kandinsky5` arm (encode once, then time `denoise` + `decode`), and `doctor --capability` reports the family as running.

### Phase 4 — Quantized tier

- `vendored_qwen2q.rs`, the GGUF descriptor, and the DiT NF4 path.
- **Gate:** the §10.2 acceptance numbers; a peak of ≤ 7 GB on a 16 GB Mac. G3 met.

**P4 as built** (`vendored_qwen2q.rs`, `kandinsky_text.rs`, `kandinsky_dit.rs`, `nf4_codec.rs`, `kandinsky.rs`). Both gates pass on Metal, the acceptance one in a reduced form; the 16 GB machine is simulated on the 36 GB host.

- **The Qwen tower** (`--quantize-qwen`) is `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` from `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` (4.68 GB, the language model only). `vendored_qwen2q::Model` is `vendored_qwen2` with `QMatMul` layers: one causal forward, every position's hidden state after the final norm, F32 activations. Two things differ from candle's `quantized_qwen2`: the metadata prefix is read from `general.architecture` (llama.cpp writes `qwen2vl` for this model, not `qwen2`), and the token embedding is dequantized once and kept on the CPU in F16 (1.1 GB) instead of on the device in F32.
- **On Metal the quantized tower runs on the CPU.** Measured on the P1 reference prompt: CPU +5.7 GB, 3.8 s for two texts, per-token cosine 0.9940; GPU +7.9 GB, 1.0 s, 0.9950. The CPU is 2.2 GB cheaper and the 3 seconds are nothing against the denoise.
- **The DiT** (`--dit-nf4`) is quantized at load from the BF16 checkpoint: there is no NF4 checkpoint of this model to download. `Lin::Nf4` covers the attention and feed-forward linears of all 52 blocks; the embeddings, modulations, time layers and the output layer stay dense BF16. The layout is bitsandbytes' (two codes a byte, absmax per 64 values). Against the reference's `velocity_500` the NF4 forward is at cosine 0.998836.
- **The weights are dequantized on the device, not on the CPU.** A 256 × 2 table maps a packed byte to its pair of codes, so a layer's weight is `pairs.index_select(packed) · absmax` — all in the op-output pool, which is reused. The first build dequantized on the CPU and uploaded: every upload is a fresh buffer in candle's weight pool, which is swept only every 50 computes, and the denoise peaked at 15.5 GB instead of 8.1 GB.
- **Low-memory mode** (`kandinsky::low_memory()`): on under 24 GB of RAM, or with `PLAKAT_K5_LOW_MEMORY=1` (`=0` forbids it). Attention runs in 128 MB chunks instead of 1 GB (denoise peak 8.1 → 6.5 GB; 32 MB chunks gain nothing more), and the VAE decodes on the CPU in tiles (+2.9 GB instead of +10 GB, at P3's 35.7 dB against the whole decode's 101.8 dB). Machines with 24 GB or more keep the whole decode.
- **The memory gate passes: 6.97 GB.** 1024², both flags, 10 steps, `/usr/bin/time -l`, release build, OOM guard on:

  | Run | Denoise | Wall | Peak footprint |
  |---|---|---|---|
  | `PLAKAT_K5_LOW_MEMORY=1` | 184.8 s (18.5 s a step) | 553 s | **6.97 GB** |
  | `PLAKAT_K5_LOW_MEMORY=0` | 184.1 s (18.4 s a step) | 226 s | 11.39 GB |
  | full tier (P3) | 15.3 s a step | — | 16.1 GB |

  By stage (probe): text 5.56 GB, DiT loaded 4.68 GB, denoising 6.47 GB, then the decode on top of 0.7 GB. The peak is the last tile of the decode or the denoise, within 0.5 GB of each other, and it is 0.03 GB under the bar: there is no margin. NF4 costs 20 % a step (18.4 s against 15.3 s) for the per-call dequantization; the smaller attention chunks cost nothing measurable. This is the 36 GB machine with the mode forced, not a 16 GB machine: footprint is the same quantity on both, but swap behaviour and the OOM guard's verdict there are unmeasured.
- **§10.2's encoder bar passes: 0.9937** mean per-token cosine of the Q4 tower against the BF16 one over 64 prompts (worst prompt 0.9921; the bar is 0.98). The empty negative prompt alone is at 0.9799 — one token sequence, under the bar on its own, and it is half of every CFG step.
- **§10.2's adherence bar passes in a reduced run.** 4 prompts, 20 steps, 1024², seeds 42–45, full tier against `--quantize-qwen --dit-nf4`:

  | Prompt | Full | Quantized |
  |---|---|---|
  | a red fox at dawn in soft golden light | 0.2903 | 0.3014 |
  | an old fisherman mending a net in heavy rain at night, neon reflections | 0.3087 | 0.3046 |
  | a glass teapot as a watercolor painting | 0.3293 | 0.3312 |
  | a lighthouse on a cliff in a studio, black background, product photo | 0.3164 | 0.3322 |
  | mean | 0.3112 | 0.3174 |

  The quantized tier scores 2.0 % *higher*; the bar is a drop of ≤ 2 %. On four images that is noise in either direction, not a finding that quantization helps. The images are different pictures of the same prompt (the fisherman gains a hat in one and loses it in the other), both clean.
- **Where the acceptance departs from §10.2.** There is no "64-prompt verify corpus" in the tree; `kandinsky_p4_acceptance` builds its own 8 subjects × 8 settings grid. The adherence half ran on 4 of the 64 prompts at 20 steps because 64 × 2 images at 50 steps is about 28 hours on this machine; `PLAKAT_P4_IMAGES=64 PLAKAT_P4_STEPS=50` runs it in full.
- **The NF4 pack is cached on disk** (added with Phase 6). One file a layer — 512 files, 2.8 GB — under `plakat-nf4` beside the downloaded models, so it follows `--cache-dir`; `--nf4-cache <dir|off>` (env `PLAKAT_NF4_CACHE`, a scenario's `nf4-cache:`) moves it to another disk or turns it off. Keyed by the checkpoint file; a layer a LoRA changes neither reads nor writes it. A file a layer because the packs are produced one layer at a time and holding them all to write one file would cost the low-memory tier 3.3 GB it does not have. Measured on the SSD with the checkpoint warm, 8 steps: 199 s a run reading the cache against 219 s without it, and 243 s for the run that writes it — the quantization is a small part of a warm run, and the gain should be larger where reading 12 GB is slow, which is not measured. The image is the same to the pixel with the cache, without it, and on the run that writes it.
- **Not done, with the reason.** Before that cache, every `--dit-nf4` run read the 12 GB BF16 checkpoint and quantized it, 319 s with the checkpoint cold on the busy USB volume and under 40 s warm (the whole 226 s run above holds a 184 s denoise). So the quantized tier saves memory, not download or disk: a 16 GB machine still needs the 12 GB checkpoint. The `kandinsky5-q4` alias (§11.1) is not registered — the two flags are the interface. The Q8 fallback was not needed. CUDA is unmeasured. Only 1024² was measured in this tier.

### Phase 5 — img2img, inpaint, compile, polish

- `FlowSpace` init with the §9 encode, and `masked_denoise::step_blend` inpaint.
- The `compile` profile, a step-count quality sweep (20/30/40/50) on the verify corpus to set the `--steps` guidance, and README, Book, and `--help` updates.
- **Gate:** G4 and G5. Release notes state plainly that there are no LoRA or ControlNet adapters yet and that this is the slowest family.

**P5 as built** (`kandinsky.rs`, `cli/img2img.rs`, `cli/scenario.rs`, `compile/`, `verify/tier2.rs`, `ui/tui/services/model_service.rs`, `imaging/io.rs`). G4 is met as function, with quality caveats; G5 is met except for the TUI, which was built and unit-tested but not run. Everything below was measured on Metal at 1024², release build.

- **img2img** (`plakat img2img --model kandinsky5`). The source is encoded by the Flux VAE on the decode device (§9: `·scale`, no shift) in its own stage 1b, between the text stage and the DiT. The latent enters at `x = (1 − σ)·z0 + σ·noise`, at the σ the strength maps to (below; the first build entered at step `steps − round(strength · steps)`, σ 0.88 at 0.6, and the table's runs were made that way). The default strength is 0.6.
- **Inpaint** (`--mask`, with `--mask-feather` and `--mask-invert`). The mask is reduced to the latent grid (factor 8); after every Euler step `masked_denoise::step_blend` puts the source, re-noised to that step's σ by a `FlowSpace` over the remaining sigmas, back outside the mask. The default strength is 1.0 — the masked region starts from pure noise. `step_hook::refine` is called after the blend, so the existing step hooks see the blended latent.
- **Measured.**

  | Run | Steps run | A step | Peak footprint |
  |---|---|---|---|
  | img2img, strength 0.6, 30 steps | 18 | 15.38 s | 16.41 GB |
  | inpaint, 30 steps | 30 | 15.58 s | 16.40 GB |
  | img2img, both quantized flags, low-memory mode, whole CPU encode | 6 of 10 | 18.42 s | 8.98 GB |
  | the same, tiled encode | 6 of 10 | 18.36 s | **7.21 GB** |

  The whole CPU encode of a 1024² image cost 2 GB over the denoise's peak, so low-memory mode encodes in tiles (`encode_tiled`: 64-latent tiles at stride 48, ramp-weighted; `PLAKAT_K5_VAE_TILED` overrides either way). The remaining 0.2 GB over P4's 7 GB bar is not explained; txt2img in the same mode is 6.97 GB.
- **Quality, first pass, on one source image.** img2img at strength 0.6 kept the composition almost exactly and did not apply the "watercolor" the prompt asked for. Inpaint left everything outside the mask untouched and filled the mask with a crude flat-sided cabin with a hard edge at the mask boundary. Both were then tuned (two batches, 19 images at 30 steps):
  - **The strength scale was the img2img problem.** Entering the schedule at `strength` of its steps, a photograph came back as the same photograph up to strength 0.9 (σ 0.978; only the snow's detail moved) and first turned into a painting at 0.95 (σ 0.986): at a megapixel the model fixes composition and look while almost no signal is left, which is what shift 5.0 is for. At 30 steps that left three usable settings. `--strength` now maps to the start sigma by `shift_t(strength, 23)` (`STRENGTH_SHIFT`, fitted to put that transition at 0.75), and the run takes the reference schedule's spacing over the part below it (`run_sigmas`): 0.3 → σ 0.908, 0.5 → 0.958, 0.6 → 0.972, 0.75 → 0.986, 0.85 → 0.992; strength 1 is the reference schedule exactly. The start sigma no longer depends on the step count. Re-run on two sources: the fox is the same photograph at 0.3 and 0.6, a watercolour in the same pose at 0.75, a looser one at 0.85; a photograph of a fisherman is an oil painting with its composition at 0.6 and recomposed at 0.75 and 0.85. So the scale is spread over its range, and where the medium flips depends on the picture (0.6 for one, 0.75 for the other). The default stays 0.6. The cost: 26 of 30 steps run at 0.6, so a low strength no longer saves much time.
  - **The prompt and the mask edge were the inpaint problem, not the blend.** The model has no inpaint conditioning; with a prompt for the patch alone ("a small wooden cabin…") it plans a whole cabin picture and the mask keeps a fragment of it. With a prompt for the whole picture (the fox, and a cabin behind it) and a 48 px feather the cabin is in the scene — scale, light and depth of field right, no seam; the same prompt at 8 px leaves a faint seam, and the patch prompt at 48 px leaves a haze around the cabin. Strength 0.9 (old scale) was no better than 1.0. `DEFAULT_MASK_FEATHER = 48` for this family, in the CLI (an unset `--mask-feather`, read as its default of 8) and the TUI; the prompt rule is in IMG2IMG.md.
  - **Limits of the tuning.** Two sources, one mask, one seed, 1024² only, judged by eye. `STRENGTH_SHIFT` is a fit to one transition on one picture.
- **`compile`.** `ModelFamily::Kandinsky5` (any model name containing `kandinsky`, or `k5`): long-form prose, no weight syntax, no quality boosters, a 512-token budget, the prose and relationship reinforcement the other transformer families get, and a lint when a verbatim prompt still carries `(term:N)`.
- **`scenario`.** A Kandinsky arm at the head of the dispatch: each task snaps to its bucket and takes the family's steps and guidance when the scenario sets none. LoRAs in a task are refused. The first build ran each task as its own `run_jobs` batch: a two-task scenario (1024² and 1280×768, 8 steps, `--etch`) took 2023 s of wall time for 241 s of denoising, peak 16.20 GB — the checkpoints were read from disk again for each task.
- **One run for the whole scenario.** A task's prompt is only known inside the task loop (the enhancer runs there), so the encode cannot be hoisted in front of it; the generation is moved behind it instead. Each Kandinsky task queues its jobs, and one `run_jobs` after the last task renders them all: one load of the text encoders, one of the DiT, one of the VAE. The same scenario: **603 s** for 237 s of denoising, peak 16.13 GB. A task with a pass of its own right after generation (ranking, artefacts, `style`, upscale) needs its images at once and still runs in place. Costs: the images appear at the end, not task by task; and if the one run fails, every queued task is marked failed after the status board has already shown it as finished.
- **`--etch`, run** (on the first build's scenario). `doctor --if-plakat` reads both images as generated: L0 manifest present, L1 pixel etch 16/16 tiles, L3 fingerprint match at cosine 1.000. The L2 latent etch was not checked (its read needs a model).
- **`generate` on prompts written for other families.** Found on a real prompt with `(blue grass:1.6)` and `--enhance deepseek`. The model has no weight parser, so `generate` now takes the `(term:N)` wrappers off and appends `prose_reinforcement`'s restatement, as `compile` does, and prints the prompt it uses. And `--enhance` used the generic brief ("add detail, under 70 tokens"), which returned SD-style tags; on this family it now uses `assembler::enhance_system` — the family section plus "keep every element and colour, turn weights into words" — unless `--enhance-system` is given. On that prompt DeepSeek returned three paragraphs with every element kept, and the 30-step image has all of them (blue grass, white flowers, umbrella acacias, the two-rut road, blue bushes, a green sky, an orange sun). The first wording of the language rule ("the model reads Russian as well as English") made it translate an English prompt into Russian; reworded to "the same language as the input", English stays English and Russian stays Russian (one prompt each).
- **TUI.** `UiFamily::Kandinsky5` loads per generation through `run_hooked` with the TUI's step hook, and passes an init image and mask through. Covered by unit tests; **not run** — the TUI cannot be driven from the development session.
- **Verify tier 2.** `kandinsky5` renders 512², 8 steps, guidance 3.5. The harness's shared deterministic latent is uniform in [−1, 1), which a flow model turns into a flat grey field — the first golden was one, and passed while testing nothing. Under `PLAKAT_VERIFY_DET_INIT` the family now draws a unit Gaussian from SplitMix64 and Box–Muller instead. The golden authored that way on the CPU (32.95 s a step, 29.2 GB footprint) is a real picture, and the Metal render matches it at SSIM 0.9969 and mean_abs 1.648 against bars of 0.97 and 4 (3.47 s a step).
- **Refused, with a message:** in `generate`, `--quality`, `--adetailer`, `--hires-fix`, `--artefact`, `--grid` and non-PNG `--format` (added to P0's list); in `img2img`, `--lora`, `--control*`, `--tiled`, `--artefact` and `--grid`. `naturalize`'s repaint falls back to SDXL for this family, as for the other transformers.
- **A bug in every family, found by the Russian prompts.** `save_rgb_u8_with_metadata` wrote the `parameters` chunk as `tEXt`, which is Latin-1: a Cyrillic prompt failed the PNG header after the whole render (the first step sweep lost an hour to it). `imaging::io::add_png_text` now writes `iTXt` when the text is not Latin-1 and `png_text` reads either; `etch::detect` and the book-art canvas use them.
- **A correction to what follows, found in Phase 6.** candle's Metal generator does not repeat a seeded draw (see Phase 6), so the images below that were meant to share a seed's noise did not: the step counts of the sweep and the two languages of a pair each started from different noise. The adherence and aesthetic means stand as measurements of four images each; the PSNR column compares different noise and says nothing about what the step count changes. Re-checked on one prompt at one seed after the fix: 20 steps against 50 is 20.2 dB, the same composition and pose with shifted details — a 20-step draft is a preview of the 50-step picture, which the P5 note below denied. The img2img and inpaint tuning is not affected in kind — it was judged by eye across strengths — but its ladders were not one noise either.
- **Step sweep** (4 prompts, seeds 42–45, meant to be the same noise at every step count; 16 images, 10 245 s):

  | Steps | Adherence (CLIP) | Aesthetic | PSNR to the 50-step image |
  |---|---|---|---|
  | 20 | 0.3028 | 6.055 | 15.6 dB |
  | 30 | 0.3025 | 6.067 | 17.2 dB |
  | 40 | 0.2889 | 5.929 | 15.9 dB |
  | 50 | 0.2872 | 5.986 | — |

  No step count is measurably or visibly worse: the 20-step images are finished pictures, not drafts of the 50-step ones. The low PSNR is the other finding — the composition holds across step counts but the details do not (a samovar's tap changes side, a pier gains a railing), and 40 steps is no closer to 50 than 20 is. So there is no convergence to buy with more steps. **Guidance: the default stays 50, the reference's; `--steps 30` for ordinary work, 20 for drafts.** Four prompts cannot rank the counts, and the differences in the table are within their spread (per-prompt adherence at 20 steps runs 0.283–0.350).
- **Q3, the bilingual set.** The same four prompts in Russian, at 20 steps, scored against the *English* text: 0.3055 from Russian, 0.3028 from English (+0.9 %; per prompt 0.300 / 0.284 / 0.287 / 0.351 against 0.283 / 0.286 / 0.292 / 0.350). By eye the Russian set is as faithful — the fox, the fisherman with his net, the samovar with its string of баранки, the watercolour church. `compile`'s profile therefore keeps a Russian source in Russian. Four pairs support "Russian prompts work"; they do not support a claim of a cultural advantage, and none is made.

### Phase 6 — LoRA (after the RFC: N4 lifted)

N4 said no public adapters existed. For images that is still so — the 25 Kandinsky 5 LoRAs on the Hub are all for the video models — but the model's authors have published a trainer (`kandinskylab/kandinsky-5-lora-train`, MIT), so there is a format to be compatible with and a recipe to follow.

**P6 as built** (`kandinsky_lora.rs`, `kandinsky_train.rs`, `kandinsky_dit.rs`, `kandinsky.rs`, `cli/style.rs`). The mathematics is verified on the tiny random DiT of the unit tests, on the CPU and on Metal; memory, speed and one trial run are measured on the real weights (36 GB M5 Max).

- **Format.** The reference's: PEFT adapters on the native module names, `base_model.model.<module>.lora_A.default.weight` `(r, in)` and `lora_B…` `(out, r)`, BF16, no alpha (the reference trains at `lora_alpha = r = 32`). The diffusers checkpoint keeps the native names, so the key is the layer's path. Also read: no adapter name, a `transformer.` or `diffusion_model.` prefix, `lora_down` / `lora_up`, and an `.alpha` beside a pair.
- **Loading** (`--lora file[:scale]` on `generate` and `img2img`). Merged into the weight as each layer is read, `W += s · B · A`, before the NF4 quantization when that is on: no per-step cost, and §11.4's wrapper is not needed. A file none of whose layers are in the model is an error; layers the model has no place for are reported. Not wired into the TUI.
- **Training** (`plakat style train --base kandinsky5`). The reference's recipe: the ten target layers (attention projections and feed-forward, text and visual blocks), `t = sigmoid(N(0,1))` shifted by 3.0 — the reference's `scheduler_scale`, not sampling's 5.0 — `x_t = (1 − t)·z + t·ε`, target `ε − z`, MSE, AdamW β 0.9 / 0.95 without weight decay, a 100-step warm-up, gradient norm clipped at 1, the empty caption in half the steps. Captions come from `<image>.txt`. One image a step, square, resized.
- **The backward goes block by block.** candle keeps a forward's activations until `backward` and has no checkpointing (`GRADIENT_CHECKPOINTING.md` calls it a dead end, and the other transformer trainers OOM above 256² for it). The DiT is a chain, so: forward once with every block's output detached and kept; take the loss on the output layer from the last kept output as a variable, which gives the gradient `g` of the visual stream; then walk back, running block `i` again from its kept input as a variable `x`, and differentiate the scalar `Σ(block(x) · g)` — its gradient in `x` is the next `g`, its gradients in the block's adapters are theirs of the loss. The text stream's gradient is summed over the visual blocks and walked back through the text blocks. Each block is run twice and differentiated once; the graph in memory is one block's. Only `backward` is used.
- **Checked:** the block-by-block gradient equals one backward through the whole model, for all 84 adapter matrices of the tiny DiT, to 2e-3 of each one's largest entry on the CPU and 5e-3 on Metal; 40 steps lower the loss by over 20 %; the saved file merges back into the trained model (to BF16 rounding); a LoRA merged before NF4 quantization is the model quantized from merged weights; a checkpoint restores into live adapters.
- **Found by the first test:** `candle_nn::ops::softmax_last_dim`, which the attention used, has no backward — the query and key projections got no gradient at all. With adapters installed the attention takes the softmax built from `exp` and `sum`.
- **Attention is one node of the graph under training.** Left to the tape, a block's self-attention keeps every head's `L × L` scores and the softmax's intermediates until the block is differentiated. A `CustomOp3` holds the three inputs instead: its forward keeps nothing, its backward rebuilds the graph a few heads at a time. Its gradients equal the tape's (unit test).
- **Metal's buffer pool decides the memory, not the graph.** candle's Metal backend never releases the pool that op outputs come from, and hands a freed buffer to the next request that fits in it. Under training small tensors live long, and each one that lands in a large freed buffer pins it. Three fixes came from this, each measured: tensors kept across the step (block inputs, `g`, adapter gradients) are parked on the CPU — 512² fell from 42.6 GB to 25.4 GB and stopped swapping; the attention chunk under training is 128 MB of scores, not 1 GB — a 1024² step fell from 254 s to 86–100 s; and a LoRA is merged into a dense weight on the CPU and uploaded once — generation with a LoRA fell from 38.1 GB (stopped by the OOM guard) to 24.4 GB.
- **Measured** (rank 32: 512 adapters, 109.4 M parameters, 219 MB):

  | | peak | a step |
  |---|---|---|
  | training, 512² | 22.1 GB (was 25.5 GB) | 11 s |
  | training, 1024² | 30.2 GB (was 43.1 GB) | 66 s (was 86–100 s, swapping) |
  | generation 1024², no LoRA | 16.1 GB | 15.5 s |
  | generation 1024², LoRA, dense | 24.4 GB | 15.6 s |
  | generation 1024², LoRA, `--dit-nf4` | 16.1 GB | 18.3 s |

  The loss is the same to four digits before and after each memory fix (0.5602, 0.3534, 0.4447 at 512²), so none of them changed the computation.
- **1024² training fits in 36 GB: where the 43 GB went.** A copy of candle with its pool made to report (each time it grew by a GB: buffers by size, how many in use, the bytes asked of them) showed that at the peak of a block's backward pass the pool held 21.5 GB of buffers, every one in use, for 8.8 GB of tensors. Three things, all the pool's: a buffer is rounded up to a power of two (a 42 MB stream tensor takes 64 MB, a 168 MB feed-forward one 256 MB); a free buffer goes to any smaller request when nothing nearer is free (the 48 buffers of 128 MB held tensors of 42 MB on average, the 26 of 256 MB tensors of 62 MB); and nothing is ever released, so the first step's peak is the run's. The pool cannot be reached from outside candle, so the fixes shrink what is alive at once:
  - **A block is differentiated one sublayer at a time** (self-attention, cross-attention, feed-forward — the block is the chain of them): 43.3 → 34.7 GB.
  - **The feed-forward goes 1024 tokens at a time**, forward and backward — it works on each token alone, so its 168 MB tensors and their 256 MB buffers no longer exist — and the sublayers' inputs are kept from the forward pass when they are under 2 GB (512²) and made again on the way back otherwise: → 31.1 GB, 512² back at 11 s.
  - **A frozen weight is one node of the graph.** candle differentiates a matmul with respect to both factors, so each backward pass also computed the gradient of every frozen weight — a product the size of the layer's own, held at the weight's F32 size until the pass ended, read by nothing. The node's backward is `grad · W` alone: → 30.2 GB and 66 s a step.

  After the first step the footprint is flat (26–27 GB at 1024², 19.5 GB at 512²) and the step time steady. The loss is the same to four digits throughout (0.4998, 0.3448, 0.2688 at 1024²). Measured over four steps; a long run at 1024² has not been made. The pool still holds about twice what its tensors need; taking that back means changing candle's allocator (exact sizes, a cap, releasing free buffers), which is a fork of the dependency and was not done.
- **One trial.** 200 steps at 512² on three near-identical images (38 minutes; loss 0.385 → 0.358, noisy). The file loads into all 512 layers, dense and NF4. At one seed's noise (after the fix below) and a prompt without the trigger, 8 steps, NF4: the LoRA changes the picture — the same layout, another pose and other trees — and the fox is worse drawn, smeared. So the path works end to end and the adapters act; a set of three pictures of one subject shows nothing about how well a style transfers, and that needs a real set.
- **In `scenario`.** `loras:` at the top of the file and on a task (the scenario's first, then the task's; `lora-scale` as elsewhere). LoRAs are merged as the transformer is read, so a set of LoRAs is a load: the queued tasks render one load for each distinct set. Run: three tasks, two of them with the trial LoRA — two loads, 640 s, 16.2 GB peak with `dit-nf4: true`. The scenario also takes `dit-nf4`, `quantize-qwen` and `nf4-cache`. A look's auto-discovered LoRAs belong to other families and are left out.
- **A seed did not repeat its image on Metal — fixed.** Found when the run that wrote the NF4 cache and the run that read it gave different pictures (15 dB apart). Four runs at one seed, with and without the cache, were all different, so it was not the cache. candle's Metal generator, seeded alike, agrees between two draws in their first values and not in the rest: the sum of a `(1, 16, 128, 128)` draw at seed 42 was −770, −864, −893 and −880 in four draws. The starting noise now comes from the family's own generator on the CPU (`seeded_noise`: SplitMix64 through Box–Muller, as verify's `deterministic_noise`), the same on every device. After it: three runs at one seed — two reading the cache, one with it off — are identical to the pixel. Every comparison "at the same seed" made on Metal before this compared different noise; the P5 notes above are marked.
- **Open.** The 8 GB a LoRA adds to dense generation (the weights read before the merge are probably released late; reading them on the CPU would avoid it). A style set of 10–20 varied images, judged. Whether a LoRA from the reference trainer loads (none is published to try). Prior preservation, aspect-ratio buckets and the optimizer's state in a checkpoint are not implemented; LoRAs are not wired into the TUI.

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| diffusers' no-shift VAE decode is itself a bug | Low | Every image off | §9 check against upstream in P1, before any DiT work |
| Tokenizer drift breaks offset 41 | Low | Garbage conditioning | T7 asserts it; on mismatch, find `<\|im_start\|>user\n` by token search instead of a fixed offset |
| Q4 encoder shifts embeddings too far | Medium | Quality loss in the quantized tier | §10.2 acceptance gate; fall back to Q8 (~8 GB) |
| Users perceive the family as "slow" | High | Adoption | Honest capability text; P5 step sweep; the quantized tier |
| Metal SDPA at 4,096+ tokens × 20 heads | Medium | OOM / slowness | Existing chunked-attention path from the Flux work; measure in P3 |
| candle 0.10.2 Metal breaks on tensors of rank > 4 (4-D batched matmul, rank-5 non-trailing reductions — found on the Sana and SD3 ports) | High | Garbage or a crash on Metal only; CPU parity passes | Keep every Metal op ≤ 3-D: flatten `B·heads` for attention, and do the 5-D channels-last latents and the 8-D patchify/unpatchify permutes as reshape + 3-D ops (or on CPU, once per step). P2's gate runs parity stages 3–6 on **Metal** as well as CPU |
| On-disk repo (35.7 GB) far exceeds what is loaded | Certain | Download size and disk | `hf` allow-list patterns: skip `assets/`, and fetch only the needed shards. (The `visual.*` tensors share shards with LM tensors, so they cannot be skipped at download time; document the cost.) |
| The development machine is a 24 GB Mac | Certain | P1–P3 runs fit the ≈15 GB staged peak only with nothing else running; the 35.7 GB download needs the disk | One heavy job at a time (render, build, gate or parity run) on the 24 GB machine; the end-to-end and peak-RSS gates (P3) are run by the author on a separate 36 GB host |
| Upstream ships a distilled T2I later | Medium | Wasted P5 step tuning | Cheap to absorb: a new alias, different default steps and guidance |

## 15. Open questions

- Q1. **Prompt-embedding disk cache.** Should `$PLAKAT_HOME/cache/k5emb/` key embeddings by `sha256(model ‖ template ‖ max_seq ‖ prompt)`? It would make TUI re-renders with only seed or size changes skip the encode stage entirely. *Lean: yes, but in 7.3, after real usage shows the hit rate.* **Postponed.** (The other cache — the NF4 pack of the transformer — is built: §13, Phase 4.)
- Q2. **`--keep-encoders` default in the TUI.** Default it on when `hw` reports ≥ 40 GB? *Lean: yes, decided by `capability`, not hard-coded.* **Postponed.**
- Q3. **`compile` Cyrillic handling.** Is the model's Russian-language and cultural strength real enough to advertise? Treat it as unverified until P5 runs a bilingual prompt set through `clip_adherence` and human review. *Answered in P5: Russian prompts score the same as English ones on four pairs and look as faithful; `compile` passes a Russian source through. No cultural-strength claim is made.*
- Q4. **KANDINSKY-2 scope.** I2I-Lite needs a Qwen2.5-VL vision encoder plus true M-RoPE. candle's `qwen3_vl` is the nearest template. Should KANDINSKY-2 vendor a `qwen2_5_vl` vision tower, or wait for candle upstream? The answer also decides whether plakat ever gets VLM-native prompt understanding beyond OWL-ViT and CLIP. **Postponed.**
- Q5. **Pretrain alias.** Keep `kandinsky5-pretrain` in N5, or drop it until a training RFC needs it? **Kept, as a training base:** `plakat style train --base kandinsky5-pretrain` trains a LoRA on the pretrain checkpoint (the alias resolves to its repo). The two repos share the text encoders and the VAE byte for byte (the same LFS objects), so the trainer reads those from the generation repo and downloads the pretrain transformer alone (12 GB, not 30). Run once: 200 steps at 512² on the three-image trial set, 27.4 GB peak and 11.0 s a step — the cost of training on the sft checkpoint — loss 0.39 → 0.36; the LoRA merges into `kandinsky5` (512 layers) and changes the picture at a fixed seed. Whether a LoRA trained on pretrain carries a style better than one trained on sft is not measured. It stays out of generation.
- Q6. **Is Russian better than English, and does the model know Russian culture?** P5's four prompt pairs support "Russian prompts work" (0.3055 against 0.3028 adherence, as faithful by eye). They do not support "Russian is better than English", and they say nothing about culture-specific knowledge: neither was tested. To answer: a larger bilingual set (the P4 corpus of 64 prompts, in both languages, several seeds), scored against the English text and reviewed by eye; and a separate set of culture-specific subjects (objects, dress, architecture, folk art, named places and works) that have no short English name, prompted in Russian, in transliteration and in an English paraphrase, judged for whether the specific thing is drawn and not a generic stand-in. Until then no claim of either kind is made in the docs. **A first answer** (`kandinsky_q6_culture`: ten subjects, each named briefly in Russian and described at length in English, one seed, 20 steps, 1024²; all twenty images looked at). *The model knows the culture.* A five-word Russian prompt with no description gave the right thing for Khokhloma (black, red and gold, with the spoon the English image left out), Gzhel, the hut on chicken legs, matryoshkas, the burning Maslenitsa effigy and the Russian stove — where the Russian image has the arched firebox and the cat on the ledge, and the English one a modern fireplace with a cast-iron door. *It does not always fill in what a name implies:* «Жар-птица» gave an orange bird, not a glowing one, and «богатырь» a rider without the helmet and chain mail the English description asked for. *Cyrillic lettering works for a short word:* «ХЛЕБ» is right from both languages; «МИР» is right from the Russian prompt and garbled from the English one («ИР», «ЛИР»); secondary lettering is gibberish in both. *Russian is not shown to be better.* CLIP adherence to the English text is 0.3311 from English against 0.3149 from Russian, Russian ahead in 3 of 10 — but the English prompts were the longer ones and are the text scored against, so the number favours them by construction. Still open: several seeds, the 64-prompt corpus in both languages, and prompts of equal detail in the two languages, which is the comparison that would say whether one language is better.

## 16. What this does not do

It does not add a new attention backend, offload framework, or adapter system. It does not touch any existing family's numerics. It does not promise seed-for-seed equality with diffusers.

It adds one DiT, one encoder adapter, and one memory-staging pattern local to the family.

---

## Appendix A — Porting traps checklist

| # | Trap | Where it bites | Test |
|---|---|---|---|
| T1 | VAE decode `/scale` **without** `+shift` (encode: `·scale`, no `−shift`) | §9 | VAE round-trip vs dump |
| T2 | DiT timestep is raw `σ·1000` | §8.2 | time-embed parity at t ∈ {1000, 500, 1} |
| T3 | candle qwen2 with `attn_mask` is **bidirectional** | §7.1 | Qwen hidden-state parity |
| T4 | RoPE is interleaved-pair 2×2 rotation, not rotate-half | §6.2 | 4-dim hand case |
| T5 | Time embedding is `cat[cos, sin]` (cos first) | §6.1 | time-embed parity |
| T6 | Modulate and residual math in F32; modulation weights loaded as F32 | §6.4, §6.6 | block parity at BF16 |
| T7 | Template offset 41 depends on the tokenizer | §7.1 | token-count assert |
| T8 | Text blocks are time-modulated; never cache across steps | §6.4 | velocity parity at steps 25 and 49 |
| T9 | Latents are channels-last `(B, 1, H, W, C)` inside the DiT | §6.1 | patchify round-trip |
| T10 | Cross-attention has no RoPE; positive and negative have independent text positions | §6.2, §8.2 | CFG velocity parity |
| T11 | Load `qwen2::Model`, not `ModelForCausalLM` (skip 1.1 GB `lm_head`) | §7.1 | peak-RSS gate in P3 |
| T12 | Template typo "promt" is load-bearing | §7.1 | string constant test |
| T13 | Reference rounds RoPE output to BF16 even in F32 | §12.3 | F32 parity, flag on/off |

## Appendix B — References

- Model: https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers
- diffusers docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky5_image
- diffusers integration PR: https://github.com/huggingface/diffusers/pull/12664
- Upstream: https://github.com/kandinskylab/kandinsky-5
- Paper: arXiv 2511.14993
