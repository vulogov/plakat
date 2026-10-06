# RFC KANDINSKY-1 — Kandinsky 5.0 Image Lite as plakat's eighth model family

| | |
|---|---|
| **RFC** | KANDINSKY-1 |
| **Status** | Accepted — Phase 0 and P1 built (7.2.0); P1's weight gates pending on the 36 GB host; P2 next |
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
- N4. LoRA and ControlNet. No public adapters exist yet. The LoRA loader seam is noted in §11.4 but not implemented.
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
- **Not yet run:** the Qwen tower has not been loaded anywhere (15.5 GB download, ≈14 GB resident), so parity stages 1–2 and the VAE decode against a reference dump are open. They run on the 36 GB host: `python tools/kandinsky_dump.py --out DIR`, then the ignored test `kandinsky_parity_p1` with `PLAKAT_PARITY_DIR=DIR`.
- **One thing for P3:** upstream's own sampler defaults to `scheduler_scale = 3.0` in `t2i_pipeline.py`, while the diffusers repo's scheduler config says `shift: 5.0`. The port follows the diffusers config (§8.1); worth a look when the first images exist.

### Phase 2 — DiT

- `kandinsky_dit.rs` per §6, with F32-island loading.
- **Gate:** parity stages 3–6 at both precisions, and on Metal as well as CPU (§14, the rank > 4 risk).

### Phase 3 — End-to-end and staged residency

- `kandinsky.rs` orchestration, §7.4 batching and the scenario `pre_encode` hook, `--keep-encoders`, and tiled decode fallback.
- `parameters` sidecar fields: `family=kandinsky5`, `max_seq`, `bucket`.
- **Gate:** parity stages 7–8. Peak RSS measured on 1024² and 1408×640 at ≤ 16 GB plus activations on CUDA and Metal. G1 and G2 met.

### Phase 4 — Quantized tier

- `vendored_qwen2q.rs`, the GGUF descriptor, and the DiT NF4 path.
- **Gate:** the §10.2 acceptance numbers; a peak of ≤ 7 GB on a 16 GB Mac. G3 met.

### Phase 5 — img2img, inpaint, compile, polish

- `FlowSpace` init with the §9 encode, and `masked_denoise::step_blend` inpaint.
- The `compile` profile, a step-count quality sweep (20/30/40/50) on the verify corpus to set the `--steps` guidance, and README, Book, and `--help` updates.
- **Gate:** G4 and G5. Release notes state plainly that there are no LoRA or ControlNet adapters yet and that this is the slowest family.

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

- Q1. **Prompt-embedding disk cache.** Should `$PLAKAT_HOME/cache/k5emb/` key embeddings by `sha256(model ‖ template ‖ max_seq ‖ prompt)`? It would make TUI re-renders with only seed or size changes skip the encode stage entirely. *Lean: yes, but in 7.3, after real usage shows the hit rate.*
- Q2. **`--keep-encoders` default in the TUI.** Default it on when `hw` reports ≥ 40 GB? *Lean: yes, decided by `capability`, not hard-coded.*
- Q3. **`compile` Cyrillic handling.** Is the model's Russian-language and cultural strength real enough to advertise? Treat it as unverified until P5 runs a bilingual prompt set through `clip_adherence` and human review.
- Q4. **KANDINSKY-2 scope.** I2I-Lite needs a Qwen2.5-VL vision encoder plus true M-RoPE. candle's `qwen3_vl` is the nearest template. Should KANDINSKY-2 vendor a `qwen2_5_vl` vision tower, or wait for candle upstream? The answer also decides whether plakat ever gets VLM-native prompt understanding beyond OWL-ViT and CLIP.
- Q5. **Pretrain alias.** Keep `kandinsky5-pretrain` in N5, or drop it until a training RFC needs it?

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
