# Watercolour style corpus (for `plakat style train`)

Nine watercolour illustrations — the training corpus for the in-progress
**`plakat style train`** feature (`watercolour/*.jpeg`). The goal:
*creation*, not detection — turn these into a kohya-format LoRA
`.safetensors` that loads into the model via `--lora` / a catalog, so
generation paints in this style.

`catalog.hjson` is the `plakat style init` output (one style, these as
exemplars). It's detection-only today (`models: {}`); the trainer will
fill in `models: { sdxl: { loras: [...] } }` with the trained adapter.

## Status

The training subsystem is being built. Foundation banked + proven:

- **candle Metal backward works** — verified with `src/bin/train_spike.rs`
  (`cargo run --release --features metal --bin train_spike`): AdamW
  converges and conv2d / softmax / layernorm / silu all backprop on Metal.
  (Only `rand_uniform` F64 is unimplemented on Metal → init in F32.)
- **`LoraLinear` has a trainable adapter** (`set_train_adapter`) — Var-
  backed, `None` by default so inference is byte-identical.

Remaining: wire `LoraLinear` into `SdxlUNet` attention (it currently
merges LoRAs for inference, no per-layer hooks), the diffusion training
loop (512², encode-then-free VAE/text-encoders), kohya save, and the
`plakat style train` subcommand. See the seeded task list.

Why a LoRA and not these images directly: plakat detection routes on CLIP
fingerprints of these images; *painting* in the style needs a LoRA. The
old route (the bundled generic `ink-watercolor` LoRA) was "watercolour-ish
but not these" — hence training our own.
