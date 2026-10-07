//! Kandinsky 5 text conditioning and VAE (RFC KANDINSKY-1, P1).
//!
//! * **Qwen2.5-VL-7B, text only.** The checkpoint's language tower loads as candle's plain
//!   `qwen2::Model` (`model.embed_tokens`, `model.layers.N`, `model.norm`); `lm_head` and `visual.*` are
//!   never read from the mmapped shards. For text-only input Qwen2.5-VL's multimodal RoPE collapses to
//!   1-D RoPE, so the qwen2 decoder is exact here. It is vendored (`vendored_qwen2`): the weights rest in
//!   BF16 on a GPU but activations are computed in F32 — all-BF16 compute fails parity on this tower.
//!   Each prompt is encoded as a batch of one with NO attention mask — candle's masked path is
//!   bidirectional, the model is causal — and the prompt's hidden states are the last layer's, sliced
//!   from the end of the template's system prefix.
//! * **CLIP-L pooled.** The final-layer-norm hidden state at the EOT position, as `pooler_output`.
//! * **VAE.** The repo ships the Flux VAE in the DIFFUSERS key layout (`AutoencoderKL`), so it loads with
//!   candle's `stable_diffusion::vae::AutoEncoderKL` exactly as plakat's SD3 path does — not with the
//!   BFL-layout `flux::autoencoder`. Both the upstream code and diffusers scale latents by
//!   `scaling_factor` and apply NO `shift_factor`, on decode and on encode; this module does the same.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use crate::pipelines::vendored_qwen2 as qwen2;
use candle_transformers::models::stable_diffusion::vae as sdvae;
use tokenizers::Tokenizer;

use crate::pipelines::vendored_clip as vclip;

/// A prompt template: the system prefix every prompt is wrapped in, and how many tokens that prefix is
/// under the reference tokenizer (the hidden states before it are discarded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Template {
    pub name: &'static str,
    /// Everything before the user's prompt, up to and including `<|im_start|>user\n`.
    pub prefix: &'static str,
    /// Token length of `prefix` under the reference tokenizer.
    pub crop_start: usize,
}

/// The template the model was TRAINED with (`kandinskylab/kandinsky-5`, `text_embedders.py`, the `image`
/// entry; `crop_start` 41). The typo "promt" is the upstream's and is load-bearing: it is two tokens
/// where "prompt" is one, and every later hidden state is conditioned on it. The default.
pub const TEMPLATE_UPSTREAM: Template = Template {
    name: "upstream",
    prefix: "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n",
    crop_start: 41,
};

/// The template diffusers' `Kandinsky5T2IPipeline` uses today: the typo corrected, and so one token
/// shorter (`prompt_template_encode_start_idx` 40). NOT what the weights were trained with; here so a
/// parity run against diffusers tensor dumps compares like with like (`PLAKAT_K5_TEMPLATE=diffusers`).
pub const TEMPLATE_DIFFUSERS: Template = Template {
    name: "diffusers",
    prefix: "<|im_start|>system\nYou are a prompt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n",
    crop_start: 40,
};

const TEMPLATE_SUFFIX: &str = "<|im_end|>";

impl Template {
    /// The template in force: the upstream one unless `PLAKAT_K5_TEMPLATE=diffusers`.
    pub fn active() -> &'static Template {
        match std::env::var("PLAKAT_K5_TEMPLATE").ok().as_deref() {
            Some("diffusers") => &TEMPLATE_DIFFUSERS,
            _ => &TEMPLATE_UPSTREAM,
        }
    }

    pub fn by_name(name: &str) -> Option<&'static Template> {
        [&TEMPLATE_UPSTREAM, &TEMPLATE_DIFFUSERS].into_iter().find(|t| t.name == name)
    }

    /// The full string the tokenizer sees for a prompt.
    pub fn format(&self, prompt: &str) -> String {
        format!("{}{}{}", self.prefix, prompt, TEMPLATE_SUFFIX)
    }

    /// The prefix's token length under `tok`. Must equal [`Template::crop_start`]; a tokenizer that
    /// disagrees would slice the hidden states at the wrong place and feed the DiT garbage.
    pub fn measured_crop(&self, tok: &Tokenizer) -> Result<usize> {
        Ok(tok.encode(self.prefix, false).map_err(|e| anyhow!("tokenizing the template prefix: {e}"))?.len())
    }
}

/// A prompt tokenized for the Qwen tower.
#[derive(Debug, Clone)]
pub struct PromptTokens {
    /// The whole templated sequence, truncated to `crop_start + max_seq`.
    pub ids: Vec<u32>,
    /// Where the prompt's own tokens start.
    pub crop_start: usize,
    /// The text that fell past the budget, if any.
    pub dropped: Option<String>,
}

/// Tokenize `prompt` inside `template`, keeping at most `max_seq` tokens after the prefix — the
/// reference's plain right-truncation (a prompt that fills the budget loses its closing `<|im_end|>`,
/// as it does upstream).
pub fn tokenize_prompt(tok: &Tokenizer, template: &Template, prompt: &str, max_seq: usize) -> Result<PromptTokens> {
    let crop_start = template.measured_crop(tok)?;
    if crop_start != template.crop_start {
        anyhow::bail!(
            "the Kandinsky 5 template prefix is {crop_start} tokens under this tokenizer, but the `{}` template expects {} — \
             the tokenizer does not match the reference one (RFC KANDINSKY-1, trap T7)",
            template.name,
            template.crop_start
        );
    }
    let enc = tok.encode(template.format(prompt), false).map_err(|e| anyhow!("tokenizing the prompt: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    let budget = crop_start + max_seq;
    let dropped = if ids.len() > budget {
        let tail = tok.decode(&ids[budget..], true).unwrap_or_default();
        ids.truncate(budget);
        (!tail.trim().is_empty()).then_some(tail)
    } else {
        None
    };
    Ok(PromptTokens { ids, crop_start, dropped })
}

/// The Qwen2.5-VL language-model config, read from `text_encoder/config.json`. Depending on the
/// transformers version that wrote the file the fields sit at the top level (the shipped repo) or
/// nested under `text_config`; both are accepted, the nested one winning.
pub fn qwen_config_from_json(json: &str) -> Result<qwen2::Config> {
    let v: serde_json::Value = serde_json::from_str(json).context("parsing text_encoder/config.json")?;
    let nested = v.get("text_config");
    let get = |k: &str| nested.and_then(|n| n.get(k)).filter(|x| !x.is_null()).or_else(|| v.get(k).filter(|x| !x.is_null()));
    let need = |k: &str| -> Result<usize> { get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| anyhow!("text_encoder/config.json has no `{k}`")) };
    let opt_usize = |k: &str, d: usize| get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d);
    let act = get("hidden_act").and_then(|x| x.as_str()).unwrap_or("silu");
    if act != "silu" {
        anyhow::bail!("Qwen text tower with hidden_act `{act}` is not supported (expected silu)");
    }
    Ok(qwen2::Config {
        vocab_size: need("vocab_size")?,
        hidden_size: need("hidden_size")?,
        intermediate_size: need("intermediate_size")?,
        num_hidden_layers: need("num_hidden_layers")?,
        num_attention_heads: need("num_attention_heads")?,
        num_key_value_heads: need("num_key_value_heads")?,
        max_position_embeddings: opt_usize("max_position_embeddings", 32768),
        sliding_window: opt_usize("sliding_window", 32768),
        max_window_layers: opt_usize("max_window_layers", 28),
        tie_word_embeddings: get("tie_word_embeddings").and_then(|x| x.as_bool()).unwrap_or(false),
        rope_theta: get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(1_000_000.0),
        rms_norm_eps: get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6),
        use_sliding_window: get("use_sliding_window").and_then(|x| x.as_bool()).unwrap_or(false),
        hidden_act: candle_nn::Activation::Silu,
    })
}

/// Qwen2.5-VL's multimodal RoPE position ids (temporal, height, width) for a TEXT-ONLY sequence. All
/// three sections are the same `0..len`, which is what makes plain 1-D RoPE exact for it — an invariant
/// this port relies on and a test asserts, rather than an assumption.
pub fn mrope_text_positions(len: usize) -> [Vec<usize>; 3] {
    let p: Vec<usize> = (0..len).collect();
    [p.clone(), p.clone(), p]
}

const CLIP_BOS: u32 = 49406;
const CLIP_EOT: u32 = 49407;
const CLIP_LEN: usize = 77;

/// CLIP-L ids as the reference tokenizer builds them (`max_length=77`, `padding="max_length"`,
/// truncation): a prompt longer than 77 tokens is cut and its LAST kept token is the EOT, the rest is
/// padded with EOT. Returns the 77 ids and the position `pooler_output` reads (the first EOT).
pub fn clip_ids_77(raw: &[u32]) -> (Vec<u32>, usize) {
    let mut ids: Vec<u32> = raw.to_vec();
    if ids.first() != Some(&CLIP_BOS) {
        ids.insert(0, CLIP_BOS);
    }
    if ids.last() != Some(&CLIP_EOT) {
        ids.push(CLIP_EOT);
    }
    if ids.len() > CLIP_LEN {
        ids.truncate(CLIP_LEN);
        ids[CLIP_LEN - 1] = CLIP_EOT;
    }
    let eot = ids.iter().position(|&t| t == CLIP_EOT).unwrap_or(ids.len() - 1);
    ids.resize(CLIP_LEN, CLIP_EOT);
    (ids, eot)
}

/// One prompt's conditioning.
pub struct Embeds {
    /// Qwen last hidden states of the prompt's own tokens: `(1, L, 3584)`.
    pub qwen: Tensor,
    /// CLIP-L pooled: `(1, 768)`.
    pub pooled: Tensor,
    /// The token ids the Qwen tower saw (template included) — the first parity stage.
    pub ids: Vec<u32>,
    /// Text that fell past `max_seq`, if any.
    pub dropped: Option<String>,
}

/// The two text encoders. Owned separately from the DiT so the pipeline can drop them after encoding
/// (staged residency, RFC §10.1).
/// The quantized Qwen tower (`--quantize-qwen`): the language model of Qwen2.5-VL-7B-Instruct at
/// Q4_K_M, 4.7 GB. The tokenizer stays the Kandinsky repo's own.
pub const QWEN_GGUF: (&str, &str) = ("ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");

/// The Qwen tower as loaded: the checkpoint's BF16 weights, or a GGUF's quantized ones.
enum Qwen {
    Dense(qwen2::Model),
    Quantized(crate::pipelines::vendored_qwen2q::Model),
}

impl Qwen {
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Qwen::Dense(m) => m.forward(ids),
            Qwen::Quantized(m) => m.forward(ids),
        }
    }
}

pub struct TextEncoders {
    qwen: Qwen,
    qwen_tok: Tokenizer,
    clip: vclip::ClipTextTransformer,
    clip_tok: Tokenizer,
    device: Device,
    dtype: DType,
    pub template: &'static Template,
}

/// The shards of a sharded safetensors checkpoint that hold tensors under `prefix` (read from the
/// index, so a shard that carries none of them is never downloaded).
async fn shards_with_prefix(repo: &str, subfolder: &str, prefix: &str) -> Result<Vec<std::path::PathBuf>> {
    let index = crate::hf::download::get_file(repo, &format!("{subfolder}/model.safetensors.index.json")).await.with_context(|| format!("{subfolder}/model.safetensors.index.json"))?;
    let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(&index)?)?;
    let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| anyhow!("{subfolder}: the safetensors index has no weight_map"))?;
    let mut names: Vec<String> = map.iter().filter(|(k, _)| k.starts_with(prefix)).filter_map(|(_, f)| f.as_str().map(str::to_string)).collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        anyhow::bail!("{subfolder}: no tensors under `{prefix}` in the safetensors index");
    }
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        out.push(crate::hf::download::get_file(repo, &format!("{subfolder}/{n}")).await?);
    }
    Ok(out)
}

impl TextEncoders {
    /// Load the Qwen text tower and CLIP-L from a Kandinsky 5 diffusers repo. BF16 on a GPU, F32 on CPU.
    pub async fn load(repo: &str, device: &Device) -> Result<Self> {
        Self::load_with(repo, device, false).await
    }

    /// As [`Self::load`]; with `quantized` the Qwen tower comes from [`QWEN_GGUF`] instead of the repo.
    pub async fn load_with(repo: &str, device: &Device, quantized: bool) -> Result<Self> {
        let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
        let qwen_tok_path = crate::hf::download::get_file(repo, "tokenizer/tokenizer.json").await.context("Kandinsky 5 tokenizer/tokenizer.json")?;
        let qwen_tok = Tokenizer::from_file(&qwen_tok_path).map_err(|e| anyhow!("loading the Qwen tokenizer: {e}"))?;
        let template = Template::active();
        let crop = template.measured_crop(&qwen_tok)?;
        if crop != template.crop_start {
            anyhow::bail!("the `{}` template prefix is {crop} tokens under the repo's tokenizer, expected {} (RFC KANDINSKY-1, trap T7)", template.name, template.crop_start);
        }
        let qwen = if quantized {
            let (gguf_repo, gguf_file) = QWEN_GGUF;
            let path = crate::hf::download::get_file(gguf_repo, gguf_file).await.with_context(|| format!("{gguf_repo}/{gguf_file}"))?;
            let mut file = std::fs::File::open(&path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
            Qwen::Quantized(crate::pipelines::vendored_qwen2q::Model::from_gguf(content, &mut file, device).context("building the quantized Qwen2.5-VL text tower")?)
        } else {
            let cfg_path = crate::hf::download::get_file(repo, "text_encoder/config.json").await?;
            let cfg = qwen_config_from_json(&std::fs::read_to_string(&cfg_path)?)?;
            // Only the language tower: `model.*`. `lm_head` and `visual.*` stay on disk.
            let shards = shards_with_prefix(repo, "text_encoder", "model.").await?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, device)? };
            Qwen::Dense(qwen2::Model::new(&cfg, vb).context("building the Qwen2.5-VL text tower")?)
        };

        let clip_weights = crate::hf::download::get_file(repo, "text_encoder_2/model.safetensors").await?;
        let clip_tok_path = crate::hf::download::get_file(repo, "tokenizer_2/tokenizer.json").await?;
        let clip_tok = Tokenizer::from_file(&clip_tok_path).map_err(|e| anyhow!("loading the CLIP tokenizer: {e}"))?;
        // The checkpoint is a whole CLIP ViT-L/14 (text + vision); only `text_model.*` is read.
        let clip = vclip::build_clip_transformer(&vclip::Config::v1_5(), &clip_weights, device, dtype)?;
        Ok(Self { qwen, qwen_tok, clip, clip_tok, device: device.clone(), dtype, template })
    }

    /// Encode one prompt (a batch of one, causal, unpadded).
    pub fn encode(&mut self, prompt: &str, max_seq: usize) -> Result<Embeds> {
        let toks = tokenize_prompt(&self.qwen_tok, self.template, prompt, max_seq)?;
        let n = toks.ids.len();
        let ids = Tensor::new(toks.ids.as_slice(), &self.device)?.unsqueeze(0)?;
        // The vendored tower is causal and takes the sequence whole (no mask: trap T3); it computes in F32
        // whatever the weights are stored as, and the embeds are handed on in the pipeline's dtype.
        let hidden = self.qwen.forward(&ids)?.to_dtype(self.dtype)?;
        let qwen = hidden.narrow(1, toks.crop_start, n - toks.crop_start)?;

        let raw = self.clip_tok.encode(prompt, true).map_err(|e| anyhow!("CLIP tokenize: {e}"))?.get_ids().to_vec();
        let (clip_ids, eot) = clip_ids_77(&raw);
        let clip_in = Tensor::new(clip_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let pooled = self.clip.forward(&clip_in)?.i((.., eot, ..))?.to_dtype(self.dtype)?;
        Ok(Embeds { qwen, pooled, ids: toks.ids, dropped: toks.dropped })
    }
}

/// `scaling_factor` of the Flux VAE. The reference applies it alone — no `shift_factor` (trap T1).
pub const VAE_SCALE: f64 = 0.3611;

/// The Kandinsky 5 VAE: the Flux 16-channel autoencoder, in the diffusers layout, always F32.
pub struct Vae {
    inner: sdvae::AutoEncoderKL,
}

impl Vae {
    pub async fn load(repo: &str, device: &Device) -> Result<Self> {
        let path = crate::hf::download::get_file(repo, "vae/diffusion_pytorch_model.safetensors").await.context("Kandinsky 5 vae")?;
        // Same shape as the SD3 autoencoder plakat already loads this way: 4 blocks, 16 latent channels, no
        // quant / post-quant convs.
        let cfg = sdvae::AutoEncoderKLConfig { block_out_channels: vec![128, 256, 512, 512], layers_per_block: 2, latent_channels: 16, norm_num_groups: 32, use_quant_conv: false, use_post_quant_conv: false };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, device)? };
        Ok(Self { inner: sdvae::AutoEncoderKL::new(vb, 3, 3, cfg)? })
    }

    /// Latents `(B, 16, H/8, W/8)` → pixels `(B, 3, H, W)` in `[-1, 1]`: `decode(z / scale)`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        Ok(self.inner.decode(&(latents.to_dtype(DType::F32)? / VAE_SCALE)?)?)
    }

    /// Pixels in `[-1, 1]` → latents: `encode(x).sample() · scale` (the img2img init).
    pub fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        Ok((self.inner.encode(&pixels.to_dtype(DType::F32)?)?.sample()? * VAE_SCALE)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_templates_are_the_references_strings() {
        // The upstream's typo is load-bearing (trap T12); diffusers corrected it and lost a token.
        assert!(TEMPLATE_UPSTREAM.prefix.contains("You are a promt engineer."));
        assert!(TEMPLATE_DIFFUSERS.prefix.contains("You are a prompt engineer."));
        assert_eq!((TEMPLATE_UPSTREAM.crop_start, TEMPLATE_DIFFUSERS.crop_start), (41, 40));
        for t in [&TEMPLATE_UPSTREAM, &TEMPLATE_DIFFUSERS] {
            assert!(t.prefix.starts_with("<|im_start|>system\n") && t.prefix.ends_with("<|im_end|>\n<|im_start|>user\n"));
            assert_eq!(t.format("a cat"), format!("{}a cat<|im_end|>", t.prefix));
            assert_eq!(Template::by_name(t.name), Some(t));
        }
    }

    #[test]
    fn the_config_shim_reads_both_layouts() {
        let flat = r#"{"hidden_size":3584,"intermediate_size":18944,"num_hidden_layers":28,"num_attention_heads":28,
            "num_key_value_heads":4,"vocab_size":152064,"rope_theta":1000000.0,"rms_norm_eps":1e-06,"hidden_act":"silu",
            "tie_word_embeddings":false,"use_sliding_window":false,"max_position_embeddings":128000,"vision_config":{"hidden_size":1280}}"#;
        let nested = r#"{"model_type":"qwen2_5_vl","hidden_size":null,"text_config":{"hidden_size":3584,"intermediate_size":18944,
            "num_hidden_layers":28,"num_attention_heads":28,"num_key_value_heads":4,"vocab_size":152064,"rope_theta":1000000.0},
            "vision_config":{"hidden_size":1280}}"#;
        for j in [flat, nested] {
            let c = qwen_config_from_json(j).unwrap();
            assert_eq!((c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.vocab_size), (3584, 18944, 28, 28, 4, 152064));
            assert_eq!(c.rope_theta, 1e6);
            assert!(!c.tie_word_embeddings && !c.use_sliding_window);
        }
        assert!(qwen_config_from_json(r#"{"hidden_size":3584}"#).is_err());
    }

    #[test]
    fn text_only_mrope_collapses_to_one_dimension() {
        let [t, h, w] = mrope_text_positions(57);
        assert_eq!(t, h);
        assert_eq!(h, w);
        assert_eq!(t, (0..57).collect::<Vec<_>>());
    }

    #[test]
    fn clip_ids_pad_and_truncate_as_the_reference_tokenizer() {
        let (ids, eot) = clip_ids_77(&[CLIP_BOS, 320, 2368, CLIP_EOT]);
        assert_eq!((ids.len(), eot), (77, 3));
        assert!(ids[3..].iter().all(|&t| t == CLIP_EOT));
        // A long prompt: cut to 77 with the EOT as its last token, pooled there.
        let mut long = vec![CLIP_BOS];
        long.extend(std::iter::repeat(320).take(200));
        long.push(CLIP_EOT);
        let (ids, eot) = clip_ids_77(&long);
        assert_eq!((ids.len(), eot, ids[76], ids[75]), (77, 76, CLIP_EOT, 320));
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let (a, b) = (a.to_dtype(DType::F32).unwrap().flatten_all().unwrap(), b.to_dtype(DType::F32).unwrap().flatten_all().unwrap());
        let dot = (&a * &b).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let na = a.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        let nb = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        dot / (na * nb).max(1e-12)
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// P1's parity gate (RFC §12, stages 1–2 and the VAE): against `tools/kandinsky_dump.py` output in
    /// `PLAKAT_PARITY_DIR`. Loads the real weights — run it alone, on a machine that fits them:
    /// `PLAKAT_PARITY_DIR=<dir> cargo test --release --features metal --lib kandinsky_parity -- --ignored --nocapture`
    /// (`PLAKAT_PARITY_DEVICE=cpu` for the F32 tolerances; the F32 Qwen tower is ≈28 GB resident).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_parity_p1() {
        let dir = std::path::PathBuf::from(std::env::var("PLAKAT_PARITY_DIR").expect("set PLAKAT_PARITY_DIR to a tools/kandinsky_dump.py output directory"));
        let meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("meta.json")).unwrap()).unwrap();
        let s = |k: &str| meta[k].as_str().unwrap().to_string();
        let template = Template::by_name(&s("template")).expect("meta.json names a known template");
        assert_eq!(meta["crop_start"].as_u64().unwrap() as usize, template.crop_start);
        let max_seq = meta["max_seq"].as_u64().unwrap() as usize;
        let device = crate::device::select(&std::env::var("PLAKAT_PARITY_DEVICE").unwrap_or_else(|_| "auto".into())).unwrap();
        let f32_run = device.is_cpu();
        let reference = candle_core::safetensors::load(dir.join("p1.safetensors"), &device).unwrap();
        // SAFETY of the env write: a single ignored test, run alone.
        unsafe { std::env::set_var("PLAKAT_K5_TEMPLATE", template.name) };
        let repo = s("repo");
        // Every stage is measured before the gate is judged, so one run (minutes of loading) reports all.
        let failed = std::cell::RefCell::new(Vec::new());
        let check = |ok: bool, what: String| {
            if !ok {
                failed.borrow_mut().push(what)
            }
        };
        let peak = |t: &Tensor| t.to_dtype(DType::F32).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        // `PLAKAT_PARITY_STAGE=vae` skips the encoders (the VAE check alone is seconds, not minutes).
        if std::env::var("PLAKAT_PARITY_STAGE").as_deref() != Ok("vae") {
            let mut enc = TextEncoders::load(&repo, &device).await.unwrap();
            for (tag, text) in [("pos", s("prompt")), ("neg", s("negative"))] {
                let e = enc.encode(&text, max_seq).unwrap();
                let want_ids: Vec<i64> = reference[&format!("qwen_ids_{tag}")].to_dtype(DType::I64).unwrap().to_vec1().unwrap();
                assert_eq!(e.ids.iter().map(|&i| i as i64).collect::<Vec<_>>(), want_ids, "{tag}: token ids");
                let (h, p) = (&reference[&format!("qwen_hidden_{tag}")], &reference[&format!("clip_pooled_{tag}")]);
                assert_eq!(e.qwen.dims(), h.dims(), "{tag}: Qwen hidden shape");
                let (ch, cp, mh, mp) = (cosine(&e.qwen, h), cosine(&e.pooled, p), max_abs(&e.qwen, h), max_abs(&e.pooled, p));
                println!("{tag}: Qwen hidden cosine {ch:.6} max-abs {mh:.2e} · CLIP pooled cosine {cp:.6} max-abs {mp:.2e}");
                // F32 is held to max-abs RELATIVE to the reference's largest value (Qwen's hidden states
                // reach ~100, so an absolute bound would be asking for more than F32 has after 28 layers).
                let (rh, rp) = (mh / peak(h), mp / peak(p));
                if f32_run {
                    check(rh <= 1e-4, format!("{tag}: Qwen hidden relative max-abs {rh:.2e}"));
                    check(rp <= 1e-5, format!("{tag}: CLIP pooled relative max-abs {rp:.2e}"));
                } else {
                    check(ch >= 0.999, format!("{tag}: Qwen hidden cosine {ch}"));
                    check(cp >= 0.9999, format!("{tag}: CLIP pooled cosine {cp}"));
                }
            }
        } // the encoders are released before the VAE loads
        let vae = Vae::load(&repo, &device).await.unwrap();
        let got = vae.decode(&reference["vae_latent"]).unwrap();
        let want = reference["vae_decoded"].to_dtype(DType::F32).unwrap();
        let mse = (&got - &want).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap() / 4.0;
        // A NaN decode must not pass: `f32::max` drops a NaN operand, which would read as 120 dB.
        check(mse.is_finite(), format!("VAE decode is not finite (mse {mse})"));
        let psnr = -10.0 * mse.max(1e-12).log10();
        println!("VAE decode vs reference: PSNR {psnr:.1} dB (mse {mse:.3e})");
        check(psnr >= if f32_run { 40.0 } else { 32.0 }, format!("VAE decode PSNR {psnr}"));
        let failed = failed.into_inner();
        assert!(failed.is_empty(), "parity gate failed: {failed:#?}");
    }

    /// Trap T7: the template prefix must be exactly `crop_start` tokens under the repo's tokenizer.
    /// Needs `tokenizer/tokenizer.json` from the Kandinsky repo: `PLAKAT_K5_TOKENIZER=<path>`.
    #[test]
    #[ignore]
    fn the_template_prefix_is_its_crop_start_under_the_reference_tokenizer() {
        let path = std::env::var("PLAKAT_K5_TOKENIZER").expect("set PLAKAT_K5_TOKENIZER to tokenizer/tokenizer.json");
        let tok = Tokenizer::from_file(path).unwrap();
        for t in [&TEMPLATE_UPSTREAM, &TEMPLATE_DIFFUSERS] {
            assert_eq!(t.measured_crop(&tok).unwrap(), t.crop_start, "{}", t.name);
        }
        let p = tokenize_prompt(&tok, &TEMPLATE_UPSTREAM, "a red fox in the snow", 512).unwrap();
        assert_eq!(p.crop_start, 41);
        assert!(p.dropped.is_none());
        // The slice keeps the prompt and its closing <|im_end|>.
        let im_end = tok.token_to_id("<|im_end|>").unwrap();
        assert_eq!(*p.ids.last().unwrap(), im_end);
        assert!(p.ids.len() - p.crop_start >= 6);
        // The empty negative still has one token after the crop: the template's tail.
        let neg = tokenize_prompt(&tok, &TEMPLATE_UPSTREAM, "", 512).unwrap();
        assert_eq!(neg.ids.len() - neg.crop_start, 1);
        // Past the budget the tail is dropped and reported.
        let long = "a very long prompt ".repeat(400);
        let cut = tokenize_prompt(&tok, &TEMPLATE_UPSTREAM, &long, 512).unwrap();
        assert_eq!(cut.ids.len(), 41 + 512);
        assert!(cut.dropped.is_some());
    }
}
