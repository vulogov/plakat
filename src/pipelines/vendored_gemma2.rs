//! Vendored Gemma-2 (Google) — the text encoder for Sana (ROADMAP_4.5.0 Phase 2).
//!
//! A copy of candle-transformers 0.10.2 `models::gemma2`, adapted for **encoder** use: Sana feeds
//! Gemma-2-2B's **last hidden state** (all positions, no `lm_head`) into the DiT cross-attention.
//! candle's stock `Model::forward` returns last-token *logits* and its fields are private, so we
//! vendor + add [`Model::forward_hidden`] (embed → layers → final norm, over all positions) with a
//! padding-mask argument (Sana passes an attention mask that the `[0]+last-299` re-slice preserves).
//!
//! Based on implementations from Google and OpenLLM (via candle-transformers).

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear, Activation, Linear, VarBuilder};

/// GQA key/value head expansion (candle-transformers `utils::repeat_kv`, vendored to avoid the
/// `crate::utils` path). Repeats each KV head `n_rep×` consecutively.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    // TODO: Handle the sliding window in the attention mask.
    pub sliding_window: Option<usize>,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        _use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        // plakat runs Gemma as a single-shot encoder (no flash-attn feature): the standard
        // softmax path only. `attn_logit_softcapping` (Gemma-2) and the additive causal+padding
        // mask are applied before softmax.
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, use_flash_attn, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: Option<usize>,
}

impl Model {
    /// Load a Gemma-2 model. NB: Sana's `text_encoder/` is a standalone `Gemma2Model` whose
    /// weights sit at the **root** (`embed_tokens`, `layers.*`, `norm`) — no `model.` prefix — so
    /// this vendored copy loads at the VarBuilder root (unlike the upstream `GemmaForCausalLM`,
    /// which nests under `model.`). Pass a VarBuilder rooted at the text-encoder weights.
    pub fn new(use_flash_attn: bool, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb;
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb_m.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), use_flash_attn, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb_m.device().clone(),
            dtype: vb_m.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = match self.sliding_window {
            None => (0..tgt_len)
                .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
                .collect(),
            Some(sliding_window) => (0..tgt_len)
                .flat_map(|i| {
                    (0..tgt_len).map(move |j| {
                        if i < j || j + sliding_window < i {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect(),
        };
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };

        Ok(logits)
    }

    /// Encoder-style forward for Sana: returns the **last hidden state** over ALL positions
    /// (`embed → layers → final RMSNorm`; no `lm_head`, no last-token narrow), shape
    /// `(B, L, hidden)`. `attention_mask` is `(B, L)` with `1.0` for real tokens / `0.0` for
    /// padding; it is combined with the causal mask so padding KEYS are ignored. (Sana
    /// right-pads and the DiT caption re-slice keeps some padding positions, so the mask matters.)
    pub fn forward_hidden(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.clear_kv_cache(); // idempotent re-encode (single-shot; kv-cache is unused here)
        let (b_size, seq_len) = input_ids.dims2()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            let causal = self.prepare_decoder_attention_mask(b_size, seq_len, 0)?; // (B,1,L,L)
            let combined = match attention_mask {
                None => causal,
                Some(pad) => {
                    // (B,L) 1/0 → key bias (B,1,1,L): real→0, pad→-1e30 (finite, softmax→0).
                    let bias = pad
                        .to_dtype(DType::F32)?
                        .affine(1e30, -1e30)?
                        .reshape((b_size, 1, 1, seq_len))?
                        .to_dtype(self.dtype)?;
                    causal.broadcast_add(&bias)?
                }
            };
            Some(combined)
        };
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref(), 0)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1().unwrap();
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in a.iter().zip(&b) {
            num += (x - ma) * (y - mb);
            da += (x - ma).powi(2);
            db += (y - mb).powi(2);
        }
        num / (da.sqrt() * db.sqrt() + 1e-12)
    }

    /// Verify `forward_hidden` (all-position last hidden state + padding mask) against a diffusers
    /// dump (`tools/reference/sana_gemma_dump.py`). Opt-in: `PLAKAT_GEMMA_VERIFY=1`; the Sana
    /// text_encoder must be cached. F32/CPU canonical.
    #[test]
    fn gemma_forward_hidden_matches_diffusers() {
        if std::env::var("PLAKAT_GEMMA_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let home = std::env::var("HOME").unwrap();
        let te = {
            let base = format!(
                "{home}/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots"
            );
            let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
            snap.join("text_encoder")
        };
        let cfg: Config = serde_json::from_reader(
            std::fs::File::open(te.join("config.json")).unwrap(),
        )
        .unwrap();
        let shards = [
            te.join("model-00001-of-00002.safetensors"),
            te.join("model-00002-of-00002.safetensors"),
        ];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &dev).unwrap() };
        let mut model = Model::new(false, &cfg, vb).unwrap();

        let g = candle_core::safetensors::load(
            "tools/reference/out/sana-gemma/goldens.safetensors",
            &dev,
        )
        .unwrap();
        let input_ids = g["input_ids"].to_dtype(DType::U32).unwrap();
        let mask = &g["attention_mask"];
        let raw_ref = &g["raw_hidden"];
        let final_ref = &g["final_embeds"];

        let hidden = model.forward_hidden(&input_ids, Some(mask)).unwrap();
        let c = corr(&hidden, raw_ref);
        eprintln!("raw_hidden:   corr={c:.6} shape={:?}", hidden.dims());
        assert!(c > 0.999, "gemma forward_hidden corr {c} < 0.999");

        // encode_prompt re-slice: [0] + last 299 → (1,300,2304).
        const MAX_SEQ: usize = 300;
        let l = hidden.dim(1).unwrap();
        let bos = hidden.narrow(1, 0, 1).unwrap();
        let tail = hidden.narrow(1, l - (MAX_SEQ - 1), MAX_SEQ - 1).unwrap();
        let resliced = Tensor::cat(&[bos, tail], 1).unwrap();
        let c2 = corr(&resliced, final_ref);
        eprintln!("final_embeds: corr={c2:.6} shape={:?}", resliced.dims());
        assert!(c2 > 0.999, "gemma reslice corr {c2} < 0.999");
    }
}
