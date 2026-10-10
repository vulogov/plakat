//! Vendored Qwen2 decoder — the text tower of Kandinsky 5's Qwen2.5-VL (RFC KANDINSKY-1, P1).
//!
//! Adapted from candle-transformers 0.10.2 `models::qwen2::Model` (same tensor keys, candle's own
//! `qwen2::Config`) for **encoder** use, and forked for one reason: precision. The weights stay in the
//! dtype they are loaded in (BF16 on a GPU — what the checkpoint stores, ≈14 GB), but every activation is
//! computed in **F32**: each layer's weights are widened for the duration of that layer's forward and
//! dropped again (≈1 GB transient). Widening BF16 weights is exact, so the result is the F32 reference's.
//!
//! Why: with everything in BF16, candle on Metal reaches only cosine 0.94 / 0.90 against the F32
//! reference on the real tower (measured, P1 parity), where transformers' own BF16 run holds 0.9999 /
//! 0.9996. Attention logits, the SwiGLU product and the residual stream rounded to 8 bits of mantissa
//! at every op do not survive 28 layers of this model's activations.
//!
//! Encoder-only: one causal forward over the whole sequence, all positions returned after the final
//! norm. No KV cache, no padding mask (candle's masked path is bidirectional — RFC trap T3), no
//! `ModelForCausalLM`.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, VarBuilder};
pub use candle_transformers::models::qwen2::Config;

/// The dtype activations are computed in, whatever the weights are stored as.
const COMPUTE: DType = DType::F32;

/// GQA key/value head expansion (candle-transformers `utils::repeat_kv`).
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// `xs` (F32) through a linear layer whose weights are widened to F32 for this call only.
fn wide(l: &Linear, xs: &Tensor) -> Result<Tensor> {
    let bias = l.bias().map(|b| b.to_dtype(COMPUTE)).transpose()?;
    Linear::new(l.weight().to_dtype(COMPUTE)?, bias).forward(xs)
}

/// RMSNorm over the last dim, in F32, with the stored weight widened.
fn rms_norm(weight: &Tensor, eps: f64, xs: &Tensor) -> Result<Tensor> {
    let hidden = xs.dim(D::Minus1)?;
    let ms = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    xs.broadcast_div(&(ms + eps)?.sqrt()?)?.broadcast_mul(&weight.to_dtype(COMPUTE)?)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim).step_by(2).map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32).collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?.to_dtype(COMPUTE)?.reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        Ok((candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?, candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?))
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    act_fn: Activation,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let (num_heads, num_kv_heads) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let head_dim = h / num_heads;
        let (va, vm) = (vb.pp("self_attn"), vb.pp("mlp"));
        Ok(Self {
            q_proj: linear(h, num_heads * head_dim, va.pp("q_proj"))?,
            k_proj: linear(h, num_kv_heads * head_dim, va.pp("k_proj"))?,
            v_proj: linear(h, num_kv_heads * head_dim, va.pp("v_proj"))?,
            o_proj: linear_no_bias(num_heads * head_dim, h, va.pp("o_proj"))?,
            gate_proj: linear_no_bias(h, inter, vm.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, inter, vm.pp("up_proj"))?,
            down_proj: linear_no_bias(inter, h, vm.pp("down_proj"))?,
            input_layernorm: vb.pp("input_layernorm").get(h, "weight")?,
            post_attention_layernorm: vb.pp("post_attention_layernorm").get(h, "weight")?,
            act_fn: cfg.hidden_act,
            num_heads,
            num_kv_heads,
            head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn attention(&self, xs: &Tensor, rope: &RotaryEmbedding, causal: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, hidden) = xs.dims3()?;
        let heads = |t: Tensor, n: usize| t.reshape((b_sz, q_len, n, self.head_dim))?.transpose(1, 2);
        let q = heads(wide(&self.q_proj, xs)?, self.num_heads)?;
        let k = heads(wide(&self.k_proj, xs)?, self.num_kv_heads)?;
        let v = heads(wide(&self.v_proj, xs)?, self.num_kv_heads)?;
        let (q, k) = rope.apply(&q, &k)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, groups)?.contiguous()?;
        let v = repeat_kv(v, groups)?.contiguous()?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let weights = match causal {
            None => weights,
            Some(mask) => weights.broadcast_add(mask)?,
        };
        let out = candle_nn::ops::softmax_last_dim(&weights)?.matmul(&v)?;
        wide(&self.o_proj, &out.transpose(1, 2)?.reshape((b_sz, q_len, hidden))?)
    }

    fn forward(&self, xs: &Tensor, rope: &RotaryEmbedding, causal: Option<&Tensor>) -> Result<Tensor> {
        let attn = self.attention(&rms_norm(&self.input_layernorm, self.eps, xs)?, rope, causal)?;
        let xs = (xs + attn)?;
        let n = rms_norm(&self.post_attention_layernorm, self.eps, &xs)?;
        let mlp = wide(&self.down_proj, &(wide(&self.gate_proj, &n)?.apply(&self.act_fn)? * wide(&self.up_proj, &n)?)?)?;
        xs + mlp
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: Tensor,
    rope: RotaryEmbedding,
    eps: f64,
    device: Device,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let vb_l = vb_m.pp("layers");
        let layers = (0..cfg.num_hidden_layers).map(|i| DecoderLayer::new(cfg, vb_l.pp(i))).collect::<Result<Vec<_>>>()?;
        let norm = vb_m.pp("norm").get(cfg.hidden_size, "weight")?;
        Ok(Self { embed_tokens, layers, norm, rope: RotaryEmbedding::new(cfg, vb.device())?, eps: cfg.rms_norm_eps, device: vb.device().clone() })
    }

    fn causal_mask(&self, len: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..len).flat_map(|i| (0..len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. })).collect();
        Tensor::from_slice(&mask, (1, 1, len, len), &self.device)
    }

    /// Token ids `(B, L)` → the last hidden states `(B, L, hidden)` after the final norm, in F32. Causal,
    /// unpadded: every sequence in the batch is taken whole.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let causal = if seq_len <= 1 { None } else { Some(self.causal_mask(seq_len)?) };
        let mut xs = self.embed_tokens.forward(input_ids)?.to_dtype(COMPUTE)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rope, causal.as_ref())?
        }
        rms_norm(&self.norm, self.eps, &xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let (a, b) = (a.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap(), b.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap());
        let dot = (&a * &b).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        let n = |t: &Tensor| t.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
        dot / (n(&a) * n(&b)).max(1e-12)
    }

    /// A small random tower with BF16-representable weights, CPU F32 against the GPU with the weights
    /// stored as F32, BF16 and F16 — no checkpoint needed. Activations are F32 whatever the storage, so
    /// BF16 storage must agree with F32 (all-BF16 compute gave 0.9992 here at 12 layers, 0.94 on the real
    /// tower): `cargo test --release --features metal --lib qwen2_gpu_dtypes -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn qwen2_gpu_dtypes_track_cpu_f32() {
        let gpu = crate::device::select("auto").unwrap();
        for layers in [1usize, 4, 12] {
            let cfg = Config { vocab_size: 512, hidden_size: 448, intermediate_size: 2368, num_hidden_layers: layers, num_attention_heads: 7, num_key_value_heads: 1, max_position_embeddings: 1024, sliding_window: 1024, max_window_layers: layers, tie_word_embeddings: false, rope_theta: 1e6, rms_norm_eps: 1e-6, use_sliding_window: false, hidden_act: Activation::Silu };
            let vm = VarMap::new();
            Model::new(&cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap(); // fills the VarMap
            let weights: std::collections::HashMap<String, Tensor> = vm.data().lock().unwrap().iter().map(|(k, v)| {
                    // A VarMap fills the bare norm weights with zeros; a norm's weight starts at one.
                    let t = if k.ends_with("norm.weight") { v.as_tensor().ones_like().unwrap() } else { v.as_tensor().clone() };
                    (k.clone(), t.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap())
                })
                .collect();
            let cpu = Model::new(&cfg, VarBuilder::from_tensors(weights.clone(), DType::F32, &Device::Cpu)).unwrap();
            let ids: Vec<u32> = (0..42u32).map(|i| (i * 37 + 11) % 512).collect();
            let want = cpu.forward(&Tensor::new(ids.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap()).unwrap();
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let on_gpu: std::collections::HashMap<String, Tensor> = weights.iter().map(|(k, v)| (k.clone(), v.to_device(&gpu).unwrap().to_dtype(dtype).unwrap())).collect();
                let m = Model::new(&cfg, VarBuilder::from_tensors(on_gpu, dtype, &gpu)).unwrap();
                let got = m.forward(&Tensor::new(ids.as_slice(), &gpu).unwrap().unsqueeze(0).unwrap()).unwrap();
                let c = cosine(&got, &want);
                println!("{layers:>2} layers · weights {dtype:?}: cosine {c:.6}");
                assert_eq!(got.dtype(), DType::F32);
                if dtype != DType::F16 {
                    assert!(c >= 0.99999, "{layers} layers, {dtype:?} weights: cosine {c}");
                }
            }
        }
    }
}
