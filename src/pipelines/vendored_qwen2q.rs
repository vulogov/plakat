//! Vendored quantized Qwen2 decoder — Kandinsky 5's text tower from a GGUF (RFC KANDINSKY-1, P4).
//!
//! Adapted from candle-transformers 0.10.2 `models::quantized_qwen2::ModelWeights` (same GGUF tensor
//! names and metadata keys) for **encoder** use. candle's forward returns the last token's logits; this
//! one returns every position's hidden state after the final norm, which is what the DiT is conditioned
//! on. It is [`super::vendored_qwen2`] with quantized matmuls: one causal forward over the whole
//! sequence, no KV cache, no padding mask, no output head.
//!
//! The quantized weights stay packed (≈4.7 GB at Q4_K_M) and are dequantized to F32 a matmul at a time;
//! activations are F32 throughout. The token embedding is the one table that is not a matmul: it is
//! dequantized once, held on the CPU in F16 (≈1.1 GB) and only the rows a prompt needs are widened.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Result, Tensor, D};

/// GQA key/value head expansion (candle-transformers `utils::repeat_kv`).
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// RMSNorm over the last dim, in F32.
fn rms_norm(weight: &Tensor, eps: f64, xs: &Tensor) -> Result<Tensor> {
    let hidden = xs.dim(D::Minus1)?;
    let ms = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    xs.broadcast_div(&(ms + eps)?.sqrt()?)?.broadcast_mul(weight)
}

struct Layer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    bq: Tensor,
    bk: Tensor,
    bv: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    attn_norm: Tensor,
    ffn_norm: Tensor,
}

pub struct Model {
    /// `(vocab, hidden)` in F16, on the CPU.
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f32,
    eps: f64,
    device: Device,
}

impl Model {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(ct: gguf_file::Content, reader: &mut R, device: &Device) -> Result<Self> {
        let md = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in the GGUF metadata"),
            Some(v) => Ok(v),
        };
        // llama.cpp names the architecture `qwen2` for a text Qwen2 and `qwen2vl` for the language
        // model of a Qwen2-VL; the keys below are the same under either prefix.
        let arch = md("general.architecture")?.to_string()?.clone();
        let md = |s: &str| md(&s.replacen("qwen2", &arch, 1));
        let heads = md("qwen2.attention.head_count")?.to_u32()? as usize;
        let kv_heads = md("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let hidden = md("qwen2.embedding_length")?.to_u32()? as usize;
        let blocks = md("qwen2.block_count")?.to_u32()? as usize;
        let eps = md("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_base = md("qwen2.rope.freq_base").and_then(|m| m.to_f32()).unwrap_or(10000f32);

        // First, and on the CPU: its F32 form (2.2 GB) is gone before the layers are read.
        let embed = ct.tensor(reader, "token_embd.weight", &Device::Cpu)?.dequantize(&Device::Cpu)?.to_dtype(DType::F16)?;
        let dense = |reader: &mut R, name: &str| -> Result<Tensor> { ct.tensor(reader, name, device)?.dequantize(device)?.to_dtype(DType::F32) };
        let quant = |reader: &mut R, name: &str| -> Result<QMatMul> { QMatMul::from_qtensor(ct.tensor(reader, name, device)?) };
        let mut layers = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let p = format!("blk.{i}");
            layers.push(Layer {
                wq: quant(reader, &format!("{p}.attn_q.weight"))?,
                wk: quant(reader, &format!("{p}.attn_k.weight"))?,
                wv: quant(reader, &format!("{p}.attn_v.weight"))?,
                wo: quant(reader, &format!("{p}.attn_output.weight"))?,
                bq: dense(reader, &format!("{p}.attn_q.bias"))?,
                bk: dense(reader, &format!("{p}.attn_k.bias"))?,
                bv: dense(reader, &format!("{p}.attn_v.bias"))?,
                gate: quant(reader, &format!("{p}.ffn_gate.weight"))?,
                up: quant(reader, &format!("{p}.ffn_up.weight"))?,
                down: quant(reader, &format!("{p}.ffn_down.weight"))?,
                attn_norm: dense(reader, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: dense(reader, &format!("{p}.ffn_norm.weight"))?,
            });
        }
        let norm = dense(reader, "output_norm.weight")?;
        Ok(Self { embed, layers, norm, heads, kv_heads, head_dim: hidden / heads, rope_base, eps, device: device.clone() })
    }

    /// `cos` and `sin` of the rotary angles for the first `len` positions: `(len, head_dim / 2)`.
    fn rope(&self, len: usize) -> Result<(Tensor, Tensor)> {
        let inv: Vec<f32> = (0..self.head_dim).step_by(2).map(|i| 1f32 / self.rope_base.powf(i as f32 / self.head_dim as f32)).collect();
        let n = inv.len();
        let inv = Tensor::from_vec(inv, (1, n), &self.device)?;
        let t = Tensor::arange(0u32, len as u32, &self.device)?.to_dtype(DType::F32)?.reshape((len, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    fn attention(&self, l: &Layer, xs: &Tensor, cos: &Tensor, sin: &Tensor, causal: Option<&Tensor>) -> Result<Tensor> {
        let (b, len, hidden) = xs.dims3()?;
        let heads = |t: Tensor, n: usize| t.reshape((b, len, n, self.head_dim))?.transpose(1, 2)?.contiguous();
        let q = heads(l.wq.forward(xs)?.broadcast_add(&l.bq)?, self.heads)?;
        let k = heads(l.wk.forward(xs)?.broadcast_add(&l.bk)?, self.kv_heads)?;
        let v = heads(l.wv.forward(xs)?.broadcast_add(&l.bv)?, self.kv_heads)?;
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(k, groups)?.contiguous()?;
        let v = repeat_kv(v, groups)?.contiguous()?;
        let weights = (q.matmul(&k.transpose(2, 3)?)? * (1f64 / (self.head_dim as f64).sqrt()))?;
        let weights = match causal {
            None => weights,
            Some(mask) => weights.broadcast_add(mask)?,
        };
        let out = candle_nn::ops::softmax_last_dim(&weights)?.matmul(&v)?;
        l.wo.forward(&out.transpose(1, 2)?.reshape((b, len, hidden))?)
    }

    /// Token ids `(B, L)` → the last hidden states `(B, L, hidden)` after the final norm, in F32. Causal,
    /// unpadded: every sequence in the batch is taken whole.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, len) = input_ids.dims2()?;
        let ids = input_ids.to_device(&Device::Cpu)?.flatten_all()?;
        let hidden = self.embed.dim(1)?;
        let mut xs = self.embed.index_select(&ids, 0)?.to_dtype(DType::F32)?.reshape((b, len, hidden))?.to_device(&self.device)?;
        let causal = if len <= 1 {
            None
        } else {
            let mask: Vec<f32> = (0..len).flat_map(|i| (0..len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. })).collect();
            Some(Tensor::from_slice(&mask, (1, 1, len, len), &self.device)?)
        };
        let (cos, sin) = self.rope(len)?;
        for l in &self.layers {
            let attn = self.attention(l, &rms_norm(&l.attn_norm, self.eps, &xs)?, &cos, &sin, causal.as_ref())?;
            xs = (xs + attn)?;
            let n = rms_norm(&l.ffn_norm, self.eps, &xs)?;
            let mlp = l.down.forward(&(candle_nn::ops::silu(&l.gate.forward(&n)?)? * l.up.forward(&n)?)?)?;
            xs = (xs + mlp)?;
        }
        rms_norm(&self.norm, self.eps, &xs)
    }
}
