//! Sana Linear-DiT — `SanaTransformer2DModel` in candle (ROADMAP_4.5.0 Phase 3; variants 4.6.0).
//!
//! A DiT whose dimensions come from a [`Config`] read from `transformer/config.json`, so every base
//! Sana variant loads (1.6B/1024, 0.6B, 512, 2K — same DC-AE + Gemma-2). Two novelties vs PixArt's
//! DiT. First, **ReLU linear
//! self-attention** — `relu(Q)·(relu(K)ᵀV)` with a ones-row denominator, F32 reduction island (not
//! self-normalizing, would NaN in F16); cross-attention to the caption stays **vanilla softmax**.
//! Second, a **GLU-MBConv Mix-FFN** (pointwise-expand → 3×3 depthwise → GLU gate → pointwise
//! project) operating on the tokens reshaped back to the 2D latent grid.
//!
//! Timestep conditioning is **AdaLN-single** (a shared 6-chunk `scale_shift_table` per block, like
//! PixArt). Patch size 1 → the 32×32×32 DC-AE latent is 1024 tokens; patchify/unpatchify are trivial.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder, conv2d, conv2d_no_bias, linear,
    linear_no_bias,
};

use super::pixart_dit::TimestepEmbedder;

const NORM_EPS: f64 = 1e-6;
const CAPTION_RMS_EPS: f64 = 1e-5;
const ATTN_EPS: f64 = 1e-15;
const QK_NORM_EPS: f64 = 1e-5; // Sana-1.5 qk_norm (Attention default eps)

/// DiT dimensions, read from `transformer/config.json` so every Sana variant loads (1.6B, 600M,
/// 512/2K share arches). `hidden = heads·head_dim`; `mlp_hidden = ⌊mlp_ratio·hidden⌋` (diffusers `int()`).
#[derive(Clone, Copy)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub cross_heads: usize,
    pub cross_head_dim: usize,
    pub caption_ch: usize,
    pub out_ch: usize,
    pub mlp_hidden: usize,
    /// Sana-1.5: `qk_norm = rms_norm_across_heads` → an RMSNorm over the full inner dim on q/k.
    pub qk_norm: bool,
}

impl Config {
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)
            .with_context(|| format!("parsing Sana transformer config {}", path.display()))?;
        let g = |k: &str| -> Result<usize> {
            v[k].as_u64().map(|x| x as usize).with_context(|| format!("config field {k}"))
        };
        let heads = g("num_attention_heads")?;
        let head_dim = g("attention_head_dim")?;
        let mlp_ratio = v["mlp_ratio"].as_f64().unwrap_or(2.5);
        let hidden = heads * head_dim;
        // Only `rms_norm_across_heads` (Sana-1.5) is supported; other qk_norm kinds bail.
        let qk_norm = match v.get("qk_norm").and_then(|x| x.as_str()) {
            None | Some("null") => false,
            Some("rms_norm_across_heads") => true,
            Some(other) => bail!("Sana qk_norm {other:?} not supported (only rms_norm_across_heads)."),
        };
        Ok(Config {
            layers: g("num_layers")?,
            hidden,
            heads,
            head_dim,
            cross_heads: g("num_cross_attention_heads")?,
            cross_head_dim: g("cross_attention_head_dim")?,
            caption_ch: g("caption_channels")?,
            out_ch: g("out_channels")?,
            mlp_hidden: (mlp_ratio * hidden as f64) as usize,
            qk_norm,
        })
    }
}

fn cfg1(groups: usize) -> Conv2dConfig {
    Conv2dConfig { padding: 0, stride: 1, dilation: 1, groups, cudnn_fwd_algo: None }
}

/// LayerNorm with `elementwise_affine=False` (no weight/bias) over the last dim.
fn layer_norm_noaffine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc.broadcast_div(&(var + eps)?.sqrt()?)?)
}

/// RMSNorm over the last dim with a weight (no `+1`, unlike Gemma), eps 1e-5.
fn rms_norm_last(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let xn = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(xn.broadcast_mul(weight)?)
}

// ── ReLU linear self-attention ───────────────────────────────────────────────────────────────
struct LinearSelfAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Option<Tensor>, // Sana-1.5 qk_norm (RMSNorm over the full inner dim)
    norm_k: Option<Tensor>,
    c: Config,
}
impl LinearSelfAttn {
    fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        let (norm_q, norm_k) = if c.qk_norm {
            (Some(vb.get(c.hidden, "norm_q.weight")?), Some(vb.get(c.hidden, "norm_k.weight")?))
        } else {
            (None, None)
        };
        // attention_bias = False → q/k/v have no bias; to_out.0 has bias.
        Ok(Self {
            to_q: linear_no_bias(c.hidden, c.hidden, vb.pp("to_q"))?,
            to_k: linear_no_bias(c.hidden, c.hidden, vb.pp("to_k"))?,
            to_v: linear_no_bias(c.hidden, c.hidden, vb.pp("to_v"))?,
            to_out: linear(c.hidden, c.hidden, vb.pp("to_out.0"))?,
            norm_q,
            norm_k,
            c,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (heads, hd, hidden) = (self.c.heads, self.c.head_dim, self.c.hidden);
        let (b, n, _) = x.dims3()?;
        let orig = x.dtype();
        // (B,N,inner) → (B·heads, head_dim, N), collapsing the batch: candle 0.10.2's Metal matmul
        // rejects 4-D batched matmuls, so the linear-attention reduction runs 3-D batched.
        let bh = b * heads;
        let to3 = |t: Tensor| -> Result<Tensor> {
            Ok(t.transpose(1, 2)?.reshape((b, heads, hd, n))?.contiguous()?.reshape((bh, hd, n))?)
        };
        // qk_norm (Sana-1.5): RMSNorm over the full inner dim on q/k, before the head reshape.
        let qk = |t: Tensor, w: &Option<Tensor>| -> Result<Tensor> {
            match w {
                Some(w) => rms_norm_last(&t, w, QK_NORM_EPS),
                None => Ok(t),
            }
        };
        let q = to3(qk(self.to_q.forward(x)?, &self.norm_q)?)?.relu()?.to_dtype(DType::F32)?;
        let k = to3(qk(self.to_k.forward(x)?, &self.norm_k)?)?.relu()?.to_dtype(DType::F32)?;
        let v = to3(self.to_v.forward(x)?)?.to_dtype(DType::F32)?;
        let ones = Tensor::ones((bh, 1, n), DType::F32, v.device())?;
        let v = Tensor::cat(&[v, ones], 1)?; // (bh, head_dim+1, N)
        let scores = v.matmul(&k.transpose(1, 2)?.contiguous()?)?; // (bh, head_dim+1, head_dim)
        let out = scores.matmul(&q)?; // (bh, head_dim+1, N)
        let num = out.narrow(1, 0, hd)?;
        let den = (out.narrow(1, hd, 1)? + ATTN_EPS)?;
        let attn = num.broadcast_div(&den)?; // (bh, head_dim, N)
        let attn = attn.reshape((b, hidden, n))?.transpose(1, 2)?.contiguous()?.to_dtype(orig)?; // (B,N,inner)
        Ok(self.to_out.forward(&attn)?)
    }
}

// ── vanilla softmax cross-attention ──────────────────────────────────────────────────────────
struct CrossAttn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: Option<Tensor>, // Sana-1.5 qk_norm
    norm_k: Option<Tensor>,
    c: Config,
}
impl CrossAttn {
    fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        // attn2: bias = True (q/k/v), out_bias = True. inner = cross_heads·cross_head_dim (= hidden).
        let inner = c.cross_heads * c.cross_head_dim;
        let (norm_q, norm_k) = if c.qk_norm {
            (Some(vb.get(inner, "norm_q.weight")?), Some(vb.get(inner, "norm_k.weight")?))
        } else {
            (None, None)
        };
        Ok(Self {
            to_q: linear(c.hidden, inner, vb.pp("to_q"))?,
            to_k: linear(c.hidden, inner, vb.pp("to_k"))?,
            to_v: linear(c.hidden, inner, vb.pp("to_v"))?,
            to_out: linear(inner, c.hidden, vb.pp("to_out.0"))?,
            norm_q,
            norm_k,
            c,
        })
    }
    /// `enc`: (B, L, HIDDEN) caption; `mask_bias`: (B,1,1,L) additive (0 keep / -10000 pad) or None.
    fn forward(&self, x: &Tensor, enc: &Tensor, mask_bias: Option<&Tensor>) -> Result<Tensor> {
        let (heads, hd, hidden) = (self.c.cross_heads, self.c.cross_head_dim, self.c.hidden);
        let (b, n, _) = x.dims3()?;
        let l = enc.dim(1)?;
        let bh = b * heads;
        // qk_norm (Sana-1.5): RMSNorm over the full inner dim on q/k, before the head reshape.
        let qk = |t: Tensor, w: &Option<Tensor>| -> Result<Tensor> {
            match w {
                Some(w) => rms_norm_last(&t, w, QK_NORM_EPS),
                None => Ok(t),
            }
        };
        // Collapse (B,heads) → 3-D batched matmul (candle Metal rejects 4-D batched).
        let q = qk(self.to_q.forward(x)?, &self.norm_q)?.reshape((b, n, heads, hd))?.transpose(1, 2)?.contiguous()?.reshape((bh, n, hd))?;
        let k = qk(self.to_k.forward(enc)?, &self.norm_k)?.reshape((b, l, heads, hd))?.transpose(1, 2)?.contiguous()?.reshape((bh, l, hd))?;
        let v = self.to_v.forward(enc)?.reshape((b, l, heads, hd))?.transpose(1, 2)?.contiguous()?.reshape((bh, l, hd))?;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // (bh, N, L)
        if let Some(bias) = mask_bias {
            // (B,1,1,L) → (bh,1,L): same mask for every head of a batch item.
            let m = bias.reshape((b, 1, l))?.broadcast_as((b, heads, l))?.reshape((bh, 1, l))?;
            scores = scores.broadcast_add(&m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // (bh, N, head_dim)
        let out = out.reshape((b, heads, n, hd))?.transpose(1, 2)?.contiguous()?.reshape((b, n, hidden))?;
        Ok(self.to_out.forward(&out)?)
    }
}

// ── GLU-MBConv Mix-FFN (norm_type=None, residual handled by the block) ───────────────────────
struct MixFfn {
    conv_inverted: Conv2d,
    conv_depth: Conv2d,
    conv_point: Conv2d,
}
impl MixFfn {
    fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        let (h, m) = (c.hidden, c.mlp_hidden);
        Ok(Self {
            conv_inverted: conv2d(h, m * 2, 1, cfg1(1), vb.pp("conv_inverted"))?,
            conv_depth: conv2d(m * 2, m * 2, 3, Conv2dConfig { padding: 1, stride: 1, dilation: 1, groups: m * 2, cudnn_fwd_algo: None }, vb.pp("conv_depth"))?,
            conv_point: conv2d_no_bias(m, h, 1, cfg1(1), vb.pp("conv_point"))?,
        })
    }
    /// `x`: (B, HIDDEN, H, W).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv_inverted.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv_depth.forward(&h)?;
        let half = h.dim(1)? / 2;
        let a = h.narrow(1, 0, half)?;
        let gate = h.narrow(1, half, half)?;
        let h = (a * candle_nn::ops::silu(&gate)?)?;
        Ok(self.conv_point.forward(&h)?)
    }
}

// ── SanaTransformerBlock ─────────────────────────────────────────────────────────────────────
struct Block {
    attn1: LinearSelfAttn,
    attn2: CrossAttn,
    ff: MixFfn,
    scale_shift_table: Tensor, // (6, hidden)
    c: Config,
}
impl Block {
    fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn1: LinearSelfAttn::load(c, vb.pp("attn1"))?,
            attn2: CrossAttn::load(c, vb.pp("attn2"))?,
            ff: MixFfn::load(c, vb.pp("ff"))?,
            scale_shift_table: vb.get((6, c.hidden), "scale_shift_table")?,
            c,
        })
    }
    /// `temb`: (B, 6*HIDDEN) AdaLN-single timestep. `hw`: (H,W) latent grid.
    fn forward(&self, x: &Tensor, enc: &Tensor, mask_bias: Option<&Tensor>, temb: &Tensor, hw: (usize, usize)) -> Result<Tensor> {
        let hidden = self.c.hidden;
        let (b, _n, _) = x.dims3()?;
        // modulation: (scale_shift_table[None] + temb.reshape(B,6,-1)).chunk(6)
        let sst = self.scale_shift_table.reshape((1, 6, hidden))?;
        let temb6 = temb.reshape((b, 6, hidden))?;
        let m = sst.broadcast_add(&temb6)?; // (B,6,HIDDEN)
        let chunk = |i: usize| m.narrow(1, i, 1)?.squeeze(1); // (B,HIDDEN)
        let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
        let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);

        // self-attention
        let norm = layer_norm_noaffine(x, NORM_EPS)?;
        let norm = norm.broadcast_mul(&(scale_msa.unsqueeze(1)? + 1.0)?)?.broadcast_add(&shift_msa.unsqueeze(1)?)?;
        let attn = self.attn1.forward(&norm)?;
        let x = (x + attn.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;

        // cross-attention (on raw x, no pre-norm — matches diffusers)
        let attn = self.attn2.forward(&x, enc, mask_bias)?;
        let x = (attn + x)?;

        // feed-forward
        let (h, w) = hw;
        let norm = layer_norm_noaffine(&x, NORM_EPS)?;
        let norm = norm.broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?.broadcast_add(&shift_mlp.unsqueeze(1)?)?;
        // tokens (B,N,HIDDEN) → grid (B,HIDDEN,H,W)
        let grid = norm.reshape((b, h, w, hidden))?.permute((0, 3, 1, 2))?.contiguous()?;
        let ff = self.ff.forward(&grid)?; // (B,HIDDEN,H,W)
        let ff = ff.flatten(2, 3)?.permute((0, 2, 1))?.contiguous()?; // (B,N,HIDDEN)
        Ok((x + ff.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)?)
    }
}

/// The Sana 1.6B Linear-DiT. Predicts the flow-matching velocity for a `(B,32,H,W)` latent.
pub struct SanaTransformer {
    patch_proj: Conv2d, // patch_size 1 → 1×1 conv, in 32 → HIDDEN
    ts_embedder: TimestepEmbedder, // time_embed.emb.timestep_embedder
    time_linear: Linear, // time_embed.linear → 6*HIDDEN
    cap_linear1: Linear, // caption_projection.linear_1
    cap_linear2: Linear, // caption_projection.linear_2
    caption_norm_w: Tensor,
    blocks: Vec<Block>,
    scale_shift_table: Tensor, // final (2, hidden)
    norm_out: LayerNorm,       // SanaModulatedNorm's inner LayerNorm (no affine)
    proj_out: Linear,          // hidden → out_ch
    c: Config,
}

impl SanaTransformer {
    pub fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        let patch_proj = conv2d(c.out_ch, c.hidden, 1, cfg1(1), vb.pp("patch_embed.proj")).context("patch_embed")?;
        let te = vb.pp("time_embed");
        let ts_embedder = TimestepEmbedder::new(c.hidden, te.pp("emb.timestep_embedder"))
            .map_err(|e| anyhow::anyhow!("time_embed timestep_embedder: {e}"))?;
        let time_linear = linear(c.hidden, 6 * c.hidden, te.pp("linear"))?;
        let cap = vb.pp("caption_projection");
        let cap_linear1 = linear(c.caption_ch, c.hidden, cap.pp("linear_1"))?;
        let cap_linear2 = linear(c.hidden, c.hidden, cap.pp("linear_2"))?;
        let caption_norm_w = vb.get(c.hidden, "caption_norm.weight")?;
        let bvb = vb.pp("transformer_blocks");
        let mut blocks = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            blocks.push(Block::load(c, bvb.pp(i)).with_context(|| format!("block {i}"))?);
        }
        // SanaModulatedNorm's LayerNorm is elementwise_affine=False → build a zero/one LN.
        let norm_out = LayerNorm::new_no_bias(
            Tensor::ones(c.hidden, vb.dtype(), vb.device())?,
            NORM_EPS,
        );
        Ok(Self {
            patch_proj,
            ts_embedder,
            time_linear,
            cap_linear1,
            cap_linear2,
            caption_norm_w,
            blocks,
            scale_shift_table: vb.get((2, c.hidden), "scale_shift_table")?,
            norm_out,
            proj_out: linear(c.hidden, c.out_ch, vb.pp("proj_out"))?,
            c,
        })
    }

    /// The DiT hidden dim (`heads·head_dim`) — used to check ControlNet residual-width compatibility.
    pub fn hidden(&self) -> usize {
        self.c.hidden
    }

    /// `latent`: (B,32,H,W). `caption`: (B,L,2304). `timestep`: (B,). `enc_mask`: (B,L) 1/0 or None.
    pub fn forward(&self, latent: &Tensor, caption: &Tensor, timestep: &Tensor, enc_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_control(latent, caption, timestep, enc_mask, None)
    }

    /// Like [`forward`], plus optional ControlNet residuals: `residuals[i-1]` is added after block
    /// `i` for `1 ≤ i ≤ residuals.len()` (diffusers' injection window — block 0 and any block past the
    /// ControlNet's depth are untouched). `None` is byte-identical to [`forward`].
    pub fn forward_control(&self, latent: &Tensor, caption: &Tensor, timestep: &Tensor, enc_mask: Option<&Tensor>, controlnet_residuals: Option<&[Tensor]>) -> Result<Tensor> {
        let (b, _c, h, w) = latent.dims4()?;
        // patchify (patch_size 1): conv → (B,HIDDEN,H,W) → tokens (B,N,HIDDEN)
        let tokens = self.patch_proj.forward(latent)?.flatten(2, 3)?.permute((0, 2, 1))?.contiguous()?;

        // AdaLN-single: embedded_timestep (B,HIDDEN); temb (B,6*HIDDEN).
        let embedded_timestep = self.ts_embedder.forward(timestep).map_err(|e| anyhow::anyhow!("ts embed: {e}"))?;
        let temb = self.time_linear.forward(&candle_nn::ops::silu(&embedded_timestep)?)?;

        // caption projection + RMSNorm.
        let enc = self.cap_linear1.forward(caption)?;
        let enc = enc.gelu()?; // gelu_tanh
        let enc = self.cap_linear2.forward(&enc)?;
        let enc = rms_norm_last(&enc, &self.caption_norm_w, CAPTION_RMS_EPS)?;

        // encoder mask → additive bias (B,1,1,L).
        let mask_bias = match enc_mask {
            None => None,
            Some(m) => {
                let l = m.dim(1)?;
                Some(m.to_dtype(DType::F32)?.affine(10000.0, -10000.0)?.reshape((b, 1, 1, l))?.to_dtype(tokens.dtype())?)
            }
        };

        let mut x = tokens;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward(&x, &enc, mask_bias.as_ref(), &temb, (h, w))?;
            // ControlNet residual injection (diffusers window: after blocks 1..=len).
            if let Some(res) = controlnet_residuals {
                if i >= 1 && i <= res.len() {
                    x = x.broadcast_add(&res[i - 1])?;
                }
            }
        }

        // SanaModulatedNorm: LN(no affine) then shift/scale from scale_shift_table(2) + embedded_timestep.
        let (hidden, out_ch) = (self.c.hidden, self.c.out_ch);
        let x = self.norm_out.forward(&x)?;
        let sst = self.scale_shift_table.reshape((1, 2, hidden))?;
        let et = embedded_timestep.reshape((b, 1, hidden))?;
        let m = sst.broadcast_add(&et)?; // (B,2,hidden)
        let shift = m.narrow(1, 0, 1)?; // (B,1,hidden)
        let scale = m.narrow(1, 1, 1)?;
        let x = x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;

        let x = self.proj_out.forward(&x)?; // (B,N,out_ch)
        // unpatchify (patch 1): (B,N,out_ch) → (B,out_ch,H,W)
        Ok(x.reshape((b, h, w, out_ch))?.permute((0, 3, 1, 2))?.contiguous()?)
    }
}

// ── SanaControlNet ───────────────────────────────────────────────────────────────────────────
/// `SanaControlNetModel` — a truncated copy of the Sana DiT (the first `layers` blocks) that
/// consumes a **DC-AE-encoded control latent** and emits one residual per block, added into the
/// main DiT's hidden state (after blocks 1..=layers). Shares the block/embed machinery above; the
/// only extra weights are `input_block` (a Linear on the patch-embedded control) and one zero-init
/// `controlnet_blocks[i]` Linear per block that projects the block output into a residual.
///
/// The public ControlNets are 600M-dim (`inner=1152`, 7 blocks) — pair with `sana-600m`.
pub struct SanaControlNet {
    c: Config,
    patch_proj: Conv2d,
    input_block: Linear,
    ts_embedder: TimestepEmbedder,
    time_linear: Linear,
    cap_linear1: Linear,
    cap_linear2: Linear,
    caption_norm_w: Tensor,
    blocks: Vec<Block>,
    controlnet_blocks: Vec<Linear>,
}

impl SanaControlNet {
    pub fn load(c: Config, vb: VarBuilder) -> Result<Self> {
        let patch_proj = conv2d(c.out_ch, c.hidden, 1, cfg1(1), vb.pp("patch_embed.proj")).context("cn patch_embed")?;
        let input_block = linear(c.hidden, c.hidden, vb.pp("input_block")).context("cn input_block")?;
        let te = vb.pp("time_embed");
        let ts_embedder = TimestepEmbedder::new(c.hidden, te.pp("emb.timestep_embedder"))
            .map_err(|e| anyhow::anyhow!("cn time_embed timestep_embedder: {e}"))?;
        let time_linear = linear(c.hidden, 6 * c.hidden, te.pp("linear"))?;
        let cap = vb.pp("caption_projection");
        let cap_linear1 = linear(c.caption_ch, c.hidden, cap.pp("linear_1"))?;
        let cap_linear2 = linear(c.hidden, c.hidden, cap.pp("linear_2"))?;
        let caption_norm_w = vb.get(c.hidden, "caption_norm.weight")?;
        let bvb = vb.pp("transformer_blocks");
        let cbvb = vb.pp("controlnet_blocks");
        let mut blocks = Vec::with_capacity(c.layers);
        let mut controlnet_blocks = Vec::with_capacity(c.layers);
        for i in 0..c.layers {
            blocks.push(Block::load(c, bvb.pp(i)).with_context(|| format!("cn block {i}"))?);
            controlnet_blocks.push(linear(c.hidden, c.hidden, cbvb.pp(i)).with_context(|| format!("cn controlnet_block {i}"))?);
        }
        Ok(Self {
            c,
            patch_proj,
            input_block,
            ts_embedder,
            time_linear,
            cap_linear1,
            cap_linear2,
            caption_norm_w,
            blocks,
            controlnet_blocks,
        })
    }

    /// `latent`: (B,32,H,W) noisy latent. `control`: (B,32,H,W) DC-AE control latent. `caption`,
    /// `timestep`, `enc_mask` as the main DiT. Returns one residual per block, each `× scale`.
    pub fn forward(
        &self,
        latent: &Tensor,
        control: &Tensor,
        caption: &Tensor,
        timestep: &Tensor,
        enc_mask: Option<&Tensor>,
        scale: f64,
    ) -> Result<Vec<Tensor>> {
        let (b, _c, h, w) = latent.dims4()?;
        let patch = |t: &Tensor| -> Result<Tensor> {
            Ok(self.patch_proj.forward(t)?.flatten(2, 3)?.permute((0, 2, 1))?.contiguous()?)
        };
        // patch-embed both the noisy latent and the control latent (same proj); add input_block(control).
        let tokens = patch(latent)?;
        let ctrl = self.input_block.forward(&patch(control)?)?;
        let mut x = (tokens + ctrl)?;

        let embedded_timestep = self.ts_embedder.forward(timestep).map_err(|e| anyhow::anyhow!("cn ts embed: {e}"))?;
        let temb = self.time_linear.forward(&candle_nn::ops::silu(&embedded_timestep)?)?;

        let enc = self.cap_linear1.forward(caption)?;
        let enc = enc.gelu()?;
        let enc = self.cap_linear2.forward(&enc)?;
        let enc = rms_norm_last(&enc, &self.caption_norm_w, CAPTION_RMS_EPS)?;

        let mask_bias = match enc_mask {
            None => None,
            Some(m) => {
                let l = m.dim(1)?;
                Some(m.to_dtype(DType::F32)?.affine(10000.0, -10000.0)?.reshape((b, 1, 1, l))?.to_dtype(x.dtype())?)
            }
        };

        // Run the blocks, projecting each output into a scaled residual.
        let mut residuals = Vec::with_capacity(self.blocks.len());
        for (blk, cn) in self.blocks.iter().zip(&self.controlnet_blocks) {
            x = blk.forward(&x, &enc, mask_bias.as_ref(), &temb, (h, w))?;
            let r = cn.forward(&x)?;
            residuals.push((r * scale)?);
        }
        let _ = self.c;
        Ok(residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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

    /// Verify the candle Sana DiT against a diffusers dump (`tools/reference/sana_dit_dump.py`):
    /// a single forward with fixed latent / caption / timestep / mask. Opt-in
    /// (`PLAKAT_SANADIT_VERIFY=1`); the Sana transformer must be cached. F32/CPU canonical.
    #[test]
    fn sana_dit_matches_diffusers() {
        if std::env::var("PLAKAT_SANADIT_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots"
        );
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
        let tdir = snap.join("transformer");
        let shards = [
            tdir.join("diffusion_pytorch_model-00001-of-00002.safetensors"),
            tdir.join("diffusion_pytorch_model-00002-of-00002.safetensors"),
        ];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &dev).unwrap() };
        let cfg = Config::from_json(&tdir.join("config.json")).unwrap();
        let model = SanaTransformer::load(cfg, vb).unwrap();

        let g = candle_core::safetensors::load("tools/reference/out/sana-dit/goldens.safetensors", &dev).unwrap();
        let out = model
            .forward(&g["latent"], &g["caption"], &g["timestep"], Some(&g["mask"]))
            .unwrap();
        let c = corr(&out, &g["output"]);
        eprintln!("sana_dit output: corr={c:.6} shape={:?}", out.dims());
        assert!(c > 0.999, "sana dit corr {c} < 0.999");
    }

    /// Verify the Sana-1.5 DiT (qk_norm = rms_norm_across_heads) against a diffusers dump.
    /// Opt-in (`PLAKAT_SANA15_VERIFY=1`); SANA1.5 transformer cached + `sana-dit-15` goldens present.
    #[test]
    fn sana15_dit_matches_diffusers() {
        if std::env::var("PLAKAT_SANA15_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--Efficient-Large-Model--SANA1.5_1.6B_1024px_diffusers/snapshots"
        );
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
        let tdir = snap.join("transformer");
        let shards = [tdir.join("diffusion_pytorch_model.safetensors")]; // SANA1.5 is a single file
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &dev).unwrap() };
        let cfg = Config::from_json(&tdir.join("config.json")).unwrap();
        assert!(cfg.qk_norm, "SANA1.5 config should set qk_norm");
        let model = SanaTransformer::load(cfg, vb).unwrap();

        let g = candle_core::safetensors::load("tools/reference/out/sana-dit-15/goldens.safetensors", &dev).unwrap();
        let out = model.forward(&g["latent"], &g["caption"], &g["timestep"], Some(&g["mask"])).unwrap();
        let c = corr(&out, &g["output"]);
        eprintln!("sana1.5_dit output: corr={c:.6} shape={:?}", out.dims());
        assert!(c > 0.999, "sana-1.5 dit corr {c} < 0.999");
    }

    /// Verify the Sana ControlNet residuals against a diffusers dump. Opt-in
    /// (`PLAKAT_SANACN_VERIFY=1`); needs the 600M ControlNet cached + `sana-controlnet` goldens.
    #[test]
    fn sana_controlnet_matches_diffusers() {
        if std::env::var("PLAKAT_SANACN_VERIFY").is_err() {
            return;
        }
        let dev = Device::Cpu;
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--ishan24--Sana_600M_1024px_ControlNetPlus_diffusers/snapshots"
        );
        let snap = std::fs::read_dir(&base).unwrap().next().unwrap().unwrap().path();
        let cdir = snap.join("controlnet");
        let weights = [cdir.join("diffusion_pytorch_model.safetensors")];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &dev).unwrap() };
        let cfg = Config::from_json(&cdir.join("config.json")).unwrap();
        let model = SanaControlNet::load(cfg, vb).unwrap();

        let g = candle_core::safetensors::load("tools/reference/out/sana-controlnet/goldens.safetensors", &dev).unwrap();
        let res = model
            .forward(&g["latent"], &g["control"], &g["caption"], &g["timestep"], Some(&g["mask"]), 1.0)
            .unwrap();
        assert_eq!(res.len(), cfg.layers, "expected {} residuals", cfg.layers);
        let mut worst = 1.0f32;
        for (i, r) in res.iter().enumerate() {
            let c = corr(r, &g[&format!("res_{i}")]);
            eprintln!("sana_controlnet res_{i}: corr={c:.6} shape={:?}", r.dims());
            worst = worst.min(c);
        }
        assert!(worst > 0.999, "sana controlnet worst residual corr {worst} < 0.999");
    }
}
