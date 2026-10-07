//! The Kandinsky 5 DiT — `Kandinsky5Transformer3DModel` at T = 1 (RFC KANDINSKY-1 §6, P2).
//!
//! A 6B flow-matching transformer: two time-modulated text blocks (self-attention with 1-D RoPE), fifty
//! visual blocks (self-attention with 3-D RoPE, cross-attention to the text stream, feed-forward), and a
//! modulated output layer that predicts **velocity**. Module names are the diffusers ones, which are the
//! checkpoint's keys.
//!
//! * **Layout.** plakat hands latents over as NCHW `(B, 16, H/8, W/8)`; the reference works channels-last
//!   over 2×2 patches. The patchify and unpatchify permutes are done once per forward on the CPU (they
//!   are rank 6, and candle's Metal backend is only trusted up to rank 4 — RFC §14). Note the two orders
//!   differ: a patch goes IN as `(ph, pw, C)` and comes OUT as `(C, ph, pw)`.
//! * **Precision.** The checkpoint is BF16 throughout and the weights rest in BF16 on every device. The
//!   activations run in a *compute* dtype — F32 by default, BF16 on request on a GPU (see
//!   [`Dit::compute_dtype`]) — and a layer whose weights are narrower than its input widens them for
//!   that call. Whatever the compute
//!   dtype, the reference's F32 islands are F32 here too: the time embedding, every modulation, each
//!   modulate and residual add, the Q/K RMSNorm and the RoPE rotation.
//! * **RoPE** is a 2×2 rotation of ADJACENT channel pairs (not the half-split form), and the reference
//!   rounds the rotated Q and K to BF16 even when it runs in F32 (RFC §12.3). That rounding is mirrored
//!   by default; `PLAKAT_K5_ROPE_ROUND=0` turns it off.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

/// `transformer/config.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub in_visual_dim: usize,
    pub out_visual_dim: usize,
    pub time_dim: usize,
    pub model_dim: usize,
    pub ff_dim: usize,
    pub num_text_blocks: usize,
    pub num_visual_blocks: usize,
    /// RoPE dims of the (t, h, w) axes; their sum is the head dim.
    pub axes_dims: [usize; 3],
    pub in_text_dim: usize,
    pub in_text_dim2: usize,
}

/// The spatial patch (the temporal one is 1).
pub const PATCH: usize = 2;
const LN_EPS: f64 = 1e-5;
/// `nn.RMSNorm(eps=None)` uses the input dtype's epsilon, and the input is always cast to F32.
const RMS_EPS: f64 = f32::EPSILON as f64;
const ROPE_MAX_TEXT: usize = 1024;
const ROPE_MAX_AXIS: usize = 128;
/// Attention scores are computed a few heads at a time, so that one score tensor stays under this.
const ATTN_CHUNK_BYTES: usize = 1 << 30;
/// The same on a low-memory host ([`crate::pipelines::kandinsky::low_memory`]): at 1024² the denoise's
/// pool is 1.8 GB instead of 3.4 GB, for 18 % more time a step (measured with the NF4 DiT).
const ATTN_CHUNK_BYTES_LOW: usize = 1 << 27;

impl Config {
    /// Kandinsky 5.0 T2I Lite (RFC §4.2).
    pub fn lite() -> Self {
        Self { in_visual_dim: 16, out_visual_dim: 16, time_dim: 512, model_dim: 2560, ff_dim: 10240, num_text_blocks: 2, num_visual_blocks: 50, axes_dims: [32, 48, 48], in_text_dim: 3584, in_text_dim2: 768 }
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json).context("parsing transformer/config.json")?;
        let need = |k: &str| -> Result<usize> { v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| anyhow!("transformer/config.json has no `{k}`")) };
        let list = |k: &str| -> Result<Vec<usize>> { Ok(v.get(k).and_then(|x| x.as_array()).ok_or_else(|| anyhow!("transformer/config.json has no `{k}`"))?.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect()) };
        if list("patch_size")? != [1, PATCH, PATCH] {
            anyhow::bail!("Kandinsky 5 DiT with patch_size {:?} is not supported (expected [1, 2, 2])", list("patch_size")?);
        }
        if v.get("visual_cond").and_then(|x| x.as_bool()).unwrap_or(false) {
            anyhow::bail!("this is a visual-conditioned Kandinsky 5 DiT (image-to-image / video); only the T2I model is supported");
        }
        let attention = v.get("attention_type").and_then(|x| x.as_str()).unwrap_or("regular");
        if attention != "regular" {
            anyhow::bail!("Kandinsky 5 DiT with attention_type `{attention}` is not supported (expected regular)");
        }
        let axes: [usize; 3] = list("axes_dims")?.try_into().map_err(|_| anyhow!("transformer/config.json: axes_dims must have three entries"))?;
        let cfg = Self {
            in_visual_dim: need("in_visual_dim")?,
            out_visual_dim: need("out_visual_dim")?,
            time_dim: need("time_dim")?,
            model_dim: need("model_dim")?,
            ff_dim: need("ff_dim")?,
            num_text_blocks: need("num_text_blocks")?,
            num_visual_blocks: need("num_visual_blocks")?,
            axes_dims: axes,
            in_text_dim: need("in_text_dim")?,
            in_text_dim2: need("in_text_dim2")?,
        };
        if axes.iter().any(|a| a % 2 != 0) || cfg.model_dim % cfg.head_dim() != 0 {
            anyhow::bail!("Kandinsky 5 DiT: axes_dims {axes:?} do not form a head dim that divides model_dim {}", cfg.model_dim);
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.axes_dims.iter().sum()
    }

    pub fn heads(&self) -> usize {
        self.model_dim / self.head_dim()
    }
}

/// `exp(−ln(10000) · i / dim)` for `i` in `0..dim`, in F32 as the reference computes it.
fn freqs(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (-(10000f32.ln()) * i as f32 / dim as f32).exp()).collect()
}

/// The sinusoidal time features: `cat([cos(t·f), sin(t·f)])` — COS FIRST (trap T5). `t` is the raw
/// scheduler timestep, `sigma · 1000` (trap T2).
pub fn time_features(t: f32, model_dim: usize) -> Vec<f32> {
    let f = freqs(model_dim / 2);
    f.iter().map(|f| (t * f).cos()).chain(f.iter().map(|f| (t * f).sin())).collect()
}

/// RoPE angles for a text sequence: row `p` is `p · freqs(head_dim / 2)`.
pub fn rope_angles_1d(len: usize, head_dim: usize) -> Result<Vec<f32>> {
    if len > ROPE_MAX_TEXT {
        anyhow::bail!("Kandinsky 5: {len} text tokens is past the DiT's {ROPE_MAX_TEXT}-position RoPE table");
    }
    let f = freqs(head_dim / 2);
    Ok((0..len).flat_map(|p| f.iter().map(move |f| p as f32 * f)).collect())
}

/// RoPE angles for an `h × w` grid of patches at T = 1, row-major: each row is `[t | h | w]` with
/// `axes / 2` angles per axis (and `t` is always position 0).
pub fn rope_angles_3d(h: usize, w: usize, axes: [usize; 3]) -> Result<Vec<f32>> {
    if h > ROPE_MAX_AXIS || w > ROPE_MAX_AXIS {
        anyhow::bail!("Kandinsky 5: a {w}x{h} patch grid is past the DiT's {ROPE_MAX_AXIS}-position RoPE table (at most {} px a side)", ROPE_MAX_AXIS * 8 * PATCH);
    }
    let (ft, fh, fw) = (freqs(axes[0] / 2), freqs(axes[1] / 2), freqs(axes[2] / 2));
    let mut out = Vec::with_capacity(h * w * (ft.len() + fh.len() + fw.len()));
    for y in 0..h {
        for x in 0..w {
            out.extend(ft.iter().map(|f| 0.0 * f));
            out.extend(fh.iter().map(|f| y as f32 * f));
            out.extend(fw.iter().map(|f| x as f32 * f));
        }
    }
    Ok(out)
}

/// The cos and sin tables of a RoPE, each `(1, L, 1, head_dim / 2)` in F32.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(angles: Vec<f32>, len: usize, device: &Device) -> Result<Self> {
        let half = angles.len() / len.max(1);
        let a = Tensor::from_vec(angles, (1, len, 1, half), device)?;
        Ok(Self { cos: a.cos()?, sin: a.sin()? })
    }

    /// Rotate adjacent channel pairs of `x` `(B, L, H, D)` (F32): `(x0, x1) → (c·x0 − s·x1, s·x0 + c·x1)`
    /// — the reference's `[[cos, −sin], [sin, cos]]` matrices (trap T4). Every op stays at rank ≤ 4.
    fn rotate(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, h, d) = x.dims4()?;
        let n = b * l * h;
        let pairs = x.reshape((n, d / 2, 2))?;
        let x0 = pairs.narrow(2, 0, 1)?.reshape((b, l, h, d / 2))?;
        let x1 = pairs.narrow(2, 1, 1)?.reshape((b, l, h, d / 2))?;
        let y0 = (x0.broadcast_mul(&self.cos)? - x1.broadcast_mul(&self.sin)?)?;
        let y1 = (x0.broadcast_mul(&self.sin)? + x1.broadcast_mul(&self.cos)?)?;
        Ok(Tensor::cat(&[y0.reshape((n, d / 2, 1))?, y1.reshape((n, d / 2, 1))?], 2)?.reshape((b, l, h, d))?)
    }
}

/// NCHW latents `(B, C, H, W)` → patch tokens `(B, H/2 · W/2, 4C)`, each patch laid out `(ph, pw, C)`
/// — the reference's channels-last patchify (trap T9). Runs wherever `x` lives; call it on the CPU.
pub fn patchify(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if h % PATCH != 0 || w % PATCH != 0 {
        anyhow::bail!("Kandinsky 5: a {w}x{h} latent does not divide into {PATCH}x{PATCH} patches");
    }
    let (gh, gw) = (h / PATCH, w / PATCH);
    Ok(x.reshape(&[b, c, gh, PATCH, gw, PATCH][..])?.permute([0, 2, 4, 3, 5, 1])?.contiguous()?.reshape((b, gh * gw, PATCH * PATCH * c))?)
}

/// The inverse for the OUTPUT layer, whose patches are laid out `(C, ph, pw)`: tokens
/// `(B, gh · gw, 4C)` → NCHW `(B, C, 2·gh, 2·gw)`.
pub fn unpatchify(x: &Tensor, gh: usize, gw: usize) -> Result<Tensor> {
    let (b, n, f) = x.dims3()?;
    if n != gh * gw || f % (PATCH * PATCH) != 0 {
        anyhow::bail!("Kandinsky 5: {n} tokens of {f} features is not a {gw}x{gh} grid of {PATCH}x{PATCH} patches");
    }
    let c = f / (PATCH * PATCH);
    Ok(x.reshape(&[b, gh, gw, c, PATCH, PATCH][..])?.permute([0, 3, 1, 4, 2, 5])?.contiguous()?.reshape((b, c, gh * PATCH, gw * PATCH))?)
}

fn lin(vb: VarBuilder, input: usize, output: usize, bias: bool) -> Result<Linear> {
    let w = vb.get((output, input), "weight")?;
    let b = if bias { Some(vb.get(output, "bias")?) } else { None };
    Ok(Linear::new(w, b))
}

/// `xs` through a linear layer, in `xs`'s dtype: weights stored narrower are widened for this call.
fn wide(l: &Linear, xs: &Tensor) -> Result<Tensor> {
    let dt = xs.dtype();
    if l.weight().dtype() == dt {
        return Ok(l.forward(xs)?);
    }
    let bias = l.bias().map(|b| b.to_dtype(dt)).transpose()?;
    Ok(Linear::new(l.weight().to_dtype(dt)?, bias).forward(xs)?)
}

/// A block's linear layer: the checkpoint's dense weights, or NF4 (`--dit-nf4`, RFC §10.2) — 4-bit
/// codes, two to a byte, and an absmax per 64 values, both on the model's device. A call dequantizes
/// there: the bytes index a 256-row table of code pairs and the blocks are scaled by their absmax, so
/// the dense weight exists for one matmul and its buffer is the pool's to reuse. (Dequantizing on the
/// CPU and uploading was measured first: as fast, but each upload is a fresh buffer and the denoise
/// peaked 7 GB higher.) The bias stays dense.
enum Lin {
    Dense(Linear),
    Nf4 { packed: Tensor, absmax: Tensor, pairs: Tensor, bias: Option<Tensor>, shape: (usize, usize) },
}

impl Lin {
    /// `vb` is on the model's device. `quant`, when given, is the same path on the CPU: the weight is
    /// read through it one layer at a time and quantized, so the dense checkpoint is never resident.
    fn load(vb: VarBuilder, quant: Option<VarBuilder>, input: usize, output: usize, bias: bool) -> Result<Self> {
        use crate::pipelines::nf4_codec::{quantize_nf4_cpu, NF4_BLOCK_SIZE, NF4_CODEBOOK};
        let Some(q) = quant else {
            return Ok(Lin::Dense(lin(vb, input, output, bias)?));
        };
        let device = vb.device();
        let w: Vec<f32> = q.get((output, input), "weight")?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (packed, absmax) = quantize_nf4_cpu(&w)?;
        let blocks = absmax.len();
        debug_assert_eq!(blocks * NF4_BLOCK_SIZE, input * output);
        // Row `b` is the two values a byte `b` packs: low nibble first.
        let pairs: Vec<f32> = (0..256usize).flat_map(|b| [NF4_CODEBOOK[b & 0x0F], NF4_CODEBOOK[b >> 4]]).collect();
        Ok(Lin::Nf4 {
            packed: Tensor::from_vec(packed, input * output / 2, device)?,
            absmax: Tensor::from_vec(absmax, (blocks, 1), device)?,
            pairs: Tensor::from_vec(pairs, (256, 2), device)?,
            bias: if bias { Some(vb.get(output, "bias")?) } else { None },
            shape: (output, input),
        })
    }

    /// `xs` through the layer, in `xs`'s dtype.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Lin::Dense(l) => wide(l, xs),
            Lin::Nf4 { packed, absmax, pairs, bias, shape } => {
                let dt = xs.dtype();
                let blocks = absmax.dim(0)?;
                let w = pairs.index_select(packed, 0)?.reshape((blocks, ()))?.broadcast_mul(absmax)?.reshape(*shape)?.to_dtype(dt)?;
                let bias = bias.as_ref().map(|b| b.to_dtype(dt)).transpose()?;
                Ok(Linear::new(w, bias).forward(xs)?)
            }
        }
    }
}

/// Non-affine LayerNorm over the last dim, in F32.
fn layer_norm(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let x = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(var + LN_EPS)?.sqrt()?)?)
}

/// `Linear → affine LayerNorm` (`Kandinsky5TextEmbeddings`). The norm is done in F32; the result is F32.
struct Embed {
    in_layer: Linear,
    weight: Tensor,
    bias: Tensor,
}

impl Embed {
    fn new(vb: VarBuilder, input: usize, output: usize) -> Result<Self> {
        Ok(Self { in_layer: lin(vb.pp("in_layer"), input, output, true)?, weight: vb.pp("norm").get(output, "weight")?, bias: vb.pp("norm").get(output, "bias")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = layer_norm(&wide(&self.in_layer, x)?)?;
        Ok(n.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.broadcast_add(&self.bias.to_dtype(DType::F32)?)?)
    }
}

/// `SiLU → Linear(time_dim → k · model_dim)`, an F32 island. [`Modulation::params`] takes the already
/// activated time vector and returns the `k` parameters as `(B, 1, model_dim)` each.
struct Modulation {
    out_layer: Linear,
    k: usize,
}

impl Modulation {
    fn new(vb: VarBuilder, time_dim: usize, model_dim: usize, k: usize) -> Result<Self> {
        Ok(Self { out_layer: lin(vb.pp("out_layer"), time_dim, k * model_dim, true)?, k })
    }

    fn params(&self, time_act: &Tensor) -> Result<Vec<Tensor>> {
        let all = wide(&self.out_layer, time_act)?;
        let c = all.dim(D::Minus1)? / self.k;
        (0..self.k).map(|i| Ok(all.narrow(D::Minus1, i * c, c)?.unsqueeze(1)?)).collect()
    }
}

/// `LN(x) · (1 + scale) + shift`, in F32, back in `x`'s dtype (trap T6).
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Ok(layer_norm(x)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?.to_dtype(x.dtype())?)
}

/// `x + gate · out`, in F32, back in `x`'s dtype (trap T6).
fn residual(x: &Tensor, gate: &Tensor, out: &Tensor) -> Result<Tensor> {
    Ok((x.to_dtype(DType::F32)? + out.to_dtype(DType::F32)?.broadcast_mul(gate)?)?.to_dtype(x.dtype())?)
}

/// Plain scaled-dot-product attention over `(B·heads, L, D)`, no mask, a few heads at a time.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (bh, lq, d) = q.dims3()?;
    let budget = if crate::pipelines::kandinsky::low_memory() { ATTN_CHUNK_BYTES_LOW } else { ATTN_CHUNK_BYTES };
    let per = (budget / (lq * k.dim(1)? * q.dtype().size_in_bytes()).max(1)).clamp(1, bh);
    let scale = 1.0 / (d as f64).sqrt();
    let kt = k.transpose(1, 2)?.contiguous()?;
    let mut out = Vec::new();
    let mut i = 0;
    while i < bh {
        let n = per.min(bh - i);
        let scores = (q.narrow(0, i, n)?.matmul(&kt.narrow(0, i, n)?)? * scale)?;
        out.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.narrow(0, i, n)?)?);
        i += n;
    }
    Ok(if out.len() == 1 { out.remove(0) } else { Tensor::cat(&out, 0)? })
}

struct Attention {
    to_query: Lin,
    to_key: Lin,
    to_value: Lin,
    out_layer: Lin,
    query_norm: Tensor,
    key_norm: Tensor,
    heads: usize,
    rope_round: bool,
}

impl Attention {
    fn new(vb: VarBuilder, quant: Option<VarBuilder>, cfg: &Config, rope_round: bool) -> Result<Self> {
        let (c, hd) = (cfg.model_dim, cfg.head_dim());
        let load = |name: &str| Lin::load(vb.pp(name), quant.as_ref().map(|q| q.pp(name)), c, c, true);
        Ok(Self {
            to_query: load("to_query")?,
            to_key: load("to_key")?,
            to_value: load("to_value")?,
            out_layer: load("out_layer")?,
            query_norm: vb.pp("query_norm").get(hd, "weight")?,
            key_norm: vb.pp("key_norm").get(hd, "weight")?,
            heads: cfg.heads(),
            rope_round,
        })
    }

    /// Q or K `(B, L, C)`: per-head RMSNorm in F32, then the rotation if there is one (rounded to BF16
    /// as the reference does), handed back as `(B·heads, L, head_dim)` in the input's dtype.
    fn prepare(&self, x: &Tensor, norm: &Tensor, rope: Option<&Rope>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let hd = c / self.heads;
        let y = x.to_dtype(DType::F32)?.reshape((b, l * self.heads, hd))?;
        let rms = (y.sqr()?.mean_keepdim(D::Minus1)? + RMS_EPS)?.sqrt()?;
        let mut y = y.broadcast_div(&rms)?.broadcast_mul(&norm.to_dtype(DType::F32)?)?.reshape((b, l, self.heads, hd))?;
        if let Some(rope) = rope {
            y = rope.rotate(&y)?;
            if self.rope_round {
                y = y.to_dtype(DType::BF16)?;
            }
        }
        Ok(y.to_dtype(x.dtype())?.transpose(1, 2)?.contiguous()?.reshape((b * self.heads, l, hd))?)
    }

    /// Self-attention when `context` is `None`, cross-attention to it otherwise (which has no RoPE —
    /// trap T10).
    fn forward(&self, x: &Tensor, context: Option<&Tensor>, rope: Option<&Rope>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let src = context.unwrap_or(x);
        let (lk, hd) = (src.dim(1)?, c / self.heads);
        let q = self.prepare(&self.to_query.forward(x)?, &self.query_norm, rope)?;
        let k = self.prepare(&self.to_key.forward(src)?, &self.key_norm, rope)?;
        let v = self.to_value.forward(src)?.reshape((b, lk, self.heads, hd))?.transpose(1, 2)?.contiguous()?.reshape((b * self.heads, lk, hd))?;
        let out = sdpa(&q, &k, &v)?.reshape((b, self.heads, l, hd))?.transpose(1, 2)?.contiguous()?.reshape((b, l, c))?;
        self.out_layer.forward(&out)
    }
}

/// `Linear (no bias) → exact GELU → Linear (no bias)`.
struct FeedForward {
    in_layer: Lin,
    out_layer: Lin,
}

impl FeedForward {
    fn new(vb: VarBuilder, quant: Option<VarBuilder>, cfg: &Config) -> Result<Self> {
        let q = |name: &str| quant.as_ref().map(|q| q.pp(name));
        Ok(Self { in_layer: Lin::load(vb.pp("in_layer"), q("in_layer"), cfg.model_dim, cfg.ff_dim, false)?, out_layer: Lin::load(vb.pp("out_layer"), q("out_layer"), cfg.ff_dim, cfg.model_dim, false)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out_layer.forward(&self.in_layer.forward(x)?.gelu_erf()?)
    }
}

/// `Kandinsky5TransformerEncoderBlock`: the text stream. Time-modulated, so it runs every step (T8).
struct TextBlock {
    modulation: Modulation,
    attention: Attention,
    feed_forward: FeedForward,
}

impl TextBlock {
    fn forward(&self, x: &Tensor, time_act: &Tensor, rope: &Rope) -> Result<Tensor> {
        let p = self.modulation.params(time_act)?;
        let out = self.attention.forward(&modulate(x, &p[0], &p[1])?, None, Some(rope))?;
        let x = residual(x, &p[2], &out)?;
        let out = self.feed_forward.forward(&modulate(&x, &p[3], &p[4])?)?;
        residual(&x, &p[5], &out)
    }
}

/// `Kandinsky5TransformerDecoderBlock`: self-attention, cross-attention to the text, feed-forward.
struct VisualBlock {
    modulation: Modulation,
    self_attention: Attention,
    cross_attention: Attention,
    feed_forward: FeedForward,
}

impl VisualBlock {
    fn forward(&self, x: &Tensor, text: &Tensor, time_act: &Tensor, rope: &Rope) -> Result<Tensor> {
        let p = self.modulation.params(time_act)?;
        let out = self.self_attention.forward(&modulate(x, &p[0], &p[1])?, None, Some(rope))?;
        let x = residual(x, &p[2], &out)?;
        let out = self.cross_attention.forward(&modulate(&x, &p[3], &p[4])?, Some(text), None)?;
        let x = residual(&x, &p[5], &out)?;
        let out = self.feed_forward.forward(&modulate(&x, &p[6], &p[7])?)?;
        residual(&x, &p[8], &out)
    }
}

/// Intermediate tensors a parity run asks for: the text stream after its blocks (`text_stream`) and
/// the visual stream after each block listed in `blocks` (`visual_block_{i}`), all F32 on the CPU.
#[derive(Default)]
pub struct Taps {
    pub blocks: Vec<usize>,
    pub seen: HashMap<String, Tensor>,
}

impl Taps {
    fn record(&mut self, name: String, t: &Tensor) -> Result<()> {
        self.seen.insert(name, t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
        Ok(())
    }
}

pub struct Dit {
    pub cfg: Config,
    time_in: Linear,
    time_out: Linear,
    text_embeddings: Embed,
    pooled_text_embeddings: Embed,
    visual_in: Linear,
    text_blocks: Vec<TextBlock>,
    visual_blocks: Vec<VisualBlock>,
    out_modulation: Modulation,
    out_layer: Linear,
    device: Device,
    compute: DType,
}

impl Dit {
    /// Build from a VarBuilder over the checkpoint (any stored dtype); activations run in `compute`.
    pub fn new(cfg: Config, vb: VarBuilder, compute: DType) -> Result<Self> {
        Self::build(cfg, vb, None, compute)
    }

    /// As [`Self::new`], with the blocks' attention and feed-forward weights in NF4 — 95 % of the model.
    /// `quant` is a VarBuilder over the same checkpoint on the CPU, which those weights are read
    /// through. The embeddings, the modulations (the reference's F32 islands) and the output layer stay
    /// dense.
    pub fn new_nf4(cfg: Config, vb: VarBuilder, quant: VarBuilder, compute: DType) -> Result<Self> {
        Self::build(cfg, vb, Some(quant), compute)
    }

    fn build(cfg: Config, vb: VarBuilder, quant: Option<VarBuilder>, compute: DType) -> Result<Self> {
        let round = std::env::var("PLAKAT_K5_ROPE_ROUND").ok().as_deref() != Some("0");
        let (c, td) = (cfg.model_dim, cfg.time_dim);
        let text_blocks = (0..cfg.num_text_blocks)
            .map(|i| {
                let vb = vb.pp("text_transformer_blocks").pp(i);
                let q = |name: &str| quant.as_ref().map(|q| q.pp("text_transformer_blocks").pp(i).pp(name));
                Ok(TextBlock { modulation: Modulation::new(vb.pp("text_modulation"), td, c, 6)?, attention: Attention::new(vb.pp("self_attention"), q("self_attention"), &cfg, round)?, feed_forward: FeedForward::new(vb.pp("feed_forward"), q("feed_forward"), &cfg)? })
            })
            .collect::<Result<Vec<_>>>()?;
        let visual_blocks = (0..cfg.num_visual_blocks)
            .map(|i| {
                let vb = vb.pp("visual_transformer_blocks").pp(i);
                let q = |name: &str| quant.as_ref().map(|q| q.pp("visual_transformer_blocks").pp(i).pp(name));
                Ok(VisualBlock {
                    modulation: Modulation::new(vb.pp("visual_modulation"), td, c, 9)?,
                    self_attention: Attention::new(vb.pp("self_attention"), q("self_attention"), &cfg, round)?,
                    cross_attention: Attention::new(vb.pp("cross_attention"), q("cross_attention"), &cfg, round)?,
                    feed_forward: FeedForward::new(vb.pp("feed_forward"), q("feed_forward"), &cfg)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_in: lin(vb.pp("time_embeddings").pp("in_layer"), c, td, true)?,
            time_out: lin(vb.pp("time_embeddings").pp("out_layer"), td, td, true)?,
            text_embeddings: Embed::new(vb.pp("text_embeddings"), cfg.in_text_dim, c)?,
            pooled_text_embeddings: Embed::new(vb.pp("pooled_text_embeddings"), cfg.in_text_dim2, td)?,
            visual_in: lin(vb.pp("visual_embeddings").pp("in_layer"), PATCH * PATCH * cfg.in_visual_dim, c, true)?,
            text_blocks,
            visual_blocks,
            out_modulation: Modulation::new(vb.pp("out_layer").pp("modulation"), td, c, 2)?,
            out_layer: lin(vb.pp("out_layer").pp("out_layer"), c, PATCH * PATCH * cfg.out_visual_dim, true)?,
            device: vb.device().clone(),
            compute,
            cfg,
        })
    }

    /// The compute dtype for a device: F32, unless `PLAKAT_K5_DIT_COMPUTE=bf16` on a GPU. F32 is the
    /// default there too because it is nearly free and far closer: at 1024² on Metal a forward is 8.1 s
    /// against 7.4 s, and the velocity's cosine to the F32 reference is 1.000000 against 0.9995–0.9999
    /// (P2 parity). The weights are BF16 either way, so the resident size is the same.
    pub fn compute_dtype(device: &Device) -> DType {
        let bf16_asked = std::env::var("PLAKAT_K5_DIT_COMPUTE").ok().as_deref() == Some("bf16");
        if !device.is_cpu() && bf16_asked { DType::BF16 } else { DType::F32 }
    }

    /// Load `transformer/` from a Kandinsky 5 diffusers repo. The weights rest in BF16 (what the
    /// checkpoint stores, ≈12 GB) on every device.
    pub async fn load(repo: &str, device: &Device) -> Result<Self> {
        Self::load_with(repo, device, false).await
    }

    /// As [`Self::load`]; with `nf4` the block weights are quantized as they are read (≈3.7 GB resident).
    pub async fn load_with(repo: &str, device: &Device, nf4: bool) -> Result<Self> {
        let cfg_path = crate::hf::download::get_file(repo, "transformer/config.json").await.context("Kandinsky 5 transformer/config.json")?;
        let cfg = Config::from_json(&std::fs::read_to_string(&cfg_path)?)?;
        let weights = crate::hf::download::get_file(repo, "transformer/diffusion_pytorch_model.safetensors").await.context("Kandinsky 5 transformer weights")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::BF16, device)? };
        let quant = if nf4 { Some(unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::BF16, &Device::Cpu)? }) } else { None };
        Self::build(cfg, vb, quant, Self::compute_dtype(device)).context("building the Kandinsky 5 DiT")
    }

    pub fn compute(&self) -> DType {
        self.compute
    }

    /// `time_embeddings` alone: the sinusoidal features through its two layers, `(1, time_dim)` in F32.
    pub fn time_embed(&self, t: f64) -> Result<Tensor> {
        let feats = Tensor::from_vec(time_features(t as f32, self.cfg.model_dim), (1, self.cfg.model_dim), &self.device)?;
        wide(&self.time_out, &wide(&self.time_in, &feats)?.silu()?)
    }

    /// One forward: NCHW latents `(1, 16, H/8, W/8)`, the Qwen hidden states `(1, L, 3584)`, the CLIP
    /// pooled vector `(1, 768)` and the raw timestep `sigma · 1000` → the velocity, NCHW in F32.
    pub fn forward(&self, latents: &Tensor, text: &Tensor, pooled: &Tensor, t: f64) -> Result<Tensor> {
        self.forward_tapped(latents, text, pooled, t, None)
    }

    pub fn forward_tapped(&self, latents: &Tensor, text: &Tensor, pooled: &Tensor, t: f64, mut taps: Option<&mut Taps>) -> Result<Tensor> {
        let (b, _c, h, w) = latents.dims4()?;
        if b != 1 || text.dim(0)? != 1 {
            anyhow::bail!("the Kandinsky 5 DiT takes one image and one prompt per forward (the two CFG branches differ in length)");
        }
        let (gh, gw) = (h / PATCH, w / PATCH);
        let tokens = patchify(&latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?.to_device(&self.device)?.to_dtype(self.compute)?;
        let mut visual = wide(&self.visual_in, &tokens)?;
        let mut text = self.text_embeddings.forward(&text.to_device(&self.device)?.to_dtype(self.compute)?)?.to_dtype(self.compute)?;
        // The one global conditioning vector: time + pooled text. Every modulation applies the same SiLU.
        let time = (self.time_embed(t)? + self.pooled_text_embeddings.forward(&pooled.to_device(&self.device)?.to_dtype(DType::F32)?)?)?;
        let time_act = time.silu()?;

        let len = text.dim(1)?;
        let text_rope = Rope::new(rope_angles_1d(len, self.cfg.head_dim())?, len, &self.device)?;
        for block in &self.text_blocks {
            text = block.forward(&text, &time_act, &text_rope)?;
        }
        if let Some(taps) = taps.as_deref_mut() {
            taps.record("text_stream".into(), &text)?;
        }
        let visual_rope = Rope::new(rope_angles_3d(gh, gw, self.cfg.axes_dims)?, gh * gw, &self.device)?;
        for (i, block) in self.visual_blocks.iter().enumerate() {
            visual = block.forward(&visual, &text, &time_act, &visual_rope)?;
            if let Some(taps) = taps.as_deref_mut() {
                if taps.blocks.contains(&i) {
                    taps.record(format!("visual_block_{i}"), &visual)?;
                }
            }
        }
        let p = self.out_modulation.params(&time_act)?;
        let out = wide(&self.out_layer, &modulate(&visual, &p[0], &p[1])?)?;
        Ok(unpatchify(&out.to_dtype(DType::F32)?.to_device(&Device::Cpu)?, gh, gw)?.to_device(&self.device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let (a, b) = (v(a), v(b));
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let n = |x: &[f32]| x.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (n(&a) * n(&b)).max(1e-30)) as f32
    }

    /// max |a − b| relative to the reference's largest value.
    fn rel(a: &Tensor, b: &Tensor) -> f32 {
        let (a, b) = (v(a), v(b));
        let peak = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / peak.max(1e-30)
    }

    #[test]
    fn the_config_is_read_and_checked() {
        let json = r#"{"_class_name":"Kandinsky5Transformer3DModel","in_visual_dim":16,"out_visual_dim":16,"time_dim":512,"patch_size":[1,2,2],
            "model_dim":2560,"ff_dim":10240,"num_text_blocks":2,"num_visual_blocks":50,"axes_dims":[32,48,48],"visual_cond":false,
            "in_text_dim":3584,"in_text_dim2":768,"attention_type":"regular"}"#;
        let c = Config::from_json(json).unwrap();
        assert_eq!(c, Config::lite());
        assert_eq!((c.head_dim(), c.heads()), (128, 20));
        assert!(Config::from_json(&json.replace("\"visual_cond\":false", "\"visual_cond\":true")).is_err());
        assert!(Config::from_json(&json.replace("regular", "nabla")).is_err());
        assert!(Config::from_json(&json.replace("[1,2,2]", "[1,1,1]")).is_err());
    }

    #[test]
    fn time_features_are_cosine_first() {
        // Trap T5. At t = 0 every cos is 1 and every sin 0; the first frequency is exactly 1.
        let z = time_features(0.0, 8);
        assert_eq!(z, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let f = time_features(2.0, 8);
        assert!((f[0] - 2f32.cos()).abs() < 1e-6 && (f[4] - 2f32.sin()).abs() < 1e-6);
        // freqs = 10000^(−i/4): the second is 0.1.
        assert!((f[1] - 0.2f32.cos()).abs() < 1e-6 && (f[5] - 0.2f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_adjacent_pairs() {
        // Trap T4: a 4-dim head is two (even, odd) pairs, each turned by its own angle — not the
        // half-split form, which would pair channel 0 with channel 2.
        let (a, b) = (0.3f32, 1.1f32);
        let rope = Rope::new(vec![a, b], 1, &Device::Cpu).unwrap();
        let x = Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu).unwrap().reshape((1, 1, 1, 4)).unwrap();
        let got = v(&rope.rotate(&x).unwrap());
        let want = [a.cos() * 1. - a.sin() * 2., a.sin() * 1. + a.cos() * 2., b.cos() * 3. - b.sin() * 4., b.sin() * 3. + b.cos() * 4.];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} vs {want:?}");
        }
        // A rotation keeps each pair's length.
        assert!(((got[0].powi(2) + got[1].powi(2)) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn rope_angles_follow_the_axes() {
        let one = rope_angles_1d(3, 8).unwrap();
        assert_eq!(one.len(), 3 * 4);
        assert_eq!(&one[..4], &[0.0; 4]);
        assert!((one[4] - 1.0).abs() < 1e-6 && (one[8] - 2.0).abs() < 1e-6); // position · 1
        assert!((one[5] - 0.1).abs() < 1e-6); // 10000^(−1/4)
        assert!(rope_angles_1d(1025, 128).is_err());

        // A 2×3 grid with axes [4, 4, 8]: rows are [t(2) | h(2) | w(4)], row-major, t always 0.
        let g = rope_angles_3d(2, 3, [4, 4, 8]).unwrap();
        assert_eq!(g.len(), 6 * 8);
        let row = |y: usize, x: usize| &g[(y * 3 + x) * 8..(y * 3 + x + 1) * 8];
        assert_eq!(row(0, 0), &[0.0; 8]);
        assert_eq!(&row(1, 2)[..2], &[0.0, 0.0]);
        assert!((row(1, 2)[2] - 1.0).abs() < 1e-6, "h position 1");
        assert!((row(1, 2)[4] - 2.0).abs() < 1e-6, "w position 2");
        assert!((row(0, 2)[2]).abs() < 1e-6 && (row(1, 0)[4]).abs() < 1e-6);
        // The largest bucket fits; one past the table does not.
        assert!(rope_angles_3d(88, 40, [32, 48, 48]).is_ok());
        assert!(rope_angles_3d(129, 8, [32, 48, 48]).is_err());
    }

    #[test]
    fn patches_go_in_channels_last_and_come_out_channels_first() {
        // Trap T9. x[c, y, x] = 100c + 10y + x on a 2-channel 4×4 latent.
        let (c, h, w) = (2usize, 4usize, 4usize);
        let data: Vec<f32> = (0..c).flat_map(|ci| (0..h).flat_map(move |y| (0..w).map(move |x| (100 * ci + 10 * y + x) as f32))).collect();
        let x = Tensor::from_vec(data, (1, c, h, w), &Device::Cpu).unwrap();
        let p = patchify(&x).unwrap();
        assert_eq!(p.dims(), [1, 4, 8]);
        let p = v(&p);
        // Token 3 is the patch at grid (1, 1) — pixels y ∈ {2, 3}, x ∈ {2, 3} — laid out (ph, pw, C).
        assert_eq!(&p[3 * 8..4 * 8], &[22., 122., 23., 123., 32., 132., 33., 133.]);

        // The output layer's patches are (C, ph, pw): build tokens that way and get the image back.
        let tokens: Vec<f32> = (0..2).flat_map(|gy| (0..2).flat_map(move |gx| (0..c).flat_map(move |ci| (0..2).flat_map(move |py| (0..2).map(move |px| (100 * ci + 10 * (2 * gy + py) + 2 * gx + px) as f32))))).collect();
        let back = unpatchify(&Tensor::from_vec(tokens, (1, 4, 8), &Device::Cpu).unwrap(), 2, 2).unwrap();
        assert_eq!(back.dims(), [1, c, h, w]);
        assert_eq!(v(&back), v(&x));
        assert!(patchify(&Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap()).is_err());
    }

    fn tiny() -> Config {
        Config { in_visual_dim: 4, out_visual_dim: 4, time_dim: 32, model_dim: 64, ff_dim: 128, num_text_blocks: 2, num_visual_blocks: 3, axes_dims: [4, 6, 6], in_text_dim: 24, in_text_dim2: 12 }
    }

    /// Random BF16-representable weights for a tiny DiT. The reference zero-initialises its modulations;
    /// here they are random too, so the modulate and gate paths are exercised.
    fn tiny_weights(cfg: &Config) -> HashMap<String, Tensor> {
        let vm = candle_nn::VarMap::new();
        Dit::new(cfg.clone(), VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), DType::F32).unwrap();
        let mut seed = 1u64;
        vm.data()
            .lock()
            .unwrap()
            .iter()
            .map(|(k, var)| {
                seed += 1;
                let n = var.elem_count();
                let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
                let data: Vec<f32> = (0..n)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.25
                    })
                    .collect();
                let t = Tensor::from_vec(data, var.shape(), &Device::Cpu).unwrap();
                // Norm weights sit near one, as trained ones do.
                let t = if k.ends_with("norm.weight") { (t + 1.0).unwrap() } else { t };
                (k.clone(), t.to_dtype(DType::BF16).unwrap().to_dtype(DType::F32).unwrap())
            })
            .collect()
    }

    fn tiny_inputs(cfg: &Config) -> (Tensor, Tensor, Tensor) {
        let f = |n: usize, k: f32| (0..n).map(|i| (i as f32 * k).sin()).collect::<Vec<f32>>();
        let lat = Tensor::from_vec(f(4 * 8 * 12, 0.37), (1, 4, 8, 12), &Device::Cpu).unwrap();
        let text = Tensor::from_vec(f(5 * cfg.in_text_dim, 0.11), (1, 5, cfg.in_text_dim), &Device::Cpu).unwrap();
        let pooled = Tensor::from_vec(f(cfg.in_text_dim2, 0.53), (1, cfg.in_text_dim2), &Device::Cpu).unwrap();
        (lat, text, pooled)
    }

    #[test]
    fn a_tiny_dit_runs_and_widening_is_exact() {
        let cfg = tiny();
        let w = tiny_weights(&cfg);
        let (lat, text, pooled) = tiny_inputs(&cfg);
        let f32_stored = Dit::new(cfg.clone(), VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu), DType::F32).unwrap();
        let mut taps = Taps { blocks: vec![0, 2], ..Default::default() };
        let want = f32_stored.forward_tapped(&lat, &text, &pooled, 500.0, Some(&mut taps)).unwrap();
        assert_eq!(want.dims(), lat.dims());
        assert_eq!(want.dtype(), DType::F32);
        assert!(v(&want).iter().all(|x| x.is_finite()));
        assert_eq!(taps.seen["text_stream"].dims(), [1, 5, cfg.model_dim]);
        assert_eq!(taps.seen["visual_block_2"].dims(), [1, 4 * 6, cfg.model_dim]);
        assert!(!taps.seen.contains_key("visual_block_1"));

        // BF16 at rest, F32 compute: the same numbers, since widening BF16 is exact.
        let bf16: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), t.to_dtype(DType::BF16).unwrap())).collect();
        let stored_bf16 = Dit::new(cfg.clone(), VarBuilder::from_tensors(bf16, DType::BF16, &Device::Cpu), DType::F32).unwrap();
        let got = stored_bf16.forward(&lat, &text, &pooled, 500.0).unwrap();
        assert!(rel(&got, &want) < 1e-6, "relative max-abs {}", rel(&got, &want));

        // The text stream is time-modulated (trap T8) and the prompt matters: both change the velocity.
        let other_t = f32_stored.forward(&lat, &text, &pooled, 250.0).unwrap();
        assert!(rel(&other_t, &want) > 1e-3);
        let other_text = f32_stored.forward(&lat, &(&text * 0.5).unwrap(), &pooled, 500.0).unwrap();
        assert!(rel(&other_text, &want) > 1e-3);
        // Two images or two prompts in one forward are refused.
        assert!(f32_stored.forward(&Tensor::cat(&[&lat, &lat], 0).unwrap(), &text, &pooled, 500.0).is_err());
    }

    /// The tiny DiT on the GPU, in both compute dtypes, against the CPU — the rank > 4 risk (RFC §14)
    /// without the checkpoint: `cargo test --release --features metal --lib a_tiny_dit_on_the_gpu -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn a_tiny_dit_on_the_gpu_tracks_the_cpu() {
        let gpu = crate::device::select("auto").unwrap();
        let cfg = tiny();
        let w = tiny_weights(&cfg);
        let (lat, text, pooled) = tiny_inputs(&cfg);
        let cpu = Dit::new(cfg.clone(), VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu), DType::F32).unwrap();
        let want = cpu.forward(&lat, &text, &pooled, 500.0).unwrap();
        for compute in [DType::F32, DType::BF16] {
            let on_gpu: HashMap<String, Tensor> = w.iter().map(|(k, t)| (k.clone(), t.to_device(&gpu).unwrap().to_dtype(DType::BF16).unwrap())).collect();
            let dit = Dit::new(cfg.clone(), VarBuilder::from_tensors(on_gpu, DType::BF16, &gpu), compute).unwrap();
            let got = dit.forward(&lat, &text, &pooled, 500.0).unwrap();
            let (c, r) = (cosine(&got, &want), rel(&got, &want));
            println!("compute {compute:?}: cosine {c:.6}, relative max-abs {r:.2e}");
            assert!(if compute == DType::F32 { r < 1e-4 } else { c > 0.999 }, "compute {compute:?}: cosine {c}, relative max-abs {r}");
        }
    }

    /// P2's parity gate (RFC §12, stages 3–6) against a `tools/kandinsky_dump.py --stage p2` dump in
    /// `PLAKAT_PARITY_DIR` (which also holds P1's `p1.safetensors`, the embeddings both sides use).
    /// Loads the 12 GB DiT — run it alone:
    /// `PLAKAT_PARITY_DIR=<dir> cargo test --release --features metal --lib kandinsky_parity_p2 -- --ignored --nocapture`
    /// (`PLAKAT_PARITY_DEVICE=cpu` for the CPU; `PLAKAT_K5_DIT_COMPUTE=bf16` for BF16 compute on the GPU).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore]
    async fn kandinsky_parity_p2() {
        let dir = std::path::PathBuf::from(std::env::var("PLAKAT_PARITY_DIR").expect("set PLAKAT_PARITY_DIR to a tools/kandinsky_dump.py output directory"));
        let meta: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("meta_p2.json")).unwrap()).unwrap();
        let device = crate::device::select(&std::env::var("PLAKAT_PARITY_DEVICE").unwrap_or_else(|_| "auto".into())).unwrap();
        let p1 = candle_core::safetensors::load(dir.join("p1.safetensors"), &Device::Cpu).unwrap();
        let p2 = candle_core::safetensors::load(dir.join("p2.safetensors"), &Device::Cpu).unwrap();
        let t0 = std::time::Instant::now();
        let dit = Dit::load(meta["repo"].as_str().unwrap(), &device).await.unwrap();
        let exact = dit.compute() == DType::F32;
        println!("DiT loaded in {:.1}s on {device:?}, compute {:?}", t0.elapsed().as_secs_f64(), dit.compute());

        let failed = std::cell::RefCell::new(Vec::new());
        // F32 compute: `Exact` stages are held to max-abs relative to the reference's peak. Past the first
        // block that bound is not reachable against a reference run on another backend — the BF16
        // rounding of the rotated Q/K (RFC §12.3) turns 1e-6 differences into 4e-3 ones, and fifty blocks
        // carry them — so the deep stages are held to a cosine instead. BF16 compute: a cosine, always.
        enum F32Bar {
            Exact(f32),
            Cosine(f32),
        }
        let check = |name: &str, got: &Tensor, f32_bar: F32Bar, bf16_cos: f32| {
            let want = &p2[name];
            assert_eq!(got.dims(), want.dims(), "{name}: shape");
            let (c, r) = (cosine(got, want), rel(got, want));
            println!("{name:<20} cosine {c:.6} · relative max-abs {r:.2e}");
            let ok = match (exact, f32_bar) {
                (true, F32Bar::Exact(bound)) => r <= bound,
                (true, F32Bar::Cosine(bound)) => c >= bound,
                (false, _) => c >= bf16_cos,
            };
            if !ok {
                failed.borrow_mut().push(format!("{name}: cosine {c}, relative max-abs {r}"));
            }
        };

        // Stage 3: the time embedding (traps T2, T5).
        for t in [1000.0, 500.0, 1.0] {
            check(&format!("time_embed_{}", t as u32), &dit.time_embed(t).unwrap(), F32Bar::Exact(1e-4), 0.9999);
        }
        // Stages 4 and 5: the text stream and the visual stream at t = 500, on the dumped noise.
        let text = |tag: &str| (p1[&format!("qwen_hidden_{tag}")].clone(), p1[&format!("clip_pooled_{tag}")].clone());
        let (pos, neg) = (text("pos"), text("neg"));
        let blocks: Vec<usize> = meta["tap_blocks"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap() as usize).collect();
        let mut taps = Taps { blocks: blocks.clone(), ..Default::default() };
        let t1 = std::time::Instant::now();
        let velocity = dit.forward_tapped(&p2["noise"], &pos.0, &pos.1, 500.0, Some(&mut taps)).unwrap();
        println!("one forward at {:?}: {:.1}s", p2["noise"].dims(), t1.elapsed().as_secs_f64());
        check("text_stream_500", &taps.seen["text_stream"], F32Bar::Exact(1e-3), 0.998);
        for b in blocks {
            let bar = if b == 0 { F32Bar::Exact(1e-3) } else { F32Bar::Cosine(0.9999) };
            check(&format!("visual_block_{b}_500"), &taps.seen[&format!("visual_block_{b}")], bar, 0.998);
        }
        check("velocity_500", &velocity, F32Bar::Cosine(0.9999), 0.995);
        // Stage 6: both CFG branches at the tapped steps, from the reference's own latents (T8, T10).
        for step in meta["tap_steps"].as_array().unwrap().iter().map(|s| s.as_u64().unwrap()) {
            let x = &p2[&format!("x_step{step}")];
            let t = p2[&format!("t_step{step}")].to_vec1::<f32>().unwrap()[0] as f64;
            check(&format!("vcond_step{step}"), &dit.forward(x, &pos.0, &pos.1, t).unwrap(), F32Bar::Cosine(0.9999), 0.995);
            check(&format!("vuncond_step{step}"), &dit.forward(x, &neg.0, &neg.1, t).unwrap(), F32Bar::Cosine(0.9999), 0.995);
        }
        let failed = failed.into_inner();
        assert!(failed.is_empty(), "parity gate failed: {failed:#?}");
    }
}
