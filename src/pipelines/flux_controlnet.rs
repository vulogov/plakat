//! Flux ControlNet — v0.12 phase 2b.
//!
//! Mirrors diffusers' `FluxControlNetModel`: a partial Flux transformer
//! (~5 DoubleStream blocks) with two extra pieces of plumbing —
//!
//!   * `controlnet_x_embedder` — a Linear that ingests the conditioning
//!     image (VAE-encoded + packed to Flux's 64-d token shape) and adds
//!     it to the main hidden state at the input of the network.
//!   * `controlnet_blocks` — per-block zero-conv Linear "residual heads"
//!     that project each DoubleStream block's output to the residual
//!     tensor the main Flux's `forward_with_residuals` adds onto its
//!     own DoubleStream pass.
//!
//! ## State-dict naming
//!
//! Stock Flux ships in BFL's naming convention (fused
//! `img_attn.qkv.weight` etc.). Community Flux ControlNets — including
//! every InstantX / Shakker-Labs / XLabs-AI variant — ship in
//! **diffusers** naming (separate `attn.to_q.weight`, `to_k.weight`,
//! `to_v.weight`). The two formats are structurally identical apart
//! from QKV fusion + a half-dozen renames.
//!
//! [`remap_diffusers_to_bfl`] converts a diffusers state_dict into the
//! BFL-shaped HashMap that plakat's vendored DoubleStreamBlock expects.
//! Then `VarBuilder::from_tensors` builds the model directly — no temp
//! safetensors file needed.
//!
//! ## What this commit doesn't support
//!
//! * `num_single_layers > 0` — most community ControlNets ship with 0,
//!   but XLabs's v3 line has SingleStream blocks. Code path is stubbed
//!   with a clear bail.
//! * Union ControlNets that take a `mode` integer index — needs an
//!   extra `controlnet_mode_embedder` and a conditioning-aware encoder
//!   hidden state. Tracked for a follow-up.
//! * Multi-Flux-ControlNet — pipeline accepts at most one
//!   `--control-spec` when `--model` is Flux.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use std::collections::HashMap;

use crate::pipelines::flux_inner::{
    self as fi, DoubleStreamBlock, EmbedNd, MlpEmbedder,
};

/// FluxControlNet shape config. Most community models trained against
/// `FLUX.1-dev` use these defaults; the only common variation is
/// `num_layers` (5 in InstantX, 6 in Shakker-Labs).
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of DoubleStream ControlNet blocks. Diffusers default 5.
    pub num_layers: usize,
    /// Whether the ControlNet has a guidance-aware time embedding.
    /// True for FLUX.1-dev derivatives, false for FLUX.1-schnell ones.
    pub guidance_embed: bool,
}

impl Config {
    pub fn instantx_canny() -> Self {
        Self {
            num_layers: 5,
            guidance_embed: true,
        }
    }
}

/// The ControlNet. Built from a diffusers-format safetensors via
/// [`FluxControlNet::load_from_hf`], or directly from a remapped
/// [`VarBuilder`] via [`FluxControlNet::new`].
#[derive(Debug)]
pub struct FluxControlNet {
    /// img_in equivalent — projects the 64-d packed-latent tokens to
    /// the 3072-d hidden state.
    x_embedder: Linear,
    /// Extra ingestion of the conditioning image (VAE-encoded + packed
    /// to the same 64-d token shape). Added to `x_embedder`'s output
    /// at the start of forward.
    controlnet_x_embedder: Linear,
    /// txt_in equivalent — projects 4096-d T5 tokens to 3072-d.
    txt_in: Linear,
    /// time / vector / guidance embedders — re-used from the main Flux
    /// pipeline structurally. ControlNets ship their own (trained
    /// weights identical to base Flux at init).
    time_in: MlpEmbedder,
    vector_in: MlpEmbedder,
    guidance_in: Option<MlpEmbedder>,
    pe_embedder: EmbedNd,
    double_blocks: Vec<DoubleStreamBlock>,
    /// Zero-conv Linear heads — one per `double_blocks` entry. Each
    /// projects a block's output to the residual tensor the main Flux
    /// adds onto its own DoubleStream pass.
    controlnet_blocks: Vec<Linear>,
}

impl FluxControlNet {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // We force the SAME hidden/attn/etc. shapes as base Flux.1-dev
        // since every published Flux ControlNet matches those.
        let flux_cfg = fi::Config::dev();
        let h = flux_cfg.hidden_size;

        let x_embedder = candle_nn::linear(flux_cfg.in_channels, h, vb.pp("img_in"))?;
        let controlnet_x_embedder = candle_nn::linear(
            flux_cfg.in_channels,
            h,
            vb.pp("controlnet_x_embedder"),
        )?;
        let txt_in = candle_nn::linear(flux_cfg.context_in_dim, h, vb.pp("txt_in"))?;
        let time_in = MlpEmbedder::new(256, h, vb.pp("time_in"))?;
        let vector_in = MlpEmbedder::new(flux_cfg.vec_in_dim, h, vb.pp("vector_in"))?;
        let guidance_in = if cfg.guidance_embed {
            Some(MlpEmbedder::new(256, h, vb.pp("guidance_in"))?)
        } else {
            None
        };

        let mut double_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_d = vb.pp("double_blocks");
        for i in 0..cfg.num_layers {
            double_blocks.push(DoubleStreamBlock::new(&flux_cfg, vb_d.pp(i))?);
        }

        let mut controlnet_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_cb = vb.pp("controlnet_blocks");
        for i in 0..cfg.num_layers {
            // Zero-init Linear at training; at inference we just load
            // whatever the trained weight is.
            controlnet_blocks.push(candle_nn::linear(h, h, vb_cb.pp(i))?);
        }

        let pe_dim = h / flux_cfg.num_heads;
        let pe_embedder = EmbedNd::new(pe_dim, flux_cfg.theta, flux_cfg.axes_dim.clone());

        Ok(Self {
            x_embedder,
            controlnet_x_embedder,
            txt_in,
            time_in,
            vector_in,
            guidance_in,
            pe_embedder,
            double_blocks,
            controlnet_blocks,
        })
    }

    /// Run the ControlNet. Returns the per-block DoubleStream
    /// residuals the main Flux's `forward_with_residuals` consumes.
    /// `conditioning` — the conditioning image packed to the same
    /// `(B, img_seq_len, 64)` shape as the main pipeline's `img`.
    /// `conditioning_scale` — diffusers `controlnet_conditioning_scale`,
    /// applied as a multiplier to each residual.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Tensor,
        conditioning: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
        conditioning_scale: f32,
    ) -> Result<Vec<Tensor>> {
        let dtype = img.dtype();
        // Position embeddings cover the concat'd [txt, img] sequence.
        let pe = {
            let ids = Tensor::cat(&[txt_ids, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
        let mut txt = txt.apply(&self.txt_in)?;
        // Inject conditioning at the input: hidden = x_embedder(img) + controlnet_x_embedder(cond).
        let mut img = (img.apply(&self.x_embedder)?
            + conditioning.apply(&self.controlnet_x_embedder)?)?;

        let vec_ = fi::timestep_embedding(timesteps, 256, dtype)?.apply(&self.time_in)?;
        let vec_ = match (self.guidance_in.as_ref(), guidance) {
            (Some(g_in), Some(g)) => {
                (vec_ + fi::timestep_embedding(g, 256, dtype)?.apply(g_in))?
            }
            _ => vec_,
        };
        let vec_ = (vec_ + y.apply(&self.vector_in))?;

        let mut block_outs: Vec<Tensor> = Vec::with_capacity(self.double_blocks.len());
        for block in &self.double_blocks {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
            block_outs.push(img.clone());
        }

        // Project each block output through its zero-conv head and
        // scale by the user's conditioning_scale. The result is what
        // the main Flux's `forward_with_residuals` will add onto its
        // own DoubleStream output at the matching block index.
        let scale = conditioning_scale as f64;
        let mut residuals: Vec<Tensor> = Vec::with_capacity(block_outs.len());
        for (sample, head) in block_outs.iter().zip(self.controlnet_blocks.iter()) {
            let projected = head.forward(sample)?;
            residuals.push((projected * scale)?);
        }
        Ok(residuals)
    }
}

// =====================================================================
// Diffusers → BFL key remapping
// =====================================================================
//
// Community Flux ControlNets ship in diffusers naming. Convert at load
// time to the BFL-style HashMap the vendored DoubleStreamBlock expects.

/// Convert a diffusers-format Flux ControlNet state_dict to the BFL
/// naming convention the vendored types load from. The conversion is
/// purely renames + QKV fusion (concat Q, K, V along dim 0).
///
/// Returns a new HashMap suitable for `VarBuilder::from_tensors`.
pub fn remap_diffusers_to_bfl(
    diffusers: HashMap<String, Tensor>,
    num_layers: usize,
    guidance_embed: bool,
) -> Result<HashMap<String, Tensor>> {
    let get = |k: &str| -> Result<Tensor> {
        diffusers
            .get(k)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing diffusers key {k}"))
    };
    let fuse_qkv = |q_key: &str, k_key: &str, v_key: &str| -> Result<Tensor> {
        let q = get(q_key)?;
        let k = get(k_key)?;
        let v = get(v_key)?;
        Ok(Tensor::cat(&[&q, &k, &v], 0)?)
    };
    let mut out: HashMap<String, Tensor> = HashMap::new();

    // Embedders.
    out.insert("img_in.weight".into(), get("x_embedder.weight")?);
    out.insert("img_in.bias".into(), get("x_embedder.bias")?);
    out.insert(
        "controlnet_x_embedder.weight".into(),
        get("controlnet_x_embedder.weight")?,
    );
    out.insert(
        "controlnet_x_embedder.bias".into(),
        get("controlnet_x_embedder.bias")?,
    );
    out.insert("txt_in.weight".into(), get("context_embedder.weight")?);
    out.insert("txt_in.bias".into(), get("context_embedder.bias")?);
    out.insert(
        "time_in.in_layer.weight".into(),
        get("time_text_embed.timestep_embedder.linear_1.weight")?,
    );
    out.insert(
        "time_in.in_layer.bias".into(),
        get("time_text_embed.timestep_embedder.linear_1.bias")?,
    );
    out.insert(
        "time_in.out_layer.weight".into(),
        get("time_text_embed.timestep_embedder.linear_2.weight")?,
    );
    out.insert(
        "time_in.out_layer.bias".into(),
        get("time_text_embed.timestep_embedder.linear_2.bias")?,
    );
    out.insert(
        "vector_in.in_layer.weight".into(),
        get("time_text_embed.text_embedder.linear_1.weight")?,
    );
    out.insert(
        "vector_in.in_layer.bias".into(),
        get("time_text_embed.text_embedder.linear_1.bias")?,
    );
    out.insert(
        "vector_in.out_layer.weight".into(),
        get("time_text_embed.text_embedder.linear_2.weight")?,
    );
    out.insert(
        "vector_in.out_layer.bias".into(),
        get("time_text_embed.text_embedder.linear_2.bias")?,
    );
    if guidance_embed {
        out.insert(
            "guidance_in.in_layer.weight".into(),
            get("time_text_embed.guidance_embedder.linear_1.weight")?,
        );
        out.insert(
            "guidance_in.in_layer.bias".into(),
            get("time_text_embed.guidance_embedder.linear_1.bias")?,
        );
        out.insert(
            "guidance_in.out_layer.weight".into(),
            get("time_text_embed.guidance_embedder.linear_2.weight")?,
        );
        out.insert(
            "guidance_in.out_layer.bias".into(),
            get("time_text_embed.guidance_embedder.linear_2.bias")?,
        );
    }

    // DoubleStream blocks. Diffusers stores Q/K/V separately for both
    // image and text streams; BFL fuses them.
    for i in 0..num_layers {
        let d = |k: &str| format!("transformer_blocks.{i}.{k}");
        let b = |k: &str| format!("double_blocks.{i}.{k}");

        // Modulation Linears (norm1.linear → img_mod.lin).
        out.insert(b("img_mod.lin.weight"), get(&d("norm1.linear.weight"))?);
        out.insert(b("img_mod.lin.bias"), get(&d("norm1.linear.bias"))?);
        out.insert(b("txt_mod.lin.weight"), get(&d("norm1_context.linear.weight"))?);
        out.insert(b("txt_mod.lin.bias"), get(&d("norm1_context.linear.bias"))?);

        // Image stream QKV fusion + projection + QkNorm + MLP.
        out.insert(
            b("img_attn.qkv.weight"),
            fuse_qkv(
                &d("attn.to_q.weight"),
                &d("attn.to_k.weight"),
                &d("attn.to_v.weight"),
            )?,
        );
        out.insert(
            b("img_attn.qkv.bias"),
            fuse_qkv(
                &d("attn.to_q.bias"),
                &d("attn.to_k.bias"),
                &d("attn.to_v.bias"),
            )?,
        );
        out.insert(
            b("img_attn.proj.weight"),
            get(&d("attn.to_out.0.weight"))?,
        );
        out.insert(b("img_attn.proj.bias"), get(&d("attn.to_out.0.bias"))?);
        out.insert(
            b("img_attn.norm.query_norm.scale"),
            get(&d("attn.norm_q.weight"))?,
        );
        out.insert(
            b("img_attn.norm.key_norm.scale"),
            get(&d("attn.norm_k.weight"))?,
        );
        out.insert(b("img_mlp.0.weight"), get(&d("ff.net.0.proj.weight"))?);
        out.insert(b("img_mlp.0.bias"), get(&d("ff.net.0.proj.bias"))?);
        out.insert(b("img_mlp.2.weight"), get(&d("ff.net.2.weight"))?);
        out.insert(b("img_mlp.2.bias"), get(&d("ff.net.2.bias"))?);

        // Text stream parallel set.
        out.insert(
            b("txt_attn.qkv.weight"),
            fuse_qkv(
                &d("attn.add_q_proj.weight"),
                &d("attn.add_k_proj.weight"),
                &d("attn.add_v_proj.weight"),
            )?,
        );
        out.insert(
            b("txt_attn.qkv.bias"),
            fuse_qkv(
                &d("attn.add_q_proj.bias"),
                &d("attn.add_k_proj.bias"),
                &d("attn.add_v_proj.bias"),
            )?,
        );
        out.insert(
            b("txt_attn.proj.weight"),
            get(&d("attn.to_add_out.weight"))?,
        );
        out.insert(b("txt_attn.proj.bias"), get(&d("attn.to_add_out.bias"))?);
        out.insert(
            b("txt_attn.norm.query_norm.scale"),
            get(&d("attn.norm_added_q.weight"))?,
        );
        out.insert(
            b("txt_attn.norm.key_norm.scale"),
            get(&d("attn.norm_added_k.weight"))?,
        );
        out.insert(
            b("txt_mlp.0.weight"),
            get(&d("ff_context.net.0.proj.weight"))?,
        );
        out.insert(
            b("txt_mlp.0.bias"),
            get(&d("ff_context.net.0.proj.bias"))?,
        );
        out.insert(b("txt_mlp.2.weight"), get(&d("ff_context.net.2.weight"))?);
        out.insert(b("txt_mlp.2.bias"), get(&d("ff_context.net.2.bias"))?);
    }

    // controlnet_blocks zero-conv heads.
    for i in 0..num_layers {
        out.insert(
            format!("controlnet_blocks.{i}.weight"),
            get(&format!("controlnet_blocks.{i}.weight"))?,
        );
        out.insert(
            format!("controlnet_blocks.{i}.bias"),
            get(&format!("controlnet_blocks.{i}.bias"))?,
        );
    }

    Ok(out)
}

/// Load a community Flux ControlNet from HuggingFace. Downloads the
/// diffusers-format safetensors, remaps to BFL naming, and constructs
/// via `VarBuilder::from_tensors`.
pub async fn load_from_hf(
    repo: &str,
    file: &str,
    cfg: Config,
    device: &Device,
    dtype: DType,
) -> Result<FluxControlNet> {
    let path = crate::hf::download::get_file(repo, file)
        .await
        .with_context(|| format!("downloading Flux ControlNet {repo}/{file}"))?;
    let diffusers: HashMap<String, Tensor> =
        candle_core::safetensors::load(&path, device).with_context(|| {
            format!("loading Flux ControlNet safetensors from {}", path.display())
        })?;
    // Bail loud if SingleStream blocks are present — not yet supported.
    if diffusers
        .keys()
        .any(|k| k.starts_with("single_transformer_blocks."))
    {
        anyhow::bail!(
            "Flux ControlNet {repo}/{file}: SingleStream blocks present in \
             state_dict. plakat's current Flux ControlNet supports DoubleStream \
             blocks only — this checkpoint will be supported in a follow-up. \
             Try a ControlNet with `num_single_layers = 0` (e.g. InstantX/canny)."
        );
    }
    let bfl =
        remap_diffusers_to_bfl(diffusers, cfg.num_layers, cfg.guidance_embed)?;
    let bfl_cast: HashMap<String, Tensor> = bfl
        .into_iter()
        .map(|(k, v)| {
            let v = v.to_dtype(dtype).map_err(anyhow::Error::from)?;
            Ok::<_, anyhow::Error>((k, v))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let vb = VarBuilder::from_tensors(bfl_cast, dtype, device);
    FluxControlNet::new(&cfg, vb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_reasonable() {
        let c = Config::instantx_canny();
        assert_eq!(c.num_layers, 5);
        assert!(c.guidance_embed);
    }
}
