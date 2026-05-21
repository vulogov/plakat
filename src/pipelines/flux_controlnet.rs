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
use candle_nn::{Embedding, Linear, VarBuilder};
use std::collections::HashMap;

use crate::pipelines::flux_inner::{
    self as fi, DoubleStreamBlock, EmbedNd, MlpEmbedder, SingleStreamBlock,
};

/// FluxControlNet shape config. Two common topologies in the wild:
///
/// * **Specialised** (InstantX/canny, Shakker-Labs/depth):
///   `num_layers = 5`, `num_single_layers = 0`, `num_mode = None`.
///   Each conditioning type gets its own checkpoint.
/// * **Union** (Shakker-Labs/Union-Pro, InstantX/Union):
///   `num_layers = 5`, `num_single_layers = 10`, `num_mode = Some(N)`.
///   One checkpoint handles N conditioning types via the mode
///   embedder.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of DoubleStream ControlNet blocks. 5 in every shipped
    /// community model so far.
    pub num_layers: usize,
    /// Number of SingleStream blocks. 0 for specialised CNs, 10 for
    /// Union variants.
    pub num_single_layers: usize,
    /// Whether the ControlNet has a guidance-aware time embedding.
    /// True for FLUX.1-dev derivatives, false for FLUX.1-schnell ones.
    pub guidance_embed: bool,
    /// `Some(N)` for Union ControlNets — N conditioning modes share
    /// one checkpoint via a `controlnet_mode_embedder`. `None` for
    /// specialised CNs.
    pub num_mode: Option<usize>,
}

impl Config {
    pub fn instantx_canny() -> Self {
        Self {
            num_layers: 5,
            num_single_layers: 0,
            guidance_embed: true,
            num_mode: None,
        }
    }

    /// Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro (v1).
    /// 7 modes: canny, tile, depth, blur, pose, gray, lq.
    pub fn shakker_union_pro() -> Self {
        Self {
            num_layers: 5,
            num_single_layers: 10,
            guidance_embed: true,
            num_mode: Some(7),
        }
    }

    /// Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0.
    /// 5 modes: canny, soft_edge, pose, depth, gray.
    pub fn shakker_union_pro_v2() -> Self {
        Self {
            num_layers: 5,
            num_single_layers: 10,
            guidance_embed: true,
            num_mode: Some(5),
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
    /// SingleStream blocks (Union ControlNets only). Empty for
    /// specialised CNs.
    single_blocks: Vec<SingleStreamBlock>,
    /// Zero-conv heads for the SingleStream residuals. Same length
    /// as `single_blocks`.
    controlnet_single_blocks: Vec<Linear>,
    /// Union ControlNet mode embedder. `None` for specialised CNs.
    /// When present, `forward` takes a mode index and prepends the
    /// looked-up embedding to the encoder hidden states.
    mode_embedder: Option<Embedding>,
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
            controlnet_blocks.push(candle_nn::linear(h, h, vb_cb.pp(i))?);
        }

        // SingleStream blocks (Union only).
        let mut single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_s = vb.pp("single_blocks");
        for i in 0..cfg.num_single_layers {
            single_blocks.push(SingleStreamBlock::new(&flux_cfg, vb_s.pp(i))?);
        }
        let mut controlnet_single_blocks = Vec::with_capacity(cfg.num_single_layers);
        let vb_csb = vb.pp("controlnet_single_blocks");
        for i in 0..cfg.num_single_layers {
            controlnet_single_blocks.push(candle_nn::linear(h, h, vb_csb.pp(i))?);
        }

        // Mode embedder (Union only).
        let mode_embedder = match cfg.num_mode {
            Some(n) => Some(candle_nn::embedding(n, h, vb.pp("controlnet_mode_embedder"))?),
            None => None,
        };

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
            single_blocks,
            controlnet_single_blocks,
            mode_embedder,
        })
    }

    /// Run the ControlNet. Returns `(double_residuals,
    /// single_residuals)` — the second vec is empty for specialised
    /// (non-Union) CNs. Both flow into the main Flux's
    /// `forward_with_residuals`.
    ///
    /// `mode` — Union ControlNet mode index. Required when the CN
    /// has a mode embedder, rejected when it doesn't.
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
        mode: Option<u32>,
        conditioning_scale: f32,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let dtype = img.dtype();
        // Encoder hidden states: project T5 to hidden_size first;
        // then if the ControlNet is Union, prepend the mode embedding
        // and extend txt_ids with a leading zero row so the position
        // embedder accounts for the extra leading token.
        let mut txt = txt.apply(&self.txt_in)?;
        let mut txt_ids_eff: Tensor = txt_ids.clone();
        if let Some(embedder) = self.mode_embedder.as_ref() {
            let mode = mode.ok_or_else(|| {
                anyhow::anyhow!(
                    "Union FluxControlNet requires a mode index — caller \
                     supplied none."
                )
            })?;
            if (mode as usize) >= embedder.embeddings().dim(0)? {
                anyhow::bail!(
                    "Union FluxControlNet mode {mode} out of range; this \
                     checkpoint has {} modes",
                    embedder.embeddings().dim(0)?
                );
            }
            let mode_t =
                Tensor::from_slice(&[mode], 1, txt.device())?.to_dtype(DType::U32)?;
            let mode_emb = embedder.forward(&mode_t)?; // (1, hidden)
            let mode_emb = mode_emb.unsqueeze(0)?.to_dtype(dtype)?; // (1, 1, hidden)
            txt = Tensor::cat(&[&mode_emb, &txt], 1)?;
            let zero_row = Tensor::zeros(
                (txt_ids_eff.dim(0)?, 1, txt_ids_eff.dim(2)?),
                txt_ids_eff.dtype(),
                txt_ids_eff.device(),
            )?;
            txt_ids_eff = Tensor::cat(&[&zero_row, &txt_ids_eff], 1)?;
        } else if mode.is_some() {
            anyhow::bail!(
                "Caller supplied a mode index but this is a specialised \
                 (non-Union) FluxControlNet."
            );
        }

        // Position embeddings cover the concat'd [txt, img] sequence
        // — extended txt seq when Union mode is active.
        let pe = {
            let ids = Tensor::cat(&[&txt_ids_eff, img_ids], 1)?;
            ids.apply(&self.pe_embedder)?
        };
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

        // DoubleStream pass, collecting per-block img outputs.
        let mut block_outs: Vec<Tensor> = Vec::with_capacity(self.double_blocks.len());
        for block in &self.double_blocks {
            (img, txt) = block.forward(&img, &txt, &vec_, &pe)?;
            block_outs.push(img.clone());
        }
        let scale = conditioning_scale as f64;
        let mut double_residuals: Vec<Tensor> = Vec::with_capacity(block_outs.len());
        for (sample, head) in block_outs.iter().zip(self.controlnet_blocks.iter()) {
            let projected = head.forward(sample)?;
            double_residuals.push((projected * scale)?);
        }

        // SingleStream pass (Union only). Mirror the main Flux's
        // single-block loop: concat [txt, img] along seq dim, run
        // each block on the combined hidden state. The zero-conv
        // head reads only the image tail (txt tokens at positions
        // 0..txt_len are pass-through and don't get residualised).
        let mut single_residuals: Vec<Tensor> = Vec::with_capacity(self.single_blocks.len());
        if !self.single_blocks.is_empty() {
            let txt_len = txt.dim(1)?;
            let mut img_single = Tensor::cat(&[&txt, &img], 1)?;
            for (block, head) in self
                .single_blocks
                .iter()
                .zip(self.controlnet_single_blocks.iter())
            {
                img_single = block.forward(&img_single, &vec_, &pe)?;
                let img_tail = img_single
                    .narrow(1, txt_len, img_single.dim(1)? - txt_len)?;
                let projected = head.forward(&img_tail)?;
                single_residuals.push((projected * scale)?);
            }
        }

        Ok((double_residuals, single_residuals))
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
    num_single_layers: usize,
    guidance_embed: bool,
    has_mode_embedder: bool,
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

    // SingleStream blocks (Union variants). Fuses Q/K/V/MLP_up into
    // `linear1` (BFL convention) and renames the rest.
    for i in 0..num_single_layers {
        let d = |k: &str| format!("single_transformer_blocks.{i}.{k}");
        let b = |k: &str| format!("single_blocks.{i}.{k}");

        // norm.linear is the modulation Linear (3*hidden output).
        out.insert(b("modulation.lin.weight"), get(&d("norm.linear.weight"))?);
        out.insert(b("modulation.lin.bias"), get(&d("norm.linear.bias"))?);

        // QKV + MLP_up fusion into linear1.
        let q_w = get(&d("attn.to_q.weight"))?;
        let k_w = get(&d("attn.to_k.weight"))?;
        let v_w = get(&d("attn.to_v.weight"))?;
        let mlp_up_w = get(&d("proj_mlp.weight"))?;
        out.insert(
            b("linear1.weight"),
            Tensor::cat(&[&q_w, &k_w, &v_w, &mlp_up_w], 0)?,
        );
        let q_b = get(&d("attn.to_q.bias"))?;
        let k_b = get(&d("attn.to_k.bias"))?;
        let v_b = get(&d("attn.to_v.bias"))?;
        let mlp_up_b = get(&d("proj_mlp.bias"))?;
        out.insert(
            b("linear1.bias"),
            Tensor::cat(&[&q_b, &k_b, &v_b, &mlp_up_b], 0)?,
        );

        // linear2 = proj_out (output projection from concat'd
        // attn+mlp back to hidden).
        out.insert(b("linear2.weight"), get(&d("proj_out.weight"))?);
        out.insert(b("linear2.bias"), get(&d("proj_out.bias"))?);

        // QkNorm.
        out.insert(
            b("norm.query_norm.scale"),
            get(&d("attn.norm_q.weight"))?,
        );
        out.insert(b("norm.key_norm.scale"), get(&d("attn.norm_k.weight"))?);

        // controlnet_single_blocks zero-conv heads.
        out.insert(
            format!("controlnet_single_blocks.{i}.weight"),
            get(&format!("controlnet_single_blocks.{i}.weight"))?,
        );
        out.insert(
            format!("controlnet_single_blocks.{i}.bias"),
            get(&format!("controlnet_single_blocks.{i}.bias"))?,
        );
    }

    // Mode embedder (Union only). Diffusers stores it as a Linear-
    // style `.weight` tensor of shape (num_modes, hidden) — same
    // shape candle's `embedding` loader expects.
    if has_mode_embedder {
        out.insert(
            "controlnet_mode_embedder.weight".into(),
            get("controlnet_mode_embedder.weight")?,
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
    // Quick sanity — bail if the safetensors carries more
    // single_transformer_blocks than the config expects. Catches the
    // "loaded Union with specialised config" mistake early.
    let observed_single = diffusers
        .keys()
        .filter_map(|k| {
            k.strip_prefix("single_transformer_blocks.")
                .and_then(|tail| tail.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    if observed_single > cfg.num_single_layers {
        anyhow::bail!(
            "Flux ControlNet {repo}/{file}: state_dict carries {observed_single} \
             single_transformer_blocks but Config specifies num_single_layers={}. \
             Pick a Union config (e.g. shakker_union_pro / shakker_union_pro_v2).",
            cfg.num_single_layers
        );
    }
    let bfl = remap_diffusers_to_bfl(
        diffusers,
        cfg.num_layers,
        cfg.num_single_layers,
        cfg.guidance_embed,
        cfg.num_mode.is_some(),
    )?;
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
