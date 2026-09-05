//! SD3 / SD3.5 ControlNet — InstantX-style residual producer.
//!
//! Modelled on the InstantX SD3 ControlNet collection (`InstantX/SD3-Controlnet-*`
//! repos) and the diffusers `SD3ControlNetModel` reference. The model
//! is a smaller MMDiT transformer (typically 12-18 joint blocks vs
//! the base 24/38) that takes:
//!
//!   * `hidden_states` — the same noise latent the main MMDiT sees
//!     (after patchify + positional embed).
//!   * `controlnet_cond` — the conditioning latent (VAE-encoded edge
//!     map / depth map / pose map). Added to `hidden_states` after
//!     the CN's own patchify.
//!   * `encoder_hidden_states` — the (T5) context tokens.
//!   * `pooled_projections` — the (CLIP-G || CLIP-L) pooled `y`.
//!   * `timestep` — broadcast scalar.
//!
//! and emits one residual per joint block. Each residual goes through
//! a zero-initialised `controlnet_blocks[i]` linear before being added
//! to the corresponding base MMDiT joint block's `x` output (the
//! `forward_with_residuals` hook from v0.15 phase 6a consumes the
//! residual list with the same `ceil(blocks/residuals)` interleave
//! the Flux CN uses).
//!
//! ## State-dict naming
//!
//! Diffusers SD3 ControlNet checkpoints (e.g. `InstantX/SD3-Controlnet-Canny`)
//! use these key prefixes under the safetensors root:
//!
//! ```text
//!     pos_embed.proj              — controlnet_x_embedder (Conv2d)
//!     pos_embed_input.proj        — conditioning embedder (Conv2d)
//!     time_text_embed.timestep_embedder.linear_1 / linear_2
//!     time_text_embed.text_embedder.linear_1 / linear_2
//!     context_embedder            — Linear projecting 4096 → hidden
//!     transformer_blocks.{i}.*    — joint blocks
//!     controlnet_blocks.{i}       — zero-conv residual heads
//! ```
//!
//! The InstantX repos ship variants of this with the same convention;
//! state-dict remap to plakat's vendored MMDiT joint-block path
//! happens in the loader (subsequent commit). This module defines the
//! model structure + forward; the loader is `sd3_controlnet::loader`.

use anyhow::{Context, Result};
use candle_core::{D, Module, Tensor};
use candle_nn::{Conv2d, Linear, VarBuilder};

use crate::pipelines::lora_linear::LoraRegistry;
use crate::pipelines::mmdit_inner::{
    JointBlock, MMDiTJointBlock, MMDiTXJointBlock, PatchEmbedder, PositionEmbedder,
    TimestepEmbedder, VectorEmbedder,
};

/// Configuration for an SD3 ControlNet model. Mirrors the diffusers
/// `SD3ControlNetModel` config fields, with the same defaults the
/// InstantX checkpoints publish.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of joint blocks the CN ships. Determines how many
    /// residuals the model emits per forward. Typical values:
    ///   * `12` — InstantX Canny / Pose / Tile (compact)
    ///   * `23` — full-depth alternative
    pub num_layers: usize,
    /// 2 — SD3 patch size; should match the base MMDiT.
    pub patch_size: usize,
    /// 16 — latent channel count (SD3 VAE).
    pub in_channels: usize,
    /// `head_size * num_heads`. SD3/SD3.5-Medium: 1536. SD3.5-Large:
    /// 2432.
    pub hidden_size: usize,
    /// 24 for both Medium variants; 38 for SD3.5-Large.
    pub num_heads: usize,
    /// 2048 — pooled-CLIP `y` vector dim.
    pub adm_in_channels: usize,
    /// 192 (SD3 / SD3.5-Large) or 384 (SD3.5-Medium).
    pub pos_embed_max_size: usize,
    /// 4096 — T5 hidden dim.
    pub context_embed_size: usize,
    /// 256 — sinusoidal embedding dim for timesteps.
    pub frequency_embedding_size: usize,
}

impl Config {
    /// 12-layer compact CN matching InstantX's SD3-Controlnet-Canny / -Pose / -Tile / -Depth. These are
    /// trained for the ORIGINAL SD3-medium, whose MMDiT uses `pos_embed_max_size = 192` (verified against
    /// the checkpoint: `pos_embed.pos_embed` is `[1, 192², 1536]`). NOTE: SD3.5-medium's base uses 384, so
    /// pairing this CN with an sd35-medium base is an architecture mismatch — it loads but the positional
    /// grids differ, so spatial control is imprecise. There is no InstantX ControlNet for SD3.5-medium.
    pub fn instantx_sd35_medium() -> Self {
        Self {
            num_layers: 12,
            patch_size: 2,
            in_channels: 16,
            hidden_size: 1536,
            num_heads: 24,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }

    /// 12-layer compact CN matching SD3 / SD3.5-Large (smaller
    /// `pos_embed_max_size = 192` than 3.5-Medium).
    pub fn instantx_sd35_large() -> Self {
        Self {
            num_layers: 12,
            patch_size: 2,
            in_channels: 16,
            hidden_size: 2432,
            num_heads: 38,
            adm_in_channels: 2048,
            pos_embed_max_size: 192,
            context_embed_size: 4096,
            frequency_embedding_size: 256,
        }
    }
}

/// The ControlNet model. Constructed from a `Config` + a
/// `candle_nn::VarBuilder` rooted at the safetensors location for the
/// model (e.g. the `transformer/` subdir of an InstantX repo, or the
/// repo root depending on file layout).
pub struct Sd3ControlNet {
    /// Patchifies the input latent. Same architecture as the main
    /// MMDiT's `x_embedder` — separate weights.
    x_embedder: PatchEmbedder,
    /// Conv2d that patchifies the conditioning latent (canny / depth
    /// map etc., VAE-encoded into 16-channel latent space). Output
    /// added to `x_embedder`'s output at the start of forward.
    pos_embed_input: Conv2d,
    /// Positional embedding (own copy, not shared with main MMDiT).
    pos_embedder: PositionEmbedder,
    /// Timestep + pooled-CLIP combined modulation signal — same
    /// architecture as the main MMDiT.
    timestep_embedder: TimestepEmbedder,
    vector_embedder: VectorEmbedder,
    /// T5 context projection from 4096 → hidden_size.
    context_embedder: Linear,
    /// Joint blocks — the CN's `transformer_blocks` in diffusers
    /// naming. Subset (typically 12) of what the main MMDiT carries.
    joint_blocks: Vec<Box<dyn JointBlock>>,
    /// Zero-initialised residual heads — one per `joint_blocks`. Each
    /// projects a joint block's `x` output to a residual tensor the
    /// main MMDiT consumes.
    controlnet_blocks: Vec<Linear>,
}

impl Sd3ControlNet {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // v0.16 phase 3a: the CN's joint blocks reuse mmdit_inner's
        // joint-block types, which take a LoRA registry. ControlNets
        // are separate models; user LoRAs target the base MMDiT, not
        // the CN. Build a throwaway registry — same pattern flux_controlnet
        // uses (the discarded registry is a few-dozen-entry HashMap).
        let cn_registry = std::sync::Arc::new(std::sync::RwLock::new(
            LoraRegistry::new(),
        ));

        let x_embedder = PatchEmbedder::new(
            cfg.patch_size,
            cfg.in_channels,
            cfg.hidden_size,
            vb.pp("pos_embed"),
        )
        .context("loading CN x_embedder (pos_embed.proj)")?;

        // The conditioning patchifier — diffusers calls it
        // `pos_embed_input.proj`. Same shape as `pos_embed.proj` (a
        // Conv2d with kernel/stride = patch_size, channels = 16 → hidden).
        let pos_embed_input = candle_nn::conv2d(
            cfg.in_channels,
            cfg.hidden_size,
            cfg.patch_size,
            candle_nn::Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb.pp("pos_embed_input.proj"),
        )
        .context("loading CN pos_embed_input.proj")?;

        // The learned positional embedding lives at `pos_embed.pos_embed` in the diffusers checkpoint (the
        // `pos_embed` PatchEmbed module's buffer), NOT at the root `pos_embed`. Load it from that prefix.
        let pos_embedder = PositionEmbedder::new(
            cfg.hidden_size,
            cfg.patch_size,
            cfg.pos_embed_max_size,
            vb.pp("pos_embed"),
        )
        .context("loading CN pos_embedder (pos_embed.pos_embed)")?;

        // Diffusers' `time_text_embed` packs a timestep MLP + a text
        // (pooled) MLP. We split into the two standard MMDiT
        // embedders, each at its diffusers path.
        let timestep_embedder = TimestepEmbedder::new(
            cfg.hidden_size,
            cfg.frequency_embedding_size,
            vb.pp("time_text_embed.timestep_embedder"),
            &cn_registry,
        )
        .context("loading CN timestep_embedder")?;
        let vector_embedder = VectorEmbedder::new(
            cfg.adm_in_channels,
            cfg.hidden_size,
            vb.pp("time_text_embed.text_embedder"),
            &cn_registry,
        )
        .context("loading CN text_embedder (pooled-CLIP projection)")?;

        let context_embedder = candle_nn::linear(
            cfg.context_embed_size,
            cfg.hidden_size,
            vb.pp("context_embedder"),
        )
        .context("loading CN context_embedder")?;

        let mut joint_blocks: Vec<Box<dyn JointBlock>> =
            Vec::with_capacity(cfg.num_layers);
        let vb_tb = vb.pp("transformer_blocks");
        for i in 0..cfg.num_layers {
            let vb_i = vb_tb.pp(i);
            // Detect MMDiT-X (has `attn2.qkv` per x_block) vs plain
            // MMDiT joint block. SD3.5-Medium CNs typically use
            // MMDiT-X; SD3 / SD3.5-Large use plain MMDiT.
            let attn2_probe = format!("{}.x_block.attn2.qkv.weight", i);
            let block: Box<dyn JointBlock> =
                if vb_tb.contains_tensor(&attn2_probe) {
                    Box::new(MMDiTXJointBlock::new(
                        cfg.hidden_size,
                        cfg.num_heads,
                        false,
                        vb_i,
                        &cn_registry,
                    )?)
                } else {
                    Box::new(MMDiTJointBlock::new(
                        cfg.hidden_size,
                        cfg.num_heads,
                        false,
                        vb_i,
                        &cn_registry,
                    )?)
                };
            joint_blocks.push(block);
        }

        let mut controlnet_blocks = Vec::with_capacity(cfg.num_layers);
        let vb_cb = vb.pp("controlnet_blocks");
        for i in 0..cfg.num_layers {
            controlnet_blocks.push(
                candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb_cb.pp(i))
                    .with_context(|| format!("loading CN controlnet_blocks.{i}"))?,
            );
        }

        Ok(Self {
            x_embedder,
            pos_embed_input,
            pos_embedder,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            joint_blocks,
            controlnet_blocks,
        })
    }

    /// Forward pass — produces one residual tensor per joint block.
    /// Caller passes the residual list to
    /// `mmdit_inner::MMDiT::forward_with_residuals` on the base model.
    ///
    /// Shapes:
    /// * `hidden_states`: `(B, 16, H, W)` — the noise latent (pre-patchify).
    /// * `controlnet_cond`: `(B, 16, H, W)` — VAE-encoded conditioning.
    /// * `encoder_hidden_states`: `(B, T, 4096)` — T5 + CLIP context.
    /// * `pooled_projections`: `(B, 2048)` — pooled CLIP-G || CLIP-L.
    /// * `timestep`: `(B,)` — broadcast across the batch.
    ///
    /// Output: `Vec<Tensor>` with shape `(B, N, hidden_size)` per
    /// entry, where `N = (H/patch) * (W/patch)` — the residuals
    /// applied to the corresponding base MMDiT joint block's `x`
    /// stream after each block.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        controlnet_cond: &Tensor,
        encoder_hidden_states: &Tensor,
        pooled_projections: &Tensor,
        timestep: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let h = hidden_states.dim(D::Minus2)?;
        let w = hidden_states.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;

        // Patchify the noise latent (same as base MMDiT) + add the
        // conditioning patchify on top. Both go through the same
        // (1, N, hidden) shape after the broadcast_add.
        let x_main = self
            .x_embedder
            .forward(hidden_states)
            .context("CN x_embedder forward")?
            .broadcast_add(&cropped_pos_embed)
            .context("CN broadcast_add positional")?;
        let x_cond_raw = self
            .pos_embed_input
            .forward(controlnet_cond)
            .context("CN pos_embed_input.proj forward")?;
        // pos_embed_input.proj outputs (B, hidden, h/patch, w/patch);
        // flatten to (B, N, hidden) so we can add to x_main.
        let (b, c, h_p, w_p) = x_cond_raw.dims4()?;
        let x_cond = x_cond_raw.reshape((b, c, h_p * w_p))?.transpose(1, 2)?;
        let x = (x_main + x_cond)?;

        // Modulation signal: timestep + pooled-CLIP combined.
        let c_t = self.timestep_embedder.forward(timestep)?;
        let c_y = self.vector_embedder.forward(pooled_projections)?;
        let c = (c_t + c_y)?;

        // Context (T5 + CLIP halves) projected to hidden.
        let context = self.context_embedder.forward(encoder_hidden_states)?;

        // Run the joint blocks, capturing each block's x output, then
        // project through the zero-conv head to produce a residual.
        let mut residuals = Vec::with_capacity(self.joint_blocks.len());
        let (mut ctx, mut x) = (context, x);
        for (i, block) in self.joint_blocks.iter().enumerate() {
            (ctx, x) = block.forward(&ctx, &x, &c, false)?;
            let res = x.apply(&self.controlnet_blocks[i])?;
            residuals.push(res);
        }
        Ok(residuals)
    }

    /// Number of joint blocks — equals the residual count emitted by
    /// `forward`. Callers use this to size the residual interleave
    /// when passing to base MMDiT's `forward_with_residuals`.
    pub fn n_residuals(&self) -> usize {
        self.joint_blocks.len()
    }
}

/// Per-instance ControlNet load spec — mirrors `FluxControlNetLoad`.
/// Carries the repo / file / config plus the runtime knobs (scale,
/// conditioning path, step-gating window) that the dispatch sets
/// per-task or per-call.
#[derive(Debug, Clone)]
pub struct Sd3ControlNetLoad {
    /// HuggingFace repo id, e.g. `"InstantX/SD3-Controlnet-Canny"`.
    pub repo: String,
    /// File within the repo. Default for InstantX repos:
    /// `"diffusion_pytorch_model.safetensors"`.
    pub file: String,
    /// Model architecture config.
    pub cfg: Config,
    /// Residual scale applied uniformly across all joint blocks.
    /// 1.0 = full strength, 0.0 = disable. Diffusers calls this
    /// `controlnet_conditioning_scale`.
    pub scale: f32,
    /// Path to the conditioning image (canny edges / depth map /
    /// pose map). The image is VAE-encoded at dispatch time and
    /// patchified by the CN's `pos_embed_input.proj`.
    pub conditioning: Option<std::path::PathBuf>,
    /// Step-gating window in `[0, 1]` schedule fractions. CN
    /// residuals contribute only when `start <= progress < end`.
    /// Diffusers calls these `control_guidance_start` /
    /// `control_guidance_end`.
    pub start: f32,
    pub end: f32,
}

/// A loaded SD3 ControlNet bound to a conditioning image. The
/// scenario / CLI dispatcher holds one of these per active CN slot
/// and mutates `scale` / `conditioning` / `start` / `end` per task.
pub struct LoadedSd3ControlNet {
    pub net: Sd3ControlNet,
    pub scale: f32,
    pub conditioning_path: Option<std::path::PathBuf>,
    pub start: f32,
    pub end: f32,
}

impl LoadedSd3ControlNet {
    /// `true` if the CN's step-gating window includes the given
    /// progress fraction. `progress` is the fraction through the
    /// denoise schedule in `[0, 1)` — same convention diffusers uses.
    pub fn active_at(&self, progress: f32) -> bool {
        progress >= self.start && progress < self.end
    }
}

/// Download a single InstantX SD3 ControlNet safetensors from
/// HuggingFace and construct the model. Async because the first
/// call on a cold cache downloads ~2-3 GB.
///
/// Returns the model — the dispatcher wraps it in a
/// [`LoadedSd3ControlNet`] with the per-instance runtime knobs.
///
/// The InstantX repos publish weights with key names that match the
/// diffusers `SD3ControlNetModel` convention (`pos_embed.proj`,
/// `pos_embed_input.proj`, `time_text_embed.timestep_embedder.*`,
/// `transformer_blocks.{i}.*`, `controlnet_blocks.{i}`). Our
/// `Sd3ControlNet::new` resolves directly against those paths, so
/// no state-dict remap is needed (unlike the diffusers→BFL remap
/// Flux ControlNet requires).
pub async fn load_from_hf(
    repo: &str,
    file: &str,
    cfg: &Config,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Sd3ControlNet> {
    let path = crate::hf::download::get_file(repo, file)
        .await
        .with_context(|| format!("downloading SD3 ControlNet {repo}/{file}"))?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&path], dtype, device)?
    };
    Sd3ControlNet::new(cfg, vb).with_context(|| {
        format!("constructing SD3 ControlNet from {repo}/{file}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantx_sd35_medium_config_matches_known_shape() {
        let c = Config::instantx_sd35_medium();
        assert_eq!(c.num_layers, 12);
        assert_eq!(c.hidden_size, 1536);
        assert_eq!(c.num_heads, 24);
        // 192², not 384: this config loads InstantX/SD3-Controlnet-Canny, whose `pos_embed.pos_embed` is
        // `[1, 192², 1536]` (verified against the checkpoint). It's an original-SD3-medium CN.
        assert_eq!(c.pos_embed_max_size, 192);
        assert_eq!(c.in_channels, 16);
        assert_eq!(c.context_embed_size, 4096);
    }

    #[test]
    fn instantx_sd35_large_config_matches_known_shape() {
        let c = Config::instantx_sd35_large();
        assert_eq!(c.hidden_size, 2432);
        assert_eq!(c.num_heads, 38);
        assert_eq!(c.pos_embed_max_size, 192);
    }

    #[test]
    fn configs_match_main_mmdit_invariants() {
        // The CN must share patch_size, adm_in_channels,
        // frequency_embedding_size, and context_embed_size with the
        // main MMDiT — those are protocol constants between the two
        // models. Anything that diverges would produce mis-shaped
        // residuals that the base model rejects at add time.
        for c in [
            Config::instantx_sd35_medium(),
            Config::instantx_sd35_large(),
        ] {
            assert_eq!(c.patch_size, 2);
            assert_eq!(c.in_channels, 16);
            assert_eq!(c.adm_in_channels, 2048);
            assert_eq!(c.frequency_embedding_size, 256);
            assert_eq!(c.context_embed_size, 4096);
        }
    }

    /// Standalone gating predicate to exercise the same math
    /// `LoadedSd3ControlNet::active_at` runs without needing a real
    /// `Sd3ControlNet`. The struct can't be `mem::zeroed` safely
    /// (Vec / Linear fields have ownership invariants), so we test
    /// the bool by mirroring its implementation here.
    fn gate_active_at(start: f32, end: f32, progress: f32) -> bool {
        progress >= start && progress < end
    }

    #[test]
    fn gating_window_full_range() {
        // Standard start=0.0, end=1.0: active throughout, exclusive
        // at the right edge.
        assert!(gate_active_at(0.0, 1.0, 0.0));
        assert!(gate_active_at(0.0, 1.0, 0.5));
        assert!(!gate_active_at(0.0, 1.0, 1.0));
    }

    #[test]
    fn gating_window_excludes_outside() {
        let (s, e) = (0.2, 0.6);
        assert!(!gate_active_at(s, e, 0.0));
        assert!(!gate_active_at(s, e, 0.1));
        assert!(gate_active_at(s, e, 0.2));
        assert!(gate_active_at(s, e, 0.5));
        assert!(!gate_active_at(s, e, 0.6));
        assert!(!gate_active_at(s, e, 0.9));
    }
}
