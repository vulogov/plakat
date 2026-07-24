//! Sana 1.6B (1024px) — the sixth model family (ROADMAP_4.5.0).
//!
//! A linear-attention DiT with two components unlike anything else in plakat: **DC-AE**, a
//! deep-compression autoencoder (32× spatial, 32 latent channels — not the `AutoEncoderKL`
//! every other pipeline uses; phase 1), and **Gemma-2-2B**, a decoder-only LLM used as the
//! text encoder instead of T5/CLIP (phase 2). The DiT (phase 3) pairs ReLU **linear**
//! self-attention with vanilla softmax cross-attention, a GLUMBConv Mix-FFN, and AdaLN-single,
//! sampled with a flow-matching scheduler (phase 4).
//!
//! **Phase 0 (this file):** dispatch wiring only. `run()` resolves the alias then errors with a
//! clear "not implemented yet" so `plakat generate --model sana` routes here correctly. The
//! per-component phases replace the stub with a real `Pipeline::load` + denoise loop.

use anyhow::{Result, bail};
use candle_core::Device;

use crate::pipelines::scheduler::SchedulerKind;

/// A Sana text-to-image run. Mirrors [`pixart::RunRequest`](crate::pipelines::pixart::RunRequest)
/// so the `t2i::run` dispatch fan-out treats every DiT family the same way.
pub struct RunRequest {
    pub model: String,
    pub device: Device,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub scheduler: SchedulerKind,
    pub out_dir: std::path::PathBuf,
    /// Count of images (per-image seed = base + idx).
    pub count: u32,
    /// LoRA stack (resolved or unresolved); `run()` resolves before load.
    pub loras: Vec<crate::pipelines::lora::LoraSpec>,
    pub lora_scale: f32,
}

/// Run a Sana text-to-image generation.
///
/// **Phase 0 stub.** Confirms the dispatch routes `--model sana` here, then bails with the
/// build status. Phases 1–4 replace this with the real DC-AE / Gemma / DiT / flow-match path.
pub async fn run(req: RunRequest) -> Result<()> {
    let repo = crate::hf::resolve_alias(&req.model);
    bail!(
        "Sana ({repo}) is not implemented yet — the pipeline is being built across \
         ROADMAP_4.5.0 phases (DC-AE autoencoder, Gemma-2-2B encoder, Linear-DiT, \
         flow-matching t2i). Track progress in ROADMAP_4.5.0.md.\n\
         Requested: {}×{} · {} steps · guidance {} · seed {:?} · {} image(s).",
        req.width,
        req.height,
        req.steps,
        req.guidance,
        req.seed,
        req.count,
    )
}
