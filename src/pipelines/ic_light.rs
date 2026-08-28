//! IC-Light — "Imposing Consistent Light" relighting (lllyasviel).
//!
//! Relights a foreground subject according to a text prompt describing
//! the desired lighting ("sunset glow from the left", "neon city
//! ambience", …). The architecture is plain SD 1.5 with two
//! modifications, both handled at load time:
//!
//!   1. **Widened input conv** — the UNet's `conv_in.weight` goes from
//!      (320, 4, 3, 3) to (320, 8, 3, 3). The first 4 input channels are
//!      the usual noisy latent; the extra 4 carry the VAE latent of the
//!      subject-on-grey conditioning image. At every denoise step we
//!      concat the subject latent onto the noisy latent along the
//!      channel dim → 8 channels.
//!
//!   2. **Offset weights** — the IC-Light release ships an *offset*
//!      (`iclight_sd15_fc.safetensors`), not a full UNet. Each tensor is
//!      added on top of the matching base SD 1.5 UNet tensor. The
//!      offset's own `conv_in.weight` is already 8-channel, so it adds
//!      onto the zero-padded widened base conv.
//!
//! This module is the "FC" (foreground-conditioned) variant: the model
//! is told what the subject is and invents lighting from the prompt.
//! The foreground is matted off its background (U2Net), composited onto
//! a neutral grey field, and VAE-encoded to produce the conditioning
//! latent.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

use candle_transformers::models::stable_diffusion::{
    StableDiffusionConfig, clip as sdclip, vae::AutoEncoderKL,
};

use crate::pipelines::sdxl_unet::SdUNet;
use crate::pipelines::vendored_clip::{self, ClipTextTransformer};
use crate::ui::progress;

/// SD 1.5 VAE scaling factor (diffusers `vae.config.scaling_factor`).
const SD15_VAE_SCALE: f64 = 0.18215;

/// IC-Light offset weights (foreground-conditioned variant).
const ICLIGHT_REPO: &str = "lllyasviel/ic-light";
const ICLIGHT_FILE: &str = "iclight_sd15_fc.safetensors";

// =====================================================================
// Offset merge — the load-bearing part.
// =====================================================================

/// Build a merged 8-channel SD 1.5 UNet on disk from the base diffusers
/// UNet plus the IC-Light offset, returning the tempfile (the caller
/// must keep it alive so the mmap stays valid).
///
/// Steps:
///   * Load base UNet + offset into `HashMap<String, Tensor>`.
///   * Widen the base `conv_in.weight` (320,4,3,3) → (320,8,3,3) by
///     concatenating 4 zero input-channels.
///   * For every base key, add the offset tensor if present (after
///     prefix alignment), doing the arithmetic in F32 then casting back
///     to the base dtype.
///   * Save the merged map to a `.safetensors` tempfile.
///
/// Key alignment: some IC-Light releases prefix their state-dict keys
/// (`unet.` / `model.diffusion_model.`). For each base key we look up
/// the offset under the bare key first, then under each known prefix,
/// so a prefixed offset still merges. The merged/total count is logged
/// so a wholesale key mismatch (0 merged) is obvious.
fn merge_iclight_unet(
    base_unet_path: &Path,
    offset_path: &Path,
    device: &Device,
) -> Result<tempfile::NamedTempFile> {
    let base = candle_core::safetensors::load(base_unet_path, device)
        .with_context(|| format!("loading base UNet {}", base_unet_path.display()))?;
    let offset = candle_core::safetensors::load(offset_path, device)
        .with_context(|| format!("loading IC-Light offset {}", offset_path.display()))?;

    // Known prefixes an offset key might carry on top of the diffusers
    // state-dict naming. Tried in order; the first that resolves wins.
    const PREFIXES: &[&str] = &["", "unet.", "model.diffusion_model."];
    let lookup = |k: &str| -> Option<&Tensor> {
        for p in PREFIXES {
            if p.is_empty() {
                if let Some(t) = offset.get(k) {
                    return Some(t);
                }
            } else {
                let pk = format!("{p}{k}");
                if let Some(t) = offset.get(&pk) {
                    return Some(t);
                }
            }
        }
        None
    };

    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(base.len());
    let mut n_merged = 0usize;
    let conv_in_key = "conv_in.weight";

    for (k, base_t) in base.iter() {
        let base_dtype = base_t.dtype();
        if k == conv_in_key {
            // Widen (out, 4, 3, 3) → (out, 8, 3, 3): real weights in the
            // first 4 input channels, zeros in the last 4.
            let (oc, ic, kh, kw) = base_t.dims4().with_context(|| {
                format!("conv_in.weight expected 4-D, got {:?}", base_t.shape())
            })?;
            let base_f32 = base_t.to_dtype(DType::F32)?;
            let widened = if ic == 8 {
                // Already 8-channel (unlikely for a stock base UNet, but
                // handle gracefully): use as-is.
                base_f32
            } else {
                let zeros =
                    Tensor::zeros((oc, 8 - ic, kh, kw), DType::F32, device)?;
                Tensor::cat(&[&base_f32, &zeros], 1)?
            };
            let merged_t = match lookup(conv_in_key) {
                Some(off) => {
                    n_merged += 1;
                    let off = off.to_dtype(DType::F32)?;
                    (widened + off)?
                }
                None => widened,
            };
            merged.insert(k.clone(), merged_t.to_dtype(base_dtype)?);
            continue;
        }

        let merged_t = match lookup(k) {
            Some(off) => {
                n_merged += 1;
                let a = base_t.to_dtype(DType::F32)?;
                let b = off.to_dtype(DType::F32)?;
                (a + b)?.to_dtype(base_dtype)?
            }
            None => base_t.clone(),
        };
        merged.insert(k.clone(), merged_t);
    }

    tracing::info!(
        target: "plakat",
        "IC-Light merge: {}/{} base UNet tensors received an offset ({} offset tensors total)",
        n_merged,
        base.len(),
        offset.len()
    );
    if n_merged == 0 {
        anyhow::bail!(
            "IC-Light merge matched 0 of {} base UNet tensors against the offset \
             ({} offset tensors). The offset key naming did not align with the \
             diffusers UNet state-dict even after prefix stripping ({:?}).",
            base.len(),
            offset.len(),
            PREFIXES,
        );
    }

    let tmp = tempfile::Builder::new()
        .prefix("plakat-iclight-unet-")
        .suffix(".safetensors")
        .tempfile()?;
    candle_core::safetensors::save(&merged, tmp.path())
        .context("writing merged IC-Light UNet safetensors")?;
    Ok(tmp)
}

// =====================================================================
// Pipeline.
// =====================================================================

pub struct Pipeline {
    unet: SdUNet,
    vae: AutoEncoderKL,
    text_encoder: ClipTextTransformer,
    tokenizer: Tokenizer,
    cfg: StableDiffusionConfig,
    device: Device,
    dtype: DType,
    /// Kept alive so the merged-UNet mmap stays valid for the
    /// pipeline's lifetime.
    _unet_tmp: tempfile::NamedTempFile,
}

impl Pipeline {
    /// Download SD 1.5 + the IC-Light offset, merge into a widened
    /// 8-channel UNet, and build the VAE / CLIP-L text encoder.
    pub async fn load(device: Device) -> Result<Self> {
        let base_repo = crate::hf::resolve_alias("sd15").to_string();
        let dtype = if matches!(device, Device::Cpu) {
            DType::F32
        } else {
            DType::F16
        };

        let dl = progress::spinner("Resolving SD 1.5 + IC-Light weights");
        let tokenizer_path = crate::hf::download::get_first_of(&[
            (&base_repo, "tokenizer/tokenizer.json"),
            ("openai/clip-vit-large-patch14", "tokenizer.json"),
        ])
        .await
        .with_context(|| format!("tokenizer (CLIP-L) for {base_repo}"))?;
        let text_enc_path = crate::hf::download::get_first_of(&[
            (&base_repo, "text_encoder/model.fp16.safetensors"),
            (&base_repo, "text_encoder/model.safetensors"),
        ])
        .await
        .with_context(|| format!("text_encoder for {base_repo}"))?;
        let unet_path = crate::hf::download::get_first_of(&[
            (&base_repo, "unet/diffusion_pytorch_model.fp16.safetensors"),
            (&base_repo, "unet/diffusion_pytorch_model.safetensors"),
        ])
        .await
        .with_context(|| format!("unet for {base_repo}"))?;
        let vae_path = crate::hf::download::get_first_of(&[
            (&base_repo, "vae/diffusion_pytorch_model.fp16.safetensors"),
            (&base_repo, "vae/diffusion_pytorch_model.safetensors"),
        ])
        .await
        .with_context(|| format!("vae for {base_repo}"))?;
        let offset_path = crate::hf::download::get_file(ICLIGHT_REPO, ICLIGHT_FILE)
            .await
            .with_context(|| {
                format!("downloading IC-Light offset {ICLIGHT_REPO}/{ICLIGHT_FILE}")
            })?;
        dl.finish_with_message("✓ weights ready");

        let cfg = StableDiffusionConfig::v1_5(None, None, None);

        // Merge the offset onto the widened base UNet → tempfile.
        let merge = progress::spinner("Merging IC-Light offset into widened SD 1.5 UNet");
        let unet_tmp = merge_iclight_unet(&unet_path, &offset_path, &device)?;
        merge.finish_with_message("✓ IC-Light UNet merged (8-channel input)");

        let build = progress::spinner("Loading IC-Light core");
        // 8-channel conv_in. SD 1.5 → candle's stock UNet (no
        // add_embedding), wrapped as SdUNet::Sd.
        let unet = SdUNet::Sd(cfg.build_unet(unet_tmp.path(), &device, 8, false, dtype)?);
        let vae = cfg.build_vae(&vae_path, &device, dtype)?;
        let text_encoder = vendored_clip::build_clip_transformer(
            &vendored_clip::Config::v1_5(),
            &text_enc_path,
            &device,
            dtype,
        )?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("tokenizer (CLIP-L): {e}"))?;
        build.finish_with_message("✓ IC-Light core loaded");

        Ok(Self {
            unet,
            vae,
            text_encoder,
            tokenizer,
            cfg,
            device,
            dtype,
            _unet_tmp: unet_tmp,
        })
    }

    /// Encode one prompt to CLIP-L `encoder_hidden_states` `(1, 77, 768)`.
    fn encode_text(&self, text: &str) -> Result<Tensor> {
        let ids = tokenize_padded(&self.tokenizer, &self.cfg.clip, text, &self.device)?;
        Ok(self.text_encoder.forward(&ids)?.to_dtype(self.dtype)?)
    }

    /// Relight `subject_png` under `prompt`. Returns `(rgb_u8, width,
    /// height)`.
    ///
    /// The subject is matted (U2Net), composited onto a neutral grey
    /// (0.5) field at `width × height`, VAE-encoded, and fed as the
    /// extra 4 input channels at every denoise step.
    #[allow(clippy::too_many_arguments)]
    pub fn relight(
        &self,
        subject_png: &Path,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f64,
        seed: u64,
        backdrop: Backdrop,
    ) -> Result<(Vec<u8>, u32, u32)> {
        let (w, h) = (width as usize, height as usize);
        let do_cfg = guidance > 1.0;

        // ---- 1. matte → composite over the (possibly directional) backdrop ----
        let subject_rgb = self.subject_on_grey(subject_png, width, height, backdrop)?;

        // ---- 2. preprocess to [-1,1] (1,3,H,W) ----
        let cond_px =
            crate::imaging::preprocess::sd_image_tensor(&subject_rgb, width, height, &self.device, self.dtype)
                .context("preprocessing IC-Light conditioning image")?;

        // ---- 3. VAE-encode the conditioning → subject latents (1,4,H/8,W/8) ----
        let subject_latents = {
            let dist = self.vae.encode(&cond_px)?;
            (dist.sample()? * SD15_VAE_SCALE)?.to_dtype(self.dtype)?
        };

        // ---- 4. text embeddings, CFG batch [uncond, cond] ----
        let cond = self.encode_text(prompt)?;
        let text_embeds = if do_cfg {
            let uncond = self.encode_text(negative)?;
            Tensor::cat(&[&uncond, &cond], 0)?
        } else {
            cond
        };

        // ---- 5. init noise + scheduler ----
        let prepared = crate::pipelines::seeds::prepare_seed(seed, &self.device);
        if let Err(e) = self.device.set_seed(prepared) {
            tracing::debug!(target: "plakat", "set_seed not supported ({e}); using global RNG");
        }
        let mut scheduler = crate::pipelines::scheduler::build(
            crate::pipelines::scheduler::SchedulerKind::Default,
            &self.cfg,
            steps,
        )?;
        let timesteps = scheduler.timesteps().to_vec();
        let (latent_h, latent_w) = (h / 8, w / 8);
        let mut latents = Tensor::randn(0f32, 1f32, (1, 4, latent_h, latent_w), &self.device)?
            .to_dtype(self.dtype)?;
        latents = (latents * scheduler.init_noise_sigma())?;

        // ---- 6. denoise ----
        let bar = progress::step_bar(timesteps.len() as u64, "relight");
        for &t in timesteps.iter() {
            latents = self.denoise_step(
                &latents,
                t,
                &text_embeds,
                &subject_latents,
                &mut scheduler,
                guidance,
                do_cfg,
            )?;
            bar.inc(1);
            bar.set_message(format!("t={t} seed={seed}"));
        }
        bar.finish_and_clear();

        // ---- 7. decode ----
        let image = self.vae.decode(&(&latents / SD15_VAE_SCALE)?)?;
        let image = ((image / 2.0)? + 0.5)?.clamp(0f32, 1f32)?;
        let image = (image * 255.0)?.to_dtype(DType::U8)?.i(0)?.permute((1, 2, 0))?;
        let (oh, ow, _) = image.dims3()?;
        let buf = image.flatten_all()?.to_vec1::<u8>()?;
        Ok((buf, ow as u32, oh as u32))
    }

    /// One CFG denoise step with the 8-channel concat. `subject_latents`
    /// is `(1, 4, h/8, w/8)`; it's tiled across the CFG batch and
    /// concatenated on the channel dim onto the scaled noisy latent.
    fn denoise_step(
        &self,
        latents: &Tensor,
        timestep: usize,
        text_embeds: &Tensor,
        subject_latents: &Tensor,
        scheduler: &mut Box<dyn candle_transformers::models::stable_diffusion::schedulers::Scheduler>,
        guidance: f64,
        do_cfg: bool,
    ) -> Result<Tensor> {
        let latent_in = if do_cfg {
            Tensor::cat(&[latents, latents], 0)?
        } else {
            latents.clone()
        };
        let latent_in = scheduler.scale_model_input(latent_in, timestep)?;
        // Tile the subject latent across the (CFG) batch and concat on
        // the channel dim → 8 channels.
        let subj = if do_cfg {
            Tensor::cat(&[subject_latents, subject_latents], 0)?
        } else {
            subject_latents.clone()
        };
        let latent_in = Tensor::cat(&[&latent_in, &subj], 1)?;
        let noise_pred =
            self.unet
                .forward(&latent_in, timestep as f64, text_embeds, None, None)?;
        let noise_pred = if do_cfg {
            let chunks = noise_pred.chunk(2, 0)?;
            let uncond = &chunks[0];
            let text = &chunks[1];
            (uncond + ((text - uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        Ok(scheduler.step(&noise_pred, timestep, latents)?)
    }

    /// Matte the subject off its background and composite it over a
    /// neutral grey (0.5) field, returning a temp PNG path at
    /// `width × height`. This is the FC conditioning image.
    fn subject_on_grey(&self, subject_png: &Path, width: u32, height: u32, backdrop: Backdrop) -> Result<std::path::PathBuf> {
        // U2Net cut-out → RGBA temp.
        let cut = tempfile::Builder::new()
            .prefix("plakat-iclight-cut-")
            .suffix(".png")
            .tempfile()?;
        let cut_path = cut.path().to_path_buf();
        // matting::cutout is async; we're inside a sync `relight` called
        // from the async CLI on a multi-threaded runtime. Bridge with the
        // repo's established block_in_place + Handle::block_on pattern.
        let device = self.device.clone();
        let in_path = subject_png.to_path_buf();
        let cut_for_async = cut_path.clone();
        let handle = tokio::runtime::Handle::try_current().map_err(|e| {
            anyhow!(
                "IC-Light: no tokio runtime in scope (relight must run under \
                 a multi-threaded runtime). Underlying error: {e}"
            )
        })?;
        tokio::task::block_in_place(|| {
            handle.block_on(crate::pipelines::matting::cutout(
                &in_path,
                &cut_for_async,
                false,
                &device,
            ))
        })
        .context("matting subject for IC-Light conditioning")?;

        // Composite RGBA over grey (128), resize to target.
        let rgba = image::open(&cut_path)
            .context("reading IC-Light matte")?
            .to_rgba8();
        let mut composited = image::RgbImage::new(rgba.width(), rgba.height());
        let (rw, rh) = (rgba.width().max(2), rgba.height().max(2));
        for (x, y, p) in rgba.enumerate_pixels() {
            let a = p.0[3] as f32 / 255.0;
            // RELIGHT-1 P2: composite over the directional backdrop (a spatial light cue) instead of flat grey.
            let bg_v = backdrop_value(backdrop, x as f32 / (rw - 1) as f32, y as f32 / (rh - 1) as f32);
            let mix = |fg: u8| -> u8 {
                ((fg as f32 * a) + bg_v * (1.0 - a)).round().clamp(0.0, 255.0) as u8
            };
            composited.put_pixel(x, y, image::Rgb([mix(p.0[0]), mix(p.0[1]), mix(p.0[2])]));
        }
        let resized = image::imageops::resize(
            &composited,
            width,
            height,
            image::imageops::FilterType::Triangle,
        );
        let grey_tmp = tempfile::Builder::new()
            .prefix("plakat-iclight-grey-")
            .suffix(".png")
            .tempfile()?;
        let grey_path = grey_tmp.path().to_path_buf();
        resized.save(&grey_path).context("saving IC-Light conditioning image")?;
        // Persist the tempfile (keep_path returns the path + drops the
        // delete-on-drop guard) so the file survives until we read it
        // back via sd_image_tensor.
        let (_file, kept) = grey_tmp.keep().context("persisting IC-Light conditioning temp")?;
        Ok(kept)
    }
}

// =====================================================================
// Tokenisation helper — mirrors portrait::tokenize_padded / t2i.
// =====================================================================

fn tokenize_padded(
    tokenizer: &Tokenizer,
    cfg: &sdclip::Config,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let pad_id: u32 = match &cfg.pad_with {
        Some(s) => tokenizer
            .token_to_id(s)
            .ok_or_else(|| anyhow!("tokenizer missing pad token {s:?}"))?,
        None => tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow!("tokenizer missing <|endoftext|>"))?,
    };
    let mut ids = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    ids.resize(cfg.max_position_embeddings, pad_id);
    Ok(Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?)
}

// ── Named lighting presets + directional backdrop (RFC RELIGHT-1) ────────────────────────────────────

/// The conditioning **backdrop** the subject is composited on — IC-Light reads it as a spatial light cue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backdrop {
    /// Flat neutral grey (the original behaviour).
    Flat,
    /// Dark centre, bright edges — a rim/backlight cue.
    Rim,
    /// A linear gradient with the light coming from `angle` degrees (0 = left, 90 = top, 180 = right,
    /// 270 = bottom, CCW).
    Directional(f32),
}

/// A named lighting preset: a curated prompt + negative + the backdrop direction.
#[derive(Debug, Clone, Copy)]
pub struct LightPreset {
    pub name: &'static str,
    pub prompt: &'static str,
    pub negative: &'static str,
    pub backdrop: Backdrop,
}

/// The curated lighting-preset table (RFC RELIGHT-1 P1).
pub fn light_presets() -> &'static [LightPreset] {
    &[
        LightPreset { name: "key-left", prompt: "dramatic key light from the left, soft falloff to the right, cinematic portrait lighting", negative: "flat lighting, evenly lit, overexposed", backdrop: Backdrop::Directional(0.0) },
        LightPreset { name: "key-right", prompt: "dramatic key light from the right, soft falloff to the left, cinematic portrait lighting", negative: "flat lighting, evenly lit, overexposed", backdrop: Backdrop::Directional(180.0) },
        LightPreset { name: "top", prompt: "soft light from above, gentle overhead illumination, natural shadows", negative: "harsh uplighting, flat", backdrop: Backdrop::Directional(90.0) },
        LightPreset { name: "rim", prompt: "rim light, backlit subject with a glowing edge, dark background, moody", negative: "flat frontal light, washed out", backdrop: Backdrop::Rim },
        LightPreset { name: "softbox", prompt: "soft even studio softbox lighting, clean product shot, gentle shadows", negative: "harsh shadows, hard light, colored light", backdrop: Backdrop::Flat },
        LightPreset { name: "golden-hour", prompt: "warm golden hour sunlight from a low angle, long soft shadows, amber glow", negative: "cool light, blue tint, midday sun", backdrop: Backdrop::Directional(20.0) },
        LightPreset { name: "sunset", prompt: "warm orange sunset light from the side, dramatic warm glow, dusk atmosphere", negative: "cool light, flat, midday", backdrop: Backdrop::Directional(160.0) },
        LightPreset { name: "moonlight", prompt: "cool blue moonlight, soft night illumination, low key, calm", negative: "warm light, bright daylight, orange", backdrop: Backdrop::Directional(75.0) },
        LightPreset { name: "candlelight", prompt: "warm flickering candlelight, low key, intimate orange glow from below", negative: "cool light, bright, daylight, overhead", backdrop: Backdrop::Directional(280.0) },
        LightPreset { name: "neon", prompt: "colorful neon lighting, cyberpunk city glow, magenta and cyan rim light", negative: "natural daylight, warm sunlight, flat", backdrop: Backdrop::Rim },
        LightPreset { name: "overcast", prompt: "soft diffuse overcast daylight, even and shadowless, gentle cool tone", negative: "hard shadows, direct sun, colored light", backdrop: Backdrop::Flat },
    ]
}

/// Look up a preset by name (case-insensitive).
pub fn light_preset(name: &str) -> Option<&'static LightPreset> {
    let n = name.trim().to_ascii_lowercase();
    light_presets().iter().find(|p| p.name == n)
}

/// The backdrop grey level (0..255) at normalised pixel `(fx, fy)` in `0..1`. `Flat` → 128; `Directional`
/// brightens toward the light; `Rim` darkens the centre and brightens the edges. Pure.
pub fn backdrop_value(bg: Backdrop, fx: f32, fy: f32) -> f32 {
    const BASE: f32 = 128.0;
    const AMP: f32 = 74.0;
    let v = match bg {
        Backdrop::Flat => BASE,
        Backdrop::Rim => {
            let d = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt() / std::f32::consts::FRAC_1_SQRT_2; // 0 centre → 1 corner
            BASE - AMP * 0.5 + AMP * d
        }
        Backdrop::Directional(deg) => {
            let r = deg.to_radians();
            // light unit vector = (-cos, -sin): angle 0 lights from the left, 90 from the top.
            BASE + AMP * (-(fx - 0.5) * r.cos() - (fy - 0.5) * r.sin())
        }
    };
    v.clamp(36.0, 224.0)
}

#[cfg(test)]
mod relight_preset_tests {
    use super::*;

    #[test]
    fn presets_resolve_and_carry_a_backdrop() {
        assert!(light_preset("key-left").is_some());
        assert!(light_preset("KEY-LEFT").is_some(), "case-insensitive");
        assert!(light_preset("nope").is_none());
        assert_eq!(light_preset("softbox").unwrap().backdrop, Backdrop::Flat);
        assert!(matches!(light_preset("key-left").unwrap().backdrop, Backdrop::Directional(_)));
    }

    #[test]
    fn backdrop_directional_brightens_toward_the_light() {
        // Light from the left (0°): the left edge is brighter than the right.
        let left = backdrop_value(Backdrop::Directional(0.0), 0.0, 0.5);
        let right = backdrop_value(Backdrop::Directional(0.0), 1.0, 0.5);
        assert!(left > right + 40.0, "left {left} brighter than right {right}");
        // Light from the top (90°): top brighter than bottom.
        assert!(backdrop_value(Backdrop::Directional(90.0), 0.5, 0.0) > backdrop_value(Backdrop::Directional(90.0), 0.5, 1.0));
        // Flat is uniform 128; rim darkens the centre vs a corner.
        assert_eq!(backdrop_value(Backdrop::Flat, 0.3, 0.7), 128.0);
        assert!(backdrop_value(Backdrop::Rim, 0.5, 0.5) < backdrop_value(Backdrop::Rim, 0.0, 0.0));
    }
}
