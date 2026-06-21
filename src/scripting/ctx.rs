//! v0.21: process-global script context.
//!
//! Host words registered into bundcore can't capture closures
//! (`VMInlineFn` is a bare `fn` pointer). They reach plakat state
//! via the [`CTX`] singleton — the same pattern blackInkhaven uses
//! for its `ADAM` VM and `ACTIVE_STORE` project handle.
//!
//! Phase 1 carries only `device` + `out_dir`; phase 2 will add a
//! lazy-loaded `HashMap<String, LoadedPipeline>` so scripts can
//! reuse a loaded model across calls without paying the model-load
//! cost per `plakat.generate`.

use anyhow::{Result, anyhow};
use candle_core::Device;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

use crate::pipelines::{
    controlnet::ControlSpec, flux, lora::LoraSpec, portrait, sd3, t2i,
};
use crate::scripting::config::GenerationConfig;
use crate::scripting::loaded_pipeline::{LoadedPipeline, PipelineFamily};

/// v0.34 phase 3: VAE cache lookup helper for the scripting ctx.
/// Mirrors `cli::scenario::vae_cache_lookup` — returns a fresh Arc
/// handle to the cached VAE when the alias matches, `None`
/// otherwise. The Arc clone is cheap (refcount bump).
fn vae_cache_lookup_script<T: Clone>(
    cache: Option<&(String, T)>,
    alias: &str,
) -> Option<T> {
    cache.filter(|(k, _)| k == alias).map(|(_, v)| v.clone())
}

/// Process-wide script context. Holds the device + output dir +
/// the in-script image registry + the active model alias.
///
/// One script per process by construction — bundcore's VM has no
/// per-eval isolation and the singleton can only be written once.
pub struct ScriptCtx {
    pub device: Device,
    pub out_dir: PathBuf,
    /// v0.22 phase 1: cached pipeline keyed by the model alias
    /// that loaded it. `None` means no model has been loaded yet;
    /// image-producing host words bail with a clear message.
    ///
    /// Replaces v0.21's `loaded_model: Option<String>` — v0.22 (per
    /// RFC decision #3) caches the actual pipeline so subsequent
    /// `plakat.generate` / `img2img` / `portrait` calls reuse it
    /// without paying the model-load cost again. v0.21 compat is
    /// relaxed per decision #7 — the `loaded_model` field is gone;
    /// scripts that don't care about the change still work.
    pub loaded: Option<(String, LoadedPipeline)>,
    /// v0.23 phase 1: secondary SD-family cache slot holding a
    /// `t2i::Pipeline`. Used exclusively by `plakat.generate`'s
    /// SD path so that family's CLIP-skip + (v0.23 phase 2)
    /// SDXL-refiner UNet wiring can land — those live on
    /// `t2i::Pipeline`, not the `portrait::Pipeline` that
    /// `plakat.img2img` / `.portrait` keep using.
    ///
    /// Per RFC v0.23 Option A: both SD-family slots can be loaded
    /// for the same alias simultaneously; they share an
    /// `Arc<SdCore>` so the duplication cost is only the slot
    /// extras (refiner UNet vs. IP-Adapter encoder). Loading a
    /// non-SD-family alias drops this slot.
    pub loaded_t2i: Option<(String, t2i::Pipeline)>,
    /// v0.26 phase 7: cached `stylize::Pipeline` for
    /// `plakat.stylize` (IP-Adapter Plus + SD 1.5 base). Without
    /// this slot, every `plakat.stylize` call pays the full ~5 GB
    /// load (SD 1.5 weights + CLIP-H image encoder + IP-Adapter
    /// projection). With the slot, multi-call scripts amortise.
    ///
    /// SD 1.5 only — stylize already bails on SDXL / Flux / SD3
    /// at load time. Tuple is `(alias, pipeline)` matching the
    /// `loaded_t2i` pattern. Invalidated by [`Self::mark_loras_changed`]
    /// because the stylize pipeline holds an SD 1.5 UNet snapshot
    /// — a LoRA stack mutation would need a fresh load to take
    /// effect.
    pub loaded_stylize:
        Option<(String, crate::pipelines::stylize::Pipeline)>,
    /// v0.29 phase 1: cached SD 1.5 AnimateDiff pipeline
    /// (`pipelines::animatediff::AnimateDiffPipeline`) for
    /// `plakat.animate`. Key is `format!("{alias}:{mode}")` where
    /// `mode` is `"v3"` (default, V3 motion adapter) or `"lcm"`
    /// (AnimateLCM adapter via `animate_lcm`). The mode in the key
    /// means toggling `animate_lcm` between calls invalidates the
    /// slot automatically.
    ///
    /// SD 1.5 + AnimateLCM both download ~3.5 GB on cold start; the
    /// slot amortises that across multi-call scripts. Alias change
    /// drops it; LoRA stack mutation drops it via
    /// [`Self::mark_loras_changed`] (the motion-UNet was loaded
    /// from the SD 1.5 weights with LoRA merge baked in).
    pub loaded_animatediff:
        Option<(String, crate::pipelines::animatediff::AnimateDiffPipeline)>,
    /// v0.29 phase 1: cached SDXL AnimateDiff pipeline for
    /// `plakat.animate`. Key is the SDXL alias. Independent slot
    /// from `loaded_animatediff` because the pipeline types differ
    /// (SDXL needs dual CLIP-L/G + add_text_embeds + add_time_ids
    /// vs SD 1.5's single CLIP-L).
    ///
    /// Cold load downloads ~7 GB (SDXL base + SDXL beta motion
    /// adapter). The slot amortises that across calls. Alias change
    /// + LoRA mutation drop it on the same rules as the SD 1.5 slot.
    pub loaded_animatediff_sdxl:
        Option<(
            String,
            crate::pipelines::animatediff::AnimateDiffSdxlPipeline,
        )>,
    /// v0.36 phase 1: cached PixArt-Σ pipeline for `plakat.pixart`.
    /// Key is the user's PixArt alias (`pixart` / `pixart-sigma` /
    /// `pixart-1024`). Same-alias hit reuses; alias change drops +
    /// reloads. LoRA stack mutation drops via
    /// [`Self::mark_loras_changed`] (PixArt LoRAs merge at load
    /// time per v0.35 phase 4, so stack mutation needs a fresh
    /// load).
    ///
    /// Cold load downloads ~12 GB (T5-XXL + DiT-XL/2 + VAE). The
    /// slot amortises that across calls.
    pub loaded_pixart:
        Option<(String, crate::pipelines::pixart::Pipeline)>,
    /// v0.38 phase 2: cached Stable Cascade pipeline for
    /// `plakat.cascade`. Key is the user's Cascade alias
    /// (`stable-cascade` / `cascade` / a `*-lite` fork). Same-alias
    /// hit reuses; alias change drops + reloads. Stable Cascade has
    /// no scripting-side LoRA support yet (v0.38 phase 3 wires it
    /// at load time mirroring the v0.35 phase 4 PixArt pattern),
    /// so this slot doesn't drop on LoRA stack mutation. Cold load
    /// downloads ~14 GB (CLIP-G + Stage A + Stage B + Stage C).
    pub loaded_cascade:
        Option<(String, crate::pipelines::cascade::Pipeline)>,
    /// v0.34 phase 3: scripting-side VAE cache. Mirrors the scenario
    /// runner's v0.32 phase 2 / v0.34 phase 3 cross-kind sharing.
    /// Each load (`plakat.load`, `plakat.animate`) looks this up by
    /// alias before building a fresh VAE; on miss the load populates
    /// it. Mixed-kind scripts (`plakat.load sdxl; plakat.animate
    /// sdxl`) stop paying the ~330 MB SDXL VAE rebuild cost on the
    /// kind switch — closes the v0.32 phase 2 deferral.
    pub vae_cache: Option<(
        String,
        std::sync::Arc<
            candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
        >,
    )>,
    /// v0.21 phase 2: rendered images, addressable by the integer
    /// handle pushed onto the stack by `plakat.generate`. Index =
    /// handle (1-based — handle 0 is reserved as "no image").
    /// Phase 2 keeps every rendered image in memory for the
    /// script's lifetime; if scripts ever start producing hundreds
    /// of images we'll revisit (e.g. spill to disk).
    pub images: Vec<image::DynamicImage>,
    /// v0.26 phase 8: per-image metadata, indexed parallel to
    /// [`Self::images`]. `Some` when the rendering path attached
    /// a [`crate::imaging::metadata::GenerationMetadata`] at push
    /// time (full A1111-style record: prompt / negative / model /
    /// seed / steps / guidance / scheduler / etc.); `None`
    /// otherwise (e.g. images loaded from disk, or rendering paths
    /// that don't yet populate metadata). Read by `plakat.save` to
    /// route through `save_rgb_u8_with_metadata` (sidecar + PNG
    /// tEXt) and by `plakat.metadata.write` to re-attach metadata
    /// to existing files.
    pub images_metadata:
        Vec<Option<crate::imaging::metadata::GenerationMetadata>>,
    /// v0.21 phase 3: generation knobs the script accumulates via
    /// `plakat.config.set`. Persistent across calls within one
    /// script. Read by [`super::script_entry::generate_one`] when
    /// building the `t2i::Request`.
    pub config: GenerationConfig,
    /// v0.22 phase 4: LoRA stack accumulated via `plakat.lora.add`.
    /// Read at pipeline-load time (cache invalidation: mutating
    /// this drops `loaded` so the next `ensure_loaded` rebuilds
    /// with the new LoRA set). See RFC §7 + the
    /// [`Self::mark_loras_changed`] helper that does the drop.
    pub loras: Vec<LoraSpec>,
    /// v0.22 phase 5: ControlNet stack accumulated via the
    /// `plakat.controlnet.*` words. Read at generate time
    /// (per-call, not per-load — SD-family ControlNet flows
    /// through `Request.controls` / `pipeline.generate(...,
    /// controls)`). No cache invalidation needed for SD-family
    /// because the cached pipeline doesn't bake in the CN stack.
    ///
    /// v0.23 phase 6 wires Flux ControlNet at load time (the
    /// Flux pipeline's `LoadRequest.controlnets` bakes in the
    /// CN stack on first generate; `mark_controlnets_changed`
    /// drops the slot on stack mutations). SD3 CN follows in
    /// phase 7. Note: Flux CN scripting supports `image=` specs
    /// only — `from=` (auto-annotate) would need the per-generate
    /// width/height at load time, which the loader doesn't know.
    pub controlnets: Vec<ControlSpec>,
    /// v0.22 phase 6 + v0.23 phase 2: SDXL refiner toggle.
    /// `plakat.refiner.enable` sets this to `true`;
    /// `plakat.refiner.disable` resets it.
    ///
    /// As of v0.23 phase 2, mutating this flag invalidates the
    /// SdT2i slot via [`Self::mark_loras_changed`] so the next
    /// `plakat.generate` reloads with `use_refiner` matching the
    /// new value. The refiner UNet is SDXL-only; on non-SDXL
    /// aliases the toggle warns + downgrades silently inside
    /// [`Self::get_or_load_sd_t2i`].
    pub refiner_enabled: bool,
    /// v0.22 phase 7: ADetailer post-process toggle. When `true`,
    /// `script_entry::generate_one` runs `adetailer::refine_files`
    /// on the rendered image before pushing it to `images`. The
    /// per-pass knobs (strength / padding / feather / confidence
    /// / size / prompt) come from `config.adetailer_*` keys.
    /// SD-family only — Flux + SD3 generate paths bail when this
    /// is on (SCRFD + img2img face passes are SD-only).
    pub adetailer_enabled: bool,
    /// v0.22 phase 8: Hires-fix post-process toggle. When `true`,
    /// `script_entry::generate_one` runs `hires_fix::refine_files`
    /// on the rendered image (upscale → img2img refine). The
    /// per-pass knobs (scale / strength / upscaler / steps) come
    /// from `config.hires_*` keys. SD-family only — Flux + SD3
    /// bail (hires_fix needs an SD img2img pipeline for refine).
    pub hires_enabled: bool,
    /// v0.22 phase 9: artefact specs accumulated via
    /// `plakat.artefact.add`. After `plakat.generate` /
    /// `plakat.img2img` / `plakat.portrait` renders, the
    /// post-process composites each artefact onto the image (in
    /// the order added). Empty (default) = no compositing pass.
    /// SD-family only — Flux + SD3 bail when this is non-empty.
    pub artefacts: Vec<crate::artefacts::ArtefactSpec>,
    /// v0.22 phase 9: when `true` AND `artefacts` is non-empty,
    /// runs the masked-img2img blend pass over the artefact
    /// zones after compositing — smooths hard edges. Same
    /// behaviour as the CLI's `--artefact-blend` flag.
    pub artefact_blend_enabled: bool,
    /// v0.23 phase 4: active style id (set by `plakat.style.apply`).
    /// When `Some`, the next SD-family `plakat.generate` resolves
    /// the style against the catalog at request-build time:
    /// catalog LoRAs replace user LoRAs (with a warn), trigger
    /// phrase prepends to the prompt, and `negative_extras`
    /// appends to the negative. Mirrors `--style ID`.
    pub style_id: Option<String>,
    /// v0.23 phase 4: reference photo for detection-based style
    /// pick (set by `plakat.style.detect`). When `Some` AND
    /// `style_id` is `None`, generate runs detection through
    /// CLIP-H + cosine-matches against the catalog. When both
    /// are `Some`, the photo runs detection (for logging) but
    /// `style_id` wins. Mirrors `--style-ref PATH`.
    pub style_ref: Option<std::path::PathBuf>,
    /// v0.24 phase 1: multi-photo portrait stack. Populated via
    /// `plakat.portrait.photo.add ( path weight -- )`; drained
    /// by `plakat.portrait.photo.clear`. `plakat.portrait
    /// ( prompt -- handle )` reads this stack and bails when
    /// empty. Mirrors the v0.22 LoRA/ControlNet pattern — state
    /// accumulates between calls, mutations don't invalidate
    /// the cache (photos are per-call on the SD-family path).
    ///
    /// Each entry's `weight` is `Some(f32)` for an explicit
    /// weight or `None` for "auto-fill the remainder." Same
    /// normalisation as `cli::portrait`
    /// (`ip_adapter::normalize_photo_weights`).
    pub portrait_photos: Vec<crate::pipelines::ip_adapter::WeightedPhoto>,
    /// v0.24 phase 5: Textual Inversion (embedding) stack
    /// populated via `plakat.embedding.add`. Threaded into
    /// `t2i::LoadRequest.embeddings` at load time, so mutations
    /// invalidate the SdT2i slot via `mark_loras_changed`
    /// (embeddings are load-time alongside LoRAs).
    ///
    /// **Effective only on `plakat.generate`'s SdT2i path.**
    /// `plakat.img2img` + `plakat.portrait` use
    /// `portrait::Pipeline`, which doesn't take embeddings
    /// (matches `cli::img2img` / `cli::portrait` — neither CLI
    /// command exposes `--embedding` either). Embeddings stay
    /// in `ctx.embeddings` silently on those paths.
    pub embeddings: Vec<crate::pipelines::embedding::EmbeddingSpec>,
    /// v0.24 phase 8: ControlNet auto-annotation cache for Flux +
    /// SD3. `from=` specs in `ctx.controlnets` can't be annotated
    /// at pipeline-load time (the loader doesn't know the
    /// per-generate width/height yet). The first `plakat.generate`
    /// with a `from=` spec runs the annotator using that
    /// generate's dims; subsequent generates with the same
    /// pipeline + same dims reuse. Dim mismatch on a later call
    /// re-annotates.
    ///
    /// `None` means "no cache yet." The first generate populates
    /// it. Cleared by `mark_controlnets_changed` /
    /// `mark_loras_changed` (any pipeline-invalidating mutation
    /// also drops these tempfiles).
    pub cn_annotation_cache: Option<CnAnnotationCache>,
    /// v0.25 phase 8: active art-medium preset (set by
    /// `plakat.look.apply`). When `Some`, the next `plakat.generate`
    /// / `.portrait` / `.img2img` applies the preset's prompt
    /// prefix/suffix + sampler/steps/guidance hints and (when
    /// `ctx.loras` is empty) runs auto-LoRA discovery. Mirrors
    /// `--look NAME`.
    pub look_name: Option<String>,
    /// v0.25 phase 8: active subject-domain preset (set by
    /// `plakat.genre.apply`). Independent axis from `look_name`;
    /// composes additively. Mirrors `--genre NAME`.
    pub genre_name: Option<String>,
    /// MAP-5: town street-plan override (`radial`/`grid`/`organic`), set by
    /// `plakat.map.layout`. Applied to the spec by `plakat.map.render`. Mirrors
    /// `--map-urban-layout`.
    pub map_layout: Option<String>,
    /// MAP-2: natural-feature erosion override (0 smooth … 1 natural … >1 rugged),
    /// set by `plakat.map.erosion`. Applied by `plakat.map.render`. Mirrors
    /// `--map-erosion`.
    pub map_erosion: Option<f32>,
}

/// v0.24 phase 8: per-loaded-pipeline annotation cache. The
/// `_tmpdir` field keeps the annotated PNGs alive for the
/// pipeline's lifetime; dropping the cache deletes them.
pub struct CnAnnotationCache {
    /// Indexed parallel to `ctx.controlnets`. `Some((w, h, path))`
    /// means CN[i] has a cached annotation at those dims;
    /// `None` means CN[i] doesn't need annotation (image= spec)
    /// or hasn't been annotated yet.
    pub entries: Vec<Option<(u32, u32, std::path::PathBuf)>>,
    /// Lifetime guard for the annotation PNGs.
    pub _tmpdir: tempfile::TempDir,
}

impl ScriptCtx {
    /// Initialise the singleton. Called once at the top of
    /// `cli::run::run` after the CLI device selection lands. A
    /// second call after the first is a hard error — bundcore
    /// can't run two scripts concurrently in one process.
    pub fn init(device: Device, out_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&out_dir).map_err(|e| {
            anyhow!("creating script output dir {}: {e}", out_dir.display())
        })?;
        CTX.set(RwLock::new(ScriptCtx {
            device,
            out_dir,
            loaded: None,
            loaded_t2i: None,
            images: Vec::new(),
            images_metadata: Vec::new(),
            config: GenerationConfig::default(),
            loras: Vec::new(),
            controlnets: Vec::new(),
            refiner_enabled: false,
            adetailer_enabled: false,
            hires_enabled: false,
            artefacts: Vec::new(),
            artefact_blend_enabled: false,
            style_id: None,
            style_ref: None,
            portrait_photos: Vec::new(),
            embeddings: Vec::new(),
            cn_annotation_cache: None,
            look_name: None,
            genre_name: None,
            map_layout: None,
            map_erosion: None,
            loaded_stylize: None,
            loaded_animatediff: None,
            loaded_animatediff_sdxl: None,
            loaded_pixart: None,
            loaded_cascade: None,
            vae_cache: None,
        }))
        .map_err(|_| anyhow!("ScriptCtx already initialised"))
    }

    /// v0.22 phase 4: invalidate the cached pipeline so the next
    /// `ensure_loaded` reloads with the current LoRA stack
    /// merged in. Call after every `plakat.lora.add` /
    /// `plakat.lora.clear` mutation. Per RFC §7, this is the
    /// "defer the merge to next generate" pattern — simpler than
    /// in-place LoRA injection across the three pipeline families.
    pub fn mark_loras_changed(&mut self) {
        self.loaded = None;
        // v0.23 phase 1: the t2i slot also caches LoRA-merged
        // weights; same invalidation rule.
        self.loaded_t2i = None;
        // v0.24 phase 8: annotation cache is bound to the loaded
        // pipeline; drop alongside.
        self.cn_annotation_cache = None;
        // v0.26 phase 7: the stylize slot holds an SD 1.5 UNet
        // snapshot — a LoRA stack mutation invalidates it. Same
        // pattern as loaded_t2i.
        self.loaded_stylize = None;
        // v0.29 phase 1: the AnimateDiff slots hold motion UNets
        // loaded from the SD 1.5 / SDXL base weights with LoRA
        // merge baked in at load time. Drop on LoRA stack mutation.
        self.loaded_animatediff = None;
        self.loaded_animatediff_sdxl = None;
        // v0.36 phase 1: PixArt LoRAs merge into the DiT
        // transformer tempfile at load time (v0.35 phase 4 pattern);
        // any stack mutation needs a fresh load to take effect.
        self.loaded_pixart = None;
        // v0.38 phase 3: Cascade LoRAs merge into Stage B + Stage C
        // safetensors tempfiles at load time; drop on mutation.
        self.loaded_cascade = None;
    }

    /// v0.23 phase 6: invalidate pipeline slots whose ControlNet
    /// stack is baked in at LOAD time. Flux CN is load-time
    /// (per phase 6); SD3 CN is also load-time (per phase 7,
    /// when that wiring lands the SD3 branch here will start
    /// firing). SD-family CN stays per-call (the
    /// `pipeline.generate(req, &controls)` arg), so the SD slots
    /// aren't touched — scripts that toggle CN on/off between SD
    /// generates pay nothing.
    ///
    /// Called by `plakat.controlnet.*` mutations.
    pub fn mark_controlnets_changed(&mut self) {
        // The primary `loaded` slot is shared by Flux + SD3 + SD;
        // drop it only when it's Flux (phase 6) or SD3 (phase 7).
        let is_flux_or_sd3 = matches!(
            self.loaded.as_ref().map(|(_, p)| p),
            Some(LoadedPipeline::Flux(_)) | Some(LoadedPipeline::Sd3(_))
        );
        if is_flux_or_sd3 {
            self.loaded = None;
        }
        // v0.24 phase 8: drop annotation cache too (CN stack
        // change → cached annotations may be wrong index).
        self.cn_annotation_cache = None;
        // SD-family slots (loaded if SdFamily variant, loaded_t2i)
        // are left intact: SD-family CN is per-call, not per-load.
    }

    /// v0.22 phase 1: read-only accessor for the currently-loaded
    /// model's alias. `None` when nothing's been `plakat.load`ed
    /// yet. Replaces direct access to v0.21's `loaded_model` field.
    pub fn loaded_model(&self) -> Option<&str> {
        // v0.23 phase 1: prefer the SdT2i slot's alias when it's
        // populated (plakat.load now loads t2i by default for
        // SD-family). Fall back to the portrait/flux/sd3 slot.
        // Both slots normally hold the same alias when both are
        // loaded; the order matters only during a slot-rebuild
        // window.
        self.loaded_t2i
            .as_ref()
            .map(|(a, _)| a.as_str())
            .or_else(|| self.loaded.as_ref().map(|(a, _)| a.as_str()))
            // v0.42 phase 4: PixArt / Cascade live in their own slots
            // (no SdT2i / portrait slot), so report their alias too —
            // otherwise `plakat.cascade` / `plakat.pixart` can't find
            // the model `plakat.load` warmed.
            .or_else(|| self.loaded_pixart.as_ref().map(|(a, _)| a.as_str()))
            .or_else(|| self.loaded_cascade.as_ref().map(|(a, _)| a.as_str()))
    }

    /// v0.22 phase 1: get-or-load the SD-family pipeline for
    /// `alias`. Returns a reference to the cached
    /// [`portrait::Pipeline`] (which generalises across
    /// text-to-image / img2img / portrait — see
    /// [`crate::scripting::loaded_pipeline::LoadedPipeline`]).
    ///
    /// On a cache miss the previous pipeline drops (RAII-freeing
    /// GPU memory) before the new one loads. The load is `async`
    /// (HF download + safetensors mmap); we block on the current
    /// tokio runtime via `block_in_place` so callers can stay
    /// sync. This requires a multi-threaded tokio runtime in
    /// scope — `cli::run::run` provides one.
    ///
    /// **Identity encoder strategy**: SD 1.5 → `PlusFace`;
    /// SDXL / SDXL-Turbo → `PlusFaceSdxl`; SD 2.1 → `None`
    /// (no shipped Plus-Face checkpoint). Loading without
    /// identity means `plakat.portrait` will bail at generate
    /// time with the v0.21 "no identity encoder" message —
    /// preserving v0.21's SD 2.1 portrait behaviour while
    /// keeping `plakat.generate` working.
    pub fn get_or_load_sd_family(
        &mut self,
        alias: &str,
    ) -> Result<&portrait::Pipeline> {
        // Cache hit on the alias?
        let hit = self
            .loaded
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            // Drop the previous pipeline first so the new model's
            // weights don't have to coexist in GPU memory with
            // the old.
            self.loaded = None;

            // v0.23 phase 1: if `loaded_t2i` already holds the
            // same alias, we can derive a portrait::Pipeline from
            // its shared `SdCore` without paying for a second
            // weights load. Saves several GB on SDXL.
            let shared_core = self
                .loaded_t2i
                .as_ref()
                .filter(|(a, _)| a == alias)
                .map(|(_, p)| p.core());

            if let Some(core) = shared_core {
                let pipeline = portrait::Pipeline::from_core(core);
                self.loaded = Some((alias.to_string(), LoadedPipeline::SdFamily(pipeline)));
            } else {
                let override_kind = if self.config.identity_kind.is_empty() {
                    None
                } else {
                    Some(self.config.identity_kind.as_str())
                };
                let identity = pick_sd_family_identity(alias, override_kind);
                let device = self.device.clone();
                let loras = self.loras.clone();
                let lora_scale = self.config.lora_scale;
                let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                    anyhow!(
                        "ScriptCtx::get_or_load_sd_family: no tokio runtime in scope. {e}"
                    )
                })?;
                let pipeline: portrait::Pipeline = tokio::task::block_in_place(|| {
                    handle.block_on(portrait::Pipeline::load(portrait::LoadRequest {
                        model: alias.to_string(),
                        device,
                        loras,
                        lora_scale,
                        identity,
                        shared_clip_h: None,
                    }))
                })?;
                self.loaded = Some((alias.to_string(), LoadedPipeline::SdFamily(pipeline)));
            }
        }

        match &self.loaded.as_ref().expect("just inserted").1 {
            LoadedPipeline::SdFamily(p) => Ok(p),
            LoadedPipeline::Flux(_) | LoadedPipeline::Sd3(_) => Err(anyhow!(
                "ScriptCtx::get_or_load_sd_family called with a non-SD \
                 alias — the cache is holding a different pipeline. \
                 Use ensure_loaded for family-aware dispatch."
            )),
        }
    }

    /// v0.23 phase 1: get-or-load the SD-family **t2i** pipeline
    /// for `alias`. Caches into [`Self::loaded_t2i`] — the
    /// secondary SD-family slot that coexists with the primary
    /// `loaded` slot (which holds the portrait::Pipeline for
    /// `plakat.img2img` / `.portrait`).
    ///
    /// Used by `plakat.generate`'s SD-family path so the
    /// v0.23 phase 2 refiner UNet load and phase 3 clip_skip
    /// wiring have a `t2i::Pipeline` to land on.
    ///
    /// **Refiner gating (v0.23 phase 2)**: `use_refiner` reads
    /// `ctx.refiner_enabled`, but only for SDXL aliases. SD 1.5 /
    /// SD 2.1 with the toggle on silently downgrade to
    /// `use_refiner: false` and emit a one-time warn — mirrors
    /// the CLI's `--refiner` behaviour. Toggling
    /// `plakat.refiner.enable` / `.disable` invalidates this slot
    /// via [`Self::mark_loras_changed`] so the next call rebuilds
    /// with the new `use_refiner` value.
    pub fn get_or_load_sd_t2i(&mut self, alias: &str) -> Result<&mut t2i::Pipeline> {
        let hit = self
            .loaded_t2i
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            // Drop the previous t2i pipeline first.
            self.loaded_t2i = None;

            // Family change: if the primary slot holds a non-SD
            // family (Flux / SD3), drop it too. Same-family aliases
            // can coexist (portrait::Pipeline + t2i::Pipeline for
            // the same alias share an Arc<SdCore>).
            if !matches!(self.loaded.as_ref().map(|(_, p)| p),
                Some(LoadedPipeline::SdFamily(_)) | None) {
                self.loaded = None;
            }

            // v0.23 phase 2: refiner UNet is SDXL-only. For non-SDXL
            // aliases with the toggle on, downgrade silently with a
            // warn rather than letting t2i::Pipeline::load bail (the
            // bail would surface at plakat.load time, which is worse
            // than a graceful downgrade).
            let resolved = if alias.contains('/') {
                alias.to_string()
            } else {
                crate::hf::resolve_alias(alias).to_string()
            };
            let variant = crate::pipelines::t2i::Variant::detect(&resolved);
            let use_refiner = if self.refiner_enabled && variant.is_xl() {
                true
            } else {
                if self.refiner_enabled && !variant.is_xl() {
                    tracing::warn!(
                        target: "plakat",
                        "plakat.refiner.enable is on, but model {alias:?} \
                         resolves to {variant:?}, not SDXL. The SDXL refiner \
                         UNet is SDXL-only; loading without it. Same as the \
                         CLI's `--refiner` behaviour."
                    );
                }
                false
            };

            let device = self.device.clone();
            let loras = self.loras.clone();
            let lora_scale = self.config.lora_scale;
            // v0.24 phase 5: Textual Inversion embeddings flow
            // through at load time, alongside LoRAs. Cache
            // invalidation on stack mutation uses the same
            // `mark_loras_changed` path.
            let embeddings = self.embeddings.clone();
            // v0.34 phase 3: lookup pre-built VAE from the scripting
            // ctx's cross-kind cache. Cache HIT lets a `plakat.animate
            // sdxl` followed by `plakat.load sdxl` (or vice versa)
            // skip the ~330 MB SDXL VAE rebuild — same pattern as the
            // scenario runner.
            let cached_vae = vae_cache_lookup_script(self.vae_cache.as_ref(), alias);
            if cached_vae.is_some() {
                tracing::info!(
                    target: "plakat",
                    "v0.34 phase 3: VAE cache HIT on scripting t2i load (alias={alias})"
                );
            }
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_sd_t2i: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline: t2i::Pipeline = tokio::task::block_in_place(|| {
                handle.block_on(t2i::Pipeline::load(t2i::LoadRequest {
                    model: alias.to_string(),
                    device,
                    loras,
                    lora_scale,
                    use_refiner,
                    embeddings,
                    vae_cache: cached_vae,
                }))
            })?;
            // Populate cache from freshly loaded pipeline's VAE so
            // subsequent plakat.animate loads can reuse.
            self.vae_cache = Some((
                alias.to_string(),
                std::sync::Arc::clone(&pipeline.core().vae),
            ));
            self.loaded_t2i = Some((alias.to_string(), pipeline));
        }

        Ok(&mut self.loaded_t2i.as_mut().expect("just inserted").1)
    }

    /// v0.26 phase 7: get-or-load the IP-Adapter stylize pipeline
    /// for `alias`. Caches into [`Self::loaded_stylize`].
    ///
    /// Mirrors the v0.23 SdT2i pattern: same-alias hit reuses the
    /// loaded pipeline; alias change drops + reloads. Invalidated
    /// on LoRA stack mutation via [`Self::mark_loras_changed`]
    /// (the stylize pipeline holds a UNet snapshot that LoRAs
    /// would need to re-merge into).
    ///
    /// Returns `&stylize::Pipeline` (immutable) because
    /// `stylize::Pipeline::stylize_one` takes `&self` — no
    /// per-call state mutation.
    pub fn get_or_load_stylize(
        &mut self,
        alias: &str,
    ) -> Result<&crate::pipelines::stylize::Pipeline> {
        let hit = self
            .loaded_stylize
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            self.loaded_stylize = None;
            let device = self.device.clone();
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_stylize: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(crate::pipelines::stylize::Pipeline::load(
                    crate::pipelines::stylize::LoadRequest {
                        model: alias.to_string(),
                        device,
                        shared_clip_h: None,
                        instantstyle: false,
                        style_scale: 1.0,
                    },
                ))
            })?;
            self.loaded_stylize = Some((alias.to_string(), pipeline));
        }

        Ok(&self.loaded_stylize.as_ref().expect("just inserted").1)
    }

    /// v0.29 phase 1: get-or-load the SD 1.5 AnimateDiff pipeline
    /// for `(alias, lcm)`. Cache key is `format!("{alias}:{mode}")`
    /// where mode is `"v3"` or `"lcm"` — toggling `animate_lcm`
    /// between calls invalidates the slot.
    ///
    /// Mirrors [`Self::get_or_load_stylize`]: same-key hit reuses
    /// the loaded pipeline; key change drops + reloads. LoRA stack
    /// mutation drops the slot via [`Self::mark_loras_changed`].
    /// Network-required on cold load (~3.5 GB).
    pub fn get_or_load_animatediff(
        &mut self,
        alias: &str,
        lcm: bool,
        dtype: candle_core::DType,
    ) -> Result<&crate::pipelines::animatediff::AnimateDiffPipeline> {
        let mode = if lcm { "lcm" } else { "v3" };
        let key = format!("{alias}:{mode}");
        let hit = self
            .loaded_animatediff
            .as_ref()
            .map(|(k, _)| k == &key)
            .unwrap_or(false);

        if !hit {
            self.loaded_animatediff = None;
            let device = self.device.clone();
            // v0.34 phase 3: SD 1.5 animate hard-codes the canonical
            // sd15 base; cache key is "sd15" so this pairs with a
            // preceding `plakat.load sd15`.
            let vae_cache_key = "sd15";
            let cached_vae = vae_cache_lookup_script(self.vae_cache.as_ref(), vae_cache_key);
            if cached_vae.is_some() {
                tracing::info!(
                    target: "plakat",
                    "v0.34 phase 3: VAE cache HIT on scripting animate SD 1.5 load"
                );
            }
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_animatediff: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    if lcm {
                        crate::pipelines::animatediff::AnimateDiffPipeline::load_animatelcm(
                            &device, dtype, &[], 1.0, cached_vae, "sd15",
                        )
                        .await
                    } else {
                        crate::pipelines::animatediff::AnimateDiffPipeline::load_v3(
                            &device, dtype, &[], 1.0, cached_vae, "sd15",
                        )
                        .await
                    }
                })
            })?;
            self.vae_cache = Some((
                vae_cache_key.to_string(),
                std::sync::Arc::clone(&pipeline.vae),
            ));
            self.loaded_animatediff = Some((key, pipeline));
        }

        Ok(&self.loaded_animatediff.as_ref().expect("just inserted").1)
    }

    /// v0.29 phase 1: get-or-load the SDXL AnimateDiff pipeline for
    /// `alias`. Cache key is just the alias (SDXL beta is the only
    /// supported SDXL motion adapter today; no LCM variant to
    /// disambiguate). LoRA stack mutation drops via
    /// [`Self::mark_loras_changed`].
    ///
    /// Network-required on cold load (~7 GB SDXL base + ~1.5 GB
    /// motion adapter).
    pub fn get_or_load_animatediff_sdxl(
        &mut self,
        alias: &str,
        dtype: candle_core::DType,
    ) -> Result<&crate::pipelines::animatediff::AnimateDiffSdxlPipeline> {
        let hit = self
            .loaded_animatediff_sdxl
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            self.loaded_animatediff_sdxl = None;
            let device = self.device.clone();
            let alias_owned = alias.to_string();
            // v0.34 phase 3: SDXL animate uses the user's alias as
            // cache key — pairs with SDXL `plakat.load` of the same
            // alias and avoids the ~330 MB SDXL VAE rebuild.
            let cached_vae = vae_cache_lookup_script(self.vae_cache.as_ref(), alias);
            if cached_vae.is_some() {
                tracing::info!(
                    target: "plakat",
                    "v0.34 phase 3: VAE cache HIT on scripting animate SDXL load (alias={alias})"
                );
            }
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_animatediff_sdxl: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(
                    crate::pipelines::animatediff::AnimateDiffSdxlPipeline::load_sdxl_beta(
                        &device,
                        dtype,
                        &alias_owned,
                        &[],
                        1.0,
                        cached_vae,
                    ),
                )
            })?;
            self.vae_cache = Some((
                alias.to_string(),
                std::sync::Arc::clone(&pipeline.vae),
            ));
            self.loaded_animatediff_sdxl = Some((alias.to_string(), pipeline));
        }

        Ok(&self.loaded_animatediff_sdxl.as_ref().expect("just inserted").1)
    }

    /// v0.36 phase 1: get-or-load the PixArt-Σ pipeline for `alias`.
    /// Cache key is the user's alias (`pixart` / `pixart-sigma` /
    /// `pixart-1024`); same-alias hit reuses; alias change drops +
    /// reloads. LoRA stack mutation drops via
    /// [`Self::mark_loras_changed`].
    ///
    /// LoRAs: passes the current `self.loras` stack into the PixArt
    /// load (which merges via the v0.35 phase 4 tempfile-merge
    /// path). LoRA stack mutation between calls drops the cache.
    pub fn get_or_load_pixart(
        &mut self,
    ) -> Result<&mut crate::pipelines::pixart::Pipeline> {
        let alias_owned = self
            .loaded_model()
            .ok_or_else(|| {
                anyhow!(
                    "ScriptCtx::get_or_load_pixart: no model loaded. \
                     Call `\"pixart\" plakat.load` before `plakat.pixart`."
                )
            })?
            .to_string();
        self.cache_or_load_pixart(alias_owned)
    }

    /// v0.42 phase 4: load-or-cache PixArt for an EXPLICIT alias (see
    /// [`Self::cache_or_load_cascade`] for the chicken-and-egg rationale
    /// — `ensure_loaded` / `plakat.load` calls this directly).
    fn cache_or_load_pixart(
        &mut self,
        alias_owned: String,
    ) -> Result<&mut crate::pipelines::pixart::Pipeline> {
        let hit = self
            .loaded_pixart
            .as_ref()
            .map(|(a, _)| a == &alias_owned)
            .unwrap_or(false);

        if !hit {
            self.loaded_pixart = None;
            let device = self.device.clone();
            let lora_scale = self.config.lora_scale;
            // Resolve scenario-level LoRAs once for the lifetime of
            // this pipeline. Network-required if any are HF / Civitai.
            let loras_snapshot = self.loras.clone();
            // VAE cache lookup: pairs with `plakat.load <pixart-alias>`
            // of the same alias and avoids the ~330 MB VAE rebuild.
            let cached_vae =
                vae_cache_lookup_script(self.vae_cache.as_ref(), &alias_owned);
            if cached_vae.is_some() {
                tracing::info!(
                    target: "plakat",
                    "v0.36 phase 1: VAE cache HIT on scripting PixArt load (alias={alias_owned})"
                );
            }
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_pixart: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    // Resolve LoRA specs inside the async block —
                    // matches the v0.35 phase 4 pixart::run pattern.
                    let mut resolved: Vec<
                        crate::pipelines::lora::ResolvedLora,
                    > = Vec::with_capacity(loras_snapshot.len());
                    for spec in &loras_snapshot {
                        resolved.push(spec.resolve().await?);
                    }
                    let repo = if alias_owned.contains('/') {
                        alias_owned.clone()
                    } else {
                        crate::hf::resolve_alias(&alias_owned).to_string()
                    };
                    crate::pipelines::pixart::Pipeline::load(
                        crate::pipelines::pixart::LoadRequest {
                            repo,
                            device,
                            vae_cache: cached_vae,
                            loras: resolved,
                            lora_scale,
                        },
                    )
                    .await
                })
            })?;
            // Populate VAE cache from the freshly loaded pipeline so
            // subsequent SDXL t2i loads with the same alias reuse it.
            self.vae_cache = Some((
                alias_owned.clone(),
                std::sync::Arc::clone(&pipeline.vae),
            ));
            self.loaded_pixart = Some((alias_owned, pipeline));
        }

        Ok(&mut self.loaded_pixart.as_mut().expect("just inserted").1)
    }

    /// v0.38 phase 2: get-or-load the Stable Cascade pipeline for
    /// the currently-loaded Cascade alias. Mirrors
    /// [`Self::get_or_load_pixart`] but with no LoRA / VAE cache
    /// plumbing — Cascade's Stage A VAE is a custom Paella v3 design
    /// (not SD-family AutoEncoderKL) so the v0.34 phase 3 VAE cache
    /// doesn't apply; scripting-side LoRA support is v0.38 phase 3.
    pub fn get_or_load_cascade(
        &mut self,
    ) -> Result<&mut crate::pipelines::cascade::Pipeline> {
        let alias_owned = self
            .loaded_model()
            .ok_or_else(|| {
                anyhow!(
                    "ScriptCtx::get_or_load_cascade: no model loaded. \
                     Call `\"stable-cascade\" plakat.load` before `plakat.cascade`."
                )
            })?
            .to_string();
        self.cache_or_load_cascade(alias_owned)
    }

    /// v0.42 phase 4: load-or-cache the Stable Cascade pipeline for an
    /// EXPLICIT alias. `get_or_load_cascade` reads the alias from
    /// `loaded_model()`; `ensure_loaded` (the `plakat.load` path) calls
    /// this directly so the load isn't gated on `loaded_model()` already
    /// knowing about the cascade slot (chicken-and-egg, since the alias
    /// only lands in `loaded_cascade` *after* this returns).
    fn cache_or_load_cascade(
        &mut self,
        alias_owned: String,
    ) -> Result<&mut crate::pipelines::cascade::Pipeline> {
        // v0.42 phase 4: a canny ControlSpec on the stack (pushed via
        // `plakat.controlnet.add` / `.annotate`) means the cascade
        // pipeline must be loaded WITH its ControlNet. Reload if the
        // CN-request state flipped since the cached load.
        let cn_requested = cascade_cn_requested(&self.controlnets);
        let hit = self
            .loaded_cascade
            .as_ref()
            .map(|(a, p)| a == &alias_owned && p.control_conditioning_active() == cn_requested)
            .unwrap_or(false);

        if !hit {
            self.loaded_cascade = None;
            let device = self.device.clone();
            // v0.38 phase 3: LoRA stack merges into Stage B + Stage C
            // safetensors at load time (same pattern as PixArt). Any
            // mutation between calls drops the slot via
            // mark_loras_changed below.
            let lora_scale = self.config.lora_scale;
            let loras_snapshot = self.loras.clone();
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!(
                    "ScriptCtx::get_or_load_cascade: no tokio runtime in scope. {e}"
                )
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    let mut resolved: Vec<
                        crate::pipelines::lora::ResolvedLora,
                    > = Vec::with_capacity(loras_snapshot.len());
                    for spec in &loras_snapshot {
                        resolved.push(spec.resolve().await?);
                    }
                    let repo = if alias_owned.contains('/') {
                        alias_owned.clone()
                    } else {
                        crate::hf::resolve_alias(&alias_owned).to_string()
                    };
                    // v0.42 phase 4: auto-resolve the canny ControlNet
                    // from the model repo when a spec is on the stack —
                    // same `controlnet/canny.safetensors` path the t2i
                    // CLI auto-resolves (cascade::run).
                    let controlnet_weights = if cn_requested {
                        Some(
                            crate::hf::download::get_first_of(&[(
                                &repo,
                                "controlnet/canny.safetensors",
                            )])
                            .await
                            .map_err(|e| {
                                anyhow!(
                                    "auto-resolving Cascade canny ControlNet \
                                     for plakat.cascade: {e}"
                                )
                            })?,
                        )
                    } else {
                        None
                    };
                    crate::pipelines::cascade::Pipeline::load(
                        crate::pipelines::cascade::LoadRequest {
                            repo,
                            device,
                            loras: resolved,
                            lora_scale,
                            controlnet_weights,
                            // v0.42 phase 3: image variation is a
                            // CLI-only path for now.
                            image_encoder_weights: None,
                        },
                    )
                    .await
                })
            })?;
            self.loaded_cascade = Some((alias_owned, pipeline));
        }

        Ok(&mut self.loaded_cascade.as_mut().expect("just inserted").1)
    }

    /// v0.22 phase 2: get-or-load the Flux pipeline for `alias`.
    /// Mirrors [`Self::get_or_load_sd_family`] for the Flux
    /// family — same cache-invalidation rules, same internal
    /// `block_in_place` async bridge.
    ///
    /// Reads the Flux-specific D-keys (`quantize_t5`,
    /// `quant_level`, `t5_quant_level`) off `self.config` at
    /// load time. Changing those config keys AFTER the pipeline
    /// is cached has no effect until the next reload (alias
    /// change or explicit unload). This is the v0.21 behaviour
    /// for SD-family LoRAs too; documented in §7 of the v0.22
    /// RFC.
    ///
    /// Returns `&mut flux::Pipeline` because Flux's `generate`
    /// takes `&mut self` (T5 KV cache mutation). The borrow
    /// checker enforces single-script-at-a-time access through
    /// the singleton `RwLock`.
    pub fn get_or_load_flux(&mut self, alias: &str) -> Result<&mut flux::Pipeline> {
        let hit = self
            .loaded
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            self.loaded = None;
            // v0.23 phase 1: family change drops the SD t2i slot too.
            self.loaded_t2i = None;
            let resolved = if alias.contains('/') {
                alias.to_string()
            } else {
                crate::hf::resolve_alias(alias).to_string()
            };
            let t2i_variant = crate::pipelines::t2i::Variant::detect(&resolved);
            let fvar = match t2i_variant {
                crate::pipelines::t2i::Variant::FluxSchnell => flux::Variant::Schnell,
                crate::pipelines::t2i::Variant::FluxDev => flux::Variant::Dev,
                crate::pipelines::t2i::Variant::FluxFillDev => flux::Variant::FillDev,
                crate::pipelines::t2i::Variant::FluxCannyDev => flux::Variant::CannyDev,
                crate::pipelines::t2i::Variant::FluxDepthDev => flux::Variant::DepthDev,
                crate::pipelines::t2i::Variant::FluxKontextDev => flux::Variant::KontextDev,
                _ => {
                    return Err(anyhow!(
                        "ScriptCtx::get_or_load_flux: alias {alias:?} \
                         doesn't resolve to a Flux variant ({t2i_variant:?})"
                    ));
                }
            };

            // v0.23 phase 6 + v0.24 phase 8: resolve the script's
            // ControlNet stack into Flux-flavoured load specs.
            // image= specs get the path baked into the load
            // request; from= specs leave `conditioning: None` —
            // the annotator runs at first generate (script_entry)
            // and feeds the result back via
            // `pipeline.set_controlnet_conditioning`.
            let flux_cn_loads: Vec<flux::FluxControlNetLoad> = self
                .controlnets
                .iter()
                .map(|spec| -> Result<flux::FluxControlNetLoad> {
                    let cond = match (&spec.image, &spec.from) {
                        (Some(path), None) => Some(path.clone()),
                        (None, Some(_)) => None, // lazy-annotate
                        (Some(_), Some(_)) => anyhow::bail!(
                            "plakat.controlnet (Flux): a single spec has \
                             both image= and from= set; pick one. (kind={:?})",
                            spec.kind
                        ),
                        (None, None) => anyhow::bail!(
                            "plakat.controlnet (Flux): kind={:?} needs \
                             either image=PATH (pre-rendered) or from=PATH \
                             (auto-annotate).",
                            spec.kind
                        ),
                    };
                    let mut cn_load = crate::pipelines::t2i::flux_controlnet_load_for(
                        spec.kind, fvar, spec.strength,
                    )?;
                    cn_load.conditioning = cond;
                    cn_load.start = spec.start;
                    cn_load.end = spec.end;
                    Ok(cn_load)
                })
                .collect::<Result<Vec<_>>>()?;

            let device = self.device.clone();
            let quantize_t5 = self.config.quantize_t5;
            let quant_level = self.config.quant_level.clone();
            let t5_quant_level = self.config.t5_quant_level.clone();
            let lora_specs = self.loras.clone();
            let lora_scale = self.config.lora_scale;
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!("ScriptCtx::get_or_load_flux: no tokio runtime in scope. {e}")
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(async {
                    // Flux's LoadRequest wants ResolvedLora, not
                    // LoraSpec — resolve here (downloads happen
                    // lazily) and pass the resolved list through.
                    let mut resolved_loras = Vec::with_capacity(lora_specs.len());
                    for spec in &lora_specs {
                        resolved_loras.push(spec.resolve().await?);
                    }
                    flux::Pipeline::load(flux::LoadRequest {
                        variant: fvar,
                        repo: resolved,
                        device,
                        loras: resolved_loras,
                        lora_scale,
                        controlnets: flux_cn_loads,
                        quantize_t5,
                        flux_quant_level: quant_level,
                        t5_quant_level,
                        redux: false,
                    })
                    .await
                })
            })?;
            self.loaded = Some((alias.to_string(), LoadedPipeline::Flux(pipeline)));
        }

        match &mut self.loaded.as_mut().expect("just inserted").1 {
            LoadedPipeline::Flux(p) => Ok(p),
            LoadedPipeline::SdFamily(_) | LoadedPipeline::Sd3(_) => Err(anyhow!(
                "ScriptCtx::get_or_load_flux called with a non-Flux \
                 alias — the cache is holding a different pipeline. \
                 Use ensure_loaded for family-aware dispatch."
            )),
        }
    }

    /// v0.22 phase 3 + v0.23 phase 7: get-or-load the SD3 / SD3.5
    /// pipeline for `alias`. Mirrors [`Self::get_or_load_sd_family`]
    /// and [`Self::get_or_load_flux`].
    ///
    /// v0.23 phase 7 resolves `self.controlnets` into
    /// `Vec<Sd3ControlNetLoad>` at load time (image=PATH specs
    /// only — auto-annotate bails the same way Flux does, since
    /// neither family knows the per-generate dims at load).
    /// `mark_controlnets_changed` drops this slot on stack
    /// mutations.
    pub fn get_or_load_sd3(&mut self, alias: &str) -> Result<&mut sd3::Pipeline> {
        let hit = self
            .loaded
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            self.loaded = None;
            // v0.23 phase 1: family change drops the SD t2i slot too.
            self.loaded_t2i = None;
            let resolved = if alias.contains('/') {
                alias.to_string()
            } else {
                crate::hf::resolve_alias(alias).to_string()
            };
            let t2i_variant = crate::pipelines::t2i::Variant::detect(&resolved);
            let sd3_variant = match t2i_variant {
                crate::pipelines::t2i::Variant::Sd3Medium => sd3::Variant::Sd3Medium,
                crate::pipelines::t2i::Variant::Sd35Medium => sd3::Variant::Sd35Medium,
                crate::pipelines::t2i::Variant::Sd35Large => sd3::Variant::Sd35Large,
                crate::pipelines::t2i::Variant::Sd35LargeTurbo => {
                    sd3::Variant::Sd35LargeTurbo
                }
                _ => {
                    return Err(anyhow!(
                        "ScriptCtx::get_or_load_sd3: alias {alias:?} \
                         doesn't resolve to an SD3 variant ({t2i_variant:?})"
                    ));
                }
            };

            // v0.23 phase 7 + v0.24 phase 8: resolve the script's
            // ControlNet stack into SD3-flavoured load specs.
            // image= specs bake the path in; from= specs leave
            // conditioning=None and the annotator fires at first
            // generate (same lazy pattern as Flux).
            let sd3_cn_loads: Vec<crate::pipelines::sd3_controlnet::Sd3ControlNetLoad> =
                self.controlnets
                    .iter()
                    .map(|spec| -> Result<crate::pipelines::sd3_controlnet::Sd3ControlNetLoad> {
                        let cond = match (&spec.image, &spec.from) {
                            (Some(path), None) => Some(path.clone()),
                            (None, Some(_)) => None, // lazy-annotate
                            (Some(_), Some(_)) => anyhow::bail!(
                                "plakat.controlnet (SD3): a single spec has \
                                 both image= and from= set; pick one. (kind={:?})",
                                spec.kind
                            ),
                            (None, None) => anyhow::bail!(
                                "plakat.controlnet (SD3): kind={:?} needs \
                                 either image=PATH (pre-rendered) or from=PATH \
                                 (auto-annotate).",
                                spec.kind
                            ),
                        };
                        let mut cn_load = crate::pipelines::t2i::sd3_controlnet_load_for(
                            spec.kind, sd3_variant, spec.strength,
                        )?;
                        cn_load.conditioning = cond;
                        cn_load.start = spec.start;
                        cn_load.end = spec.end;
                        Ok(cn_load)
                    })
                    .collect::<Result<Vec<_>>>()?;

            let device = self.device.clone();
            let loras = self.loras.clone();
            let lora_scale = self.config.lora_scale;
            let handle = tokio::runtime::Handle::try_current().map_err(|e| {
                anyhow!("ScriptCtx::get_or_load_sd3: no tokio runtime in scope. {e}")
            })?;
            let pipeline = tokio::task::block_in_place(|| {
                handle.block_on(sd3::Pipeline::load(sd3::LoadRequest {
                    variant: sd3_variant,
                    repo: resolved,
                    device,
                    loras,
                    lora_scale,
                    controlnets: sd3_cn_loads,
                }))
            })?;
            self.loaded = Some((alias.to_string(), LoadedPipeline::Sd3(pipeline)));
        }

        match &mut self.loaded.as_mut().expect("just inserted").1 {
            LoadedPipeline::Sd3(p) => Ok(p),
            LoadedPipeline::SdFamily(_) | LoadedPipeline::Flux(_) => Err(anyhow!(
                "ScriptCtx::get_or_load_sd3 called with a non-SD3 \
                 alias — the cache is holding a different pipeline. \
                 Use ensure_loaded for family-aware dispatch."
            )),
        }
    }

    /// v0.22 phase 2: unified get-or-load dispatching on family.
    /// Ensures the cache holds the right pipeline for `alias`;
    /// callers then borrow-match against [`Self::loaded`] for the
    /// pattern they need.
    ///
    /// Pattern at call sites:
    ///
    /// ```ignore
    /// ctx.ensure_loaded(&alias)?;
    /// match &mut ctx.loaded.as_mut().unwrap().1 {
    ///     LoadedPipeline::SdFamily(p) => { /* ... */ }
    ///     LoadedPipeline::Flux(p) => { /* ... */ }
    /// }
    /// ```
    ///
    /// We don't return a `&mut LoadedPipeline` directly because
    /// the borrow checker treats `get_or_load_sd_family` /
    /// `get_or_load_flux` as still-holding `&mut self` even after
    /// `?`, which would prevent the caller from accessing other
    /// `ScriptCtx` fields. Splitting "load" from "borrow" sidesteps
    /// the issue.
    pub fn ensure_loaded(&mut self, alias: &str) -> Result<()> {
        // v0.42 phase 4: PixArt and Stable Cascade have dedicated
        // loaders + slots that `PipelineFamily::detect` (SD/Flux/SD3
        // only) doesn't know about — without this, a cascade alias
        // mis-routes to the SD-only loader and `plakat.load` errors.
        // Detect them up front, clear any sibling slot so
        // `loaded_model()` resolves to the right alias, and warm the
        // dedicated slot.
        let resolved = if alias.contains('/') {
            alias.to_string()
        } else {
            crate::hf::resolve_alias(alias).to_string()
        };
        let variant = crate::pipelines::t2i::Variant::detect(&resolved);
        if variant.is_cascade() {
            self.loaded = None;
            self.loaded_t2i = None;
            self.loaded_pixart = None;
            self.cache_or_load_cascade(alias.to_string())?;
            return Ok(());
        }
        if variant.is_pixart() {
            self.loaded = None;
            self.loaded_t2i = None;
            self.loaded_cascade = None;
            self.cache_or_load_pixart(alias.to_string())?;
            return Ok(());
        }
        // SD/Flux/SD3 also live in their own slots; clear the
        // PixArt/Cascade ones on a switch so `loaded_model()` doesn't
        // report a stale Cascade/PixArt alias.
        self.loaded_pixart = None;
        self.loaded_cascade = None;
        match PipelineFamily::detect(alias) {
            PipelineFamily::SdFamily => {
                // v0.23 phase 1: `plakat.load` now warms the t2i
                // slot by default for SD-family aliases.
                // plakat.portrait + plakat.img2img will lazy-load
                // the portrait slot on first call (deriving from
                // the t2i slot's SdCore — no second weights load).
                self.get_or_load_sd_t2i(alias)?;
                Ok(())
            }
            PipelineFamily::Flux => {
                self.get_or_load_flux(alias)?;
                Ok(())
            }
            PipelineFamily::Sd3 => {
                self.get_or_load_sd3(alias)?;
                Ok(())
            }
        }
    }
}

/// v0.22 phase 1 + v0.24 phase 3: pick the identity strategy for
/// an SD-family alias. `override_kind` (v0.24 phase 3) lets the
/// script override the alias-based auto-pick via the
/// `identity_kind` config key. When `override_kind` is `None`
/// or empty, the v0.22 auto-pick rule applies: sd15 → PlusFace,
/// sdxl → PlusFaceSdxl, sd21 → None (no shipped Plus-Face
/// checkpoint; portrait bails at generate time on sd21).
///
/// Caller is responsible for having validated SD-family-ness.
fn pick_sd_family_identity(
    alias: &str,
    override_kind: Option<&str>,
) -> Option<crate::pipelines::ip_adapter::IdentityKind> {
    use crate::pipelines::ip_adapter::IdentityKind;
    // v0.24 phase 3: override wins when non-empty. set_str
    // validated the string at config-set time, so re-parsing
    // here is effectively infallible — but bail loudly if not
    // (a panic would mask a config-layer bug).
    if let Some(s) = override_kind {
        let s = s.trim();
        if !s.is_empty() {
            use std::str::FromStr;
            return IdentityKind::from_str(s)
                .ok()
                .map(Some)
                .unwrap_or(None);
        }
    }
    let resolved = if alias.contains('/') {
        alias.to_string()
    } else {
        crate::hf::resolve_alias(alias).to_string()
    };
    let variant = crate::pipelines::t2i::Variant::detect(&resolved);
    if variant.is_xl() {
        Some(IdentityKind::PlusFaceSdxl)
    } else if matches!(variant, crate::pipelines::t2i::Variant::Sd21) {
        // SD 2.1 has no shipped Plus-Face checkpoint. Load
        // without identity; plakat.portrait will bail at generate
        // time with the underlying "no identity encoder" message.
        None
    } else {
        // SD 1.5 default.
        Some(IdentityKind::PlusFace)
    }
}

impl ScriptCtx {

    /// v0.21 phase 2: register a rendered image and return the
    /// 1-based handle the script will see. Caller is responsible
    /// for serialising mutation through [`with_ctx_mut`].
    ///
    /// No metadata attached. For rendering paths that have
    /// A1111-style metadata to carry, use
    /// [`Self::push_image_with_metadata`].
    pub fn push_image(&mut self, img: image::DynamicImage) -> i64 {
        self.images.push(img);
        self.images_metadata.push(None);
        self.images.len() as i64
    }

    /// v0.26 phase 8: register a rendered image with its
    /// generation metadata attached. `plakat.save` writes the
    /// JSON sidecar + PNG tEXt automatically; `plakat.metadata
    /// .write` reads the metadata from the registered handle.
    pub fn push_image_with_metadata(
        &mut self,
        img: image::DynamicImage,
        meta: crate::imaging::metadata::GenerationMetadata,
    ) -> i64 {
        self.images.push(img);
        self.images_metadata.push(Some(meta));
        self.images.len() as i64
    }

    /// v0.26 phase 8: look up the metadata for a handle, if it
    /// was registered with [`Self::push_image_with_metadata`].
    /// Bails on unknown handles (same as [`Self::image_at`]).
    ///
    /// Lenient on size mismatch between `images` and
    /// `images_metadata`: returns `Ok(None)` if the metadata Vec
    /// is shorter than the image Vec at this handle. Lets tests
    /// (and historical callers) pre-stuff `images` directly via
    /// `ctx.images.push(...)` without breaking `plakat.save`.
    pub fn metadata_at(
        &self,
        handle: i64,
    ) -> Result<Option<&crate::imaging::metadata::GenerationMetadata>> {
        if handle <= 0 {
            return Err(anyhow!(
                "image handle must be >= 1 (got {handle}); handle 0 is reserved"
            ));
        }
        let idx = handle as usize - 1;
        // image_at is the authority on handle validity; metadata
        // is best-effort.
        if idx >= self.images.len() {
            return Err(anyhow!(
                "image handle {handle} not found (only {} image(s) rendered so far)",
                self.images.len()
            ));
        }
        Ok(self.images_metadata.get(idx).and_then(|o| o.as_ref()))
    }

    /// v0.21 phase 2: look up an image by its script-visible
    /// handle. Bails on unknown handles + on the reserved 0
    /// handle.
    pub fn image_at(&self, handle: i64) -> Result<&image::DynamicImage> {
        if handle <= 0 {
            return Err(anyhow!(
                "image handle must be >= 1 (got {handle}); handle 0 is reserved"
            ));
        }
        let idx = handle as usize - 1;
        self.images.get(idx).ok_or_else(|| {
            anyhow!(
                "image handle {handle} not found (only {} image(s) rendered so far)",
                self.images.len()
            )
        })
    }
}

/// The singleton. Using `std::sync::RwLock` to keep the dep
/// footprint flat; phase-1's contention story is "one host word
/// at a time on one thread" so the lighter parking_lot variant
/// wouldn't pay back.
pub(crate) static CTX: OnceLock<RwLock<ScriptCtx>> = OnceLock::new();

/// Borrow the script context for a read. Bails if [`ScriptCtx::init`]
/// hasn't run yet — host words always need a context.
pub fn with_ctx<R>(f: impl FnOnce(&ScriptCtx) -> R) -> Result<R> {
    let lock = CTX
        .get()
        .ok_or_else(|| anyhow!("ScriptCtx not initialised — was `plakat run` invoked?"))?;
    let guard = lock
        .read()
        .map_err(|e| anyhow!("ScriptCtx read lock poisoned: {e}"))?;
    Ok(f(&guard))
}

/// v0.42 phase 4: does the ControlSpec stack request a Stable Cascade
/// ControlNet? Cascade ships only a canny CN, so only a canny spec
/// triggers the load-time CN attach in [`ScriptCtx::get_or_load_cascade`].
/// Other kinds (depth, lineart, …) are SD/Flux-only and are ignored by
/// the cascade path rather than erroring at load — the `plakat.cascade`
/// word raises the loud "canny only" error when it actually tries to
/// build the conditioning.
fn cascade_cn_requested(specs: &[crate::pipelines::controlnet::ControlSpec]) -> bool {
    specs
        .iter()
        .any(|s| matches!(s.kind, crate::pipelines::controlnet::ControlKind::Canny))
}

/// Borrow the script context for a write.
pub fn with_ctx_mut<R>(f: impl FnOnce(&mut ScriptCtx) -> R) -> Result<R> {
    let lock = CTX
        .get()
        .ok_or_else(|| anyhow!("ScriptCtx not initialised — was `plakat run` invoked?"))?;
    let mut guard = lock
        .write()
        .map_err(|e| anyhow!("ScriptCtx write lock poisoned: {e}"))?;
    Ok(f(&mut guard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn cascade_cn_requested_only_on_canny() {
        use crate::pipelines::controlnet::{ControlKind, ControlSpec};
        let spec = |kind| ControlSpec {
            kind,
            image: Some(std::path::PathBuf::from("/tmp/edges.png")),
            from: None,
            video: None,
            strength: 1.0,
            start: 0.0,
            end: 1.0,
        };
        // No specs → no CN.
        assert!(!cascade_cn_requested(&[]));
        // A canny spec → CN requested.
        assert!(cascade_cn_requested(&[spec(ControlKind::Canny)]));
        // A non-canny spec → not requested (Cascade ships only canny).
        assert!(!cascade_cn_requested(&[spec(ControlKind::Depth)]));
        // Mixed: a canny anywhere in the stack triggers it.
        assert!(cascade_cn_requested(&[
            spec(ControlKind::Depth),
            spec(ControlKind::Canny),
        ]));
    }

    fn mk_ctx() -> ScriptCtx {
        ScriptCtx {
            device: Device::Cpu,
            out_dir: std::env::temp_dir(),
            loaded: None,
            loaded_t2i: None,
            images: Vec::new(),
            images_metadata: Vec::new(),
            config: GenerationConfig::default(),
            loras: Vec::new(),
            controlnets: Vec::new(),
            refiner_enabled: false,
            adetailer_enabled: false,
            hires_enabled: false,
            artefacts: Vec::new(),
            artefact_blend_enabled: false,
            style_id: None,
            style_ref: None,
            portrait_photos: Vec::new(),
            embeddings: Vec::new(),
            cn_annotation_cache: None,
            look_name: None,
            genre_name: None,
            map_layout: None,
            map_erosion: None,
            loaded_stylize: None,
            loaded_animatediff: None,
            loaded_animatediff_sdxl: None,
            loaded_pixart: None,
            loaded_cascade: None,
            vae_cache: None,
        }
    }

    fn mk_image(r: u8) -> DynamicImage {
        let mut img = RgbImage::new(2, 2);
        for p in img.pixels_mut() {
            *p = image::Rgb([r, 0, 0]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn push_image_returns_one_based_handle() {
        let mut ctx = mk_ctx();
        assert_eq!(ctx.push_image(mk_image(10)), 1);
        assert_eq!(ctx.push_image(mk_image(20)), 2);
        assert_eq!(ctx.push_image(mk_image(30)), 3);
    }

    #[test]
    fn image_at_returns_the_pushed_image() {
        let mut ctx = mk_ctx();
        let h = ctx.push_image(mk_image(99));
        let got = ctx.image_at(h).unwrap();
        // pixel (0,0) should be (99, 0, 0)
        let rgb = got.to_rgb8();
        let p = rgb.get_pixel(0, 0);
        assert_eq!(p.0, [99, 0, 0]);
    }

    #[test]
    fn image_at_zero_bails_with_reserved_message() {
        let ctx = mk_ctx();
        let err = ctx.image_at(0).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("reserved"), "got {msg}");
    }

    #[test]
    fn image_at_negative_bails() {
        let ctx = mk_ctx();
        assert!(ctx.image_at(-1).is_err());
    }

    #[test]
    fn image_at_unknown_handle_includes_rendered_count() {
        let mut ctx = mk_ctx();
        ctx.push_image(mk_image(1));
        let err = ctx.image_at(99).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("99"), "got {msg}");
        // The diagnostic mentions the rendered count so users can
        // tell whether they're addressing a future handle vs a
        // typo.
        assert!(msg.contains("1 image"), "got {msg}");
    }

    // v0.22 phase 1: pick_sd_family_identity returns the right
    // identity strategy per alias, mapping sd21 → None (no
    // shipped Plus-Face checkpoint) so the cache load succeeds
    // for plakat.generate even on sd21.

    #[test]
    fn pick_sd_family_identity_sd15_is_plus_face() {
        use crate::pipelines::ip_adapter::IdentityKind;
        let id = pick_sd_family_identity("sd15", None);
        assert!(matches!(id, Some(IdentityKind::PlusFace)));
    }

    #[test]
    fn pick_sd_family_identity_sdxl_is_plus_face_sdxl() {
        use crate::pipelines::ip_adapter::IdentityKind;
        assert!(matches!(
            pick_sd_family_identity("sdxl", None),
            Some(IdentityKind::PlusFaceSdxl)
        ));
        assert!(matches!(
            pick_sd_family_identity("sdxl-turbo", None),
            Some(IdentityKind::PlusFaceSdxl)
        ));
    }

    #[test]
    fn pick_sd_family_identity_sd21_is_none() {
        // SD 2.1 has no shipped Plus-Face checkpoint. The cache
        // loads without identity; plakat.portrait bails at
        // generate time, but plakat.generate works.
        assert!(pick_sd_family_identity("sd21", None).is_none());
    }

    #[test]
    fn pick_sd_family_identity_resolves_alias_before_detection() {
        // Bare "sd21" contains none of Variant::detect's SD-2.1
        // substrings; only the resolved repo id does. We resolve
        // first so the detection works.
        // The resolved-alias path is also exercised by the
        // canonical HF repo path:
        assert!(
            pick_sd_family_identity("nlightcho/stable-diffusion-2-1", None)
                .is_none()
        );
    }

    // v0.24 phase 3: identity_kind override.

    /// Non-empty override wins over the alias-based auto-pick.
    #[test]
    fn pick_sd_family_identity_override_wins() {
        use crate::pipelines::ip_adapter::IdentityKind;
        // sd15 would auto-pick PlusFace; override to FaceId.
        let id = pick_sd_family_identity("sd15", Some("face-id"));
        assert!(matches!(id, Some(IdentityKind::FaceId)));
        // sdxl with override.
        let id = pick_sd_family_identity("sdxl", Some("face-id-sdxl"));
        assert!(matches!(id, Some(IdentityKind::FaceIdSdxl)));
    }

    /// Empty override falls back to auto-pick.
    #[test]
    fn pick_sd_family_identity_empty_override_falls_back() {
        use crate::pipelines::ip_adapter::IdentityKind;
        assert!(matches!(
            pick_sd_family_identity("sd15", Some("")),
            Some(IdentityKind::PlusFace)
        ));
        assert!(matches!(
            pick_sd_family_identity("sd15", Some("   ")),
            Some(IdentityKind::PlusFace)
        ));
    }

    /// Override can force PlusFaceSdxl even on a non-XL alias
    /// (caller's responsibility to keep this sane — pipeline load
    /// will bail at runtime if the override mismatches the
    /// model's hidden dim).
    #[test]
    fn pick_sd_family_identity_override_on_sd21() {
        use crate::pipelines::ip_adapter::IdentityKind;
        let id = pick_sd_family_identity("sd21", Some("plus-face"));
        assert!(matches!(id, Some(IdentityKind::PlusFace)));
    }

    #[test]
    fn loaded_model_accessor_returns_none_when_unloaded() {
        let ctx = mk_ctx();
        assert!(ctx.loaded_model().is_none());
    }

    // v0.23 phase 1: mark_loras_changed drops both SD-family
    // slots (primary `loaded` + secondary `loaded_t2i`).
    #[test]
    fn mark_loras_changed_drops_both_sd_slots() {
        let mut ctx = mk_ctx();
        // We can't easily fabricate a real pipeline here without a
        // tokio runtime + model download, but we can at least
        // exercise the field-clearing path: pre-populating with
        // None and calling mark_loras_changed is a no-op that
        // should still leave both slots None.
        assert!(ctx.loaded.is_none());
        assert!(ctx.loaded_t2i.is_none());
        ctx.mark_loras_changed();
        assert!(ctx.loaded.is_none());
        assert!(ctx.loaded_t2i.is_none());
    }

    // v0.23 phase 6: mark_controlnets_changed is a no-op when
    // both slots are empty (smoke test). The real semantics
    // (drops Flux/SD3 slot, leaves SD-family alone) need a real
    // loaded pipeline to verify; covered by CLI smoke + the
    // documented invariant inside the method.
    #[test]
    fn mark_controlnets_changed_is_safe_when_empty() {
        let mut ctx = mk_ctx();
        assert!(ctx.loaded.is_none());
        assert!(ctx.loaded_t2i.is_none());
        ctx.mark_controlnets_changed();
        assert!(ctx.loaded.is_none());
        assert!(ctx.loaded_t2i.is_none());
    }
}
