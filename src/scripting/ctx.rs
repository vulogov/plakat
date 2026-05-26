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
    controlnet::ControlSpec, flux, lora::LoraSpec, portrait, sd3,
};
use crate::scripting::config::GenerationConfig;
use crate::scripting::loaded_pipeline::{LoadedPipeline, PipelineFamily};

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
    /// v0.21 phase 2: rendered images, addressable by the integer
    /// handle pushed onto the stack by `plakat.generate`. Index =
    /// handle (1-based — handle 0 is reserved as "no image").
    /// Phase 2 keeps every rendered image in memory for the
    /// script's lifetime; if scripts ever start producing hundreds
    /// of images we'll revisit (e.g. spill to disk).
    pub images: Vec<image::DynamicImage>,
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
    /// Flux + SD3 ControlNet need load-time wiring that doesn't
    /// fit phase 5's scope — the SD-family generate / img2img
    /// paths bail if `controlnets` is non-empty when running on
    /// Flux or SD3 with a clear "v0.23" pointer.
    pub controlnets: Vec<ControlSpec>,
    /// v0.22 phase 6: SDXL refiner toggle. `plakat.refiner.enable`
    /// sets this to `true`; `plakat.refiner.disable` resets it.
    ///
    /// The actual SDXL refiner UNet load is **not yet wired**:
    /// the cached `portrait::Pipeline` doesn't expose the
    /// refiner-UNet slot the way `t2i::Pipeline` does. When
    /// `refiner_enabled` is `true`, `script_entry::generate_one`
    /// bails with a v0.23 deferral message + remediation hint.
    /// The toggle is shipped today so the surface is stable for
    /// when the cache switches to `t2i::Pipeline`.
    pub refiner_enabled: bool,
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
            images: Vec::new(),
            config: GenerationConfig::default(),
            loras: Vec::new(),
            controlnets: Vec::new(),
            refiner_enabled: false,
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
    }

    /// v0.22 phase 1: read-only accessor for the currently-loaded
    /// model's alias. `None` when nothing's been `plakat.load`ed
    /// yet. Replaces direct access to v0.21's `loaded_model` field.
    pub fn loaded_model(&self) -> Option<&str> {
        self.loaded.as_ref().map(|(alias, _)| alias.as_str())
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

            let identity = pick_sd_family_identity(alias);
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

        match &self.loaded.as_ref().expect("just inserted").1 {
            LoadedPipeline::SdFamily(p) => Ok(p),
            LoadedPipeline::Flux(_) | LoadedPipeline::Sd3(_) => Err(anyhow!(
                "ScriptCtx::get_or_load_sd_family called with a non-SD \
                 alias — the cache is holding a different pipeline. \
                 Use ensure_loaded for family-aware dispatch."
            )),
        }
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
                        controlnets: Vec::new(),
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

    /// v0.22 phase 3: get-or-load the SD3 / SD3.5 pipeline for
    /// `alias`. Mirrors [`Self::get_or_load_sd_family`] and
    /// [`Self::get_or_load_flux`].
    ///
    /// SD3 doesn't have LoRA / ControlNet load-time D-keys
    /// (those land in phases 4-5); the LoadRequest fields are
    /// empty for now.
    pub fn get_or_load_sd3(&mut self, alias: &str) -> Result<&mut sd3::Pipeline> {
        let hit = self
            .loaded
            .as_ref()
            .map(|(a, _)| a == alias)
            .unwrap_or(false);

        if !hit {
            self.loaded = None;
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
                    controlnets: Vec::new(),
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
        match PipelineFamily::detect(alias) {
            PipelineFamily::SdFamily => {
                self.get_or_load_sd_family(alias)?;
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

/// v0.22 phase 1: pick the identity strategy for an SD-family
/// alias without bailing — sd21 returns `None` so the pipeline
/// loads without an identity encoder. Caller is responsible for
/// having validated SD-family-ness via
/// [`crate::scripting::script_entry::validate_supported_for_phase_2`].
fn pick_sd_family_identity(
    alias: &str,
) -> Option<crate::pipelines::ip_adapter::IdentityKind> {
    use crate::pipelines::ip_adapter::IdentityKind;
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
    pub fn push_image(&mut self, img: image::DynamicImage) -> i64 {
        self.images.push(img);
        self.images.len() as i64
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

    fn mk_ctx() -> ScriptCtx {
        ScriptCtx {
            device: Device::Cpu,
            out_dir: std::env::temp_dir(),
            loaded: None,
            images: Vec::new(),
            config: GenerationConfig::default(),
            loras: Vec::new(),
            controlnets: Vec::new(),
            refiner_enabled: false,
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
        let id = pick_sd_family_identity("sd15");
        assert!(matches!(id, Some(IdentityKind::PlusFace)));
    }

    #[test]
    fn pick_sd_family_identity_sdxl_is_plus_face_sdxl() {
        use crate::pipelines::ip_adapter::IdentityKind;
        assert!(matches!(
            pick_sd_family_identity("sdxl"),
            Some(IdentityKind::PlusFaceSdxl)
        ));
        assert!(matches!(
            pick_sd_family_identity("sdxl-turbo"),
            Some(IdentityKind::PlusFaceSdxl)
        ));
    }

    #[test]
    fn pick_sd_family_identity_sd21_is_none() {
        // SD 2.1 has no shipped Plus-Face checkpoint. The cache
        // loads without identity; plakat.portrait bails at
        // generate time, but plakat.generate works.
        assert!(pick_sd_family_identity("sd21").is_none());
    }

    #[test]
    fn pick_sd_family_identity_resolves_alias_before_detection() {
        // Bare "sd21" contains none of Variant::detect's SD-2.1
        // substrings; only the resolved repo id does. We resolve
        // first so the detection works.
        // The resolved-alias path is also exercised by the
        // canonical HF repo path:
        assert!(pick_sd_family_identity("stabilityai/stable-diffusion-2-1")
            .is_none());
    }

    #[test]
    fn loaded_model_accessor_returns_none_when_unloaded() {
        let ctx = mk_ctx();
        assert!(ctx.loaded_model().is_none());
    }
}
