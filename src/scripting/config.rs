//! v0.21 phase 3: per-script generation config.
//!
//! Held on [`ScriptCtx`](super::ctx::ScriptCtx) and mutated by the
//! `plakat.config.set` host word. Read by
//! [`super::script_entry::generate_one`] when building the
//! `t2i::Request`. Persistent across calls within one script — the
//! whole point is letting a script say `steps: 50` once and have
//! every subsequent `plakat.generate` honour it.
//!
//! Defaults mirror `cli::generate`'s clap defaults so scripts and
//! the CLI produce the same output for the same inputs.

use anyhow::{Context, Result, anyhow, bail};

use crate::pipelines::scheduler::SchedulerKind;

/// Mutable config the script accumulates through `plakat.config.set`
/// calls. Phase 3 covers the seven keys named in the RFC (steps,
/// guidance, seed, width, height, negative, scheduler). Phase 4+
/// may extend (clip_skip, refine, refine_strength) but only when
/// a host word actually needs them.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub steps: usize,
    pub guidance: f64,
    /// `None` → pipeline picks a random seed per call. Setting an
    /// explicit seed via `plakat.config.set` pins it across calls.
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub negative: String,
    pub scheduler: SchedulerKind,
    /// v0.21 phase 4: img2img denoise strength in `[0, 1]`. 1.0 =
    /// fully re-noised input (output ignores the input). 0.0 = no
    /// denoise (output == input). Default 0.75 matches the
    /// `cli::img2img` default. Ignored by `plakat.generate`.
    pub strength: f32,
    /// v0.21 phase 5: IP-Adapter image-token contribution scale in
    /// `[0, 1]`. 1.0 = image tokens carry full weight; 0.0 collapses
    /// portrait into a text-only generate. Default 0.8 matches
    /// `cli::portrait`'s default. Read by `plakat.portrait`; ignored
    /// by `plakat.generate` / `plakat.img2img`.
    pub face_strength: f32,
    /// v0.22 phase 2: load T5-XXL as a quantized GGUF (~3 GB instead
    /// of ~10 GB BF16). Default `false`. Flux-only — ignored on
    /// SD-family. Combined with a `*-gguf` model alias, total Flux
    /// footprint drops from ~17 GB to ~10 GB (fits 12 GB GPUs).
    pub quantize_t5: bool,
    /// v0.22 phase 2: Flux transformer GGUF quant level (`Q4_K_S`,
    /// `Q4_K_M`, `Q5_K_M`, `Q8_0`, …). `None` → city96's `Q4_K_S`
    /// default. Validated against the published city96 quant list.
    /// Flux-only.
    pub quant_level: Option<String>,
    /// v0.22 phase 2: T5-XXL GGUF quant level. `None` → `Q4_K_M`.
    /// Honoured only when `quantize_t5` is `true`.
    pub t5_quant_level: Option<String>,
    /// v0.22 phase 2: distillation preset name. Maps to the same
    /// `--fast PRESET` table the CLI exposes: `hyper-8`, `hyper-16`,
    /// `turbo-alpha` (Flux), `lcm-sdxl`, `lcm-sd15`. `None` → no
    /// preset. Validated at apply time so unknown names bail with
    /// the supported list.
    pub fast: Option<String>,
    /// v0.22 phase 2: opt-in Kontext aspect-bucket snap. When `true`
    /// AND the loaded model is `flux-kontext-dev`, the requested
    /// (width, height) snaps to the nearest of 17 BFL-recommended
    /// resolutions before VAE encoding. No-op on every other model.
    pub kontext_bucket: bool,
    /// v0.22 phase 3: tiled MultiDiffusion-style denoise. When `true`,
    /// the backbone only ever sees `tile_size`-sized tiles per step;
    /// overlapping tiles are blended via a 2D Hann window. Lets the
    /// model produce 4K+ outputs without exceeding its trained
    /// working resolution.
    ///
    /// Supported on Flux (`flux-dev` / `flux-schnell`) and SD3 /
    /// SD3.5 in v0.22 phase 3 — SDXL tiled is a follow-up
    /// (needs the t2i::Pipeline cache path which isn't wired yet).
    /// SD 1.5 / 2.1 / Flux concept variants bail loud.
    pub tiled: bool,
    /// v0.22 phase 3: tile side length in pixels. Default 1024 (the
    /// SDXL native + Flux working scale). Must be a multiple of 8 on
    /// SD, 16 on Flux + SD3. Ignored when `tiled` is `false`.
    pub tile_size: u32,
    /// v0.22 phase 3: stride between tile origins. Smaller = more
    /// overlap = smoother seams + more compute. Default 768
    /// (`tile_size - tile_size/4`). Ignored when `tiled` is `false`.
    pub tile_stride: u32,
    /// v0.22 phase 4: global LoRA scale multiplier applied on top
    /// of each individual `plakat.lora.add` weight. Default 1.0.
    /// At 0.5 every LoRA's effective scale is halved.
    pub lora_scale: f32,
    /// v0.22 phase 6: same-model polish refine steps. `Some(N)`
    /// appends N extra denoise steps at low strength after the
    /// main loop, using the same model — useful for sharpening.
    /// `None` (the default) disables the polish pass.
    /// SD-family only; ignored on Flux + SD3.
    pub refine_steps: Option<usize>,
    /// v0.22 phase 6: polish-pass denoise strength in [0, 1].
    /// Default 0.3. Lower = subtler (mostly preserves the main
    /// pass); higher = more rework of the existing image.
    pub refine_strength: f32,
    /// v0.22 phase 6 + v0.23 phase 2: fraction of the schedule at
    /// which the SDXL refiner UNet takes over (default 0.8 = last
    /// 20% of steps). Wired into `t2i::GenRequest.refiner_frac`
    /// by [`super::script_entry::build_t2i_gen_request`]. Only
    /// meaningful when `plakat.refiner.enable` is in effect AND
    /// the loaded model is SDXL — otherwise the t2i pipeline's
    /// refiner_unet slot is None and this field is ignored.
    pub refiner_frac: f32,
    /// v0.22 phase 6: `plakat.style.*`-related strength multiplier
    /// in [0, 1]. Declared today; the apply / detect / clear / list
    /// words land in v0.23 phase 4 (catalog-integration scope is
    /// bigger than the v0.22 phase-6 budget). Documented as a
    /// known shipping-but-no-op key — same approach as Flux's
    /// `kontext_bucket` before phase 2 wired it.
    pub style_strength: f32,
    /// v0.23 phase 4: optional override of the style catalog
    /// directory. Empty (default) → CLI default
    /// `assets/style_catalog`. Set with `plakat.config.set
    /// "style_catalog" "path/to/catalog"`. Read by
    /// `plakat.style.apply` / `.detect` / `.list` at resolve time.
    pub style_catalog: String,
    /// v0.25 phase 8: skip remote LoRA discovery for `plakat.look.*`
    /// / `plakat.genre.*` (use cache + local scan only). Mirrors the
    /// CLI `--offline` flag. Default `false`. Set with
    /// `plakat.config.set "offline_discovery" "true"`.
    pub offline_discovery: bool,
    /// v0.22 phase 7: ADetailer face img2img strength in [0, 1].
    /// Default 0.4 (Auto1111's ADetailer default). Lower preserves
    /// identity / colour; higher = more rework. Ignored when
    /// `ctx.adetailer_enabled == false`.
    pub adetailer_strength: f32,
    /// v0.22 phase 7: bbox expansion fraction (each side) in [0, 1].
    /// Default 0.25 (25% on each side = 50% total dim growth).
    /// Trade-off: more = better blending, less = sharper detail.
    pub adetailer_padding: f32,
    /// v0.22 phase 7: feather fraction in [0, 1]. Default 0.25 —
    /// the outer 25% of the bbox fades 1.0 → 0.0. Larger = softer
    /// seam.
    pub adetailer_feather: f32,
    /// v0.22 phase 7: SCRFD confidence threshold in [0, 1]. Faces
    /// scored below this are skipped. Default 0.5 (InsightFace's).
    pub adetailer_confidence: f32,
    /// v0.22 phase 7: working resolution of the face img2img pass
    /// (square). Default 512 (SD 1.5 native); 1024 fits SDXL.
    /// Snapped to /8 by the img2img pipeline.
    pub adetailer_size: u32,
    /// v0.22 phase 7: prompt for the face pass. Default matches
    /// `adetailer::Config::defaults`. Set to a portrait-flavoured
    /// override (e.g. "detailed eyes, intricate iris, sharp
    /// focus") for stronger refinements.
    pub adetailer_prompt: String,
    /// v0.22 phase 8: hires-fix upscale factor in (1, 4]. Default
    /// 2.0 (matches `--hires-scale`). Ignored when the picked
    /// upscaler is ML (Real-ESRGAN: native 2× / 4×).
    pub hires_scale: f32,
    /// v0.22 phase 8: img2img strength on the upscaled image in
    /// [0, 1]. Default 0.5 — preserves the t2i composition while
    /// adding refinement. `0.7+` allows more reinterpretation.
    pub hires_strength: f32,
    /// v0.22 phase 8: upscaler token. Same grammar as
    /// `plakat upscale --method` / `--hires-upscaler`. Default
    /// `"lanczos"` (fast + sharp). `real-esrgan-x2` etc download
    /// weights on first use.
    pub hires_upscaler: String,
    /// v0.22 phase 8: optional step-count override for the refine
    /// pass. `None` (default) falls back to `config.steps`. The
    /// main-pass step count is usually a reasonable refine count;
    /// callers wanting a cheaper refine can drop this to ~12.
    pub hires_steps: Option<usize>,
    /// v0.22 phase 9: optional override of the artefact library
    /// directory. Empty (default) → CLI default
    /// `assets/artefact_library`. Set with `plakat.config.set
    /// "artefact_library" "path/to/library"`.
    pub artefact_library: String,
    /// v0.22 phase 9: img2img strength for the optional
    /// post-composite blend pass in [0, 1]. Default 0.3. Same
    /// semantics as `--artefact-blend-strength`. Only meaningful
    /// when `plakat.artefact.blend.enable` is in effect AND the
    /// artefact stack is non-empty.
    pub artefact_blend_strength: f32,
    /// v0.22 phase 9: enable smart-zones (depth + luminance) for
    /// per-image artefact placement. Default false (rigid grid).
    /// Mirrors `--smart-zones`. Loads the Depth-Anything-V2-Small
    /// checkpoint on first use; falls back to the grid with a
    /// warning on inference failure.
    pub artefact_smart_zones: bool,
    /// v0.22 phase 10: prompt-enhancer provider. Same grammar as
    /// `--enhance`: `"auto"` (default), `"deepseek"`, `"gemini"`,
    /// `"local"`, `"local:<alias>"` (e.g. `"local:smollm2-360m"`).
    /// Read at `plakat.enhance` time.
    pub enhance_provider: String,
    /// v0.22 phase 10: local-LLM sampling temperature. `None`
    /// (default) → greedy decode (reproducible). `Some(t)` with
    /// `t > 0` enables sampling. Ignored on DeepSeek / Gemini
    /// providers. Mirrors `--enhance-temp`.
    pub enhance_temp: Option<f64>,
    /// v0.22 phase 10: local-LLM max-new-tokens cap. `None`
    /// (default) → 96 (the same default as `--enhance-max-tokens`).
    /// Ignored on DeepSeek / Gemini.
    pub enhance_max_tokens: Option<usize>,
    /// v0.22 phase 10: opt-in disk cache for the local enhancer.
    /// SHA-256 of (alias, system, user, temp, max_tokens) keys an
    /// on-disk lookup at `~/.cache/plakat/enhance/`. Default
    /// false. Mirrors `--enhance-cache`. Ignored on the API
    /// providers.
    pub enhance_cache: bool,
    /// v0.22 phase 10: optional path to a custom enhancer system
    /// prompt file. Empty (default) → built-in
    /// `prompt::SYSTEM`. Mirrors `--enhance-system`. Applies to
    /// all three providers (the API providers honour the system
    /// override even though they ignore temp / max_tokens / cache).
    pub enhance_system: String,
    /// v0.22 phase 10: join the enhancer rewrite with the original
    /// prompt via the A1111 `BREAK` keyword so each chunk gets
    /// its own 77-token CLIP slot. Default false. SD-family only
    /// (Flux / SD3's T5 ignores BREAK; the flag warn-no-ops on
    /// those families when a model is loaded). Mirrors
    /// `--enhance-keep-original`.
    pub enhance_keep_original: bool,
    /// v0.22 phase 11: aspect ratio (`"16:9"`, `"1:1"`, `"2:3"`,
    /// etc). Empty (default) → no aspect resolution. When set
    /// and `size_explicit` is false, the working size becomes
    /// `aspect` × `base` (shorter side = `base`). Mirrors
    /// `--aspect`.
    pub aspect: String,
    /// v0.22 phase 11: base resolution for the shorter side
    /// (pixels) when `aspect` is set. Default 768 (same as CLI).
    /// Mirrors `--base`.
    pub base: u32,
    /// v0.22 phase 11: feather radius (pixels) applied to the
    /// img2img mask edge. Default 8 (same as CLI). Only
    /// meaningful when img2img is invoked with a mask.
    /// Wired into `img2img::Request.mask_feather` by
    /// `script_entry::img2img_or_inpaint_one` (v0.23 phase 5);
    /// the img2img pipeline honours it only when a mask is
    /// supplied via `plakat.inpaint`.
    pub mask_feather: u32,
    /// v0.22 phase 11: invert mask polarity (treat black as
    /// inpaint). Default false. Same wiring as `mask_feather`
    /// — fires through `plakat.inpaint` (v0.23 phase 5).
    pub mask_invert: bool,
    /// v0.22 phase 11 + v0.23 phase 3: CLIP-skip layer index.
    /// `1` (default) uses the last hidden state. `2` uses the
    /// penultimate (Auto1111 / NovelAI SD 1.5 anime default).
    /// SD 1.5 / SD 2.1 only — SDXL / Flux / SD3 ignore (SDXL
    /// already uses penultimate by training default; Flux + SD3
    /// use T5 and have no equivalent knob). Wired into
    /// `t2i::GenRequest.clip_skip` by
    /// [`super::script_entry::build_t2i_gen_request`]; t2i's
    /// `encode_prompt` reads it during the CLIP-L hidden-state
    /// pick. Honoured by `plakat.generate` only.
    pub clip_skip: usize,
    /// v0.22 phase 11: wildcard directory for `__name__` prompt
    /// expansion. Empty (default) → no file-wildcard expansion
    /// (inline `{a|b|c}` still works). Wired into
    /// [`super::script_entry::expand_prompt`], called at the top of
    /// `generate_one` / `img2img_one` / `portrait_one` so all
    /// three image-producing words honour it.
    pub wildcard_dir: String,
    /// v0.22 phase 11: bundled negative-prompt preset name. One
    /// of `photo` / `painting` / `anime` / `cinematic` (or any
    /// user-installed preset). Empty (default) → no preset.
    /// When set, the resolved preset text is comma-joined with
    /// `negative` at generate-request time (preset first, then
    /// user negative). Mirrors `--negative-preset`. Validated
    /// against `prompt::negative_presets::PRESETS` at config-set
    /// time.
    pub negative_preset: String,
    /// v0.24 phase 2: optional face bounding box override for
    /// `plakat.portrait`. CSV grammar `"x0,y0,x1,y1"` (4 floats
    /// in [0,1]; x0<x1, y0<y1). Mirrors `--face-bbox`. Empty
    /// string clears; non-empty validates at set-time via
    /// `cli::portrait::parse_face_bbox`. Threaded into
    /// `portrait::GenRequest.face_bbox` at request-build time.
    /// `face_landmarks` takes precedence when both are set.
    pub face_bbox: Option<[f32; 4]>,
    /// v0.24 phase 2: optional 5-point face landmarks override
    /// for `plakat.portrait`. CSV grammar
    /// `"LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY"` (10 floats in [0,1])
    /// — left_eye, right_eye, nose, left_mouth, right_mouth.
    /// Mirrors `--face-landmarks`. Empty string clears; non-empty
    /// validates at set-time via
    /// `cli::portrait::parse_face_landmarks`. Threaded into
    /// `portrait::GenRequest.face_landmarks`. **Takes precedence
    /// over `face_bbox` when both are set** — same precedence as
    /// the CLI's `--face-landmarks > --face-bbox` rule.
    pub face_landmarks: Option<[[f32; 2]; 5]>,
    /// v0.24 phase 3: optional identity-encoder override for
    /// `plakat.portrait`. Empty (default) → auto-pick by alias
    /// (`pick_sd_family_identity`'s rule: sd15 → PlusFace,
    /// sdxl → PlusFaceSdxl, sd21 → None). Non-empty values:
    /// `"plus-face"`, `"plus-face-sdxl"`, `"face-id"`,
    /// `"face-id-sdxl"` (validated via `IdentityKind::from_str`
    /// — accepts the same aliases as `--identity-kind`).
    /// Mutating this drops both SD slots via
    /// `mark_loras_changed` since identity is a load-time
    /// pipeline feature.
    pub identity_kind: String,
    /// `true` while the script hasn't called `plakat.config.set` for
    /// width/height yet. When still `true` at generate time,
    /// [`super::script_entry::generate_one`] picks the SD-family
    /// default for the loaded model (SDXL → 1024², everything else
    /// → 512²). Once the script sets either dim explicitly, this
    /// flips and the explicit values apply.
    pub size_explicit: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            // Mirror cli::generate defaults so scripts behave like
            // the CLI does for the same inputs.
            steps: 28,
            guidance: 7.5,
            seed: None,
            width: 0,  // sentinel — see `size_explicit`
            height: 0, // sentinel — see `size_explicit`
            negative: String::new(),
            scheduler: SchedulerKind::Default,
            strength: 0.75,
            face_strength: 0.8,
            quantize_t5: false,
            quant_level: None,
            t5_quant_level: None,
            fast: None,
            kontext_bucket: false,
            tiled: false,
            tile_size: 1024,
            tile_stride: 768,
            lora_scale: 1.0,
            refine_steps: None,
            refine_strength: 0.3,
            refiner_frac: 0.8,
            style_strength: 1.0,
            style_catalog: String::new(),
            offline_discovery: false,
            adetailer_strength: 0.4,
            adetailer_padding: 0.25,
            adetailer_feather: 0.25,
            adetailer_confidence: 0.5,
            adetailer_size: 512,
            adetailer_prompt: "detailed face, sharp focus, high quality".to_string(),
            hires_scale: 2.0,
            hires_strength: 0.5,
            hires_upscaler: "lanczos".to_string(),
            hires_steps: None,
            artefact_library: String::new(),
            artefact_blend_strength: 0.3,
            artefact_smart_zones: false,
            enhance_provider: "auto".to_string(),
            enhance_temp: None,
            enhance_max_tokens: None,
            enhance_cache: false,
            enhance_system: String::new(),
            enhance_keep_original: false,
            aspect: String::new(),
            base: 768,
            mask_feather: 8,
            mask_invert: false,
            clip_skip: 1,
            wildcard_dir: String::new(),
            negative_preset: String::new(),
            face_bbox: None,
            face_landmarks: None,
            identity_kind: String::new(),
            size_explicit: false,
        }
    }
}

impl GenerationConfig {
    /// Apply one `(key, value-string)` mutation. Returns `Err` on
    /// unknown key OR on a value that can't be parsed into the
    /// expected type. The host word renders user `Value`s to
    /// strings first (via the helpers in [`super::helpers`]); this
    /// keeps the value-parsing in one place and gives a uniform
    /// error message.
    pub fn set_str(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "steps" => {
                self.steps = parse_pos_int(value, key)? as usize;
            }
            "guidance" => {
                self.guidance = parse_finite_float(value, key)?;
            }
            "seed" => {
                self.seed = Some(parse_pos_int(value, key)?);
            }
            "width" => {
                self.width = parse_dim(value, key)?;
                self.size_explicit = true;
            }
            "height" => {
                self.height = parse_dim(value, key)?;
                self.size_explicit = true;
            }
            "negative" => {
                self.negative = value.to_string();
            }
            "scheduler" => {
                self.scheduler = value.parse::<SchedulerKind>()?;
            }
            "strength" => {
                self.strength = parse_unit_float(value, key)? as f32;
            }
            "face_strength" => {
                self.face_strength = parse_unit_float(value, key)? as f32;
            }
            "quantize_t5" => {
                self.quantize_t5 = parse_bool(value, key)?;
            }
            "quant_level" => {
                validate_flux_quant_level(value, "quant_level")?;
                self.quant_level = Some(value.to_string());
            }
            "t5_quant_level" => {
                validate_t5_quant_level(value, "t5_quant_level")?;
                self.t5_quant_level = Some(value.to_string());
            }
            "fast" => {
                validate_fast_preset(value)?;
                self.fast = Some(value.to_string());
            }
            "kontext_bucket" => {
                self.kontext_bucket = parse_bool(value, key)?;
            }
            "tiled" => {
                self.tiled = parse_bool(value, key)?;
            }
            "tile_size" => {
                self.tile_size = parse_tile_dim(value, key)?;
            }
            "tile_stride" => {
                self.tile_stride = parse_tile_dim(value, key)?;
            }
            "lora_scale" => {
                // 0.0 = no LoRA effect; > 1.0 amplifies. Cap at
                // 2.0 to avoid silently zeroing weights or
                // exploding gradients; matches the CLI's
                // `--lora-scale` documented range.
                let f = parse_finite_float(value, key)?;
                if !(0.0..=2.0).contains(&f) {
                    bail!(
                        "plakat.config.set: lora_scale must be in \
                         [0, 2] (got {f})"
                    );
                }
                self.lora_scale = f as f32;
            }
            "refine_steps" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 {
                    bail!(
                        "plakat.config.set: refine_steps must be > 0 (got 0). \
                         Use `plakat.refiner.disable` to skip the polish pass."
                    );
                }
                if n > 100 {
                    bail!(
                        "plakat.config.set: refine_steps {n} is past any \
                         practical polish budget. Try 4-16."
                    );
                }
                self.refine_steps = Some(n as usize);
            }
            "refine_strength" => {
                self.refine_strength = parse_unit_float(value, key)? as f32;
            }
            "refiner_frac" => {
                self.refiner_frac = parse_unit_float(value, key)? as f32;
            }
            "style_strength" => {
                self.style_strength = parse_unit_float(value, key)? as f32;
            }
            "adetailer_strength" | "adetailer_padding" | "adetailer_feather"
            | "adetailer_confidence" => {
                let f = parse_unit_float(value, key)? as f32;
                match key {
                    "adetailer_strength" => self.adetailer_strength = f,
                    "adetailer_padding" => self.adetailer_padding = f,
                    "adetailer_feather" => self.adetailer_feather = f,
                    "adetailer_confidence" => self.adetailer_confidence = f,
                    _ => unreachable!(),
                }
            }
            "adetailer_size" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 || n > 2048 {
                    bail!(
                        "plakat.config.set: adetailer_size must be in \
                         (0, 2048] (got {n})"
                    );
                }
                if n % 8 != 0 {
                    bail!(
                        "plakat.config.set: adetailer_size must be a \
                         multiple of 8 (VAE); got {n}"
                    );
                }
                self.adetailer_size = n as u32;
            }
            "adetailer_prompt" => {
                self.adetailer_prompt = value.to_string();
            }
            "hires_scale" => {
                let f = value.parse::<f32>().with_context(|| {
                    format!(
                        "plakat.config.set: hires_scale expects a float, \
                         got {value:?}"
                    )
                })?;
                if !(f > 1.0 && f <= 4.0) {
                    bail!(
                        "plakat.config.set: hires_scale must be in (1, 4] \
                         (got {f})"
                    );
                }
                self.hires_scale = f;
            }
            "hires_strength" => {
                let f = parse_unit_float(value, key)? as f32;
                self.hires_strength = f;
            }
            "hires_upscaler" => {
                // Validate now so the failure surfaces at config-set
                // time instead of generate time. We don't store the
                // parsed Method (keeps the config plain-data); the
                // post-process call re-parses.
                use std::str::FromStr;
                crate::imaging::upscale::Method::from_str(value)
                    .with_context(|| {
                        format!(
                            "plakat.config.set: hires_upscaler {value:?} not \
                             recognised. Accepted: nearest, bilinear, bicubic, \
                             lanczos, real-esrgan-x2, real-esrgan-x4, \
                             real-esrgan-anime-x4"
                        )
                    })?;
                self.hires_upscaler = value.to_string();
            }
            "hires_steps" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 || n > 500 {
                    bail!(
                        "plakat.config.set: hires_steps must be in (0, 500] \
                         (got {n})"
                    );
                }
                self.hires_steps = Some(n as usize);
            }
            "artefact_library" => {
                // Empty resets to default; non-empty is stored as-is
                // (path validation happens at generate time when the
                // library actually loads).
                self.artefact_library = value.to_string();
            }
            "style_catalog" => {
                // v0.23 phase 4: same shape as artefact_library —
                // empty resets to default. The catalog directory is
                // loaded lazily inside the plakat.style.* words.
                self.style_catalog = value.to_string();
            }
            "offline_discovery" => {
                // v0.25 phase 8: mirrors the CLI --offline flag for
                // plakat.look.* / plakat.genre.* auto-discovery.
                // Accepts "true"/"false" / "1"/"0" / "yes"/"no" via
                // parse_bool below.
                self.offline_discovery = parse_bool(value, key)?;
            }
            "artefact_blend_strength" => {
                let f = parse_unit_float(value, key)? as f32;
                self.artefact_blend_strength = f;
            }
            "artefact_smart_zones" => {
                self.artefact_smart_zones = parse_bool(value, key)?;
            }
            "enhance_provider" => {
                // Validate against the same grammar `prompt::enhance_with_args`
                // accepts. The bail surfaces at config-set time so a typo
                // doesn't wait until plakat.enhance to fail.
                let v = value.to_lowercase();
                let ok = matches!(v.as_str(), "auto" | "deepseek" | "gemini" | "local")
                    || v.starts_with("local:");
                if !ok {
                    bail!(
                        "plakat.config.set: enhance_provider {value:?} not \
                         recognised. Accepted: auto, deepseek, gemini, local, \
                         local:<alias> (e.g. local:smollm2-360m)"
                    );
                }
                self.enhance_provider = v;
            }
            "enhance_temp" => {
                let f = parse_finite_float(value, key)?;
                if !(0.0..=2.0).contains(&f) {
                    bail!(
                        "plakat.config.set: enhance_temp must be in [0, 2] \
                         (got {f})"
                    );
                }
                self.enhance_temp = Some(f);
            }
            "enhance_max_tokens" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 || n > 1024 {
                    bail!(
                        "plakat.config.set: enhance_max_tokens must be in \
                         (0, 1024] (got {n})"
                    );
                }
                self.enhance_max_tokens = Some(n as usize);
            }
            "enhance_cache" => {
                self.enhance_cache = parse_bool(value, key)?;
            }
            "enhance_system" => {
                self.enhance_system = value.to_string();
            }
            "enhance_keep_original" => {
                self.enhance_keep_original = parse_bool(value, key)?;
            }
            "aspect" => {
                // Empty resets (clear the aspect override). Non-empty
                // must parse as `W:H` with positive integers.
                if !value.is_empty() {
                    let (w, h) = value.split_once(':').ok_or_else(|| {
                        anyhow!(
                            "plakat.config.set: aspect {value:?} must be \
                             `W:H` with positive integers (e.g. 16:9)"
                        )
                    })?;
                    let w_n: u32 = w.parse().with_context(|| {
                        format!(
                            "plakat.config.set: aspect {value:?} width \
                             must be an integer"
                        )
                    })?;
                    let h_n: u32 = h.parse().with_context(|| {
                        format!(
                            "plakat.config.set: aspect {value:?} height \
                             must be an integer"
                        )
                    })?;
                    if w_n == 0 || h_n == 0 {
                        bail!(
                            "plakat.config.set: aspect {value:?} components \
                             must be > 0"
                        );
                    }
                }
                self.aspect = value.to_string();
            }
            "base" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 || n > 4096 {
                    bail!(
                        "plakat.config.set: base must be in (0, 4096] \
                         (got {n})"
                    );
                }
                if n % 8 != 0 {
                    bail!(
                        "plakat.config.set: base must be a multiple of 8 \
                         (VAE); got {n}"
                    );
                }
                self.base = n as u32;
            }
            "mask_feather" => {
                let n = parse_pos_int(value, key)?;
                if n > 256 {
                    bail!(
                        "plakat.config.set: mask_feather must be in [0, 256] \
                         pixels (got {n})"
                    );
                }
                self.mask_feather = n as u32;
            }
            "mask_invert" => {
                self.mask_invert = parse_bool(value, key)?;
            }
            "clip_skip" => {
                let n = parse_pos_int(value, key)?;
                if n == 0 || n > 12 {
                    bail!(
                        "plakat.config.set: clip_skip must be in [1, 12] \
                         (got {n}). Common values: 1 (default), 2 (SD 1.5 \
                         anime)."
                    );
                }
                self.clip_skip = n as usize;
            }
            "wildcard_dir" => {
                self.wildcard_dir = value.to_string();
            }
            "negative_preset" => {
                if !value.is_empty() {
                    // Validate against built-in + user-installed presets.
                    let valid = crate::prompt::negative_presets::resolve(value)
                        .is_some();
                    if !valid {
                        bail!(
                            "plakat.config.set: negative_preset {value:?} not \
                             recognised. Supported: {}",
                            crate::prompt::negative_presets::supported_names()
                        );
                    }
                }
                self.negative_preset = value.to_string();
            }
            "face_bbox" => {
                // v0.24 phase 2: empty clears; non-empty validates
                // via the CLI's parse_face_bbox (4-CSV grammar).
                if value.is_empty() {
                    self.face_bbox = None;
                } else {
                    let parsed =
                        crate::cli::portrait::parse_face_bbox(value).map_err(|e| {
                            anyhow!(
                                "plakat.config.set: face_bbox {value:?}: {e}"
                            )
                        })?;
                    self.face_bbox = Some(parsed);
                }
            }
            "face_landmarks" => {
                // v0.24 phase 2: empty clears; non-empty validates
                // via the CLI's parse_face_landmarks (10-CSV grammar,
                // 5 landmark pairs).
                if value.is_empty() {
                    self.face_landmarks = None;
                } else {
                    let parsed = crate::cli::portrait::parse_face_landmarks(value)
                        .map_err(|e| {
                            anyhow!(
                                "plakat.config.set: face_landmarks {value:?}: {e}"
                            )
                        })?;
                    self.face_landmarks = Some(parsed);
                }
            }
            "identity_kind" => {
                // v0.24 phase 3: empty resets to auto-pick; non-empty
                // validates via IdentityKind::FromStr. The stored
                // string keeps the user's canonical form ("plus-face"
                // etc.) since pick_sd_family_identity re-parses it.
                if !value.is_empty() {
                    use std::str::FromStr;
                    crate::pipelines::ip_adapter::IdentityKind::from_str(value)
                        .map_err(|e| {
                            anyhow!(
                                "plakat.config.set: identity_kind {value:?}: \
                                 {e}. Accepted: plus-face, plus-face-sdxl, \
                                 face-id, face-id-sdxl (plus aliases — see \
                                 IdentityKind::from_str)."
                            )
                        })?;
                }
                self.identity_kind = value.to_string();
            }
            other => {
                return Err(anyhow!(
                    "plakat.config.set: unknown key {other:?}. \
                     Supported keys: steps, guidance, seed, width, \
                     height, negative, scheduler, strength, \
                     face_strength, quantize_t5, quant_level, \
                     t5_quant_level, fast, kontext_bucket, tiled, \
                     tile_size, tile_stride, lora_scale, \
                     refine_steps, refine_strength, refiner_frac, \
                     style_strength, adetailer_strength, \
                     adetailer_padding, adetailer_feather, \
                     adetailer_confidence, adetailer_size, \
                     adetailer_prompt, hires_scale, hires_strength, \
                     hires_upscaler, hires_steps, artefact_library, \
                     artefact_blend_strength, artefact_smart_zones, \
                     enhance_provider, enhance_temp, enhance_max_tokens, \
                     enhance_cache, enhance_system, enhance_keep_original, \
                     aspect, base, mask_feather, mask_invert, clip_skip, \
                     wildcard_dir, negative_preset, style_catalog, \
                     face_bbox, face_landmarks, identity_kind, \
                     offline_discovery."
                ));
            }
        }
        Ok(())
    }

    /// Apply an integer key directly (avoids the string round-trip
    /// when the script pushed an int). Falls back to `set_str` for
    /// keys that don't accept ints.
    pub fn set_int(&mut self, key: &str, value: i64) -> Result<()> {
        match key {
            "steps" | "guidance" | "seed" | "width" | "height"
            | "strength" | "face_strength" | "tile_size" | "tile_stride"
            | "lora_scale" | "refine_steps" | "refine_strength"
            | "refiner_frac" | "style_strength"
            | "adetailer_strength" | "adetailer_padding"
            | "adetailer_feather" | "adetailer_confidence"
            | "adetailer_size"
            | "hires_scale" | "hires_strength" | "hires_steps"
            | "artefact_blend_strength" | "enhance_temp"
            | "enhance_max_tokens" | "base" | "mask_feather"
            | "clip_skip" => {
                self.set_str(key, &value.to_string())
            }
            "quantize_t5" | "kontext_bucket" | "tiled"
            | "artefact_smart_zones" | "enhance_cache"
            | "enhance_keep_original" | "mask_invert" => {
                // Permissive bool ↔ int: accept 0 / 1 only.
                match value {
                    0 => self.set_str(key, "false"),
                    1 => self.set_str(key, "true"),
                    _ => Err(anyhow!(
                        "plakat.config.set: key {key:?} expects a bool \
                         (true/false or 0/1); got integer {value}"
                    )),
                }
            }
            "negative" | "scheduler" | "adetailer_prompt"
            | "hires_upscaler" | "artefact_library"
            | "enhance_provider" | "enhance_system"
            | "aspect" | "wildcard_dir" | "negative_preset"
            | "style_catalog" | "face_bbox" | "face_landmarks"
            | "identity_kind" => Err(anyhow!(
                "plakat.config.set: key {key:?} expects a string value, got integer {value}"
            )),
            other => Err(anyhow!(
                "plakat.config.set: unknown key {other:?}. \
                 Supported keys: steps, guidance, seed, width, \
                 height, negative, scheduler, strength, \
                 face_strength, quantize_t5, quant_level, \
                 t5_quant_level, fast, kontext_bucket, tiled, \
                 tile_size, tile_stride."
            )),
        }
    }

    /// Same as [`set_int`] for floats.
    pub fn set_float(&mut self, key: &str, value: f64) -> Result<()> {
        match key {
            "guidance" => {
                if !value.is_finite() {
                    bail!(
                        "plakat.config.set: guidance {value} isn't finite \
                         (NaN / Infinity / -Infinity rejected)"
                    );
                }
                self.guidance = value;
                Ok(())
            }
            "strength" => {
                if !value.is_finite() {
                    bail!(
                        "plakat.config.set: strength {value} isn't finite"
                    );
                }
                if !(0.0..=1.0).contains(&value) {
                    bail!(
                        "plakat.config.set: strength must be in [0, 1] \
                         (got {value})"
                    );
                }
                self.strength = value as f32;
                Ok(())
            }
            "face_strength" => {
                if !value.is_finite() {
                    bail!(
                        "plakat.config.set: face_strength {value} isn't finite"
                    );
                }
                if !(0.0..=1.0).contains(&value) {
                    bail!(
                        "plakat.config.set: face_strength must be in [0, 1] \
                         (got {value})"
                    );
                }
                self.face_strength = value as f32;
                Ok(())
            }
            "lora_scale" => {
                if !value.is_finite() {
                    bail!("plakat.config.set: lora_scale {value} isn't finite");
                }
                if !(0.0..=2.0).contains(&value) {
                    bail!(
                        "plakat.config.set: lora_scale must be in [0, 2] \
                         (got {value})"
                    );
                }
                self.lora_scale = value as f32;
                Ok(())
            }
            "refine_strength" | "refiner_frac" | "style_strength"
            | "adetailer_strength" | "adetailer_padding"
            | "adetailer_feather" | "adetailer_confidence"
            | "hires_strength" | "artefact_blend_strength" => {
                if !value.is_finite() {
                    bail!(
                        "plakat.config.set: {key} {value} isn't finite"
                    );
                }
                if !(0.0..=1.0).contains(&value) {
                    bail!(
                        "plakat.config.set: {key} must be in [0, 1] (got {value})"
                    );
                }
                match key {
                    "refine_strength" => self.refine_strength = value as f32,
                    "refiner_frac" => self.refiner_frac = value as f32,
                    "style_strength" => self.style_strength = value as f32,
                    "adetailer_strength" => self.adetailer_strength = value as f32,
                    "adetailer_padding" => self.adetailer_padding = value as f32,
                    "adetailer_feather" => self.adetailer_feather = value as f32,
                    "adetailer_confidence" => self.adetailer_confidence = value as f32,
                    "hires_strength" => self.hires_strength = value as f32,
                    "artefact_blend_strength" => self.artefact_blend_strength = value as f32,
                    _ => unreachable!(),
                }
                Ok(())
            }
            "hires_scale" => {
                if !value.is_finite() {
                    bail!("plakat.config.set: hires_scale {value} isn't finite");
                }
                if !(value > 1.0 && value <= 4.0) {
                    bail!(
                        "plakat.config.set: hires_scale must be in (1, 4] \
                         (got {value})"
                    );
                }
                self.hires_scale = value as f32;
                Ok(())
            }
            "enhance_temp" => {
                if !value.is_finite() {
                    bail!("plakat.config.set: enhance_temp {value} isn't finite");
                }
                if !(0.0..=2.0).contains(&value) {
                    bail!(
                        "plakat.config.set: enhance_temp must be in [0, 2] \
                         (got {value})"
                    );
                }
                self.enhance_temp = Some(value);
                Ok(())
            }
            "steps" | "seed" | "width" | "height" | "refine_steps"
            | "adetailer_size" | "hires_steps" | "enhance_max_tokens"
            | "base" | "mask_feather" | "clip_skip" => {
                // Permissive: round int-valued floats so `7.0` → 7.
                // Strictly-non-integer floats are an error.
                if value.fract() != 0.0 {
                    bail!(
                        "plakat.config.set: key {key:?} expects an integer; \
                         got float {value} with non-zero fractional part"
                    );
                }
                self.set_int(key, value as i64)
            }
            "negative" | "scheduler" | "adetailer_prompt"
            | "hires_upscaler" | "artefact_library"
            | "enhance_provider" | "enhance_system"
            | "aspect" | "wildcard_dir" | "negative_preset"
            | "style_catalog" | "face_bbox" | "face_landmarks"
            | "identity_kind" => Err(anyhow!(
                "plakat.config.set: key {key:?} expects a string value, got float {value}"
            )),
            other => Err(anyhow!(
                "plakat.config.set: unknown key {other:?}. \
                 Supported keys: steps, guidance, seed, width, \
                 height, negative, scheduler, strength, \
                 face_strength, quantize_t5, quant_level, \
                 t5_quant_level, fast, kontext_bucket, tiled, \
                 tile_size, tile_stride."
            )),
        }
    }
}

fn parse_pos_int(s: &str, key: &str) -> Result<u64> {
    let n: i64 = s
        .parse()
        .map_err(|e| anyhow!("plakat.config.set: {key} = {s:?} isn't an integer ({e})"))?;
    if n < 0 {
        bail!("plakat.config.set: {key} must be >= 0 (got {n})");
    }
    Ok(n as u64)
}

fn parse_finite_float(s: &str, key: &str) -> Result<f64> {
    let f: f64 = s
        .parse()
        .map_err(|e| anyhow!("plakat.config.set: {key} = {s:?} isn't a number ({e})"))?;
    if !f.is_finite() {
        bail!("plakat.config.set: {key} {f} isn't finite");
    }
    Ok(f)
}

fn parse_unit_float(s: &str, key: &str) -> Result<f64> {
    let f = parse_finite_float(s, key)?;
    if !(0.0..=1.0).contains(&f) {
        bail!("plakat.config.set: {key} must be in [0, 1] (got {f})");
    }
    Ok(f)
}

fn parse_bool(s: &str, key: &str) -> Result<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(anyhow!(
            "plakat.config.set: {key} = {s:?} isn't a bool (try true/false or 1/0)"
        )),
    }
}

/// v0.22 phase 2: validate a Flux transformer GGUF quant level
/// against the published city96 list. Same check `pipelines::flux`
/// applies at load time — we surface the error earlier (at
/// `plakat.config.set` time) so scripts fail loudly the moment
/// they typo a quant string.
fn validate_flux_quant_level(value: &str, key: &str) -> Result<()> {
    let allowed = crate::pipelines::flux::FLUX_QUANT_LEVELS;
    if allowed.iter().any(|l| l.eq_ignore_ascii_case(value)) {
        Ok(())
    } else {
        Err(anyhow!(
            "plakat.config.set: {key} = {value:?} isn't a published Flux \
             GGUF quant level. Supported: {}",
            allowed.join(", ")
        ))
    }
}

fn validate_t5_quant_level(value: &str, key: &str) -> Result<()> {
    let allowed = crate::pipelines::flux::T5_QUANT_LEVELS;
    if allowed.iter().any(|l| l.eq_ignore_ascii_case(value)) {
        Ok(())
    } else {
        Err(anyhow!(
            "plakat.config.set: {key} = {value:?} isn't a published T5 \
             GGUF quant level. Supported: {}",
            allowed.join(", ")
        ))
    }
}

/// v0.22 phase 2: validate a `--fast` preset name. Accepts the
/// five published presets; bails with the supported list on
/// anything else.
fn validate_fast_preset(value: &str) -> Result<()> {
    const VALID: &[&str] =
        &["hyper-8", "hyper-16", "turbo-alpha", "lcm-sdxl", "lcm-sd15"];
    if VALID.iter().any(|p| p.eq_ignore_ascii_case(value)) {
        return Ok(());
    }
    bail!(
        "plakat.config.set: fast = {value:?} not recognised. Supported: \
         {}",
        VALID.join(", ")
    )
}

/// v0.22 phase 3: tile-size / tile-stride validator. Same as
/// `parse_dim` but the upper bound is 4096 + must be a multiple
/// of 16 (Flux + SD3's patching granularity — strictest of the
/// three families). SD's relaxed /8 constraint isn't worth its
/// own validator yet because v0.22 phase 3 only ships tiled on
/// Flux + SD3.
fn parse_tile_dim(s: &str, key: &str) -> Result<u32> {
    let n = parse_pos_int(s, key)?;
    if n == 0 {
        bail!("plakat.config.set: {key} must be > 0 (got 0)");
    }
    if n % 16 != 0 {
        bail!(
            "plakat.config.set: {key} must be a multiple of 16 \
             (Flux + SD3 patching constraint); got {n}"
        );
    }
    if n > 4096 {
        bail!(
            "plakat.config.set: {key} {n} > 4096 is past any \
             practical tile size"
        );
    }
    Ok(n as u32)
}

fn parse_dim(s: &str, key: &str) -> Result<u32> {
    let n = parse_pos_int(s, key)?;
    if n == 0 {
        bail!("plakat.config.set: {key} must be > 0 (got 0)");
    }
    if n % 8 != 0 {
        bail!(
            "plakat.config.set: {key} must be a multiple of 8 (VAE constraint); got {n}"
        );
    }
    if n > 4096 {
        bail!(
            "plakat.config.set: {key} {n} > 4096 is well past any \
             practical SD/SDXL size; pass --tiled at the CLI level \
             if you really mean it"
        );
    }
    Ok(n as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_cli_defaults() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.steps, 28);
        assert!((cfg.guidance - 7.5).abs() < 1e-9);
        assert!(cfg.seed.is_none());
        assert_eq!(cfg.negative, "");
        assert!(matches!(cfg.scheduler, SchedulerKind::Default));
        // size_explicit starts false so script_entry picks the
        // model-family default at generate time.
        assert!(!cfg.size_explicit);
    }

    #[test]
    fn set_int_accepts_integer_keys() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("steps", 50).unwrap();
        cfg.set_int("seed", 42).unwrap();
        cfg.set_int("width", 512).unwrap();
        cfg.set_int("height", 768).unwrap();
        assert_eq!(cfg.steps, 50);
        assert_eq!(cfg.seed, Some(42));
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 768);
        assert!(cfg.size_explicit);
    }

    #[test]
    fn set_int_rejects_string_keys() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_int("negative", 0).is_err());
        assert!(cfg.set_int("scheduler", 0).is_err());
    }

    #[test]
    fn set_float_accepts_guidance_and_rounded_ints() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("guidance", 3.5).unwrap();
        cfg.set_float("steps", 40.0).unwrap();
        assert!((cfg.guidance - 3.5).abs() < 1e-9);
        assert_eq!(cfg.steps, 40);
    }

    #[test]
    fn set_float_rejects_non_integer_for_int_keys() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_float("steps", 40.5).unwrap_err();
        assert!(format!("{err}").contains("fractional"));
    }

    #[test]
    fn set_float_rejects_nan_guidance() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_float("guidance", f64::NAN).is_err());
        assert!(cfg.set_float("guidance", f64::INFINITY).is_err());
    }

    #[test]
    fn set_str_dim_must_be_multiple_of_eight() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("width", "513").unwrap_err();
        assert!(format!("{err}").contains("multiple of 8"));
    }

    #[test]
    fn set_str_dim_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("width", "0").is_err());
        assert!(cfg.set_str("width", "8000").is_err());
    }

    #[test]
    fn set_str_scheduler_parses() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("scheduler", "euler-a").unwrap();
        assert!(matches!(cfg.scheduler, SchedulerKind::EulerA));
        cfg.set_str("scheduler", "dpmpp-2m").unwrap();
        assert!(matches!(cfg.scheduler, SchedulerKind::DpmppKarras));
    }

    #[test]
    fn set_str_unknown_scheduler_bails_with_supported_list() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("scheduler", "not-a-real-scheduler").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown scheduler"), "got {msg}");
    }

    #[test]
    fn set_str_negative_is_passthrough_string() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("negative", "blurry, low quality").unwrap();
        assert_eq!(cfg.negative, "blurry, low quality");
    }

    #[test]
    fn set_str_unknown_key_lists_supported_keys() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("definitely_not_a_key", "x").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown key"), "got {msg}");
        assert!(msg.contains("steps"), "got {msg}");
        assert!(msg.contains("scheduler"), "got {msg}");
    }

    #[test]
    fn set_str_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("strength", "0.0").unwrap();
        assert!((cfg.strength - 0.0).abs() < 1e-9);
        cfg.set_str("strength", "1.0").unwrap();
        assert!((cfg.strength - 1.0).abs() < 1e-9);
        cfg.set_str("strength", "0.55").unwrap();
        assert!((cfg.strength - 0.55).abs() < 1e-6);
    }

    #[test]
    fn set_str_strength_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("strength", "-0.1").is_err());
        assert!(cfg.set_str("strength", "1.1").is_err());
        assert!(cfg.set_str("strength", "2.0").is_err());
    }

    #[test]
    fn set_float_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("strength", 0.5).unwrap();
        assert!((cfg.strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn set_float_strength_rejects_out_of_range_and_nan() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_float("strength", -0.01).is_err());
        assert!(cfg.set_float("strength", 1.01).is_err());
        assert!(cfg.set_float("strength", f64::NAN).is_err());
    }

    #[test]
    fn set_int_strength_accepts_zero_and_one() {
        // The int path routes through set_str, which accepts "0" + "1".
        let mut cfg = GenerationConfig::default();
        cfg.set_int("strength", 0).unwrap();
        cfg.set_int("strength", 1).unwrap();
        assert_eq!(cfg.strength, 1.0);
    }

    #[test]
    fn default_strength_matches_cli_default() {
        let cfg = GenerationConfig::default();
        assert!((cfg.strength - 0.75).abs() < 1e-9);
    }

    #[test]
    fn default_face_strength_matches_cli_default() {
        let cfg = GenerationConfig::default();
        assert!((cfg.face_strength - 0.8).abs() < 1e-9);
    }

    #[test]
    fn set_str_face_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("face_strength", "0.0").unwrap();
        assert!((cfg.face_strength - 0.0).abs() < 1e-9);
        cfg.set_str("face_strength", "1.0").unwrap();
        assert!((cfg.face_strength - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_str_face_strength_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("face_strength", "-0.1").is_err());
        assert!(cfg.set_str("face_strength", "1.5").is_err());
    }

    #[test]
    fn set_float_face_strength_rejects_nan_and_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_float("face_strength", -0.01).is_err());
        assert!(cfg.set_float("face_strength", 1.01).is_err());
        assert!(cfg.set_float("face_strength", f64::NAN).is_err());
    }

    // v0.22 phase 2: Flux-specific D-keys.

    #[test]
    fn set_str_quantize_t5_accepts_bool_forms() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("quantize_t5", "true").unwrap();
        assert!(cfg.quantize_t5);
        cfg.set_str("quantize_t5", "false").unwrap();
        assert!(!cfg.quantize_t5);
        cfg.set_str("quantize_t5", "1").unwrap();
        assert!(cfg.quantize_t5);
        cfg.set_str("quantize_t5", "0").unwrap();
        assert!(!cfg.quantize_t5);
        cfg.set_str("quantize_t5", "yes").unwrap();
        assert!(cfg.quantize_t5);
        cfg.set_str("quantize_t5", "on").unwrap();
        assert!(cfg.quantize_t5);
    }

    #[test]
    fn set_str_quantize_t5_rejects_garbage() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("quantize_t5", "maybe").is_err());
        assert!(cfg.set_str("quantize_t5", "2").is_err());
    }

    #[test]
    fn set_int_quantize_t5_accepts_zero_and_one_only() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("quantize_t5", 1).unwrap();
        assert!(cfg.quantize_t5);
        cfg.set_int("quantize_t5", 0).unwrap();
        assert!(!cfg.quantize_t5);
        // Anything else bails.
        assert!(cfg.set_int("quantize_t5", 2).is_err());
        assert!(cfg.set_int("quantize_t5", -1).is_err());
    }

    #[test]
    fn set_str_quant_level_accepts_published_values() {
        let mut cfg = GenerationConfig::default();
        for level in &["Q4_K_S", "Q8_0", "F16", "Q6_K"] {
            cfg.set_str("quant_level", level).unwrap_or_else(|e| {
                panic!("quant level {level:?} should be accepted: {e}")
            });
        }
        // Case-insensitive.
        cfg.set_str("quant_level", "q4_k_s").unwrap();
    }

    #[test]
    fn set_str_quant_level_rejects_unknown() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("quant_level", "Q1_K").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("isn't a published Flux"), "got {msg}");
        assert!(msg.contains("Q4_K_S"), "got {msg}");
    }

    #[test]
    fn set_str_fast_accepts_published_presets() {
        let mut cfg = GenerationConfig::default();
        for preset in &[
            "hyper-8",
            "hyper-16",
            "turbo-alpha",
            "lcm-sdxl",
            "lcm-sd15",
        ] {
            cfg.set_str("fast", preset).unwrap_or_else(|e| {
                panic!("preset {preset:?} should be accepted: {e}")
            });
        }
    }

    #[test]
    fn set_str_fast_rejects_unknown_preset() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("fast", "ultra-9000").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not recognised"), "got {msg}");
        assert!(msg.contains("hyper-8"), "got {msg}");
    }

    #[test]
    fn set_str_kontext_bucket_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("kontext_bucket", "true").unwrap();
        assert!(cfg.kontext_bucket);
        cfg.set_str("kontext_bucket", "false").unwrap();
        assert!(!cfg.kontext_bucket);
    }

    #[test]
    fn unknown_key_error_lists_new_v022_keys() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("definitely-not-a-key", "x").unwrap_err();
        let msg = format!("{err}");
        // Phases 2 + 3 added these; the error message should
        // advertise them so users can self-correct on typos.
        for new_key in &[
            "quantize_t5",
            "quant_level",
            "t5_quant_level",
            "fast",
            "kontext_bucket",
            "tiled",
            "tile_size",
            "tile_stride",
            "lora_scale",
            "refine_steps",
            "refine_strength",
            "refiner_frac",
            "style_strength",
            // Phase 7 adetailer keys:
            "adetailer_strength",
            "adetailer_padding",
            "adetailer_feather",
            "adetailer_confidence",
            "adetailer_size",
            "adetailer_prompt",
            // Phase 8 hires keys:
            "hires_scale",
            "hires_strength",
            "hires_upscaler",
            "hires_steps",
            // Phase 9 artefact keys:
            "artefact_library",
            "artefact_blend_strength",
            "artefact_smart_zones",
            // Phase 10 enhance keys:
            "enhance_provider",
            "enhance_temp",
            "enhance_max_tokens",
            "enhance_cache",
            "enhance_system",
            "enhance_keep_original",
            // Phase 11 misc keys:
            "aspect",
            "base",
            "mask_feather",
            "mask_invert",
            "clip_skip",
            "wildcard_dir",
            "negative_preset",
            // v0.23 phase 4 style key:
            "style_catalog",
            // v0.24 phase 2 face keys:
            "face_bbox",
            "face_landmarks",
            // v0.24 phase 3 identity key:
            "identity_kind",
        ] {
            assert!(
                msg.contains(new_key),
                "key {new_key:?} should be in the supported-keys list: {msg}"
            );
        }
    }

    #[test]
    fn defaults_for_v022_d_keys() {
        let cfg = GenerationConfig::default();
        assert!(!cfg.quantize_t5);
        assert!(cfg.quant_level.is_none());
        assert!(cfg.t5_quant_level.is_none());
        assert!(cfg.fast.is_none());
        assert!(!cfg.kontext_bucket);
        // Phase 3 D-keys:
        assert!(!cfg.tiled);
        assert_eq!(cfg.tile_size, 1024);
        assert_eq!(cfg.tile_stride, 768);
    }

    // v0.22 phase 3: tiled D-keys.

    #[test]
    fn set_str_tiled_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("tiled", "true").unwrap();
        assert!(cfg.tiled);
        cfg.set_str("tiled", "false").unwrap();
        assert!(!cfg.tiled);
    }

    #[test]
    fn set_int_tiled_accepts_zero_and_one() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("tiled", 1).unwrap();
        assert!(cfg.tiled);
        cfg.set_int("tiled", 0).unwrap();
        assert!(!cfg.tiled);
        assert!(cfg.set_int("tiled", 2).is_err());
    }

    #[test]
    fn set_str_tile_size_accepts_multiple_of_16() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("tile_size", "512").unwrap();
        assert_eq!(cfg.tile_size, 512);
        cfg.set_str("tile_size", "1024").unwrap();
        assert_eq!(cfg.tile_size, 1024);
        cfg.set_str("tile_size", "768").unwrap();
        assert_eq!(cfg.tile_size, 768);
    }

    #[test]
    fn set_str_tile_size_rejects_non_multiple_of_16() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("tile_size", "513").unwrap_err();
        assert!(format!("{err}").contains("multiple of 16"));
    }

    #[test]
    fn set_str_tile_size_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("tile_size", "0").is_err());
        assert!(cfg.set_str("tile_size", "8000").is_err());
    }

    #[test]
    fn set_int_tile_stride_accepts_768() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("tile_stride", 768).unwrap();
        assert_eq!(cfg.tile_stride, 768);
    }

    // v0.22 phase 4: lora_scale config key.

    #[test]
    fn default_lora_scale_is_one() {
        let cfg = GenerationConfig::default();
        assert!((cfg.lora_scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_str_lora_scale_accepts_zero_to_two() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("lora_scale", "0.0").unwrap();
        assert!((cfg.lora_scale - 0.0).abs() < 1e-9);
        cfg.set_str("lora_scale", "1.5").unwrap();
        assert!((cfg.lora_scale - 1.5).abs() < 1e-6);
        cfg.set_str("lora_scale", "2.0").unwrap();
        assert!((cfg.lora_scale - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_str_lora_scale_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("lora_scale", "-0.1").is_err());
        assert!(cfg.set_str("lora_scale", "2.1").is_err());
    }

    #[test]
    fn set_float_lora_scale_accepts_unit_and_amplified() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("lora_scale", 0.5).unwrap();
        assert!((cfg.lora_scale - 0.5).abs() < 1e-6);
    }

    // v0.22 phase 6: refiner + style config keys.

    #[test]
    fn defaults_for_phase_6_keys() {
        let cfg = GenerationConfig::default();
        assert!(cfg.refine_steps.is_none());
        assert!((cfg.refine_strength - 0.3).abs() < 1e-6);
        assert!((cfg.refiner_frac - 0.8).abs() < 1e-6);
        assert!((cfg.style_strength - 1.0).abs() < 1e-6);
    }

    #[test]
    fn set_str_refine_steps_accepts_positive_int() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("refine_steps", "8").unwrap();
        assert_eq!(cfg.refine_steps, Some(8));
        cfg.set_str("refine_steps", "16").unwrap();
        assert_eq!(cfg.refine_steps, Some(16));
    }

    #[test]
    fn set_str_refine_steps_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("refine_steps", "0").unwrap_err();
        assert!(format!("{err}").contains("must be > 0"));
        assert!(cfg.set_str("refine_steps", "999").is_err());
    }

    #[test]
    fn set_str_refine_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("refine_strength", "0.5").unwrap();
        assert!((cfg.refine_strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn set_str_refine_strength_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("refine_strength", "1.5").is_err());
    }

    #[test]
    fn set_str_refiner_frac_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("refiner_frac", "0.75").unwrap();
        assert!((cfg.refiner_frac - 0.75).abs() < 1e-6);
    }

    #[test]
    fn set_str_style_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("style_strength", "0.6").unwrap();
        assert!((cfg.style_strength - 0.6).abs() < 1e-6);
    }

    #[test]
    fn set_float_refine_strength_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("refine_strength", 0.4).unwrap();
        assert!((cfg.refine_strength - 0.4).abs() < 1e-6);
    }

    #[test]
    fn set_int_refine_steps_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("refine_steps", 10).unwrap();
        assert_eq!(cfg.refine_steps, Some(10));
    }

    #[test]
    fn set_int_seed_rejects_negative() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("seed", -1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(">= 0"), "got {msg}");
    }

    // v0.22 phase 7: adetailer config keys.

    #[test]
    fn defaults_for_v022_phase7_adetailer_keys() {
        let cfg = GenerationConfig::default();
        assert!((cfg.adetailer_strength - 0.4).abs() < 1e-6);
        assert!((cfg.adetailer_padding - 0.25).abs() < 1e-6);
        assert!((cfg.adetailer_feather - 0.25).abs() < 1e-6);
        assert!((cfg.adetailer_confidence - 0.5).abs() < 1e-6);
        assert_eq!(cfg.adetailer_size, 512);
        assert_eq!(
            cfg.adetailer_prompt,
            "detailed face, sharp focus, high quality"
        );
    }

    #[test]
    fn set_str_adetailer_unit_floats_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("adetailer_strength", "0.6").unwrap();
        assert!((cfg.adetailer_strength - 0.6).abs() < 1e-6);
        cfg.set_str("adetailer_padding", "0.35").unwrap();
        assert!((cfg.adetailer_padding - 0.35).abs() < 1e-6);
        cfg.set_str("adetailer_feather", "0.5").unwrap();
        assert!((cfg.adetailer_feather - 0.5).abs() < 1e-6);
        cfg.set_str("adetailer_confidence", "0.7").unwrap();
        assert!((cfg.adetailer_confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn set_str_adetailer_unit_floats_reject_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("adetailer_strength", "1.5").is_err());
        assert!(cfg.set_str("adetailer_padding", "-0.1").is_err());
        assert!(cfg.set_str("adetailer_confidence", "2.0").is_err());
    }

    #[test]
    fn set_str_adetailer_size_requires_multiple_of_eight() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("adetailer_size", "768").unwrap();
        assert_eq!(cfg.adetailer_size, 768);
        let err = cfg.set_str("adetailer_size", "513").unwrap_err();
        assert!(format!("{err}").contains("multiple of 8"));
    }

    #[test]
    fn set_str_adetailer_size_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("adetailer_size", "0").is_err());
        assert!(cfg.set_str("adetailer_size", "4096").is_err());
    }

    #[test]
    fn set_str_adetailer_prompt_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("adetailer_prompt", "sharp eyes, detailed skin").unwrap();
        assert_eq!(cfg.adetailer_prompt, "sharp eyes, detailed skin");
    }

    #[test]
    fn set_int_adetailer_prompt_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("adetailer_prompt", 42).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expects a string"), "got {msg}");
    }

    #[test]
    fn set_int_adetailer_size_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("adetailer_size", 1024).unwrap();
        assert_eq!(cfg.adetailer_size, 1024);
    }

    #[test]
    fn set_float_adetailer_strength_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("adetailer_strength", 0.55).unwrap();
        assert!((cfg.adetailer_strength - 0.55).abs() < 1e-6);
    }

    // v0.22 phase 8: hires-fix config keys.

    #[test]
    fn defaults_for_v022_phase8_hires_keys() {
        let cfg = GenerationConfig::default();
        assert!((cfg.hires_scale - 2.0).abs() < 1e-6);
        assert!((cfg.hires_strength - 0.5).abs() < 1e-6);
        assert_eq!(cfg.hires_upscaler, "lanczos");
        assert!(cfg.hires_steps.is_none());
    }

    #[test]
    fn set_str_hires_scale_accepts_open_one_to_four() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("hires_scale", "1.5").unwrap();
        assert!((cfg.hires_scale - 1.5).abs() < 1e-6);
        cfg.set_str("hires_scale", "4.0").unwrap();
        assert!((cfg.hires_scale - 4.0).abs() < 1e-6);
    }

    #[test]
    fn set_str_hires_scale_rejects_one_and_below_or_above_four() {
        let mut cfg = GenerationConfig::default();
        // 1.0 is excluded — no upscaling makes hires-fix a no-op.
        assert!(cfg.set_str("hires_scale", "1.0").is_err());
        assert!(cfg.set_str("hires_scale", "0.5").is_err());
        assert!(cfg.set_str("hires_scale", "4.5").is_err());
    }

    #[test]
    fn set_str_hires_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("hires_strength", "0.0").unwrap();
        assert!((cfg.hires_strength - 0.0).abs() < 1e-9);
        cfg.set_str("hires_strength", "0.7").unwrap();
        assert!((cfg.hires_strength - 0.7).abs() < 1e-6);
        cfg.set_str("hires_strength", "1.0").unwrap();
        assert!((cfg.hires_strength - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_str_hires_strength_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("hires_strength", "-0.1").is_err());
        assert!(cfg.set_str("hires_strength", "1.5").is_err());
    }

    #[test]
    fn set_str_hires_upscaler_accepts_canonical_methods() {
        let mut cfg = GenerationConfig::default();
        for m in &[
            "lanczos",
            "lanczos3",
            "bicubic",
            "bilinear",
            "nearest",
            "real-esrgan-x2",
            "real-esrgan-x4",
            "real-esrgan-anime-x4",
        ] {
            cfg.set_str("hires_upscaler", m)
                .unwrap_or_else(|e| panic!("upscaler {m:?} should parse: {e}"));
            assert_eq!(cfg.hires_upscaler, *m);
        }
    }

    #[test]
    fn set_str_hires_upscaler_rejects_unknown() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("hires_upscaler", "ultra-9000").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ultra-9000"), "got {msg}");
        assert!(msg.contains("lanczos"), "got {msg}");
    }

    #[test]
    fn set_str_hires_steps_accepts_positive_int() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("hires_steps", "12").unwrap();
        assert_eq!(cfg.hires_steps, Some(12));
    }

    #[test]
    fn set_str_hires_steps_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("hires_steps", "0").is_err());
        assert!(cfg.set_str("hires_steps", "1000").is_err());
    }

    #[test]
    fn set_int_hires_upscaler_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("hires_upscaler", 42).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }

    #[test]
    fn set_float_hires_scale_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("hires_scale", 2.5).unwrap();
        assert!((cfg.hires_scale - 2.5).abs() < 1e-6);
    }

    // v0.22 phase 9: artefact config keys.

    #[test]
    fn defaults_for_v022_phase9_artefact_keys() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.artefact_library, "");
        assert!((cfg.artefact_blend_strength - 0.3).abs() < 1e-6);
        assert!(!cfg.artefact_smart_zones);
    }

    #[test]
    fn set_str_artefact_library_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("artefact_library", "/custom/lib").unwrap();
        assert_eq!(cfg.artefact_library, "/custom/lib");
        // Empty is allowed; resets to default at use-site.
        cfg.set_str("artefact_library", "").unwrap();
        assert_eq!(cfg.artefact_library, "");
    }

    #[test]
    fn set_str_artefact_blend_strength_accepts_unit_interval() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("artefact_blend_strength", "0.4").unwrap();
        assert!((cfg.artefact_blend_strength - 0.4).abs() < 1e-6);
        cfg.set_str("artefact_blend_strength", "0.0").unwrap();
        assert!((cfg.artefact_blend_strength - 0.0).abs() < 1e-9);
        cfg.set_str("artefact_blend_strength", "1.0").unwrap();
        assert!((cfg.artefact_blend_strength - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_str_artefact_blend_strength_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("artefact_blend_strength", "-0.1").is_err());
        assert!(cfg.set_str("artefact_blend_strength", "1.5").is_err());
    }

    #[test]
    fn set_str_artefact_smart_zones_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("artefact_smart_zones", "true").unwrap();
        assert!(cfg.artefact_smart_zones);
        cfg.set_str("artefact_smart_zones", "false").unwrap();
        assert!(!cfg.artefact_smart_zones);
    }

    #[test]
    fn set_int_artefact_library_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("artefact_library", 0).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }

    #[test]
    fn set_int_artefact_smart_zones_accepts_zero_and_one() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("artefact_smart_zones", 1).unwrap();
        assert!(cfg.artefact_smart_zones);
        cfg.set_int("artefact_smart_zones", 0).unwrap();
        assert!(!cfg.artefact_smart_zones);
        assert!(cfg.set_int("artefact_smart_zones", 2).is_err());
    }

    #[test]
    fn set_float_artefact_blend_strength_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("artefact_blend_strength", 0.45).unwrap();
        assert!((cfg.artefact_blend_strength - 0.45).abs() < 1e-6);
    }

    // v0.22 phase 10: enhance config keys.

    #[test]
    fn defaults_for_v022_phase10_enhance_keys() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.enhance_provider, "auto");
        assert!(cfg.enhance_temp.is_none());
        assert!(cfg.enhance_max_tokens.is_none());
        assert!(!cfg.enhance_cache);
        assert_eq!(cfg.enhance_system, "");
        assert!(!cfg.enhance_keep_original);
    }

    #[test]
    fn set_str_enhance_provider_accepts_grammar() {
        let mut cfg = GenerationConfig::default();
        for p in &[
            "auto",
            "deepseek",
            "gemini",
            "local",
            "local:smollm2-360m",
            "local:qwen2.5-1.5b",
        ] {
            cfg.set_str("enhance_provider", p)
                .unwrap_or_else(|e| panic!("provider {p:?} should parse: {e}"));
            assert_eq!(cfg.enhance_provider, p.to_lowercase());
        }
    }

    #[test]
    fn set_str_enhance_provider_rejects_unknown() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("enhance_provider", "claude").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("claude"), "got {msg}");
        assert!(msg.contains("local"), "got {msg}");
    }

    #[test]
    fn set_str_enhance_temp_accepts_zero_to_two() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("enhance_temp", "0.0").unwrap();
        assert_eq!(cfg.enhance_temp, Some(0.0));
        cfg.set_str("enhance_temp", "1.5").unwrap();
        assert!((cfg.enhance_temp.unwrap() - 1.5).abs() < 1e-6);
        cfg.set_str("enhance_temp", "2.0").unwrap();
        assert_eq!(cfg.enhance_temp, Some(2.0));
    }

    #[test]
    fn set_str_enhance_temp_rejects_out_of_range() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("enhance_temp", "-0.1").is_err());
        assert!(cfg.set_str("enhance_temp", "2.5").is_err());
    }

    #[test]
    fn set_str_enhance_max_tokens_accepts_positive() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("enhance_max_tokens", "128").unwrap();
        assert_eq!(cfg.enhance_max_tokens, Some(128));
    }

    #[test]
    fn set_str_enhance_max_tokens_rejects_zero_and_huge() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("enhance_max_tokens", "0").is_err());
        assert!(cfg.set_str("enhance_max_tokens", "5000").is_err());
    }

    #[test]
    fn set_str_enhance_cache_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("enhance_cache", "true").unwrap();
        assert!(cfg.enhance_cache);
        cfg.set_str("enhance_cache", "false").unwrap();
        assert!(!cfg.enhance_cache);
    }

    #[test]
    fn set_str_enhance_system_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("enhance_system", "/path/to/sys.txt").unwrap();
        assert_eq!(cfg.enhance_system, "/path/to/sys.txt");
    }

    #[test]
    fn set_str_enhance_keep_original_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("enhance_keep_original", "true").unwrap();
        assert!(cfg.enhance_keep_original);
        cfg.set_str("enhance_keep_original", "false").unwrap();
        assert!(!cfg.enhance_keep_original);
    }

    #[test]
    fn set_int_enhance_provider_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("enhance_provider", 0).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }

    #[test]
    fn set_float_enhance_temp_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_float("enhance_temp", 0.7).unwrap();
        assert!((cfg.enhance_temp.unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn set_int_enhance_max_tokens_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("enhance_max_tokens", 200).unwrap();
        assert_eq!(cfg.enhance_max_tokens, Some(200));
    }

    #[test]
    fn set_int_enhance_cache_accepts_zero_and_one() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("enhance_cache", 1).unwrap();
        assert!(cfg.enhance_cache);
        cfg.set_int("enhance_cache", 0).unwrap();
        assert!(!cfg.enhance_cache);
    }

    // v0.22 phase 11: misc config keys.

    #[test]
    fn defaults_for_v022_phase11_misc_keys() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.aspect, "");
        assert_eq!(cfg.base, 768);
        assert_eq!(cfg.mask_feather, 8);
        assert!(!cfg.mask_invert);
        assert_eq!(cfg.clip_skip, 1);
        assert_eq!(cfg.wildcard_dir, "");
        assert_eq!(cfg.negative_preset, "");
    }

    #[test]
    fn set_str_aspect_accepts_w_colon_h() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("aspect", "16:9").unwrap();
        assert_eq!(cfg.aspect, "16:9");
        cfg.set_str("aspect", "2:3").unwrap();
        assert_eq!(cfg.aspect, "2:3");
        // Empty clears.
        cfg.set_str("aspect", "").unwrap();
        assert_eq!(cfg.aspect, "");
    }

    #[test]
    fn set_str_aspect_rejects_malformed() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("aspect", "169").is_err()); // no colon
        assert!(cfg.set_str("aspect", "16:0").is_err()); // zero
        assert!(cfg.set_str("aspect", "0:9").is_err()); // zero
        assert!(cfg.set_str("aspect", "abc:def").is_err()); // non-int
    }

    #[test]
    fn set_str_base_accepts_multiple_of_eight() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("base", "512").unwrap();
        assert_eq!(cfg.base, 512);
        cfg.set_str("base", "1024").unwrap();
        assert_eq!(cfg.base, 1024);
    }

    #[test]
    fn set_str_base_rejects_non_multiple_of_eight() {
        let mut cfg = GenerationConfig::default();
        assert!(cfg.set_str("base", "513").is_err());
        assert!(cfg.set_str("base", "0").is_err());
        assert!(cfg.set_str("base", "9000").is_err());
    }

    #[test]
    fn set_str_mask_feather_accepts_zero_to_256() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("mask_feather", "0").unwrap();
        assert_eq!(cfg.mask_feather, 0);
        cfg.set_str("mask_feather", "16").unwrap();
        assert_eq!(cfg.mask_feather, 16);
        cfg.set_str("mask_feather", "256").unwrap();
        assert_eq!(cfg.mask_feather, 256);
        assert!(cfg.set_str("mask_feather", "500").is_err());
    }

    #[test]
    fn set_str_mask_invert_accepts_bool() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("mask_invert", "true").unwrap();
        assert!(cfg.mask_invert);
        cfg.set_str("mask_invert", "false").unwrap();
        assert!(!cfg.mask_invert);
    }

    #[test]
    fn set_str_clip_skip_accepts_one_to_twelve() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("clip_skip", "1").unwrap();
        assert_eq!(cfg.clip_skip, 1);
        cfg.set_str("clip_skip", "2").unwrap();
        assert_eq!(cfg.clip_skip, 2);
        assert!(cfg.set_str("clip_skip", "0").is_err());
        assert!(cfg.set_str("clip_skip", "13").is_err());
    }

    #[test]
    fn set_str_wildcard_dir_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("wildcard_dir", "/path/to/wildcards").unwrap();
        assert_eq!(cfg.wildcard_dir, "/path/to/wildcards");
    }

    #[test]
    fn set_str_negative_preset_accepts_built_ins() {
        let mut cfg = GenerationConfig::default();
        for name in &["photo", "painting", "anime", "cinematic"] {
            cfg.set_str("negative_preset", name)
                .unwrap_or_else(|e| panic!("preset {name:?} should parse: {e}"));
            assert_eq!(cfg.negative_preset, *name);
        }
        // Empty clears.
        cfg.set_str("negative_preset", "").unwrap();
        assert_eq!(cfg.negative_preset, "");
    }

    #[test]
    fn set_str_negative_preset_rejects_unknown() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("negative_preset", "ultra-9000").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ultra-9000"), "got {msg}");
        assert!(msg.contains("photo"), "got {msg}");
    }

    #[test]
    fn set_int_aspect_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("aspect", 169).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }

    #[test]
    fn set_int_base_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("base", 512).unwrap();
        assert_eq!(cfg.base, 512);
    }

    #[test]
    fn set_int_mask_invert_accepts_zero_and_one() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("mask_invert", 1).unwrap();
        assert!(cfg.mask_invert);
        cfg.set_int("mask_invert", 0).unwrap();
        assert!(!cfg.mask_invert);
    }

    #[test]
    fn set_int_clip_skip_round_trip() {
        let mut cfg = GenerationConfig::default();
        cfg.set_int("clip_skip", 2).unwrap();
        assert_eq!(cfg.clip_skip, 2);
    }

    // v0.24 phase 2: face_bbox + face_landmarks config keys.

    #[test]
    fn defaults_for_v024_phase2_face_keys() {
        let cfg = GenerationConfig::default();
        assert!(cfg.face_bbox.is_none());
        assert!(cfg.face_landmarks.is_none());
    }

    #[test]
    fn set_str_face_bbox_round_trips_csv() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("face_bbox", "0.2,0.1,0.8,0.7").unwrap();
        let bbox = cfg.face_bbox.expect("bbox set");
        assert!((bbox[0] - 0.2).abs() < 1e-6);
        assert!((bbox[1] - 0.1).abs() < 1e-6);
        assert!((bbox[2] - 0.8).abs() < 1e-6);
        assert!((bbox[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn set_str_face_bbox_empty_clears() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("face_bbox", "0.2,0.1,0.8,0.7").unwrap();
        assert!(cfg.face_bbox.is_some());
        cfg.set_str("face_bbox", "").unwrap();
        assert!(cfg.face_bbox.is_none());
    }

    #[test]
    fn set_str_face_bbox_rejects_malformed() {
        let mut cfg = GenerationConfig::default();
        // Wrong arity.
        assert!(cfg.set_str("face_bbox", "0.2,0.1,0.8").is_err());
        // Out-of-range component.
        assert!(cfg.set_str("face_bbox", "0.2,0.1,1.5,0.7").is_err());
        // x0 >= x1.
        assert!(cfg.set_str("face_bbox", "0.8,0.1,0.2,0.7").is_err());
        // y0 >= y1.
        assert!(cfg.set_str("face_bbox", "0.2,0.7,0.8,0.1").is_err());
    }

    #[test]
    fn set_str_face_landmarks_round_trips_csv() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str(
            "face_landmarks",
            "0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68",
        )
        .unwrap();
        let lm = cfg.face_landmarks.expect("landmarks set");
        assert!((lm[0][0] - 0.40).abs() < 1e-6);
        assert!((lm[4][1] - 0.68).abs() < 1e-6);
    }

    #[test]
    fn set_str_face_landmarks_empty_clears() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str(
            "face_landmarks",
            "0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68",
        )
        .unwrap();
        assert!(cfg.face_landmarks.is_some());
        cfg.set_str("face_landmarks", "").unwrap();
        assert!(cfg.face_landmarks.is_none());
    }

    #[test]
    fn set_str_face_landmarks_rejects_wrong_arity() {
        let mut cfg = GenerationConfig::default();
        // 8 values instead of 10.
        let err =
            cfg.set_str("face_landmarks", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8").unwrap_err();
        assert!(format!("{err}").contains("10 comma-separated"));
    }

    #[test]
    fn set_int_face_bbox_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("face_bbox", 42).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }

    // v0.24 phase 3: identity_kind config key.

    #[test]
    fn defaults_for_v024_phase3_identity_key() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.identity_kind, "");
    }

    #[test]
    fn set_str_identity_kind_accepts_canonical_variants() {
        let mut cfg = GenerationConfig::default();
        for name in &["plus-face", "plus-face-sdxl", "face-id", "face-id-sdxl"] {
            cfg.set_str("identity_kind", name)
                .unwrap_or_else(|e| panic!("identity_kind {name:?} should parse: {e}"));
            assert_eq!(cfg.identity_kind, *name);
        }
    }

    #[test]
    fn set_str_identity_kind_accepts_aliases() {
        // IdentityKind::from_str takes plus-face / plusface /
        // plus_face all as PlusFace; we store the user's input
        // verbatim, but the set should succeed.
        let mut cfg = GenerationConfig::default();
        cfg.set_str("identity_kind", "plusface").unwrap();
        cfg.set_str("identity_kind", "face_id").unwrap();
        cfg.set_str("identity_kind", "sdxl-faceid").unwrap();
    }

    #[test]
    fn set_str_identity_kind_empty_clears() {
        let mut cfg = GenerationConfig::default();
        cfg.set_str("identity_kind", "face-id").unwrap();
        assert_eq!(cfg.identity_kind, "face-id");
        cfg.set_str("identity_kind", "").unwrap();
        assert_eq!(cfg.identity_kind, "");
    }

    #[test]
    fn set_str_identity_kind_rejects_unknown() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_str("identity_kind", "instant-id").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("instant-id"), "got {msg}");
        // Error should mention accepted variants.
        assert!(msg.contains("plus-face"), "got {msg}");
    }

    #[test]
    fn set_int_identity_kind_is_type_error() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("identity_kind", 0).unwrap_err();
        assert!(format!("{err}").contains("expects a string"));
    }
}
