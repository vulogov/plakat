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

use anyhow::{Result, anyhow, bail};

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
            other => {
                return Err(anyhow!(
                    "plakat.config.set: unknown key {other:?}. \
                     Supported keys: steps, guidance, seed, width, \
                     height, negative, scheduler, strength, \
                     face_strength, quantize_t5, quant_level, \
                     t5_quant_level, fast, kontext_bucket, tiled, \
                     tile_size, tile_stride, lora_scale."
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
            | "lora_scale" => self.set_str(key, &value.to_string()),
            "quantize_t5" | "kontext_bucket" | "tiled" => {
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
            "negative" | "scheduler" => Err(anyhow!(
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
            "steps" | "seed" | "width" | "height" => {
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
            "negative" | "scheduler" => Err(anyhow!(
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

    #[test]
    fn set_int_seed_rejects_negative() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("seed", -1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(">= 0"), "got {msg}");
    }
}
