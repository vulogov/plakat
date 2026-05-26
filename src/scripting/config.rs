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
            other => {
                return Err(anyhow!(
                    "plakat.config.set: unknown key {other:?}. \
                     Supported keys: steps, guidance, seed, width, \
                     height, negative, scheduler, strength, face_strength."
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
            | "strength" | "face_strength" => self.set_str(key, &value.to_string()),
            "negative" | "scheduler" => Err(anyhow!(
                "plakat.config.set: key {key:?} expects a string value, got integer {value}"
            )),
            other => Err(anyhow!(
                "plakat.config.set: unknown key {other:?}. \
                 Supported keys: steps, guidance, seed, width, \
                 height, negative, scheduler, strength, face_strength."
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
                 height, negative, scheduler, strength, face_strength."
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

    #[test]
    fn set_int_seed_rejects_negative() {
        let mut cfg = GenerationConfig::default();
        let err = cfg.set_int("seed", -1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(">= 0"), "got {msg}");
    }
}
