//! Generation metadata for PNG tEXt chunks + JSON sidecars.
//!
//! Two output forms:
//!
//! 1. **PNG tEXt chunk** with key `parameters` carrying the
//!    Auto1111-compatible format:
//!
//!    ```text
//!    <prompt>
//!    Negative prompt: <negative>
//!    Steps: 28, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, Size: 512x512, Model: sd15
//!    ```
//!
//!    Every viewer in the SD ecosystem (Auto1111 Web UI, Civitai
//!    image uploader, ComfyUI drag-to-load, SD-Prompt-Reader,
//!    image-info Chrome extension, ...) recognises this key and
//!    surfaces the recipe inline.
//!
//! 2. **JSON sidecar** `<base>.json` next to the PNG carrying the
//!    structured equivalent — full LoRA stack, scheduler enum,
//!    refiner config, ADetailer / Hires fix flags, etc. Useful
//!    when you want to script around the recipe.
//!
//! The struct is `Serialize` + `Deserialize` so users can pipe the
//! sidecar back through any consumer.

use serde::{Deserialize, Serialize};

/// Everything needed to reproduce one image. Fields are
/// best-effort — anything the pipeline knows at save time gets
/// captured, anything it doesn't (e.g. img2img init paths during
/// a t2i call) stays `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub negative: String,
    pub model: String,
    pub seed: u64,
    pub steps: usize,
    pub guidance: f64,
    pub scheduler: String,
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_skip: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refiner_frac: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    /// Free-form notes — plakat features that have no Auto1111
    /// counterpart (ADetailer / Hires fix / Civitai sourcing) get
    /// recorded here so the sidecar carries them faithfully.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extras: Vec<(String, String)>,
    /// Plakat version that produced the image. Read at runtime
    /// from `env!("CARGO_PKG_VERSION")` so updates roll forward
    /// automatically.
    pub generator: String,
}

impl GenerationMetadata {
    /// Build a minimal metadata record covering the t2i common
    /// fields. Callers can add LoRAs / controls / mode flags via
    /// the field-setting helpers below.
    pub fn new(
        prompt: impl Into<String>,
        model: impl Into<String>,
        seed: u64,
        steps: usize,
        guidance: f64,
        scheduler: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            prompt: prompt.into(),
            negative: String::new(),
            model: model.into(),
            seed,
            steps,
            guidance,
            scheduler: scheduler.into(),
            width,
            height,
            loras: Vec::new(),
            lora_scale: None,
            clip_skip: None,
            controls: Vec::new(),
            refiner_frac: None,
            mode: None,
            strength: None,
            extras: Vec::new(),
            generator: format!("plakat {}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Serialise to Auto1111's `parameters` PNG tEXt chunk
    /// format. Three (or two) lines: prompt, negative prompt (only
    /// if non-empty), key:value pairs.
    ///
    /// Auto1111 parsers are lenient about ordering and trailing
    /// commas; the canonical order is Steps / Sampler / CFG /
    /// Seed / Size / Model. plakat-specific fields tail along
    /// after that as `Key: value` pairs the viewers will display
    /// but won't necessarily parse.
    pub fn to_a1111_parameters_string(&self) -> String {
        let mut out = String::with_capacity(256 + self.prompt.len() + self.negative.len());
        out.push_str(&self.prompt);
        if !self.negative.is_empty() {
            out.push('\n');
            out.push_str("Negative prompt: ");
            out.push_str(&self.negative);
        }
        out.push('\n');
        // Comma-separated key:value pairs. No trailing comma.
        let mut pairs: Vec<(&str, String)> = Vec::new();
        pairs.push(("Steps", self.steps.to_string()));
        pairs.push(("Sampler", self.scheduler.clone()));
        pairs.push(("CFG scale", format_float(self.guidance)));
        pairs.push(("Seed", self.seed.to_string()));
        pairs.push(("Size", format!("{}x{}", self.width, self.height)));
        pairs.push(("Model", self.model.clone()));
        if let Some(cs) = self.clip_skip {
            if cs > 1 {
                pairs.push(("Clip skip", cs.to_string()));
            }
        }
        if !self.loras.is_empty() {
            pairs.push(("LoRAs", self.loras.join(" | ")));
        }
        if let Some(s) = self.lora_scale {
            if (s - 1.0).abs() > f32::EPSILON {
                pairs.push(("LoRA scale", format_float(s as f64)));
            }
        }
        if !self.controls.is_empty() {
            pairs.push(("Controls", self.controls.join(" | ")));
        }
        if let Some(f) = self.refiner_frac {
            pairs.push(("Refiner frac", format_float(f as f64)));
        }
        if let Some(m) = &self.mode {
            pairs.push(("Mode", m.clone()));
        }
        if let Some(s) = self.strength {
            pairs.push(("Strength", format_float(s as f64)));
        }
        for (k, v) in &self.extras {
            pairs.push((k.as_str(), v.clone()));
        }
        pairs.push(("Generator", self.generator.clone()));
        let line = pairs
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&line);
        out
    }

    /// Serialise to pretty JSON for the sidecar file.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Render a float for the A1111 parameters string. Drops trailing
/// zeros to match Auto1111's own formatting (`CFG scale: 7` not
/// `CFG scale: 7.0`).
fn format_float(v: f64) -> String {
    if v.fract().abs() < 1e-6 {
        format!("{}", v as i64)
    } else {
        // 4 decimal places is enough resolution for every plakat
        // numeric and matches A1111's default.
        let s = format!("{v:.4}");
        // Trim trailing zeros + a trailing dot if any.
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk() -> GenerationMetadata {
        GenerationMetadata::new(
            "a fox",
            "sd15",
            42,
            28,
            7.5,
            "euler-a",
            512,
            512,
        )
    }

    #[test]
    fn a1111_parameters_minimal() {
        let m = mk();
        let s = m.to_a1111_parameters_string();
        // Prompt is the first line.
        let first = s.lines().next().unwrap();
        assert_eq!(first, "a fox");
        // Negative absent → no "Negative prompt:" line.
        assert!(!s.contains("Negative prompt:"));
        // The KV line is the second line.
        let kv = s.lines().nth(1).unwrap();
        assert!(kv.contains("Steps: 28"));
        assert!(kv.contains("Sampler: euler-a"));
        assert!(kv.contains("CFG scale: 7.5"));
        assert!(kv.contains("Seed: 42"));
        assert!(kv.contains("Size: 512x512"));
        assert!(kv.contains("Model: sd15"));
        assert!(kv.contains("Generator: plakat"));
    }

    #[test]
    fn a1111_parameters_with_negative() {
        let mut m = mk();
        m.negative = "blurry, low quality".to_string();
        let s = m.to_a1111_parameters_string();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "a fox");
        assert_eq!(lines[1], "Negative prompt: blurry, low quality");
        assert!(lines[2].contains("Steps: 28"));
    }

    #[test]
    fn a1111_parameters_with_loras_and_clipskip() {
        let mut m = mk();
        m.loras = vec!["my/style:0.7".into(), "my/character:0.5".into()];
        m.lora_scale = Some(0.8);
        m.clip_skip = Some(2);
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("LoRAs: my/style:0.7 | my/character:0.5"));
        assert!(s.contains("LoRA scale: 0.8"));
        assert!(s.contains("Clip skip: 2"));
    }

    #[test]
    fn clip_skip_1_is_omitted() {
        // The diffusers default doesn't need to be repeated in
        // the metadata.
        let mut m = mk();
        m.clip_skip = Some(1);
        let s = m.to_a1111_parameters_string();
        assert!(!s.contains("Clip skip"));
    }

    #[test]
    fn lora_scale_default_is_omitted() {
        let mut m = mk();
        m.lora_scale = Some(1.0);
        m.loras = vec!["foo".into()];
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("LoRAs:"));
        assert!(!s.contains("LoRA scale"));
    }

    #[test]
    fn float_formatter_drops_trailing_zeros() {
        assert_eq!(format_float(7.0), "7");
        assert_eq!(format_float(7.5), "7.5");
        assert_eq!(format_float(0.75), "0.75");
        assert_eq!(format_float(0.1), "0.1");
    }

    #[test]
    fn json_sidecar_roundtrips() {
        let mut m = mk();
        m.negative = "blurry".to_string();
        m.loras = vec!["x:0.7".into()];
        let json = m.to_json_pretty().unwrap();
        let m2: GenerationMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.prompt, m.prompt);
        assert_eq!(m2.negative, m.negative);
        assert_eq!(m2.loras, m.loras);
        assert_eq!(m2.seed, m.seed);
    }

    #[test]
    fn json_sidecar_omits_default_optional_fields() {
        // The serde `skip_serializing_if` for None / empty Vec /
        // empty String keeps the sidecar tidy.
        let m = mk();
        let json = m.to_json_pretty().unwrap();
        assert!(!json.contains("negative"));
        assert!(!json.contains("loras"));
        assert!(!json.contains("controls"));
        assert!(!json.contains("clip_skip"));
        assert!(json.contains("\"prompt\""));
        assert!(json.contains("\"seed\""));
    }

    #[test]
    fn extras_round_trip() {
        let mut m = mk();
        m.extras.push(("ADetailer".into(), "on".into()));
        m.extras.push(("Hires upscaler".into(), "real-esrgan-x2".into()));
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("ADetailer: on"));
        assert!(s.contains("Hires upscaler: real-esrgan-x2"));
    }
}
