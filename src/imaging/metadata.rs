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

/// v0.33 phase 0: structured LoRA entry. Companion to the flat
/// `loras: Vec<String>` field which keeps the A1111-style
/// representation for legacy tooling. `lora_stack` when present
/// carries richer per-LoRA detail downstream consumers can use.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraEntry {
    /// Human-readable display name (matches `ResolvedLora.display`).
    pub display: String,
    pub scale: f32,
    /// Source kind: `"local"` / `"hub"` / `"civitai"`. Optional so
    /// downstream tooling can keep an entry without committing to
    /// a taxonomy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// HF pinned revision (SHA / tag / branch) when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// v0.33 phase 0: structured Textual Inversion (embedding) entry.
/// Records what TIs were applied at load time per v0.30 phase 0 /
/// v0.31 phase 0 runtime injection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingEntry {
    pub trigger: String,
    pub embed_dim: usize,
    /// Number of vectors per trigger (multi-vector TIs render as N
    /// consecutive tokens — `trigger`, `trigger_1`, ...).
    pub num_tokens: usize,
    /// `true` when this is a v0.31 SDXL dual-encoder TI (clip_l +
    /// clip_g in the same file). `false` for single-encoder TIs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dual_encoder: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// v0.33 phase 0: structured ControlNet entry. Captures per-CN
/// detail beyond the flat `controls: Vec<String>` shorthand. Used
/// when downstream tooling needs to know the window / strength
/// per conditioner.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlEntry {
    pub kind: String,
    /// User-provided pre-rendered conditioning image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Auto-annotation source (v0.10+).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Per-frame video source (v0.30 phase 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<String>,
    pub strength: f32,
    pub start: f32,
    pub end: f32,
}

/// v0.33 phase 0: prompt enhancement details. Captures the v0.19
/// `--enhance` workflow so a sidecar carries the original prompt +
/// the enhancer's output + provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnhancementMetadata {
    /// `"deepseek"` / `"gemini"` / `"local"` / `"local:<alias>"`.
    pub provider: String,
    /// v0.19: optional system-prompt override file name (without
    /// directory). Empty when the bundled system prompt was used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_name: Option<String>,
    /// `true` when the local-enhancer cache served this prompt.
    pub cache_hit: bool,
    /// Pre-enhancement prompt (the user's actual `--prompt` value).
    /// `prompt` on the parent `GenerationMetadata` holds the
    /// post-enhancement version that fed the pipeline.
    pub original_prompt: String,
}

/// Everything needed to reproduce one image. Fields are
/// best-effort — anything the pipeline knows at save time gets
/// captured, anything it doesn't (e.g. img2img init paths during
/// a t2i call) stays `None`.
///
/// v0.33 phase 0 added structured stacks (`lora_stack`,
/// `embedding_stack`, `control_stack`) + look / genre / enhancer
/// fields. All additions are ADDITIVE — existing v0.32 sidecars
/// still parse via `#[serde(default)]` on the new fields.
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

    // ---------------------------------------------------------------
    // v0.33 phase 0: structured polish fields. All additive.
    // ---------------------------------------------------------------
    /// v0.25 `--look` preset name when applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<String>,
    /// v0.25 `--genre` preset name when applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub genre: Option<String>,
    /// v0.19 `--negative-preset` name when applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_preset: Option<String>,
    /// Structured LoRA stack — companion to the flat `loras` Vec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_stack: Option<Vec<LoraEntry>>,
    /// Flat A1111-style trigger list for embeddings. Mirrors
    /// `loras` shape — viewers / Civitai uploaders that recognise
    /// the field surface it inline.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embeddings: Vec<String>,
    /// Structured embedding stack — companion to `embeddings`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_stack: Option<Vec<EmbeddingEntry>>,
    /// Structured ControlNet stack — companion to `controls`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_stack: Option<Vec<ControlEntry>>,
    /// v0.19 prompt enhancement details when active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enhancement: Option<EnhancementMetadata>,
    /// v0.32 phase 0 `--free-noise` opt-in for long-form animate.
    /// Captured here so a sidecar from a long-form run records the
    /// noise-sharing mode that produced the seams (or lack thereof).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_noise: Option<bool>,
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
            // v0.33 phase 0 — structured polish fields. All None /
            // empty by default; callers populate via field setters
            // or the new with_* helpers.
            look: None,
            genre: None,
            negative_preset: None,
            lora_stack: None,
            embeddings: Vec::new(),
            embedding_stack: None,
            control_stack: None,
            enhancement: None,
            free_noise: None,
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
        // v0.33 phase 0: surface the new polish fields in the A1111
        // tEXt string so viewers (Civitai, A1111, ...) display them
        // inline. JSON sidecar consumers get the full structured
        // versions via the parent fields.
        if !self.embeddings.is_empty() {
            pairs.push(("Embeddings", self.embeddings.join(" | ")));
        }
        if let Some(look) = &self.look {
            pairs.push(("Look", look.clone()));
        }
        if let Some(genre) = &self.genre {
            pairs.push(("Genre", genre.clone()));
        }
        if let Some(np) = &self.negative_preset {
            pairs.push(("Negative preset", np.clone()));
        }
        if let Some(enh) = &self.enhancement {
            pairs.push(("Enhancer", enh.provider.clone()));
            if enh.cache_hit {
                pairs.push(("Enhancer cache", "hit".to_string()));
            }
        }
        if let Some(true) = self.free_noise {
            pairs.push(("FreeNoise", "on".to_string()));
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

    /// v0.18 phase 6: stamp `plakat animate` frame context into
    /// `extras`. The `prompt` field already carries the synthetic
    /// "lerp(t): from | to" description for the A1111 tEXt chunk;
    /// this helper records the structured `t`, `from`, and `to`
    /// values in the JSON sidecar so callers can re-render any
    /// frame from its sidecar alone.
    pub fn with_animate_lerp(
        &mut self,
        t: f64,
        from_prompt: &str,
        to_prompt: &str,
    ) -> &mut Self {
        self.extras.push(("Lerp t".into(), format!("{t:.4}")));
        self.extras
            .push(("Animate from".into(), from_prompt.to_string()));
        self.extras
            .push(("Animate to".into(), to_prompt.to_string()));
        self
    }

    // -----------------------------------------------------------------
    // v0.33 phase 0: builder-style helpers for the new polish fields.
    // -----------------------------------------------------------------

    /// Record the v0.25 look / genre presets applied. Pass `None`
    /// for either when the user didn't supply it.
    pub fn with_look_genre(
        &mut self,
        look: Option<&str>,
        genre: Option<&str>,
    ) -> &mut Self {
        self.look = look.map(|s| s.to_string());
        self.genre = genre.map(|s| s.to_string());
        self
    }

    /// Record the structured LoRA stack alongside the existing flat
    /// `loras` Vec<String>. The flat field stays the A1111-style
    /// short form; `lora_stack` carries the richer per-entry info.
    pub fn with_lora_stack(&mut self, stack: Vec<LoraEntry>) -> &mut Self {
        if !stack.is_empty() {
            self.lora_stack = Some(stack);
        }
        self
    }

    /// Record the structured embedding (TI) stack. Also populates
    /// the flat `embeddings` Vec<String> with the trigger words so
    /// the A1111 tEXt chunk surfaces them.
    pub fn with_embedding_stack(&mut self, stack: Vec<EmbeddingEntry>) -> &mut Self {
        if !stack.is_empty() {
            self.embeddings = stack.iter().map(|e| e.trigger.clone()).collect();
            self.embedding_stack = Some(stack);
        }
        self
    }

    /// Record the structured ControlNet stack. Also populates the
    /// flat `controls` Vec<String> with `kind` summaries when that
    /// field is currently empty.
    pub fn with_control_stack(&mut self, stack: Vec<ControlEntry>) -> &mut Self {
        if !stack.is_empty() {
            if self.controls.is_empty() {
                self.controls = stack.iter().map(|c| c.kind.clone()).collect();
            }
            self.control_stack = Some(stack);
        }
        self
    }

    /// Record v0.19 prompt enhancement details.
    pub fn with_enhancement(&mut self, e: EnhancementMetadata) -> &mut Self {
        self.enhancement = Some(e);
        self
    }

    /// Record the v0.32 phase 0 `--free-noise` flag for long-form
    /// animate. Pass `true` when the flag was set so the sidecar
    /// records how the seam handling worked.
    pub fn with_free_noise(&mut self, on: bool) -> &mut Self {
        self.free_noise = Some(on);
        self
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

    // v0.18 phase 6 — animate frame metadata.

    #[test]
    fn with_animate_lerp_stamps_three_extras() {
        let mut m = mk();
        m.with_animate_lerp(0.375, "a fox in a meadow", "a cat in a meadow");
        // Order is t, from, to.
        assert_eq!(m.extras.len(), 3);
        assert_eq!(m.extras[0].0, "Lerp t");
        assert_eq!(m.extras[0].1, "0.3750");
        assert_eq!(m.extras[1], ("Animate from".into(), "a fox in a meadow".into()));
        assert_eq!(m.extras[2], ("Animate to".into(), "a cat in a meadow".into()));
    }

    #[test]
    fn with_animate_lerp_surfaces_in_a1111_chunk() {
        let mut m = mk();
        m.with_animate_lerp(0.5, "a", "b");
        let s = m.to_a1111_parameters_string();
        // The A1111 third line concatenates all the key:value pairs;
        // the lerp entries should land there alongside Steps / Sampler.
        assert!(s.contains("Lerp t: 0.5000"), "got: {s}");
        assert!(s.contains("Animate from: a"));
        assert!(s.contains("Animate to: b"));
    }

    #[test]
    fn with_animate_lerp_json_carries_structured_extras() {
        let mut m = mk();
        m.with_animate_lerp(0.25, "from", "to");
        let json = m.to_json_pretty().unwrap();
        assert!(json.contains("\"Lerp t\""), "{json}");
        assert!(json.contains("\"0.2500\""), "{json}");
        assert!(json.contains("\"Animate from\""));
    }

    // -----------------------------------------------------------------
    // v0.33 phase 0: backward-compat + new field tests.
    // -----------------------------------------------------------------

    /// CRITICAL: a sidecar saved by v0.32 must still parse cleanly
    /// in v0.33. Every new field is Optional / has a serde default;
    /// this test simulates a pre-v0.33 sidecar by hand and confirms
    /// the new fields land at their default values.
    #[test]
    fn v032_sidecar_still_parses() {
        // Minimal v0.32 sidecar shape (no v0.33 polish fields).
        let v032_json = r#"{
            "prompt": "a watercolor cottage",
            "model": "sd15",
            "seed": 42,
            "steps": 28,
            "guidance": 7.5,
            "scheduler": "euler-a",
            "width": 512,
            "height": 512,
            "loras": ["my/style:0.7"],
            "generator": "plakat 0.32.0"
        }"#;
        let m: GenerationMetadata = serde_json::from_str(v032_json).unwrap();
        assert_eq!(m.prompt, "a watercolor cottage");
        assert_eq!(m.loras, vec!["my/style:0.7".to_string()]);
        // All v0.33 fields default to None / empty.
        assert!(m.look.is_none());
        assert!(m.genre.is_none());
        assert!(m.negative_preset.is_none());
        assert!(m.lora_stack.is_none());
        assert!(m.embeddings.is_empty());
        assert!(m.embedding_stack.is_none());
        assert!(m.control_stack.is_none());
        assert!(m.enhancement.is_none());
        assert!(m.free_noise.is_none());
    }

    #[test]
    fn with_look_genre_records_both() {
        let mut m = mk();
        m.with_look_genre(Some("watercolor"), Some("anime"));
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("Look: watercolor"), "got {s}");
        assert!(s.contains("Genre: anime"), "got {s}");
    }

    #[test]
    fn with_look_genre_handles_none() {
        let mut m = mk();
        m.with_look_genre(None, Some("anime"));
        let s = m.to_a1111_parameters_string();
        assert!(!s.contains("Look:"));
        assert!(s.contains("Genre: anime"));
    }

    #[test]
    fn lora_stack_round_trips_through_json() {
        let mut m = mk();
        m.with_lora_stack(vec![
            LoraEntry {
                display: "my/style".into(),
                scale: 0.7,
                source: Some("hub".into()),
                revision: Some("abc123".into()),
            },
            LoraEntry {
                display: "civitai:12345".into(),
                scale: 0.5,
                source: Some("civitai".into()),
                revision: None,
            },
        ]);
        let json = m.to_json_pretty().unwrap();
        let m2: GenerationMetadata = serde_json::from_str(&json).unwrap();
        let stack = m2.lora_stack.expect("lora_stack");
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0].display, "my/style");
        assert!((stack[0].scale - 0.7).abs() < f32::EPSILON);
        assert_eq!(stack[0].revision.as_deref(), Some("abc123"));
        assert_eq!(stack[1].source.as_deref(), Some("civitai"));
        assert!(stack[1].revision.is_none());
    }

    #[test]
    fn embedding_stack_populates_flat_triggers() {
        let mut m = mk();
        m.with_embedding_stack(vec![
            EmbeddingEntry {
                trigger: "my-style".into(),
                embed_dim: 768,
                num_tokens: 2,
                dual_encoder: false,
                source: Some("./my-style.safetensors".into()),
            },
            EmbeddingEntry {
                trigger: "pony-style".into(),
                embed_dim: 768,
                num_tokens: 1,
                dual_encoder: true,
                source: Some("AstraliteHeart/pony-ti".into()),
            },
        ]);
        // Flat field auto-populated from stack triggers.
        assert_eq!(m.embeddings, vec!["my-style".to_string(), "pony-style".to_string()]);
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("Embeddings: my-style | pony-style"), "got {s}");
    }

    #[test]
    fn embedding_dual_encoder_flag_round_trips() {
        let mut m = mk();
        m.with_embedding_stack(vec![EmbeddingEntry {
            trigger: "dual".into(),
            embed_dim: 768,
            num_tokens: 1,
            dual_encoder: true,
            source: None,
        }]);
        let json = m.to_json_pretty().unwrap();
        let m2: GenerationMetadata = serde_json::from_str(&json).unwrap();
        let stack = m2.embedding_stack.unwrap();
        assert!(stack[0].dual_encoder);
    }

    #[test]
    fn control_stack_populates_flat_controls_when_empty() {
        let mut m = mk();
        assert!(m.controls.is_empty());
        m.with_control_stack(vec![
            ControlEntry {
                kind: "canny".into(),
                image: Some("./edges.png".into()),
                from: None,
                video: None,
                strength: 0.8,
                start: 0.0,
                end: 1.0,
            },
            ControlEntry {
                kind: "openpose".into(),
                image: None,
                from: None,
                video: Some("./drive.mp4".into()),
                strength: 0.9,
                start: 0.0,
                end: 0.8,
            },
        ]);
        // Flat `controls` auto-populated from stack kinds.
        assert_eq!(m.controls, vec!["canny".to_string(), "openpose".to_string()]);
        // Structured stack preserved.
        let stack = m.control_stack.as_ref().unwrap();
        assert_eq!(stack[1].video.as_deref(), Some("./drive.mp4"));
        assert!((stack[1].end - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn control_stack_preserves_existing_flat_controls() {
        // When the user populated `controls` before calling
        // with_control_stack (legacy callers), don't clobber it.
        let mut m = mk();
        m.controls = vec!["legacy-control".into()];
        m.with_control_stack(vec![ControlEntry {
            kind: "canny".into(),
            image: None,
            from: None,
            video: None,
            strength: 1.0,
            start: 0.0,
            end: 1.0,
        }]);
        assert_eq!(m.controls, vec!["legacy-control".to_string()]);
        assert!(m.control_stack.is_some());
    }

    #[test]
    fn enhancement_round_trips_with_a1111_visibility() {
        let mut m = mk();
        m.with_enhancement(EnhancementMetadata {
            provider: "local:llama-3-8b".into(),
            system_prompt_name: Some("anime-prompts.txt".into()),
            cache_hit: true,
            original_prompt: "a fox".into(),
        });
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("Enhancer: local:llama-3-8b"), "got {s}");
        assert!(s.contains("Enhancer cache: hit"), "got {s}");
        let json = m.to_json_pretty().unwrap();
        let m2: GenerationMetadata = serde_json::from_str(&json).unwrap();
        let e = m2.enhancement.unwrap();
        assert_eq!(e.original_prompt, "a fox");
        assert_eq!(e.system_prompt_name.as_deref(), Some("anime-prompts.txt"));
    }

    #[test]
    fn free_noise_flag_surfaces_in_a1111() {
        let mut m = mk();
        m.with_free_noise(true);
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("FreeNoise: on"));

        let mut m2 = mk();
        m2.with_free_noise(false);
        let s2 = m2.to_a1111_parameters_string();
        assert!(!s2.contains("FreeNoise"), "off should be omitted, got: {s2}");
    }

    #[test]
    fn negative_preset_surfaces_in_a1111() {
        let mut m = mk();
        m.negative_preset = Some("photo".into());
        let s = m.to_a1111_parameters_string();
        assert!(s.contains("Negative preset: photo"));
    }

    #[test]
    fn empty_optional_fields_omitted_from_json() {
        let m = mk();
        let json = m.to_json_pretty().unwrap();
        // Empty / None fields should not appear in the sidecar.
        assert!(!json.contains("\"look\""));
        assert!(!json.contains("\"genre\""));
        assert!(!json.contains("\"negative_preset\""));
        assert!(!json.contains("\"lora_stack\""));
        assert!(!json.contains("\"embeddings\""));
        assert!(!json.contains("\"embedding_stack\""));
        assert!(!json.contains("\"control_stack\""));
        assert!(!json.contains("\"enhancement\""));
        assert!(!json.contains("\"free_noise\""));
    }
}
