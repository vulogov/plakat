//! The three prompt layers of a multiperson scene, kept separate at the type
//! level so they can't be mixed up:
//!
//!   * **scene** — the spatial/relational description ("three friends at tea").
//!     The only thing the LLM scene analyser ever sees.
//!   * **style** — visual aesthetic ("oil painting, Rembrandt lighting"). From a
//!     `// ` inline clause in the scene prompt, or a `style:` field. Enhancer only.
//!   * **per-person** — what each persona is doing / how they face. Appended to
//!     that persona's region prompt only.
//!
//! A `// ` inside the scene prompt splits scene (left) from inline style (right).

/// A parsed scene prompt: scene clause + an optional inline style clause.
#[derive(Debug, Clone)]
pub struct MultipersonPrompt {
    pub scene_clause: String,
    pub style_clause: Option<String>,
}

impl MultipersonPrompt {
    pub fn parse(raw: &str) -> Self {
        match raw.split_once("//") {
            Some((s, st)) => Self {
                scene_clause: s.trim().to_string(),
                style_clause: Some(st.trim().to_string()).filter(|s| !s.is_empty()),
            },
            None => Self { scene_clause: raw.trim().to_string(), style_clause: None },
        }
    }

    /// Sent to the scene analyser LLM — scene clause ONLY (no style/weather).
    pub fn for_analyser(&self) -> &str {
        &self.scene_clause
    }

    /// The shared scene+aesthetic base for the SD enhancer: scene + inline style
    /// + weather + effective style (task style overrides the global style).
    pub fn enhancer_base(
        &self,
        weather: Option<&str>,
        global_style: Option<&str>,
        task_style: Option<&str>,
    ) -> String {
        let effective_style = task_style.or(global_style);
        [
            Some(self.scene_clause.as_str()),
            self.style_clause.as_deref(),
            weather,
            effective_style,
        ]
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
    }

    /// One persona's region prompt: the shared base + their behavioural prompt +
    /// their facing phrase. Empty pieces are dropped.
    ///
    /// NOTE: this frames the region with the *whole* scene clause ("three people
    /// …"), which is wrong for a per-persona inpaint — it asks the region to paint
    /// the whole crowd, diluting that one persona's identity. Prefer
    /// [`single_region_prompt`](Self::single_region_prompt) for the inpaint pass;
    /// this is kept for callers that genuinely want the full scene base.
    pub fn region_prompt(
        &self,
        base: &str,
        person_prompt: Option<&str>,
        facing_phrase: Option<&str>,
    ) -> String {
        [Some(base), person_prompt, facing_phrase]
            .into_iter()
            .flatten()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A SINGLE-person region prompt for the per-persona inpaint pass: one person,
    /// their behavioural prompt, their facing, and the shared *style* clause for
    /// aesthetic consistency — but NOT the scene's head-count clause. The scene
    /// base already established the setting; the inpaint only needs to render one
    /// coherent person into the masked region with matching style. Including
    /// "three people …" here makes the region paint the whole crowd and dilutes
    /// the identity injected by the reference photos.
    pub fn single_region_prompt(
        &self,
        person_prompt: Option<&str>,
        facing_phrase: Option<&str>,
    ) -> String {
        [
            Some("a single person, one face"),
            person_prompt,
            facing_phrase,
            self.style_clause.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
    }

    /// A face-focused prompt for the identity-refinement pass: a portrait framing
    /// (plus-face's sweet spot) plus the persona's facing and the shared style.
    pub fn face_region_prompt(&self, facing_phrase: Option<&str>) -> String {
        [
            Some("a detailed face portrait, head and shoulders, sharp focus"),
            facing_phrase,
            self.style_clause.as_deref(),
        ]
        .into_iter()
        .flatten()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_inline_style_clause() {
        let p = MultipersonPrompt::parse("three friends at tea // oil painting, dappled light");
        assert_eq!(p.scene_clause, "three friends at tea");
        assert_eq!(p.style_clause.as_deref(), Some("oil painting, dappled light"));
        // analyser sees only the scene clause
        assert_eq!(p.for_analyser(), "three friends at tea");
    }

    #[test]
    fn no_separator_means_no_style_clause() {
        let p = MultipersonPrompt::parse("a couple in a cafe");
        assert!(p.style_clause.is_none());
    }

    #[test]
    fn enhancer_base_joins_layers_task_style_wins() {
        let p = MultipersonPrompt::parse("friends at tea // impressionist");
        let base = p.enhancer_base(Some("golden light"), Some("global oil"), Some("watercolor"));
        // scene + inline style + weather + TASK style (not global)
        assert_eq!(base, "friends at tea, impressionist, golden light, watercolor");
        // falls back to global when no task style
        let base2 = p.enhancer_base(Some("golden light"), Some("global oil"), None);
        assert!(base2.ends_with("global oil"));
    }

    #[test]
    fn region_prompt_appends_person_and_facing() {
        let p = MultipersonPrompt::parse("friends at tea");
        let base = p.enhancer_base(None, None, None);
        let rp = p.region_prompt(&base, Some("laughing"), Some("in profile, side view"));
        assert_eq!(rp, "friends at tea, laughing, in profile, side view");
    }

    #[test]
    fn single_region_prompt_drops_scene_keeps_style() {
        let p = MultipersonPrompt::parse(
            "three people playing chess // detailed digital painting",
        );
        let rp = p.single_region_prompt(Some("hand on chin"), Some("front view"));
        // never mentions the crowd ("three people"); keeps the style clause.
        assert!(!rp.contains("three people"));
        assert!(rp.starts_with("a single person, one face"));
        assert!(rp.contains("hand on chin"));
        assert!(rp.contains("front view"));
        assert!(rp.ends_with("detailed digital painting"));
    }

    #[test]
    fn face_region_prompt_is_portrait_framed() {
        let p = MultipersonPrompt::parse("three people // oil painting");
        let fp = p.face_region_prompt(Some("front view"));
        assert!(fp.starts_with("a detailed face portrait"));
        assert!(!fp.contains("three people"));
        assert!(fp.ends_with("oil painting"));
    }
}
