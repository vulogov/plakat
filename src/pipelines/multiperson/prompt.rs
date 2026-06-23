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
}
