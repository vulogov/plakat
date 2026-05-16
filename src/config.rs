use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub deepseek_api_key: Option<String>,
    #[serde(default)]
    pub gemini_api_key: Option<String>,
    #[serde(default)]
    pub hf_token: Option<String>,
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let mut cfg = match config_path() {
            Some(p) if p.exists() => {
                let s = std::fs::read_to_string(&p)?;
                toml::from_str::<Self>(&s)?
            }
            _ => Self::default(),
        };
        if cfg.deepseek_api_key.is_none() {
            cfg.deepseek_api_key = std::env::var("DEEPSEEK_API_KEY").ok();
        }
        if cfg.gemini_api_key.is_none() {
            cfg.gemini_api_key = std::env::var("GEMINI_API_KEY").ok();
        }
        if cfg.hf_token.is_none() {
            cfg.hf_token = std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
        }
        Ok(cfg)
    }
}

pub fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("ai", "plakat", "plakat")
        .map(|d| d.config_dir().join("config.toml"))
}
