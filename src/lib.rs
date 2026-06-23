//! plakat — local text-to-image, style transfer, LoRA, upscale, and
//! color-key CLI built on candle.
//!
//! This crate is primarily consumed as the `plakat` binary defined in
//! `src/main.rs`. The library target exists so examples (under
//! `examples/`) and integration tests (under `tests/`) can reach into
//! the same module surface the binary uses, without duplicating
//! non-trivial code like CLIP-H preprocessing or model loading.

pub mod artefacts;
pub mod capability;
pub mod civitai;
pub mod cli;
pub mod compile;
pub mod config;
pub mod device;
pub mod error_hints;
pub mod hf;
pub mod hw;
pub mod imaging;
pub mod instance_guard;
pub mod map;
pub mod memwatch;
pub mod llm;
pub mod pipelines;
pub mod preset;
pub mod prompt;
pub mod scripting;
pub mod style;
pub mod ui;
