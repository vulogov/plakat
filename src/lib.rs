//! plakat — local text-to-image, style transfer, LoRA, upscale, and
//! color-key generation built on candle.
//!
//! # Using plakat as a library
//!
//! **[`plakat::api`](crate::api) is the supported, stable, documented surface.** Start there:
//! builder types (`Generate`, `Img2img`, `Upscale`, `Relight`, `Stylize`, `Transparent`,
//! `Segment`, `Portrait`, `Multiperson`, `Map`, `Animate`, `StyleTrain`, `EmbeddingTrain`,
//! `Verify`) cover everything the CLI does except the interactive UI.
//!
//! ```no_run
//! # async fn ex() -> anyhow::Result<()> {
//! let images = plakat::api::Generate::new("sdxl").prompt("a fox").run().await?;
//! images[0].save("fox.png")?;
//! # Ok(()) }
//! ```
//!
//! Everything else below is **implementation detail**. Those modules are `pub` because the
//! `plakat` binary (`src/main.rs`), examples, and tests share them, but they are `#[doc(hidden)]`
//! and carry **no semver stability promise** — they churn between releases. Do not build on
//! them; if `plakat::api` is missing something you need, please open an issue.

pub mod api;

#[doc(hidden)]
pub mod artefacts;
#[doc(hidden)]
pub mod capability;
#[doc(hidden)]
pub mod civitai;
#[doc(hidden)]
pub mod cli;
#[doc(hidden)]
pub mod compile;
#[doc(hidden)]
pub mod config;
#[doc(hidden)]
pub mod device;
#[doc(hidden)]
pub mod error_hints;
#[doc(hidden)]
pub mod hf;
#[doc(hidden)]
pub mod hw;
#[doc(hidden)]
pub mod imaging;
#[doc(hidden)]
pub mod instance_guard;
#[doc(hidden)]
pub mod map;
#[doc(hidden)]
pub mod memwatch;
#[doc(hidden)]
pub mod llm;
#[doc(hidden)]
pub mod pipelines;
/// Local, model-free TF-IDF cosine text search (History + `plakat photos`). Feature-agnostic.
pub mod textsearch;
/// v3.0 flagship: `plakat photos` TUI collection manager (RFC PHOTOS-1). Behind `--features photos`.
#[cfg(feature = "photos")]
pub mod photos;
/// v4.1 flagship: `plakat fractals` — non-AI + AI-assisted fractal generation (RFC FRACTALS-1).
/// Behind `--features fractals`.
#[cfg(feature = "fractals")]
#[doc(hidden)]
pub mod fractals;
/// v5.0 flagship: `plakat persona` — controllable synthetic-person composition (RFC PERSONA-1).
/// The deterministic compiler half (spec/lexicon/resolver/emitters) is pure and weight-free.
pub mod persona;
pub mod bookart;
/// v6.3 flagship: `plakat texture` — seamless PBR material synthesis (RFC TEXTURE-1). Always compiled
/// (the deterministic front half + derivation are weight-free).
pub mod texture;
/// v6.8 flagship: `plakat comic` — multi-panel comic pages (RFC COMIC-1). Always compiled (the layout /
/// balloon / page-composite half is weight-free).
pub mod comic;
/// v6.7 flagship: provenance etching (RFC ETCH-1) — `--etch` + `doctor --if-plakat`. Always compiled
/// (not behind a cargo feature) so the `--no-default-features` gate covers it; the `--etch` flag opts in.
pub mod etch;
#[doc(hidden)]
pub mod preset;
#[doc(hidden)]
pub mod prompt;
#[doc(hidden)]
pub mod scripting;
#[doc(hidden)]
pub mod style;
#[doc(hidden)]
pub mod ui;
#[doc(hidden)]
pub mod verify;
