//! v0.16 phase 7: Civitai browser + downloader.
//!
//! [Civitai](https://civitai.com) is the major community hub for
//! Stable Diffusion checkpoints, LoRAs, embeddings, and ControlNet
//! variants. plakat doesn't depend on it (every existing flag still
//! works against HF / local paths), but a built-in browser cuts the
//! "find a LoRA → copy URL → curl → reference local path" loop down
//! to one CLI invocation.
//!
//! Two surfaces:
//!
//! * [`api`] — typed wrappers over the public REST endpoints
//!   (search + model + model-version lookups). Optional
//!   `CIVITAI_API_KEY` env var for rate-limit lift + gated-model
//!   access; works without one for public assets.
//!
//! * [`download`] — file downloader writing into
//!   `<plakat-cache>/civitai/<model-id>/<version-id>/`. Returns the
//!   absolute path the user can drop into `--lora` /
//!   `--model PATH` directly.

pub mod api;
pub mod download;
