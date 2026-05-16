//! Scheduler wrappers.
//!
//! Currently the T2I pipeline uses the DDIM-style scheduler built by
//! `StableDiffusionConfig::build_scheduler(steps)` directly. This module
//! is reserved for adding Euler-Ancestral / DPM-Solver++ wrappers when
//! candle gains them or when we want a custom step rule (e.g. img2img
//! noise scheduling for the stylize pipeline).
