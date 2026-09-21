//! `plakat paint` — a stroke-space painting engine (RFC PAINT-1, the 7.0.0 flagship).
//!
//! The diffusion model never renders the deliverable. It produces a low-resolution structural **armature**;
//! a deterministic, weight-free **stroke engine** paints the full-resolution image from it in a declared
//! medium. The canonical artifact is a replayable **stroke score**. Everything below the armature merge is a
//! pure function of `(plan, seed)` and runs with no GPU.
//!
//! **Governing principle:** structure is low-resolution and model-derived; surface is full-resolution and
//! stroke-derived; the two never meet at the same scale. An armature has no detail to trace, so the brushwork
//! must invent the surface — which is what makes the strokes generative rather than a painterly filter.
//!
//! P0 (this milestone) builds the engine foundation and the filter-gate: the pigment/Kubelka-Munk colour
//! model and limited palettes here, then the pigment canvas, the bristle brush with pickup, the coarse-to-fine
//! painter over an input image, and the halo/cutout/traceability detectors that decide whether the thesis
//! holds.

pub mod canvas;
pub mod color;
pub mod mixer;
pub mod palette;
pub mod pigment;
pub mod stroke;

pub use canvas::Canvas;
pub use color::{Lab, Srgb};
pub use mixer::{solve_mixture, Mixture};
pub use palette::Palette;
pub use pigment::Pigment;
pub use stroke::{BrushConfig, Stroke};
