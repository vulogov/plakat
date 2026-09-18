//! LAYERED-1 (`plakat layers`) — plan-guided layered generation (RFC LAYERED-1).
//!
//! A complex prompt is split into a **plan**: a backdrop plus independent subject layers, each with its own
//! full-budget prompt, box and depth. Each layer is drafted ALONE (binding is trivial with one subject),
//! the drafts become a **low-frequency guide latent**, and the finish is ONE ordinary denoising trajectory
//! of the user's model, steered toward the guide only in its low frequencies during early steps (the
//! `refine_latent` seam wired across every family in P0). Layers are constraints, never pixels.
//!
//! Named `layered` (not `layers`) to avoid colliding with `photos::layers`. This module is the P1 engine
//! core (plan / lint / classifier / draft / guide / hook / render with hand-written plans); verify/repair
//! (S4) and lift (S5) land in P3.

pub mod draft;
pub mod hook;
pub mod lint;
pub mod plan;
