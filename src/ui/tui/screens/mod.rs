//! TUI screen bodies (RFC TUI-1 §6–§13). Release 1: Models + Chat. Each screen
//! owns its state struct and a `render` + `handle_key`; the `App` holds them and
//! dispatches to the active one.

pub mod chat;
pub mod models;
pub mod prompt_editor;
pub mod scenarios;
