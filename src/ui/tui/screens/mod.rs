//! TUI screen bodies (RFC TUI-1 §6–§13). Release 1: Models + Chat. Each screen
//! owns its state struct and a `render` + `handle_key`; the `App` holds them and
//! dispatches to the active one.

pub mod canvas;
pub mod chat;
pub mod history;
pub mod lorahub;
pub mod models;
pub mod naturalize;
pub mod palette;
pub mod people;
pub mod prompts;
pub mod prompt_editor;
pub mod scenarios;
