//! Background services for the TUI (RFC TUI-1 §4, §16). Long-running work (model
//! load/unload, generation, downloads, LLM calls) runs off the event-loop thread
//! and talks back over channels the `App` drains each tick.

pub mod model_service;
pub mod search_cache;
pub mod semantic;
