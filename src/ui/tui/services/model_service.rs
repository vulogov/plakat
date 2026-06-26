//! `ModelService` — a dedicated background thread that owns the loaded model and
//! serialises load/unload (RFC TUI-1 §16). A multi-GB load can't run on the event
//! loop without freezing the UI, and the loaded pipeline isn't moved across threads
//! (so no `Send` gymnastics): one thread owns it, receives [`ModelCommand`]s, and
//! reports [`ModelMessage`]s the `App` drains each tick. The same thread will run
//! generations later (Metal device exclusivity for free).
//!
//! This increment loads the SD-family t2i pipeline (sd15 / sd21 / sdxl). Other
//! families return a friendly "not yet wired" message rather than a cryptic error.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use candle_core::Device;
use tokio::runtime::Handle;

use crate::pipelines::t2i;
use crate::preset::discovery::BaseFamily;

/// A request to the model thread.
pub enum ModelCommand {
    Load(String),
    Unload,
    Shutdown,
}

/// A status update from the model thread.
#[derive(Debug, Clone)]
pub enum ModelMessage {
    /// A load has begun (the multi-GB download/map runs in the background).
    LoadStarted(String),
    /// Loaded; `used_gb` is the system memory in use just after the load.
    Loaded { alias: String, used_gb: f64 },
    /// Unloaded — no model resident.
    Unloaded,
    /// A load/unload failed (or the model family isn't wired yet).
    Error(String),
}

/// Whether the t2i loader handles this model's family. Pure — unit-tested. Returns
/// `Err(message)` with friendly guidance for families not yet wired in the TUI.
pub fn t2i_load_check(alias: &str) -> Result<(), String> {
    match BaseFamily::from_model_arg(alias) {
        BaseFamily::Sd15 | BaseFamily::Sd21 | BaseFamily::Sdxl => Ok(()),
        other => Err(format!(
            "'{alias}' is a {other:?} model — the TUI loads SD-family models \
             (sd15 / sd21 / sdxl) in this release; {other:?} support is a follow-up."
        )),
    }
}

/// Handle to the model thread. Drop signals shutdown and joins.
pub struct ModelService {
    cmd_tx: Sender<ModelCommand>,
    msg_rx: Receiver<ModelMessage>,
    handle: Option<JoinHandle<()>>,
}

impl ModelService {
    /// Spawn the model thread. `rt` is the app's tokio runtime handle, used to
    /// `block_on` the async pipeline load from this (non-runtime) thread.
    pub fn spawn(device: Device, rt: Handle) -> Self {
        let (cmd_tx, cmd_rx) = channel::<ModelCommand>();
        let (msg_tx, msg_rx) = channel::<ModelMessage>();
        let handle = std::thread::Builder::new()
            .name("plakat-model-svc".into())
            .spawn(move || model_loop(cmd_rx, msg_tx, device, rt))
            .expect("spawn model service");
        Self { cmd_tx, msg_rx, handle: Some(handle) }
    }

    pub fn load(&self, alias: impl Into<String>) {
        let _ = self.cmd_tx.send(ModelCommand::Load(alias.into()));
    }

    pub fn unload(&self) {
        let _ = self.cmd_tx.send(ModelCommand::Unload);
    }

    /// Non-blocking drain of one status message (called from the event-loop tick).
    pub fn try_recv(&self) -> Option<ModelMessage> {
        self.msg_rx.try_recv().ok()
    }
}

impl Drop for ModelService {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(ModelCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn model_loop(
    cmd_rx: Receiver<ModelCommand>,
    msg_tx: Sender<ModelMessage>,
    device: Device,
    rt: Handle,
) {
    // The loaded pipeline lives here and never leaves this thread.
    let mut loaded: Option<(String, t2i::Pipeline)> = None;
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            ModelCommand::Load(alias) => {
                if let Err(msg) = t2i_load_check(&alias) {
                    let _ = msg_tx.send(ModelMessage::Error(msg));
                    continue;
                }
                let _ = msg_tx.send(ModelMessage::LoadStarted(alias.clone()));
                // Free the current model first — unified memory means we can't hold
                // two large models at once.
                loaded = None;
                let req = t2i::LoadRequest {
                    model: alias.clone(),
                    device: device.clone(),
                    loras: Vec::new(),
                    lora_scale: 1.0,
                    use_refiner: false,
                    embeddings: Vec::new(),
                    vae_cache: None,
                };
                match rt.block_on(t2i::Pipeline::load(req)) {
                    Ok(p) => {
                        let used = (crate::hw::total_ram_gb() - crate::hw::available_ram_gb()).max(0.0);
                        loaded = Some((alias.clone(), p));
                        let _ = msg_tx.send(ModelMessage::Loaded { alias, used_gb: used });
                    }
                    Err(e) => {
                        let _ = msg_tx.send(ModelMessage::Error(format!("load {alias}: {e:#}")));
                    }
                }
            }
            ModelCommand::Unload => {
                loaded = None;
                let _ = msg_tx.send(ModelMessage::Unloaded);
            }
            ModelCommand::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd_family_passes_the_load_check() {
        assert!(t2i_load_check("sdxl").is_ok());
        assert!(t2i_load_check("sd15").is_ok());
        assert!(t2i_load_check("sd21").is_ok());
    }

    #[test]
    fn other_families_get_a_friendly_error() {
        let e = t2i_load_check("flux-dev").unwrap_err();
        assert!(e.contains("Flux"));
        assert!(e.contains("follow-up"));
        assert!(t2i_load_check("sd35-medium").is_err());
        assert!(t2i_load_check("pixart-sigma").is_err());
    }
}
