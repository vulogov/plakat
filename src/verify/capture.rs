//! `TensorTap` — the capture abstraction pipelines use to record named intermediate tensors
//! during a Tier-1 verification run. Mirrors `StepHook`: a pipeline takes an
//! `Option<&mut dyn TensorTap>` and, in production, it's `None` so every capture call is
//! elided (zero cost, no behavior change). Wiring the actual capture points into the
//! pipelines is Phase 1b; this lands the mechanism + the collector.

use std::collections::{HashMap, HashSet};

use candle_core::Tensor;

/// A sink for named intermediate tensors.
pub trait TensorTap {
    /// Whether a capture named `name` is wanted — lets a pipeline skip materializing a
    /// capture it won't need.
    fn wants(&self, name: &str) -> bool;
    /// Record `tensor` under `name` (called only when `wants(name)` is true).
    fn capture(&mut self, name: &str, tensor: &Tensor);
}

/// The helper pipelines call at a capture point: `tap(&mut tap, "clip_l.penultimate", &t)`.
/// No-ops when there's no tap or the name isn't wanted — the whole thing folds away in a
/// production (`None`) run.
pub fn tap(sink: &mut Option<&mut dyn TensorTap>, name: &str, tensor: &Tensor) {
    if let Some(s) = sink {
        if s.wants(name) {
            s.capture(name, tensor);
        }
    }
}

/// Collects exactly the captures a manifest asked for into a name→tensor map.
pub struct CaptureBag {
    wanted: HashSet<String>,
    captured: HashMap<String, Tensor>,
}

impl CaptureBag {
    pub fn new(wanted: impl IntoIterator<Item = String>) -> Self {
        Self { wanted: wanted.into_iter().collect(), captured: HashMap::new() }
    }
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.captured.get(name)
    }
    pub fn captured(&self) -> &HashMap<String, Tensor> {
        &self.captured
    }
    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }
}

impl TensorTap for CaptureBag {
    fn wants(&self, name: &str) -> bool {
        self.wanted.contains(name)
    }
    fn capture(&mut self, name: &str, tensor: &Tensor) {
        if self.wanted.contains(name) {
            self.captured.insert(name.to_string(), tensor.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn bag_records_only_wanted_captures() {
        let mut bag = CaptureBag::new(["a".to_string(), "b".to_string()]);
        {
            let mut sink: Option<&mut dyn TensorTap> = Some(&mut bag);
            let t = Tensor::new(&[1f32, 2.], &Device::Cpu).unwrap();
            tap(&mut sink, "a", &t); // wanted → stored
            tap(&mut sink, "c", &t); // not wanted → dropped
        }
        assert!(bag.get("a").is_some());
        assert!(bag.get("c").is_none(), "unwanted captures are ignored");
        assert_eq!(bag.captured().len(), 1);
    }

    #[test]
    fn none_sink_is_a_noop() {
        // The production path: no tap → capture folds away.
        let mut sink: Option<&mut dyn TensorTap> = None;
        let t = Tensor::new(&[1f32], &Device::Cpu).unwrap();
        tap(&mut sink, "a", &t); // must not panic / do anything
    }
}
