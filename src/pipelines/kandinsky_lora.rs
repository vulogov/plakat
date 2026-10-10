//! LoRA files for the Kandinsky 5 DiT (RFC KANDINSKY-1, §11.4's seam).
//!
//! The format is the reference trainer's (`kandinskylab/kandinsky-5-lora-train`): PEFT adapters on the
//! native module names, one pair of matrices a layer —
//!
//! ```text
//! base_model.model.visual_transformer_blocks.7.self_attention.to_query.lora_A.default.weight   (r, in)
//! base_model.model.visual_transformer_blocks.7.self_attention.to_query.lora_B.default.weight   (out, r)
//! ```
//!
//! — and nothing else: no alpha, since the reference trains at `lora_alpha = r`, a scale of one. The
//! diffusers checkpoint plakat loads keeps the native names, so a layer's key is its path in the model.
//! Read here as well: the same pairs without the adapter name (`….lora_A.weight`), under a
//! `transformer.` or `diffusion_model.` prefix, with kohya's `lora_down` / `lora_up`, and with an
//! `….alpha` scalar beside them (then the scale is `alpha / r`).
//!
//! At inference the adapters are merged into the weights as the DiT is read (`W += s · B · A`), before
//! NF4 quantization when that is on, so a LoRA costs nothing a step.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// The linear layers the reference trainer adapts, as the tail of a module path
/// (`configs/trainer/lora_image.yaml`). The text blocks have no cross-attention.
pub const TARGETS: [&str; 10] = [
    "self_attention.to_query",
    "self_attention.to_key",
    "self_attention.to_value",
    "self_attention.out_layer",
    "cross_attention.to_query",
    "cross_attention.to_key",
    "cross_attention.to_value",
    "cross_attention.out_layer",
    "feed_forward.in_layer",
    "feed_forward.out_layer",
];

/// The reference trainer's rank (and alpha).
pub const DEFAULT_RANK: usize = 32;

const PREFIXES: [&str; 3] = ["base_model.model.", "transformer.", "diffusion_model."];

/// One layer's adapter, F32 on the CPU: `a` is `(r, in)`, `b` is `(out, r)`.
#[derive(Debug, Clone)]
pub struct Pair {
    pub a: Tensor,
    pub b: Tensor,
    /// `alpha / r`; one when the file carries no alpha.
    pub scale: f64,
}

/// A LoRA file, keyed by module path (`visual_transformer_blocks.7.self_attention.to_query`).
#[derive(Debug, Clone, Default)]
pub struct LoraFile {
    pub layers: HashMap<String, Pair>,
}

/// Which half of a pair a tensor name is, and the module it belongs to.
fn classify(name: &str) -> Option<(&str, char)> {
    let name = PREFIXES.iter().find_map(|p| name.strip_prefix(p)).unwrap_or(name);
    for (marker, half) in [(".lora_A.", 'a'), (".lora_B.", 'b'), (".lora_down.", 'a'), (".lora_up.", 'b')] {
        if let Some(at) = name.find(marker) {
            return Some((&name[..at], half));
        }
    }
    name.strip_suffix(".alpha").map(|module| (module, 's'))
}

impl LoraFile {
    pub fn from_tensors(tensors: HashMap<String, Tensor>) -> Result<Self> {
        let (mut a, mut b, mut alpha) = (HashMap::new(), HashMap::new(), HashMap::new());
        for (name, t) in tensors {
            let Some((module, half)) = classify(&name) else { continue };
            let t = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            match half {
                'a' => a.insert(module.to_string(), t),
                'b' => b.insert(module.to_string(), t),
                _ => alpha.insert(module.to_string(), t),
            };
        }
        let mut layers = HashMap::new();
        for (module, a) in a {
            let b = b.remove(&module).with_context(|| format!("the LoRA has `lora_A` for {module} but no `lora_B`"))?;
            let (r, _) = a.dims2().with_context(|| format!("{module}: `lora_A` is not a matrix"))?;
            let (_, rb) = b.dims2().with_context(|| format!("{module}: `lora_B` is not a matrix"))?;
            anyhow::ensure!(r == rb && r > 0, "{module}: `lora_A` has rank {r} and `lora_B` has rank {rb}");
            let scale = match alpha.get(&module) {
                Some(t) => t.flatten_all()?.to_vec1::<f32>()?[0] as f64 / r as f64,
                None => 1.0,
            };
            layers.insert(module, Pair { a, b, scale });
        }
        anyhow::ensure!(!layers.is_empty(), "no `lora_A` / `lora_B` pairs in the file — not a Kandinsky 5 (PEFT) LoRA");
        Ok(Self { layers })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("reading the LoRA {}", path.display()))?;
        Self::from_tensors(tensors).with_context(|| format!("the LoRA {}", path.display()))
    }

    pub fn rank(&self) -> usize {
        self.layers.values().next().map_or(0, |p| p.a.dim(0).unwrap_or(0))
    }
}

/// The LoRAs of a run, each at its strength, and which of their layers the model took.
#[derive(Debug, Default)]
pub struct LoraSet {
    files: Vec<(LoraFile, f64, String)>,
    used: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl LoraSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file at `strength` (`--lora file:0.8`); `name` is what messages call it.
    pub fn push(&mut self, file: LoraFile, strength: f64, name: impl Into<String>) {
        self.files.push((file, strength, name.into()));
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The sum of every file's `strength · scale · B · A` for the layer at `module`, `(out, in)` in F32 on
    /// the CPU — `None` when no file adapts it.
    pub fn delta(&self, module: &str, out: usize, input: usize) -> Result<Option<Tensor>> {
        let mut sum: Option<Tensor> = None;
        for (file, strength, name) in &self.files {
            let Some(p) = file.layers.get(module) else { continue };
            anyhow::ensure!(p.b.dim(0)? == out && p.a.dim(1)? == input, "the LoRA {name} has a {}x{} adapter for {module}, which is {out}x{input} in this model", p.b.dim(0)?, p.a.dim(1)?);
            let d = (p.b.matmul(&p.a)? * (strength * p.scale))?;
            sum = Some(match sum {
                Some(s) => (s + d)?,
                None => d,
            });
            self.used.lock().unwrap().insert(module.to_string());
        }
        Ok(sum)
    }

    /// After the model is built: `(layers merged, layers in the files the model has no place for)`.
    pub fn report(&self) -> (usize, Vec<String>) {
        let used = self.used.lock().unwrap();
        let mut unmatched: Vec<String> = self.files.iter().flat_map(|(f, _, _)| f.layers.keys()).filter(|k| !used.contains(*k)).cloned().collect();
        unmatched.sort();
        unmatched.dedup();
        (used.len(), unmatched)
    }
}

/// Write adapters in the reference trainer's layout: BF16, `base_model.model.<module>.lora_{A,B}.default.weight`.
/// The scale is not stored — it is one (`alpha = r`), as in the reference.
pub fn save<'a>(adapters: impl IntoIterator<Item = (&'a str, &'a Tensor, &'a Tensor)>, out: &Path) -> Result<()> {
    let mut tensors = HashMap::new();
    for (module, a, b) in adapters {
        for (half, t) in [("lora_A", a), ("lora_B", b)] {
            tensors.insert(format!("base_model.model.{module}.{half}.default.weight"), t.detach().to_device(&Device::Cpu)?.to_dtype(DType::BF16)?);
        }
    }
    anyhow::ensure!(!tensors.is_empty(), "no adapters to save");
    if let Some(dir) = out.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    crate::pipelines::atomic_safetensors_save(&tensors, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, k: f32) -> Tensor {
        Tensor::from_vec((0..rows * cols).map(|i| (i as f32 * k).sin()).collect::<Vec<f32>>(), (rows, cols), &Device::Cpu).unwrap()
    }

    #[test]
    fn the_reference_and_the_other_spellings_name_the_same_layer() {
        let module = "visual_transformer_blocks.7.self_attention.to_query";
        for (name, half) in [
            (format!("base_model.model.{module}.lora_A.default.weight"), 'a'),
            (format!("base_model.model.{module}.lora_B.default.weight"), 'b'),
            (format!("transformer.{module}.lora_A.weight"), 'a'),
            (format!("diffusion_model.{module}.lora_up.weight"), 'b'),
            (format!("{module}.lora_down.weight"), 'a'),
            (format!("{module}.alpha"), 's'),
        ] {
            assert_eq!(classify(&name), Some((module, half)), "{name}");
        }
        assert_eq!(classify("visual_transformer_blocks.7.self_attention.to_query.weight"), None);
    }

    #[test]
    fn a_saved_file_reads_back_and_merges_at_its_strength() {
        let module = "visual_transformer_blocks.0.feed_forward.in_layer";
        let (a, b) = (mat(4, 8, 0.3), mat(16, 4, 0.7));
        let dir = std::env::temp_dir().join(format!("plakat-k5-lora-{}", std::process::id()));
        let path = dir.join("x.safetensors");
        save([(module, &a, &b)], &path).unwrap();
        let file = LoraFile::load(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!((file.rank(), file.layers.len()), (4, 1));
        let mut set = LoraSet::new();
        set.push(file, 0.5, "x");
        let d = set.delta(module, 16, 8).unwrap().unwrap();
        // BF16 on disk: the product is close to the F32 one, not equal to it.
        let want = (b.matmul(&a).unwrap() * 0.5).unwrap();
        let worst = (d - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(worst < 0.02, "{worst}");
        assert!(set.delta("visual_transformer_blocks.1.feed_forward.in_layer", 16, 8).unwrap().is_none());
        assert_eq!(set.report(), (1, Vec::new()));
        // A layer of the wrong size is an error, not a silent skip.
        assert!(set.delta(module, 16, 9).is_err());
    }

    #[test]
    fn alpha_scales_a_pair_and_two_files_add() {
        let module = "text_transformer_blocks.0.self_attention.to_key";
        let (a, b) = (mat(2, 6, 0.3), mat(6, 2, 0.7));
        let with_alpha = LoraFile::from_tensors(HashMap::from([
            (format!("{module}.lora_down.weight"), a.clone()),
            (format!("{module}.lora_up.weight"), b.clone()),
            (format!("{module}.alpha"), Tensor::new(1f32, &Device::Cpu).unwrap()),
        ]))
        .unwrap();
        assert_eq!(with_alpha.layers[module].scale, 0.5);
        let plain = LoraFile::from_tensors(HashMap::from([(format!("{module}.lora_A.weight"), a.clone()), (format!("{module}.lora_B.weight"), b.clone())])).unwrap();
        let mut set = LoraSet::new();
        set.push(with_alpha, 1.0, "half");
        set.push(plain, 1.0, "whole");
        let d = set.delta(module, 6, 6).unwrap().unwrap();
        let want = (b.matmul(&a).unwrap() * 1.5).unwrap();
        let worst = (d - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
        assert!(worst < 1e-6, "{worst}");
        // A file with one half of a pair, or with no pair at all, is refused.
        assert!(LoraFile::from_tensors(HashMap::from([(format!("{module}.lora_A.weight"), a)])).is_err());
        assert!(LoraFile::from_tensors(HashMap::from([("weight".to_string(), b)])).is_err());
    }
}
