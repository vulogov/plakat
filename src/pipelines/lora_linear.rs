//! Runtime LoRA infrastructure — v0.15 phase 7b-1.
//!
//! A `LoraLinear` wraps a regular `candle_nn::Linear` and applies a
//! mutable stack of LoRA delta-pairs at forward time:
//!
//! ```text
//!     y = base(x) + Σ_i scale_i · (B_i @ (A_i @ x))
//! ```
//!
//! The LoRA stack lives behind an `Arc<RwLock<Vec<LoraSlot>>>` so the
//! model can be called via `&self` (the candle `Module` shape) while
//! the dispatch layer swaps LoRAs through `set_loras` / `clear_loras`.
//!
//! ## Design notes
//!
//! * **B is pre-padded to full output dim** at `set_loras` time. PEFT
//!   LoRAs commonly target a slice of a fused Linear (e.g. the Q rows
//!   of a fused QKV). Rather than carrying slice metadata through the
//!   forward path and rebuilding `y` via slice-add-cat per step, we
//!   pay a small memory cost up front: zero-pad the B matrix to
//!   `(out_dim, rank)` so the forward is a uniform `broadcast_add`.
//!   For a typical rank-32 LoRA targeting Flux's QKV (out=9216,
//!   in=3072, slice_size=3072), the padding adds ~590 KB per slot —
//!   negligible across the ~500 Linears in a typical Flux LoRA target
//!   set.
//!
//! * **Effective scale = `(alpha / rank) * user_scale`** is folded
//!   into `LoraSpec.scale` before `set_loras`. Forward does one
//!   `tensor * scale_f64` per slot.
//!
//! * **Internal mutability** uses `RwLock`, not `RefCell`, so the
//!   same `LoraLinear` can be shared across threads if a future
//!   release wires concurrent inference. Current plakat inference is
//!   single-threaded; the RwLock is uncontended in practice.
//!
//! ## What's NOT here yet
//!
//! Phase 7b-1 ships just the wrapper + tests. The plumbing to
//! actually substitute every model `nn::Linear` for a `LoraLinear`,
//! and to drive `set_loras` from a scenario's per-task LoRA stack,
//! lives in 7b-2 (NF4), 7b-3 (Flux BF16), 7b-4 (Flux GGUF),
//! 7b-5 (MMDiT), 7b-6 (SD UNet), and 7b-7 (scenario dispatch).

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn as nn;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// One active LoRA, stored in the runtime LoRA stack of a
/// [`LoraLinear`]. Pre-cast to the base Linear's dtype and pre-padded
/// to the full output dim — `forward` is a single uniform broadcast
/// add over the result of `B @ A @ x`.
#[derive(Debug, Clone)]
pub struct LoraSlot {
    /// LoRA-A matrix, shape `(rank, in_dim)`. Stored in the runtime
    /// dtype the base Linear uses (typically BF16 on GPU).
    pub a: Tensor,
    /// LoRA-B matrix, shape `(out_dim, rank)`. Zero-padded outside
    /// the target row slice — see module docs.
    pub b: Tensor,
    /// Effective scale = `(alpha / rank) * user_scale`. Applied via
    /// scalar multiply on the delta tensor.
    pub scale: f32,
}

/// One pre-pad LoRA specification — what the dispatch layer hands to
/// [`LoraLinear::set_loras`]. The B matrix is the LoRA's natural
/// shape (out rows match the row slice's size); padding to the full
/// base output dim happens inside `set_loras` based on `row_slice`.
#[derive(Debug, Clone)]
pub struct LoraSpec {
    /// LoRA-A, shape `(rank, in_dim)`.
    pub a: Tensor,
    /// LoRA-B, shape `(slice_size, rank)` where `slice_size` matches
    /// the target row slice (or `out_dim` for `Full`).
    pub b: Tensor,
    /// Effective scale.
    pub scale: f32,
    /// Row slice of the base Linear's output this LoRA targets.
    /// `None` is equivalent to `Full` — the LoRA covers the entire
    /// output dim and no padding is needed.
    pub row_slice: Option<(usize, usize)>,
}

/// Runtime-LoRA-enabled wrapper around `candle_nn::Linear`. Holds the
/// base Linear immutably and a mutable stack of [`LoraSlot`] applied
/// at forward time. Initial stack is empty — the LoraLinear behaves
/// byte-identically to the wrapped Linear until `set_loras` is
/// called.
///
/// `Clone` clones the base Linear (cheap — its weight is a `Tensor`
/// which is internally an `Arc`) and the `slots` Arc (shared
/// reference; cloned LoraLinears see the same active LoRAs).
#[derive(Debug, Clone)]
pub struct LoraLinear {
    base: nn::Linear,
    /// Cached output dim (extracted from `base.weight().dim(0)` at
    /// construction so we don't re-read it inside `set_loras`).
    out_dim: usize,
    /// Cached input dim (`base.weight().dim(1)`).
    in_dim: usize,
    /// Shared mutable LoRA stack. Cloning the `Arc` is cheap and
    /// gives external code a handle for keyed registries (see
    /// future 7b-2+ subphases).
    slots: Arc<RwLock<Vec<LoraSlot>>>,
}

impl LoraLinear {
    /// Wrap a Linear loaded from safetensors. The runtime LoRA stack
    /// starts empty.
    pub fn from_linear(base: nn::Linear) -> Result<Self> {
        let w = base.weight();
        let (out_dim, in_dim) = w.dims2()?;
        Ok(Self {
            base,
            out_dim,
            in_dim,
            slots: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Hand out an `Arc` handle to the shared LoRA stack. Used by the
    /// future `LoraRegistry` (7b-7) to drive `set_loras` from a
    /// path-keyed map without holding direct references to every
    /// LoraLinear in the model.
    pub fn slots_handle(&self) -> Arc<RwLock<Vec<LoraSlot>>> {
        self.slots.clone()
    }

    /// Cached output / input dims.
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Replace the active LoRA stack. Padding the B matrices to the
    /// full output dim happens here. `dtype` should match the base
    /// Linear's weight dtype (typically BF16 on GPU); `device` likewise.
    pub fn set_loras(
        &self,
        specs: Vec<LoraSpec>,
        dtype: DType,
        device: &Device,
    ) -> Result<()> {
        let mut new_slots = Vec::with_capacity(specs.len());
        for spec in specs {
            let b_padded = pad_b_to_out_dim(
                &spec.b,
                spec.row_slice,
                self.out_dim,
                dtype,
                device,
            )?;
            new_slots.push(LoraSlot {
                a: spec.a.to_dtype(dtype)?,
                b: b_padded,
                scale: spec.scale,
            });
        }
        *self.slots.write().expect("LoraLinear slots poisoned") = new_slots;
        Ok(())
    }

    /// Drop every active LoRA. Forward reverts to pure base-Linear
    /// behaviour (byte-identical to wrapping the same Linear without
    /// LoraLinear at all).
    pub fn clear_loras(&self) {
        self.slots
            .write()
            .expect("LoraLinear slots poisoned")
            .clear();
    }

    /// Number of currently-active LoRAs. Cheap read lock.
    pub fn n_loras(&self) -> usize {
        self.slots
            .read()
            .expect("LoraLinear slots poisoned")
            .len()
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut y = self.base.forward(x)?;
        let slots = self
            .slots
            .read()
            .map_err(|_| candle_core::Error::Msg("LoraLinear slots poisoned".into()))?;
        for slot in slots.iter() {
            // x: (..., in_dim)
            // A: (rank, in_dim) → A.T: (in_dim, rank)
            // lo = x @ A.T: (..., rank)
            let lo = x.broadcast_matmul(&slot.a.t()?)?;
            // B: (out_dim, rank) → B.T: (rank, out_dim)
            // delta = lo @ B.T: (..., out_dim)
            let delta = lo.broadcast_matmul(&slot.b.t()?)?;
            let delta = (delta * slot.scale as f64)?;
            y = y.broadcast_add(&delta)?;
        }
        Ok(y)
    }
}

// =====================================================================
// v0.15 phase 7b-3: shared LoRA registry types — promoted from
// `flux_nf4_inner` so every backbone (NF4 / Flux BF16 / Flux GGUF /
// MMDiT / SD UNet) uses the same path-keyed structure.
// =====================================================================

/// One LoRA-registry entry — a handle into the target Linear's runtime
/// LoRA stack plus the metadata needed to pad LoRA-B matrices to the
/// right size in the backbone's `apply_loras`. Storing `out_dim` /
/// `in_dim` here avoids walking the model structure at apply time.
///
/// `Clone` clones the `Arc<RwLock<...>>` handle (cheap, shared
/// reference). `Debug` is derived so consuming backbones can
/// derive `Debug` on their top-level model struct.
#[derive(Debug, Clone)]
pub struct LoraRegistryEntry {
    pub handle: Arc<RwLock<Vec<LoraSlot>>>,
    pub out_dim: usize,
    pub in_dim: usize,
}

/// Path → entry map, keyed by full safetensors key (including the
/// trailing `.weight`). Each backbone's model constructor populates
/// this during weight loading; `apply_loras` consumes it.
pub type LoraRegistry = HashMap<String, LoraRegistryEntry>;

// =====================================================================
// pad_b_to_out_dim — exported for backbone runtime-LoRA wrappers.
// =====================================================================

/// Zero-pad a LoRA-B matrix from the natural slice shape
/// `(slice_size, rank)` to the full output shape `(out_dim, rank)`.
/// `row_slice = None` is the identity (B already covers the full
/// output; just dtype-cast).
///
/// `pub(crate)` so the per-backbone runtime LoRA wrappers (7b-2:
/// NF4, 7b-3: BF16, 7b-4: GGUF, 7b-5: MMDiT, 7b-6: SD UNet) can
/// build `LoraSlot`s with the same padding rules `LoraLinear` uses.
pub(crate) fn pad_b_to_out_dim(
    b: &Tensor,
    row_slice: Option<(usize, usize)>,
    out_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let b = b.to_dtype(dtype)?;
    let (start, end) = match row_slice {
        None => {
            // Full slice — sanity-check that B's row count matches.
            let (got_rows, _rank) = b.dims2()?;
            if got_rows != out_dim {
                anyhow::bail!(
                    "LoRA B full-slice mismatch: B has {got_rows} rows but base \
                     out_dim is {out_dim}"
                );
            }
            return Ok(b);
        }
        Some(s) => s,
    };
    let (slice_rows, rank) = b.dims2()?;
    let want = end.saturating_sub(start);
    if slice_rows != want {
        anyhow::bail!(
            "LoRA B partial-slice mismatch: B has {slice_rows} rows but slice \
             [{start}, {end}) wants {want}"
        );
    }
    if end > out_dim {
        anyhow::bail!(
            "LoRA B slice end {end} exceeds base out_dim {out_dim}"
        );
    }
    let head = Tensor::zeros((start, rank), dtype, device)?;
    let tail = Tensor::zeros((out_dim - end, rank), dtype, device)?;
    Ok(Tensor::cat(&[&head, &b, &tail], 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Build a `(out, in)` zero-Linear and wrap it as LoraLinear.
    /// Used as the test stub — base contribution is always zero, so
    /// `forward(x)` returns exactly the LoRA delta sum.
    fn zero_linear(out: usize, in_: usize) -> LoraLinear {
        let vmap = candle_nn::VarMap::new();
        // Insert zero tensors for weight + bias under the VarMap.
        vmap.get((out, in_), "weight", candle_nn::Init::Const(0.0), DType::F32, &cpu())
            .unwrap();
        vmap.get((out,), "bias", candle_nn::Init::Const(0.0), DType::F32, &cpu())
            .unwrap();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, &cpu());
        let base = candle_nn::linear(in_, out, vb).unwrap();
        LoraLinear::from_linear(base).unwrap()
    }

    #[test]
    fn empty_stack_returns_base_output() {
        // No LoRAs active → forward equals base.forward (which is
        // zeros for the zero_linear stub).
        let ll = zero_linear(4, 3);
        let x = Tensor::ones((1, 3), DType::F32, &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(yvec, vec![0.0; 4]);
        assert_eq!(ll.n_loras(), 0);
    }

    #[test]
    fn single_lora_full_slice_adds_to_output() {
        // out=2, in=2. A = identity_2x2, B = identity_2x2, scale=1.
        // delta = B @ A @ x = x. So y = 0 + 1*x = x.
        let ll = zero_linear(2, 2);
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![LoraSpec {
                a,
                b,
                scale: 1.0,
                row_slice: None,
            }],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        assert_eq!(ll.n_loras(), 1);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yvec[0] - 3.0).abs() < 1e-5);
        assert!((yvec[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn lora_scale_multiplies_delta() {
        // Same identity LoRA but scale=0.5 → y = 0 + 0.5x.
        let ll = zero_linear(2, 2);
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![LoraSpec {
                a,
                b,
                scale: 0.5,
                row_slice: None,
            }],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        let x = Tensor::from_vec(vec![4.0f32, 6.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yvec[0] - 2.0).abs() < 1e-5);
        assert!((yvec[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn two_loras_compose_additively() {
        // Two LoRAs, each contributing 1*identity*x → y = 2x.
        let ll = zero_linear(2, 2);
        let id = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![
                LoraSpec {
                    a: id.clone(),
                    b: id.clone(),
                    scale: 1.0,
                    row_slice: None,
                },
                LoraSpec {
                    a: id.clone(),
                    b: id.clone(),
                    scale: 1.0,
                    row_slice: None,
                },
            ],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        let x = Tensor::from_vec(vec![7.0f32, 11.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yvec[0] - 14.0).abs() < 1e-5);
        assert!((yvec[1] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn partial_row_slice_zero_outside_slice() {
        // Base out=4, in=2. LoRA targets rows [1, 3) only.
        // A = I_2, B = I_2 (slice_size=2, rank=2). x = [1, 1].
        // delta_slice = B @ A @ x = [1, 1] over rows [1, 3).
        // Padded delta = [0, 1, 1, 0]. y = 0 + 1*[0,1,1,0].
        let ll = zero_linear(4, 2);
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![LoraSpec {
                a,
                b,
                scale: 1.0,
                row_slice: Some((1, 3)),
            }],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((yvec[0] - 0.0).abs() < 1e-5, "row 0 should stay 0, got {}", yvec[0]);
        assert!((yvec[1] - 1.0).abs() < 1e-5, "row 1 (slice start) got {}", yvec[1]);
        assert!((yvec[2] - 1.0).abs() < 1e-5, "row 2 got {}", yvec[2]);
        assert!((yvec[3] - 0.0).abs() < 1e-5, "row 3 should stay 0, got {}", yvec[3]);
    }

    #[test]
    fn clear_loras_restores_base() {
        // Apply a LoRA, then clear; forward should match the empty-
        // stack output again.
        let ll = zero_linear(2, 2);
        let id = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![LoraSpec {
                a: id.clone(),
                b: id.clone(),
                scale: 1.0,
                row_slice: None,
            }],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        assert_eq!(ll.n_loras(), 1);
        ll.clear_loras();
        assert_eq!(ll.n_loras(), 0);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &cpu()).unwrap();
        let y = ll.forward(&x).unwrap();
        let yvec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(yvec, vec![0.0, 0.0]);
    }

    #[test]
    fn set_loras_replaces_stack() {
        // First call sets 2 LoRAs. Second call sets 1. n_loras should
        // reflect the latest set.
        let ll = zero_linear(2, 2);
        let id = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &cpu()).unwrap();
        ll.set_loras(
            vec![
                LoraSpec {
                    a: id.clone(),
                    b: id.clone(),
                    scale: 1.0,
                    row_slice: None,
                },
                LoraSpec {
                    a: id.clone(),
                    b: id.clone(),
                    scale: 1.0,
                    row_slice: None,
                },
            ],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        assert_eq!(ll.n_loras(), 2);
        ll.set_loras(
            vec![LoraSpec {
                a: id.clone(),
                b: id.clone(),
                scale: 1.0,
                row_slice: None,
            }],
            DType::F32,
            &cpu(),
        )
        .unwrap();
        assert_eq!(ll.n_loras(), 1);
    }

    #[test]
    fn pad_b_full_slice_is_identity() {
        let b =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &cpu()).unwrap();
        let p = pad_b_to_out_dim(&b, None, 3, DType::F32, &cpu()).unwrap();
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn pad_b_partial_zero_pads_head_and_tail() {
        // B is (2, 2) = [[1,2],[3,4]]. Target slice [1, 3) of an
        // out_dim=5 base. Expected padded: [[0,0], [1,2], [3,4], [0,0], [0,0]].
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &cpu()).unwrap();
        let p = pad_b_to_out_dim(&b, Some((1, 3)), 5, DType::F32, &cpu()).unwrap();
        let v = p.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_b_rejects_slice_size_mismatch() {
        // B has 3 rows but slice [1, 3) wants 2 rows.
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &cpu()).unwrap();
        let err = pad_b_to_out_dim(&b, Some((1, 3)), 5, DType::F32, &cpu()).unwrap_err();
        assert!(format!("{err}").contains("partial-slice mismatch"));
    }

    #[test]
    fn pad_b_rejects_slice_out_of_bounds() {
        // Slice [3, 6) but out_dim is 5.
        let b = Tensor::from_vec(vec![1.0f32; 6], (3, 2), &cpu()).unwrap();
        let err = pad_b_to_out_dim(&b, Some((3, 6)), 5, DType::F32, &cpu()).unwrap_err();
        assert!(format!("{err}").contains("exceeds base out_dim"));
    }
}
