# Gradient checkpointing in plakat — investigated, dead end on candle 0.10.2

**Status: not feasible without re-implementing autograd. Parked.** This note exists so
the idea stops being re-attempted every cycle.

## Why we wanted it

The transformer trainers (PixArt-Σ / SD 3.5 MMDiT / Stable Cascade Stage-C LoRA) OOM at
the **first backward** on a 24 GB unified-memory box: the autograd graph retains every
intermediate activation of the denoiser so `loss.backward()` can use them, and a full DiT /
MMDiT forward's activations don't fit alongside the weights + optimizer state. Gradient
(activation) checkpointing is the standard fix: **don't keep** a block's activations; in the
backward pass, **recompute** the block's forward on the fly to get them. It trades compute
for a large drop in peak memory.

## Why candle 0.10.2 can't express it

candle's autograd is a single, monolithic reverse graph walk:

- `Tensor::backward()` (`candle-core/src/backprop.rs`) topologically sorts the recorded op
  graph from the loss and walks it once, accumulating gradients into a `GradStore`. A
  parameter (`Var`) receives a gradient **only if it appears in that recorded graph**
  (`node.is_variable()` → `track_grad`).
- There is **no checkpoint / recompute / rematerialize API** anywhere in candle, and no hook
  to inject "re-run this subgraph's forward here" during the backward walk.

The two things you'd reach for don't work:

1. **`detach()` the block input + recompute in backward** — manually. There is no callback
   into `backward()` where you could run the recompute, so this can't be wired in.
2. **Wrap the block as a `CustomOp`** whose `fwd` runs it without building a graph and whose
   `bwd` recomputes. `CustomOp::bwd(arg, res, grad_res) -> Option<Tensor>` returns the
   gradient **w.r.t. the input arg only** — it has no channel to accumulate gradients for the
   block's **parameters** (the `Var`s you're training), and those params aren't in the outer
   graph because the forward was detached. So the checkpointed block's weights would get
   **no gradient** → it can't train. This is the load-bearing blocker.

Implementing real checkpointing would mean re-implementing a parameter-aware backward for the
checkpointed region (re-run forward with the params tracked, run a local backward, splice the
param grads into the outer `GradStore`) — i.e. a custom autograd layer. That's a large,
fragile undertaking and out of scope for a feature trainer.

## What actually works on 24 GB today (use these instead)

- **Lower training resolution** — 256² fits where 512² OOMs (the existing trainer default).
  This is the practical lever and already in the `*_train.sh` drivers.
- **LoRA, not full fine-tune** — already the case; only the adapter + its optimizer state are
  trainable, which is what makes the smaller trainers fit at all.
- **Batch size 1** — already the default.
- **CPU offload** — correct but far too slow to be useful here.
- **A CUDA box** — the transformer trainers are verified *correct* (loss decreases on real
  weights); they're only **memory-bound** on Metal/24 GB. They run as-is with more VRAM.
- **Wait for candle** — if candle gains native activation checkpointing (or a custom-op
  backward that exposes parameter grads), revisit. Until then this is parked.

## Bottom line

Don't re-attempt gradient checkpointing on the current candle. The transformer trainers stay
**verified-correct but on-box-unverified at >256²** until either candle adds the primitive or
a higher-VRAM device is available.
