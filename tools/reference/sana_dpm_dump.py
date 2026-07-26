"""DPM++ 2M flow scheduler trajectory dump (ROADMAP_4.6.0 Phase 1).

Isolates the scheduler math from the model: feeds a FIXED sequence of seeded "velocities" through
diffusers' DPMSolverMultistepScheduler (Sana's config) and records each step's output latent, so
the candle `SanaSched::Dpm` step (flow x0-conversion + 1st/2nd-order update) can be verified
tensor-for-tensor without running the DiT.

Run:  python3 tools/reference/sana_dpm_dump.py --out tools/reference/out
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from diffusers import DPMSolverMultistepScheduler

STEPS = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    args = ap.parse_args()

    sch = DPMSolverMultistepScheduler(
        num_train_timesteps=1000, algorithm_type="dpmsolver++", solver_order=2,
        solver_type="midpoint", use_flow_sigmas=True, prediction_type="flow_prediction",
        flow_shift=3.0, final_sigmas_type="zero", lower_order_final=True, timestep_spacing="linspace",
    )
    sch.set_timesteps(STEPS, device="cpu")

    g = torch.Generator().manual_seed(0)
    latent = torch.randn(1, 32, 8, 8, generator=g)
    tensors = {"init": latent.clone()}
    for i, t in enumerate(sch.timesteps):
        v = torch.randn(1, 32, 8, 8, generator=g)
        tensors[f"v{i}"] = v.clone()
        latent = sch.step(v, t, latent, return_dict=False)[0]
        tensors[f"out{i}"] = latent.clone()

    out_dir = Path(args.out) / "sana-dpm"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().to(torch.float32) for k, v in tensors.items()}, str(out_dir / "goldens.safetensors"))
    (out_dir / "manifest.json").write_text(
        json.dumps({"steps": STEPS, "note": "DPM++2M flow trajectory; out{i} = step(v{i}, t_i, prev)."}, indent=2) + "\n"
    )
    print(f"[dpm] wrote {len(tensors)} tensors → {out_dir} (final std={latent.std():.4f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
