"""SanaControlNet reference dump (ROADMAP_4.8.0 Phase 2).

Loads the Sana ControlNet (`SanaControlNetModel`, a 7-block copy of the DiT), runs a single
forward on fixed seeded inputs (latent + timestep + caption + mask + controlnet_cond), and dumps
inputs + the N per-block residuals so the candle `sana_controlnet.rs` port can be verified
tensor-for-tensor.

Run:  python3 tools/reference/sana_controlnet_dump.py --out tools/reference/out
      (default repo = ishan24/Sana_600M_1024px_ControlNetPlus_diffusers, subfolder controlnet/)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "ishan24/Sana_600M_1024px_ControlNetPlus_diffusers"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--subfolder", default="controlnet")
    ap.add_argument("--name", default="sana-controlnet", help="output subdir name")
    args = ap.parse_args()

    from diffusers import SanaControlNetModel

    print(f"[cn] loading {args.repo} ({args.subfolder}/) F32/CPU…", file=sys.stderr)
    model = SanaControlNetModel.from_pretrained(
        args.repo, subfolder=args.subfolder, torch_dtype=torch.float32
    ).to("cpu").eval()
    cfg = dict(model.config)
    print(
        f"[cn] layers={cfg.get('num_layers')} heads={cfg.get('num_attention_heads')} "
        f"head_dim={cfg.get('attention_head_dim')} inner={cfg.get('num_attention_heads')*cfg.get('attention_head_dim')} "
        f"caption_ch={cfg.get('caption_channels')} in={cfg.get('in_channels')}",
        file=sys.stderr,
    )

    g = torch.Generator().manual_seed(0)
    latent = torch.randn(1, cfg["in_channels"], 32, 32, generator=g)  # noisy latent
    caption = torch.randn(1, 300, cfg["caption_channels"], generator=g)
    control = torch.randn(1, cfg["in_channels"], 32, 32, generator=g)  # DC-AE control latent
    mask = torch.zeros(1, 300)
    mask[:, :40] = 1.0
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        res = model(
            hidden_states=latent,
            encoder_hidden_states=caption,
            timestep=timestep,
            controlnet_cond=control,
            conditioning_scale=1.0,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]  # tuple of N residuals, each (1, N_tokens, inner_dim)

    tensors = {
        "latent": latent.contiguous(),
        "caption": caption.contiguous(),
        "control": control.contiguous(),
        "mask": mask.contiguous(),
        "timestep": timestep.contiguous(),
    }
    for i, r in enumerate(res):
        tensors[f"res_{i}"] = r.contiguous()
    print(f"[cn] {len(res)} residual(s), each {list(res[0].shape)}", file=sys.stderr)

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.float32) for k, v in tensors.items()}, str(out_dir / "goldens.safetensors"))
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "num_residuals": len(res),
                "config": {k: cfg.get(k) for k in ["num_layers", "num_attention_heads", "attention_head_dim", "num_cross_attention_heads", "cross_attention_head_dim", "cross_attention_dim", "caption_channels", "in_channels", "patch_size", "mlp_ratio", "norm_eps", "qk_norm"]},
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "note": "F32/CPU. timestep=500, mask=first-40-real, conditioning_scale=1.0. res_i = per-block controlnet residual.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[cn] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
