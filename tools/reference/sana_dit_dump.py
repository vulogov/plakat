"""Sana Linear-DiT (SanaTransformer2DModel) reference dump (ROADMAP_4.5.0 Phase 3).

Loads the Sana 1.6B `transformer/` (F32/CPU canonical), runs a single forward on fixed
seeded inputs (latent + timestep + caption embeds + mask), and dumps inputs + the output
velocity so the candle `sana_dit.rs` can be verified tensor-for-tensor.

Run:  python3 tools/reference/sana_dit_dump.py --out tools/reference/out
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    ap.add_argument("--repo", default=REPO, help="Sana diffusers repo (e.g. SANA1.5 for qk_norm)")
    ap.add_argument("--name", default="sana-dit", help="output subdir name")
    args = ap.parse_args()
    repo = args.repo

    from diffusers import SanaTransformer2DModel

    print(f"[dit] loading {repo} transformer/ (F32/CPU)…", file=sys.stderr)
    model = SanaTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.float32).to("cpu").eval()
    cfg = dict(model.config)
    print(
        f"[dit] layers={cfg.get('num_layers')} heads={cfg.get('num_attention_heads')} "
        f"head_dim={cfg.get('attention_head_dim')} caption_ch={cfg.get('caption_channels')} "
        f"in={cfg.get('in_channels')} patch={cfg.get('patch_size')} guidance={cfg.get('guidance_embeds')}",
        file=sys.stderr,
    )

    g = torch.Generator().manual_seed(0)
    latent = torch.randn(1, cfg["in_channels"], 32, 32, generator=g)  # (1,32,32,32)
    caption = torch.randn(1, 300, cfg["caption_channels"], generator=g)  # (1,300,2304)
    # a realistic mask: first 40 tokens real, rest padding.
    mask = torch.zeros(1, 300)
    mask[:, :40] = 1.0
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        out = model(
            hidden_states=latent,
            encoder_hidden_states=caption,
            timestep=timestep,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]

    tensors = {
        "latent": latent.contiguous(),
        "caption": caption.contiguous(),
        "mask": mask.contiguous(),
        "timestep": timestep.contiguous(),
        "output": out.contiguous(),
    }
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.float32) for k, v in tensors.items()}, str(out_dir / "goldens.safetensors"))
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "config": {k: cfg.get(k) for k in ["num_layers", "num_attention_heads", "attention_head_dim", "num_cross_attention_heads", "cross_attention_head_dim", "cross_attention_dim", "caption_channels", "in_channels", "out_channels", "patch_size", "sample_size", "mlp_ratio", "norm_eps", "guidance_embeds", "qk_norm", "timestep_scale"]},
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "note": "F32/CPU canonical. timestep=500, mask=first-40-real. output = velocity.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[dit] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    for k, v in tensors.items():
        info = f"{list(v.shape)}"
        if v.dtype.is_floating_point:
            info += f"  mean={v.float().mean():+.4f} std={v.float().std():.4f}"
        print(f"      {k:10} {info}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
