"""Sana DC-AE (AutoencoderDC) reference dump for plakat's Rust port (ROADMAP_4.5.0 Phase 1).

Loads `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers` (the Sana deep-compression autoencoder),
runs deterministic encode + decode on CPU in F32 (the canonical reference tier), and writes
`goldens.safetensors` + `manifest.json` so the candle `dc_ae.rs` can be verified tensor-for-tensor.

Fixtures (all shapes for a 256x256 image → 8x8x32 latent, 32x compression):
  * `image_in`      (1,3,256,256)  — a fixed deterministic RGB pattern in [-1,1]
  * `latent_enc`    (1,32,8,8)     — encoder(image_in), UN-scaled (raw encoder output)
  * `recon_dec`     (1,3,256,256)  — decoder(latent_enc)
  * `latent_fixed`  (1,32,8,8)     — a fixed pseudo-random latent (seed 0), for decode-in-isolation
  * `decode_fixed`  (1,3,256,256)  — decoder(latent_fixed)

Run:  python3 tools/reference/sana_dcae_dump.py --out tools/reference/out
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers"


def fixed_image(h=256, w=256) -> torch.Tensor:
    """A deterministic RGB test image in [-1,1] — smooth gradients + a couple of hard edges
    so both low-freq (ResBlock) and high-freq (EfficientViT) paths are exercised."""
    ys = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    r = xs
    g = ys
    b = (xs + ys) / 2
    img = torch.stack([r, g, b], dim=0).unsqueeze(0)  # (1,3,h,w) in [0,1]
    # a bright square + a dark bar (hard edges)
    img[:, :, h // 4 : h // 2, w // 4 : w // 2] = 1.0
    img[:, :, 3 * h // 4 :, :] = 0.0
    return img * 2.0 - 1.0  # → [-1,1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    args = ap.parse_args()

    from diffusers import AutoencoderDC

    torch.manual_seed(0)
    dev = "cpu"
    dtype = torch.float32

    print(f"[dcae] loading {REPO} (F32/CPU)…", file=sys.stderr)
    vae = AutoencoderDC.from_pretrained(REPO, torch_dtype=dtype).to(dev).eval()
    cfg = dict(vae.config)
    print(
        f"[dcae] scaling_factor={cfg.get('scaling_factor')} "
        f"latent_channels={cfg.get('latent_channels')} "
        f"compression={vae.spatial_compression_ratio}x",
        file=sys.stderr,
    )

    image_in = fixed_image().to(dev, dtype)
    latent_fixed = torch.randn(1, cfg["latent_channels"], 8, 8, generator=torch.Generator().manual_seed(0)).to(dev, dtype)

    with torch.no_grad():
        latent_enc = vae.encode(image_in).latent  # raw encoder output (unscaled)
        recon_dec = vae.decode(latent_enc).sample
        decode_fixed = vae.decode(latent_fixed).sample

    tensors = {
        "image_in": image_in.contiguous(),
        "latent_enc": latent_enc.contiguous(),
        "recon_dec": recon_dec.contiguous(),
        "latent_fixed": latent_fixed.contiguous(),
        "decode_fixed": decode_fixed.contiguous(),
    }
    out_dir = Path(args.out) / "sana-dcae"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.to(torch.float32) for k, v in tensors.items()}, str(out_dir / "goldens.safetensors"))
    manifest = {
        "repo": REPO,
        "scaling_factor": cfg.get("scaling_factor"),
        "latent_channels": cfg.get("latent_channels"),
        "spatial_compression": vae.spatial_compression_ratio,
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "note": "F32/CPU canonical. latent_enc is UN-scaled (raw encoder). decode expects un-scaled z.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[dcae] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    for k, v in tensors.items():
        print(f"       {k:14} {list(v.shape)}  mean={v.mean():+.4f} std={v.std():.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
