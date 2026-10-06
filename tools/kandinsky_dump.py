#!/usr/bin/env python3
"""Reference tensor dumps for the Kandinsky 5 port (RFC KANDINSKY-1, section 12.1).

Dev-only; never shipped. Needs `diffusers` (a build that has Kandinsky5T2IPipeline), `transformers`,
`torch` and `safetensors`. Writes into --out:

  p1.safetensors   stages 1-2 and the VAE check:
      qwen_ids_pos / qwen_ids_neg        token ids of the templated prompt (int64)
      qwen_hidden_pos / qwen_hidden_neg  last hidden states, sliced from the template's crop start
      clip_pooled_pos / clip_pooled_neg  CLIP-L pooler_output
      vae_latent                         a seeded N(0,1) latent, (1, 16, H/8, W/8)
      vae_decoded                        vae.decode(vae_latent / scaling_factor), no shift_factor
  meta.json        the prompt, negative, template name, crop start, max_seq and size used

Templates. `upstream` is what the model was trained with (kandinskylab/kandinsky-5: the typo "promt",
crop start 41) and is plakat's default. `diffusers` is what the diffusers pipeline ships today (typo
corrected, crop start 40). The parity test reads meta.json and uses the same one.

  python tools/kandinsky_dump.py --out /tmp/k5_parity --template upstream
  PLAKAT_PARITY_DIR=/tmp/k5_parity cargo test --release --features metal --lib kandinsky_parity -- --ignored --nocapture
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

TEMPLATES = {
    "upstream": ("<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>", 41),
    "diffusers": ("<|im_start|>system\nYou are a prompt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>", 40),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="A red fox sitting in fresh snow at dawn, soft golden light, detailed fur, shallow depth of field")
    ap.add_argument("--negative", default="")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="upstream")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from diffusers import Kandinsky5T2IPipeline

    dtype = getattr(torch, args.dtype)
    pipe = Kandinsky5T2IPipeline.from_pretrained(args.repo, torch_dtype=dtype)
    template, crop = TEMPLATES[args.template]
    pipe.prompt_template = template
    pipe.prompt_template_encode_start_idx = crop
    os.makedirs(args.out, exist_ok=True)
    out = {}

    pipe.text_encoder.to(args.device)
    pipe.text_encoder_2.to(args.device)
    with torch.no_grad():
        for tag, text in (("pos", args.prompt), ("neg", args.negative)):
            ids = pipe.tokenizer(text=[template.format(text)], images=None, videos=None, max_length=crop + args.max_seq, truncation=True, return_tensors="pt", padding=True)["input_ids"]
            hidden, _ = pipe._encode_prompt_qwen(prompt=[text], device=args.device, max_sequence_length=args.max_seq, dtype=dtype)
            pooled = pipe._encode_prompt_clip(prompt=[text], device=args.device, dtype=dtype)
            out[f"qwen_ids_{tag}"] = ids[0].to(torch.int64).cpu()
            out[f"qwen_hidden_{tag}"] = hidden.float().cpu().contiguous()
            out[f"clip_pooled_{tag}"] = pooled.float().cpu().contiguous()
            print(f"{tag}: {ids.shape[1]} tokens -> hidden {tuple(hidden.shape)}, pooled {tuple(pooled.shape)}")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")

    vae = pipe.vae.to(args.device, torch.float32)
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn((1, 16, args.height // 8, args.width // 8), generator=g, dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(latent.to(args.device) / vae.config.scaling_factor).sample
    out["vae_latent"] = latent.contiguous()
    out["vae_decoded"] = decoded.float().cpu().contiguous()
    print(f"vae: latent {tuple(latent.shape)} -> decoded {tuple(decoded.shape)} (scaling_factor {vae.config.scaling_factor}, shift_factor {vae.config.shift_factor} NOT applied)")

    save_file(out, os.path.join(args.out, "p1.safetensors"))
    meta = {"repo": args.repo, "prompt": args.prompt, "negative": args.negative, "template": args.template, "crop_start": crop, "max_seq": args.max_seq, "width": args.width, "height": args.height, "seed": args.seed, "dtype": args.dtype}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.join(args.out, "p1.safetensors"), "and meta.json")


if __name__ == "__main__":
    main()
