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

`--stage p2` is the DiT's (section 12.1, stages 3-8). It reads the embeddings back from p1.safetensors in
--out, so the text tower is not loaded again, and writes:

  p2.safetensors   latents and velocities are NCHW (1, 16, H/8, W/8), blocks are (1, N, 2560):
      time_embed_{1000,500,1}            time_embeddings alone (before the pooled text is added)
      noise                              the seeded N(0,1) start latent
      text_stream_500                    the text stream after its blocks, at t=500 on `noise`, positive prompt
      visual_block_{i}_500               the visual stream after block i (first, middle, last)
      velocity_500                       that forward's output
      x_step{i} / t_step{i}              the latent and the timestep going into tapped step i
      vcond_step{i} / vuncond_step{i}    the two velocities at that step
      sigmas                             the schedule (steps + 1 values)
      final_latent / decoded             the end of the loop and vae.decode(final_latent / scaling_factor)
  meta_p2.json     steps, guidance, tapped steps and blocks, size, seed, dtype

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


def nchw(x):
    """The DiT's channels-last (B, 1, H, W, C) as plakat's (B, C, H, W)."""
    return x[:, 0].permute(0, 3, 1, 2).float().cpu().contiguous()


def dump_p2(args):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, Kandinsky5Transformer3DModel
    from safetensors.torch import load_file

    dtype = getattr(torch, args.dtype)
    dev = args.device
    p1 = load_file(os.path.join(args.out, "p1.safetensors"))
    text = {tag: (p1[f"qwen_hidden_{tag}"].to(dev, dtype), p1[f"clip_pooled_{tag}"].to(dev, dtype)) for tag in ("pos", "neg")}
    tr = Kandinsky5Transformer3DModel.from_pretrained(args.repo, subfolder="transformer", torch_dtype=dtype).to(dev).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(args.repo, subfolder="scheduler")
    h, w = args.height // 8, args.width // 8
    rope_pos = [torch.arange(1, device=dev), torch.arange(h // 2, device=dev), torch.arange(w // 2, device=dev)]
    out = {}

    def forward(x, tag, t):
        hidden, pooled = text[tag]
        return tr(hidden_states=x.to(dtype), encoder_hidden_states=hidden, pooled_projections=pooled, timestep=t.to(dtype), visual_rope_pos=rope_pos, text_rope_pos=torch.arange(hidden.shape[1], device=dev), scale_factor=[1.0, 1.0, 1.0], sparse_params=None, return_dict=True).sample

    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn((1, 1, h, w, 16), generator=g, dtype=torch.float32).to(dev)
    out["noise"] = nchw(noise)
    n_blocks = len(tr.visual_transformer_blocks)
    tap_blocks = sorted({0, n_blocks // 2 - 1, n_blocks - 1})
    tap_steps = sorted({0, args.steps // 2, args.steps - 1})
    with torch.no_grad():
        for t in (1000.0, 500.0, 1.0):
            out[f"time_embed_{int(t)}"] = tr.time_embeddings(torch.tensor([t], device=dev)).float().cpu().contiguous()

        hooks = [tr.text_transformer_blocks[-1].register_forward_hook(lambda m, a, o: out.__setitem__("text_stream_500", o.float().cpu().contiguous()))]
        for i in tap_blocks:
            hooks.append(tr.visual_transformer_blocks[i].register_forward_hook(lambda m, a, o, i=i: out.__setitem__(f"visual_block_{i}_500", o.float().cpu().contiguous())))
        out["velocity_500"] = nchw(forward(noise, "pos", torch.tensor([500.0], device=dev)))
        for hk in hooks:
            hk.remove()
        print("taps at t=500:", {k: tuple(v.shape) for k, v in out.items() if k.endswith("_500")})

        sched.set_timesteps(args.steps, device=dev)
        out["sigmas"] = sched.sigmas.float().cpu().contiguous()
        x = noise
        for i, t in enumerate(sched.timesteps):
            ts = t.unsqueeze(0)
            v_cond = forward(x, "pos", ts)
            v_uncond = forward(x, "neg", ts)
            if i in tap_steps:
                out[f"x_step{i}"] = nchw(x)
                out[f"t_step{i}"] = ts.float().cpu()
                out[f"vcond_step{i}"] = nchw(v_cond)
                out[f"vuncond_step{i}"] = nchw(v_uncond)
            v = v_uncond + args.guidance * (v_cond - v_uncond)
            x = sched.step(v, t, x, return_dict=False)[0]
            print(f"step {i + 1}/{args.steps} t={float(t):.2f}", flush=True)
        out["final_latent"] = nchw(x)
    del tr
    vae = AutoencoderKL.from_pretrained(args.repo, subfolder="vae", torch_dtype=torch.float32).to(dev)
    with torch.no_grad():
        out["decoded"] = vae.decode(out["final_latent"].to(dev) / vae.config.scaling_factor).sample.float().cpu().contiguous()

    save_file(out, os.path.join(args.out, "p2.safetensors"))
    meta = {"repo": args.repo, "steps": args.steps, "guidance": args.guidance, "tap_steps": tap_steps, "tap_blocks": tap_blocks, "width": args.width, "height": args.height, "seed": args.seed, "dtype": args.dtype, "device": dev}
    with open(os.path.join(args.out, "meta_p2.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.join(args.out, "p2.safetensors"), "and meta_p2.json")


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
    ap.add_argument("--stage", choices=["p1", "p2"], default="p1")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.5)
    args = ap.parse_args()
    if args.stage == "p2":
        return dump_p2(args)

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
