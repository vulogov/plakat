"""
v0.41 phase 2f reference harness.

Runs a single Stable Cascade Stage C (prior) forward through diffusers'
StableCascadeUNet on FIXED deterministic inputs, and dumps the inputs +
key intermediate activations + final output to a safetensors file. The
Rust side (cascade_prior reference test) loads the same inputs, runs our
forward, and diffs against these tensors to localize where our forward
diverges from upstream.

Single forward at 24x24 — runs in seconds on CPU.
"""

import json
import os
import tempfile
import shutil

import torch
from safetensors.torch import save_file, load_file
from diffusers import StableCascadeUNet

torch.manual_seed(0)

WEIGHTS = os.path.expanduser(
    "~/weights/stable-cascade/prior/diffusion_pytorch_model.safetensors"
)
CONFIG_URL_LOCAL = "/tmp/cascade_prior_config.json"
OUT = "/tmp/cascade_ref.safetensors"

# Build a from_pretrained-able dir: config.json + the local weights.
tmpdir = tempfile.mkdtemp(prefix="cascade-prior-")
shutil.copy(CONFIG_URL_LOCAL, os.path.join(tmpdir, "config.json"))
os.symlink(WEIGHTS, os.path.join(tmpdir, "diffusion_pytorch_model.safetensors"))

unet = StableCascadeUNet.from_pretrained(tmpdir, torch_dtype=torch.float32)
unet.eval()

# ---- Fixed deterministic inputs ----
B = 1
latents = torch.randn(B, 16, 24, 24)
timestep_ratio = torch.tensor([0.5] * B)
clip_text = torch.randn(B, 77, 1280)
clip_text_pooled = torch.randn(B, 1, 1280)
clip_img = torch.randn(B, 1, 768)

dumps = {}

# Hook the modules we can name cleanly.
def hook(name):
    def fn(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor):
            dumps[name] = t.detach().float().contiguous()
    return fn

unet.embedding.register_forward_hook(hook("emb"))
unet.clf.register_forward_hook(hook("clf"))
# Last block of each down/up level — localizes which level diverges.
for li, lvl in enumerate(unet.down_blocks):
    lvl[-1].register_forward_hook(hook(f"down_lvl{li}"))
for li, lvl in enumerate(unet.up_blocks):
    lvl[-1].register_forward_hook(hook(f"up_lvl{li}"))

with torch.no_grad():
    out = unet(
        sample=latents,
        timestep_ratio=timestep_ratio,
        clip_text_pooled=clip_text_pooled,
        clip_text=clip_text,
        clip_img=clip_img,
        return_dict=False,
    )[0]

# Also dump the conditioning embedding the UNet builds internally, so we
# can compare our build_clip_conditioning against it.
with torch.no_grad():
    clip = unet.get_clip_embeddings(
        clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img
    ) if hasattr(unet, "get_clip_embeddings") else None

tensors = {
    "in_latents": latents,
    "in_timestep_ratio": timestep_ratio,
    "in_clip_text": clip_text,
    "in_clip_text_pooled": clip_text_pooled,
    "in_clip_img": clip_img,
    "out_final": out.detach().float().contiguous(),
}
if clip is not None:
    tensors["clip_cond"] = clip.detach().float().contiguous()
for k, v in dumps.items():
    tensors[k] = v

save_file(tensors, OUT)
print("Saved", OUT)
for k, v in tensors.items():
    print(f"  {k:18} {tuple(v.shape)}  mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")
