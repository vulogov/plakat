"""
v0.41 phase 2g reference harness for Stage B (decoder).

Single forward through diffusers' StableCascadeUNet (decoder config) on
fixed inputs incl. effnet conditioning, dumping intermediates for the
Rust side to diff against.
"""
import os, tempfile, shutil
import torch
from safetensors.torch import save_file
from diffusers import StableCascadeUNet

torch.manual_seed(0)

WEIGHTS = os.path.expanduser(
    "~/weights/stable-cascade/decoder/diffusion_pytorch_model.safetensors"
)
CONFIG = "/tmp/cascade_decoder_config.json"
OUT = "/tmp/cascade_ref_b.safetensors"

tmpdir = tempfile.mkdtemp(prefix="cascade-decoder-")
shutil.copy(CONFIG, os.path.join(tmpdir, "config.json"))
os.symlink(WEIGHTS, os.path.join(tmpdir, "diffusion_pytorch_model.safetensors"))

unet = StableCascadeUNet.from_pretrained(tmpdir, torch_dtype=torch.float32)
unet.eval()

B = 1
# Decoder sample is the 4-channel Stage A VQ latent; the decoder
# patchifies (PixelUnshuffle patch_size=2) internally to 16ch.
latents = torch.randn(B, 4, 24, 24)
timestep_ratio = torch.tensor([0.5] * B)
clip_text_pooled = torch.randn(B, 1, 1280)
# effnet at the internal embedding spatial (24/patch_size=12) so the
# interpolate is identity — isolates the effnet_mapper + blocks from
# the resampling.
effnet = torch.randn(B, 16, 12, 12)

dumps = {}
def hook(name):
    def fn(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        if isinstance(t, torch.Tensor):
            dumps[name] = t.detach().float().contiguous()
    return fn

unet.embedding.register_forward_hook(hook("emb"))
unet.clf.register_forward_hook(hook("clf"))
for li, lvl in enumerate(unet.down_blocks):
    lvl[-1].register_forward_hook(hook(f"down_lvl{li}"))
for li, lvl in enumerate(unet.up_blocks):
    lvl[-1].register_forward_hook(hook(f"up_lvl{li}"))

with torch.no_grad():
    out = unet(
        sample=latents,
        timestep_ratio=timestep_ratio,
        clip_text_pooled=clip_text_pooled,
        effnet=effnet,
        return_dict=False,
    )[0]

with torch.no_grad():
    clip = unet.get_clip_embeddings(clip_txt_pooled=clip_text_pooled)

tensors = {
    "in_latents": latents,
    "in_timestep_ratio": timestep_ratio,
    "in_clip_text_pooled": clip_text_pooled,
    "in_effnet": effnet,
    "out_final": out.detach().float().contiguous(),
    "clip_cond": clip.detach().float().contiguous(),
}
for k, v in dumps.items():
    tensors[k] = v

save_file(tensors, OUT)
print("Saved", OUT)
for k, v in tensors.items():
    print(f"  {k:14} {tuple(v.shape)}  mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")
