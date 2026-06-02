"""
v0.41 phase 2h reference harness for Stage A (Paella VQ-GAN) DECODE.

Runs diffusers PaellaVQModel.decode on a fixed 4-channel latent and
dumps input + intermediates + final image. The Rust side feeds the
same latent through StageAVae.decode and diffs.
"""
import os, tempfile, shutil
import torch
from safetensors.torch import save_file
from diffusers.pipelines.deprecated.wuerstchen import PaellaVQModel

torch.manual_seed(0)
W = os.path.expanduser("~/weights/stable-cascade/vqgan/diffusion_pytorch_model.safetensors")
CONFIG = "/tmp/cascade_vqgan_config.json"
OUT = "/tmp/cascade_ref_a.safetensors"

td = tempfile.mkdtemp(prefix="cascade-vqgan-")
shutil.copy(CONFIG, td + "/config.json")
os.symlink(W, td + "/diffusion_pytorch_model.safetensors")

vq = PaellaVQModel.from_pretrained(td, torch_dtype=torch.float32).eval()

# Decode input: 4-channel latent. Use a modest 16x16 so decode is fast
# (-> 64x64 image after 4x). scale_factor is applied OUTSIDE (in the
# pipeline) per upstream; here we feed the post-scale latent directly
# to .decode and also save the pre-scale version for the Rust side
# which applies scale_factor itself.
latent = torch.randn(1, 4, 16, 16)

dumps = {}
def hook(name):
    def fn(_m, _i, o):
        t = o.sample if hasattr(o, "sample") else (o[0] if isinstance(o, tuple) else o)
        if isinstance(t, torch.Tensor):
            dumps[name] = t.detach().float().contiguous()
    return fn

# Hook up_blocks (pre-out_block) and out_block to localize.
vq.up_blocks.register_forward_hook(hook("up_blocks_out"))
vq.out_block.register_forward_hook(hook("out_block_out"))

with torch.no_grad():
    # Upstream Cascade pipeline: vqgan.decode(scale_factor * latents).
    img = vq.decode(0.3764 * latent).sample

tensors = {
    "in_latent": latent,                 # pre-scale; Rust applies 0.3764
    "out_image": img.detach().float().contiguous(),
}
for k, v in dumps.items():
    tensors[k] = v.clone()
save_file(tensors, OUT)
print("Saved", OUT)
for k, v in tensors.items():
    print(f"  {k:14} {tuple(v.shape)}  mean={v.mean():.4f} min={v.min():.4f} max={v.max():.4f}")
