"""
v0.41 phase 3 reference harness for the Stable Cascade ControlNet
(canny). The backbone is torchvision EfficientNetV2-S with a 1-channel
stem; projections are Conv(1x1,no-bias) -> LeakyReLU(0.2) -> Conv(1x1,
no-bias). Dumps backbone feature + the 8 projection residuals on a
fixed input for the Rust side to diff against.
"""
import struct, json
import torch
import torch.nn as nn
import torchvision
from safetensors.torch import load_file, save_file

torch.manual_seed(0)
W = "/Users/gandalf/weights/stable-cascade/controlnet/canny.safetensors"
OUT = "/tmp/cascade_ref_cn.safetensors"

sd = load_file(W)

# Backbone: efficientnet_v2_s.features, 1-channel stem.
feats = torchvision.models.efficientnet_v2_s(weights=None).features
feats[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
bb_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
feats.load_state_dict(bb_sd)
feats.eval()

# Projections: 8 heads, each Conv(1280->1280,no-bias) LeakyReLU Conv(1280->2048,no-bias)
projections = nn.ModuleList()
for i in range(8):
    proj = nn.Sequential(
        nn.Conv2d(1280, 1280, 1, bias=False),
        nn.LeakyReLU(0.2),
        nn.Conv2d(1280, 2048, 1, bias=False),
    )
    proj[0].weight.data = sd[f"projections.{i}.0.weight"]
    proj[2].weight.data = sd[f"projections.{i}.2.weight"]
    projections.append(proj.eval())

# Fixed input: 1-channel canny-like, 224x224 (CannyFilter resize=224).
x = torch.randn(1, 1, 224, 224)

with torch.no_grad():
    feat = feats(x)
    residuals = [projections[i](feat) for i in range(8)]

tensors = {"in_cond": x, "backbone_feat": feat.contiguous()}
for i, r in enumerate(residuals):
    tensors[f"residual_{i}"] = r.contiguous()
save_file(tensors, OUT)
print("Saved", OUT)
print(f"  backbone_feat {tuple(feat.shape)}  mean={feat.mean():.4f} min={feat.min():.4f} max={feat.max():.4f}")
for i in range(8):
    r = residuals[i]
    print(f"  residual_{i} {tuple(r.shape)}  mean={r.mean():.4f} min={r.min():.4f} max={r.max():.4f}")
