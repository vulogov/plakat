#!/usr/bin/env python3
"""
Convert Carve/u2net-universal full_weights.pth (legacy pickle, verified-firing,
original stage/rebnconv/conv_s1/bn_s1/side/outconv naming) -> safetensors.

Why: candle's VarBuilder reads safetensors or ZIP-format .pth only. Carve's
full_weights.pth is a LEGACY (non-zip) pickle, so from_pth fails with
"invalid Zip archive: Could not find EOCD". Re-serialising the same state_dict
to safetensors fixes that while keeping every key byte-for-byte identical.

Then it verifies the converted file LOADS into the official xuebinqin U2NET and
FIRES (d0 max ~0.9-1.0) on a real photo.

Deps: pip install torch safetensors huggingface_hub pillow numpy
Run:  python convert_u2net_to_safetensors.py [/path/to/real_photo.jpg]
"""
import sys, os
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download

OUT = os.path.expanduser("~/.cache/plakat/u2net/u2net-universal.safetensors")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# 1. fetch the verified-firing legacy pickle (Apache-2.0)
pth = hf_hub_download(repo_id="Carve/u2net-universal", filename="full_weights.pth")
print("downloaded:", pth, os.path.getsize(pth) // (1024 * 1024), "MB")

# 2. load the state_dict. weights_only=True keeps it to plain tensors.
sd = torch.load(pth, map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd and all(
    not torch.is_tensor(v) for k, v in sd.items() if k != "state_dict"
):
    sd = sd["state_dict"]

# 3. clean: drop a leading "module." (DataParallel) if present; keep tensors only;
#    make contiguous + f32 (candle-friendly).
clean = {}
dropped = []
for k, v in sd.items():
    if not torch.is_tensor(v):
        dropped.append(k)
        continue
    nk = k[len("module."):] if k.startswith("module.") else k
    clean[nk] = v.detach().contiguous().to(torch.float32)
if dropped:
    print("dropped non-tensor entries:", dropped)

# 4. write safetensors (keys unchanged)
save_file(clean, OUT)
print("wrote:", OUT, os.path.getsize(OUT) // (1024 * 1024), "MB,", len(clean), "tensors")

# report naming sample
sample = [k for k in clean.keys()]
print("\nKEY NAMING (first 12 of %d):" % len(sample))
for k in sample[:12]:
    print("  ", k, tuple(clean[k].shape))
print("  ... side/outconv keys:")
for k in sample:
    if k.startswith("side") or k.startswith("outconv"):
        print("  ", k, tuple(clean[k].shape))
has_module = any(k.startswith("module.") for k in clean)
print("module. prefix present:", has_module, "(your candle loader wants False)")

# 5. round-trip + FIRE test against the official U2NET forward
roundtrip = load_file(OUT)
print("\nreloaded safetensors:", len(roundtrip), "tensors")

# pull the official model def
import importlib.util, urllib.request, tempfile
src = urllib.request.urlopen(
    "https://raw.githubusercontent.com/xuebinqin/U-2-Net/master/model/u2net.py"
).read().decode()
mp = os.path.join(tempfile.gettempdir(), "u2net_ref.py")
open(mp, "w").write(src)
spec = importlib.util.spec_from_file_location("u2net_ref", mp)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

net = mod.U2NET(3, 1)
missing, unexpected = net.load_state_dict(roundtrip, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
if missing:
    print("  missing sample:", missing[:5])
if unexpected:
    print("  unexpected sample:", unexpected[:5])
net.eval()

# real photo (arg, or synthesise a high-contrast subject if none given)
if len(sys.argv) > 1:
    from PIL import Image
    img = Image.open(sys.argv[1]).convert("RGB").resize((320, 320))
    arr = np.asarray(img).astype(np.float32) / 255.0
else:
    print("\n[no photo arg] using a synthetic centred-disk subject")
    yy, xx = np.mgrid[0:320, 0:320]
    disk = ((xx - 160) ** 2 + (yy - 160) ** 2) < 90 ** 2
    arr = np.zeros((320, 320, 3), np.float32)
    arr[disk] = [0.9, 0.3, 0.2]
    arr[~disk] = [0.05, 0.05, 0.08]

# official preprocessing: /max then per-channel mean/std (RescaleT+ToTensorLab)
t = arr / max(arr.max(), 1e-6)
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
t = (t - mean) / std
x = torch.from_numpy(t.transpose(2, 0, 1)[None]).float()

with torch.no_grad():
    d0 = net(x)[0]
d0 = d0.squeeze().numpy()
print("\n*** d0 max = %.4f  mean = %.4f ***" % (d0.max(), d0.mean()))
print("FIRES" if d0.max() > 0.5 else "DEAD (does not fire)")
