#!/usr/bin/env python3
"""Grayscale + auto-contrast the bookart training corpus (ROADMAP_BOOKART_1 G0.3).

Book-ornament LoRAs must learn the *black-and-white* idiom, so colour plates (esp.
Bilibin) are desaturated before training; already-B/W scans (Beardsley, Hokusai) are
unchanged apart from exposure normalisation. Writes `<artist>_gray/` next to each raw
`<artist>/` dir. Requires Pillow (`pip install pillow`).

    python3 tools/bookart/prep_grayscale.py
"""
import glob
import os

from PIL import Image, ImageOps

ARTISTS = ["beardsley", "hokusai", "bilibin"]
ROOT = "datasets/bookart_training"


def main():
    for a in ARTISTS:
        src, dst = f"{ROOT}/{a}", f"{ROOT}/{a}_gray"
        os.makedirs(dst, exist_ok=True)
        n = 0
        for f in sorted(glob.glob(src + "/*")):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                im = Image.open(f).convert("L")
                im = ImageOps.autocontrast(im, cutoff=1)  # normalise scan exposure
                im.convert("RGB").save(f"{dst}/{a}_{n:02d}.jpg", quality=92)
                n += 1
            except Exception as e:  # skip unreadable scans
                print("skip", f, e)
        print(f"{a}: {n} grayscale images -> {dst}")


if __name__ == "__main__":
    main()
