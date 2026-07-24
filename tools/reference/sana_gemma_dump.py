"""Sana Gemma-2-2B text-encoder reference dump (ROADMAP_4.5.0 Phase 2).

Replicates diffusers `SanaPipeline._get_gemma_prompt_embeds` + the `encode_prompt` select-index
re-slice, using the **Sana repo's own** `text_encoder/` + `tokenizer/` (ungated, unlike
google/gemma-2-2b-it), in **F32 on CPU** (canonical). Dumps input_ids / attention_mask / the raw
last_hidden_state (pre-reslice) / the final (1,300,2304) embeds, so the candle `vendored_gemma2`
`forward_hidden` can be verified tensor-for-tensor and the `[0]+last-299` re-slice checked in Rust.

Run:  python3 tools/reference/sana_gemma_dump.py --out tools/reference/out
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
MAX_SEQ = 300
PROMPT = "a lighthouse on a rocky cliff at golden hour, dramatic clouds"
CHI = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/reference/out")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    print(f"[gemma] fetching {REPO} text_encoder/ + tokenizer/…", file=sys.stderr)
    local = snapshot_download(REPO, allow_patterns=["text_encoder/*", "tokenizer/*"])
    tok = AutoTokenizer.from_pretrained(f"{local}/tokenizer")
    tok.padding_side = "right"
    enc = AutoModel.from_pretrained(f"{local}/text_encoder", torch_dtype=torch.float32).to("cpu").eval()
    print(f"[gemma] hidden={enc.config.hidden_size} layers={enc.config.num_hidden_layers}", file=sys.stderr)

    # _get_gemma_prompt_embeds: CHI prepend + right-pad to num_chi + 300 - 2.
    chi_prompt = "\n".join(CHI)
    num_chi = len(tok.encode(chi_prompt))
    max_len = num_chi + MAX_SEQ - 2
    full = chi_prompt + PROMPT
    ti = tok([full], padding="max_length", max_length=max_len, truncation=True, add_special_tokens=True, return_tensors="pt")
    input_ids, mask = ti.input_ids, ti.attention_mask
    print(f"[gemma] num_chi={num_chi} max_len={max_len} seq={input_ids.shape[1]}", file=sys.stderr)

    with torch.no_grad():
        raw_hidden = enc(input_ids, attention_mask=mask)[0]  # (1, max_len, 2304)

    # encode_prompt select-index re-slice: keep BOS + last 299 → (1,300,2304).
    select = [0] + list(range(-MAX_SEQ + 1, 0))
    final_embeds = raw_hidden[:, select]
    final_mask = mask[:, select]

    tensors = {
        "input_ids": input_ids.to(torch.int64),
        "attention_mask": mask.to(torch.float32),
        "raw_hidden": raw_hidden.to(torch.float32).contiguous(),
        "final_embeds": final_embeds.to(torch.float32).contiguous(),
        "final_mask": final_mask.to(torch.float32).contiguous(),
    }
    out_dir = Path(args.out) / "sana-gemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "goldens.safetensors"))
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "prompt": PROMPT,
                "num_chi_tokens": num_chi,
                "max_length_all": max_len,
                "max_sequence_length": MAX_SEQ,
                "hidden": enc.config.hidden_size,
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "note": "F32/CPU canonical. raw_hidden = forward_hidden(input_ids, mask) target; "
                "final_embeds = raw_hidden[:, [0]+last-299].",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[gemma] wrote {len(tensors)} tensor(s) → {out_dir}", file=sys.stderr)
    for k, v in tensors.items():
        info = f"{list(v.shape)}"
        if v.dtype.is_floating_point:
            info += f"  mean={v.float().mean():+.4f} std={v.float().std():.4f}"
        print(f"       {k:14} {info}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
