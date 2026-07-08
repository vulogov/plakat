"""Canonical, deterministic verification fixtures.

A fixture is the EXACT input a golden was captured under. It must be reproduced identically
on the plakat (Rust) side — same prompt, negative, seed, size, steps — or the golden tensors
won't correspond to what plakat captures. The fixture id is the contract between this harness
and `plakat verify`.

**Keep in sync** with plakat's Rust fixture definitions (a future `src/verify/fixtures.rs`).
Neither side reads the other (tools/ is excluded from the crate), so the values are
duplicated by discipline. Change a fixture ⇒ re-author its goldens ⇒ bump nothing here but
record the new capture in the manifest.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    id: str
    prompt: str
    negative: str
    seed: int
    width: int
    height: int
    steps: int


# The fixtures. Small + fixed so goldens are stable and cheap to store. Prompts are plain
# (no A1111 attention / wildcards) so the input path has no ambiguity.
FIXTURES = {
    "portrait_v1": Fixture(
        id="portrait_v1",
        prompt="a portrait of a red fox in a sunlit forest, detailed fur",
        negative="blurry, watermark, text",
        seed=42,
        width=512,
        height=512,
        steps=4,  # a few steps is enough — Tier 1 taps ENCODER / one UNet forward / VAE, not final quality
    ),
    # A deliberately SHORTER prompt (~7 tokens) → far more padding, which stresses the pad
    # attention masking (the v2.1 bug class) at a different prompt length. Mirrors
    # src/verify/fixtures.rs::STILL_LIFE_V1 exactly (the fixture id is the contract).
    "still_life_v1": Fixture(
        id="still_life_v1",
        prompt="a red apple on a wooden table",
        negative="blurry",
        seed=43,
        width=512,
        height=512,
        steps=4,
    ),
    # Tokenization edge cases — numbers, hyphens, an ampersand, mixed punctuation — a
    # distinctly different token stream. No A1111 weighting syntax (no diffusers reference).
    # Mirrors src/verify/fixtures.rs::EMBLEM_V1 exactly.
    "emblem_v1": Fixture(
        id="emblem_v1",
        prompt="a neon-lit 1980s arcade, 8-bit sprites & CRT glow",
        negative="blurry, low-res",
        seed=44,
        width=512,
        height=512,
        steps=4,
    ),
}


def get(fixture_id: str) -> Fixture:
    if fixture_id not in FIXTURES:
        raise SystemExit(
            f"unknown fixture {fixture_id!r}; known: {', '.join(sorted(FIXTURES))}"
        )
    return FIXTURES[fixture_id]


def deterministic_tensor(shape, seed: int = 1):
    """A DETERMINISTIC tensor of `shape` via a tiny LCG — NOT a seeded RNG.

    Byte-for-byte identical to plakat's `verify::deterministic_tensor` (same LCG, same
    C-order flattening, same `seed` stream selector). Values in [-1, 1). float32.
    Used to feed identical synthetic conditioning (caption / context / pooled) to both
    sides so a transformer-block tap isolates the block math from the text encoders.
    """
    import torch

    n = 1
    for d in shape:
        n *= d
    x = seed
    vals = [0.0] * n
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        vals[i] = (x % 2000) / 1000.0 - 1.0
    return torch.tensor(vals, dtype=torch.float32).reshape(*shape)


def deterministic_latent(channels: int, lh: int, lw: int):
    """A DETERMINISTIC latent (1, channels, lh, lw) — LCG seed 1. See `deterministic_tensor`.
    Byte-identical to plakat's `verify::deterministic_latent`; both sides decode/denoise the
    SAME input. Values in [-1, 1). Returns a torch.Tensor (float32)."""
    return deterministic_tensor((1, channels, lh, lw), seed=1)
