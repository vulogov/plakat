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
}


def get(fixture_id: str) -> Fixture:
    if fixture_id not in FIXTURES:
        raise SystemExit(
            f"unknown fixture {fixture_id!r}; known: {', '.join(sorted(FIXTURES))}"
        )
    return FIXTURES[fixture_id]


def deterministic_latent(channels: int, lh: int, lw: int):
    """A DETERMINISTIC latent (1, channels, lh, lw) via a tiny LCG — NOT a seeded RNG.

    Byte-for-byte identical to plakat's `verify::deterministic_latent` (same LCG, same
    C-order flattening), so both sides decode the SAME input for the `vae.decoded` golden.
    Values in [-1, 1). Returns a torch.Tensor (float32).
    """
    import torch

    n = channels * lh * lw
    x = 1
    vals = [0.0] * n
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        vals[i] = (x % 2000) / 1000.0 - 1.0
    return torch.tensor(vals, dtype=torch.float32).reshape(1, channels, lh, lw)
