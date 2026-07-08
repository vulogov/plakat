//! Canonical verification fixtures (Rust side). **Must mirror** `tools/reference/fixtures.py`
//! — the golden tensors are captured under these exact inputs, so any drift here silently
//! breaks the correspondence. Neither side reads the other (tools/ is excluded from the
//! crate), so they're duplicated by discipline; the fixture id is the contract.

/// A deterministic verification input.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub id: &'static str,
    pub prompt: &'static str,
    pub negative: &'static str,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
}

/// The pilot fixture — mirrors `fixtures.py::FIXTURES["portrait_v1"]`.
pub const PORTRAIT_V1: Fixture = Fixture {
    id: "portrait_v1",
    prompt: "a portrait of a red fox in a sunlit forest, detailed fur",
    negative: "blurry, watermark, text",
    seed: 42,
    width: 512,
    height: 512,
    steps: 4,
};

/// A second, deliberately DIFFERENT-shaped fixture — mirrors `fixtures.py::FIXTURES["still_life_v1"]`.
/// A much shorter prompt (~7 tokens vs portrait's ~15) means far more padding, which stresses
/// the pad attention masking (the class of bug found in v2.1) with a different prompt length.
pub const STILL_LIFE_V1: Fixture = Fixture {
    id: "still_life_v1",
    prompt: "a red apple on a wooden table",
    negative: "blurry",
    seed: 43,
    width: 512,
    height: 512,
    steps: 4,
};

/// A third fixture stressing **tokenization edge cases** — numbers, hyphens, an ampersand,
/// mixed punctuation — a distinctly different token stream from the two natural-language
/// prompts. (A1111 attention-weighting syntax is deliberately avoided: it has no diffusers
/// reference, so it can't be Tier-1-verified.) Mirrors `fixtures.py::FIXTURES["emblem_v1"]`.
pub const EMBLEM_V1: Fixture = Fixture {
    id: "emblem_v1",
    prompt: "a neon-lit 1980s arcade, 8-bit sprites & CRT glow",
    negative: "blurry, low-res",
    seed: 44,
    width: 512,
    height: 512,
    steps: 4,
};

/// Every canonical fixture, in a stable order. Tier 1 iterates these per model (a missing
/// golden for a (model, fixture) pair simply skips).
pub fn all() -> &'static [&'static Fixture] {
    &[&PORTRAIT_V1, &STILL_LIFE_V1, &EMBLEM_V1]
}

/// Look up a fixture by id.
pub fn get(id: &str) -> Option<&'static Fixture> {
    match id {
        "portrait_v1" => Some(&PORTRAIT_V1),
        "still_life_v1" => Some(&STILL_LIFE_V1),
        "emblem_v1" => Some(&EMBLEM_V1),
        _ => None,
    }
}
