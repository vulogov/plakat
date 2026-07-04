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

/// Look up a fixture by id.
pub fn get(id: &str) -> Option<&'static Fixture> {
    match id {
        "portrait_v1" => Some(&PORTRAIT_V1),
        _ => None,
    }
}
