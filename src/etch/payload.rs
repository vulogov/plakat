//! `EtchId` derivation + (later phases) CRC/ECC/tile framing. Phase 1 provides the keyed id derivation.

use super::EtchId;
use sha2::{Digest, Sha256};

/// `EtchId = SHA-256(key ‖ "plakat-etch-v1" ‖ canonical_manifest)[0..8]`, big-endian (RFC §"EtchId
/// derivation"; we use the in-tree `sha2` rather than adding `blake3`, which the RFC explicitly allows).
/// Reproducible: identical recipe + key → identical id. Opaque: reveals nothing without the manifest.
pub fn derive_id(key: &str, canonical_manifest: &str) -> EtchId {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    h.update(b"plakat-etch-v1");
    h.update(canonical_manifest.as_bytes());
    let d = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&d[..8]);
    EtchId(u64::from_be_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_reproducible_and_key_sensitive() {
        let a = derive_id("k", "recipe");
        assert_eq!(a, derive_id("k", "recipe"), "same key+recipe → same id");
        assert_ne!(a, derive_id("k2", "recipe"), "different key → different id");
        assert_ne!(a, derive_id("k", "recipe2"), "different recipe → different id");
        assert_eq!(a.hex().len(), 16);
    }
}
