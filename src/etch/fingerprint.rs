//! L3 — **fingerprint** (RFC ETCH-1 §L3): give up on bit recovery and match on what survives a heavy
//! edit — the *semantics*. At generation time a CLIP image embedding is stored `embedding → EtchId` in a
//! local, append-only store; verification is a nearest-cosine query, not an extraction. This is the
//! layer that covers img2img (C2PA-style *soft binding*: the perceptual hash is the lookup key, not the
//! proof). The store is a plain local directory — never a network service; `--etch-db none` disables it.

use super::EtchId;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"ETCHDB01";
/// CLIP ViT-L/14 image-embedding dimension.
pub const DIM: usize = 768;

/// Thresholds (RFC §L3, placeholders pending Phase-5 calibration).
pub const STRONG: f32 = 0.92; // ≥ → strong match
pub const PROBABLE: f32 = 0.85; // [PROBABLE, STRONG) → probable derivative

/// A nearest-neighbour result.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub id: EtchId,
    pub cosine: f32,
}

/// A local, append-only fingerprint store at `<dir>/index.etchdb`.
pub struct Store {
    file: PathBuf,
}

impl Store {
    /// Open (creating the directory + header if needed) the store rooted at `dir`.
    pub fn open(dir: &Path) -> anyhow::Result<Store> {
        std::fs::create_dir_all(dir)?;
        let file = dir.join("index.etchdb");
        if !file.exists() {
            let mut f = std::fs::File::create(&file)?;
            f.write_all(MAGIC)?;
            f.write_all(&(DIM as u32).to_le_bytes())?;
        }
        Ok(Store { file })
    }

    /// Append one `embedding → id` record. `embedding` must be `DIM` L2-normalized floats.
    pub fn add(&self, id: EtchId, embedding: &[f32]) -> anyhow::Result<()> {
        if embedding.len() != DIM {
            anyhow::bail!("embedding dim {} != {DIM}", embedding.len());
        }
        let mut f = std::fs::OpenOptions::new().append(true).open(&self.file)?;
        f.write_all(&id.0.to_be_bytes())?;
        let mut bytes = Vec::with_capacity(DIM * 4);
        for &v in embedding {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes)?;
        Ok(())
    }

    /// Nearest record to `embedding` by cosine (a dot product on unit vectors). `None` if the store is
    /// empty. Linear scan — an ANN index is a later refinement (RFC).
    pub fn query(&self, embedding: &[f32]) -> anyhow::Result<Option<Match>> {
        if embedding.len() != DIM {
            anyhow::bail!("query dim {} != {DIM}", embedding.len());
        }
        let mut f = std::fs::File::open(&self.file)?;
        let mut header = [0u8; 12];
        if f.read_exact(&mut header).is_err() || &header[..8] != MAGIC {
            return Ok(None);
        }
        let rec = 8 + DIM * 4;
        let mut buf = vec![0u8; rec];
        let mut best: Option<Match> = None;
        while f.read_exact(&mut buf).is_ok() {
            let id = u64::from_be_bytes(buf[..8].try_into().unwrap());
            let mut dot = 0f32;
            for i in 0..DIM {
                let off = 8 + i * 4;
                let v = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                dot += v * embedding[i];
            }
            if best.map(|m| dot > m.cosine).unwrap_or(true) {
                best = Some(Match { id: EtchId(id), cosine: dot });
            }
        }
        Ok(best)
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        let meta = std::fs::metadata(&self.file).map(|m| m.len() as usize).unwrap_or(0);
        meta.saturating_sub(12) / (8 + DIM * 4)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Classify a match's cosine into an L3 strength (RFC thresholds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L3Strength {
    Strong,
    Probable,
    None,
}
pub fn classify(cosine: f32) -> L3Strength {
    if cosine >= STRONG {
        L3Strength::Strong
    } else if cosine >= PROBABLE {
        L3Strength::Probable
    } else {
        L3Strength::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(seed: u64) -> Vec<f32> {
        // deterministic pseudo-embedding, L2-normalized.
        let mut v: Vec<f32> = (0..DIM).map(|i| (((seed.wrapping_mul(2654435761).wrapping_add(i as u64)) % 1000) as f32 / 500.0) - 1.0).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        v
    }

    #[test]
    fn store_add_and_query_nearest() {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(dir.path()).unwrap();
        assert!(s.is_empty());
        s.add(EtchId(0xaaaa), &unit(1)).unwrap();
        s.add(EtchId(0xbbbb), &unit(2)).unwrap();
        s.add(EtchId(0xcccc), &unit(3)).unwrap();
        assert_eq!(s.len(), 3);
        // querying with a stored embedding returns its exact id at cosine ≈ 1.
        let m = s.query(&unit(2)).unwrap().unwrap();
        assert_eq!(m.id, EtchId(0xbbbb));
        assert!(m.cosine > 0.999, "self-match cosine {}", m.cosine);
        assert_eq!(classify(m.cosine), L3Strength::Strong);
    }

    #[test]
    fn query_survives_reopen_and_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        {
            let s = Store::open(dir.path()).unwrap();
            s.add(EtchId(0x1234), &unit(7)).unwrap();
        }
        // reopen (append-only persistence).
        let s = Store::open(dir.path()).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.query(&unit(7)).unwrap().unwrap().id, EtchId(0x1234));
        // a fresh empty store returns None. (Bind the tempdir so it isn't dropped before the query.)
        let d2 = tempfile::tempdir().unwrap();
        let empty = Store::open(d2.path()).unwrap();
        assert!(empty.query(&unit(1)).unwrap().is_none());
    }
}
