use anyhow::{Result, anyhow};
use image::{ImageBuffer, Rgb};
use std::path::Path;

pub fn save_rgb_u8(buf: &[u8], width: u32, height: u32, path: &Path) -> Result<()> {
    let expected = (width as usize) * (height as usize) * 3;
    if buf.len() != expected {
        return Err(anyhow!(
            "buffer size mismatch: got {}, expected {} for {}x{}",
            buf.len(),
            expected,
            width,
            height
        ));
    }
    let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(width, height, buf.to_vec())
        .ok_or_else(|| anyhow!("failed to construct ImageBuffer"))?;
    img.save(path)?;
    Ok(())
}
