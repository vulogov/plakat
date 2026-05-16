use anyhow::{Result, anyhow};
use candle_core::Device;

pub fn select(spec: &str) -> Result<Device> {
    let s = spec.trim().to_lowercase();
    match s.as_str() {
        "auto" => Ok(auto()),
        "cpu" => Ok(Device::Cpu),
        x if x.starts_with("cuda") => {
            let idx = x
                .strip_prefix("cuda")
                .and_then(|t| t.strip_prefix(':'))
                .unwrap_or("0")
                .parse::<usize>()
                .unwrap_or(0);
            cuda(idx)
        }
        "metal" => metal(),
        other => Err(anyhow!("unknown device spec: {other}")),
    }
}

#[cfg(feature = "cuda")]
fn cuda(idx: usize) -> Result<Device> {
    Device::new_cuda(idx).map_err(|e| anyhow!("CUDA init failed (device {idx}): {e}"))
}
#[cfg(not(feature = "cuda"))]
fn cuda(_idx: usize) -> Result<Device> {
    Err(anyhow!(
        "binary was not built with --features cuda; rebuild with `cargo build --release --features cuda`"
    ))
}

#[cfg(feature = "metal")]
fn metal() -> Result<Device> {
    Device::new_metal(0).map_err(|e| anyhow!("Metal init failed: {e}"))
}
#[cfg(not(feature = "metal"))]
fn metal() -> Result<Device> {
    Err(anyhow!(
        "binary was not built with --features metal; rebuild with `cargo build --release --features metal`"
    ))
}

fn auto() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        tracing::info!(target: "plakat", "Using CUDA device 0");
        return d;
    }
    #[cfg(feature = "metal")]
    if let Ok(d) = Device::new_metal(0) {
        tracing::info!(target: "plakat", "Using Metal device");
        return d;
    }
    tracing::warn!(target: "plakat", "No GPU backend available — falling back to CPU. Inference will be slow.");
    Device::Cpu
}
