//! Vendored Stable Diffusion UNet attention + blocks + assembly (copied
//! from candle_transformers 0.10.2) with `LoraLinear` hooks on the
//! `CrossAttention` projections, for `plakat style train` on SDXL / SD 1.5.
//!
//! A SEPARATE **trainable** UNet — plakat's inference path keeps using
//! candle's UNet unchanged, so this adds no inference regression. Only the
//! `CrossAttention` q/k/v/out Linears are swapped for `LoraLinear` + a
//! registry; everything else is candle verbatim (resnets/sampling reused
//! from candle directly).
pub mod attention;
pub mod blocks;
pub mod unet;

pub mod trainer;
