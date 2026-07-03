pub mod adetailer;
pub mod artefact_blend;
pub mod controlnet;
pub(crate) mod train_progress;
pub mod matting;
pub mod ic_light;
pub mod instantstyle;
pub mod multiperson;
pub mod controlnet_annotator;
pub mod depth;
pub mod embedding;
pub mod hed;
pub mod lineart;
pub mod openpose;
pub mod openpose_post;
pub mod sd_core;
pub mod sdxl_clip;
pub mod sd_train;
pub mod sdxl_unet;
pub mod vendored_clip;
pub mod vendored_t5;
pub mod tiled;
pub mod extra_schedulers;
pub mod hires_fix;
pub mod img2img;
pub mod animatediff;
pub mod motion_adapter;
pub mod motion_module;
pub mod sd15_motion_unet;
pub mod face_models;
pub mod faceid_lora;
pub mod identity_quality;
pub mod flux;
pub mod flux_controlnet;
pub mod flux_fast;
pub mod flux_inner;
pub mod flux_ip_adapter;
pub mod flux_lora;
pub mod flux_nf4_inner;
pub mod flux_quantized_inner;
pub mod flux_redux;
pub mod ip_adapter;
pub mod lcm_scheduler;
pub mod lora;
pub mod lora_linear;
pub mod nf4_codec;
pub mod nf4_loader;
pub mod portrait;
pub mod real_esrgan;
pub mod cascade;
pub mod cascade_blocks;
pub mod cascade_cn;
pub mod cascade_lora;
pub mod cascade_prior;
pub mod cascade_scheduler;
pub mod cascade_vae;
pub mod pixart;
pub mod pixart_dit;
pub mod pixart_lora;
pub mod sam;
pub mod gen_channel;
pub mod gen_queue;
pub mod scheduler;
pub mod step_hook;
pub mod seeds;
pub mod scrfd;
pub mod inswapper;
pub mod faceswap;
pub mod mmdit_inner;
pub mod sd3;
pub mod sd3_controlnet;
pub mod sd3_lora;
pub mod stylize;
pub mod t2i;
pub mod ti_train;

/// Atomically write a safetensors map: serialize to a sibling temp file, then rename it
/// into place (atomic on the same filesystem). Training is memory-bound and the OOM guard
/// self-aborts mid-run, so a straight `safetensors::save` to the destination can truncate
/// a checkpoint — and in single-file mode (`PLAKAT_TRAIN_SINGLE_FILE=1`) that would be the
/// only artifact. The temp-then-rename keeps the previous file intact until the new one is
/// complete. Use for every training-checkpoint / trained-artifact save.
pub(crate) fn atomic_safetensors_save(
    tensors: &std::collections::HashMap<String, candle_core::Tensor>,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::Context;
    let tmp = out.with_extension("safetensors.tmp");
    candle_core::safetensors::save(tensors, &tmp)
        .with_context(|| format!("writing checkpoint temp {}", tmp.display()))?;
    std::fs::rename(&tmp, out)
        .with_context(|| format!("finalizing checkpoint {}", out.display()))?;
    Ok(())
}

#[cfg(test)]
mod atomic_save_tests {
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    #[test]
    fn atomic_save_writes_the_file_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("plakat-atomic-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("ckpt.safetensors");
        let _ = std::fs::remove_file(&out);
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert("w".into(), Tensor::new(&[1f32, 2., 3.], &Device::Cpu).unwrap());
        super::atomic_safetensors_save(&m, &out).unwrap();
        // The destination exists and the sibling temp was renamed away (not left behind).
        assert!(out.exists(), "checkpoint written");
        assert!(!out.with_extension("safetensors.tmp").exists(), "temp cleaned up by rename");
        // It round-trips.
        let loaded = candle_core::safetensors::load(&out, &Device::Cpu).unwrap();
        assert!(loaded.contains_key("w"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
