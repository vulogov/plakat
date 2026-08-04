//! v0.21: `plakat.*` host words registered into the bundcore VM.
//!
//! Each word lives in its own file (`echo.rs`, `load.rs`,
//! `generate.rs`, …). [`register_plakat_words`] wires every word
//! into the VM via [`VM::register_inline`]. The host fns are
//! plain `fn` pointers; see `super::ctx` for the state-sharing
//! singleton that gets around the no-closures constraint.

use anyhow::{Result, anyhow};
use rust_multistackvm::multistackvm::VM;

pub mod adetailer;
pub mod animate;
pub mod artefact;
pub mod bookart;
pub mod cascade;
pub mod config;
pub mod controlnet;
pub mod echo;
pub mod embedding;
pub mod enhance;
#[cfg(feature = "fractals")]
pub mod fractals;
pub mod generate;
pub mod genre;
pub mod hires;
pub mod tiled;
pub mod img2img;
pub mod inpaint;
pub mod load;
pub mod look;
pub mod lora;
pub mod map;
pub mod metadata;
pub mod multiperson;
pub mod outpaint;
pub mod pixart;
pub mod portrait;
pub mod portrait_photo;
pub mod compose;
pub mod refiner;
pub mod relight;
pub mod save;
pub mod segment;
pub mod style;
pub mod stylize;
pub mod transparent;
pub mod upscale;

/// Register every `plakat.*` word into `vm`. v0.21 shipped 7
/// MVP words. v0.22 phase 4 adds the `plakat.lora.*` namespace.
pub fn register_plakat_words(vm: &mut VM) -> Result<()> {
    vm.register_inline("plakat.echo".to_string(), echo::plakat_echo)
        .map_err(|e| anyhow!("registering plakat.echo: {e}"))?;
    vm.register_inline("plakat.load".to_string(), load::plakat_load)
        .map_err(|e| anyhow!("registering plakat.load: {e}"))?;
    vm.register_inline("plakat.generate".to_string(), generate::plakat_generate)
        .map_err(|e| anyhow!("registering plakat.generate: {e}"))?;
    vm.register_inline("plakat.map.render".to_string(), map::plakat_map_render)
        .map_err(|e| anyhow!("registering plakat.map.render: {e}"))?;
    vm.register_inline("plakat.map.layout".to_string(), map::plakat_map_layout)
        .map_err(|e| anyhow!("registering plakat.map.layout: {e}"))?;
    vm.register_inline("plakat.map.erosion".to_string(), map::plakat_map_erosion)
        .map_err(|e| anyhow!("registering plakat.map.erosion: {e}"))?;
    vm.register_inline("plakat.map.paint".to_string(), map::plakat_map_paint)
        .map_err(|e| anyhow!("registering plakat.map.paint: {e}"))?;
    vm.register_inline("plakat.map.tiles".to_string(), map::plakat_map_tiles)
        .map_err(|e| anyhow!("registering plakat.map.tiles: {e}"))?;
    #[cfg(feature = "fractals")]
    {
        vm.register_inline("plakat.fractal.size".to_string(), fractals::plakat_fractal_size)
            .map_err(|e| anyhow!("registering plakat.fractal.size: {e}"))?;
        vm.register_inline("plakat.fractal.render".to_string(), fractals::plakat_fractal_render)
            .map_err(|e| anyhow!("registering plakat.fractal.render: {e}"))?;
        vm.register_inline("plakat.fractal.compose".to_string(), fractals::plakat_fractal_compose)
            .map_err(|e| anyhow!("registering plakat.fractal.compose: {e}"))?;
        vm.register_inline("plakat.fractal.paint".to_string(), fractals::plakat_fractal_paint)
            .map_err(|e| anyhow!("registering plakat.fractal.paint: {e}"))?;
        vm.register_inline("plakat.fractal.animate".to_string(), fractals::plakat_fractal_animate)
            .map_err(|e| anyhow!("registering plakat.fractal.animate: {e}"))?;
    }
    // 6.1.0 (A4): plakat.bookart.* namespace — render a book ornament into an image handle.
    vm.register_inline("plakat.bookart.origin".to_string(), bookart::plakat_bookart_origin)
        .map_err(|e| anyhow!("registering plakat.bookart.origin: {e}"))?;
    vm.register_inline("plakat.bookart.technique".to_string(), bookart::plakat_bookart_technique)
        .map_err(|e| anyhow!("registering plakat.bookart.technique: {e}"))?;
    vm.register_inline("plakat.bookart.render".to_string(), bookart::plakat_bookart_render)
        .map_err(|e| anyhow!("registering plakat.bookart.render: {e}"))?;
    vm.register_inline("plakat.bookart.illustrate".to_string(), bookart::plakat_bookart_illustrate)
        .map_err(|e| anyhow!("registering plakat.bookart.illustrate: {e}"))?;
    vm.register_inline("plakat.multiperson".to_string(), multiperson::plakat_multiperson)
        .map_err(|e| anyhow!("registering plakat.multiperson: {e}"))?;
    vm.register_inline("plakat.img2img".to_string(), img2img::plakat_img2img)
        .map_err(|e| anyhow!("registering plakat.img2img: {e}"))?;
    vm.register_inline("plakat.portrait".to_string(), portrait::plakat_portrait)
        .map_err(|e| anyhow!("registering plakat.portrait: {e}"))?;
    vm.register_inline("plakat.save".to_string(), save::plakat_save)
        .map_err(|e| anyhow!("registering plakat.save: {e}"))?;
    vm.register_inline("plakat.upscale".to_string(), upscale::plakat_upscale)
        .map_err(|e| anyhow!("registering plakat.upscale: {e}"))?;
    vm.register_inline(
        "plakat.config.set".to_string(),
        config::plakat_config_set,
    )
    .map_err(|e| anyhow!("registering plakat.config.set: {e}"))?;
    // v0.22 phase 4: plakat.lora.* namespace.
    vm.register_inline("plakat.lora.add".to_string(), lora::plakat_lora_add)
        .map_err(|e| anyhow!("registering plakat.lora.add: {e}"))?;
    vm.register_inline("plakat.lora.clear".to_string(), lora::plakat_lora_clear)
        .map_err(|e| anyhow!("registering plakat.lora.clear: {e}"))?;
    vm.register_inline("plakat.lora.list".to_string(), lora::plakat_lora_list)
        .map_err(|e| anyhow!("registering plakat.lora.list: {e}"))?;
    // v0.22 phase 5: plakat.controlnet.* namespace.
    vm.register_inline(
        "plakat.controlnet.add".to_string(),
        controlnet::plakat_controlnet_add,
    )
    .map_err(|e| anyhow!("registering plakat.controlnet.add: {e}"))?;
    vm.register_inline(
        "plakat.controlnet.annotate".to_string(),
        controlnet::plakat_controlnet_annotate,
    )
    .map_err(|e| anyhow!("registering plakat.controlnet.annotate: {e}"))?;
    vm.register_inline(
        "plakat.controlnet.spec".to_string(),
        controlnet::plakat_controlnet_spec,
    )
    .map_err(|e| anyhow!("registering plakat.controlnet.spec: {e}"))?;
    vm.register_inline(
        "plakat.controlnet.clear".to_string(),
        controlnet::plakat_controlnet_clear,
    )
    .map_err(|e| anyhow!("registering plakat.controlnet.clear: {e}"))?;
    vm.register_inline(
        "plakat.controlnet.list".to_string(),
        controlnet::plakat_controlnet_list,
    )
    .map_err(|e| anyhow!("registering plakat.controlnet.list: {e}"))?;
    // v0.22 phase 6: plakat.refiner.* namespace.
    vm.register_inline(
        "plakat.refiner.enable".to_string(),
        refiner::plakat_refiner_enable,
    )
    .map_err(|e| anyhow!("registering plakat.refiner.enable: {e}"))?;
    vm.register_inline(
        "plakat.refiner.disable".to_string(),
        refiner::plakat_refiner_disable,
    )
    .map_err(|e| anyhow!("registering plakat.refiner.disable: {e}"))?;
    // v0.22 phase 7: plakat.adetailer.* namespace.
    vm.register_inline(
        "plakat.adetailer.enable".to_string(),
        adetailer::plakat_adetailer_enable,
    )
    .map_err(|e| anyhow!("registering plakat.adetailer.enable: {e}"))?;
    vm.register_inline(
        "plakat.adetailer.disable".to_string(),
        adetailer::plakat_adetailer_disable,
    )
    .map_err(|e| anyhow!("registering plakat.adetailer.disable: {e}"))?;
    // v0.22 phase 8: plakat.hires.* namespace.
    vm.register_inline(
        "plakat.hires.enable".to_string(),
        hires::plakat_hires_enable,
    )
    .map_err(|e| anyhow!("registering plakat.hires.enable: {e}"))?;
    vm.register_inline(
        "plakat.hires.disable".to_string(),
        hires::plakat_hires_disable,
    )
    .map_err(|e| anyhow!("registering plakat.hires.disable: {e}"))?;
    // v1.0: plakat.tiled.* namespace (SDXL tiled hi-res scripting).
    vm.register_inline(
        "plakat.tiled.enable".to_string(),
        tiled::plakat_tiled_enable,
    )
    .map_err(|e| anyhow!("registering plakat.tiled.enable: {e}"))?;
    vm.register_inline(
        "plakat.tiled.disable".to_string(),
        tiled::plakat_tiled_disable,
    )
    .map_err(|e| anyhow!("registering plakat.tiled.disable: {e}"))?;
    // v0.22 phase 9: plakat.artefact.* namespace.
    vm.register_inline(
        "plakat.artefact.add".to_string(),
        artefact::plakat_artefact_add,
    )
    .map_err(|e| anyhow!("registering plakat.artefact.add: {e}"))?;
    vm.register_inline(
        "plakat.artefact.clear".to_string(),
        artefact::plakat_artefact_clear,
    )
    .map_err(|e| anyhow!("registering plakat.artefact.clear: {e}"))?;
    vm.register_inline(
        "plakat.artefact.list".to_string(),
        artefact::plakat_artefact_list,
    )
    .map_err(|e| anyhow!("registering plakat.artefact.list: {e}"))?;
    vm.register_inline(
        "plakat.artefact.blend.enable".to_string(),
        artefact::plakat_artefact_blend_enable,
    )
    .map_err(|e| anyhow!("registering plakat.artefact.blend.enable: {e}"))?;
    vm.register_inline(
        "plakat.artefact.blend.disable".to_string(),
        artefact::plakat_artefact_blend_disable,
    )
    .map_err(|e| anyhow!("registering plakat.artefact.blend.disable: {e}"))?;
    // v0.22 phase 10: plakat.enhance host word.
    vm.register_inline("plakat.enhance".to_string(), enhance::plakat_enhance)
        .map_err(|e| anyhow!("registering plakat.enhance: {e}"))?;
    // v0.23 phase 4: plakat.style.* namespace.
    vm.register_inline("plakat.style.apply".to_string(), style::plakat_style_apply)
        .map_err(|e| anyhow!("registering plakat.style.apply: {e}"))?;
    vm.register_inline(
        "plakat.style.detect".to_string(),
        style::plakat_style_detect,
    )
    .map_err(|e| anyhow!("registering plakat.style.detect: {e}"))?;
    vm.register_inline("plakat.style.clear".to_string(), style::plakat_style_clear)
        .map_err(|e| anyhow!("registering plakat.style.clear: {e}"))?;
    vm.register_inline("plakat.style.list".to_string(), style::plakat_style_list)
        .map_err(|e| anyhow!("registering plakat.style.list: {e}"))?;
    vm.register_inline("plakat.style.train".to_string(), style::plakat_style_train)
        .map_err(|e| anyhow!("registering plakat.style.train: {e}"))?;
    // v0.25 phase 8: plakat.look.* + plakat.genre.* namespaces.
    vm.register_inline("plakat.look.apply".to_string(), look::plakat_look_apply)
        .map_err(|e| anyhow!("registering plakat.look.apply: {e}"))?;
    vm.register_inline("plakat.look.clear".to_string(), look::plakat_look_clear)
        .map_err(|e| anyhow!("registering plakat.look.clear: {e}"))?;
    vm.register_inline("plakat.look.list".to_string(), look::plakat_look_list)
        .map_err(|e| anyhow!("registering plakat.look.list: {e}"))?;
    vm.register_inline("plakat.genre.apply".to_string(), genre::plakat_genre_apply)
        .map_err(|e| anyhow!("registering plakat.genre.apply: {e}"))?;
    vm.register_inline("plakat.genre.clear".to_string(), genre::plakat_genre_clear)
        .map_err(|e| anyhow!("registering plakat.genre.clear: {e}"))?;
    vm.register_inline("plakat.genre.list".to_string(), genre::plakat_genre_list)
        .map_err(|e| anyhow!("registering plakat.genre.list: {e}"))?;
    // v0.23 phase 5: plakat.inpaint host word.
    vm.register_inline("plakat.inpaint".to_string(), inpaint::plakat_inpaint)
        .map_err(|e| anyhow!("registering plakat.inpaint: {e}"))?;
    // v0.24 phase 1: plakat.portrait.photo.* multi-photo namespace.
    vm.register_inline(
        "plakat.portrait.photo.add".to_string(),
        portrait_photo::plakat_portrait_photo_add,
    )
    .map_err(|e| anyhow!("registering plakat.portrait.photo.add: {e}"))?;
    vm.register_inline(
        "plakat.portrait.photo.clear".to_string(),
        portrait_photo::plakat_portrait_photo_clear,
    )
    .map_err(|e| anyhow!("registering plakat.portrait.photo.clear: {e}"))?;
    vm.register_inline(
        "plakat.portrait.photo.list".to_string(),
        portrait_photo::plakat_portrait_photo_list,
    )
    .map_err(|e| anyhow!("registering plakat.portrait.photo.list: {e}"))?;
    // v0.24 phase 4: plakat.outpaint host word.
    vm.register_inline("plakat.outpaint".to_string(), outpaint::plakat_outpaint)
        .map_err(|e| anyhow!("registering plakat.outpaint: {e}"))?;
    // v0.24 phase 5: plakat.embedding.* (Textual Inversion).
    vm.register_inline(
        "plakat.embedding.add".to_string(),
        embedding::plakat_embedding_add,
    )
    .map_err(|e| anyhow!("registering plakat.embedding.add: {e}"))?;
    vm.register_inline(
        "plakat.embedding.clear".to_string(),
        embedding::plakat_embedding_clear,
    )
    .map_err(|e| anyhow!("registering plakat.embedding.clear: {e}"))?;
    vm.register_inline(
        "plakat.embedding.list".to_string(),
        embedding::plakat_embedding_list,
    )
    .map_err(|e| anyhow!("registering plakat.embedding.list: {e}"))?;
    vm.register_inline(
        "plakat.embedding.train".to_string(),
        embedding::plakat_embedding_train,
    )
    .map_err(|e| anyhow!("registering plakat.embedding.train: {e}"))?;
    // v0.24 phase 6: plakat.stylize (IP-Adapter style transfer).
    vm.register_inline("plakat.stylize".to_string(), stylize::plakat_stylize)
        .map_err(|e| anyhow!("registering plakat.stylize: {e}"))?;
    // v2.3: image-producing words that mirror the like-named CLI subcommands.
    vm.register_inline("plakat.relight".to_string(), relight::plakat_relight)
        .map_err(|e| anyhow!("registering plakat.relight: {e}"))?;
    vm.register_inline("plakat.transparent".to_string(), transparent::plakat_transparent)
        .map_err(|e| anyhow!("registering plakat.transparent: {e}"))?;
    vm.register_inline("plakat.segment".to_string(), segment::plakat_segment)
        .map_err(|e| anyhow!("registering plakat.segment: {e}"))?;
    vm.register_inline("plakat.compose".to_string(), compose::plakat_compose)
        .map_err(|e| anyhow!("registering plakat.compose: {e}"))?;
    // v0.24 phase 7: plakat.metadata.read (JSON sidecar reader).
    vm.register_inline(
        "plakat.metadata.read".to_string(),
        metadata::plakat_metadata_read,
    )
    .map_err(|e| anyhow!("registering plakat.metadata.read: {e}"))?;
    // v0.26 phase 8: plakat.metadata.write (re-attach metadata
    // to an existing file from an image handle).
    vm.register_inline(
        "plakat.metadata.write".to_string(),
        metadata::plakat_metadata_write,
    )
    .map_err(|e| anyhow!("registering plakat.metadata.write: {e}"))?;
    // v0.28 phase 2: plakat.animate ( prompt out_dir -- )
    // — single-prompt AnimateDiff via the V3 or AnimateLCM motion
    // adapter; writes frame-NNNN.png to the given dir.
    vm.register_inline(
        "plakat.animate".to_string(),
        animate::plakat_animate,
    )
    .map_err(|e| anyhow!("registering plakat.animate: {e}"))?;
    // v0.36 phase 1: plakat.pixart ( prompt -- handle )
    // — single-image PixArt-Σ generation. Cached on
    // ScriptCtx.loaded_pixart; shares VAE with `plakat.load
    // <pixart-alias>` via the v0.34 phase 3 cache.
    vm.register_inline(
        "plakat.pixart".to_string(),
        pixart::plakat_pixart,
    )
    .map_err(|e| anyhow!("registering plakat.pixart: {e}"))?;
    // v0.38 phase 2: plakat.cascade ( prompt -- handle )
    // — single-image Stable Cascade generation via the 3-stage
    // pipeline. Cached on ScriptCtx.loaded_cascade. Honours
    // stage_c_steps / stage_b_steps config keys (or falls back to
    // splitting `steps` 2/3 + 1/3 the same way the CLI does).
    vm.register_inline(
        "plakat.cascade".to_string(),
        cascade::plakat_cascade,
    )
    .map_err(|e| anyhow!("registering plakat.cascade: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_plakat_words_is_idempotent_on_fresh_vm() {
        let mut vm = VM::new();
        register_plakat_words(&mut vm).unwrap();
        // A second call would re-register (bundcore allows upsert).
        // We don't enforce idempotency yet; this test just pins
        // that "fresh VM + register" doesn't fail with the v0.21
        // word set.
    }
}
