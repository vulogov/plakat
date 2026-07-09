//! Public-API surface lock for `plakat::api`.
//!
//! This test does not *run* anything — it exists so the **stable surface fails to compile** if
//! a builder, method, or re-exported type is renamed or removed. That turns an accidental
//! breaking change into a red test instead of a silent semver break. (For a richer diff of the
//! full public surface, `cargo public-api` can be layered on later; this is the zero-dependency
//! floor.)
//!
//! Everything here is behind `if false` so no model weights or devices are touched.

#![allow(unreachable_code, unused_must_use, clippy::diverging_sub_expression)]

use plakat::api::{
    device, Animate, EmbeddingTrain, Generate, IdentityKind, Image, Img2img, Map, MapSpec,
    Multiperson, Person, Portrait, Relight, SchedulerKind, Segment, StyleTrain, Stylize,
    Transparent, UpscaleMethod, Upscale, VideoFormat, Verify, Distance, Facing, Position,
};

#[test]
fn public_api_surface_is_stable() {
    if false {
        // Free items.
        let _: fn(&str) -> anyhow::Result<candle_core::Device> = device;
        let _img: Image = unreachable!();
        let _: &[u8] = _img.pixels();
        let _: u32 = _img.width();
        let _: u32 = _img.height();
        _img.save("x.png");
        let _ = Image::open("x.png");

        // Generation.
        let _ = async {
            Generate::new("sdxl")
                .prompt("p")
                .negative("n")
                .size(512, 512)
                .steps(20)
                .guidance(7.5)
                .seed(1)
                .count(1)
                .clip_skip(2)
                .scheduler(SchedulerKind::default())
                .device("auto")
                .lora("l", 0.8)
                .run()
                .await
        };

        // Editing.
        let _ = async {
            Img2img::new("sd15", "in.png")
                .prompt("p")
                .strength(0.5)
                .mask("m.png")
                .mask_feather(4)
                .mask_invert(true)
                .run()
                .await
        };
        let _ = async { Upscale::new("in.png").scale(2.0).method(UpscaleMethod::RealEsrganX4).run().await };

        // Style / relight.
        let _ = async { Relight::new("s.png").prompt("light").size(512, 512).run().await };
        let _ = async { Stylize::new("in.png", "ref.png").model("sdxl").instantstyle(true).run().await };

        // People / masks.
        let _ = async { Transparent::new("in.png").crop(true).run("out.png").await };
        let _ = async { Segment::new("in.png").point(10.0, 20.0, true).feather(3).run("m.png").await };
        let _ = async {
            Portrait::new("sdxl").prompt("p").photo("a.png", 0.9).identity(IdentityKind::FaceIdSdxl).run().await
        };
        let _ = async {
            Multiperson::new("scene")
                .person(Person::new("a").photo("a.png", 1.0).place(Position::Left, Distance::Mid, Facing::Front))
                .identity(IdentityKind::PlusFace)
                .run()
                .await
        };

        // Map / animate.
        let _ = async { Map::from_spec(MapSpec::minimal("w", 4, 4, 3)).style("inked").seed(1).render().await };
        let _ = async { Map::from_prose("a rainy archipelago").provider("none").render().await };
        let _ = async { Animate::new("sd15", "a", "b").frames(16).format(VideoFormat::Frames).run().await };

        // Training / verify.
        let _ = async { StyleTrain::new("sd15", vec!["a.png".into()], "o.safetensors").trigger("t").run().await };
        let _ = async { EmbeddingTrain::new("sdxl", vec!["a.png".into()], "<t>", "e.safetensors").run().await };
        let _ = async { Verify::new().tier(1).model("sdxl").json(true).run().await };
    }
}
