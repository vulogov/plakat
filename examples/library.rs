//! Using plakat as a library — text-to-image, img2img/inpaint, and upscaling.
//!
//! Run with: `cargo run --example library` (needs the model weights; first run downloads
//! them from Hugging Face, exactly like the CLI).

use plakat::api::{Generate, Img2img, Upscale, UpscaleMethod};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Text-to-image.
    let images = Generate::new("sd15")
        .prompt("a portrait of a red fox in a sunlit forest, detailed fur")
        .negative("blurry, watermark")
        .size(512, 512)
        .steps(20)
        .guidance(7.5)
        .seed(42)
        .run()
        .await?;
    images[0].save("fox.png")?;
    println!("wrote fox.png ({}x{})", images[0].width(), images[0].height());

    // 2. Img2img — transform an existing image (add `.mask(path)` to inpaint instead).
    let edited = Img2img::new("sd15", "fox.png")
        .prompt("a fox in a snowy forest, winter")
        .strength(0.55)
        .seed(42)
        .run()
        .await?;
    edited[0].save("fox_winter.png")?;

    // 3. Upscale — classical or Real-ESRGAN.
    let big = Upscale::new("fox_winter.png")
        .method(UpscaleMethod::RealEsrganX4)
        .run()
        .await?;
    big.save("fox_winter_4x.png")?;
    println!("upscaled to {}x{}", big.width(), big.height());

    Ok(())
}
