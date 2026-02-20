use std::time::Instant;

use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::Vector2F;
use image::{DynamicImage, GrayImage, Luma, Pixel, Rgb, RgbImage, Rgba};
use clap::Parser;

const DEFAULT_ALPHABET: &str =
    ">=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
}

fn render(font: &Font, char: char, offset: [f32; 2], render_options: RenderOptions) -> Canvas {
    assert!(render_options.format == Format::A8);

    let glyph_id = font.glyph_for_char(char).unwrap();
    let pos = Vector2F::new(offset[0], offset[1]);

    let bounds = font
        .raster_bounds(
            glyph_id,
            render_options.size,
            Transform2F::from_translation(pos),
            render_options.hinting,
            render_options.rasterization,
        ).unwrap();

    let mut canvas = Canvas::new(bounds.size(), render_options.format);

    font.rasterize_glyph(
        &mut canvas,
        glyph_id,
        render_options.size,
        Transform2F::from_translation(-bounds.to_f32().origin()).translate(pos),
        render_options.hinting,
        render_options.rasterization,
    )
    .unwrap();
    canvas
}

fn canvas_to_lum8(canvas: &Canvas) -> GrayImage {
    assert!(canvas.format == Format::A8);
    let w = canvas.size.x() as u32;
    let h = canvas.size.y() as u32;
    let mut pixels = canvas.pixels.clone();
    for px in pixels.iter_mut() {
        *px = 255 - *px;
    }
    GrayImage::from_raw(w, h, pixels).unwrap()
}

fn se(a: Luma<u8>, b: Luma<u8>) -> u32 {
    ((a[0] as i16 - b[0] as i16) as i32).pow(2) as u32
}

#[derive(Debug)]
struct Match {
    x: f32,
    y: f32,
    mse: f32,
}

fn search(needle: &GrayImage, reference: &GrayImage, threshold: f32) -> Vec<Match> {
    let mut ret = vec![];
    //let (w_n, h_n) = (needle.width(), needle.height());
    //let (w_r, h_r) = (reference.width(), reference.height());
    let x_searches = reference.width() - needle.width() + 1;
    let y_searches = reference.height() - needle.height() + 1;
    let divisor = (needle.width() as f32) * (needle.height() as f32);
    let half_width = (needle.width() as f32) / 2.;
    let half_height = (needle.height() as f32) / 2.;
    let mut acc = vec![0u32; x_searches as usize];
    let mut min = f32::MAX;

    for y in 0..y_searches {
        for needle_y in 0..needle.height() {
            for (x, acc) in acc.iter_mut().enumerate() {
                for needle_x in 0..needle.width() {
                    let npx = needle.get_pixel(needle_x, needle_y);
                    let rpx = reference.get_pixel((x as u32) + needle_x, y + needle_y);
                    *acc += se(*npx, *rpx);
                }
            }
        }
        for (x, acc) in acc.iter().enumerate() {
            let mse = *acc as f32 / divisor;
            if mse < min {
                min = mse;
            }
            if mse < threshold {
                let x = (x as f32) + half_width;
                let y = (y as f32) + half_height;
                ret.push(Match{x, y, mse}); // TODO make into centered coords
            }
        }
        acc.fill(0);
    }
    eprintln!("got min of {min}");
    ret
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, num_args=1..)]
    img: Vec<String>,

    #[arg(short, long)]
    font: String,

    #[arg(short, long)]
    text_size: f32,

    #[arg(long)]
    hinting: bool,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,
}

fn main() {
    let args = Args::parse();

    let hinting = if args.hinting {
        HintingOptions::Full(args.text_size)
    } else {
        HintingOptions::None
    };

    let render_options = RenderOptions {
        format: Format::A8,
        rasterization: RasterizationOptions::GrayscaleAa,
        size: args.text_size,
        hinting,
    };

    let offsets = [
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [0.25, 0.0],
        [0.0, 0.25],
        [0.25, 0.25],
    ];


    let font = Font::from_path(args.font, 0).unwrap();
    let img = image::open(args.img.first().unwrap()).unwrap().into_luma8();
    let t00 = Instant::now();
    for letter in args.alphabet.chars() {
        for offset in offsets {

            let needle = canvas_to_lum8(&render(&font, letter, offset, render_options));
            eprintln!("`{letter}` {offset:?} needle size {}x{}", needle.width(), needle.height());
            let t0 = Instant::now();
            let hits = search(&needle, &img, 1000.);
            let t1 = Instant::now();
            for hit in hits {
                //println!("{hit:?}");
                println!("{},{}", hit.x + offset[0], hit.y + offset[1]);
            }
            eprintln!("took {:.4}ms", (t1 - t0).as_millis());
        }
    }
    let t11 = Instant::now();
    eprintln!("overall {:.4}ms", (t11 - t00).as_millis());
}
