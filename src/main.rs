use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{DynamicImage, GrayImage, Rgba, Pixel};
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};
use rayon::prelude::*;

const DEFAULT_ALPHABET: &str =
    "> =ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
    kern_x: f32, // scaler of regular advance
}

#[derive(Clone, Copy)]
struct DecodeOptions {
    x_start: u32,
    y_start: u32,
    line_height: u32,
    line_advance: u32,
    width: u32,
}

#[allow(unused)]
fn render(font: &Font, text: &str, render_options: RenderOptions) -> Canvas {
    assert!(render_options.format == Format::A8);

    let mut glyph_pos = Vec::with_capacity(text.len());

    let mut pos = Vector2F::default();

    for char in text.chars() {
        let glyph_id = font.glyph_for_char(char).unwrap();
        glyph_pos.push((glyph_id, pos));
        pos += font.advance(glyph_id).unwrap() * render_options.size / 24. / 96.
            * render_options.kern_x;
    }

    let bounds = glyph_pos
        .iter()
        .fold(RectF::default(), |bounds, (glyph_id, pos)| {
            let raster_rect = font
                .raster_bounds(
                    *glyph_id,
                    render_options.size,
                    Transform2F::from_translation(*pos),
                    render_options.hinting,
                    render_options.rasterization,
                )
                .unwrap();
            bounds.union_rect(raster_rect.to_f32())
        });

    let mut canvas = Canvas::new(bounds.round().to_i32().size(), render_options.format);

    for (glyph_id, pos) in glyph_pos {
        font.rasterize_glyph(
            &mut canvas,
            glyph_id,
            render_options.size,
            Transform2F::from_translation(-bounds.origin()).translate(pos),
            render_options.hinting,
            render_options.rasterization,
        )
        .unwrap();
    }
    canvas
}

fn sum_of_squares(xs: &[u8], ys: &[u8]) -> i64 {
    assert!(xs.len() == ys.len());
    xs.iter()
        .zip(ys)
        .map(|(x, y)| (*x as i32 - *y as i32).pow(2) as i64)
        .sum::<i64>()
}

fn score_glyph(
    canvas: &mut Canvas,
    font: &Font,
    reference_pixels: &[u8],
    (_char, gid): (char, u32),
    origin: Vector2F,
    pos: Vector2F,
    render_options: RenderOptions,
) -> i64 {
    canvas.pixels.fill(0);

    let _raster_rect = font
        .raster_bounds(
            gid,
            render_options.size,
            Transform2F::from_translation(pos),
            render_options.hinting,
            render_options.rasterization,
        )
        .unwrap();

    font.rasterize_glyph(
        canvas,
        gid,
        render_options.size,
        Transform2F::from_translation(origin).translate(pos),
        render_options.hinting,
        render_options.rasterization,
    )
    .unwrap();

    // TODO can you just look at the rastered part? not sure how to get it
    sum_of_squares(reference_pixels, &canvas.pixels)
}

fn decode_line(
    reference: &GrayImage,
    font: &Font,
    alphabet: &str,
    render_options: RenderOptions,
) -> String {
    let (w, h) = reference.dimensions();
    let mut canvas = Canvas::new(Vector2I::new(w as i32, h as i32), render_options.format);

    let mut pos = Vector2F::default();

    let char_gids: Vec<_> = alphabet
        .chars()
        .map(|c| (c, font.glyph_for_char(c).unwrap()))
        .collect();

    // compute the biggest bbox we need for all chars in the alphabet. The
    // font.metrics().bounding_box seems to be too big? not sure if this is totally right
    // in general, but works for initial testing
    let bbox = char_gids
        .iter()
        .fold(RectF::default(), |bounds, (_char, gid)| {
            let raster_rect = font
                .raster_bounds(
                    *gid,
                    render_options.size,
                    Transform2F::default(),
                    render_options.hinting,
                    render_options.rasterization,
                )
                .unwrap();
            bounds.union_rect(raster_rect.to_f32())
        });
    let origin = -bbox.origin();

    // invert the reference since raster prints white text on black bg
    let reference_pixels: Vec<_> = reference.as_raw().iter().map(|x| 255 - x).collect();

    // the diff score we calculate is very high because the canvas is always mostly empty; can
    // subtract this baseline_diff to get a bit of a more sensible score, and the correct
    // character will have a negative score, but it ultimately doesn't matter
    // let baseline_diff = reference_pixels.iter().map(|v| (*v as i32).pow(2) as i64).sum::<i64>();

    let mut s = String::new();
    while pos.x() < w as f32 {
        let (c, gid) = char_gids
            .iter()
            .min_by_key(|(_char, gid)| {
                score_glyph(
                    &mut canvas,
                    font,
                    &reference_pixels,
                    (*_char, *gid),
                    origin,
                    pos,
                    render_options,
                )
            })
            .unwrap();

        s.push(*c);

        pos +=
            font.advance(*gid).unwrap() * render_options.size / 24. / 96. * render_options.kern_x;
    }
    s
}

fn decode_image(
    ref_img: &DynamicImage,
    font: &Font,
    alphabet: &str,
    decode_options: DecodeOptions,
    render_options: RenderOptions,
    mut cb: impl FnMut(String)
) {
    let DecodeOptions {
        x_start,
        y_start,
        line_advance,
        width,
        line_height,
    } = decode_options;

    for i in 0..u32::MAX {
        let img_line = ref_img
            .crop_imm(x_start, y_start + i * line_advance, width, line_height)
            .into_luma8();

        if img_line.height() == 0 {
            break;
        }
        if img_line.as_raw().iter().all(|c| *c == 255) {
            //eprintln!("whitespace line");
            continue;
        }
        let line = decode_line(&img_line, &font, alphabet, render_options);
        if line.is_empty() {
            break;
        }
        cb(line);
    }
}

fn draw_test_rectangles(img: &DynamicImage, decode_options: DecodeOptions) -> DynamicImage {
    let DecodeOptions {
        x_start,
        y_start,
        line_advance,
        width,
        line_height,
    } = decode_options;
    let mut ret = img.clone().into_rgba8();
    for i in 0..u32::MAX {
        let img_line = img
            .crop_imm(x_start, y_start + i * line_advance, width, line_height)
            .into_luma8();
        if img_line.height() == 0 {
            break;
        }
        if img_line.as_raw().iter().all(|x| *x == 255) {
            continue;
        }
        // horizontal
        let y = y_start + i * line_advance;
        for x in x_start..=x_start+width {
            ret.get_pixel_mut(x, y).blend(&Rgba([255, 0, 0, 128]));
            ret.get_pixel_mut(x, y + line_height).blend(&Rgba([255, 0, 0, 128]));
        }
        // vertical
        for y in y..=(y + line_height) {
            ret.get_pixel_mut(x_start, y).blend(&Rgba([255, 0, 0, 128]));
            ret.get_pixel_mut(x_start + width, y).blend(&Rgba([255, 0, 0, 128]));
        }
    }
    ret.into()
}

fn draw_test_text(font: &Font, text: &str, img: &DynamicImage, decode_options: DecodeOptions, render_options: RenderOptions) -> DynamicImage {
    let DecodeOptions {
        x_start,
        y_start,
        width,
        line_height,
        line_advance: _line_advance,
    } = decode_options;
    let mut img_line = img
        .crop_imm(x_start, y_start, width, line_height)
        .into_rgba8();
    let img_text = canvas_to_lum8(&render(font, text, render_options));
    for x in 0..img_line.width().min(img_text.width()) {
        for y in 0..img_line.height().min(img_text.height()) {
            let c = img_text.get_pixel(x, y)[0];
            if c == 255 {
                continue;
            }
            img_line.get_pixel_mut(x, y).blend(&Rgba([c, 0, 0, 128]));
        }
    }
    img_line.into()
}

fn decode_image_vec(
    ref_img: &DynamicImage,
    font: &Font,
    alphabet: &str,
    decode_options: DecodeOptions,
    render_options: RenderOptions,
) -> Vec<String> {
    let mut ret = Vec::with_capacity(128);
    decode_image(ref_img, font, alphabet, decode_options, render_options, |line| {
        ret.push(line);
    });
    ret
}

fn canvas_to_lum8(canvas: &Canvas) -> GrayImage {
    assert!(canvas.format == Format::A8);
    let w = canvas.size.x() as u32;
    let h = canvas.size.y() as u32;
    let mut pixels = canvas.pixels.clone();
    for px in pixels.iter_mut() {
        *px = 255 - *px;
    }
    GrayImage::from_raw(w as u32, h as u32, pixels).unwrap()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, num_args=1..)]
    img: Vec<String>,

    #[arg(short, long)]
    font: String,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,

    #[arg(long)]
    hinting: bool,

    #[arg(short, long)]
    text_size: f32,

    #[arg(short, long, default_value_t = 1.0)]
    kerning: f32,

    #[arg(short, long, default_value_t = 0)]
    x: u32,

    #[arg(short, long, default_value_t = 0)]
    y: u32,

    #[arg(short, long)]
    width: u32,

    #[arg(long)]
    line_height: u32,

    #[arg(long)]
    line_advance: u32,

    #[arg(long)]
    debug: bool,

    #[arg(long)]
    test: Option<String>,
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
        kern_x: args.kerning,
        hinting,
    };

    let decode_options = DecodeOptions {
        x_start: args.x,
        y_start: args.y,
        width: args.width,
        line_height: args.line_height,
        line_advance: args.line_advance,
    };

    if let Some(test_outfile) = args.test {
        let img = image::open(args.img.first().unwrap()).unwrap();
        let out = draw_test_rectangles(&img, decode_options);
        out.save(format!("{test_outfile}-rect.png")).unwrap();

        let font = Font::from_path(args.font, 0).unwrap();
        let out = draw_test_text(&font, &args.alphabet, &img, decode_options, render_options);
        out.save(format!("{test_outfile}-rect.png")).unwrap();
        return;
    }

    if args.img.len() == 1 {
        let font = Font::from_path(args.font.clone(), 0).unwrap();
        let img = image::open(args.img.first().unwrap()).unwrap();

        decode_image(&img, &font, &args.alphabet, decode_options, render_options, |line| {
            println!("{line}");
        });
    } else {
        let imgs: Vec<_> = args.img.into_iter().enumerate().collect();
        let mut liness: Vec<_> = imgs.into_par_iter().map(|(i, img)| {
            let font = Font::from_path(args.font.clone(), 0).unwrap();
            let img = image::open(img).unwrap();
            let lines = decode_image_vec(&img, &font, &args.alphabet, decode_options, render_options);
            (i, lines)
        }).collect();
        liness.sort();
        for (_, lines) in liness {
            for line in lines {
                println!("{line}");
            }
        }
    }
}

//fn test_image() {
//    let size = 13.0;
//    let kern_x = 1.125;
//
//
//    let font = Font::from_path("Courier New.otf", 0).unwrap();
//
//    let canvas = render(&font, text, render_options);
//
//    let image = canvas_to_rgba8(&canvas);
//    image.save("font_kit.png").unwrap();
//
//    let mut pixels = canvas.pixels.clone();
//    for px in pixels.iter_mut() {
//        *px = 255 - *px;
//    }
//    canvas_to_lum8(&canvas).save("font_kit-l8.png").unwrap();
//
//    let mut _base_image = image::open("imgs-000.png")
//        .unwrap();
//    let _base_image = _base_image
//        //.crop_imm(45, 48 - 9, 608, 12)
//        //.crop_imm(45, 48 - 9 + 0, 608, 12)
//        .crop_imm(45, 48 - 9 + 0, 608, 12)
//        .into_rgba8();
//
//    let x_start = 45;
//    let y_start = 48 - 9;
//    let width = 608;
//    let height = 12;
//    let reference_image = image::open("imgs-000.png")
//        .unwrap()
//        .crop_imm(x_start, y_start, width, height)
//        .into_luma8();
//    reference_image.save("foo.png").unwrap();
//
//}
//
