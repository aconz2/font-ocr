use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{DynamicImage, GrayImage, Rgba};
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
    kern_x: f32, // scaler of regular advance
}

fn render(font: &Font, text: &str, start_pos: Vector2F, render_options: RenderOptions) -> Canvas {
    let mut glyph_pos = Vec::with_capacity(text.len());

    let mut pos = start_pos;

    for char in text.chars() {
        let glyph_id = font.glyph_for_char(char).unwrap();
        glyph_pos.push((glyph_id, pos));
        pos += font.advance(glyph_id).unwrap() * render_options.size / 24. / 96.
            * render_options.kern_x;
    }

    let mut bounds = RectF::new(Vector2F::splat(0.), Vector2F::splat(0.));
    for (glyph_id, pos) in &glyph_pos {
        let raster_rect = font
            .raster_bounds(
                *glyph_id,
                render_options.size,
                Transform2F::from_translation(*pos),
                render_options.hinting,
                render_options.rasterization,
            )
            .unwrap();
        bounds = bounds.union_rect(raster_rect.to_f32());
    }

    println!("bounds={:?}", bounds);

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

    //println!("{_char} raster rect {:?} {:?}", raster_rect.origin(), raster_rect.size());
    font.rasterize_glyph(
        canvas,
        gid,
        render_options.size,
        // have to use the origin from as if we had a whole line, not just a single char...
        Transform2F::from_translation(origin).translate(pos),
        render_options.hinting,
        render_options.rasterization,
    )
    .unwrap();

    //let mut pixels = canvas.pixels.clone();
    //for px in pixels.iter_mut() {
    //    *px = 255 - *px;
    //}
    //GrayImage::from_raw(w, h, pixels).unwrap().save("foo.png").unwrap();

    // TODO can you just look at the rastered part? not sure how to get it
    sum_of_squares(reference_pixels, &canvas.pixels)
}

fn decode_line(
    reference: &GrayImage,
    font: &Font,
    nchars: usize,
    start_pos: Vector2F,
    alphabet: &str,
    render_options: RenderOptions,
) -> String {
    let origin = Vector2F::new(0., 9.); // TODO

    let (w, h) = reference.dimensions();
    let mut canvas = Canvas::new(Vector2I::new(w as i32, h as i32), render_options.format);

    let mut pos = start_pos;

    let char_gids: Vec<_> = alphabet
        .chars()
        .map(|c| (c, font.glyph_for_char(c).unwrap()))
        .collect();

    // invert the reference since raster prints white text on black bg
    let reference_pixels: Vec<_> = reference.as_raw().iter().map(|x| 255 - x).collect();

    // the diff score we calculate is very high because the canvas is always mostly empty; can
    // subtract this baseline_diff to get a bit of a more sensible score, and the correct
    // character will have a negative score, but it ultimately doesn't matter
    // let baseline_diff = reference_pixels.iter().map(|v| (*v as i32).pow(2) as i64).sum::<i64>();

    (0..nchars)
        .map(|_| {
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
            pos += font.advance(*gid).unwrap() * render_options.size / 24. / 96.
                * render_options.kern_x;
            c
        })
        .collect()
}

fn test_image() {
    let size = 13.0;
    let kern_x = 1.125;
    let _line_height = 15.; // amount to skip down to write next line
    //let start_pos = Vector2F::new(45.0, 48. + 15.); // < this matches the start pos of the whole page
    let start_pos = Vector2F::new(0., 0.);

    let render_options = RenderOptions {
        format: Format::A8,
        rasterization: RasterizationOptions::GrayscaleAa,
        hinting: HintingOptions::Full(size),
        size,
        kern_x,
    };

    let font = Font::from_path("Courier New.otf", 0).unwrap();
    let metrics = font.metrics();
    let height = metrics.ascent - metrics.descent;
    println!("height {height} {}", height / 24. / 96.);

    println!("metrics {:?}", font.metrics());
    println!("mono? {}", font.is_monospace());

    let text = "> MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0+Pg1lbmRvYmoNICAgICAgICAgICAgICAg";
    //let text = "> DQo1MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0-----------------------------";
    //let text = ">";

    let canvas = render(&font, text, start_pos, render_options);
    let w = canvas.size.x() as usize;
    let h = canvas.size.y() as usize;
    println!("render canvas size w={w} h={h}");

    let mut image = DynamicImage::new_rgba8(w as u32, h as u32).to_rgba8();
    for y in 0..h {
        let (row_start, row_end) = (y * canvas.stride, (y + 1) * canvas.stride);
        let row = &canvas.pixels[row_start..row_end];
        for x in 0..w {
            let c = 255 - row[x];
            if c != 255 {
                let color = Rgba([c, c, c, 255]);
                image.put_pixel(x as u32, y as u32, color);
            }
        }
    }
    image.save("font_kit.png").unwrap();

    let mut pixels = canvas.pixels.clone();
    for px in pixels.iter_mut() {
        *px = 255 - *px;
    }
    GrayImage::from_raw(w as u32, h as u32, pixels)
        .unwrap()
        .save("font_kit-l8.png")
        .unwrap();

    let mut _base_image = image::open("imgs-000.png")
        .unwrap()
        //.crop_imm(45, 48 - 9, 608, 12)
        .crop_imm(45, 48 - 9 + 0, 608, 12)
        .into_rgba8();

    //image::imageops::overlay(&mut base_image, &image, 0, 0);
    _base_image.save("overlay.png").unwrap();

    let x_start = 45;
    let y_start = 48 - 9;
    let width = 608;
    let height = 12;
    let reference_image = image::open("imgs-000.png")
        .unwrap()
        .crop_imm(x_start, y_start, width, height)
        .into_luma8();

    let start_pos = Vector2F::new(0., 0.);
    let alphabet = "> =ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    //let alphabet = "> M";
    let n_chars = 78;
    //let n_chars = 1;
    let line = decode_line(
        &reference_image,
        &font,
        n_chars,
        start_pos,
        alphabet,
        render_options,
    );
    println!("yo decoded line to `{line}`");
    println!("match? {}", line == text);

    if false {
        let reference_image = image::open("imgs-000.png").unwrap();
        let x_start = 45;
        let y_start = 48 - 9;
        let width = 608;
        let height = 12;
        let n_chars = 78;
        let n_lines = 65;

        for i in 0..n_lines {
            let img_line = reference_image
                .crop_imm(x_start, y_start + i * 15, width, height)
                .into_luma8();
            if img_line.height() == 0 {
                break;
            }
            let line = decode_line(
                &img_line,
                &font,
                n_chars,
                start_pos,
                alphabet,
                render_options,
            );
            if line.is_empty() {
                break;
            }
            println!("{line}");
        }
    }
}

fn main() {
    test_image();
}
