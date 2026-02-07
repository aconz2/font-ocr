#![allow(unused)]

use std::fs::File;
use std::io::Read;
use image::{DynamicImage, Rgba, Rgb, Pixel, Luma, GrayImage};

// so it looks like 0.42 kerning

//const FONT_FILE: &str = "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf";
const FONT_FILE: &str = "Courier New.ttf";
//const FONT_FILE: &str = "Courier.ttf";

//fn test_image() {
//    let text = "Rust Typography!";
//    let font_size = 48;
//
//    // 1. Initialize FreeType and load the face
//    let lib = Library::init().unwrap();
//    let ft_face = lib.new_face(FONT_FILE, 0).unwrap();
//    ft_face.set_char_size(font_size << 6, 0, 96, 96).unwrap();
//
//    // 2. Initialize HarfBuzz shaping
//    let hb_face = Face::from_file(FONT_FILE, 0).unwrap();
//    let mut hb_font = Font::new(hb_face);
//    let buffer = UnicodeBuffer::new().add_str(text);
//    let glyphs = shape(&mut hb_font, buffer, &[]);
//
//    let positions = glyphs.get_glyph_positions();
//    let infos = glyphs.get_glyph_infos();
//
//    // 3. Prepare the Image canvas
//    let mut img = GrayImage::new(800, 200);
//    let mut cursor_x = 20.0;
//    let cursor_y = 100.0;
//
//    // 4. Render each glyph
//    for (info, pos) in infos.iter().zip(positions.iter()) {
//        let gid = info.codepoint;
//        ft_face.load_glyph(gid, freetype::face::LoadFlag::RENDER).unwrap();
//
//        let glyph = ft_face.glyph();
//        let bitmap = glyph.bitmap();
//        let x = cursor_x + (glyph.bitmap_left() as f32) + (pos.x_offset as f32 / 64.0);
//        let y = cursor_y - (glyph.bitmap_top() as f32) - (pos.y_offset as f32 / 64.0);
//
//        // Draw the bitmap pixels onto our GrayImage
//        let width = bitmap.width() as u32;
//        let rows = bitmap.rows() as u32;
//        let buffer = bitmap.buffer();
//
//        for row in 0..rows {
//            for col in 0..width {
//                let pixel_val = buffer[(row * width + col) as usize];
//                if pixel_val > 0 {
//                    let px = (x as u32 + col) as u32;
//                    let py = (y as u32 + row) as u32;
//                    if px < img.width() && py < img.height() {
//                        img.put_pixel(px, py, Luma([pixel_val]));
//                    }
//                }
//            }
//        }
//
//        // Advance the cursor
//        cursor_x += pos.x_advance as f32 / 64.0;
//    }
//
//    img.save("rendered_text.png").unwrap();
//    println!("Text rendered to rendered_text.png");
//}
//

use font_kit::canvas::{Canvas, RasterizationOptions, Format};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F,Vector2I};
use pathfinder_geometry::rect::{RectI,RectF};

fn render(font: &Font, text: &str, size: f32, kern_x: f32, start_pos: Vector2F) -> Canvas {
    let format = Format::A8;
    let rasterization = RasterizationOptions::GrayscaleAa;
    let hinting = HintingOptions::Full(size);

    let mut glyph_pos = vec![];
    glyph_pos.reserve(text.len());

    let mut pos = start_pos;

    for char in text.chars() {
        let glyph_id = font.glyph_for_char(char).unwrap();
        let advance = font.advance(glyph_id).unwrap();
        glyph_pos.push((glyph_id, pos));
        pos += font.advance(glyph_id).unwrap() * size / 24. / 96. * kern_x;
    }

    let mut bounds = RectF::new(Vector2F::splat(0.), Vector2F::splat(0.));
    for (glyph_id, pos) in &glyph_pos {
        let raster_rect = font
            .raster_bounds(
                *glyph_id,
                size,
                Transform2F::from_translation(*pos),
                hinting,
                rasterization,
            )
            .unwrap();
        bounds = bounds.union_rect(raster_rect.to_f32());
    }

    println!("bounds={:?}", bounds);

    let mut canvas = Canvas::new(bounds.round().to_i32().size(), format);

    for (glyph_id, pos) in glyph_pos {
        font.rasterize_glyph(
            &mut canvas,
            glyph_id,
            size,
            Transform2F::from_translation(-bounds.origin()).translate(pos),
            hinting,
            rasterization,
        )
        .unwrap();
    }
    canvas
}

fn decode_line(reference: &GrayImage, font: &Font, nchars: usize, size: f32, kern_x: f32, start_pos: Vector2F, alphabet: &str) -> String {
    let format = Format::A8;
    let rasterization = RasterizationOptions::GrayscaleAa;
    let transform = Transform2F::default();
    let hinting = HintingOptions::Full(size);

    let (w, h) = reference.dimensions();
    let mut canvas = Canvas::new(Vector2I::new(w as i32, h as i32), format);
    println!("decode canvas size {:?}", canvas.size);

    let mut pos = start_pos;
    println!("(pos {pos:?})");

    let char_gids: Vec<_> = alphabet.chars().map(|c| (c, font.glyph_for_char(c).unwrap())).collect();
    println!("{char_gids:?}");

    let mut score_glyph = |(_char, gid): &&(char, u32)| {
        canvas.pixels.fill(0);

        let raster_rect = font
            .raster_bounds(
                *gid,
                size,
                Transform2F::from_translation(pos),
                hinting,
                rasterization,
            )
            .unwrap();
        //println!("raster rect {raster_rect:?}");
        font.rasterize_glyph(
            &mut canvas,
            *gid,
            size,
            // have to use the origin from as if we had a whole line, not just a single char...
            Transform2F::from_translation(Vector2F::new(0., 9.)).translate(pos),
            hinting,
            rasterization,
        )
        .unwrap();

        let mut pixels = canvas.pixels.clone();
        for px in pixels.iter_mut() {
            *px = 255 - *px;
        }
        GrayImage::from_raw(w, h, pixels).unwrap().save("foo.png").unwrap();

        let sqr = |x| { x * x };

        let diff: i64 = reference
            .as_raw()
            .iter()
            .zip(canvas.pixels.iter())
            .map(|(reference, canvas)| sqr(*reference as i32 - ((255 - canvas) as i32)) as i64)
            .sum();
        diff
    };

    (0..nchars)
        .map(|_| {
            let (c, _) = char_gids.iter().min_by_key(&mut score_glyph).unwrap();
            c
        })
        .collect()
}

fn test_image() {
    let size = 13.0;
    let kern_x = 1.125;
    let line_height = 15.; // amount to skip down to write next line
    //let start_pos = Vector2F::new(45.0, 48. + 15.); // < this matches the start pos of the whole page
    let start_pos = Vector2F::new(0., 0.);

    let font = Font::from_path("Courier New.otf", 0).unwrap();
    let metrics = font.metrics();
    let height = metrics.ascent - metrics.descent;
    println!("height {height} {}", height as f32 / 24. / 96.);

    println!("metrics {:?}", font.metrics());
    println!("mono? {}", font.is_monospace());

    let text = "> MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0+Pg1lbmRvYmoNICAgICAgICAgICAgICAg";
    //let text = "> DQo1MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0-----------------------------";
    //let text = ">";

    let canvas = render(&font, text, size, kern_x, start_pos);
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

    //let mut image = DynamicImage::new_luma8(w as u32, h as u32).to_luma8();
    //for px in image.pixels_mut() {
    //    *px = Luma([255]);
    //}
    //for y in 0..h {
    //    let (row_start, row_end) = (y * canvas.stride, (y + 1) * canvas.stride);
    //    let row = &canvas.pixels[row_start..row_end];
    //    for x in 0..w {
    //        let c = 255 - row[x];
    //        if c != 255 {
    //            let color = Luma([c]);
    //            image.put_pixel(x as u32, y as u32, color);
    //        }
    //    }
    //}
    let mut pixels = canvas.pixels.clone();
    for px in pixels.iter_mut() {
        *px = 255 - *px;
    }
    GrayImage::from_raw(w as u32, h as u32, pixels).unwrap().save("font_kit-l8.png").unwrap();

    let mut base_image = image::open("imgs-000.png")
        .unwrap()
        //.crop_imm(45, 48 - 9, 608, 12)
        .crop_imm(45, 48 - 9 + 0, 608, 12)
        .into_rgba8();

    //image::imageops::overlay(&mut base_image, &image, 0, 0);
    base_image.save("overlay.png").unwrap();


    let x_start = 45;
    let y_start = 48 - 9;
    let width = 608;
    let height = 12;
    let mut reference_image = image::open("imgs-000.png")
        .unwrap()
        .crop_imm(x_start, y_start, width, height)
        .into_luma8();

    let start_pos = Vector2F::new(0., 0.);
    let alphabet = "> =ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    //let alphabet = "> M";
    let line = decode_line(&reference_image, &font, 10, size, kern_x, start_pos, alphabet);
    println!("yo decoded line to `{line}`");

}

fn main() {
    test_image();
}

//use rusttype;
//use rusttype::{Font, Scale};
//fn load_font() -> Font<'static> {
//    let mut buf = vec![];
//    let mut f = File::open(FONT_FILE).unwrap();
//    f.read_to_end(&mut buf).unwrap();
//    Font::try_from_vec(buf).unwrap()
//}
//
//fn test_image() {
//    let font = load_font();
//
//    //let scale = Scale::uniform(15.0); // in pixels
//    let scale = Scale{x: 14.0, y: 14.0}; // in pixels
//
//    //let text = "> ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
//    let text = "> MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0+Pg1lbmRvYmoNICAgICAgICAgICAgICAg";
//    let color = (0, 0, 0);
//
//    let v_metrics = font.v_metrics(scale);
//    println!("ascent={} descent={}", v_metrics.ascent, v_metrics.descent);
//
//    //let glyphs: Vec<_> = font
//    //    .layout(text, scale, rusttype::point(0.0, 0.0 + v_metrics.ascent))
//    //    .collect();
//    //{
//    //
//    //    use harfbuzz_rs::{Face, UnicodeBuffer, shape, Font};
//    //    let hb_face = Face::from_file(FONT_FILE, 0).unwrap();
//    //    let mut hb_font = Font::new(hb_face);
//    //    let buffer = UnicodeBuffer::new().add_str(text);
//    //    let glyphs = shape(&mut hb_font, buffer, &[]);
//    //
//    //    let positions = glyphs.get_glyph_positions();
//    //    let infos = glyphs.get_glyph_infos();
//    //    for (info, pos) in infos.iter().zip(positions.iter()) {
//    //        println!("{info:?} {pos:?}");
//    //    }
//    //}
//
//    let mut base_image = image::open("imgs-000.png")
//        .unwrap()
//        .crop_imm(0, 0, 816, 100)
//        .into_rgba8();
//
//    // 46 for the >
//    let start = rusttype::point(46.0, 48.25 - 10.0);
//    let glyphs: Vec<_> = font.glyphs_for(text.chars())
//        .scan((None, 0.0), |(last, x), g| {
//            let g = g.scaled(scale);
//            if let Some(last) = last {
//                *x += font.pair_kerning(scale, *last, g.id());
//            }
//            let w = g.h_metrics().advance_width;
//            //println!("w={w}");
//            let next = g.positioned(start + rusttype::vector(*x, 0.0));
//            *last = Some(next.id());
//            *x += w * 0.95;
//            Some(next)
//        })
//        .collect();
//
//    let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
//    let glyphs_width = {
//        let min_x = glyphs
//            .first()
//            .map(|g| g.pixel_bounding_box().unwrap().min.x)
//            .unwrap();
//        let max_x = glyphs
//            .last()
//            .map(|g| g.pixel_bounding_box().unwrap().max.x)
//            .unwrap();
//        (max_x - min_x) as u32
//    };
//    println!("glyphs_height={glyphs_height} glyphs_width={glyphs_width}");
//
//    //let mut image = DynamicImage::new_rgba8(glyphs_width + 2, glyphs_height + 0).to_rgba8();
//    let mut image = DynamicImage::new_rgba8(816, 100).to_rgba8();
//    println!("image is {}x{}", image.width(), image.height());
//
//    for px in image.pixels_mut() {
//        *px = Rgba([0, 0, 0, 0]);
//    }
//
//    let x_scaler = 1.00;
//
//    for glyph in glyphs {
//        if let Some(bounding_box) = glyph.pixel_bounding_box() {
//            glyph.draw(|x, y, v| {
//                let x = x + bounding_box.min.x as u32;
//                let x = (x as f32 * x_scaler) as u32;
//                let y = y + bounding_box.min.y as u32;
//                image.get_pixel_mut(x, y).blend(&Rgba([color.0, color.1, color.2, (v * 255.0) as u8]));
//            });
//        }
//    }
//
//    image.save("image_example.png").unwrap();
//
//    image::imageops::overlay(&mut base_image, &image, 0, 0);
//    //DynamicImage::ImageRgba8(base_image).into_luma8().save("overlay.png").unwrap();
//    DynamicImage::ImageRgba8(base_image).crop_imm(40, 20, 20, 40).save("overlay.png").unwrap();
//}
//
//
        //Format::Rgb24 => {
        //    let mut image = DynamicImage::new_rgba8(raster_rect.width() as u32, raster_rect.height() as u32).to_rgb8();
        //    for px in image.pixels_mut() {
        //        *px = Rgb([255, 255, 255]);
        //    }
        //    for y in 0..raster_rect.height() {
        //        let (row_start, row_end) = (y as usize * canvas.stride, (y + 1) as usize * canvas.stride);
        //        let row = &canvas.pixels[row_start..row_end];
        //        for x in 0..raster_rect.width() {
        //            let i = x as usize * 3;
        //            let mut color = Rgb([row[i], row[i + 1], row[i + 2]]);
        //            color.invert();
        //            image.put_pixel(x.try_into().unwrap(), y.try_into().unwrap(), color);
        //        }
        //    }
        //    image.save("font_kit.png").unwrap();
