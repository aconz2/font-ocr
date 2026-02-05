use std::fs::File;
use std::io::Read;
use image::{DynamicImage, Rgba, Rgb, Pixel};

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
fn main() {
    test_image();
    println!("Hello, world!");
}

use rusttype;
use rusttype::{Font, Scale};
fn load_font() -> Font<'static> {
    let mut buf = vec![];
    let mut f = File::open(FONT_FILE).unwrap();
    f.read_to_end(&mut buf).unwrap();
    Font::try_from_vec(buf).unwrap()
}

fn test_image() {
    let font = load_font();

    let scale = Scale::uniform(15.0);

    //let text = "> ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let text = ">MTcxODE";
    let text = "> MTcxODExL04gMi9UIDI3NTY3OC9IIFsgNTIyIDIzMV0+Pg1lbmRvYmoNICAgICAgICAgICAgICAg";
    let color = (0, 0, 0);

    let v_metrics = font.v_metrics(scale);
    println!("ascent={} descent={}", v_metrics.ascent, v_metrics.descent);

    //let glyphs: Vec<_> = font
    //    .layout(text, scale, rusttype::point(0.0, 0.0 + v_metrics.ascent))
    //    .collect();
    //{
    //
    //    use harfbuzz_rs::{Face, UnicodeBuffer, shape, Font};
    //    let hb_face = Face::from_file(FONT_FILE, 0).unwrap();
    //    let mut hb_font = Font::new(hb_face);
    //    let buffer = UnicodeBuffer::new().add_str(text);
    //    let glyphs = shape(&mut hb_font, buffer, &[]);
    //
    //    let positions = glyphs.get_glyph_positions();
    //    let infos = glyphs.get_glyph_infos();
    //    for (info, pos) in infos.iter().zip(positions.iter()) {
    //        println!("{info:?} {pos:?}");
    //    }
    //}

    let start = rusttype::point(0.0, 0.0 + v_metrics.ascent);
    let glyphs: Vec<_> = font.glyphs_for(text.chars())
        .scan((None, 0.0), |(last, x), g| {
            let g = g.scaled(scale);
            if let Some(last) = last {
                *x += font.pair_kerning(scale, *last, g.id());
            }
            let w = g.h_metrics().advance_width;
            //println!("w={w}");
            let next = g.positioned(start + rusttype::vector(*x, 0.0));
            *last = Some(next.id());
            *x += w * 0.98;
            Some(next)
        })
        .collect();

    let glyphs_height = (v_metrics.ascent - v_metrics.descent).ceil() as u32;
    let glyphs_width = {
        let min_x = glyphs
            .first()
            .map(|g| g.pixel_bounding_box().unwrap().min.x)
            .unwrap();
        let max_x = glyphs
            .last()
            .map(|g| g.pixel_bounding_box().unwrap().max.x)
            .unwrap();
        (max_x - min_x) as u32
    };
    println!("glyphs_height={glyphs_height} glyphs_width={glyphs_width}");

    let mut image = DynamicImage::new_rgba8(glyphs_width + 2, glyphs_height + 0).to_rgba8();
    println!("image is {}x{}", image.width(), image.height());

    for px in image.pixels_mut() {
        *px = Rgba([0, 0, 0, 0]);
    }

    let x_scaler = 1.01;

    for glyph in glyphs {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x + bounding_box.min.x as u32;
                let x = (x as f32 * x_scaler) as u32;
                let y = y + bounding_box.min.y as u32;
                image.get_pixel_mut(x, y).blend(&Rgba([color.0, color.1, color.2, (v * 255.0) as u8]));
            });
        }
    }

    image.save("image_example.png").unwrap();
}

