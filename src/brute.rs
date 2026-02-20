use std::time::Instant;

use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::Vector2F;
use image::{GrayImage, Luma};
use clap::Parser;

const DEFAULT_ALPHABET: &str =
    ">=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/;:-";

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

//struct SimpleImage {
//    pixels: Box<[i16]>,
//    w: usize,
//    h: usize,
//}
//
//fn canvas_to_i16(canvas: &Canvas) -> SimpleImage {
//    assert!(canvas.format == Format::A8);
//    let w = canvas.size.x() as usize;
//    let h = canvas.size.y() as usize;
//    let pixels = canvas.pixels.iter().map(|p| p as i16).collect();
//    SimpleImage {w, h, pixels}
//}

#[derive(Debug)]
struct Match {
    x: f32,
    y: f32,
    mse: f32,
}

fn copy_needle8(out: &mut [[i16; 8]], img: &GrayImage) {
    for (x, y, px) in img.enumerate_pixels() {
        out[y as usize][x as usize] = px[0] as i16;
    }
}

fn get_8(arr: &[u8], w: usize, x: usize, y: usize) -> [i16; 8] {
    let mut ret = [0i16; 8];
    for i in 0..8 {
        ret[i] = arr[w * y + x + i] as i16;
    }
    ret
}

//fn get_8_shifted1(arr: &[u8], w: usize, x: usize, y: usize) -> [i16; 8] {
//    let mut ret = [0i16; 8];
//    for i in 1..8 {
//        ret[i] = arr[w * y + x + i] as i16;
//    }
//    ret
//}

//fn shift_in_8(arr: [i16; 8], x: u8) -> [i16; 8] {
//    let mut ret = [0i16; 8];
//    for i in 0..7 {
//        ret[i] = arr[i + 1];
//    }
//    ret[7] = x as i16;
//    ret
//}

fn get_row(arr: &[i16], w: usize, y: usize) -> &[i16] {
    &arr[y*w..(y+1)*w]
}

struct Searcher {
    reference: Box<[i16]>,
    r_w: usize,
    r_h: usize,
    acc: Vec<u32>,
    needle: Vec<[i16; 8]>,
    matches: Vec<Match>
}

impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let reference = image_to_i16(img);
        let r_w = img.width() as usize;
        let r_h = img.height() as usize;
        let matches = Vec::with_capacity(1024);
        let acc = vec![0u32; img.width() as usize];
        let needle = vec![[0; 8]; 12];

        Searcher {
            reference: reference.into(),
            r_w,
            r_h,
            needle,
            acc: acc.into(),
            matches,
        }
    }

    #[inline(never)]
    fn search(&mut self, needle: &GrayImage, threshold: f32) -> &[Match] {
        self.matches.clear();

        let n_h = needle.height() as usize;
        let n_w = needle.width() as usize;

        let x_searches = self.r_w - n_w + 1;
        let y_searches = self.r_h - n_h + 1;

        self.acc.resize(x_searches, 0);
        self.acc.fill(0);

        let divisor = (needle.width() as f32) * (needle.height() as f32);
        let half_width = (needle.width() as f32) / 2.;
        let half_height = (needle.height() as f32) / 2.;
        let mut min = f32::MAX;

        let needle_raw = needle.as_raw();

        for y in 0..y_searches {
            for needle_y in 0..n_h {
                for (x, acc) in self.acc.iter_mut().enumerate() {
                    for needle_x in 0..n_w {
                        //let npx = unsafe { needle_raw.get_unchecked(needle_y * n_w + needle_x) };
                        //let rpx = unsafe { self.reference.get_unchecked((y + needle_y) * self.r_w + needle_x + x) };
                        let npx = needle_raw.get(needle_y * n_w + needle_x).unwrap();
                        let rpx = self.reference.get((y + needle_y) * self.r_w + needle_x + x).unwrap();
                        *acc += ((*npx as i16 - *rpx) as i32).pow(2) as u32;
                    }
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let mse = *acc as f32 / divisor;
                if mse < min {
                    min = mse;
                }
                if mse < threshold {
                    let x = (x as f32) + half_width;
                    let y = (y as f32) + half_height;
                    self.matches.push(Match{x, y, mse}); // TODO make into centered coords
                }
            }
            self.acc.fill(0);
        }
        eprintln!("got min of {min}");
        &self.matches
    }

    #[inline(never)]
    fn search8(&mut self, needle: &GrayImage, threshold: f32) -> &[Match] {
        assert!(needle.width() <= 8);
        self.matches.clear();

        let n_h = needle.height() as usize;
        let n_w = needle.width() as usize;

        let x_searches = self.r_w - 8 + 1;
        let y_searches = self.r_h - n_h + 1;

        self.acc.resize(x_searches, 0);
        self.acc.fill(0);

        let divisor = 8. * (n_h as f32);
        let half_width = 8. / 2.;
        let half_height = (n_h as f32) / 2.;
        let mut min = f32::MAX;

        self.needle.fill([255i16; 8]);
        self.needle.resize(n_h as usize, [255i16; 8]);
        copy_needle8(&mut self.needle, needle);

        for y in 0..y_searches {
            for (_needle_y, needle_row) in self.needle.iter().enumerate() {
                let ref_windows = get_row(&self.reference, self.r_w, y).array_windows::<8>();
                for ((x, acc), ref_row) in self.acc.iter_mut().enumerate().zip(ref_windows) {
                    *acc += se_i168(*needle_row, *ref_row);
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let mse = *acc as f32 / divisor;
                if mse < min {
                    min = mse;
                }
                if mse < threshold {
                    let x = (x as f32) + half_width;
                    let y = (y as f32) + half_height;
                    self.matches.push(Match{x, y, mse});
                }
            }
            self.acc.fill(0);
        }
        eprintln!("got min of {min}");
        &self.matches
    }

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

    #[arg(long, default_value_t=1000.)]
    threshold: f32,

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
    let mut searcher = Searcher::new(&img);
    let t00 = Instant::now();

    for letter in args.alphabet.chars() {
        for offset in offsets {

            let needle = canvas_to_lum8(&render(&font, letter, offset, render_options));
            eprintln!("`{letter}` {offset:?} needle size {}x{}", needle.width(), needle.height());
            let t0 = Instant::now();
            let hits = if needle.width() <= 8 {
                searcher.search8(&needle, args.threshold)
            } else {
                searcher.search(&needle, args.threshold)
            };
            let t1 = Instant::now();
            for hit in hits {
                //println!("{hit:?}");
                //println!("{},{}", hit.x + offset[0], hit.y + offset[1]);
                println!("{},{}", hit.x, hit.y);
            }
            eprintln!("took {:.4}ms", (t1 - t0).as_millis());
        }
    }
    let t11 = Instant::now();
    eprintln!("overall {:.4}ms", (t11 - t00).as_millis());
}

fn image_to_i16(img: &GrayImage) -> Vec<i16> {
    img.as_raw().iter().map(|x| *x as i16).collect()
}

fn se(a: Luma<u8>, b: Luma<u8>) -> u32 {
    ((a[0] as i8 - b[0] as i8) as i32).pow(2) as u32
}

fn se_u8(a: u8, b: u8) -> u32 {
    ((a as i8 - b as i8) as i32).pow(2) as u32
}

fn se_i168(a: [i16; 8], b: [i16; 8]) -> u32 {
    let mut ret = 0u32;
    for i in 0..8 {
        ret += ((a[i] - b[i]) as i32).pow(2) as u32
    }
    ret
}

