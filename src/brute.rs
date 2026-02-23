#![feature(iter_array_chunks)]

use std::time::Instant;

use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{DynamicImage, GrayImage, Luma};
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::Vector2F;

const DEFAULT_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
}

#[derive(Debug)]
struct Match {
    rect: RectF,
    mse: f32,
}

struct Searcher {
    reference: Box<[i16]>,
    r_w: usize,
    r_h: usize,
    acc: Vec<f32>,
    //needle8: Vec<[i16; 8]>,
    //needle16: Vec<[i16; 16]>,
    //needlex2: Vec<[i16; 16]>,
    matches: Vec<Match>,
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
        )
        .unwrap();

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

fn render_string(
    font: &Font,
    text: &str,
    offset: [f32; 2],
    render_options: RenderOptions,
) -> Canvas {
    assert!(render_options.format == Format::A8);
    let units_per_em = font.metrics().units_per_em as f32;

    let mut glyph_pos = Vec::with_capacity(text.len());

    let mut pos = Vector2F::new(offset[0], offset[1]);

    for char in text.chars() {
        let glyph_id = font.glyph_for_char(char).unwrap();
        glyph_pos.push((glyph_id, pos));
        pos += font.advance(glyph_id).unwrap() / units_per_em * render_options.size;
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

fn copy_needle_n<const N: usize>(out: &mut [[i16; N]], img: &GrayImage) {
    for (x, y, px) in img.enumerate_pixels() {
        out[y as usize][x as usize] = px[0] as i16;
    }
}

//fn copy_needle8x2(out: &mut [[i16; 16]], img: &GrayImage) {
//    for (x, y, px) in img.enumerate_pixels() {
//        out[y as usize][x as usize] = px[0] as i16;
//        out[y as usize][x as usize + 8] = px[0] as i16;
//    }
//}

fn get_row(arr: &[i16], w: usize, y: usize) -> &[i16] {
    &arr[y * w..(y + 1) * w]
}

fn get_mask_n<const N: usize>(n: usize) -> [i16; N] {
    //const MASKS: [[i16; 8]; 8] = [
    //    [1, 0, 0, 0, 0, 0, 0, 0],
    //    [1, 1, 0, 0, 0, 0, 0, 0],
    //    [1, 1, 1, 0, 0, 0, 0, 0],
    //    [1, 1, 1, 1, 0, 0, 0, 0],
    //    [1, 1, 1, 1, 1, 0, 0, 0],
    //    [1, 1, 1, 1, 1, 1, 0, 0],
    //    [1, 1, 1, 1, 1, 1, 1, 0],
    //    [1, 1, 1, 1, 1, 1, 1, 1],
    //];
    assert!(n >= 1 && n <= N);
    //MASKS[n - 1]
    let mut arr = [0i16; N];
    for i in 0..n {
        arr[i] = 1;
    }
    arr
}

impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let reference = image_to_i16(img);
        let r_w = img.width() as usize;
        let r_h = img.height() as usize;
        let matches = Vec::with_capacity(1024);
        let acc = vec![0f32; img.width() as usize];
        //let needle = vec![[0; 8]; 12];
        //let needlex2 = vec![[0; 16]; 12];

        Searcher {
            reference: reference.into(),
            r_w,
            r_h,
            //needle,
            //needlex2,
            acc,
            matches,
        }
    }

    fn search(&mut self, needle: &GrayImage, threshold: f32) -> (f32, &[Match]) {
        let w = needle.width();
        if w <= 8 {
            self.search_n::<8>(needle, threshold)
        } else if w <= 16 {
            // TODO this isn't generating avx2 for some reason
            self.search_n::<16>(needle, threshold)
        } else {
            self.search_var(needle, threshold)
        }
    }

    #[inline(never)]
    fn search_var(&mut self, needle: &GrayImage, threshold: f32) -> (f32, &[Match]) {
        self.matches.clear();

        let n_h = needle.height() as usize;
        let n_w = needle.width() as usize;

        let x_searches = self.r_w - n_w + 1;
        let y_searches = self.r_h - n_h + 1;

        self.acc.fill(0.);
        self.acc.resize(x_searches, 0.);

        let divisor = (needle.width() as f32) * (needle.height() as f32);
        let mut min = f32::MAX;

        let needle_raw = needle.as_raw();

        for y in 0..y_searches {
            for needle_y in 0..n_h {
                for (x, acc) in self.acc.iter_mut().enumerate() {
                    for needle_x in 0..n_w {
                        //let npx = unsafe { needle_raw.get_unchecked(needle_y * n_w + needle_x) };
                        //let rpx = unsafe { self.reference.get_unchecked((y + needle_y) * self.r_w + needle_x + x) };
                        let npx = needle_raw.get(needle_y * n_w + needle_x).unwrap();
                        let rpx = self
                            .reference
                            .get((y + needle_y) * self.r_w + needle_x + x)
                            .unwrap();
                        *acc += ((*npx as i16 - *rpx) as i32).pow(2) as f32;
                    }
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let mse = *acc as f32 / divisor;
                min = f32::min(mse, min);
                if mse < threshold {
                    let upper_left = Vector2F::new(x as f32, y as f32);
                    let rect = RectF::new(upper_left, Vector2F::new(n_w as f32, n_h as f32));
                    self.matches.push(Match { mse, rect });
                }
            }
            self.acc.fill(0.);
        }
        (min, &self.matches)
    }

    #[inline(never)]
    fn search_n<const N: usize>(&mut self, needle: &GrayImage, threshold: f32) -> (f32, &[Match]) {
        assert!(needle.width() <= N as u32);
        self.matches.clear();

        let n_h = needle.height() as usize;
        let n_w = needle.width() as usize;

        let x_searches = self.r_w - N + 1;
        let y_searches = self.r_h - n_h + 1;

        self.acc.fill(0.);
        self.acc.resize(x_searches, 0.);

        let divisor = (N as f32) * (n_h as f32);
        //let scaled_threshold = (divisor * threshold).ceil();
        let mut min = f32::MAX;

        //self.needle.fill([255i16; 8]);
        //self.needle.resize(n_h as usize, [255i16; 8]);
        //copy_needle8(&mut self.needle, needle);
        let mut needle_buf = vec![[255i16; N]; n_h];
        copy_needle_n(&mut needle_buf, needle);
        let mask = get_mask_n(n_w);

        for y in 0..y_searches {
            for (needle_y, needle_row) in needle_buf.iter().enumerate() {
                let ref_windows =
                    get_row(&self.reference, self.r_w, y + needle_y).array_windows::<N>();
                for (acc, ref_row) in self.acc.iter_mut().zip(ref_windows) {
                    // trying early termination, but not that helpful
                    //if *acc < scaled_threshold {
                    *acc += sqerror_n(*needle_row, *ref_row, mask) as f32;
                    //}
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let mse = *acc as f32 / divisor;
                min = f32::min(mse, min);
                if mse < threshold {
                    let upper_left = Vector2F::new(x as f32, y as f32);
                    let rect = RectF::new(upper_left, Vector2F::new(n_w as f32, n_h as f32));
                    self.matches.push(Match { mse, rect });
                }
            }
            self.acc.fill(0.);
        }
        //eprintln!("got min of {min}");
        (min, &self.matches)
    }
    //#[inline(never)]
    //fn search8x2(&mut self, needle: &GrayImage, threshold: f32) -> (f32, &[Match]) {
    //    assert!(needle.width() <= 8);
    //    self.matches.clear();
    //
    //    let n_h = needle.height() as usize;
    //    let n_w = needle.width() as usize;
    //
    //    let x_searches = self.r_w - 8 + 1;
    //    let y_searches = self.r_h - n_h + 1;
    //
    //    self.acc.fill(0.);
    //    self.acc.resize(x_searches, 0.);
    //
    //    let divisor = 8. * (n_h as f32);
    //    let mut min = f32::MAX;
    //
    //    self.needlex2.fill([255i16; 16]);
    //    self.needlex2.resize(n_h as usize, [255i16; 16]);
    //    copy_needle8x2(&mut self.needlex2, needle);
    //
    //    let (acc_chunks, rem) = self.acc.as_chunks_mut::<2>();
    //    if rem.is_empty() {
    //        eprintln!("WARN x2 rem not empty");
    //    }
    //
    //    for y in 0..y_searches {
    //        for (needle_y, needle_row) in self.needlex2.iter().enumerate() {
    //            let ref_windows = get_row(&self.reference, self.r_w, y + needle_y).array_windows::<16>().array_chunks::<8>().step_by(2);
    //            for ((_x, acc), ref_row) in acc_chunks.iter_mut().enumerate().zip(ref_windows) {
    //                let (a0, a1) = se_i168x2(*needle_row, *ref_row);
    //                acc[0] += a0;
    //                acc[1] += a1;
    //            }
    //        }
    //        for (x, acc) in acc_chunks.as_flattened().iter().enumerate() {
    //            let mse = *acc as f32 / divisor;
    //            min = f32::min(mse, min);
    //            if mse < threshold {
    //                let upper_left = Vector2F::new(x as f32, y as f32);
    //                let rect = RectF::new(upper_left, Vector2F::new(n_w as f32, n_h as f32));
    //                self.matches.push(Match{mse, rect});
    //            }
    //        }
    //        acc_chunks.fill([0., 0.]);
    //    }
    //    //eprintln!("got min of {min}");
    //    (min, &self.matches)
    //}
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

    #[arg(long, default_value_t = 1000.)]
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
        //[0.0, 0.5],
        //[0.5, 0.5],
        [0.25, 0.0],
        //[0.0, 0.25],
        //[0.25, 0.25],
        //
        [0.125, 0.0],
        //[0.0, 0.125],
        //[0.125, 0.125],
        //
        //[0.1, 0.0],
        //[0.0, 0.1],
        //[0.1, 0.1],

        //[-0.5, -0.0],
        //[0.0, -0.5],
        //[-0.5, -0.5],
        //[-0.25, 0.0],
        //[-0.0, -0.25],
        //[-0.25, -0.25],
    ];

    let font = Font::from_path(args.font, 0).unwrap();
    let metrics = font.metrics();
    eprintln!("metrics {:?}", metrics);
    let linespace = metrics.ascent - metrics.descent + metrics.line_gap;
    eprintln!("linescape {} {}px", linespace, linespace / 96.);
    let img = image::open(args.img.first().unwrap()).unwrap().into_luma8();
    let mut searcher = Searcher::new(&img);
    let mut total_mse = 0.;
    let mut n_hits = 0;

    if false {
        let mut s = String::new();
        for c1 in args.alphabet.chars() {
            for c2 in args.alphabet.chars() {
                s.push(c1);
                s.push(c2);
                let canvas = render_string(&font, &s, [0., 0.], render_options);
                s.clear();
            }
        }
        return;
    }

    let t00 = Instant::now();

    //for letter in args.alphabet.chars() {
    for word in ["RJ", "JY"] {
        for offset in offsets {
            //let canvas = render(&font, letter, offset, render_options);
            let canvas = render_string(&font, word, offset, render_options);
            let needle = canvas_to_lum8(&canvas);
            //if false {
            //    let x = (offset[0] * 1000.) as usize;
            //    let y = (offset[1] * 1000.) as usize;
            //    DynamicImage::ImageLuma8(needle.clone()).save(format!("letters/{letter}-{x}_{y}.png")).unwrap();
            //}
            let t0 = Instant::now();
            let (min, hits) = searcher.search(&needle, args.threshold);
            //let (min, hits) = searcher.search(&needle, args.threshold);
            //let (min, hits) = searcher.search8(&needle, args.threshold);
            //let (min, hits) = searcher.search8x2(&needle, args.threshold);
            //eprintln!("`{letter}` {offset:?} needle size {}x{} min mse {}", needle.width(), needle.height(), min);
            let t1 = Instant::now();
            eprintln!(
                "`{word}` {offset:?} needle size {}x{} min mse {} elapsed: {}ms",
                needle.width(),
                needle.height(),
                min,
                (t1 - t0).as_millis()
            );
            n_hits += hits.len();
            for hit in hits {
                total_mse += hit.mse;
                //println!("{hit:?}");
                //println!("{},{}", hit.x + offset[0], hit.y + offset[1]);
                let ul = hit.rect.origin();
                let pt = hit.rect.center();
                println!(
                    "{},{},{},{},{},{}",
                    pt.x(),
                    pt.y(),
                    ul.x(),
                    ul.y(),
                    hit.rect.width(),
                    hit.rect.height()
                );
                //eprintln!("HIT {},{} offset {:?}", pt.x(), pt.y(), offset);
            }
            //eprintln!("took {:.4}ms", (t1 - t0).as_millis());
        }
    }
    let t11 = Instant::now();
    eprintln!("overall {:.4}ms", (t11 - t00).as_millis());
    let avg_mse = total_mse / n_hits as f32;
    eprintln!("hits: {n_hits} mse per hit:{avg_mse}");
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

fn sqerror_n<const N: usize>(a: [i16; N], b: [i16; N], mask: [i16; N]) -> u32 {
    let mut ret = 0u32;
    for i in 0..N {
        ret += (((a[i] - b[i]) * mask[i]) as i32).pow(2) as u32;
    }
    ret
}

//fn se_i168x2(a: [i16; 16], b: [i16; 16]) -> (f32, f32) {
//    let mut ra = 0u32;
//    let mut rb = 0u32;
//    for i in 0..8 {
//        ra += ((a[i] - b[i]) as i32).pow(2) as u32;
//        rb += ((a[i + 8] - b[i + 8]) as i32).pow(2) as u32;
//    }
//    (ra as f32, rb as f32)
//}
