#![feature(iter_array_chunks)]

use std::time::Instant;

use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{DynamicImage, GrayImage, Luma, imageops};
use pathfinder_geometry::rect::{RectF,RectI};
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F,Vector2I};

const DEFAULT_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+(){};:";

//#[derive(Clone, Copy)]
//enum BoxSize {
//    Font,
//    Alphabet,
//    Char
//}
//
//impl TryFrom<&str> for BoxSize {
//    type Error = ();
//    fn try_from(s: &str) -> Result<BoxSize, ()> {
//        match s {
//            "font" => Ok(BoxSize::Font),
//            "alphabet" => Ok(BoxSize::Alphabet),
//            "char" => Ok(BoxSize::Char),
//            _ => Err(())
//        }
//    }
//}
//
//#[derive(Clone, Copy)]
//enum Baseline {
//    AlphabetAscentCeil,
//    FontAscentCeil,
//    LetterAscentCeil,
//}

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
}

#[derive(Debug)]
struct Match {
    rect: RectI,
    mse: f32,
}

struct Searcher {
    reference: Box<[i16]>,
    reference_rot: Box<[i16]>,
    r_w: usize,
    r_h: usize,
    acc: Vec<f32>,
    needle: Vec<i16>,
    matches: Vec<Match>,
}

fn render(font: &Font, char: char, offset: [f32; 2], render_options: RenderOptions, canvas: Option<Canvas>) -> Canvas {
    assert!(render_options.format == Format::A8);

    let glyph_id = font.glyph_for_char(char).unwrap();
    let pos = Vector2F::new(offset[0], offset[1]);

    let to_px = (1. / font.metrics().units_per_em as f32) * render_options.size;

    let bounds = font
        .raster_bounds(
            glyph_id,
            render_options.size,
            Transform2F::from_translation(pos),
            render_options.hinting,
            render_options.rasterization,
        )
        .unwrap();
    let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * to_px;
    let _bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
    let bearing_x = glyph_bounds.origin().x();
    let ascent_px = font.metrics().ascent * to_px;

    let size = if true {
        bounds.size()
    } else {
        let font_bbox = font.metrics().bounding_box * to_px;
        Vector2I::new(
            font_bbox.width().ceil() as i32,
            font_bbox.height().ceil() as i32,
            )
    };

    let mut canvas = match canvas {
        Some(mut canvas) if canvas.size == size && canvas.format == Format::A8 => {
            canvas.pixels.fill(0);
            canvas
        },
        _ => Canvas::new(size, render_options.format),
    };

    font.rasterize_glyph(
        &mut canvas,
        glyph_id,
        render_options.size,
        Transform2F::from_translation(-bounds.to_f32().origin()).translate(pos),
        //Transform2F::from_translation(Vector2F::new(bearing_x.floor(), ascent_px.ceil())).translate(pos),
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
    canvas: Option<Canvas>,
) -> Canvas {
    assert!(render_options.format == Format::A8);
    let units_per_em = font.metrics().units_per_em as f32;

    let mut glyph_pos = Vec::with_capacity(text.len());

    let mut pos = Vector2F::new(offset[0], offset[1]);

    for char in text.chars() {
        let glyph_id = font.glyph_for_char(char).unwrap();
        glyph_pos.push((glyph_id, pos));
        pos += font.advance(glyph_id).unwrap() / units_per_em * render_options.size;
        let bounds = font.typographic_bounds(glyph_id).unwrap() * (1. / units_per_em) * render_options.size;
        let bounds2 = font.typographic_bounds(glyph_id).unwrap();
        eprintln!("{char} {bounds:?} {bounds2:?}");
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
            eprintln!("raster rect {raster_rect:?}");
            bounds.union_rect(raster_rect.to_f32())
        });

    eprintln!("bounds `{}` {:?}", text, bounds);
    let font_bbox = font.metrics().bounding_box * (1. / units_per_em) * render_options.size;
    eprintln!("font_bbox {:?} height={}", font_bbox, font_bbox.height());

    let size = Vector2I::new(
        bounds.width().ceil() as i32,
        //font_bbox.height().ceil() as i32,
        font_bbox.height().ceil() as i32,
        //bounds.height().ceil() as i32,
    );

    let mut canvas = match canvas {
        Some(canvas) if canvas.size == size => { canvas },
        _ => Canvas::new(size, render_options.format),
    };

    for (glyph_id, pos) in glyph_pos {
        let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * (1. / units_per_em) * render_options.size;
        let bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
        eprintln!("bearing_y {bearing_y}");
        //eprintln!("pos={:?}", pos);
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

fn rotate(img: &GrayImage) -> GrayImage {
    let mut ret = GrayImage::new(img.height(), img.width());
    imageops::rotate270_in(img, &mut ret).unwrap();
    ret
}

impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let reference = image_to_i16(img);
        let reference_rot = image_to_i16(&rotate(img));
        let r_w = img.width() as usize;
        let r_h = img.height() as usize;
        let matches = Vec::with_capacity(1024);
        let acc = vec![0f32; img.width() as usize];
        let needle = vec![0; 1024];
        //let needlex2 = vec![[0; 16]; 12];

        Searcher {
            reference: reference.into(),
            reference_rot: reference_rot.into(),
            r_w,
            r_h,
            needle,
            acc,
            matches,
        }
    }

    fn search(&mut self, needle: &GrayImage, threshold: f32) -> (f32, &[Match]) {
        let w = needle.width();
        let h = needle.height();
        if w > 16 || h > 16 {
            self.search_var(needle, threshold)
        } else if w >= h {
            if w <= 8 {
                self.search_n::<8>(needle, threshold, false)
            } else {
                // TODO this isn't generating avx2 for some reason
                self.search_n::<16>(needle, threshold, false)
            }
        } else {
            let needle_rot = rotate(needle);
            if h <= 8 {
                self.search_n::<8>(&needle_rot, threshold, true)
            } else {
                assert!(h <= 16);
                self.search_n::<16>(&needle_rot, threshold, true)
            }
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
                    let upper_left = Vector2I::new(x as i32, y as i32);
                    let rect = RectI::new(upper_left, Vector2I::new(n_w as i32, n_h as i32));
                    self.matches.push(Match { mse, rect });
                }
            }
            self.acc.fill(0.);
        }
        (min, &self.matches)
    }

    #[inline(never)]
    fn search_n<const N: usize>(&mut self, needle: &GrayImage, threshold: f32, rot: bool) -> (f32, &[Match]) {
        assert!(needle.width() <= N as u32);
        self.matches.clear();

        let n_h = needle.height() as usize;
        let n_w = needle.width() as usize;

        let (r_w, r_h) = if rot {
            (self.r_h, self.r_w)
        } else {
            (self.r_w, self.r_h)
        };

        let x_searches = r_w - N + 1;
        let y_searches = r_h - n_h + 1;

        let reference = if rot {
            &self.reference_rot
        } else {
            &self.reference
        };

        self.acc.fill(0.);
        self.acc.resize(x_searches, 0.);

        let divisor = (N as f32) * (n_h as f32);
        //let scaled_threshold = (divisor * threshold).ceil();
        let mut min = f32::MAX;

        //self.needle.fill([255i16; 8]);
        self.needle.resize(N * n_h as usize, 0);
        {
            let (rows, _rem) = self.needle.as_chunks_mut::<N>();
            copy_needle_n(rows, needle);
        }
        //let mut needle_buf = vec![[255i16; N]; n_h];
        //copy_needle_n(&mut needle_buf, needle);
        let mask = get_mask_n(n_w);
        let (needle_buf, _rem) = self.needle.as_chunks::<N>();

        for y in 0..y_searches {
            for (needle_y, needle_row) in needle_buf.iter().enumerate() {
                let ref_windows =
                    get_row(&reference, r_w, y + needle_y).array_windows::<N>();
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
                    let rect = if rot {
                        RectI::new(
                            Vector2I::new((r_h - y - n_h) as i32, x as i32),
                            Vector2I::new(n_h as i32, n_w as i32)
                            )
                    } else {
                        RectI::new(
                            Vector2I::new(x as i32, y as i32),
                            Vector2I::new(n_w as i32, n_h as i32)
                            )
                    };
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

    #[arg(short, long, default_value_t = 1)]
    bits: u32,

    #[arg(long)]
    hinting: bool,

    #[arg(long, default_value_t = 1000.)]
    threshold: f32,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,

    //#[arg(long, default_value="char")]
    //box_size: String,
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
        //box_size: args.box_size.as_str().try_into().unwrap()
    };

    //let offsets = [
    //    [0.0, 0.0],
    //    //[0.5, 0.0],
    //    //[0.0, 0.5],
    //    //[0.5, 0.5],
    //    [0.25, 0.0],
    //    //[0.0, 0.25],
    //    //[0.25, 0.25],
    //    //
    //    [0.125, 0.0],
    //    //[0.0, 0.125],
    //    //[0.125, 0.125],
    //    //
    //    //[0.1, 0.0],
    //    //[0.0, 0.1],
    //    //[0.1, 0.1],
    //
    //    //[-0.5, -0.0],
    //    //[0.0, -0.5],
    //    //[-0.5, -0.5],
    //    //[-0.25, 0.0],
    //    //[-0.0, -0.25],
    //    //[-0.25, -0.25],
    //];
    let offsets = {
        let mut acc = vec![];
        let divisor = 1. / 2usize.pow(args.bits) as f32;
        for x in 0..2usize.pow(args.bits) {
            for y in 0..2usize.pow(args.bits) {
                acc.push([x as f32 * divisor, y as f32 * divisor]);
            }
        }
        acc
    };
    let font = Font::from_path(args.font, 0).unwrap();
    if true {
        let metrics = font.metrics();
        let to_px = (1. / metrics.units_per_em as f32) * args.text_size;
        let font_bbox = metrics.bounding_box * to_px;
        let line_space = metrics.ascent - metrics.descent + metrics.line_gap;
        let line_space_px = line_space * to_px;
        eprintln!("metrics {:?}", metrics);
        eprintln!("ascent  {}px", metrics.ascent * to_px);
        eprintln!("descent {}px", metrics.descent * to_px);
        eprintln!("font_bbox size {:?}px", font_bbox.size());
        eprintln!("line_space {} {}px", line_space, line_space_px);

        for char in args.alphabet.chars() {
            let glyph_id = font.glyph_for_char(char).unwrap();
            let typo_bounds_px = font.typographic_bounds(glyph_id).unwrap() * to_px;
            let advance = (font.advance(glyph_id).unwrap() * to_px).x();
            let bearing_y = typo_bounds_px.origin().y() + typo_bounds_px.height();
            let bearing_x = typo_bounds_px.origin().x();
            eprintln!("`{char}` {typo_bounds_px:?}px advance={advance} bearing_x={bearing_x} bearing_y={bearing_y}");
        }

        let alphabet_bbox = args.alphabet.chars()
            .fold(RectF::default(), |bounds, c| {
                let gid = font.glyph_for_char(c).unwrap();
                let raster_rect = font
                    .raster_bounds(
                        gid,
                        render_options.size,
                        Transform2F::default(),
                        render_options.hinting,
                        render_options.rasterization,
                    )
                    .unwrap();
                bounds.union_rect(raster_rect.to_f32())
            });
        eprintln!("alphabet_bbox size {:?}", alphabet_bbox.size());
    }
    let img = image::open(args.img.first().unwrap()).unwrap().into_luma8();
    let mut searcher = Searcher::new(&img);
    let mut total_mse = 0.;
    let mut n_hits = 0;

    // this shows all offset are unique
    //if false {
    //    println!("({offsets:#?})");
    //
    //    for letter in args.alphabet.chars() {
    //        let mut set = std::collections::HashSet::new();
    //        for offset in &offsets {
    //            let canvas = render(&font, letter, *offset, render_options, None);
    //            set.insert(canvas.pixels);
    //        }
    //        println!("`{letter}` {}/{}", set.len(), offsets.len());
    //    }
    //    return;
    //}
    //
    //if false {
    //    let mut s = String::new();
    //    for c1 in args.alphabet.chars() {
    //        for c2 in args.alphabet.chars() {
    //            s.push(c1);
    //            s.push(c2);
    //            let canvas = render_string(&font, &s, [0., 0.], render_options, None);
    //            s.clear();
    //        }
    //    }
    //    return;
    //}

    let t00 = Instant::now();

    let mut canvas_cache = None;
    for letter in args.alphabet.chars() {
    //for word in ["J"] {
        for offset in &offsets {
            let canvas = render(&font, letter, *offset, render_options, canvas_cache.take());
            //let canvas = render_string(&font, word, offset, render_options);
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
            let t1 = Instant::now();
            let _ = canvas_cache.insert(canvas);
            eprintln!("`{letter}` {offset:?} needle size {}x{} min mse {} elapsed {}ms", needle.width(), needle.height(), min, (t1 - t0).as_millis());
            //eprintln!(
            //    "`{word}` {offset:?} needle size {}x{} min mse {} elapsed: {}ms",
            //    needle.width(),
            //    needle.height(),
            //    min,
            //    (t1 - t0).as_millis()
            //);
            n_hits += hits.len();

            let glyph_id = font.glyph_for_char(letter).unwrap();
            let units_per_em = font.metrics().units_per_em as f32;
            let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * (1. / units_per_em) * render_options.size;
            let bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
            let bearing_x = glyph_bounds.origin().x();
            let ascent_px = font.metrics().ascent * (1. / units_per_em) * render_options.size;
            for hit in hits {
                total_mse += hit.mse;
                //println!("{hit:?}");
                //println!("{},{}", hit.x + offset[0], hit.y + offset[1]);
                let ul = hit.rect.origin();
                let pt = hit.rect.to_f32().center();
                println!(
                    "{},{},{},{},{},{},{},{},{},{}",
                    pt.x(),
                    pt.y(),
                    ul.x(),
                    ul.y(),
                    hit.rect.width(),
                    hit.rect.height(),
                    bearing_x,
                    ascent_px,
                    //bearing_y,
                    offset[0],
                    offset[1],
                );
                //for offset in [1., -1.] {
                //    println!(
                //        "{},{},{},{},{},{}",
                //        pt.x(),
                //        pt.y() + line_space_px,
                //        ul.x(),
                //        ul.y() as f32 + offset * line_space_px.floor(),
                //        hit.rect.width(),
                //        hit.rect.height()
                //    );
                //}
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

fn sqerror_n<const N: usize>(a: [i16; N], b: [i16; N], mask: [i16; N]) -> u32 {
    let mut ret = 0u32;
    for i in 0..N {
        ret += (((a[i] - b[i]) * mask[i]) as i32).pow(2) as u32;
    }
    ret
}
