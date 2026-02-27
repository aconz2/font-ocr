#![feature(iter_array_chunks)]

use std::time::Instant;
use std::collections::HashMap;

use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{GrayImage, imageops};
use pathfinder_geometry::rect::{RectF, RectI};
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};

const DEFAULT_ALPHABET: &str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+(){};:/_-";
//"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

#[derive(Clone, Copy)]
enum BoxSize {
    Font,
    Alphabet,
    Char,
}

impl TryFrom<&str> for BoxSize {
    type Error = ();
    fn try_from(s: &str) -> Result<BoxSize, ()> {
        match s {
            "font" => Ok(BoxSize::Font),
            "alphabet" => Ok(BoxSize::Alphabet),
            "char" => Ok(BoxSize::Char),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
}

#[derive(Clone, Copy, Debug)]
struct Match {
    rect: RectI,
    similarity: AccType,
}

#[derive(Clone, Copy, Debug)]
struct MatchWithBaseline {
    rect: RectI,
    similarity: AccType,
    baseline: f32,
}

// seeing a large speedup using u32 over u64 for accumulating the sum of squared errors
// and as the threshold type
// the max total error we can get is W*H*255^2, for up to say 16x16, that fits in 24 bits
type AccType = f32;

struct Searcher {
    reference: Box<[f32]>,
    r_w: usize,
    r_h: usize,
    acc: Vec<AccType>,
    needle: Vec<f32>,
    matches: Vec<Match>,
}

fn render(
    font: &Font,
    char: char,
    offset: [f32; 2],
    render_options: RenderOptions,
    canvas_size: Option<Vector2I>,
    canvas: Option<Canvas>,
    padding: Vector2I,
) -> Canvas {
    assert!(render_options.format == Format::A8);

    let glyph_id = font.glyph_for_char(char).unwrap();
    let pos = Vector2F::new(offset[0], offset[1]);

    let raster_bounds = font
        .raster_bounds(
            glyph_id,
            render_options.size,
            Transform2F::from_translation(pos),
            render_options.hinting,
            render_options.rasterization,
        )
        .unwrap();

    let size = canvas_size.unwrap_or_else(|| raster_bounds.size()) + padding * 2;

    let origin = if canvas_size.is_some() {
        // BoxSize::Font | Alphabet
        Vector2F::new(0., 0.)
    } else {
        //let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * to_px;
        //let bearing_x = glyph_bounds.origin().x();
        //let bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
        //bearing_y.ceil()
        -raster_bounds.origin().to_f32()
    };

    let mut canvas = match canvas {
        Some(mut canvas) if canvas.size == size && canvas.format == Format::A8 => {
            canvas.pixels.fill(0);
            canvas
        }
        _ => Canvas::new(size, render_options.format),
    };

    font.rasterize_glyph(
        &mut canvas,
        glyph_id,
        render_options.size,
        Transform2F::from_translation(origin)
            .translate(padding.to_f32())
            .translate(pos),
        render_options.hinting,
        render_options.rasterization,
    )
    .unwrap();
    canvas
}

//fn render_string(
//    font: &Font,
//    text: &str,
//    offset: [f32; 2],
//    render_options: RenderOptions,
//    canvas: Option<Canvas>,
//) -> Canvas {
//    assert!(render_options.format == Format::A8);
//    let units_per_em = font.metrics().units_per_em as f32;
//
//    let mut glyph_pos = Vec::with_capacity(text.len());
//
//    let mut pos = Vector2F::new(offset[0], offset[1]);
//
//    for char in text.chars() {
//        let glyph_id = font.glyph_for_char(char).unwrap();
//        glyph_pos.push((glyph_id, pos));
//        pos += font.advance(glyph_id).unwrap() / units_per_em * render_options.size;
//        let bounds = font.typographic_bounds(glyph_id).unwrap() * (1. / units_per_em) * render_options.size;
//        let bounds2 = font.typographic_bounds(glyph_id).unwrap();
//        eprintln!("{char} {bounds:?} {bounds2:?}");
//    }
//
//    let bounds = glyph_pos
//        .iter()
//        .fold(RectF::default(), |bounds, (glyph_id, pos)| {
//            let raster_rect = font
//                .raster_bounds(
//                    *glyph_id,
//                    render_options.size,
//                    Transform2F::from_translation(*pos),
//                    render_options.hinting,
//                    render_options.rasterization,
//                )
//                .unwrap();
//            eprintln!("raster rect {raster_rect:?}");
//            bounds.union_rect(raster_rect.to_f32())
//        });
//
//    eprintln!("bounds `{}` {:?}", text, bounds);
//    let font_bbox = font.metrics().bounding_box * (1. / units_per_em) * render_options.size;
//    eprintln!("font_bbox {:?} height={}", font_bbox, font_bbox.height());
//
//    let size = Vector2I::new(
//        bounds.width().ceil() as i32,
//        //font_bbox.height().ceil() as i32,
//        font_bbox.height().ceil() as i32,
//        //bounds.height().ceil() as i32,
//    );
//
//    let mut canvas = match canvas {
//        Some(canvas) if canvas.size == size => { canvas },
//        _ => Canvas::new(size, render_options.format),
//    };
//
//    for (glyph_id, pos) in glyph_pos {
//        let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * (1. / units_per_em) * render_options.size;
//        let bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
//        eprintln!("bearing_y {bearing_y}");
//        //eprintln!("pos={:?}", pos);
//        font.rasterize_glyph(
//            &mut canvas,
//            glyph_id,
//            render_options.size,
//            Transform2F::from_translation(-bounds.origin()).translate(pos),
//            render_options.hinting,
//            render_options.rasterization,
//        )
//        .unwrap();
//    }
//    canvas
//}

fn canvas_to_lum8(canvas: &Canvas) -> GrayImage {
    assert!(canvas.format == Format::A8);
    let w = canvas.size.x() as u32;
    let h = canvas.size.y() as u32;
    let pixels = canvas.pixels.clone();
    GrayImage::from_raw(w, h, pixels).unwrap()
}

fn copy_needle_n<const N: usize>(out: &mut [[f32; N]], needle: &[f32], size: Vector2I) {
    let w = size.x() as usize;
    for y in 0..(size.y() as usize) {
        for x in 0..(size.x() as usize) {
            out[y][x] = needle[y * w + x];
        }
    }
}

fn get_row<T>(arr: &[T], w: usize, y: usize) -> &[T] {
    &arr[y * w..(y + 1) * w]
}

fn get_mask_n<const N: usize>(n: usize) -> [f32; N] {
    assert!(n >= 1 && n <= N);
    let mut arr = [0f32; N];
    for i in 0..n {
        arr[i] = 1.;
    }
    arr
}

impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let mut reference = image_to_f32(img);
        normalize_image(&mut reference);

        let norm = reference.iter().fold(0f64, |acc, x| acc + (*x as f64) * (*x as f64)).sqrt();
        let r_w = img.width() as usize;
        let r_h = img.height() as usize;
        let matches = Vec::with_capacity(1024);
        let acc = vec![0.; img.width() as usize];
        let needle = vec![0.; 1024];

        Searcher {
            reference: reference.into(),
            r_w,
            r_h,
            needle,
            acc,
            matches,
        }
    }

    fn search(&mut self, needle: &[f32], size: Vector2I, threshold: AccType) -> &[Match] {
        let w = size.x();
        if w <= 8 {
            self.search_n::<8>(needle, size, threshold)
        } else if w <= 16 {
            // TODO this isn't generating avx2 for some reason
            self.search_n::<16>(needle, size, threshold)
        } else {
            todo!()
        }
    }

    #[inline(never)]
    fn search_n<const N: usize>(
        &mut self,
        needle: &[f32],
        size: Vector2I,
        threshold: AccType,
    ) -> &[Match] {
        assert!(size.x() <= N as i32);
        self.matches.clear();

        let n_h = size.y() as usize;
        let n_w = size.x() as usize;

        let (r_w, r_h) = (self.r_w, self.r_h);

        let x_searches = r_w - N + 1;
        let y_searches = r_h - n_h + 1;

        let mut min_sim = 1.;
        let mut max_sim = -1.;
        let divisor = 1. / (n_w as f32 * n_h as f32);

        self.acc.fill(0.);
        self.acc.resize(x_searches, 0.);

        self.needle.resize(N * n_h as usize, 0.);
        {
            let (rows, _rem) = self.needle.as_chunks_mut::<N>();
            copy_needle_n(rows, needle, size);
        }
        let mask = get_mask_n(n_w);
        let (needle_buf, _rem) = self.needle.as_chunks::<N>();

        for y in 0..y_searches {
            for (needle_y, needle_row) in needle_buf.iter().enumerate() {
                let ref_windows = get_row(&self.reference, r_w, y + needle_y).array_windows::<N>();
                for (acc, ref_row) in self.acc.iter_mut().zip(ref_windows) {
                    *acc += cross_corr_n(*needle_row, *ref_row, mask) as AccType;
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let similarity = *acc * divisor;
                max_sim = f32::max(max_sim, similarity);
                min_sim = f32::min(max_sim, similarity);
                if similarity > threshold {
                    let rect = RectI::new(
                            Vector2I::new(x as i32, y as i32),
                            Vector2I::new(n_w as i32, n_h as i32),
                        );
                    self.matches.push(Match { similarity, rect });
                }
            }
            self.acc.fill(0.);
        }
        eprintln!("max sim {max_sim} min sim {min_sim}");
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

    #[arg(long, default_value_t = 0)]
    x_bits: u32,

    #[arg(long, default_value_t = 0)]
    y_bits: u32,

    #[arg(long)]
    hinting: bool,

    // this threshold value is the non-squared distance in pixels for each letter, excluding
    // background pixels. this keeps a `-` from matching whitespace because otherwise the large
    // majority of white matching brings the mean or sqrt/euclidean distance so low
    #[arg(long, default_value_t = 0.95)]
    threshold: f32,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,

    #[arg(long, default_value = "char")]
    box_size: String,

    #[arg(long, default_value_t = 0)]
    padding_x: usize,

    #[arg(long, default_value_t = 0)]
    padding_y: usize,

    #[arg(long)]
    count_unique: bool,

    #[arg(long)]
    save_letters: bool,
}

fn main() {
    let args = Args::parse();

    let padding = Vector2I::new(args.padding_x as i32, args.padding_y as i32);

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
    let box_size = args.box_size.as_str().try_into().unwrap();

    let offsets = {
        let mut acc = vec![];
        let x_divisor = 1. / 2usize.pow(args.x_bits) as f32;
        let y_divisor = 1. / 2usize.pow(args.y_bits) as f32;
        for x in 0..2usize.pow(args.x_bits) {
            for y in 0..2usize.pow(args.y_bits) {
                acc.push([x as f32 * x_divisor, y as f32 * y_divisor]);
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
            eprintln!(
                "`{char}` {typo_bounds_px:?}px advance={advance} bearing_x={bearing_x} bearing_y={bearing_y}"
            );
        }

        let alphabet_bbox = args.alphabet.chars().fold(RectF::default(), |bounds, c| {
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
    //let mut total_error = 0;
    let mut n_hits = 0;

    let to_px = (1. / font.metrics().units_per_em as f32) * render_options.size;

    let t00 = Instant::now();

    let mut unique_rows = std::collections::HashSet::new();
    let mut total_rows = 0;

    let mut hits_by_char: HashMap<_, _> = args.alphabet.chars().map(|c| (c, 0)).collect();
    let mut all_hits = Vec::with_capacity(4096);
    let mut canvas_cache = None;
    for offset in &offsets {
        let (y_offset, canvas_size) = match box_size {
            BoxSize::Font => {
                // TODO this doesn't take into account the bbox
                let bbox = font.metrics().bounding_box * to_px;
                //let size = Vector2I::new(
                //    font_bbox.width().ceil() as i32,
                //    font_bbox.height().ceil() as i32,
                //    );
                let size = bbox.round_out().to_i32().size();
                let y_offset = (font.metrics().ascent * to_px).ceil();
                (y_offset, Some(size))
            }
            BoxSize::Alphabet => {
                let (y_offset, bbox) =
                    args.alphabet
                        .chars()
                        .fold((0., RectF::default()), |(y_offset, bounds), c| {
                            let gid = font.glyph_for_char(c).unwrap();
                            let glyph_bounds = font.typographic_bounds(gid).unwrap() * to_px;
                            let bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
                            let raster_rect = font
                                .raster_bounds(
                                    gid,
                                    render_options.size,
                                    Transform2F::from_translation(Vector2F::new(
                                        offset[0], offset[1],
                                    )),
                                    render_options.hinting,
                                    render_options.rasterization,
                                )
                                .unwrap();
                            (
                                f32::max(y_offset, bearing_y.ceil()),
                                bounds.union_rect(raster_rect.to_f32()),
                            )
                        });
                let size = bbox.round_out().to_i32().size();
                (y_offset, Some(size))
            }
            _ => (0., None),
        };
        let corrected_offset = [offset[0], offset[1] + y_offset];
        for letter in args.alphabet.chars() {
            let t0 = Instant::now();
            let canvas = render(
                &font,
                letter,
                corrected_offset,
                render_options,
                canvas_size,
                canvas_cache.take(),
                padding,
            );
            let mut needle = canvas_to_f32(&canvas);
            normalize_image(&mut needle);
            // canvas has black as background, count # of pixels which actually have some letter in
            // them
            let size = canvas.size;
            if args.count_unique {
                total_rows += size.y() as usize;
                for y in 0..size.y() as usize {
                    let row =
                        canvas.pixels[y * size.x() as usize..(y + 1) * size.x() as usize].to_vec();
                    unique_rows.insert(row);
                }
            }
            if args.save_letters {
                let x = (offset[0] * 1000.) as usize;
                let y = (offset[1] * 1000.) as usize;
                let im = canvas_to_lum8(&canvas);
                image::DynamicImage::ImageLuma8(im).save(format!("letters/{letter}-{x}_{y}.png")).unwrap();
            }
            let hits = searcher.search(&needle, canvas.size, args.threshold);
            let t1 = Instant::now();
            let _ = canvas_cache.insert(canvas);
            eprintln!(
                "`{letter}` {offset:?} needle size {}x{} hits {} elapsed {}ms",
                size.x(),
                size.y(),
                hits.len(),
                (t1 - t0).as_millis()
            );
            n_hits += hits.len();

            let glyph_id = font.glyph_for_char(letter).unwrap();
            let units_per_em = font.metrics().units_per_em as f32;
            let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * to_px;
            let _bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
            let bearing_x = glyph_bounds.origin().x();
            let _ascent_px = font.metrics().ascent * (1. / units_per_em) * render_options.size;
            let mut last_rect = RectI::default();
            for hit in hits {
                if hit.rect.intersects(last_rect) {
                    continue;
                }
                if let Some(entry) = hits_by_char.get_mut(&letter) {
                    *entry += 1;
                } else {
                    assert!(false);
                };
                all_hits.push(MatchWithBaseline {
                    similarity: hit.similarity,
                    rect: hit.rect,
                    baseline: corrected_offset[1],
                });
                last_rect = hit.rect;
                //total_error += hit.error;
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
                    corrected_offset[1],
                    offset[0],
                    offset[1],
                );
                //eprintln!("HIT {},{} offset {:?}", pt.x(), pt.y(), offset);
            }
            //eprintln!("took {:.4}ms", (t1 - t0).as_millis());
        }
    }
    let t11 = Instant::now();
    eprintln!("overall {:.4}ms", (t11 - t00).as_millis());
    if args.count_unique {
        eprintln!(
            "saw {} unique rows out of {} == {:.2}%",
            unique_rows.len(),
            total_rows,
            unique_rows.len() as f32 / total_rows as f32 * 100.
        );
    }
    //let avg_error = total_error as f32 / n_hits as f32;
    //eprintln!("hits: {n_hits} error per hit:{avg_error}");
    eprintln!("hits: {n_hits}");

    for (char, count) in hits_by_char {
        eprintln!("`{char}` {count}");
    }

    if false {
        let baselines = {
            let mut tmp: Vec<_> = all_hits
                .iter()
                .map(|h| h.rect.lower_left().y() as f32 + h.baseline)
                .collect();
            tmp.sort_by(f32::total_cmp);
            tmp.dedup();
            tmp
        };
        let baseline_diff_counts = {
            let mut counts = std::collections::HashMap::new();
            for [b1, b2] in baselines.array_windows::<2>() {
                let diff = b2 - b1;
                let key = format!("{diff}");
                //eprintln!("{diff}");
                if let Some(count) = counts.get_mut(&key) {
                    *count += 1;
                } else {
                    counts.insert(key, 1);
                }
            }
            let mut counts: Vec<_> = counts
                .into_iter()
                .map(|(diff, count)| (count, diff))
                .collect();
            counts.sort();
            counts
        };
        for (count, diff) in baseline_diff_counts {
            eprintln!("{diff} {count}");
        }
        // so we jump between 12 and 13 pixels between baselines
        // so either the text is getting snapped
        // or the true line spacing is somewhere between
    }
}

fn image_to_f32(img: &GrayImage) -> Vec<f32> {
    img.as_raw().iter().map(|x| (255 - *x) as f32 / 255.).collect()
}

fn canvas_to_f32(canvas: &Canvas) -> Vec<f32> {
    canvas.pixels.iter().map(|x| *x as f32 / 255.).collect()
}

// for NCC, the norm has to be of the window, which is the same for the needle each time,
// but varies for the image
fn normalize_image(pixels: &mut [f32]) {
    let mean = pixels.iter().map(|x| *x as f64).sum::<f64>() / pixels.len() as f64;
    let mean = mean as f32;
    pixels.iter_mut().for_each(|x| *x -= mean);
    //let norm = pixels.iter().fold(0f64, |acc, x| acc + (*x as f64) * (*x as f64)).sqrt();
    //let norm = (1. / norm) as f32;
    //pixels.iter_mut().for_each(|x| *x *= norm);
}

fn cross_corr_n<const N: usize>(a: [f32; N], b: [f32; N], mask: [f32; N]) -> f32 {
    let mut ret = 0f32;
    for i in 0..N {
        ret += a[i] * b[i] * mask[i];
    }
    ret
}
