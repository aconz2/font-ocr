#![feature(iter_array_chunks)]

use std::time::Instant;
use std::collections::HashMap;

use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::{GrayImage};
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
    similarity: f32,
}

#[derive(Clone, Copy, Debug)]
struct MatchWithBaseline {
    rect: RectI,
    similarity: f32,
    baseline: f32,
}

type SumTableT = u32;
type SumSqrTableT = u64;

struct Searcher {
    reference: Array2<u8>,
    sum_table: Array2<SumTableT>,
    sumsqr_table: Array2<SumSqrTableT>,
    r_w: usize,
    r_h: usize,
    acc: Vec<u32>,
    needle: Vec<u8>,
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

#[derive(Clone)]
struct Array2<T: Copy> {
    data: Box<[T]>,
    rows: usize,
    cols: usize,
}

impl<T: Copy + Default> Array2<T> {
    fn new(rows: usize, cols: usize) -> Self {
        let data = vec![T::default(); rows * cols].into();
        Array2 { data, rows, cols }
    }
}

impl<T: Copy> std::ops::Index<(usize, usize)> for Array2<T> {
    type Output = T;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[y * self.cols + x]
    }
}

impl<T: Copy> std::ops::IndexMut<(usize, usize)> for Array2<T> {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.data[y * self.cols + x]
    }
}

impl<T: Copy> Array2<T> {
    //fn len(&self) -> usize { self.data.len() }
    fn get_row(&self, y: usize) -> &[T] {
        &self.data[y * self.cols..(y+1) * self.cols]
    }

    fn ravel_block(&self, (x, y): (usize, usize), (w, h): (usize, usize)) -> Vec<T> {
        let mut ret = Vec::with_capacity(w * h);
        for row in 0..h {
            ret.extend(&self.get_row(y + row)[x..x+w]);
        }
        ret
    }
}

fn canvas_to_lum8(canvas: &Canvas) -> GrayImage {
    assert!(canvas.format == Format::A8);
    let w = canvas.size.x() as u32;
    let h = canvas.size.y() as u32;
    let pixels = canvas.pixels.clone();
    GrayImage::from_raw(w, h, pixels).unwrap()
}

fn copy_needle_n<const N: usize>(out: &mut [[u8; N]], needle: &[u8], size: Vector2I) {
    let w = size.x() as usize;
    for y in 0..(size.y() as usize) {
        for x in 0..(size.x() as usize) {
            out[y][x] = needle[y * w + x];
        }
    }
}

fn get_mask_n<const N: usize>(n: usize) -> [u8; N] {
    assert!(n >= 1 && n <= N);
    let mut arr = [0; N];
    for i in 0..n {
        arr[i] = 1;
    }
    arr
}

// https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf
fn ncc_sum_table(pixels: &Array2<u8>) -> Array2<SumTableT> {
    let mut ret = Array2::<SumTableT>::new(pixels.rows, pixels.cols);
    // init first row and col
    ret[(0, 0)] = pixels[(0, 0)] as SumTableT;
    for x in 1..pixels.cols {
        ret[(x, 0)] = pixels[(x, 0)] as SumTableT + ret[(x - 1, 0)];
    }
    for y in 1..pixels.rows {
        ret[(0, y)] = pixels[(0, y)] as SumTableT + ret[(0, y - 1)];
    }
    for y in 1..pixels.rows {
        for x in 1..pixels.cols {
            ret[(x, y)] = pixels[(x, y)] as SumTableT + ret[(x - 1, y)] + ret[(x, y - 1)] - ret[(x - 1, y - 1)];
        }
    }
    ret
}

fn ncc_sumsqr_table(pixels: &Array2<u8>) -> Array2<SumSqrTableT> {
    let mut ret = Array2::<SumSqrTableT>::new(pixels.rows, pixels.cols);
    for x in 0..pixels.cols {
        let p = pixels[(x, 0)] as SumSqrTableT;
        ret[(x, 0)] = p * p;
    }
    for y in 0..pixels.rows {
        let p = pixels[(0, y)] as SumSqrTableT;
        ret[(0, y)] = p * p;
    }
    for y in 1..pixels.rows {
        for x in 1..pixels.cols {
            let p = pixels[(x, y)] as SumSqrTableT;
            ret[(x, y)] = p * p + ret[(x - 1, y)] + ret[(x, y - 1)] - ret[(x - 1, y - 1)];
        }
    }
    ret
}

fn ncc_sum_table_sum(s: &Array2<u32>, (x, y): (usize, usize), (w, h): (usize, usize)) -> u32 {
    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
    let a = s[(x + w - 1, y + h - 1)] as i64;
    let b = s[(x     - 1, y + h - 1)] as i64;
    let c = s[(x + w - 1, y     - 1)] as i64;
    let d = s[(x     - 1, y     - 1)] as i64;
    (a - b - c + d) as u32
}

fn ncc_sumsqr_table_sum(s: &Array2<SumSqrTableT>, (x, y): (usize, usize), (w, h): (usize, usize)) -> SumSqrTableT {
    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
    let a = s[(x + w - 1, y + h - 1)] as i64;
    let b = s[(x     - 1, y + h - 1)] as i64;
    let c = s[(x + w - 1, y     - 1)] as i64;
    let d = s[(x     - 1, y     - 1)] as i64;
    (a - b - c + d) as u64
}

//fn ncc_recip_norm(sum: f32, sumsqr: f32, n: usize) -> f32 {
//    // this is ||(r - r_mean)|| == sum(r_i^2 - 2 * r_i * r_mean + r_mean_i^2)
//    // == sum(r_i^2) - sum(r_i)^2 / n
//    let normsqr = sumsqr - sum.powf(2.) / n as f32;
//    let recip_norm = normsqr.powf(-0.5).clamp(0f32.next_up(), f32::INFINITY);
//    recip_norm
//}
//
//fn ncc_norm(sum: f32, sumsqr: f32, n: usize) -> f32 {
//    // this is ||(r - r_mean)|| == sum(r_i^2 - 2 * r_i * r_mean + r_mean_i^2)
//    // == sum(r_i^2) - sum(r_i)^2 / n
//    let normsqr = sumsqr - sum * sum / n as f32;
//    //let norm = normsqr.powf(0.5).clamp(0f32.next_up(), f32::INFINITY);
//    let norm = normsqr.powf(0.5);
//    norm
//}

impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let reference = image_to_u8(img);
        let sum_table = ncc_sum_table(&reference);
        let sumsqr_table = ncc_sumsqr_table(&reference);

        let r_w = img.width() as usize;
        let r_h = img.height() as usize;
        let matches = Vec::with_capacity(1024);
        let acc = vec![0; img.width() as usize];
        let needle = vec![0; 1024];

        Searcher {
            reference,
            sum_table,
            sumsqr_table,
            r_w,
            r_h,
            needle,
            acc,
            matches,
        }
    }

    fn search(&mut self, needle: &[u8], size: Vector2I, threshold: f32) -> &[Match] {
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

    // note that we skip the first row and col of reference to make the indexing easier
    // and because the needle is padded up to N, we won't search the last N - needle.width()
    // cols
    #[inline(never)]
    fn search_n<const N: usize>(
        &mut self,
        needle: &[u8],
        size: Vector2I,
        threshold: f32,
    ) -> &[Match] {
        assert!(size.x() <= N as i32);
        self.matches.clear();

        let n_h = size.y() as usize;
        let n_w = size.x() as usize;

        let (r_w, r_h) = (self.r_w, self.r_h);

        let x_searches = r_w - N + 1;
        let y_searches = r_h - n_h + 1;

        let mut min_sim = f32::INFINITY;
        let mut max_sim = -f32::INFINITY;

        self.acc.fill(0);
        self.acc.resize(x_searches - 1, 0);

        self.needle.resize(N * n_h as usize, 0);
        {
            let (rows, _rem) = self.needle.as_chunks_mut::<N>();
            copy_needle_n(rows, needle, size);
        }
        let mask = get_mask_n(n_w);
        let (needle_buf, _rem) = self.needle.as_chunks::<N>();

        let n = n_w * n_h;

        // sum_needle sumsqr_needle
        let (s_n, s2_n) = image_centered_var(needle);

        // we calculate NCC, which is normally
        //     (x-x') (y-y')
        // ---------------------
        // ||(x-x')|| ||(y-y')||
        //
        // the numerator becomes xy - SxSy/n
        // where xy is the dot product and Sx is the sum(x)
        // the norm of the mean-centered vector x-x' is the standard deviation
        // which can be calculated by sqrt(S2 - S**2/n)
        // where S2 is the sum of squares and S is the sum

        for y in 1..y_searches {
            for (needle_y, needle_row) in needle_buf.iter().enumerate() {

                let row = self.reference.get_row(y + needle_y);
                let ref_windows = row[1..].array_windows::<N>();
                for (_x, (acc, ref_row)) in self.acc.iter_mut().zip(ref_windows).enumerate() {
                    // x is offset by 1
                    *acc += cross_corr_n(*needle_row, *ref_row, mask) as u32;
                }
            }
            for (x, acc) in self.acc.iter().enumerate() {
                let x = x + 1;
                let s_p = ncc_sum_table_sum(&self.sum_table, (x, y), (n_w, n_h));
                let s2_p = ncc_sumsqr_table_sum(&self.sumsqr_table, (x, y), (n_w, n_h));
                if s_p == 0 {
                    continue;
                }
                let num = *acc as f64 - (s_n * s_p) as f64 / n as f64;
                if num < 0. {
                    continue;
                }
                let norm2_n = s2_n as f64 - (s_n * s_n) as f64 / n as f64;
                let norm2_p = s2_p as f64 - (s_p * s_p) as f64 / n as f64;
                assert!(norm2_n > 0.);
                assert!(norm2_p > 0.);
                let den = (norm2_n * norm2_p).sqrt();
                //let den = norm2_n.sqrt() * norm2_p.sqrt();
                let similarity = (num as f64 / den) as f32;
                //let similarity = ((*acc as f64) / den - ((s_n * s_p) as f64 / n as f64) / den) as f32;
                assert!(similarity >= -1.01 && similarity <= 1.01, "got bad similarity={similarity} norm2_n={norm2_n} norm2_p={norm2_p} acc={acc} num={num} s_n={s_n} s_p={s_p}");
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
            self.acc.fill(0);
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

    #[arg(long, default_value = "alphabet")]
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
    //let img = image::open(args.img.first().unwrap()).unwrap().into_luma8();
    let img = image::open(args.img.first().unwrap()).unwrap();
    //let img = img.crop_imm(0, 0, img.width(), 200);
    let img = img.into_luma8();
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
            let mut needle = canvas_to_u8(&canvas);
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
        if count == 0 {
            continue;
        }
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
    }
}

fn image_to_f32(img: &GrayImage) -> Array2<f32> {
    let data = img.as_raw().iter().map(|x| (255 - *x) as f32 / 255.).collect();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    Array2 { data, rows, cols }
}

fn image_to_i16(img: &GrayImage) -> Array2<i16> {
    let data = img.as_raw().iter().map(|x| (255 - *x) as i16).collect();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    Array2 { data, rows, cols }
}

fn image_to_u8(img: &GrayImage) -> Array2<u8> {
    let data = img.as_raw().iter().map(|x| (255 - *x) as u8).collect();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    Array2 { data, rows, cols }
}

fn canvas_to_f32(canvas: &Canvas) -> Vec<f32> {
    canvas.pixels.iter().map(|x| *x as f32 / 255.).collect()
}

fn canvas_to_i16(canvas: &Canvas) -> Vec<i16> {
    canvas.pixels.iter().map(|x| *x as i16).collect()
}

fn canvas_to_u8(canvas: &Canvas) -> Vec<u8> {
    canvas.pixels.iter().map(|x| *x as u8).collect()
}

fn image_centered_var(pixels: &[u8]) -> (u32, u32) {
    let mut sum = 0u32;
    let mut sum2 = 0u32;
    for p in pixels {
        let p = *p as u32;
        sum += p;
        sum2 += p * p;
    }
    (sum, sum2)
}

//fn image_norm(pixels: &[f32]) -> f32 {
//    pixels.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32
//}
//
//fn normalize_image(pixels: &mut [f32]) {
//    let mean = pixels.iter().map(|x| *x as f64).sum::<f64>() / pixels.len() as f64;
//    let mean = mean as f32;
//    pixels.iter_mut().for_each(|x| *x -= mean);
//    let norm = pixels.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
//    // this should only be called for needles which have at least 1 pixel
//    assert!(norm >= 0.);
//    let norm = (1. / norm) as f32;
//    pixels.iter_mut().for_each(|x| *x *= norm);
//}
//
//fn center_image(pixels: &mut [f32]) {
//    let mean = pixels.iter().map(|x| *x as f64).sum::<f64>() / pixels.len() as f64;
//    pixels.iter_mut().for_each(|x| *x -= mean as f32);
//}
//
//
fn cross_corr_n<const N: usize>(a: [u8; N], b: [u8; N], mask: [u8; N]) -> u32 {
    let mut ret = 0u32;
    for i in 0..N {
        ret += a[i] as u32 * b[i] as u32 * mask[i] as u32;
    }
    ret
}
