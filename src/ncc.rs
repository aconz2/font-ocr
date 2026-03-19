#![feature(iter_array_chunks)]

use std::collections::HashMap;
use std::time::Instant;

use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use image::GrayImage;
use pathfinder_geometry::rect::{RectF, RectI};
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};
use rayon::prelude::*;

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

// _ underscore is a tricky character b/c it has no bottom whitespace and so wants to match lines
const DEFAULT_ALPHABET: &str =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=+<>(){};:/-";

const MAX_MATCHES: usize = 1024;

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

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct MatchC {
    x: u16,
    y: u16,
    similarity: f32,
}

#[derive(Clone, Copy, Debug)]
struct MatchWithLetter {
    rect: RectI,
    similarity: f32,
    letter: char,
}

impl Match {
    fn from_matchc(m: MatchC, w: u32, h: u32) -> Self {
        let rect = RectI::new(
            Vector2I::new(m.x as i32, m.y as i32),
            Vector2I::new(w as i32, h as i32),
        );
        let similarity = m.similarity;
        Match { rect, similarity }
    }
}

unsafe extern "C" {
    fn ncc_8_u8(
        reference: *const u8,
        r_w: usize,
        r_h: usize,
        needle_u8: *const u8,
        n_w: usize,
        n_h: usize,
        acc: *mut u32,
        acc_len: usize,
        patch_sum: *const u32,
        patch_rnorm: *const f64,
        start_end: *const u16,
        threshold: f32,
        out: *mut MatchC,
        n_out: usize,
    ) -> usize;

    fn ncc_16_u8(
        reference: *const u8,
        r_w: usize,
        r_h: usize,
        needle_u8: *const u8,
        n_w: usize,
        n_h: usize,
        acc: *mut u32,
        acc_len: usize,
        patch_sum: *const u32,
        patch_rnorm: *const f64,
        start_end: *const u16,
        threshold: f32,
        out: *mut MatchC,
        n_out: usize,
    ) -> usize;
}

struct Searcher {
    reference_f32: Array2<f32>,
    reference_u8: Array2<u8>,
    patch_sum: Array2<u32>,
    patch_rnorm: Array2<f64>,
    sum_table: Array2<u32>,
    sumsqr_table: Array2<u64>,
    acc_u32: Vec<u32>,
    needle_u8: Vec<u8>,
    matches: Vec<Match>,
    matches_c: Vec<MatchC>,
    last_patch_size: Option<Vector2I>,
    start_end: Vec<u16>,
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
    fn get_row(&self, y: usize) -> &[T] {
        &self.data[y * self.cols..(y + 1) * self.cols]
    }
}
impl Searcher {
    fn new(img: &GrayImage) -> Searcher {
        let reference_u8 = image_to_u8(img);
        let sum_table = ncc_sum_table(&reference_u8);
        let sumsqr_table = ncc_sumsqr_table(&reference_u8);
        let reference_f32 = image_to_f32(img);
        let patch_sum = Array2::new(img.height() as usize, img.width() as usize);
        let patch_norm = Array2::new(img.height() as usize, img.width() as usize);

        let matches = Vec::with_capacity(1024);
        let matches_c = vec![MatchC::default(); MAX_MATCHES];
        // big enough to keep 8 partial sums (for 16) and 32 bytes extra to align
        let acc_u32 = vec![0; img.width() as usize * 8 + 8];
        let needle_u8 = vec![0; 128];
        let last_patch_size = None;
        let start_end = vec![0; img.height() as usize * 2];

        Searcher {
            reference_f32,
            reference_u8,
            sum_table,
            sumsqr_table,
            needle_u8,
            acc_u32,
            matches,
            matches_c,
            last_patch_size,
            patch_sum,
            patch_rnorm: patch_norm,
            start_end,
        }
    }

    fn prepare_for_size(&mut self, size: Vector2I) {
        if let Some(s) = self.last_patch_size
            && s == size
        {
            return;
        }

        let n_h = size.y() as usize;
        let n_w = size.x() as usize;
        let n = n_h * n_w;

        let (r_w, r_h) = (self.reference_u8.cols, self.reference_u8.rows);

        let x_searches = r_w - n_w + 1;
        let y_searches = r_h - n_h + 1;

        for y in 1..y_searches {
            let start = {
                let mut x = 1;
                while x < x_searches {
                    let s_p = ncc_sum_table_sum_nz(&self.sum_table, (x, y), (n_w, n_h));
                    if s_p != 0 {
                        break;
                    }
                    x += 1;
                }
                x
            };
            let end = {
                let mut x = x_searches - 1;
                while x > start {
                    let s_p = ncc_sum_table_sum_nz(&self.sum_table, (x, y), (n_w, n_h));
                    if s_p != 0 {
                        break;
                    }
                    x -= 1;
                }
                x + 1
            };
            //if start != end {
            //    debug_assert!(ncc_sum_table_sum_nz(&self.sum_table, (start, y), (n_w, n_h)) != 0, "start y={y} start={start} end={end}");
            //    debug_assert!(ncc_sum_table_sum_nz(&self.sum_table, (end - 1, y), (n_w, n_h)) != 0, "end y={y} start={start} end={end}");
            //}
            for x in start..end {
                let s_p = ncc_sum_table_sum_nz(&self.sum_table, (x, y), (n_w, n_h));
                let s2_p = ncc_sumsqr_table_sum_nz(&self.sumsqr_table, (x, y), (n_w, n_h));
                let norm = s2_p as f64 - ((s_p as u64 * s_p as u64) as f64) / n as f64;
                self.patch_sum[(x, y)] = s_p;
                self.patch_rnorm[(x, y)] = 1. / norm.sqrt();
            }
            self.start_end[y * 2 + 0] = start.try_into().unwrap();
            self.start_end[y * 2 + 1] = end.try_into().unwrap();
        }

        self.last_patch_size = Some(size);
    }

    fn search_u8(&mut self, needle: &[u8], size: Vector2I, threshold: f32) -> &[Match] {
        self.prepare_for_size(size);
        let w = size.x();
        if w <= 8 {
            self.search_n_u8::<8>(needle, size, threshold)
        } else if w <= 16 {
            self.search_n_u8::<16>(needle, size, threshold)
        } else {
            todo!()
        }
    }

    fn search_c_u8(&mut self, needle: &[u8], size: Vector2I, threshold: f32) -> &[Match] {
        // TODO I think I need to prepare_for_size with the width as 8
        self.prepare_for_size(size);
        let n_h = size.y() as usize;
        let n_w = size.x() as usize;
        let n_matches = if n_w <= 8 {
            const N: usize = 8;

            self.needle_u8.resize(N * n_h, 0);
            {
                let (rows, _rem) = self.needle_u8.as_chunks_mut::<N>();
                copy_needle_n_u8(rows, needle, size);
            }

            unsafe {
                ncc_8_u8(
                    self.reference_u8.data.as_ptr(),
                    self.reference_u8.cols,
                    self.reference_u8.rows,
                    self.needle_u8.as_ptr(),
                    n_w,
                    n_h,
                    self.acc_u32.as_mut_ptr(),
                    self.acc_u32.len(),
                    self.patch_sum.data.as_ptr(),
                    self.patch_rnorm.data.as_ptr(),
                    self.start_end.as_ptr(),
                    threshold,
                    self.matches_c.as_mut_ptr(),
                    self.matches_c.len(),
                )
            }
        } else if n_w <= 16 {
            const N: usize = 16;

            self.needle_u8.resize(N * n_h, 0);
            {
                let (rows, _rem) = self.needle_u8.as_chunks_mut::<N>();
                copy_needle_n_u8(rows, needle, size);
            }

            unsafe {
                ncc_16_u8(
                    self.reference_u8.data.as_ptr(),
                    self.reference_u8.cols,
                    self.reference_u8.rows,
                    self.needle_u8.as_ptr(),
                    n_w,
                    n_h,
                    self.acc_u32.as_mut_ptr(),
                    self.acc_u32.len(),
                    self.patch_sum.data.as_ptr(),
                    self.patch_rnorm.data.as_ptr(),
                    self.start_end.as_ptr(),
                    threshold,
                    self.matches_c.as_mut_ptr(),
                    self.matches_c.len(),
                )
            }
        } else {
            panic!("not handled")
        };

        if n_matches == MAX_MATCHES {
            eprintln!("WARN got >= {n_matches} matches");
        }
        self.matches.clear();
        for m in self.matches_c.iter().take(n_matches) {
            self.matches
                .push(Match::from_matchc(*m, n_w as u32, n_h as u32));
        }
        &self.matches
    }

    #[inline(never)]
    fn search_n_u8<const N: usize>(
        &mut self,
        needle: &[u8],
        size: Vector2I,
        threshold: f32,
    ) -> &[Match] {
        assert!(size.x() <= N as i32);
        self.matches.clear();

        let n_h = size.y() as usize;
        let n_w = size.x() as usize;
        let r_h = self.reference_f32.rows;
        let y_searches = r_h - n_h + 1;

        self.needle_u8.resize(N * n_h, 0);
        {
            let (rows, _rem) = self.needle_u8.as_chunks_mut::<N>();
            copy_needle_n_u8(rows, needle, size);
        }
        let (needle_buf, _rem) = self.needle_u8.as_chunks::<N>();

        let n = n_w * n_h;

        // sum_needle sumsqr_needle
        let (s_n, s2_n) = image_sum_sumsqr(needle);
        if s_n == 0 {
            return &self.matches;
        }

        for y in 1..y_searches {
            let start = self.start_end[y * 2] as usize;
            let end = self.start_end[y * 2 + 1] as usize;

            for (needle_y, needle_row) in needle_buf.iter().enumerate() {
                let row = self.reference_u8.get_row(y + needle_y);
                let ref_windows = row.array_windows::<N>().skip(start).take(end - start);
                for (acc, ref_row) in self.acc_u32.iter_mut().zip(ref_windows) {
                    if needle_y == 0 {
                        *acc = cross_corr_n_u8(*needle_row, *ref_row);
                    } else {
                        *acc += cross_corr_n_u8(*needle_row, *ref_row);
                    }
                }
            }
            for (x, acc) in (start..end).zip(self.acc_u32.iter()) {
                let s_p = ncc_sum_table_sum_nz(&self.sum_table, (x, y), (n_w, n_h));
                let s2_p = ncc_sumsqr_table_sum_nz(&self.sumsqr_table, (x, y), (n_w, n_h));
                if s_p == 0 {
                    continue;
                }
                let num = *acc as f64 - (s_n as u64 * s_p as u64) as f64 / n as f64;
                if num < 0. {
                    continue;
                }
                //let num_i64 = n as i64 * *acc as i64 - s_n as i64 * s_p as i64;
                let norm2_n = s2_n as f64 - (s_n as u64 * s_n as u64) as f64 / n as f64;
                let norm2_p = s2_p as f64 - (s_p as u64 * s_p as u64) as f64 / n as f64;
                debug_assert!(norm2_n > 0.);
                debug_assert!(norm2_p > 0.);
                let den = (norm2_n * norm2_p).sqrt();
                let similarity_f64 = num as f64 / den;
                let similarity = (num as f64 / den) as f32;
                debug_assert!(
                    similarity >= -1.01 && similarity <= 1.01,
                    "got bad similarity={similarity} norm2_n={norm2_n} norm2_p={norm2_p} acc={acc} num={num} s_n={s_n} s_p={s_p}"
                );
                if similarity_f64 != f64::INFINITY && similarity_f64 > threshold as f64 {
                    let rect = RectI::new(
                        Vector2I::new(x as i32, y as i32),
                        Vector2I::new(n_w as i32, n_h as i32),
                    );
                    self.matches.push(Match { similarity, rect });
                }
            }
        }
        &self.matches
    }
}

#[derive(Parser, Debug, Clone)]
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

    #[arg(long, default_value_t = 0.8)]
    threshold: f32,

    #[arg(long, default_value_t = 0.95)]
    anchor_threshold: f32,

    #[arg(long, default_value_t = 5)]
    overlap: i32,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,

    #[arg(long, default_value = "alphabet")]
    box_size: String,

    #[arg(long, default_value_t = 0)]
    x_padding: usize,

    #[arg(long, default_value_t = 0)]
    y_padding: usize,

    #[arg(long)]
    save_letters: bool,

    #[arg(long)]
    rust: bool,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    csv: bool,

    #[arg(long)]
    raw: bool,
}

fn get_hits(args: &Args, image_index: usize) -> Vec<MatchWithLetter> {
    let padding = Vector2I::new(args.x_padding as i32, args.y_padding as i32);

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

    let font = Font::from_path(&args.font, 0).unwrap();

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

    let img = image::open(&args.img[image_index]).unwrap().into_luma8();
    let mut searcher = Searcher::new(&img);
    let mut n_hits = 0;

    let to_px = (1. / font.metrics().units_per_em as f32) * render_options.size;

    let t00 = Instant::now();

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
            let canvas = render(
                &font,
                letter,
                corrected_offset,
                render_options,
                canvas_size,
                canvas_cache.take(),
                padding,
            );
            let needle = canvas_to_u8(&canvas);
            let size = canvas.size;
            if args.save_letters {
                let x = (offset[0] * 1000.) as usize;
                let y = (offset[1] * 1000.) as usize;
                let im = canvas_to_lum8(&canvas);
                image::DynamicImage::ImageLuma8(im)
                    .save(format!("letters/{letter}-{x}_{y}.png"))
                    .unwrap();
            }
            let t0 = Instant::now();
            let hits = if args.rust {
                searcher.search_u8(&needle, size, args.threshold)
            } else {
                searcher.search_c_u8(&needle, size, args.threshold)
            };
            let t1 = Instant::now();
            if args.verbose {
                eprintln!(
                    "`{letter}` {offset:?} needle size {}x{} hits {} elapsed {}ms ({:.2} ns/pixel)",
                    size.x(),
                    size.y(),
                    hits.len(),
                    (t1 - t0).as_millis(),
                    (t1 - t0).as_nanos() as f64 / (img.width() * img.height()) as f64,
                );
            }
            n_hits += hits.len();

            let glyph_id = font.glyph_for_char(letter).unwrap();
            let units_per_em = font.metrics().units_per_em as f32;
            let glyph_bounds = font.typographic_bounds(glyph_id).unwrap() * to_px;
            let _bearing_y = glyph_bounds.origin().y() + glyph_bounds.height();
            let bearing_x = glyph_bounds.origin().x();
            let _ascent_px = font.metrics().ascent * (1. / units_per_em) * render_options.size;
            for hit in hits {
                all_hits.push(MatchWithLetter {
                    letter,
                    similarity: hit.similarity,
                    rect: hit.rect,
                });
                let ul = hit.rect.origin();
                let pt = hit.rect.to_f32().center();
                if args.raw {
                    println!(
                        "{},{},{},{},{},{},{},{},{},{},{}",
                        letter as usize,
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
                }
            }
            *hits_by_char.get_mut(&letter).unwrap() += hits.len();
        }
    }
    let t11 = Instant::now();
    if args.verbose {
        eprintln!("overall {:.4}ms", (t11 - t00).as_millis());
        eprintln!("hits: {n_hits}");
    }

    if args.verbose {
        let mut hits_by_char = hits_by_char.into_iter().collect::<Vec<_>>();
        hits_by_char.sort_by_key(|(char, count)| (*count, *char));
        for (char, count) in hits_by_char {
            if count == 0 {
                continue;
            }
            eprintln!("`{char}` {count}");
        }
    }

    all_hits
}

fn process_hits(args: &Args, all_hits: &[MatchWithLetter]) -> Vec<Vec<MatchWithLetter>> {
    let mut hits = {
        use std::collections::HashSet;
        let mut keep_y = HashSet::with_capacity(512);
        for h in all_hits {
            if h.similarity >= args.anchor_threshold {
                keep_y.insert(h.rect.origin().y());
            }
        }
        let mut acc = vec![];
        for h in all_hits {
            if keep_y.contains(&h.rect.origin().y()) {
                acc.push(*h);
            }
        }
        acc
    };
    let t0 = Instant::now();
    hits.sort_by_key(|m| m.rect.origin().y());
    if args.verbose {
        for hit in &hits {
            eprintln!("{} {hit:?}", hit.rect.origin().y());
        }
    }
    let line_slices = partition_by(&hits, |a, b| a.rect.origin().y() == b.rect.origin().y());
    let mut lines = vec![];
    for (i, j) in &line_slices {
        let slice = &mut hits[*i..*j];
        slice.sort_by_key(|m| m.rect.origin().x());
    }
    for (i, j) in line_slices {
        let slice = &hits[i..j];
        let duplicate_slices = partition_by(slice, |a, b| {
            (a.rect.origin().x() - b.rect.origin().x()).abs() <= args.overlap
        });
        let mut dedup = vec![];
        for (i, j) in duplicate_slices {
            // this is slightly maddening
            let m = slice[i..j]
                .iter()
                .max_by(|a, b| f32::total_cmp(&a.similarity, &b.similarity))
                .unwrap();
            dedup.push(*m);
        }
        if args.verbose {
            let mut dx_counts = HashMap::with_capacity(16);
            for [a, b] in dedup.array_windows::<2>() {
                let dx = b.rect.origin().x() - a.rect.origin().x();
                if let Some(c) = dx_counts.get_mut(&dx) {
                    *c += 1
                } else {
                    dx_counts.insert(dx, 1);
                }
            }
            eprintln!("{dx_counts:?}");
        }
        lines.push(dedup);
    }
    let t1 = Instant::now();
    if args.verbose {
        eprintln!("processing took {}ms", (t1 - t0).as_millis());
    }
    lines
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        let font = Font::from_path(&args.font, 0).unwrap();
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
            if false {
                eprintln!(
                    "`{char}` {typo_bounds_px:?}px advance={advance} bearing_x={bearing_x} bearing_y={bearing_y}"
                );
            }
        }

        //let alphabet_bbox = args.alphabet.chars().fold(RectF::default(), |bounds, c| {
        //    let gid = font.glyph_for_char(c).unwrap();
        //    let raster_rect = font
        //        .raster_bounds(
        //            gid,
        //            render_options.size,
        //            Transform2F::default(),
        //            render_options.hinting,
        //            render_options.rasterization,
        //        )
        //        .unwrap();
        //    bounds.union_rect(raster_rect.to_f32())
        //});
        //eprintln!("alphabet_bbox size {:?}", alphabet_bbox.size());
    }

    if args.raw {
        assert!(args.img.len() == 1);
        get_hits(&args, 0);
        return;
    }

    let mut pages: Vec<_> = (0..args.img.len())
        .into_par_iter()
        .map(|i| {
            let hits = get_hits(&args, i);
            let lines = process_hits(&args, &hits);
            (i, lines)
        })
        .collect();
    pages.sort_by_key(|(i, _)| *i);

    if args.csv {
        for (i, lines) in pages {
            for line in lines {
                for m in line {
                    let pt = m.rect.to_f32().center();
                    println!(
                        "{},{},{},{},{},{},{},{}",
                        i,
                        m.letter as usize,
                        pt.x(),
                        pt.y(),
                        m.rect.origin().x(),
                        m.rect.origin().y(),
                        m.rect.width(),
                        m.rect.height()
                    );
                }
            }
        }
    } else {
        for (_, lines) in pages {
            for line in lines {
                for m in line {
                    print!("{}", m.letter);
                }
                println!();
            }
        }
    }
}

fn image_to_f32(img: &GrayImage) -> Array2<f32> {
    let data = img.as_raw().iter().map(|x| (255 - *x) as f32).collect();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    Array2 { data, rows, cols }
}

fn image_to_u8(img: &GrayImage) -> Array2<u8> {
    let data = img.as_raw().iter().map(|x| 255 - *x).collect();
    let rows = img.height() as usize;
    let cols = img.width() as usize;
    Array2 { data, rows, cols }
}

fn canvas_to_u8(canvas: &Canvas) -> Vec<u8> {
    canvas.pixels.clone()
}

fn image_sum_sumsqr(pixels: &[u8]) -> (u32, u32) {
    let mut sum = 0u32;
    let mut sum2 = 0u32;
    for p in pixels {
        let p = *p as u32;
        sum += p;
        sum2 += p * p;
    }
    (sum, sum2)
}

fn cross_corr_n_u8<const N: usize>(a: [u8; N], b: [u8; N]) -> u32 {
    let mut ret = 0u32;
    for i in 0..N {
        ret += (a[i] as u16 * b[i] as u16) as u32;
    }
    ret
}

fn canvas_to_lum8(canvas: &Canvas) -> GrayImage {
    assert!(canvas.format == Format::A8);
    let w = canvas.size.x() as u32;
    let h = canvas.size.y() as u32;
    let pixels = canvas.pixels.clone();
    GrayImage::from_raw(w, h, pixels).unwrap()
}

fn copy_needle_n_u8<const N: usize>(out: &mut [[u8; N]], needle: &[u8], size: Vector2I) {
    let w = size.x() as usize;
    for y in 0..(size.y() as usize) {
        for x in 0..w {
            out[y][x] = needle[y * w + x];
        }
        for x in w..N {
            out[y][x] = 0;
        }
    }
}

// https://isas.iar.kit.edu/pdf/SPIE01_BriechleHanebeck_CrossCorr.pdf
fn ncc_sum_table(pixels: &Array2<u8>) -> Array2<u32> {
    let mut ret = Array2::<u32>::new(pixels.rows, pixels.cols);
    // init first row and col
    ret[(0, 0)] = pixels[(0, 0)] as u32;
    for x in 1..pixels.cols {
        ret[(x, 0)] = pixels[(x, 0)] as u32 + ret[(x - 1, 0)];
    }
    for y in 1..pixels.rows {
        ret[(0, y)] = pixels[(0, y)] as u32 + ret[(0, y - 1)];
    }
    for y in 1..pixels.rows {
        for x in 1..pixels.cols {
            ret[(x, y)] =
                pixels[(x, y)] as u32 + ret[(x - 1, y)] + ret[(x, y - 1)] - ret[(x - 1, y - 1)];
        }
    }
    ret
}

fn ncc_sumsqr_table(pixels: &Array2<u8>) -> Array2<u64> {
    let mut ret = Array2::<u64>::new(pixels.rows, pixels.cols);
    for x in 0..pixels.cols {
        let p = pixels[(x, 0)] as u64;
        ret[(x, 0)] = p * p;
    }
    for y in 0..pixels.rows {
        let p = pixels[(0, y)] as u64;
        ret[(0, y)] = p * p;
    }
    for y in 1..pixels.rows {
        for x in 1..pixels.cols {
            let p = pixels[(x, y)] as u64;
            ret[(x, y)] = p * p + ret[(x - 1, y)] + ret[(x, y - 1)] - ret[(x - 1, y - 1)];
        }
    }
    ret
}

fn ncc_sum_table_sum_nz(s: &Array2<u32>, (x, y): (usize, usize), (w, h): (usize, usize)) -> u32 {
    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
    let a = s[(x + w - 1, y + h - 1)] as i64;
    let b = s[(x - 1, y + h - 1)] as i64;
    let c = s[(x + w - 1, y - 1)] as i64;
    let d = s[(x - 1, y - 1)] as i64;
    (a - b + d - c) as u32
}

//fn ncc_sum_table_sum(s: &Array2<u32>, (x, y): (usize, usize), (w, h): (usize, usize)) -> u32 {
//    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
//    let a = s[(x + w - 1, y + h - 1)] as i64;
//    let b = if x == 0 {
//        0
//    } else {
//        s[(x - 1, y + h - 1)] as i64
//    };
//    let c = if y == 0 {
//        0
//    } else {
//        s[(x + w - 1, y - 1)] as i64
//    };
//    let d = if x == 0 || y == 0 {
//        0
//    } else {
//        s[(x - 1, y - 1)] as i64
//    };
//    (a - b + d - c) as u32
//}

fn ncc_sumsqr_table_sum_nz(s: &Array2<u64>, (x, y): (usize, usize), (w, h): (usize, usize)) -> u64 {
    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
    let a = s[(x + w - 1, y + h - 1)] as i64;
    let b = s[(x - 1, y + h - 1)] as i64;
    let c = s[(x + w - 1, y - 1)] as i64;
    let d = s[(x - 1, y - 1)] as i64;
    (a - b + d - c) as u64
}

//fn ncc_sumsqr_table_sum(s: &Array2<u64>, (x, y): (usize, usize), (w, h): (usize, usize)) -> u64 {
//    //s[(x + w - 1, y + h - 1)] - s[(x - 1, y + h - 1)] - s[(x + w - 1, y - 1)] + s[(x - 1, y - 1)]
//    let a = s[(x + w - 1, y + h - 1)] as i64;
//    let b = if x == 0 {
//        0
//    } else {
//        s[(x - 1, y + h - 1)] as i64
//    };
//    let c = if y == 0 {
//        0
//    } else {
//        s[(x + w - 1, y - 1)] as i64
//    };
//    let d = if x == 0 || y == 0 {
//        0
//    } else {
//        s[(x - 1, y - 1)] as i64
//    };
//    (a - b + d - c) as u64
//}

fn partition_by<T>(xs: &[T], pred: impl Fn(&T, &T) -> bool) -> Vec<(usize, usize)> {
    let mut it = xs.iter();
    let mut i = 0;
    let mut j = 0;
    let mut last = it.next().unwrap();
    let mut slices = vec![];
    for next in it {
        j += 1;
        if !pred(last, next) {
            slices.push((i, j));
            i = j;
            last = next;
        }
    }
    slices.push((i, j + 1));
    slices
}
