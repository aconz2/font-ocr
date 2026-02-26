use clap::Parser;
use font_kit::canvas::{Canvas, Format, RasterizationOptions};
use font_kit::hinting::HintingOptions;
use font_kit::loaders::freetype::Font;
use pathfinder_geometry::rect::RectF;
use pathfinder_geometry::transform2d::Transform2F;
use pathfinder_geometry::vector::{Vector2F, Vector2I};
use petal_neighbors::BallTree;

const DEFAULT_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

#[derive(Clone, Copy)]
struct RenderOptions {
    rasterization: RasterizationOptions,
    hinting: HintingOptions,
    format: Format,
    size: f32,
}

fn render(
    font: &Font,
    char: char,
    offset: Vector2F,
    render_options: RenderOptions,
    size: Vector2I,
) -> Canvas {
    assert!(render_options.format == Format::A8);

    let glyph_id = font.glyph_for_char(char).unwrap();

    let mut canvas = Canvas::new(size, render_options.format);

    font.rasterize_glyph(
        &mut canvas,
        glyph_id,
        render_options.size,
        Transform2F::from_translation(offset),
        render_options.hinting,
        render_options.rasterization,
    )
    .unwrap();
    canvas
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

    #[arg(short, long, default_value_t = 0)]
    bits: u32,

    #[arg(long)]
    hinting: bool,

    #[arg(short, long, default_value_t=DEFAULT_ALPHABET.to_string())]
    alphabet: String,

    #[arg(long, default_value_t = 0)]
    padding_x: usize,

    #[arg(long, default_value_t = 0)]
    padding_y: usize,

    #[arg(long, default_value_t = 1000.)]
    threshold: f32,
}

fn main() {
    let args = Args::parse();

    let _padding = Vector2I::new(args.padding_x as i32, args.padding_y as i32);

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

    let offsets = {
        let mut acc = vec![];
        let divisor = 1. / 2usize.pow(args.bits) as f32;
        for x in 0..2usize.pow(args.bits) {
            for y in 0..2usize.pow(args.bits) {
                acc.push(Vector2F::new(x as f32 * divisor, y as f32 * divisor));
            }
        }
        acc
    };

    let font = Font::from_path(args.font, 0).unwrap();
    let to_px = (1. / font.metrics().units_per_em as f32) * args.text_size;

    let img = image::open(args.img.first().unwrap()).unwrap().into_luma8();
    let (r_w, r_h) = (img.width() as usize, img.height() as usize);
    let f_img: Vec<_> = img.as_raw().iter().map(|p| *p as f32 / 255.).collect();

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
                        Transform2F::default(),
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
    let y_offset = Vector2F::new(0., y_offset);
    let mut chars = vec![];
    let mut pixels = vec![];
    let row_length = (size.x() * size.y()) as usize;
    for offset in offsets {
        for char in args.alphabet.chars() {
            chars.push(char);
            let canvas = render(&font, char, offset + y_offset, render_options, size);
            pixels.extend(canvas.pixels.into_iter().map(|x| 1. - x as f32 / 255.));
        }
    }
    let points = ndarray::Array::from_shape_vec((chars.len(), row_length), pixels).unwrap();
    let tree = BallTree::euclidean(&points).unwrap();
    //let (indices, distance) = tree.query(&points.row(0), 2);
    //println!("indices {indices:?} distance {distance:?}");

    let (n_w, n_h) = (size.x() as usize, size.y() as usize);

    let t00 = std::time::Instant::now();

    let mut n_hits = 0;
    let mut query = vec![0.; row_length];
    for y in 0..(r_h - n_h + 1) {
        for x in 0..(r_w - n_w + 1) {
            load_into(&mut query, &f_img, (r_w, r_h), size, (x, y));
            //let (indices, distances) = tree.query(&ndarray::aview1(&query), 5);
            //if distances[0] < args.threshold {
            //    let matched_chars: Vec<_> = indices.into_iter().map(|i| chars[i]).collect();
            //    println!("x={x} y={y} matched={matched_chars:?} distances={distances:?}");
            //}
            //let t0 = std::time::Instant::now();
            let (index, distance) = tree.query_nearest(&ndarray::aview1(&query));
            //let t1 = std::time::Instant::now();
            //eprintln!("query {}", (t1 - t0).as_millis());
            if distance < args.threshold {
                n_hits += 1;
                println!(
                    "{},{}",
                    x as f32 + (n_w as f32 / 2.),
                    y as f32 + (n_h as f32 / 2.)
                );
            }
        }
    }
    let t11 = std::time::Instant::now();
    eprintln!("overall {}", (t11 - t00).as_millis());
    eprintln!("{n_hits} hits");
}

fn load_into(
    out: &mut [f32],
    src: &[f32],
    (src_w, _src_h): (usize, usize),
    needle_size: Vector2I,
    (x, y): (usize, usize),
) {
    let mut i = 0;
    for dy in 0..(needle_size.y() as usize) {
        for dx in 0..(needle_size.x() as usize) {
            out[i] = src[(y + dy) * src_w + (x + dx)];
            i += 1;
        }
    }
}
