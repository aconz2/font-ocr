# Font OCR

A specialized tool that does OCR on text rendered by a computer into an image (and then possibly included in a PDF) with a known font.

It looks at a series of rectangles located at `(x, y + i * line_advance)` of size `(width, line_height)`. For each rectangle, try rendering each letter in the given alphabet with FreeType, pick the one with the least squared distance to the reference. Advance the cursor to the right and repeat. Assumes black text on white background. An all white rectangle is skipped. Should work on more fonts than just monospace but only tested with Courier New right now.

# Example

Requires `pdfimages` from `poppler-utils` and `Courier New.otf`

```bash
# wget https://github.com/maseyyi/font-courier-new/raw/refs/heads/master/Courier%20New.otf
# let's say you have a certain input.pdf with embedded images
mkdir -p imgs verify
pdfimages input.pdf imgs/page
cargo run --release -- -i imgs/* -f 'Courier New.otf' -t 13 -k 1.125 -x 45 -y 39 -w 608 --line-height 12 --line-advance 15 --verify verify > out

# cleanup out file by removing garbage from beginning and end
# manually add first line as
# > JVBERi0xLjUNJeLjz9MNCjM0IDAgb2JqDTw8L0xpbmVhcm16ZWQgMS9MIDI3NjAyOC9PIDM2L0Ug
sed 's/> //' out | base64 -d > attachment.pdf
```

Since we passed `--verify`, we generate an overlaid image of the reference image in the red channel and the rendered text in the blue channel, as well as print out the mean squared error between the two channels. You can then open this in eg. GIMP and choose Colors > Components > Decompose to split RGB to layers. Page 74 stands out as particularly bad for some reason, about 3 times the MSE as the others.

# Usage

Add the `--test mytest` flag to the invocation to create mytest-rect.png and mytest-text.png

This draws red rectangles of where will be processed and red text on top of the first processed rectangle be able to find the parameters.

# Notes

I couldn't get the example to work with the `ttf`, only `otf`. The recreated rendered text is very close but not identical, I assume either because of a different font file or the original was likely rendered on Windows.

I am not a font rendering expert.

I thought there should be a way to only render a small character-sized canvas and then check against the corresponding part of the reference image, but instead I render to a line-sized canvas and check the whole line. This seems excessive but doing the former is tricky and I couldn't figure it out.

I made no attempts to cache rendered glyphs. This would work better if rendering to a character-sized canvas, though the cache would still need to be keyed off `(glyph, subpixel_position)` (is that what real renderers do?). In general, caching a glyph per position on the line is possible, but could get large in size if the font isn't monospace.

There is no font shaping.

A kerning scaler proved essential to getting this to work in the example. This eliminated a lot of easier looking API's.
