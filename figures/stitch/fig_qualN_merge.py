# !/usr/bin/env python3
"""Join selected N=50 and N=2,500 qualitative panels.

The shared RGB/reference/teacher columns must be pixel-identical; the script
checks this before joining model columns. Promote the selected panels to
qual_N50.png and qual_N2500.png under assets/sturm_orig/figures/ first.
Offsets were measured for that layout. Recheck TOP_CUT, SHARED, BAND, GAP
and RELABEL when changing DPI, font size or columns."""
from __future__ import annotations
import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path("assets/sturm_orig/figures")
# px, measured on the source-repo panels and asserted below. Rows 0-64 are the per-panel
# title, which is dropped (it showed the budget, the new header band does that now). Below
# it the first 1,668 columns (row labels, RGB, reference, teacher) are byte-identical in both
# panels. Recalibrate on this repo's own qual_N50.png/qual_N2500.png once they exist.
TOP_CUT, SHARED = 65, 1668
BAND = 62                          # px: height of the added "N = ..." header band
# The squeeze centres each gutter's surviving white on the gutter, but the block boundary
# x1 lands at that gutter's RIGHT edge, so the rule ends up flush against the first N=2,500
# column (measured: 13 px clear on the left, 0 px on the right). GAP px of white are spliced
# in immediately after the rule to restore a symmetric margin.
GAP = 14
INK, RULE = (38, 38, 38), (150, 150, 150)


def _pt(ink_h):
    """Nominal font size from a measured ink height.

        The measured run spans cap-height to baseline, roughly 0.72 of the nominal size, so
        passing the ink height straight to truetype() renders visibly smaller type than the
        neighbouring headers it has to match.
    """
    return max(8, int(round(ink_h / 0.72)))


def _font(px, bold=True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (f"/usr/share/fonts/truetype/dejavu/{name}",
              f"/usr/share/fonts/dejavu/{name}"):
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def _squeeze_gutters(im, keep=26, floor=46):
    """Narrow the all-white vertical bands between columns to `keep` px.

        Eleven columns at the source panels' own margins do not fit a two-column page at a
        readable size. Only columns that are white over the FULL image height are touched, and
        only where the run is wider than `floor`, so this can never cut into a mask or into the
        relabelled headers (both leave ink somewhere in the column). Returns the squeezed image
        and the boolean keep-mask, which is what maps a pre-squeeze x to a post-squeeze x.
    """
    white = (im > 250).all(axis=(0, 2))
    drop = np.zeros(im.shape[1], bool)
    st = None
    for i, v in enumerate(np.append(white, False)):
        if v and st is None:
            st = i
        elif not v and st is not None:
            if i - st > floor:
                lo = st + keep // 2
                drop[lo:lo + (i - st - keep)] = True
            st = None
    return im[:, ~drop], ~drop


def _header_rows(im, x0, x1):
    """Row spans of the FIRST header line above each of the four scene rows."""
    ink = (im[:, x0:x1] < 250).any(axis=2).any(axis=1)
    runs, st = [], None
    for i, v in enumerate(ink):
        if v and st is None:
            st = i
        if not v and st is not None:
            runs.append((st, i - 1)); st = None
    # header lines are short runs; the masks are tall. Two short runs precede each mask
    # (model name, then flood-IoU) -- take the first of each pair.
    short = [r for r in runs if r[1] - r[0] < 40]
    assert len(short) % 2 == 0 and len(short) >= 8, f"unexpected header structure: {short}"
    return short[0::2]


def _ink_span(im, r0, r1, x0, x1):
    cols = (im[r0:r1 + 1, x0:x1] < 250).any(axis=2).any(axis=0)
    nz = np.nonzero(cols)[0]
    return (x0 + int(nz[0]), x0 + int(nz[-1])) if len(nz) else (x0, x1)


# Column headers rewritten in place. The parameter counts are dropped from the EfficientViT
# columns (they are in the text, and the long strings overflowed once the gutters were
# narrowed) and the first column is named by its label source, as everywhere else in the
# manuscript. The UNet column keeps its 32.5 M, which is the contrast the figure is making.
RELABEL = {0: "EViT, Sen1Floods11", 1: "EViT-LogitKD", 3: "EViT-OFA (ours)"}


def _relabel(im, d, block_x0, block_w):
    """Rewrite the model-name header line of the columns named in RELABEL."""
    cw = block_w // 4
    for ci, text in RELABEL.items():
        x0, x1 = block_x0 + ci * cw, block_x0 + (ci + 1) * cw
        for r0, r1 in _header_rows(im, x0 + 10, x1 - 10):
            span = _ink_span(im, r0, r1, x0 + 3, x1 - 3)
            d.rectangle([x0 + 2, r0 - 3, x1 - 2, r1 + 2], fill=(255, 255, 255))
            d.text(((span[0] + span[1]) // 2, r1), text,
                   font=_font(_pt(r1 - r0 + 1), bold=False), fill=INK, anchor="ms")


def main():
    for name in ("qual_N50.png", "qual_N2500.png"):
        if not (SRC / name).exists():
            sys.exit(f"{SRC / name} not found -- this figure needs a human-promoted panel "
                     "(see module docstring's PRECONDITION note), not just a script run.")
    a = np.array(Image.open(SRC / "qual_N50.png").convert("RGB"))
    b = np.array(Image.open(SRC / "qual_N2500.png").convert("RGB"))
    assert a.shape == b.shape, f"panels differ in size: {a.shape} vs {b.shape}"
    a, b = a[TOP_CUT:], b[TOP_CUT:]
    assert np.array_equal(a[:, :SHARED], b[:, :SHARED]), \
        "the shared columns are not identical -- the two panels no longer show the same scenes"
    assert not np.array_equal(a[:, SHARED:SHARED + 40], b[:, SHARED:SHARED + 40]), \
        "SHARED reaches into a column the two panels disagree on -- recheck the cut"

    w = a.shape[1]
    out = np.concatenate([a[:, :SHARED], a[:, SHARED:], b[:, SHARED:]], axis=1)

    # Relabel FIRST, while the column geometry is still the source panels' own; squeezing
    # the gutters afterwards moves every x and would invalidate it.
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    blk = w - SHARED
    for bi in range(2):
        _relabel(np.array(im), d, SHARED + bi * blk, blk)
    out = np.array(im)

    out, keep = _squeeze_gutters(out)
    x1 = int(keep[:SHARED + blk].sum())          # block boundary, post-squeeze
    x0 = int(keep[:SHARED].sum())
    out = np.concatenate([out[:, :x1],
                          np.full((out.shape[0], GAP, 3), 255, np.uint8),
                          out[:, x1:]], axis=1)
    out = np.concatenate([np.full((BAND, out.shape[1], 3), 255, np.uint8), out], axis=0)

    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    f = _font(38)
    for lab, lo, hi in (("N = 50   (data-scarce)", x0, x1),
                        ("N = 2,500   (deployment operating point)",
                         x1 + GAP, out.shape[1])):
        d.text(((lo + hi) // 2, BAND - 20), lab, font=f, fill=INK, anchor="ms")
        d.line([(lo + 8, BAND - 6), (hi - 8, BAND - 6)], fill=RULE, width=3)
    d.line([(x1, 4), (x1, out.shape[0] - 4)], fill=RULE, width=3)

    # The margin either side of the rule must now be non-zero, or the splice missed.
    a2 = np.array(im)[BAND:]
    ink = (a2 < 250).any(axis=2).any(axis=0)
    lft = x1 - 2 - max(i for i in range(x1 - 1) if ink[i])
    rgt = min(i for i in range(x1 + 2, len(ink)) if ink[i]) - (x1 + 1)
    assert lft > 4 and rgt > 4, f"rule still touches a column: {lft} px left, {rgt} px right"
    print(f"[chk] rule clearance {lft} px left / {rgt} px right")

    p = SRC / "qual_N50_N2500_merged.png"
    im.save(p)
    print(f"[ok] {p}  {im.size[0]}x{im.size[1]}")


if __name__ == "__main__":
    main()
