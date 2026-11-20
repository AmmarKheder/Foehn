#!/usr/bin/env python3
"""Overlay check of fig_gc1_schematic_rt: are panels a, b, c the Nature figure, pixel for pixel?

A. Render of the original against the new figure (the proof asked for). fig_gc1_schematic.main() is run as it is (only
   its save() is redirected to check/, the Nature folder is never written) and saved as a whole page, 183 x 136 mm at
   300 dpi, without tight crop: check/orig_full_300.png. The new figure, built by fig_gc1_schematic_rt.build(), is saved
   the same way: check/new_full_300.png. Band d adds 192 pt = 800 pixels under the old page, so a, b, c must sit on the
   same pixels counted from the top-left corner. The two images are compared over the whole old page (RGBA, every row).
   check/overlay_abc_diff.png: the new figure in grey, differing pixels in magenta, the limit of the old page dashed.
B. The published PNG of 15 October (paper/nature/figs/fig_gc1_schematic.png, 600 dpi, tight) against (i) this render of
   the original and (ii) the new figure saved with the same settings, over the rows of the published image; and the
   delivered preview fig_gc1_schematic_rt.png (300 dpi, tight) against the original saved the same way.
C. The submitted PDF (ref_nature_final/figs/fig_gc1_schematic.pdf) against the new PDF, both rasterised by MuPDF at
   300 dpi, over the old page (the two pages share their top-left corner). Differences of a few grey levels on
   anti-aliased edges are rounding of the rasteriser; the count above 8 levels is the one that matters.
D. Band d: the letter d against the letters a and c (same left edge, same offset from the title), fonts and smallest
   font size, pairs of labels of the band whose boxes meet.

    unset PYTHONPATH; MPLCONFIGDIR=<writable dir> /projappl/project_462001804/envs/world/bin/python -B check_overlay.py
Needs fig_gc1_schematic_rt.pdf (run fig_gc1_schematic_rt.py first). Writes only into check/."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_gc1_schematic_rt as N                                # noqa: E402  (sets nature_style.OUT to this directory, bundled fonts)
from fig_gc1_schematic_rt import F, plt                         # noqa: E402

CHK = HERE / "check"
REF_PDF = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/ref_nature_final/figs/fig_gc1_schematic.pdf")
PUB_PNG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/nature/figs/fig_gc1_schematic.png")
NEW_PDF = HERE / "fig_gc1_schematic_rt.pdf"


def rgba(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.int16)


def compare(a, b):
    d = np.abs(a - b).max(-1)
    return dict(pixels=int(d.size), differing=int((d > 0).sum()), differing_by_more_than_8_levels=int((d > 8).sum()),
                max_abs_difference=int(d.max())), d > 0


def main():
    import matplotlib
    import pymupdf
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_dilation
    CHK.mkdir(exist_ok=True)
    out = dict(environment=dict(python=sys.version.split()[0], matplotlib=matplotlib.__version__, numpy=np.__version__,
                                liberation_sans=sorted({f.fname for f in N._fm.fontManager.ttflist if f.name == "Liberation Sans"})))

    # ---- the original, unchanged, saved to check/ only
    def save_orig(fig, name):
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(CHK / "orig_full_300.png", dpi=300)
        fig.savefig(CHK / "_orig_tight_600.png")                                 # as nature_style.save(): tight, 600 dpi
        fig.savefig(CHK / "_orig_tight_300.png", dpi=300)                        # as the delivered preview
    keep = F.save
    F.save = save_orig
    try:
        F.main()
    finally:
        F.save = keep
    plt.close("all")

    # ---- the new figure, same page settings
    fig = N.build()
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(CHK / "new_full_300.png", dpi=300)
    fig.savefig(CHK / "_new_tight_600.png")
    # D. labels of band d (boxes of the drawn texts, in mm from the lower-left corner of the page)
    r = fig.canvas.get_renderer(); k = 25.4 / fig.dpi
    band = [t for ax in fig.subfigs[1].axes for t in ax.texts if t.get_text()]
    boxes = [(t.get_text().replace("\n", " "), t.get_window_extent(r)) for t in band]
    meet = [(a, b) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
            if ba.x0 < bb.x1 and bb.x0 < ba.x1 and ba.y0 < bb.y1 and bb.y0 < ba.y1]
    out["band_d_labels"] = dict(count=len(boxes), pairs_whose_boxes_meet=meet,
                                smallest_font_pt=min(t.get_fontsize() for t in band), largest_font_pt=max(t.get_fontsize() for t in band),
                                right_edge_of_the_rightmost_label_mm=round(max(b.x1 for _, b in boxes) * k, 2),
                                left_edge_of_the_leftmost_label_mm=round(min(b.x0 for _, b in boxes) * k, 2))
    plt.close("all")

    # ---- A. overlay over the whole old page
    o, n = rgba(CHK / "orig_full_300.png"), rgba(CHK / "new_full_300.png")
    H0 = o.shape[0]
    res, diff = compare(o, n[:H0])
    out["A_render_of_the_original_vs_new_figure_300dpi_whole_old_page"] = dict(
        original_px=list(o.shape[:2][::-1]), new_px=list(n.shape[:2][::-1]), rows_compared=H0, offset_from_the_top_left_corner_px=[0, 0],
        added_below_px=int(n.shape[0] - H0), added_below_mm=round(float(N.DY), 3), **res)
    grey = (255 - 0.45 * (255 - n[..., :3].mean(-1)) * (n[..., 3] / 255.0)).clip(0, 255).astype(np.uint8)
    vis = np.repeat(grey[..., None], 3, -1)
    m = np.zeros(n.shape[:2], bool); m[:H0] = binary_dilation(diff, iterations=2)
    vis[m] = (230, 0, 120)
    dash = (np.arange(n.shape[1]) // 9) % 2 == 0
    vis[H0 - 1:H0 + 1, dash] = (30, 30, 30)
    head = np.full((84, n.shape[1], 3), 255, np.uint8)                           # a header line above the page
    im = Image.fromarray(np.vstack([head, vis])); dr = ImageDraw.Draw(im)
    from PIL import ImageFont
    font = ImageFont.truetype(str(N.FIG / "fonts" / "LiberationSans-Regular.ttf"), 30)
    dr.text((24, 12), f"Overlay, 300 dpi. Above the dashed line: the page of the original figure, where {res['differing']} of {res['pixels']:,} "
                      f"pixels differ from a render of the original", fill=(20, 20, 20), font=font)
    dr.text((24, 46), "(differing pixels would be magenta). Below the dashed line: band d, new.", fill=(20, 20, 20), font=font)
    im.save(CHK / "overlay_abc_diff.png")

    # ---- B. the published PNG of 15 October
    if PUB_PNG.exists():
        p, q, w = rgba(PUB_PNG), rgba(CHK / "_orig_tight_600.png"), rgba(CHK / "_new_tight_600.png")
        B = dict(published_png=str(PUB_PNG), published_px=list(p.shape[:2][::-1]), render_of_the_original_px=list(q.shape[:2][::-1]),
                 new_px=list(w.shape[:2][::-1]))
        if p.shape == q.shape:
            B["published_vs_render_of_the_original"] = compare(p, q)[0]
        if p.shape[1] == w.shape[1] and w.shape[0] >= p.shape[0]:
            B["published_vs_new_figure_rows_of_the_published_image"] = compare(p, w[:p.shape[0]])[0]
        out["B_published_png_600dpi_tight"] = B
    pv = HERE / "fig_gc1_schematic_rt.png"
    if pv.exists():
        e, g = rgba(CHK / "_orig_tight_300.png"), rgba(pv)
        if e.shape[1] == g.shape[1] and g.shape[0] >= e.shape[0]:
            out["B_delivered_preview_300dpi_tight_vs_render_of_the_original"] = dict(
                preview=str(pv), preview_px=list(g.shape[:2][::-1]), original_px=list(e.shape[:2][::-1]), **compare(e, g[:e.shape[0]])[0])
    for f in ("_orig_tight_600.png", "_new_tight_600.png", "_orig_tight_300.png"):   # only needed here
        (CHK / f).unlink()

    # ---- C. the vector files, rasterised by the same engine
    def pdf_rgb(path, dpi=300):
        pg = pymupdf.open(path)[0]
        pm = pg.get_pixmap(dpi=dpi, alpha=False)
        return (np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3).astype(np.int16),
                [round(pg.rect.width / 72 * 25.4, 2), round(pg.rect.height / 72 * 25.4, 2)])
    a, sa = pdf_rgb(REF_PDF); b, sb = pdf_rgb(NEW_PDF)
    out["C_submitted_pdf_vs_new_pdf_mupdf_300dpi_whole_old_page"] = dict(
        submitted_pdf=str(REF_PDF), submitted_page_mm=sa, new_page_mm=sb, rows_compared=int(a.shape[0]),
        **compare(a, b[:a.shape[0], :a.shape[1]])[0])

    # ---- D. letters and fonts, read in the new PDF
    doc = pymupdf.open(NEW_PDF); pg = doc[0]
    spans = [s for blk in pg.get_text("dict")["blocks"] for ln in blk.get("lines", []) for s in ln["spans"]]
    pos = {}
    for i, s in enumerate(spans[:-1]):
        if s["text"] in ("a", "b", "c", "d") and abs(s["size"] - 8) < 0.01 and "Bold" in s["font"]:
            t = spans[i + 1]                                                      # the panel title follows its letter
            pos[s["text"]] = dict(letter_left_edge_mm=round(s["bbox"][0] / 72 * 25.4, 3), title=t["text"],
                                  title_left_edge_mm=round(t["bbox"][0] / 72 * 25.4, 3),
                                  letter_baseline_below_title_baseline_pt=round(s["origin"][1] - t["origin"][1], 3))
    out["D_panel_letters_in_the_new_pdf"] = pos
    import io
    from fontTools.ttLib import TTFont
    fonts = []
    for f in doc.get_page_fonts(0):
        buf = doc.extract_font(f[0])[3]
        fonts.append(dict(name=f[3].split("+")[-1], type=f[2], embedded_as=f[1], revision=round(TTFont(io.BytesIO(buf), lazy=True)["head"].fontRevision, 3)))
    out["D_fonts_in_the_new_pdf"] = fonts
    out["D_smallest_font_in_the_new_pdf_pt"] = dict(whole_page=round(min(s["size"] for s in spans), 2),
                                                     note="3.5 and 4.1 pt are the subscripts of PM2.5 (a-c and d); labels of band d are 5.0 pt or more")

    (CHK / "overlay_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
