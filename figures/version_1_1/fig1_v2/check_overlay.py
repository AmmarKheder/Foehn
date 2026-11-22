#!/usr/bin/env python3
"""Overlay check of fig_gc1_schematic_rt2 (version 2 of the lower band): are panels a, b, c the Nature figure, pixel for pixel?

A. Render of the original against the new figure (the proof). fig_gc1_schematic.main() is run as it is (only
   its save() is redirected to check/, the Nature folder is never written) and saved as a whole page, 183 x 136 mm at
   300 dpi, without tight crop: check/orig_full_300.png. The new figure, built by fig_gc1_schematic_rt2.build(), is saved
   the same way: check/new_full_300.png. The lower band adds 234 pt = 975 pixels under the old page, so a, b, c must sit on the
   same pixels counted from the top-left corner. The two images are compared over the whole old page (RGBA, every row).
   check/overlay_abc_diff.png: the new figure in grey, differing pixels in magenta, the limit of the old page dashed.
B. The published PNG of 15 October (paper/nature/figs/fig_gc1_schematic.png, 600 dpi, tight) against (i) this render of
   the original and (ii) the new figure saved with the same settings, over the rows of the published image; and the
   delivered preview fig_gc1_schematic_rt2.png (300 dpi, tight) against the original saved the same way.
C. The submitted PDF (ref_nature_final/figs/fig_gc1_schematic.pdf) against the new PDF, both rasterised by MuPDF at
   300 dpi, over the old page (the two pages share their top-left corner). Differences of a few grey levels on
   anti-aliased edges are rounding of the rasteriser; the pixels above 6 levels are listed. The merged image of the two
   small discs of a and c (331 pixels wide) must have the size and place it has in the submitted PDF: this is what
   depends on the height of the lower band (check/scan_band_height.txt).
D. Lower band: the letters d and e against the letters a, b and c (same distance from the edge of their panel, same
   offset from the title), fonts and smallest font size, pairs of labels of the band whose boxes meet, distance of the
   labels to the edges of their panel.
E. The lower band of the PDF, rasterised by MuPDF, against the lower band of the PNG preview (are the images of the
   band at the same place in the vector file?), and the size of the page.

    unset PYTHONPATH; MPLCONFIGDIR=<writable dir> /projappl/project_462001804/envs/world/bin/python -B check_overlay.py
Needs fig_gc1_schematic_rt2.pdf (run fig_gc1_schematic_rt2.py first). Writes only into check/."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_gc1_schematic_rt2 as N                               # noqa: E402  (sets nature_style.OUT to this directory, bundled fonts)
from fig_gc1_schematic_rt2 import F, plt                        # noqa: E402

CHK = HERE / "check"
REF_PDF = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/ref_nature_final/figs/fig_gc1_schematic.pdf")
PUB_PNG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/nature/figs/fig_gc1_schematic.png")
NEW_PDF = HERE / "fig_gc1_schematic_rt2.pdf"


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
    # D. labels of the lower band (boxes of the drawn texts, in mm from the lower-left corner of the page)
    r = fig.canvas.get_renderer(); k = 25.4 / fig.dpi
    band = [t for ax in fig.subfigs[1].axes for t in ax.texts if t.get_text()]
    boxes = [(t.get_text().replace("\n", " "), t.get_window_extent(r)) for t in band]
    meet = [(a, b) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
            if ba.x0 < bb.x1 and bb.x0 < ba.x1 and ba.y0 < bb.y1 and bb.y0 < ba.y1]
    gaps = sorted((round(max(bb.x0 - ba.x1, ba.x0 - bb.x1, bb.y0 - ba.y1, ba.y0 - bb.y1) * k, 2), a, b)
                  for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:])[:6]
    inside = {}
    for name, (xa, xb) in (("d", N.XD), ("e", N.XE)):                           # labels of each panel against its edges
        bs_ = [b for _, b in boxes if b.y0 * k > N.YB and xa < 0.5 * (b.x0 + b.x1) * k < xb]
        inside[name] = dict(labels=len(bs_), smallest_distance_to_the_left_edge_mm=round(min(b.x0 for b in bs_) * k - xa, 2),
                            smallest_distance_to_the_right_edge_mm=round(xb - max(b.x1 for b in bs_) * k, 2),
                            smallest_distance_to_the_lower_edge_mm=round(min(b.y0 for b in bs_) * k - N.YB, 2),
                            smallest_distance_to_the_upper_edge_mm=round(N.DY - 0.5 - max(b.y1 for b in bs_) * k, 2))
    out["band_labels"] = dict(count=len(boxes), texts=[a for a, _ in boxes], pairs_whose_boxes_meet=meet,
                              six_smallest_gaps_between_two_labels_mm=gaps, labels_against_the_edges_of_their_panel=inside,
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
    dr.text((24, 46), "(differing pixels would be magenta). Below the dashed line: panels d and e and the time strip, new.", fill=(20, 20, 20), font=font)
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
    pv = HERE / "fig_gc1_schematic_rt2.png"
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
    dC = np.abs(a - b[:a.shape[0], :a.shape[1]]).max(-1); yC, xC = np.nonzero(dC > 6)
    merged = lambda path: [(im["width"], im["height"], round(im["bbox"][1], 3), round(im["bbox"][3], 3))
                           for im in pymupdf.open(path)[0].get_image_info() if im["width"] == 331]
    out["C_details"] = dict(
        by_6_levels_or_less=int(((dC > 0) & (dC <= 6)).sum()), above_6_levels=int((dC > 6).sum()),
        pixels_above_6_levels=[dict(x_mm=round(x / 300 * 25.4, 1), y_mm_from_top=round(y / 300 * 25.4, 1), levels=int(dC[y, x])) for y, x in zip(yC, xC)],
        note="anti-aliased edges; the pixels above 6 levels lie on the shafts of two arrows (a to b; flow-time axis of the disc grid of c)",
        merged_image_of_the_two_small_discs_px_and_top_bottom_pt=merged(NEW_PDF), the_same_in_the_submitted_pdf=merged(REF_PDF))

    # ---- D. letters and fonts, read in the new PDF
    doc = pymupdf.open(NEW_PDF); pg = doc[0]
    spans = [s for blk in pg.get_text("dict")["blocks"] for ln in blk.get("lines", []) for s in ln["spans"]]
    pos = {}
    for i, s in enumerate(spans[:-1]):
        if s["text"] in ("a", "b", "c", "d", "e") and abs(s["size"] - 8) < 0.01 and "Bold" in s["font"]:
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
                                                     note="3.5 and 4.1 pt are the subscripts of PM2.5 (a-c, and d-e); labels of the lower band are 5.0 pt or more")
    # ---- E. the lower band in the vector file against the preview, and the page
    pm = pg.get_pixmap(dpi=300, alpha=False)
    v = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3).astype(np.int16)
    q = rgba(pv); q = (q[..., :3] * (q[..., 3:] / 255.0) + 255 * (1 - q[..., 3:] / 255.0)).round().astype(np.int16)
    h = min(v.shape[0], q.shape[0]); w = min(v.shape[1], q.shape[1]); y0 = a.shape[0]
    dd = np.abs(v[y0:h, :w] - q[y0:h, :w]).max(-1)
    out["E_lower_band_pdf_mupdf_vs_png_preview_300dpi"] = dict(
        pdf_px=[int(v.shape[1]), int(v.shape[0])], png_px=[int(q.shape[1]), int(q.shape[0])], rows_compared=int(h - y0),
        mean_abs_difference_levels=round(float(dd.mean()), 3), share_of_pixels_differing_by_more_than_64_levels=round(float((dd > 64).mean()), 5),
        note="two rasterisers: anti-aliased edges and resampled images differ by a few levels; a displaced image would show as a large share")
    Image.fromarray(v[y0:h, :w].astype(np.uint8)).save(CHK / "band_from_pdf_300.png")
    out["E_page"] = dict(pdf_page_mm=sb, limit_mm=[183.0 + 0.51, 225.0], band_height_mm=round(float(N.DY), 3),
                         figure_mm=[183.0, round(float(F.H_MM + N.DY), 3)])

    (CHK / "overlay_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
