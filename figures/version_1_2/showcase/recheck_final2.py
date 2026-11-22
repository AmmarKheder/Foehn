#!/usr/bin/env python3
"""RECHECK (2026-11-21) of the second versions of Fig. 3 and Fig. 5 in N/rebuild/figs_show/final2. Independent code; read only
on every input. Writes only: print/show/recheck_final2.json, figs_show/final2/sheet_fig3.png, figs_show/final2/sheet_fig5.png
(temporary file + fsync + rename + read-back).
  A  Fig. 3 numbers from analyses/circularity/data_ar/station_day.npz (paper mask, independent of compare_fair's sums):
     RMSE by lead (six networks, China, outside China) and the six by-network reductions against CAMS 1 day ahead with the
     paper's bootstrap (compare_fair.py: default_rng(0), blocks of 7 start days, 10,000 resamples, 2.5/97.5 percentiles);
  B  Fig. 3 v2 PDF geometry: what is drawn in c, d (log axes, curves) and e (bars, whiskers, hatching) read back in data units;
  C  specifications of both PDFs (page, fonts, text sizes, line widths, raster dpi);
  D  identity with the first final versions (rasters bit for bit, renders pixel by pixel outside the panels meant to change);
  E  Fig. 5: crop files <-> keys (start day = target - lead), box levels / RMSE / ranks / probabilities, spectra counts, u band
     and axis read back from the PDF, CNEMC stations of the plain;
  F  contact sheets (present manuscript figure left, second version right, same scale 200 dpi, labels outside).
    FOEHN_DESIGN=ar MPLCONFIGDIR=<scratch> OMP_NUM_THREADS=4 python -I -B recheck_final2.py"""
import glob
import hashlib
import io
import json
import re
import os
from pathlib import Path

import numpy as np
import pymupdf as fz
import zarr
from PIL import Image, ImageDraw, ImageFont

N = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms"))
F1, F2, WF = N / "rebuild/figs_show/final", N / "rebuild/figs_show/final2", N / "work/figs"
OUTJ = N / "rebuild/print/show/recheck_final2.json"
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
CASE = RUN / "eval_case_eastasia_crps_e01_ar"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
AURD = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/benchmark/aurora"))
CAMSZ = (os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr")
R = {}


def atomic(path, data):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    assert path.read_bytes() == data, path


# ---------------------------------------------------------------- A: Fig. 3 numbers (station_day.npz, paper mask)
sd = np.load(N / "analyses/circularity/data_ar/station_day.npz")
P, O, src = sd["paper"], sd["obs"], sd["src"]
NETC = {1: "China", 3: "USA", 5: "Japan", 0: "Europe", 6: "India", 4: "Korea"}
SYS = {"Foehn": "foehn", "CAMS": "cams1", "Aurora": "aur1"}
E2, NN = {}, {}
for s, v in SYS.items():
    X = sd[v]
    assert np.isfinite(X[P]).all() and np.isfinite(O[P]).all()
    e2 = np.where(P, (X.astype(np.float64) - O) ** 2, 0.0)
    E2[s] = np.stack([e2[:, :, src == g].sum(-1) for g in NETC], -1)          # (day, lead, network)
    NN[s] = np.stack([P[:, :, src == g].sum(-1) for g in NETC], -1).astype(np.float64)
NETS = list(NETC.values())
GRP = {"six networks": NETS, "China": ["China"], "outside China": [n for n in NETS if n != "China"]}
gi = lambda names: [NETS.index(n) for n in names]
R["A_band_rmse"] = {}
for gname, names in GRP.items():
    k = gi(names)
    R["A_band_rmse"][gname] = {s: [round(float(np.sqrt(E2[s][:, h][:, k].sum() / NN[s][:, h][:, k].sum())), 4) for h in range(4)] for s in SYS}
    R["A_band_rmse"][gname]["station_days"] = [int(NN["Foehn"][:, h][:, k].sum()) for h in range(4)]
    assert all(NN[s][:, :, k].sum() == NN["Foehn"][:, :, k].sum() for s in SYS)
fz_ = np.load(FIG / "data/compare_fair_crps_e01_ar.npz")
okd_cf = np.isfinite(fz_["PD"][0, :, 0, 0])
okd = sd["used"][:, 0]
R["A_days"] = {"station_day_days": int(len(sd["days"])), "used_lead1": int(okd.sum()), "compare_fair_ok_d": int(okd_cf.sum()),
               "same_days": bool(np.array_equal(sd["days"], fz_["days"]) and np.array_equal(okd, okd_cf))}
rng = np.random.default_rng(0)                                  # the paper's bootstrap (compare_fair.py lines 181-191)
nd, B = int(okd.sum()), 7
idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]


def reduction(k, h, ref):
    a0, a1 = E2["Foehn"][okd, h][:, k].sum(-1), NN["Foehn"][okd, h][:, k].sum(-1)
    b0, b1 = E2[ref][okd, h][:, k].sum(-1), NN[ref][okd, h][:, k].sum(-1)
    ratio = lambda ix: np.sqrt(a0[ix].sum(-1) / a1[ix].sum(-1)) / np.sqrt(b0[ix].sum(-1) / b1[ix].sum(-1)) - 1
    bs = ratio(idx); pt = ratio(np.arange(nd))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"reduction_pct": round(float(-100 * pt), 3), "ci95_pct": [round(float(-100 * hi), 3), round(float(-100 * lo), 3)],
            "contains_zero": bool(-100 * hi <= 0 <= -100 * lo)}


R["A_e_by_network_vs_CAMS_lead1"] = {n: reduction(gi([n]), 0, "CAMS") for n in NETS}
R["A_pooled"] = {g: {ref: [reduction(gi(names), h, ref) for h in range(4)] for ref in ("CAMS", "Aurora")} for g, names in GRP.items()}
kc, ka = gi(["China"]), gi(NETS)
R["A_china_share"] = {
    "station_days_pct": [round(float(100 * NN["Foehn"][:, h][:, kc].sum() / NN["Foehn"][:, h][:, ka].sum()), 2) for h in range(4)],
    "sq_error_reduction_vs_CAMS_pct": [round(float(100 * (E2["CAMS"][:, h][:, kc].sum() - E2["Foehn"][:, h][:, kc].sum())
                                                    / (E2["CAMS"][:, h][:, ka].sum() - E2["Foehn"][:, h][:, ka].sum())), 1) for h in range(4)]}
# comparison with the json of the figure script
W3 = json.loads((F2 / "fig3_numbers.json").read_text())
cmp = {}
for n, v in R["A_e_by_network_vs_CAMS_lead1"].items():
    w = W3["band_e"]["values"][n]
    cmp[n] = {"recomputed": [v["reduction_pct"], *v["ci95_pct"]], "figure_numbers": [w["reduction_pct"], *w["ci95_pct"]],
              "max_abs_diff": round(max(abs(a - b) for a, b in zip([v["reduction_pct"], *v["ci95_pct"]], [w["reduction_pct"], *w["ci95_pct"]])), 4),
              "hatched_recomputed": v["contains_zero"], "hatched_figure_numbers": n in W3["band_e"]["hatched_lighter"]}
R["A_e_vs_fig3_numbers"] = cmp
R["A_band_vs_fig3_numbers"] = {p: {s: max(abs(a - b) for a, b in zip(R["A_band_rmse"][g][s], W3["band"][p][nm]["rmse"]))
                                  for s, nm in (("Foehn", "Foehn (1 km)"), ("CAMS", "CAMS forecast (40 km)"), ("Aurora", "Aurora (40 km)"))}
                              for p, g in (("Stations of the six networks", "six networks"), ("China", "China"))}

# ---------------------------------------------------------------- B: Fig. 3 v2 geometry read back from the PDF
def spans(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip():
                    out.append(dict(t=s["text"], size=s["size"], font=s["font"], bbox=tuple(s["bbox"])))
    return out


close = lambda c, ref, tol=0.02: c is not None and all(abs(a - b) < tol for a, b in zip(c, ref))
FOE, CAM, AUR, BARC = (0x1F / 255, 0x5A / 255, 0xA6 / 255), (0x11 / 255,) * 3, (0xD5 / 255, 0x5E / 255, 0), (0x2F / 255, 0x6F / 255, 0xB7 / 255)
pg3 = fz.open(F2 / "fig_overview_final2.pdf")[0]
dr3, sp3 = pg3.get_drawings(), spans(pg3)
band_y0 = min(s["bbox"][1] for s in sp3 if s["t"] == "Lead time (days)") - 90
spines = sorted([g for g in dr3 if g["rect"].y0 > band_y0 and len(g["items"]) == 1 and g["items"][0][0] == "l" and g.get("color") is not None
                 and abs(g["rect"].x1 - g["rect"].x0) < 0.01 and 60 < g["rect"].y1 - g["rect"].y0 < 80], key=lambda g: g["rect"].x0)
assert len(spines) == 3, [g["rect"] for g in spines]
geo = {}
for name, sp, labels in (("c", spines[0], ("12", "15", "20")), ("d", spines[1], ("15", "20", "25")), ("e", spines[2], ("0", "20", "40"))):
    x, top, bot = sp["rect"].x0, sp["rect"].y0, sp["rect"].y1
    ticks = sorted(g["rect"].y0 for g in dr3 if len(g["items"]) == 1 and g["items"][0][0] == "l" and abs(g["rect"].x1 - x) < 0.05
                   and 1.5 < g["rect"].x1 - g["rect"].x0 < 2.5 and abs(g["rect"].y1 - g["rect"].y0) < 0.01 and top - 0.5 <= g["rect"].y0 <= bot + 0.5)
    lab = sorted([s for s in sp3 if s["t"] in labels and abs(s["bbox"][2] - x) < 6 and top - 3 < (s["bbox"][1] + s["bbox"][3]) / 2 < bot + 3],
                 key=lambda s: s["bbox"][1])
    assert len(ticks) == 3 and len(lab) == 3, (name, ticks, [s["t"] for s in lab])
    vals = [float(s["t"]) for s in lab]                                    # top to bottom, ticks top to bottom
    if name in "cd":
        A = np.vstack([np.ones(3), np.log10(vals)]).T; coef, res, *_ = np.linalg.lstsq(A, np.array(ticks), rcond=None)
        v_of = lambda y, c=coef: 10 ** ((y - c[0]) / c[1])
        resid = float(np.max(np.abs(A @ coef - np.array(ticks))))
        ytop, ybot = v_of(top), v_of(bot)
        g = {"axis_pt": [round(top, 2), round(bot, 2)], "ticks": dict(zip(labels[::-1], [round(t, 2) for t in ticks])),
             "log_fit_max_residual_pt": round(resid, 4), "y_range_read": [round(ybot, 3), round(ytop, 3)], "top_bottom_ratio": round(ytop / ybot, 4)}
        curves = {}
        for sname, col, w in (("Foehn", FOE, 1.5), ("CAMS", CAM, 0.8), ("Aurora", AUR, 0.8)):
            cand = [d for d in dr3 if d["rect"].x0 > x and d["rect"].x1 < x + 160 and d["rect"].y0 > top - 1 and d["rect"].y1 < bot + 1
                    and close(d.get("color"), col) and d.get("width") and abs(d["width"] - w) < 0.01 and len(d["items"]) == 3]
            assert len(cand) == 1, (name, sname, len(cand))
            pts = [cand[0]["items"][0][1]] + [it[2] for it in cand[0]["items"]]
            curves[sname] = [round(float(v_of(p.y)), 3) for p in pts]
        ref = [d for d in dr3 if d["rect"].x0 > x and d["rect"].x1 < x + 160 and close(d.get("color"), CAM) and d.get("width")
               and abs(d["width"] - 0.6) < 0.01 and len(d["items"]) == 1]
        g["drawn_values"] = curves
        g["dotted_line_value"] = round(float(v_of(ref[0]["rect"].y0)), 3) if ref else None
        gname = {"c": "six networks", "d": "China"}[name]
        g["max_abs_drawn_minus_recomputed"] = round(max(abs(a - b) for s in curves for a, b in zip(curves[s], R["A_band_rmse"][gname][s])), 4)
        lab_txt = [s for s in sp3 if abs(s["size"] - 5.3) < 0.01 and x < s["bbox"][0] < x + 165]
        g["printed_values"] = sorted(s["t"] for s in lab_txt)
        exp = sorted(f"{R['A_band_rmse'][gname][s][h]:.1f}" for s in SYS for h in (0, 3))
        g["printed_equal_recomputed_rounded"] = g["printed_values"] == exp
        geo[name] = g
    else:
        A = np.vstack([np.ones(3), vals]).T; coef, *_ = np.linalg.lstsq(A, np.array(ticks), rcond=None)
        v_of = lambda y, c=coef: (y - c[0]) / c[1]
        xl = {s["t"]: (s["bbox"][0] + s["bbox"][2]) / 2 for s in sp3 if s["t"] in NETS and s["bbox"][1] > bot - 2}
        bars = {}
        for d in dr3:
            r = d["rect"]
            if not (r.x0 > x and r.y0 > top - 1 and r.y1 < bot + 1 and r.x1 - r.x0 > 8 and len(d["items"]) == 1 and d["items"][0][0] == "re"):
                continue
            solid = d["type"] == "f" and close(d.get("fill"), BARC)
            hatched = d["type"] == "s" and close(d.get("color"), BARC)
            if solid or hatched:
                n = min(xl, key=lambda k: abs(xl[k] - (r.x0 + r.x1) / 2))
                y0v, y1v = v_of(r.y1), v_of(r.y0)
                bars[n] = {"style": "hatched" if hatched else "solid", "height": round(float(y1v if abs(y0v) < abs(y1v) else y0v), 3),
                           "base": round(float(y0v if abs(y0v) < abs(y1v) else y1v), 3)}
        for d in dr3:
            r = d["rect"]
            if (r.x0 > x and len(d["items"]) == 1 and d["items"][0][0] == "l" and close(d.get("color"), (0.2, 0.2, 0.2), 0.01)
                    and abs(r.x1 - r.x0) < 0.01 and r.y1 - r.y0 > 1):
                n = min(xl, key=lambda k: abs(xl[k] - r.x0))
                bars[n]["whisker"] = [round(float(v_of(r.y1)), 3), round(float(v_of(r.y0)), 3)]
        pl = {}
        for s in sp3:
            if abs(s["size"] - 5.5) < 0.01 and x < s["bbox"][0] and s["bbox"][3] < bot - 2 and s["t"].replace("−", "").isdigit() and s["bbox"][0] > x + 3:
                n = min(xl, key=lambda k: abs(xl[k] - (s["bbox"][0] + s["bbox"][2]) / 2))
                pl[n] = {"text": s["t"], "bold": "Bold" in s["font"]}
        rec = R["A_e_by_network_vs_CAMS_lead1"]
        for n in bars:
            bars[n]["printed"] = pl.get(n)
            bars[n]["recomputed"] = [rec[n]["reduction_pct"], *rec[n]["ci95_pct"]]
            bars[n]["max_abs_drawn_minus_recomputed"] = round(max(abs(bars[n]["height"] - rec[n]["reduction_pct"]),
                                                                  *[abs(a - b) for a, b in zip(bars[n]["whisker"], rec[n]["ci95_pct"])]), 3)
            bars[n]["printed_ok"] = pl[n]["text"] == f"{rec[n]['reduction_pct']:.0f}".replace("-", "−")
            bars[n]["hatched_ok"] = (bars[n]["style"] == "hatched") == rec[n]["contains_zero"]
            bars[n]["base_on_zero"] = abs(bars[n]["base"]) < 0.05
        geo["e"] = {"order_left_to_right": sorted(bars, key=lambda n: xl[n]), "bars": bars,
                    "y_range_read": [round(float(v_of(bot)), 3), round(float(v_of(top)), 3)]}
R["B_fig3_drawn"] = geo

# ---------------------------------------------------------------- C: specifications
def specs(path):
    doc = fz.open(path); p = doc[0]
    sp = spans(p)
    fonts = p.get_fonts(full=True)
    sizes = sorted({round(s["size"], 2) for s in sp})
    small = sorted({(round(s["size"], 2), s["t"]) for s in sp if s["size"] < 4.995})
    ws = [d["width"] for d in p.get_drawings() if d.get("width") is not None and "s" in d["type"]]
    imgs = p.get_image_info(xrefs=True)
    dpi = [min(i["width"] / ((i["bbox"][2] - i["bbox"][0]) / 72), i["height"] / ((i["bbox"][3] - i["bbox"][1]) / 72)) for i in imgs]
    letters = sorted({(round(s["size"], 2), s["t"]) for s in sp if "Bold" in s["font"] and s["size"] >= 7.9})
    words = " ".join(s["t"] for s in sp).lower()
    bad = [w for w in (r"\bapp\b", r"this work", r"persistence") if re.search(w, words)]
    return {"page_mm": [round(p.rect.width / 72 * 25.4, 2), round(p.rect.height / 72 * 25.4, 2)],
            "fonts": [(f[3], f[1]) for f in fonts], "all_fonts_embedded": all(f[1] not in ("n/a", "") for f in fonts),
            "text_sizes_pt": sizes, "below_5pt": [list(t) for t in small],
            "below_5pt_all_sub_superscripts": all(t in ("−3", "2.5", "−6", "2") for _, t in small),
            "panel_letters": [list(t) for t in letters],
            "min_stroke_pt": round(min(ws), 3), "strokes_below_0p25": int(sum(w < 0.2499 for w in ws)), "n_strokes": len(ws),
            "rasters": len(imgs), "min_raster_dpi": round(min(dpi), 1) if dpi else None,
            "forbidden_words": bad}


R["C_specs"] = {"fig3_v2": specs(F2 / "fig_overview_final2.pdf"), "fig5_v2": specs(F2 / "fig_gc2_realism_final2.pdf")}
for nm in ("fig_overview_final2.png", "fig_gc2_realism_final2.png"):
    im = Image.open(F2 / nm); R["C_specs"][nm] = {"px": im.size, "dpi_tag": im.info.get("dpi"), "width_mm_at_300dpi": round(im.size[0] / 300 * 25.4, 2)}

# ---------------------------------------------------------------- D: identity with the first final versions
def raster_md5(path):
    doc = fz.open(path); p = doc[0]; out = []
    for i in p.get_image_info(xrefs=True):
        pix = fz.Pixmap(doc, i["xref"]); h_ = hashlib.md5(pix.samples)
        sm = doc.extract_image(i["xref"]).get("smask", 0)
        if sm:
            h_.update(fz.Pixmap(doc, sm).samples)                             # the alpha mask too
        out.append({"md5": h_.hexdigest(), "px": [pix.width, pix.height],
                    "bbox_pt": [round(v, 2) for v in i["bbox"]], "size_mm": [round((i["bbox"][2] - i["bbox"][0]) / 72 * 25.4, 3),
                                                                             round((i["bbox"][3] - i["bbox"][1]) / 72 * 25.4, 3)]})
    return out


def render(path, dpi=300):
    pix = fz.open(path)[0].get_pixmap(dpi=dpi, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


r1, r2 = raster_md5(F1 / "fig_overview_final.pdf"), raster_md5(F2 / "fig_overview_final2.pdf")
m1 = {r["md5"]: r for r in r1}
R["D_fig3_rasters"] = {"n_v1": len(r1), "n_v2": len(r2), "identical_data": sum(r["md5"] in m1 for r in r2),
                       "identical_data_and_size": sum(r["md5"] in m1 and m1[r["md5"]]["size_mm"] == r["size_mm"] for r in r2)}
# FIG. 3 rasters not bit-identical: same data resampled? (pairs by place; exact share at the best offset of +-2 px, and share of
# v2 pixels whose colour is present in the 3 x 3 neighbourhood of the v1 raster at the scaled position; colour sets)
def raster_arrays(path):
    doc = fz.open(path); p = doc[0]; out = []
    for i in sorted(p.get_image_info(xrefs=True), key=lambda i: (round(i["bbox"][1]), i["bbox"][0])):
        pix = fz.Pixmap(doc, i["xref"])
        out.append((np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3], [round(v, 2) for v in i["bbox"]]))
    return out


pairs = []
for (X1, b1), (X2, b2) in zip(raster_arrays(F1 / "fig_overview_final.pdf"), raster_arrays(F2 / "fig_overview_final2.pdf")):
    c1 = X1.astype(np.int32) @ np.array([65536, 256, 1]); c2 = X2.astype(np.int32) @ np.array([65536, 256, 1])
    h2, w2 = c2.shape
    ri_ = np.clip(np.rint(np.arange(h2) * c1.shape[0] / h2).astype(int), 0, c1.shape[0] - 1)
    ci_ = np.clip(np.rint(np.arange(w2) * c1.shape[1] / w2).astype(int), 0, c1.shape[1] - 1)
    hit = np.zeros(c2.shape, bool); ex = 0.0
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            m = c1[np.clip(ri_ + a, 0, c1.shape[0] - 1)[:, None], np.clip(ci_ + b, 0, c1.shape[1] - 1)[None, :]] == c2
            hit |= m; ex = max(ex, float(m.mean()))
    pairs.append({"px_v1": [int(X1.shape[1]), int(X1.shape[0])], "px_v2": [int(w2), int(h2)],
                  "size_mm_v1": [round((b1[2] - b1[0]) / 72 * 25.4, 3), round((b1[3] - b1[1]) / 72 * 25.4, 3)],
                  "size_mm_v2": [round((b2[2] - b2[0]) / 72 * 25.4, 3), round((b2[3] - b2[1]) / 72 * 25.4, 3)],
                  "exact_share_best_shift": round(ex, 4), "share_in_3x3_neighbourhood": round(float(hit.mean()), 5),
                  "same_colour_set": bool(set(np.unique(c1).tolist()) == set(np.unique(c2).tolist()))})
R["D_fig3_rasters"]["pairs_by_place"] = pairs
# panel a: renders compared (a is the top of both pages), best integer offset, where the pixels differ
A1, A2 = render(F1 / "fig_overview_final.pdf"), render(F2 / "fig_overview_final2.pdf")
ya = int(round(222 / 72 * 300))                                           # a's lowest labels end near 216 pt
best = None
for dy in range(-4, 5):
    for dx in range(-8, 9):
        a = A1[10 + dy:ya + dy, 10 + dx:A1.shape[1] - 10 + dx]; b = A2[10:ya, 10:A2.shape[1] - 10]
        if a.shape != b.shape:
            continue
        frac = float(np.mean(np.any(a != b, -1)))
        if best is None or frac < best[0]:
            best = (frac, dy, dx)
dy, dx = best[1], best[2]
diff = np.any(A1[10 + dy:ya + dy, 10 + dx:A1.shape[1] - 10 + dx] != A2[10:ya, 10:A2.shape[1] - 10], -1)
yy, xx = np.nonzero(diff)
mm = lambda px: round(float(px) / 300 * 25.4, 2)
cb = [s for s in spans(fz.open(F2 / "fig_overview_final2.pdf")[0]) if s["t"].startswith("Daily mean")][0]["bbox"]
blk = (cb[0] / 72 * 300 - 400, cb[1] / 72 * 300 - 30, cb[2] / 72 * 300 + 900, 216 / 72 * 300)   # generous box around the colour-bar block
outside = int(np.sum(~((xx + 10 >= blk[0]) & (xx + 10 <= blk[2]) & (yy + 10 >= blk[1]) & (yy + 10 <= blk[3]))))
R["D_fig3_panel_a_render"] = {"offset_px": [dy, dx], "differing_px_share": round(best[0], 6),
                              "diff_bbox_mm_xyxy": [mm(xx.min() + 10), mm(yy.min() + 10), mm(xx.max() + 10), mm(yy.max() + 10)] if len(xx) else None,
                              "differing_px_outside_colour_bar_area": outside}
# panel b: texts unchanged (multiset of text, size, font in b's band of the page)
def b_texts(path):
    p = fz.open(path)[0]; sp = spans(p)
    y0 = [s for s in sp if s["t"].startswith("Daily mean")][0]["bbox"][3] + 8
    y1 = min(s["bbox"][1] for s in sp if s["t"] == "Foehn (1 km)" and s["size"] == 6.0 and "Bold" not in s["font"])
    return sorted((s["t"], round(s["size"], 2), s["font"]) for s in sp if y0 < s["bbox"][1] < y1)
t1, t2 = b_texts(F1 / "fig_overview_final.pdf"), b_texts(F2 / "fig_overview_final2.pdf")
R["D_fig3_panel_b_texts"] = {"n_v1": len(t1), "n_v2": len(t2), "identical": t1 == t2,
                             "only_v1": [list(t) for t in t1 if t not in t2][:10], "only_v2": [list(t) for t in t2 if t not in t1][:10]}
# Fig. 5: rasters by place, renders outside the regions meant to change
s1, s2 = raster_md5(F1 / "fig_gc2_realism_final.pdf"), raster_md5(F2 / "fig_gc2_realism_final2.pdf")
pl5 = []
for a, b in zip(sorted(s1, key=lambda r: (r["bbox_pt"][1], r["bbox_pt"][0])), sorted(s2, key=lambda r: (r["bbox_pt"][1], r["bbox_pt"][0]))):
    pl5.append({"bbox_pt": b["bbox_pt"], "same_place": a["bbox_pt"] == b["bbox_pt"], "identical": a["md5"] == b["md5"]})
R["D_fig5_rasters"] = {"n": len(pl5), "identical": sum(r["identical"] for r in pl5), "all_same_place": all(r["same_place"] for r in pl5),
                       "changed_bbox_pt": [r["bbox_pt"] for r in pl5 if not r["identical"]]}
B1, B2 = render(F1 / "fig_gc2_realism_final.pdf"), render(F2 / "fig_gc2_realism_final2.pdf")
assert B1.shape == B2.shape
d5 = np.any(B1 != B2, -1)
allowed = np.zeros_like(d5)
k = 300 / 72
for r in pl5:
    if not r["identical"]:
        x0, y0, x1, y1 = r["bbox_pt"]; allowed[int(y0 * k) - 3:int(y1 * k) + 4, int(x0 * k) - 3:int(x1 * k) + 4] = True
p5 = fz.open(F2 / "fig_gc2_realism_final2.pdf")[0]; sp5 = spans(p5)
hd = [s for s in sp5 if "issued 3 days ahead" in s["t"]][0]["bbox"]
allowed[int(hd[1] * k) - 3:int(hd[3] * k) + 4, int(hd[0] * k) - 3:int(hd[2] * k) + 4] = True
xsp = min(s["bbox"][0] for s in sp5 if s["t"] == "Variance") - 2
ysp = max(s["bbox"][3] for s in sp5 if s["t"] == "Wavelength (km)") + 2
allowed[:int(ysp * k), int(xsp * k):] = True                                  # spectra h and p (shared y range)
lg = [s for s in sp5 if s["t"] == "Foehn, ensemble mean" and s["size"] < 5.2][0]["bbox"]
allowed[int(lg[1] * k) - 3:int(lg[3] * k) + 4, int(lg[0] * k) - 3:int(lg[2] * k) + 40] = True
yu = min(s["bbox"][1] for s in sp5 if s["t"].startswith("Foehn, 16 members")) - 3
allowed[int(yu * k):, :] = True                                              # panel u (and its legend)
R["D_fig5_render"] = {"differing_px": int(d5.sum()), "differing_px_outside_regions_meant_to_change": int((d5 & ~allowed).sum()),
                      "regions_meant_to_change": "row i-o rasters that changed, its descriptor, spectra h and p, the spectrum legend entry, panel u"}
del A1, A2, B1, B2

# ---------------------------------------------------------------- E: Fig. 5 numbers
PEAK = 23                                                                   # 0-based doy of 24 January 2022
SUB = (41.5, 35.5, 113.5, 121.5)
with open(G / "ghap1km_tiles.npy", "rb") as f:
    np.lib.format.read_magic(f); TSH, _, TDT = np.lib.format.read_array_header_1_0(f); TH0 = f.tell()


def ghap_sub(doy):
    n, s, w, e = SUB; r0, r1, c0, c1 = round((84 - n) * 100), round((84 - s) * 100), round((w + 180) * 100), round((e + 180) * 100)
    out = np.empty((r1 - r0, c1 - c0), np.float32); fd = os.open(G / "ghap1km_tiles.npy", os.O_RDONLY)
    for tr in range(r0 // 512, (r1 - 1) // 512 + 1):
        for tc in range(c0 // 512, (c1 - 1) // 512 + 1):
            off = TH0 + (((1826 + doy) * TSH[1] + tr) * TSH[2] + tc) * 512 * 512 * TDT.itemsize
            t = np.frombuffer(os.pread(fd, 512 * 512 * TDT.itemsize, off), TDT).reshape(512, 512)
            a0, a1, b0, b1 = max(r0, tr * 512), min(r1, tr * 512 + 512), max(c0, tc * 512), min(c1, tc * 512 + 512)
            out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = t[a0 - tr * 512:a1 - tr * 512, b0 - tc * 512:b1 - tc * 512]
    os.close(fd)
    return out


gh = ghap_sub(PEAK); land = gh > 0; g = gh[land].astype(np.float64); y75 = g > 75
la = (SUB[0] - (np.arange(gh.shape[0]) + 0.5) * 0.01)[:, None].repeat(gh.shape[1], 1)[land]
lo = (SUB[2] + (np.arange(gh.shape[1]) + 0.5) * 0.01)[None, :].repeat(gh.shape[0], 0)[land]
near = lambda f: np.asarray(f, np.float64)[np.rint((90 - la) / 0.4).astype(int), np.rint((lo % 360) / 0.4).astype(int) % 900]
cams = zarr.open(CAMSZ, mode="r")
aur_idx = {}
for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
    for kk, d in enumerate(np.load(p)["days"]):
        aur_idx[int(d)] = (p, kk)
_ap = {}


def aurora(d, h):
    p, kk = aur_idx[d + 1]
    if p not in _ap:
        _ap.clear(); _ap[p] = np.load(p)["pred"]
    A = _ap[p][kk].astype(np.float32)                                         # steps 12, 24, ..., 120 h after 00 UTC d+1
    first = np.load(AURD / "raw" / f"{d + 1:03d}.npz")["surf"][0, 5, :450].astype(np.float32) * 1e9 if h == 1 else A[2 * h - 3]
    return 0.25 * first + 0.5 * A[2 * h - 2] + 0.25 * A[2 * h - 1]


st = np.load(G / "stations_2022.npz")
parts = [np.load(q) for q in sorted(glob.glob(str(CASE / "part_*.npz")))]
keys = np.concatenate([q["keys"] for q in parts]); ens = np.concatenate([q["st_ens"] for q in parts]).astype(np.float32)
kpos = {tuple(int(v) for v in kk): i for i, kk in enumerate(keys)}
E5 = {"ghap_level": round(float(g.mean()), 2), "ghap_share_gt75": round(float(y75.mean()), 4), "land_px": int(land.sum()), "leads": {},
      "crop_files_vs_keys": {}}
inbox = (st["lat"] < 42) & (st["lat"] > 33) & (st["lon"] > 112) & (st["lon"] < 131)
ri = np.floor((42 - st["lat"][inbox]) * 100).astype(int); ci = np.floor((st["lon"][inbox] - 112) * 100).astype(int)
for h in (1, 3, 5, 7):
    d = PEAK - h
    z = np.load(CASE / f"crop0_d{d}_h{h}.npz"); assert np.allclose(z["box"], [42, 33, 112, 131])
    mem = z["members"].astype(np.float32)
    # file <-> key: the crop at the station pixels against the evaluation's station values of key (d, h, d + h)
    at = mem[:, ri, ci].T                                                   # (stations, 16)
    for kk in [(d, h, PEAK)] + ([(PEAK - 1, 1, PEAK)] if h == 3 else []):
        ev = ens[kpos[kk]][inbox]; ok = np.isfinite(ev).all(1) & np.isfinite(at).all(1)
        E5["crop_files_vs_keys"][f"crop0_d{d}_h{h} vs key {list(kk)}"] = {"stations": int(ok.sum()),
                                                                          "median_abs_diff": round(float(np.median(np.abs(at[ok] - ev[ok]))), 4),
                                                                          "share_within_0p5": round(float(np.mean(np.abs(at[ok] - ev[ok]) < 0.5)), 4)}
    ml = mem[:, 50:650, 150:950][:, land].astype(np.float64)
    mu, bm = ml.mean(0), ml.mean(1); rk = [int(np.sum(bm > bm[j])) + 1 for j in range(16)]
    pr = (ml > 75).mean(0)
    rec = {"start_day_calendar": f"{d + 1} January", "target": "24 January", "ens_mean_level": round(float(mu.mean()), 2),
           "lowest_member": [round(float(bm.min()), 2), int(np.argmin(bm)) + 1], "highest_member": [round(float(bm.max()), 2), int(np.argmax(bm)) + 1],
           "members_1_3_level": [round(float(bm[j]), 2) for j in range(3)], "members_1_3_rank": rk[:3],
           "rmse_ens_mean_vs_ghap": round(float(np.sqrt(np.mean((mu - g) ** 2))), 2),
           "p_gt75_where_ghap_gt75": round(float(pr[y75].mean()), 3), "p_gt75_elsewhere": round(float(pr[~y75].mean()), 3)}
    if h <= 5:
        cf, af = near(cams[d + 1, 0, h - 1]), near(aurora(d, h))
        rec.update({"cams_level": round(float(cf.mean()), 2), "aurora_level": round(float(af.mean()), 2),
                    "rmse_cams": round(float(np.sqrt(np.mean((cf - g) ** 2))), 2), "rmse_aurora": round(float(np.sqrt(np.mean((af - g) ** 2))), 2)})
    # spectra (own code of the stated definition): GHAP inside the members' range, 2-100 km, at the leads of h and p
    if h in (1, 3):
        def spec(x):
            x = np.where(np.isfinite(x), x, np.nanmean(x)); x = x - x.mean(); ny, nx = x.shape
            w2 = np.outer(np.hanning(ny), np.hanning(nx))
            Pw = np.abs(np.fft.fftshift(np.fft.fft2(x * w2))) ** 2 / (nx * ny) ** 2 / np.mean(w2 ** 2)
            K = np.hypot(*np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ny, 1.0)), np.fft.fftshift(np.fft.fftfreq(nx, 1.0)), indexing="ij"))
            bins = np.logspace(np.log10(1.0 / min(nx, ny)), np.log10(0.5), 24); ix = np.digitize(K.ravel(), bins)
            band = np.array([Pw.ravel()[ix == i].sum() if np.any(ix == i) else np.nan for i in range(1, 24)])
            return 1 / np.sqrt(bins[1:] * bins[:-1]), band / np.diff(np.log10(bins))
        msub = mem[:, 50:650, 150:950].copy(); msub[:, ~land] = np.nan
        lam, PG = spec(np.where(land, gh, np.nan))
        PM = np.array([spec(msub[j])[1] for j in range(16)]); lo_, hi_ = np.nanmin(PM, 0), np.nanmax(PM, 0)
        sm = (lam >= 2) & (lam <= 100)
        rec["spectra_2_100km"] = {"bands": int(sm.sum()), "ghap_inside": int(np.sum(sm & (PG >= lo_) & (PG <= hi_))),
                                  "ghap_below": int(np.sum(sm & (PG < lo_))), "ghap_above": int(np.sum(sm & (PG > hi_)))}
    E5["leads"][f"{h}"] = rec
# u: band of the 16 members over the 12 central-Beijing CNEMC stations, 1 day ahead
selb = (st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4)
m1 = np.stack([np.nanmean(ens[kpos[(t - 1, 1, t)]][selb], 0) for t in range(1, 29)])
E5["u_band"] = {"stations": int(selb.sum()), "band_max": round(float(m1.max()), 2), "band_max_day": f"{int(np.argmax(m1.max(1))) + 2} January",
                "band_24jan": [round(float(m1[22].min()), 2), round(float(m1[22].max()), 2)]}
# u read back from the PDF: y axis top, band top, vertical line position
dr5 = p5.get_drawings()
xlab = [s for s in sp5 if s["t"] == "January 2022"][0]["bbox"]
uspine = [d for d in dr5 if d["rect"].y0 > yu and len(d["items"]) == 1 and d["items"][0][0] == "l" and abs(d["rect"].x1 - d["rect"].x0) < 0.01
          and d["rect"].y1 - d["rect"].y0 > 40 and d.get("color") is not None and not close(d.get("color"), (0x9A / 255,) * 3)]
uspine = min(uspine, key=lambda d: d["rect"].x0)
ux, utop, ubot = uspine["rect"].x0, uspine["rect"].y0, uspine["rect"].y1
yt = sorted([(float(s["t"]), (s["bbox"][1] + s["bbox"][3]) / 2) for s in sp5 if s["t"] in ("0", "50", "100", "150", "200") and abs(s["bbox"][2] - ux) < 8
             and s["bbox"][1] > yu])
cy = np.polyfit([v for v, _ in yt], [y for _, y in yt], 1)
vy = lambda y: (y - cy[1]) / cy[0]
fills = [d for d in dr5 if d["rect"].y0 > yu - 1 and d["type"] == "f" and close(d.get("fill"), FOE) and d["rect"].x1 - d["rect"].x0 > 200]
xt = sorted([(float(s["t"]), (s["bbox"][0] + s["bbox"][2]) / 2) for s in sp5 if s["t"] in ("5", "9", "13", "17", "21", "25", "29") and s["bbox"][1] > ubot])
cx = np.polyfit([v for v, _ in xt], [x for _, x in xt], 1)
vline = [d for d in dr5 if d["rect"].y0 >= utop - 1 and close(d.get("color"), (0x9A / 255,) * 3) and abs(d["rect"].x1 - d["rect"].x0) < 0.01
         and d["rect"].y1 - d["rect"].y0 > 40]
E5["u_drawn"] = {"axis_top_value": round(float(vy(utop)), 2), "axis_bottom_value": round(float(vy(ubot)), 2),
                 "band_top_value": round(float(vy(min(f["rect"].y0 for f in fills))), 2) if fills else None,
                 "band_whole": bool(fills and min(f["rect"].y0 for f in fills) >= utop - 0.01),
                 "vertical_line_day": round(float((vline[0]["rect"].x0 - cx[1]) / cx[0]), 3) if vline else None,
                 "vertical_line_width_pt": vline[0].get("width") if vline else None, "x_ticks": [int(v) for v, _ in xt]}
# CNEMC stations of the plain (Methods box 36-41 N, 114-119.5 E), 24 January, bilinear 0.4 deg reads
obs = st["daily"][:, PEAK]
sel = (st["source"] == 1) & (st["lat"] >= 36) & (st["lat"] <= 41) & (st["lon"] >= 114) & (st["lon"] <= 119.5) & np.isfinite(obs)
fi = (90 - st["lat"][sel]) / 0.4; fj = (st["lon"][sel] % 360) / 0.4; i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int); wi, wj = fi - i0, fj - j0
bil = lambda f: (1 - wi) * ((1 - wj) * f[i0, j0] + wj * f[i0, j0 + 1]) + wi * ((1 - wj) * f[i0 + 1, j0] + wj * f[i0 + 1, j0 + 1])
stn = {"stations": int(sel.sum())}
for h in (1, 3, 5):
    d = PEAK - h; F = ens[kpos[(d, h, PEAK)]][sel].mean(1); o = obs[sel]
    C = bil(np.asarray(cams[d + 1, 0, h - 1], np.float64)); A = bil(aurora(d, h).astype(np.float64))
    ok = np.isfinite(F) & np.isfinite(C) & np.isfinite(A)
    stn[f"{h}"] = {"n": int(ok.sum()), "foehn": round(float(np.sqrt(np.mean((F - o)[ok] ** 2))), 2), "cams": round(float(np.sqrt(np.mean((C - o)[ok] ** 2))), 2),
                   "aurora": round(float(np.sqrt(np.mean((A - o)[ok] ** 2))), 2)}
E5["stations_plain_24jan"] = stn
W5 = json.loads((F2 / "fig5_numbers.json").read_text())["box_numbers"]["leads"]
dif = []
for h, nm in (("1", "1_day"), ("3", "3_days"), ("5", "5_days"), ("7", "7_days")):
    a, b = E5["leads"][h], W5[nm]
    pairs = [(a["ens_mean_level"], b["level"]["ensemble_mean"]), (a["lowest_member"][0], b["level"]["lowest_member"]),
             (a["highest_member"][0], b["level"]["highest_member"]), (a["rmse_ens_mean_vs_ghap"], b["rmse_vs_ghap"]["ensemble_mean"]),
             (a["p_gt75_where_ghap_gt75"], b["prob_gt75_mean"]["where_ghap_gt75"]), (a["p_gt75_elsewhere"], b["prob_gt75_mean"]["elsewhere"])]
    if "cams_level" in a:
        pairs += [(a["cams_level"], b["level"]["cams"]), (a["aurora_level"], b["level"]["aurora"]), (a["rmse_cams"], b["rmse_vs_ghap"]["cams"]),
                  (a["rmse_aurora"], b["rmse_vs_ghap"]["aurora"])]
    dif.append(max(abs(x - y) for x, y in pairs))
    E5["leads"][h]["ranks_equal_fig5_numbers"] = a["members_1_3_rank"] == b["members_1_2_3"]["rank_of_16_1=highest"]
E5["max_abs_diff_vs_fig5_numbers"] = round(max(dif), 4)
R["E_fig5"] = E5

# ---------------------------------------------------------------- F: contact sheets (200 dpi both sides, labels outside)
FONT = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 30)
FONTB = ImageFont.truetype(str(Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 36)


def sheet(left, right, ltxt, rtxt, title, out):
    ims = []
    for pth in (left, right):
        p = fz.open(pth)[0]; pix = p.get_pixmap(dpi=200, alpha=False)
        ims.append((Image.frombytes("RGB", (pix.width, pix.height), pix.samples), round(p.rect.width / 72 * 25.4, 1), round(p.rect.height / 72 * 25.4, 1)))
    m, gap, head, foot = 50, 90, 150, 70
    Wd = 2 * m + gap + sum(i[0].width for i in ims) + 4; Hd = head + max(i[0].height for i in ims) + foot + 4
    S = Image.new("RGB", (Wd, Hd), "white"); dr = ImageDraw.Draw(S)
    dr.text((m, 20), title, fill="#111111", font=FONTB)
    x = m
    for (im, w, h), txt in zip(ims, (ltxt, rtxt)):
        dr.text((x, 85), txt, fill="#111111", font=FONT)
        S.paste(im, (x + 2, head + 2)); dr.rectangle([x, head, x + im.width + 3, head + im.height + 3], outline="#9A9A9A", width=2)
        dr.text((x, head + im.height + 18), f"page {w} x {h} mm, rendered at 200 dpi (same scale left and right)", fill="#555555", font=FONT)
        x += im.width + gap
    buf = io.BytesIO(); S.save(buf, "PNG", optimize=True); atomic(out, buf.getvalue())
    return list(S.size)


def day_of(path, pat):
    m = re.search(pat, " ".join(s["t"] for s in spans(fz.open(path)[0])))
    return m.group(1) if m else "?"


R["F_sheets"] = {
    "sheet_fig3": sheet(WF / "fig_overview.pdf", F2 / "fig_overview_final2.pdf",
                        "Manuscript now: N/work/figs/fig_overview.pdf  (" + day_of(WF / "fig_overview.pdf", r"(\d{1,2} \w+ 2022)") + ")",
                        "Second version: final2/fig_overview_final2.pdf  (9 February 2022)",
                        "Fig. 3: present manuscript figure (left) and second version (right)", F2 / "sheet_fig3.png"),
    "sheet_fig5": sheet(WF / "fig_gc2_realism.pdf", F2 / "fig_gc2_realism_final2.pdf",
                        "Manuscript now: N/work/figs/fig_gc2_realism.pdf  (" + day_of(WF / "fig_gc2_realism.pdf", r"Forecasts for (\d{1,2} \w+)") + " 2022)",
                        "Second version: final2/fig_gc2_realism_final2.pdf  (24 January 2022)",
                        "Fig. 5: present manuscript figure (left) and second version (right)", F2 / "sheet_fig5.png")}

atomic(OUTJ, (json.dumps(R, indent=1, ensure_ascii=False) + "\n").encode())
print(json.dumps(R, indent=1, ensure_ascii=False))
