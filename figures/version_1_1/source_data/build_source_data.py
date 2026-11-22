#!/usr/bin/env python3
"""Source Data of "Global probabilistic forecasting of fine particulate air pollution at the kilometre scale".

Builds, from the data files that the figure scripts read:
    Source_Data.xlsx   one sheet per figure panel with graph data (first sheet README)
    csv/<sheet>.csv    the same tables as plain CSV (UTF-8); csv/README.csv lists them
    MANIFEST.md        one row per panel of every figure: exported sheet, or the reason for no export
    MANIFEST_internal.md   internal working file: script and data file behind each sheet (full paths)

    python -B build_source_data.py                            everything (about two minutes, CPU only, 2 GB)
    python -B build_source_data.py figED_fair fig_leadtime    only these figure files -> CSV files only (development)

The figure numbers are NOT written here: they are read from the manuscript at every run (order of the \includegraphics of
work/sections/main_text.tex, work/sections/methods.tex and work/supplementary.tex), so that a figure that moves between the
main text and the Supplementary Information keeps the right sheet names (Fig4a, SFig6a, ...) after a new run.

Rules followed here:
  * every value is read from a data file that the figure script reads, or recomputed with the lines of the figure script
    itself; these lines are COPIED here (the scripts draw and save at import, so they are not imported), and each copy
    names the script and the lines it comes from;
  * no model code is imported; numpy, zarr and openpyxl only;
  * every path is derived from the place of this file (no absolute path is written here);
  * files are written through a temporary file, flushed, synced, read back and then renamed.
The monitoring stations are evaluation data: they are only read to score and to draw the forecasts.
"""
import csv
import datetime as dt
import glob
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent                    # .../source_data
W = HERE.parent                                           # workspace of the manuscript (work/, figures_new/, analyses/)
PROJ = W.parents[1]                                       # tree that holds experiments/ and results/
EXP = PROJ / "experiments"
FIG = EXP / "figures"                                     # the figure scripts of the paper
DATA = FIG / "data"                                       # their data files
RUNS = EXP / "global1km" / "runs"                         # evaluation folders of the global models
WORLD = PROJ.parent / "data" / "world"                    # 1 km stores, operational forecasts
G1 = WORLD / "global1km"
NEW = W / "figures_new"
ANA = W / "analyses"
for _z in sorted((HERE / "_pylib").glob("*.zip")):        # private copy of openpyxl (one zip archive, imported from the zip)
    sys.path.insert(0, str(_z))

# the final model of the paper (figures/final_model.py: LABEL = "crps_e01")
LABEL = "crps_e01"
RUN_FINAL = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
RUN_E02 = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
EVAL_DIR, JAN21_DIR, CASE_DIR, SEL21_DIR = (RUN_FINAL / k for k in ("eval_final_crps_e01", "eval_jan21_crps_e01",
                                                                     "eval_case_eastasia_crps_e01", "eval2021_sel_crps_e01"))

SHEETS = []          # one dict per sheet, in the order of the workbook
PANELS = []          # one dict per panel of every figure (MANIFEST.md)
USED = {}            # data file -> modification time when read (MANIFEST_internal.md)
CURRENT = {"stem": ""}   # the figure file being exported
NOTES = []           # remarks on the state of the manuscript at build time (both manifests)
T0 = time.time()


# ------------------------------------------------------------------------------------------------ helpers
def sig(x, n=4):
    """a number with n significant digits (None for a missing value)"""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return 0.0 if x == 0 else float(f"{x:.{n}g}")


def dec(x, n=2):
    """a number with n decimals (None for a missing value)"""
    if x is None:
        return None
    x = float(x)
    return round(x, n) if math.isfinite(x) else None


def used(*paths):
    """remember which data files were read, with their modification times"""
    for p in paths:
        p = Path(p)
        USED[str(p)] = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if p.exists() else "MISSING"
    return paths[0] if len(paths) == 1 else paths


def npz(path, **kw):
    used(path)
    return np.load(path, **kw)


def jload(path):
    used(path)
    return json.loads(Path(path).read_text())


def sheet(name, figure, panel, desc, cols, rows, script, files, note=""):
    """register one sheet: row 1 = desc (one sentence with units), row 2 = cols, then rows (tidy: one row per plotted point)"""
    assert len(name) <= 31 and not any(c in name for c in "[]:*?/\\"), name
    assert name not in [s["name"] for s in SHEETS], name
    rows = [list(r) for r in rows]
    assert rows and all(len(r) == len(cols) for r in rows), (name, len(cols), [len(r) for r in rows][:3])
    clean = []
    for r in rows:
        o = []
        for v in r:
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            if isinstance(v, bool):
                v = "yes" if v else "no"
            o.append(v)
        clean.append(o)
    SHEETS.append(dict(name=name, figure=figure, panel=panel, desc=desc, cols=list(cols), rows=clean, script=str(script),
                       files=[str(f) for f in files], note=note, stem=CURRENT["stem"]))
    print(f"  sheet {name:22s} {len(clean):6d} rows x {len(cols):2d} columns   [{time.time() - T0:6.1f} s]", flush=True)


def panel(figure, letters, what, export):
    """one line of MANIFEST.md: export = sheet name(s), or the reason for no export"""
    PANELS.append(dict(figure=figure, panel=letters, what=what, export=export, stem=CURRENT["stem"]))


def safe_write(path, data):
    """temporary file, flush + fsync, read back, rename (the file system dropped writes on 2026-11-19)"""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        if f.read() != data:
            raise IOError(f"read-back differs: {path}")
    os.replace(tmp, path)
    with open(path, "rb") as f:
        if f.read() != data:
            raise IOError(f"read-back after rename differs: {path}")


def npy_header(path):
    """offset of the data, shape and dtype of a .npy file (as figures/fig_exposure_zoom.py:hdr)"""
    with open(path, "rb") as f:
        v = np.lib.format.read_magic(f)
        sh, _, dtp = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
        return f.tell(), sh, dtp


TILE, NTR, NTC = 512, 29, 71                              # the 1 km tile stores: (index, 29, 71, 512, 512), float16


def tile_box(name, k, r0, r1, c0, c1):
    """rows r0:r1, columns c0:c1 of field k of a 1 km tile store, reading only the tiles needed.
    COPY of figures/fig_exposure_zoom.py:ghap_box (L20-34), with the file name as an argument; it returns the same
    values as figures/fig1_teaser_v6.py:tiles(name, k)[r0:r1, c0:c1], which reads the whole globe."""
    path = G1 / name; used(path)
    h0, sh, dtp = npy_header(path)
    out = np.zeros((r1 - r0, c1 - c0), np.float32)
    fd = os.open(path, os.O_RDONLY)
    try:
        for tr in range(r0 // TILE, (r1 - 1) // TILE + 1):
            for tc in range(c0 // TILE, (c1 - 1) // TILE + 1):
                off = h0 + ((((k * NTR + tr) * NTC + tc) * TILE * TILE) * dtp.itemsize)
                b = np.frombuffer(os.pread(fd, TILE * TILE * dtp.itemsize, off), dtp).reshape(TILE, TILE)
                rr0, rr1 = max(r0, tr * TILE), min(r1, (tr + 1) * TILE); cc0, cc1 = max(c0, tc * TILE), min(c1, (tc + 1) * TILE)
                out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = b[rr0 - tr * TILE:rr1 - tr * TILE, cc0 - tc * TILE:cc1 - tc * TILE]
    finally:
        os.close(fd)
    return out


def ghap_box(d, r0, r1, c0, c1):
    """the 1 km estimates (GHAP) of global day d (day 1826 = 1 January 2022)"""
    return tile_box("ghap1km_tiles.npy", d, r0, r1, c0, c1)


def land_box(r0, r1, c0, c1):
    """the 1 km land mask of the model (channel 1 of the static store)"""
    return tile_box("static1km_tiles.npy", 1, r0, r1, c0, c1) > 0


def day2022(d):
    """0-based day of 2022 -> date"""
    return dt.date(2022, 1, 1) + dt.timedelta(int(d))


_CAMS_Z = []


def cams_daily():
    """the operational CAMS forecasts of 2022 as daily means: [run day, 0, lead day] on 451 x 900 (90 N .. 90 S, 0 .. 359.6 E)"""
    if not _CAMS_Z:
        import zarr
        p = WORLD / "cams_forecast_daily" / "2022.zarr"; used(p)
        _CAMS_Z.append(zarr.open(str(p), mode="r"))
    return _CAMS_Z[0]


def bilinear04(field, lat, lon):
    """a 0.4 degree field (row 0 = 90 N, column 0 = 0 E) read bilinearly at points; lines of figures/fig_overview.py L46-52"""
    fi = (90.0 - lat) / 0.4; fj = (lon % 360.0) / 0.4
    i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900
    wi, wj = fi - i0, fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])


# ================================================================================================ MAIN FIGURES
def fig1(fg, sh):
    panel(fg, "all panels", "schematic of the method (fields of single forecasts drawn as globes, discs and lenses; no graph)", "schematic, no graph data")


# ------------------------------------------------------------------------------------------------ hidden exposure
def hidden_ae(fg, sh, scr=None):
    """Panels a-e of the 'hidden exposure' figure: figures/fig_hidden.bak_20261015_warned.py (the state that drew
    fig_hidden.pdf; the current figures/fig_hidden.py draws another panel e), copied unchanged for a-e into
    figures_new/fig2_v3/fig_hidden_v4.py."""
    scr = scr or FIG / "fig_hidden.bak_20261015_warned.py"
    f = DATA / "exposure_2022.npz"
    ex = npz(f); XT = list(ex["thr"]); XR = [str(r) for r in ex["regions"]]
    res = ex["res"]                                       # (day, region, [above at 1 km, above at 40 km, hidden, spurious], threshold)
    # --- e: lines 61-70 of the script
    regs = ["World", "South Asia", "East Asia", "Africa", "Middle East", "Southeast Asia", "Europe", "North America", "South America"]
    who = {15.0: "WHO 24-h guideline level", 37.5: "WHO interim target 3", 75.0: "WHO interim target 1"}
    rows = []
    for thr in (15.0, 37.5, 75.0):
        k = XT.index(thr)
        for r in regs:
            above = res[:, XR.index(r), 0, k].sum(); hid = res[:, XR.index(r), 2, k].sum()
            share = 100 * hid / max(above, 1)
            keep = above > 1e9                            # at least a billion person-days above: the bar is drawn
            rows.append([r, thr, who[thr], sig(above / 1e9), sig(hid / 1e9), sig(share), bool(keep)])
    sheet(f"{sh}e", fg, "e",
          "Share of the person-days of 2022 above each threshold in the residents' 1 km grid cell that lie in 40 km cells whose mean is "
          "below it (%), by region (latitude-longitude boxes) and threshold (µg m-3), with the person-days behind it (billions); a bar is "
          "drawn when the region has more than one billion person-days above the threshold.",
          ["region", "threshold (µg m-3)", "WHO level", "person-days above the threshold at 1 km (billions)",
           "person-days hidden by the 40 km cells (billions)", "share hidden (%)", "bar drawn"], rows, scr, [f])
    panel(fg, "a", "map of the residents per day hidden by the 40 km grid, per 40 km cell", "map, not exported")
    panel(fg, "b-d", "Beijing-Tianjin on 21 January 2022: 1 km estimates, the same on 40 km cells, residents seen or hidden", "maps, not exported")
    panel(fg, "e", "share of the exposure hidden by a 40 km grid, by region and threshold (bars)", f"{sh}e")


def hidden_v4(fg, sh):
    """figures_new/fig2_v3/fig_hidden_v4.py: a-e as above, and f, the warnings issued 1 day ahead (bars with their values)."""
    scr = NEW / "fig2_v3" / "fig_hidden_v4.py"
    hidden_ae(fg, sh, scr)
    f = ANA / "hidden" / "v2" / "results" / "numbers_v2.json"
    N = jload(f)["systems"]
    rows = []
    for thr, key, lab in (("37.5", "reach_pct", "share of the people exposed who are warned (reached)"),
                          ("37.5", "correct_pct", "share of the warnings that are correct"),
                          ("75", "reach_pct", "share of the people exposed who are warned (reached)"),
                          ("75", "correct_pct", "share of the warnings that are correct")):       # L83-91
        for sk, name in (("foehn_p50", "Foehn"), ("cams", "CAMS forecast"), ("aurora", "Aurora")):
            v = N[f"thr{thr}"]["lead1"]["World"][sk][key]
            rows.append([float(thr), lab, name, sig(v[0]), sig(v[1]), sig(v[2])])
    sheet(f"{sh}f", fg, "f",
          "Warnings issued 1 day ahead in the retrospective forecasts of 2022, above 37.5 and 75 µg m-3: share of the people exposed who "
          "are warned and share of the warnings that are correct (%), for Foehn (at least two of its four members above the threshold), the "
          "CAMS forecast and Aurora; the bars carry the rounded value; bounds, 95% interval over days (not drawn).",
          ["threshold (µg m-3)", "quantity", "system", "share (%)", "lower bound (%)", "upper bound (%)"], rows, scr, [f])
    panel(fg, "f", "warnings issued 1 day ahead: people reached and correct warnings, three systems, two thresholds (bars with values)", f"{sh}f")


def warned_v2(fg, sh, letters="abc"):
    """analyses/hidden/v2/fig_hidden_warned_v2.py (people warned, version 2), every number from results/numbers_v2.json."""
    v2 = ANA / "hidden" / "v2"
    scr = v2 / "fig_hidden_warned_v2.py"
    f = v2 / "results" / "numbers_v2.json"
    N = jload(f)["systems"]
    A_, B_, C_ = letters
    SYSTEMS = [("foehn_p50", "Foehn (at least 2 of 4 members)"), ("foehn_40km", "Foehn on 40 km cells (ensemble mean)"),
               ("cams", "CAMS forecast"), ("aurora", "Aurora"), ("cams_qm", "CAMS forecast, bias-corrected"),
               ("cams_bcsd", "CAMS forecast, bias-corrected + 1 km pattern")]
    val = lambda t, h, s, m: N[f"thr{t}"][f"lead{h}"]["World"][s][m]
    rows = []
    for t in ("15", "37.5", "75"):                        # markers (lines 83-89), then the tails (lines 90-101)
        for s, name in SYSTEMS:
            for h in (1, 3):
                c, r = val(t, h, s, "correct_pct"), val(t, h, s, "reach_pct")
                rows.append([name, float(t), h, "marker (as issued)", None, sig(c[0]), sig(c[1]), sig(c[2]), sig(r[0]), sig(r[1]), sig(r[2])])
        for s in ("cams_qm", "cams_bcsd"):
            name = dict(SYSTEMS)[s]
            q = N[f"thr{t}"]["lead1"]["World"]["equal_number_warned"][s]
            sw = N[f"thr{t}"]["lead1"]["World"]["decision_threshold_sweep"][s]
            mu_eq = q["decision_threshold_factor"]
            for mu, v in sorted(sw.items(), key=lambda kv: -float(kv[0])):
                if mu_eq < float(mu) < 1.0:
                    rows.append([name, float(t), 1, "thin line (decision threshold lowered)", sig(float(mu)), sig(v["correct_pct"]), None, None,
                                 sig(v["reach_pct"]), None, None])
            rows.append([name, float(t), 1, "small dot (as many people warned as Foehn)", sig(mu_eq), sig(q["correct_pct"][0]), sig(q["correct_pct"][1]),
                         sig(q["correct_pct"][2]), sig(q["reach_pct"][0]), sig(q["reach_pct"][1]), sig(q["reach_pct"][2])])
    sheet(f"{sh}{A_}", fg, A_,
          "Share of the warnings that are correct (%) and share of the exposed people whom the warnings reach (%), by system, threshold "
          "(µg m-3) and lead time (days), world, 2022, with 95% intervals over days where the analysis gives them (the figure draws the "
          "values without intervals); thin lines lead from each bias-corrected reference issued 1 day ahead to the point reached when its "
          "decision threshold is lowered until it warns as many people as Foehn.",
          ["system", "threshold (µg m-3)", "lead time (days)", "element", "decision threshold as a factor of the threshold",
           "warnings that are correct (%)", "correct, lower bound (%)", "correct, upper bound (%)", "exposed people reached (%)",
           "reached, lower bound (%)", "reached, upper bound (%)"], rows, scr, [f])
    S = N["thr37.5"]
    rows = []
    for m, lab in (("hidden_warned_pct", "all hidden people"), ("hidden_margin_warned_pct", "hidden with a margin of 10%")):
        for s in ("foehn_p50", "foehn_40km"):
            for h in (1, 3):
                v = S[f"lead{h}"]["World"][s][m]
                rows.append([lab, {"foehn_p50": "Foehn, 1 km", "foehn_40km": "Foehn on 40 km cells"}[s], h, sig(v[0]), sig(v[1]), sig(v[2])])
    sheet(f"{sh}{B_}", fg, B_,
          "Share of the hidden people who are warned above 37.5 µg m-3 (%), by Foehn at 1 km (at least 2 of 4 members) and by its ensemble "
          "mean on 40 km cells, 1 and 3 days ahead, for all hidden people and for those hidden with a margin of 10% on both sides of the "
          "threshold; bounds, 95% interval over days.",
          ["hidden people counted", "system", "lead time (days)", "hidden people warned (%)", "lower bound (%)", "upper bound (%)"], rows, scr, [f])
    rows = []
    for s, name in SYSTEMS:
        for h in (1, 3):
            d = S[f"lead{h}"]["World"][s]
            hid, fa, tot, per = d["hidden_warned_M"], d["false_in_silent40_M"], d["warned_in_silent40_M"], d["false_per_hidden_warned"]
            rows.append([name, h, sig(hid[0]), sig(fa[0]), sig(tot[0]), sig(per[0])])
    sheet(f"{sh}{C_}", fg, C_,
          "People warned a day above 37.5 µg m-3 where the mean of the 40 km cell is below it (millions), by system and lead time (days): "
          "hidden people warned (pink bars) and people wrongly warned (hatched bars), with their sum and their ratio.",
          ["system", "lead time (days)", "hidden people warned (millions a day)", "people wrongly warned (millions a day)",
           "people warned where the 40 km mean is below (millions a day)", "wrongly warned per hidden person warned"], rows, scr, [f])
    panel(fg, A_, "reach against correct warnings, six systems, three thresholds, two lead times (points and thin lines; inset = the points of 15 µg m-3)", f"{sh}{A_}")
    panel(fg, B_, "hidden people warned: Foehn at 1 km against its mean on 40 km cells (bars with intervals)", f"{sh}{B_}")
    panel(fg, C_, "people warned where the 40 km mean is below the threshold (stacked bars)", f"{sh}{C_}")


# ------------------------------------------------------------------------------------------------ Fig. 3
def fig3(fg, sh):
    """figures/fig_overview.py. b: the RMSE written in the 12 maps (lines 39-61 and 88-117) and the station dots (line 111)."""
    scr = FIG / "fig_overview.py"
    D0, H = 19, 1; DT = D0 + H
    CENTRES = {"Delhi and the Ganges plain": (28.7, 77.6, 700), "Po Valley and the Alps": (45.35, 10.0, 600),
               "Los Angeles basin": (34.05, -117.9, 300), "Seoul and the Han basin": (37.56, 127.0, 300)}
    ASPECT = 1.5
    fs, fc = G1 / "stations_2022.npz", DATA / f"overview_cache_{LABEL}.npz"
    st = npz(fs); off = st["source"] != 2                 # the six official networks
    obs = st["daily"][:, DT]; okst = np.isfinite(obs) & off
    slat, slon, sobs = st["lat"][okst], st["lon"][okst], obs[okst]
    la_all, lo_all = st["lat"], st["lon"]
    zc = npz(fc); SSE, NSD = zc["SSE"], zc["NSD"]         # squared errors and station-days of 2022, lead 1 day: Foehn, CAMS, GHAP
    rows, pts = [], []
    names = {"cams": "CAMS forecast (40 km)", "ours": "Foehn (1 km)", "ghap": "GHAP (1 km estimates, not a forecast)"}
    for j, (nm, (la, lo, wkm)) in enumerate(CENTRES.items()):
        dla = wkm / ASPECT / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
        n, s, w, e = la + dla, la - dla, lo - dlo, lo + dlo
        sel = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e)
        inbox = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & (NSD > 0) & off
        for key in ("cams", "ours", "ghap"):
            rm = np.sqrt(SSE[inbox, {"ours": 0, "cams": 1, "ghap": 2}[key]].sum() / NSD[inbox].sum())
            rows.append([j + 1, nm, names[key], sig(rm), int(inbox.sum()), int(NSD[inbox].sum())])
        for a, b, c in zip(slat[sel], slon[sel], sobs[sel]):
            pts.append([j + 1, nm, dec(a, 4), dec(b, 4), sig(c)])
    sheet(f"{sh}b", fg, "b",
          "Root-mean-square error (µg m-3) over 2022 at the stations of each basin, 1 day ahead, on identical station-days: the numbers "
          "written in the twelve maps (the CAMS forecast, Foehn at 1 km and the 1 km estimates read at the station pixel).",
          ["basin number", "basin", "system", "RMSE (µg m-3)", "stations", "station-days"], rows, scr, [fc, fs])
    sheet(f"{sh}b_stations", fg, "b",
          "Daily mean PM2.5 (µg m-3) measured on 21 January 2022 at the stations of the official networks drawn as circles in the maps "
          "of each basin (latitude and longitude in degrees).",
          ["basin number", "basin", "latitude (°N)", "longitude (°E)", "measured PM2.5 (µg m-3)"], pts, scr, [fs])
    panel(fg, "a", "world map of Foehn's 1 km forecast for 21 January 2022", "map, not exported")
    panel(fg, "b", "four basins x (CAMS forecast, Foehn, 1 km estimates): maps with the RMSE over 2022 written in each and the stations as circles",
          f"maps not exported; printed numbers in {sh}b; station circles in {sh}b_stations")


# ------------------------------------------------------------------------------------------------ Fig. 4
def spread_skill_2021(ev):
    """COPY of figures/figED_fair.py:spread_skill_2021 (L33-42): spread / RMSE of the ensemble mean by lead day"""
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    z0 = npz(parts[0]); C = list(z0["rec_cols"]); M = int(z0["info"][5])
    R = np.concatenate([np.load(p)["recs"] for p in parts])
    out = []
    for h in range(1, 8):
        r = R[R[:, C.index("lead")] == h]
        n, sse, var = r[:, C.index("n")].sum(), r[:, C.index("sse")].sum(), r[:, C.index("var")].sum()
        out.append(np.sqrt((M + 1) / M * var / n) / np.sqrt(sse / n))
    return np.array(out), len(parts)


def fig4(fg, sh):
    """figures/figED_fair.py (drew figED_fair.pdf: the stations of six official networks)."""
    scr = FIG / "figED_fair.py"
    f = DATA / f"compare_fair_{LABEL}.npz"
    fz = npz(f); SYS = [str(s) for s in fz["systems"]]; S = fz["S"]      # (network, lead, system, [n, sum err, sum sq err, ...])
    NETS = [str(x) for x in fz["nets"]]
    I = {"Foehn": SYS.index(f"{LABEL} (1 km)"), "CAMS forecast": SYS.index("CAMS (run d+1, equal issuance)"),
         "Aurora": SYS.index("Aurora (00 UTC d+1, equal issuance)"), "Climatology of the 1 km estimates": SYS.index("climatology (GHAP 2017-2020)")}
    r = lambda g, h, i: np.sqrt(S[g, h, i, 2] / S[g, h, i, 0])
    b = lambda g, h, i: S[g, h, i, 1] / S[g, h, i, 0]
    ORDER = [1, 2, 0, 3, 4, 5]                            # China, USA, Europe, Korea, Japan, India
    for k, g in enumerate(ORDER):
        rows = []
        for nm, i in I.items():
            for h in range(4):
                rows.append([NETS[g], nm, h + 1, sig(r(g, h, i)),
                             dec(b(g, 0, i), 1) if (h == 0 and nm != "Climatology of the 1 km estimates") else None, int(S[g, h, i, 0])])
        sheet(f"{sh}{'abcdef'[k]}", fg, "abcdef"[k],
              f"Root-mean-square error (µg m-3) of the daily PM2.5 forecasts of 2022 by lead time (days) at the stations of {NETS[g]}, on "
              "identical station-days, and the mean bias 1 day ahead written in the panel (µg m-3, forecast minus measurement).",
              ["network", "system", "lead time (days)", "RMSE (µg m-3)", "bias written in the panel, 1 day ahead (µg m-3)", "station-days"],
              rows, scr, [f])
        panel(fg, "abcdef"[k], f"RMSE by lead time at the stations of {NETS[g]} (lines) and bias 1 day ahead (text)", f"{sh}{'abcdef'[k]}")
    g = S.shape[0] - 1
    rows = []
    for nm, key in (("Foehn, 1 km", f"{LABEL} (1 km)"), ("Foehn at 0.25°", f"{LABEL} at 0.25 deg"), ("Foehn at 0.5°", f"{LABEL} at 0.5 deg"),
                    ("CAMS forecast", "CAMS (run d+1, equal issuance)"), ("Aurora", "Aurora (00 UTC d+1, equal issuance)")):
        for h in range(4):
            rows.append([nm, h + 1, sig(r(g, h, SYS.index(key))), int(S[g, h, SYS.index(key), 0])])
    sheet(f"{sh}g", fg, "g",
          "Root-mean-square error (µg m-3) by lead time (days) at all stations of the six networks in 2022: Foehn read at its 1 km pixel and "
          "averaged to 0.25° and 0.5° before being read like the CAMS forecast and Aurora.",
          ["system", "lead time (days)", "RMSE, all stations (µg m-3)", "station-days"], rows, scr, [f])
    panel(fg, "g", "RMSE at all stations, Foehn at 1 km, 0.25° and 0.5°, CAMS, Aurora (lines)", f"{sh}g")
    # h: calibration (lines 81-89)
    ss, n1 = spread_skill_2021(SEL21_DIR)
    sb, n2 = spread_skill_2021(RUN_E02 / "eval2021_sel_e02")
    fc = DATA / "cams12h_scores_foehn_emis.npz"
    cz = npz(fc); gg = list(cz["regions"]).index("Global")
    assert cz["spread_skill"].shape[0] == 10
    rows = [["Foehn, 1 km (2021, against the 1 km estimates), after the CRPS training", h, sig(v)] for h, v in zip(range(1, 8), ss)]
    rows += [["Foehn, 1 km (2021, against the 1 km estimates), before the CRPS training", h, sig(v)] for h, v in zip(range(1, 8), sb)]
    rows += [["Foehn-CAMS (May-November 2022, against the CAMS analysis)", float(h), sig(v)] for h, v in zip(cz["leads_h"] / 24, cz["spread_skill"][:, gg])]
    sheet(f"{sh}h", fg, "h",
          "Ratio of the ensemble spread to the root-mean-square error of the ensemble mean (dimensionless; 1 for an ensemble whose spread "
          "matches its error) by lead time (days).",
          ["ensemble", "lead time (days)", "spread / RMSE"], rows, scr,
          [SEL21_DIR / "part_*.npz", RUN_E02 / "eval2021_sel_e02" / "part_*.npz", fc],
          note=f"the 2021 values are computed from the evaluation records ({n1} and {n2} part files) with the function of the figure script")
    panel(fg, "h", "spread / RMSE by lead time for three ensembles (lines)", f"{sh}h")


# ------------------------------------------------------------------------------------------------ Fig. 5
NY, NX = 361, 900                                         # the 0.4 degree domain of the Aurora protocol (global1km/eval_cams12h.py L23-28)
LAT04 = 84.0 - 0.4 * np.arange(NY)
REG12 = {"Global": None, "Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
         "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
         "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}


def to_domain(a, lat, lon):
    """COPY of global1km/eval_cams12h.py:to_domain (L31-38): a 0.4 degree global field -> 361 x 900 at 84 N .. 60 S, 180 W .. 179.6 E"""
    i0 = int(np.argmin(np.abs(np.asarray(lat) - 84.0)))
    assert abs(lat[i0] - 84.0) < 1e-3 and abs(lat[i0 + NY - 1] + 60.0) < 1e-3, (lat[:3], lat[-3:])
    a = a[..., i0:i0 + NY, :]
    j0 = int(np.argmin(np.abs(((np.asarray(lon) + 180.0) % 360.0))))
    assert abs(((lon[j0] + 180.0) % 360.0)) < 1e-3 and abs((lon[1] - lon[0]) - 0.4) < 1e-4
    return np.roll(a, -j0, axis=-1)


_F12 = {}


def fields12(leads, D0=243):
    """COPY of figures/fig_gc5_aurora_setup.py:fields (L28-44): the four fields of the forecasts initialised at 00 UTC on
    1 September 2022 (D0 = 243) at the given lead times (hours), and the latitude-weighted RMSE of each map over the domain."""
    key = (tuple(leads), D0)
    if key in _F12:
        return _F12[key]
    run = RUNS / "cams04_flow_shift2.6_emis" / "cams12h"
    AUR = EXP / "benchmark" / "aurora" / "out_12h"; RAW = EXP / "benchmark" / "aurora" / "raw"; CFC = WORLD / "cams_forecast_12h"
    w = np.cos(np.deg2rad(LAT04))[:, None] * np.ones((1, NX))
    fd = list(np.load(used(run / "doys.npy")))
    E = np.asarray(np.load(used(run / "pred.npy"), mmap_mode="r")[fd.index(D0)], np.float32).mean(0)
    A = None
    for p in sorted(AUR.glob("part_*.npz")):
        z = np.load(p)
        if D0 in list(z["days"]):
            used(p); A = to_domain(z["pred"][list(z["days"]).index(D0)].astype(np.float32), z["lat"], z["lon"])
    zc = npz(CFC / "2022_09.npz"); Cf = to_domain(zc["pm25"][0].astype(np.float32), zc["lat"], zc["lon"])
    out = {}
    for L in leads:
        k = L // 12 - 1; vt = 2 * D0 + L // 12
        zt = npz(RAW / f"{vt // 2:03d}.npz"); T = to_domain(zt["surf"][:, 5] * 1e9, zt["lat"], zt["lon"])[vt % 2].astype(np.float32)
        out[L] = {"Foehn-CAMS": E[k], "Aurora": A[k], "CAMS forecast": Cf[k], "CAMS analysis": T}
        for nm in ("Foehn-CAMS", "Aurora", "CAMS forecast"):
            out[L][nm + "_rmse"] = float(np.sqrt((w * (out[L][nm] - T) ** 2).sum() / w.sum()))
    out["files"] = [run / "pred.npy", AUR / "part_*.npz", CFC / "2022_09.npz", RAW / "<day of year>.npz"]
    _F12[key] = out
    return out


def fig5(fg, sh):
    """figures/fig_gc5_aurora_setup.py (the submitted render differs from the current script only by the place of the panel letters)."""
    scr = FIG / "fig_gc5_aurora_setup.py"
    F = fields12((24, 120))
    rows = []
    for L in (24, 120):
        vt = dt.datetime(2022, 9, 1) + dt.timedelta(hours=L)
        for nm in ("Foehn-CAMS", "Aurora", "CAMS forecast"):
            rows.append([nm, L // 24, f"{vt:%Y-%m-%d} 00 UTC", sig(F[L][nm + "_rmse"])])
    sheet(f"{sh}a", fg, "a",
          "Latitude-weighted root-mean-square error (µg m-3) against the CAMS analysis, over the whole domain (84°N-60°S), of the "
          "near-surface PM2.5 maps of the forecasts initialized at 00 UTC on 1 September 2022, 1 and 5 days ahead: the numbers written in the maps.",
          ["system", "lead time (days)", "valid time", "RMSE of the map (µg m-3)"], rows, scr, F["files"])
    panel(fg, "a", "eight world maps (three forecasts and the analysis, 1 and 5 days ahead) with the RMSE of each map written in it",
          f"maps not exported; printed numbers in {sh}a")
    f = DATA / "aurora_grid.npz"
    gz = npz(f); GS = [str(x) for x in gz["systems"]]; GR = [str(x) for x in gz["regions"]]
    k00 = np.arange(1, len(gz["leads_h"]), 2)             # 24, 48, ..., 120 h: valid at 00 UTC, as the initializations (L89)
    Lh = gz["leads_h"][k00] / 24
    rows = []
    for key, lab in (("rmse", "RMSE"), ("crps", "CRPS")):
        for r in GR:
            for sn in ("Climatology", "CAMS forecast", "Aurora", "Foehn-CAMS"):
                v = gz[key][GS.index(sn), k00, GR.index(r)]
                for L, x in zip(Lh, v):
                    rows.append([lab, r, sn, int(L), sig(x), "yes" if (key == "rmse" and L >= 3 and sn != "Climatology") else "no"])
    sheet(f"{sh}b", fg, "b",
          "Latitude-weighted root-mean-square error and continuous ranked probability score (µg m-3) against the CAMS analysis by lead time "
          "(days), for the forecasts initialized at 00 UTC from May to November 2022 (214 initializations) and valid at 00 UTC, over the whole "
          "domain and four continents; the insets repeat the RMSE of the three forecasts 3 to 5 days ahead.",
          ["score", "region", "system", "lead time (days)", "value (µg m-3)", "also drawn in the inset"], rows, scr, [f],
          note=f"n_init = {int(gz['n_init'])}")
    panel(fg, "b", "RMSE (top) and CRPS (bottom) by lead time in five regions, four systems (lines), with insets for days 3-5", f"{sh}b")


# ------------------------------------------------------------------------------------------------ Fig. 6
def spectrum(x, dx_km):
    """COPY of figures/fig_gc2_realism.py:spectrum (L60-73): radially averaged power spectrum -> wavelengths (km), variance per decade"""
    x = np.where(np.isfinite(x), x, np.nanmean(x)); x = x - x.mean()
    ny, nx = x.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2 / (nx * ny) ** 2 / np.mean(win ** 2)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dx_km)); kx = np.fft.fftshift(np.fft.fftfreq(nx, dx_km))
    K = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    nb = 24 if dx_km < 5 else 7
    bins = np.logspace(np.log10(1.0 / (min(nx, ny) * dx_km)), np.log10(0.5 / dx_km), nb)
    idx = np.digitize(K.ravel(), bins)
    band = np.array([P.ravel()[idx == i].sum() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    kc = np.sqrt(bins[1:] * bins[:-1])
    return 1.0 / kc, band / np.diff(np.log10(bins))


def fig6(fg, sh):
    """figures/fig_gc2_realism.py (the submitted render differs from the current script only by the place of the letters h and p)."""
    scr = FIG / "fig_gc2_realism.py"
    AURD = EXP / "benchmark" / "aurora"
    SUB = (41.5, 35.5, 113.5, 121.5)
    fcase = DATA / f"case_eastasia_{RUN_FINAL.name}.npz"
    Z = npz(fcase); peak = int(Z["peak"])
    D = CASE_DIR
    c0 = npz(D / f"crop0_d{peak - 1}_h1.npz"); box = c0["box"]; (r0, r1), (cc0, cc1) = c0["rows"], c0["cols"]
    n, s, w, e = SUB
    sr0, sr1 = int(round((box[0] - n) * 100)), int(round((box[0] - s) * 100)); sc0, sc1 = int(round((w - box[2]) * 100)), int(round((e - box[2]) * 100))
    gh = ghap_box(1826 + peak, int(r0), int(r1), int(cc0), int(cc1))[sr0:sr1, sc0:sc1]; land = gh > 0; gh = np.where(land, gh, np.nan)
    cams = cams_daily()
    aur = {}                                              # aurora_all (L36-42)
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        for k, d in enumerate(z["days"]):
            aur[int(d)] = (p, k)

    def aur_field(d, h):                                  # L45-49: Aurora issued with Foehn's forecast of day d, daily mean of day d + h
        p, k = aur[d + 1]; used(p); A = np.load(p)["pred"][k].astype(np.float32); k0 = 2 * (h - 1)
        first = (npz(AURD / "raw" / f"{d + 1:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9) if h == 1 else A[k0 - 1]
        return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]

    def sub40(field, lat0=90.0, dlat=0.4, lon0=0.0, dlon=0.4):       # L52-57
        lats = lat0 - dlat * np.arange(field.shape[0]); lons = (lon0 + dlon * np.arange(field.shape[1]) + 180) % 360 - 180
        ri = np.where((lats <= n + 0.2) & (lats >= s - 0.2))[0]; ci = np.where((lons >= w - 0.2) & (lons <= e + 0.2))[0]
        return field[np.ix_(ri, ci)]

    day = day2022(peak)
    for row, h in enumerate((1, 5)):                      # L105-124
        m = npz(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        m[:, ~land] = np.nan
        cf = sub40(np.asarray(cams[peak - h + 1, 0, h - 1], np.float32))
        af = sub40(aur_field(peak - h, h))
        dx40 = 0.4 * 111 * np.cos(np.deg2rad(38.5))
        rows = []
        for f_, dx, lab in ((gh, 1.0, "GHAP (1 km estimates)"), (m[0], 1.0, "Foehn, member 1"), (np.nanmean(m, 0), 1.0, "Foehn, ensemble mean"),
                            (cf, dx40, "CAMS forecast (0.4°)"), (af, dx40, "Aurora (0.4°)")):
            lam, P = spectrum(f_, dx)
            for a, b in zip(lam, P):
                rows.append([lab, h, sig(a), sig(b), "yes" if (np.isfinite(b) and 2 <= a <= 600) else "no"])
        L = "hp"[row]
        sheet(f"{sh}{L}", fg, L,
              f"Variance per decade of wavelength ((µg m-3)^2) of each field of daily PM2.5 over the North China Plain panel "
              f"(35.5-41.5°N, 113.5-121.5°E) on {day:%-d %B %Y}, forecasts issued {h} day{'s' if h > 1 else ''} ahead: radially averaged power "
              "spectrum by wavelength (km); the 0.4° forecasts on their own grid.",
              ["field", "issued ahead (days)", "wavelength (km)", "variance per decade ((µg m-3)^2)", "inside the axis range 600-2 km"], rows, scr,
              [D / f"crop0_d{peak - h}_h{h}.npz", G1 / "ghap1km_tiles.npy", WORLD / "cams_forecast_daily" / "2022.zarr",
               AURD / "out_12h_all" / "part_*.npz", AURD / "raw" / "<day of year>.npz"])
    # u: the monitors of central Beijing (L158-184)
    fs = G1 / "stations_2022.npz"
    st = npz(fs)
    sel = (st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4)
    obs = np.nanmean(st["daily"][sel].astype(np.float32), 0)
    parts = [np.load(q) for q in sorted(glob.glob(str(CASE_DIR / "part_*.npz")))]
    kk = np.concatenate([q["keys"] for q in parts]); ee = np.concatenate([q["st_ens"][:, sel] for q in parts]); gg = np.concatenate([q["st_ghap"][:, sel] for q in parts])
    E, GH = {}, {}
    for n_, (d_, h_, t_) in enumerate(kk):
        E[(int(d_), int(h_))] = np.nanmean(ee[n_].astype(np.float32), 0)      # member means over the stations (16,)
        GH[int(t_)] = float(np.nanmean(gg[n_].astype(np.float32)))
    la_, lo_ = st["lat"][sel], st["lon"][sel]
    days = np.arange(1, 29); cal = days + 1               # target days 1..28 = 2-29 January
    m1 = np.stack([E[(t - 1, 1)] for t in days]); m3 = {t: E[(t - 3, 3)] for t in days if t >= 3}
    rows = []
    for i, t in enumerate(days):
        cs = bilinear04(np.asarray(cams[int(t), 0, 0], np.float32), la_, lo_)
        rows.append([f"{day2022(t):%Y-%m-%d}", int(cal[i]), sig(obs[t]), sig(GH[int(t)]), sig(m1[i].mean()), sig(m1[i].min()), sig(m1[i].max()),
                     sig(m3[t].mean()) if t in m3 else None, sig(float(np.nanmean(cs)))])
    sheet(f"{sh}u", fg, "u",
          "Daily mean PM2.5 (µg m-3, UTC day) in central Beijing in January 2022: mean of the measurements at the 12 CNEMC monitors, the 1 km "
          "estimates (GHAP) at their grid cells, and the forecasts read at the monitors, averaged over them: Foehn issued 1 day earlier "
          "(ensemble mean, minimum and maximum of the 16 members), Foehn's ensemble mean issued 3 days earlier and the CAMS forecast issued "
          "at the same time as the 1-day forecast.",
          ["date", "day of January 2022", "measured, mean of the monitors", "GHAP at the monitor grid cells", "Foehn 1 day ahead, ensemble mean",
           "Foehn 1 day ahead, lowest member", "Foehn 1 day ahead, highest member", "Foehn 3 days ahead, ensemble mean", "CAMS forecast"],
          rows, scr, [fs, CASE_DIR / "part_*.npz", WORLD / "cams_forecast_daily" / "2022.zarr"], note=f"{int(sel.sum())} monitors selected")
    panel(fg, "a-g", "North China Plain, 25 January 2022: the 1 km estimates and six forecast fields issued 1 day ahead", "maps, not exported")
    panel(fg, "h", "power spectrum of the fields of a-g (lines, logarithmic axes)", f"{sh}h")
    panel(fg, "i-o", "the same fields issued 5 days ahead", "maps, not exported")
    panel(fg, "p", "power spectrum of the fields of i-o (lines)", f"{sh}p")
    panel(fg, "q-t", "probability of exceeding 75 µg m-3 from 16 members, issued 7, 5, 3 and 1 days ahead", "maps, not exported")
    panel(fg, "u", "daily series at the 12 monitors of central Beijing, January 2022 (points, lines, band)", f"{sh}u")


# ------------------------------------------------------------------------------------------------ Fig. 7
def fig7(fg, sh):
    """figures/fig_transport.py (the submitted render differs from the current script only by the place of the letters h-j)."""
    scr = FIG / "fig_transport.py"
    f1, f2, f3 = DATA / "transport_fig.npz", DATA / "transport_gain.npz", DATA / "transport_kano.npz"
    z, tg, zs = npz(f1), npz(f2), npz(f3)
    R = [str(r) for r in z["regions"]]; GR = [str(r) for r in tg["regions"]]
    T = int(zs["T"])                                      # arrival day: 16 May 2022 (0-based day 135)
    # a-c: where the air reaching Kano on the arrival day was on each of the three days before (L67-73)
    for k, h in enumerate((3, 2, 1)):
        o = zs[f"org_{h}"]; keep = np.nan_to_num(o[:, 3]) > 0.05
        rows = [[f"{day2022(T - h):%Y-%m-%d}", h, dec(a, 3), dec(b, 3), sig(c, 3), sig(d, 3)] for a, b, c, d in o[keep]]
        L = "abc"[k]
        sheet(f"{sh}{L}", fg, L,
              f"Where the air reaching Kano (12.00°N, 8.52°E) on 16 May 2022 was on {day2022(T - h):%-d %B} (black dots): origins of the "
              f"particles released in the 3 x 3 receptor cells around the city in the forecast started that day ({h} day{'s' if h > 1 else ''} "
              "before), longitude and latitude in degrees, height relative to the mixed layer of the start day and survival (fraction not "
              "removed on the way; particles with a survival above 0.05 are drawn).",
              ["day", "days before arrival", "longitude (°E)", "latitude (°N)", "height / mixing height", "survival"], rows, scr, [f3],
              note=f"{int(keep.sum())} of {len(o)} particles drawn; the 1 km map under the dots is not exported")
    # h: share of the arriving air that was in the mixed layer on the start day (L110-118)
    L7 = z["leads"]; g0 = R.index("Global")
    near = 100 - 100 * np.nanmean(z["aloft"][:, :, g0], 0)
    rows = [[int(h), sig(v), sig(100 - v)] for h, v in zip(L7, near)]
    sheet(f"{sh}h", fg, "h",
          "Share of the air arriving over land on the target day (%) that was in the mixed layer on the start day (blue) or higher up or "
          "farther back (pink), by lead time (days); all land, mean over 90 start days of 2022.",
          ["lead time (days)", "in the mixed layer on the start day (%)", "higher up or farther back (%)"], rows, scr, [f1],
          note=f"{z['aloft'].shape[0]} start days")
    # j: reduction of the CRPS by the computed transport (L144-153)
    ga = GR.index("All regions"); gain, lo, hi = tg["gain"][ga, :, 0], tg["gain"][ga, :, 1], tg["gain"][ga, :, 2]
    rows = [[int(h), sig(a), sig(b), sig(c)] for h, a, b, c in zip(L7, gain, lo, hi)]
    sheet(f"{sh}j", fg, "j",
          "Reduction of the continuous ranked probability score against the 1 km estimates brought by the computed transport (%), over all "
          "forecasts of 2022 and all regions, by lead time (days): dots, with the bounds of the 95% interval (bars).",
          ["lead time (days)", "CRPS reduction (%)", "lower bound (%)", "upper bound (%)"], rows, scr, [f2])
    gd = tg["gain_day"][ga]                               # (leads, start days)
    rows = []
    for h, v in zip(L7, gd):
        fin = np.isfinite(v); qa, qb = np.percentile(v[fin], [2, 98])
        for d, x in zip(tg["days"][fin], v[fin]):
            rows.append([int(h), f"{day2022(d):%Y-%m-%d}", sig(x), "yes" if qa <= x <= qb else "no"])
    sheet(f"{sh}j_violins", fg, "j",
          "Reduction of the continuous ranked probability score brought by the computed transport (%), for each start day of 2022 and lead "
          "time (days), all regions: the values whose distribution the violins show (the violins are drawn from the values between the 2nd "
          "and 98th percentiles of each lead time).",
          ["lead time (days)", "start day", "CRPS reduction (%)", "inside the 2nd-98th percentiles (drawn)"], rows, scr, [f2])
    panel(fg, "a-c", "1 km estimates over the central Sahel on 13-15 May 2022 with the origins of the air reaching Kano (dots)",
          f"maps not exported; dots in {sh}a, {sh}b, {sh}c")
    panel(fg, "d", "1 km estimates on 16 May 2022, the arrival day", "map, not exported")
    panel(fg, "e-g", "probability of PM2.5 above 100 µg m-3 on 16 May from 16 members, issued 3, 2 and 1 days before", "maps, not exported")
    panel(fg, "h", "share of the arriving air in the mixed layer on the start day, by lead time (areas and line)", f"{sh}h")
    panel(fg, "i", "world map of the same share 3 days ahead", "map, not exported")
    panel(fg, "j", "CRPS reduction by the computed transport: violins per start day, dots and 95% intervals", f"{sh}j (dots, intervals); {sh}j_violins (values per start day)")


# ------------------------------------------------------------------------------------------------ Fig. 8
def fig8(fg, sh):
    """figures/fig_smoke.py (the submitted render differs from the current script only by the place of the letters f-h)."""
    scr = FIG / "fig_smoke.py"
    SHORT, LONG = RUN_FINAL / "eval_case_gironde_crps_e01", RUN_FINAL / "eval_case_gironde_long_crps_e01"
    N_, S_, W_, E_ = 46.0, 43.4, -2.6, 1.0
    CR0, CC0 = 3700, 17700
    T, THR = 198, 25.0                                    # 18 July 2022; WHO 24-h interim target 4
    TEICH = (44.64, -1.02)
    ffire, fs = G1 / "fires.npy", G1 / "stations_2022.npz"
    fires = np.load(used(ffire), mmap_mode="r"); st = npz(fs)
    sel = (st["lat"] <= N_) & (st["lat"] >= S_) & (st["lon"] >= W_) & (st["lon"] <= E_) & (st["source"] == 0)
    fr0, fc0 = int((84 - N_) / 0.25), int((W_ + 180) / 0.25); fr1, fc1 = int((84 - S_) / 0.25) + 1, int((E_ + 180) / 0.25) + 1

    def fire_pts(d):                                      # L79-81
        fr = np.asarray(fires[1826 + d, 0, fr0:fr1, fc0:fc1], np.float32)
        return np.array([(84 - 0.25 * (fr0 + i) - 0.125, -180 + 0.25 * (fc0 + j) + 0.125) for i, j in np.argwhere(fr > 3)])

    def members(t, h):                                    # L40-41
        return np.load((SHORT if h <= 3 else LONG) / f"crop0_d{t - h}_h{h}.npz")["members"].astype(np.float32)

    tr, tc = int((84 - TEICH[0]) * 100), int((TEICH[1] + 180) * 100)

    def at_reserve(m):                                    # L85-86: members averaged over the 20 km box around the reserve
        return m[:, tr - CR0 - 10:tr - CR0 + 10, tc - CC0 - 10:tc - CC0 + 10].mean((1, 2))

    # a: the EEA stations (circles) and the fire cells of a-e (triangles)
    rows = [[dec(a, 4), dec(b, 4), sig(c)] for a, b, c in zip(st["lat"][sel], st["lon"][sel], st["daily"][sel, T].astype(np.float32))]
    sheet(f"{sh}a_stations", fg, "a",
          "Daily mean PM2.5 (µg m-3) measured on 18 July 2022 at the stations of the European Environment Agency inside the panel "
          "(43.4-46.0°N, 2.6°W-1.0°E), drawn as circles; empty cell, no measurement that day.",
          ["latitude (°N)", "longitude (°E)", "measured PM2.5 (µg m-3)"], rows, scr, [fs])
    rows = []
    for L, d in (("a", T), ("b", T - 7), ("c", T - 6), ("d", T - 5), ("e", T - 1)):
        for la, lo in fire_pts(d):
            rows.append([L, f"{day2022(d):%Y-%m-%d}", dec(la, 3), dec(lo, 3)])
    sheet(f"{sh}a-e_fire_cells", fg, "a-e",
          "Fire cells of the Global Fire Assimilation System drawn as triangles: centres (degrees) of the 0.25° cells with fire emissions "
          "above the drawing level, on 18 July 2022 (a) and on the start day of each forecast (b-e); no row for a panel without fire cell.",
          ["panel", "day of the fire data", "latitude (°N)", "longitude (°E)"], rows, scr, [ffire])
    # f: the 1 km estimates over the 20 km surroundings of the reserve (L134-150)
    days_f = np.arange(193, 207)                          # 13-26 July
    gr = []
    for d in days_f:
        v = ghap_box(1826 + d, tr - 10, tr + 10, tc - 10, tc + 10).astype(np.float32)
        gr.append(float(v[v > 0].mean()))

    def on_fires(d):                                      # L140-142: GFAS cells over the two fires
        pts = fire_pts(d)
        return len(pts) > 0 and bool(np.any((pts[:, 0] > 44.25) & (pts[:, 0] < 44.75) & (pts[:, 1] > -1.25) & (pts[:, 1] < -0.25)))
    rows = [[f"{day2022(d):%Y-%m-%d}", int(d - 181 + 1), sig(v), bool(on_fires(d))] for d, v in zip(days_f, gr)]
    sheet(f"{sh}f", fg, "f",
          "Daily PM2.5 of the 1 km estimates (GHAP) averaged over the 20 km surroundings of the Teich bird reserve (µg m-3), 13-26 July 2022, "
          "and the days with fire cells of the Global Fire Assimilation System at the two fires (shading).",
          ["date", "day of July 2022", "GHAP, 20 km around the reserve (µg m-3)", "fire cells at the two fires (shaded)"], rows, scr,
          [G1 / "ghap1km_tiles.npy", ffire])
    # g: probability of exceeding 25 ug m-3 by target day and lead time (L165-178); h: the 16 members for 18 July (L190-196)
    LD = np.arange(1, 8); rows, rows_h = [], []
    for h in LD:
        for t in days_f:
            f_ = (SHORT if h <= 3 else LONG) / f"crop0_d{t - h}_h{h}.npz"
            if f_.exists():
                v = at_reserve(members(t, h)); p = float(np.mean(v > THR))
                rows.append([f"{day2022(t):%Y-%m-%d}", int(t - 181 + 1), int(h), sig(p), int((v > THR).sum()), "yes" if p >= 0.18 else "no"])
                if t == T:
                    for i, x in enumerate(v):
                        rows_h.append([int(h), "member", i + 1, sig(x)])
                    rows_h.append([int(h), "ensemble mean", None, sig(v.mean())])
            else:
                rows.append([f"{day2022(t):%Y-%m-%d}", int(t - 181 + 1), int(h), None, None, "no"])
    sheet(f"{sh}g", fg, "g",
          "Foehn's probability that PM2.5 exceeds 25 µg m-3 over the 20 km surroundings of the reserve (share of the 16 members, each averaged "
          "over the surroundings), by target day of July 2022 and lead time (days); empty cell, no forecast in this evaluation (grey).",
          ["target day", "day of July 2022", "issued before (days)", "probability", "members above 25 µg m-3 (of 16)", "number written in the cell"],
          rows, scr, [SHORT / "crop0_d<start day>_h<lead>.npz", LONG / "crop0_d<start day>_h<lead>.npz"])
    sheet(f"{sh}h", fg, "h",
          "PM2.5 over the 20 km surroundings of the reserve on 18 July 2022 (µg m-3) in each of Foehn's 16 members, by lead time (days), and "
          f"their mean (bars); the dotted line of the panel is the GHAP value of 18 July in {sh}f, the dashed line 25 µg m-3.",
          ["issued before (days)", "element", "member", "PM2.5 at the reserve (µg m-3)"], rows_h, scr,
          [SHORT / "crop0_d<start day>_h<lead>.npz", LONG / "crop0_d<start day>_h<lead>.npz"],
          note="the horizontal position of the dots inside a lead time is a random jitter, not data")
    panel(fg, "a", "1 km estimates on 18 July 2022 with the fire cells (triangles) and the EEA stations (circles)",
          f"map not exported; circles in {sh}a_stations; triangles in {sh}a-e_fire_cells")
    panel(fg, "b-e", "probability of PM2.5 above 25 µg m-3 on 18 July from 16 members, issued 7, 6, 5 and 1 days before, with the fire cells of the inputs",
          f"maps not exported; triangles in {sh}a-e_fire_cells")
    panel(fg, "f", "1 km estimates over the surroundings of the reserve, 13-26 July (line, shading)", f"{sh}f")
    panel(fg, "g", "probability of exceedance by target day and lead time (coloured table with numbers)", f"{sh}g")
    panel(fg, "h", "the 16 members for 18 July by lead time (dots, bars)", f"{sh}h")


# ------------------------------------------------------------------------------------------------ Fig. 9
def md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def variant_in_manuscript(folder, stem, default=""):
    """which render of a new figure the manuscript holds: the tag of the file of `folder` that is identical to work/figs/<stem>.pdf"""
    target = md5(W / "work" / "figs" / f"{stem}.pdf")
    if target is None:
        return default, f"work/figs/{stem}.pdf does not exist yet: default render of the figure folder used"
    for p in sorted(folder.glob(f"{stem}*.pdf")):
        if md5(p) == target:
            return p.stem[len(stem):].lstrip("_"), f"work/figs/{stem}.pdf is identical to {p.name}"
    return default, f"work/figs/{stem}.pdf matches no render of the figure folder: default render used"


def fig9(fg, sh):
    """figures_new/fig_rt_day/fig_rt_day.py; it reads data/rt_day[_variant].npz and numbers[_variant].json."""
    d = NEW / "fig_rt_day"; scr = d / "fig_rt_day.py"
    tag, why = variant_in_manuscript(d, "fig_rt_day")
    suf = f"_{tag}" if tag else ""
    fz, fn = d / "data" / f"rt_day{suf}.npz", d / f"numbers{suf}.json"
    Z, num = npz(fz), jload(fn)
    day1, day2 = dt.date.fromisoformat(str(Z["day1"])), dt.date.fromisoformat(str(Z["day2"]))
    city1, cityn, city2 = str(Z["city1"]), str(Z["neighbour"]), str(Z["city2"])
    NAME = {"cams": "CAMS forecast (40 km)", "rt04": "Foehn in real time (40 km)", "km": "Foehn in real time (1 km)"}
    sc = num["case1"]["region"]["scores_at_monitors"]     # L99, L125-129, L141-144
    nmon = int(num["case1"]["region"]["monitors"])
    rows = []
    for L, key in zip("bcd", ("cams", "rt04", "km")):
        q = sc[f"{key}_lead1"]
        rows.append([L, NAME[key], f"{day1}", 1, sig(q["rmse"]), sig(q["r"]), sig(q["bias"]), nmon])
    for L_ in (3, 5):
        q = sc[f"km_lead{L_}"]
        rows.append(["e", NAME["km"], f"{day1 - dt.timedelta(L_ - 1)}", L_, sig(q["rmse"]), None, None, nmon])
    sheet(f"{sh}b-e", fg, "b-e",
          f"Scores of the forecasts of daily mean PM2.5 for {day1:%-d %B %Y} at the {nmon} monitors of the {city1}-{cityn} window, written in the "
          "maps: root-mean-square error (µg m-3), correlation between monitors and mean bias (µg m-3, forecast minus measurement); panel e "
          "prints the RMSE only.",
          ["panel", "system", "cycle (day of issue, 00 UTC)", "lead day", "RMSE (µg m-3)", "R", "bias (µg m-3)", "monitors"], rows, scr, [fn],
          note=why)
    rows = [[dec(a, 4), dec(b, 4), sig(c)] for a, b, c in zip(Z["reg_lon"], Z["reg_lat"], Z["reg_obs"])]
    sheet(f"{sh}b-e_monitors", fg, "b-e",
          f"Daily mean PM2.5 (µg m-3) measured on {day1:%-d %B %Y} at the monitors drawn as circles in panels b-e (longitude and latitude in degrees).",
          ["longitude (°E)", "latitude (°N)", "measured PM2.5 (µg m-3)"], rows, scr, [fz])
    days = [str(s) for s in Z["days"]]
    late = num["rule"]["lead1_computed_on_issue_day"]; late04 = num["rule"]["rt04_computed_on_issue_day"]
    for L, city in (("f", city1), ("g", cityn)):          # L149-165
        S = {q: Z[f"s_{city}_{q}"] for q in ("n", "obs", "km1", "km1_p10", "km1_p90", "km3", "rt1", "cams1")}
        rows = []
        for i, dd in enumerate(days):
            has1 = np.isfinite(S["km1"][i])
            rows.append([dd, int(S["n"][i]) if np.isfinite(S["n"][i]) else None, sig(S["obs"][i]), sig(S["km1"][i]), sig(S["km1_p10"][i]),
                         sig(S["km1_p90"][i]), sig(S["km3"][i]), sig(S["rt1"][i]), sig(S["cams1"][i]),
                         ("no" if late.get(dd, True) else "yes") if has1 else None,
                         ("no" if late04.get(dd, True) else "yes") if np.isfinite(S["rt1"][i]) else None])
        sheet(f"{sh}{L}", fg, L,
              f"Daily mean PM2.5 (µg m-3) over the monitors of {city}, by valid day of November 2026: measurements; 1 km forecast at lead day 1 "
              "(mean of eight members and mean over the monitors of their 10th and 90th percentiles) and at lead day 3; 0.4° forecast and CAMS "
              "forecast at lead day 1; the last two columns mark the cycles computed one day late (open squares).",
              ["valid day", "monitors with a daily mean", "measured", "1 km forecast, lead day 1", "1 km, 10th percentile of members",
               "1 km, 90th percentile of members", "1 km forecast, lead day 3", "0.4° forecast, lead day 1", "CAMS forecast, lead day 1",
               "1 km cycle of lead day 1 computed one day late", "0.4° cycle of lead day 1 computed one day late"], rows, scr, [fz, fn])
    # h-j: the residents counted in panel j (L183-190), recomputed from the fields of the panel, and the monitors
    THR = 37.5
    hx = Z["h_km"].astype(np.float32); ok = np.isfinite(hx); pop = Z["h_pop"].astype(np.float64); c04 = Z["h_rt04"]
    above = ok & (hx > THR); seen = above & (c04 > THR)
    tot, ab, se, hi = (pop * ok).sum() / 1e6, (pop * above).sum() / 1e6, (pop * seen).sum() / 1e6, (pop * (above & ~seen)).sum() / 1e6
    c2 = num["case2"]
    rows = [["residents of the panel area with a forecast", sig(tot), sig(c2["residents_million"])],
            ["above 37.5 µg m-3 in the 1 km forecast", sig(ab), sig(c2["above_million"])],
            ["of whom in 0.4° cells forecast above it (grey: seen)", sig(se), sig(c2["seen_million"])],
            ["of whom in 0.4° cells forecast below it (pink: hidden)", sig(hi), sig(c2["hidden_million"])]]
    sheet(f"{sh}j", fg, "j",
          f"Residents of the {city2} panel (millions) on {day2:%-d %B %Y}, lead day 1: all, above WHO interim target 3 (37.5 µg m-3) in the 1 km "
          "forecast, and of these the residents whose 0.4° cell is forecast above it (seen) or below it (hidden); recomputed from the fields "
          "of the panel and as stored with the figure.",
          ["residents", "millions (recomputed from the panel fields)", "millions (numbers file of the figure)"], rows, scr, [fz, fn])
    rows = [[dec(a, 4), dec(b, 4), sig(c)] for a, b, c in zip(Z["h_lon"], Z["h_lat"], Z["h_obs"])]
    sheet(f"{sh}h-j_monitors", fg, "h-j",
          f"Daily mean PM2.5 (µg m-3) measured on {day2:%-d %B %Y} at the monitors of {city2} drawn as circles in panels h-j (longitude and latitude in degrees).",
          ["longitude (°E)", "latitude (°N)", "measured PM2.5 (µg m-3)"], rows, scr, [fz])
    panel(fg, "a", f"world map of the real-time 1 km forecast for {day1:%-d %B %Y}", "map, not exported")
    panel(fg, "b-d", f"{city1} and {cityn}: CAMS forecast, 0.4° forecast and 1 km forecast with the monitors (circles) and RMSE, R, bias written in each map",
          f"maps not exported; printed numbers in {sh}b-e; circles in {sh}b-e_monitors")
    panel(fg, "e", "the 1 km forecast from the cycles two and four days earlier, with its RMSE written", f"maps not exported; printed numbers in {sh}b-e")
    panel(fg, "f", f"daily series over the monitors of {city1} (points, lines, band)", f"{sh}f")
    panel(fg, "g", f"daily series over the monitors of {cityn} (points, lines, band)", f"{sh}g")
    panel(fg, "h-i", f"{city2}: 1 km forecast with the 37.5 µg m-3 contour and 0.4° forecast, with the monitors (circles)",
          f"maps not exported; circles in {sh}h-j_monitors")
    panel(fg, "j", "residents above 37.5 µg m-3 at 1 km, seen or hidden by the 0.4° cells (map)", f"map not exported; the residents counted in {sh}j")


# ------------------------------------------------------------------------------------------------ Fig. 10
def rt_skill(fg, sh, stem="fig_rt_skill", with_f=True):
    """figures_new/fig_rt_skill/fig_rt_skill.py (panels a-f) and its main-text variant fig_rt_skill_main.py (the same without
    panel f); both read numbers[_tag].json and data/rt_skill[_tag].npz."""
    d = NEW / "fig_rt_skill"; scr = d / f"{stem}.py"
    tag, why = variant_in_manuscript(d, stem)
    suf = f"_{tag}" if tag else ""
    fn, fz = d / f"numbers{suf}.json", d / "data" / f"rt_skill{suf}.npz"
    num, Z = jload(fn), npz(fz)
    A = num["arm"]
    NAME = {A: "Foehn in real time (1 km)", "rt04": "Foehn in real time (40 km)", "cams": "CAMS forecast", "clim": "Climatology (1 km)"}
    REF = {"cams": "CAMS forecast", "rt04": "Foehn in real time (40 km), its input", "clim": "Climatology (1 km)"}
    est = lambda v: [sig(v["est"]), sig(v["ci"][0]), sig(v["ci"][1])]
    # a: the 2-degree cells (L94-106) and the share of the coloured cells in each colour class (L119-133)
    M = num["a_map"]
    chg = 100.0 * (Z[f"cell_rmse_{A}"] / Z["cell_rmse_cams"] - 1.0)
    okc = Z["cell_n"] >= M["min_station_days_to_colour"]
    rows = [[dec(la, 1), dec(lo, 1), int(n), int(m), sig(a), sig(b), sig(c), bool(k)] for la, lo, n, m, a, b, c, k in
            zip(Z["cell_lat0"], Z["cell_lon0"], Z["cell_n"], Z["cell_monitors"],
                Z[f"cell_rmse_{A}"], Z["cell_rmse_cams"], chg, okc)]
    sheet(f"{sh}a", fg, "a",
          "Change in root-mean-square error of the real-time 1 km forecast against the CAMS forecast of the same cycle (%), by 2° cell "
          "(southern and western edges in degrees), with the station-days, the monitors and the RMSE of both forecasts (µg m-3); cells "
          f"with fewer than {int(M['min_station_days_to_colour'])} station-days are grey in the map.",
          ["southern edge (°N)", "western edge (°E)", "station-days", "monitors", "RMSE, 1 km forecast (µg m-3)", "RMSE, CAMS forecast (µg m-3)",
           "change in RMSE (%)", "coloured (not grey)"], rows, scr, [fz], note=why)
    cnt = np.array(M["cells_per_class"], float)
    shr = 100.0 * np.concatenate([[cnt[0] + cnt[1]], cnt[2:-2], [cnt[-2] + cnt[-1]]]) / cnt.sum()
    cb = M["colour_class_bounds"]
    labs = [f"below {cb[0]}"] + [f"{a} to {b}" for a, b in zip(cb[:-1], cb[1:])] + [f"above {cb[-1]}"]
    cc = np.concatenate([[cnt[0] + cnt[1]], cnt[2:-2], [cnt[-2] + cnt[-1]]])
    rows = [[lab, int(c), sig(s)] for lab, c, s in zip(labs, cc, shr)]
    sheet(f"{sh}a_classes", fg, "a",
          "Share of the coloured 2° cells in each class of the change in RMSE against the CAMS forecast (%): the bars above the colour "
          "classes of the map and the percentages written on them.",
          ["class of the change in RMSE (%)", "cells", "share of the coloured cells (%)"], rows, scr, [fn])
    # b: by lead day (L148-166)
    Bl = num["b_lead"]; lead = range(1, 6)
    rows = [["RMSE (µg m-3)", NAME[k], None, L, sig(v), None, None, int(n)] for k in (A, "rt04", "cams", "clim")
            for L, v, n in zip(lead, Bl["rmse"][k], Bl["station_days"])]
    rows += [["change in RMSE of the 1 km forecast (%)", NAME[A], REF[b], L] + est(v) + [int(v["n"])] for b in ("cams", "rt04", "clim")
             for L, v in zip(lead, Bl["change_pooled"][b])]
    sheet(f"{sh}b", fg, "b",
          "Root-mean-square error (µg m-3) of the four systems at the monitors by lead day (top), and change in RMSE of the real-time 1 km "
          "forecast against each of the three others (%) with its 95% interval (bottom); lead day 1 is the day of issue.",
          ["quantity", "system", "against", "lead day", "value", "lower bound", "upper bound", "station-days"], rows, scr, [fn])
    # c: by region (L173-202)
    Cr = num["c_regions"]
    regs = sorted(Cr["order_of_scorer"], key=lambda g: Cr["change"]["cams"][g]["est"])
    rowsc = [("all", "All monitors"), ("regmean", "Mean of the regions")] + [(g, Cr["figure_names"][g]) for g in regs]
    mons = {"all": num["sample"]["monitors"], **Cr["monitors"]}
    rows = [[name, REF[b]] + est(Cr["change"][b][key]) + [mons.get(key), int(Cr["change"][b][key]["n"]) if key not in ("regmean",) else None]
            for b in ("cams", "clim") for key, name in rowsc]
    sheet(f"{sh}c", fg, "c",
          "Change in root-mean-square error of the real-time 1 km forecast against the CAMS forecast and against the climatology (%), over "
          "all monitors, as the mean of the regions and by region, with 95% intervals and the number of monitors.",
          ["region", "against", "change in RMSE (%)", "lower bound (%)", "upper bound (%)", "monitors", "station-days"], rows, scr, [fn])
    # d: away from the monitor (L205-214)
    Dr = num["d_rings"]
    where = ["pixel of the monitor (0 km)", "ring 1 km away", "ring 3 km away", "ring 10 km away", "mean of the 0.25° cell"]
    rows = [[w_, REF[b]] + est(v) + [int(v["n"])] for b in ("cams", "rt04") for w_, v in zip(where, Dr["vs"][b]["pooled"])]
    sheet(f"{sh}d", fg, "d",
          "Change in root-mean-square error (%) of the real-time 1 km forecast read at the monitor's pixel, on rings 1, 3 and 10 km around it "
          "and as the mean of its 0.25° cell, against the CAMS forecast and against its 0.4° input, with 95% intervals.",
          ["where the 1 km forecast is read", "against", "change in RMSE (%)", "lower bound (%)", "upper bound (%)", "station-days"], rows, scr, [fn])
    # e: by day of issue (L223-251)
    E = num["e_days"]
    rows = [[dd, "live" if lv else "reforecast", REF[b]] + est(v) + ["yes" if (v["ci"][1] < 0 or v["ci"][0] > 0) else "no", int(v["n"])]
            for b in ("cams", "clim") for dd, lv, v in zip(E["issue_day"], E["live"], E["change_pooled"][b])]
    sheet(f"{sh}e", fg, "e",
          "Change in root-mean-square error of the real-time 1 km forecast against the CAMS forecast and against the climatology (%), one "
          "value per day of issue (lead days 1-5 pooled), with its 95% interval over 2° cells (the figure draws the values; filled symbol, "
          "interval excluding zero).",
          ["day of issue", "production", "against", "change in RMSE (%)", "lower bound (%)", "upper bound (%)", "interval excludes zero (filled)",
           "station-days"], rows, scr, [fn])
    panel(fg, "a", "world map of 2° cells coloured by the change in RMSE against the CAMS forecast, with the share of cells per colour class (bars)",
          f"{sh}a (cells); {sh}a_classes (bars)")
    panel(fg, "b", "RMSE by lead day of four systems (lines) and change of the 1 km forecast against each (points with intervals)", f"{sh}b")
    panel(fg, "c", "change in RMSE by region against the CAMS forecast and the climatology (forest plot)", f"{sh}c")
    panel(fg, "d", "change in RMSE with the distance from the monitor (points with intervals)", f"{sh}d")
    panel(fg, "e", "change in RMSE by day of issue (lines and points)", f"{sh}e")
    if not with_f:
        return
    # f: the members at the monitors (L267-287)
    F = num["f_ensemble"]; arms = list(F["arms"])
    rows = [[F["names"][x], k, sig(v), sig(F["expected"]["rank_share_pct"])] for x in arms for k, v in enumerate(F[x]["rank_hist_pct"])]
    sheet(f"{sh}f_ranks", fg, "f",
          "Share of the station-days (%) by the number of the eight members that are below the measurement (rank histogram, left part of "
          "the panel), with the share expected from a calibrated ensemble (dashed line).",
          ["ensemble", "members below the measurement", "station-days (%)", "calibrated ensemble (%)"], rows, scr, [fn])
    rows = [[F["names"][x], sig(s), sig(r)] for x in arms for s, r in zip(F[x]["spread_classes"]["spread"], F[x]["spread_classes"]["rmse"])]
    sheet(f"{sh}f_spread", fg, "f",
          "Root-mean-square error of the ensemble mean (µg m-3) by class of ensemble spread (µg m-3) at the monitors (right part of the "
          "panel, logarithmic axes); a calibrated ensemble lies on the diagonal.",
          ["ensemble", "ensemble spread (µg m-3)", "RMSE of the ensemble mean (µg m-3)"], rows, scr, [fn])
    panel(fg, "f", "rank histogram of the eight members (bars) and RMSE against spread (line)", f"{sh}f_ranks; {sh}f_spread")


def rt_skill_main(fg, sh):
    rt_skill(fg, sh, stem="fig_rt_skill_main", with_f=False)


def rt_ensemble(fg, sh):
    """figures_new/fig_rt_skill/figS_rt_ensemble.py: the members at the monitors, as issued and with the calibrated noise
    (numbers_cal.json, key f_ensemble)."""
    d = NEW / "fig_rt_skill"; scr = d / "figS_rt_ensemble.py"
    fn = d / "numbers_cal.json"
    F = jload(fn)["f_ensemble"]
    arms = list(F["arms"]); assert len(arms) == 2, arms
    names = {arms[0]: "As issued", arms[1]: "Calibrated noise"}
    rows = [[names[x], k, sig(v), sig(F["expected"]["rank_share_pct"])] for x in arms for k, v in enumerate(F[x]["rank_hist_pct"])]
    sheet(f"{sh}a", fg, "a",
          "Share of the station-days (%) by the number of the eight members of the real-time 1 km forecast that are below the measurement "
          "(rank histogram), as issued and with the calibrated noise, with the share expected from a calibrated ensemble (dashed line).",
          ["ensemble", "members below the measurement", "station-days (%)", "calibrated ensemble (%)"], rows, scr, [fn],
          note=f"station-days {F.get('station_days')}")
    rows = [[names[x], sig(s_), sig(r)] for x in arms for s_, r in zip(F[x]["spread_classes"]["spread"], F[x]["spread_classes"]["rmse"])]
    sheet(f"{sh}b", fg, "b",
          "Root-mean-square error of the ensemble mean (µg m-3) by class of ensemble spread (µg m-3) at the monitors (logarithmic axes), as "
          "issued and with the calibrated noise; a calibrated ensemble lies on the diagonal.",
          ["ensemble", "ensemble spread (µg m-3)", "RMSE of the ensemble mean (µg m-3)"], rows, scr, [fn])
    panel(fg, "a", "rank histogram of the eight members, as issued and calibrated (bars)", f"{sh}a")
    panel(fg, "b", "RMSE of the ensemble mean against ensemble spread, as issued and calibrated (lines)", f"{sh}b")


# ================================================================================================ SUPPLEMENTARY FIGURES
# The European benchmark (Supplementary Figs. 1-3 and 11): figures/fig_style.py names and evaluation folders
SEEDS = ("", "_s43", "_s44")
EU_NAME = {"foehn": "Foehn",                 # the labels of the figures; there "Foehn" is the 0.25° network of the European benchmark
            "nofp": "Foehn, no transport layer", "convlstm": "ConvLSTM", "earthformer": "Earthformer",
           "simvp": "SimVP", "climax": "ClimaX", "topoflow": "TopoFlow", "aurora": "Aurora", "cams": "CAMS (operational)",
           "climatology": "Climatology"}
EU_RUNS = {"foehn": [EXP / f"dejavu/runs/noise_t31_fpstab{s}/eval1km_hres_none" for s in SEEDS],
           "nofp": [EXP / f"dejavu/runs/noise_t31{s}/eval1km_hres_none" for s in SEEDS],
           **{m: [EXP / f"retrain/runs/B_{m}_mettarget31{s}/eval1km_hres_none" for s in SEEDS]
              for m in ("convlstm", "earthformer", "simvp", "climax", "topoflow")},
           "cams": [EXP / "benchmark/eval1km_cams"], "aurora": [EXP / "benchmark/eval1km_aurora"],
           "climatology": [EXP / "benchmark/eval1km_climatology"]}
EU_LEADS = (1, 2, 3, 4)


def eu_load(model, key):
    """COPY of figures/fig_style.py:metric and load (L83-106): mean and standard deviation over the trainings, per lead time"""
    get = {"rmse_025": lambda v: v["coarse_025deg"]["rmse"], "rmse_1km": lambda v: v["grid"]["model"]["rmse"],
           "rmse_eea": lambda v: v["stations"]["model_rmse"], "csi_025": lambda v: v["coarse_025deg"]["csi25"]}[key]
    vals = []
    for d in EU_RUNS[model]:
        f = Path(d) / "metrics.json"
        if f.exists():
            res = jload(f); vals.append([get(res[f"T+{h}"]) for h in EU_LEADS])
    if not vals:
        return np.full(4, np.nan), np.full(4, np.nan), 0
    v = np.array(vals, float)
    return v.mean(0), v.std(0), len(vals)


def sfig1(fg, sh):
    """figures/fig_spectra_cost.py (rendered with NATURE=1)."""
    scr = FIG / "fig_spectra_cost.py"

    def radial(a, win, kr, bins):                         # L25-29
        p = np.abs(np.fft.fft2(a * win)) ** 2
        idx = np.digitize(kr.ravel(), bins) - 1
        ok = (idx >= 0) & (idx < len(bins) - 1)
        return np.bincount(idx[ok], p.ravel()[ok], len(bins) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(bins) - 1), 1)

    def setup(nr, nc, nbins):                             # L32-37
        win = np.outer(np.hanning(nr), np.hanning(nc))
        ky, kx = np.meshgrid(np.fft.fftfreq(nr), np.fft.fftfreq(nc), indexing="ij")
        kr = np.sqrt(kx ** 2 + ky ** 2)
        bins = np.logspace(np.log10(1.0 / max(nr, nc)), np.log10(0.5), nbins)
        return win, kr, bins, np.sqrt(bins[1:] * bins[:-1])

    def err_ratio(pairs, ok, win, kr, bins):              # L40-47
        ae = at = 0
        for f, t in pairs:
            m = ok & np.isfinite(t) & (t > 0)
            ae = ae + radial(np.where(m, f - t, 0.0), win, kr, bins)
            at = at + radial(np.where(m, t - t[m].mean(), 0.0), win, kr, bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ae / at

    # a: 0.25 degree, central Europe, 61 start days of 2022, lead 3 days (L50-63, L111-116)
    R0, C0, NR, NC, LEAD_I = 48, 104, 64, 128, 1

    def load25(name):
        z = npz(DATA / f"{name}.npz"); return {int(d): i for i, d in enumerate(z["days"])}, z["pred"].astype(np.float32)
    tdays, truth = load25("truth")
    land25 = np.isfinite(truth[0, 0, 0, R0:R0 + NR, C0:C0 + NC])
    win25, kr25, bins25, kc25 = setup(NR, NC, 22)
    wl25 = 27.8 / kc25
    rows = []
    for name in ("topoflow", "earthformer", "simvp", "convlstm", "nofp", "foehn"):
        days, pred = load25(name)
        pairs = [(pred[i, LEAD_I, :, R0:R0 + NR, C0:C0 + NC].mean(0), truth[tdays[d], LEAD_I, 0, R0:R0 + NR, C0:C0 + NC]) for d, i in days.items() if d in tdays]
        r = err_ratio(pairs, land25, win25, kr25, bins25)
        for a, b in zip(wl25, r):
            if np.isfinite(b):
                rows.append([EU_NAME[name], sig(a), sig(b), len(pairs)])
    sheet(f"{sh}a", fg, "a",
          "Power of the forecast error divided by the power of the 1 km estimates (deviations from their mean over the area; dimensionless), "
          "by wavelength (km), at 0.25° over central Europe, 3 days ahead, summed over the start days of 2022; as in the figure, the label Foehn "
          "denotes the 0.25° network of the European benchmark, not the global 1 km model.",
          ["network", "wavelength (km)", "error power / signal power", "start days"], rows, scr, [DATA / "<network>.npz", DATA / "truth.npz"])
    # b: the same at 1 km on the Paris-Florence window (L66-90, L122-131)
    WR0, WC0, WNR, WNC = 2250, 2500, 768, 1536
    fst = EXP / "fullimage/data/static1km.npz"; FINE = EXP / "fullimage/data/ghap1km.npy"
    land1 = npz(fst)["land"][WR0:WR0 + WNR, WC0:WC0 + WNC] > 0
    win1, kr1, bins1, kc1 = setup(WNR, WNC, 24)
    HDR = npy_header(used(FINE))[0]; OFF22 = 365 * 4 + 1 + 365
    tcache = {}

    def truth1(d):
        if d not in tcache:
            a = np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + d) * 4192 * 6992 * 2, shape=(4192, 6992))
            tcache[d] = np.asarray(a[WR0:WR0 + WNR, WC0:WC0 + WNC], np.float32)
        return tcache[d]
    REG = EXP / "regional/runs"
    rows = []
    for evaldir, lab in [(EXP / "retrain/runs/B_convlstm_mettarget31/eval_region", "ConvLSTM (interp.)"),
                         (REG / "simvp/eval_region", "SimVP (1 km)"),
                         (REG / "earthformer/eval_region", "Earthformer (1 km)"),
                         (EXP / "dejavu/runs/noise_t31_fpstab/eval_region", "Foehn (interp.)"),
                         (REG / "flow/eval_region", "Foehn (1 km window)")]:
        pairs = []
        for p in sorted(glob.glob(str(evaldir / "part_*.npz"))):
            z = np.load(p)
            for k in z.files:
                if k.startswith("pred_"):
                    used(p)
                    d, h = (int(x[1:]) for x in k[5:].split("_"))
                    pairs.append((z[k][WR0:WR0 + WNR, WC0:WC0 + WNC].astype(np.float32), truth1(d + h)))
        r = err_ratio(pairs, land1, win1, kr1, bins1)
        for a, b in zip(1.0 / kc1, r):
            if np.isfinite(b):
                rows.append([lab, sig(a), sig(b), len(pairs)])
    sheet(f"{sh}b", fg, "b",
          "Power of the forecast error divided by the power of the 1 km estimates (dimensionless), by wavelength (km), at 1 km on the window "
          "from Paris to Florence, summed over 16 forecasts of 2022 (four start days, 1 to 4 days ahead); labels as in the figure: (1 km), "
          "networks trained at 1 km on the window; (interp.), 0.25° networks interpolated to 1 km; Foehn, networks of the European benchmark.",
          ["network", "wavelength (km)", "error power / signal power", "forecasts"], rows, scr,
          [EXP / "<network run>" / "eval_region" / "part_*.npz", FINE, fst])
    # c: skill against cost (L139-146)
    rows = []
    for name in ("foehn", "convlstm", "earthformer", "simvp", "topoflow", "climax", "aurora"):
        c = jload(DATA / f"cost_{name}.json")
        mean, _, n = eu_load(name, "rmse_1km")
        rows.append([EU_NAME[name], sig(c["seconds"]), sig(float(np.mean(mean))), sig(c["peak_gb"]), n])
    sheet(f"{sh}c", fg, "c",
          "Root-mean-square error against the 1 km estimates (µg m-3; forecasts interpolated to 1 km, mean over lead times of 1 to 4 days, "
          "Europe, 2022) against the time to compute one member for one lead time on one MI250X graphics compute die (s), with the peak GPU "
          "memory (GB) that sets the marker area; the label Foehn denotes the 0.25° network of the European benchmark.",
          ["system", "inference time (s per member per lead time)", "RMSE at 1 km (µg m-3)", "peak GPU memory (GB)", "trainings averaged"],
          rows, scr, [DATA / "cost_<system>.json", EXP / "<run>" / "eval1km_hres_none" / "metrics.json"])
    panel(fg, "a", "error power relative to signal power by wavelength at 0.25°, six networks (lines)", f"{sh}a")
    panel(fg, "b", "the same at 1 km on a window, five networks (lines)", f"{sh}b")
    panel(fg, "c", "RMSE at 1 km against inference time, seven systems (points; inset = the same points)", f"{sh}c")


def sfig2(fg, sh):
    """figures/fig_leadtime.py (rendered with NATURE=1)."""
    scr = FIG / "fig_leadtime.py"
    ORDER = ["climatology", "cams", "aurora", "climax", "topoflow", "simvp", "earthformer", "convlstm", "nofp", "foehn"]
    PAN = [("a", "rmse_025", "Root-mean-square error (µg m-3) against the 0.25° means of the 1 km estimates"),
           ("b", "rmse_1km", "Root-mean-square error (µg m-3) against the 1 km estimates, the 0.25° forecasts being interpolated"),
           ("c", "rmse_eea", "Root-mean-square error (µg m-3) against the measurements of the European Environment Agency stations"),
           ("d", "csi_025", "Critical success index of exceedances of 25 µg m-3 at 0.25° (dimensionless)")]
    for L, key, what in PAN:
        rows = []
        for m in ORDER:
            mean, sd, n = eu_load(m, key)
            if n == 0 or np.all(np.isnan(mean)):
                continue
            for h, a, b in zip(EU_LEADS, mean, sd):
                rows.append([EU_NAME[m], h, sig(a), sig(b) if n > 1 else None, n, "yes" if (n > 1 and m == "foehn") else "no"])
        sheet(f"{sh}{L}", fg, L,
              f"{what}, by lead time (days), forecasts of 2022 over Europe: mean over the trainings of each network, with the standard "
              "deviation over trainings (drawn as a band for the network labelled Foehn, the 0.25° network of the European benchmark, only).",
              ["system", "lead time (days)", "score (mean over trainings)", "standard deviation over trainings", "trainings", "band drawn"],
              rows, scr, [EXP / "<run>" / "eval1km_hres_none" / "metrics.json"])
        panel(fg, L, f"{what.split(' (')[0].lower()} by lead time, ten systems (lines)", f"{sh}{L}")


def sfig3(fg, sh):
    """figures/fig_extremes.py (rendered with NATURE=1): the scorecard, lines 13-99."""
    scr = FIG / "fig_extremes.py"
    EPIS = [(69, 79), (83, 89), (187, 201), (217, 225)]
    fm, fp = EXP / "dejavu/data/meta.npz", EXP / "dejavu/data/pm.npy"
    meta = npz(fm); land = meta["land"].astype(bool)
    i0 = int(np.where(meta["year"] == 2022)[0][0])
    pm = np.load(used(fp), mmap_mode="r")
    TRUTH = {d: np.asarray(pm[i0 + d], np.float32) for d in range(365)}
    FRAC = {d: float(np.nanmean(np.where(land, TRUTH[d], np.nan) > 25)) for d in range(365)}

    def rows_flow(run):
        r = np.load(used(EXP / "dejavu/runs" / run / "records_2022_hres_rec8.npy"))
        return {(int(x[1]), int(x[2])): np.array([x[3], x[4], x[5], x[10], x[11]]) for x in r}

    def rows_base(run):
        r = np.load(used(EXP / "retrain/runs" / run / "records_coarse_2022_hres.npy"))
        return {(int(x[0]), int(x[1])): np.array([x[2], x[3], x[4], x[5], x[6]]) for x in r}

    def rows_grid(npy):
        arr = np.load(used(npy), mmap_mode="r")
        out = {}
        for d in range(365):
            for h in EU_LEADS:
                if d + h >= 365 or not np.isfinite(arr[d, h - 1]).all():
                    continue
                f, t = np.asarray(arr[d, h - 1], np.float32), TRUTH[d + h]
                m = land & np.isfinite(t) & (t > 0)
                e, g = t[m] > 25, f[m] > 25
                out[d, h] = np.array([m.sum(), ((f[m] - t[m]) ** 2).sum(), np.abs(f[m] - t[m]).sum(), (e & g).sum(), (e | g).sum()])
        return out

    MODELS = {"foehn": [rows_flow(f"noise_t31_fpstab{s}") for s in SEEDS], "nofp": [rows_flow(f"noise_t31{s}") for s in SEEDS],
              **{m: [rows_base(f"B_{m}_mettarget31{s}") for s in SEEDS] for m in ("convlstm", "earthformer", "simvp", "topoflow", "climax")},
              "aurora": [rows_grid(EXP / "benchmark/aurora/out/aurora_2022.npy")], "cams": [rows_grid(EXP / "benchmark/cams_forecast_2022.npy")],
              "climatology": [rows_grid(EXP / "benchmark/climatology_2022.npy")]}
    common = set.intersection(*[set(r) for rs in MODELS.values() for r in rs])

    def dayset(name, h):
        keys = sorted(k for k in common if k[1] == h)
        fr = np.array([FRAC[d + h] for d, _ in keys])
        if name == "all":
            return keys
        if name == "top10":
            return [k for k, f in zip(keys, fr) if f >= np.quantile(fr, 0.9)]
        return [k for k in keys if any(a <= k[0] <= b for a, b in EPIS)]

    def score(rows, keys, metric):
        s = np.sum([rows[k] for k in keys], 0)
        return {"rmse": np.sqrt(s[1] / s[0]), "crps": s[2] / s[0], "csi": s[3] / max(s[4], 1)}[metric]

    ORDER = ["climatology", "cams", "aurora", "climax", "topoflow", "simvp", "earthformer", "convlstm", "nofp", "foehn"]
    COLS = [("all", "All days"), ("top10", "10% most polluted days"), ("episodes", "Transported episodes")]
    ROWS = [("rmse", "RMSE (µg m-3)"), ("crps", "CRPS (µg m-3)"), ("csi", "CSI, > 25 µg m-3")]
    summary, nfc = {}, {}
    for dset, _ in COLS:
        for h in EU_LEADS:
            nfc[dset, h] = len(dayset(dset, h))
        for metric, _ in ROWS:
            for m in ORDER:
                vals = np.array([[score(r, dayset(dset, h), metric) for h in EU_LEADS] for r in MODELS[m]])
                summary[m, dset, metric] = vals.mean(0)
    SYSR = ["nofp", "convlstm", "earthformer", "simvp", "topoflow", "climax", "aurora", "cams", "climatology"]
    rows = []
    for m in SYSR:
        for dset, dname in COLS:
            for metric, mname in ROWS:
                for h in EU_LEADS:
                    f, o = summary["foehn", dset, metric][h - 1], summary[m, dset, metric][h - 1]
                    D = 100 * (f / o - 1) * (-1 if metric == "csi" else 1)         # negative = the network is better, for every score
                    rows.append([EU_NAME[m], dname, mname, h, sig(D), sig(f), sig(o), nfc[dset, h]])
    sheet(f"{sh}", fg, "single panel",
          "Scorecard: change of the score of the 0.25° network of the European benchmark (labelled Foehn in the figure) relative to each system (%; sign reversed for the "
          "critical success index, so that a negative value always means that the network is better), by group of days, score and lead time "
          "(days), with the two scores behind each cell; land cells of Europe, 2022, 0.25°, the same start days for every system.",
          ["compared system (row)", "days", "score", "lead time (days)", "change of the network's score (%)", "score of the network",
           "score of the compared system", "forecasts in the group"], rows, scr,
          [EXP / "dejavu/runs/<run>/records_2022_hres_rec8.npy", EXP / "retrain/runs/<run>/records_coarse_2022_hres.npy",
           EXP / "benchmark/aurora/out/aurora_2022.npy", EXP / "benchmark/cams_forecast_2022.npy", EXP / "benchmark/climatology_2022.npy", fp, fm],
          note=f"common samples (start day, lead): {len(common)}")
    panel(fg, "single panel", "scorecard of 9 systems x 3 groups of days x 3 scores x 4 lead times (coloured cells)", f"{sh}")


def sedi(c):
    """COPY of figures/fig_gc4_extremes.py:sedi (L35-39): symmetric extremal dependence index from (hits, misses, false alarms, correct negatives)"""
    h, m, f, n = np.moveaxis(c, -1, 0)
    with np.errstate(all="ignore"):
        H = np.clip(h / (h + m), 1e-6, 1 - 1e-6); F = np.clip(f / (f + n), 1e-6, 1 - 1e-6)
        return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))


def rev_curves(cnt, alphas):
    """COPY of figures/fig_gc4_extremes.py:rev_curves (L23-32): relative economic value, envelope over probability thresholds"""
    n0, n1 = cnt[:, 0].astype(float), cnt[:, 1].astype(float)
    s = n1.sum() / (n0.sum() + n1.sum())
    H = np.cumsum(n1[::-1])[::-1] / n1.sum(); F = np.cumsum(n0[::-1])[::-1] / n0.sum()
    keep = (H > 0) | (F > 0)
    H, F = H[keep][1:] if keep.sum() > 1 else H[keep], F[keep][1:] if keep.sum() > 1 else F[keep]
    a = alphas[:, None]
    V = (np.minimum(a, s) - (a * (H * s + F * (1 - s)) + (1 - H) * s)) / (np.minimum(a, s) - a * s)
    return np.max(V, 1)


def sfig4(fg, sh):
    """analyses/issuance/scripts/fig_exposure_eqiss.py and extremes_panels_eqiss.py (drew work/figs/figS_exposure.pdf)."""
    scr = ANA / "issuance" / "scripts" / "extremes_panels_eqiss.py"
    ft = ANA / "issuance" / "data" / f"tail_events_{LABEL}_d1.npz"
    te = npz(ft); CT = te["C"]                           # (region, lead, tier, system, [hits, misses, false alarms, correct negatives])
    TS = [str(x) for x in te["systems"]]; TR = [str(r) for r in te["regions"]]
    SYSE = [("Foehn", f"{LABEL} (mean)"), ("CAMS", "CAMS"), ("Aurora", "Aurora")]
    L = np.arange(1, 8); gl = TR.index("Global")
    rows = []
    for nm, key in SYSE:                                  # g: L36-39
        c = CT[gl, :, 2, TS.index(key)]; v = sedi(c); ok = np.isfinite(v) & (c[:, :2].sum(-1) > 0)
        for h, x, cc in zip(L[ok], v[ok], c[ok]):
            rows.append([nm, int(h), sig(x), int(cc[0]), int(cc[1]), int(cc[2]), int(cc[3])])
    sheet(f"{sh}g", fg, "g",
          "Symmetric extremal dependence index (dimensionless) of the days above the local 99th percentile of 2017-2020, over all land in "
          "2022, by lead time (days), for Foehn's ensemble mean and the CAMS and Aurora forecasts issued at the same time (which cover 5 "
          "days), with the counts of 0.25° cell-days behind it.",
          ["system", "lead time (days)", "SEDI", "hits", "misses", "false alarms", "correct negatives"], rows, scr, [ft])
    rows = []
    for r in [r for r in TR if r != "Global"]:            # h: L45-52
        for nm, key in SYSE:
            rows.append([r, nm, sig(float(sedi(CT[TR.index(r), 0, 2, TS.index(key)])))])
    sheet(f"{sh}h", fg, "h",
          "Symmetric extremal dependence index (dimensionless) of the days above the local 99th percentile, 1 day ahead, by region.",
          ["region", "system", "SEDI, 1 day ahead"], rows, scr, [ft])
    fe = DATA / f"extremes_stations_{LABEL}.npz"
    es = npz(fe); CE = es["C"]; ES = [str(x) for x in es["systems"]]
    alphas = np.logspace(np.log10(0.003), np.log10(0.6), 250)
    for Lr, tier, pct in (("i", 1, "95th"), ("j", 2, "99th")):       # i, j: L60-68
        rows = []
        for nm, key in (("Foehn", LABEL), ("CAMS", "CAMS"), ("Aurora", "Aurora")):
            for hh, lead in ((0, 1), (2, 3)):
                v = rev_curves(CE[0, tier, hh, ES.index(key)], alphas)
                for a, x in zip(alphas, v):
                    rows.append([nm, lead, sig(a), sig(x), "yes" if a >= 0.01 else "no"])
        sheet(f"{sh}{Lr}", fg, Lr,
              f"Retrospective potential relative economic value (dimensionless) of warnings of days above the local {pct} percentile at the "
              "monitoring stations, against the ratio of the cost of protection to the loss avoided, 1 and 3 days ahead.",
              ["system", "lead time (days)", "cost / avoided loss", "relative economic value", "inside the axis range (0.01-0.6)"], rows, scr, [fe])
    panel(fg, "a-c", "Cairo on 2 January 2022: 1 km estimates, the same on 40 km cells, residents seen or hidden", "maps, not exported")
    panel(fg, "d-f", "Santiago de Chile on 20 June 2022: the same three maps", "maps, not exported")
    panel(fg, "g", "SEDI of local extremes by lead time, three systems (lines)", f"{sh}g")
    panel(fg, "h", "SEDI 1 day ahead by region, three systems (points)", f"{sh}h")
    panel(fg, "i", "relative economic value of warnings, days above the local 95th percentile (lines)", f"{sh}i")
    panel(fg, "j", "relative economic value of warnings, days above the local 99th percentile (lines)", f"{sh}j")


def warned_v1(fg, sh):
    """analyses/hidden/fig_hidden_warned.py (drew work/figs/figS_hidden_warned.pdf); every number from results/numbers.json."""
    scr = ANA / "hidden" / "fig_hidden_warned.py"
    f = ANA / "hidden" / "results" / "numbers.json"
    N = jload(f)["systems"]
    SYSTEMS = (("foehn_p50", "Foehn (at least 2 of 4 members)"), ("foehn_40km", "Foehn on 40 km cells (ensemble mean)"),
               ("cams", "CAMS forecast"), ("aurora", "Aurora"))
    S = N["thr37.5"]
    rows = []
    for s, name in SYSTEMS:                               # a: L66-76
        for h in (1, 3):
            d = S[f"lead{h}"]["World"][s]
            hid, fa = d["hidden_warned_M"][0], d["false_M"][0]; seen = d["right_M"][0] - hid
            rows.append([name, h, sig(seen), sig(hid), sig(fa), sig(d["right_M"][0]), sig(d["warned_M"][0]), sig(d["exposed_M"][0])])
    sheet(f"{sh}a", fg, "a",
          "People warned a day above 37.5 µg m-3 (millions), by system, 1 and 3 days ahead: rightly warned and not hidden (solid), rightly "
          "warned and hidden by the 40 km grid (pink), wrongly warned (hatched), with their sums and the people exposed a day (dashed line).",
          ["system", "lead time (days)", "rightly warned, not hidden (millions)", "rightly warned, hidden by the 40 km grid (millions)",
           "wrongly warned (millions)", "rightly warned, total (millions)", "warned, total (millions)", "people exposed (millions)"],
          rows, scr, [f])
    rows = []
    for s, name in SYSTEMS:                               # b: L89-97
        for h in (1, 3):
            d = S[f"lead{h}"]["World"][s]
            v, m = d["hidden_warned_pct"], d["hidden_margin_warned_pct"]
            rows.append([name, h, sig(v[0]), sig(v[1]), sig(v[2]), sig(m[0])])
    sheet(f"{sh}b", fg, "b",
          "Share of the hidden people who are warned above 37.5 µg m-3 (%), by system, 1 and 3 days ahead, with its 95% interval over days "
          "(whiskers), and the same share when a resident counts as hidden only with a margin of 10% on both sides of the threshold (diamonds).",
          ["system", "lead time (days)", "hidden people warned (%)", "lower bound (%)", "upper bound (%)", "hidden with a margin of 10%, warned (%)"],
          rows, scr, [f])
    rows = []
    for t in ("15", "37.5", "75"):                        # c: L106-112
        for s, name in SYSTEMS:
            for h in (1, 3):
                d = N[f"thr{t}"][f"lead{h}"]["World"][s]
                rows.append([name, float(t), h, sig(d["correct_pct"][0]), sig(d["reach_pct"][0])])
    sheet(f"{sh}c", fg, "c",
          "Share of the warnings that are correct (%) and share of the exposed people whom the warnings reach (%), by system, threshold "
          "(µg m-3) and lead time (days).",
          ["system", "threshold (µg m-3)", "lead time (days)", "warnings that are correct (%)", "exposed people reached (%)"], rows, scr, [f])
    panel(fg, "a", "people warned a day, rightly, hidden, wrongly, four systems and two lead times (stacked bars)", f"{sh}a")
    panel(fg, "b", "share of the hidden people warned (bars, whiskers, diamonds)", f"{sh}b")
    panel(fg, "c", "reach against correct warnings at three thresholds (points)", f"{sh}c")


def aurora_daily_d(T):
    """Aurora initialised at 00 UTC of day T, daily mean of day T (lines 39-44 of figures/figED_regions.py)"""
    AURD = EXP / "benchmark" / "aurora"
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        if T in list(z["days"]):
            used(p); A = z["pred"][list(z["days"]).index(T)].astype(np.float32); alat, alon = z["lat"], z["lon"]
    an = npz(AURD / "raw" / f"{T:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9
    return np.maximum(0.25 * an + 0.5 * A[0] + 0.25 * A[1], 0), alat, alon


def sfig6(fg, sh):
    """figures/figED_regions.py: the mean absolute error and the spatial correlation written in the maps (lines 52-58, 75-78)."""
    scr = FIG / "figED_regions.py"
    T = 20
    REGIONS = {"North China Plain": (41.0, 38.0, 114.5, 118.5), "Delhi and the Ganges plain": (30.3, 27.3, 75.6, 79.6),
               "Po Valley": (46.3, 44.0, 7.3, 12.3), "Seoul and the Han basin": (38.3, 36.5, 126.0, 128.3)}

    def coarse_at(field, lat0, dlat, lon0, dlon, lats, lons):        # L27-30
        i = np.clip(np.round((lat0 - lats) / dlat).astype(int), 0, field.shape[0] - 1)
        j = np.round(((lons - lon0) % 360.0) / dlon).astype(int) % field.shape[1]
        return field[i[:, None], j[None, :]]
    ffm = JAN21_DIR / "map1km_d19_h1.npy"
    fm = np.load(used(ffm), mmap_mode="r")
    cams = np.asarray(cams_daily()[T, 0, 0], np.float32)
    aur, alat, alon = aurora_daily_d(T)
    rows = []
    for i, (rn, (N, S, Wl, E)) in enumerate(REGIONS.items()):
        r0, r1, c0, c1 = int(round((84 - N) * 100)), int(round((84 - S) * 100)), int(round((Wl + 180) * 100)), int(round((E + 180) * 100))
        lats, lons = 84 - 0.01 * (np.arange(r0, r1) + 0.5), -180 + 0.01 * (np.arange(c0, c1) + 0.5)
        g = ghap_box(1826 + T, r0, r1, c0, c1).astype(np.float32); g[(g <= 0) | ~land_box(r0, r1, c0, c1)] = np.nan
        ok = np.isfinite(g)
        f = np.where(ok, np.asarray(fm[r0:r1, c0:c1], np.float32), np.nan)
        c = np.where(ok, coarse_at(cams, 90.0, 0.4, 0.0, 0.4, lats, lons), np.nan)
        a = np.where(ok, coarse_at(aur, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]), lats, lons), np.nan)
        for j, (fld, nm) in enumerate(((c, "CAMS forecast, 0.4°"), (a, "Aurora, 0.4°"), (f, "Foehn, 1 km"))):
            m = ok & np.isfinite(fld)
            mae = np.mean(np.abs(fld[m] - g[m])); rr = np.corrcoef(fld[m], g[m])[0, 1]
            rows.append(["abcdefghijklmnopqrstuvwxy"[5 * i + j + 1], rn, nm, sig(mae), sig(rr), int(m.sum())])
    sheet(f"{sh}", fg, "b-d, g-i, l-n, q-s",
          "Mean absolute error (µg m-3) and spatial correlation against the 1 km estimates over the land pixels of each panel, daily mean "
          "PM2.5 on 21 January 2022: the numbers written in the maps of the CAMS forecast, Aurora and Foehn in four regions.",
          ["panel", "region", "system", "MAE (µg m-3)", "R", "land pixels"], rows, scr,
          [ffm, G1 / "ghap1km_tiles.npy", G1 / "static1km_tiles.npy", WORLD / "cams_forecast_daily" / "2022.zarr",
           EXP / "benchmark" / "aurora" / "out_12h_all" / "part_*.npz", EXP / "benchmark" / "aurora" / "raw" / "020.npz"])
    panel(fg, "a-t", "four regions x (1 km estimates, CAMS, Aurora, Foehn, Foehn minus the estimates): maps with MAE and R written in the forecast maps",
          f"maps not exported; printed numbers in {sh}")


def boot_inits(a, b, B=7, n=10000, seed=0):
    """COPY of figures/fig_gc3_scorecards.py:boot_inits (L31-40): relative change in mean per-initialization RMSE, 7-day moving blocks"""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    nd = len(a)
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]
    r = a[idx].mean(1) / b[idx].mean(1) - 1
    return 100 * (a.mean() / b.mean() - 1), 100 * np.percentile(r, 2.5), 100 * np.percentile(r, 97.5)


def sfig7(fg, sh):
    """figures/figED_aurora_regions.py (the submitted render differs from the current script only by the place of the panel letters)."""
    scr = FIG / "figED_aurora_regions.py"
    REG = ["Europe", "North America", "South America", "Africa", "South Asia", "East Asia"]
    Fm = fields12((72,))[72]
    f1, f2 = DATA / "cams12h_scores_foehn_emis.npz", DATA / "cams12h_clim.npz"
    cz, cl = npz(f1), npz(f2)
    CS = [str(s) for s in cz["systems"]]; CR = [str(r) for r in cz["regions"]]
    Lh = cz["leads_h"] / 24; k72 = list(cz["leads_h"]).index(72)
    w = np.cos(np.deg2rad(LAT04))[:, None] * np.ones((1, NX))
    LO2 = -180.0 + 0.4 * np.arange(NX)
    T = Fm["CAMS analysis"]
    rows_m, rows_c, rows_d = [], [], []
    for i, r in enumerate(REG):
        N, S, Wl, E = REG12[r]
        box = (LAT04[:, None] <= N) & (LAT04[:, None] > S) & (LO2[None, :] >= Wl) & (LO2[None, :] < E)
        for nm in ("Foehn-CAMS", "Aurora", "CAMS forecast"):          # L53-55
            e = np.sqrt((w * box * (Fm[nm] - T) ** 2).sum() / (w * box).sum())
            rows_m.append(["abcdef"[i], r, nm, sig(e)])
        g = CR.index(r)
        for key, nm in (("CAMS forecast", "CAMS forecast"), ("Aurora", "Aurora"), ("Foehn-CAMS (mean)", "Foehn-CAMS")):   # L68-69, L73-75
            for L, v in zip(Lh, cz["rmse"][CS.index(key), :, g]):
                rows_c.append(["abcdef"[i], r, nm, float(L), sig(v), "yes" if L >= 2.9 else "no"])
        for L, v in zip(Lh, cl["rmse"][:, list(cl["regions"]).index(r)]):
            rows_c.append(["abcdef"[i], r, "Climatology", float(L), sig(v), "no"])
        pt, lo, hi = boot_inits(cz["rmse_per_init"][CS.index("Foehn-CAMS (mean)"), :, k72, g], cz["rmse_per_init"][CS.index("Aurora"), :, k72, g])
        rows_d.append(["abcdef"[i], r, sig(pt), sig(lo), sig(hi), "yes" if hi < 0 else "no"])
    sheet(f"{sh}_maps", fg, "a-f",
          "Latitude-weighted root-mean-square error (µg m-3) against the CAMS analysis over each region of the near-surface PM2.5 forecast "
          "for 4 September 2022 at 00 UTC issued three days earlier: the numbers written in the maps.",
          ["panel", "region", "system", "RMSE of the map over the region (µg m-3)"], rows_m, scr, fields12((72,))["files"])
    sheet(f"{sh}_curves", fg, "a-f",
          "Latitude-weighted root-mean-square error (µg m-3) of each region by lead time (days, 12-hour steps) over May-November 2022 "
          "(214 initializations), for the three forecasts and the climatology of the CAMS analyses of 2018-2021; the enlarged panels repeat "
          "the three forecasts from 3 to 5 days.",
          ["panel", "region", "system", "lead time (days)", "RMSE (µg m-3)", "also drawn in the enlarged panel (days 3-5)"], rows_c, scr, [f1, f2])
    sheet(f"{sh}_change", fg, "a-f",
          "Change of the RMSE of Foehn-CAMS relative to Aurora three days ahead (%) with its 95% interval (7-day moving-block bootstrap over "
          "initializations, 10,000 resamples): the numbers written in the enlarged panels (blue when the interval excludes zero).",
          ["panel", "region", "change of RMSE, Foehn-CAMS against Aurora, 3 days ahead (%)", "lower bound (%)", "upper bound (%)",
           "interval below zero (blue)"], rows_d, scr, [f1])
    panel(fg, "a-f", "six regions: four maps (analysis and three forecasts, 3 days ahead) with the RMSE written, RMSE by lead time (lines) and the days 3-5 enlarged with the change against Aurora",
          f"maps not exported; printed RMSE in {sh}_maps; lines in {sh}_curves; printed change and interval in {sh}_change")


def sfig8(fg, sh):
    """analyses/reliability/scripts/fig_reliability.py (drew work/figs/figS_reliability.pdf); it reads four result files."""
    d = ANA / "reliability"; scr = d / "scripts" / "fig_reliability.py"
    f1, f2, f3, f4 = (d / "results" / n for n in ("station_scores_final.json", "station_scores_before.json", "estimates_scores.json", "members16_noise.json"))
    S, SB, EST, M16 = jload(f1), jload(f2), jload(f3), jload(f4)
    A, PP, NINE = "All official networks", "All populated pixels", "Nine regions pooled"
    THR = S["thresholds"]; PK = np.arange(5) / 4
    arr = lambda x: np.array([np.nan if v is None else v for v in x], float)
    st = lambda h: S["leads"][str(h)][A]
    pop = lambda h: EST["pop"]["leads"][str(h)][PP]["pixels"]
    V_EST, V_PIX, V_MEA = "1 km estimates, populated pixels", "1 km estimates, station pixels", "measurements, stations"
    for j, T in enumerate(THR):                           # a-c: L72-99
        rows = []
        for h in (1, 3):
            r = pop(h); f = arr(r["rel_freq"]["v"][j]); lo, hi = arr(r["rel_freq"]["ci_day"][0][j]), arr(r["rel_freq"]["ci_day"][1][j]); n = arr(r["rel_n"]["v"][j])
            for k in range(5):
                rows.append([V_EST, h, float(PK[k]), k, sig(f[k]), sig(lo[k]), sig(hi[k]), int(n[k]), sig(n[k] / n.sum()) if h == 1 else None])
        for h in (1, 3, 7):
            r = st(h); f = arr(r["rel_freq_0"]["v"][j]); lo, hi = arr(r["rel_freq_0"]["ci_day"][0][j]), arr(r["rel_freq_0"]["ci_day"][1][j]); n = arr(r["rel_n"]["v"][j])
            for k in range(5):
                rows.append([V_MEA, h, float(PK[k]), k, sig(f[k]), sig(lo[k]), sig(hi[k]), int(n[k]), sig(n[k] / n.sum()) if h == 1 else None])
        L = "abc"[j]
        sheet(f"{sh}{L}", fg, L,
              f"Reliability of the forecast probability that daily PM2.5 exceeds {T:g} µg m-3: observed frequency against the forecast "
              "probability (share of the four members above the threshold), with its 95% interval (block bootstrap over start days), against "
              "the 1 km estimates at the populated pixels and against the measurements at the stations, by lead time (days); the last column "
              "is the share of the forecasts in each probability class 1 day ahead (inset bars).",
              ["verification", "lead time (days)", "forecast probability", "members above the threshold", "observed frequency", "lower bound",
               "upper bound", "forecasts in the class", "share of the forecasts, 1 day ahead (inset)"], rows, scr, [f1, f3])
        panel(fg, L, f"reliability diagram, exceedance of {T:g} µg m-3 (points with intervals; inset bars)", f"{sh}{L}")
    # d: rank histogram (L111-119)
    series = [(V_EST, 1, "bar", pop(1)["rank"]["v"]), (V_EST, 3, "diamond", pop(3)["rank"]["v"]), (V_PIX, 1, "bar", st(1)["rank_1"]["v"]),
              (V_PIX, 7, "diamond", st(7)["rank_1"]["v"]), (V_MEA, 1, "bar", st(1)["rank_0"]["v"]), (V_MEA, 7, "diamond", st(7)["rank_0"]["v"])]
    rows = [[v, h, el, k + 1, sig(x)] for v, h, el, vals in series for k, x in enumerate(arr(vals))]
    sheet(f"{sh}d", fg, "d",
          "Frequency of each rank of the verification among the four members (1 = below all members, 5 = above all; 0.2 for a reliable "
          "ensemble), 1 day ahead (bars) and 7 days ahead or, at the populated pixels, 3 days ahead (diamonds).",
          ["verification", "lead time (days)", "drawn as", "rank of the verification", "frequency"], rows, scr, [f1, f3])
    # e: spread-to-error ratio by lead time (L128-144)
    leads = np.arange(1, 8)
    land = EST["land"]["2022 final"]["leads"]; land_b = EST["land"]["2022 before CRPS training"]["leads"]
    rows = [["1 km estimates, land (nine regions pooled)", int(h), sig(land[str(h)][NINE]["ss"]), sig(land[str(h)][NINE]["ss_ci"][0]), sig(land[str(h)][NINE]["ss_ci"][1])] for h in leads]
    rows += [["1 km estimates, land, before the CRPS training", int(h), sig(land_b[str(h)][NINE]["ss"]), None, None] for h in leads]
    rows += [[V_EST, h, sig(pop(h)["ss"]["v"]), sig(pop(h)["ss"]["ci_day"][0]), sig(pop(h)["ss"]["ci_day"][1])] for h in (1, 3)]
    rows += [[V_PIX, int(h), sig(st(h)["ss_1"]["v"]), None, None] for h in leads]
    rows += [[V_MEA, int(h), sig(st(h)["ss_0"]["v"]), sig(st(h)["ss_0"]["ci_day"][0]), sig(st(h)["ss_0"]["ci_day"][1])] for h in leads]
    rows += [["measurements, stations, before the CRPS training", int(h), sig(SB["leads"][str(h)][A]["ss_0"]["v"]), None, None] for h in leads]
    sheet(f"{sh}e", fg, "e",
          "Ratio of the spread of the ensemble to the root-mean-square error of its mean (dimensionless; 1 for a reliable ensemble) by lead "
          "time (days), with the 95% interval where the figure draws one.",
          ["verification", "lead time (days)", "spread / RMSE", "lower bound", "upper bound"], rows, scr, [f1, f2, f3])
    # f: by network (L153-161)
    nets = [g for g in S["groups"] if g != A]
    nets.sort(key=lambda g: S["leads"]["1"][g]["ss_0"]["v"])
    rows = []
    for g in nets:
        r1, r7 = S["leads"]["1"][g], S["leads"]["7"][g]
        for key, v in (("ss_1", V_PIX), ("ss_0", V_MEA)):
            rows.append([g, v, 1, sig(r1[key]["v"]), sig(r1[key]["ci_day"][0]), sig(r1[key]["ci_day"][1]), int(r1["n_stations"])])
            rows.append([g, v, 7, sig(r7[key]["v"]), None, None, int(r7["n_stations"]) if "n_stations" in r7 else None])
    sheet(f"{sh}f", fg, "f",
          "Ratio of the spread of the ensemble to the RMSE of its mean at the stations of each network, 1 day ahead (points with 95% "
          "intervals) and 7 days ahead (diamonds), against the measurements and against the 1 km estimates at the same pixels and days.",
          ["network", "verification", "lead time (days)", "spread / RMSE", "lower bound", "upper bound", "stations"], rows, scr, [f1])
    # g: share of the verifications inside the range of the members (L171-182)
    ea, gi = M16["A"]["ea16"]["leads"]["all"], M16["A"]["gir16"]["leads"]["all"]
    cats = [(4, "2022, 1 day ahead", 0.6, st(1)["cover_1"]["v"], st(1)["cover_0"]["v"], pop(1)["cover"]["v"]),
            (4, "East Asia, January 2022 (first four members)", 0.6, ea["M4_v1"]["cover"], ea["M4_v0"]["cover"], None),
            (16, "East Asia, January 2022", 15 / 17, ea["M16_v1"]["cover"], ea["M16_v0"]["cover"], None),
            (4, "Europe, July 2022 (first four members)", 0.6, gi["M4_v1"]["cover"], gi["M4_v0"]["cover"], None),
            (16, "Europe, July 2022", 15 / 17, gi["M16_v1"]["cover"], gi["M16_v0"]["cover"], None)]
    rows = []
    for m_, per, exp, ce, cm, cp in cats:
        rows.append([per, m_, "expected from a reliable ensemble, (M - 1) / (M + 1)", sig(exp)])
        if cp is not None:
            rows.append([per, m_, V_EST, sig(cp)])
        rows.append([per, m_, V_PIX, sig(ce)])
        rows.append([per, m_, V_MEA, sig(cm)])
    sheet(f"{sh}g", fg, "g",
          "Share of the verifications inside the range of the members (fraction; the axis of the figure is in %), with the share expected "
          "from a reliable ensemble of M members, for the four members over 2022 and for the 16-member evaluations with their first four members.",
          ["evaluation", "members", "verification", "share inside the range of the members"], rows, scr, [f1, f3, f4])
    panel(fg, "d", "rank histogram of the verification among four members (bars, diamonds)", f"{sh}d")
    panel(fg, "e", "spread / RMSE by lead time, six series (lines, bands, points)", f"{sh}e")
    panel(fg, "f", "spread / RMSE by network, 1 and 7 days ahead (points with intervals)", f"{sh}f")
    panel(fg, "g", "share of the verifications inside the member range (points)", f"{sh}g")


def sfig9(fg, sh):
    """analyses/issuance/scripts/fig_sedi_definitions.py P99 (drew work/figs/figS_sedi_definitions.pdf); it reads data/derived.json."""
    scr = ANA / "issuance" / "scripts" / "fig_sedi_definitions.py"
    f = ANA / "issuance" / "data" / "derived.json"
    R = jload(f); TIER = "P99"
    COLS = ["Local climate, same threshold", "Local climate, matched frequency", "Year 2022, same threshold", "Year 2022, matched frequency"]
    ROWS = [("land", "Global", ["ghap", "matched", "truth2022", "own"], "the 1 km estimates over all land"),
            ("stations", "all", ["ghap", "matched", "station", "own"], "the measurements at the stations")]
    letters = iter("abcdefgh")
    for dom, reg, kinds, truth in ROWS:
        for c, kind in enumerate(kinds):
            L = next(letters); d = R[dom][kind][TIER][reg]
            rows = []
            for nm, key in (("Foehn", "Foehn"), ("CAMS forecast", "CAMS"), ("Aurora", "Aurora")):
                v = np.array([np.nan if x is None else x for x in d[key]["sedi"]], float); n = int(np.isfinite(v).sum())
                for h in range(n):
                    rows.append([nm, h + 1, sig(v[h]), sig(d[key]["ci_low"][h]), sig(d[key]["ci_high"][h])])
            sheet(f"{sh}{L}", fg, L,
                  f"Symmetric extremal dependence index (dimensionless) of the days above the 99th percentile by lead time (days), against "
                  f"{truth}, definition '{COLS[c].lower()}', with the 95% interval (moving-block bootstrap over start days); the dotted "
                  "line of the panel is the value of the CAMS forecast 1 day ahead.",
                  ["system", "lead time (days)", "SEDI", "lower bound", "upper bound"], rows, scr, [f])
            panel(fg, L, f"SEDI by lead time against {truth}, {COLS[c].lower()} (lines with bands)", f"{sh}{L}")


def sfig10(fg, sh):
    """analyses/emissions/fig/figS_emission_lead.py (drew work/figs/figS_emission_lead.pdf); it reads control_gains.csv and control_scores.csv."""
    d = ANA / "emissions" / "fig"; scr = d / "figS_emission_lead.py"
    fgain, fscore = d / "control_gains.csv", d / "control_scores.csv"
    used(fgain, fscore)
    EMIS, TRAN = "emissions (e02 vs control)", "transport (A vs C)"
    G, S, R = {}, {}, {}                                  # read(): L32-40
    for r in csv.DictReader(open(fgain)):
        G[(r["region"], r["comparison"], r["metric"], int(r["lead_days"]))] = (float(r["gain_pct"]), float(r["ci95_lo"]), float(r["ci95_hi"]))
    for r in csv.DictReader(open(fscore)):
        S[(r["region"], r["model"], int(r["lead_days"]))] = float(r["spread_skill"])
        R[(r["region"], r["model"], int(r["lead_days"]))] = float(r["rmse"])
    L = range(1, 8)
    NAMEC = {TRAN: "computed transport (base model against the same network without the transport inputs)",
             EMIS: "emissions (model with the emission terms against the control without them)"}
    for k, reg in enumerate(["East Asia", "South Asia", "Europe", "North America", "South America", "Africa"]):
        rows = [[reg, NAMEC[comp], h] + [sig(x) for x in G[(reg, comp, "rmse", h)]] for comp in (TRAN, EMIS) for h in L]
        # first crossing, transport -> emissions (L61-68): the dotted line and the lead time written beside it
        t = np.array([G[(reg, TRAN, "rmse", h)][0] for h in L]); e = np.array([G[(reg, EMIS, "rmse", h)][0] for h in L])
        xf = np.linspace(1, 7, 601); tf, ef = np.interp(xf, np.arange(1, 8), t), np.interp(xf, np.arange(1, 8), e)
        dd = tf - ef; kk = np.where((dd[:-1] > 0) & (dd[1:] <= 0))[0]
        cross = float(xf[kk[0]]) if (len(kk) and ef[kk[0]] > 0) else None
        Lp = "abcdef"[k]
        sheet(f"{sh}{Lp}", fg, Lp,
              f"Gain in the accuracy of the ensemble mean in {reg} (reduction of the root-mean-square error against the 1 km estimates, %), "
              "brought by the computed transport and by the emissions taken up along the back-trajectories, by lead time (days), with 95% "
              "intervals (paired moving-block bootstrap over start days)."
              + (f" The dotted line marks the lead time at which the emissions overtake the transport ({cross:.1f} days)." if cross else ""),
              ["region", "contribution", "lead time (days)", "gain in accuracy (%)", "lower bound (%)", "upper bound (%)"], rows, scr, [fgain])
        panel(fg, Lp, f"gain in accuracy by lead time in {reg}: transport and emissions (lines with bands)", f"{sh}{Lp}")
    for k, reg in enumerate(["East Asia", "South Asia", "Europe"]):
        rows = [[reg, lab, h, sig(R[(reg, key, h)])] for key, lab in (("ctrl", "without emissions (control)"), ("e02", "with emissions")) for h in L]
        Lp = "ghi"[k]
        sheet(f"{sh}{Lp}", fg, Lp,
              f"Root-mean-square error of the ensemble mean against the 1 km estimates in {reg} (µg m-3) by lead time (days), for the control "
              "and for the model with the emission terms.",
              ["region", "model", "lead time (days)", "RMSE (µg m-3)"], rows, scr, [fscore])
        panel(fg, Lp, f"RMSE by lead time in {reg}, control and model with emissions (lines)", f"{sh}{Lp}")
    series = [("A", "base model"), ("ctrl", "control, no emissions"), ("e02", "with emissions")]
    rows = [[lab, h, sig(S[("All regions", key, h)])] for key, lab in series for h in L]
    sheet(f"{sh}j", fg, "j",
          "Ratio of the ensemble spread to the RMSE of the ensemble mean over all regions together (dimensionless), by lead time (days), for "
          "the base model, the control and the model with emissions.",
          ["model", "lead time (days)", "spread / RMSE"], rows, scr, [fscore])
    panel(fg, "j", "spread / RMSE by lead time, three models (lines)", f"{sh}j")


def sfig11(fg, sh):
    """figures/fig_km.py (rendered with NATURE=1). b, c: the RMSE written in the maps (L15-39); d: the bars, whose values are
    written in the figure script (L81-83), with the same ratio recomputed here from the saved forecasts with the function of
    regional/tiled_seams.py (L50-53: mean absolute jump across tile edges / elsewhere, over the land of Europe)."""
    scr = FIG / "fig_km.py"
    D, H = 40, 1
    RLINES, CLINES = [768, 1536, 2304, 3072, 3424], [1536, 3072, 4608, 5456]
    FINE = EXP / "fullimage/data/ghap1km.npy"; fst = EXP / "fullimage/data/static1km.npz"
    HDR = npy_header(used(FINE))[0]; OFF22 = 365 * 4 + 1 + 365
    truth_day = lambda d: np.asarray(np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + d) * 4192 * 6992 * 2, shape=(4192, 6992)), np.float32)
    land = npz(fst)["land"]

    def preds(run):                                       # every saved 1 km forecast of a tile-free run: {(start day, lead): map}
        out = {}
        for p in sorted(glob.glob(str(EXP / "fullimage/runs" / run / "eval_hres" / "part_*.npz"))):
            z = np.load(p)
            for k in z.files:
                if k.startswith("pred_") and k.endswith("_h1"):
                    used(p); out[int(k[6:].split("_")[0])] = z[k].astype(np.float32)
        return out
    free_all = preds("flow_shift25")
    truth = truth_day(D + H)
    tiled = np.load(used(EXP / f"regional/runs/earthformer/tiled_europe_d{D}_h{H}.npy")).astype(np.float32)
    free = free_all[D]
    R, C, Wd = 1536, 5860, 260
    sl = (slice(R - Wd, R + Wd), slice(C - Wd, C + Wd))
    ok = (land[sl] > 0) & (truth[sl] > 0)
    rmse = lambda a: float(np.sqrt(((a[sl] - truth[sl])[ok] ** 2).mean()))
    rows = [["b", "Earthformer, tiled", sig(rmse(tiled)), int(ok.sum())], ["c", "Foehn, tile-free", sig(rmse(free)), int(ok.sum())]]
    sheet(f"{sh}b-c", fg, "b, c",
          "Root-mean-square error (µg m-3) against the 1 km estimates over the land of the panel (around 56.6°N, 33.6°E), PM2.5 on 11 February "
          "2022 forecast 1 day ahead: the numbers written in the maps.",
          ["panel", "network", "RMSE over the land of the panel (µg m-3)", "land pixels"], rows, scr,
          [FINE, EXP / f"regional/runs/earthformer/tiled_europe_d{D}_h{H}.npy", EXP / "fullimage/runs/flow_shift25/eval_hres/part_*.npz", fst])

    def ratio(a):                                         # regional/tiled_seams.py:ratio (L50-53)
        a = np.where(land > 0, a, np.nan)
        dc, dr = np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0))
        return 0.5 * (np.nanmean(dc[:, [c - 1 for c in CLINES]]) / np.nanmean(dc) + np.nanmean(dr[[r - 1 for r in RLINES]]) / np.nanmean(dr))
    DAYS = [40, 100, 200, 300]
    reg_all = preds("reg")
    rec = {"SimVP, tiled": [ratio(np.load(used(EXP / f"regional/runs/simvp/tiled_europe_d{d}_h1.npy")).astype(np.float32)) for d in DAYS],
           "Earthformer, tiled": [ratio(np.load(used(EXP / f"regional/runs/earthformer/tiled_europe_d{d}_h1.npy")).astype(np.float32)) for d in DAYS],
           "Same net, regression": [ratio(reg_all[d]) for d in DAYS], "Foehn, tile-free": [ratio(free_all[d]) for d in DAYS],
           "Observed (GHAP)": [ratio(truth_day(d + 1)) for d in DAYS]}
    bars = [("SimVP, tiled", 4.82), ("Earthformer, tiled", 1.77), ("Same net, regression", 1.19), ("Foehn, tile-free", 1.01), ("Observed (GHAP)", 1.04)]
    rows = [[lab, v, sig(float(np.mean(rec[lab])))] + [sig(x) for x in rec[lab]] for lab, v in bars]
    sheet(f"{sh}d", fg, "d",
          "Mean absolute jump between neighbouring grid cells across tile edges divided by the mean jump elsewhere, over the land of Europe "
          "(dimensionless; 1, no seam), forecasts 1 day ahead: the value drawn and written in the panel (mean of four forecasts), and the "
          "same ratio recomputed from the saved 1 km maps for each of the four start days.",
          ["field", "ratio drawn (value written in the figure script)", "ratio recomputed, mean of the four forecasts",
           "recomputed, start day 10 February 2022", "recomputed, start day 11 April 2022", "recomputed, start day 20 July 2022",
           "recomputed, start day 28 October 2022"], rows, scr,
          [EXP / "regional/runs/<network>/tiled_europe_d<day>_h1.npy", EXP / "fullimage/runs/<run>/eval_hres/part_*.npz", FINE, fst],
          note="the bar values are literals in fig_km.py L81-83; the recomputation uses regional/tiled_seams.py:ratio")
    panel(fg, "a-c", "1 km maps around a tile edge: the estimates, Earthformer applied tile by tile, the tile-free network, with the RMSE written in b and c",
          f"maps not exported; printed numbers in {sh}b-c")
    panel(fg, "d", "seam ratio of five fields (bars with their values)", f"{sh}d")


# ================================================================================================ OUTPUT
TITLE = "Global probabilistic forecasting of fine particulate air pollution at the kilometre scale"
FORBIDDEN = ("/scr" + "atch/", "/p" + "fs/", "/proj" + "appl/")       # never in a delivered file (built from pieces: this file is delivered too)
FIGLIST = []                                              # (figure label, sheet prefix, figure file, remark), filled by main()
README_COLS = ["sheet", "figure", "panel", "figure file", "what the rows are (units in the text)", "columns", "rows"]


def check_clean(text, where):
    low = text.lower()
    for w in FORBIDDEN:
        if w in low:
            raise SystemExit(f"forbidden string {w!r} in {where}")


def csv_text(cols, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return buf.getvalue()


def readme_rows():
    return [[s["name"], s["figure"], s["panel"], s["stem"] + ".pdf", s["desc"], "; ".join(s["cols"]), len(s["rows"])] for s in SHEETS]


def write_csvs(only=None):
    out = HERE / "csv"; out.mkdir(exist_ok=True)
    for s in SHEETS:
        t = csv_text(s["cols"], s["rows"]); check_clean(t, s["name"])
        safe_write(out / f"{s['name']}.csv", t)
    if only is None:
        t = csv_text(README_COLS, readme_rows())
        check_clean(t, "README.csv"); safe_write(out / "README.csv", t)
        keep = {f"{s['name']}.csv" for s in SHEETS} | {"README.csv"}
        for p in out.glob("*.csv"):                       # a sheet that no longer exists (a figure renumbered) leaves no stale file
            if p.name not in keep:
                p.unlink()


def write_xlsx():
    import openpyxl
    from openpyxl.styles import Alignment, Font
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "README"
    ws.append([f"Source Data. {TITLE}"]); ws["A1"].font = Font(bold=True, size=12)
    ws.append(["Kheder, Boy and Liu. One sheet per figure panel with graph data. In every data sheet: row 1, what the sheet holds, with units; "
               "row 2, column headers; then one row per plotted point, bar or printed number. Empty cell: no value. Maps and images are not "
               "exported pixel by pixel; for maps, the numbers printed in the panels and the stations or monitors drawn as circles are given. "
               "The monitoring stations are evaluation data only."])
    ws.append([])
    ws.append(README_COLS)
    for c in ws[4]:
        c.font = Font(bold=True)
    for r in readme_rows():
        ws.append(r)
    for col, wd in zip("ABCDEFG", (22, 22, 14, 26, 110, 90, 8)):
        ws.column_dimensions[col].width = wd
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A5"
    for s in SHEETS:
        w = wb.create_sheet(s["name"])
        w.append([s["desc"]]); w["A1"].font = Font(italic=True)
        w.append(s["cols"])
        for c in w[2]:
            c.font = Font(bold=True)
        for r in s["rows"]:
            w.append(r)
        for i, cname in enumerate(s["cols"]):
            wd = max([len(str(cname))] + [len(str(r[i])) for r in s["rows"][:200] if r[i] is not None])
            w.column_dimensions[openpyxl.utils.get_column_letter(i + 1)].width = min(max(10, wd + 2), 60)
        w.freeze_panes = "A3"
    wb.properties.title = "Source Data"; wb.properties.creator = "Kheder, Boy and Liu"
    buf = io.BytesIO(); wb.save(buf)
    safe_write(HERE / "Source_Data.xlsx", buf.getvalue())
    # read back with a fresh parser: same sheets, same values, numbers stored as numbers
    wb2 = openpyxl.load_workbook(HERE / "Source_Data.xlsx", read_only=True)
    assert wb2.sheetnames == ["README"] + [s["name"] for s in SHEETS], "sheet list differs after read-back"
    for s in SHEETS:
        got = [list(r) for r in wb2[s["name"]].iter_rows(values_only=True)]
        assert got[0][0] == s["desc"] and list(got[1][:len(s["cols"])]) == s["cols"], s["name"]
        assert len(got) == len(s["rows"]) + 2, (s["name"], len(got), len(s["rows"]))
        for a, b in zip(got[2:], s["rows"]):
            a = list(a[:len(b)]) + [None] * (len(b) - len(a))
            for x, y in zip(a, b):
                if isinstance(y, (int, float)) and not isinstance(y, bool):
                    assert isinstance(x, (int, float)) and x == y, (s["name"], x, y)
                else:
                    assert x == y, (s["name"], x, y)
    wb2.close()
    print(f"Source_Data.xlsx: {len(SHEETS)} data sheets, read back identical", flush=True)


def write_manifests():
    L = ["# Source Data: what is exported for each panel", "",
         f"Paper: *{TITLE}*. Built {dt.datetime.now():%Y-%m-%d %H:%M}.", "",
         "`Source_Data.xlsx` holds one sheet per figure panel with graph data (lines, bars, points, histograms, forest plots, violins) "
         "and `csv/<sheet>.csv` the same tables as plain text (header row, then the values; `csv/README.csv` lists the sheets). "
         "Maps and images are not exported pixel by pixel. For a map whose message is a number printed in the panel, the printed "
         "numbers are exported; where stations or monitors are drawn as circles on a map, their positions and measured values are "
         "exported too. Every value is read from the data file that the figure script reads, or recomputed with the lines of that "
         "script (`build_source_data.py`). Scores are given with four significant digits. The monitoring stations are evaluation data only.", "",
         "## Figures of the manuscript when the Source Data were built", "",
         "The figure numbers are read from the manuscript files at every run of `build_source_data.py` (order of the figures in the "
         "main text and in the Supplementary Information); the sheet names follow them.", "",
         "| Figure | Figure file | Sheets | Remark |", "|---|---|---|---|"]
    for fg, sh, stem, remark in FIGLIST:
        names = [s["name"] for s in SHEETS if s["stem"] == stem]
        L.append(f"| {fg} | `{stem}.pdf` | {', '.join(names) if names else 'none'} | {remark} |")
    if NOTES:
        L += ["", "Remarks:", ""] + [f"- {n}" for n in NOTES]
    L += ["", "## Panel by panel", "", "| Figure | Panel | Content | Exported as |", "|---|---|---|---|"]
    for p in PANELS:
        L.append(f"| {p['figure']} | {p['panel']} | {p['what']} | {p['export']} |")
    L += ["", "## Sheets", "", "| Sheet | Figure | Panel | Rows | Columns |", "|---|---|---|---|---|"]
    for s in SHEETS:
        L.append(f"| {s['name']} | {s['figure']} | {s['panel']} | {len(s['rows'])} | {'; '.join(s['cols'])} |")
    t = "\n".join(L) + "\n"; check_clean(t, "MANIFEST.md"); safe_write(HERE / "MANIFEST.md", t)
    L = ["# Source Data: internal manifest (working file, NOT for the journal: it holds internal paths and names)", "",
         f"Built {dt.datetime.now():%Y-%m-%d %H:%M:%S} by `{HERE / 'build_source_data.py'}` in {time.time() - T0:.0f} s.", "",
         "Roots derived from the place of the build script:", "",
         f"- workspace `{W}`", f"- figure scripts `{FIG}`", f"- data of the figure scripts `{DATA}`", f"- evaluation folders `{RUNS}`",
         f"- 1 km stores and operational forecasts `{WORLD}`", "",
         "## State of the manuscript at build time", "",
         "| Figure | Figure file | Modified | Remark |", "|---|---|---|---|"]
    for fg, sh, stem, remark in FIGLIST:
        p = W / "work" / "figs" / f"{stem}.pdf"
        mt = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if p.exists() else "not in work/figs"
        L.append(f"| {fg} | `{p}` | {mt} | {remark} |")
    L += [""] + [f"- {n}" for n in NOTES]
    L += ["", "## Sheet -> figure script and data files", "", "| Sheet | Figure script (source of the plotted numbers) | Data files read | Note |", "|---|---|---|---|"]
    for s in SHEETS:
        L.append(f"| {s['name']} | `{s['script']}` | {'<br>'.join('`' + f + '`' for f in s['files'])} | {s['note']} |")
    L += ["", "## Data files read, with their modification time at the moment of reading", "", "| File | Modified |", "|---|---|"]
    for f, m in sorted(USED.items()):
        L.append(f"| `{f}` | {m} |")
    safe_write(HERE / "MANIFEST_internal.md", "\n".join(L) + "\n")


def rt_proof(fg, sh):
    """figures_new/fig_rt_skill/figS_rt_proof.py: one panel, RMSE by lead day of the four systems at the monitors (numbers.json)."""
    d = NEW / "fig_rt_skill"; scr = d / "figS_rt_proof.py"; fn = d / "numbers.json"
    num = jload(fn); r = num["b_lead"]["rmse"]
    NAME = {"km": "Foehn in real time (1 km)", "rt04": "Foehn in real time (40 km)", "cams": "CAMS forecast", "clim": "Climatology (1 km)"}
    rows = [[NAME[k], L + 1, sig(r[k][L])] for k in ("km", "rt04", "cams", "clim") for L in range(5)]
    sheet(sh, fg, "",
          "Root-mean-square error (µg m-3) of the daily mean PM2.5 at 11,121 monitors by lead day (lead day 1 is the day of issue), on "
          "identical station-days (1,819,647; 48 issue days of autumn 2026), for the real-time 1 km forecast (mean of eight members), the "
          "0.4° forecast from which it is drawn, the CAMS forecast of the same cycle (daily mean of its eight 3-hourly steps) and the "
          "climatology of the 1 km estimates.",
          ["system", "lead day", "RMSE (µg m-3)"], rows, scr, [fn])
    panel(fg, "", "RMSE by lead day of the four systems (lines)", sh)


# ================================================================================================ FIGURES OF THE MANUSCRIPT
# figure file (stem of the PDF included by the manuscript) -> function that exports its panels
BUILDERS = {
    "fig_hidden": hidden_ae, "fig_hidden_v4": hidden_v4, "figS_hidden_warned": warned_v1, "figS_hidden_warned_v2": warned_v2,
    "fig_overview": fig3, "figED_fair": fig4, "fig_gc5_aurora_setup": fig5, "fig_gc2_realism": fig6, "fig_transport": fig7,
    "fig_smoke": fig8, "fig_rt_day": fig9, "fig_rt_skill": rt_skill, "fig_rt_skill_main": rt_skill_main, "figS_rt_ensemble": rt_ensemble, "figS_rt_proof": rt_proof,
    "fig_spectra_cost": sfig1, "fig_leadtime": sfig2, "fig_extremes": sfig3, "figS_exposure": sfig4, "figED_regions": sfig6,
    "figED_aurora_regions": sfig7, "figS_reliability": sfig8, "figS_sedi_definitions": sfig9, "figS_emission_lead": sfig10, "fig_km": sfig11,
}
SCHEMATIC = "fig_gc1_schematic"                           # every version of the schematic (Fig. 1): no graph data
PENDING_MAIN = []                                         # (the verification figure is no longer announced as a main figure)
_PENDING_MAIN_UNTIL_20261120 = [("fig_rt_skill", ("fig_rt_skill", "fig_rt_skill_main"))]   # main figure announced: exported under a provisional number
                                                          # while none of its versions is in the manuscript files


def manuscript_figures():
    """[(figure label, sheet prefix, figure file, remark)] in the order of the manuscript, read from the .tex files"""
    import re
    pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+?)(?:\.pdf)?\}")

    def stems(path):
        out = []
        if path.exists():
            used(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                line = re.sub(r"(?<!\\)%.*", "", line)    # comments
                out += pat.findall(line)
        return out
    work = W / "work"
    main_ = stems(work / "sections" / "main_text.tex") + stems(work / "sections" / "methods.tex")
    supp = stems(work / "supplementary.tex")
    figs = [(f"Fig. {i + 1}", f"Fig{i + 1}", s, "") for i, s in enumerate(main_)]
    for s, versions in PENDING_MAIN:
        if not any(v in main_ + supp for v in versions):
            n = len(figs) + 1
            figs.append((f"Fig. {n}", f"Fig{n}", s, "NOT YET in the manuscript files at build time: provisional number (next main figure)"))
            NOTES.append(f"`{s}.pdf` is not included by the manuscript files at build time; it is exported under the provisional number Fig. {n} "
                         "(the next main figure). Run the build again once the figure is in the manuscript.")
    figs += [(f"Supplementary Fig. {i + 1}", f"SFig{i + 1}", s, "") for i, s in enumerate(supp)]
    return figs


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    figs = manuscript_figures()
    for fg, sh, stem, remark in figs:
        if want and stem not in want:
            continue
        CURRENT["stem"] = stem
        fn = fig1 if stem.startswith(SCHEMATIC) else BUILDERS.get(stem)
        print(f"{fg} = {stem} ...", flush=True)
        if fn is None:
            panel(fg, "all", "figure file unknown to build_source_data.py", "NOT EXPORTED: no export function for this figure file")
            NOTES.append(f"`{stem}.pdf` ({fg}) has no export function in build_source_data.py: NOTHING is exported for it.")
            print(f"  WARNING: no export function for {stem}", flush=True)
            continue
        fn(fg, sh)
    FIGLIST[:] = figs
    write_csvs(only=want or None)
    if want:
        print("development run: CSV files only"); return
    write_xlsx()
    write_manifests()
    import zipfile
    with zipfile.ZipFile(HERE / "Source_Data.xlsx") as z:  # the workbook is a zip: scan its members too
        for n in z.namelist():
            check_clean(z.read(n).decode("utf-8", "replace"), f"Source_Data.xlsx:{n}")
    print(f"done in {time.time() - T0:.0f} s: {len(SHEETS)} sheets, {len(PANELS)} panel rows", flush=True)
    for n in NOTES:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
