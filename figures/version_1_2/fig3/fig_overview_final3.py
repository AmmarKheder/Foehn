#!/usr/bin/env python3
"""Main figure: one forecast of the whole planet at 1 km (the final model; after fig1_overview.py).
a, Foehn's forecast (ensemble mean of four members) issued at the end of 8 February 2022 for 9 February, with the four
   regions of b;
b, the same forecast in four regions on four continents, next to the CAMS forecast issued at the same time (00 UTC run of
   9 February, 0.4 deg) and GHAP; dots: the measurements of that day at the stations of the six official networks.
   Numbers: RMSE at those stations ON THE DAY SHOWN (1 day ahead), with the pairing of the yearly values (stations where the
   measurement and the three fields exist);
c-d, RMSE at the stations by lead time (1-4 days), 2022, paper mask, identical station-days for the three systems:
   c the six networks, d China (CNEMC), on logarithmic y axes of the same top/bottom ratio;
e, RMSE reduction against the CAMS forecast 1 day ahead by network (four members as run, paper mask), 95 % interval of
   compare_fair's bootstrap (blocks of 7 start days, 10,000 resamples, default_rng(0)).
    FOEHN_DESIGN=ar python -I -B fig_overview_final2.py         (FINAL_FAST=1: no map rasters, for the layout only)"""
# FIX (natcomms/rebuild/figs_show/final2, 2026-11-21): a marked copy of final/fig_overview_final.py (its FINAL-<n> and
# SHOW-<n> marks kept); changed or added lines carry 'FIX-<n>:'. Decisions after the figure checks:
#   FIX-0: writes only under figs_show/final2 (or FINAL_OUT for a layout preview); outputs fig_overview_final2.*, fig3_numbers.json;
#   FIX-1: (F3-1) c and d on logarithmic y axes with the same top/bottom ratio (1.95), plain ticks, no minor ticks;
#   FIX-2: e = one bar panel: RMSE reduction against the CAMS forecast 1 day ahead by network, compare_fair's bootstrap
#          (whisker = 95 % interval; interval containing zero: hatched, lighter bar); pooled outside-China values -> json;
#   FIX-3: (F3-5) opaque boxes and scale-bar labels in b;
#   FIX-4: (F3-2) a white block (no edge) behind a's colour bar, above the graticule only (land and coasts stay above it);
#   FIX-5: page height <= 205 mm: smaller white gaps (rows of b, a->b and b->band rhythm, legend), lower band axes (>= 24 mm).
# FINAL (natcomms/rebuild/figs_show/final, 2026-11-21): a marked copy of figs_show/fig_overview_show.py, variant A (itself a
# marked copy of experiments/figures/fig_overview.py; its SHOW-<n> marks are kept); changed or added lines carry 'FINAL-<n>':
#   FINAL-0  own write guard: writes only under figs_show/final (or FINAL_OUT for a layout preview), MPLCONFIGDIR, /tmp;
#   FINAL-1  variant A fixed (9 February 2022, start 0-based 38, lead 1, the re-run 1 km map of figs_show/run/...);
#   FINAL-2  boxes of b: RMSE on the day shown, the cache's pairing restricted to key (38, 1); yearly values -> json only;
#   FINAL-3  band c-e (three line panels) under b, on the grid of b; one legend above the band;
#   FINAL-4  page exactly 180 mm wide (bbox centred on the content), PDF + PNG at 300 dpi; numbers -> final/fig3_numbers.json.
import glob
import json  # FINAL-4
import os  # SHOW-1
import sys  # SHOW-1
from pathlib import Path  # SHOW-1

FINAL = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/figs_show/final2"))  # FIX-0: final -> final2
OUTDIR = Path(os.environ.get("FINAL_OUT", str(FINAL)))  # FINAL-0 (SHOW-3: the output folder)
FAST = os.environ.get("FINAL_FAST") == "1"  # FINAL-0 (layout preview: no map rasters)
_ALLOW = tuple(p.rstrip("/") + "/" for p in (str(OUTDIR), "/tmp", "/dev", os.environ.get("MPLCONFIGDIR", "/tmp")))  # FINAL-0
_WB = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND  # FINAL-0


def _ok(p):  # FINAL-0
    return (os.path.abspath(os.fsdecode(p)) + "/").startswith(_ALLOW)


def _hook(ev, args):  # FINAL-0 (as render_show.py: any write outside the allowed folders raises, nothing is written)
    if ev == "open":
        p, mode, flags = args
        if p is None or isinstance(p, int):
            return
        w = bool(flags & _WB) if isinstance(flags, int) else bool(mode and any(c in mode for c in "wax+"))
        if w and not _ok(p):
            raise PermissionError(f"fig_overview_final write guard: write {p} refused")
    elif ev in ("os.rename", "os.replace", "shutil.copyfile"):
        if not _ok(args[1]):
            raise PermissionError(f"fig_overview_final write guard: {ev} to {args[1]} refused")
    elif ev in ("os.remove", "os.rmdir", "os.truncate", "shutil.rmtree", "os.chmod", "os.utime"):
        if isinstance(args[0], (str, bytes, os.PathLike)) and not _ok(args[0]):
            raise PermissionError(f"fig_overview_final write guard: {ev} {args[0]} refused")
    elif ev == "os.mkdir":
        if not _ok(args[0]) and not os.path.exists(args[0]):
            raise PermissionError(f"fig_overview_final write guard: mkdir {args[0]} refused")


sys.addaudithook(_hook)  # FINAL-0
sys.path.insert(0, (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))  # SHOW-1 (this copy lives outside FIG)

import cartopy.crs as ccrs
import numpy as np
import zarr
from matplotlib.lines import Line2D  # FINAL-3
from matplotlib.patches import Rectangle  # FIX-4:
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # FIX-1:
from matplotlib.transforms import Bbox  # FINAL-4

import final_model as FM
from fig1_teaser_v6 import tiles
from gc_style import AURORA, CAMS, FOEHN, geo_ticks, map_axes, world_coords  # FINAL-3 (colours of the systems)
from nature_style import DOUBLE, HERE, PM_BOUNDS, letter, plt, pm_cmap

assert FM.LABEL == "crps_e01_ar", FM.LABEL  # FINAL-1 (FOEHN_DESIGN=ar)
D0, H = 38, 1  # FINAL-1 start day 8 February (0-based 38), lead 1: target 9 February 2022 (SHOW variant A)
SHOWRUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/figs_show/run/eval_show_d38_crps_e01_ar"))  # SHOW-2
DATE = "9 February 2022"  # FINAL-1
DT = D0 + H
G = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km")
CENTRES = {"Delhi and the Ganges plain": (28.7, 77.6, 700), "Po Valley and the Alps": (45.35, 10.0, 600),
           "Los Angeles basin": (34.05, -117.9, 300), "Seoul and the Han basin": (37.56, 127.0, 300)}
ASPECT = 1.5                                                    # width / height of every zoom (km)
MM = 1 / 25.4  # FINAL-3
NUM = {"figure": "fig_overview_final3", "design": FM.LABEL, "day": {"target": "2022-02-09", "start_0based": D0, "lead_days": H}}  # FIX-0: name
ROWGAP_MM = 1.4  # FIX-5: white gap between the rows of b (was 0.012 x 183 = 2.20 mm)


def day_pairing(st, cz, off):  # FINAL-2 (whole function)
    """the pairing of the cached yearly values (the cache branch of main, identical lines) restricted to the key (D0, H):
    Foehn = mean of the four members as run, CAMS = run of 00 UTC of the target day read bilinearly at the station, GHAP at
    the station pixel; kept: official networks where the measurement and the three fields are finite"""
    rows = []
    for p in sorted(glob.glob(str(FM.EVAL_DIR / "part_*.npz"))):
        if ".tmp" in p:
            continue
        z = np.load(p); w = np.where((z["keys"][:, 0] == D0) & (z["keys"][:, 1] == H))[0]
        rows += [(p, int(k), z) for k in w]
    assert len(rows) == 1, [r[:2] for r in rows]
    p, n, z = rows[0]
    t = int(z["keys"][n, 2]); assert t == DT, (t, DT)
    la_all, lo_all = st["lat"], st["lon"]
    fi = (90.0 - la_all) / 0.4; fj = (lo_all % 360.0) / 0.4
    i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900; wi, wj = fi - i0, fj - np.floor(fj)
    c = np.asarray(cz[t, 0, 0], np.float32)
    cs = (1 - wi) * ((1 - wj) * c[i0, j0] + wj * c[i0, j1]) + wi * ((1 - wj) * c[i0 + 1, j0] + wj * c[i0 + 1, j1])
    X = np.stack([z["st_ens"][n].astype(np.float32).mean(-1), cs, z["st_ghap"][n].astype(np.float32)]); o = st["daily"][:, t]
    m = np.isfinite(o) & np.all(np.isfinite(X), 0) & off
    return X, o, m, f"{p} row {n} keys {z['keys'][n].tolist()}"


def main():
    ours = np.load(SHOWRUN / f"map1km_d{D0}_h{H}.npy", mmap_mode="r")  # FINAL-1 (SHOW-2, variant A)
    ghap = tiles("ghap1km_tiles.npy", 1826 + DT); land = tiles("static1km_tiles.npy", 1) > 0
    cz = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
    cams = np.asarray(cz[DT, 0, 0], np.float32)                 # run of 00 UTC 9 February, its lead day 0: equal issuance
    st = np.load(f"{G}/stations_2022.npz"); off = st["source"] != 2  # the six official networks (OpenAQ excluded, as everywhere)
    obs = st["daily"][:, DT]; okst = np.isfinite(obs) & off
    slat, slon, sobs = st["lat"][okst], st["lon"][okst], obs[okst]
    cmap, norm = pm_cmap()
    # RMSE over 2022, lead 1 day, at every station: Foehn (ensemble mean), CAMS (equal issuance), GHAP; identical station-days
    la_all, lo_all = st["lat"], st["lon"]
    cache = HERE / "data" / f"overview_cache_{FM.LABEL}.npz"
    assert cache.exists(), cache  # FINAL-2 (the stored cache is read, never rebuilt here)
    zc_ = np.load(cache); SSE, NSD = zc_["SSE"], zc_["NSD"]
    print("station RMSE (lead 1) on", int(NSD.sum()), "station-days")
    XD, OD, MD, DSRC = day_pairing(st, cz, off)  # FINAL-2
    NUM["boxes_day"] = {"source": f"{DSRC}; CAMS data/world/cams_forecast_daily/2022.zarr[{DT}, 0, 0] read bilinearly; "  # FINAL-4
                                  f"stations_2022.npz daily[:, {DT}], source != 2; pairing = the code of the cached yearly values (overview_cache)",
                        "rows": ["Foehn (four-member mean)", "CAMS forecast (00 UTC run of the target day)", "GHAP (target day)"]}
    NUM["boxes_yearly_for_legend"] = {"source": str(cache), "station_days": int(NSD.sum())}  # FINAL-4
    REG = {}
    for nm, (la, lo, wkm) in CENTRES.items():
        dla = wkm / ASPECT / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
        REG[nm] = (la + dla, la - dla, lo - dlo, lo + dlo, wkm)
    FH = 183 / 25.4
    fig = plt.figure(figsize=(DOUBLE, FH))
    # a: the globe at 1 km (ensemble mean) and the regions of b
    ax = map_axes(fig, [0.035, 91 / 183, 0.93, 88 / 183])                 # room for the latitudes on both sides
    m = np.asarray(ours, np.float32); m[~land] = np.nan
    if not FAST:  # FINAL-0
        ax.imshow(m, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
                  regrid_shape=4400, zorder=1, interpolation="nearest")
    del m
    for k, (nm, (n, s, w, e, _)) in enumerate(REG.items()):
        t_ = np.linspace(0, 1, 40)
        bx = np.concatenate([w + (e - w) * t_, np.full(40, e), e - (e - w) * t_, np.full(40, w)])
        by = np.concatenate([np.full(40, n), n - (n - s) * t_, np.full(40, s), s + (n - s) * t_])
        ax.plot(bx, by, transform=ccrs.PlateCarree(), color="#111111", lw=0.7, zorder=6)
        ax.text(e + 1.2, n + 0.5, str(k + 1), transform=ccrs.PlateCarree(), fontsize=6.5, fontweight="bold", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#111111", lw=0.5))
    world_coords(ax, size=5.5)
    letter(ax, "a", x=-0.005, y=0.985)
    cax = fig.add_axes([0.385, 101 / 183, 0.285, 0.0075])   # fully inside the Southern Ocean of a
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("Daily mean PM$_{2.5}$ on " + DATE + " (µg m$^{-3}$)", fontsize=5.8, labelpad=2)  # SHOW-2
    # b: four regions x (CAMS 40 km, Foehn 1 km); the GHAP zooms are in the ED regions figure
    rows = (("CAMS forecast (40 km)", "cams"), ("Foehn (1 km)", "ours"), ("GHAP (1 km)", "ghap"))
    cw = 0.205; ch = cw * DOUBLE / FH / ASPECT; x0 = 0.056; y0 = 82 / 183 - ch       # 7 mm gaps: each region has its latitudes
    AXB, ROWL = [], []  # FINAL-3 (the axes of b and its row labels, for the grid of the band)
    for j, (nm, (n, s, w, e, wkm)) in enumerate(REG.items()):
        r0, r1 = int((84 - n) * 100), int((84 - s) * 100); c0, c1 = int((w + 180) * 100), int((e + 180) * 100)
        sel = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e)
        inbox = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & (NSD > 0) & off
        inday = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & MD  # FINAL-2 (stations scored on the day)
        for i, (rt, key) in enumerate(rows):
            a = fig.add_axes([x0 + j * 0.2435, y0 - i * (ch + ROWGAP_MM / 183), cw, ch]); a.set_facecolor("white"); AXB.append(a)  # FIX-5: row gap
            if FAST:  # FINAL-0
                pass
            elif key == "cams":
                ci0, ci1 = int((90 - n) / 0.4), int((90 - s) / 0.4) + 1; cj0, cj1 = int((w % 360) / 0.4), int((e % 360) / 0.4) + 1
                lw0 = cj0 * 0.4 - 0.2; lw0 = lw0 - 360.0 if lw0 > 180 else lw0
                cc = cams[ci0:ci1, cj0:cj1].copy()
                la_c = 90 - 0.4 * np.arange(ci0, ci1); lo_c = (0.4 * np.arange(cj0, cj1) + 180) % 360 - 180
                rr = np.clip(((84 - la_c) * 100).astype(int), 0, land.shape[0] - 1); cc_ = np.clip(((lo_c + 180) * 100).astype(int), 0, land.shape[1] - 1)
                lm = land[rr[:, None], cc_[None, :]]                  # the 40 km cells shown over land, like the 1 km rows
                cc[~lm] = np.nan
                a.imshow(cc, extent=(lw0, lw0 + (cj1 - cj0) * 0.4, 90 - ci1 * 0.4 + 0.2, 90 - ci0 * 0.4 + 0.2),
                         cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            else:
                src = ours if key == "ours" else ghap
                v = np.asarray(src[r0:r1, c0:c1], np.float32); v[~land[r0:r1, c0:c1]] = np.nan
                if key == "ghap":
                    v[v <= 0] = np.nan
                a.imshow(v, extent=(w, e, s, n), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            if sel.any():
                a.scatter(slon[sel], slat[sel], c=sobs[sel], cmap=cmap, norm=norm, s=6, linewidths=0.35, edgecolors="#111111", zorder=5)
            k3 = {"ours": 0, "cams": 1, "ghap": 2}[key]  # FINAL-2
            if inbox.sum() >= 3:  # FINAL-2 (yearly value: json, for the legend)
                ry = np.sqrt(SSE[inbox, k3].sum() / NSD[inbox].sum())
                NUM["boxes_yearly_for_legend"].setdefault(nm, {})[key] = round(float(ry), 1)
                NUM["boxes_yearly_for_legend"][nm]["stations"] = int(inbox.sum())
            if inday.sum() >= 3:  # FINAL-2 (day value: printed in the box)
                rm = np.sqrt(((XD[k3, inday] - OD[inday]) ** 2).sum() / inday.sum())
                a.text(0.03, 0.04, f"RMSE {rm:.1f} µg m$^{{-3}}$" + (f" ({int(inday.sum())} stations)" if i == 0 else ""),
                       transform=a.transAxes, fontsize=5.2, zorder=8, fontweight="bold" if key == "ours" else "normal",
                       bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=1.0))  # FIX-3: opaque (was 0.85)
                b_ = NUM["boxes_day"].setdefault(nm, {"stations_scored": int(inday.sum()), "dots_drawn": int(sel.sum())})  # FINAL-4
                b_[key] = {"printed": f"{rm:.1f}", "value": float(rm)}
                print(f"{nm:28s} {key:5s} RMSE {DATE} {rm:.2f} ({int(inday.sum())} stations; {int(sel.sum())} dots)")
            a.set_xlim(w, e); a.set_ylim(s, n); a.set_aspect(1 / np.cos(np.deg2rad(0.5 * (n + s))), adjustable="box")
            if i == 2:                                                 # scale bar
                L = {700: 200, 600: 100, 400: 100, 300: 50}[wkm]; dlo = L / (111.2 * np.cos(np.deg2rad(0.5 * (n + s))))
                xb, yb = e - 0.06 * (e - w) - dlo, s + 0.08 * (n - s)
                a.plot([xb, xb + dlo], [yb, yb], color="#111111", lw=1.0, zorder=9, solid_capstyle="butt")
                a.text(xb + dlo / 2, yb + 0.03 * (n - s), f"{L} km", ha="center", va="bottom", fontsize=5.2, zorder=9,
                       bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=1.0))  # FIX-3: opaque (was 0.7)
            geo_ticks(a, left=True, bottom=i == 2)                         # latitudes on every panel, longitudes under the last row
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#333333")
            if i == 0:
                a.set_title(f"{j + 1}  {nm}", fontsize=6.5, pad=2, fontweight="bold")
            if j == 0:
                ROWL.append(a.text(-0.17, 0.5, rt, transform=a.transAxes, rotation=90, ha="right", va="center", fontsize=6,  # FINAL-3
                                   fontweight="bold" if key == "ours" else "normal"))
            if i == 0 and j == 0:
                letter(a, "b", x=-0.12, y=1.04)
                a_b = a  # SHOW-4 (the axes that carries letter b)
    paper_mask_day(REG)  # FINAL-4 (the same day under the paper mask, for the report; not printed)
    band(fig, FH, a_b, AXB, ROWL, [ax, cax])  # FINAL-3
    cbar_block(fig, ax, cax)  # FIX-4:
    save_final(fig)  # FINAL-4


def cbar_block(fig, ax, cax, margin_mm=0.8):  # FIX-4: (whole function)
    """a white rectangle without edge in a's axes behind the whole colour-bar block (title, bar, extension, tick labels, +
    margin_mm), in cax's axes coordinates (it follows cax when the layout moves it), zorder 0.8: above the graticule
    (world_coords, zorder 0.6) and the land shading (0), below the 1 km field (1) and the coastlines (4)"""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = cax.get_tightbbox(r); m = margin_mm * MM * fig.dpi
    (x0, y0), (x1, y1) = cax.transAxes.inverted().transform([[tb.x0 - m, tb.y0 - m], [tb.x1 + m, tb.y1 + m]])
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=cax.transAxes, facecolor="white", edgecolor="none", lw=0, zorder=0.8)
    ax.add_artist(rect)
    NUM["colour_bar_block_mm"] = {"width": round((tb.width + 2 * m) / fig.dpi / MM, 2), "height": round((tb.height + 2 * m) / fig.dpi / MM, 2),
                                  "margin": margin_mm}


def paper_mask_day(REG):  # FINAL-4 (whole function)
    """the boxes' day values under the paper mask (circularity station_day.npz: official, ten systems finite, one station per
    pixel), and the tally behind the choice of the day (stored by the independent check print/show/verify_BC.json)"""
    sd = np.load((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/analyses/circularity/data_ar/station_day.npz"))
    i = int(np.where(sd["days"] == D0)[0][0]); P, O = sd["paper"][i, H - 1], sd["obs"][i, H - 1]
    R = {"source": f"analyses/circularity/data_ar/station_day.npz, start {D0}, lead {H}, mask 'paper'"}
    for nm, (n, s, w, e, _) in REG.items():
        b = P & (sd["lat"] <= n) & (sd["lat"] >= s) & (sd["lon"] >= w) & (sd["lon"] <= e)
        R[nm] = {k: round(float(np.sqrt(((sd[v][i, H - 1][b] - O[b]) ** 2).mean())), 2) for k, v in
                 (("Foehn", "foehn"), ("CAMS", "cams1"), ("Aurora", "aur1"), ("GHAP_target_day", "ghapt"))}
        R[nm]["stations"] = int(b.sum())
    NUM["boxes_day_paper_mask"] = R
    vb = json.load(open((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/print/show/verify_BC.json")))
    NUM["choice_of_day"] = {"source": "natcomms/rebuild/print/show/verify_BC.json (independent check, 2026-11-21)",
                            "paper_mask": {k: vb["B_paper_mask"][k] for k in ("days_all_four_valid", "below_cams_all_four", "below_both_all_four",
                                                                               "rank_min_ratio", "rank_min_margin")},
                            "figure_pairing": {k: vb["B_figure_pairing"][k] for k in ("days_all_four_valid", "below_cams_all_four",
                                                                                       "below_both_all_four", "rank_min_ratio", "rank_min_margin")},
                            "per_basin_days_below_CAMS_paper_mask": {k: vb["B_paper_mask"][k]["below_cams"] for k in ("Delhi", "Po", "LA", "Seoul")},
                            "per_basin_days_below_CAMS_figure_pairing": {k: vb["B_figure_pairing"][k]["below_cams"] for k in ("Delhi", "Po", "LA", "Seoul")},
                            "per_basin_days_above_CAMS_paper_mask": {k: vb["B_paper_mask"][k]["above_cams"] for k in ("Delhi", "Po", "LA", "Seoul")},  # FIX-6:
                            "per_basin_days_above_CAMS_figure_pairing": {k: vb["B_figure_pairing"][k]["above_cams"] for k in ("Delhi", "Po", "LA", "Seoul")}}  # FIX-6:


# ---- FINAL-3: the band c-e ------------------------------------------------------------------------------------------------
NETS_EXPECT = {"Stations of the six networks": ([13.00, 13.91, 14.21, 14.77], [15.38, 15.93, 16.03, 16.49], [15.86, 16.46, 16.87, 17.42]),
               "China": ([15.46, 17.47, 18.47, 19.40], [22.20, 23.09, 23.58, 24.28], [23.54, 24.86, 25.99, 26.92])}
SYSN = {"Foehn (1 km)": "crps_e01_ar (1 km)", "CAMS forecast (40 km)": "CAMS (run d+1, equal issuance)",
        "Aurora (40 km)": "Aurora (00 UTC d+1, equal issuance)"}
STY = {"CAMS forecast (40 km)": dict(color=CAMS, marker="s", ms=2.5, lw=0.8, zorder=4),        # references first, thin
       "Aurora (40 km)": dict(color=AURORA, marker="D", ms=2.3, lw=0.8, zorder=4),
       "Foehn (1 km)": dict(color=FOEHN, marker="o", ms=3.3, lw=1.5, zorder=6)}                 # Foehn last and thicker
BAND = dict(gap_mm=2.0, legend_mm=3.0, legend_gap_mm=0.9, title_mm=3.0, axes_mm=25.0, below_mm=7.6)   # FIX-5: was 3.6 / 1.6 / 30.0
RATIO = 1.95                                                     # FIX-1: top/bottom ratio of the log y axes of c and d (check F3-1)
YT = {"Stations of the six networks": [12, 15, 20], "China": [15, 20, 25]}   # FIX-1: plain ticks, no minor ticks
NETE = {"China (CNEMC)": "China", "USA (EPA)": "USA", "Japan (NIES)": "Japan", "Europe (EEA)": "Europe", "India (CPCB)": "India",
        "Korea (AirKorea)": "Korea"}                             # FIX-2: the six networks of compare_fair, short names
EXPECT_E = {"China": "30", "USA": "23", "Japan": "12", "Europe": "6", "India": "2", "Korea": "−6"}   # FIX-2: expected values (check only)
BAR, BAR_LIGHT, BAR_TXT, WHISK = "#2F6FB7", "#C9DAEF", "#1F4E86", "#333333"   # FIX-2: soft tint of Fig. 2f; lighter tint if not significant
CLASSU = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a14/classU/results_ar/st_cf_classU_179.json"))  # FIX-2: cross-check


def band_values():
    """RMSE by lead of the three systems on identical station-days (compare_fair, paper mask), six networks / China / the five
    other networks (sums of squared errors and counts of the five), cross-checked against circularity station_day.npz"""
    fz = np.load(HERE / "data" / f"compare_fair_{FM.LABEL}.npz"); SYS = [str(s) for s in fz["systems"]]; S = fz["S"]
    nets = [str(s) for s in fz["nets"]]
    G_ = {"Stations of the six networks": [nets.index("all")], "China": [nets.index("China (CNEMC)")],
          "Outside China": [k for k, s in enumerate(nets) if s not in ("all", "China (CNEMC)")]}
    assert len(G_["Outside China"]) == 5 and np.allclose(S[:6].sum(0), S[nets.index("all")]), nets
    R, NSD_ = {}, {}
    for pn, gg in G_.items():
        R[pn] = {nm: [float(np.sqrt(S[gg, h, SYS.index(sn), 2].sum() / S[gg, h, SYS.index(sn), 0].sum())) for h in range(4)] for nm, sn in SYSN.items()}
        cnt = {nm: [int(S[gg, h, SYS.index(sn), 0].sum()) for h in range(4)] for nm, sn in SYSN.items()}
        assert cnt["Foehn (1 km)"] == cnt["CAMS forecast (40 km)"] == cnt["Aurora (40 km)"], cnt   # identical station-days
        NSD_[pn] = cnt["Foehn (1 km)"]
        if pn in NETS_EXPECT:
            for nm, ex in zip(SYSN, NETS_EXPECT[pn]):
                assert [f"{v:.2f}" for v in R[pn][nm]] == [f"{v:.2f}" for v in ex], (pn, nm, R[pn][nm])
    sd = np.load((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/analyses/circularity/data_ar/station_day.npz"))
    P, O, src = sd["paper"], sd["obs"], sd["src"]
    SEL = {"Stations of the six networks": src != 2, "China": src == 1, "Outside China": (src != 1) & (src != 2)}
    for pn, sel in SEL.items():
        for nm, v in zip(SYSN, ("foehn", "cams1", "aur1")):
            mk = P & sel[None, None]
            r = [float(np.sqrt(np.where(mk[:, h], (sd[v][:, h] - O[:, h]) ** 2, 0).sum() / mk[:, h].sum())) for h in range(4)]
            assert np.allclose(r, R[pn][nm], rtol=0, atol=1e-6) and [int(mk[:, h].sum()) for h in range(4)] == NSD_[pn], (pn, nm, r)
    return R, NSD_


def bars_values():  # FIX-2: (whole function)
    """RMSE reduction (%) of Foehn (four-member mean as run) against the CAMS forecast issued at the same time, 100 (1 -
    RMSE_F / RMSE_C), by network and pooled, with compare_fair.py's bootstrap verbatim (lines 181-191: default_rng(0),
    blocks of 7 start days, 10,000 resamples, percentiles 2.5 / 97.5 of the ratio) on its stored per-day sums PDN
    (data/compare_fair_<label>.npz), as A14 class U (a14/classU/scripts/st_cf_score.py sys_day); cross-checked against
    a14/classU/results_ar/st_cf_classU_179.json to 1e-6"""
    fz = np.load(HERE / "data" / f"compare_fair_{FM.LABEL}.npz"); SYS = [str(s) for s in fz["systems"]]; nets = [str(s) for s in fz["nets"]]
    PD, PDN, S = fz["PD"], fz["PDN"], fz["S"]
    kF, kC, kA = (SYS.index(SYSN[nm]) for nm in ("Foehn (1 km)", "CAMS forecast (40 km)", "Aurora (40 km)"))
    fin = np.isfinite(PD)
    assert kF == 0 and np.allclose(PDN[:, :6].sum(1), PDN[:, 6]) and np.allclose(PDN[:, 6][fin], PD[fin])
    rng = np.random.default_rng(0)                                   # compare_fair.py lines 181-184, verbatim
    ok_d = np.isfinite(PD[0, :, 0, 0])
    nd, B = int(ok_d.sum()), 7
    idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]

    def red(gl, h, k):
        a, b = PDN[kF][gl].sum(0)[ok_d, h - 1], PDN[k][gl].sum(0)[ok_d, h - 1]
        r = lambda ix: np.sqrt(np.nansum(a[ix, 0], 1) / np.nansum(a[ix, 1], 1)) / np.sqrt(np.nansum(b[ix, 0], 1) / np.nansum(b[ix, 1], 1)) - 1
        bs = r(idx); pt = np.sqrt(np.nansum(a[:, 0]) / np.nansum(a[:, 1])) / np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1])) - 1
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return {"reduction_pct": round(float(-100 * pt), 4), "ci95_pct": [round(float(-100 * hi), 4), round(float(-100 * lo), 4)],
                "rmse_foehn": round(float(np.sqrt(np.nansum(a[:, 0]) / np.nansum(a[:, 1]))), 4),
                "rmse_ref": round(float(np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1]))), 4), "station_days": int(np.nansum(a[:, 1]))}

    J = json.loads(CLASSU.read_text())["groups"]
    E = {}
    for g, net in enumerate(nets[:6]):
        E[NETE[net]] = red([g], 1, kC); E[NETE[net]]["network"] = net
        j = J[net]["1"]["Foehn 1 km"]["gain_pct"]["CAMS d+1 (equal issuance)"]["as_run"]
        assert np.allclose([E[NETE[net]]["reduction_pct"], *E[NETE[net]]["ci95_pct"]], [-j["v"], -j["ci"][1], -j["ci"][0]], atol=1e-3), net
    order = sorted(E, key=lambda n: -E[n]["reduction_pct"])
    for n in order:
        e = E[n]; lo, hi = e["ci95_pct"]
        e["printed"] = f"{e['reduction_pct']:.0f}".replace("-", "−"); e["interval_contains_zero"] = bool(lo <= 0 <= hi)
        e["printed_interval"] = [f"{lo:.0f}", f"{hi:.0f}"]
        assert e["printed"] == EXPECT_E[n], (n, e["printed"], EXPECT_E[n])
    oc = [g for g, s in enumerate(nets[:6]) if s != "China (CNEMC)"]; ch = [nets.index("China (CNEMC)")]
    P = {"outside China (five networks pooled)": oc, "China (CNEMC)": ch, "six networks": [6]}
    pooled = {pn: {"vs_CAMS": [red(gl, h, kC) for h in range(1, 5)], "vs_Aurora": [red(gl, h, kA) for h in range(1, 5)]} for pn, gl in P.items()}
    for h in range(1, 5):                                            # cross-check of the pooled outside-China row (A14 row 21)
        j = J["outside China"][str(h)]["Foehn 1 km"]["gain_pct"]["CAMS d+1 (equal issuance)"]["as_run"]
        o = pooled["outside China (five networks pooled)"]["vs_CAMS"][h - 1]
        assert np.allclose([o["reduction_pct"], *o["ci95_pct"]], [-j["v"], -j["ci"][1], -j["ci"][0]], atol=1e-3), h
    # China's share of the station-days and of the reduction of the summed squared error against CAMS (S: identical station-days)
    gC, gA = nets.index("China (CNEMC)"), nets.index("all")
    share = {"station_days_pct": [round(float(100 * S[gC, h, kF, 0] / S[gA, h, kF, 0]), 2) for h in range(4)],
             "reduction_of_summed_squared_error_vs_CAMS_pct": [round(float(100 * (S[gC, h, kC, 2] - S[gC, h, kF, 2]) / (S[gA, h, kC, 2] - S[gA, h, kF, 2])), 1)
                                                                for h in range(4)]}
    NUM["band_e"] = {"panel": "e, RMSE reduction against the CAMS forecast 1 day ahead by network (%), bars in decreasing order",
                     "source": f"{HERE}/data/compare_fair_{FM.LABEL}.npz PDN[system, network, start day, lead, (SSE, count)]; bootstrap = "
                               "compare_fair.py lines 181-191 verbatim (default_rng(0), blocks of 7 start days, 10,000 resamples, "
                               "percentiles 2.5 / 97.5); reduction = -100 (ratio - 1); identical to a14/classU/results_ar/"
                               "st_cf_classU_179.json (as run) to 1e-3 points (asserted)",
                     "mask": "paper mask of compare_fair (official networks, finite measurement, the ten systems finite), four members as run",
                     "order": order, "values": {n: E[n] for n in order},
                     "hatched_lighter": [n for n in order if E[n]["interval_contains_zero"]]}
    NUM["pooled_for_legend"] = {"note": "intervals: compare_fair's bootstrap (default_rng(0)); the intervals of the independent check (own seed) differ by "
                                        "<= 0.1 point (e.g. outside China 1 day ahead 0.5-9.7 there, 0.4-9.7 here)",
                                "leads": [1, 2, 3, 4], **pooled, "China_share": share}
    return E, order


def spread(y, gap):
    """positions (mm) moved apart to at least gap, order kept, symmetric pushes"""
    y = np.asarray(y, float); o = np.argsort(y); z = y[o].copy()
    for _ in range(200):
        moved = False
        for k in range(1, len(z)):
            if z[k] - z[k - 1] < gap - 1e-9:
                d = (gap - (z[k] - z[k - 1])) / 2; z[k - 1] -= d; z[k] += d; moved = True
        if not moved:
            break
    out = np.empty_like(z); out[o] = z
    return out


def band(fig, FH, a_b, AXB, ROWL, AXA):
    """c-e: three line panels of equal size under b, outer edges on the outer edges of b's columns; the figure grows by EXTRA
    mm at the bottom and every panel of a and b keeps its size in mm (shifted up by EXTRA, as variant B of figs_show); then a
    (map and colour bar, unchanged) moves down so that the white gap between a's lowest labels and b's highest text equals
    the gap between b's lowest labels and the legend of the band (one vertical rhythm).
    FIX-1/FIX-2: c and d keep their place and width (log y axes, same ratio); e is the bar panel, its left edge set so that
    the white gap between d and e equals the one between c and d, its right edge on b's right edge"""
    R, NSD_ = band_values()
    E, ORDER = bars_values()  # FIX-2:
    r = fig.canvas.get_renderer(); fig.canvas.draw()
    lowest = min(a_.get_tightbbox(r).y0 for a_ in AXB) / fig.dpi / MM          # mm above the canvas bottom: b's lowest label
    EXTRA = (BAND["below_mm"] + BAND["axes_mm"] + BAND["title_mm"] + BAND["legend_gap_mm"] + BAND["legend_mm"] + BAND["gap_mm"] - lowest) * MM
    fig.set_size_inches(DOUBLE, FH + EXTRA)
    for ax_ in fig.axes:
        p = ax_.get_position(original=True)
        ax_.set_position([p.x0, (p.y0 * FH + EXTRA) / (FH + EXTRA), p.width, p.height * FH / (FH + EXTRA)], which="both")
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    Hm = (FH + EXTRA) / MM; Wm = DOUBLE / MM
    pos = [a_.get_position() for a_ in AXB]
    L, Rr = min(p.x0 for p in pos) * Wm, max(p.x1 for p in pos) * Wm                 # outer edges of b's columns (mm)
    xs = sorted({(round(p.x0, 6), round(p.x1, 6)) for p in pos})                    # the four columns of b
    g = float(np.mean([(b0 - a1) * Wm for (_, a1), (b0, _) in zip(xs[:-1], xs[1:])]))  # gutter of b's columns (mm), used by the band
    pw = (Rr - L - 2 * g) / 3; y_ax = BAND["below_mm"]; h_ax = BAND["axes_mm"]
    pb = a_b.get_position(); dlet = (pb.x0 - (pb.x0 - 0.12 * pb.width)) * Wm         # letter b: its right edge dlet mm left of its axes
    rowx = min(t.get_window_extent(r).x0 for t in ROWL) / fig.dpi / MM               # left edge of b's row labels (mm)
    AX, TT = [], []
    for k, pn in enumerate(YT):  # FIX-2: c and d only
        vals = R[pn]  # FIX-2:
        x_ax = L + k * (pw + g)
        ax = fig.add_axes([x_ax / Wm, y_ax / Hm, pw / Wm, h_ax / Hm]); AX.append(ax)
        allv = [v for nm in STY for v in vals[nm]]  # FIX-1: limits g / sqrt(RATIO), g sqrt(RATIO), g = geometric mean of min and max
        gm = float(np.sqrt(min(allv) * max(allv))); y0, y1 = gm / np.sqrt(RATIO), gm * np.sqrt(RATIO); yt = YT[pn]  # FIX-1:
        lg = float(np.log10(y1 / y0))  # FIX-1:
        mm_of = lambda v: (np.log10(v) - np.log10(y0)) / lg * h_ax  # FIX-1: height (mm) above the bottom of the axes
        v_of = lambda mm: y0 * 10 ** (mm / h_ax * lg)  # FIX-1: value at that height
        # x range: room for the values printed left of lead 1 and right of lead 4 (5.3 pt, at most 4 characters)
        lab_mm, off_mm = 4.6, 1.3
        a_ = (lab_mm + off_mm + 0.6) / pw * 3 / (1 - 2 * (lab_mm + off_mm + 0.6) / pw)
        xl0, xl1 = 1 - a_, 4 + a_; xpm = pw / (xl1 - xl0)
        ref = vals["CAMS forecast (40 km)"][0]
        ax.plot([1, 4], [ref, ref], color=CAMS, lw=0.6, ls=(0, (1, 1.6)), zorder=2, solid_capstyle="butt")
        for nm in STY:
            ax.plot(range(1, 5), vals[nm], color=STY[nm]["color"], marker=STY[nm]["marker"], ms=STY[nm]["ms"], lw=STY[nm]["lw"],
                    zorder=STY[nm]["zorder"], mew=0, clip_on=False)
        for h, side in ((0, "left"), (3, "right")):                                  # values at leads 1 and 4
            ys = [vals[nm][h] for nm in STY]
            yy = spread([mm_of(v) for v in ys], 2.25)  # FIX-1: spacing in mm on the log axis
            for nm, yv in zip(STY, yy):
                xv = (1 - off_mm / xpm) if side == "left" else (4 + off_mm / xpm)
                ax.text(xv, v_of(yv), f"{vals[nm][h]:.1f}", ha="right" if side == "left" else "left", va="center",  # FIX-1: y
                        fontsize=5.3, color=STY[nm]["color"], fontweight="bold" if nm == "Foehn (1 km)" else "normal", zorder=8)
        if k == 0:                                                                   # the reference line, labelled once
            ax.text(2.05, v_of(mm_of(ref) - 0.5), "CAMS forecast, 1 day ahead", ha="center", va="top", fontsize=5.2, color=CAMS, zorder=3)  # FIX-1: x 2.5 -> 2.05, 0.7 -> 0.5 mm: clear of Foehn's line on the log axis
        ax.set_yscale("log")  # FIX-1:
        ax.set_xlim(xl0, xl1); ax.set_ylim(y0, y1); ax.set_xticks(range(1, 5))
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))  # FIX-1: plain
        ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())  # FIX-1: no minor ticks
        ax.tick_params(labelsize=5.5, length=2, width=0.5, pad=1.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_linewidth(0.5)
        ax.set_xlabel("Lead time (days)", fontsize=6, labelpad=1.5)
        TT.append(ax.set_title(pn, fontsize=6.5, pad=2, fontweight="bold"))
        NUM.setdefault("band", {})[pn] = {nm: {"rmse": [round(v, 4) for v in vals[nm]], "printed_lead1": f"{vals[nm][0]:.1f}",
                                               "printed_lead4": f"{vals[nm][3]:.1f}"} for nm in STY}
        NUM["band"][pn]["station_days_by_lead"] = NSD_[pn]
        NUM["band"][pn]["y_range"] = [round(y0, 3), round(y1, 3)]  # FIX-1:
        NUM["band"][pn]["y_axis"] = {"scale": "log", "top_bottom_ratio": RATIO, "ticks": yt}  # FIX-1:
    # y label of c on the x of b's row labels (left edge), one line
    AX[0].text((rowx - L) / pw, 0.5, "RMSE (µg m$^{-3}$)", transform=AX[0].transAxes, rotation=90,  # MAIN-1: shorter than the axis, centred (re-check R3-1)
               ha="left", va="center", fontsize=6)  # FIX-5: with h_ax < the label's length its top stays level with the axes, below letter c
    ae = fig.add_axes([(L + 2 * (pw + g)) / Wm, y_ax / Hm, pw / Wm, h_ax / Hm])  # FIX-2: e, provisional place (moved below)
    bars(ae, E, ORDER, h_ax)  # FIX-2:
    TT.append(ae.set_title("By network, 1 day ahead", fontsize=6.5, pad=2, fontweight="bold"))  # FIX-2: descriptor
    fig.canvas.draw(); r = fig.canvas.get_renderer()  # FIX-2: e's left edge: the white gap c -> d repeated d -> e
    tbm = lambda a_: a_.get_tightbbox(r)  # FIX-2:
    gap_cd = (tbm(AX[1]).x0 - tbm(AX[0]).x1) / fig.dpi / MM  # FIX-2: white gap between c's content and d's tick labels (mm)
    dec_e = ae.get_position().x0 * Wm - tbm(ae).x0 / fig.dpi / MM  # FIX-2: width of e's tick labels and y label (mm)
    xe0 = tbm(AX[1]).x1 / fig.dpi / MM + gap_cd + dec_e  # FIX-2:
    ae.set_position([xe0 / Wm, y_ax / Hm, (Rr - xe0) / Wm, h_ax / Hm]); AX.append(ae)  # FIX-2: right edge on b's right edge
    fig.canvas.draw()
    for k, (ax, tt, s) in enumerate(zip(AX, TT, "cde")):                            # letters: b's offset from its axes, title baseline
        xb = (ax.get_position().x0 * Wm - dlet) / Wm
        yb = fig.transFigure.inverted().transform(tt.get_transform().transform(tt.get_position()))[1]
        fig.text(xb, yb, s, fontsize=8, fontweight="bold", ha="right", va="baseline")
    hd = [Line2D([], [], color=STY[nm]["color"], marker=STY[nm]["marker"], ms=STY[nm]["ms"], lw=STY[nm]["lw"], mew=0)
          for nm in ("Foehn (1 km)", "CAMS forecast (40 km)", "Aurora (40 km)")]
    yleg = (y_ax + h_ax + BAND["title_mm"] + BAND["legend_gap_mm"] + BAND["legend_mm"] / 2) / Hm
    xcd = (L + L + 2 * pw + g) / 2  # FIX-2: the legend belongs to the line panels c-d: centred over them
    leg = fig.legend(hd, ["Foehn (1 km)", "CAMS forecast (40 km)", "Aurora (40 km)"], loc="center", bbox_to_anchor=(xcd / Wm, yleg),  # FIX-2:
                     ncol=3, frameon=False, fontsize=6, handlelength=2.4, columnspacing=2.2, borderaxespad=0, borderpad=0)
    fig.canvas.draw(); r = fig.canvas.get_renderer()                                 # one vertical rhythm: a -> b as b -> band
    b_low = min(a_.get_tightbbox(r).y0 for a_ in AXB); gap_bc = b_low - leg.get_window_extent(r).y1
    b_top = max([a_.get_tightbbox(r).y1 for a_ in AXB] + [t.get_window_extent(r).y1 for t in a_b.texts])
    a_low = min(a_.get_tightbbox(r).y0 for a_ in AXA)
    dy = (a_low - b_top - gap_bc) / fig.dpi / (FH + EXTRA)                             # figure fraction
    assert dy >= 0, dy
    for a_ in AXA:
        p = a_.get_position(original=True); a_.set_position([p.x0, p.y0 - dy, p.width, p.height], which="both")
    NUM["layout_mm_gaps"] = {"a_to_b_before": round((a_low - b_top) / fig.dpi / MM, 2), "b_to_legend": round(gap_bc / fig.dpi / MM, 2),
                             "a_moved_down": round(dy * (FH + EXTRA) / MM, 2)}
    NUM["band"]["source"] = (f"{HERE}/data/compare_fair_{FM.LABEL}.npz S[net, lead, system, (0 count, 2 SSE)]; 'Outside China' = sums over "
                             "Europe (EEA), USA (EPA), Korea (AirKorea), Japan (NIES), India (CPCB); identical to analyses/circularity/data_ar/"
                             "station_day.npz (paper mask) to 1e-6")
    NUM["band"]["Outside China (not drawn; for the legend)"] = {nm: {"rmse": [round(v, 4) for v in R["Outside China"][nm]]} for nm in STY}  # FIX-2:
    NUM["band"]["Outside China (not drawn; for the legend)"]["station_days_by_lead"] = NSD_["Outside China"]  # FIX-2:
    NUM["layout_mm"] = {"band_left_right_canvas": [round(L, 2), round(Rr, 2)], "panel_width_c_d": round(pw, 2), "axes_height": h_ax,  # FIX-2:
                        "gutter": round(g, 2), "letter_offset_from_axes": round(dlet, 2), "extra_height": round(EXTRA / MM, 2),
                        "e_left_width": [round(xe0, 2), round(Rr - xe0, 2)], "white_gap_c_d": round(gap_cd, 2), "row_gap_b": ROWGAP_MM}  # FIX-2:


def bars(ae, E, order, h_ax):  # FIX-2: (whole function)
    """e: one bar per network, decreasing reduction; thin whisker = 95 % interval; value (integer per cent) above the whisker;
    a bar whose interval contains zero hatched and lighter; zero line"""
    lo = [E[n]["ci95_pct"][0] for n in order]; hi = [E[n]["ci95_pct"][1] for n in order]; v = [E[n]["reduction_pct"] for n in order]
    top_mm, bot_mm = 3.1, 0.6                                        # room for the value above the highest whisker, below the lowest
    u = (max(hi) - min(lo)) / (h_ax - top_mm - bot_mm)               # per cent per mm
    y0, y1 = min(lo) - bot_mm * u, max(hi) + top_mm * u
    for i, n in enumerate(order):
        ns = E[n]["interval_contains_zero"]
        with plt.rc_context({"hatch.linewidth": 0.45}):
            ae.bar(i, v[i], width=0.62, color=BAR_LIGHT if ns else BAR, edgecolor=BAR if ns else "none", lw=0.45 if ns else 0,
                   hatch="/////" if ns else None, zorder=3)
        ae.vlines(i, lo[i], hi[i], color=WHISK, lw=0.6, zorder=5)
        ae.hlines([lo[i], hi[i]], i - 0.09, i + 0.09, color=WHISK, lw=0.6, zorder=5)
        ae.text(i, hi[i] + 0.6 * u, E[n]["printed"], ha="center", va="bottom", fontsize=5.5, color=BAR_TXT,
                fontweight="normal" if ns else "bold", zorder=6)
    ae.axhline(0, color="#111111", lw=0.5, zorder=4)
    ae.set_xlim(-0.6, len(order) - 0.4); ae.set_ylim(y0, y1)
    ae.set_xticks(range(len(order))); ae.set_xticklabels(order, fontsize=5.5)
    ae.set_yticks([t for t in (-20, 0, 20, 40) if y0 <= t <= y1])
    ae.tick_params(axis="y", labelsize=5.5, length=2, width=0.5, pad=1.5); ae.tick_params(axis="x", length=0, pad=2)
    for s_ in ("top", "right", "bottom"):
        ae.spines[s_].set_visible(False)
    ae.spines["left"].set_linewidth(0.5)
    ae.set_ylabel("RMSE reduction against\nthe CAMS forecast (%)", fontsize=6, labelpad=1.5, linespacing=1.05)
    NUM["band_e"]["y_range"] = [round(y0, 2), round(y1, 2)]


def save_final(fig):  # FINAL-4 (whole function)
    """page exactly 180 mm wide: the content's tight box, centred, widened to 180 mm (refused if the content is wider)"""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r); pad = 0.3 * MM                                          # inches
    wc = tb.x1 - tb.x0; pad = min(pad, (180 * MM - wc) / 2)
    cx = (tb.x0 + tb.x1) / 2
    bb = Bbox([[cx - 90 * MM, tb.y0 - pad], [cx + 90 * MM, tb.y1 + pad]])
    NUM["page_mm"] = [round(bb.width / MM, 2), round(bb.height / MM, 2)]
    NUM["content_width_mm"] = round(wc / MM, 2)
    assert bb.height / MM <= 205.0, NUM["page_mm"]  # FIX-5: page height at most 205 mm
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = OUTDIR / f"fig_overview_final3.{ext}"; tmp = OUTDIR / f".fig_overview_final3.tmp.{ext}"  # FIX-0: names
        fig.savefig(tmp, bbox_inches=bb, **kw)
        fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
        os.replace(tmp, out)
        print("saved", out)
    txt = json.dumps(NUM, indent=1, ensure_ascii=False) + "\n"
    out = OUTDIR / "fig3_numbers.json"; tmp = OUTDIR / ".fig3_numbers.json.tmp"
    with open(tmp, "w") as f:
        f.write(txt); f.flush(); os.fsync(f.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, out)
    print("page", NUM["page_mm"], "mm; content", NUM["content_width_mm"], "mm")


if __name__ == "__main__":
    main()
