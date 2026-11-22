#!/usr/bin/env python3
# COPY made 2026-11-21 of N/rebuild/figs_show/final/fig_gc2_realism_final.py (Fig. 5, second and last version, 24 January 2022).
# CHANGES against it are marked FIX:
#   FIX:1  map rows a-g and i-o and the spectra h, p: forecasts issued 1 and 3 days ahead (was 1 and 5); q-t keep 7, 5, 3, 1;
#   FIX:2  u: y range up to the top of the 16-member band (it was cut at 195 on 23-24 January, check F5-1), ticks 0-200 by 50;
#   FIX:3  u: thin light-grey vertical line (#9A9A9A, 0.4 pt) at 24 January, behind the curves, no text (check F5-3);
#   FIX:4  u: x ticks 5, 9, ..., 29 (check F5-5; was 2, 5, 9, ...);
#   FIX:5  spectrum legend 'Foehn, ensemble mean' (check F5-4); y label stays 'Variance', its unit is written to fig5_numbers.json;
#   FIX:6  numbers for the legend and the text in <out>/fig5_numbers.json: box-mean levels, RMSE against GHAP, ranks of members
#          1-3 by level, mean probability of > 75 where GHAP > 75 and elsewhere (leads 1, 3, 5, 7);
#   FIX:7  output name fig_gc2_realism_final2.
# Run only through render_fig5_final2.py (same folder).
# ---- header of the copied script (kept) ----
# COPY made 2026-11-21 (Fig. 5 final candidate, 24 January 2022) of N/rebuild/figs_show/fig_gc2_realism_show.py (itself a copy of
# N/rebuild/print/figs_letters/fig_gc2_realism_letters.py, the script of the present Fig. 5; its SHOW-1/SHOW-2 lines are kept).
# CHANGES against fig_gc2_realism_show.py are marked FINAL:
#   FINAL:1  maps drawn at the aspect the slots were already sized for (1 deg lat = 1/cos 38.5 deg lon): every map fills its
#            slot, so the map rows, the spectra and the panel titles sit on one grid (before: plate carree squeezed the maps
#            into the middle of taller slots, 2 mm of white above and below);
#   FINAL:2  spectra h, p: Foehn's 16 members as their MEDIAN spectrum (solid blue) and a light-blue band from the lowest to
#            the highest member at each wavelength (before: member 1 alone); ensemble mean, GHAP, CAMS, Aurora unchanged;
#   FINAL:3  spectra h, p: one y range for both, labelled decades only, written as plain numbers (1, 10, 100, 1000, like the
#            wavelength axis: the exponents of 10^n were 3.5 pt), tick labels 5 pt;
#   FINAL:4  contour of q-t: GHAP smoothed over 5 km as in Fig. 7 (FIG/fig_smoke.py: sea = 0, 5 x 5 km box mean), contoured
#            only where the 5 x 5 km box lies entirely on GHAP's land data inside the panel (no segment along the coast or the
#            frame); then not drawn: closed loops enclosing < 25 km2 (the box), open pieces < 5 km long or never > 5 km from
#            the frame (before: the raw 1 km field, a clutter of coast-hugging segments);
#   FINAL:5  panel letters 8 pt bold through the gc_style helpers: letter_before (1 mm before the row descriptors and
#            before 'Power spectrum'), label (u, at its present place);
#   FINAL:6  all text >= 5 pt (4.8 / 4.9 pt -> 5 pt);
#   FINAL:7  right-hand block of the q-t row (PM2.5 colour bar on top of the q-t maps, spectrum legend with a title, its bottom
#            on their bottom), right edges of the colour bar and of the series u on the spectra's (181 mm);
#   FINAL:8  output name fig_gc2_realism_final; the values the legend needs are written to <out>/fig5_numbers.json.
# Run only through render_fig5_final.py (same folder), which puts experiments/figures first on sys.path.
"""Fig. 2 (GenCast style): realism of Foehn's samples, the January 2022 haze over the North China Plain.
Peak day of the haze (highest Beijing-Tianjin-Hebei station mean, figdata_case.py), daily PM2.5:
a-g, forecasts issued one day ahead: a, GHAP (1 km, the reference); b-d, three Foehn members (1 km); e, Foehn's ensemble mean
     (16 members); f, CAMS and g, Aurora (0.4 deg), issued at the same time (end of the start day); h, radially averaged power
     spectrum of each field over the panel area (the 0.4 deg forecasts on their own grid);
i-p, the same three days ahead (FIX:1; was five);
q-t, probability of PM2.5 above 75 ug m-3 (WHO 24-h interim target 1) from Foehn's 16 members, issued 7, 5, 3 and 1 days
     ahead; black contour: 75 ug m-3 in GHAP on that day.
    python fig_gc2_realism.py <run> <eval dir>"""
import json                                                                    # FINAL:8
import os                                                                      # FINAL:8
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import zarr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple                              # FINAL:2 (legend: band + median)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator   # FINAL:3
from scipy.ndimage import uniform_filter                                       # FINAL:4

from fig_exposure_zoom import ghap_box
from gc_style import AURORA, CAMS, FOEHN, GHAP, clean, geo_ticks, label, letter_before   # FINAL:5: + letter_before
import final_model as FM
import nature_style                                                            # FINAL:8 (output folder)
from nature_style import DOUBLE, HERE, MM, PM_BOUNDS, plt, pm_cmap, save

RUN = sys.argv[1] if len(sys.argv) > 1 else FM.M["run"]
EV = sys.argv[2] if len(sys.argv) > 2 else FM.M["case"]
D = HERE.parent / "global1km" / "runs" / RUN / EV
AURD = HERE.parent / "benchmark" / "aurora"
SUB = (41.5, 35.5, 113.5, 121.5)                                              # North China Plain: N, S, W, E
TH = 75.0
PC = ccrs.PlateCarree()
XT, YT = (115, 120), (36, 38, 40)                                            # coordinates of every map (2 mm clear of the edges)
X0 = 7.5                                                                       # left edge of the maps (mm): latitudes beside it
SHOW_DAY = 23                                                                  # SHOW-1: target day of a-t (doy, 0 = 1 Jan): 24 January 2022
LAT0 = 38.5                                                                    # FINAL:1: mid-latitude of SUB
ASPECT = 1.0 / np.cos(np.deg2rad(LAT0))                                        # FINAL:1: 1 deg lat / 1 deg lon on the page
SMOOTH_KM = 5                                                                  # FINAL:4: box of the contour smoothing (Fig. 7)
MIN_LOOP_KM2 = 25.0                                                            # FINAL:4: closed loops enclosing less than the 5 x 5 km box: not drawn
MIN_OPEN_KM = 5.0                                                              # FINAL:4: open pieces shorter than 5 km: not drawn
FRAME_KM = 5.0                                                                 # FINAL:4: open pieces never farther than 5 km from the frame: not drawn
FS_TICK = 5.0                                                                  # FINAL:6: smallest text (was 4.8 / 4.9)


def aurora_all():
    aur = {}
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        for k, d in enumerate(z["days"]):
            aur[int(d)] = (p, k)
    return aur


def aur_field(aur, d, h):
    """Aurora issued with Foehn's forecast of day d (00 UTC of d + 1), daily mean of day d + h (as compare_fair.py)"""
    p, k = aur[d + 1]; A = np.load(p)["pred"][k].astype(np.float32); k0 = 2 * (h - 1)
    first = (np.load(AURD / "raw" / f"{d + 1:03d}.npz")["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9) if h == 1 else A[k0 - 1]
    return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]


def sub40(field, lat0=90.0, dlat=0.4, lon0=0.0, dlon=0.4):
    """the 0.4 deg cells covering SUB (+ their extent)"""
    n, s, w, e = SUB
    lats = lat0 - dlat * np.arange(field.shape[0]); lons = (lon0 + dlon * np.arange(field.shape[1]) + 180) % 360 - 180
    ri = np.where((lats <= n + 0.2) & (lats >= s - 0.2))[0]; ci = np.where((lons >= w - 0.2) & (lons <= e + 0.2))[0]
    return field[np.ix_(ri, ci)], (lons[ci].min() - 0.2, lons[ci].max() + 0.2, lats[ri].min() - 0.2, lats[ri].max() + 0.2)


def spectrum(x, dx_km):
    """radially averaged power spectrum of x (2-D, NaN filled with the mean) -> wavelengths (km), power"""
    x = np.where(np.isfinite(x), x, np.nanmean(x)); x = x - x.mean()
    ny, nx = x.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2 / (nx * ny) ** 2 / np.mean(win ** 2)   # variance per mode
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dx_km)); kx = np.fft.fftshift(np.fft.fftfreq(nx, dx_km))
    K = np.hypot(*np.meshgrid(ky, kx, indexing="ij"))
    nb = 24 if dx_km < 5 else 7                                                      # coarse grids: few modes, fewer bands
    bins = np.logspace(np.log10(1.0 / (min(nx, ny) * dx_km)), np.log10(0.5 / dx_km), nb)
    idx = np.digitize(K.ravel(), bins)
    band = np.array([P.ravel()[idx == i].sum() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    kc = np.sqrt(bins[1:] * bins[:-1])
    return 1.0 / kc, band / np.diff(np.log10(bins))                                  # variance per decade of wavelength


def mapax(fig, rect, title=None, left=False, bottom=False):
    ax = fig.add_axes(rect, projection=PC)
    n, s, w, e = SUB
    ax.set_extent([w, e, s, n], crs=PC)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=2)   # white sea on every panel, 40 km too
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.3, edgecolor="#6E6E6E", zorder=4)
    ax.spines["geo"].set_linewidth(0.4)
    geo_ticks(ax, XT, YT, left=left, bottom=bottom)
    if title:
        ax.set_title(title, fontsize=5.5, pad=1.5)
    return ax


def fill_slot(ax):
    """FINAL:1: the map at 1 deg lat = ASPECT x 1 deg lon, which is the height the slot was computed for (imshow resets it)"""
    ax.set_aspect(ASPECT, adjustable="box")


def write_json(path, obj):
    """FINAL:8: temporary file, fsync, rename, read back"""
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.rename(tmp, path)
    assert json.loads(path.read_text()) == json.loads(json.dumps(obj))


def box_numbers(D, peak, gh, land, sr0, sr1, sc0, sc1, cams, aur):
    """FIX:6: numbers of the legend and the text over the maps' box (land = GHAP > 0, as drawn), target day of a-t. The 0.4 deg
    fields are read at each 1 km pixel in the cell drawn there (nearest grid point); levels are means over the land pixels."""
    n, s, w, e = SUB
    la = (n - (np.arange(gh.shape[0]) + 0.5) * 0.01)[:, None] * np.ones((1, gh.shape[1]))
    lo = (w + (np.arange(gh.shape[1]) + 0.5) * 0.01)[None, :] * np.ones((gh.shape[0], 1))
    la, lo, g = la[land], lo[land], gh[land].astype(np.float64); y = g > TH
    near = lambda f: np.asarray(f, np.float64)[np.rint((90.0 - la) / 0.4).astype(int), np.rint((lo % 360.0) / 0.4).astype(int) % 900]
    r2 = lambda v: round(float(v), 2)
    rmse = lambda v: r2(np.sqrt(np.mean((v - g) ** 2)))
    res = {"definitions": {
        "box": "North China Plain box of the maps, N 41.5 S 35.5 W 113.5 E 121.5; land = GHAP > 0 (%d pixels of 1 km)" % int(land.sum()),
        "level": "mean over the land pixels of the box (ug m-3); for a member, its own box mean",
        "ensemble_mean": "mean of the 16 members pixel by pixel",
        "lowest_member_highest_member": "the members with the lowest and the highest box mean (any index)",
        "rmse": "RMSE against GHAP over the land pixels of the box (ug m-3); CAMS and Aurora read in the 0.4 deg cell drawn at each pixel",
        "cams_aurora": "issued at the same time as Foehn (end of the start day): CAMS run cams[d+1, 0, h-1], Aurora initialised 00 UTC d+1, "
                       "daily mean 0.25/0.5/0.25 of its 12-h steps (as compare_fair.py); both stop at 5 days (CAMS 5 daily leads, "
                       "Aurora 10 steps of 12 h), so null at 7 days",
        "rank": "rank by box-mean level among the 16 members, 1 = highest, 16 = lowest; members 1-3 are fixed indices 0-2, not selected",
        "probability": "P(PM2.5 > 75) = share of the 16 members above 75 at each pixel; mean over the land pixels where GHAP > 75 "
                       "(raw 1 km GHAP, not the smoothed contour) and over the other land pixels"},
        "ghap_level": r2(g.mean()), "ghap_share_of_land_gt75": round(float(y.mean()), 4), "leads": {}}
    for h in (1, 3, 5, 7):
        ml = np.load(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1][:, land].astype(np.float64)
        mu = ml.mean(0); bm = ml.mean(1); rk = [int(np.sum(bm > bm[k])) + 1 for k in range(ml.shape[0])]
        p = (ml > TH).mean(0)
        cf = near(cams[peak - h + 1, 0, h - 1]) if h <= 5 else None
        af = near(aur_field(aur, peak - h, h)) if h <= 5 else None
        res["leads"][f"{h}_day{'s' if h > 1 else ''}"] = {
            "shown_in": {1: "maps a-g, spectrum h, probability t", 3: "maps i-o, spectrum p, probability s",
                         5: "probability r", 7: "probability q"}[h],
            "level": {"ghap": r2(g.mean()), "ensemble_mean": r2(mu.mean()), "lowest_member": r2(bm.min()), "highest_member": r2(bm.max()),
                      "lowest_member_index_1based": int(np.argmin(bm)) + 1, "highest_member_index_1based": int(np.argmax(bm)) + 1,
                      "cams": None if cf is None else r2(cf.mean()), "aurora": None if af is None else r2(af.mean())},
            "rmse_vs_ghap": {"ensemble_mean": rmse(mu), "cams": None if cf is None else rmse(cf), "aurora": None if af is None else rmse(af)},
            "members_1_2_3": {"level": [r2(bm[k]) for k in range(3)], "rank_of_16_1=highest": rk[:3]},
            "prob_gt75_mean": {"where_ghap_gt75": round(float(p[y].mean()), 3), "elsewhere": round(float(p[~y].mean()), 3)},
        }
    return res


def main():
    Z = np.load(HERE / "data" / f"case_eastasia_{RUN}{FM.SFX}.npz")
    peak = int(Z["peak"]) if SHOW_DAY is None else SHOW_DAY                    # SHOW-2: override the data file's peak (24)
    day = (np.datetime64("2022-01-01") + np.timedelta64(peak, "D")).astype("datetime64[D]").astype(object)
    cmap, norm = pm_cmap()
    c0 = np.load(D / f"crop0_d{peak - 1}_h1.npz"); box = c0["box"]; (r0, r1), (cc0, cc1) = c0["rows"], c0["cols"]
    n, s, w, e = SUB
    sr0, sr1 = int(round((box[0] - n) * 100)), int(round((box[0] - s) * 100)); sc0, sc1 = int(round((w - box[2]) * 100)), int(round((e - box[2]) * 100))
    gh = ghap_box(1826 + peak, int(r0), int(r1), int(cc0), int(cc1))[sr0:sr1, sc0:sc1]; land = gh > 0; gh = np.where(land, gh, np.nan)
    cams = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
    aur = aurora_all()
    W, H = DOUBLE, 127 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    pw = 20.4; ph = pw * 6.0 / (8.0 * np.cos(np.deg2rad(LAT0)))                # FINAL:1 (LAT0 = 38.5, as before)
    ext = (w, e, s, n)
    marks = []                                                                     # FINAL:5: (text, letter), placed after the draw
    spec_axes = []; spec_handles = None                                        # FINAL:2-3
    NUM = {"rows": {}}                                                         # FINAL:8
    dx40 = 0.4 * 111 * np.cos(np.deg2rad(LAT0))                                # FINAL:2 (grid step of the 0.4 deg fields, as before)
    for row, h in enumerate((1, 3)):                                          # FIX:1 (was (1, 5))
        y0 = 127 - 7 - (row + 1) * ph - row * 9
        m = np.load(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        m[:, ~land] = np.nan
        cf, cext = sub40(np.asarray(cams[peak - h + 1, 0, h - 1], np.float32))
        af, aext = sub40(aur_field(aur, peak - h, h))
        panels = [("GHAP (1 km)", gh, ext), ("Foehn, member 1", m[0], ext), ("member 2", m[1], ext), ("member 3", m[2], ext),
                  ("Foehn, ensemble mean", np.nanmean(m, 0), ext), ("CAMS (40 km)", cf, cext), ("Aurora (40 km)", af, aext)]
        for j, (title, f, ex_) in enumerate(panels):
            ax = mapax(fig, [fx(X0 + j * (pw + 1.2)), fy(y0), fx(pw), fy(ph)], title, left=j == 0, bottom=row == 1)
            ax.imshow(np.ma.masked_invalid(f), extent=ex_, origin="upper", transform=PC, cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
            fill_slot(ax)                                                          # FINAL:1
        hd = fig.text(fx(X0), fy(y0 + ph + 4.0), f"Forecasts for {day.day} {day:%B} issued {h} day{'s' if h > 1 else ''} ahead",
                      fontsize=6.5, va="baseline")                                 # FINAL:5: descriptor flush with the maps, letter added later
        marks.append((hd, ("a–g", "i–o")[row]))
        # spectrum
        sa = fig.add_axes([fx(167), fy(y0), fx(14), fy(ph)])
        Pm = []
        for k in range(m.shape[0]):                                                # FINAL:2: all 16 members
            lam1, P_ = spectrum(m[k], 1.0); Pm.append(P_)
        Pm = np.array(Pm); med, lo, hi = np.nanmedian(Pm, 0), np.nanmin(Pm, 0), np.nanmax(Pm, 0)
        sa.fill_between(lam1, lo, hi, color=FOEHN, alpha=0.2, lw=0, zorder=1)
        curves = {}
        for f_, dx, col, lw, ls, lab, zo in ((gh, 1.0, GHAP, 1.2, "-", "GHAP", 3), (None, 1.0, FOEHN, 1.0, "-", "Foehn members (median, range)", 4),
                                             (np.nanmean(m, 0), 1.0, FOEHN, 0.8, (0, (2, 1)), "Foehn, ensemble mean", 4),   # FIX:5 (was "Foehn mean")
                                             (cf, dx40, CAMS, 0.8, "-", "CAMS", 5), (af, dx40, AURORA, 0.8, "-", "Aurora", 5)):
            lam, P = (lam1, med) if f_ is None else spectrum(f_, dx)
            sa.loglog(lam, P, color=col, lw=lw, ls=ls, label=lab, zorder=zo)
            curves[lab] = (lam, P)
        sa.set_xlim(600, 2); sa.set_xticks([100, 10]); sa.set_xticklabels(["100", "10"]); sa.xaxis.set_minor_formatter(NullFormatter())
        sa.tick_params(labelsize=FS_TICK, length=1.5, width=0.4, pad=1.0)        # FINAL:6
        sa.tick_params(which="minor", length=0.9, width=0.3)
        sa.set_xlabel("Wavelength (km)", fontsize=5, labelpad=0.5)
        sa.set_ylabel("Variance", fontsize=5, labelpad=0.5)
        for s_ in ("top", "right"):
            sa.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            sa.spines[s_].set_linewidth(0.4)
        sa.set_title("Power spectrum", fontsize=5.5, pad=1.5)
        marks.append((sa.title, "hp"[row]))                                        # FINAL:5: letter_before 'Power spectrum'
        spec_axes.append(sa)
        if row == 0:
            hh, ll = sa.get_legend_handles_labels()
            i_ = ll.index("Foehn members (median, range)")
            hh[i_] = (Patch(facecolor=FOEHN, alpha=0.2, lw=0), Line2D([], [], color=FOEHN, lw=1.0))   # FINAL:2: band + median
            spec_handles = (hh, ll)
        # FINAL:8: spectrum values for the legend (wavelength band nearest 10 km, and where GHAP sits in the members' range)
        i10 = int(np.nanargmin(np.abs(np.log(lam1 / 10.0))))
        sm = (lam1 >= 2) & (lam1 <= 100) & np.isfinite(lo) & np.isfinite(curves["GHAP"][1])
        gP = curves["GHAP"][1]
        NUM["rows"][f"lead{h}"] = {
            "wavelength_band_near_10km_km": round(float(lam1[i10]), 2),
            "variance_per_decade_at_that_band": {"ghap": round(float(gP[i10]), 3), "members_min": round(float(lo[i10]), 3),
                                                  "members_median": round(float(med[i10]), 3), "members_max": round(float(hi[i10]), 3),
                                                  "ensemble_mean": round(float(curves["Foehn, ensemble mean"][1][i10]), 3)},
            "member1_rank_at_that_band_1=highest": int(1 + np.sum(Pm[:, i10] > Pm[0, i10])),
            "bands_2_to_100km": int(sm.sum()),
            "bands_2_to_100km_ghap_inside_member_range": int(np.sum(sm & (gP >= lo) & (gP <= hi))),
            "bands_2_to_100km_ghap_above_member_max": int(np.sum(sm & (gP > hi))),
            "bands_2_to_100km_ghap_below_member_min": int(np.sum(sm & (gP < lo))),
            "bands_2_to_100km_ensemble_mean_below_member_min": int(np.sum(sm & (curves["Foehn, ensemble mean"][1] < lo))),
        }
    # FINAL:3: one y range for h and p (all curves and bands, x 2-600 km, plus a small margin), labelled decades only
    vals = np.concatenate([ln.get_ydata()[(ln.get_xdata() >= 2) & (ln.get_xdata() <= 600)] for a_ in spec_axes for ln in a_.get_lines()]
                          + [np.asarray(pa.vertices[:, 1]) for a_ in spec_axes for pc in a_.collections for pa in pc.get_paths()])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ylo, yhi = vals.min() / 1.4, vals.max() * 1.25
    for sa in spec_axes:
        sa.set_ylim(ylo, yhi)
        sa.yaxis.set_major_locator(LogLocator(base=10, numticks=10)); sa.yaxis.set_minor_locator(NullLocator())
        sa.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    # probability of > 75 by lead
    y0 = 35; pw2 = 27.0; ph2 = pw2 * 6.0 / (8.0 * np.cos(np.deg2rad(LAT0)))   # FINAL:1 (LAT0 = 38.5, as before)
    pcm = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=10)
    pn = BoundaryNorm(np.linspace(0, 1, 11), 10)
    hd = fig.text(fx(X0), fy(y0 + ph2 + 4.0), f"Probability of PM$_{{2.5}}$ > {TH:g} µg m$^{{-3}}$ on {day.day} {day:%B} "
                  f"(16 members); contour: {TH:g} µg m$^{{-3}}$ in GHAP", fontsize=6.5, va="baseline")   # FINAL:5
    marks.append((hd, "q–t"))
    # FINAL:4: the truth contour as Fig. 7 draws it (sea = 0, 5 x 5 km box mean), kept only where the whole box lies on GHAP's land
    # data inside the panel (the coast and the panel frame are both edges of the data), then without the features smaller than the box
    gsm = uniform_filter(np.where(land, gh, 0.0).astype(np.float32), SMOOTH_KM)
    inside = uniform_filter(land.astype(np.float32), SMOOTH_KM, mode="constant", cval=0.0) > 1 - 1e-6
    gsm = np.ma.masked_where(~inside, gsm)
    xs_, ys_ = np.linspace(w, e, gh.shape[1]), np.linspace(n, s, gh.shape[0])
    import contourpy                                                               # matplotlib's own contouring (mpl2014, corner mask)
    from matplotlib.collections import LineCollection
    kx_, ky_ = 111.32 * np.cos(np.deg2rad(LAT0)), 111.32
    seg_len = lambda sg: float(np.hypot(np.diff(sg[:, 0]) * kx_, np.diff(sg[:, 1]) * ky_).sum())
    seg_area = lambda sg: 0.5 * abs(float(np.dot(sg[:, 0] * kx_, np.roll(sg[:, 1] * ky_, 1)) - np.dot(sg[:, 1] * ky_, np.roll(sg[:, 0] * kx_, 1))))
    seg_frame = lambda sg: float(np.max(np.minimum(np.minimum(sg[:, 0] - w, e - sg[:, 0]) * kx_, np.minimum(sg[:, 1] - s, n - sg[:, 1]) * ky_)))
    pts_, cod_ = contourpy.contour_generator(xs_, ys_, gsm, name="mpl2014", corner_mask=True, line_type="SeparateCode").lines(TH)
    allsegs = [(sg, bool(cd[-1] == 79 or np.allclose(sg[0], sg[-1]))) for sg, cd in zip(pts_, cod_) if len(sg) > 1]   # 79: CLOSEPOLY

    def dropped(sg, closed):
        if closed:
            return "closed_loop_lt_%g_km2" % MIN_LOOP_KM2 if seg_area(sg) < MIN_LOOP_KM2 else None
        if seg_len(sg) < MIN_OPEN_KM:
            return "open_piece_lt_%g_km" % MIN_OPEN_KM
        return "open_piece_within_%g_km_of_frame" % FRAME_KM if seg_frame(sg) < FRAME_KM else None
    why = [dropped(sg, c_) for sg, c_ in allsegs]
    keep = [sg for (sg, c_), y_ in zip(allsegs, why) if y_ is None]
    for j, h in enumerate((7, 5, 3, 1)):
        m = np.load(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        p = np.where(land, (m > TH).mean(0), np.nan)
        ax = mapax(fig, [fx(X0 + j * (pw2 + 1.5)), fy(y0), fx(pw2), fy(ph2)], f"{h} day{'s' if h > 1 else ''} ahead", left=j == 0, bottom=True)
        ax.imshow(np.ma.masked_invalid(p), extent=ext, origin="upper", transform=PC, cmap=pcm, norm=pn, interpolation="nearest", zorder=1)
        ax.add_collection(LineCollection(keep, colors="#111111", linewidths=0.5, transform=PC, zorder=5))   # FINAL:4
        fill_slot(ax)                                                              # FINAL:1
    NUM["contour"] = {"pieces_after_smoothing_and_mask": len(allsegs), "closed_loops_after_smoothing_and_mask": int(sum(c_ for _, c_ in allsegs)),
                      "pieces_not_drawn": {k: int(sum(y_ == k for y_ in why)) for k in sorted(set(why) - {None})},
                      "pieces_drawn": len(keep), "length_drawn_km": round(sum(seg_len(sg) for sg in keep)),
                      "length_not_drawn_km": round(sum(seg_len(sg) for sg, _ in allsegs) - sum(seg_len(sg) for sg in keep)),
                      "share_of_land_px_masked_near_coast_or_frame": round(float(np.mean(~inside[land])), 4),
                      "share_of_land_gt75_raw": round(float(np.mean(gh[land] > TH)), 4),
                      "share_of_unmasked_land_gt75_smoothed": round(float(np.mean(gsm[inside & land] > TH)), 4)}
    pr_right = X0 + 4 * (pw2 + 1.5) - 1.5                                      # FINAL:7 (right edge of t; colour bar as before)
    cax = fig.add_axes([fx(pr_right + 1.5), fy(y0 + 2), fx(2), fy(ph2 - 4)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pn, cmap=pcm), cax=cax, ticks=[0, 0.5, 1]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.4); cb.set_label("probability", fontsize=5.5, labelpad=1)
    # FINAL:7: PM2.5 colour bar and spectrum legend in one block right of the colour bar, top on the maps' top, right edge on the spectra's
    bx0, bx1 = 134.0, 181.0
    cax = fig.add_axes([fx(bx0), fy(y0 + ph2 - 1.8), fx(bx1 - bx0), fy(1.8)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform",
                      ticks=PM_BOUNDS, extendfrac=0.05)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=FS_TICK, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$), a–g, i–o", fontsize=5.5, labelpad=1)
    hh, ll = spec_handles
    leg = fig.legend(hh, ll, fontsize=FS_TICK, frameon=False, loc="lower left", bbox_to_anchor=(fx(bx0), fy(y0)), ncol=2,
                     handlelength=1.8, columnspacing=1.0, borderaxespad=0, borderpad=0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
                     title="Power spectra, h and p", title_fontsize=5.5, alignment="left")
    # u: the forecasts track the observed peaks at the stations (12 CNEMC monitors of central Beijing, January 2022)
    import glob
    import zarr as _zarr
    st = np.load((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km/stations_2022.npz"))
    sel = (st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4)
    obs = np.nanmean(st["daily"][sel].astype(np.float32), 0)
    parts = [np.load(q) for q in sorted(glob.glob(str(FM.CASE_DIR / "part_*.npz")))]
    kk = np.concatenate([q["keys"] for q in parts]); ee = np.concatenate([q["st_ens"] for q in parts]); gg = np.concatenate([q["st_ghap"] for q in parts])
    E = {}; GH = {}
    for n_, (d_, h_, t_) in enumerate(kk):
        E[(int(d_), int(h_))] = np.nanmean(ee[n_, sel].astype(np.float32), 0)          # member means over the stations (16,)
        GH[int(t_)] = float(np.nanmean(gg[n_, sel].astype(np.float32)))
    cz = _zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
    la_, lo_ = st["lat"][sel], st["lon"][sel]
    fi = (90.0 - la_) / 0.4; fj = (lo_ % 360.0) / 0.4
    i0_ = np.floor(fi).astype(int); j0_ = np.floor(fj).astype(int) % 900; wi = fi - i0_; wj = fj - np.floor(fj)
    au = fig.add_axes([fx(15), fy(6), fx(166), fy(19)])                         # FINAL:7: right edge on the spectra's and the colour bar's (181 mm; was 178)
    days = np.arange(1, 29); cal = days + 1                         # target doys 1..28 = 2-29 January
    m1 = np.stack([E[(t - 1, 1)] for t in days]); m3 = np.stack([E[(t - 3, 3)] for t in days if t >= 3])
    cam = []
    for t in days:
        c_ = np.asarray(cz[int(t), 0, 0], np.float32)
        cs = (1 - wi) * ((1 - wj) * c_[i0_, j0_] + wj * c_[i0_, (j0_ + 1) % 900]) + wi * ((1 - wj) * c_[i0_ + 1, j0_] + wj * c_[i0_ + 1, (j0_ + 1) % 900])
        cam.append(float(np.nanmean(cs)))
    au.fill_between(cal, m1.min(1), m1.max(1), color=FOEHN, alpha=0.18, lw=0, zorder=2, label="Foehn, 16 members (1 day ahead)")
    au.plot(cal, m1.mean(1), color=FOEHN, lw=1.5, zorder=5, label="Foehn, ensemble mean (1 day ahead)")
    au.plot(cal[2:], m3.mean(1), color="#8FB2DD", lw=1.0, ls=(0, (3, 1.6)), zorder=4, label="Foehn, ensemble mean (3 days ahead)")
    au.plot(cal, cam, color=CAMS, lw=0.9, zorder=3, label="CAMS, issued at the same time")
    au.plot(cal, [GH[t] for t in days], color=GHAP, lw=0.9, ls=(0, (1, 1.2)), zorder=3, label="GHAP at the station pixels")
    au.plot(cal, obs[days], marker="o", ms=2.6, color="#111111", mfc="white", mew=0.7, lw=0, zorder=6, label="Measured (12 stations)")
    au.axvline(peak + 1, color="#9A9A9A", lw=0.4, zorder=1)                     # FIX:3: 24 January (cal = doy + 1), behind the curves
    au.set_xlim(1.5, 29.5); au.set_xticks([5, 9, 13, 17, 21, 25, 29])           # FIX:4 (was 2, 5, 9, ...)
    utop = max(float(m1.max()), float(np.nanmax(obs[days])), float(max(cam)), float(m3.mean(1).max()),
               float(max(GH[t] for t in days)))                                    # FIX:2: top of everything drawn (the band's top)
    au.set_yticks([0, 50, 100, 150, 200]); au.set_ylim(0, 1.02 * utop)            # FIX:2 (was 195; limits after the ticks)
    au.set_xlabel("January 2022", fontsize=6, labelpad=1); au.set_ylabel("Daily PM$_{2.5}$,\ncentral Beijing (µg m$^{-3}$)", fontsize=6)
    clean(au); au.tick_params(labelsize=5.5)
    au.legend(fontsize=FS_TICK, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.8, columnspacing=1.2)   # FINAL:6
    label(fig, fx(1.5), fy(28.2), "u")                                             # FINAL:5: gc_style.label at the present place
    fig.canvas.draw()                                                              # FINAL:5: letters once the layout is drawn
    for t_, s_ in marks:
        letter_before(fig, t_, s_)
    print("obs:", np.round(obs[days], 0))
    print("band at peak doy24:", round(float(m1[23].min())), round(float(m1[23].max())), "| collapse doy25 obs", round(float(obs[25])), "mean1", round(float(m1[24].mean())))
    print("mean1:", np.round(m1.mean(1), 0))
    NUM.update({
        "figure": "Fig. 5 second and last version (fig_gc2_realism_final2.pdf / .png), script fig_gc2_realism_final2.py, FOEHN_DESIGN=" + os.environ.get("FOEHN_DESIGN", ""),
        "eval_folder": str(D),
        "day": {"date": f"{day:%Y-%m-%d}", "doy0": peak, "choice": "fixed before rendering (SHOW_DAY = 23); the tally over all days is in "
                "N/rebuild/print/show/fig5_24jan_numbers.json and the days shown before stay as Supplementary figures"},
        "maps_a_to_g_lead_days": 1, "maps_i_to_o_lead_days": 3, "probability_maps_q_to_t_lead_days": [7, 5, 3, 1],
        "series_u": {"leads_days": [1, 3], "days": "2-29 January 2022", "stations": int(sel.sum()),
                     "band": "lowest to highest of the 16 members (each member averaged over the stations), 1 day ahead",
                     "y_axis_range": [0.0, round(1.02 * utop, 2)], "band_max_ug_m3": round(float(m1.max()), 2),       # FIX:2
                     "band_max_day": f"{int(cal[int(np.argmax(m1.max(1)))])} January",
                     "vertical_line": "24 January, #9A9A9A, 0.4 pt, no text", "x_ticks": [5, 9, 13, 17, 21, 25, 29]},   # FIX:3-4
        "members": {"count": int(m.shape[0]), "shown_b_to_d_and_j_to_l": "members with fixed indices 1, 2, 3 (array indices 0, 1, 2)",
                    "ensemble_mean": "mean of the 16 members, pixel by pixel"},
        "spectrum_band": {"solid_line": "median over the 16 members of their spectra, wavelength band by wavelength band",
                          "band": "lowest to highest of the 16 member spectra in each wavelength band (full range, min-max)",
                          "method": "radially averaged power spectrum over the panel area, Hann window, NaN (sea) filled with the field mean; "
                                    "1 km fields: 23 log-spaced wavelength bands 2-600 km; 0.4 deg fields: 6 bands on their own grid; "
                                    "variance per decade of wavelength", "y_axis_range": [float(ylo), float(yhi)],
                          "y_label": "Variance",                                                    # FIX:5
                          "y_unit_for_the_figure_legend": "µg² m⁻⁶ per decade of wavelength ((µg m⁻³)² per decade)",
                          "legend_entries": ["GHAP", "Foehn members (median, range)", "Foehn, ensemble mean", "CAMS", "Aurora"]},
        "contour": dict(NUM.get("contour", {}), threshold_ug_m3=TH, field=f"GHAP on {day:%-d %B %Y}",
                        smoothing=f"{SMOOTH_KM} x {SMOOTH_KM} km box mean on the 1 km grid, sea set to 0 (as Fig. 7, FIG/fig_smoke.py)",
                        masking=f"contoured only where the {SMOOTH_KM} x {SMOOTH_KM} km box lies entirely on GHAP land data inside the panel "
                                "(no segment along the coast or the frame)",
                        not_drawn=f"closed loops enclosing < {MIN_LOOP_KM2:g} km2, open pieces < {MIN_OPEN_KM:g} km long or never more than "
                                  f"{FRAME_KM:g} km from the frame (features smaller than the smoothing box)",
                        legend_wording=f"black contour: {TH:g} ug m-3 in GHAP on {day:%-d %B}, smoothed over {SMOOTH_KM} km as in Fig. 7; "
                                       "features smaller than 25 km2 and segments along the coast or the panel edge not drawn"),
        "threshold_q_to_t_ug_m3": TH,
    })
    NUM["box_numbers"] = box_numbers(D, peak, gh, land, sr0, sr1, sc0, sc1, cams, aur)         # FIX:6
    bb = fig.get_tightbbox(fig.canvas.get_renderer())                              # FINAL:8: printed size (savefig bbox tight, pad 0.01 in)
    NUM["printed_size_mm"] = [round((bb.width + 0.02) * 25.4, 1), round((bb.height + 0.02) * 25.4, 1)]
    out = Path(getattr(nature_style, "OUT", HERE))
    save(fig, "fig_gc2_realism_final2")                                            # FIX:7
    write_json(out / "fig5_numbers.json", NUM)                                     # FINAL:8
    print("peak", peak, day)


if __name__ == "__main__":
    main()
