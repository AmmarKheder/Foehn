#!/usr/bin/env python3
"""Fig. 2 (GenCast style): realism of Foehn's samples, the January 2022 haze over the North China Plain.
Peak day of the haze (highest Beijing-Tianjin-Hebei station mean, figdata_case.py), daily PM2.5:
a-g, forecasts issued one day ahead: a, GHAP (1 km, the reference); b-d, three Foehn members (1 km); e, Foehn's ensemble mean
     (16 members); f, CAMS and g, Aurora (0.4 deg), issued at the same time (end of the start day); h, radially averaged power
     spectrum of each field over the panel area (the 0.4 deg forecasts on their own grid);
i-p, the same five days ahead;
q-t, probability of PM2.5 above 75 ug m-3 (WHO 24-h interim target 1) from Foehn's 16 members, issued 7, 5, 3 and 1 days
     ahead; black contour: 75 ug m-3 in GHAP on that day.
    python fig_gc2_realism.py <run> <eval dir>"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import zarr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from fig_exposure_zoom import ghap_box
from gc_style import AURORA, CAMS, FOEHN, GHAP, clean, geo_ticks, label
import final_model as FM
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


def main():
    Z = np.load(HERE / "data" / f"case_eastasia_{RUN}.npz")
    peak = int(Z["peak"]); day = (np.datetime64("2022-01-01") + np.timedelta64(peak, "D")).astype("datetime64[D]").astype(object)
    cmap, norm = pm_cmap()
    c0 = np.load(D / f"crop0_d{peak - 1}_h1.npz"); box = c0["box"]; (r0, r1), (cc0, cc1) = c0["rows"], c0["cols"]
    n, s, w, e = SUB
    sr0, sr1 = int(round((box[0] - n) * 100)), int(round((box[0] - s) * 100)); sc0, sc1 = int(round((w - box[2]) * 100)), int(round((e - box[2]) * 100))
    gh = ghap_box(1826 + peak, int(r0), int(r1), int(cc0), int(cc1))[sr0:sr1, sc0:sc1]; land = gh > 0; gh = np.where(land, gh, np.nan)
    cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    aur = aurora_all()
    W, H = DOUBLE, 127 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    pw = 20.4; ph = pw * 6.0 / (8.0 * np.cos(np.deg2rad(38.5)))
    ext = (w, e, s, n)
    for row, h in enumerate((1, 5)):
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
        fig.text(fx(3), fy(y0 + ph + 3.5), f"$\\bf{{{'ai'[row]}}}$–$\\bf{{{'go'[row]}}}$  Forecasts for {day.day} {day:%B} issued "
                 f"{h} day{'s' if h > 1 else ''} ahead", fontsize=6.5, va="bottom")
        # spectrum
        sa = fig.add_axes([fx(167), fy(y0), fx(14), fy(ph)])
        for f_, dx, col, lw, lab in ((gh, 1.0, GHAP, 1.2, "GHAP"), (m[0], 1.0, FOEHN, 1.0, "Foehn member"),
                                     (np.nanmean(m, 0), 1.0, FOEHN, 0.8, "Foehn mean"), (cf, 0.4 * 111 * np.cos(np.deg2rad(38.5)), CAMS, 0.8, "CAMS"),
                                     (af, 0.4 * 111 * np.cos(np.deg2rad(38.5)), AURORA, 0.8, "Aurora")):
            lam, P = spectrum(f_, dx)
            sa.loglog(lam, P, color=col, lw=lw, ls=(0, (2, 1)) if lab == "Foehn mean" else "-", label=lab)
        sa.set_xlim(600, 2); sa.set_xticks([100, 10]); sa.set_xticklabels(["100", "10"]); sa.xaxis.set_minor_formatter(plt.NullFormatter())
        sa.tick_params(labelsize=4.8, length=1.5); sa.set_xlabel("Wavelength (km)", fontsize=5, labelpad=0.5)
        sa.set_ylabel("Variance", fontsize=5, labelpad=0.5)
        for s_ in ("top", "right"):
            sa.spines[s_].set_visible(False)
        sa.set_title("Power spectrum", fontsize=5.5, pad=1.5)
        if row == 1:
            fig.legend(*sa.get_legend_handles_labels(), fontsize=5, frameon=False, loc="lower left", bbox_to_anchor=(fx(132), fy(32)),
                       ncol=3, handlelength=1.8, columnspacing=1.0)
        fig.text(fx(160), fy(y0 + ph + 3.5), f"$\\bf{{{'hp'[row]}}}$", fontsize=6.5, va="bottom")
    # probability of > 75 by lead
    y0 = 35; pw2 = 27.0; ph2 = pw2 * 6.0 / (8.0 * np.cos(np.deg2rad(38.5)))
    pcm = LinearSegmentedColormap.from_list("prob", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"], N=10)
    pn = BoundaryNorm(np.linspace(0, 1, 11), 10)
    fig.text(fx(3), fy(y0 + ph2 + 3.5), f"$\\bf{{q}}$–$\\bf{{t}}$  Probability of PM$_{{2.5}}$ > {TH:g} µg m$^{{-3}}$ on {day.day} {day:%B} "
             f"(16 members); contour: {TH:g} µg m$^{{-3}}$ in GHAP", fontsize=6.5, va="bottom")
    for j, h in enumerate((7, 5, 3, 1)):
        m = np.load(D / f"crop0_d{peak - h}_h{h}.npz")["members"].astype(np.float32)[:, sr0:sr1, sc0:sc1]
        p = np.where(land, (m > TH).mean(0), np.nan)
        ax = mapax(fig, [fx(X0 + j * (pw2 + 1.5)), fy(y0), fx(pw2), fy(ph2)], f"{h} day{'s' if h > 1 else ''} ahead", left=j == 0, bottom=True)
        ax.imshow(np.ma.masked_invalid(p), extent=ext, origin="upper", transform=PC, cmap=pcm, norm=pn, interpolation="nearest", zorder=1)
        ax.contour(np.linspace(w, e, gh.shape[1]), np.linspace(n, s, gh.shape[0]), np.nan_to_num(gh), levels=[TH], colors="#111111",
                   linewidths=0.5, transform=PC, zorder=5)
    cax = fig.add_axes([fx(X0 + 4 * (pw2 + 1.5) + 1.5), fy(y0 + 2), fx(2), fy(ph2 - 4)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=pn, cmap=pcm), cax=cax, ticks=[0, 0.5, 1]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5, length=1.5); cb.set_label("probability", fontsize=5.5, labelpad=1)
    cax = fig.add_axes([fx(134), fy(y0 + 17), fx(46), fy(1.8)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$), a–g, i–o", fontsize=5.5, labelpad=1)
    # u: the forecasts track the observed peaks at the stations (12 CNEMC monitors of central Beijing, January 2022)
    import glob
    import zarr as _zarr
    st = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/stations_2022.npz")
    sel = (st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4)
    obs = np.nanmean(st["daily"][sel].astype(np.float32), 0)
    parts = [np.load(q) for q in sorted(glob.glob(str(FM.CASE_DIR / "part_*.npz")))]
    kk = np.concatenate([q["keys"] for q in parts]); ee = np.concatenate([q["st_ens"] for q in parts]); gg = np.concatenate([q["st_ghap"] for q in parts])
    E = {}; GH = {}
    for n_, (d_, h_, t_) in enumerate(kk):
        E[(int(d_), int(h_))] = np.nanmean(ee[n_, sel].astype(np.float32), 0)          # member means over the stations (16,)
        GH[int(t_)] = float(np.nanmean(gg[n_, sel].astype(np.float32)))
    cz = _zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    la_, lo_ = st["lat"][sel], st["lon"][sel]
    fi = (90.0 - la_) / 0.4; fj = (lo_ % 360.0) / 0.4
    i0_ = np.floor(fi).astype(int); j0_ = np.floor(fj).astype(int) % 900; wi = fi - i0_; wj = fj - np.floor(fj)
    au = fig.add_axes([fx(15), fy(6), fx(163), fy(19)])
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
    au.set_xlim(1.5, 29.5); au.set_xticks([2, 5, 9, 13, 17, 21, 25, 29]); au.set_ylim(0, 195)
    au.set_xlabel("January 2022", fontsize=6, labelpad=1); au.set_ylabel("Daily PM$_{2.5}$,\ncentral Beijing (µg m$^{-3}$)", fontsize=6)
    clean(au); au.tick_params(labelsize=5.5)
    au.legend(fontsize=4.9, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.text(fx(1.5), fy(28.2), "$\\bf{u}$", fontsize=8, va="bottom")
    print("obs:", np.round(obs[days], 0))
    print("band at peak doy24:", round(float(m1[23].min())), round(float(m1[23].max())), "| collapse doy25 obs", round(float(obs[25])), "mean1", round(float(m1[24].mean())))
    print("mean1:", np.round(m1.mean(1), 0))
    save(fig, "fig_gc2_realism")
    print("peak", peak, day)


if __name__ == "__main__":
    main()
