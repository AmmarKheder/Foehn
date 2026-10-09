#!/usr/bin/env python3
"""A compact, reproducible redesign of the exposure teaser, preserving its data.

GHAP 2022 concentrations and GHS-POP 2020 residents only; no model forecasts.
Coarse concentrations are unweighted means of valid GHAP pixels in aligned
0.4-degree cells. City-centred areas and dates match teaser_cities.json exactly.

Run in the existing figure environment:
    ./render.sh fig1_teaser_exposure_redesign

Writes a separate PDF/PNG, a caption, and the numerical provenance; never
overwrites the original figure or modifies the underlying data.
"""
import json
from datetime import date, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

from fig_exposure_zoom import ghap_box, pop_box
from nature_style import HERE, MM, OUT, plt


NAME = "fig1_teaser_exposure_redesign"
WIDTH, HEIGHT = 183.0, 210.0
INK, MUTED, ACCENT = "#24333B", "#596870", "#B75038"
ORDER = ("Cairo", "Dhaka", "Seoul", "Lima", "Kinshasa", "Jakarta")


def rect(x, y, w, h):
    return [x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT]


def text(fig, x, y, value, **kw):
    defaults = dict(fontsize=7.2, color=INK, ha="left", va="baseline")
    defaults.update(kw)
    return fig.text(x / WIDTH, y / HEIGHT, value, **defaults)


def load_city(c, threshold):
    n, s, w, e = c["box"]
    r0, r1 = round((84 - n) * 100), round((84 - s) * 100)
    c0, c1 = round((w + 180) * 100), round((e + 180) * 100)
    assert all(k % 40 == 0 for k in (r0, r1, c0, c1)), c["city"]
    fine = ghap_box(1826 + int(c["doy"]), r0, r1, c0, c1)
    pop = pop_box(r0, r1, c0, c1)
    valid = fine > 0
    nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
    count = valid.reshape(nr, 40, nc, 40).sum((1, 3))
    total = np.where(valid, fine, 0).reshape(nr, 40, nc, 40).sum((1, 3))
    coarse = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    coarse = np.repeat(np.repeat(coarse, 40, 0), 40, 1)
    residents = np.where(valid & np.isfinite(coarse), pop, 0.0)
    hidden = float((residents * ((fine > threshold) & ~(coarse > threshold))).sum())
    # The JSON rounds the original calculation to 0.01 million residents.
    assert abs(hidden / 1e6 - c["hidden_M"]) < 0.011, (c["city"], hidden)
    return fine, coarse, valid, (w, e, s, n), hidden


def draw_crop(fig, x, y, field, valid, extent, cmap, norm, scale_bar=False):
    ax = fig.add_axes(rect(x, y, 26, 27))
    ax.imshow(np.where(valid, field, np.nan), origin="upper", extent=extent,
              cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
    west, east, south, north = extent
    for lon in np.arange(west + 0.4, east - 1e-6, 0.4):
        ax.axvline(lon, color="white", lw=0.22, alpha=0.65)
    for lat in np.arange(south + 0.4, north - 1e-6, 0.4):
        ax.axhline(lat, color="white", lw=0.22, alpha=0.65)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / np.cos(np.deg2rad((north + south) / 2)))
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F2F5F6")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#A8B2B7")
        spine.set_linewidth(0.35)
    if scale_bar:
        lat = south + 0.13 * (north - south)
        lon = west + 0.08 * (east - west)
        degrees = 50 / (111.195 * np.cos(np.deg2rad(lat)))
        ax.plot([lon, lon + degrees], [lat, lat], color=INK, lw=1.25,
                solid_capstyle="butt", path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
        ax.text(lon + degrees / 2, lat + 0.045 * (north - south), "50 km", ha="center",
                va="bottom", fontsize=5.8, color=INK,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    return ax


def main():
    meta = json.loads((HERE / "data" / "teaser_cities.json").read_text())
    threshold = float(meta["threshold"])
    cities = {c["city"]: c for c in meta["cities"]}
    with np.load(HERE / "data" / "exposure_2022.npz") as z:
        days = z["days"].copy()
        kt = list(z["thr"]).index(threshold)
        region = list(map(str, z["regions"])).index("World")
        per_day = z["maps"][1, kt].astype(np.float64) / len(days)
        world_hidden = float(z["res"][:, region, 2, kt].sum() / len(days))
        world_exposed = float(z["res"][:, region, 0, kt].sum() / len(days))
    assert np.array_equal(np.sort(days), np.arange(365)), "Expected the full 2022 year"
    assert np.isclose(per_day.sum(), world_hidden, rtol=2e-5), "Map and world total disagree"
    share = 100 * world_hidden / world_exposed

    fig = plt.figure(figsize=(WIDTH * MM, HEIGHT * MM), facecolor="white")
    text(fig, 6, 204.7, "Pollution exposure hidden by coarse grids", fontsize=11, fontweight="bold")
    text(fig, 6, 198.7,
         f"Residents exposed above {threshold:g} µg m$^{{-3}}$ at 1 km while their 0.4° cell mean remains below the threshold.",
         fontsize=7.1, color=MUTED)

    # A quiet world map, with one numbered key for each city-centred example.
    text(fig, 6, 192.0, "a", fontsize=8.5, fontweight="bold")
    text(fig, 11, 192.0, "Mean affected residents per day, per coarse cell", fontsize=7.4)
    ax = fig.add_axes(rect(5, 133, 130, 56), projection=ccrs.Robinson())
    ax.set_extent((-179.99, 179.99, -60, 84), crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F2F2EF", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#8E9697", linewidth=0.23, zorder=4)
    ax.spines["geo"].set_visible(False)
    world_cmap = LinearSegmentedColormap.from_list(
        "exposure_copper_plum", ["#FFF3DB", "#F9C886", "#EA8C60", "#C54C51", "#79284D", "#321638"], N=256)
    world_norm = LogNorm(vmin=30, vmax=1e7)
    lon_edges, lat_edges = -180 + 0.4 * np.arange(901), 84 - 0.4 * np.arange(361)
    mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_less(per_day, 30),
                         cmap=world_cmap, norm=world_norm, transform=ccrs.PlateCarree(),
                         rasterized=True, zorder=2)
    for index, name in enumerate(ORDER):
        n, s, w, e = cities[name]["box"]
        ax.text((w + e) / 2, (n + s) / 2, "bcdefg"[index], transform=ccrs.PlateCarree(),
                fontsize=6.4, fontweight="bold", ha="center", va="center", color=INK,
                bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor=INK, linewidth=0.55),
                zorder=8)
    cax = fig.add_axes(rect(14, 127, 112, 2.2))
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_ticks([30, 1e3, 1e4, 1e5, 1e6, 1e7])
    cb.set_ticklabels(["30", "1,000", "10,000", "100,000", "1 million", "10 million"])
    cb.outline.set_visible(False)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=6, length=2, width=0.45, pad=2)

    # One headline statistic; its denominator and temporal meaning are explicit.
    text(fig, 143, 164, f"{world_hidden / 1e6:.0f}", fontsize=34, fontweight="bold", color=ACCENT)
    text(fig, 144, 159, "million residents", fontsize=8.4, fontweight="bold")
    text(fig, 144, 153.5, "per day, on average", fontsize=7.2, color=MUTED)
    text(fig, 144, 148.5, "in 2022", fontsize=7.2, color=MUTED)
    text(fig, 144, 135, f"{share:.1f}%", fontsize=16, fontweight="bold", color=ACCENT)
    text(fig, 144, 129.5, "of above-threshold", fontsize=6.8, color=MUTED)
    text(fig, 144, 125, "person-days", fontsize=6.8, color=MUTED)
    fig.add_artist(Line2D([6 / WIDTH, 177 / WIDTH], [119 / HEIGHT] * 2,
                          transform=fig.transFigure, color="#CCD2D5", lw=0.6))
    text(fig, 6, 114.8, "Local detail lost after spatial averaging", fontsize=8.5, fontweight="bold")

    pm_cmap = LinearSegmentedColormap.from_list(
        "pm_threshold", [(0, "#F4F6F7"), (0.4999, "#ADC3CE"),
                         (0.5, "#D95E3E"), (0.75, "#B83738"), (1, "#661B3B")], N=512)
    pm_cmap.set_bad((0, 0, 0, 0))
    pm_norm = TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=150)
    audit = []
    for index, name in enumerate(ORDER):
        row, col = divmod(index, 3)
        x0 = 6 + 58.5 * col
        base = 74 if row == 0 else 27
        city = cities[name]
        fine, coarse, valid, extent, hidden = load_city(city, threshold)
        day = date(2022, 1, 1) + timedelta(days=int(city["doy"]))
        text(fig, x0, base + 35, "bcdefg"[index], fontsize=8.4, fontweight="bold")
        text(fig, x0 + 4.8, base + 35, name, fontsize=8.2, fontweight="bold")
        text(fig, x0 + 54, base + 35, day.strftime("%d %b").lstrip("0"),
             fontsize=7.1, color=MUTED, ha="right")
        text(fig, x0 + 13, base + 30, "1 km", fontsize=6.8, ha="center", fontweight="bold")
        text(fig, x0 + 41, base + 30, "0.4° (~40 km)", fontsize=6.8, ha="center", color=MUTED)
        draw_crop(fig, x0, base, fine, valid, extent, pm_cmap, pm_norm, scale_bar=True)
        draw_crop(fig, x0 + 28, base, coarse, valid, extent, pm_cmap, pm_norm)
        text(fig, x0, base - 3.7, f"{hidden / 1e6:.1f}M", fontsize=8.4, fontweight="bold", color=ACCENT)
        text(fig, x0 + 12.2, base - 3.7, "residents hidden by averaging", fontsize=7, color=MUTED)
        audit.append(dict(city=name, date=day.isoformat(), box=city["box"],
                          hidden_residents=hidden, source_hidden_millions=city["hidden_M"]))

    text(fig, WIDTH / 2, 18.6, f"Daily PM$_{{2.5}}$ (µg m$^{{-3}}$)  ·  threshold {threshold:g}",
         fontsize=7.1, ha="center")
    pm_cax = fig.add_axes(rect(37, 13, 109, 2.2))
    pm_cb = fig.colorbar(plt.cm.ScalarMappable(norm=pm_norm, cmap=pm_cmap),
                        cax=pm_cax, orientation="horizontal", extend="max", extendfrac=0.025)
    pm_cb.set_ticks([0, 15, threshold, 75, 150])
    pm_cb.set_ticklabels(["0", "15", f"{threshold:g}", "75", "150"])
    pm_cb.outline.set_visible(False)
    pm_cb.ax.tick_params(labelsize=6.4, length=2, width=0.45, pad=2)
    pm_cb.ax.axvline(threshold, color=INK, lw=0.7, clip_on=False)
    text(fig, WIDTH / 2, 3.4,
         "GHAP 2022 × GHS-POP 2020  ·  City-centred areas on their peak-exposure day  ·  No model forecasts",
         fontsize=6, color=MUTED, ha="center")

    # Check the fixed-page layout rather than hiding overflow with a tight crop.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.texts:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.x1 and bounds.y1 <= fig.bbox.y1, artist.get_text()
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(OUT / f"{NAME}.pdf", dpi=600)
        fig.savefig(OUT / f"{NAME}.png", dpi=400)
    caption = (
        "**Pollution exposure hidden by coarse grids.** "
        f"**a,** Average daily number of residents in 2022 whose GHAP PM₂.₅ concentration exceeds {threshold:g} µg m⁻³ "
        "at 1 km while the mean of their 0.4° cell does not exceed that threshold. "
        f"The global mean is {world_hidden / 1e6:.1f} million residents per day, or {share:.1f}% of above-threshold person-days. "
        "The map omits cells with fewer than 30 affected residents per day. "
        "**b–g,** Identical city-centred areas at 1 km and after spatial averaging, on the selected date in 2022. "
        "Numbers below each pair count residents above the threshold at 1 km but not in their coarse cell. "
        "Areas and dates are unchanged from the original figure: one city-centred area per region from a predefined "
        "candidate list, with the day maximising hidden exposure; these are illustrative extreme cases. "
        "Coarse cells contain the unweighted arithmetic mean of valid 1 km GHAP pixels, anchored at 84°N and 180°W; "
        "0.4° is approximately 40 km, with physical width depending on latitude. "
        "Both concentration panels in every pair use the same colour scale and valid-pixel mask. "
        "Scale bars: 50 km. Population: GHS-POP 2020. These are data-resolution diagnostics, not model forecasts "
        "or measured operational warning outcomes.\n"
    )
    (OUT / f"{NAME}_caption.md").write_text(caption)
    provenance = dict(threshold_ug_m3=threshold, days=len(days),
                      daily_mean_hidden_residents=world_hidden,
                      daily_mean_above_threshold_residents=world_exposed,
                      hidden_share_percent=share, world_map_max=float(per_day.max()),
                      world_map_colour_limits=[30, 1e7], cities=audit,
                      data_sources=["data/exposure_2022.npz", "data/teaser_cities.json",
                                    "GHAP ghap1km_tiles.npy", "GHS-POP pop1km_global.npy"],
                      page_mm=[WIDTH, HEIGHT])
    (OUT / f"{NAME}_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.close(fig)
    print(f"saved {OUT / NAME}")
    print(f"Verified: {world_hidden / 1e6:.3f}M residents/day; {share:.3f}%; six city totals match the source.")


if __name__ == "__main__":
    main()
