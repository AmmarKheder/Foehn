#!/usr/bin/env python3
"""Income maps for crossing the exposure hidden by a 40 km grid with income (figdata_exposure_income.py), on the 5 arc-minute
grid of the GDP data: (1) the country of every cell, rasterised from the national boundaries of Kummu et al. 2025 (Sci. Data
12, 178; Zenodo 13943886, polyg_adm0_gdp_perCapita_1990_2022.gpkg) with its World Bank income group of FY2024, the
classification based on 2022 GNI per capita (OGHIST.xlsx, sheet 'Country Analytical History'); (2) the GDP per capita at
purchasing power parity of 2022 of every cell at the second administrative level (rast_adm2_gdp_perCapita_1990_2022.tif,
band 2022); (3) the residents of every cell (GHS-POP 2020 at 1 km), the global population-weighted deciles of GDP per
capita and the quintiles within each country. Inputs are read as data only (downloaded files in their own folders).
    python figdata_income.py   -> /scratch/project_462001140/ammar/eccv/data/world/income_derived/income_5arcmin.npz"""
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import shapely

D = Path("/scratch/project_462001140/ammar/eccv/data/world")
KUM, WB, OUT = D / "income_kummu2025", D / "income_worldbank", D / "income_derived"
GROUPS = ["L", "LM", "UM", "H"]                     # World Bank: low, lower-middle, upper-middle, high income
# codes of the boundaries that the World Bank lists under another code or within their state (Venezuela, unclassified since
# FY2022, Western Sahara and Northern Cyprus stay unclassified)
ALIAS = {"KSV": "XKX", "GLP": "FRA", "MTQ": "FRA", "REU": "FRA", "GUF": "FRA", "MYT": "FRA", "SPM": "FRA", "BLM": "FRA",
         "WLF": "FRA", "SJM": "NOR", "GGY": "CHI", "JEY": "CHI", "BES": "NLD"}
G1 = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")                # GHS-POP 2020 at 1 km (pop1km_global.npy)
GH, GW = 14400, 36000                               # 1 km grid, 84 N - 60 S, 0.01 deg


def xlsx_rows(path, sheet):
    """the rows of one sheet of an .xlsx file as lists (standard library only: shared strings, numbers, inline strings)"""
    import re
    import xml.etree.ElementTree as ET
    import zipfile
    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    z = zipfile.ZipFile(path)
    sst = [("".join(t.text or "" for t in si.iter(m + "t"))) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(m + "si")]
    rid = next(s.get(rel) for s in ET.fromstring(z.read("xl/workbook.xml")).iter(m + "sheet") if s.get("name") == sheet)
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = next(r.get("Target") for r in rels if r.get("Id") == rid).lstrip("/")
    target = target if target.startswith("xl/") else "xl/" + target
    out = []
    for row in ET.fromstring(z.read(target)).iter(m + "row"):
        cells = {}
        for c in row.iter(m + "c"):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                col = col * 26 + ord(ch) - 64
            v, t = c.find(m + "v"), c.get("t")
            if t == "s":
                val = sst[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(m + "t"))
            elif v is not None:
                val = v.text if t == "str" else (int(float(v.text)) if float(v.text).is_integer() else float(v.text))
            else:
                val = None
            cells[col - 1] = val
        out.append([cells.get(k) for k in range(max(cells) + 1)] if cells else [])
    return out


def wb_groups(fiscal_year_data=2022):
    """iso3 -> income group of the fiscal year whose data year is 2022 (FY2024)"""
    rows = [r + [None] * 3 for r in xlsx_rows(WB / "OGHIST.xlsx", "Country Analytical History")]
    years = next(r for r in rows if r[1] and str(r[1]).startswith("Data for calendar year"))
    fy = next(r for r in rows if r[1] and str(r[1]).startswith("Bank's fiscal year"))
    col = list(years).index(fiscal_year_data)
    print(f"World Bank classification {fy[col]} (data for {fiscal_year_data})")
    out = {}
    for r in rows:
        if r[0] and isinstance(r[0], str) and len(r[0]) == 3 and r[0].isupper() and r[col] in GROUPS:
            out[r[0]] = r[col]
    return out


def countries():
    """national boundaries (WKB inside GeoPackage blobs) -> list of (iso3, name, geometry)"""
    con = sqlite3.connect(KUM / "polyg_adm0_gdp_perCapita_1990_2022.gpkg")
    t = "polyg_adm0_gdp_perCapita_1990_2022"
    srs = con.execute(f"select srs_id from gpkg_geometry_columns where table_name='{t}'").fetchone()[0]
    assert srs == 4326, srs
    out = []
    for blob, iso, name in con.execute(f"select geom, iso3, Country from '{t}'"):
        flags = blob[3]; env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
        out.append((iso, name, shapely.from_wkb(bytes(blob[8 + env:]))))
    return out


def r5c5():
    """5 arc-minute row and column of the centre of every 1 km row and column"""
    i, j = np.arange(GH), np.arange(GW)
    return np.floor((6.0 + 0.01 * (i + 0.5)) * 12).astype(np.int64), np.floor(0.01 * (j + 0.5) * 12).astype(np.int64)


def wquant(x, w, q):
    """weighted quantiles q of x"""
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return x[o][np.minimum(np.searchsorted(cw, q), len(x) - 1)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grp = wb_groups()
    C = countries()
    with rasterio.open(KUM / "rast_adm2_gdp_perCapita_1990_2022.tif") as r:
        band = list(r.descriptions).index("gdp_pc_2022") + 1
        gdp = r.read(band).astype(np.float64); nod = r.nodata; tr = r.transform; shape = (r.height, r.width)
    gdp[gdp == nod] = np.nan
    cid = rasterio.features.rasterize(((g, k + 1) for k, (_, _, g) in enumerate(C)), out_shape=shape, transform=tr, fill=0,
                                      dtype="int16")
    iso = np.array([str(c[0] or "") for c in C]); names = np.array([str(c[1] or "") for c in C])   # "" for the small islands
    group = np.array([GROUPS.index(grp[ALIAS.get(i, i)]) if ALIAS.get(i, i) in grp else -1 for i in iso], np.int8)   # -1: none
    miss = [f"{i} {n}" for i, n, g in zip(iso, names, group) if g < 0]
    print(f"{len(C)} countries, {int((group >= 0).sum())} with a FY2024 group; not classified: {', '.join(miss)}")
    print(f"cells: country {int((cid > 0).sum())}, GDP per capita {int(np.isfinite(gdp).sum())}, both {int(((cid > 0) & np.isfinite(gdp)).sum())}")
    # residents per 5 arc-minute cell (GHS-POP 2020 inside the 1 km domain): a 1 km pixel belongs to the cell of its centre
    r5, c5 = r5c5()
    pop = np.load(G1 / "pop1km_global.npy", mmap_mode="r")
    pop5 = np.zeros(shape[0] * shape[1])
    for a in range(0, GH, 400):
        blk = np.asarray(pop[a:a + 400], np.float64)
        pop5 += np.bincount((r5[a:a + 400, None] * shape[1] + c5[None, :]).ravel(), weights=blk.ravel(), minlength=pop5.size)
    pop5 = pop5.reshape(shape)
    ok = np.isfinite(gdp) & (pop5 > 0)
    print(f"residents {pop5.sum() / 1e9:.3f} bn, with a GDP per capita {pop5[ok].sum() / 1e9:.3f} bn, in a classified country "
          f"{pop5[(cid > 0) & (group[np.maximum(cid - 1, 0)] >= 0)].sum() / 1e9:.3f} bn")
    # global deciles of GDP per capita, weighted by residents (decile 0 = the poorest tenth of the world's residents)
    dq = wquant(gdp[ok], pop5[ok], np.arange(1, 10) / 10)
    dec = np.full(shape, 10, np.int8); dec[ok] = np.searchsorted(dq, gdp[ok], side="right")
    # quintiles within each country, weighted by its residents; class = country * 6 + quintile (5: no GDP)
    cq = (cid.astype(np.int32) * 6 + 5).astype(np.int16); qthr = np.full((len(C) + 1, 4), np.nan)
    for k in range(1, len(C) + 1):
        m = ok & (cid == k)
        if pop5[m].sum() > 0:
            qthr[k] = wquant(gdp[m], pop5[m], np.array([0.2, 0.4, 0.6, 0.8]))
            cq[m] = k * 6 + np.searchsorted(qthr[k], gdp[m], side="right")
    wbc = np.where(cid > 0, group[np.maximum(cid - 1, 0)], -1); wbc = np.where(wbc < 0, 4, wbc).astype(np.int8)  # 4: none
    print("global deciles of GDP per capita PPP 2022 (int. $):", np.round(dq).astype(int))
    for g in range(5):
        print(f"  {(GROUPS + ['not classified'])[g]:15s} residents {pop5[wbc == g].sum() / 1e9:.3f} bn")
    np.savez_compressed(OUT / "income_5arcmin.npz", country=cid, gdp_pc_2022=gdp.astype(np.float32), iso3=iso, names=names,
                        wb_group=group, groups=np.array(GROUPS), transform=np.array(tr)[:6], pop5=pop5.astype(np.float32),
                        wb_class=wbc, decile=dec, decile_edges=dq, country_quintile=cq, quintile_edges=qthr)
    print("saved", OUT / "income_5arcmin.npz")


if __name__ == "__main__":
    main()
