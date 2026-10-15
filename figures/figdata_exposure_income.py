#!/usr/bin/env python3
"""The exposure that a 40 km grid hides, by income (2022): the computation of figdata_exposure.py (GHAP 1 km against its
40 km cell means, weighted by the GHS-POP 2020 residents of every 1 km pixel with a retrieval), with every 1 km pixel labelled
by the income of the 5 arc-minute cell that contains its centre (figdata_income.py):
  kind 0, wb:       World Bank income group of the country, FY2024 (2022 GNI per capita): L, LM, UM, H, not classified;
  kind 1, decile:   global decile of GDP per capita PPP 2022 at the second administrative level (Kummu et al. 2025),
                    weighted by residents (0 = the poorest tenth of the world's residents), 10 = no GDP;
  kind 2, country:  country of the boundaries of Kummu et al. (0 = none);
  kind 3, cq:       country x quintile of GDP per capita within the country (country * 6 + quintile, 5 = no GDP).
For every class the residents are summed by the number of WHO thresholds (15, 25, 37.5, 50, 75 ug m-3) that their 1 km value
exceeds and that their 40 km cell mean exceeds, which gives, for every threshold, the person-days above it at 1 km and those
hidden by the 40 km cells (above at 1 km, cell mean not above). Totals over the classes of a kind reproduce figdata_exposure.py.
One rank per process (srun); every rank takes days rank, rank + world, ...  -> data/exposure_income/part_<rank>.npz
    srun -n 24 python figdata_exposure_income.py ;  python figdata_exposure_income.py merge ;  python figdata_exposure_income.py table"""
import os
import sys
from pathlib import Path

import numpy as np

from figdata_exposure import GH, GW, I2022, THR, F, G, ghap_day

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "exposure_income"
INC = Path("/scratch/project_462001140/ammar/eccv/data/world/income_derived/income_5arcmin.npz")
KINDS = ("wb", "decile", "country", "cq")
NL = len(THR) + 1                                                           # 0..5 thresholds exceeded


def classes():
    z = np.load(INC)
    nc = int(z["country"].max())
    cls5 = {"wb": z["wb_class"].astype(np.int64), "decile": z["decile"].astype(np.int64), "country": z["country"].astype(np.int64),
            "cq": z["country_quintile"].astype(np.int64)}
    ncls = {"wb": 5, "decile": 11, "country": nc + 1, "cq": (nc + 1) * 6}
    i, j = np.arange(GH), np.arange(GW)
    r5 = np.floor((6.0 + 0.01 * (i + 0.5)) * 12).astype(np.int64); c5 = np.floor(0.01 * (j + 0.5) * 12).astype(np.int64)
    return cls5, ncls, r5, c5


def run():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    OUT.mkdir(parents=True, exist_ok=True)
    cls5, ncls, r5, c5 = classes()
    pop = np.load(G / "pop1km_global.npy", mmap_mode="r")                    # residents per 1 km pixel (page cache shared by the ranks)
    acc = {k: np.zeros(ncls[k] * NL * NL) for k in KINDS}
    days = list(range(365))[rank::world]
    test = os.environ.get("DAYS")                                             # smoke test: DAYS=0,180 (nothing saved)
    days = [int(v) for v in test.split(",")] if test else days
    for d in days:
        x = ghap_day(I2022 + d)
        for b in range(GH // F):
            xb = x[b * F:(b + 1) * F].astype(np.float32); ok = xb > 0
            with np.errstate(all="ignore"):
                cell = np.where(ok, xb, 0).reshape(F, GW // F, F).sum((0, 2)) / ok.reshape(F, GW // F, F).sum((0, 2))
            x40 = np.repeat(cell, F)[None, :]
            w = np.where(ok & np.isfinite(x40), np.asarray(pop[b * F:(b + 1) * F], np.float32), 0.0)
            if not w.any():
                continue
            lv = (np.searchsorted(THR, xb, side="left") * NL + np.searchsorted(THR, np.broadcast_to(x40, xb.shape), side="left"))
            rows = r5[b * F:(b + 1) * F]
            for k in KINDS:
                c = cls5[k][rows[:, None], c5[None, :]]
                acc[k] += np.bincount((c * (NL * NL) + lv).ravel(), weights=w.ravel(), minlength=acc[k].size)
        h = acc["wb"].reshape(-1, NL, NL)
        if test:                                                         # compare with figdata_exposure.py for this day
            ref = np.load(HERE / "data" / "exposure_2022.npz")["res"][d, 0]
            for kk, t_ in enumerate(THR):
                e_, h_ = split(h, kk)
                print(f"  day {d} > {t_:g}: above {e_.sum() / 1e6:.3f} M (ref {ref[0, kk] / 1e6:.3f}), hidden {h_.sum() / 1e6:.3f} M (ref {ref[2, kk] / 1e6:.3f})")
            acc = {k: np.zeros_like(v) for k, v in acc.items()}
        print(f"rank {rank}: day {d} done; hidden above 37.5 so far {h[:, 3:, :3].sum() / 1e9:.2f} bn person-days", flush=True)
    if not test:
        np.savez(OUT / f"part_{rank}.npz", days=np.array(days), **{k: acc[k] for k in KINDS})


def merge():
    P = [np.load(p) for p in sorted(OUT.glob("part_*.npz"))]
    days = np.sort(np.concatenate([p["days"] for p in P])); assert len(days) == 365 and (days == np.arange(365)).all(), len(days)
    z = np.load(INC)
    out = {k: sum(p[k] for p in P).reshape(-1, NL, NL) for k in KINDS}
    np.savez(HERE / "data" / "exposure_income_2022.npz", thr=THR, ndays=365, **out, iso3=z["iso3"], names=z["names"],
             wb_group=z["wb_group"], groups=z["groups"], decile_edges=z["decile_edges"],
             pop_wb=np.bincount(z["wb_class"].ravel(), weights=z["pop5"].ravel(), minlength=5),
             pop_decile=np.bincount(z["decile"].ravel().astype(np.int64), weights=z["pop5"].ravel(), minlength=11),
             pop_country=np.bincount(z["country"].ravel().astype(np.int64), weights=z["pop5"].ravel(), minlength=out["country"].shape[0]),
             pop_cq=np.bincount(z["country_quintile"].ravel().astype(np.int64), weights=z["pop5"].ravel(), minlength=out["cq"].shape[0]))
    print("saved", HERE / "data" / "exposure_income_2022.npz")
    table()


def split(a, k):
    """person-days above threshold k at 1 km, and hidden by the 40 km cells, from counts (..., level 1 km, level 40 km)"""
    return a[..., k + 1:, :].sum((-1, -2)), a[..., k + 1:, :k + 1].sum((-1, -2))


def table():
    z = np.load(HERE / "data" / "exposure_income_2022.npz"); T = list(z["thr"]); nd = int(z["ndays"])
    ref = np.load(HERE / "data" / "exposure_2022.npz")["res"]                # figdata_exposure.py: (days, region, terms, thr)
    for thr in (37.5, 75.0, 15.0):
        k = T.index(thr)
        print(f"\n=== above {thr:g} ug m-3 at the residents' 1 km pixel, 2022")
        for kind in KINDS[:2]:
            e, h = split(z[kind], k); pop = z[f"pop_{kind}"]
            tot_h = h.sum()
            print(f"  check {kind}: hidden {tot_h / nd / 1e6:.1f} M/day, paper {ref[:, 0, 2, k].sum() / nd / 1e6:.1f} M/day; "
                  f"above {e.sum() / nd / 1e6:.0f} vs {ref[:, 0, 0, k].sum() / nd / 1e6:.0f} M/day")
            names = list(z["groups"]) + ["not classified"] if kind == "wb" else [f"D{c + 1}" for c in range(10)] + ["no GDP"]
            print(f"  {'class':15s} {'residents':>9s} {'days above':>10s} {'days hidden':>11s} {'hidden %':>8s} {'of world hidden':>15s}")
            for c, nm in enumerate(names):
                if pop[c] < 1e6:
                    continue
                print(f"  {nm:15s} {pop[c] / 1e9:8.2f}b {e[c] / pop[c]:10.1f} {h[c] / pop[c]:11.2f} {100 * h[c] / max(e[c], 1):7.1f}% "
                      f"{100 * h[c] / tot_h:14.1f}%")
            if kind == "decile":
                print("  quintiles of the world's residents (poorest first):")
                for q in range(5):
                    s_ = slice(2 * q, 2 * q + 2)
                    print(f"  Q{q + 1:<14d} {pop[s_].sum() / 1e9:8.2f}b {e[s_].sum() / pop[s_].sum():10.1f} {h[s_].sum() / pop[s_].sum():11.2f} "
                          f"{100 * h[s_].sum() / max(e[s_].sum(), 1):7.1f}% {100 * h[s_].sum() / tot_h:14.1f}%")
        # within countries: the same country mix in every quintile, so differences between quintiles are within countries
        e, h = split(z["cq"], k); pop = z["pop_cq"]
        nc = len(pop) // 6
        e, h, pop = e.reshape(nc, 6), h.reshape(nc, 6), pop.reshape(nc, 6)
        print("  within countries (quintile of GDP per capita inside each country, poorest first):",
              " | ".join(f"Q{q + 1} hidden {h[1:, q].sum() / pop[1:, q].sum():.2f} d/yr, {100 * h[1:, q].sum() / max(e[1:, q].sum(), 1):.1f}% of exposure"
                         for q in range(5)))
        ec, hc = split(z["country"], k); pc = z["pop_country"]
        top = np.argsort(-hc)[:12]
        print("  countries with the most hidden residents:", "; ".join(
            f"{z['names'][c - 1] if c > 0 else 'none'} {hc[c] / nd / 1e6:.1f} M/day ({100 * hc[c] / max(ec[c], 1):.0f}%)" for c in top if hc[c] > 0))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    merge() if arg == "merge" else table() if arg == "table" else run()
