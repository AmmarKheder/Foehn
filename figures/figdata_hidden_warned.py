#!/usr/bin/env python3
"""Would Foehn have warned the people a 40 km grid hides? For the forecasts of 2022 stored at the populated 1 km pixels
(eval_global_green.py --pop: every member at the 8.1 M pixels where >= 100 people live, 94 % of the world's residents; GHAP of
the target day at those pixels; the 40 km cell means of GHAP, cells of 40 x 40 pixels = 0.4 deg as in figdata_exposure.py):
  hidden person-days: residents whose 1 km GHAP is above the threshold while the GHAP mean of their 40 km cell is not;
  warned: the ensemble probability at the pixel, the share of the members above the threshold, is >= 0.5 (2 of 4 members);
  false alarms on the same target days: warned person-days whose 1 km GHAP is not above the threshold, among all warnings and
  among the warnings where the 40 km cell is not above it (where a perfect 40 km forecast stays silent).
Sums per lead, threshold, region (the boxes of figdata_exposure.py) and target day (for intervals) -> data/hidden_warned_<eval dir>.npz
    python figdata_hidden_warned.py [eval dir]   (default: the --pop evaluation of the final model, eval_pop_<label>)"""
import glob
import sys
from pathlib import Path

import numpy as np

import final_model as FM
from figdata_exposure import GW, REGIONS, G

HERE = Path(__file__).resolve().parent
THRS = (37.5, 75.0)
# sums: hidden, hidden & warned, exposed (1 km above), exposed & warned, warned, warned & false, warned & cell not above,
# warned & cell not above & false, residents with a valid 1 km value
TERMS = ("hid", "hid_w", "exp", "exp_w", "warn", "warn_fa", "warn_c", "warn_c_fa", "valid")


def main():
    ev = Path(sys.argv[1]) if len(sys.argv) > 1 else FM.RUN_DIR / f"eval_pop_{FM.LABEL}"
    idx = np.load(ev / "pop_idx.npy")
    pz = np.load(G / "pop_pixels.npz"); assert np.array_equal(pz["idx"], idx), "pixel sets differ"
    pop = pz["pop"].astype(np.float64)
    r, c = idx // GW, idx % GW
    lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
    cell = (r // 40) * (GW // 40) + c // 40
    R = list(REGIONS)
    reg = np.stack([np.ones(len(idx), bool) if b is None else (lat <= b[0]) & (lat > b[1]) & (lon >= b[2]) & (lon < b[3])
                    for b in REGIONS.values()]).astype(np.float64)            # (regions, pixels): sums by one matrix product
    out = {}                                                                     # (lead, thr) -> list of (target day, sums)
    for part in sorted(glob.glob(str(ev / "part_*.npz")), key=lambda p: int(p.split("_")[-1].split(".")[0])):
        k = part.split("_")[-1].split(".")[0]
        keys = np.load(part)["keys"]
        if len(keys) == 0:
            continue
        ens = np.load(ev / f"pop_{k}.npy", mmap_mode="r"); gh = np.load(ev / f"pop_ghap_{k}.npy", mmap_mode="r")
        c40 = np.load(ev / f"cell40_{k}.npy", mmap_mode="r")
        for n, (d, h, t) in enumerate(keys):
            m = np.asarray(ens[n], np.float32); g1 = np.asarray(gh[n], np.float32); g40 = np.asarray(c40[n, 1], np.float32).ravel()[cell]
            ok = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m).all(1)
            w = np.where(ok, pop, 0.0)
            for thr in THRS:
                above, cabove = g1 > thr, g40 > thr
                warn = (m > thr).mean(1) >= 0.5
                hid = above & ~cabove
                terms = (hid, hid & warn, above, above & warn, warn, warn & ~above, warn & ~cabove, warn & ~cabove & ~above, ok)
                s = reg @ (np.stack(terms, 1) * w[:, None])                                     # (regions, terms)
                out.setdefault((int(h), thr), []).append((int(t), s))
        print(f"part {k}: {len(keys)} forecasts", flush=True)
    res = {}
    for (h, thr), lst in sorted(out.items()):
        days = np.array([t for t, _ in lst]); S = np.stack([s for _, s in lst])         # (forecasts, regions, terms)
        res[f"days_h{h}_{thr:g}"] = days; res[f"sums_h{h}_{thr:g}"] = S
    np.savez(HERE / "data" / f"hidden_warned_{ev.name}.npz", regions=np.array(R), terms=np.array(TERMS), thrs=np.array(THRS), **res)
    print("saved", HERE / "data" / f"hidden_warned_{ev.name}.npz")
    table(res, R)


def boot(S, num, den, B=2000, seed=0):
    """share num/den over the target days, with a 95 % interval from resampling the target days"""
    rng = np.random.default_rng(seed); n = len(S)
    v = S[:, num].sum() / max(S[:, den].sum(), 1e-9)
    bs = [S[i, num].sum() / max(S[i, den].sum(), 1e-9) for i in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * v, 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def table(res, R):
    T = list(TERMS)
    for thr in THRS:
        for h in (1, 3):
            key = f"sums_h{h}_{thr:g}"
            if key not in res:
                continue
            S = res[key]; nf = len(S)
            print(f"\n=== > {thr:g} ug m-3, {h} day{'s' if h > 1 else ''} ahead ({nf} target days; populated pixels, >= 100 residents)")
            print(f"  {'region':15s} {'hidden M/day':>12s} {'warned of hidden':>22s} {'false alarms (all)':>22s} {'false alarms (40 km silent)':>28s} {'warned of exposed':>18s}")
            for g, r_ in enumerate(R):
                Sg = S[:, g, :]
                hid = Sg[:, T.index("hid")].sum() / nf / 1e6
                if hid < 0.05:
                    continue
                a = boot(Sg, T.index("hid_w"), T.index("hid")); b = boot(Sg, T.index("warn_fa"), T.index("warn"))
                c = boot(Sg, T.index("warn_c_fa"), T.index("warn_c")); e = 100 * Sg[:, T.index("exp_w")].sum() / Sg[:, T.index("exp")].sum()
                print(f"  {r_:15s} {hid:12.1f} {a[0]:7.1f}% [{a[1]:4.1f}, {a[2]:4.1f}] {b[0]:7.1f}% [{b[1]:4.1f}, {b[2]:4.1f}] "
                      f"{c[0]:10.1f}% [{c[1]:4.1f}, {c[2]:4.1f}] {e:14.1f}%")


if __name__ == "__main__":
    main()
