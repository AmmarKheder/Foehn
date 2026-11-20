#!/usr/bin/env python3
"""WP1a check of step A: a replayed coarse forecast against the archived production forecast of the same day.

  python -I check_coarse.py <replayed day dir> [<day>]   -> prints and returns a dict; --json FILE appends it
Compares <dir>/pred.npy with RT/out/<day>/pred.npy (members x leads x 361 x 900, float16 ug m-3): share of identical
values, absolute difference (max, 99.9th percentile, mean), correlation, per member and for the ensemble mean / spread;
and, when <dir>/green is a real directory, its three stores with RT/data/<day>/green (same seeds, so FLEXGREEN should
reproduce the particle origins up to GPU arithmetic)."""
import argparse
import json
from pathlib import Path

import numpy as np

RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")


def cmp(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[ok] - b[ok])
    return dict(n=int(ok.sum()), identical=float((a[ok] == b[ok]).mean()), max_abs=float(d.max()), p999_abs=float(np.percentile(d, 99.9)),
                mean_abs=float(d.mean()), rmsd=float(np.sqrt((d ** 2).mean())), corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                mean_a=float(a[ok].mean()), mean_b=float(b[ok].mean()), nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("day", nargs="?")
    ap.add_argument("--json")
    a = ap.parse_args()
    D = Path(a.dir)
    day = a.day or D.name
    new = np.load(D / "pred.npy")
    old = np.load(RT / "out" / day / "pred.npy")
    assert new.shape == old.shape == (8, 10, 361, 900), (new.shape, old.shape)
    out = dict(day=day, dir=str(D), archived=str(RT / "out" / day / "pred.npy"), all=cmp(new, old),
               ens_mean=cmp(new.astype(np.float32).mean(0), old.astype(np.float32).mean(0)),
               ens_sd=cmp(new.astype(np.float32).std(0), old.astype(np.float32).std(0)),
               by_lead_max_abs=[float(np.abs(new[:, k].astype(np.float32) - old[:, k].astype(np.float32)).max()) for k in range(10)],
               by_member_corr=[float(np.corrcoef(new[m].astype(np.float32).ravel(), old[m].astype(np.float32).ravel())[0, 1]) for m in range(8)],
               member_spread_reference=float(old.astype(np.float32).std(0).mean()))
    g = D / "green"
    out["green"] = "symlink to the archived store" if g.is_symlink() else {}
    if g.is_dir() and not g.is_symlink():
        for f in sorted(g.glob("*.npy")):
            ref = RT / "data" / day / "green" / f.name
            if ref.exists():
                x, y = np.load(f, mmap_mode="r"), np.load(ref, mmap_mode="r")
                out["green"][f.name] = cmp(np.asarray(x), np.asarray(y)) if x.shape == y.shape else f"shape {x.shape} vs {y.shape}"
    print(json.dumps(out, indent=1))
    if a.json:
        p = Path(a.json)
        allr = json.loads(p.read_text()) if p.exists() else []
        allr.append(out)
        p.write_text(json.dumps(allr, indent=1))


if __name__ == "__main__":
    main()
