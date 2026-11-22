#!/usr/bin/env python3
"""A6(b), proof: the copy infer_points_cal.py run with the frozen settings (tau 1) against the existing replay files
of the frozen system: replay/km/<d>.npz (downscaler part), replay/raw/<d>.npy (members at every evaluated position) and
replay/points/<d>.npz (the deliverable). Bitwise comparison (NaN = NaN), CPU, numpy only; no observation is opened.
  python compare_points.py --mine runs_ghap/pmG_cal/replay_out/proof --list days.txt --out results/replay_proof.json"""
import argparse
import json
from pathlib import Path

import numpy as np

R0 = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
KEYS = ("km", "km_ring1", "km_ring3", "km_ring10", "km_cell", "km_cell_lin")


def cmp(a, b):
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"identical": bool(same.all()) and a.dtype == b.dtype and a.shape == b.shape, "n": int(a.size), "n_differ": int((~same).sum()),
            "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0, "n_nan": int(np.isnan(a).sum()), "n_nan_ref": int(np.isnan(b).sum()),
            "dtype": str(a.dtype), "shape": list(a.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="reference directory with km/ and raw/ (default: the replay directory, with points/ too)")
    a = ap.parse_args()
    mine = Path(a.mine)
    R = Path(a.ref) if a.ref else R0
    days = sorted(x for x in Path(a.list).read_text().split() if x.strip())
    res, ok_all = {}, True
    for d in days:
        r = {}
        f = mine / "km" / f"{d}.npz"
        if not f.exists():
            res[d] = {"error": "no file from the copy"}
            ok_all = False
            continue
        m, k = np.load(f), np.load(R / "km" / f"{d}.npz")
        p = np.load(R / "points" / f"{d}.npz") if (R / "points" / f"{d}.npz").exists() else None
        for key in KEYS:
            r[f"km/{key}"] = cmp(m[key], k[key])
            if p is not None:
                r[f"points/{key}"] = cmp(m[key], p[key])
        r["valid_equal"] = bool((m["valid"] == k["valid"]).all()) and (p is None or bool((m["valid"][m["valid"] != ""] == p["valid"][m["valid"] != ""]).all()))
        r["doy_equal"] = bool((m["doy"] == k["doy"]).all())
        fr = mine / "raw" / f"{d}.npy"
        if fr.exists() and (R / "raw" / f"{d}.npy").exists():
            r["raw"] = cmp(np.load(fr, mmap_mode="r")[:], np.load(R / "raw" / f"{d}.npy", mmap_mode="r")[:])
        mm, mk = json.loads(str(m["meta"])), json.loads(str(k["meta"]))
        r["meta_same"] = {x: mm.get(x) == mk.get(x) for x in ("ckpt_sha256", "prior_sha256", "guard", "ode_steps", "members", "seed", "step", "leads_made", "n_positions")}
        r["calibration"] = mm.get("calibration")
        r["devices"] = [mm.get("device"), mk.get("device")]
        ok = all(v["identical"] for kk, v in r.items() if isinstance(v, dict) and "identical" in v) and r["valid_equal"] and r["doy_equal"] and all(r["meta_same"].values())
        r["identical"] = ok
        ok_all &= ok
        res[d] = r
        worst = max((v["max_abs_diff"] for v in r.values() if isinstance(v, dict) and "max_abs_diff" in v), default=0.0)
        ndif = sum(v["n_differ"] for v in r.values() if isinstance(v, dict) and "n_differ" in v)
        nval = sum(v["n"] for v in r.values() if isinstance(v, dict) and "n" in v)
        print(f"{d}: {'IDENTICAL' if ok else 'DIFFERENT'} ({nval:,} values compared in {sum(1 for v in r.values() if isinstance(v, dict) and 'n' in v)} arrays, "
              f"{ndif} differ, max |diff| {worst:.3g}; raw {'yes' if 'raw' in r else 'no'})", flush=True)
    out = Path(a.out)
    out.write_text(json.dumps({"mine": str(mine), "reference": str(R), "days": res, "all_identical": ok_all}, indent=1))
    n_ok = sum(1 for v in res.values() if v.get("identical"))
    worst = max((v["max_abs_diff"] for r in res.values() for v in r.values() if isinstance(v, dict) and "max_abs_diff" in v), default=0.0)
    verdict = (f"{'PASS' if ok_all else 'NOT ALL IDENTICAL'}: {n_ok} of {len(days)} issue days bit-identical ({mine} against {R}: km, raw"
               f"{', points' if not a.ref else ''}; arrays {', '.join(KEYS)}); largest |difference| {worst:.3g} ug m-3 -> {out}")
    out.with_suffix(".txt").write_text(verdict + "\n")
    print(verdict)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
