#!/usr/bin/env python3
"""A8(e)-2(ii): merge the per-rank files of psdrepro.py and compare with the stored spectra of the frozen evaluation
(pmG_base/eval_sel2021_M8_s10_P3_guard_T3/psd.npz). numpy only (host python). Writes <dir>/psdrepro_compare.json.

  python psdrepro_compare.py --dir .../a8/results/psdrepro [--ref .../psd.npz]"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

REF = ("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_base/"
       "eval_sel2021_M8_s10_P3_guard_T3/psd.npz")


def g2(k, psd, fields, lo_km=10.0, hi_km=140.0, field="member0"):
    """runs_global/validation_gates.psd_ratio: geometric mean of member-0 / truth power over 10-140 km"""
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km)
    num, den = psd[fields.index(field)][sel], psd[fields.index("truth")][sel]
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return float(np.exp(np.mean(np.log(num[ok] / den[ok])))) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()
    files = sorted(glob.glob(str(Path(a.dir) / "psdrepro_r*.npz")))
    parts = [np.load(f) for f in files]
    fields = [str(f) for f in parts[0]["fields"]]
    k = parts[0]["k_cycles_per_km"]
    s = sum(p["psd_sum"] for p in parts)
    c = sum(p["psd_cnt"] for p in parts)
    nbd = int(sum(int(p["n_box_days"]) for p in parts))
    psd = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    days = sorted(int(d) for p in parts for d in p["days"])
    info = [r for p in parts for r in json.loads(str(p["info"]))]
    z = np.load(a.ref)
    rf = [str(f) for f in z["fields"]]
    out = {"files": files, "n_ranks": len(files), "days": days, "n_box_days": nbd, "ref": a.ref,
           "ref_n_box_days": float(z["n_box_days"]), "k_equal": bool(np.allclose(k, z["k_cycles_per_km"], rtol=1e-12)),
           "fields": {}, "info": sorted(info, key=lambda r: r["date"])}
    for f in fields:
        x, y = psd[fields.index(f)], z["psd"][rf.index(f)]
        ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
        rel = np.abs(x[ok] / y[ok] - 1.0)
        out["fields"][f] = {"n_bins": int(ok.sum()), "median_rel_diff": float(np.median(rel)), "max_rel_diff": float(rel.max()),
                            "nan_pattern_equal": bool((np.isfinite(x) == np.isfinite(y)).all())}
    out["G2_recomputed"] = g2(k, psd, fields)
    out["G2_stored"] = g2(z["k_cycles_per_km"], z["psd"], rf)
    out["G2_rel_diff"] = out["G2_recomputed"] / out["G2_stored"] - 1.0
    out["ensmean_ratio_recomputed"] = g2(k, psd, fields, field="ensmean")
    out["ensmean_ratio_stored"] = g2(z["k_cycles_per_km"], z["psd"], rf, field="ensmean")
    out["pass_A8e2ii"] = bool(abs(out["G2_rel_diff"]) <= 0.005 and out["fields"]["member0"]["median_rel_diff"] <= 0.005)
    (Path(a.dir) / "psdrepro_compare.json").write_text(json.dumps(out, indent=1))
    np.savez(Path(a.dir) / "psdrepro_merged.npz", k_cycles_per_km=k, fields=np.array(fields), psd=psd, n_box_days=nbd)
    print(f"[psdrepro] {len(files)} ranks, days {days}, box-days {nbd} (stored {out['ref_n_box_days']:.0f})")
    for f, r in out["fields"].items():
        print(f"[psdrepro] {f:14s} bins {r['n_bins']:2d} median |rel diff| {r['median_rel_diff']:.2e} max {r['max_rel_diff']:.2e}")
    print(f"[psdrepro] G2 (member 0 / truth, 10-140 km): recomputed {out['G2_recomputed']:.6f} stored {out['G2_stored']:.6f} "
          f"rel diff {out['G2_rel_diff']:+.2e} | pass {out['pass_A8e2ii']}")


if __name__ == "__main__":
    main()
