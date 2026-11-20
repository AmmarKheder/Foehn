"""cdffm_aq_x.report_x -- results table of the smoke comparison (pointwise, block and spectral scores) with paired
day-block bootstrap intervals against a reference run.

  python -m cdffm_aq_x.report_x --ref base_s0=runs/s2_europe/eval_x_last_M8_s10 \
      base_s1=runs/x/base_s1/eval_x_last_M8_s10 hcf_s0=runs/x/hcf_s0/eval_x_last_M8_s10 ...
Each argument is label=path to an eval_x (or eval_aq) directory holding metrics.json (+ psd.npz).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    p = Path(path)
    m = json.loads((p / "metrics.json").read_text())
    out = {"metrics": m, "psd": None}
    if (p / "psd.npz").exists():
        z = np.load(p / "psd.npz")
        out["psd"] = {"k": z["k_cycles_per_km"], "psd": z["psd"], "fields": list(z["fields"])}
    return out


def psd_ratio(r, lo_km, hi_km, field="member0"):
    if r["psd"] is None:
        return float("nan")
    k, psd, f = r["psd"]["k"], r["psd"]["psd"], r["psd"]["fields"]
    lam = 1.0 / k
    sel = (lam >= lo_km) & (lam <= hi_km)
    i, j = f.index(field), f.index("truth")
    num, den = psd[i][sel], psd[j][sel]
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return float(np.exp(np.mean(np.log(num[ok] / den[ok])))) if ok.any() else float("nan")


def daily(r, key, system="model", scale=None, space="ug"):
    rows = r["metrics"]["daily"]
    if scale is None:
        return np.array([row[system][key] for row in rows]), [row["date"] for row in rows]
    return np.array([row[system]["blocks"][str(scale)][space][key] for row in rows]), [row["date"] for row in rows]


def boot_mean(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=eval_dir")
    ap.add_argument("--ref", default=None, help="label of the reference run (paired differences)")
    ap.add_argument("--out", default=None, help="write the markdown table here too")
    a = ap.parse_args(argv)
    R = {}
    for s in a.runs:
        lab, path = s.split("=", 1)
        try:
            R[lab] = load(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {lab}: {e}")
    lines = []
    hdr = ("| run | CRPS | RMSE_M | ssF | bias | CRPS log | blk 0.25° CRPS/ssF | blk 0.5° CRPS/ssF | blk 1° CRPS/ssF "
           "| PSD mem/truth 10-140 km | 2-5 km |")
    lines += [hdr, "|" + "---|" * 11]
    for lab, r in R.items():
        m = r["metrics"]["scores"]["model"]
        u, lg = m["ug"], m["log"]
        b = r["metrics"].get("blocks", {}).get("model")
        bl = lambda s: (f"{b[s]['ug']['crps_fair']:.3f}/{b[s]['ug'].get('spread_skill_fortin', float('nan')):.2f}"
                        if b else "-")
        lines.append(f"| {lab} | {u['crps_fair']:.3f} | {u['rmse_ensmean']:.3f} | {u.get('spread_skill_fortin', float('nan')):.2f} "
                     f"| {u['bias']:+.3f} | {lg['crps_fair']:.4f} | {bl('1')} | {bl('2')} | {bl('4')} "
                     f"| {psd_ratio(r, 10, 140):.2f} | {psd_ratio(r, 2, 5):.2f} |")
    # the baselines inside the reference run
    if a.ref and a.ref in R:
        for sysn in ("climatology_ens", "cams_bilinear"):
            m = R[a.ref]["metrics"]["scores"][sysn]
            b = R[a.ref]["metrics"].get("blocks", {}).get(sysn)
            u = m["ug"]
            bl = lambda s: (f"{b[s]['ug']['crps_fair']:.3f}/{b[s]['ug'].get('spread_skill_fortin', float('nan')):.2f}"
                            if b else "-")
            lines.append(f"| ({sysn}) | {u['crps_fair']:.3f} | {u['rmse_ensmean']:.3f} | {u.get('spread_skill_fortin', float('nan')):.2f} "
                         f"| {u['bias']:+.3f} | {m['log']['crps_fair']:.4f} | {bl('1')} | {bl('2')} | {bl('4')} | - | - |")
    if a.ref and a.ref in R:
        ref = R[a.ref]
        lines += ["", f"Paired daily differences vs {a.ref} (mean [95 % day-block bootstrap]; negative = better):", "",
                  "| run | dCRPS | dRMSE_M | dCRPS log | d blk 1° CRPS | d blk 0.25° CRPS | days better (CRPS) |",
                  "|---|---|---|---|---|---|---|"]
        for lab, r in R.items():
            if lab == a.ref:
                continue
            try:
                c0, d0 = daily(ref, "crps_fair")
                c1, d1 = daily(r, "crps_fair")
                assert d0 == d1, "different day lists"
                dc = boot_mean(c1 - c0)
                dr = boot_mean(daily(r, "rmse_ensmean")[0] - daily(ref, "rmse_ensmean")[0])
                dl = boot_mean(daily(r, "crps_fair_log")[0] - daily(ref, "crps_fair_log")[0])
                try:
                    db4 = boot_mean(daily(r, "crps_fair", scale=4)[0] - daily(ref, "crps_fair", scale=4)[0])
                    db1 = boot_mean(daily(r, "crps_fair", scale=1)[0] - daily(ref, "crps_fair", scale=1)[0])
                    s4 = f"{db4[0]:+.3f} [{db4[1]:+.3f}, {db4[2]:+.3f}]"
                    s1 = f"{db1[0]:+.3f} [{db1[1]:+.3f}, {db1[2]:+.3f}]"
                except Exception:  # noqa: BLE001
                    s4 = s1 = "-"
                lines.append(f"| {lab} | {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}] | {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}] "
                             f"| {dl[0]:+.4f} [{dl[1]:+.4f}, {dl[2]:+.4f}] | {s4} | {s1} | {int((c1 < c0).sum())}/{len(c0)} |")
            except Exception as e:  # noqa: BLE001
                lines.append(f"| {lab} | error: {e} |")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
