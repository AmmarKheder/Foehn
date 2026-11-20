#!/usr/bin/env python3
"""calib.select_cal -- stage 1 of PROTOCOL.md A6(f), CPU only (no torch): reads the V100k validation of every saved
checkpoint of every arm (runs/<arm>/metrics.jsonl), the frozen system's V100k reference (results/d0_frozen.json) and,
if present, the temperature grid (results/tgrid_frozen.json), applies the pre-stated rule and writes results/stage1.json.

  admissible  CRPS <= C0 and RMSE <= 1.02 R0 (V100k, ug m-3) and hot-spot count <= 3,534 (G0a would fail for certain)
  order       admissible with spread/skill in [0.9, 1.1] by increasing CRPS, then the other admissible ones by
              increasing |spread/skill - 1|

  python select_cal.py [--arms V1,V2,V3,V4,A1]
"""
import argparse
import json
from pathlib import Path

CAL = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal")
HOT_MAX = 3534


def rule(cands, C0, R0):
    for c in cands:
        c["admissible"] = bool(c["crps"] <= C0 and c["rmse"] <= 1.02 * R0 and c["hot"] <= HOT_MAX)
        c["why_not"] = [w for w, bad in (("crps", c["crps"] > C0), ("rmse", c["rmse"] > 1.02 * R0), ("hot", c["hot"] > HOT_MAX)) if bad]
        c["in_band"] = bool(0.9 <= c["ssF"] <= 1.1)
    adm = [c for c in cands if c["admissible"]]
    order = sorted([c for c in adm if c["in_band"]], key=lambda c: c["crps"]) + \
        sorted([c for c in adm if not c["in_band"]], key=lambda c: abs(c["ssF"] - 1.0))
    for i, c in enumerate(order):
        c["stage1_rank"] = i + 1
    return order


def fmt(c):
    return (f"{c['name']:>12s} crps {c['crps']:.4f} ({100 * (c['crps'] / c['C0'] - 1):+.2f} %) rmse {c['rmse']:.4f} "
            f"({100 * (c['rmse'] / c['R0'] - 1):+.2f} %) ssF {c['ssF']:.4f} hot {c['hot']:.0f} D {c['D']:.2f} bias {c['bias']:+.3f} "
            f"| ins2020 ssF {c.get('ins_ssF', float('nan')):.3f} | {'ADMISSIBLE' if c['admissible'] else 'no: ' + ','.join(c['why_not'])}"
            f"{' IN BAND' if c['in_band'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="V1,V2,V3,V4,A1")
    ap.add_argument("--ref", default=str(CAL / "results" / "d0_frozen.json"))
    ap.add_argument("--tgrid", default=str(CAL / "results" / "tgrid_frozen.json"))
    ap.add_argument("--out", default=str(CAL / "results" / "stage1.json"))
    a = ap.parse_args()
    ref = json.loads(Path(a.ref).read_text())["results"]
    C0, R0, S0 = (ref["sel"]["ug"][k] for k in ("crps_fair", "rmse_ensmean", "spread_skill_fortin"))
    hot0 = ref["probe"]["sanity"]["n_pix_gt_TG"]
    print(f"reference (frozen system, V100k): C0 {C0:.4f} R0 {R0:.4f} S0 {S0:.4f} hot-spot count {hot0:.0f}; "
          f"limits: crps <= {C0:.4f}, rmse <= {1.02 * R0:.4f}, hot <= {HOT_MAX}")
    cands = []
    for arm in a.arms.split(","):
        f = CAL / "runs" / arm / "metrics.jsonl"
        if not f.exists():
            print(f"  (arm {arm}: no metrics.jsonl)")
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            u = r["sets"]["sel"]["ug"]
            c = {"name": f"{arm}@{r['step']}", "arm": arm, "step": r["step"], "crps": u["crps_fair"],
                 "rmse": u["rmse_ensmean"], "ssF": u["spread_skill_fortin"], "bias": u["bias"],
                 "D": u["D_predictive_mean_sq_error"], "hot": r["sets"]["probe"]["sanity"]["n_pix_gt_TG"],
                 "log_ssF": r["sets"]["sel"]["log"]["spread_skill_fortin"], "z_ssF": r["sets"]["sel"]["z"]["spread_skill_fortin"],
                 "C0": C0, "R0": R0}
            if "ins2020" in r["sets"]:
                c["ins_ssF"] = r["sets"]["ins2020"]["ug"]["spread_skill_fortin"]
                c["ins_crps"] = r["sets"]["ins2020"]["ug"]["crps_fair"]
            cands.append(c)
    order = rule(cands, C0, R0)
    print(f"\n{len(cands)} fine-tuned checkpoints, {len(order)} admissible")
    for c in cands:
        print("  " + fmt(c))
    print("\nstage-1 order (first = first to be screened):")
    for c in order[:12]:
        print(f"  {c['stage1_rank']:2d}. " + fmt(c))
    out = {"reference": {"C0": C0, "R0": R0, "S0": S0, "hot0": hot0, "hot_max": HOT_MAX}, "candidates": cands,
           "order": [c["name"] for c in order]}
    tg = Path(a.tgrid)
    if tg.exists():
        t = json.loads(tg.read_text())
        tc = []
        for tau in t["taus"]:
            u = t["results"][f"sel_tau{tau:.2f}"]["ug"]
            tc.append({"name": f"T@{tau:.2f}", "arm": "T", "tau": tau, "crps": u["crps_fair"], "rmse": u["rmse_ensmean"],
                       "ssF": u["spread_skill_fortin"], "bias": u["bias"], "D": u["D_predictive_mean_sq_error"],
                       "hot": t["results"][f"probe_tau{tau:.2f}"]["sanity"]["n_pix_gt_TG"], "C0": C0, "R0": R0})
        to = rule(tc, C0, R0)
        print(f"\ntemperature grid: {len(tc)} values, {len(to)} admissible")
        for c in tc:
            print("  " + fmt(c))
        out["temperature"] = {"candidates": tc, "order": [c["name"] for c in to], "T_star": to[0]["tau"] if to else None}
        print("T* =", out["temperature"]["T_star"])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
