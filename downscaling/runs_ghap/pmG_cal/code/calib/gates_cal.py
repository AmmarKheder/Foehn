#!/usr/bin/env python3
"""calib.gates_cal -- reads the VALIDATION_GATES.json that runs_global/validation_gates.py wrote for an A6 evaluation and
applies the pre-stated rules of PROTOCOL.md A6 (CPU, no torch):
  screen E1 E2   stage 2 (A6(f)): E1 = eval_x at every 8th mask pixel, E2 = eval_x on the regime boxes (spectra). PASS if
                 every gate that passes for the frozen system passes: G0a, G0c, G0d, G0e, G1, G3, G4 on E1 and G2 on E2.
  full E         A6(g) on the full selection evaluation E, against the frozen system's full evaluation: ACCEPTED if (i)
                 pixel spread/skill in [0.9, 1.1], (ii) CRPS not higher, (iii) RMSE of the ensemble mean <= 1.02 x, (iv) no
                 gate that passes today fails; "CLOSEST" if (ii)-(iv) hold and (i) does not; else NOT ACCEPTED.
The last printed line is the verdict.
"""
import json
import sys
from pathlib import Path

BASE = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_base/"
            "eval_sel2021_M8_s10_P3_guard_T3")
FINAL_GATES = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_final/"
                   "VALIDATION_GATES.json")
PASS_TODAY = ("G0a", "G0c", "G0d", "G0e", "G1", "G2", "G3", "G4")


def short(k):
    return k.split("_")[0]


def load(ed):
    ed = Path(ed)
    g = json.loads((ed / "VALIDATION_GATES.json").read_text())
    m = json.loads((ed / "metrics.json").read_text())
    return g, m


def line(m):
    u = m["scores"]["model"]["ug"]
    b = m["blocks"]["model"]["4"]["ug"]
    s = m["sanity"]
    return (f"crps {u['crps_fair']:.4f} rmseM {u['rmse_ensmean']:.4f} bias {u['bias']:+.3f} pixel ssF "
            f"{u['spread_skill_fortin']:.4f} | 1deg block ssF {b['spread_skill_fortin']:.4f} crps {b['crps_fair']:.4f} | "
            f"pix-days>TG {s['n_pix_gt_440']:.0f} ({s['frac_pix_days_any_member_gt_440']:.2e}) guard-bound "
            f"{s['frac_member_pixel_days_guard_bound']:.2e} (bound {s['n_member_at_bound']:.0f}, cap "
            f"{s['n_member_at_pmax']:.0f}) max {s['max_member_ug']:.0f} cells {s['n_cells_with_pix_gt_440']} | "
            f"{m['n_pixels_per_day']:,} px/day x {m['n_days']} days, {m['minutes']:.1f} min")


def main():
    mode = sys.argv[1]
    today = {short(k): v["pass"] for k, v in json.loads(FINAL_GATES.read_text())["gates"].items()}
    assert all(today[g] for g in PASS_TODAY) and today["G0b"] is False, today
    if mode == "screen":
        g1, m1 = load(sys.argv[2])
        g2, m2 = load(sys.argv[3])
        print("stride-8:", line(m1))
        gates = {short(k): v for k, v in g1["gates"].items()}
        gates["G2"] = {short(k): v for k, v in g2["gates"].items()}["G2"]
        fails = []
        for g in PASS_TODAY + ("G0b",):
            v = gates[g]
            val = v.get("value", v.get("model"))
            print(f"  {g}: {'pass' if v['pass'] else 'FAIL'} (value {val}, reference {v.get('reference', '-')})")
            if g in PASS_TODAY and not v["pass"]:
                fails.append(g)
        print("SCREEN PASS" if not fails else "SCREEN FAIL: " + ",".join(fails))
        return
    g, m = load(sys.argv[2])
    b = json.loads((BASE / "metrics.json").read_text())
    print("frozen :", line(b))
    print("system :", line(m))
    gates = {short(k): v for k, v in g["gates"].items()}
    fails = []
    for k in PASS_TODAY + ("G0b",):
        v = gates[k]
        val = v.get("value", v.get("model"))
        print(f"  {k}: {'pass' if v['pass'] else 'FAIL'} (value {val}, reference {v.get('reference', '-')})")
        if k in PASS_TODAY and not v["pass"]:
            fails.append(k)
    u, u0 = m["scores"]["model"]["ug"], b["scores"]["model"]["ug"]
    c2 = u["crps_fair"] <= u0["crps_fair"]
    c3 = u["rmse_ensmean"] <= 1.02 * u0["rmse_ensmean"]
    c1 = 0.9 <= u["spread_skill_fortin"] <= 1.1
    print(f"  (i) pixel ssF {u['spread_skill_fortin']:.4f} in [0.9, 1.1]: {c1}")
    print(f"  (ii) CRPS {u['crps_fair']:.4f} <= {u0['crps_fair']:.4f}: {c2} ({100 * (u['crps_fair'] / u0['crps_fair'] - 1):+.2f} %)")
    print(f"  (iii) RMSE {u['rmse_ensmean']:.4f} <= {1.02 * u0['rmse_ensmean']:.4f}: {c3} "
          f"({100 * (u['rmse_ensmean'] / u0['rmse_ensmean'] - 1):+.2f} %)")
    print(f"  (iv) gates passing today that fail: {fails or 'none'}")
    g0b, g0b0 = gates["G0b"]["value"], b["sanity"]["frac_member_pixel_days_guard_bound"]
    print(f"  G0b (fails today): {g0b:.3e} vs {g0b0:.3e}" + (" HIGHER (flag)" if g0b > g0b0 else ""))
    bad = [n for n, ok in (("CRPS", c2), ("RMSE", c3), ("gates " + ",".join(fails), not fails)) if not ok]
    if bad:
        print("NOT ACCEPTED: " + "; ".join(bad) + f" (pixel ssF {u['spread_skill_fortin']:.3f})")
    elif c1:
        print(f"ACCEPTED (pixel ssF {u['spread_skill_fortin']:.3f}, CRPS {100 * (u['crps_fair'] / u0['crps_fair'] - 1):+.2f} %, "
              f"RMSE {100 * (u['rmse_ensmean'] / u0['rmse_ensmean'] - 1):+.2f} %)")
    else:
        print(f"CLOSEST THE RULE ALLOWS (pixel ssF {u['spread_skill_fortin']:.3f} outside 0.9-1.1; CRPS "
              f"{100 * (u['crps_fair'] / u0['crps_fair'] - 1):+.2f} %, RMSE {100 * (u['rmse_ensmean'] / u0['rmse_ensmean'] - 1):+.2f} %)")


if __name__ == "__main__":
    main()
