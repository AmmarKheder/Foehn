#!/usr/bin/env python3
"""Checkpoint of M and blending radius R for the final test, from the 2021 validation ONLY (val_lag tables of pmM_base:
runs/pmM_base/val_lag/sel2021_all.json; truth = the 1 km estimates of 2021; no monitor value of any year is read here).

Why a rule is needed: A11 section 4 selects stage-2 RECIPES (chain-mode CRPS, 2 % gate on the frozen model, member
amplitude >= 0.8); for M no chain-mode score and no amplitude ratio exist, and the tested system is a BLEND (M near the
monitors, the station-free member elsewhere), which section 4 does not address. The rule proposed in a11/stageM/README.md
section 10.2 (all-pixel CRPS of M at every lag not above pmG_final, M|nostations within 2 % of pmG_final) is met by NO
checkpoint (it is printed below). So ONE rule is proposed here:

RULE S. Over the 12 validated checkpoints of pmM_base and R in {10, 50, 150 km, everywhere} (the class bounds of the
2021 tables), the blended system takes M's members at the pixels whose nearest assimilated monitor with a valid value
on day t - lag is within R, and pmG_final's elsewhere. Its 2021 fair CRPS at lag l, all pixels, is the weight-averaged
CRPS of the classes (class weights n of the tables): sum_k n_k CRPS_k / sum_k n_k, with CRPS_k of M for the classes
within R and of pmG_final beyond. Choose the (checkpoint, R) with the lowest unweighted mean over the lags 1..5; accept
it only if its blended CRPS is not above pmG_final's at any lag 1..5. No tie rule (plain minimum). The no-monitor state
of M never enters the blend (it is only reached beyond 300 km), so the |nostations rows do not enter the rule.

    python -I -B select_2021.py       -> data/selection.json (and the tables on stdout)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_lib as F  # noqa: E402

CLS = ["0-10km", "10-50km", "50-150km", ">150km"]
RS = [(10.0, 1), (50.0, 2), (150.0, 3), (None, 4)]          # R, number of classes inside R (None = M everywhere)
LAGS = ["1", "2", "3", "4", "5"]


def blended(s, arm, free, lag, nin, metric):
    num = den = 0.0
    for i, k in enumerate(CLS):
        w = s[arm][lag][k]["n"]
        v = (s[arm] if i < nin else s[free])[lag][k][metric]
        num += w * (v if metric == "crps_fair" else v * v)
        den += w
    return num / den if metric == "crps_fair" else float(np.sqrt(num / den))


def main():
    j = json.loads(F.SEL2021.read_text())
    s = j["scores"]
    assert j["info"]["days"][0].startswith("2021") and all(d.startswith("2021") for d in j["info"]["days"]), "2021 only"
    steps = sorted(int(k[2:]) for k in s if k.startswith("M@") and "|" not in k)
    G = "pmG_final"
    gref = {lg: s[G][lg]["all"]["crps_fair"] for lg in LAGS}
    grm = {lg: s[G][lg]["all"]["rmse_ensmean"] for lg in LAGS}
    cand = []
    for st in steps:
        arm = f"M@{st}"
        for R, nin in RS:
            c = {lg: blended(s, arm, G, lg, nin, "crps_fair") for lg in LAGS}
            r = {lg: blended(s, arm, G, lg, nin, "rmse_ensmean") for lg in LAGS}
            rel = [100 * (c[lg] / gref[lg] - 1) for lg in LAGS]
            relr = [100 * (r[lg] / grm[lg] - 1) for lg in LAGS]
            cand.append(dict(step=st, R_km=R, crps=c, rmse=r, crps_rel_pct=rel, rmse_rel_pct=relr,
                             mean_crps=float(np.mean([c[lg] for lg in LAGS])), accepted=bool(max(rel) <= 0.0)))
    acc = [x for x in cand if x["accepted"]]
    if not acc:
        raise SystemExit("RULE S: no (checkpoint, R) is accepted: the final test is not run")
    best = min(acc, key=lambda x: x["mean_crps"])
    # README 10.2 rule, for the record (not used): all-pixel CRPS <= pmG at every lag and |nostations within 2 %
    r102 = []
    for st in steps:
        arm = f"M@{st}"
        ok1 = all(s[arm][lg]["all"]["crps_fair"] <= gref[lg] for lg in LAGS)
        ns = s[arm + "|nostations"]["1"]["all"]["crps_fair"]
        r102.append(dict(step=st, crps_all_not_above_pmG_every_lag=ok1, nostations_rel_pct=100 * (ns / gref["1"] - 1),
                         passes=bool(ok1 and ns <= 1.02 * gref["1"])))
    arm = f"M@{best['step']}"
    per_class = {lg: {k: dict(M=s[arm][lg][k]["crps_fair"], pmG=s[G][lg][k]["crps_fair"],
                              crps_rel_pct=100 * (s[arm][lg][k]["crps_fair"] / s[G][lg][k]["crps_fair"] - 1),
                              rmse_rel_pct=100 * (s[arm][lg][k]["rmse_ensmean"] / s[G][lg][k]["rmse_ensmean"] - 1),
                              weight=s[arm][lg][k]["n"]) for k in CLS} for lg in LAGS}
    ck = F.RUN_M / "ckpt" / f"step{best['step']:06d}.ckpt"
    args = json.loads((F.RUN_M / "args.json").read_text())
    unseen = dict(split=args["split"], store=args["store"], tune_stations=args["tune_stations"], init_from=args["init_from"],
                  resume=args["resume"], stn_lag=args["stn_lag"], stn_lag0_prob=args["stn_lag0_prob"],
                  stations=args["stations"], val_days=args["val_days"],
                  reading="split G = training days 2017-01-01..2020-12-31 (GhapStore asserts year <= 2020), validation = the "
                          "24 SEL days of 2021; tune_stations null (the A4 fine-tuning on 2022-2026 monitors is off; the "
                          "tune_* entries are inactive defaults); init_from null and resume false (from scratch); the training "
                          "log (logs/train_a11M_train_23893125.out) loads assimilation years 2017-2021 only (2021 for the "
                          "in-training validation); no file of 2022-2026 was opened. Nothing of 2026 was seen by M.")
    sel = dict(rule="RULE S (select_2021.py docstring), 2021 validation against the 1 km estimates only",
               source=str(F.SEL2021), source_sha256=F.sha256_file(F.SEL2021),
               checkpoint=dict(step=best["step"], path=str(ck), sha256=F.sha256_file(ck), run=str(F.RUN_M)),
               R_km=best["R_km"], chosen=best, candidates=cand, readme_10_2_rule=r102, per_class_at_choice=per_class,
               M_unseen_2026=unseen, pmG_final_2021=dict(crps=gref, rmse=grm))
    F.save_json(F.SELECTION, sel)
    out = ["| checkpoint | R = 10 km | R = 50 km | R = 150 km | M everywhere |", "|---|---|---|---|---|"]
    for st in steps:
        row = [f"M@{st}"]
        for R, _ in RS:
            x = next(c for c in cand if c["step"] == st and c["R_km"] == R)
            row.append(f"{np.mean(x['crps_rel_pct']):+.3f} / {max(x['crps_rel_pct']):+.3f}")
        out.append("| " + " | ".join(row) + " |")
    print("2021 fair CRPS of the blend against pmG_final, all pixels, % (mean over lags 1-5 / worst lag):")
    print("\n".join(out))
    print(f"\nRULE S -> {arm}, R = {best['R_km']} km; blended CRPS by lag (%): {[round(v, 3) for v in best['crps_rel_pct']]}; "
          f"RMSE by lag (%): {[round(v, 3) for v in best['rmse_rel_pct']]}")
    print("README 10.2 rule passes for:", [x["step"] for x in r102 if x["passes"]] or "no checkpoint")
    for lg in LAGS:
        print(f"lag {lg}: " + "; ".join(f"{k} CRPS {v['M']:.3f} vs {v['pmG']:.3f} ({v['crps_rel_pct']:+.1f} %), RMSE {v['rmse_rel_pct']:+.1f} %"
                                     for k, v in per_class[lg].items()))
    print(f"-> {F.SELECTION}")


if __name__ == "__main__":
    main()
