#!/usr/bin/env python3
"""Task 2: does any quantity drawn or quoted for the ensemble figure (figS_rt_ensemble.py, numbers_cal.json, key f_ensemble)
depend on the CAMS arm?

Method: (1) the station-day tables of score_cal (CAMS = three instants) and of score_cams3h (CAMS = 3-hourly mean) are
compared row by row; (2) the ensemble block is recomputed WITH THE LINES OF rt_skill_data.py and the scorer's own code from
the table of score_cal in which the CAMS arm is REPLACED IN MEMORY by the 3-hourly arm, and compared with numbers_cal.json;
(3) the contrasts of the calibrated ensemble against the ensemble as issued (CRPS, RMSE) are recomputed in the same way.
Read-only: nothing is written (results on stdout; the optional --json PATH writes them to PATH).
    python -B -W ignore check_ensemble_cams.py        (login node, about 15 s, 3 GB)"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
R = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay")
F = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/figures_new/fig_rt_skill")
sys.path.insert(0, str(R / "score"))
import sc_lib as L  # noqa: E402
import score as SC  # noqa: E402

CLEAN = ("2026-09-06", "2026-11-11")
t0 = time.time()
out = {}


def say(name, ok, extra=""):
    out[name] = dict(ok=bool(ok), detail=str(extra))
    print(f"[{time.time() - t0:5.1f} s] {'OK  ' if ok else 'FAIL'} {name} {extra}", flush=True)


# ---------------------------------------------------------------- (1) the two station-day tables
zc = np.load(R / "score_cal/work/pairs.npz")
z3 = np.load(R / "score_cams3h/work/pairs.npz")
same_rows = all(np.array_equal(zc[k], z3[k]) for k in ("issue", "lead", "mon", "vday")) and np.array_equal(zc["id"], z3["id"])
say("same rows (issue day, lead day, monitor, valid day) in score_cal and score_cams3h", same_rows, f"{zc['obs'].size:,} rows")
say("same observations", np.array_equal(zc["obs"], z3["obs"], equal_nan=True))
say("arm cams of score_cal = three-instant arm (cams_old of score_cams3h)", np.array_equal(zc["m_cams"], z3["m_cams_old"], equal_nan=True),
    f"largest difference {np.nanmax(np.abs(zc['m_cams'].astype(np.float64) - z3['m_cams_old'].astype(np.float64))):.3g}")
f_old, f_new = np.isfinite(zc["m_cams"]), np.isfinite(z3["m_cams"])
say("the 3-hourly arm is finite on exactly the rows where the three-instant arm is", np.array_equal(f_old, f_new),
    f"finite: three instants {int(f_old.sum()):,}, 3-hourly {int(f_new.sum()):,} of {f_old.size:,}")
dif = np.abs(z3["m_cams"].astype(np.float64) - zc["m_cams"].astype(np.float64))
print(f"          (the values do differ: mean |3-hourly - three instants| = {np.nanmean(dif):.3f} ug m-3, largest {np.nanmax(dif):.1f})")
for a in ("km", "kmcal", "rt04", "clim", "bcsd"):
    if f"m_{a}" in z3.files:
        say(f"arm {a}: identical in the two tables", np.array_equal(zc[f"m_{a}"], z3[f"m_{a}"], equal_nan=True))
cams3h = z3["m_cams"].astype(np.float64)
del z3

# ---------------------------------------------------------------- (2) the ensemble block, with the CAMS arm replaced
num = json.loads((F / "numbers_cal.json").read_text())
Fz = num["f_ensemble"]
A, A2 = num["arm"], num["arm2"]
D = L.Data(str(R / "score_cal/work/pairs.npz"))
B5 = [A, A2, "rt04", "cams", "clim", "bcsd"]                      # focus + base, as rt_skill_data.py builds it for score_cal
mask_old = D.finite(B5) & D.issues(*CLEAN)
D.m["cams"] = cams3h                                              # <- the only change: the CAMS arm is the 3-hourly mean
mask = D.finite(B5) & D.issues(*CLEAN)
say("primary set unchanged by the CAMS reading (same station-days)", np.array_equal(mask, mask_old), f"{int(mask.sum()):,} station-days")
smp = L.Sample(D, mask, B=2000)
print("          " + smp.describe(), flush=True)
c_new = SC.contrast(smp, A, "cams", "rmse")["all"]["all"]
say("control: with the replaced arm the contrast 1 km vs CAMS is the 3-hourly one (-14.37 [-20.73, -7.19])",
    abs(c_new["est"] + 14.36829) < 1e-4 and abs(c_new["ci"][0] + 20.72688) < 1e-4 and abs(c_new["ci"][1] + 7.18902) < 1e-4,
    f"{c_new['est']:.4f} [{c_new['ci'][0]:.4f}, {c_new['ci'][1]:.4f}]; numbers_cal.json holds {num['legend_numbers']['change_pooled']['cams']['est']:.4f}")
ens_arms = [A, A2]
E = SC.ensemble(smp, ens_arms, [])
ec = lambda d: dict(est=float(d["est"]), ci=[float(d["ci"][0]), float(d["ci"][1])])  # noqa: E731
close = lambda x, y: bool(np.allclose(np.asarray(x, float), np.asarray(y, float), rtol=0, atol=1e-9))  # noqa: E731
spread_bins = None
table = []
for x in ens_arms:
    e_ = E[x]["all|all"]
    st = D.ens_stats(x)
    M = st["M"]
    f_ = (M + 1) / M
    var = st["var"][smp.rows]
    sq = (D.m[x][smp.rows] - D.obs[smp.rows]) ** 2
    sp = np.sqrt(f_ * var)
    if spread_bins is None:
        lo_, hi_ = np.percentile(sp, [1.0, 99.9])
        spread_bins = 2.0 ** (np.arange(np.floor(2 * np.log2(lo_)), np.ceil(2 * np.log2(hi_)) + 1) / 2)
    edges = np.concatenate([[0.0], spread_bins, [np.inf]])
    k = np.digitize(sp, edges) - 1
    nb = edges.size - 1
    nk = np.bincount(k, minlength=nb)
    keep = nk >= 5000
    mine = dict(rank_hist_pct=[100.0 * float(v) for v in e_["rank_hist"]], outside_pct=float(e_["outside"]), coverage_10_90_pct=ec(e_["cover"]),
                below_pct=float(e_["below"]), above_pct=float(e_["above"]), spread_skill=ec(e_["spread_skill"]), fair_crps=ec(e_["crps"]),
                spread=float(e_["spread"]), rmse=float(e_["rmse"]),
                spread_classes=dict(spread=np.sqrt(f_ * np.bincount(k, weights=var, minlength=nb)[keep] / nk[keep]).tolist(),
                                    rmse=np.sqrt(np.bincount(k, weights=sq, minlength=nb)[keep] / nk[keep]).tolist(), station_days=nk[keep].tolist()))
    ref = Fz[x]
    for key in ("rank_hist_pct", "outside_pct", "below_pct", "above_pct", "spread", "rmse"):
        say(f"{x}: {key} identical to numbers_cal.json", close(mine[key], ref[key]))
    for key in ("coverage_10_90_pct", "spread_skill", "fair_crps"):
        say(f"{x}: {key} (estimate and interval) identical to numbers_cal.json", close([mine[key]["est"]] + mine[key]["ci"], [ref[key]["est"]] + ref[key]["ci"]),
            f"{mine[key]['est']:.4f} [{mine[key]['ci'][0]:.4f}, {mine[key]['ci'][1]:.4f}]")
    for key in ("spread", "rmse", "station_days"):
        say(f"{x}: spread classes, {key} identical to numbers_cal.json", close(mine["spread_classes"][key], ref["spread_classes"][key]), f"{len(mine['spread_classes'][key])} classes")
    table.append((x, mine))
for x in ens_arms:
    for key, nm in (("spread_skill", "spread_skill"), ("cover", "coverage_10_90_pct")):
        mine = [ec(E[x][f"{k}|all"][key]) for k in range(1, 6)]
        ref = Fz["by_lead"][x][nm]
        say(f"{x}: {nm} by lead day identical to numbers_cal.json", all(close([a["est"]] + a["ci"], [b["est"]] + b["ci"]) for a, b in zip(mine, ref)))

# ---------------------------------------------------------------- (3) calibrated against as issued: CRPS and RMSE
T03 = json.loads((R / "score_cal/tables/T03_contrasts.json").read_text())["with the CAMS forecast"]["contrasts"]
for metric in ("crps", "rmse"):
    c = SC.contrast(smp, A2, A, metric)["all"]
    key = f"{A2}_vs_{A}|{metric}"
    if key in T03:
        r = T03[key]["all"]
        say(f"{A2} vs {A}, {metric}: pooled and mean over regions identical to score_cal T03", close([c["all"]["est"]] + c["all"]["ci"], [r["all"]["est"]] + r["all"]["ci"])
            and close([c["regmean"]["est"]] + c["regmean"]["ci"], [r["regmean"]["est"]] + r["regmean"]["ci"]),
            f"pooled {c['all']['est']:+.2f} [{c['all']['ci'][0]:+.2f}, {c['all']['ci'][1]:+.2f}], regions {c['regmean']['est']:+.2f} [{c['regmean']['ci'][0]:+.2f}, {c['regmean']['ci'][1]:+.2f}]")
    else:
        print(f"          {key} not in score_cal T03 (keys: {[k for k in T03 if k.startswith(A2)]}); recomputed: pooled {c['all']['est']:+.2f} [{c['all']['ci'][0]:+.2f}, {c['all']['ci'][1]:+.2f}]")
# ---------------------------------------------------------------- what in numbers_cal.json DOES depend on the CAMS arm
old = json.loads((F / "numbers_3instants.json").read_text())


def leaves(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, d


new = json.loads((F / "numbers.json").read_text())
lo, ln = dict(leaves(old)), dict(leaves(new))
changed = sorted(k for k in lo if k in ln and lo[k] != ln[k] and k not in ("/made", "/scorer_directory") and not k.startswith("/sources/"))
lc = dict(leaves(num))
stale = [k for k in changed if k in lc and lc[k] == lo[k]]
print(f"entries of numbers.json that changed with the CAMS reading: {len(changed)}; of these, present with the three-instant value in numbers_cal.json: {len(stale)}")
blocks = {}
for k in stale:
    blocks[k.split('/')[1]] = blocks.get(k.split('/')[1], 0) + 1
print("          by block:", blocks)
print("          in f_ensemble:", [k for k in stale if k.startswith("/f_ensemble")])
out["stale_cams_entries_in_numbers_cal"] = dict(n=len(stale), by_block=blocks, in_f_ensemble=[k for k in stale if k.startswith("/f_ensemble")])
drawn = ["arms", "expected/rank_share_pct", "<arm>/rank_hist_pct", "<arm>/spread_classes/spread", "<arm>/spread_classes/rmse"]
print("keys of f_ensemble read by figS_rt_ensemble.py:", drawn)
ok_all = all(v["ok"] for v in out.values() if isinstance(v, dict) and "ok" in v)
print("ALL OK" if ok_all else "SOME CHECK FAILED")
for x, m in table:
    print(f"   {x}: outside {m['outside_pct']:.2f} %, 10-90 % range {m['coverage_10_90_pct']['est']:.2f} %, spread/skill {m['spread_skill']['est']:.3f}, fair CRPS {m['fair_crps']['est']:.3f}, RMSE {m['rmse']:.3f}")
if len(sys.argv) > 2 and sys.argv[1] == "--json":
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
