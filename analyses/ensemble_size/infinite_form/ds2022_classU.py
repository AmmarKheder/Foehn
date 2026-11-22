#!/usr/bin/env python3
"""A14 class U for the test of the real-time downscaler on 2022 against the 1 km estimates (Methods and Supplementary Methods 'Tests of the downscaler':
RMSE of the ensemble mean 9.98 against 13.50 for the coarse field and 12.37 for the quantile mapping; fair CRPS 3.13 against
3.85 for the climatological ensemble; Supplementary Note 5: 'RMSE 26 % and 19 % lower', 'fair CRPS 19 % below').
Scorer test2022_ghap/score_2022.py with runs_global/score_2024.py (metrics, boot imported unchanged) on its stored per-day sums
test2022_ghap/results/accum_2022.npz (read only); identity with results/scores_ghap_2022.json. The stored 'var' is the member
variance with divisor M (score_2024.py l. 191), so the unbiased s2 = var M/(M - 1) and MSE_inf = (sum w e^2 - sum w var/(M-1))
/ sum w. The members are drawn with seeds keyed by the day (predict_2022.py l. 159): V_draw = 0 (draw resampled with the
week blocks). The 1 deg block RMSE has no stored block variance of the members: not computed.
    python ds2022_classU.py   -> results/ds2022_classU.json"""
import json
import sys

import numpy as np

from u_common import P, RES, safe_json

sys.path.insert(0, str(P / "runs_global"))
import score_2024 as S  # noqa: E402  (unchanged)

z = np.load(P / "test2022_ghap" / "results" / "accum_2022.npz")
D = z["D"]; ARMS = [str(a) for a in z["arms"]]; RN = [str(r) for r in z["regions"]]
ref = json.loads((P / "test2022_ghap" / "results" / "scores_ghap_2022.json").read_text())
M_of = ref["members"]
days = list(range(D.shape[0]))
wk = {}
for t in days:
    wk.setdefault(t // 7, []).append(t)
B = S.boot(D, [np.array(v) for v in wk.values()])
Ssum = D.sum(0)
pm = {a: S.metrics(Ssum[i], M_of[a]) for i, a in enumerate(ARMS)}
bm = {a: S.metrics(B[:, i], M_of[a]) for i, a in enumerate(ARMS)}
ident = {}
for i, a in enumerate(ARMS):
    for ri, r in enumerate(RN):
        for k in ("crps", "rmse", "bias"):
            ident[f"{a}|{r}|{k}"] = bool(abs(pm[a][k][ri] - ref["year"]["arms"][a][r][k]) < 1e-9)
for a in ARMS[1:]:
    for ri, r in enumerate(RN):
        for k in ("crps", "rmse"):
            dlt = bm["cdffm"][k][:, ri] - bm[a][k][:, ri]
            rr = ref["year"]["diff_cdffm_minus"][a][r][k]
            ident[f"diff {a}|{r}|{k}"] = bool(abs(np.percentile(dlt, 2.5) - rr["ci95"][0]) < 1e-9 and abs(np.percentile(dlt, 97.5) - rr["ci95"][1]) < 1e-9)


def rmse_form(Sx, M, form):
    w, se, var = Sx[..., 0], Sx[..., 2], Sx[..., 6]
    if M == 1 or form == "as_run":
        return np.sqrt(se / w)
    s2 = var * M / (M - 1)
    c8 = 1 / 8
    return np.sqrt((se - s2 / M + (c8 * s2 if form == "M8" else 0.0)) / w)


OUT = {"identity": ident, "identity_all": bool(all(ident.values())), "members": M_of, "v_draw": "0 (seeds keyed by the day)", "regions": {}}
for ri, r in enumerate(RN):
    o = {}
    for i, a in enumerate(ARMS):
        o[a] = {f: float(rmse_form(Ssum[i], M_of[a], f)[ri]) for f in ("as_run", "M8", "inf")}
        o[a]["crps_fair"] = float(pm[a]["crps"][ri])
        if M_of[a] > 1:
            w, se, var = Ssum[i][ri, 0], Ssum[i][ri, 2], Ssum[i][ri, 6]
            s2 = var * M_of[a] / (M_of[a] - 1)
            o[a]["ssr_scorer_column_as_run"] = float(pm[a]["ssF"][ri])
            o[a]["ssr_fortin_unbiased_as_run"] = float(np.sqrt((M_of[a] + 1) / M_of[a] * s2 / w) / np.sqrt(se / w))
            o[a]["ssr_inf"] = float(np.sqrt(s2 / w) / np.sqrt((se - s2 / M_of[a]) / w))
    for a in ARMS[1:]:
        for f in ("as_run", "inf"):
            pa = rmse_form(Ssum[0], M_of["cdffm"], f)[ri]; pb = rmse_form(Ssum[ARMS.index(a)], M_of[a], f)[ri]
            ba = rmse_form(B[:, 0], M_of["cdffm"], f)[:, ri]; bb = rmse_form(B[:, ARMS.index(a)], M_of[a], f)[:, ri]
            rel = 100 * (pa / pb - 1); bs = 100 * (ba / bb - 1)
            o[f"rmse_change_pct_vs_{a}_{f}"] = {"v": float(rel), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        ca, cb = pm["cdffm"]["crps"][ri], pm[a]["crps"][ri]
        o[f"crps_change_pct_vs_{a}_fair_equals_inf"] = float(100 * (ca / cb - 1))
    OUT["regions"][r] = o
safe_json(RES / "ds2022_classU.json", OUT)
print("identity", OUT["identity_all"], "| global:", json.dumps(OUT["regions"]["global"])[:1500])
