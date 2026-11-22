#!/usr/bin/env python3
"""Values for the three places of the manuscript left blocked by the 'ar' pass (N/rebuild/print/apply_report.md, section 6,
'NOT legitimate: blocked'). Compute and document only: every input is read, nothing outside this folder is written, and the
only output is results.json next to this script (temporary file + fsync + read-back + rename). No GPU, no station selection:
the station values are read only to score, with the masks of the stored scorers.

  1. SI 'Implausible values at the European network': RMSE of Foehn / CAMS forecast / Aurora on the 325,072 scored European
     station-days 1 day ahead, with and without the 34 station-days above 500 ug m-3. Data: the station-day file of the
     circularity analysis (the sample of its group A, whose first half of the sentence is already printed), stored and 'ar'.
     Code of the stored 7.3 / 8.4 / 8.5: an independent recomputation, st_more_checks.py (sample b, own extraction, log
     st_more_checks.log) and reliability/results/outliers_europe_check.txt (326,862 station-days, no generator kept);
     the selection rule (obs > 500 dropped, RMSE of the ensemble mean, cams1 / aur1 = equal issuance) is that of
     s15_europe_outliers.py on the same station-day file.
  2. Methods 'Calibration fine-tuning': on the 2021 selection start days, change with the CRPS fine-tuning of the fair CRPS
     and of the RMSE of the ensemble mean, land pixels of the nine regions pooled, 1 day ahead, with the scorer of the printed
     '12%' (reliability/scripts/s34_estimates.py, section (4): sums of the recs by (start day, region)) and the A14 forms of
     a14/classU/scripts/rel_score.py section (C): MSE_M = (sse - var/4 + var/M) / n, var = member variance (ddof 1).
     Runs: RUN/eval2021_sel_crps_e01_ar (45 starts), RUN/eval2021_sel_crps_e01 (stored, 46), BEFORE/eval2021_sel_e02 (46).
     Intervals: only the scorer's own resampling (rel_common.block_weights: moving blocks of 7 start days, 10,000, seed 0),
     applied to both runs with the same weights on the matched start days; no interval for 45 against 46 days.
  3. Methods, earlier scoring code (16.7 % and 19.3 % on 6,028 stations): no recomputation; the stored sums of the backup
     of the old compare_fair file are read, and its per-station sums are compared with the stored and 'ar' files.
    $FOEHN_ENVS/world/bin/python -I -B blocked.py
"""
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm"))
N = ROOT / "paper" / "natcomms"
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
FIG = ROOT / "experiments" / "figures"
RUNS = ROOT / "experiments" / "global1km" / "runs"
RUN = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
BEFORE = RUNS / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
OUT = {"script": str(Path(__file__).resolve()), "date": "2026-11-21"}


def sha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:n]


def pct(a, b):
    return 100.0 * (a / b - 1.0)


# ===================================================================================================== 1. European >500
def place1():
    R = {"what": "European network (src 0), lead 1, station-days of the paper's mask (circularity station_day 'paper', "
                 "rows 'used'); RMSE of the 4-member mean ('foehn', 1 km pixel) and of the CAMS forecast and Aurora at equal "
                 "issuance ('cams1', 'aur1', the columns of numbers.json A 'cams' / 'aurora'); 'above 500' = obs > 500",
         "designs": {}}
    Z = {}
    for tag, sub in (("stored", "data"), ("ar", "data_ar")):
        f = N / "analyses" / "circularity" / sub / "station_day.npz"
        z = np.load(f)
        Z[tag] = z
        ud = z["used"].any(1)
        src = z["src"]
        obs = z["obs"][ud, 0].astype(np.float64)
        m = z["paper"][ud, 0] & (src == 0)[None, :]
        m_all179 = z["paper"][:, 0] & (src == 0)[None, :]
        big = m & (obs > 500)
        keep = m & (obs <= 500)
        e = {k: (z[k][ud, 0].astype(np.float64) - obs) for k in ("foehn", "cams1", "aur1")}
        r = lambda k, mm: float(np.sqrt((e[k][mm] ** 2).mean()))
        d = {"file": str(f), "sha256_16": sha(f), "station_days": int(m.sum()), "station_days_all_179_rows": int(m_all179.sum()),
             "stations": int(m.any(0).sum()), "above_500": int(big.sum()), "stations_with_a_value_above_500": int(big.any(0).sum()),
             "equal_999": int((m & (obs == 999)).sum()), "station_days_without": int(keep.sum()),
             "rmse_all": {k: r(k, m) for k in e}, "rmse_without_above_500": {k: r(k, keep) for k in e},
             "share_of_foehn_squared_error_above_500_pct": float(100 * (e["foehn"][big] ** 2).sum() / (e["foehn"][m] ** 2).sum())}
        for lab, rr in (("all", d["rmse_all"]), ("without", d["rmse_without_above_500"])):
            d[f"reduction_pct_{lab}"] = {"vs_cams": 100 * (1 - rr["foehn"] / rr["cams1"]), "vs_aurora": 100 * (1 - rr["foehn"] / rr["aur1"])}
        nj = json.loads((N / "analyses" / "circularity" / ("results" if tag == "stored" else "results_ar") / "numbers.json").read_text())
        a = nj["A"]["groups"]["Europe"]["leads"]["1"]
        d["identity_numbers_json_A_Europe_1d"] = {
            "source": str(N / "analyses" / "circularity" / ("results" if tag == "stored" else "results_ar") / "numbers.json") + " A/groups/Europe/leads/1",
            "station_days": a["station_days"], "rmse": {k: a["rmse"][k] for k in ("foehn_1km", "cams", "aurora")},
            "equal": bool(a["station_days"] == d["station_days"] and abs(a["rmse"]["foehn_1km"] - d["rmse_all"]["foehn"]) < 1e-6
                          and abs(a["rmse"]["cams"] - d["rmse_all"]["cams1"]) < 1e-6 and abs(a["rmse"]["aurora"] - d["rmse_all"]["aur1"]) < 1e-6)}
        R["designs"][tag] = d
    s, a = Z["stored"], Z["ar"]
    R["references_identical_stored_vs_ar"] = {k: bool(np.array_equal(s[k], a[k], equal_nan=True))
                                              for k in ("paper", "used", "src", "obs", "cams1", "aur1", "days")}
    R["foehn_identical_stored_vs_ar"] = bool(np.array_equal(s["foehn"], a["foehn"], equal_nan=True))
    # the stored printed values and the two stored records they round from
    R["stored_records"] = {
        "manuscript": "7.3, against 8.4 and 8.5 (13% and 14% lower) without the 34 station-days above 500",
        "st_more_checks.log line 4 (sample b, 325,072 station-days, 34 above 500)":
            "without them 7.30 8.39 8.51 (13.0%, 14.3%)",
        "analyses/reliability/results/outliers_europe_check.txt line 2 (all 326,862 station-days of the reliability files, 34 dropped)":
            "RMSE Foehn 7.32 CAMS 8.41 Aurora 8.53 | -13.0 % / -14.2 %"}
    st = R["designs"]["stored"]["rmse_without_above_500"]
    R["stored_reproduced"] = bool(round(st["foehn"], 1) == 7.3 and round(st["cams1"], 1) == 8.4 and round(st["aur1"], 1) == 8.5
                                  and R["designs"]["stored"]["station_days"] == 325072 and R["designs"]["stored"]["above_500"] == 34)
    OUT["place1_europe_above_500"] = R


# ======================================================================================== 2. 2021 calibration, lead 1
def block_weights(nd, B=7, n=10000, seed=0):
    """verbatim copy of N/analyses/reliability/scripts/rel_common.block_weights (checked against its source below)"""
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]
    W = np.zeros((n, nd), np.float64)
    np.add.at(W, (np.arange(n)[:, None], idx), 1.0)
    return W


def recs_lead(ev, lead=1):
    """s34_estimates.py section (4): (start day, region) sums of n, sse, var, crps, sum_err; pooled = sum of the nine regions
    (a pixel in two boxes counted twice), days with n > 0 at the lead"""
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    z0 = np.load(parts[0])
    C = [str(c) for c in z0["rec_cols"]]
    info = [str(x) for x in z0["info"]]
    Mx = int(info[5])
    regs = [str(r) for r in z0["regions"]]
    Rr = np.concatenate([np.load(p)["recs"] for p in parts])
    c = C.index
    days = np.unique(Rr[:, c("start_doy")]).astype(int)
    rh = Rr[Rr[:, c("lead")] == lead]
    T = np.zeros((len(days), len(regs) + 1, 5))
    di = np.searchsorted(days, rh[:, c("start_doy")].astype(int))
    ri = rh[:, c("region")].astype(int)
    vals = np.stack([rh[:, c(q)] for q in ("n", "sse", "var", "crps", "sum_err")], 1)
    np.add.at(T, (di, ri), vals)
    T[:, -1] = T[:, :-1].sum(1)
    ok = T[:, -1, 0] > 0
    return {"days": days[ok], "T": T[ok, -1], "M": Mx, "info": info, "parts": len(parts), "rows": int(len(Rr))}


def scores(s, M):
    n, sse, var, crps = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    return {"n": n, "crps_fair": crps / n, "rmse_as_run": np.sqrt(sse / n), "rmse_8": np.sqrt((sse - var / M + var / 8) / n),
            "rmse_inf": np.sqrt((sse - var / M) / n), "ss_as_run": np.sqrt((M + 1) / M * var / sse)}


KEYS = ("crps_fair", "rmse_as_run", "rmse_8", "rmse_inf")


def contrast(A, B, days=None, boot=False):
    """change (%) of the calibrated model A against the model before calibration B, pooled over the start days given
    (None = each run on its own start days)"""
    if days is None:
        sa, sb = A["T"].sum(0), B["T"].sum(0)
        da, db = A["days"], B["days"]
    else:
        ia, ib = np.searchsorted(A["days"], days), np.searchsorted(B["days"], days)
        assert np.array_equal(A["days"][ia], days) and np.array_equal(B["days"][ib], days)
        sa, sb = A["T"][ia].sum(0), B["T"][ib].sum(0)
        da = db = days
    pa, pb = scores(sa, A["M"]), scores(sb, B["M"])
    d = {"start_days_calibrated": int(len(da)), "start_days_before": int(len(db)),
         "calibrated": {k: float(v) for k, v in pa.items()}, "before": {k: float(v) for k, v in pb.items()},
         "change_pct": {k: float(pct(pa[k], pb[k])) for k in KEYS}}
    if boot and days is not None:
        W = block_weights(len(days), B=7, n=10000, seed=0)
        ba, bb = scores(W @ A["T"][ia], A["M"]), scores(W @ B["T"][ib], B["M"])
        d["change_pct_ci95_paired_block_bootstrap"] = {k: [float(x) for x in np.nanpercentile(pct(ba[k], bb[k]), [2.5, 97.5])] for k in KEYS}
    return d


def place2():
    rc = (N / "analyses" / "reliability" / "scripts" / "rel_common.py").read_text()
    body = "    rng = np.random.default_rng(seed)\n    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]\n" \
           "    W = np.zeros((n, nd), np.float64)\n    np.add.at(W, (np.arange(n)[:, None], idx), 1.0)\n    return W\n"
    R = {"what": "2021 selection start days, lead 1, land pixels of the nine regions pooled, against GHAP; scorer of "
                 "s34_estimates.py section (4) (the printed '12%'); A14 forms as rel_score.py section (C)",
         "block_weights_is_verbatim_copy_of_rel_common": body in rc}
    runs = {"ar": RUN / "eval2021_sel_crps_e01_ar", "stored": RUN / "eval2021_sel_crps_e01", "before": BEFORE / "eval2021_sel_e02"}
    D = {k: recs_lead(v) for k, v in runs.items()}
    R["runs"] = {k: {"folder": str(runs[k]), "parts": D[k]["parts"], "rows": D[k]["rows"], "members": D[k]["M"],
                     "info": D[k]["info"], "start_days": int(len(D[k]["days"])),
                     "start_days_list": [int(x) for x in D[k]["days"]]} for k in D}
    # identity with the stored records
    ide = {}
    for tag, res, key in (("ar", "results_ar", "2021 final (validation year)"), ("stored", "results", "2021 final (validation year)"),
                          ("before", "results_ar", "2021 before CRPS training")):
        f = N / "analyses" / "reliability" / res / "estimates_scores.json"
        e = json.loads(f.read_text())["land"][key]["leads"]["1"]["Nine regions pooled"]
        sc = scores(D[tag]["T"].sum(0), D[tag]["M"])
        ide[f"estimates_scores {tag}"] = {"source": f"{f} land/'{key}'/leads/1/'Nine regions pooled'",
                                          "stored": {"n": e["n"], "crps": e["crps"], "rmse": e["rmse"], "ss": e["ss"]},
                                          "here": {"n": float(sc["n"]), "crps": float(sc["crps_fair"]), "rmse": float(sc["rmse_as_run"]), "ss": float(sc["ss_as_run"])},
                                          "equal": bool(e["n"] == sc["n"] and abs(e["crps"] - sc["crps_fair"]) < 1e-9 and abs(e["rmse"] - sc["rmse_as_run"]) < 1e-9)}
    for tag, res, key in (("ar", "results_ar", "2021 final (validation year)"), ("stored", "results", "2021 final (validation year)"),
                          ("before", "results_ar", "2021 before CRPS training")):
        f = P / "a14" / "classU" / res / "rel_classU_179.json"
        e = json.loads(f.read_text())["land_recs"][key]["leads"]["1"]["Nine regions pooled"]
        sc = scores(D[tag]["T"].sum(0), D[tag]["M"])
        ide[f"rel_classU_179 {tag}"] = {"source": f"{f} land_recs/'{key}'/leads/1/'Nine regions pooled'",
                                        "stored": {k: e[k] for k in ("rmse_as_run", "rmse_8", "rmse_inf")},
                                        "here": {k: float(sc[k]) for k in ("rmse_as_run", "rmse_8", "rmse_inf")},
                                        "equal": bool(all(abs(e[k] - sc[k]) < 1e-9 for k in ("rmse_as_run", "rmse_8", "rmse_inf")))}
    R["identity"] = ide
    common = np.intersect1d(D["ar"]["days"], D["before"]["days"])
    R["matched_start_days"] = {"n": int(len(common)), "only_in_before": [int(x) for x in np.setdiff1d(D["before"]["days"], D["ar"]["days"])],
                               "only_in_ar": [int(x) for x in np.setdiff1d(D["ar"]["days"], D["before"]["days"])]}
    R["ar_vs_before_45_vs_46"] = contrast(D["ar"], D["before"])                        # the record of classU row 46 / s34
    R["ar_vs_before_matched"] = contrast(D["ar"], D["before"], common, boot=True)
    sc_common = np.intersect1d(D["stored"]["days"], D["before"]["days"])
    R["stored_vs_before_46_first_design"] = contrast(D["stored"], D["before"], sc_common, boot=True)   # the printed '12%' / '2%' before the pass
    row = (P / "a14" / "classU" / "results_ar" / "classU_table_179.md").read_text().splitlines()
    R["classU_table_179_ar_row46"] = {"source": str(P / "a14" / "classU" / "results_ar" / "classU_table_179.md"),
                                      "line": next((l[:200] for l in row if l.startswith("| 46 |")), None)}
    OUT["place2_calibration_2021"] = R


# ============================================================================== 3. earlier scoring code (no recompute)
def place3():
    old = N / "review" / "keyfix" / "backup_outputs" / "experiments" / "figures" / "data" / "compare_fair_crps_e01.npz"
    new = FIG / "data" / "compare_fair_crps_e01.npz"
    nar = FIG / "data" / "compare_fair_crps_e01_ar.npz"
    zo, zn, za = np.load(old), np.load(new), np.load(nar)
    sysn = [str(s) for s in zo["systems"]]
    iF, iC, iA = 0, sysn.index("CAMS (run d+1, equal issuance)"), sysn.index("Aurora (00 UTC d+1, equal issuance)")
    S = zo["S"][6, 0]                                                   # 'all' networks, lead 1: n, sum e, sum e2, ...
    rm = {k: float(np.sqrt(S[i, 2] / S[i, 0])) for k, i in (("foehn", iF), ("cams", iC), ("aurora", iA))}
    kept = zo["SST"][0, 0, :, 1] > 0
    # same stored run: per-station lead-1 sums of the 1 km system at the stations that the old key kept
    same_new = bool(np.allclose(zo["SST"][0, :, kept], zn["SST"][0, :, kept], rtol=1e-9, atol=0, equal_nan=True))
    same_ar = bool(np.allclose(zo["SST"][0, :, kept], za["SST"][0, :, kept], rtol=1e-9, atol=0, equal_nan=True))
    ref_same = bool(np.allclose(zo["SST"][iC, :, kept], zn["SST"][iC, :, kept], rtol=1e-9, equal_nan=True))
    nk = (zn["SST"][0, 0, :, 1] > 0)
    Sn, Sa = zn["S"][6, 0], za["S"][6, 0]
    gn = {k: 100 * (1 - np.sqrt(Sn[0, 2] / Sn[0, 0]) / np.sqrt(Sn[i, 2] / Sn[i, 0])) for k, i in (("cams", iC), ("aurora", iA))}
    ga = {k: 100 * (1 - np.sqrt(Sa[0, 2] / Sa[0, 0]) / np.sqrt(Sa[i, 2] / Sa[i, 0])) for k, i in (("cams", iC), ("aurora", iA))}
    pre = Path((os.environ["FOEHN_WORK"] + "/eccv/foehn_series/docs/preprint_arxiv_2610.25941v1.txt"))
    pt = pre.read_text(errors="ignore") if pre.exists() else ""
    m = re.search(r"RMSE of Foehn is 12\.68.{0,120}?\(reductions of 16\.7% and 19\.3%\)", pt)
    nat = (N / "ref_nature_final" / "sections" / "methods.tex").read_text()
    mn = re.search(r"RMSE of \\model\{\} is 12\.68[^\n]*?19\.3\\%\)", nat)
    info = [str(x) for x in np.load(RUN / "eval_final_crps_e01" / "part_0.npz")["info"]]
    OUT["place3_earlier_scorer"] = {
        "old_file": {"path": str(old), "sha256_16": sha(old), "mtime_utc": os.path.getmtime(old), "systems_0": sysn[0],
                     "stations_scored_1d": int(kept.sum()), "station_days_1d": int(S[iF, 0]),
                     "rmse_1d": rm, "gain_pct_vs_cams": 100 * (1 - rm["foehn"] / rm["cams"]), "gain_pct_vs_aurora": 100 * (1 - rm["foehn"] / rm["aurora"])},
        "old_scorer": {"path": str(FIG / "compare_fair.bak_20261120_key.py"), "sha256_16": sha(FIG / "compare_fair.bak_20261120_key.py"),
                       "key": "np.round(lat, 4) * 1e6 + np.round(lon, 4) on float32 (review/keyfix/README.md)"},
        "same_run_as_stored_file_at_the_6028_stations": same_new, "same_as_ar_file_at_the_6028_stations": same_ar,
        "cams_column_same_old_vs_stored": ref_same,
        "stored_file_after_keyfix_first_design": {"path": str(new), "stations_1d": int(nk.sum()), "gain_pct": gn},
        "ar_file_after_keyfix": {"path": str(nar), "gain_pct": ga},
        "run_of_the_old_file": {"label": "crps_e01 = RUN/eval_final_crps_e01 (experiments/figures/final_model.py MODELS['crps_e01'])",
                                "part_info": info, "members": int(info[5]), "euler_steps": int(info[6]),
                                "noise": "stored fields G/noise_tiles.npy, one field per member reused at every start day and lead "
                                         "(REBUILD_PLAN.md section 1, sampler S0; F1 = eval_final_crps_e01, replaced by G1 = _ar in A16)"},
        "keyfix_record": {"source": str(N / "review" / "keyfix" / "README.md") + " 'Run of 2026-11-20'",
                          "text": "gains 16.0 % [10.9, 22.9] and 18.5 % [14.6, 23.9] (was 16.7 % [11.5, 23.8] and 19.3 % [15.4, 24.9]); "
                                  "6,207 stations scored 1 day ahead (was 6,028), 1,036,052 station-days (was 1,005,327)"},
        "printed_in_preprint": {"source": str(pre), "found": bool(m), "text": m.group(0) if m else None},
        "printed_in_nature_version": {"source": str(N / "ref_nature_final" / "sections" / "methods.tex"), "found": bool(mn),
                                      "text": mn.group(0) if mn else None}}


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    assert tmp.read_text() == text
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


PROPOSED = {
    "note": "replacement texts proposed from the values above (old -> new, LaTeX); none is applied",
    "place1_SI_l454": {
        "old": r"without the 34 station-days above 500\,\textmu g\,m$^{-3}$, it is 7.3, against 8.4 and 8.5 (13\% and 14\% lower).",
        "new": r"without the 34 station-days above 500\,\textmu g\,m$^{-3}$, it is 7.4, against 8.4 and 8.5 (12\% and 13\% lower)."},
    "place2_methods_l115": {
        "old": r"for a 2\% increase in the error of the ensemble mean (the model before calibration being sampled with 12 evaluations and the first noise design, the calibrated model with four)",
        "new_recommended_matched_45_days": r"for a 1\% increase in the error of the mean of the four members, which becomes a decrease of 2\% for eight members and of 5\% for infinitely many (on the 45 start days common to both models; the model before calibration being sampled with 12 evaluations and the first noise design, the calibrated model with four)",
        "new_alternative_stored_record_45_vs_46": r"for a 2\% increase in the error of the mean of the four members, which becomes a decrease of 1\% for eight members and of 4\% for infinitely many (the calibrated model on 45 start days, the model before calibration on 46; the model before calibration being sampled with 12 evaluations and the first noise design, the calibrated model with four)",
        "crps": "'12\\%' unchanged in both samples (-12.46 % matched, -12.09 % stored record)"},
    "place3_methods_l192": {
        "old": r"left out 183 stations and gave gains of 16.7\% and 19.3\% on 6,028 stations.",
        "new_A_keep_labelled": r"left out 183 stations and, with the first noise design and four members as run, gave gains of 16.7\% and 19.3\% on 6,028 stations, against 16.0\% and 18.5\% with the corrected code (Supplementary Table~\ref{S-tab:noisedesign}).",
        "new_B_A16_4.1_4.2_no_first_design_value": r"left out 183 stations (6,028 stations were scored 1 day ahead instead of 6,207).",
        "note": "A keeps first-design values in the Methods, which A16 4.1/4.2 confine to the SI table"}}


if __name__ == "__main__":
    place1()
    place2()
    place3()
    OUT["proposed_texts"] = PROPOSED
    write_atomic(HERE / "results.json", json.dumps(OUT, indent=1, default=float) + "\n")
    print(json.dumps(OUT, indent=1, default=float)[:20000])
