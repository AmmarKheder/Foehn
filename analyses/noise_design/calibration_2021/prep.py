#!/usr/bin/env python3
"""A15 target of the calibration (a15/A15_draft.md section 4.5), CPU, before any GPU run, from stored 2021 data only:
tau_Cal = the truth's de-diluted lag-1 correlation of its run-varying deviation (log, centered per (cell, lead) over the 15
Cal runs, pooled over h = 1..6, resident weights) on the cells complete in every one of the 45 stored 2021 runs, with the
estimator of episodes_diag/code/revise.py (pass1 / p1_summary 'pass1_rho_y_dediluted/log/cen'; imported read only) and the
two-way interval (runs x 10-degree tiles) of stats_e5 'E5/log/cen/k1/rho_y_dediluted' (2000 draws, seed 20261121).
Reported: the same estimate on all 45 runs with the 2021-only cells (the two-year value was 0.588), the cell count and hash.
No 2022 file (audit hook), no station or monitor file.
    python prep.py -> results/prep.json"""
import arnoise as ar  # noqa: I001  (first)
import time

import numpy as np

ar.deny_2022()
NBOOT, SEED = 2000, 20261121


def main():
    t0 = time.time()
    D, revise = ar.episodes_2021()
    lab = [int(v) for v in D["labels"]]
    cal = [lab.index(d) for d in ar.CAL]
    Dc = ar.subset_runs(D, cal)
    C = len(Dc["pop"])
    p = revise.pass1(Dc)
    s, cen = revise.p1_summary(p, Dc)
    tau = float(s["pass1_rho_y_dediluted/log/cen"])
    print(f"Cal pass 1 {time.time() - t0:.0f}s: tau_Cal {tau:.4f}", flush=True)
    tid = Dc["tid"]
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    agg = revise.Agg(len(cal), starts, Dc["pop"])
    maps = {"cenA_log": cen["A_log"], "cenB_log": cen["B_log"], "cenA_lin": cen["A_lin"], "cenB_lin": cen["B_lin"]}
    for r in range(len(cal)):
        X, Y, Mn = (Dc[k][r].astype(np.float32) for k in ("X", "Y", "Mn"))
        agg.add(r, "E5", revise.e5_cols(X, Y, Mn, maps, C))
    res, info = revise.summarise(agg, len(cal), 1, NBOOT, SEED, fn=revise.stats_e5)
    e5 = res["E5/log/cen/k1/rho_y_dediluted"]
    print(f"E5 two-way {time.time() - t0:.0f}s: {e5}", flush=True)
    p45 = revise.pass1(D)
    s45, _ = revise.p1_summary(p45, D)
    out = {"what": __doc__, "tau_Cal": tau,
           "tau_Cal_E5": {"est": e5["est"], "est_cell": e5["est_cell"], "runs": e5["runs"], "two_way": e5["two_way"]},
           "tau_Cal_others": {k: s[k] for k in ("pass1_rho_y_dediluted/log/raw", "pass1_rho_A/log/cen", "pass1_rho_B/log/cen",
                                                "static_share/A_log/res", "static_share/B_log/res")},
           "E5_Cal_lags_2_6_rho_y_dediluted_cen": {k: res[f"E5/log/cen/k{k}/rho_y_dediluted"]["est"] for k in range(2, 7)},
           "all45_2021cells": {"pass1_rho_y_dediluted/log/cen": s45["pass1_rho_y_dediluted/log/cen"],
                               "pass1_rho_y_dediluted/log/raw": s45["pass1_rho_y_dediluted/log/raw"],
                               "reference_two_year_cells": 0.587749},
           "bootstrap": {"draws": NBOOT, "seed": SEED, "run_block": 1, "N": len(cal), **info},
           "n_cells": C, "residents": float(Dc["pop"].sum()), "cells_sha256": ar.cells_hash(Dc["cells"]),
           "n_tiles": int(len(starts)), "CAL": ar.CAL, "CONF": ar.CONF, "cal_labels_used": [int(v) for v in Dc["labels"]],
           "seconds": round(time.time() - t0, 1)}
    out["all_passed"] = bool(np.isfinite(tau) and 0.0 < tau < 1.0 and abs(tau - float(e5["est"])) < 0.01)
    ar.save_json(ar.A15 / "results" / "prep.json", out)
    print(f"tau_Cal {tau:.4f} two-way {e5['two_way']} | 45 runs {s45['pass1_rho_y_dediluted/log/cen']:.4f} | {C} cells | "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
