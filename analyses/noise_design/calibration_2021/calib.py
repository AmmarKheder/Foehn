#!/usr/bin/env python3
"""A15 rho* from GPU run C (a15/A15_draft.md section 4.5), CPU, model output only (no truth enters f-hat):
f-hat(rho) = lag-1 (leads 1-2) correlation of A_h = ln max(x_m, 0.1) - ln max(o_m, 0.1) (member m against the mean of the
other three, 0.25 deg cell means map25_mk), centered per (member choice, cell, lead) over the 15 Cal runs, resident weights,
the cells of prep.py; covariance and variances pooled over the 4 member choices before one correlation. The lead-1 rows
(rho 0.70) serve every rho (I8: z(m, d, 1) does not depend on rho). Then pool-adjacent-violators on the grid {0.50, 0.70,
0.90}, linear interpolation at tau_Cal (prep.json; smallest rho on a flat segment), rounded to 0.01; outside [f(0.50),
f(0.90)] rho* is clipped to the end and flagged (A15 then stops before run M, UNDECIDED).
Also asserts: exactly the 60 planned forecasts, and map25_ghap equal to ext_2021 Y at the cells.
    python calib.py -> results/rho.json"""
import arnoise as ar  # noqa: I001  (first)
import json
import math
import time

import numpy as np

ar.deny_2022()
GRID = (0.50, 0.70, 0.90)
EPS, M = 0.1, 4


def pav(y):
    """isotonic (non-decreasing) least squares fit, equal weights"""
    blocks = [[float(v), 1.0, [i]] for i, v in enumerate(y)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v0, w0, i0 = blocks[i]
            v1, w1, i1 = blocks[i + 1]
            blocks[i] = [(v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1, i0 + i1]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(len(y))
    for v, _, idx in blocks:
        out[idx] = v
    return out


def rho_star_of(f_iso, tau):
    if tau < f_iso[0]:
        return GRID[0], True
    if tau > f_iso[-1]:
        return GRID[-1], True
    for j in range(len(GRID) - 1):
        if f_iso[j] <= tau <= f_iso[j + 1]:
            if f_iso[j + 1] == f_iso[j]:
                r = GRID[j]
            else:
                r = GRID[j] + (tau - f_iso[j]) / (f_iso[j + 1] - f_iso[j]) * (GRID[j + 1] - GRID[j])
            return math.floor(r * 100 + 0.5) / 100, False
    raise AssertionError((f_iso, tau))


def member_dev(X4):
    """(4, C) float32 member cell means -> (4, C) float64 A of each member choice"""
    S = X4.sum(0, dtype=np.float64)
    return np.stack([np.log(np.maximum(X4[m], EPS)) - np.log(np.maximum((S - X4[m]) / (M - 1), EPS)) for m in range(M)])


def main():
    t0 = time.time()
    prep = json.loads((ar.A15 / "results" / "prep.json").read_text())
    tau = float(prep["tau_Cal"])
    D, _ = ar.episodes_2021()
    cells, w = D["cells"], D["pop"].astype(np.float64)
    assert ar.cells_hash(cells) == prep["cells_sha256"], "cells differ from prep.py"
    lab = [int(v) for v in D["labels"]]
    items, ghap_bad = {}, 0
    parts = sorted((ar.A15 / "data" / "calib").glob("part_*.npz"))
    for p in parts:
        if ".tmp" in p.name:
            continue
        with np.load(p) as z:
            keys, kr, mk, g = z["keys"], z["keys_rho"], z["map25_mk"], z["map25_ghap"]
            assert str(z["base"]) == "philox" and mk.shape[1] == M, (p, z["base"], mk.shape)
            for j, ((d, h, dt), r) in enumerate(zip(keys.tolist(), kr.tolist())):
                key = (int(d), int(h), round(float(r), 2))
                assert key not in items and dt == d + h, key
                X4 = mk[j].reshape(M, -1)[:, cells].astype(np.float32)
                assert np.isfinite(X4).all(), key
                items[key] = X4
                gy = g[j].reshape(-1)[cells]
                ghap_bad += not np.array_equal(gy.view(np.uint16), D["Y"][lab.index(d), h - 1].view(np.uint16))
    want = {(d, 1, 0.70) for d in ar.CAL} | {(d, 2, r) for d in ar.CAL for r in GRID}
    assert set(items) == want and len(items) == 60, (sorted(set(items) ^ want)[:10], len(items))
    assert ghap_bad == 0, f"map25_ghap differs from ext_2021 Y in {ghap_bad} forecasts"
    A1 = np.stack([member_dev(items[(d, 1, 0.70)]) for d in ar.CAL])           # (15, 4, C)
    A1 -= A1.mean(0)
    f_hat, per_choice, moments = {}, {}, {}
    for r in GRID:
        A2 = np.stack([member_dev(items[(d, 2, r)]) for d in ar.CAL])
        A2 -= A2.mean(0)
        cov = ((A1 * A2).sum(0) * w).sum(-1)                                     # (4,) per member choice
        v1, v2 = ((A1 * A1).sum(0) * w).sum(-1), ((A2 * A2).sum(0) * w).sum(-1)
        f_hat[r] = float(cov.sum() / np.sqrt(v1.sum() * v2.sum()))
        per_choice[r] = (cov / np.sqrt(v1 * v2)).tolist()
        moments[r] = {"var_lead1_pooled": float(v1.sum() / (15 * w.sum() * M)), "var_lead2_pooled": float(v2.sum() / (15 * w.sum() * M))}
        print(f"rho {r:.2f}: f-hat {f_hat[r]:.4f} per choice {np.round(per_choice[r], 4).tolist()}", flush=True)
    fh = [f_hat[r] for r in GRID]
    f_iso = pav(fh)
    rs, clipped = rho_star_of(f_iso, tau)
    out = {"what": __doc__, "grid": list(GRID), "f_hat": {f"{r:.2f}": f_hat[r] for r in GRID},
           "f_iso": {f"{r:.2f}": float(v) for r, v in zip(GRID, f_iso)},
           "f_hat_per_member_choice": {f"{r:.2f}": per_choice[r] for r in GRID},
           "pooled_variances": {f"{r:.2f}": moments[r] for r in GRID},
           "tau_Cal": tau, "rho_star": rs, "clipped": bool(clipped), "n_items": len(items), "n_cells": int(len(cells)),
           "cells_sha256": prep["cells_sha256"], "parts": [p.name for p in parts], "seconds": round(time.time() - t0, 1)}
    out["all_passed"] = True
    ar.save_json(ar.A15 / "results" / "rho.json", out)
    print(f"tau_Cal {tau:.4f} f_iso {np.round(f_iso, 4).tolist()} -> rho* {rs:.2f} clipped {clipped}", flush=True)


if __name__ == "__main__":
    main()
