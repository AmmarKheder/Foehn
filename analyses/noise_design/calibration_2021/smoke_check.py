#!/usr/bin/env python3
"""A15 smoke gate (a15/A15_draft.md section 6.2), run on the GPU node at the end of jobs/smoke.sh, no station file:
  (a) data/smoke/part_stored.npz (base stored, rho 1 = the stored design) against RUN/eval2021_sel_crps_e01 for (9, 1) and
      (9, 2), at A14's smoke tolerances (a14/r2/results/smoke_check.json): map25, map25_m0, map25_sd with the same finite
      pattern, bit-identical share >= 0.286, max |diff| <= 2.25 ug m-3, mean |diff| <= 0.05; map25_ghap identical; region
      sums n, sse, crps, var within 0.5 %; map25_mk[:, 0] == map25_m0. Only keys, recs, rec_cols, seams and the maps are
      read from the part files (never st_*).
  (b) data/smoke/part_cost.npz (philox, rho 0.70, Cal start 1, lead 7; never scored): finite pattern equal to the stored
      (1, 7) forecast; from logs/a15.out (lines after the last '=== A15 smoke job' marker): io wait / gpu <= 0.15 (else C
      and M use 6 workers: field 'workers') and seconds per forecast (if > 430 s, the time limits of C and M to pass to
      sbatch are reported);
  golden hashes recomputed here equal to golden.json.
    python smoke_check.py -> results/smoke_check.json; exit 1 if a gating check fails"""
import arnoise as ar  # noqa: I001  (first)
import json
import math
import re
import sys

import numpy as np

MAPS = ("map25", "map25_m0", "map25_sd", "map25_ghap")
LOG = ar.A15 / "logs" / "a15.out"


def stored_rows(want):
    found = {}
    for p in sorted(ar.EV.glob("part_*.npz")):
        with np.load(p) as z:
            for n, k in enumerate(z["keys"].tolist()):
                if (k[0], k[1]) in want:
                    assert (k[0], k[1]) not in found
                    found[(k[0], k[1])] = (p, n, k[2])
    assert set(found) == set(want), (found, want)
    return found


def read(p, n):
    with np.load(p) as z:
        o = {k: z[k][n] for k in MAPS}
        keys, cols = z["keys"], [str(c) for c in z["rec_cols"]]
        o["recs"] = {int(r[2]): r for r in z["recs"] if (int(r[0]), int(r[1])) == tuple(int(v) for v in keys[n][:2])}
        o["seam"] = float(z["seams"][n][2])
        o["info"] = [str(x) for x in z["info"]]
        if "map25_mk" in z.files:
            o["map25_mk"] = z["map25_mk"][n]
    return o, cols


def cmp(a, b):                                       # as a14/r2/smoke_check.py
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    same = (a.view(np.uint16) == b.view(np.uint16)) if a.dtype == np.float16 and b.dtype == np.float16 else (a == b)
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float(same.mean()),
                bit_identical_share_finite_cells=float(same[both].mean()) if both.any() else None,
                max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                share_above_0p5=float((d > 0.5).mean()) if d.size else 0.0)


def ok_map(c):
    return c["finite_pattern_equal"] and c["bit_identical_share"] >= 0.286 and c["max_abs"] <= 2.25 and c["mean_abs"] <= 0.05


def parse_log():
    if not LOG.exists():
        return None, []
    lines = LOG.read_text(errors="replace").splitlines()
    mk = [i for i, l in enumerate(lines) if l.startswith("=== A15 smoke job") and " start " in l]
    seg = lines[mk[-1]:] if mk else []
    pat = re.compile(r"rank \S+: (\S+) start (\d+) lead (\d+) rho ([\d.]+): io wait (\d+)s gpu (\d+)s post (\d+)s .*\| (\d+)/(\d+) (\d+)s$")
    rows = []
    for l in seg:
        m = pat.search(l)
        if m:
            g = m.groups()
            rows.append({"plan": g[0], "start": int(g[1]), "lead": int(g[2]), "rho": float(g[3]), "io": int(g[4]),
                         "gpu": int(g[5]), "post": int(g[6]), "elapsed": int(g[9])})
    return (mk[-1] if mk else None), rows


def hms(s):
    s = int(math.ceil(s / 300.0) * 300)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:00"


def main():
    R = {"what": __doc__, "tests": {}}
    T = R["tests"]
    # ---------------- (a)
    pa = ar.A15 / "data" / "smoke" / "part_stored.npz"
    st = stored_rows({(9, 1), (9, 2)})
    with np.load(pa) as z:
        ka = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
        rho_a, base_a = z["keys_rho"].tolist(), str(z["base"])
    assert sorted(ka) == [(9, 1, 10), (9, 2, 11)] and base_a == "stored" and rho_a == [1.0, 1.0], (ka, base_a, rho_a)
    a_ok = True
    for n, (d, h, t) in enumerate(ka):
        A, cols = read(pa, n)
        S, cols_s = read(st[(d, h)][0], st[(d, h)][1])
        assert cols == cols_s and st[(d, h)][2] == t
        maps = {f: cmp(A[f], S[f]) for f in MAPS}
        rel = {}
        assert set(A["recs"]) == set(S["recs"]) and A["recs"], (A["recs"].keys(), S["recs"].keys())
        for c in ("n", "sse", "crps", "var", "sae", "sum_err", "brier_35"):
            j = cols.index(c)
            rel[c] = max(abs(A["recs"][g][j] - S["recs"][g][j]) / max(abs(S["recs"][g][j]), 1e-9) for g in S["recs"])
        mk0 = bool(np.array_equal(A["map25_mk"][0].view(np.uint16), A["map25_m0"].view(np.uint16)))
        good = (all(ok_map(maps[f]) for f in ("map25", "map25_m0", "map25_sd")) and maps["map25_ghap"]["max_abs"] == 0.0
                and maps["map25_ghap"]["finite_pattern_equal"] and maps["map25_ghap"]["bit_identical_share"] == 1.0
                and all(rel[c] <= 0.005 for c in ("n", "sse", "crps", "var")) and mk0 and A["info"] == S["info"])
        a_ok &= good
        T[f"a_{d}_{h}"] = {"pass": good, "maps": maps, "region_sums_max_rel": rel, "map25_mk0_equals_map25_m0": mk0,
                           "seam_arm": A["seam"], "seam_stored": S["seam"], "info_equal": A["info"] == S["info"]}
    # ---------------- (b)
    pb = ar.A15 / "data" / "smoke" / "part_cost.npz"
    with np.load(pb) as z:
        kb = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
        rho_b, base_b = z["keys_rho"].tolist(), str(z["base"])
    assert kb == [(1, 7, 8)] and base_b == "philox" and rho_b == [0.7], (kb, base_b, rho_b)
    B, _ = read(pb, 0)
    sb = stored_rows({(1, 7)})
    S, _ = read(sb[(1, 7)][0], sb[(1, 7)][1])
    fin = {f: bool(np.array_equal(np.isfinite(B[f].astype(np.float32)), np.isfinite(S[f].astype(np.float32)))) for f in MAPS}
    fin_mk = [bool(np.array_equal(np.isfinite(B["map25_mk"][m].astype(np.float32)), np.isfinite(S["map25_m0"].astype(np.float32))))
              for m in range(4)]
    differs = cmp(B["map25_m0"], S["map25_m0"])                       # informative: another noise, not expected equal
    b_ok = all(fin.values()) and all(fin_mk) and cmp(B["map25_ghap"], S["map25_ghap"])["max_abs"] == 0.0
    T["b_finite_pattern"] = {"pass": b_ok, "finite_pattern_equal": fin, "map25_mk_finite_pattern_equal": fin_mk,
                             "member0_vs_stored_member0_informative": differs, "seam_arm": B["seam"], "seam_stored": S["seam"]}
    marker, rows = parse_log()
    cost = [r for r in rows if r["plan"] == "smoke_cost" and r["start"] == 1 and r["lead"] == 7]
    stor = [r for r in rows if r["plan"] == "smoke_stored"]
    info = {"log_marker_line": marker, "progress_lines": rows}
    if cost:
        c = cost[-1]
        ratio = c["io"] / max(c["gpu"], 1)
        sec = c["io"] + c["gpu"] + c["post"]
        info.update(io_over_gpu=ratio, seconds_per_forecast=sec, workers=4 if ratio <= 0.15 else 6)
        if sec > 430:                                                   # draft section 5: limits raised, lists unchanged
            info["new_time_limits"] = {"calib": hms(8 * sec + 80 + 180 + 900), "main": hms(27 * sec + 80 + 180 + 1800)}
    else:
        info.update(workers=4, error="no smoke_cost progress line found in logs/a15.out")
    T["b_cost"] = {"pass": bool(cost), **info}
    R["workers"] = info["workers"]
    if stor:
        R["stored_smoke_seconds"] = [r["io"] + r["gpu"] + r["post"] for r in stor]
    # ---------------- golden hashes on this node
    gold = json.loads((ar.A15 / "golden.json").read_text())
    here = ar.golden_hashes()
    T["golden_gpu_node"] = {"pass": here == gold, "mismatched": [k for k in gold if here.get(k) != gold[k]]}
    R["all_passed"] = bool(a_ok and b_ok and all(v["pass"] for v in T.values()))
    ar.save_json(ar.A15 / "results" / "smoke_check.json", R)
    print(json.dumps(ar.jsonable({k: v.get("pass") for k, v in T.items()})), "workers", R["workers"],
          "ALL PASSED" if R["all_passed"] else "SMOKE GATE FAILED", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
