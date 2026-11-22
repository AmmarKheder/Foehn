#!/usr/bin/env python3
"""A17 smoke gate (a17/A17_draft.md section 6.2), run on the GPU node at the end of code/smoke.sh; main.sh and eval17.py
--plan main refuse to start unless data/smoke_check.json says all_passed with the same code. Identity checks only: no
skill, error or pair statistic of any forecast is computed here.
  E1  the negated members run alone (data/smoke_alone, member ids 4..7) equal the negated members of an 8-member run
      (data/smoke_eight, ids 0..7) on the forecast (9, 2): map25_mk member by member at the smoke tolerances of A15 6.2(a)
      (same finite pattern, bit-identical share >= 0.286, max |diff| <= 2.25 ug m-3, mean |diff| <= 0.05); bit-identity
      is reported
  E2  members 0-3 of the 8-member run reproduce A15's run M (a15/data/main, (9, 2)) at the same tolerances
  E3  map25_ghap bit-identical in the three folders and to a15/data/main for (9, 2) and (17, 2); map25_m0 = map25_mk[0];
      keys, keys_rho = rho*, base philox, member_ids, info as planned
  E4  the negated members are not the original members (bit-identical share of the finite cells of (9, 2) < 0.5): the
      negation reached the network (a boolean only)
  cost from data/a17_gpu.out (the progress lines after '=== A17 smoke job <id> start'): seconds of the first and of the warm
      forecast of 4 negated members (smoke_alone (9, 2) and (17, 2)), of one 8-member forecast; projected wall clock and
      GCD-hours of the main run (8 ranks, at most 8 forecasts per rank) against its limit 1:20:00 and the cap of 12; the
      gate fails if either is exceeded, and no larger limit is suggested (8 x 1:20 + the smoke's 1:00 = 11.7 GCD-hours)
  golden hashes recomputed on this node equal to check17.json's golden17; the code equals the code check17 checked
    python smoke17.py <smoke job id> -> data/smoke_check.json (exit 1 if a gating check fails)"""
import noise17 as nz  # noqa: I001  (first)
import json
import math
import re
import sys

import numpy as np

MAPS = ("map25", "map25_m0", "map25_sd", "map25_ghap")
A15M = nz.A15_DIR / "data" / "main"
LIMIT_MAIN, CAP = "01:20:00", 12.0                                  # main.sh --time; smoke.sh --time=01:00:00 (1.0 GCD-h)


def find(folder, want):
    found = {}
    for p in sorted(folder.glob("part_*.npz")):
        if ".tmp" in p.name:
            continue
        with np.load(p) as z:
            for n, k in enumerate(z["keys"].tolist()):
                if (k[0], k[1]) in want:
                    assert (k[0], k[1]) not in found, (k, p)
                    found[(k[0], k[1])] = (p, n, k[2])
    assert set(found) == set(want), (folder, sorted(found), sorted(want))
    return found


def read(p, n):
    with np.load(p) as z:
        o = {k: z[k][n] for k in MAPS}
        o["map25_mk"] = z["map25_mk"][n]
        o["info"] = [str(x) for x in z["info"]]
        o["rho"], o["base"] = float(z["keys_rho"][n]), str(z["base"])
        o["mids"] = [int(m) for m in z["member_ids"]] if "member_ids" in z.files else [0, 1, 2, 3]
    return o


def cmp(a, b):                                       # as a15/smoke_check.py and a16/smoke_check.py
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    same = (a.view(np.uint16) == b.view(np.uint16)) if a.dtype == np.float16 and b.dtype == np.float16 else (a == b)
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float(same.mean()),
                max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0)


def ok_map(c):                                       # A15 6.2(a)
    return c["finite_pattern_equal"] and c["bit_identical_share"] >= 0.286 and c["max_abs"] <= 2.25 and c["mean_abs"] <= 0.05


def same16(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def identity(T, rho):
    al = find(nz.DATA / "smoke_alone", {(9, 2), (17, 2)})
    ei = find(nz.DATA / "smoke_eight", {(9, 2)})
    m15 = find(A15M, {(9, 2), (17, 2)})
    A = {k: read(*al[k][:2]) for k in al}
    E = read(*ei[(9, 2)][:2])
    M = {k: read(*m15[k][:2]) for k in m15}
    meta = (all(A[k]["mids"] == [4, 5, 6, 7] and A[k]["rho"] == rho and A[k]["base"] == "philox" and A[k]["info"] == M[k]["info"]
                for k in A) and E["mids"] == list(range(8)) and E["rho"] == rho and E["base"] == "philox"
            and E["info"][:5] == M[(9, 2)]["info"][:5] and E["info"][5] == "8" and al[(9, 2)][2] == 11 and al[(17, 2)][2] == 19
            and ei[(9, 2)][2] == 11 and all(M[k]["rho"] == rho for k in M))
    e1 = {f"id{4 + m}": cmp(E["map25_mk"][4 + m], A[(9, 2)]["map25_mk"][m]) for m in range(4)}
    e2 = {f"id{m}": cmp(E["map25_mk"][m], M[(9, 2)]["map25_mk"][m]) for m in range(4)}
    ghap = all(same16(A[k]["map25_ghap"], M[k]["map25_ghap"]) for k in A) and same16(E["map25_ghap"], M[(9, 2)]["map25_ghap"])
    m0 = all(same16(A[k]["map25_m0"], A[k]["map25_mk"][0]) for k in A) and same16(E["map25_m0"], E["map25_mk"][0])
    fin = [np.isfinite(M[(9, 2)]["map25_mk"][m].astype(np.float32)) for m in range(4)]
    e4 = all(float((A[(9, 2)]["map25_mk"][m].view(np.uint16) == M[(9, 2)]["map25_mk"][m].view(np.uint16))[fin[m]].mean()) < 0.5
             for m in range(4))
    T["E1_alone_equals_eight"] = {"pass": all(ok_map(c) for c in e1.values()), "members": e1,
                                  "bit_identical": all(c["bit_identical_share"] == 1.0 for c in e1.values())}
    T["E2_eight_members0_3_equal_a15"] = {"pass": all(ok_map(c) for c in e2.values()), "members": e2,
                                          "bit_identical": all(c["bit_identical_share"] == 1.0 for c in e2.values())}
    T["E3_truth_and_format"] = {"pass": bool(ghap and m0 and meta), "map25_ghap_identical": ghap, "map25_m0_is_first_member": m0,
                                "keys_rho_base_ids_info": meta}
    T["E4_negated_differ_from_originals"] = {"pass": bool(e4)}


def parse_log(job):
    lines = nz.GPU_LOG.read_text(errors="replace").splitlines() if nz.GPU_LOG.exists() else []
    mk = [i for i, l in enumerate(lines) if l.startswith(f"=== A17 smoke job {job} start")]
    seg = lines[mk[-1]:] if mk else []
    pe = re.compile(r"rank \S+: (\w+) start (\d+) lead (\d+) rho ([\d.]+): io wait (\d+)s gpu (\d+)s post (\d+)s .*\| (\d+)/(\d+) (\d+)s$")
    rows = {}
    for l in seg:
        m = pe.search(l)
        if m:
            g = m.groups()
            rows.setdefault(g[0], []).append(dict(start=int(g[1]), lead=int(g[2]), io=int(g[4]), gpu=int(g[5]), post=int(g[6]),
                                                  elapsed=int(g[9])))
    return (mk[-1] if mk else None), rows


def sec(t):
    h, m, s = (int(v) for v in t.split(":"))
    return 3600 * h + 60 * m + s


def cost(T, job):
    marker, rows = parse_log(job)
    info = {"log_marker_line": marker, "progress_lines": rows}
    found = len(rows.get("smoke_alone", [])) == 2 and len(rows.get("smoke_eight", [])) == 1
    if found:
        a1, a2 = rows["smoke_alone"]
        first, warm = (a1["io"] + a1["gpu"] + a1["post"]), (a2["io"] + a2["gpu"] + a2["post"])
        e = rows["smoke_eight"][0]
        boot = 300                                                   # start-up per process (model, dataset, workers), assumed
        per_rank = math.ceil(len(nz.PLANS["main"]["items"]) / 8)
        wall = boot + first + (per_rank - 1) * warm
        gcd_h = 8 * wall / 3600
        info.update(seconds_first_forecast_4_negated=first, seconds_warm_forecast_4_negated=warm,
                    seconds_first_forecast_8_members=e["io"] + e["gpu"] + e["post"], io_over_gpu=a2["io"] / max(a2["gpu"], 1),
                    main=dict(forecasts=len(nz.PLANS["main"]["items"]), forecasts_per_rank=per_rank, wall_s=round(wall),
                              gcd_h=round(gcd_h, 2), limit=LIMIT_MAIN, within_limit=wall <= sec(LIMIT_MAIN), cap_gcd_h=CAP,
                              within_cap=gcd_h + 1.0 <= CAP),
                    workers=6 if a2["io"] > 0.15 * a2["gpu"] else 4)
        found = info["main"]["within_limit"] and info["main"]["within_cap"]   # a limit above 1:20 would break the cap
    else:
        info["error"] = f"progress lines not found in {nz.GPU_LOG} for job {job}: {sorted(rows)}"
    T["cost"] = {"pass": bool(found), **info}


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    R = {"what": __doc__, "job": job, "tests": {}}
    T = R["tests"]
    rho = nz.rho_star()
    try:
        identity(T, rho)
    except Exception as e:                                           # one failing step must not hide the others
        T["identity_error"] = {"pass": False, "error": repr(e)}
    try:
        cost(T, job)
    except Exception as e:
        T["cost"] = {"pass": False, "error": repr(e)}
    chk = nz.read_json(nz.CHECK_JSON) or {}
    T["golden_gpu_node"] = {"pass": nz.golden17() == chk.get("golden17")}
    R["code_sha256"] = nz.run_code_sha()
    T["code_same_as_check17"] = {"pass": chk.get("all_passed") is True and chk.get("code_sha256") == R["code_sha256"]}
    R["workers"] = T.get("cost", {}).get("workers", 4)
    R["all_passed"] = bool(all(v.get("pass") is True for v in T.values()) and {
        "E1_alone_equals_eight", "E2_eight_members0_3_equal_a15", "E3_truth_and_format", "E4_negated_differ_from_originals"} <= set(T))
    nz.save_json(nz.SMOKE_JSON, R)
    print(json.dumps(nz.jsonable({k: v.get("pass") for k, v in T.items()})), "ALL PASSED" if R["all_passed"] else "SMOKE GATE FAILED",
          flush=True)
    if "main" in T.get("cost", {}):
        print("cost", json.dumps(nz.jsonable({k: T["cost"][k] for k in T["cost"] if k != "progress_lines"})), flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
