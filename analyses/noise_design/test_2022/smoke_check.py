#!/usr/bin/env python3
"""A16 check C2 (protocol_amendments/A16_test_year_evolving_noise.md section 3), run on the GPU node at the end of
jobs/smoke.sh; the runs G1-G9 and G11 refuse to start unless results/smoke_check.json says all_passed.
  (a)   data/smoke/a (base stored, rho 1 = the stored design; F1's forecasts (0, 1) and (0, 2)) against
        RUN/eval_final_crps_e01 at the smoke tolerances of A15 6.2(a): map25, map25_m0, map25_sd with the same finite
        pattern, bit-identical share >= 0.286, max |diff| <= 2.25 ug m-3, mean |diff| <= 0.05; map25_ghap bit-identical;
        region sums n, sse, crps, var within 0.5 %; map25_mk[0] == map25_m0; info equal. Also gating (2022 has them):
        persistence sums (n, sse_persist, sae_persist) and GHAP at the station pixels (st_ghap) identical; the members at
        the station pixels (st_ens) and map5km_d0_h1/h2 with the same finite pattern, mean |diff| <= 0.05 and 99.99th
        percentile <= 2.5 (the rule of A14a for pixel values).
  (b)   data/smoke/b4 (eval_ar22, 4 members, --pop, rho*) and data/smoke/b16 (sample16_ar, 16 members, rho*), the same
        2022 forecast (2, 3): members 0-3 at the populated pixels, each with the same finite pattern, mean |diff| <= 0.05
        and 99.99th percentile <= 2.5 (A14a); b4's map25_ghap and persistence sums identical to F1's (2, 3).
        Informative: b4's members against F2's stored members (another noise: expected to differ by their spread).
  (c)   data/smoke/c (2021, Conf start 9, lead 2, rho*) against A15's run M (a15/data/main) at the tolerances of (a),
        every member's map25_mk included.
  cost  from logs/a16.out (lines after '=== A16 smoke job <id> start'): seconds per warm forecast, 4 members (second
        forecast of (a)) and 16 members (b16: GPU time - first window + median window, plus io wait); projected wall clock
        and GCD-hours of every run against its time limit and the cap of 400 GCD-hours (reported, not gating).
  golden hashes recomputed on this node equal to a15/golden.json and golden22.json; the code equals check_cpu's.
    python smoke_check.py <smoke job id> -> results/smoke_check.json (exit 1 if a gating check fails)"""
import arnoise22 as an  # noqa: I001  (first)
import json
import math
import re
import sys

import numpy as np

sys.path.insert(0, str(an.ROOT / "a13" / "warn16"))
import check16all as c16  # noqa: E402  (cmp, unchanged)

MAPS = ("map25", "map25_m0", "map25_sd", "map25_ghap")
SM = an.A16 / "data" / "smoke"
LOG = an.A16 / "logs" / "a16.out"
EV1 = an.RUN / "eval_final_crps_e01"
EVP = an.RUN / "eval_pop_crps_e01"
A15M = an.A15_DIR / "data" / "main"
LIMITS = {"G1": "03:30:00", "G2": "01:05:00", "G3": "00:15:00", "G4": "00:15:00", "G5": "00:55:00", "G6": "00:25:00",
          "G7": "00:55:00", "G8": "01:25:00", "G9": "02:15:00", "G11": "04:05:00"}
STORED_WALL = {"G1": 8199, "G2": 2555, "G3": 541, "G4": 548, "G5": 2062, "G6": 942, "G7": 2124, "G8": 3273}


def find(folder, want):
    """{(d, h): (part path, row)} of the forecasts `want` among the part files of a folder (keys only)"""
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


def read(p, n, extra=()):
    with np.load(p) as z:
        o = {k: z[k][n] for k in MAPS + tuple(e for e in extra if e in z.files)}
        keys, cols = z["keys"], [str(c) for c in z["rec_cols"]]
        o["recs"] = {int(r[2]): r for r in z["recs"] if (int(r[0]), int(r[1])) == tuple(int(v) for v in keys[n][:2])}
        o["seam"] = float(z["seams"][n][2])
        o["info"] = [str(x) for x in z["info"]]
        if "map25_mk" in z.files:
            o["map25_mk"] = z["map25_mk"][n]
    return o, cols


def cmp(a, b):                                       # as a15/smoke_check.py (a14/r2/smoke_check.py)
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    same = (a.view(np.uint16) == b.view(np.uint16)) if a.dtype == np.float16 and b.dtype == np.float16 else (a == b)
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float(same.mean()),
                max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                p9999_abs=float(np.percentile(d, 99.99)) if d.size else 0.0)


def ok_map(c):                                       # A15 6.2(a)
    return c["finite_pattern_equal"] and c["bit_identical_share"] >= 0.286 and c["max_abs"] <= 2.25 and c["mean_abs"] <= 0.05


def ok_pix(c):                                       # A14a
    return c["finite_pattern_equal"] and c["mean_abs"] <= 0.05 and c["p9999_abs"] <= 2.5


def same16(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def rel_sums(A, S, cols, names):
    assert set(A["recs"]) == set(S["recs"]) and A["recs"], (sorted(A["recs"]), sorted(S["recs"]))
    out = {}
    for c in names:
        j = cols.index(c)
        out[c] = max(abs(A["recs"][g][j] - S["recs"][g][j]) / max(abs(S["recs"][g][j]), 1e-9) for g in S["recs"])
    return out


def compare(A, S, cols, members_mk):
    """the A15 6.2(a) comparison of one forecast; members_mk: also compare map25_mk member by member (c)"""
    maps = {f: cmp(A[f], S[f]) for f in MAPS}
    rel = rel_sums(A, S, cols, ("n", "sse", "crps", "var", "sae", "sum_err", "brier_35"))
    mk0 = same16(A["map25_mk"][0], A["map25_m0"])
    good = (all(ok_map(maps[f]) for f in ("map25", "map25_m0", "map25_sd")) and maps["map25_ghap"]["max_abs"] == 0.0
            and maps["map25_ghap"]["finite_pattern_equal"] and maps["map25_ghap"]["bit_identical_share"] == 1.0
            and all(rel[c] <= 0.005 for c in ("n", "sse", "crps", "var")) and mk0 and A["info"] == S["info"])
    out = {"maps": maps, "region_sums_max_rel": rel, "map25_mk0_equals_map25_m0": mk0, "info_equal": A["info"] == S["info"],
           "seam": A["seam"], "seam_reference": S["seam"]}
    if members_mk:
        mk = {f"m{m}": cmp(A["map25_mk"][m], S["map25_mk"][m]) for m in range(4)}
        good &= all(ok_map(c) for c in mk.values())
        out["map25_mk"] = mk
    return good, out


def step_a(T):
    pa = SM / "a" / "part_0.npz"
    with np.load(pa) as z:
        ka = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
        rho_a, base_a, plan = z["keys_rho"].tolist(), str(z["base"]), json.loads(str(z["a16"]))["plan"]
        st_a, sg_a = z["st_ens"], z["st_ghap"]
    assert ka == [(0, 1, 1), (0, 2, 2)] and base_a == "stored" and rho_a == [1.0, 1.0] and plan == "S_a", (ka, base_a, rho_a, plan)
    st = find(EV1, {(0, 1), (0, 2)})
    ok_all = True
    for n, (d, h, t) in enumerate(ka):
        A, cols = read(pa, n)
        S, cols_s = read(st[(d, h)][0], st[(d, h)][1])
        assert cols == cols_s and st[(d, h)][2] == t
        good, out = compare(A, S, cols, False)
        pers = rel_sums(A, S, cols, ("n", "sse_persist", "sae_persist"))
        with np.load(st[(d, h)][0]) as z:
            st_s, sg_s = z["st_ens"][st[(d, h)][1]], z["st_ghap"][st[(d, h)][1]]
        sg_same = same16(sg_a[n], sg_s)
        st_c = {f"m{m}": cmp(st_a[n][:, m], st_s[:, m]) for m in range(4)}
        m5 = cmp(np.load(SM / "a" / f"map5km_d{d}_h{h}.npy"), np.load(EV1 / f"map5km_d{d}_h{h}.npy"))
        good &= all(v == 0.0 for v in pers.values()) and sg_same and all(ok_pix(c) for c in st_c.values()) and ok_pix(m5)
        ok_all &= good
        T[f"a_{d}_{h}"] = {"pass": good, **out, "persistence_sums_max_rel": pers, "st_ghap_identical": sg_same,
                           "st_ens": st_c, "map5km": m5}
    return ok_all


def step_b(T):
    pb = SM / "b4" / "part_0.npz"
    with np.load(pb) as z:
        kb = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
        rho_b, base_b = z["keys_rho"].tolist(), str(z["base"])
    assert kb == [(2, 3, 5)] and base_b == "philox" and rho_b == [an.rho_star()], (kb, base_b, rho_b)
    x4 = np.load(SM / "b4" / "pop_0.npy")
    idx4 = np.load(SM / "b4" / "pop_idx.npy", mmap_mode="r")
    k16 = np.load(SM / "b16" / "keys16_s.npy")
    done16 = json.loads((SM / "b16" / "done16_s.json").read_text())
    x16 = np.load(SM / "b16" / "pop16_s.npy", mmap_mode="r")
    shape_ok = x4.shape == (1, len(idx4), 4) and x16.shape == (1, len(idx4), 16) and k16.tolist() == [[2, 3, 5]] and done16 == [0]
    c = {f"m{m}": c16.cmp(np.asarray(x16[0][:, m]), x4[0][:, m]) for m in range(4)}
    members_ok = shape_ok and all(ok_pix(v) for v in c.values())
    # references: GHAP and persistence identical to F1's (2, 3); stored F2 members (other noise, informative)
    B, cols = read(pb, 0)
    st = find(EV1, {(2, 3)})
    S, _ = read(st[(2, 3)][0], st[(2, 3)][1])
    ghap_ok = cmp(B["map25_ghap"], S["map25_ghap"])["bit_identical_share"] == 1.0
    pers = rel_sums(B, S, cols, ("n", "sse_persist", "sae_persist"))
    fp = find(EVP, {(2, 3)})
    k = int(fp[(2, 3)][0].stem.split("_")[1])
    old = np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[fp[(2, 3)][1]]
    vs_f2 = {f"m{m}": c16.cmp(x4[0][:, m], np.asarray(old[:, m])) for m in range(4)}
    good = members_ok and ghap_ok and all(v == 0.0 for v in pers.values())
    T["b_members0_3"] = {"pass": good, "forecast": [2, 3, 5], "pixels": int(len(idx4)), "shapes_ok": shape_ok, "members": c,
                         "map25_ghap_identical_to_F1": ghap_ok, "persistence_sums_max_rel_vs_F1": pers,
                         "informative_b4_vs_stored_F2_members": vs_f2}
    return good


def step_c(T):
    pc = SM / "c" / "part_0.npz"
    with np.load(pc) as z:
        kc = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
        rho_c, base_c = z["keys_rho"].tolist(), str(z["base"])
    assert kc == [(9, 2, 11)] and base_c == "philox" and rho_c == [an.rho_star()], (kc, base_c, rho_c)
    m = find(A15M, {(9, 2)})
    with np.load(m[(9, 2)][0]) as z:
        assert float(z["keys_rho"][m[(9, 2)][1]]) == an.rho_star() and str(z["base"]) == "philox"
    C, cols = read(pc, 0)
    M, cols_m = read(m[(9, 2)][0], m[(9, 2)][1])
    assert cols == cols_m
    good, out = compare(C, M, cols, True)
    T["c_9_2"] = {"pass": good, "reference": str(m[(9, 2)][0]), **out}
    return good


def parse_log(job):
    lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
    mk = [i for i, l in enumerate(lines) if l.startswith(f"=== A16 smoke job {job} start")]
    seg = lines[mk[-1]:] if mk else []
    pe = re.compile(r"rank \S+: start (\d+) lead (\d+): io wait (\d+)s gpu (\d+)s post (\d+)s .*\| (\d+)/(\d+) (\d+)s$")
    ps = re.compile(r"rank \S+: start (\d+) lead (\d+): io wait (\d+)s gpu (\d+)s \| (\d+)/(\d+) (\d+)s \| gpu first window "
                    r"([\d.]+)s median window ([\d.]+)s")
    step, rows = None, {}
    for l in seg:
        s = re.match(r"--- A16 smoke step (\w+) ", l)
        if s and " end " not in l:
            step = s.group(1)
            continue
        m = pe.search(l)
        if m and step:
            g = [int(v) for v in m.groups()]
            rows.setdefault(step, []).append(dict(start=g[0], lead=g[1], io=g[2], gpu=g[3], post=g[4], elapsed=g[7]))
            continue
        m = ps.search(l)
        if m and step:
            g = m.groups()
            rows.setdefault(step, []).append(dict(start=int(g[0]), lead=int(g[1]), io=int(g[2]), gpu=int(g[3]), elapsed=int(g[6]),
                                                  first_window=float(g[7]), median_window=float(g[8])))
    return (mk[-1] if mk else None), rows


def hms(s):
    s = int(math.ceil(s / 300.0) * 300)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:00"


def sec(t):
    h, m, s = (int(v) for v in t.split(":"))
    return 3600 * h + 60 * m + s


def cost(T, job):
    marker, rows = parse_log(job)
    info = {"log_marker_line": marker, "progress_lines": rows}
    need = {"a": 2, "b4": 1, "b16": 1, "c": 1}
    found = all(len(rows.get(k, [])) == v for k, v in need.items())
    if found:
        a1, a2 = rows["a"]
        warm4 = a2["io"] + a2["gpu"] + a2["post"]
        first4 = a1["io"] + a1["gpu"] + a1["post"]
        b4 = rows["b4"][0]
        pop_extra = max(0, (b4["io"] + b4["gpu"] + b4["post"]) - first4)
        b = rows["b16"][0]
        warm16 = b["io"] + b["gpu"] - b["first_window"] + b["median_window"]
        first16 = b["io"] + b["gpu"]
        boot = 300                                                   # start-up per process (model, dataset, workers), assumed
        proj = {}
        for g, p in an.EVAL_PLANS.items():
            if not g.startswith("G"):
                continue
            per_rank = math.ceil(p["n"] / p["world"])
            if p["members"] == 4:
                unit = warm4 + (pop_extra if p["pop"] else 0)
                wall = boot + per_rank * unit + (first4 - warm4)
            else:                                                    # 16-member crop-only: not measured (other window count)
                wall = STORED_WALL[g]
            proj[g] = dict(forecasts_per_rank=per_rank, wall_s=round(wall), gcd_h=round(p["world"] * wall / 3600, 2),
                           limit=LIMITS[g], within_limit=wall <= sec(LIMITS[g]),
                           basis="measured" if p["members"] == 4 else "stored wall clock (16-member crop-only layout not measured)")
            if wall > sec(LIMITS[g]):
                proj[g]["suggested_limit"] = hms(1.5 * wall)
        wall11 = boot + 8 * warm16 + (first16 - warm16)
        proj["G11"] = dict(forecasts_per_task=8, tasks=45, wall_s=round(wall11), gcd_h=round((boot * 45 + 358 * warm16 + 45 * (first16 - warm16)) / 3600, 2),
                           limit=LIMITS["G11"], within_limit=wall11 <= sec(LIMITS["G11"]), basis="measured")
        if wall11 > sec(LIMITS["G11"]):
            proj["G11"]["suggested_limit"] = hms(1.5 * wall11)
        total = sum(v["gcd_h"] for v in proj.values())
        info.update(seconds_per_warm_forecast_4_members=warm4, seconds_first_forecast_4_members=first4,
                    extra_seconds_pop=pop_extra, seconds_per_warm_forecast_16_members=round(warm16, 1),
                    seconds_first_forecast_16_members=first16, io_over_gpu_4=a2["io"] / max(a2["gpu"], 1),
                    io_over_gpu_16=b["io"] / max(b["gpu"], 1), projection=proj, projected_total_gcd_h=round(total, 1),
                    cap_gcd_h=400, within_cap=total + 2 <= 400)
    else:
        info["error"] = f"progress lines not found in {LOG} for job {job}: {sorted(rows)}"
    T["cost"] = {"pass": found, **info}
    return found


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    R = {"what": __doc__, "job": job, "tests": {}}
    T = R["tests"]
    for name, fn in (("a", step_a), ("b", step_b), ("c", step_c)):
        try:
            fn(T)
        except Exception as e:                                       # one failing step must not hide the others
            T[f"{name}_error"] = {"pass": False, "error": repr(e)}
    try:
        cost(T, job)
    except Exception as e:
        T["cost"] = {"pass": False, "error": repr(e)}
    g15 = an.golden_hashes() == json.loads((an.A15_DIR / "golden.json").read_text())
    g22 = an.golden22_hashes() == json.loads((an.A16 / "golden22.json").read_text())
    T["golden_gpu_node"] = {"pass": g15 and g22, "a15_golden": g15, "golden22": g22}
    R["code_sha256"] = an.run_code_sha()
    T["code_same_as_check_cpu"] = {"pass": an.same_code(an.RES / "check_cpu.json")}
    R["all_passed"] = bool(all(v.get("pass") is True for v in T.values()) and {"a_0_1", "a_0_2", "b_members0_3", "c_9_2"} <= set(T))
    an.save_json(an.RES / "smoke_check.json", R)
    print(json.dumps(an.jsonable({k: v.get("pass") for k, v in T.items()})),
          "ALL PASSED" if R["all_passed"] else "SMOKE GATE FAILED", flush=True)
    if "projection" in T.get("cost", {}):
        print("cost", json.dumps(an.jsonable({k: T["cost"][k] for k in T["cost"] if k not in ("progress_lines", "projection")})), flush=True)
        for g, v in T["cost"]["projection"].items():
            print(" ", g, json.dumps(v), flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
