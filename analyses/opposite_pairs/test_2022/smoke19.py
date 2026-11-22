#!/usr/bin/env python3
"""A19 smoke gate (a19/A19_draft.md section 6.2), run on the GPU node at the end of code/smoke.sh; main.sh and eval19.py
--plan main refuse to start unless data/smoke_check.json says all_passed with the same code. Identity checks only: no
skill, error or pair statistic of any forecast is computed here, and no member is compared with any truth.
  E1  the negated members run alone (data/smoke_alone, member ids 4..7) equal the negated members of an eight-member run
      (data/smoke_eight, ids 0..7) on the forecast (2, 3) at the populated pixels, member by member, with the rule of A14a
      (same finite pattern, mean |diff| <= 0.05 ug m-3, 99.99th percentile <= 2.5); bit-identity is reported
  E2  members 0-3 of the eight-member run equal G2's stored members of (2, 3) (RUN/eval_pop_crps_e01_ar) with the same rule
  E3  pop_ghap, cell40 and map25_ghap of both smoke folders bit-identical to G2's (2, 3) (and (2, 1) for smoke_alone);
      pop_idx equal to G2's; keys, keys_rho = rho*, base philox, member_ids, info, manifests as planned; no station column
  E4  the negated members are not copies of the original members (bit-identical share of the finite pixels of (2, 3)
      < 0.5; a boolean only): detects only a run reproducing members 0-3, not a lost negation
  cost from data/a19_gpu.out (the progress lines after '=== A19 smoke job <id> start'): seconds of the first (2, 1) and of the
      warm (2, 3) forecast of 4 negated members, and of the one eight-member forecast; projected wall clock of the main run
      (64 ranks, at most 6 forecasts per rank: start-up 300 s + first + 5 x warm) against its limit 0:55:00, and projected
      GCD-hours (64 x wall) plus the smoke's own GCD-hours (wall of the smoke job so far, rounded up to 1.0) against the cap
      of 60; the gate fails if either is exceeded
  golden hashes recomputed on this node equal to check19.json's golden19; the code equals the code check19 checked
    python smoke19.py <smoke job id> -> data/smoke_check.json (exit 1 if a gating check fails)"""
import noise19 as nz  # noqa: I001  (first)
import json
import math
import re
import sys
import time

import numpy as np

SM_A, SM_E = nz.DATA / "smoke_alone", nz.DATA / "smoke_eight"
T_START = time.time()


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


def cmp(a, b):                                       # as a16/smoke_check.py
    a, b = np.asarray(a), np.asarray(b)
    fa, fb = np.isfinite(a.astype(np.float32)), np.isfinite(b.astype(np.float32))
    both = fa & fb
    d = np.abs(a.astype(np.float32)[both] - b.astype(np.float32)[both])
    same = (a.view(np.uint16) == b.view(np.uint16)) if a.dtype == np.float16 and b.dtype == np.float16 else (a == b)
    return dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float(same.mean()),
                max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                p9999_abs=float(np.percentile(d, 99.99)) if d.size else 0.0)


def ok_pix(c):                                       # A14a
    return c["finite_pattern_equal"] and c["mean_abs"] <= 0.05 and c["p9999_abs"] <= 2.5


def same16(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def rank_of(p):
    return int(p.stem.split("_")[1])


def identity(T, rho):
    al = find(SM_A, {(2, 1), (2, 3)})
    ei = find(SM_E, {(2, 3)})
    g2 = find(nz.G2_DIR, {(2, 1), (2, 3)})
    pa, na, _ = al[(2, 3)]
    pe, ne, _ = ei[(2, 3)]
    pg, ng, _ = g2[(2, 3)]
    XA = np.load(SM_A / f"pop_{rank_of(pa)}.npy", mmap_mode="r")[na]                     # (pixels, 4): ids 4..7 alone
    XE = np.load(SM_E / f"pop_{rank_of(pe)}.npy", mmap_mode="r")[ne]                     # (pixels, 8): ids 0..7
    XG = np.load(nz.G2_DIR / f"pop_{rank_of(pg)}.npy", mmap_mode="r")[ng]                # (pixels, 4): G2 ids 0..3
    shapes = XA.shape[1] == 4 and XE.shape[1] == 8 and XG.shape[1] == 4 and XA.shape[0] == XE.shape[0] == XG.shape[0]
    e1 = {f"id{4 + m}": cmp(np.asarray(XE[:, 4 + m]), np.asarray(XA[:, m])) for m in range(4)}
    e2 = {f"id{m}": cmp(np.asarray(XE[:, m]), np.asarray(XG[:, m])) for m in range(4)}
    idx_ok = all(np.array_equal(np.load(F / "pop_idx.npy"), np.load(nz.G2_DIR / "pop_idx.npy")) for F in (SM_A, SM_E))
    truth = {}
    for nm, F, fd in (("alone", SM_A, al), ("eight", SM_E, ei)):
        for k, (p, n, t) in fd.items():
            pg_, ng_, tg_ = g2[k]
            r, rg = rank_of(p), rank_of(pg_)
            with np.load(p) as z, np.load(pg_) as zg:
                m25 = same16(z["map25_ghap"][n], zg["map25_ghap"][ng_])
                st0 = z["st_ens"].shape[1:] == (0, len(nz.PLANS[("smoke_alone" if nm == "alone" else "smoke_eight")]["mids"]))
            truth[f"{nm}_{k[0]}_{k[1]}"] = dict(
                target_equal=t == tg_,
                pop_ghap=same16(np.load(F / f"pop_ghap_{r}.npy", mmap_mode="r")[n], np.load(nz.G2_DIR / f"pop_ghap_{rg}.npy", mmap_mode="r")[ng_]),
                cell40=bool(np.array_equal(np.load(F / f"cell40_{r}.npy", mmap_mode="r")[n][1], np.load(nz.G2_DIR / f"cell40_{rg}.npy", mmap_mode="r")[ng_][1],
                                           equal_nan=True)),
                map25_ghap=m25, no_station_columns=bool(st0))
    meta = {}
    for nm, F, plan, fd in (("alone", SM_A, "smoke_alone", al), ("eight", SM_E, "smoke_eight", ei)):
        man = json.loads((F / "a19_run.json").read_text())
        p = next(iter(fd.values()))[0]
        with np.load(p) as z:
            meta[nm] = dict(plan=man["plan"] == plan, member_ids=[int(m) for m in z["member_ids"]] == list(nz.PLANS[plan]["mids"]),
                            rho=all(float(r) == rho for r in z["keys_rho"]), base=str(z["base"]) == "philox",
                            info=[str(x) for x in z["info"]][5] == str(len(nz.PLANS[plan]["mids"])),
                            keys=sorted(tuple(int(v) for v in k) for k in z["keys"].tolist()) == sorted((d, h, t) for (d, h), (_, _, t) in fd.items()))
    fin = np.isfinite(np.asarray(XG[:, 0], np.float32))
    e4 = all(float((np.asarray(XA[:, m]).view(np.uint16) == np.asarray(XG[:, m]).view(np.uint16))[fin].mean()) < 0.5 for m in range(4))
    T["E1_alone_equals_eight"] = {"pass": bool(shapes and all(ok_pix(c) for c in e1.values())), "members": e1,
                                  "bit_identical": all(c["bit_identical_share"] == 1.0 for c in e1.values())}
    T["E2_eight_members0_3_equal_G2"] = {"pass": bool(shapes and all(ok_pix(c) for c in e2.values())), "members": e2,
                                         "bit_identical": all(c["bit_identical_share"] == 1.0 for c in e2.values())}
    T["E3_truth_and_format"] = {"pass": bool(idx_ok and all(all(v.values()) for v in truth.values()) and all(all(v.values()) for v in meta.values())),
                                "pop_idx_equal_G2": idx_ok, "truth": truth, "meta": meta}
    T["E4_negated_differ_from_originals"] = {"pass": bool(e4)}


def parse_log(job):
    lines = nz.GPU_LOG.read_text(errors="replace").splitlines() if nz.GPU_LOG.exists() else []
    mk = [i for i, l in enumerate(lines) if l.startswith(f"=== A19 smoke job {job} start")]
    seg = lines[mk[-1]:] if mk else []
    pe = re.compile(r"rank \S+: start (\d+) lead (\d+): io wait (\d+)s gpu (\d+)s post (\d+)s .*\| (\d+)/(\d+) (\d+)s$")
    step, rows = None, {}
    for l in seg:
        s = re.match(r"--- A19 smoke step (\w+) ", l)
        if s and " end " not in l:
            step = s.group(1)
            continue
        m = pe.search(l)
        if m and step:
            g = [int(v) for v in m.groups()]
            rows.setdefault(step, []).append(dict(start=g[0], lead=g[1], io=g[2], gpu=g[3], post=g[4], elapsed=g[7]))
    t0 = None
    if mk:
        mt = re.search(r"epoch (\d+)", lines[mk[-1]])
        t0 = int(mt.group(1)) if mt else None
    return (mk[-1] if mk else None), rows, t0


def sec(t):
    h, m, s = (int(v) for v in t.split(":"))
    return 3600 * h + 60 * m + s


def cost(T, job):
    marker, rows, t0 = parse_log(job)
    info = {"log_marker_line": marker, "progress_lines": rows}
    found = len(rows.get("alone", [])) == 2 and len(rows.get("eight", [])) == 1
    if found:
        a1, a2 = rows["alone"]
        first, warm = (a1["io"] + a1["gpu"] + a1["post"]), (a2["io"] + a2["gpu"] + a2["post"])
        e = rows["eight"][0]
        boot = 300                                                   # start-up per process (model, dataset, workers), assumed
        n, world = nz.PLANS["main"]["n"], nz.PLANS["main"]["world"]
        per_rank = math.ceil(n / world)
        wall = boot + first + (per_rank - 1) * warm
        gcd_h = world * wall / 3600
        smoke_h = max(1.0, (time.time() - t0) / 3600) if t0 else 1.0  # the smoke's own GCD-hours (1 GCD), at least 1.0
        info.update(seconds_first_forecast_4_negated=first, seconds_warm_forecast_4_negated=warm,
                    seconds_first_forecast_8_members=e["io"] + e["gpu"] + e["post"], io_over_gpu=a2["io"] / max(a2["gpu"], 1),
                    reference_G2_warm_forecast_s=413, smoke_gcd_h_counted=round(smoke_h, 2),
                    main=dict(forecasts=n, ranks=world, forecasts_per_rank=per_rank, wall_s=round(wall), gcd_h=round(gcd_h, 2),
                              limit=nz.LIMIT_MAIN, within_limit=wall <= sec(nz.LIMIT_MAIN), cap_gcd_h=nz.CAP,
                              projected_total_gcd_h=round(gcd_h + smoke_h, 2), within_cap=gcd_h + smoke_h <= nz.CAP,
                              worst_case_at_limit_gcd_h=round(world * sec(nz.LIMIT_MAIN) / 3600 + smoke_h, 2)))
        found = info["main"]["within_limit"] and info["main"]["within_cap"]
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
    T["golden_gpu_node"] = {"pass": nz.golden19() == chk.get("golden19")}
    R["code_sha256"] = nz.run_code_sha()
    T["code_same_as_check19"] = {"pass": chk.get("all_passed") is True and chk.get("code_sha256") == R["code_sha256"]}
    R["all_passed"] = bool(all(v.get("pass") is True for v in T.values()) and {
        "E1_alone_equals_eight", "E2_eight_members0_3_equal_G2", "E3_truth_and_format", "E4_negated_differ_from_originals",
        "cost"} <= set(T))
    nz.save_json(nz.SMOKE_JSON, R)
    print(json.dumps(nz.jsonable({k: v.get("pass") for k, v in T.items()})), "ALL PASSED" if R["all_passed"] else "SMOKE GATE FAILED",
          flush=True)
    if "main" in T.get("cost", {}):
        print("cost", json.dumps(nz.jsonable({k: T["cost"][k] for k in T["cost"] if k != "progress_lines"})), flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
