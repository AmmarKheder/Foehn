#!/usr/bin/env python3
"""A20 CPU checks before the collection (a20/A20_draft.md section 6.1), in the job container (numpy 1.26.3), CPU only:
  SOURCE   common20.innovation and common20.coeffs have the same source text as a16/arnoise22.py (the generator of G2 / G11)
  NOISE    common20.z_tiles (one draw of eps_1..3 for leads 1, 2, 3) is byte-equal to arnoise22.ARNoise(2022, d, h, 0.59,
           'philox', 4 members for m <= 3, 16 for m >= 4).tile(m, tr, tc) on 48 tiles (members 0-7, leads 1-3, 6 start days)
  GOLD     arnoise22.golden22_hashes() equals a16/golden22.json, and common20 reproduces its two z hashes
  LAYOUT   the global map of common20.z_maps equals fullflow_global.full_map(ARNoise, m) (the map the evaluation cuts) and
           three of the model's windows cut by the unchanged tile_window (noise pixel (r, c) = 1 km pixel (r, c))
  FEATURES feats_at against direct box means at 400 rows of the A20 sample (z, 2 x 2 block, boxes 65 and 257 px)
  COST     wall seconds of one collection task of each kind in this container, the projection of the plan and its cost
    python -B check20.py -> code/out/check20.json (exit 1 if a check fails)"""
import common20 as C  # noqa: I001  (first: hooks)

C.install("members")
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(C.ROOT / "a16"))
import arnoise22 as an  # noqa: E402  (imported unchanged; asserts numpy 1.26.3)

egg = an.egg
R = {"what": __doc__, "numpy": np.__version__, "tests": {}}


def check(name, cond, **info):
    R["tests"][name] = dict(passed=bool(cond), **info)
    C.log(("PASS " if cond else "FAIL ") + name, json.dumps(C.jsonable(info))[:600])


def u16(a):
    return np.ascontiguousarray(a).view(np.uint16)


def main():
    t0 = time.time()
    # ---- SOURCE
    same = {f: inspect.getsource(getattr(C, f)) == inspect.getsource(getattr(an, f)) for f in ("innovation", "coeffs")}
    check("SOURCE", all(same.values()), same_source_text=same)
    # ---- NOISE
    items = [(d, m, tr, tc) for d in (0, 2, 120, 190, 300, 356) for m, (tr, tc) in
             zip(range(8), [(0, 0), (14, 35), (28, 70), (7, 30), (20, 50), (3, 66), (10, 40), (25, 5)])]
    bad, n = [], 0
    for d, m, tr, tc in items:
        mine = C.z_tiles(d, m, tr, tc, (1, 2, 3))
        for h in (1, 2, 3):
            ref = an.ARNoise(2022, d, h, C.RHO, "philox", 4 if m < 4 else 16).tile(m, tr, tc)
            n += 1
            if not (mine[h].dtype == np.float16 and np.array_equal(u16(mine[h]), u16(ref))):
                bad.append([d, m, h, tr, tc])
    check("NOISE", not bad and n == 144, tiles_compared=n, mismatches=bad)
    # ---- GOLD
    gold = json.loads((C.ROOT / "a16" / "golden22.json").read_text())
    g_a16 = an.golden22_hashes() == gold
    g1 = C.sha(C.z_tiles(0, 0, 7, 30, (7,))[7]) == gold["z_2022_0_lead7_rho0.59_philox_m0_tile7_30/f16"]
    g2 = C.sha(C.z_tiles(2, 15, 14, 35, (3,))[3]) == gold["z_2022_2_lead3_rho0.59_philox_m15_tile14_35/f16"]
    check("GOLD", g_a16 and g1 and g2, a16_golden22_reproduced=g_a16, z_m0_lead7=g1, z_m15_lead3=g2)
    # ---- LAYOUT
    t1 = time.time()
    padded = C.z_maps(2, 1, (3,), padded=True)[3]
    nz = an.ARNoise(2022, 2, 3, C.RHO, "philox", 4)
    full = egg.full_map(nz, 1)
    same_full = full.shape == (C.GH, C.GW) and np.array_equal(u16(full), u16(padded[:C.GH, :C.GW]))
    del full
    win = egg.windows(np.load(egg.G / "meta.npz")["land"])
    wins = [win[0], win[10], win[-1]]
    same_win = []
    for k0, j0 in wins:
        r0, c0 = 25 * int(k0), 25 * int(j0)
        cut = egg.tile_window(nz, 1, r0, c0, egg.PH, egg.PW)
        same_win.append(bool(np.array_equal(u16(cut), u16(padded[r0:r0 + egg.PH, c0:c0 + egg.PW]))))
    check("LAYOUT", same_full and all(same_win), full_map_equal=same_full, windows=[[int(a), int(b)] for a, b in wins],
          window_cuts_equal=same_win, seconds=round(time.time() - t1, 1))
    # ---- FEATURES
    pos = C.positions()
    rng = np.random.default_rng(7)
    sel = np.sort(rng.choice(len(pos["r"]), 400, replace=False))
    r, c = pos["r"][sel], pos["c"][sel]
    z = padded[:C.GH, :C.GW]
    f = C.feats_at(z, r, c)
    zz = lambda a: np.asarray(a, np.float64)     # noqa: E731
    dev = {"pixel": float(np.abs(f[:, 0] - zz(z[r, c])).max())}
    rb, cb = r // 2 * 2, c // 2 * 2
    blk = np.array([zz(z[a:a + 2, b:b + 2]).sum() / 2.0 for a, b in zip(rb, cb)])
    dev["block2"] = float(np.abs(f[:, 1] - blk).max())
    for w in (65, 257):
        j = C.WID.index(w)                                          # feature column 2 + j
        h = w // 2
        d = []
        for a, b in zip(r, c):
            box = zz(z[max(a - h, 0):min(a + h + 1, C.GH), max(b - h, 0):min(b + h + 1, C.GW)])
            d.append(box.sum() / np.sqrt(box.size))
        dev[f"box{w}"] = float(np.abs(f[:, 2 + j] - np.array(d)).max())
    check("FEATURES", all(v <= 1e-4 for v in dev.values()) and np.isfinite(f).all(), max_abs_dev=dev, rows=len(sel))
    del padded, z, nz
    # ---- COST (one task of each kind, the collector's own functions)
    import collect20 as col
    col.setup()
    t2 = time.time()
    s1 = col.noise_task((1, 2), dry=True)
    t3 = time.time()
    s2 = col.noise_task((1, 5), dry=True)
    t4 = time.time()
    s3 = col.member_task((1, 0), dry=True)
    t5 = time.time()
    a, b, m = t3 - t2, t4 - t3, t5 - t4
    cpu_s = 4 * C.ND * a + 4 * C.ND * b + 2 * C.ND * m
    R["cost"] = dict(task_g2_members_leads123_s=round(a, 1), task_null_member_leads13_s=round(b, 1),
                     task_members_extraction_s=round(m, 1), tasks=dict(g2=4 * C.ND, null=4 * C.ND, members=2 * C.ND),
                     projected_cpu_seconds=round(cpu_s), projected_wall_minutes_24_workers=round(cpu_s / 24 / 60, 1),
                     billing_rule="core-hours = elapsed hours x max(cores, memory GB / 2) (LUMI-C small, assumed)",
                     job=dict(cores=32, mem_gb=210), projected_core_hours=round(cpu_s / 24 / 3600 * max(32, 210 / 2), 1),
                     sample_checks=[s1, s2, s3])
    C.log("COST", json.dumps(R["cost"])[:800])
    R["all_passed"] = all(v["passed"] for v in R["tests"].values())
    R["seconds"] = round(time.time() - t0)
    R["code_sha256"] = {n: C.sha_file(C.CODE / n) for n in ("common20.py", "check20.py", "stpos20.py", "collect20.py")}
    C.save_json(C.OUT / "check20.json", R)
    C.log("ALL PASSED" if R["all_passed"] else "SOME CHECKS FAILED", f"{R['seconds']} s")
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
