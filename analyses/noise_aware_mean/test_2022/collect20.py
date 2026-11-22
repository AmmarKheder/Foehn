#!/usr/bin/env python3
"""A20 step 1 (CPU only, members and regenerated noise only; no truth, no station file): the noise features of every
member, start day and lead of G2, and the members of G2 and G11, at the rows of a20/data/pix20.npz (stpos20.py).
Writes a20/data:
  feat_g2.npy   (3 leads [1, 2, 3], 179 start days, 4 members [0-3], NALL rows, 12) float16: features (common20.feats_at)
                of z(m, d, h) = arnoise22.ARNoise(2022, d, h, 0.59, 'philox', 4).tile(...) laid out as the evaluation cuts it
                (check20 NOISE, LAYOUT); lead 2 serves only the test of the earlier leads' fields at lead 3
  feat_null.npy (2 leads [1, 3], 179, 4 members [4-7], NNULL rows (pix20 'nullrows'), 12) float16: the same for members
                4-7 of the same keys (independent noise; the noise of G11's members 4-7): the null 'another member'
  mem_g2.npy    (179, 2 leads [1, 3], NALL, 4) float16: G2's members at the rows (bit copies of RUN/eval_pop_crps_e01_ar
                pop_<r>.npy rows; only 'keys' is read from its part files)
  mem_g11.npy   (179, 2, NALL, 16) float16: G11's sixteen members (bit copies of a16/data/run16 pop16_<tag>.npy rows)
  meta20.npz    keys, targets, per-task checks, read-back checks, sha256 of the four arrays, timing
The arrays are written in place as <name>.tmp_a20.npy by the workers and renamed when complete. No truth: the audit hook
of common20 ('members') refuses any path with 'ghap', 'station', 'monitor' or 'openaq'.
    python -B -u collect20.py [--workers 24]
    python -B -u collect20.py --smoke      (2 tasks into a20/data/smoke, removed at the end; any numpy)"""
import common20 as C  # noqa: I001  (first: hooks)

C.install("members")
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

S = {}
NAMES = ("feat_g2", "feat_null", "mem_g2", "mem_g11")


def setup(work=None):
    pos = C.positions()
    S.update(r=pos["r"], c=pos["c"], rows=pos["rows"], nullrows=pos["nullrows"], NALL=len(pos["rows"]),
             NNULL=len(pos["nullrows"]), work=Path(work) if work else C.DATA)
    S["K2"], S["K11"] = C.g2_index(), C.g11_index()
    want = {(d, h) for d in C.DAYS for h in C.LEADS}
    assert set(S["K2"]) == want and set(S["K11"]) == want, "G2 / G11 do not hold the 358 forecasts"
    assert all(S["K2"][k][2] == S["K11"][k][2] for k in want), "G2 and G11 targets differ"


def shapes():
    NALL, NNULL = S["NALL"], S["NNULL"]
    return {"feat_g2": ((3, C.ND, 4, NALL, C.NF), np.float16), "feat_null": ((2, C.ND, 4, NNULL, C.NF), np.float16),
            "mem_g2": ((C.ND, 2, NALL, 4), np.float16), "mem_g11": ((C.ND, 2, NALL, 16), np.float16)}


def tmp(name):
    return S["work"] / f"{name}.tmp_a20.npy"


def noise_task(job, dry=False):
    """features of z(m, d, h) at the rows, h = 1, 2, 3 (m <= 3) or 1, 3 (m >= 4), from one draw of eps_1..3"""
    j, m = job
    d = C.DAYS[j]
    t0 = time.time()
    leads = C.LEADS3 if m < 4 else C.LEADS
    maps = C.z_maps(d, m, leads)
    info = {"j": j, "m": m}
    for h in leads:
        if m < 4:
            f = C.feats_at(maps[h], S["r"], S["c"])
        else:
            f = C.feats_at(maps[h], S["r"][S["nullrows"]], S["c"][S["nullrows"]])
        f16 = f.astype(np.float16)
        info[f"h{h}"] = dict(finite=bool(np.isfinite(f16).all()), sd_box129=float(f[:, 8].std()), sd_pixel=float(f[:, 0].std()),
                             mean_box129=float(f[:, 8].mean()), maxabs=float(np.abs(f).max()))
        if not dry:
            if m < 4:
                a = np.load(tmp("feat_g2"), mmap_mode="r+")
                a[C.LEADS3.index(h), j, m] = f16
            else:
                a = np.load(tmp("feat_null"), mmap_mode="r+")
                a[C.LEADS.index(h), j, m - 4] = f16
            a.flush()
            del a
        del f, f16
    del maps
    info["seconds"] = round(time.time() - t0, 1)
    return "noise", info


def member_task(job, dry=False):
    """G2's four and G11's sixteen members of forecast (d, h) at the rows (bit copies)"""
    j, li = job
    key = (C.DAYS[j], C.LEADS[li])
    t0 = time.time()
    r2, n2, t = S["K2"][key]
    x4 = np.asarray(np.load(C.G2D / f"pop_{r2}.npy", mmap_mode="r")[n2])
    assert x4.shape == (C.NPOP, 4) and x4.dtype == np.float16, x4.shape
    f11, n11, t11 = S["K11"][key]
    x16 = np.asarray(np.load(f11, mmap_mode="r")[n11])
    assert x16.shape == (C.NPOP, 16) and x16.dtype == np.float16 and t11 == t, (x16.shape, t11, t)
    a4, a16 = x4[S["rows"]], x16[S["rows"]]
    info = {"j": j, "li": li, "target": t, "finite_g2": float(np.isfinite(a4).all(1).mean()),
            "finite_g11": float(np.isfinite(a16).all(1).mean())}
    if not dry:
        for nm, v in (("mem_g2", a4), ("mem_g11", a16)):
            a = np.load(tmp(nm), mmap_mode="r+")
            a[j, li] = v
            a.flush()
            del a
    info["seconds"] = round(time.time() - t0, 1)
    return "members", info


def run(job):
    kind, j = job
    return noise_task(j) if kind == "noise" else member_task(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if not a.smoke:
        assert np.__version__ == "1.26.3", f"A20 collects in the job container (numpy 1.26.3), not {np.__version__}"
        chk = json.loads((C.OUT / "check20.json").read_text())
        assert chk.get("all_passed") is True, "check20 has not passed"
        assert chk["code_sha256"]["collect20.py"] == C.sha_file(C.CODE / "collect20.py"), "collect20 changed after check20"
        assert chk["code_sha256"]["common20.py"] == C.sha_file(C.CODE / "common20.py"), "common20 changed after check20"
        assert str(C.positions()["numpy"]) == "1.26.3", "pix20.npz was not drawn in the job container"
    work = C.DATA / "smoke" if a.smoke else C.DATA
    if a.smoke:
        work.mkdir(exist_ok=True)
    setup(work)
    for nm, (shp, dt) in shapes().items():
        assert not (work / f"{nm}.npy").exists(), f"{nm}.npy exists already: A20 collects once"
        np.lib.format.open_memmap(tmp(nm), mode="w+", dtype=dt, shape=shp).flush()
    if a.smoke:
        jobs = [("noise", (0, 0)), ("noise", (178, 7)), ("members", (0, 0)), ("members", (178, 1))]
    else:
        jobs = [("noise", (j, m)) for j in range(C.ND) for m in range(8)] + [("members", (j, li)) for j in range(C.ND) for li in range(2)]
    C.log(f"{len(jobs)} tasks, {a.workers} workers, rows {S['NALL']} (null {S['NNULL']})")
    infos = {"noise": [], "members": []}
    with get_context("fork").Pool(a.workers, maxtasksperchild=40) as pool:
        for n, (kind, info) in enumerate(pool.imap_unordered(run, jobs, chunksize=1)):
            infos[kind].append(info)
            if n % 25 == 0 or n == len(jobs) - 1:
                C.log(f"task {n + 1}/{len(jobs)} {kind} {json.dumps(info)[:200]} {time.time() - t0:.0f} s")
    nz = infos["noise"]
    chk = {"tasks_done": len(nz) + len(infos["members"]) == len(jobs),
           "features_finite": all(v[f"h{h}"]["finite"] for v in nz for h in (1, 2, 3) if f"h{h}" in v),
           "sd_box129_in_0.9_1.1": all(0.9 < v[f"h{h}"]["sd_box129"] < 1.1 for v in nz for h in (1, 2, 3) if f"h{h}" in v),
           "sd_pixel_in_0.98_1.02": all(0.98 < v[f"h{h}"]["sd_pixel"] < 1.02 for v in nz for h in (1, 2, 3) if f"h{h}" in v)}
    # ---- read-back: members bit-identical to their sources, features recomputed bit for bit
    rb = []
    for j, li in ([(0, 0), (178, 1)] if a.smoke else [(0, 0), (89, 1), (178, 0), (177, 1)]):
        _, i2 = member_task((j, li), dry=True)
        key = (C.DAYS[j], C.LEADS[li])
        r2, n2, _ = S["K2"][key]
        f11, n11, _ = S["K11"][key]
        m2 = np.array(np.load(tmp("mem_g2"), mmap_mode="r")[j, li])
        m11 = np.array(np.load(tmp("mem_g11"), mmap_mode="r")[j, li])
        s2 = np.asarray(np.load(C.G2D / f"pop_{r2}.npy", mmap_mode="r")[n2])[S["rows"]]
        s11 = np.asarray(np.load(f11, mmap_mode="r")[n11])[S["rows"]]
        rb.append(bool(np.array_equal(m2.view(np.uint16), s2.view(np.uint16)) and np.array_equal(m11.view(np.uint16), s11.view(np.uint16))))
        del m2, m11, s2, s11
    chk["members_bit_identical_to_source"] = rb
    for j, m in ([(0, 0)] if a.smoke else [(0, 0), (178, 7)]):
        d = C.DAYS[j]
        h = 1 if m < 4 else 3
        z = C.z_maps(d, m, (h,))[h]
        if m < 4:
            f = C.feats_at(z, S["r"], S["c"]).astype(np.float16)
            st = np.array(np.load(tmp("feat_g2"), mmap_mode="r")[C.LEADS3.index(h), j, m])
        else:
            f = C.feats_at(z, S["r"][S["nullrows"]], S["c"][S["nullrows"]]).astype(np.float16)
            st = np.array(np.load(tmp("feat_null"), mmap_mode="r")[C.LEADS.index(h), j, m - 4])
        chk[f"features_recomputed_bit_identical_j{j}_m{m}"] = bool(np.array_equal(st.view(np.uint16), f.view(np.uint16)))
        del z, f, st
    # distinct fields: member 0 vs 1 and day 0 vs day 1 differ
    fg = np.load(tmp("feat_g2"), mmap_mode="r")
    a00, a01, a10 = (np.array(fg[0, j_, m_, :1000]) for j_, m_ in ((0, 0), (0, 1), (1, 0)))
    del fg
    chk["fields_distinct"] = bool(not np.array_equal(a00, a01) and not np.array_equal(a00, a10))
    C.log("checks", json.dumps(chk))
    ok = all(v if isinstance(v, bool) else all(v) for v in chk.values())
    if a.smoke:
        shutil.rmtree(work)
        C.log("smoke done (folder removed)", "PASS" if ok else "FAIL", f"{time.time() - t0:.0f} s")
        return 0 if ok else 1
    assert ok, chk
    for nm in NAMES:
        C.finalize_memmap(tmp(nm), C.DATA / f"{nm}.npy")
    with ThreadPoolExecutor(4) as ex:
        hashes = dict(zip(NAMES, ex.map(lambda nm: C.sha_file(C.DATA / f"{nm}.npy"), NAMES)))
    keys2 = np.array([[d, h, *S["K2"][(d, h)]] for d in C.DAYS for h in C.LEADS], np.int64)
    keys11 = np.array([[d, h, S["K11"][(d, h)][1], S["K11"][(d, h)][2]] for d in C.DAYS for h in C.LEADS], np.int64)
    files11 = np.array([S["K11"][(d, h)][0] for d in C.DAYS for h in C.LEADS])
    C.save_npz(C.DATA / "meta20.npz", keys_g2=keys2, keys_g11=keys11, files_g11=files11, days=np.array(C.DAYS, np.int64),
               sha256=np.array(json.dumps(hashes)), checks=np.array(json.dumps(chk)), noise_tasks=np.array(json.dumps(nz)),
               member_tasks=np.array(json.dumps(infos["members"])), seconds=np.float64(time.time() - t0),
               numpy=np.array(np.__version__), code_sha256=np.array(json.dumps({n: C.sha_file(C.CODE / n) for n in
                                                                                  ("common20.py", "collect20.py")})))
    C.log("collection done", json.dumps(hashes), f"{time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
