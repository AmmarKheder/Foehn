#!/usr/bin/env python3
"""A16 section 4 point 5 and Addendum 1 point 1 (protocol_amendments/A16_test_year_evolving_noise.md): the inputs of the
replication of A15's statistics for one year and one pair of run folders (arm, stored), CPU. It generalises the input part
of A15's scorer (a15/prep.py: cells and their hash; a15/score.py: load_arm, load_stored_recs and the pairing asserts) to a
year and a pair of folders, and changes nothing else.
  cells   the populated 0.25 deg cells of episodes_diag/data/ext_<year>.npz (revise.load, read only) that are complete
          (revise 'ok': truth of the start day, map25_ghap, member 0 and mean finite at every lead) in every run of the
          stored extract (2021: the 45 stored runs, as a15/prep.py; 2022: the 179 runs of F1) AND finite in every scored
          forecast of the arm (map25_mk, all four members, and map25); ordered by 10-degree tile as revise.main_s12 and
          a15/arnoise.episodes_2021
  starts  the start days for which the arm holds leads 1-7 and nothing else (2021: must be the 30 Conf starts of A15;
          2022: must be the 179 starts of F1)
  truth   map25_ghap of the part files on the target days (stored and arm parts asserted bit-identical to the extract), and
          pm.npy through the extract for the start day d and the days d + 8, d + 9 (inside the year, asserted by score22)
  stored  map25_m0 (member 0) and map25 (mean) of the stored part files, asserted bit-identical to the extract
  arm     map25_mk (member m = Xa[:, m]) and map25 of the arm part files; base and keys_rho asserted as in a15/score.py
          (one base 'philox', rho = rho* of a15/results/rho.json). Null test (arm folder = stored folder): the arm is the
          stored run itself, its member 0 = map25_m0 and it has no members 1-3 (four-member items not applicable)
No station or monitor file: the audit hook of a15/arnoise.py stays installed; st_ens / st_ghap of the part files are never
read. A15 code is imported read only (setup()).
    python prep22.py --year 2022 --arm <RUN>/eval_final_crps_e01_ar --stored <RUN>/eval_final_crps_e01 \
                     --out ../results/score22_prep.json"""
import sys

sys.dont_write_bytecode = True                  # no __pycache__ (inode quota), with PYTHONDONTWRITEBYTECODE=1 in the job
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution")).resolve()
A15, A16 = ROOT / "a15", ROOT / "a16"
HERE, RESULTS = A16 / "score22", A16 / "results"
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
BASE = {2021: 1461, 2022: 1826}                 # pm.npy index of day 0 of the year (episodes_diag/code/extract_1km.py)
SAMPLE = {2021: "S1", 2022: "S2"}               # revise.load names of the extracts
EXT_LABELS = {2021: list(range(1, 354, 8)), 2022: list(range(0, 357, 2))}
H, M = 7, 4
_MOD = {}


def setup(year):
    """read-only imports, in this order:
    1. a15/arnoise.py: installs the station / monitor audit hook (kept for the whole process) and asserts numpy 1.26.3;
    2. year != 2021: arnoise._deny_on = True BEFORE a15/score.py is executed, so that the ar.deny_2022() call at the top of
       a15/score.py returns without installing its hook against 2022 files (an audit hook cannot be removed once
       installed). Year 2021 (regression T1): the hook is installed, as in A15's own scorer. No a15 file is edited;
    3. a15/score.py by file path (module 'a15_score': custom, q6, lag_pairs, constants), then episodes_diag/code/revise.py."""
    if _MOD:
        assert _MOD["year"] == year, (_MOD["year"], year)
        return _MOD["ar"], _MOD["a15s"], _MOD["revise"]
    assert year in BASE, year
    sys.path.insert(0, str(A15))
    import arnoise as ar
    assert Path(ar.__file__).resolve() == A15 / "arnoise.py", ar.__file__
    if year != 2021:
        ar._deny_on = True
    spec = importlib.util.spec_from_file_location("a15_score", str(A15 / "score.py"))
    a15s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a15s)
    assert a15s.ar is ar
    sys.path.insert(0, str(ar.EPI / "code"))
    import revise
    assert Path(revise.__file__).resolve() == (ar.EPI / "code" / "revise.py").resolve(), revise.__file__
    assert Path(sys.modules[revise.boot_counts.__module__].__file__).resolve() == (ar.EPI / "code" / "common.py").resolve()
    assert a15s.M == revise.M == M and a15s.H == revise.H == H and a15s.EPS == revise.EPS
    assert a15s.NBOOT == 2000 and a15s.SEED == 20261121
    _MOD.update(year=year, ar=ar, a15s=a15s, revise=revise)
    return ar, a15s, revise


# ======================================================================== writes (a16/score22/ and a16/results/score22* only)
def allowed(path):
    p = Path(path).resolve()
    ok = str(p).startswith(str(HERE) + os.sep) or (p.parent == RESULTS and p.name.startswith("score22"))
    assert ok, f"{p} is outside a16/score22/ and a16/results/score22*"
    return p


def write_text(path, text):
    """temporary file, fsync, read-back, rename, fsync of the folder, read-back of the final file"""
    p = allowed(path)
    data = text.encode()
    tmp = p.with_name(f"{p.name}.tmp_s22_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fd = os.open(str(p.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    assert os.path.getsize(p) == len(data), p
    with open(p, "rb") as f:
        assert f.read() == data, p


def save_json(path, obj, ar):
    write_text(path, json.dumps(ar.jsonable(obj), indent=1) + "\n")


def max_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


# ======================================================================== part files
def parts_of(folder):
    """sorted by name, as a15/score.py (load_arm, load_stored_recs): the order of the region rows summed by the guard"""
    ps = [p for p in sorted(Path(folder).glob("part_*.npz")) if ".tmp" not in p.name]
    assert ps, f"no part file in {folder}"
    return ps


def read_parts(folder, cells_all, kind):
    """kind 'arm': (map25_mk (4, C), map25, map25_ghap) per (start day, lead); kind 'stored': (map25_m0 (1, C), map25,
    map25_ghap). Plus region rows, seams and the arm's base / keys_rho. Never reads st_ens or st_ghap."""
    fc, recs, seams, cols = {}, [], [], None
    info = {"parts": [], "bases": set(), "rhos": set(), "mk0_differs_from_m0": 0}
    for p in parts_of(folder):
        with np.load(p) as z:
            c = [str(x) for x in z["rec_cols"]]
            assert cols is None or c == cols, p
            cols = c
            keys = z["keys"].tolist()
            m25, g25, m0 = z["map25"], z["map25_ghap"], z["map25_m0"]
            if kind == "arm":
                mk = z["map25_mk"]
                assert mk.shape[:2] == (len(keys), M), (p, mk.shape)
                kr = z["keys_rho"].tolist()
                assert len(kr) == len(keys), p
                info["bases"].add(str(z["base"]))
                info["rhos"].update(float(r) for r in kr)
            for j, (d, h, dt) in enumerate(keys):
                assert (d, h) not in fc and dt == d + h, (p, d, h, dt)
                b0 = m0[j].reshape(-1)[cells_all]
                if kind == "arm":
                    X = mk[j].reshape(M, -1)[:, cells_all]
                    info["mk0_differs_from_m0"] += not np.array_equal(X[0], b0, equal_nan=True)
                else:
                    X = b0[None]
                fc[(d, h)] = (X, m25[j].reshape(-1)[cells_all], g25[j].reshape(-1)[cells_all])
            recs.append(z["recs"])
            seams.append(z["seams"])
        info["parts"].append(p.name)
    info["bases"], info["rhos"] = sorted(info["bases"]), sorted(info["rhos"])
    return fc, np.concatenate(recs), np.concatenate(seams), cols, info


def u16_equal(a, b):
    return np.array_equal(np.ascontiguousarray(a).view(np.uint16), np.ascontiguousarray(b).view(np.uint16))


# ======================================================================== inputs
def build(year, arm_dir, stored_dir, log=print):
    ar, a15s, revise = setup(year)
    t0 = time.time()
    arm_dir, stored_dir = Path(arm_dir), Path(stored_dir)
    null = arm_dir.resolve() == stored_dir.resolve()
    D = revise.load(SAMPLE[year], 1)
    lab = [int(v) for v in D["labels"]]
    assert lab == EXT_LABELS[year], lab
    with np.load(revise.ROOT / "data" / f"ext_{year}.npz") as z:
        D["truth_gi0"] = int(z["truth_gi0"])
    assert D["truth_gi0"] == BASE[year] + lab[0] - 60 - 1, D["truth_gi0"]
    cells_all = D["cells"]
    log(f"extract {year}: {len(lab)} runs, {len(cells_all)} populated cells, {time.time() - t0:.0f}s", flush=True)

    st_fc, st_recs, st_seams, st_cols, st_info = read_parts(stored_dir, cells_all, "stored")
    if null:
        arm_fc, arm_recs, arm_seams, arm_cols, arm_info = st_fc, st_recs, st_seams, st_cols, dict(st_info)
        starts = list(lab)
    else:
        arm_fc, arm_recs, arm_seams, arm_cols, arm_info = read_parts(arm_dir, cells_all, "arm")
        starts = sorted({d for d, _ in arm_fc})
        want = {(d, h) for d in starts for h in range(1, H + 1)}
        assert set(arm_fc) == want, sorted(set(arm_fc) ^ want)[:10]
        rj = json.loads((A15 / "results" / "rho.json").read_text())
        assert arm_info["bases"] == ["philox"] and arm_info["rhos"] == [float(rj["rho_star"])], (arm_info, rj["rho_star"])
    assert arm_cols == st_cols
    assert starts == (ar.CONF if year == 2021 else EXT_LABELS[2022]), starts
    assert set(starts) <= set(lab) and all((d, h) in st_fc for d in starts for h in range(1, H + 1))
    log(f"parts read {time.time() - t0:.0f}s: arm {len(arm_info['parts'])}, stored {len(st_info['parts'])}, "
        f"{len(starts)} starts, null test {null}", flush=True)

    # ---------------- cells: complete in every run of the stored extract and in every scored forecast of the arm
    arm_ok = np.ones(len(cells_all), bool)
    for d in starts:
        for h in range(1, H + 1):
            X, m, _ = arm_fc[(d, h)]
            arm_ok &= np.isfinite(X).all(0) & np.isfinite(m)
    ext_ok = D["ok"].all(0)
    keep = ext_ok & arm_ok
    lat, lon = D["lat"][keep], D["lon"][keep]
    t = ((84.0 - lat) // 10).astype(int) * 36 + ((lon + 180.0) // 10).astype(int)
    _, tid = np.unique(t, return_inverse=True)
    order = np.argsort(tid, kind="stable")
    sel = np.flatnonzero(keep)[order]                                   # positions in cells_all, in scoring order
    revise.restrict(D, keep, order)
    D["tid"] = tid[order]
    Ds = ar.subset_runs(D, [lab.index(d) for d in starts])
    del D
    N, C, MA = len(starts), int(keep.sum()), (1 if null else M)
    assert np.array_equal(Ds["cells"], cells_all[sel])

    # ---------------- arm arrays (float16 of the part files, as float32) and the pairing asserts
    Xa = np.empty((N, MA, H, C), np.float32)
    Ma = np.empty((N, H, C), np.float32)
    bad = {"arm_ghap_vs_extract_Y": 0, "stored_m0_vs_extract_X": 0, "stored_mean_vs_extract_Mn": 0,
           "stored_ghap_vs_extract_Y": 0}
    for r, d in enumerate(starts):
        for h in range(1, H + 1):
            X, m, g = arm_fc[(d, h)]
            Xa[r, :, h - 1] = X[:, sel].astype(np.float32)
            Ma[r, h - 1] = m[sel].astype(np.float32)
            bad["arm_ghap_vs_extract_Y"] += not u16_equal(g[sel], Ds["Y"][r, h - 1])
            Xs, ms, gs = st_fc[(d, h)]
            bad["stored_m0_vs_extract_X"] += not u16_equal(Xs[0, sel], Ds["X"][r, h - 1])
            bad["stored_mean_vs_extract_Mn"] += not u16_equal(ms[sel], Ds["Mn"][r, h - 1])
            bad["stored_ghap_vs_extract_Y"] += not u16_equal(gs[sel], Ds["Y"][r, h - 1])
    assert not any(bad.values()), bad
    assert np.isfinite(Xa).all() and np.isfinite(Ma).all()
    del arm_fc, st_fc
    info = {"year": year, "arm": str(arm_dir), "stored": str(stored_dir), "null_test": null, "starts": starts,
            "n_runs": N, "leads": list(range(1, H + 1)), "n_cells": C, "residents": float(Ds["pop"].sum()),
            "cells_sha256": ar.cells_hash(Ds["cells"]), "n_tiles": int(len(np.unique(Ds["tid"]))),
            "cells_counts": {"populated_in_extract": int(len(cells_all)), "complete_in_every_stored_extract_run": int(ext_ok.sum()),
                             "extract_runs": len(lab), "dropped_by_arm_completeness": int((ext_ok & ~arm_ok).sum())},
            "pairing_mismatching_forecasts": bad,
            "arm_design": {"members": MA, "bases": arm_info["bases"] if not null else "stored design (no base field)",
                           "keys_rho": arm_info["rhos"] if not null else "stored design (no keys_rho field)",
                           "member0_map25_mk_differs_from_map25_m0_forecasts": arm_info["mk0_differs_from_m0"] if not null else None},
            "parts": {"arm": arm_info["parts"], "stored": st_info["parts"]}, "rec_cols": st_cols,
            "truth": "map25_ghap of the part files (target days); pm.npy through ext_%d.npz (start day, d + 8, d + 9)" % year}
    if year == 2021:
        a15p = json.loads((A15 / "results" / "prep.json").read_text())
        info["a15_prep_cells_sha256_equal"] = info["cells_sha256"] == a15p["cells_sha256"]
    log(f"inputs {time.time() - t0:.0f}s: {C} cells ({info['cells_counts']}), {N} runs, max RSS {max_rss_gb():.1f} GB", flush=True)
    return {"Ds": Ds, "Xa": Xa, "Ma": Ma, "starts": starts, "null": null, "info": info,
            "recs_arm": arm_recs, "seams_arm": arm_seams, "recs_st": st_recs, "seams_st": st_seams, "cols": st_cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, choices=sorted(BASE))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allowed(a.out)
    t0 = time.time()
    ar, _, _ = setup(a.year)
    inp = build(a.year, a.arm, a.stored)
    out = {"what": __doc__, **inp["info"], "seconds": round(time.time() - t0, 1), "max_rss_gb": round(max_rss_gb(), 2)}
    save_json(a.out, out, ar)
    print(f"prep22 {a.year}: {out['n_cells']} cells, sha256 {out['cells_sha256'][:16]}, {out['n_runs']} runs, "
          f"{out['seconds']:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
