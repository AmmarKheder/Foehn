#!/usr/bin/env python3
"""A16 checks C3 and C4 (protocol_amendments/A16_test_year_evolving_noise.md section 3), CPU only, after the runs.
No station, monitor or OpenAQ file is opened (the model's values at the station pixels are read from the part files).
  COMPLETE  every run holds exactly the forecasts of its plan once, with their target day, base philox, keys_rho = rho*
            and its manifest; G1-G8: the same file names as the stored folder (+ a16_run.json), the same (start, lead,
            target) keys; G9: the 90 forecasts of the Cal starts x leads 2-7; G11: the 358 forecasts in a16/data/run16
  C3a       G2 = G1 at the station pixels for leads 1 and 3: st_ens of every (start, lead) of G2 against G1, each member:
            the same finite pattern, mean |diff| <= 0.05 ug m-3, 99.99th percentile <= 2.5 (A14a)
  C3b       members 0-3 of G11 = G2 at the populated pixels, every forecast and member, the same rule (A14a)
  C3ref     the reference columns that the part files hold, bit-identical to the stored runs of the same (start, lead):
            map25_ghap and st_ghap (GHAP), the persistence sums n, sse_persist, sae_persist of every region row (G1-G9);
            pop_ghap and the GHAP half of cell40 (G2 against F2). The CAMS, Aurora and climatology columns are made by
            the downstream scorers (REBUILD_PLAN 4c step 1): NOT covered here.
  C4        G1 against F1 (the guard G of A15 7.1, all 179 starts): per lead fair CRPS = sum crps / sum n and RMSE of the
            mean = sqrt(sum sse / sum n) over the region rows; |G1/F1 - 1| <= 2.5 % at every lead and <= 1.5 % for the
            mean over leads (mean of the per-lead values, as a15/score.guard); mean seam per lead <= stored + 0.02
  If C3 or C4 fails, A16 is INVALID (section 3).
    python check_runs.py [workers]            -> results/check_runs.json
    python check_runs.py --dry N [workers]    the same code on the stored runs in place of the new ones (F1 for G1, F2 for
                                              G2, F11 = a13/warn16/data/run16 + a14/data/run16_13 for G11; C3b on N
                                              forecasts; C4 F1 against itself); prints, writes nothing"""
import arnoise22 as an  # noqa: I001  (first)
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(an.ROOT / "a13" / "warn16"))
import check16all as c16  # noqa: E402  (index, cmp: unchanged)

RUN = an.RUN
EV = {g: Path(p["out"]) for g, p in an.EVAL_PLANS.items() if g.startswith("G")}
ST = {g: RUN / p["stored"] for g, p in an.EVAL_PLANS.items() if g.startswith("G")}
G11 = Path(an.SAMPLE_PLANS["G11"]["out"])
MEAN_MAX, P9999_MAX = 0.05, 2.5
S = {}


def ok_pix(c):
    return bool(c["finite_pattern_equal"] and c["mean_abs"] <= MEAN_MAX and c["p9999_abs"] <= P9999_MAX)


def parts(folder):
    return [p for p in sorted(Path(folder).glob("part_*.npz")) if ".tmp" not in p.name]


def keys_of(folder):
    """{(d, h): (part, row, target)} of a folder of part files"""
    K = {}
    for p in parts(folder):
        with np.load(p) as z:
            for n, (d, h, t) in enumerate(z["keys"].tolist()):
                assert (d, h) not in K, (folder, d, h)
                K[(int(d), int(h))] = (p, n, int(t))
    return K


def load(folder, fields):
    """{(d, h): {field: row}} for the per-forecast fields of every part file"""
    out = {}
    for p in parts(folder):
        with np.load(p) as z:
            arr = {f: z[f] for f in fields}
            for n, (d, h, _) in enumerate(z["keys"].tolist()):
                out[(int(d), int(h))] = {f: arr[f][n] for f in fields}
    return out


def recs_of(folder):
    R, seams, cols = [], [], None
    for p in parts(folder):
        with np.load(p) as z:
            c = [str(x) for x in z["rec_cols"]]
            assert cols is None or c == cols
            cols = c
            R.append(z["recs"])
            seams.append(z["seams"])
    return np.concatenate(R), np.concatenate(seams), cols


# ---------------------------------------------------------------- COMPLETE
def complete(g):
    out = EV[g] if g != "G11" else G11
    if not out.exists():
        return {"pass": False, "error": f"{out} does not exist (run not done)"}
    names = sorted(os.listdir(out))
    mf = json.loads((out / "a16_run.json").read_text()) if (out / "a16_run.json").exists() else {}
    r = {"manifest_plan": mf.get("plan"), "manifest_rho": mf.get("rho"), "manifest_base": mf.get("base"),
         "tmp_files": [n for n in names if ".tmp" in n]}
    good = mf.get("plan") == g and mf.get("rho") == an.rho_star() and mf.get("base") == "philox" and not r["tmp_files"]
    if g == "G11":
        idx = c16.index(out)
        want = [tuple(int(v) for v in it.split(":")) for ln in an.pairs_g11() for it in ln]
        r.update(forecasts=len(idx), missing=[k for k in want if k not in idx], extra=[k for k in idx if k not in set(want)],
                 wrong_target=[k for k in idx if idx[k][2] != k[0] + k[1]], files=len(names))
        good &= not r["missing"] and not r["extra"] and not r["wrong_target"] and len(idx) == 358 and len(names) == 45 * 3 + 1
        return {"pass": bool(good), **r}
    K = keys_of(out)
    rho_ok = True
    for p in parts(out):
        with np.load(p) as z:
            rho_ok &= str(z["base"]) == "philox" and all(float(x) == an.rho_star() for x in z["keys_rho"]) \
                and json.loads(str(z["a16"]))["plan"] == g
    r["base_rho_plan_in_parts"] = bool(rho_ok)
    good &= rho_ok
    if g == "G9":
        want = {(d, h): d + h for d in an.CAL for h in range(2, 8)}
        r.update(forecasts=len(K), equal_to_plan=sorted((k, v[2]) for k, v in K.items()) == sorted(want.items()),
                 files=len(names), map5km=sorted(n for n in names if n.startswith("map5km")))
        good &= r["equal_to_plan"] and len(K) == 90
    else:
        Ks = keys_of(ST[g])
        snames = sorted(os.listdir(ST[g]))
        r.update(forecasts=len(K), stored_forecasts=len(Ks),
                 keys_equal_to_stored=sorted((k, v[2]) for k, v in K.items()) == sorted((k, v[2]) for k, v in Ks.items()),
                 file_names_equal_to_stored=[n for n in names if n != "a16_run.json"] == snames,
                 missing_files=[n for n in snames if n not in names][:20], extra_files=[n for n in names if n not in snames and n != "a16_run.json"][:20])
        good &= r["keys_equal_to_stored"] and r["file_names_equal_to_stored"]
    return {"pass": bool(good), **r}


# ---------------------------------------------------------------- C3
def c3a(g1, g2):
    A = load(g2, ("st_ens",))
    B = load(g1, ("st_ens",))
    rows, bad = [], []
    for k in sorted(A):
        if k not in B:
            bad.append(k)
            continue
        c = {f"m{m}": c16.cmp(np.asarray(A[k]["st_ens"][:, m], np.float16), np.asarray(B[k]["st_ens"][:, m], np.float16)) for m in range(4)}
        rows.append(dict(start=k[0], lead=k[1], ok=all(ok_pix(v) for v in c.values()),
                         worst_mean_abs=max(v["mean_abs"] for v in c.values()), worst_p9999_abs=max(v["p9999_abs"] for v in c.values()),
                         min_bit_identical_share=min(v["bit_identical_share"] for v in c.values())))
    good = not bad and len(rows) == 358 and all(r["ok"] for r in rows) and {r["lead"] for r in rows} == {1, 3}
    return {"pass": bool(good), "forecasts": len(rows), "missing_in_G1": bad, "failing": [(r["start"], r["lead"]) for r in rows if not r["ok"]],
            "worst_mean_abs": max((r["worst_mean_abs"] for r in rows), default=None),
            "worst_p9999_abs": max((r["worst_p9999_abs"] for r in rows), default=None),
            "min_bit_identical_share": min((r["min_bit_identical_share"] for r in rows), default=None)}


def _one_c3b(key):
    p, n, t = S["G11"][key]
    k, nn, t0 = S["G2"][key]
    assert t == t0, (key, t, t0)
    new = np.asarray(np.load(p, mmap_mode="r")[n], np.float16)
    old = np.asarray(np.load(Path(S["G2dir"]) / f"pop_{k}.npy", mmap_mode="r")[nn], np.float16)
    c = {f"m{m}": c16.cmp(new[:, m], old[:, m]) for m in range(4)}
    return dict(start=key[0], lead=key[1], ok=all(ok_pix(v) for v in c.values()), members=c)


def c3b(g11_dirs, g2, limit, workers):
    idx = {}
    for d in g11_dirs:
        for k, v in c16.index(d).items():
            assert k not in idx, k
            idx[k] = v
    K2 = {}
    for p in parts(g2):
        rk = int(p.stem.split("_")[1])
        with np.load(p) as z:
            for n, (d, h, t) in enumerate(z["keys"].tolist()):
                K2[(int(d), int(h))] = (rk, n, int(t))
    keys = sorted(k for k in idx if k in K2 and k[1] in (1, 3))
    if limit:
        keys = keys[:limit]
    S.update(G11=idx, G2=K2, G2dir=str(g2))
    with get_context("fork").Pool(workers) as pool:
        rows = pool.map(_one_c3b, keys, chunksize=1)
    allm = [v for r in rows for v in r["members"].values()]
    good = all(r["ok"] for r in rows) and (limit or len(rows) == 358)
    return {"pass": bool(good), "forecasts": len(rows), "failing": [(r["start"], r["lead"]) for r in rows if not r["ok"]],
            "worst_mean_abs": max((v["mean_abs"] for v in allm), default=None),
            "worst_p9999_abs": max((v["p9999_abs"] for v in allm), default=None),
            "worst_max_abs": max((v["max_abs"] for v in allm), default=None),
            "min_bit_identical_share": min((v["bit_identical_share"] for v in allm), default=None),
            "finite_pattern_equal_everywhere": all(v["finite_pattern_equal"] for v in allm)}


def same16(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def c3ref(gdir, sdir, pop):
    A = load(gdir, ("map25_ghap", "st_ghap"))
    B = load(sdir, ("map25_ghap", "st_ghap"))
    common = sorted(set(A) & set(B))
    bad_map = [k for k in common if not same16(A[k]["map25_ghap"], B[k]["map25_ghap"])]
    bad_st = [k for k in common if not same16(A[k]["st_ghap"], B[k]["st_ghap"])]
    ra, _, ca = recs_of(gdir)
    rb, _, cb = recs_of(sdir)
    assert ca == cb
    j = [ca.index(c) for c in ("n", "sse_persist", "sae_persist")]
    da = {tuple(int(v) for v in r[:3]): tuple(float(r[i]) for i in j) for r in ra}
    db = {tuple(int(v) for v in r[:3]): tuple(float(r[i]) for i in j) for r in rb}
    rk = [k for k in da if k[:2] in set(common)]
    bad_rec = [k for k in rk if db.get(k) != da[k]]
    out = {"forecasts_compared": len(common), "forecasts_in_run": len(A), "map25_ghap_mismatch": bad_map[:20],
           "st_ghap_mismatch": bad_st[:20], "region_rows": len(rk), "persistence_rows_mismatch": bad_rec[:20]}
    good = len(common) == len(A) and not bad_map and not bad_st and not bad_rec
    if pop:
        bad_pop = bad_c40 = 0
        KA, KB = keys_of(gdir), keys_of(sdir)
        for k in common:
            pa, na, _ = KA[k]
            pb, nb, _ = KB[k]
            ra_, rb_ = pa.stem.split("_")[1], pb.stem.split("_")[1]
            bad_pop += not same16(np.load(gdir / f"pop_ghap_{ra_}.npy", mmap_mode="r")[na], np.load(sdir / f"pop_ghap_{rb_}.npy", mmap_mode="r")[nb])
            bad_c40 += not same16(np.load(gdir / f"cell40_{ra_}.npy", mmap_mode="r")[na][1], np.load(sdir / f"cell40_{rb_}.npy", mmap_mode="r")[nb][1])
        out.update(pop_ghap_mismatch=bad_pop, cell40_ghap_mismatch=bad_c40)
        good &= bad_pop == 0 and bad_c40 == 0
    return {"pass": bool(good), **out}


# ---------------------------------------------------------------- C4 (a15/score.guard on all starts)
def c4(gdir, sdir):
    ra, sa, cols = recs_of(gdir)
    rs, ss, cols_s = recs_of(sdir)
    assert cols == cols_s
    j = {c: cols.index(c) for c in ("start_doy", "lead", "region", "n", "sse", "crps")}
    ka = sorted(map(tuple, ra[:, :3].astype(int).tolist()))
    ks = sorted(map(tuple, rs[:, :3].astype(int).tolist()))
    same_rows = ka == ks and len(set(ka)) == len(ka)
    out = {"per_lead": {}, "rows": len(ka), "rows_equal": same_rows}
    crps, rmse, seam = {"g1": [], "f1": []}, {"g1": [], "f1": []}, {"g1": [], "f1": []}
    for h in range(1, 8):
        for nm, r, s in (("g1", ra, sa), ("f1", rs, ss)):
            q = r[r[:, j["lead"]].astype(int) == h]
            n = q[:, j["n"]].sum()
            crps[nm].append(q[:, j["crps"]].sum() / n)
            rmse[nm].append(np.sqrt(q[:, j["sse"]].sum() / n))
            sv = s[s[:, 1].astype(int) == h][:, 2]
            assert len(sv) == 179, (nm, h, len(sv))
            seam[nm].append(float(np.mean(sv)))
    ok = same_rows
    for h in range(7):
        rc, rr = crps["g1"][h] / crps["f1"][h] - 1, rmse["g1"][h] / rmse["f1"][h] - 1
        sok = seam["g1"][h] <= seam["f1"][h] + 0.02
        good = abs(rc) <= 0.025 and abs(rr) <= 0.025 and sok
        ok &= good
        out["per_lead"][h + 1] = {"crps_g1": crps["g1"][h], "crps_f1": crps["f1"][h], "crps_ratio_m1": rc, "rmse_g1": rmse["g1"][h],
                                  "rmse_f1": rmse["f1"][h], "rmse_ratio_m1": rr, "seam_g1": seam["g1"][h], "seam_f1": seam["f1"][h],
                                  "pass": good}
    mc = np.mean(crps["g1"]) / np.mean(crps["f1"]) - 1
    mr = np.mean(rmse["g1"]) / np.mean(rmse["f1"]) - 1
    out["mean_over_leads"] = {"crps_ratio_m1": mc, "rmse_ratio_m1": mr}
    out["pass"] = bool(ok and abs(mc) <= 0.015 and abs(mr) <= 0.015)
    return out


def main():
    argv = sys.argv[1:]
    dry = "--dry" in argv
    limit = 0
    if dry:
        i = argv.index("--dry")
        limit = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    workers = int(argv[0]) if argv else 8
    t0 = time.time()
    R = {"what": __doc__, "dry": dry, "tests": {}}
    T = R["tests"]

    def run(name, fn, *a):
        t = time.time()
        try:
            T[name] = fn(*a)
        except Exception as e:                                      # one failing check must not hide the others
            T[name] = {"pass": False, "error": repr(e)}
        T[name]["seconds"] = round(time.time() - t, 1)
        print(name, json.dumps(an.jsonable({k: v for k, v in T[name].items() if k != "per_lead"}))[:900], flush=True)
    if dry:
        g1, g2 = ST["G1"], ST["G2"]
        g11 = [an.ROOT / "a13" / "warn16" / "data" / "run16", an.ROOT / "a14" / "data" / "run16_13"]
        run("C3a", c3a, g1, g2)
        run("C3b", c3b, g11, g2, limit, workers)
        run("C3ref_G1", c3ref, g1, g1, False)
        run("C4", c4, g1, g1)
        print(json.dumps(an.jsonable(T["C4"]["mean_over_leads"])) if "mean_over_leads" in T["C4"] else "", f"dry {time.time() - t0:.0f} s", flush=True)
        return 0
    for g in list(EV) + ["G11"]:
        run(f"COMPLETE_{g}", complete, g)
    run("C3a", c3a, EV["G1"], EV["G2"])
    run("C3b", c3b, [G11], EV["G2"], 0, workers)
    for g in EV:
        if EV[g].exists():
            run(f"C3ref_{g}", c3ref, EV[g], ST[g], g == "G2")
    run("C4", c4, EV["G1"], ST["G1"])
    c3 = [k for k in T if k.startswith("C3")]
    R["C3_pass"] = all(T[k]["pass"] for k in c3) and {"C3a", "C3b"} <= set(c3) and len([k for k in c3 if k.startswith("C3ref")]) == 9
    R["C4_pass"] = bool(T["C4"]["pass"])
    R["complete_pass"] = all(T[k]["pass"] for k in T if k.startswith("COMPLETE"))
    R["all_passed"] = bool(R["C3_pass"] and R["C4_pass"] and R["complete_pass"])
    R["verdict"] = ("checks passed" if R["all_passed"] else "runs incomplete: C3 and C4 not judged" if not R["complete_pass"]
                    else "A16 INVALID (section 3): C3 or C4 failed")
    R["seconds"] = round(time.time() - t0, 1)
    an.save_json(an.RES / "check_runs.json", R)
    print(R["verdict"], f"{R['seconds']:.0f}s", flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
