#!/usr/bin/env python3
"""A16 item: the 2021 evaluation of the final 1 km model under the evolving noise (rho*), assembled on CPU into ONE folder
RUN/eval2021_sel_crps_e01_ar in the format of the stored RUN/eval2021_sel_crps_e01 (F9), from three sets of forecasts that
were all made with the A15 generator (keys [1521, 2021, d, m, k, tr, tc]):
  1. a15/data/calib, the 15 Cal starts at lead 1 (run at the label rho 0.70; z(m, d, 1) = eps_1 does not depend on rho,
     A15 check I8, so these are the rho* forecasts) - only the lead-1 rows are taken;
  2. a16/data/cal2021 (run G9), the 15 Cal starts x leads 2-7 at rho*;
  3. a15/data/main (A15 run M), the 30 Conf starts x leads 1-7 at rho*.
= 45 starts x 7 leads = 315 forecasts. The stored F9 has 318: start 361 (leads 1-3) is absent by design (A15 4.4: its
leads 4-7 target 2022).
Each output part file mirrors one source part file (24 parts) and holds every field of the stored part files (recs, rec_cols,
seams, keys, st_ens, st_ghap, map25, map25_ghap, map25_sd, map25_m0, windows, regions, thresholds, info), plus map25_mk,
keys_rho (rho* for every forecast), keys_rho_as_run (0.70 for the Cal lead-1 rows), base, a16_source.
What cannot be mirrored:
  - the 42 map5km_d*_h*.npy files of F9 (starts 49, 105, 161, 217, 273, 329): A15's runs wrote none (A15 2.4 point 3);
    G9's 12 (starts 49, 217, leads 2-7) stay in a16/data/cal2021; no reader of the 2021 folder uses them (grep, 2026-11-21);
  - st_ens / st_ghap of A15's parts have 0 station columns (A15 never opened the station file): they are written as NaN
    with the station count of G9's parts, as the stored 2021 run holds them (all NaN: stations evaluate 2022 only);
  - the rank layout: 24 part files instead of 64 (readers concatenate part_*.npz).
Checks against the stored F9 (keys, recs, rec_cols and map25_ghap only): the (start, lead, target) set is F9's minus start
361; the region rows and the persistence sums (n, sse_persist, sae_persist) are identical; map25_ghap is bit-identical;
info, rec_cols, windows, regions and thresholds are equal. Writes results/assemble2021.json.
    python assemble2021.py          (after G9; CPU)
    python assemble2021.py --dry    (builds and checks in memory what exists, writes nothing)"""
import arnoise22 as an  # noqa: I001  (first)
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = an.RUN / "eval2021_sel_crps_e01_ar"
F9 = an.RUN / "eval2021_sel_crps_e01"
SRC = [("calib_lead1", an.A15_DIR / "data" / "calib"), ("G9", an.A16 / "data" / "cal2021"), ("main", an.A15_DIR / "data" / "main")]
STORED_FIELDS = ("recs", "rec_cols", "seams", "keys", "st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0",
                 "windows", "regions", "thresholds", "info")
ROWS = ("st_ens", "st_ghap", "map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk")


def parts(folder):
    return [p for p in sorted(Path(folder).glob("part_*.npz")) if ".tmp" not in p.name]


def stored_index():
    K, info = {}, {}
    for p in parts(F9):
        with np.load(p) as z:
            for n, (d, h, t) in enumerate(z["keys"].tolist()):
                K[(int(d), int(h))] = (p, n, int(t))
            info = {k: [str(x) for x in np.atleast_1d(z[k]).tolist()] for k in ("rec_cols", "info", "regions")}
            info["windows"], info["thresholds"] = z["windows"].tolist(), z["thresholds"].tolist()
            info["nst"] = int(z["st_ens"].shape[1])
    return K, info


def build(kind, p, nst):
    """the output part of one source part: its forecasts (lead 1 only for calib), stored fields + A16 extras"""
    with np.load(p) as z:
        keys = z["keys"]
        rho = z["keys_rho"].astype(np.float64)
        assert str(z["base"]) == "philox", p
        sel = np.flatnonzero(keys[:, 1] == 1) if kind == "calib_lead1" else np.arange(len(keys))
        if kind == "calib_lead1":
            assert np.all(rho[sel] == 0.70), (p, rho[sel])
        else:
            assert np.all(rho == an.rho_star()), (p, rho)
        want = {(int(d), int(h)) for d, h, _ in keys[sel].tolist()}
        rec = z["recs"]
        rsel = np.array([(int(r[0]), int(r[1])) in want for r in rec], bool) if len(rec) else np.zeros(0, bool)
        sm = z["seams"]
        ssel = np.array([(int(r[0]), int(r[1])) in want for r in sm], bool)
        out = {"recs": rec[rsel], "rec_cols": z["rec_cols"], "seams": sm[ssel], "keys": keys[sel], "windows": z["windows"],
               "regions": z["regions"], "thresholds": z["thresholds"], "info": z["info"]}
        for f in ("map25", "map25_ghap", "map25_sd", "map25_m0", "map25_mk"):
            out[f] = z[f][sel]
        se, sg = z["st_ens"][sel], z["st_ghap"][sel]
        if se.shape[1] == 0:                                           # A15 parts: no station column (never opened)
            se = np.full((len(sel), nst, se.shape[2]), np.nan, np.float16)
            sg = np.full((len(sel), nst), np.nan, np.float16)
        assert se.shape[1] == nst and np.all(np.isnan(se.astype(np.float32))) and np.all(np.isnan(sg.astype(np.float32))), p
        out["st_ens"], out["st_ghap"] = se.astype(np.float16), sg.astype(np.float16)
        out["keys_rho"] = np.full(len(sel), an.rho_star(), np.float64)
        out["keys_rho_as_run"] = rho[sel]
        out["base"] = np.array("philox")
        out["a16_source"] = np.array(json.dumps({"kind": kind, "part": str(p), "rows": sel.tolist()}))
    return out


def check(out, K, info, Z):
    """one output part against the stored F9: GHAP map and persistence sums identical, same region rows, same metadata"""
    cols = [str(c) for c in out["rec_cols"]]
    j = [cols.index(c) for c in ("n", "sse_persist", "sae_persist")]
    bad = []
    meta_ok = (cols == info["rec_cols"] and [str(x) for x in out["info"]] == info["info"]
               and [str(x) for x in out["regions"]] == info["regions"] and out["windows"].tolist() == info["windows"]
               and out["thresholds"].tolist() == info["thresholds"])
    for n, (d, h, t) in enumerate(out["keys"].tolist()):
        p, m, ts = K[(d, h)]
        if p not in Z:
            with np.load(p) as z:
                Z[p] = {"keys": z["keys"], "recs": z["recs"], "map25_ghap": z["map25_ghap"]}
        S = Z[p]
        g_ok = ts == t and np.array_equal(out["map25_ghap"][n].view(np.uint16), S["map25_ghap"][m].view(np.uint16))
        ra = {int(r[2]): tuple(float(r[i]) for i in j) for r in out["recs"] if (int(r[0]), int(r[1])) == (d, h)}
        rs = {int(r[2]): tuple(float(r[i]) for i in j) for r in S["recs"] if (int(r[0]), int(r[1])) == (d, h)}
        if not (g_ok and ra == rs and ra):
            bad.append((d, h))
    return meta_ok, bad


def main():
    dry = "--dry" in sys.argv
    t0 = time.time()
    R = {"what": __doc__, "dry": dry, "out": str(OUT), "sources": {}}
    K, info = stored_index()
    g9 = parts(an.A16 / "data" / "cal2021")
    nst = info["nst"]
    if g9:
        with np.load(g9[0]) as z:
            assert int(z["st_ens"].shape[1]) == nst, (z["st_ens"].shape, nst)
    if not dry:
        assert len(g9) == 8, "G9 (a16/data/cal2021) is not complete"
        mf = json.loads((an.A16 / "data" / "cal2021" / "a16_run.json").read_text())
        assert mf["plan"] == "G9" and mf["rho"] == an.rho_star() and mf["base"] == "philox", mf
        src_sha = {str(p): an.sha_file(p) for _, d in SRC for p in parts(d)}
        meta = {"plan": "assemble2021", "sources": src_sha, "code_sha256": {"assemble2021.py": an.sha_file(Path(__file__)),
                "arnoise22.py": an.sha_file(an.A16 / "arnoise22.py")}, "rho": an.rho_star(), "base": "philox", "year": 2021}
        an.claim(OUT, meta, fresh=False)
    done, Z, k, bad_all, meta_all = {}, {}, 0, [], True
    for kind, d in SRC:
        R["sources"][kind] = {"folder": str(d), "parts": len(parts(d))}
        for p in parts(d):
            out = build(kind, p, nst)
            for dd, h, t in out["keys"].tolist():
                assert (dd, h) not in done, (dd, h, p, done.get((dd, h)))
                done[(dd, h)] = kind
            meta_ok, bad = check(out, K, info, Z)
            meta_all &= meta_ok
            bad_all += bad
            if not dry:
                an.asavez(OUT / f"part_{k}.npz", **out)
            print(f"{kind} {p.name} -> part_{k}: {len(out['keys'])} forecasts, mismatches {len(bad)}, metadata {meta_ok}", flush=True)
            k += 1
    want = {kk for kk in K if kk[0] != 361}
    R.update(forecasts=len(done), by_source={s: sum(1 for v in done.values() if v == s) for s, _ in SRC},
             stored_forecasts=len(K), absent_by_design=sorted(kk for kk in K if kk[0] == 361),
             missing=sorted(kk for kk in want if kk not in done)[:40], extra=sorted(kk for kk in done if kk not in K),
             reference_mismatches=bad_all[:40], metadata_equal=meta_all, output_parts=k)
    R["all_passed"] = bool(not R["extra"] and not bad_all and meta_all and (dry or (not R["missing"] and len(done) == 315)))
    R["seconds"] = round(time.time() - t0, 1)
    if not dry:
        an.save_json(an.RES / "assemble2021.json", R)
    print(json.dumps(an.jsonable({k_: v for k_, v in R.items() if k_ != "what"})), flush=True)
    return 0 if R["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
