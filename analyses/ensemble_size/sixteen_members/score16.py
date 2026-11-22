#!/usr/bin/env python3
"""A13 section 4: score the warnings of Foehn with 4 and with 16 members against the start-day 1 km estimate kept unchanged,
on every sixth start day of analyses/hidden/v3 (30 start days, leads 1 and 3). No station or monitor is read.

Definitions and readers are those of analyses/hidden/v3 (paper folder, read only; imported unchanged, no bytecode written):
  population, truth  bc.setup(): the 8.1 M grid cells with >= 100 residents, residents, regions and the 16-entry code table;
                     the 1 km estimate of the target day (pop_ghap_<k>.npy) and its 40 km cell mean (cell40_<k>.npy[:, 1]) of the
                     stored evaluation eval_pop_crps_e01 (they do not depend on the members)
  valid residents    estimate, 40 km mean and every member finite (as v3; here for the 4 stored AND the 16 new members, which
                     must give v3's valid residents exactly: asserted)
  foehn4             warned when at least 2 of the 4 stored members (the paper's warnings, eval_pop_crps_e01) exceed the threshold
  foehn4new          the same rule on members 0-3 of the new 16-member run (check of the reproduction)
  foehn16            warned when at least 8 of the 16 members exceed (ensemble probability >= 0.5, the same as 2 of 4)
  persist            warned when the 1 km estimate of the start day exceeds the threshold (v3 data/start_est_v3.npz, the field of
                     v3 stage 0, read through the same rule as v3.start_pix: a value <= 0 is no estimate)
  reach / correct    tables.metrics (exposed and warned / exposed; exposed and warned / warned), sums pooled over days
  equal number warned  the start-day estimate moved (warns above mu x threshold) to the number of people warned by foehn4 or by
                     foehn16: curve on the grid mu = 0.400, 0.401, ..., 2.000 (v3.MUS) built as in v3 stage 1, point linear
                     between the two neighbouring grid factors (v3 tables_v3.at_factor, persistence branch: tied half-precision
                     values), factor found again in every bootstrap resample
  intervals          tables.mbb + Tab: 10,000 resamples, seed 0, paired (the same resampled start days for every system and
                     difference). Start days here are 12 days apart, so a block is ONE start day (v3's blocks of 4 forecasts
                     2 days apart span 7 days; one start day already spans more); blocks of 2 start days as a sensitivity.
Identity checks (asserted): foehn4 and persist tables equal the rows of v3 data/hidden_v3_pop.npz (T1 'foehn_p50', 'persist') for
these 60 forecasts; the persistence curves equal v3's H rows.

    python score16.py RUN16_DIR [workers]   -> results/numbers_warn16.json, results/tables_warn16.txt
"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HID = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/analyses/hidden"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(HID)); sys.path.insert(0, str(HID / "v3"))
import figdata_hidden_v3 as v3  # noqa: E402  (unchanged; brings bc = figdata_hidden_bc)
from tables import Tab, f, mbb, metrics, parts, tri  # noqa: E402  (hidden/tables.py, unchanged)

bc = v3.bc
EVP = bc.EVP
THR_ALL = tuple(bc.THR)                         # 15, 25, 37.5, 50, 75 (the code tables keep all five)
THR_REP = (15.0, 37.5, 75.0)
SYS = ("foehn4", "foehn4new", "foehn16", "persist")
MUS, MU0, DMU, NB = v3.MUS, v3.MU0, v3.DMU, v3.NB
DOYS = list(range(0, 349, 12))
S = {}


def load_run16(D):
    idx = {}
    for fk in sorted(Path(D).glob("keys16_*.npy")):
        rank = fk.stem.split("_")[1]
        done = set(json.loads((Path(D) / f"done16_{rank}.json").read_text()))
        for n, (d, h, t) in enumerate(np.load(fk)):
            if n in done:
                idx[(int(d), int(h))] = (str(Path(D) / f"pop16_{rank}.npy"), n, int(t))
    return idx


def one(key):
    d, h = key
    t0 = time.time()
    k, n, t = bc.S["pop_index"][key]
    p16, n16, t16 = S["R16"][key]; assert t16 == t
    m4 = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
    m16 = np.asarray(np.load(p16, mmap_mode="r")[n16], np.float32)
    g1 = np.asarray(np.load(EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
    c40 = np.asarray(np.load(EVP / f"cell40_{k}.npy", mmap_mode="r")[n], np.float32).reshape(2, -1)
    g40 = c40[1][bc.S["cell"]]
    ok4 = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m4).all(1)
    ok = ok4 & np.isfinite(m16).all(1)
    w = np.where(ok, bc.S["pop"], 0.0)
    w4 = np.where(ok4, bc.S["pop"], 0.0)
    v = S["P1"][S["POS"][d]].astype(np.float32)
    per = np.where(v > 0, v, np.nan).astype(np.float32)                       # = v3.start_pix(d)
    cache = {}

    def cnt(name, thr):
        if (name, thr) not in cache:
            x = {"foehn4": m4, "foehn4new": m16[:, :4], "foehn16": m16}[name]
            cache[(name, thr)] = (x > thr).sum(1)
        return cache[(name, thr)]
    fns = [lambda thr, C: cnt("foehn4", thr) >= 2, lambda thr, C: cnt("foehn4new", thr) >= 2,
           lambda thr, C: cnt("foehn16", thr) >= 8, lambda thr, C: per > thr]
    with np.errstate(invalid="ignore"):
        T = bc.tables(w, g1, g40, fns)                                           # (regions, 5 thresholds, systems, 16)
    Hh = np.zeros((len(THR_ALL), 8, NB + 2), np.float32)                         # persistence curve, as v3 stage 1
    for j, thr in enumerate(THR_ALL):
        code = ((g1 > thr).astype(np.int64) + 2 * (g40 > thr) + 4 * ((g1 > thr * (1 + bc.MARGIN)) & (g40 <= thr * (1 - bc.MARGIN)))) * (NB + 2)
        edges = (MUS * thr).astype(np.float32)
        i = np.searchsorted(edges, per, side="left"); i[np.isnan(per)] = 0
        Hh[j] = np.bincount(code + i, weights=w, minlength=8 * (NB + 2)).reshape(8, NB + 2)
    # members: bit identity of members 0-3 with the stored ones, and spread of the 16 members
    same = float((m16[:, :4].astype(np.float16).view(np.uint16) == m4.astype(np.float16).view(np.uint16)).mean())
    diag = dict(valid_M=float(w.sum() / 1e6), valid4_M=float(w4.sum() / 1e6), members0_3_bit_identical_share=same,
                max_abs_members0_3=float(np.nanmax(np.abs(m16[:, :4] - m4))))
    print(f"start {d} lead {h} target {t}: {w.sum() / 1e6:.0f} M valid residents, members 0-3 bit-identical {same:.6f}, "
          f"{time.time() - t0:.0f} s", flush=True)
    return key, t, T, Hh, diag


def persist_at(fixed, Hd, cnt):
    """the start-day estimate at the number of people warned by `fixed` (Tab of a Foehn variant): tables_v3.at_factor,
    persistence branch (point between the two neighbouring grid factors on the pooled curve; factor re-found per resample)"""
    n = len(Hd)
    Cd = np.cumsum(Hd.astype(np.float64)[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]   # (days, 8, NB + 1): warned at each factor
    Pw = Cd.sum(0)
    target = parts(fixed.a.sum(0))["warn"]
    mu, ncross = v3.root(Pw.sum(0), target, increasing=False)
    if mu is None:
        return None
    jm = int(round((mu - MU0) / DMU)); lo, hi = max(0, jm - 250), min(Cd.shape[2], jm + 251)
    Cw = Cd[:, :, lo:hi]; nw = hi - lo
    bW = (cnt @ Cw.reshape(n, -1)).reshape(-1, 8, nw)
    a = fixed.a; bN = cnt @ (a[:, :8] + a[:, 8:])
    tgt = parts(fixed.bsum.astype(np.float64))["warn"]; s_ = tgt[:, None] - bW.sum(1)
    cross = (s_[:, :-1] < 0) & (s_[:, 1:] >= 0)
    okb = cross.any(1); j0 = np.argmax(cross, axis=1); rr = np.arange(len(j0))
    fr = -s_[rr, j0] / (s_[rr, j0 + 1] - s_[rr, j0])
    Wb = bW[rr, :, j0] + fr[:, None] * (bW[rr, :, j0 + 1] - bW[rr, :, j0])
    A16 = np.concatenate([bN - Wb, Wb], axis=1); A16[~okb] = np.nan
    bs = metrics(parts(A16), n)
    sp = target - Pw[:, lo:hi].sum(0)
    j = int(np.flatnonzero((sp[:-1] < 0) & (sp[1:] >= 0))[0]); fr0 = -sp[j] / (sp[j + 1] - sp[j])
    P = Pw[:, lo + j] + fr0 * (Pw[:, lo + j + 1] - Pw[:, lo + j])
    N0 = a.sum(0)[:8] + a.sum(0)[8:]
    pt = metrics(parts(np.concatenate([N0 - P, P])), n)
    return dict(mu=mu, crossings=ncross, pt=pt, bs=bs, resamples_without_a_crossing=int((~okb).sum()))


KM = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "hidden_warned_M", "hidden_warned_pct", "csi")


def score(K, T1, H1, L):
    out = {}
    for h in sorted({int(x) for x in K[:, 1]}):
        sel = np.flatnonzero(K[:, 1] == h); n = len(sel); cnt = mbb(n, min(L, n))
        for thr in THR_REP:
            j = THR_ALL.index(thr)
            tab = {s: Tab(T1[sel, 0, j, q], cnt) for q, s in enumerate(SYS)}
            R = out.setdefault(f"thr{thr:g}", {}).setdefault(f"lead{h}", {})
            R["n_start_days"] = n
            R["exposed_M"] = tab["foehn4"].ci("exposed_M")
            for s in SYS:
                R[s] = {m: tab[s].ci(m) for m in KM}
            R["foehn16_minus_foehn4_as_issued"] = tab["foehn16"].diffs(tab["foehn4"], KM)
            pa = {}
            for s in ("foehn4", "foehn16"):
                r = persist_at(tab[s], H1[sel, j], cnt)
                pa[s] = r
                if r is None:
                    R[f"persist_at_number_warned_by_{s}"] = "outside the factors 0.4-2"; continue
                fb = tab[s].bs
                R[f"persist_at_number_warned_by_{s}"] = dict(
                    factor=r["mu"], crossings=r["crossings"], resamples_without_a_crossing=r["resamples_without_a_crossing"],
                    **{m: tri(r["pt"][m], r["bs"][m]) for m in KM},
                    **{f"{s}_minus_persist_{m}": tri(tab[s].pt[m] - r["pt"][m], fb[m] - r["bs"][m]) for m in KM})
            if pa["foehn4"] is not None and pa["foehn16"] is not None:
                b4, b16 = tab["foehn4"].bs, tab["foehn16"].bs
                R["gain16_minus_gain4_at_equal_number_warned"] = {
                    m: tri((tab["foehn16"].pt[m] - pa["foehn16"]["pt"][m]) - (tab["foehn4"].pt[m] - pa["foehn4"]["pt"][m]),
                           (b16[m] - pa["foehn16"]["bs"][m]) - (b4[m] - pa["foehn4"]["bs"][m])) for m in ("reach_pct", "correct_pct", "right_M", "csi")}
            R["persist_as_issued_minus_foehn16"] = tab["persist"].diffs(tab["foehn16"], KM)
            R["persist_as_issued_minus_foehn4"] = tab["persist"].diffs(tab["foehn4"], KM)
            R["foehn4new_minus_foehn4"] = tab["foehn4new"].diffs(tab["foehn4"], ("warned_M", "reach_pct", "correct_pct"))
    return out


def safe_write(path, text):
    path = str(path); data = text.encode(); tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data); fh.flush(); os.fsync(fh.fileno())
    assert open(tmp, "rb").read() == data; os.replace(tmp, path); assert open(path, "rb").read() == data


def main():
    TEST = "--test" in sys.argv                                                 # smoke: the forecasts present only, *_test outputs
    argv = [a for a in sys.argv[1:] if a != "--test"]
    D = argv[0]; workers = int(argv[1]) if len(argv) > 1 else 8
    sfx = "_test" if TEST else ""
    t0 = time.time()
    bc.setup()
    S["R16"] = load_run16(D)
    keys = [(d, h) for h in (1, 3) for d in DOYS]
    if TEST:
        keys = [k for k in keys if k in S["R16"]]
    missing = [k for k in keys if k not in S["R16"]]
    assert not missing, f"forecasts missing in {D}: {missing}"
    z = np.load(v3.DATA / "start_est_v3.npz")
    days = [int(x) for x in z["days"]]
    pos = {d: i for i, d in enumerate(days)}
    pix = z["pix"]                                                               # (179 start days, cells) float16, read once
    S["P1"] = pix[[pos[d] for d in DOYS]].copy(); del pix
    S["POS"] = {d: i for i, d in enumerate(DOYS)}
    print(f"setup {time.time() - t0:.0f} s; {len(keys)} forecasts", flush=True)
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(one, keys, chunksize=1)
    K = np.array([(k[0], k[1], t) for k, t, *_ in res])
    T1 = np.stack([r[2] for r in res]); H1 = np.stack([r[3] for r in res]); diag = [r[4] for r in res]
    np.savez(HERE / "data" / f"warn16_tables{sfx}.npz", keys=K, T1=T1, sys=np.array(SYS), H=H1, thr=np.array(THR_ALL), regions=np.array(bc.REG))
    # ---- identity with v3 (the paper's analysis) on these forecasts
    zv = np.load(v3.DATA / "hidden_v3_pop.npz")
    Kv, s1 = zv["keys"], [str(s) for s in zv["sys1"]]
    rows = [int(np.flatnonzero((Kv[:, 0] == d) & (Kv[:, 1] == h))[0]) for d, h, _ in K]
    assert np.array_equal(Kv[rows], K)
    Tv = zv["T1"][rows]; Hv = zv["H"][rows][:, :, [str(s) for s in zv["hsys"]].index("persist")]
    diagv = zv["diag"][rows]
    ident = {"foehn4 = v3 foehn_p50": bool(np.array_equal(T1[..., 0, :], Tv[..., s1.index("foehn_p50"), :])),
             "persist = v3 persist": bool(np.array_equal(T1[..., 3, :], Tv[..., s1.index("persist"), :])),
             "persist curves = v3 H": bool(np.array_equal(H1, Hv)),
             "valid residents = v3 (max abs dev, residents)": float(np.abs(np.array([x["valid_M"] for x in diag]) * 1e6 - diagv[:, 0]).max())}
    print("identity with v3:", ident, flush=True)
    assert ident["foehn4 = v3 foehn_p50"] and ident["persist = v3 persist"] and ident["persist curves = v3 H"], ident
    N = dict(what="A13 section 4: warnings of the frozen final 1 km model with 4 and 16 members against the start-day 1 km estimate, "
                  "every sixth start day of analyses/hidden/v3 (30 start days of 2022), leads 1 and 3",
             start_days=DOYS, run16=str(D), definitions=__doc__, identity_with_v3=ident,
             members=dict(min_bit_identical_share_members0_3=min(x["members0_3_bit_identical_share"] for x in diag),
                          max_abs_members0_3=max(x["max_abs_members0_3"] for x in diag),
                          valid_residents_M_4_vs_16_max_dev=max(abs(x["valid_M"] - x["valid4_M"]) for x in diag)),
             interval="[value, 2.5 %, 97.5 %]: tables.mbb, 10,000 resamples, seed 0, blocks of 1 start day (12 days), paired",
             blocks1=score(K, T1, H1, 1), blocks2_sensitivity=score(K, T1, H1, 2))
    # ---- the rule of A13 section 4
    B = N["blocks1"]
    rule = {}
    LEADS = sorted({int(x) for x in K[:, 1]})
    for thr in THR_REP:
        for h in LEADS:
            R = B[f"thr{thr:g}"][f"lead{h}"]
            g = R.get("gain16_minus_gain4_at_equal_number_warned")
            iss = R["foehn16_minus_foehn4_as_issued"]
            beats_iss = iss["reach_pct"][0] > 0 and iss["correct_pct"][0] > 0
            rule[f"thr{thr:g}_lead{h}"] = dict(
                gain16_minus_gain4_reach=g["reach_pct"] if g else None, gain16_minus_gain4_correct=g["correct_pct"] if g else None,
                sixteen_beat_four_point=bool(g and g["reach_pct"][0] > 0 and g["correct_pct"][0] > 0),
                sixteen_beat_four_interval_above_0=bool(g and g["reach_pct"][1] > 0 and g["correct_pct"][1] > 0),
                as_issued_16_minus_4_reach=iss["reach_pct"], as_issued_16_minus_4_correct=iss["correct_pct"], as_issued_16_better_on_both=beats_iss,
                **{f"{s}_minus_persist_at_equal_number_{m}": (R[f"persist_at_number_warned_by_{s}"][f"{s}_minus_persist_{m}"]
                                                              if isinstance(R[f"persist_at_number_warned_by_{s}"], dict) else None)
                   for s in ("foehn4", "foehn16") for m in ("reach_pct", "correct_pct")})
    N["rule_A13_section4"] = rule
    safe_write(HERE / "results" / f"numbers_warn16{sfx}.json", json.dumps(N, indent=1))
    L = []
    for thr in THR_REP:
        for h in LEADS:
            R = B[f"thr{thr:g}"][f"lead{h}"]
            L.append(f"\n=== > {thr:g} ug m-3, lead {h} d, {R['n_start_days']} start days; exposed {f(R['exposed_M'], 0)} M a day")
            for s in SYS:
                x = R[s]
                L.append(f"  {s:10s} warned {f(x['warned_M'], 0):>20s} M  reach {f(x['reach_pct']):>20s} %  correct {f(x['correct_pct']):>20s} %  csi {f(x['csi'], 3)}")
            d_ = R["foehn16_minus_foehn4_as_issued"]
            L.append(f"  16 - 4 as issued: warned {f(d_['warned_M'], 1)} M, reach {f(d_['reach_pct'], 2)} pts, correct {f(d_['correct_pct'], 2)} pts, csi {f(d_['csi'], 4)}")
            for s in ("foehn4", "foehn16"):
                r = R[f"persist_at_number_warned_by_{s}"]
                if not isinstance(r, dict):
                    L.append(f"  persist at {s}'s number warned: {r}"); continue
                L.append(f"  persist at {s}'s number warned (x{r['factor']:.4f}): reach {f(r['reach_pct'])} %, correct {f(r['correct_pct'])} % | "
                         f"{s} minus it: reach {f(r[f'{s}_minus_persist_reach_pct'], 2)} pts, correct {f(r[f'{s}_minus_persist_correct_pct'], 2)} pts "
                         f"(resamples without crossing {r['resamples_without_a_crossing']})")
            g = R.get("gain16_minus_gain4_at_equal_number_warned")
            if g:
                L.append(f"  gain of 16 minus gain of 4 (over persist at each one's number warned): reach {f(g['reach_pct'], 2)} pts, correct {f(g['correct_pct'], 2)} pts")
    L.append("\nidentity with v3: " + json.dumps(ident) + "\nmembers: " + json.dumps(N["members"]))
    safe_write(HERE / "results" / f"tables_warn16{sfx}.txt", "\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
