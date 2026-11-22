#!/usr/bin/env python3
"""Who is warned, version 3 (2026-11-20): three corrections of 'who is warned'
(H-1, M-1, L-6 of the code audit).

Everything else is version 2 unchanged: the same 8.1 M grid cells, the same 358 forecasts (179 per lead time, start days
every second day), the same valid residents, regions and 16-entry code table. The two files of the earlier versions are
read, never written: ../data/hidden_systems_pop.npz (../figdata_hidden_systems.py) and data/hidden_bc_pop.npz (the copy of
v2/data/hidden_bc_pop.npz; figdata_hidden_bc.py). The fields are rebuilt with the functions of those two scripts, imported
unchanged (bc = figdata_hidden_bc, sy = ../figdata_hidden_systems).

H-1  persistence: a resident is warned when the 1 km estimate of the START day d in their grid cell exceeds the threshold
     (the field from which Foehn starts; ghap1km_global.npy, day 1826 + d, read at the residents' grid cells; a value <= 0
     is no estimate and never warns). System 'persist' in table 1; in the lead ladder also 'persist_25km', the start-day
     estimate averaged over the same centred 0.25 deg blocks as Foehn's ensemble mean there (mean of the valid pixels).
M-1  exact factors: for every system that the tables move (warns above mu x threshold: foehn_mean, cams, aurora, cams_qm,
     cams_bcsd, persist), stage 1 stores, per forecast, threshold and truth code (exposed, 40 km mean above, hidden with a
     margin), the residents by the factor above which they are warned, on the grid mu = 0.400, 0.401, ..., 2.000 (exactly
     the comparison x > float32(mu x threshold) of version 2 at the grid points). Stage 2 finds, from the pooled curves, the
     factor at which the system warns as many people a day as the target (or is as often correct); stage 3 re-evaluates the
     system from the fields at that exact factor (all regions, 16 codes).
L-6  the 0.25 deg block means of Foehn (map25 of eval_*_crps_e01, blocks centred on the 0.25 deg points: fine rows
     25k-12 .. 25k+12, eval_global_green.cell_mean) are read from the block of the resident's own cell, index
     ((r + 12) // 25, ((c + 12) // 25) mod 1440); version 2 read (r // 25, c // 25), the block centred on the north-west
     corner point of the cell. 'foehn_25km_oldindex' keeps the old reading for the identity check.

    python figdata_hidden_v3.py [workers] [start days of a smoke test, comma-separated]
      -> data/start_est_v3.npz (start-day estimates at the cells and 0.25 deg block means; resumed when present)
         data/hidden_v3_pop.npz   (stage 1)        data/hidden_v3_exact.npz (stages 2 and 3)     (*_test.npz in a smoke test)"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.pop("EVP_EXTRA", None)
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import figdata_hidden_bc as bc        # noqa: E402  (v3 copy of v2/figdata_hidden_bc.py, unchanged)
import figdata_hidden_systems as sy   # noqa: E402  (../figdata_hidden_systems.py, unchanged)

G, EVP, EVF = bc.G, bc.EVP, sy.EVF
GHAP = G / "ghap1km_global.npy"
GH, GW, D0 = 14400, 36000, 1826                     # lattice of the estimates; day 1826 = 2022-01-01 (meta.npz)
THR, MARGIN, REG = bc.THR, bc.MARGIN, bc.REG
MU0, DMU, NB = 0.4, 0.001, 1600
MUS = np.round(MU0 + DMU * np.arange(NB + 1), 3)     # 1601 factors 0.400 ... 2.000
HSYS = ("foehn_mean", "cams", "aurora", "cams_qm", "cams_bcsd", "persist")
SYS1 = ("persist", "foehn_25km", "foehn_25km_oldindex", "foehn_p50", "foehn_mean", "foehn_40km", "cams", "aurora", "cams_qm", "cams_bcsd")
SYS2 = ("foehn_25km", "cams", "aurora", "persist", "persist_25km", "foehn_25km_oldindex")
LEADS2 = (1, 3, 5, 7)
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
DATA = HERE / f"data{SFX}"
S = {}


# ------------------------------------------------------------------ setup
def setup():
    bc.setup(); sy.setup()
    for k in ("pop", "LAB16", "M"):
        assert np.array_equal(bc.S[k], sy.S[k]), k
    assert bc.S["NC"] == sy.S["NC"] and list(bc.REG) == list(sy.REG) and tuple(bc.THR) == tuple(sy.THR)
    idx = np.load(EVP / "pop_idx.npy"); r, c = idx // GW, idx % GW
    rn = (r + 12) // 25
    S.update(idx=idx, rn=np.minimum(rn, 575).astype(np.int32), cn=(((c + 12) // 25) % 1440).astype(np.int32), rn_ok=rn < 576)
    with open(GHAP, "rb") as f:
        assert np.lib.format.read_magic(f) == (1, 0)
        sh, fo, dtp = np.lib.format.read_array_header_1_0(f)
        S["OFF"] = f.tell()
    assert sh[1:] == (GH, GW) and dtp == np.float16 and not fo, (sh, dtp, fo)
    meta = np.load(G / "meta.npz"); assert int(meta["year"][D0]) == 2022 and int(meta["doy"][D0]) == 0
    assert np.array_equal(MUS[[100, 200, 300, 400, 500, 700, 850, 1100, 1600]], np.array(sy.MULT)), "factor grid"
    print(f"setup: {len(idx)} cells; residents outside the 0.25 deg block rows: {bc.S['pop'][~S['rn_ok']].sum() / 1e6:.3f} M", flush=True)


# ------------------------------------------------------------------ stage 0: estimates of the start days
def read_day(day):
    a = np.fromfile(GHAP, dtype=np.float16, count=GH * GW, offset=S["OFF"] + (D0 + day) * GH * GW * 2)
    assert a.size == GH * GW
    return a.reshape(GH, GW)


def block25(a):
    """mean of the valid (> 0) pixels of each 0.25 deg block centred on a 0.25 deg point (fine rows 25k-12 .. 25k+12, columns
    25j-12 .. 25j+12, as eval_global_green.cell_mean; NaN where no valid pixel)"""
    out = np.full((GH // 25, GW // 25), np.nan, np.float32)
    for k in range(GH // 25):
        lo, hi = max(0, 25 * k - 12), min(GH, 25 * k + 13)
        x = a[lo:hi].astype(np.float32); v = x > 0
        s = np.concatenate([np.zeros(12), np.where(v, x, 0).sum(0, dtype=np.float64), np.zeros(13)]).reshape(-1, 25).sum(1)[:GW // 25]
        n = np.concatenate([np.zeros(12), v.sum(0).astype(np.float64), np.zeros(13)]).reshape(-1, 25).sum(1)[:GW // 25]
        out[k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def stage0_day(task):
    kind, day = task
    t0 = time.time()
    a = read_day(day)
    v = a.reshape(-1)[S["idx"]]
    if kind == "check":                            # the reader against the stored estimates of a target day (pop_ghap)
        key = S["check_keys"][day]; k, n, t = bc.S["pop_index"][key]
        g1 = np.asarray(np.load(EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n])
        mine = np.where(v > 0, v, np.nan).astype(np.float16)
        same = bool(np.array_equal(np.isnan(mine), np.isnan(g1)) and np.array_equal(mine[~np.isnan(mine)], g1[~np.isnan(g1)]))
        print(f"check day {day}: reader equals pop_ghap of forecast {key}: {same} ({time.time() - t0:.0f} s)", flush=True)
        return kind, day, same, None
    m25 = block25(a)
    print(f"start day {day}: read and block means {time.time() - t0:.0f} s", flush=True)
    return kind, day, v.copy(), m25


def stage0(days, workers, path):
    if path.exists():
        z = np.load(path)
        assert list(z["days"]) == list(days), "start days of the stored file differ"
        S.update(P1=z["pix"], P25=z["map25"], POS={int(d): i for i, d in enumerate(days)})
        S["CHECK0"] = json.loads(str(z["checks"]))
        print(f"stage 0 resumed from {path}", flush=True)
        return
    keys1 = sorted((k for k in bc.S["pop_index"] if k[1] == 1), key=lambda k: k[0])
    S["check_keys"] = {}
    for key in (keys1[0], keys1[len(keys1) // 2], keys1[-1]):
        S["check_keys"][bc.S["pop_index"][key][2]] = key
    tasks = [("day", int(d)) for d in days] + [("check", t) for t in S["check_keys"]]
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(stage0_day, tasks, chunksize=1)
    out = {d: (p, m) for kind, d, p, m in res if kind == "day"}
    checks = {f"reader_equals_pop_ghap_day{d}": bool(p) for kind, d, p, m in res if kind == "check"}
    P1 = np.stack([out[d][0] for d in days]); P25 = np.stack([out[d][1] for d in days])
    # the block means against the stored block means of the estimates (map25_ghap of the forecasts whose target is the day)
    dev = []
    for i, d in enumerate(days):
        if (d - 2, 2) in sy.S["fin_index"] and len(dev) < 12:
            p, nn = sy.S["fin_index"][(d - 2, 2)]
            ref = np.load(p)["map25_ghap"][nn].astype(np.float32); mine = P25[i]
            both = np.isfinite(ref) & np.isfinite(mine)
            rel = np.abs(mine[both] - ref[both]) / np.maximum(np.abs(ref[both]), 1.0)
            dev.append((int(d), float(np.percentile(rel, 99)), float(rel.max()), int((np.isfinite(ref) != np.isfinite(mine)).sum()), int(both.sum())))
    checks["block_means_vs_map25_ghap (day, p99 rel dev, max rel dev, blocks valid in one only, blocks)"] = dev
    S.update(P1=P1, P25=P25, POS={int(d): i for i, d in enumerate(days)}, CHECK0=checks)
    np.savez(path, days=np.array(days), pix=P1, map25=P25, checks=json.dumps(checks))
    print("stage 0 checks:", json.dumps(checks), flush=True)


def start_pix(day):
    v = S["P1"][S["POS"][day]].astype(np.float32)
    return np.where(v > 0, v, np.nan).astype(np.float32)


def at_block(m25):
    return np.where(S["rn_ok"], m25[S["rn"], S["cn"]], np.nan).astype(np.float32)


# ------------------------------------------------------------------ the fields of one forecast (as versions 1 and 2)
def fields(key):
    d, h = key
    k, n, t = bc.S["pop_index"][key]
    m = np.asarray(np.load(EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
    g1 = np.asarray(np.load(EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
    c40 = np.asarray(np.load(EVP / f"cell40_{k}.npy", mmap_mode="r")[n], np.float32).reshape(2, -1)
    f40, g40 = c40[0][bc.S["cell"]], c40[1][bc.S["cell"]]
    ok = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m).all(1)
    w = np.where(ok, bc.S["pop"], 0.0)
    cams, aur = sy.fields04(d, h)                                                        # = the fields of versions 1 and 2
    # the bias-corrected CAMS forecast (verbatim from figdata_hidden_bc.one)
    day = bc.D2022 + bc.dt.timedelta(t); assert day.year == 2022
    mon = day.month
    x025 = bc.regrid(cams, bc.S["WTS"]).astype(np.float16).astype(np.float64)
    xc = np.log1p(np.maximum(x025.reshape(-1)[bc.S["CELL_FLAT"]], 0.0))
    qx, qy = bc.S["QX"][:, mon - 1], bc.S["QY"][:, mon - 1]
    qm = bc.qmap_vec(xc, qx, qy)
    B = bc.harmonic_basis(t)
    U = bc.S["U"]; has = U >= 0
    qm_p = np.full(U.size, np.nan); qm_p[has] = qm[U[has]]
    mubar = bc.S["CC"] @ B
    ano = np.zeros(U.size)
    both = has & bc.S["INMASK"]
    mu_p = (bc.S["COEF"] @ B.astype(np.float32)).astype(np.float64)
    ano[both] = mu_p[both] - mubar[U[both]]
    X = dict(foehn_mean=m.mean(1), cams=bc.bil(cams), aurora=bc.bil(aur), cams_qm=np.expm1(qm_p).astype(np.float32),
             cams_bcsd=np.expm1(qm_p + ano).astype(np.float32), persist=start_pix(d))
    return t, m, g1, f40, g40, w, X


# ------------------------------------------------------------------ stage 1
def stage1(key):
    d, h = key
    t0 = time.time()
    k, n, _ = bc.S["pop_index"][key]
    t, m, g1, f40, g40, w, X = fields(key)
    m25 = np.load(EVP / f"part_{k}.npz")["map25"][n].astype(np.float32)
    f25, f25o = at_block(m25), m25[sy.S["r25"], sy.S["c25"]]
    cnt = {}

    def nab(thr):
        if thr not in cnt:
            cnt.clear(); cnt[thr] = (m > thr).sum(1)
        return cnt[thr]
    fns = [lambda thr, C: X["persist"] > thr, lambda thr, C: f25 > thr, lambda thr, C: f25o > thr, lambda thr, C: nab(thr) >= 2,
           lambda thr, C: X["foehn_mean"] > thr, lambda thr, C: f40 > thr, lambda thr, C: X["cams"] > thr, lambda thr, C: X["aurora"] > thr,
           lambda thr, C: X["cams_qm"] > thr, lambda thr, C: X["cams_bcsd"] > thr]
    assert len(fns) == len(SYS1)
    with np.errstate(invalid="ignore"):
        T1 = bc.tables(w, g1, g40, fns)
    # curves of the moved systems: residents by truth code (exposed + 2 x 40 km mean above + 4 x hidden with a margin) and by
    # the number of grid factors mu_j with x > float32(mu_j x threshold) (0 for no value)
    Hh = np.zeros((len(THR), len(HSYS), 8, NB + 2), np.float32)
    for j, thr in enumerate(THR):
        code = ((g1 > thr).astype(np.int64) + 2 * (g40 > thr) + 4 * ((g1 > thr * (1 + MARGIN)) & (g40 <= thr * (1 - MARGIN)))) * (NB + 2)
        edges = (MUS * thr).astype(np.float32)
        for s, name in enumerate(HSYS):
            x = X[name]
            i = np.searchsorted(edges, x, side="left"); i[np.isnan(x)] = 0
            Hh[j, s] = np.bincount(code + i, weights=w, minlength=8 * (NB + 2)).reshape(8, NB + 2)
    T2 = None
    if h == 1:                                                                           # lead ladder on this target day
        T2 = np.full((len(REG), len(THR), len(LEADS2), len(SYS2), 16), np.nan)
        for li, L in enumerate(LEADS2):
            s0 = t - L
            if (s0, L) not in sy.S["fin_index"]:
                continue
            p, nn = sy.S["fin_index"][(s0, L)]
            x25 = np.load(p)["map25"][nn].astype(np.float32)
            xs = {"foehn_25km": at_block(x25), "foehn_25km_oldindex": x25[sy.S["r25"], sy.S["c25"]], "persist": start_pix(s0),
                  "persist_25km": at_block(S["P25"][S["POS"][s0]])}
            if L <= 5:
                ca, au = sy.fields04(s0, L)
                xs["cams"], xs["aurora"] = bc.bil(ca), bc.bil(au)
            names = [s for s in SYS2 if s in xs]
            with np.errstate(invalid="ignore"):
                tab = bc.tables(w, g1, g40, [lambda thr, C, x=xs[s]: x > thr for s in names])
            for q, s in enumerate(names):
                T2[:, :, li, SYS2.index(s)] = tab[:, :, q]
    diag = [w.sum(), w[~np.isfinite(X["persist"])].sum(), w[~np.isfinite(f25)].sum(), w[~np.isfinite(X["cams_qm"])].sum()]
    print(f"stage 1: start {d} lead {h} target {t}: {w.sum() / 1e6:.0f} M valid residents, {time.time() - t0:.0f} s", flush=True)
    return key, t, T1, Hh, T2, diag


# ------------------------------------------------------------------ stage 2: the factors, from the pooled curves
def curves(Hs):
    """Hs (8, NB + 2) pooled -> (8, NB + 1): residents warned at each factor of the grid, by truth code"""
    return np.cumsum(Hs[:, ::-1], axis=1)[:, ::-1][:, 1:]


def pooled(T):
    """(..., 16) code table summed over days -> warned, rightly warned"""
    c16 = np.arange(16)
    return T[..., c16 >= 8].sum(-1), T[..., (c16 >= 8) & (c16 % 2 == 1)].sum(-1)


def root(y, target, increasing):
    """the factor at which the curve y (on MUS) crosses the target, linear inside one step; None outside the grid"""
    s = (y - target) if increasing else (target - y)                   # negative before the crossing, >= 0 after
    j = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    if j.size == 0:
        return None, 0
    j0 = int(j[0]); fr = -s[j0] / (s[j0 + 1] - s[j0])
    return float(MUS[j0] + fr * DMU), int(j.size)


def stage2(K, T1, Hh, clim):
    sel = {h: np.flatnonzero(K[:, 1] == h) for h in sorted(set(K[:, 1]))}
    specs = []
    W = lambda h, j, s: pooled(T1[sel[h], 0, j, SYS1.index(s)].sum(0)) if s != "clim_1km" else pooled(clim[sel[h], 0, j].sum(0))   # noqa: E731
    for h in sel:
        for j, thr in enumerate(THR):
            cv = {s: curves(Hh[sel[h], j, q].sum(0, dtype=np.float64)) for q, s in enumerate(HSYS)}
            wn = {s: cv[s].sum(0) for s in HSYS}; tp = {s: cv[s][1::2].sum(0) for s in HSYS}
            for s in HSYS:                                              # identity: the curve at mu = 1 is the system as issued
                if s in SYS1:
                    w1, r1 = W(h, j, s); jj = int(np.flatnonzero(MUS == 1.0)[0])
                    assert abs(wn[s][jj] - w1) <= 1e-5 * max(w1, 1.0) and abs(tp[s][jj] - r1) <= 1e-5 * max(r1, 1.0), (h, thr, s, wn[s][jj], w1)
            fw, fr_ = W(h, j, "foehn_p50")
            for s in ("cams", "aurora", "cams_qm", "cams_bcsd", "persist"):                 # equal number of people warned
                mu, nc = root(wn[s], fw, increasing=False)
                specs.append(dict(kind="equal_number", lead=int(h), thr=thr, moved=s, fixed="foehn_p50", target=float(fw), mu=mu, crossings=nc))
            for moved, fixed in (("cams_qm", "foehn_p50"), ("cams_bcsd", "foehn_p50"), ("cams", "foehn_p50"), ("aurora", "foehn_p50"),
                                 ("persist", "foehn_p50"), ("foehn_mean", "cams_qm"), ("foehn_mean", "cams_bcsd"), ("foehn_mean", "cams"),
                                 ("foehn_mean", "aurora"), ("foehn_mean", "clim_1km"), ("foehn_mean", "persist")):   # equal share of correct warnings
                a, b = W(h, j, fixed); tgt = b / a
                with np.errstate(invalid="ignore", divide="ignore"):
                    cs = tp[moved] / wn[moved]
                mu, nc = root(np.where(np.isfinite(cs), cs, np.inf), tgt, increasing=True)
                specs.append(dict(kind="equal_correct_share", lead=int(h), thr=thr, moved=moved, fixed=fixed, target=float(tgt), mu=mu, crossings=nc))
            for fixed in ("clim_1km", "cams_bcsd", "cams_qm", "cams", "persist"):           # Foehn's ensemble mean at their number warned
                a, _ = W(h, j, fixed)
                mu, nc = root(wn["foehn_mean"], a, increasing=False)
                specs.append(dict(kind="foehn_mean_at_number_warned_by", lead=int(h), thr=thr, moved="foehn_mean", fixed=fixed, target=float(a), mu=mu, crossings=nc))
    return specs


# ------------------------------------------------------------------ stage 3: the moved systems at the exact factors, from the fields
def stage3(key):
    d, h = key
    t0 = time.time()
    t, m, g1, f40, g40, w, X = fields(key)
    sp = [s for s in S["SPECS"] if s["lead"] == h and s["mu"] is not None]
    out = np.zeros((len(REG), len(sp), 16))
    for q, s in enumerate(sp):
        thr = s["thr"]; x = X[s["moved"]]
        E, C = g1 > thr, g40 > thr
        Hm = (g1 > thr * (1 + MARGIN)) & (g40 <= thr * (1 - MARGIN))
        with np.errstate(invalid="ignore"):
            base = bc.S["LAB16"] + E + 2 * C + 4 * Hm + 8 * (x > np.float32(s["mu"] * thr))
        out[:, q] = bc.S["M"] @ np.bincount(base, weights=w, minlength=bc.S["NC"] * 16).reshape(bc.S["NC"], 16)
    print(f"stage 3: start {d} lead {h}: {len(sp)} systems at their factors, {time.time() - t0:.0f} s", flush=True)
    return key, out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    only = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    sfx = "_test" if only else ""
    setup()
    keys = sorted(bc.S["pop_index"], key=lambda k: (k[1], k[0]))
    if only:
        keys = [k for k in keys if k[0] in only]
    days = sorted({k[0] for k in keys} | {bc.S["pop_index"][k][2] - L for k in keys if k[1] == 1 for L in LEADS2
                                          if (bc.S["pop_index"][k][2] - L, L) in sy.S["fin_index"]})
    stage0(days, workers, (DATA if sfx else HERE / "data") / f"start_est_v3{sfx}.npz")   # start-day estimates (truth only): the stored file
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(stage1, keys, chunksize=1)
    res.sort(key=lambda r: (r[0][1], r[0][0]))
    K = np.array([(k[0], k[1], t) for k, t, *_ in res])
    T1 = np.stack([r[2] for r in res]); Hh = np.stack([r[3] for r in res]); diag = np.array([r[5] for r in res])
    t2 = [(r[1], r[4]) for r in res if r[4] is not None]
    np.savez(DATA / f"hidden_v3_pop{sfx}.npz", keys=K, T1=T1, sys1=np.array(SYS1), H=Hh, hsys=np.array(HSYS), mus=MUS, targets2=np.array([t for t, _ in t2]),
             T2=np.stack([x for _, x in t2]), sys2=np.array(SYS2), leads2=np.array(LEADS2), diag=diag, regions=np.array(REG), thr=np.array(THR),
             margin=MARGIN, checks0=json.dumps(S["CHECK0"]),
             diag_cols=np.array(["valid residents", "valid residents without an estimate on the start day", "valid residents without a 0.25 deg block",
                                 "valid residents without a cell map"]),
             code=np.array("above + 2 * cell above + 4 * hidden with margin + 8 * warned"),
             hist=np.array("H[forecast, threshold, system, code8, i]: residents with truth code code8 (above + 2 cell above + 4 hidden with margin) "
                           "whose value exceeds float32(mus[j] x threshold) for exactly the j < i (i = 0: no value or below mus[0])"))
    print(f"stage 1 done ({time.time() - t0:.0f} s)", flush=True)
    zb = np.load((HERE.parent / "v2" / f"data{SFX}" if SFX else DATA) / "hidden_bc_pop.npz")   # design: the v2 output of the same design
    Kb = zb["keys"]; rows = [int(np.flatnonzero((Kb[:, 0] == d) & (Kb[:, 1] == h))[0]) for d, h, _ in K]
    clim = zb["T1"][rows][:, :, :, list(zb["sys1"]).index("clim_1km")]
    S["SPECS"] = stage2(K, T1, Hh, clim)
    print(f"stage 2: {len(S['SPECS'])} comparisons, {sum(s['mu'] is None for s in S['SPECS'])} outside the factors 0.4-2", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        res3 = pool.map(stage3, keys, chunksize=1)
    res3 = dict(res3)
    TB = {}
    for h in sorted(set(K[:, 1])):
        sp = [s for s in S["SPECS"] if s["lead"] == h and s["mu"] is not None]
        kk = [tuple(k) for k in K[K[:, 1] == h][:, :2]]
        A = np.stack([res3[k] for k in kk])                                              # (forecasts of the lead, regions, specs, 16)
        for q, s in enumerate(sp):
            s["index"] = len(TB); TB[f"spec{len(TB)}"] = A[:, :, q]
            w_, r_ = pooled(A[:, 0, q].sum(0))
            s["exact_warned"], s["exact_right"] = float(w_), float(r_)
    np.savez(DATA / f"hidden_v3_exact{sfx}.npz", specs=json.dumps(S["SPECS"]), keys=K, **TB)
    print(f"stage 3 done ({time.time() - t0:.0f} s); saved", DATA / f"hidden_v3_exact{sfx}.npz", flush=True)


if __name__ == "__main__":
    main()
