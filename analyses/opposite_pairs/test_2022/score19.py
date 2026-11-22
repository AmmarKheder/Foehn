#!/usr/bin/env python3
"""A19 scorer and decision (a19/A19_draft.md sections 4, 6.3, 6.4 and 7), CPU only, in the job container (numpy 1.26.3).
This scorer is frozen: its sha256 is written into its outputs and stated in A19_draft.md; any later change needs an addendum.

Members (ug m-3). Paired: ids 0..3 = A16 run G2 (RUN/eval_pop_crps_e01_ar, the noise z(k)), ids 4..7 = the A19 run
(a19/data/pop_neg, the noise -z(k)); pairs (k, 4 + k). Independent: the sixteen members of A16 run G11 (a16/data/run16; its
members 0..3 equal G2's at the A14a level), for the paper's forms (A14 4a) and, as run, its members 0..7.
Forms of the paper (A14 4a, from the sixteen): MSE_inf = mean[(xbar16 - y)^2 - s2_16/16], V = mean[s2_16] (ddof 1),
MSE_8 = MSE_inf + V/8. Ensembles scored as run: four (0..3), two pairs (0, 4, 1, 5; the mean over the 6 choices beside),
one pair (mean over the 4), four pairs (0..7), eight independent (G11 0..7).
  D       = MSE_8(form) - MSE(four pairs, as run)            the decision quantity (D > 0: the pairs beat the paper's form)
  O       = MSE(four, as run) - MSE(two pairs (0, 4, 1, 5))  the ORDER quantity (four members against four)
  D_asrun = MSE(G11 0..7, as run) - MSE(four pairs, as run)  (reported: the same eight-member run with members 4..7 drawn
            independently instead of negated; same particles member_particles(i, h, 4..7))
  rho_anti   within-pair correlation, reference = the mean of the OTHER pairs (A17 4.4, estimator copied from score17):
             p, t = half sum and half difference of a pair, r = mean of the K = 3 other pair means,
             rho = (A - B)/(A + B), A = K/(K + 1) sum (p - r)^2, B = sum t^2 (raw correlation kept as rho_anti_raw)
  rho_null   the same on pseudo-pairs of independent members (0, 1) | (2, 3) and (4, 5) | (6, 7), K = 1 (A17); rho_null03 =
             G2's members 0..3 alone; also in log space (ln max(x, 0.1))
  M_eq       equivalent number of independent members: V / (MSE - MSE_inf) (interval inverted from that of the inverse)
  guard G    (A17 6.3 tolerances for the two ratios, two-sided; the bias tolerance tightened from 2 % to 0.5 % after review):
             RMSE of one negated member / RMSE of one member of 0..3 within 1 +- 5 %, mean s2 of 4..7 / mean s2 of 0..3
             within 1 +- 15 %, |mean(xbar_8 - xbar_4)| <= 0.5 % of the mean truth; at the
             populated pixels (pixels, regime all), leads 1 and 3; reported at the stations
  spread     the naive spread over error sqrt((M + 1)/M mean s2)/RMSE of the eight paired members beside four (0..3), eight
             independent (G11 0..7) and the form for eight independent members sqrt(9/8 V)/sqrt(MSE_8). The sample variance of
             a paired ensemble is NOT an estimate of its forecast variance (pairs shrink the error of the mean, not the spread
             of the members); this is written into the output.
STATIONS (the decision): the station-days of the paper's station score, experiments/figures/compare_fair.py, executed from
  its unchanged source with the argument of the paper's A16 run (crps_e01_ar=<run>:eval_final_crps_e01_ar; its np.savez kept
  in memory, nothing written), its loop replayed with the same expressions as a14/classU/scripts/st_cf_pass.py (every
  system's per-day sum of squared errors and count asserted equal to the executed PD / PDN), leads 1 and 3. The members at
  the station pixels are read from the populated pixels through the station positions of a14 u_common.stations_in_pop
  (copied): 8,053 of the 8,110 stations; the 57 without a populated pixel are left out of the A19 sample S' (no member of
  G2, A19 or G11 exists there), and every A19 quantity, the paper's forms and the references (CAMS forecast and Aurora issued
  at the same time) are computed on S'. Bootstrap: compare_fair's, verbatim (default_rng(0), blocks of 7 start days,
  10,000 resamples, the 178 start days of its sample), paired (every quantity in the same resamples), leads pooled by
  start day. Identity of the scorer on the paper's full station-days: four members as run (G1) 13.00 / 14.21 and their
  reductions (PD sums, a14 st_cf_score arithmetic), the forms 12.78 / 12.55 and 13.90 / 13.59 (sixteen members of G11 at the
  populated-pixel stations, the four members of G1 at the 57 others, st_cf_score arithmetic), all against
  a14/classU/results_ar/st_cf_classU_179.json (relative 1e-9; the forms 1e-6), and the printed roundings.
POPULATED PIXELS (reported): the 8,112,762 pixels of pop_pixels.npz with at least min_pop (100) residents, estimate valid
  and every member finite, against the 1 km estimates of the target day (pop_ghap of G2), as a14 pop_pass.py and
  analyses/reliability/s3_pop.py: each pixel weight 1 (the paper's populated-pixel scores) and, beside, weighted by
  residents. Regimes of the forecast level: the mean of G11's members 8..15 (noise and particles independent of every
  member compared) < 15, 15-75, > 75 ug m-3; in a regime other than 'all' the 16-member forms are not used (members 8..15
  select the regime) and the law of the eight independent members G11 0..7 replaces them (keys *_law8). Bootstrap: blocks
  of 7 consecutive start days, 10,000 resamples, default_rng(0) (rel_common.block_weights), 179 start days. Identity: the
  sums of G2's four members equal a14 pop_pass / reliability sums_pop (relative 1e-6), those of G11 equal a14 pop16_run16.
WARNINGS (reported, never gating): 'at least 4 of 8' on the eight paired members at the populated pixels, World, 37.5 and
  75 ug m-3, leads 1 and 3, the 149 confirmatory start days of A14 (classE.sample_rows 'conf149'), with the functions of
  a14/classE/classE.py, analyses/hidden (bc.tables, tables.Tab) and a13 score16.persist_at imported unchanged (FOEHN_DESIGN=ar):
  people warned (millions a day), reach, share of correct warnings, and the CAMS forecast moved to the same number of people
  warned (eqn:cams); beside: the paper's eight-member values (classE_all2022.json M8_liberal = rebuild/print/warn8/out_ar),
  sixteen and four, and 'at least 4 of 8' of G11's eight independent members 0..7 as run. Identity: the stored four 'at
  least 2 of 4' equal v3's foehn_p50 tables, and M8_liberal recomputed from the sixteen members equals the paper's record.
Decision (A19_draft.md section 7, applied mechanically, ug m-3, stations S'): INVALID (smoke, scorer identity, format or
  guard G), then SUCCESS (D_1 > 0, D_3 > 0 and the 95 % interval of D_1 above 0), then ORDER (the pooled 95 % interval of O
  above 0), then FAIL.
    python score19.py --tests   -> code/out/tests19.json (no A19 output read: null and known-answer cases on stored data)
    python score19.py           -> code/out/score19.json (needs a19/data/pop_neg complete and data/smoke_check.json)"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

os.environ["FOEHN_DESIGN"] = "ar"                             # the A16 design of the paper's modules (classE, analyses/hidden)
for _k in ("STATIONS", "TAG", "EVP_EXTRA"):                   # compare_fair and v3 read these: the paper's defaults
    os.environ.pop(_k, None)
import argparse  # noqa: E402
import contextlib  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

# ---- A19 write guard (begin; verbatim in score19.py, check19 DIFF)
_BASES = ("/scratch/project_462001804/ammar", "/pfs/lustrep1/scratch/project_462001804/ammar")
_KEEP = ("global_model_forecasting_air_pollution/a14", "global_model_forecasting_air_pollution/a15",
         "global_model_forecasting_air_pollution/a16", "global_model_forecasting_air_pollution/a17",
         "global_model_forecasting_air_pollution/protocol_amendments", "eccv/cran-pm/experiments",
         "eccv/cran-pm/paper/natcomms", "eccv/data")
PROTECTED = tuple(f"{b}/{k}" for b in _BASES for k in _KEEP)
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_WEVENTS = {"os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.link": (1,), "os.symlink": (1,),
            "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,), "shutil.copyfile": (1,),
            "shutil.copytree": (1,), "shutil.move": (0, 1), "shutil.rmtree": (0,)}


def protected(path):
    """True if `path` lies in a folder that A19 must not modify (both spellings of the scratch mount)"""
    p = os.fsdecode(path)
    if not p.startswith("/"):
        try:
            p = os.path.join(os.getcwd(), p)
        except OSError:
            return False
    p = os.path.normpath(p)
    return any(p == q or p.startswith(q + "/") for q in PROTECTED)


def _audit19(event, args):                     # A19 writes only under a19/: never into a folder it must not modify
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and bool(flags & _WFLAGS))
        if w and protected(args[0]):
            raise PermissionError(f"A19: write refused in a protected folder: {os.fsdecode(args[0])}")
    elif event in _WEVENTS:
        for j in _WEVENTS[event]:
            a = args[j] if len(args) > j else None
            if isinstance(a, (str, bytes, os.PathLike)) and protected(a):
                raise PermissionError(f"A19: {event} refused in a protected folder: {os.fsdecode(a)}")


sys.addaudithook(_audit19)
# ---- A19 write guard (end)
import numpy as np  # noqa: E402

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution").resolve()
A19 = ROOT / "a19"
OUT, DATA = A19 / "code" / "out", A19 / "data"
CRAN = Path("/scratch/project_462001804/ammar/eccv/cran-pm")
RUN = CRAN / "experiments" / "global1km" / "runs" / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
G1D, G2D, NEGD, G11D = RUN / "eval_final_crps_e01_ar", RUN / "eval_pop_crps_e01_ar", DATA / "pop_neg", ROOT / "a16" / "data" / "run16"
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
FIGS = CRAN / "experiments" / "figures"
NATC = CRAN / "paper" / "natcomms"
A14 = ROOT / "a14"
CF_ARGV = ["crps_e01_ar=green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4:eval_final_crps_e01_ar"]
NB_, BLK = 10000, 7
SEED_SYN, EPS = 20261121, 0.1
LEADS = (1, 3)
REGS = ("all", "lt15", "15_75", "gt75")
WTS = ("pixels", "residents")
TOL_G = dict(rmse_ratio=0.05, spread_ratio=0.15, bias_rel=0.005)  # A17 6.3 ratios (>= 4.8 null sd of 2021); bias 0.5 % (review)
C_S1 = -0.85
CASES_TEST = ("N0", "S0", "S1", "S2", "S3", "S4")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


SELF_SHA256 = sha_file(__file__)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path, obj):
    """atomic: temporary file, fsync, read-back, rename; only under a19/code/out"""
    p = Path(path).resolve()
    assert str(p).startswith(str(OUT.resolve()) + os.sep) and not protected(p), p
    data = (json.dumps(jsonable(obj), indent=1) + "\n").encode()
    tmp = p.with_name(p.name + f".tmp_a19_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fd = os.open(str(p.parent), os.O_RDONLY); os.fsync(fd); os.close(fd)
    with open(p, "rb") as f:
        assert f.read() == data, p


def log(*a):
    print(f"[a19 {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ======================================================================== the paper's modules (imported unchanged)
sys.path.insert(0, str(A14 / "classE"))
import classE as ce  # noqa: E402  (a14/classE/classE.py: score16, analyses/hidden v3, bc, tables; FOEHN_DESIGN=ar)

bc, v3, s16 = ce.bc, ce.v3, ce.s16
assert ce.AR and ce.VDRAW0, "classE must be imported under FOEHN_DESIGN=ar"


# ======================================================================== stations (copies of a14 u_common, verified in the tests)
def station_pixels(lat, lon):
    """1 km pixel of each station, as eval_global_green.py (l. 336-338): row * 36000 + column"""
    GH, GW = 14400, 36000
    sr = np.clip(np.floor((84.0 - lat) / 0.01).astype(int), 0, GH - 1)
    sc = np.clip(np.floor((lon + 180.0) / 0.01).astype(int), 0, GW - 1)
    return sr.astype(np.int64) * GW + sc


def stations_in_pop():
    """position of each of the 8,110 stations in the populated-pixel list (pop_pixels.npz idx), -1 if absent"""
    st = np.load(G / "stations_2022.npz")
    idx = np.load(G / "pop_pixels.npz")["idx"]
    pix = station_pixels(st["lat"], st["lon"])
    pos = np.clip(np.searchsorted(idx, pix), 0, len(idx) - 1)
    return np.where(idx[pos] == pix, pos, -1)


def exec_compare_fair():
    """compare_fair.py executed from its unchanged source; np.savez kept in memory (no file), np.save refused"""
    src = FIGS / "compare_fair.py"
    code = src.read_text()
    ns = {"__name__": "__main__", "__file__": str(src)}
    cap = {}
    real_savez, real_save = np.savez, np.save

    def savez_cap(file, *a, **kw):
        assert not a, "positional arrays"
        cap[str(file)] = {k: np.asarray(v) for k, v in kw.items()}

    def save_refused(*a, **kw):
        raise PermissionError("A19: np.save refused while the paper's scorer runs")
    old_argv = sys.argv
    sys.argv = [str(src)] + CF_ARGV
    buf = io.StringIO()
    np.savez, np.save = savez_cap, save_refused
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, str(src), "exec"), ns)
    finally:
        np.savez, np.save = real_savez, real_save
        sys.argv = old_argv
    return ns, buf.getvalue(), hashlib.sha256(code.encode()).hexdigest(), cap


def station_prep():
    """the paper's station-days (leads 1 and 3) with y, the CAMS forecast and Aurora issued at the same time, G1's four
    members and mean; identities with the paper's stored outputs"""
    t0 = time.time()
    ns, printed, cf_sha, cap = exec_compare_fair()
    log(f"compare_fair.py executed unchanged (sha256 {cf_sha[:16]}) in {time.time() - t0:.0f} s")
    ref = np.load(FIGS / "data" / "compare_fair_crps_e01_ar.npz")
    ident = {f"executed_{k}_equals_stored_npz": bool(np.array_equal(np.asarray(ns[k]), ref[k], equal_nan=True)) for k in ("S", "PD", "PDN", "SST")}
    cv = next(iter(cap.values())) if len(cap) == 1 else {}
    ident["savez_captured_once_equal_namespace"] = len(cap) == 1 and all(
        np.array_equal(cv[k], np.asarray(ns[k]), equal_nan=True) for k in ("S", "PD", "PDN", "SST"))
    runs, common, days = ns["runs"], ns["common"], ns["days"]
    obs, official, GHd, CLIM, gid = ns["obs"], ns["official"], ns["GH"], ns["CLIM"], ns["gid"]
    cams, aur_d, aur_e, analysis00, coarse, bil = ns["cams"], ns["aur_d"], ns["aur_e"], ns["analysis00"], ns["coarse"], ns["bil"]
    sets, NS, NG, SYS, NETS, PD, PDN = ns["sets"], ns["NS"], ns["NG"], ns["SYS"], ns["NETS"], ns["PD"], ns["PDN"]
    assert SYS[4] == "CAMS (run d+1, equal issuance)" and SYS[6] == "Aurora (00 UTC d+1, equal issuance)", SYS
    index = runs[0][1]
    pc = {}

    def part(p):
        if p not in pc:
            if len(pc) > 4:
                pc.pop(next(iter(pc)))
            pc[p] = np.load(p)["st_ens"]
        return pc[p]
    rec, nrep = {}, 0
    for c, (d, h) in enumerate(common):
        if h > 4 or d + h >= 365 or d + 1 > 364 or d not in aur_d or (d + 1) not in aur_e or d not in GHd:
            continue
        t = d + h
        y = obs[:, t]
        X = [r[2][(d, h)] for r in runs]
        X += [coarse(0, d, h, 0.25)[0], coarse(0, d, h, 0.5)[0]]
        X.append(bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4))
        X.append(bil(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4))
        X.append(aur_d[d][h - 1])
        A = aur_e[d + 1]
        k0 = 2 * (h - 1)
        first = analysis00(d + 1) if h == 1 else A[k0 - 1]
        X.append(0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1])
        X.append(obs[:, d])
        X.append(GHd[d])
        X.append(CLIM[t])
        X = np.stack(X)
        ok = official & np.isfinite(y) & np.all(np.isfinite(X), 0)
        di = days.index(d)
        for k in range(len(SYS)):                                        # identity of the replay with the executed loop
            e = X[k, ok] - y[ok]
            assert PD[k, di, h - 1, 0] == (e * e).sum() and PD[k, di, h - 1, 1] == ok.sum(), (d, h, k)
            for g in range(NG):
                mg = ok & ((gid == g) if g < len(NETS) else True)
                eg = X[k, mg] - y[mg]
                assert PDN[k, g, di, h - 1, 0] == (eg * eg).sum() and PDN[k, g, di, h - 1, 1] == mg.sum(), (d, h, k, g)
        nrep += 1
        if h in LEADS:
            p, n = index[(d, h)]
            e4 = part(p); idx = sets[e4.shape[1]]
            E = np.full((NS, e4.shape[2]), np.nan, np.float32); E[idx[idx >= 0]] = e4[n, idx >= 0].astype(np.float32)
            assert np.array_equal(E.mean(-1)[ok], X[0][ok]), (d, h)
            s = np.flatnonzero(ok)
            rec[(d, h)] = dict(s=s, ok=ok, y=y[ok].astype(np.float32), cams=X[4][ok].astype(np.float32), aur=X[6][ok].astype(np.float32),
                               g1=X[0][ok].astype(np.float32), E4=E[ok])
    ident["replay_PD_PDN_identity_asserted_forecasts"] = nrep
    Z = np.load(A14 / "classU" / "data_ar" / "st_cf_terms.npz")
    MK = {(int(d), int(h)): i for i, (d, h) in enumerate(Z["keys"])}
    ident["station_day_masks_equal_a14_st_cf_terms"] = all(
        np.array_equal(np.unpackbits(Z["masks"][MK[k]])[:NS].astype(bool), r["ok"]) for k, r in rec.items()) and len(rec) == 356
    log(f"replay: {nrep} forecasts, {len(rec)} at leads 1 and 3; {time.time() - t0:.0f} s; identities {ident}")
    return dict(rec=rec, days=[int(x) for x in days], PD=np.asarray(PD), NS=int(NS), ident=ident, cf_sha=cf_sha,
                printed_tail=printed[-3000:])


# ======================================================================== per-element columns (stations and pixels)
def rho_cols(x, pairs, pre):
    """A17 score17.rho_cols (members on the LAST axis here): A = K/(K+1) sum (p - r)^2, B = sum t^2, raw products"""
    P = len(pairs)
    K = P - 1
    a_, b_ = [a for a, _ in pairs], [b for _, b in pairs]
    pm, th = 0.5 * (x[:, a_] + x[:, b_]), 0.5 * (x[:, a_] - x[:, b_])
    r = (pm.sum(1)[:, None] - pm) / K
    da, db = x[:, a_] - r, x[:, b_] - r
    return {pre + "A": (K / (K + 1) * (pm - r) ** 2).sum(1), pre + "B": (th ** 2).sum(1), pre + "dd": (da * db).sum(1),
            pre + "da2": (da ** 2).sum(1), pre + "db2": (db ** 2).sum(1)}


def null_cols(x, sets, pre):
    """A17 score17.null_cols: for every set of four independent members, the fixed pairing (a, b) | (c, d), K = 1"""
    acc = {}
    for s in sets:
        for n, v in rho_cols(x[:, list(s)], [(0, 1), (2, 3)], pre).items():
            acc[n] = acc.get(n, 0.0) + v
    return acc


PAIRS = [(k, 4 + k) for k in range(4)]


def columns(X, Y16, y):
    """X (n, 8): the eight members of the case (pairs (k, 4 + k)); Y16 (n, 16): the sixteen independent members of the forms;
    y (n,): the truth. Yields (name, column)."""
    e = lambda f: (f - y) ** 2                                         # noqa: E731
    x4, xn = X[:, :4], X[:, 4:]
    pm = 0.5 * (x4 + xn)
    yield "one", np.ones_like(y)
    yield "y", y
    yield "sse1", e(x4.T).mean(0)
    yield "sse1n", e(xn.T).mean(0)
    yield "sse4", e(x4.mean(1))
    yield "sse2p", e(0.5 * (pm[:, 0] + pm[:, 1]))
    yield "sse2p6", np.mean([e(0.5 * (pm[:, j] + pm[:, k])) for j in range(4) for k in range(j + 1, 4)], 0)
    yield "sse1p", e(pm.T).mean(0)
    yield "sse8p", e(X.mean(1))
    yield "s2_4", x4.var(1, ddof=1)
    yield "s2_n", xn.var(1, ddof=1)
    yield "s2_8", X.var(1, ddof=1)
    yield "bias", X.mean(1) - x4.mean(1)
    yield "sse16", e(Y16.mean(1))
    yield "s2_16", Y16.var(1, ddof=1)
    yield "ssei8", e(Y16[:, :8].mean(1))
    yield "s2_i8", Y16[:, :8].var(1, ddof=1)
    sets = [(0, 1, 2, 3), (4, 5, 6, 7)]
    lx = np.log(np.maximum(X, EPS))
    for d_ in (rho_cols(X, PAIRS, "ra_"), null_cols(X, sets, "rn_"), null_cols(X, sets[:1], "r3_"),
               rho_cols(lx, PAIRS, "la_"), null_cols(lx, sets, "ln_")):
        yield from d_.items()


def build_case(case, E4, X16, N4, Z):
    """(X8, Y16) of a case. real: G2 0..3 + A19 4..7, G11; N0: G2 0..3 + G11 4..7 (independent), G11; S*: synthetic members
    around the mean of G2's four with their spread (Z: standard normals (n, 24): 16 for the forms, 4 + 4 for the pairs)"""
    if case == "real":
        return np.concatenate([E4, N4], 1), X16
    if case == "N0":
        return np.concatenate([E4, X16[:, 4:8]], 1), X16
    mu = E4.mean(1, keepdims=True)
    sd = np.sqrt(E4.var(1, ddof=1, keepdims=True))
    Y16 = mu + sd * Z[:, :16]
    x03 = mu + sd * Z[:, 16:20]
    ep = Z[:, 20:24]
    if case == "S0":
        x47 = mu + sd * ep
    elif case == "S1":
        x47 = mu + sd * (np.float32(C_S1) * Z[:, 16:20] + np.float32(np.sqrt(1 - C_S1 ** 2)) * ep)
    elif case == "S2":
        x47 = np.concatenate([2 * mu - x03[:, :2], x03[:, 2:]], 1)
    elif case == "S3":
        x47 = 2 * mu - x03
    elif case == "S4":
        x47 = x03.copy()
    else:
        raise ValueError(case)
    return np.concatenate([x03, x47], 1), Y16


# ======================================================================== the populated-pixel pass (one forecast per task)
S = {}


def part_index(D):
    out = {}
    for p in sorted(glob.glob(str(D / "part_*.npz"))):
        assert ".tmp" not in p, p
        r = int(Path(p).stem.split("_")[1])
        with np.load(p) as z:
            for n, (d, h, t) in enumerate(z["keys"].tolist()):
                assert (int(d), int(h)) not in out, (p, d, h)
                out[(int(d), int(h))] = (r, n, int(t))
    return out


def index16(D):
    """(start, lead) -> (pop16 file, row, target) of the finished rows of a sixteen-member folder (as a14 u_common.index16)"""
    out = {}
    for fk in sorted(Path(D).glob("keys16_*.npy")):
        tag = fk.stem.split("_", 1)[1]
        done = set(json.loads((Path(D) / f"done16_{tag}.json").read_text()))
        for n, (d, h, t) in enumerate(np.load(fk)):
            if n in done:
                key = (int(d), int(h))
                assert key not in out, key
                out[key] = (str(Path(D) / f"pop16_{tag}.npy"), n, int(t))
    return out


def one(key):
    d, h = key
    t0 = time.time()
    r2, n2, t = S["K2"][key]
    E4 = np.asarray(np.load(G2D / f"pop_{r2}.npy", mmap_mode="r")[n2], np.float32)
    g16raw = np.load(G2D / f"pop_ghap_{r2}.npy", mmap_mode="r")[n2]
    g = np.asarray(g16raw, np.float32)
    c40 = np.asarray(np.load(G2D / f"cell40_{r2}.npy", mmap_mode="r")[n2], np.float32).reshape(2, -1)
    p16, n16, t16 = S["K16"][key]
    assert t16 == t, (key, t16, t)
    X16 = np.asarray(np.load(p16, mmap_mode="r")[n16], np.float32)
    diag = {}
    N4 = None
    if S["final"]:
        rn, nn, tn = S["KN"][key]
        N4 = np.asarray(np.load(NEGD / f"pop_{rn}.npy", mmap_mode="r")[nn], np.float32)
        gN = np.load(NEGD / f"pop_ghap_{rn}.npy", mmap_mode="r")[nn]
        cN = np.asarray(np.load(NEGD / f"cell40_{rn}.npy", mmap_mode="r")[nn], np.float32).reshape(2, -1)
        diag["format"] = dict(same_rank_and_row_as_G2=(rn, nn, tn) == (r2, n2, t),
                              pop_ghap_bit_identical=bool(np.array_equal(np.asarray(gN).view(np.uint16), np.asarray(g16raw).view(np.uint16))),
                              cell40_ghap_identical=bool(np.array_equal(cN[1], c40[1], equal_nan=True)),
                              finite_pattern_equal=bool(np.array_equal(np.isfinite(N4), np.isfinite(E4))), shape=list(N4.shape))
    fin4, fin16 = np.isfinite(E4).all(1), np.isfinite(X16).all(1)
    finN = np.isfinite(N4).all(1) if N4 is not None else np.ones(len(g), bool)
    POP, POPOK = S["POP"], S["POPOK"]
    ok4 = np.isfinite(g) & fin4 & POPOK
    valid = ok4 & fin16 & finN
    diag["ok4_not_valid"] = int((ok4 & ~valid).sum())
    # ---- identity sums of G2's four (a14 pop_pass.one_stored arithmetic) and of G11 (pop_pass.one16)
    Ez = np.where(ok4[:, None], E4, 0.0); gz = np.where(ok4, g, 0.0)
    mean = Ez.mean(1); var = Ez.var(1, ddof=1); err = mean - gz
    idn = {}
    for nm, w in (("", ok4.astype(np.float64)), ("w", POP * ok4)):
        idn[nm] = dict(n=float(w.sum()), sse=float(np.dot(w, np.asarray(err * err, np.float64))), var=float(np.dot(w, np.asarray(var, np.float64))))
    del Ez, gz, mean, var, err
    ok16 = ok4 & fin16
    D16 = X16[ok16].astype(np.float64) - g[ok16].astype(np.float64)[:, None]
    for nm, w in (("", np.ones(int(ok16.sum()))), ("w", POP[ok16])):
        Dm = D16.mean(1)
        idn["g16" + nm] = dict(n=float(w.sum()), sse16=float(w @ (Dm * Dm)),
                               s2_16=float(w @ (((D16 - Dm[:, None]) ** 2).sum(1) / 15.0)))
    del D16
    # ---- the members at the station pixels (for the station scorer)
    sp = S["STPOS"]; has = sp >= 0; spc = np.clip(sp, 0, None)
    st = {"g2": E4[spc].astype(np.float16), "g16": X16[spc].astype(np.float16)}
    if N4 is not None:
        st["neg"] = N4[spc].astype(np.float16)
    for v in st.values():
        v[~has] = np.nan
    # ---- the cases at the populated pixels
    E4v, X16v, gv, popv = E4[valid], X16[valid], g[valid], POP[valid]
    N4v = N4[valid] if N4 is not None else None
    cls = X16v[:, 8:].mean(1)
    lab = np.where(cls < 15.0, 0, np.where(cls <= 75.0, 1, 2))
    ones = np.ones(len(gv))
    Z = None
    if not S["final"]:
        rng = np.random.default_rng([SEED_SYN, 2, d, h])
        Z = rng.standard_normal((len(gv), 24), dtype=np.float32)
    sums = {}
    for case in S["cases"]:
        X8, Y16 = build_case(case, E4v, X16v, N4v, Z)
        cs = {}
        for name, col in columns(X8, Y16, gv):
            arr = np.zeros((2, 4))
            for wi, w in enumerate((ones, popv)):
                b = np.bincount(lab, weights=w * col, minlength=3)
                arr[wi, 1:] = b[:3]; arr[wi, 0] = b[:3].sum()
            cs[name] = arr
        sums[case] = cs
        del X8, Y16
    del E4v, X16v, gv, popv, N4v, Z, cls, lab
    # ---- warnings (World): at least 4 of 8 paired, at least 2 of 4 (G2), at least 4 of 8 of G11 0..7; k of 16 by truth code
    g40 = c40[1][bc.S["cell"]]
    okw = np.isfinite(g) & np.isfinite(g40) & fin4 & fin16 & finN
    w = np.where(okw, bc.S["pop"], 0.0)
    X8p = np.concatenate([E4, N4], 1) if N4 is not None else np.concatenate([E4, X16[:, 4:8]], 1)
    fns = [lambda thr, C: (X8p > thr).sum(1) >= 4, lambda thr, C: (E4 > thr).sum(1) >= 2,
           lambda thr, C: (X16[:, :8] > thr).sum(1) >= 4]
    with np.errstate(invalid="ignore"):
        T = bc.tables(w, g, g40, fns)[0]                                 # World: (5 thresholds, 3 systems, 16)
        HK = np.zeros((len(bc.THR), 8, 17))
        for j, thr in enumerate(bc.THR):
            Ex, Cx = g > thr, g40 > thr
            Hm = (g > thr * (1 + bc.MARGIN)) & (g40 <= thr * (1 - bc.MARGIN))
            code = Ex.astype(np.int64) + 2 * Cx + 4 * Hm
            HK[j] = np.bincount(code * 17 + (X16 > thr).sum(1), weights=w, minlength=8 * 17).reshape(8, 17)
    diag["valid_residents_M_warnings"] = float(w.sum() / 1e6)
    diag["seconds"] = round(time.time() - t0, 1)
    return key, t, sums, idn, st, T, HK, diag


# ======================================================================== bootstrap and statistics of summed columns
def boot_counts(nd):
    """counts (10,000, nd) of compare_fair's moving-block bootstrap (= rel_common.block_weights): default_rng(0), blocks of 7"""
    rng = np.random.default_rng(0)
    idx = (rng.integers(0, nd - BLK + 1, size=(NB_, -(-nd // BLK)))[:, :, None] + np.arange(BLK)).reshape(NB_, -1)[:, :nd]
    W = np.zeros((NB_, nd))
    np.add.at(W, (np.arange(NB_)[:, None], idx), 1.0)
    return W


def summed(Sd, W):
    """Sd {col: (days, 2 leads)} per-day sums -> {L: {col: (10,001,) point then the resamples}} for L = 1, 3, 13"""
    out = {}
    for L in ("1", "3", "13"):
        out[L] = {}
        for k, a in Sd.items():
            v = a[:, 0] if L == "1" else (a[:, 1] if L == "3" else a[:, 0] + a[:, 1])
            out[L][k] = np.concatenate([[v.sum()], W @ v])
    return out


def stats(g, law16=True, refs=()):
    """every statistic, for the point and each resample at once; g {col: (10,001,)}"""
    with np.errstate(all="ignore"):
        n = g["one"]
        m = {k: g[k] / n for k in ("sse1", "sse1n", "sse4", "sse2p", "sse2p6", "sse1p", "sse8p", "sse16", "ssei8")}
        o = {"n": n, "mean_y": g["y"] / n}
        for k, v in m.items():
            o["mse_" + k[3:]] = v
            o["rmse_" + k[3:]] = np.sqrt(v)
        V16, Vi8 = g["s2_16"] / n, g["s2_i8"] / n
        inf16, inf8 = m["sse16"] - V16 / 16, m["ssei8"] - Vi8 / 8
        if law16:
            o.update(V16=V16, mse_inf=inf16, mse_form8=inf16 + V16 / 8, mse_form4=inf16 + V16 / 4, rmse_inf=np.sqrt(inf16),
                     rmse_form8=np.sqrt(inf16 + V16 / 8), D=inf16 + V16 / 8 - m["sse8p"], D_rel_V=(inf16 + V16 / 8 - m["sse8p"]) / V16,
                     inv_meq_8p=(m["sse8p"] - inf16) / V16, inv_meq_2p=(m["sse2p"] - inf16) / V16, inv_meq_4=(m["sse4"] - inf16) / V16,
                     inv_meq_1p=(m["sse1p"] - inf16) / V16, O_rel_V=(m["sse4"] - m["sse2p"]) / V16,
                     soe_law8=np.sqrt(9 / 8 * V16) / np.sqrt(inf16 + V16 / 8))
        o.update(V_i8=Vi8, mse_inf_law8=inf8, inv_meq_8p_law8=(m["sse8p"] - inf8) / Vi8, D_asrun=m["ssei8"] - m["sse8p"],
                 O=m["sse4"] - m["sse2p"], O6=m["sse4"] - m["sse2p6"])
        for nm, p_ in (("rho_anti", "ra_"), ("rho_null", "rn_"), ("rho_null03", "r3_"), ("rho_anti_log", "la_"), ("rho_null_log", "ln_")):
            A, B = g[p_ + "A"], g[p_ + "B"]
            o[nm] = (A - B) / (A + B)
            o[nm + "_raw"] = g[p_ + "dd"] / np.sqrt(g[p_ + "da2"] * g[p_ + "db2"])
        V4, Vn, V8 = g["s2_4"] / n, g["s2_n"] / n, g["s2_8"] / n
        o.update(V4=V4, V_neg=Vn, V8_paired_naive=V8, rmse_ratio_neg_over_orig=np.sqrt(m["sse1n"] / m["sse1"]),
                 spread_ratio_neg_over_orig=Vn / V4, bias=g["bias"] / n, bias_rel=g["bias"] / g["y"],
                 soe_4=np.sqrt(5 / 4 * V4) / np.sqrt(m["sse4"]), soe_8p_naive=np.sqrt(9 / 8 * V8) / np.sqrt(m["sse8p"]),
                 soe_i8=np.sqrt(9 / 8 * Vi8) / np.sqrt(m["ssei8"]))
        for r in refs:                                                   # reductions (%) against the references (stations)
            mr = g["sse_" + r] / n
            o["rmse_" + r] = np.sqrt(mr)
            sysm = {"four": m["sse4"], "two_pairs": m["sse2p"], "four_pairs": m["sse8p"], "eight_indep_asrun": m["ssei8"],
                    "g1_four": g["sse_g1"] / n}
            if law16:
                sysm.update(form8=inf16 + V16 / 8, inf=inf16)
            for s_, v in sysm.items():
                o[f"red_{s_}_vs_{r}"] = 100 * (1 - np.sqrt(v / mr))
        if "sse_g1" in g:
            o["rmse_g1_four"] = np.sqrt(g["sse_g1"] / n)
    return o


def summarize(o):
    """{stat: {est, ci (95 % percentile)}} with M_eq inverted from its inverse"""
    out = {}
    for k, v in o.items():
        v = np.asarray(v, np.float64)
        bs = v[1:]
        lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan))
        out[k] = {"est": float(v[0]), "ci": [float(lo), float(hi)]}
    for k in [k for k in out if k.startswith("inv_meq")]:
        e, (lo, hi) = out[k]["est"], out[k]["ci"]
        out["meq" + k[7:]] = {"est": 1 / e if e > 0 else float("inf"), "ci": [1 / hi if hi > 0 else float("inf"), 1 / lo if lo > 0 else float("inf")]}
    return out


# ======================================================================== decision (section 7, fixed in advance)
def guard_of(px):
    """guard G (A17 6.3 ratio tolerances, bias 0.5 %) at the populated pixels, pixels weight, regime all, leads 1 and 3"""
    G_ = {}
    for L in ("1", "3"):
        s = px[L]
        G_[f"rmse_ratio_L{L}_within_5pct"] = abs(s["rmse_ratio_neg_over_orig"]["est"] - 1) <= TOL_G["rmse_ratio"]
        G_[f"spread_ratio_L{L}_within_15pct"] = abs(s["spread_ratio_neg_over_orig"]["est"] - 1) <= TOL_G["spread_ratio"]
        G_[f"bias_L{L}_within_0p5pct_of_truth"] = abs(s["bias_rel"]["est"]) <= TOL_G["bias_rel"]
    return G_


def decide(st, invalid_reasons):
    est = lambda L, k: float(st[L][k]["est"])                            # noqa: E731
    lo = lambda L, k: float(st[L][k]["ci"][0])                           # noqa: E731
    cond = {"D_1_above_0": est("1", "D") > 0, "D_3_above_0": est("3", "D") > 0, "D_1_interval_above_0": lo("1", "D") > 0}
    order = {"pooled_interval_of_O_above_0": lo("13", "O") > 0}
    if invalid_reasons:
        outcome, reasons = "INVALID", list(invalid_reasons)
    elif all(cond.values()):
        outcome, reasons = "SUCCESS", ["every SUCCESS condition of section 7 holds"]
    elif all(order.values()):
        outcome, reasons = "ORDER", ["pooled interval of MSE(four) - MSE(two pairs) above 0; SUCCESS not met:"] + [k for k, v in cond.items() if not v]
    else:
        outcome, reasons = "FAIL", ["SUCCESS not met: " + ", ".join(k for k, v in cond.items() if not v), "pooled interval of O not above 0"]
    return dict(outcome=outcome, reasons=reasons, success_conditions=cond, order_conditions=order)


# ======================================================================== assembling the stations
FORM_COLS = ("one", "sseF", "s2MF", "s2rawF", "sse_g1", "s2_g1")


def station_block(prep, res, cases, final):
    """per-day sums at the stations: S' (stations with a populated pixel, every member finite) for each case, and the
    paper's full station-days for the identity (G1 as run, the A14 forms with the 57-station fallback)"""
    days = prep["days"]
    ND = len(days)
    Sd = {c: {} for c in cases}
    Fd = {k: np.zeros((ND, 2)) for k in FORM_COLS}
    STP = S["STPOS"]
    nst = {"S_prime_station_days": [0, 0], "full_station_days": [0, 0], "dropped_nonfinite_at_populated_stations": [0, 0],
           "fallback_station_days_57": [0, 0]}
    st16_ok = True
    Z16 = np.load(A14 / "classU" / "data_ar" / "st16_run16.npz")
    K16 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(Z16["keys"])}
    stpos_equal = bool(np.array_equal(Z16["stpos"], STP))
    for key in sorted(prep["rec"], key=lambda k: (k[1], k[0])):
        d, h = key
        r = prep["rec"][key]
        li, di = LEADS.index(h), days.index(d)
        st = res[key]["st"]
        st16_ok &= bool(np.array_equal(st["g16"].view(np.uint16), Z16["ens16"][K16[key]].view(np.uint16)))
        s = r["s"]
        y64 = r["y"].astype(np.float64)
        # ---- the paper's full station-days (identity): G1 as run, A14 st_cf_score sixteen-member forms with fallback
        E16 = st["g16"][s].astype(np.float32); E4 = r["E4"]
        f16 = np.isfinite(E16).all(1); f4 = ~f16
        x16 = E16.mean(1).astype(np.float64); v16 = E16.var(1, ddof=1).astype(np.float64)
        x4 = E4.mean(1).astype(np.float64); v4 = E4.var(1, ddof=1).astype(np.float64)
        Fd["one"][di, li] = len(s)
        Fd["sseF"][di, li] = ((x16[f16] - y64[f16]) ** 2).sum() + ((x4[f4] - y64[f4]) ** 2).sum()
        Fd["s2MF"][di, li] = v16[f16].sum() / 16 + v4[f4].sum() / 4
        Fd["s2rawF"][di, li] = v16[f16].sum() + v4[f4].sum()
        Fd["sse_g1"][di, li] = ((r["g1"].astype(np.float64) - y64) ** 2).sum()
        Fd["s2_g1"][di, li] = v4.sum()
        nst["full_station_days"][li] += len(s); nst["fallback_station_days_57"][li] += int(f4.sum())
        # ---- S'
        G2s = st["g2"][s].astype(np.float64); G16s = st["g16"][s].astype(np.float64)
        Ns = st["neg"][s].astype(np.float64) if final else None
        fin = (STP[s] >= 0) & np.isfinite(G2s).all(1) & np.isfinite(G16s).all(1)
        if final:
            fin &= np.isfinite(Ns).all(1)
        nst["dropped_nonfinite_at_populated_stations"][li] += int(((STP[s] >= 0) & ~fin).sum())
        nst["S_prime_station_days"][li] += int(fin.sum())
        Zs = None
        if not final:
            Zs = np.random.default_rng([SEED_SYN, 1, d, h]).standard_normal((int(fin.sum()), 24))
        extra = {"sse_cams": (r["cams"].astype(np.float64) - y64) ** 2, "sse_aur": (r["aur"].astype(np.float64) - y64) ** 2,
                 "sse_g1": (r["g1"].astype(np.float64) - y64) ** 2}
        for c in cases:
            X8, Y16 = build_case(c, G2s[fin], G16s[fin], Ns[fin] if final else None, Zs)
            cols = dict(columns(X8, Y16, y64[fin]))
            cols.update({k: v[fin] for k, v in extra.items()})
            for k, v in cols.items():
                if k not in Sd[c]:
                    Sd[c][k] = np.zeros((ND, 2))
                Sd[c][k][di, li] = float(np.sum(v))
    ok_d = np.isfinite(prep["PD"][0, :, 0, 0])
    return Sd, Fd, ok_d, nst, dict(stpos_equal_a14_st16=stpos_equal, g11_station_members_equal_a14_st16=bool(st16_ok))


def station_identity(prep, Fd, ok_d):
    """the paper's station values from this scorer: G1 as run (PD), the A14 forms, against st_cf_classU_179.json"""
    PD = prep["PD"]
    W = boot_counts(int(ok_d.sum()))
    ref = json.loads((A14 / "classU" / "results_ar" / "st_cf_classU_179.json").read_text())["groups"]["all"]
    out, ok = {}, True
    for h in LEADS:
        li = LEADS.index(h)
        R = ref[str(h)]["Foehn 1 km"]
        a = PD[0, ok_d, h - 1]; sa, ca = a[:, 0], a[:, 1]
        pt = np.sqrt(np.nansum(sa) / np.nansum(ca))
        o = {"rmse_four_as_run_G1": pt, "a14": R["rmse"]["as_run"]["v"]}
        for nm, j, rk in (("cams", 4, "CAMS d+1 (equal issuance)"), ("aurora", 6, "Aurora 00 UTC d+1 (equal issuance)")):
            b = PD[j, ok_d, h - 1]
            o[f"gain_as_run_{nm}"] = 100 * (pt / np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1])) - 1)
            o[f"a14_gain_as_run_{nm}"] = R["gain_pct"][rk]["as_run"]["v"]
        F = {k: v[ok_d, li] for k, v in Fd.items()}
        N_ = F["one"].sum()
        inf_ = np.sqrt((F["sseF"].sum() - F["s2MF"].sum()) / N_)
        m8 = np.sqrt((F["sseF"].sum() - F["s2MF"].sum() + F["s2rawF"].sum() / 8) / N_)
        sub = R["sixteen_members_provisional"]
        o.update(form8=m8, inf=inf_, a14_form8=sub["sixteen_M8"], a14_inf=sub["sixteen_inf"])
        for form, num in (("inf", F["sseF"] - F["s2MF"]), ("M8", F["sseF"] - F["s2MF"] + F["s2rawF"] / 8)):
            bs = np.sqrt((W @ num) / (W @ F["one"]))
            o[f"rmse16_{form}_ci"] = list(np.nanpercentile(bs, [2.5, 97.5]))
            o[f"a14_rmse16_{form}_ci"] = sub[f"rmse16_{form}"]["ci_boot"]
            for nm, j, rk in (("cams", 4, "CAMS d+1 (equal issuance)"), ("aurora", 6, "Aurora 00 UTC d+1 (equal issuance)")):
                b = PD[j, ok_d, h - 1]
                o[f"gain16_{form}_{nm}"] = 100 * (np.sqrt(num.sum() / F["one"].sum()) / np.sqrt(np.nansum(b[:, 0]) / np.nansum(b[:, 1])) - 1)
                o[f"a14_gain16_{form}_{nm}"] = sub[f"gain16_{form}_{rk}"]["v"]
        o["rmse_four_as_run_G1_recomputed_float64"] = float(np.sqrt(F["sse_g1"].sum() / N_))
        rel = lambda a_, b_: abs(a_ - b_) / abs(b_)                         # noqa: E731
        chk = {"as_run_1e-9": rel(o["rmse_four_as_run_G1"], o["a14"]) <= 1e-9,
               "gains_as_run_1e-9": all(rel(o[f"gain_as_run_{nm}"], o[f"a14_gain_as_run_{nm}"]) <= 1e-9 for nm in ("cams", "aurora")),
               "forms_1e-6": rel(m8, o["a14_form8"]) <= 1e-6 and rel(inf_, o["a14_inf"]) <= 1e-6,
               "form_intervals_1e-6": all(rel(x_, y_) <= 1e-6 for f_ in ("inf", "M8") for x_, y_ in zip(o[f"rmse16_{f_}_ci"], o[f"a14_rmse16_{f_}_ci"])),
               "form_gains_1e-6": all(rel(o[f"gain16_{f_}_{nm}"], o[f"a14_gain16_{f_}_{nm}"]) <= 1e-6 for f_ in ("inf", "M8") for nm in ("cams", "aurora")),
               "printed": (f"{pt:.2f}", f"{m8:.2f}", f"{inf_:.2f}") == {1: ("13.00", "12.78", "12.55"), 3: ("14.21", "13.90", "13.59")}[h],
               "printed_reductions_cams": (f"{-o['gain_as_run_cams']:.1f}", f"{-o['gain16_M8_cams']:.1f}", f"{-o['gain16_inf_cams']:.1f}")
               == {1: ("15.5", "16.9", "18.4"), 3: ("11.4", "13.3", "15.2")}[h],
               "float64_recomputation_1e-6": rel(o["rmse_four_as_run_G1_recomputed_float64"], pt) <= 1e-6}
        o["checks"] = chk
        ok &= all(chk.values())
        out[f"L{h}"] = o
    return out, bool(ok)


# ======================================================================== warnings
WQ = ("issued:warned_M", "issued:reach_pct", "issued:correct_pct", "issued:hidden_warned_pct", "eqn:cams:reach_pct",
      "eqn:cams:correct_pct", "eqn:cams:hidden_warned_pct", "diff:cams:reach_pct", "diff:cams:correct_pct")


def warnings_block(K, T, HK):
    """'at least 4 of 8' of the paired members (T[..., 0]) beside the paper's record; identity of the stored four and of the
    eight-member expectation from the sixteen (classE functions imported unchanged)"""
    zv = np.load(v3.DATA / "hidden_v3_pop.npz")
    s1, hs = [str(x) for x in zv["sys1"]], [str(x) for x in zv["hsys"]]
    Kv = zv["keys"]
    rows_v = [int(np.flatnonzero((Kv[:, 0] == d) & (Kv[:, 1] == h))[0]) for d, h, _ in K]
    assert np.array_equal(Kv[rows_v], K)
    T1 = zv["T1"][rows_v]
    ident = {"stored_four_2_of_4_equals_v3_foehn_p50": bool(np.array_equal(T[:, :, 1, :], T1[:, 0, :, s1.index("foehn_p50"), :]))}
    RT = T1[:, 0, :, s1.index("cams"), :]
    RH = zv["H"][rows_v][:, :, hs.index("cams")]
    del zv, T1
    rec = json.loads((A14 / "classE" / "results_ar" / "classE_all2022.json").read_text())["results"]["world"]["conf149"]
    f2 = json.loads((NATC / "rebuild" / "print" / "warn8" / "out_ar" / "fig2f.json").read_text())["systems"]
    out, dev = {}, 0.0
    for h in LEADS:
        for thr in (37.5, 75.0):
            j = list(bc.THR).index(thr)
            rows, cnt, n, how = ce.sample_rows(K, h, "conf149")
            systems = {"paired_4_of_8": T[rows, j, 0], "independent_G11_0_7_4_of_8": T[rows, j, 2], "stored_four_2_of_4": T[rows, j, 1],
                       "M8_liberal_recomputed": ce.tables_from_k(HK[rows, j], ce.pi_table(16, 8, "liberal"))}
            RIT = {"cams": ce.Tab(RT[rows, j], cnt)}
            res = {}
            for sname, a in systems.items():
                tab = ce.Tab(a, cnt)
                q1, _, _ = ce.boot_quantities(tab, RIT, {}, cnt, n)
                q2, info, _ = ce.boot_quantities(tab, {}, {"cams": RH[rows, j]}, cnt, n, full=None)
                q1.update(q2)
                res[sname] = {q: ce.summarize(*q1[q], None) for q in WQ}
                res[sname]["eqn_cams_info"] = info.get("eqn:cams")
            R_ = rec[f"thr{thr:g}"][f"lead{h}"]["quantities"]
            paper = {q: {s: {"value": R_[q][s]["value"], "interval95": R_[q][s]["interval95"]} for s in ("M8_liberal", "M16_liberal_measured", "M4_stored")}
                     for q in WQ}
            for q in WQ:
                a_, b_ = res["M8_liberal_recomputed"][q]["value"], paper[q]["M8_liberal"]["value"]
                dev = max(dev, abs(a_ - b_) / max(abs(b_), 1e-9))
                for x_, y_ in zip(res["M8_liberal_recomputed"][q]["interval95"], paper[q]["M8_liberal"]["interval95"]):
                    dev = max(dev, abs(x_ - y_) / max(abs(y_), 1e-9))
            key = f"lead{h}_thr{thr:g}"
            out[key] = dict(n_start_days=int(n), bootstrap=how, systems={k: {q: dict(value=v[q]["value"], interval95=v[q]["interval95"])
                                                                             for q in WQ} for k, v in res.items()},
                            paired_eqn_cams_factor=res["paired_4_of_8"]["eqn_cams_info"], paper_record=paper)
            if h == 1:
                fg = f2[f"thr{thr:g}"]["lead1"]["World"]["foehn_m8"]
                ident[f"warn8_fig2f_equals_record_{key}"] = all(abs(fg[m][0] - paper[f"issued:{m}"]["M8_liberal"]["value"]) <= 1e-9 for m in ("reach_pct", "correct_pct"))
    ident["M8_liberal_recomputed_equals_record_max_rel_dev"] = dev
    ident["M8_liberal_recomputed_equals_record"] = dev <= 1e-6
    return out, ident


# ======================================================================== tests (stored data only) and the decision run
def near(v, t):
    """known-answer rule (A17): |estimate - target| <= 1.5 x the half-width of the 95 % interval"""
    lo, hi = v["ci"]
    return dict(est=v["est"], target=t, ci=[lo, hi], interval_holds=bool(lo <= t <= hi),
                ok=bool(abs(v["est"] - t) <= max(1.5 * (hi - lo) / 2, 1e-9)))


EXPECT = {   # case: (description, {stat: target}, exact {stat: target}, class at the stations)
    "N0": ("the real members of 2022, no A19 output: G2's 0..3 with G11's independent 4..7 in place of the negated members; "
           "rho_null03 = G2's members 0..3 against themselves as pseudo-pairs", {"rho_anti": 0.0, "rho_null": 0.0, "rho_null03": 0.0,
           "inv_meq_8p": 1 / 8, "D_rel_V": 0.0, "O_rel_V": 0.0, "bias_rel": 0.0}, {}, "FAIL"),
    "S0": ("synthetic independent pairs (c = 0) around G2's mean with G2's spread", {"rho_anti": 0.0, "rho_null": 0.0,
           "inv_meq_8p": 1 / 8, "inv_meq_2p": 1 / 4, "D_rel_V": 0.0, "O_rel_V": 0.0}, {}, "FAIL"),
    "S1": (f"synthetic pairs with within-pair correlation c = {C_S1} (the A17 value of 2021)", {"rho_anti": C_S1, "rho_null": 0.0,
           "inv_meq_8p": (1 + C_S1) / 8, "inv_meq_2p": (1 + C_S1) / 4, "D_rel_V": -C_S1 / 8, "O_rel_V": -C_S1 / 4}, {}, "SUCCESS"),
    "S2": ("pairs 0, 1 exact reflections about the mean, pairs 2, 3 copies: two pairs (0, 4, 1, 5) are exact, four pairs are "
           "worth eight independent members", {"rho_anti": 0.0, "inv_meq_8p": 1 / 8, "inv_meq_2p": 0.0, "D_rel_V": 0.0, "O_rel_V": 0.25},
           {}, "ORDER"),
    "S3": ("exact reflections x(4 + k) = 2 mu - x(k): the eight-member mean is mu", {"inv_meq_8p": 0.0, "D_rel_V": 1 / 8, "O_rel_V": 0.25},
           {"rho_anti": -1.0}, "SUCCESS"),
    "S4": ("copies x(4 + k) = x(k): the eight-member mean is the four-member mean", {"inv_meq_8p": 1 / 4, "D_rel_V": -1 / 8, "O_rel_V": -1 / 4},
           {"rho_anti": 1.0}, "FAIL"),
}


def evaluate(final):
    t0 = time.time()
    prep = station_prep()
    bc.setup()
    pp = np.load(G / "pop_pixels.npz")
    S["POP"] = pp["pop"].astype(np.float64); S["POPOK"] = S["POP"] >= float(pp["min_pop"])
    assert np.array_equal(pp["idx"], np.load(G2D / "pop_idx.npy"))
    S["STPOS"] = stations_in_pop()
    S["K2"], S["K16"] = part_index(G2D), index16(G11D)
    S["final"] = final
    S["cases"] = ("real",) if final else CASES_TEST
    keys = sorted(S["K2"], key=lambda k: (k[1], k[0]))
    assert len(keys) == 358 and set(keys) == set(S["K16"]), (len(keys), len(S["K16"]))
    fmt = {}
    if final:
        fmt = folder_checks()
        S["KN"] = part_index(NEGD)
        assert set(S["KN"]) == set(keys), "pop_neg does not hold G2's 358 forecasts"
        assert np.array_equal(pp["idx"], np.load(NEGD / "pop_idx.npy"))
    del pp
    log(f"pixel pass: {len(keys)} forecasts, cases {S['cases']}")
    with get_context("fork").Pool(16) as pool:
        out = pool.map(one, keys, chunksize=1)
    res = {k: dict(t=t, sums=s, idn=i, st=st, T=T, HK=HK, diag=dg) for k, t, s, i, st, T, HK, dg in out}
    del out
    log(f"pixel pass done {time.time() - t0:.0f} s")
    # ---- identities at the populated pixels
    p4 = np.load(A14 / "classU" / "data_ar" / "pop4_terms.npz"); P4 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(p4["keys"])}
    sp = np.load(NATC / "analyses" / "reliability" / "data_ar" / "sums_pop.npz"); SP = {(int(d), int(h)): i for i, (d, h, t) in enumerate(sp["keys"])}
    q16 = np.load(A14 / "classU" / "data_ar" / "pop16_run16.npz"); Q16 = {(int(d), int(h)): i for i, (d, h, t) in enumerate(q16["keys"])}
    dev = {"pop4_terms": 0.0, "sums_pop": 0.0, "pop16_run16": 0.0}
    for k in keys:
        I = res[k]["idn"]
        for nm in ("", "w"):
            for q in ("n", "sse", "var"):
                a = I[nm][q]
                dev["pop4_terms"] = max(dev["pop4_terms"], abs(a - float(p4[nm + q][P4[k]])) / max(abs(float(p4[nm + q][P4[k]])), 1e-9))
                b = float(sp[nm + q][SP[k], -1])
                dev["sums_pop"] = max(dev["sums_pop"], abs(a - b) / max(abs(b), 1e-9))
            wi = 0 if nm == "" else 1
            Dm = q16["D"][Q16[k], wi]
            sse16 = Dm.sum() / 256.0; s2 = (np.trace(Dm) - 16 * sse16) / 15.0
            for a, b in ((I["g16" + nm]["n"], float(q16["n"][Q16[k], wi])), (I["g16" + nm]["sse16"], sse16), (I["g16" + nm]["s2_16"], s2)):
                dev["pop16_run16"] = max(dev["pop16_run16"], abs(a - b) / max(abs(b), 1e-9))
    ok4_not_valid = sum(res[k]["diag"]["ok4_not_valid"] for k in keys)
    pix_ident = dict(max_rel_dev=dev, ok4_pixel_days_not_valid=ok4_not_valid,
                     passed=dev["pop4_terms"] <= 1e-9 and dev["sums_pop"] <= 1e-6 and dev["pop16_run16"] <= 1e-5 and ok4_not_valid == 0)
    # ---- the populated pixels: per-day sums by case, weight, regime -> statistics
    days = sorted({d for d, _ in keys}); assert len(days) == 179
    Wp = boot_counts(179)
    PX = {}
    for c in S["cases"]:
        PX[c] = {}
        for wi, wn in enumerate(WTS):
            PX[c][wn] = {}
            for ri, rn in enumerate(REGS):
                Sd = {}
                for k in keys:
                    li, di = LEADS.index(k[1]), days.index(k[0])
                    for col, arr in res[k]["sums"][c].items():
                        if col not in Sd:
                            Sd[col] = np.zeros((179, 2))
                        Sd[col][di, li] = arr[wi, ri]
                g_ = summed(Sd, Wp)
                PX[c][wn][rn] = {L: summarize(stats(g_[L], law16=(rn == "all"))) for L in g_}
    log(f"pixel statistics {time.time() - t0:.0f} s")
    # ---- the stations
    Sd, Fd, ok_d, nst, st_ident = station_block(prep, res, S["cases"], final)
    Ws = boot_counts(int(ok_d.sum()))
    ST = {}
    for c in S["cases"]:
        g_ = summed({k: v[ok_d] for k, v in Sd[c].items()}, Ws)
        ST[c] = {L: summarize(stats(g_[L], law16=True, refs=("cams", "aur"))) for L in g_}
    sid, sid_ok = station_identity(prep, Fd, ok_d)
    log(f"station statistics {time.time() - t0:.0f} s")
    # ---- warnings
    K = np.array([(d, h, res[(d, h)]["t"]) for d, h in keys], np.int64)
    T = np.stack([res[k]["T"] for k in keys]); HK = np.stack([res[k]["HK"] for k in keys])
    try:
        warn, wident = warnings_block(K, T, HK)
    except Exception as e:                                                   # reported, never gating
        warn, wident = {"error": repr(e)}, {"error": repr(e)}
    log(f"warnings {time.time() - t0:.0f} s")
    identity = dict(compare_fair=prep["ident"], compare_fair_sha256=prep["cf_sha"], stations=sid, stations_passed=sid_ok,
                    station_members=st_ident, pixels=pix_ident, warnings=wident)
    id_ok = (all(v is True or (isinstance(v, int) and not isinstance(v, bool)) for v in prep["ident"].values()) and sid_ok
             and all(st_ident.values()) and pix_ident["passed"])
    return dict(prep=prep, res=res, ST=ST, PX=PX, nst=nst, identity=identity, id_ok=bool(id_ok), warn=warn, fmt=fmt,
                seconds=round(time.time() - t0, 1), keys=keys)


def folder_checks():
    """the A19 run folder: names, manifest, parts (format guard G0; read before any member value)"""
    names = sorted(os.listdir(NEGD))
    want = {"a19_run.json", "pop_idx.npy"} | {f"{p}_{r}.{e}" for r in range(64) for p, e in (("part", "npz"), ("pop", "npy"), ("pop_ghap", "npy"), ("cell40", "npy"))}
    man = json.loads((NEGD / "a19_run.json").read_text())
    chk = json.loads((OUT / "check19.json").read_text())
    G2K = part_index(G2D)
    parts_ok = True
    jobs = set()
    for r in range(64):
        with np.load(NEGD / f"part_{r}.npz") as z:
            parts_ok &= (json.loads(str(z["a19"])) == man and [int(m) for m in z["member_ids"]] == [4, 5, 6, 7]
                         and str(z["base"]) == "philox" and all(float(x) == 0.59 for x in z["keys_rho"]) and z["st_ens"].shape[1] == 0)
            jobs |= {str(x) for x in z["a19_jobs"]}
            kk = [tuple(int(v) for v in k) for k in z["keys"].tolist()]
            parts_ok &= all(G2K[(d, h)][0] == r and G2K[(d, h)][2] == t for d, h, t in kk)
    return dict(names_exact=set(names) == want, manifest_plan_main=man["plan"] == "main", manifest_ids=man["member_ids"] == [4, 5, 6, 7],
                manifest_rho_base=float(man["rho"]) == 0.59 and man["base"] == "philox",
                code_of_the_run_is_the_checked_code=man["code_sha256"] == chk.get("code_sha256"), parts_ok=bool(parts_ok),
                jobs=sorted(jobs))


SPREAD_NOTE = ("A paired ensemble's sample variance is not an estimate of its forecast variance: the two members of a pair "
               "are negatively correlated, so the mean of the pairs is closer to the infinite-ensemble mean than the members' "
               "spread suggests (E[s2] = V (7 - c)/7 for eight paired members, while MSE of the mean = MSE_inf + V (1 + c)/8). "
               "The naive spread over error of the paired members is printed for information only; calibrating an ensemble "
               "made of pairs needs its own estimators (A17 7.4).")
HEAD = ("n", "rmse_4", "rmse_2p", "rmse_8p", "rmse_form8", "rmse_inf", "rmse_i8", "D", "D_rel_V", "O", "D_asrun", "rho_anti",
        "rho_anti_raw", "rho_null", "rho_null03", "meq_8p", "meq_2p", "red_four_vs_cams", "red_two_pairs_vs_cams",
        "red_four_pairs_vs_cams", "red_form8_vs_cams", "red_inf_vs_cams", "red_four_vs_aur", "red_four_pairs_vs_aur",
        "red_form8_vs_aur", "red_inf_vs_aur", "rmse_ratio_neg_over_orig", "spread_ratio_neg_over_orig", "bias_rel",
        "soe_4", "soe_8p_naive", "soe_i8", "soe_law8")


def headline(st):
    return {L: {k: st[L][k] for k in HEAD if k in st[L]} for L in st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.tests:
        E = evaluate(False)
        cases = {}
        for c in CASES_TEST:
            st, px = E["ST"][c], E["PX"][c]["pixels"]["all"]
            G_ = guard_of(px)
            d = decide(st, [] if all(G_.values()) else ["guard G: " + ", ".join(k for k, v in G_.items() if not v)])
            desc, tg, ex, want = EXPECT[c]
            chk = {f"stations_{k}": near(st["13"][k], v) for k, v in tg.items()}
            chk.update({f"pixels_{k}": near(px["13"][k], v) for k, v in tg.items()})
            for k, v in ex.items():
                chk[f"stations_{k}_exact"] = dict(est=st["13"][k]["est"], target=v, ok=abs(st["13"][k]["est"] - v) < 1e-6)
                chk[f"pixels_{k}_exact"] = dict(est=px["13"][k]["est"], target=v, ok=abs(px["13"][k]["est"] - v) < 1e-6)
            if c == "N0":                                                      # the null of the ORDER and D intervals
                chk["stations_D_1_interval_holds_0"] = dict(ci=st["1"]["D"]["ci"], ok=st["1"]["D"]["ci"][0] <= 0 <= st["1"]["D"]["ci"][1])
                chk["stations_O_13_interval_holds_0"] = dict(ci=st["13"]["O"]["ci"], ok=st["13"]["O"]["ci"][0] <= 0 <= st["13"]["O"]["ci"][1])
            pxd = decide(px, [])
            cases[c] = dict(expect=desc + f"; class at the stations {want}", checks=chk, outcome_stations=d["outcome"], decision=d,
                            outcome_pixels_reported=pxd["outcome"], guard_pixels=G_,
                            passed=all(v["ok"] for v in chk.values()) and d["outcome"] == want,
                            stations=headline(st), pixels=headline(px))
        Rr = {"what": __doc__, "score19_sha256": SELF_SHA256, "mode": "tests (no A19 output read)", "identity": E["identity"],
              "identity_passed": E["id_ok"], "station_sample": E["nst"], "cases": cases,
              "paper_values_on_S_prime_N0": {L: {k: E["ST"]["N0"][L][k] for k in ("rmse_4", "rmse_g1_four", "rmse_form8", "rmse_inf",
                                                                                   "rmse_i8", "red_four_vs_cams", "red_form8_vs_cams", "red_inf_vs_cams",
                                                                                   "red_four_vs_aur", "red_form8_vs_aur", "red_inf_vs_aur", "n")}
                                             for L in ("1", "3", "13")},
              "pixels_all_cases": E["PX"], "stations_all_cases": E["ST"], "warnings_real_data": E["warn"],
              "spread_note": SPREAD_NOTE, "seconds": E["seconds"]}
        Rr["all_passed"] = bool(E["id_ok"] and all(v["passed"] for v in cases.values()))
        save_json(OUT / "tests19.json", Rr)
        log(json.dumps(jsonable({k: (v["passed"], v["outcome_stations"]) for k, v in cases.items()})), "identity", E["id_ok"],
            "ALL PASSED" if Rr["all_passed"] else "SOME TESTS FAILED", f"score19 sha256 {SELF_SHA256}")
        return 0 if Rr["all_passed"] else 1
    sm = json.loads((DATA / "smoke_check.json").read_text()) if (DATA / "smoke_check.json").exists() else {}
    E = evaluate(True)
    st, px = E["ST"]["real"], E["PX"]["real"]
    G_ = guard_of(px["pixels"]["all"])
    fmt_ok = all(v for k, v in E["fmt"].items() if k != "jobs")
    G0 = dict(folder=E["fmt"], every_forecast={k_: all(E["res"][k]["diag"]["format"][k_] for k in E["keys"])
                                               for k_ in ("same_rank_and_row_as_G2", "pop_ghap_bit_identical", "cell40_ghap_identical", "finite_pattern_equal")},
              station_days_dropped=E["nst"]["dropped_nonfinite_at_populated_stations"])
    g0_ok = fmt_ok and all(G0["every_forecast"].values()) and sum(G0["station_days_dropped"]) == 0
    inv = ([] if sm.get("all_passed") is True else ["smoke gate not passed"]) + ([] if E["id_ok"] else ["identity checks of the scorer"]) \
        + ([] if g0_ok else ["format guard G0"]) + [k for k, v in G_.items() if not v]
    d = decide(st, inv)
    Rr = {"what": __doc__, "score19_sha256": SELF_SHA256, "mode": "decision", **d, "guard_G_pixels": G_, "guard_G0_format": G0,
          "headline_stations": headline(st), "headline_pixels": headline(px["pixels"]["all"]), "station_sample": E["nst"],
          "identity": E["identity"], "identity_passed": E["id_ok"], "stations": st, "pixels": px, "warnings": E["warn"],
          "spread_note": SPREAD_NOTE, "seconds": E["seconds"]}
    save_json(OUT / "score19.json", Rr)
    log(f"OUTCOME {d['outcome']}: {d['reasons']} | score19 sha256 {SELF_SHA256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
