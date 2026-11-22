#!/usr/bin/env python3
"""A14 class E, count rules (protocol_amendments/A14_infinite_ensemble.md: section 4b 'Count rules', 4c for class E,
section 6 K1, decision D2). Warnings of the frozen final 1 km model at the 8.1 M populated pixels, leads 1 and 3, 2022, from
the 16 members of a13/warn16/data/run16 (the 30 seen start days), a14/data/run16_13 (run R1) and the rows 2:1 2:3 4:1 4:3 of
a13/warn16/data/run16all (task 0 of job 23898998; its rows at leads 5 and 7 are never read). Version 2 (2026-11-21, after
an internal check of version 1; changelog in README.md).

Definitions, readers and identity checks are those of a13/warn16/score16.py, imported UNCHANGED (it brings analyses/hidden/v3
= figdata_hidden_v3, bc = figdata_hidden_bc and tables.py, all unchanged): valid residents, truth codes (exposed, 40 km mean
above, hidden with the 10 % margin), the start-day 1 km estimate kept unchanged ('persist'), score16.persist_at (a reference
moved to the number of people a system warns, factor found again in every bootstrap resample inside +-0.25 of the point
factor, as tables_v3.at_factor) and tables.Tab / mbb / metrics / parts (pooled sums, paired bootstrap). The references of
the paper (analyses/hidden/v3: CAMS, Aurora, CAMS bias-corrected with and without the 1 km pattern, the start-day estimate,
Foehn's mean on 40 km cells and the 1 km climatology) are read from v3's own tables and factor curves (hidden_v3_pop.npz,
hidden_bc_pop.npz for clim_1km). persist_at_correct below is a transcription of the equal_correct_share branch of
tables_v3.at_factor (a closure inside tables_v3.main that cannot be imported).

Count rules (A14 4b). At a pixel where k of the N = 16 members exceed the threshold, the expected warning weight of an
M-member system drawn without replacement from the N members is pi_M(k) = sum_j g_M(j) C(k, j) C(N - k, M - j) / C(N, M):
  liberal    (even M) g = 1 for j >= M/2            the issued rule ('at least half'; M = 8: at least 4 of 8)
  symmetric  (even M) g = 1 for j > M/2, 1/2 at j = M/2
  strict     (odd M)  g = 1 for j >= (M + 1)/2
Counts are population-weighted sums of these weights (exact expected counts). Every system is a code table (16 entries:
truth code x warned); ratios (reach, share of correct warnings, CSI, ...), the comparisons with the references at an equal
number warned and at an equal share of correct warnings, and the paired differences and ratios with the references as issued
are plug-ins formed from the sums of that table (A14 4b 'Ratios and equal-number comparisons are formed from the sums').
The INFINITE system of each fit is the table with the signed weights h(k) = sum_M w0_M pi_M(k) (w0: the least-squares
intercept row of Q(M) = Q_inf + b x(M) on the fit grid; A14 4b 'the infinite count is a sum with signed weights'): its counts
are the fitted intercepts of the counts (linearity), and every ratio and comparison of the infinite value is formed from its
sums, the reference moved to the infinite system's own number warned / share of correct warnings. Because the kernel is
linear in the daily tables, the fit is inside every bootstrap resample. The fitted curves of the ratios and comparisons
(version 1) are kept as a labelled sensitivity ('sensitivity_fitted_curve').
  reported    M = 8 liberal (headline, D2); M = 16 liberal (measured); M = 4 = the stored paper members (eval_pop_crps_e01,
              'at least 2 of 4'); the infinite value: six fits (liberal, symmetric on even M = 4..16; strict on odd M = 5..15;
              each in 1/M and 1/(M + 1)); point = symmetric 1/M; range = the six; interval = union of their intervals; K1 = the
              three 1/M intercepts and their spread. Regions and country groups at 4, 8 and 16 members only (never infinite).
  quantities  issued:<metric>, diff:<ref>:<metric> (system minus reference, as issued), ratio:<ref>:<metric> (system over
              reference), eqn:<ref>:<metric> (system minus the reference moved to the system's number warned), eqc:<ref>:<metric>
              (system minus the reference moved to the system's share of correct warnings); lead3v1: Foehn 3 days ahead minus
              the references 1 day ahead on v3's common target days.
  intervals   the paired moving-block bootstrap of the paper (tables.mbb: 10,000 resamples, seed 0): 179 start days in
              blocks of 4 forecasts (analyses/hidden/v3); 149 unseen start days: the same resamples with the 30 seen days
              removed, each resample rescaled to 149 days; 30 seen days: blocks of one start day (score16), and beside them the
              paper's resamples restricted to the 30 seen columns ('seen30_paperblocks'). Every half-width is multiplied by
              sqrt(1 + V_draw / V_boot), V_draw the draw variance of the fixed noise fields:
                M = 8          delete-one-field jackknife over the 16 fields (pi_8 of the 15 others)
                M = 16, inf    the same jackknife with one grid for the full estimate and every replicate (even M = 4..14,
                               odd M = 5..13; M = 16: the liberal 1/M fit of that grid evaluated at M = 16), approximate
                M = 4 stored   the larger of (i) the jackknife over the four stored fields (the issued rule on the three
                               others: at least 2 of 3) and (ii) the variance over the 1,820 quartets of the 16 fields plus
                               the jackknife variance of their mean (drawing 4 of 16 without replacement)
              The reference moved in a resample is searched inside +-0.25 of the point factor (v3, score16): resamples without
              a crossing there are dropped and counted ('resamples_nan', interval 'truncated'); the whole factor grid
              0.4-2.0 (first crossing) is given beside as a sensitivity ('full_grid').
  samples     'conf149' the 149 unseen start days (confirmatory), 'seen30', 'seen30_paperblocks', 'all179'.
No station or monitor file is read. The infinite counts are sums with signed weights: no map, region or income group of them.

    python classE.py pass seen30|all2022 [workers]       -> data/hist_<set>_v2.npz
    python classE.py stats all2022 [workers]             -> results/classE_all2022.json/.md, classE_regions_all2022.json/.md
    python classE.py run2022 [workers] [gate]            gate (exit 2 if R1 is not usable), then pass + stats; gate =
                                                         'check13' (default: check13.json "passed" and the A13 rounding
                                                         level) or 'check13b' (amendment A14a: check13b "passed" and the
                                                         A13 rounding level of check13.json); see gate_check13"""
import json
import os
import sys
import time
from fractions import Fraction
from itertools import combinations
from math import comb
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
A14 = ROOT / "a14"
W16 = ROOT / "a13" / "warn16"
RUN16, RUN16ALL, RUN16_13 = W16 / "data" / "run16", W16 / "data" / "run16all", A14 / "data" / "run16_13"
# noise design (a14/DESIGN_SWITCH.md; A16 section 4 point 3). Unset: as stored. 'ar': the 358 sixteen-member forecasts of A16
# run G11 in one folder (the 30 seen start days among them, still seen), the 4 members of eval_pop_crps_e01_ar and the v3 tables
# of the design (S1 step 6); V_draw = 0 and factor 1. 'check': the stored runs, labelled outputs. 'check0': 'check' with V_draw = 0.
DESIGN = os.environ.get("FOEHN_DESIGN", "")
assert DESIGN in ("", "ar", "check", "check0"), f"FOEHN_DESIGN={DESIGN!r}: A14 knows only '', 'ar', 'check', 'check0'"
SFX = f"_{DESIGN}" if DESIGN else ""
AR, VDRAW0 = DESIGN == "ar", DESIGN in ("ar", "check0")
RUN16_AR = ROOT / "a16" / "data" / "run16"
NPLAN = 358 if AR else 298                                                          # forecasts checked by check13 / check13b
CHECK13 = A14 / f"results{SFX}" / "check13.json"
CHECK13B = (A14 / f"results{SFX}" / "check13b.json", A14 / f"results{SFX}" / "check13b.txt")    # A14a (protocol_amendments/A14a_check13_rounding.md)
GATES = ("check13", "check13b")
DATA, RES = HERE / f"data{SFX}", HERE / f"results{SFX}"
TASK0 = ((2, 1), (2, 3), (4, 1), (4, 3))
sys.dont_write_bytecode = True
sys.path.insert(0, str(W16))
_held = os.environ.pop("FOEHN_DESIGN", None) if DESIGN in ("check", "check0") else None   # 'check': the paper's modules read the stored files
import score16 as s16  # noqa: E402  (unchanged: brings v3, bc and the helpers of tables.py; their design switch is read at import)
if _held is not None:
    os.environ["FOEHN_DESIGN"] = _held

bc, v3 = s16.bc, s16.v3
Tab, mbb, metrics, parts = s16.Tab, s16.mbb, s16.metrics, s16.parts
THR_ALL, THR_REP, DOYS = s16.THR_ALL, s16.THR_REP, s16.DOYS
MUS, MU0, DMU, NB = s16.MUS, s16.MU0, s16.DMU, s16.NB
NF = 16                                                               # fields of the 16-member runs
RULES = ("liberal", "symmetric", "strict")
FORMS = ("1/M", "1/(M+1)")
GRID = {"liberal": (4, 6, 8, 10, 12, 14, 16), "symmetric": (4, 6, 8, 10, 12, 14, 16), "strict": (5, 7, 9, 11, 13, 15)}
GRID_JK = {"liberal": (4, 6, 8, 10, 12, 14), "symmetric": (4, 6, 8, 10, 12, 14), "strict": (5, 7, 9, 11, 13)}
FITS = tuple(f"{r} {fm}" for fm in FORMS for r in RULES)
POINT = "symmetric 1/M"
MQ = ("warned_M", "right_M", "false_M", "reach_pct", "correct_pct", "false_pct", "csi", "hidden_warned_M", "hidden_warned_pct",
      "hidden_margin_warned_M", "hidden_margin_warned_pct", "seen_warned_pct", "warned_in_silent40_M", "false_in_silent40_M",
      "false_pct_in_silent40", "false_per_hidden_warned")
RATIO = ("right_M", "false_M", "hidden_warned_M", "warned_M", "false_in_silent40_M", "hidden_margin_warned_M")   # = tables_v3.RATIO
CROSS = ("reach_pct", "correct_pct", "right_M", "false_M", "warned_M", "hidden_warned_M", "hidden_warned_pct",
         "hidden_margin_warned_pct", "csi")                                                                      # = tables_v3.CROSS
REFS_MOVED = ("persist", "cams", "aurora", "cams_qm", "cams_bcsd")            # v3: equal_number_warned / equal_correct_share
REFS_ISSUED = ("persist", "cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km", "clim_1km")   # v3 REFS (differences, ratios)
REFS_LEAD1 = ("cams", "aurora", "cams_qm", "cams_bcsd", "foehn_40km")       # v3 foehn_3_days_against_references_1_day
REF_NOTE = {"persist": "start-day 1 km estimate kept unchanged", "cams": "CAMS forecast", "aurora": "Aurora",
            "cams_qm": "CAMS forecast, bias-corrected", "cams_bcsd": "CAMS forecast, bias-corrected + 1 km pattern",
            "foehn_40km": "Foehn's mean of the 4 stored members on 40 km cells (a four-member device, A14 4b)",
            "clim_1km": "1 km climatology alone"}
SAMPLES = ("conf149", "seen30", "seen30_paperblocks", "all179")
REGION_SAMPLES = ("conf149", "seen30", "all179")
SAMPLE_NAMES = {"conf149": "confirmatory: the 149 start days of 2022 not in a13/warn16/data/run16",
                "seen30": "the 30 seen start days of a13/warn16/data/run16 (not confirmatory; score16 blocks of one start day)",
                "seen30_paperblocks": "the 30 seen start days with the paper's resamples restricted to them (not confirmatory)",
                "all179": "all 179 start days"}
TOL_MEMBERS = dict(min_bit_identical_share=0.25, max_abs=2.5, mean_abs=0.05)   # a14/r2/smoke_check.py ok_members (fixed before any score)
S = {}


# ------------------------------------------------------------------ expected warning weights and fits
def rule_g2(j, M, rule):
    """twice the warning weight of j exceeding members out of M (integer, exact)"""
    if rule == "liberal":
        assert M % 2 == 0, (rule, M)
        return 2 if 2 * j >= M else 0
    if rule == "symmetric":
        assert M % 2 == 0, (rule, M)
        return 2 if 2 * j > M else (1 if 2 * j == M else 0)
    if rule == "strict":
        assert M % 2 == 1, (rule, M)
        return 2 if 2 * j > M else 0
    raise ValueError(rule)


_PI = {}


def pi_table(N, M, rule):
    """(N + 1,) float64: pi_M(k), the expected warning weight of an M-member system drawn without replacement from N members
    of which k exceed (exact rational arithmetic, then float)"""
    key = (N, M, rule)
    if key not in _PI:
        assert 1 <= M <= N, key
        den = 2 * comb(N, M)
        _PI[key] = np.array([float(Fraction(sum(rule_g2(j, M, rule) * comb(k, j) * comb(N - k, M - j)
                                                 for j in range(max(0, M - (N - k)), min(k, M) + 1)), den)) for k in range(N + 1)])
    return _PI[key]


def xform(M, form):
    return 1.0 / M if form == "1/M" else 1.0 / (M + 1)


def ls_rows(Ms, form):
    """least squares of Q(M) = Q_inf + b x(M) on the grid Ms: rows (w0, w1) with Q_inf = w0 @ Q, b = w1 @ Q"""
    x = np.array([xform(M, form) for M in Ms])
    X = np.c_[np.ones_like(x), x]
    W = np.linalg.solve(X.T @ X, X.T)
    return W[0], W[1]


def kernel(N, rule, form, grid, at=None):
    """signed weights h(k) of the fitted system: at=None the intercept (infinite system), Q_inf = sum residents x h(k);
    at=M the fitted value at M"""
    w0, w1 = ls_rows(grid, form)
    c = w0 if at is None else w0 + w1 * xform(at, form)
    return sum(ci * pi_table(N, M, rule) for ci, M in zip(c, grid))


def tables_from_k(HK, pi):
    """HK (..., 8, K + 1) residents by truth code and k; pi (K + 1,) -> code tables (..., 16) (codes 0-7 not warned, 8-15 warned)"""
    pi = np.asarray(pi, np.float64)
    return np.concatenate([HK @ (1.0 - pi), HK @ pi], axis=-1)


def jk_var(th):
    """delete-one jackknife variance from the replicates th (n,); NaN if any replicate is undefined"""
    th = np.asarray(th, np.float64)
    if not np.isfinite(th).all():
        return float("nan")
    n = len(th)
    return float((n - 1) / n * ((th - th.mean()) ** 2).sum())


def jk_var_finite(th):
    """the jackknife formula over the finite replicates only (information when some are undefined): (value, n finite)"""
    th = np.asarray(th, np.float64); f = th[np.isfinite(th)]
    if len(f) < 2:
        return float("nan"), int(len(f))
    return float((len(f) - 1) / len(f) * ((f - f.mean()) ** 2).sum()), int(len(f))


# ------------------------------------------------------------------ a reference moved (score16 / tables_v3)
def pooled_curve(Hd):
    """(days, 8, NB + 2) curves -> (8, NB + 1) residents warned at each factor, pooled over days (as score16.persist_at)"""
    Cd = np.cumsum(Hd.astype(np.float64)[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
    return Cd.sum(0)


def persist_at_correct(fixed, Hd, cnt):
    """a reference (curves Hd) at the share of correct warnings of `fixed` (a Tab): the equal_correct_share branch of
    tables_v3.at_factor, with the root on the pooled curve as score16.persist_at; factor re-found in every resample inside
    +-0.25 of the point factor; None outside the factors 0.4-2"""
    n = len(Hd)
    Cd = np.cumsum(Hd.astype(np.float64)[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
    Pw = Cd.sum(0)
    p0 = parts(fixed.a.sum(0)); target = p0["tp"] / p0["warn"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = Pw[1::2].sum(0) / Pw.sum(0)
    mu, ncross = v3.root(np.where(np.isfinite(cs), cs, np.inf), target, increasing=True)
    if mu is None:
        return None
    jm = int(round((mu - MU0) / DMU)); lo, hi = max(0, jm - 250), min(Cd.shape[2], jm + 251)
    Cw = Cd[:, :, lo:hi]; nw = hi - lo
    bW = (cnt @ Cw.reshape(n, -1)).reshape(-1, 8, nw)
    a = fixed.a; bN = cnt @ (a[:, :8] + a[:, 8:])
    fb = fixed.bsum.astype(np.float64); tgt = parts(fb)["tp"] / parts(fb)["warn"]
    yc = bW[:, 1::2].sum(1) / np.maximum(bW.sum(1), 1e-9); s_ = yc - tgt[:, None]
    cross = (s_[:, :-1] < 0) & (s_[:, 1:] >= 0)
    okb = cross.any(1); j0 = np.argmax(cross, axis=1); rr = np.arange(len(j0))
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = -s_[rr, j0] / (s_[rr, j0 + 1] - s_[rr, j0])
        Wb = bW[rr, :, j0] + fr[:, None] * (bW[rr, :, j0 + 1] - bW[rr, :, j0])
    A16 = np.concatenate([bN - Wb, Wb], axis=1); A16[~okb] = np.nan
    bs = metrics(parts(A16), n)
    Pc = Cw.sum(0)
    sp = Pc[1::2].sum(0) / np.maximum(Pc.sum(0), 1e-9) - target
    jj = np.flatnonzero((sp[:-1] < 0) & (sp[1:] >= 0))
    if jj.size == 0:
        return None
    j = int(jj[0]); fr0 = -sp[j] / (sp[j + 1] - sp[j])
    P = Pc[:, j] + fr0 * (Pc[:, j + 1] - Pc[:, j])
    N0 = a.sum(0)[:8] + a.sum(0)[8:]
    pt = metrics(parts(np.concatenate([N0 - P, P])), n)
    return dict(mu=mu, crossings=ncross, pt=pt, bs=bs, resamples_without_a_crossing=int((~okb).sum()))


def full_curves(Hd, cnt):
    """the resampled curves of a reference on the WHOLE factor grid (sensitivity of the +-0.25 window): bW (B, 8, NB + 1),
    residents warned Ws (B, NB + 1) and share of correct warnings cs (B, NB + 1) at each factor"""
    n = len(Hd)
    Cd = np.cumsum(Hd.astype(np.float64)[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:]
    bW = (cnt @ Cd.reshape(n, -1)).reshape(-1, 8, NB + 1)
    Ws = bW.sum(1)
    return dict(bW=bW, Ws=Ws, cs=bW[:, 1::2].sum(1) / np.maximum(Ws, 1e-9))


def moved_full(kind, fixed, cnt, n, F):
    """the bootstrap of a moved reference with the root of every resample searched on the whole grid 0.4-2.0 (first
    crossing, as v3.root for the point): metrics (B,) and the number of resamples without a crossing"""
    a = fixed.a; bN = cnt @ (a[:, :8] + a[:, 8:])
    p = parts(fixed.bsum.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        s_ = (p["warn"][:, None] - F["Ws"]) if kind == "eqn" else (F["cs"] - (p["tp"] / p["warn"])[:, None])
    cross = (s_[:, :-1] < 0) & (s_[:, 1:] >= 0)
    okb = cross.any(1); j0 = np.argmax(cross, axis=1); rr = np.arange(len(j0))
    bW = F["bW"]
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = -s_[rr, j0] / (s_[rr, j0 + 1] - s_[rr, j0])
        Wb = bW[rr, :, j0] + fr[:, None] * (bW[rr, :, j0 + 1] - bW[rr, :, j0])
    A16 = np.concatenate([bN - Wb, Wb], axis=1); A16[~okb] = np.nan
    return metrics(parts(A16), n), int((~okb).sum())


def pt_eqn(a_sum, Pc, n):
    """point of score16.persist_at from a pooled code table (16,) and the pooled curve Pc (8, NB + 1)"""
    target = parts(a_sum)["warn"]
    mu, _ = v3.root(Pc.sum(0), target, increasing=False)
    if mu is None:
        return None, None
    jm = int(round((mu - MU0) / DMU)); lo, hi = max(0, jm - 250), min(Pc.shape[1], jm + 251)
    sp = target - Pc[:, lo:hi].sum(0)
    jj = np.flatnonzero((sp[:-1] < 0) & (sp[1:] >= 0))
    if jj.size == 0:
        return None, mu
    j = int(jj[0]); fr0 = -sp[j] / (sp[j + 1] - sp[j])
    P = Pc[:, lo + j] + fr0 * (Pc[:, lo + j + 1] - Pc[:, lo + j])
    N0 = a_sum[:8] + a_sum[8:]
    return metrics(parts(np.concatenate([N0 - P, P])), n), mu


def pt_eqc(a_sum, Pc, n):
    """point of persist_at_correct from a pooled code table and the pooled curve"""
    p = parts(a_sum)
    with np.errstate(invalid="ignore", divide="ignore"):
        target = p["tp"] / p["warn"]
        cs = Pc[1::2].sum(0) / Pc.sum(0)
    if not np.isfinite(target):
        return None, None
    mu, _ = v3.root(np.where(np.isfinite(cs), cs, np.inf), target, increasing=True)
    if mu is None:
        return None, None
    jm = int(round((mu - MU0) / DMU)); lo, hi = max(0, jm - 250), min(Pc.shape[1], jm + 251)
    Cw = Pc[:, lo:hi]
    sp = Cw[1::2].sum(0) / np.maximum(Cw.sum(0), 1e-9) - target
    jj = np.flatnonzero((sp[:-1] < 0) & (sp[1:] >= 0))
    if jj.size == 0:
        return None, mu
    j = int(jj[0]); fr0 = -sp[j] / (sp[j + 1] - sp[j])
    P = Cw[:, j] + fr0 * (Cw[:, j + 1] - Cw[:, j])
    N0 = a_sum[:8] + a_sum[8:]
    return metrics(parts(np.concatenate([N0 - P, P])), n), mu


# ------------------------------------------------------------------ the quantities of a system
QS_WORLD = dict(diff=MQ, ratio=RATIO)                                 # quantity sets: World; regions and lead3v1 below
QS_SHORT = dict(diff=CROSS, ratio=())                                 # lead3v1 (= tables_v3 CROSS)
QS_REGION = dict(diff=MQ, ratio=())                                   # regions (v3 regions: DIFF2, contained in MQ)


def qnames(RI, RM, qs=QS_WORLD):
    return ([f"issued:{m}" for m in MQ] + [f"diff:{r}:{m}" for r in RI for m in qs["diff"]] + [f"ratio:{r}:{m}" for r in RI for m in qs["ratio"]]
            + [f"{p}:{r}:{m}" for r in RM for p in ("eqn", "eqc") for m in MQ])


def pts_many(A, n, RIp, RMp, qs=QS_WORLD):
    """A (R, 16) pooled code tables of R systems or replicates -> {quantity: (R,) points}; RIp {ref: pooled metrics of the
    reference as issued}, RMp {ref: pooled curve of a moved reference}"""
    A = np.atleast_2d(np.asarray(A, np.float64)); R_ = len(A)
    pt = metrics(parts(A), n)
    out = {f"issued:{m}": np.asarray(pt[m], np.float64) for m in MQ}
    with np.errstate(all="ignore"):
        for r, rp in RIp.items():
            for m in qs["diff"]:
                out[f"diff:{r}:{m}"] = out[f"issued:{m}"] - rp[m]
            for m in qs["ratio"]:
                out[f"ratio:{r}:{m}"] = out[f"issued:{m}"] / rp[m]
    for r, Pc in RMp.items():
        for pre, fn in (("eqn", pt_eqn), ("eqc", pt_eqc)):
            vals = {m: np.full(R_, np.nan) for m in MQ}
            for i in range(R_):
                with np.errstate(all="ignore"):
                    res, _ = fn(A[i], Pc, n)
                if res is not None:
                    for m in MQ:
                        vals[m][i] = out[f"issued:{m}"][i] - res[m]
            for m in MQ:
                out[f"{pre}:{r}:{m}"] = vals[m]
    return out


def boot_quantities(tab, RIt, RM, cnt, n, full=None, qs=QS_WORLD):
    """the bootstrap of every quantity of one system (Tab of its daily tables): {q: (point, replicates (B,))}, info of the moved
    references and, for each moved reference, the whole-grid sensitivity {q: replicates} (full: {ref: full_curves})"""
    bs = tab.bs
    out = {f"issued:{m}": (float(tab.pt[m]), np.asarray(bs[m], np.float64)) for m in MQ}
    with np.errstate(all="ignore"):
        for r, rt in RIt.items():
            rb = rt.bs
            for m in qs["diff"]:
                out[f"diff:{r}:{m}"] = (float(tab.pt[m] - rt.pt[m]), np.asarray(bs[m] - rb[m], np.float64))
            for m in qs["ratio"]:
                out[f"ratio:{r}:{m}"] = (float(tab.pt[m] / rt.pt[m]), np.asarray(bs[m] / rb[m], np.float64))
    info, fullq = {}, {}
    nanb = np.full(cnt.shape[0], np.nan)
    for r, Hd in RM.items():
        for pre, fn in (("eqn", s16.persist_at), ("eqc", persist_at_correct)):
            with np.errstate(all="ignore"):
                res = fn(tab, Hd, cnt)
            if res is None:
                info[f"{pre}:{r}"] = "outside the factors 0.4-2"
                for m in MQ:
                    out[f"{pre}:{r}:{m}"] = (float("nan"), nanb)
                continue
            info[f"{pre}:{r}"] = dict(factor=float(res["mu"]), crossings=int(res["crossings"]),
                                      resamples_without_a_crossing=int(res["resamples_without_a_crossing"]))
            with np.errstate(all="ignore"):
                for m in MQ:
                    out[f"{pre}:{r}:{m}"] = (float(tab.pt[m] - res["pt"][m]), np.asarray(bs[m] - res["bs"][m], np.float64))
            if full is not None:
                with np.errstate(all="ignore"):
                    fb, nf = moved_full(pre, tab, cnt, n, full[r])
                    for m in MQ:
                        fullq[f"{pre}:{r}:{m}"] = np.asarray(bs[m] - fb[m], np.float64)
                info[f"{pre}:{r}"]["full_grid_resamples_without_a_crossing"] = nf
    return out, info, fullq


def summarize(pt, bs, vdraw, vnote=None, where="within +-0.25 of the point factor"):
    """value, bootstrap percentile interval, V_boot, V_draw, the inflated 95 % interval and its status"""
    pt = float(pt) if pt is not None else float("nan")
    nnan = None if bs is None else int((~np.isfinite(bs)).sum())
    e = dict(value=pt, boot95=[None, None], V_boot=None, V_draw=0.0 if VDRAW0 else vdraw, inflation=None, interval95=[None, None],
             resamples_nan=nnan)
    if VDRAW0:                                                             # A16 4.3: the fixed-design V_draw, information only
        e["V_draw_fixed_design"] = vdraw
    if vnote:
        e["V_draw_note"] = vnote
    if not np.isfinite(pt):
        e["interval_status"] = "no interval: value undefined"
        return e
    if bs is None or not np.isfinite(bs).any():
        e["interval_status"] = "no interval: no finite bootstrap replicate"
        return e
    lo, hi = (float(v) for v in np.nanpercentile(bs, [2.5, 97.5]))
    vb = float(np.nanvar(bs, ddof=1)) if np.isfinite(bs).sum() > 1 else float("nan")
    e.update(boot95=[lo, hi], V_boot=vb)
    if VDRAW0:                                                             # A16 4.3: V_draw = 0, factor 1
        e.update(inflation=1.0, interval95=[lo, hi], interval_status="factor 1 (A16 4.3: V_draw = 0): bootstrap percentile interval")
    elif vdraw is not None and np.isfinite(vdraw) and np.isfinite(vb) and vb > 0:
        f = float(np.sqrt(1.0 + vdraw / vb))
        e.update(inflation=f, interval95=[pt - (pt - lo) * f, pt + (hi - pt) * f], interval_status="inflated")
    else:
        why = "V_draw undefined" if vdraw is None or not np.isfinite(vdraw) else "V_boot = 0"
        e.update(interval95=[lo, hi], interval_status=f"NOT inflated ({why}): bootstrap percentile interval only")
    if nnan:
        e["interval_status"] += f"; truncated: {nnan} of {len(bs)} resamples without a crossing {where} dropped"
    return e


def summarize_full(pt, bsf, vdraw, nf):
    """the whole-grid sensitivity of a moved comparison"""
    e = summarize(pt, bsf, vdraw, where="on the whole grid 0.4-2.0")
    e["resamples_without_a_crossing_full_grid"] = nf
    return e


# ------------------------------------------------------------------ the Foehn systems of one sample / lead / threshold / region
def foehn_systems(HKw, HJw, HQw, H3w, T4w, with_inf=True):
    """daily code tables and pooled draw replicates of the Foehn systems:
      M8_liberal            pi_8 of 16; replicates pi_8 of the 15 others (16)
      M16_liberal_measured  pi_16; replicates: the liberal 1/M fit on even M = 4..14 of the 15 others evaluated at M = 16 (16)
      M4_stored             the stored 'at least 2 of 4'; replicates (i) 'at least 2 of 3' of the 3 other stored fields (4),
                            (ii) the 1,820 quartets of the 16 fields ('at least 2 of 4')
      inf <rule> <form>     the signed kernel of the fit on GRID; replicates: the same fit on GRID_JK of the 15 others (16)"""
    HJs = HJw.sum(0)                                                     # (8, 16, 16)
    tot = HKw.sum(0).sum(-1)                                             # (8,) residents by truth code
    out = {}

    def reps(h15):
        return np.stack([tables_from_k(HJs[:, i, :], h15) for i in range(NF)])
    out["M8_liberal"] = dict(a=tables_from_k(HKw, pi_table(NF, 8, "liberal")), reps=reps(pi_table(NF - 1, 8, "liberal")))
    out["M16_liberal_measured"] = dict(a=tables_from_k(HKw, pi_table(NF, 16, "liberal")),
                                       reps=reps(kernel(NF - 1, "liberal", "1/M", GRID_JK["liberal"], at=16)))
    HQs, H3s = HQw.sum(0), H3w.sum(0)                                    # (1820, 8), (4, 8)
    out["M4_stored"] = dict(a=T4w, reps=np.concatenate([tot[None] - H3s, H3s], -1),
                            quartets=np.concatenate([tot[None] - HQs, HQs], -1))
    if with_inf:
        for r in RULES:
            for fm in FORMS:
                out[f"inf {r} {fm}"] = dict(a=tables_from_k(HKw, kernel(NF, r, fm, GRID[r])), reps=reps(kernel(NF - 1, r, fm, GRID_JK[r])))
    return out


CONTAINS = np.array([[i in qq for i in range(NF)] for qq in combinations(range(NF), 4)])   # (1820, 16)


def vdraw_of(name, sysd, q, RP):
    """V_draw of quantity q of system `name` from its replicate points RP[name][q] (and the quartets); note"""
    if name == "M4_stored":
        th4 = RP[name]["reps"][q]; tq = RP[name]["quartets"][q]
        v1 = jk_var(th4)
        if np.isfinite(tq).all():
            vq = float(tq.var()); vu = jk_var([tq[~CONTAINS[:, i]].mean() for i in range(NF)]); v2 = vq + vu
        else:
            vq = vu = v2 = float("nan")
        parts_ = dict(i_jackknife_4_stored_fields_2_of_3=v1, ii_quartets=vq, ii_mean_jackknife=vu, ii_total=v2,
                      quartets_undefined=int((~np.isfinite(tq)).sum()), stored_replicates_undefined=int((~np.isfinite(th4)).sum()))
        ok1, ok2 = np.isfinite(v1), np.isfinite(v2)
        if ok1 and ok2:
            return max(v1, v2), dict(parts_, used="max((i), (ii))", which="(i)" if v1 > v2 else "(ii)")
        if ok1 or ok2:
            return (v1 if ok1 else v2), dict(parts_, used=f"{'(i)' if ok1 else '(ii)'} only: the other is undefined")
        return float("nan"), dict(parts_, used="undefined")
    th = RP[name]["reps"][q]
    v = jk_var(th)
    if np.isfinite(v):
        return v, None
    vf, nfin = jk_var_finite(th)
    return float("nan"), dict(undefined_replicates=int(len(th) - nfin), jackknife_over_the_finite_replicates=vf, finite=nfin)


def foehn_block(HKw, HJw, HQw, H3w, T4w, cnt, n, RIt, RM, with_inf=True, with_full=True, with_curves=True, qs=QS_WORLD):
    """every system and quantity of one block: RIt {ref: daily tables} as issued, RM {ref: daily curves} moved"""
    t0 = time.time()
    SY = foehn_systems(HKw, HJw, HQw, H3w, T4w, with_inf=with_inf)
    RIT = {r: Tab(a, cnt) for r, a in RIt.items()}
    RIp = {r: t.pt for r, t in RIT.items()}
    RMp = {r: pooled_curve(Hd) for r, Hd in RM.items()}
    TB = {s: Tab(d["a"], cnt) for s, d in SY.items()}
    BQ, INFO, FULL = {s: {} for s in SY}, {s: {} for s in SY}, {s: {} for s in SY}
    for s, tab in TB.items():                                              # issued, differences, ratios
        q_, _, _ = boot_quantities(tab, RIT, {}, cnt, n, qs=qs)
        BQ[s].update(q_)
    for r, Hd in RM.items():                                               # moved references, one at a time (memory)
        F = full_curves(Hd, cnt) if with_full else None
        for s, tab in TB.items():
            q_, i_, f_ = boot_quantities(tab, {}, {r: Hd}, cnt, n, full=None if F is None else {r: F})
            BQ[s].update(q_); INFO[s].update(i_); FULL[s].update(f_)
        del F
    QN = qnames(RIt, RM, qs)
    RP = {}
    for s, d in SY.items():                                                # points of the draw replicates
        RP[s] = {"reps": pts_many(d["reps"], n, RIp, RMp, qs)}
        if "quartets" in d:
            RP[s]["quartets"] = pts_many(d["quartets"], n, RIp, RMp, qs)
    PT = {s: {q: float(v[0]) for q, v in pts_many(TB[s].a.sum(0), n, RIp, RMp, qs).items()} for s in SY}
    dev = 0.0                                                              # point-only = bootstrap point (check)
    for s in SY:
        for q in QN:
            a_, b_ = BQ[s][q][0], PT[s][q]
            if np.isfinite(a_) or np.isfinite(b_):
                dev = max(dev, abs(a_ - b_) / max(1.0, abs(a_)) if (np.isfinite(a_) and np.isfinite(b_)) else np.inf)
    CV = {}
    if with_curves:                                                        # Q(M) of the three rules (points)
        HKs = HKw.sum(0)
        keys = [(r, M) for r in RULES for M in GRID[r]]
        cp = pts_many(np.stack([tables_from_k(HKs, pi_table(NF, M, r)) for r, M in keys]), n, RIp, RMp, qs)
        CV = {q: {r: {str(M): float(cp[q][i]) for i, (r_, M) in enumerate(keys) if r_ == r} for r in RULES} for q in QN}
    out = {}
    for q in QN:
        e = {}
        for s in SY:
            vd, note = vdraw_of(s, SY[s], q, RP)
            ent = summarize(*BQ[s][q], vd, note)
            if q in FULL[s]:
                inf_ = INFO[s].get(":".join(q.split(":")[:2]))
                ent["full_grid"] = summarize_full(BQ[s][q][0], FULL[s][q], vd,
                                                  inf_.get("full_grid_resamples_without_a_crossing") if isinstance(inf_, dict) else None)
            if s.startswith("inf "):
                e.setdefault("fits", {})[s[4:]] = ent
            else:
                e[s] = ent
        tq = RP["M4_stored"]["quartets"][q]
        e["M4_quartets_of_the_16_fields"] = dict(mean=float(np.nanmean(tq)) if np.isfinite(tq).any() else None,
                                                 sd=float(np.nanstd(tq)) if np.isfinite(tq).any() else None,
                                                 min=float(np.nanmin(tq)) if np.isfinite(tq).any() else None,
                                                 max=float(np.nanmax(tq)) if np.isfinite(tq).any() else None,
                                                 fields_0_3_of_the_16=float(tq[0]), undefined=int((~np.isfinite(tq)).sum()))
        if with_curves:
            e["curve"] = CV[q]
            e["M4_expected_from_16"] = CV[q]["liberal"]["4"]
            sens = {}
            for r in RULES:
                Qv = np.array([CV[q][r][str(M)] for M in GRID[r]])
                for fm in FORMS:
                    sens[f"{r} {fm}"] = float(ls_rows(GRID[r], fm)[0] @ Qv) if np.isfinite(Qv).all() else None
            e["sensitivity_fitted_curve"] = sens
        if with_inf:
            e["infinite"] = infinite_summary(e["fits"])
        out[q] = e
    info = {s: INFO[s] for s in SY}
    if with_inf:
        tabs_inf = {s: SY[s]["a"].sum(0) for s in SY if s.startswith("inf ")}
        info["infinite_tables_min_entry"] = {s[4:]: float(v.min()) for s, v in tabs_inf.items()}
    info.update(check_point_only_max_rel_dev=float(dev), seconds=round(time.time() - t0, 1))
    return out, info


def infinite_summary(fits):
    """point (symmetric 1/M), range of the defined fits among the six, union of their inflated intervals, K1"""
    vals = {k: fits[k]["value"] for k in FITS}
    ok = [k for k in FITS if vals[k] is not None and np.isfinite(vals[k])]
    und = [k for k in FITS if k not in ok]
    oki = [k for k in ok if all(v is not None and np.isfinite(v) for v in fits[k]["interval95"])]
    k1 = {r: vals[f"{r} 1/M"] for r in RULES}; k1p = {r: vals[f"{r} 1/(M+1)"] for r in RULES}
    k1v = [v for v in k1.values() if v is not None and np.isfinite(v)]; k1pv = [v for v in k1p.values() if v is not None and np.isfinite(v)]
    pv = vals[POINT]
    st = "the six fits" if not und else f"{len(ok)} of the 6 fits ({', '.join(und)}: outside the factors 0.4-2 / undefined)"
    notinf = [k for k in oki if fits[k]["interval_status"] != "inflated"]
    trunc = [k for k in oki if "truncated" in fits[k]["interval_status"]]
    return dict(point=pv if pv is not None and np.isfinite(pv) else None, point_label="symmetric-rule intercept, fit in 1/M",
                range=[min(vals[k] for k in ok), max(vals[k] for k in ok)] if ok else [None, None], range_over=st,
                undefined_fits=und,
                interval95=[min(fits[k]["interval95"][0] for k in oki), max(fits[k]["interval95"][1] for k in oki)] if oki else [None, None],
                interval_over=[k for k in oki], interval_parts_not_inflated=notinf, interval_parts_truncated=trunc,
                K1={"intercepts_1/M": k1, "spread_1/M": (max(k1v) - min(k1v)) if len(k1v) == 3 else None,
                    "intercepts_1/(M+1)": k1p, "spread_1/(M+1)": (max(k1pv) - min(k1pv)) if len(k1pv) == 3 else None,
                    "spread_of_the_six": (max(vals[k] for k in ok) - min(vals[k] for k in ok)) if len(ok) == 6 else None})


# ------------------------------------------------------------------ the pass over the forecasts
ALLP = np.arange(1 << NF, dtype=np.int64)
POPC = np.zeros(1 << NF, np.uint8)
for _i in range(NF):
    POPC += ((ALLP >> _i) & 1).astype(np.uint8)
QUARTETS = list(combinations(range(NF), 4))
QMASK = np.array([sum(1 << i for i in qq) for qq in QUARTETS], np.int64)
SUBS = [(M, j, tuple(range(j * M, (j + 1) * M))) for M in (1, 2, 4, 8, 16) for j in range(16 // M)]   # = extrap16.SUBS
SUBNAMES = [f"m{M}_{j}" for M, j, _ in SUBS]


def patterns(m, thr):
    """16-bit exceedance pattern of every pixel (int64) from the members (pixels, 16)"""
    pk = np.packbits(m > thr, axis=1, bitorder="little")
    return pk[:, 0].astype(np.int64) | (pk[:, 1].astype(np.int64) << 8)


KMI = [POPC[ALLP & ~(1 << _i)] for _i in range(NF)]                    # k of the 15 other fields, per deleted field


def ph_jack(ph):
    """residents by 16-bit pattern ph (65536,) -> (16, 16): by deleted field and k of the 15 other fields"""
    return np.stack([np.bincount(KMI[i], weights=ph, minlength=NF) for i in range(NF)])


def ph_derived(PH):
    """from the residents by truth code and 16-bit pattern PH (8, 65536): HJ (8, 16, 16) by deleted field and k of the 15
    others, HQ (1820, 8) warned by 'at least 2 of 4' of each quartet, HS (31, 8) warned by the direct rule of each disjoint
    sub-ensemble of extrap16 (at least ceil(M/2) of M)"""
    HJ = np.stack([ph_jack(PH[c]) for c in range(8)])
    return HJ, S["G4"] @ PH.T, S["SUBI"] @ PH.T


def quartet_indicator():
    G4 = np.zeros((len(QMASK), 1 << NF))
    for a in range(0, len(QMASK), 140):
        G4[a:a + 140] = POPC[ALLP[None, :] & QMASK[a:a + 140, None]] >= 2
    return G4


def subset_indicator():
    return np.stack([(POPC[ALLP & sum(1 << i for i in idx)] >= (M + 1) // 2).astype(np.float64) for M, j, idx in SUBS])


def index16(D, keep=None):
    """{(start, lead): (pop16 file, row, target)} of the finished rows of a 16-member folder (keys16_*/done16_*)"""
    D = Path(D); idx = {}
    for fk in sorted(D.glob("keys16_*.npy")):
        tag = fk.stem.split("_", 1)[1]
        done = set(json.loads((D / f"done16_{tag}.json").read_text()))
        for n, (d, h, t) in enumerate(np.load(fk)):
            key = (int(d), int(h))
            if n in done and (keep is None or key in keep):
                assert key not in idx, f"{key} twice in {D}"
                idx[key] = (str(D / f"pop16_{tag}.npy"), n, int(t))
    return idx


def check13_verdict(R, ignore_passed=False):
    """(usable, details) from the content of check13.json: "passed": true AND members 0-3 against the stored members at the
    rounding level that A13 accepted (a14/r2/smoke_check.py tolerance, fixed before any score: bit-identical share >= 0.25,
    max |difference| <= 2.5 ug m-3, mean |difference| <= 0.05 ug m-3 for every member of every forecast, finite patterns
    equal) for all 298 forecasts"""
    try:
        vs = R["vs_stored"]
        rows = [r for r in R["rows"] if "vs_stored" in r]
        mean_abs = max(r["vs_stored"][f"m{m}"]["mean_abs"] for r in rows for m in range(4))
        ok = dict(complete=not R.get("missing") and not R.get("extra") and not R.get("wrong_target") and R.get("done") == NPLAN,
                  all_forecasts=vs["forecasts"] == NPLAN and len(rows) == NPLAN,
                  finite_pattern_equal_everywhere=vs["finite_pattern_equal_everywhere"] is True,
                  min_bit_identical_share=vs["min_bit_identical_share"] >= TOL_MEMBERS["min_bit_identical_share"],
                  worst_max_abs=vs["worst_max_abs"] <= TOL_MEMBERS["max_abs"], worst_mean_abs=mean_abs <= TOL_MEMBERS["mean_abs"])
        if not ignore_passed:
            ok["check13_passed"] = R.get("passed") is True
        det = dict(ok, check13_passed_as_coded=R.get("passed"), min_share=vs["min_bit_identical_share"], max_abs=vs["worst_max_abs"],
                   mean_abs=mean_abs)
    except Exception as e:                                                  # noqa: BLE001
        return False, dict(error=f"unreadable or incomplete ({e!r})")
    return all(ok.values()), det


def gate_check13(mode="check13"):
    """exit 2 unless R1 is usable (A14 section 3 order of work, K4). Two gates, chosen by the caller, never by this code:
      'check13'   (the original gate, default): check13.json "passed": true AND check13_verdict (A13 rounding level, R-10);
      'check13b'  (amendment A14a of 01:45, written before check13 ran: 'the 179-day class-U and class-E jobs are gated on
                  results/check13b.txt'): check13b.txt and check13b.json "passed": true AND check13_verdict without its
                  "passed" (completeness, finite patterns, A13 rounding level of members 0-3 in check13.json)."""
    assert mode in GATES, mode
    if not CHECK13.exists():
        print(f"EXIT 2: {CHECK13} is missing: run R1 is not checked, the class-E values of 2022 are not computed", flush=True)
        sys.exit(2)
    try:
        R = json.loads(CHECK13.read_text())
    except Exception as e:                                                  # noqa: BLE001
        print(f"EXIT 2: {CHECK13} unreadable ({e!r})", flush=True)
        sys.exit(2)
    usable, det = check13_verdict(R, ignore_passed=(mode == "check13b"))
    det["gate"] = mode
    if mode == "check13b":
        try:
            b = [json.loads(f.read_text()).get("passed") is True for f in CHECK13B]
        except Exception as e:                                              # noqa: BLE001
            b = [False]; det["check13b_error"] = repr(e)
        det["check13b_passed (json, txt)"] = b
        usable = usable and all(b) and len(b) == 2
    if not usable:
        print(f"EXIT 2: run R1 is not usable under the gate '{mode}': {json.dumps(det)}", flush=True)
        sys.exit(2)
    print(f"R1 accepted under the gate '{mode}': {json.dumps(det)}", flush=True)


def triples_table(B4, idx, w, nbins):
    """(4, nbins): residents (weights w, bins idx) warned by 'at least 2 of 3' of the 3 OTHER stored members, for each deleted
    stored member (the replicates of the jackknife over the four stored fields with the issued rule 'at least half')"""
    n4t = B4.sum(1)
    return np.stack([np.bincount(idx, weights=np.where(n4t - B4[:, i] >= 2, w, 0.0), minlength=nbins) for i in range(4)])


def members_vs_stored(m16, m4):
    """members 0-3 of the 16-member forecast against the 4 stored members, member by member (as check16all.cmp)"""
    out = []
    for m in range(4):
        a = m16[:, m].astype(np.float16); b = m4[:, m].astype(np.float16)
        fa, fb = np.isfinite(a), np.isfinite(b); both = fa & fb
        d = np.abs(a[both].astype(np.float32) - b[both].astype(np.float32))
        out.append(dict(finite_pattern_equal=bool(np.array_equal(fa, fb)), bit_identical_share=float((a.view(np.uint16) == b.view(np.uint16)).mean()),
                        max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0))
    return out


def one(key):
    """one forecast: the stored-4 and start-day tables of score16 (bc.tables), the start-day curves, and the residents by
    truth code and k of the 16 members (all regions, five thresholds), with the jackknife, quartet, stored-triple tables (all
    regions, the three reported thresholds) and the sub-ensemble tables (World)"""
    d, h = key
    t0 = time.time()
    k, n, t = bc.S["pop_index"][key]
    p16, n16, t16 = S["R16"][key]; assert t16 == t, (key, t16, t)
    m4 = np.asarray(np.load(bc.EVP / f"pop_{k}.npy", mmap_mode="r")[n], np.float32)
    m16 = np.asarray(np.load(p16, mmap_mode="r")[n16], np.float32)
    assert m16.shape[1] == NF
    g1 = np.asarray(np.load(bc.EVP / f"pop_ghap_{k}.npy", mmap_mode="r")[n], np.float32)
    c40 = np.asarray(np.load(bc.EVP / f"cell40_{k}.npy", mmap_mode="r")[n], np.float32).reshape(2, -1)
    g40 = c40[1][bc.S["cell"]]
    ok4 = np.isfinite(g1) & np.isfinite(g40) & np.isfinite(m4).all(1)
    ok = ok4 & np.isfinite(m16).all(1)                                      # = score16.one
    w = np.where(ok, bc.S["pop"], 0.0)
    v = S["P1"][S["POS"][d]].astype(np.float32)
    per = np.where(v > 0, v, np.nan).astype(np.float32)                    # = v3.start_pix(d)
    cnt4 = {}

    def n4(thr):
        if thr not in cnt4:
            cnt4.clear(); cnt4[thr] = (m4 > thr).sum(1)
        return cnt4[thr]
    with np.errstate(invalid="ignore"):
        T = bc.tables(w, g1, g40, [lambda thr, C: n4(thr) >= 2, lambda thr, C: per > thr])   # (regions, 5, 2, 16)
    Hh = np.zeros((len(THR_ALL), 8, NB + 2), np.float32)                   # start-day curves, verbatim from score16.one
    for j, thr in enumerate(THR_ALL):
        code = ((g1 > thr).astype(np.int64) + 2 * (g40 > thr) + 4 * ((g1 > thr * (1 + bc.MARGIN)) & (g40 <= thr * (1 - bc.MARGIN)))) * (NB + 2)
        edges = (MUS * thr).astype(np.float32)
        i = np.searchsorted(edges, per, side="left"); i[np.isnan(per)] = 0
        Hh[j] = np.bincount(code + i, weights=w, minlength=8 * (NB + 2)).reshape(8, NB + 2)
    lab, NC, Mreg = bc.S["LAB16"] // 16, bc.S["NC"], bc.S["M"]
    nreg = len(bc.REG); nt = len(THR_REP)
    HK = np.zeros((nreg, len(THR_ALL), 8, NF + 1))
    HJ = np.zeros((nreg, nt, 8, NF, NF)); HQ = np.zeros((nreg, nt, len(QMASK), 8)); H3 = np.zeros((nreg, nt, 4, 8))
    HS = np.zeros((nt, len(SUBS), 8))
    dev_ph = 0.0
    for j, thr in enumerate(THR_ALL):
        E, C = g1 > thr, g40 > thr
        Hm = (g1 > thr * (1 + bc.MARGIN)) & (g40 <= thr * (1 - bc.MARGIN))
        code = E.astype(np.int64) + 2 * C + 4 * Hm                          # truth code of bc.tables
        pat = patterns(m16, thr); kk = POPC[pat].astype(np.int64)
        hk = np.bincount((lab * 8 + code) * (NF + 1) + kk, weights=w, minlength=NC * 8 * (NF + 1)).reshape(NC, 8 * (NF + 1))
        HK[:, j] = (Mreg @ hk).reshape(nreg, 8, NF + 1)
        if thr in THR_REP:
            jj = THR_REP.index(thr)
            PHc = np.bincount((lab * 8 + code) * (1 << NF) + pat, weights=w, minlength=NC * 8 << NF).reshape(NC, 8 << NF)
            PHr = (Mreg @ PHc).reshape(nreg, 8, 1 << NF); del PHc            # residents by region, truth code and pattern
            for rg in range(nreg):
                HJ[rg, jj] = np.stack([ph_jack(PHr[rg, c]) for c in range(8)])
            HQ[:, jj] = (S["G4"] @ PHr.reshape(nreg * 8, 1 << NF).T).reshape(len(QMASK), nreg, 8).transpose(1, 0, 2)
            HS[jj] = S["SUBI"] @ PHr[0].T
            hk0 = np.stack([np.bincount(POPC, weights=PHr[0, c], minlength=NF + 1) for c in range(8)])
            dev_ph = max(dev_ph, float(np.abs(hk0 - HK[0, j]).max() / max(1.0, HK[0, j].max())))
            del PHr
            t3 = triples_table(m4 > thr, lab * 8 + code, w, NC * 8)          # (4, NC x 8): the stored triples, at least 2 of 3
            H3[:, jj] = (Mreg @ t3.reshape(4, NC, 8).transpose(1, 0, 2).reshape(NC, 32)).reshape(nreg, 4, 8)
    mv = members_vs_stored(m16, m4)
    same = min(x["bit_identical_share"] for x in mv)
    diag = dict(valid_M=float(w.sum() / 1e6), valid4_M=float(np.where(ok4, bc.S["pop"], 0.0).sum() / 1e6),
                members0_3_bit_identical_share=same, members0_3=mv, pattern_vs_k_hist_max_rel_dev=dev_ph)
    print(f"start {d} lead {h} target {t}: {w.sum() / 1e6:.0f} M valid residents, members 0-3 bit-identical >= {same:.4f}, max |diff| "
          f"{max(x['max_abs'] for x in mv):.3f}, {time.time() - t0:.0f} s", flush=True)
    return dict(key=key, t=t, T=T, Hh=Hh, HK=HK, HJ=HJ, HQ=HQ, H3=H3, HS=HS, diag=diag)


def safe_savez(path, **arrs):
    """np.savez to a temporary file of the same folder + fsync + read-back + rename"""
    path = Path(path); tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrs)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    z = np.load(tmp)
    for k_, v_ in arrs.items():
        assert np.array_equal(z[k_], np.asarray(v_), equal_nan=np.asarray(v_).dtype.kind == "f"), k_
    os.replace(tmp, path)
    assert path.exists()


def run_pass(setname, workers, gate="check13"):
    t0 = time.time()
    bc.setup()
    keys_seen = [(d, h) for h in (1, 3) for d in DOYS]
    seen = index16(RUN16) if not AR else index16(RUN16_AR, keep=set(keys_seen))   # 'ar': the seen start days from the one folder
    assert all(k in seen for k in keys_seen), "run16 incomplete"
    if setname == "seen30":
        R16 = {k: seen[k] for k in keys_seen}
        keys = sorted(keys_seen, key=lambda k: (k[1], k[0]))
    elif setname == "all2022":
        gate_check13(gate)
        if AR:
            R16 = index16(RUN16_AR)                                          # every forecast of the folder (asserted = the 358 below)
        else:
            unseen = index16(RUN16_13); extra = index16(RUN16ALL, keep=set(TASK0))
            assert set(extra) == set(TASK0), sorted(extra)
            assert not set(unseen) & set(extra) and not (set(unseen) | set(extra)) & set(keys_seen)
            R16 = {**{k: seen[k] for k in keys_seen}, **unseen, **extra}
        keys = sorted(bc.S["pop_index"], key=lambda k: (k[1], k[0]))
        assert len(keys) == 358 and set(R16) == set(keys), (len(keys), sorted(set(keys) ^ set(R16))[:10])
    else:
        raise SystemExit(f"unknown set {setname}")
    S["R16"] = R16
    z = np.load((v3.HERE / "data" if AR else v3.DATA) / "start_est_v3.npz")      # truth only: the stored file under 'ar' (as v3 does)
    days = [int(x) for x in z["days"]]; pos = {d: i for i, d in enumerate(days)}
    need = sorted({d for d, _ in keys})
    pix = z["pix"]; S["P1"] = pix[[pos[d] for d in need]].copy(); del pix
    S["POS"] = {d: i for i, d in enumerate(need)}
    S["G4"], S["SUBI"] = quartet_indicator(), subset_indicator()
    print(f"setup {time.time() - t0:.0f} s; {len(keys)} forecasts, {workers} workers", flush=True)
    with get_context("fork").Pool(workers) as pool:
        res = pool.map(one, keys, chunksize=1)
    res.sort(key=lambda r: (r["key"][1], r["key"][0]))
    K = np.array([(r["key"][0], r["key"][1], r["t"]) for r in res], np.int64)
    diag = [r["diag"] for r in res]
    # ---- members 0-3 against the stored members, every forecast (A14 K4; tolerance of a14/r2/smoke_check.py)
    bad = [(int(k_[0]), int(k_[1]), m, x) for k_, dg in zip(K, diag) for m, x in enumerate(dg["members0_3"])
           if not (x["finite_pattern_equal"] and x["bit_identical_share"] >= TOL_MEMBERS["min_bit_identical_share"]
                   and x["max_abs"] <= TOL_MEMBERS["max_abs"] and x["mean_abs"] <= TOL_MEMBERS["mean_abs"])]
    memb = dict(tolerance=TOL_MEMBERS, forecasts=len(K), failing=bad[:20], n_failing=len(bad),
                min_bit_identical_share=min(x["bit_identical_share"] for dg in diag for x in dg["members0_3"]),
                worst_max_abs=max(x["max_abs"] for dg in diag for x in dg["members0_3"]),
                worst_mean_abs=max(x["mean_abs"] for dg in diag for x in dg["members0_3"]))
    print("members 0-3 vs stored:", json.dumps(memb), flush=True)
    if bad:
        print(f"EXIT 2: members 0-3 of {len(bad)} forecast-members are not at the A13 rounding level: the 16-member files are not usable", flush=True)
        sys.exit(2)
    T = np.stack([r["T"] for r in res]); Hh = np.stack([r["Hh"] for r in res])
    HK = np.stack([r["HK"] for r in res]); HJ = np.stack([r["HJ"] for r in res]); HQ = np.stack([r["HQ"] for r in res])
    H3 = np.stack([r["H3"] for r in res]); HS = np.stack([r["HS"] for r in res])
    del res
    # ---- identity with analyses/hidden/v3 (the paper's analysis), as score16
    zv = np.load(v3.DATA / "hidden_v3_pop.npz")
    Kv, s1, hs = zv["keys"], [str(s) for s in zv["sys1"]], [str(s) for s in zv["hsys"]]
    rows = [int(np.flatnonzero((Kv[:, 0] == d) & (Kv[:, 1] == h))[0]) for d, h, _ in K]
    assert np.array_equal(Kv[rows], K)
    Tv = zv["T1"][rows]; Hv = zv["H"][rows][:, :, hs.index("persist")]; dv = zv["diag"][rows]
    tot16 = HK.sum(-1); totp = T[..., 1, :8] + T[..., 1, 8:]
    G = [i for i, r in enumerate(bc.REG) if r.startswith("G:")]
    reps = [i for i, r in enumerate(bc.REG) if r != "World"]
    ident = {"foehn4 (stored, 2 of 4) = v3 foehn_p50": bool(np.array_equal(T[..., 0, :], Tv[..., s1.index("foehn_p50"), :])),
             "persist = v3 persist": bool(np.array_equal(T[..., 1, :], Tv[..., s1.index("persist"), :])),
             "persist curves = v3 H": bool(np.array_equal(Hh, Hv)),
             "valid residents = v3 (max abs dev, residents)": float(np.abs(np.array([x["valid_M"] for x in diag]) * 1e6 - dv[:, 0]).max()),
             "residents by truth code from the 16-member k tables = persist table, all regions (max rel dev)":
                 float(np.abs(tot16 - totp).max() / max(1.0, np.abs(totp).max())),
             "pattern histograms = k tables (max rel dev)": max(x["pattern_vs_k_hist_max_rel_dev"] for x in diag),
             "country groups G: partition the World: HK, HJ, HQ, H3 (max rel dev)": max(
                 float(np.abs(A_[:, G].sum(1) - A_[:, 0]).max() / max(1.0, np.abs(A_[:, 0]).max())) for A_ in (HK, HJ, HQ, H3)),
             "jackknife tables of every region sum to its k tables (max rel dev)": float(
                 np.abs(HJ.sum(-1) - HK[:, :, [THR_ALL.index(t_) for t_ in THR_REP]].sum(-1)[..., None]).max() / max(1.0, HK.max())),
             "regions other than World: checked": len(reps)}
    print("identity with v3:", json.dumps(ident), flush=True)
    assert ident["foehn4 (stored, 2 of 4) = v3 foehn_p50"] and ident["persist = v3 persist"] and ident["persist curves = v3 H"], ident
    assert ident["valid residents = v3 (max abs dev, residents)"] < 1.0, ident
    assert ident["residents by truth code from the 16-member k tables = persist table, all regions (max rel dev)"] < 1e-12, ident
    assert ident["pattern histograms = k tables (max rel dev)"] < 1e-12, ident
    assert ident["country groups G: partition the World: HK, HJ, HQ, H3 (max rel dev)"] < 1e-12, ident
    assert ident["jackknife tables of every region sum to its k tables (max rel dev)"] < 1e-12, ident
    DATA.mkdir(exist_ok=True)
    out = DATA / f"hist_{setname}_v2.npz"
    safe_savez(out, keys=K, T=T, Hh=Hh, HK=HK, HJ=HJ, HQ=HQ, H3=H3, HS=HS,
               valid_M=np.array([x["valid_M"] for x in diag]), valid4_M=np.array([x["valid4_M"] for x in diag]),
               members0_3=json.dumps([x["members0_3"] for x in diag]), members_check=json.dumps(memb),
               thr_all=np.array(THR_ALL), thr_rep=np.array(THR_REP), regions=np.array(bc.REG), quartets=np.array(QUARTETS),
               subsets=np.array(SUBNAMES), identity=json.dumps(ident),
               sources=json.dumps({f"{k_[0]}:{k_[1]}": R16[k_][0] + f"[{R16[k_][1]}]" for k_ in keys}),
               layout=np.array("T (forecast, region, threshold, [stored 4 'at least 2 of 4', start-day estimate], 16 codes) of bc.tables; "
                               "Hh (forecast, threshold, code8, NB + 2) start-day curves of score16; HK (forecast, region, threshold, code8, k of 16); "
                               "HJ (forecast, region, reported threshold, code8, deleted field, k of the 15 others); HQ (forecast, region, reported "
                               "threshold, quartet, code8) warned by 'at least 2 of 4' of the quartet; H3 (forecast, region, reported threshold, "
                               "deleted stored member, code8) warned by 'at least 2 of 3' of the 3 other STORED members; HS (forecast, reported "
                               "threshold, sub-ensemble of extrap16, code8) warned by its direct rule, World; code8 = exposed + 2 x 40 km mean "
                               "above + 4 x hidden with the 10 % margin"))
    print(f"pass {setname} done {time.time() - t0:.0f} s -> {out}", flush=True)


# ------------------------------------------------------------------ statistics
def sample_rows(K, h, sname):
    """rows of one lead in start-day order, bootstrap count matrix, number of days and the bootstrap of a sample"""
    sel = np.flatnonzero(K[:, 1] == h)
    sel = sel[np.argsort(K[sel, 0], kind="stable")]
    days = K[sel, 0]
    seen = np.isin(days, DOYS)
    if sname == "seen30":
        assert seen.sum() == 30
        return sel[seen], mbb(30, 1), 30, "score16 blocks: tables.mbb(30, 1), blocks of one start day (12 days apart)"
    assert len(sel) == 179 and (np.diff(days) == 2).all(), "the 179 start days every second day are needed"
    c = mbb(179, 4)
    if sname == "all179":
        return sel, c, 179, "tables.mbb(179, 4): blocks of 4 consecutive forecasts (7 days), as analyses/hidden/v3"
    if sname in ("conf149", "seen30_paperblocks"):
        keep = ~seen if sname == "conf149" else seen
        nk = int(keep.sum())
        c = c[:, keep].copy()
        assert c.sum(1).min() > 0, "a resample without any day of the sample"
        c *= (nk / c.sum(1))[:, None]
        return sel[keep], c, nk, (f"the resamples of tables.mbb(179, 4) (blocks of 4 consecutive forecasts, as analyses/hidden/v3) with the "
                                  f"{'30 seen' if sname == 'conf149' else '149 unseen'} start days removed, each resample rescaled to {nk} days")
    raise ValueError(sname)


def lead3v1_rows(K, sname):
    """v3's common target days of the forecasts issued 3 days and 1 day ahead: rows of Foehn at lead 3 and of the references at
    lead 1, in target order, and the bootstrap (sample by Foehn's lead-3 start day)"""
    s1 = np.flatnonzero(K[:, 1] == 1); s1 = s1[np.argsort(K[s1, 2], kind="stable")]
    s3 = np.flatnonzero(K[:, 1] == 3); s3 = s3[np.argsort(K[s3, 2], kind="stable")]
    t1, t3 = K[s1, 2], K[s3, 2]
    common = np.intersect1d(t1, t3)
    assert len(common) == 178 and (np.diff(common) == 2).all()
    i1, i3 = s1[np.searchsorted(t1, common)], s3[np.searchsorted(t3, common)]
    seen = np.isin(K[i3, 0], DOYS)
    if sname == "seen30":
        return i3[seen], i1[seen], mbb(int(seen.sum()), 1), int(seen.sum()), "blocks of one start day (score16), Foehn's seen lead-3 start days"
    c = mbb(178, 4)
    if sname == "all179":
        return i3, i1, c, 178, "tables.mbb(178, 4) over the 178 common target days, as analyses/hidden/v3"
    keep = ~seen if sname == "conf149" else seen
    nk = int(keep.sum()); c = c[:, keep].copy(); assert c.sum(1).min() > 0; c *= (nk / c.sum(1))[:, None]
    return i3[keep], i1[keep], c, nk, (f"the resamples of tables.mbb(178, 4) with the common target days of Foehn's "
                                       f"{'seen' if sname == 'conf149' else 'unseen'} lead-3 start days removed, rescaled to {nk} days")


def load_refs():
    """v3's reference tables (forecast, region, threshold, ref, 16) and curves (forecast, threshold, ref, 8, NB + 2), rows in the
    order of the pass (asserted equal for the stored-4 and persist tables in the pass and again here)"""
    K = S["Z"]["keys"]
    zv = np.load(v3.DATA / "hidden_v3_pop.npz")
    zb = np.load((v3.HERE.parent / "v2" / f"data{SFX}" if AR else v3.DATA) / "hidden_bc_pop.npz")   # 'ar': the v2 output of the design, as v3
    s1, hs, sb = [str(s) for s in zv["sys1"]], [str(s) for s in zv["hsys"]], [str(s) for s in zb["sys1"]]
    assert np.array_equal(zv["keys"], zb["keys"])
    Kv = zv["keys"]; rows = [int(np.flatnonzero((Kv[:, 0] == d) & (Kv[:, 1] == h))[0]) for d, h, _ in K]
    assert np.array_equal(Kv[rows], K)
    T1, Tb = zv["T1"], zb["T1"]
    RT = np.stack([T1[rows][:, :, :, s1.index(r)] if r != "clim_1km" else Tb[rows][:, :, :, sb.index(r)] for r in REFS_ISSUED], 3)
    RH = zv["H"][rows][:, :, [hs.index(r) for r in REFS_MOVED]]
    assert np.array_equal(RT[..., REFS_ISSUED.index("persist"), :], S["Z"]["T"][..., 1, :]), "persist table of the pass differs from v3"
    assert np.array_equal(T1[rows][:, :, :, s1.index("foehn_p50")], S["Z"]["T"][..., 0, :]), "stored-4 table of the pass differs from v3"
    assert np.array_equal(RH[:, :, REFS_MOVED.index("persist")], S["Z"]["Hh"]), "persist curves of the pass differ from v3"
    S["RT"], S["RH"] = RT, RH


def _world_task(args):
    sname, h, thr = args
    Z = S["Z"]
    rows, cnt, n, how = sample_rows(Z["keys"], h, sname)
    j, jj = THR_ALL.index(thr), THR_REP.index(thr)
    RIt = {r: S["RT"][rows, 0, j, q] for q, r in enumerate(REFS_ISSUED)}
    RM = {r: S["RH"][rows, j, q] for q, r in enumerate(REFS_MOVED)}
    out, info = foehn_block(Z["HK"][rows, 0, j], Z["HJ"][rows, 0, jj], Z["HQ"][rows, 0, jj], Z["H3"][rows, 0, jj], Z["T"][rows, 0, j, 0],
                            cnt, n, RIt, RM)
    info.update(n_start_days=int(n), bootstrap=how, start_days=[int(x) for x in Z["keys"][rows, 0]])
    print(f"world {sname} lead {h} > {thr:g}: {info['seconds']} s", flush=True)
    return ("world", sname, h, thr), out, info


def _region_task(args):
    sname, h, thr = args
    Z = S["Z"]
    rows, cnt, n, how = sample_rows(Z["keys"], h, sname)
    j, jj = THR_ALL.index(thr), THR_REP.index(thr)
    t0 = time.time(); out = {}
    for rg, name in enumerate(bc.REG):
        RIt = {r: S["RT"][rows, rg, j, q] for q, r in enumerate(REFS_ISSUED)}
        o_, i_ = foehn_block(Z["HK"][rows, rg, j], Z["HJ"][rows, rg, jj], Z["HQ"][rows, rg, jj], Z["H3"][rows, rg, jj], Z["T"][rows, rg, j, 0],
                             cnt, n, RIt, {}, with_inf=False, with_full=False, with_curves=False, qs=QS_REGION)
        out[name] = dict(quantities={q: {s: e[s] for s in ("M8_liberal", "M16_liberal_measured", "M4_stored")} for q, e in o_.items()},
                         exposed_M=float(Tab(Z["T"][rows, rg, j, 0], cnt).pt["exposed_M"]), check_point_only_max_rel_dev=i_["check_point_only_max_rel_dev"])
    info = dict(n_start_days=int(n), bootstrap=how, seconds=round(time.time() - t0, 1))
    print(f"regions {sname} lead {h} > {thr:g}: {info['seconds']} s", flush=True)
    return ("regions", sname, h, thr), out, info


def _lead3v1_task(args):
    sname, thr = args
    Z = S["Z"]
    i3, i1, cnt, n, how = lead3v1_rows(Z["keys"], sname)
    j, jj = THR_ALL.index(thr), THR_REP.index(thr)
    RIt = {r: S["RT"][i1, 0, j, REFS_ISSUED.index(r)] for r in REFS_LEAD1}
    out, info = foehn_block(Z["HK"][i3, 0, j], Z["HJ"][i3, 0, jj], Z["HQ"][i3, 0, jj], Z["H3"][i3, 0, jj], Z["T"][i3, 0, j, 0],
                            cnt, n, RIt, {}, with_full=False, with_curves=False, qs=QS_SHORT)
    keep = {q: e for q, e in out.items() if q.startswith("diff:") or (q.startswith("issued:") and q.split(":")[1] in CROSS)}
    info.update(n_common_target_days=int(n), bootstrap=how, lead3_start_days=[int(x) for x in Z["keys"][i3, 0]])
    print(f"lead3v1 {sname} > {thr:g}: {info['seconds']} s", flush=True)
    return ("lead3v1", sname, 3, thr), keep, info


def clean(v):
    if isinstance(v, dict):
        return {str(k_): clean(x) for k_, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    return None if isinstance(v, float) and not np.isfinite(v) else v


FMT = {"warned_M": 1, "right_M": 1, "false_M": 1, "reach_pct": 2, "correct_pct": 2, "false_pct": 2, "csi": 4, "hidden_warned_M": 2,
       "hidden_warned_pct": 2, "hidden_margin_warned_M": 2, "hidden_margin_warned_pct": 2, "seen_warned_pct": 2,
       "warned_in_silent40_M": 1, "false_in_silent40_M": 1, "false_pct_in_silent40": 2, "false_per_hidden_warned": 3}
MD_ROWS = ("issued:warned_M", "issued:right_M", "issued:false_M", "issued:reach_pct", "issued:correct_pct", "issued:csi",
           "issued:hidden_warned_M", "issued:hidden_warned_pct", "issued:hidden_margin_warned_M", "issued:false_in_silent40_M",
           "issued:false_per_hidden_warned", "eqn:persist:reach_pct", "eqn:persist:correct_pct", "eqc:persist:warned_M",
           "eqc:persist:reach_pct", "eqn:cams_bcsd:reach_pct", "eqn:cams_bcsd:correct_pct", "eqn:cams_bcsd:hidden_warned_M",
           "eqn:cams:reach_pct", "eqc:cams:reach_pct", "diff:cams:reach_pct", "diff:cams:correct_pct", "diff:cams:right_M",
           "diff:aurora:reach_pct", "diff:aurora:correct_pct", "diff:cams:false_in_silent40_M")
LABEL = {"issued": "as issued", "eqn": "minus {ref} at equal number warned", "eqc": "minus {ref} at equal share correct",
         "diff": "minus {ref} as issued", "ratio": "over {ref} as issued"}


def qlabel(q):
    p = q.split(":")
    if p[0] == "issued":
        return f"{p[1]} as issued"
    return f"{p[2]} {LABEL[p[0]].format(ref=p[1])}"


def _f(v, d):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def _cell(e, d):
    lo, hi = e["interval95"]
    st = e.get("interval_status") or ""
    mark = ("" if st == "inflated" else "‡" if st.startswith("NOT inflated") else "") + (f"†{e['resamples_nan']}" if "truncated" in st else "")
    return f"{_f(e['value'], d)} [{_f(lo, d)}, {_f(hi, d)}]{mark}"


LEGEND = ("Counts in millions of people a day; shares in %; differences in points. † n: the interval is truncated: n of the 10,000 "
          "resamples had no crossing within +-0.25 of the point factor and were dropped (v3/score16 convention; the whole-grid "
          "sensitivity is in the JSON, entry 'full_grid'); ‡: not inflated (V_draw undefined or V_boot = 0), bootstrap percentile "
          "interval only. Infinite value: the signed-kernel system of each fit, every ratio and comparison formed from its sums.")
DESIGN_MD = (f"**Noise design FOEHN_DESIGN={DESIGN}: A16 section 4 point 3, V_draw = 0 and factor 1. Every interval below is the bootstrap "
             "percentile interval (the inflation described in this header belongs to the fixed design); the fixed-design V_draw is kept in "
             "the JSON as 'V_draw_fixed_design', information only.**\n")


def markdown(R, title):
    L = [f"# {title}", "",
         "Headline (A14 D2): eight members, liberal rule (at least 4 of 8), exact expected value from the 16 fields. Beside it: 16 members "
         "measured (at least 8 of 16) and the 4 stored paper members (at least 2 of 4). Infinite value: point = symmetric-rule intercept "
         "(fit in 1/M), range = the six fits (three rules, 1/M and 1/(M+1)), interval = union of their inflated bootstrap intervals. "
         "Intervals: 95 %, paired moving-block bootstrap with the fits inside every resample, half-widths multiplied by "
         "sqrt(1 + V_draw/V_boot). " + LEGEND, ""]
    if VDRAW0:
        L.insert(2, DESIGN_MD)
    for sname in SAMPLES:
        if sname not in R.get("world", {}):
            continue
        L += [f"## {SAMPLE_NAMES[sname]}", ""]
        for thr in THR_REP:
            for h in (1, 3):
                B = R["world"][sname].get(f"thr{thr:g}", {}).get(f"lead{h}")
                if B is None:
                    continue
                L += [f"### > {thr:g} ug m-3, lead {h} d ({B['info']['n_start_days']} start days; {B['info']['bootstrap']})", "",
                      "| quantity | 8 members (headline) | 16 members (measured) | 4 members (stored) | infinite: point | infinite: range | infinite: interval |",
                      "|---|---|---|---|---|---|---|"]
                for q in MD_ROWS:
                    m = q.split(":")[-1]; d = FMT[m]; e = B["quantities"].get(q)
                    if e is None:
                        continue
                    inf = e["infinite"]
                    L.append(f"| {qlabel(q)} | {_cell(e['M8_liberal'], d)} | {_cell(e['M16_liberal_measured'], d)} | {_cell(e['M4_stored'], d)} | "
                             f"{_f(inf['point'], d)} | [{_f(inf['range'][0], d)}, {_f(inf['range'][1], d)}]{'' if not inf['undefined_fits'] else ' (' + str(6 - len(inf['undefined_fits'])) + ' of 6)'} | "
                             f"[{_f(inf['interval95'][0], d)}, {_f(inf['interval95'][1], d)}]{'†' if inf['interval_parts_truncated'] else ''} |")
                L.append("")
        LB = R.get("lead3v1", {}).get(sname)
        if LB:
            L += [f"### Foehn 3 days ahead minus the references 1 day ahead ({sname}; v3's common target days)", "",
                  "| threshold | quantity | 8 members | 16 members | 4 members (stored) | infinite: point [range] |", "|---|---|---|---|---|---|"]
            for thr in THR_REP:
                B = LB.get(f"thr{thr:g}")
                if B is None:
                    continue
                for ref in ("cams", "aurora", "cams_bcsd"):
                    for m in ("reach_pct", "correct_pct"):
                        e = B["quantities"].get(f"diff:{ref}:{m}")
                        if e is None:
                            continue
                        inf = e["infinite"]; d = FMT[m]
                        L.append(f"| > {thr:g} | {m}, minus {ref} 1 d | {_cell(e['M8_liberal'], d)} | {_cell(e['M16_liberal_measured'], d)} | "
                                 f"{_cell(e['M4_stored'], d)} | {_f(inf['point'], d)} [{_f(inf['range'][0], d)}, {_f(inf['range'][1], d)}] |")
            L.append("")
    return "\n".join(L) + "\n"


def markdown_regions(RR, title):
    L = [f"# {title}", "", "Regions and country groups (bc.REG) at 4, 8 and 16 members only (A14 4b: never infinite). " + LEGEND, ""]
    if VDRAW0:
        L.insert(2, DESIGN_MD)
    for sname in REGION_SAMPLES:
        if sname not in RR:
            continue
        L += [f"## {SAMPLE_NAMES[sname]}", ""]
        for thr in (37.5, 75.0):
            for h in (1, 3):
                B = RR[sname].get(f"thr{thr:g}", {}).get(f"lead{h}")
                if B is None:
                    continue
                L += [f"### > {thr:g} ug m-3, lead {h} d", "",
                      "| region | exposed M | reach % (8 / 16 / 4 members) | correct % (8 / 16 / 4) | hidden warned M (8 / 16 / 4) | reach minus CAMS (8 members) |",
                      "|---|---|---|---|---|---|"]
                for name, x in B["regions"].items():
                    if x["exposed_M"] is None or x["exposed_M"] < 0.05:
                        continue
                    Q = x["quantities"]

                    def tri(q, d):
                        return " / ".join(_f(Q[q][s]["value"], d) for s in ("M8_liberal", "M16_liberal_measured", "M4_stored"))
                    L.append(f"| {name} | {x['exposed_M']:.1f} | {tri('issued:reach_pct', 1)} | {tri('issued:correct_pct', 1)} | "
                             f"{tri('issued:hidden_warned_M', 2)} | {_cell(Q['diff:cams:reach_pct']['M8_liberal'], 1)} |")
                L.append("")
    return "\n".join(L) + "\n"


def run_stats(setname, workers, samples=None):
    t0 = time.time()
    z = np.load(DATA / f"hist_{setname}_v2.npz")
    S["Z"] = {k_: z[k_] for k_ in ("keys", "HK", "HJ", "HQ", "H3", "T", "Hh")}
    load_refs()
    samples = samples or SAMPLES
    tasks = ([(_world_task, (s_, h, thr)) for s_ in samples for h in (1, 3) for thr in THR_REP]
             + [(_lead3v1_task, (s_, thr)) for s_ in samples for thr in THR_REP]
             + [(_region_task, (s_, h, thr)) for s_ in samples if s_ in REGION_SAMPLES for h in (1, 3) for thr in THR_REP])
    with get_context("fork").Pool(min(workers, len(tasks))) as pool:
        res = pool.map(_dispatch, tasks, chunksize=1)
    R, RR = {"world": {}, "lead3v1": {}}, {}
    for (kind, s_, h, thr), out, info in res:
        if kind == "world":
            R["world"].setdefault(s_, {}).setdefault(f"thr{thr:g}", {})[f"lead{h}"] = dict(info=info, quantities=out)
        elif kind == "lead3v1":
            R["lead3v1"].setdefault(s_, {})[f"thr{thr:g}"] = dict(info=info, quantities=out)
        else:
            RR.setdefault(s_, {}).setdefault(f"thr{thr:g}", {})[f"lead{h}"] = dict(info=info, regions=out)
    head = dict(what=__doc__, set=setname, samples={s_: SAMPLE_NAMES[s_] for s_ in samples}, identity_of_the_pass=json.loads(str(z["identity"])),
                members_check=json.loads(str(z["members_check"])),
                rules={"liberal": "even M, warn when at least M/2 of M exceed (issued rule)", "symmetric": "even M, weight 1/2 at exactly M/2",
                       "strict": "odd M, at least (M+1)/2 of M"},
                grids={"fit": GRID, "jackknife": GRID_JK}, fits=list(FITS), point=POINT, references=REF_NOTE,
                quantities={"issued": "the system as issued (metrics of tables.py)",
                            "diff:<ref>": "system minus reference as issued (paired)", "ratio:<ref>": "system over reference as issued",
                            "eqn:<ref>": "system minus the reference moved to the system's number of people warned (score16.persist_at; point on "
                                         "the reference's factor curve, linear inside one step of 0.001)",
                            "eqc:<ref>": "system minus the reference moved to the system's share of correct warnings (tables_v3.at_factor, "
                                         "equal_correct_share branch; point on the curve)",
                            "lead3v1 diff:<ref>": "Foehn 3 days ahead minus the reference 1 day ahead, v3's common target days"},
                V_draw={"M8_liberal": "jackknife over the 16 fields (pi_8 of the 15 others)",
                        "M16_liberal_measured": "jackknife over the 16 fields of the liberal 1/M fit on even M = 4..14 evaluated at M = 16 "
                                                "(one grid for the full estimate and every replicate; approximate)",
                        "M4_stored": "max((i) jackknife over the 4 stored fields, 'at least 2 of 3' of the 3 others; (ii) variance over the "
                                     "1,820 quartets of the 16 fields + jackknife variance of their mean)",
                        "fits": "jackknife over the 16 fields of the same fit on the reduced grid (even M = 4..14, odd M = 5..13), approximate"},
                kernels={f"{r} {fm}": kernel(NF, r, fm, GRID[r]).tolist() for r in RULES for fm in FORMS},
                fit_weights={f"{r} {fm}": dict(zip([str(M) for M in GRID[r]], ls_rows(GRID[r], fm)[0].tolist())) for r in RULES for fm in FORMS},
                relative_economic_value="NOT DELIVERED in this build (A14 4b; the paper reports it at the monitoring stations, Supplementary "
                                        "Fig. S-fig:exposure i,j): the paper's value stays labelled 'four members'; see README",
                k3="NOT DONE (A14 4b 'Functionals of the ensemble mean', K3); see README")
    if VDRAW0:
        head["design"] = dict(FOEHN_DESIGN=DESIGN, rule="A16 section 4 point 3: V_draw = 0, factor 1 in every interval; estimators, fits, "
                              "count rules, samples (conf149 = the 149 unseen start days, the 30 seen days stay seen) and bootstrap unchanged",
                              sixteen_members=str(RUN16_AR if AR else "the stored folders"))
    RES.mkdir(exist_ok=True)
    s16.safe_write(RES / f"classE_{setname}.json", json.dumps(clean(dict(head, results=R)), allow_nan=False))
    s16.safe_write(RES / f"classE_{setname}.md", markdown(clean(R), f"A14 class E, count rules ({setname})"))
    s16.safe_write(RES / f"classE_regions_{setname}.json", json.dumps(clean(dict(head, results=RR)), allow_nan=False))
    s16.safe_write(RES / f"classE_regions_{setname}.md", markdown_regions(clean(RR), f"A14 class E, regions at 4, 8, 16 members ({setname})"))
    print(f"stats {setname} done {time.time() - t0:.0f} s -> {RES / f'classE_{setname}.json'}", flush=True)
    return R, RR


def _dispatch(task):
    fn, args = task
    return fn(args)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("pass", "stats", "run2022"):
        raise SystemExit(__doc__)
    if a[0] == "run2022":
        workers = int(a[1]) if len(a) > 1 else 16
        gate = a[2] if len(a) > 2 else "check13"
        gate_check13(gate)
        run_pass("all2022", workers, gate)
        run_stats("all2022", workers)
        return
    setname = a[1]; workers = int(a[2]) if len(a) > 2 else 16
    gate = a[3] if len(a) > 3 else "check13"
    if a[0] == "stats" and setname != "all2022":
        raise SystemExit("stats only on all2022 (after the check13 gate): no further class-E value of 2022 before R1 passed its check")
    if setname == "all2022":
        gate_check13(gate)
    if a[0] == "pass":
        run_pass(setname, workers, gate)
    else:
        run_stats(setname, workers)


if __name__ == "__main__":
    main()
