#!/usr/bin/env python3
"""Detection of local extremes over land against the 1 km estimates (GHAP 0.25 deg block means, 2022) with thresholds
matched in frequency, at equal issuance. Reads the fields written by `ISSUANCE=d1 DUMP=1 tail_events_eqiss.py`
(../data/fields_d1_h<lead>.npy: truth, Foehn mean, one member, CAMS, Aurora, persistence; start days x scored cells).

Four kinds of thresholds, three tiers (P90, P95, P99), leads 1-7 (CAMS and Aurora 1-5), six regions:
  'ghap'      the paper's: local percentile of GHAP 2017-2020 for the cell and calendar day (+-15 days), the same threshold
              for the truth and every system. Recomputed here from the fields; must equal tail_events_<label>_d1.npz.
  'matched'   the paper's EVENTS (truth above the 2017-2020 local percentile of the calendar day), and for every system a
              threshold rescaled at each cell by one factor so that the system warns exactly as often as the event occurs
              at that cell over the scored days (yes on the n_e days with the largest forecast / threshold ratio, n_e =
              number of event days of the cell). Frequency bias 1 at every cell; only the ranking of the days counts.
  'own'       as experiments/figures/extremes_stations_own.py does at the stations: the truth and every system have their
              own percentile at each cell over the scored days of the lead (annual, 2022). Frequency bias 1.
  'truth2022' the percentile of the truth (GHAP 2022) at each cell over the scored days, applied to every system
              (the land analogue of 'each station's own percentiles').
Counts per start day are kept (thresholds fixed) for the moving-block bootstrap.
    python land_freqmatch.py <label> <lead>     -> ../data/land_freqmatch_<label>_h<lead>.npz   (one job per lead)
    python land_freqmatch.py <label> merge      -> ../data/land_freqmatch_<label>.npz"""
import os
import sys
from pathlib import Path

import numpy as np

G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
THR = G / "clim" / "tail_thresholds_2017_2020.npy"
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
OUT = Path(__file__).resolve().parent.parent / f"data{SFX}"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "crps_e01"
MODE = sys.argv[2] if len(sys.argv) > 2 else "merge"
NR, NC = 576, 1440
PCTS = (90, 95, 99)
KINDS = ["ghap", "matched", "own", "truth2022"]
REG = {"Global": None, "East Asia": (54, 18, 97, 146), "South Asia": (37, 5, 60, 97), "Europe": (72, 35, -25, 45),
       "North America": (72, 15, -170, -50), "Africa": (37, -35, -20, 52)}


def to_centres(a):                                                    # as tail_events.py
    b = np.concatenate([a, a[..., -1:, :]], -2); b = 0.5 * (b[..., :-1, :] + b[..., 1:, :])
    c = np.concatenate([b, b[..., :1]], -1); return 0.5 * (c[..., :-1] + c[..., 1:])


def pcts(x):
    """percentiles PCTS along axis 0 (linear, as np.percentile); x has no NaN -> (3, cells)"""
    xs = np.sort(x, axis=0); n = x.shape[0]; out = []
    for q in PCTS:
        pos = q / 100.0 * (n - 1); lo = int(np.floor(pos)); hi = min(lo + 1, n - 1); w = np.float32(pos - lo)
        out.append(xs[lo] * (1 - w) + xs[hi] * w)
    return np.stack(out)


te = np.load(OUT / f"tail_events_{LABEL}_d1.npz")
days = te["days"]; DSYS = [str(s) for s in te["dump_systems"]]
SYS = DSYS[1:]                                                        # Foehn mean, one member, CAMS, Aurora, persistence

if MODE == "merge":
    C = np.zeros((len(KINDS), 3, 7, len(SYS), len(REG), 4)); CD = np.zeros((len(days),) + C.shape)
    FB = np.full((len(KINDS), 3, 7, len(SYS)), np.nan)
    for h in range(1, 8):
        z = np.load(OUT / f"land_freqmatch_{LABEL}_h{h}.npz")
        C[:, :, h - 1] = z["C"]; CD[:, :, :, h - 1] = z["CD"]; FB[:, :, h - 1] = z["FB"]
    f = OUT / f"land_freqmatch_{LABEL}.npz"
    np.savez(f, C=C, CD=CD, FB=FB, days=days, systems=np.array(SYS), tiers=np.array(["P90", "P95", "P99"]), kinds=np.array(KINDS),
             regions=np.array(list(REG)))
    print("saved", f); sys.exit(0)

h = int(MODE)
cells = np.load(OUT / "fields_d1_cells.npy")
LATC = 84.0 - 0.25 * np.arange(NR) - 0.125; LONC = -180.0 + 0.25 * np.arange(NC) + 0.125
la, lo = LATC[cells // NC], LONC[cells % NC]
RM = [np.ones(len(cells), bool) if b is None else (la <= b[0]) & (la > b[1]) & (lo >= b[2]) & (lo < b[3]) for b in REG.values()]
RI = [np.flatnonzero(m) for m in RM[1:]]                              # cell indices of the five regions
thr = np.load(THR, mmap_mode="r")
C = np.zeros((len(KINDS), 3, len(SYS), len(REG), 4)); CD = np.zeros((len(days),) + C.shape)
FB = np.full((len(KINDS), 3, len(SYS)), np.nan)
F = np.load(OUT / f"fields_d1_h{h}.npy", mmap_mode="r")               # (6, days, cells)
vd = np.flatnonzero(np.isfinite(np.asarray(F[0][:, :64])).any(1))     # start days with a score (the first start day has no persistence)
Y = np.asarray(F[0][vd]); assert np.isfinite(Y).all()
T = np.stack([to_centres(np.asarray(thr[min(int(days[i]) + h, 364)], np.float32)).reshape(3, -1)[:, cells] for i in vd], 1)   # (3, days, cells)
nd = len(vd); TY = pcts(Y)
print(f"lead {h}: {nd} start days, {len(cells)} cells", flush=True)
for s in range(len(SYS)):
    X = np.asarray(F[1 + s][vd])
    if not np.isfinite(X).all():                                      # CAMS and Aurora beyond their last lead
        assert not np.isfinite(X).any(); continue
    TX = pcts(X)
    for k_t in range(3):
        ev_g = Y > T[k_t]                                             # the paper's events
        n_e = ev_g.sum(0)
        order = np.argsort(-(X / T[k_t]), axis=0, kind="stable")
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(nd)[:, None], 0); del order
        ev_o = Y > TY[k_t][None]
        pairs = ((ev_g, X > T[k_t]),                                  # ghap
                 (ev_g, rank < n_e[None]),                            # matched
                 (ev_o, X > TX[k_t][None]),                           # own
                 (ev_o, X > TY[k_t][None]))                           # truth2022
        for kind, (e_, g_) in enumerate(pairs):
            code = (e_.astype(np.int8) << 1) + g_                     # 3 hit, 2 miss, 1 false alarm, 0 correct negative
            for g in range(len(REG)):
                cg = code if g == 0 else code[:, RI[g - 1]]
                v = np.stack([(cg == 3).sum(1), (cg == 2).sum(1), (cg == 1).sum(1), (cg == 0).sum(1)], -1)   # (days, 4)
                CD[vd, kind, k_t, s, g] = v; C[kind, k_t, s, g] = v.sum(0)
            FB[kind, k_t, s] = g_.sum() / max(e_.sum(), 1)
        del rank
    print(f"lead {h} {SYS[s]:22s} done; frequency bias P99 by kind:", np.round(FB[:, 2, s], 3), flush=True)
f = OUT / f"land_freqmatch_{LABEL}_h{h}.npz"
np.savez(f, C=C, CD=CD, FB=FB, days=days, systems=np.array(SYS), tiers=np.array(["P90", "P95", "P99"]), kinds=np.array(KINDS),
         regions=np.array(list(REG)), lead=h)
print("saved", f)
