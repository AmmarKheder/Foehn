#!/usr/bin/env python3
"""Under which definition does "Foehn h days ahead detects local extremes as well as the CAMS forecast 1 day ahead" hold?
One table for every definition of the threshold, over land (truth: the 1 km estimates, 0.25 deg block means) and at the
stations of the six official networks (truth: the measurements), equal issuance everywhere, tiers P99 / P95 / P90:
  SEDI by lead of Foehn's ensemble mean, the CAMS forecast and Aurora, with hit rate, false-alarm ratio, frequency bias;
  Foehn minus CAMS / Aurora at equal lead; Foehn at lead h minus CAMS / Aurora at 1 day;
  'match_point'    the longest lead h (consecutive from 1) at which Foehn's SEDI is not below that of the baseline at 1 day;
  'match_interval' the longest lead h (consecutive from 1) at which the 95 % interval of the difference is not entirely below 0;
  'crossing'       the lead at which Foehn's SEDI curve, interpolated linearly between lead days, falls to the baseline's
                   1-day value (first crossing; 7 if never within the seven leads), with its bootstrap interval.
Intervals: paired moving-block bootstrap over start days (blocks of 7 start days, 10,000 resamples, seed 0; common.py),
thresholds kept fixed. Reads ../data/{tail_events_<L>_d1,land_freqmatch_<L>,extremes_stations_kinds4_<L>}.npz (the last one from
extremes_stations_matched.py; extremes_stations_own_eqiss_<L>.npz, without the kind 'matched', if it is missing).
    python derive.py [label]   -> ../data/derived.json, printed table"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from common import block_index, boot_counts, ci, rates, sedi

SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
D = Path(__file__).resolve().parent.parent / f"data{SFX}"
L = sys.argv[1] if len(sys.argv) > 1 else "crps_e01"
TT = ["P90", "P95", "P99"]
r3 = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in np.atleast_1d(a)]
OUT = {}


def crossing(f, ref):
    """f (..., 7) SEDI of Foehn by lead, ref (...) the baseline's 1-day value -> lead at which f falls to ref (first crossing)"""
    f = np.asarray(f, float); ref = np.asarray(ref, float)[..., None]
    below = f < ref                                                     # (..., 7)
    first = np.where(below.any(-1), below.argmax(-1), 7)                # index of the first lead below; 7: never
    out = np.full(first.shape, 7.0)
    i0 = first == 0; out[i0] = 1.0                                      # already below at 1 day
    mid = (first > 0) & (first < 7)
    if mid.any():
        k = first[mid]; fa = np.take_along_axis(f[mid], (k - 1)[:, None], -1)[:, 0]; fb = np.take_along_axis(f[mid], k[:, None], -1)[:, 0]
        out[mid] = k + (fa - ref[mid][:, 0]) / (fa - fb)                # lead of index k-1 is k days
    return out


def block(name, CF, CB, BF, BB, nleadB=5, note=""):
    """CF (7, 4) counts of Foehn; CB {name: (7, 4)} baselines; BF (n, 7) bootstrap SEDI of Foehn; BB {name: (n, 7)}"""
    rf = rates(CF); res = {"note": note, "Foehn": {k: r3(rf[k]) for k in ("sedi", "pod", "far", "fb")}}
    lo, hi = ci(BF); res["Foehn"]["ci_low"], res["Foehn"]["ci_high"] = r3(lo), r3(hi)
    line = [f"    Foehn   SEDI " + " ".join(f"{v:.3f}" for v in rf["sedi"]) + "   hit " + " ".join(f"{v:.2f}" for v in rf["pod"][:5])
            + "   FAR " + " ".join(f"{v:.2f}" for v in rf["far"][:5]) + "   fbias " + " ".join(f"{v:.2f}" for v in rf["fb"][:5])]
    for nm, C in CB.items():
        rb = rates(C[:nleadB]); B = BB[nm]
        res[nm] = {k: r3(rb[k]) for k in ("sedi", "pod", "far", "fb")}
        lo, hi = ci(B[:, :nleadB]); res[nm]["ci_low"], res[nm]["ci_high"] = r3(lo), r3(hi)
        line.append(f"    {nm:7s} SEDI " + " ".join(f"{v:.3f}" for v in rb["sedi"]) + "               hit " + " ".join(f"{v:.2f}" for v in rb["pod"])
                    + "   FAR " + " ".join(f"{v:.2f}" for v in rb["far"]) + "   fbias " + " ".join(f"{v:.2f}" for v in rb["fb"]))
        de = rf["sedi"][:nleadB] - rb["sedi"]; lo, hi = ci(BF[:, :nleadB] - B[:, :nleadB])
        res[f"Foehn minus {nm}, equal lead"] = {"diff": r3(de), "ci_low": r3(lo), "ci_high": r3(hi)}
        line.append(f"      Foehn - {nm} at equal lead 1..5:   " + "  ".join(f"{a:+.3f} [{b:+.3f},{c:+.3f}]" for a, b, c in zip(de, lo, hi)))
        d1 = rf["sedi"] - rb["sedi"][0]; lo1, hi1 = ci(BF - B[:, :1])
        res[f"Foehn at lead h minus {nm} at 1 day"] = {"diff": r3(d1), "ci_low": [round(float(v), 4) for v in lo1], "ci_high": [round(float(v), 4) for v in hi1]}
        line.append(f"      Foehn(h) - {nm}(1 day), h = 1..7:  " + "  ".join(f"{a:+.3f} [{b:+.3f},{c:+.4f}]" for a, b, c in zip(d1, lo1, hi1)))
        mp = 0
        while mp < 7 and d1[mp] >= 0:
            mp += 1
        mi = 0
        while mi < 7 and hi1[mi] >= 0:
            mi += 1
        cr = float(crossing(rf["sedi"][None], np.array([rb["sedi"][0]]))[0]); cb = crossing(BF, B[:, 0]); clo, chi = np.percentile(cb, [2.5, 97.5])
        res[f"match with {nm} at 1 day"] = {"match_point_lead": mp, "match_interval_lead": mi, "crossing_lead": round(cr, 2),
                                            "crossing_ci": [round(float(clo), 2), round(float(chi), 2)]}
        line.append(f"      Foehn matches {nm} of 1 day up to: {mp} d (point estimate), {mi} d (interval); crossing at {cr:.2f} d [{clo:.2f}, {chi:.2f}]")
    print(f"\n  {name}" + (f"   ({note})" if note else "")); print("\n".join(line))
    return res


# ---------------------------------------------------------------- land
te = np.load(D / f"tail_events_{L}_d1.npz"); TS = [str(s) for s in te["systems"]]; TR = [str(r) for r in te["regions"]]
fm = np.load(D / f"land_freqmatch_{L}.npz"); FS = [str(s) for s in fm["systems"]]; FK = [str(s) for s in fm["kinds"]]; FR = [str(s) for s in fm["regions"]]
assert FR == TR
jF, jC, jA = FS.index(f"{L} (mean)"), FS.index("CAMS"), FS.index("Aurora")
vf = np.flatnonzero(fm["CD"][:, 0, 2, 0, jF, 0].sum(-1) > 0); idf = block_index(len(vf))
LAND_KINDS = {"ghap": "same threshold for all: local percentile of the 1 km estimates of 2017-2020, cell and calendar day (the paper's)",
              "matched": "the paper's events; each system's threshold rescaled at each cell so that it warns as often as the event occurs",
              "own": "truth and each system: their own percentile of 2022 at each cell (frequency-matched, annual)",
              "truth2022": "same threshold for all: percentile of the 1 km estimates of 2022 at each cell"}
OUT["land"] = {}
print("=" * 120, "\nLAND, truth = 1 km estimates (0.25 deg block means), 178 start days x 251,310 cells, equal issuance\n", "=" * 120, sep="")
for kind, kn in enumerate(FK):
    OUT["land"][kn] = {"definition": LAND_KINDS[kn]}
    BSk = sedi(boot_counts(fm["CD"][vf][:, kind], idf))                 # (n, tier, lead, system, region)
    for kt in (2, 1, 0):
        OUT["land"][kn][TT[kt]] = {}
        for g, rn in enumerate(FR):
            if g > 0 and not (kt == 2 and kn in ("ghap", "matched")):
                continue
            Ck = fm["C"][kind, kt]                                      # (lead, system, region, 4)
            OUT["land"][kn][TT[kt]][rn] = block(f"land | {kn} | {TT[kt]} | {rn}", Ck[:, jF, g], {"CAMS": Ck[:, jC, g], "Aurora": Ck[:, jA, g]},
                                                BSk[:, kt, :, jF, g], {"CAMS": BSk[:, kt, :, jC, g], "Aurora": BSk[:, kt, :, jA, g]},
                                                note=LAND_KINDS[kn] if g == 0 else "")
    del BSk
# ---------------------------------------------------------------- stations
st4 = D / f"extremes_stations_kinds4_{L}.npz"                         # with the kind 'matched' (extremes_stations_matched.py)
st = np.load(st4 if st4.exists() else D / f"extremes_stations_own_eqiss_{L}.npz"); SK = [str(s) for s in st["kinds"]]; SN = [str(s) for s in st["nets"]]
ids = block_index(st["CD"].shape[0])
ST_KINDS = {"ghap": "same threshold for all: local percentile of the 1 km estimates of 2017-2020 at the station pixel (as over land)",
            "station": "same threshold for all: the station's own percentile of the measurements of 2022",
            "own": "measurements and each system: their own percentile at each station over the same station-days (frequency-matched)",
            "matched": "the events of 'ghap'; each system's threshold rescaled at each station so that it warns as often as the event occurs"}
OUT["stations"] = {}
print("\n" + "=" * 120, "\nSTATIONS of the six official networks, truth = measurements, equal issuance (as submitted)\n", "=" * 120, sep="")
for kind, kn in enumerate(SK):
    OUT["stations"][kn] = {"definition": ST_KINDS[kn], "stations_1day": int(st["NST"][kind, 0, -1])}
    BSk = sedi(boot_counts(st["CD"][:, kind], ids))                     # (n, tier, lead, system, net)
    for kt in (2, 1, 0):
        OUT["stations"][kn][TT[kt]] = {}
        for n_, nn in enumerate(SN):
            if nn != "all" and kt != 2:
                continue
            Ck = st["C"][kind, kt]                                      # (lead, system, net, 4)
            if nn != "all":                                             # networks: numbers only, no print
                import io, contextlib
                with contextlib.redirect_stdout(io.StringIO()):
                    OUT["stations"][kn][TT[kt]][nn] = block(nn, Ck[:, 0, n_], {"CAMS": Ck[:, 1, n_], "Aurora": Ck[:, 2, n_]},
                                                            BSk[:, kt, :, 0, n_], {"CAMS": BSk[:, kt, :, 1, n_], "Aurora": BSk[:, kt, :, 2, n_]})
                continue
            OUT["stations"][kn][TT[kt]][nn] = block(f"stations | {kn} | {TT[kt]} | all networks, {int(Ck[0, 0, n_].sum())} station-days at 1 day",
                                                    Ck[:, 0, n_], {"CAMS": Ck[:, 1, n_], "Aurora": Ck[:, 2, n_]},
                                                    BSk[:, kt, :, 0, n_], {"CAMS": BSk[:, kt, :, 1, n_], "Aurora": BSk[:, kt, :, 2, n_]}, note=ST_KINDS[kn])
    del BSk
# ---------------------------------------------------------------- land r / bias with the mean of the truth
cg = np.load(D / f"compare_ghap_global_{L}_d1.npz"); S = cg["S"]; SG = [str(s) for s in cg["systems"]]
n = S[0, :, 0, 0]; my = S[0, :, 0, 5] / n
OUT["land_truth_mean_by_lead"] = r3(my)
print("\nmean of the 1 km estimates on the scored cells and days, by lead:", np.round(my, 3))
for nm in ("CAMS", "Aurora", L):
    k = SG.index(nm); nn = S[0, :, k, 0]
    with np.errstate(all="ignore"):
        b = S[0, :, k, 1] / nn
    print(f"  {nm:9s} bias {np.round(b, 3)}  = {np.round(100 * b / my, 1)} % of the mean")
    OUT[f"land_bias_percent_{nm}"] = r3(100 * b / my)
(D / "derived.json").write_text(json.dumps(OUT, indent=1))
print("\nsaved", D / "derived.json")
