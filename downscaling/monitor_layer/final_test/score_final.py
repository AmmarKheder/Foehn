#!/usr/bin/env python3
"""FROZEN scorer of the final test of the monitor-fed layer on the autumn 2026 record (draft Addendum 8 of A11).

    python -I -B score_final.py --tests     known-answer tests on synthetic truths: unit cases of the class rule, and the
                                            decision path itself (run_decision) end to end with the ft_lib paths redirected
                                            to a scratch folder; no file of observations can be opened
    python -I -B score_final.py --hashes    SHA-256 of this file (the content of FILED) and of its frozen dependencies
    python -I -B score_final.py             THE DECISION: refuses (exit 3) unless the file FILED exists and holds the
                                            SHA-256 of this file, and (exit 4) if result/TRUTH_READ exists (truth read before)

ONE READING of the held-out monitors (review finding 1). Every check that needs no observation runs first (checks 1-4 and
6 below); if one fails, the class is INVALID, result.json says so and the truth file is NOT opened (nothing was read: the
run may be completed with the same frozen code and the decision started again; any change of a frozen file is a new
addendum). Otherwise result/TRUTH_READ is created with O_CREAT | O_EXCL (SHA-256 of this file and of forecasts.npz, time,
host) immediately before the truth is opened, and every later run is refused, whatever happened after (a crash after
TRUTH_READ is a reading: the main session records it, no silent re-run). If check 5 fails after the reading, the class
INVALID and the failing check are written, and no table.

Truth: the record's own station-day table (realtime_1km/replay/score_cams3h/work/pairs.npz, the observations that scored the
paper's real-time record: UTC daily means of the OpenAQ monitors, valid days to 2026-10-21 from the 2026 station archive,
later from the re-read archive). Sample (review finding 11): the record's PRIMARY rows (observation, km, rt04, cams, clim and
bcsd finite, issue days from 2026-09-06) of the HELD-OUT monitors by the place rule of Addendum 3 (catalogue monitors inside
the 188 held-out 2-degree cells of split_v1) on the 48 issue days 2026-09-06..09-20, 2026-10-10..11-11, on which every arm
of this test is finite (identical station-days for all arms).
Arms:
  mfed   THE MONITOR-FED SYSTEM (primary): member m = M's member m (monitors of day d - 1, lag L) where an assimilated
         monitor with a valid value on day d - 1 lies within R km of the monitor's pixel, the record's station-free member
         m elsewhere (R and the checkpoint: data/selection.json, RULE S, 2021 validation only);
  meve   M everywhere (monitors of day d - 1); msens  the blend with the monitors of day d - 2 (lead day 5: station-free);
  fz     DIAGNOSTIC (review finding 4, not in the decision): pmG_final WITHOUT the fitted prior (free0 of infer_m.py: the
         base noise of m1) within R, the record's member elsewhere: mfed vs fz = the monitors (and M's network) at matched
         noise and no prior; fz vs km = the effect of the fitted prior P3 at the same pixels;
  km     the station-free chain of the record (points/<d>.npz km); cams  the CAMS forecast of the same 00 UTC cycle, mean of
         its eight 3-hourly steps; clim  the 1 km climatology. Every arm floored at 0, as the record's build_pairs.py.
Scores (the record's library realtime_1km/replay/score/sc_lib.py and score.py, imported unchanged, SHA-256 asserted): RMSE of
the 8-member mean, fair CRPS, per-monitor temporal correlation (inside each period first, then the record's T04 over all
issue days), SEDI (T07: above 37.5 and 75 ug m-3 and the monitors' own 95th / 99th percentiles), by lead day 1-5, by region,
by distance from the monitor's pixel to the nearest assimilated monitor with a value on day d - 1 (0-10, 10-50, 50-150,
> 150 km), and for the held-out monitors with no assimilated monitor within 10 km (Addendum 3); 95 % intervals by the
record's two-way block bootstrap (2-degree cells x calendar weeks of the valid day, 2,000 resamples, seed 20261119; 400 for
the temporal statistics).

DECISION (fixed before the test; RMSE change of mfed against km, pooled over the held-out station-days, with the bootstrap
interval; review finding 3: the literal reading of "not worse with an interval excluding zero"):
  INVALID  any check below failed;
  SUCCESS  lead day 1: change < 0 with the interval entirely below 0, AND lead days 2-5 pooled: change < 0 with the interval
           entirely below 0;
  PARTIAL  lead day 1 as SUCCESS, lead days 2-5 pooled NOT shown better (interval reaching 0 or above): the layer is proposed
           for lead day 1 only;
  FAIL     otherwise (lead day 1 not shown better).
Checks that make the class INVALID. Before the truth: (1) the frozen files (DEPS) unchanged, R of selection.json = R_KM;
(2) forecasts.npz: the 48 days, all done; every day's provenance: the SHA-256 of DEPS, of the selected checkpoint of M, of
pmG_final's checkpoint and prior, arms free/free0/m1/m2, seed 0, 10 Euler steps, 8 members, 11,278 points, guard 10 / 1000 /
3, source days d - 1 and d - 2 with the inputs' monitor counts, lead days made 1-5 (2-5 for 2026-10-10); members finite
exactly on the made lead days (m2: 1-4) and NaN elsewhere; free0, m1 and m2 share the base noise (equal truncation counts);
(3) the station-free arm recomputed by infer_m.py equals the record's km on EVERY made lead day (max relative difference
<= 1e-3, r >= 0.999999, same missing values); (4) guard: member points of m1, m2 at the state bound or the member cap <= 1e-4
of all, base-noise truncations <= 0.6 %; (6) the network's distances (d - 1 and d - 2) agree with the inputs' (class identical
on >= 99.9 % of the monitor-days). After the truth: (5) the truth reader reproduces the record (1,819,647 station-days, RMSE
16.4784 for km and 19.2434 for the CAMS forecast, +-5e-4) and the record's km at the held-out rows equals points/<d>.npz bit
for bit (no table if it fails); (7) lead day 1 sample: >= 5,000 station-days in >= 20 cells.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FT = Path(__file__).resolve().parent
sys.path.insert(0, str(FT))
import ft_lib as F  # noqa: E402

sys.path.insert(0, str(F.R / "score"))
import sc_lib as L  # noqa: E402
import score as S  # noqa: E402
from sc_lib import REGIONS, Sample, fci, md_table, pct, rel_mean, rel_rmse  # noqa: E402

B, BT, SEED = 2000, 400, 20261119
R_KM = 150.0
ARMS = ["mfed", "km", "meve", "msens", "fz", "cams", "clim"]
KINDS = {"mfed": "ens", "km": "ens", "meve": "ens", "msens": "ens", "fz": "ens", "cams": "det", "clim": "det"}
LABEL = {"mfed": "monitor-fed (blend)", "km": "station-free (record)", "meve": "M everywhere", "msens": "monitor-fed, monitors of d-2",
         "fz": "station-free net, no fitted prior (<= R)", "cams": "CAMS forecast", "clim": "1 km climatology"}
PAIRS = [("mfed", "km"), ("meve", "km"), ("msens", "km"), ("mfed", "fz"), ("fz", "km"), ("mfed", "cams"), ("km", "cams"),
         ("mfed", "clim"), ("km", "clim")]
DCLS = [c[0] for c in F.DIST_CLASSES]
S.LABEL.update(LABEL)                                   # table labels of the record's functions (in memory; score.py unchanged)
TOL = dict(identity_max_rel=1e-3, identity_min_r=0.999999, guard_bound_cap_share=1e-4, guard_trunc_share=0.006,
           record_n=1819647, record_rmse_km=16.4784, record_rmse_cams=19.2434, record_tol=5e-4, dist_agree=0.999,
           min_l1_station_days=5000, min_l1_cells=20)
# frozen dependencies: path -> SHA-256 (filled when the code was frozen; --hashes prints the current values)
DEPS = {
    "realtime_1km/replay/score/sc_lib.py": "03ddc9ea3597ebe05974c7d4391f279c4ec082defa3e4341a3f76394ede4d9e9",
    "realtime_1km/replay/score/score.py": "2a5cda3740a203bc5421e27a1095fe17ae8b9962a93044f62d7d781d4c1b3317",
    "a11/stageM/final_test/ft_lib.py": "ad729fb44f59c622ab22825c0c34c6fc7021ef93dcac6216c396d06696fde40f",
    "a11/stageM/final_test/infer_m.py": "e667c5d6e69928e29507617e51b3fc100e9cd55c5b0530401d9455d666c4aeb1",
    "a11/stageM/final_test/build_inputs.py": "f3fb582a633addfcd5ded26f3df116b890277370a45fb3f46fc238d4c9157081",
    "a11/stageM/final_test/select_2021.py": "496a1e096b4eab33dcc334b3e8d0784e8a6190a365505f2792088aa25da2f137",
    "a11/stageM/final_test/data/selection.json": "b401972f45de21f7dbe5bc292f9175627517dd45b7bc4c2d7896ed5fb0498e52",
    "a11/stageM/final_test/data/inputs.npz": "00dc650aa9dfd5e5897c870b476ba12824ab50acbb8cf8a8735901dca8e00eae",
}
PROV_SHA = {"ft_lib": "a11/stageM/final_test/ft_lib.py", "infer_m": "a11/stageM/final_test/infer_m.py",
            "inputs": "a11/stageM/final_test/data/inputs.npz", "selection": "a11/stageM/final_test/data/selection.json"}
# the station-free system of the record (review finding 2): pmG_final best.ckpt and its fitted prior (runs_ghap/pmG_final)
FREE_SHA = {"ckpt_free": "6ff79794bc846086dd0ceffd4abc9383dd317687d898a6af7ebc3349fe66544a",
            "prior": "edb59d16d0c8816f5c794a0c9625e33b4f63c0e756a2a13dcae1ef91c9af342a"}
# the frozen configuration of the full run (infer_m.py FULL_RUN), as its provenance records it
EXPECT = dict(arms=["free", "free0", "m1", "m2"], seed=0, ode_steps=10, members=8, points_per_member=11278, pixels=11278,
              guard=dict(z_bound=10.0, pmax_ug=1000.0, z0_trunc=3.0), step_free=22000, prior_m=None, prior_free0=None)
MEANING = {"SUCCESS": "lead day 1 and lead days 2-5 pooled both better than the station-free chain (intervals entirely below 0)",
           "PARTIAL": "lead day 1 better (interval entirely below 0), lead days 2-5 pooled not shown better: the layer is proposed "
                      "for lead day 1 only",
           "FAIL": "lead day 1 not shown better than the station-free chain: the layer is not proposed",
           "INVALID": "a check failed: no claim of any kind"}


# ---- decision ------------------------------------------------------------------------------------------------------------
def pooled(smp, a, b, leads):
    """relative RMSE change (%) of arm a against arm b over the given lead days, pooled, with the two-way block bootstrap"""
    D = smp.D
    fn = lambda x: (D.m[x][smp.rows] - D.obs[smp.rows]) ** 2
    pa, Ra = smp.q(f"rmse_{a}", lambda: fn(a))
    pb, Rb = smp.q(f"rmse_{b}", lambda: fn(b))
    pn, _ = smp.q("n", lambda: np.ones(smp.n))
    li = [x - 1 for x in leads]
    n = float(pn[li].sum())
    lr = smp.lead + 1
    cells = int(np.unique(smp.ci[np.isin(lr, leads)]).size)
    return dict(est=float(rel_rmse(pa[li].sum(), pb[li].sum())), ci=pct(rel_rmse(Ra[:, li].sum((1, 2)), Rb[:, li].sum((1, 2)))),
                n=int(n), cells=cells, rmse_a=float(np.sqrt(pa[li].sum() / n)) if n else None,
                rmse_b=float(np.sqrt(pb[li].sum() / n)) if n else None)


def shown_better(r):
    """a change shown negative: point change < 0 and the 95 % interval entirely below 0"""
    return r["n"] > 0 and r["est"] < 0 and r["ci"][1] < 0


def classify(l1, l25, invalid):
    if invalid:
        return "INVALID"
    if not shown_better(l1):
        return "FAIL"
    return "SUCCESS" if shown_better(l25) else "PARTIAL"


def decide(D, mask, invalid=()):
    smp = Sample(D, mask, B=B, seed=SEED)
    l1, l25 = pooled(smp, "mfed", "km", [1]), pooled(smp, "mfed", "km", [2, 3, 4, 5])
    inv = list(invalid)
    if l1["n"] < TOL["min_l1_station_days"] or l1["cells"] < TOL["min_l1_cells"]:
        inv.append(f"lead day 1 sample too small ({l1['n']} station-days, {l1['cells']} cells)")
    return dict(cls=classify(l1, l25, inv), lead1=l1, lead2_5=l25, invalid=inv, sample=smp.describe()), smp


# ---- the station-day table of the test ------------------------------------------------------------------------------
def write_pairs(path, issue_days, issue, lead, mon, vday, obs, cat, arms):
    """the record's pairs.npz format (sc_lib.Data), arms: {name: (rows, 8) or (rows,)}"""
    lat, lon = cat["lat"], cat["lon"]
    cell = (np.floor(lat / 2).astype(np.int64) + 90) * 1000 + (np.floor(lon / 2).astype(np.int64) + 180)
    save = dict(issue_days=np.array(issue_days), arms=np.array(list(arms)), kinds=np.array([KINDS.get(a, "ens" if arms[a].ndim == 2 else "det") for a in arms]),
                id=cat["id"], lat=lat, lon=lon, region=cat["region"], cell=cell, issue=issue.astype(np.int16), lead=lead.astype(np.int8),
                mon=mon.astype(np.int32), vday=vday.astype(np.int16), obs=obs.astype(np.float32), src=np.zeros(obs.size, np.int8))
    for a, x in arms.items():
        x = np.maximum(x.astype(np.float32), 0.0)
        if x.ndim == 2:
            save[f"e_{a}"], save[f"m_{a}"] = x, x.mean(1)
        else:
            save[f"m_{a}"] = x
    F.save_npz(path, **save)
    return L.Data(path)


def tables(D, mask, dcls, far10, Bt=BT, B_=B):
    """every table of the test (markdown, json); the record's functions"""
    md, js = "", {}
    smp = Sample(D, mask, B=B_, seed=SEED)
    intro = lambda s: f"Sample: {s.describe()}. Time blocks: {'; '.join(s.blocks)}.\n\n"
    res = S.contrasts(smp, PAIRS, cells_only=False)
    js["contrasts"] = res
    md += "## Relative change of RMSE (8-member mean) and fair CRPS, % (negative = first arm better)\n\n" + intro(smp)
    md += ("'station-free net, no fitted prior' (fz) is a diagnostic, not in the decision: monitor-fed vs fz = the monitors and M's "
           "network at matched noise without prior; fz vs station-free (record) = the fitted prior P3 at the same pixels.\n\n")
    md += "RMSE:\n\n" + S.md_contrasts(smp, res, PAIRS, "rmse") + "Fair CRPS (absolute error for the deterministic arms):\n\n" + \
        S.md_contrasts(smp, res, PAIRS, "crps", grid=False)
    sc, scj = S.scores(smp, ARMS)
    js["scores"] = scj
    md += "## Scores of the ensemble mean\n\n" + S.md_scores(smp, sc, ARMS)
    crps = {}
    for a in ARMS:
        c = D.crps(a)[smp.rows]
        crps[a] = {"all": float(c.mean()), **{str(Ld): float(c[smp.lead == Ld - 1].mean()) for Ld in range(1, 6)}}
    js["fair_crps"] = crps
    md += "Mean fair CRPS (ug m-3; absolute error for cams, clim):\n\n" + md_table(
        ["arm", "all", "lead 1", "lead 2", "lead 3", "lead 4", "lead 5"],
        [[LABEL[a]] + [f"{crps[a][k]:.3f}" for k in ["all", "1", "2", "3", "4", "5"]] for a in ARMS]) + "\n"
    # by distance class
    sd = Sample(D, mask, group=dcls, gnames=DCLS, B=B_, seed=SEED)
    dp = [("mfed", "km"), ("meve", "km"), ("mfed", "fz"), ("fz", "km"), ("mfed", "cams"), ("km", "cams")]
    rd = S.contrasts(sd, dp, metrics=("rmse", "crps"), cells_only=False)
    js["by_distance"] = rd
    gl = [(g, g) for g in DCLS]
    md += ("## By distance from the monitor's pixel to the nearest assimilated monitor with a value on day d - 1\n\n" + intro(sd) +
           "RMSE:\n\n" + S.md_contrasts(sd, rd, dp, "rmse", by_lead=False, grid=True, groups=gl) +
           "Fair CRPS:\n\n" + S.md_contrasts(sd, rd, dp, "crps", by_lead=False, grid=False, groups=gl))
    # Addendum 3: held-out monitors with no assimilated monitor within 10 km
    sf = Sample(D, mask & far10, B=B_, seed=SEED)
    rf = S.contrasts(sf, dp, metrics=("rmse", "crps"), cells_only=False)
    js["no_assimilated_within_10km"] = dict(contrasts=rf, lead1=pooled(sf, "mfed", "km", [1]), lead2_5=pooled(sf, "mfed", "km", [2, 3, 4, 5]))
    md += ("## Held-out monitors with no assimilated monitor within 10 km (Addendum 3)\n\n" + intro(sf) +
           "RMSE:\n\n" + S.md_contrasts(sf, rf, dp, "rmse", grid=False) + "Fair CRPS:\n\n" + S.md_contrasts(sf, rf, dp, "crps", grid=False))
    # day-to-day signal (review finding 12): inside each period first (one season each), then the record's T04 over all days
    md += "## Day-to-day signal: per (monitor, lead day) temporal correlation inside one period (read these first)\n\n"
    for p in ("sep", "octnov"):
        sp = Sample(D, mask & D.issues(*S.PERIODS[p][:2]), B=B_, seed=SEED)
        if sp.n >= 1000:
            tp = S.temporal(sp, ARMS, ["mfed", "km"], Bt=Bt)
            js[f"temporal_{p}"] = tp
            md += f"### Inside one period: {S.PERIODS[p][2]}\n\n" + intro(sp) + S.md_temporal(sp, tp, ARMS, ["mfed", "km"])
    t = S.temporal(smp, ARMS, ["mfed", "km"], Bt=Bt)
    js["temporal"] = t
    md += ("## The record's T04 over all 48 issue days (raw series across September and October-November: it mixes the seasonal "
           "step between the two periods, not only day-to-day anomalies)\n\n" + intro(smp) + S.md_temporal(smp, t, ARMS, ["mfed", "km"]))
    r7 = S.sedi_table(smp, ARMS, ["mfed"])
    js["sedi"] = r7
    md += "## Extremes: SEDI (the record's T07)\n\n" + intro(smp) + S.md_sedi(smp, r7, ARMS, ["mfed"])
    return md, js


# ---- checks of the decision mode ----------------------------------------------------------------------------------------
def check_filed(this=None, filed=None):
    this = Path(this or __file__)
    filed = Path(filed or F.FILED)
    if not filed.exists():
        return False, f"{filed} does not exist: the addendum is not filed"
    want = F.sha256_file(this)
    got = filed.read_text().split()
    if not got or got[0] != want:
        return False, f"{filed} holds {got[:1]}, not the SHA-256 of {this.name} ({want})"
    return True, want


def dep_hashes():
    return {k: F.sha256_file(F.P / k) for k in DEPS}


def made_mask(days):
    """(48, 5) lead days that must be made: 1-5, 2-5 for 2026-10-10 (no lead day 1 in the record)"""
    m = np.ones((len(days), 5), bool)
    m[[i for i, d in enumerate(days) if d == "2026-10-10"], 0] = False
    return m


def check_forecasts(days, sel, I):
    """every check of forecasts.npz that needs no observation (checks 2, 3, 4, 6); returns (invalid list, info or None)"""
    inv = []
    z = np.load(F.FORECASTS)
    need = {"days", "done", "prov", "free0", "m1", "m2", "dmin1", "dmin2", "nmon"}
    if not need <= set(z.files):
        return [f"forecasts.npz lacks {sorted(need - set(z.files))}"], None
    if [str(x) for x in z["days"]] != days:
        return ["forecasts.npz days differ from the record's 48"], None
    n = int(I["cat_id"].size)
    for k in ("free0", "m1", "m2"):
        if z[k].shape != (len(days), 5, 8, n):
            return [f"forecasts.npz {k} has shape {z[k].shape}"], None
    if not z["done"].all():
        inv.append(f"forecasts.npz: {int((~z['done']).sum())} days not done")
    cnt = np.diff(I["src_off"])
    want_mask = made_mask(days)
    nb = ntr = npts = 0
    for i, s in enumerate(z["prov"]):
        d, dd = days[i], dt.date.fromisoformat(days[i])
        want = [Ld + 1 for Ld in range(5) if want_mask[i, Ld]]
        try:
            p = json.loads(str(s))
        except ValueError:
            inv.append(f"{d}: no provenance")
            continue
        bad = []
        if p.get("issue") != d:
            bad.append(f"issue {p.get('issue')}")
        sha = p.get("sha256", {})
        for k, path in PROV_SHA.items():
            if sha.get(k) != DEPS[path]:
                bad.append(f"sha256 of {k} differs from the frozen file")
        if sha.get("ckpt_m") != sel["checkpoint"]["sha256"] or p.get("step_m") != sel["checkpoint"]["step"]:
            bad.append("checkpoint of M is not the selected one")
        for k, v in FREE_SHA.items():
            if sha.get(k) != v:
                bad.append(f"sha256 of {k} is not pmG_final's")
        for k, v in EXPECT.items():
            if p.get(k, "missing") != v:
                bad.append(f"{k} = {p.get(k, 'missing')!r} (frozen: {v!r})")
        if p.get("leads_made") != want:
            bad.append(f"lead days made {p.get('leads_made')} (expected {want})")
        if p.get("src_days") != {"1": str(dd - dt.timedelta(days=1)), "2": str(dd - dt.timedelta(days=2))}:
            bad.append(f"source days {p.get('src_days')}")
        nm = [int(cnt[I["issue_src"][i, 0]]), int(cnt[I["issue_src"][i, 1]])]
        if p.get("n_monitors") != nm or z["nmon"][i].tolist() != nm:
            bad.append(f"monitor counts {p.get('n_monitors')} / {z['nmon'][i].tolist()} (inputs: {nm})")
        idn = p.get("identity", {})
        if set(idn) != {str(Ld) for Ld in want}:
            bad.append(f"station-free identity on lead days {sorted(idn)} (expected every made lead day)")
        for Ld, v in idn.items():
            if not (v.get("n", 0) > 0 and v.get("same_nan") and v.get("max_rel", np.inf) <= TOL["identity_max_rel"]
                    and v.get("r", 0.0) >= TOL["identity_min_r"]):
                bad.append(f"lead {Ld}: station-free identity failed {v}")
        gh = p.get("guard_hits", {})
        if set(gh) != {str(Ld) for Ld in want}:
            bad.append(f"guard hits on lead days {sorted(gh)}")
        for Ld, hl in gh.items():
            if set(hl) != set(EXPECT["arms"]):
                bad.append(f"lead {Ld}: guard hits of arms {sorted(hl)}")
                continue
            tr = hl["free0"]["z0_trunc"], hl["m1"]["z0_trunc"], hl["m2"]["z0_trunc"]
            if tr[0] != tr[1] or (int(Ld) <= 4 and tr[2] != tr[1]):
                bad.append(f"lead {Ld}: free0 / m1 / m2 do not share the base noise (truncations {tr})")
            for arm in ("m1", "m2"):
                if arm == "m2" and int(Ld) == 5:
                    continue
                nb += hl[arm]["at_bound"] + hl[arm]["at_pmax"]
                ntr += hl[arm]["z0_trunc"]
                npts += EXPECT["points_per_member"] * EXPECT["members"]          # member points of one arm and lead day
        if bad:
            inv.append(f"{d}: " + "; ".join(bad))
    for k in ("free0", "m1", "m2"):
        mk = want_mask.copy()
        if k == "m2":
            mk[:, 4] = False
        x = z[k]
        if not np.isfinite(x[mk]).all():
            inv.append(f"{k}: non-finite members on made lead days")
        if not np.isnan(x[~mk]).all():
            inv.append(f"{k}: values on lead days not made")
    if not npts or nb / npts > TOL["guard_bound_cap_share"] or ntr / npts > TOL["guard_trunc_share"]:
        inv.append(f"guard: bound/cap share {nb / max(npts, 1):.2e}, truncation share {ntr / max(npts, 1):.4f} ({npts} member points)")
    agree = {}
    for k, col in (("dmin1", 0), ("dmin2", 1)):
        ref = I["dday"][I["issue_src"][:, col]]
        agree[k] = float((F.dist_class(z[k]) == F.dist_class(ref)).mean())
        if agree[k] < TOL["dist_agree"]:
            inv.append(f"distance classes ({k}) of the network and of the inputs agree on {agree[k]:.5f} only")
    return inv, dict(z=z, guard=dict(bound_or_cap=nb, trunc=ntr, member_points=npts), distance_agreement=agree)


def log_attempt(entry):
    p = F.RESULT / "attempts.jsonl"
    old = p.read_text() if p.exists() else ""
    F.save_text(p, old + json.dumps(entry, default=F._jdef) + "\n")


def take_truth_lock(stamp):
    """result/TRUTH_READ, created exclusively: False if it exists (the truth was read before)"""
    lock = F.RESULT / "TRUTH_READ"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o660)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(stamp) + "\n")
        f.flush()
        os.fsync(f.fileno())
    dfd = os.open(F.RESULT, os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)
    return True


def run_decision(fast_tables=False):
    ok, msg = check_filed()
    if not ok:
        print(f"[score_final] REFUSED: {msg}")
        sys.exit(3)
    lock = F.RESULT / "TRUTH_READ"
    if lock.exists():
        print(f"[score_final] REFUSED: the held-out monitors were read before ({lock}: {lock.read_text().strip()[:400]}); "
              "a second reading is not allowed")
        sys.exit(4)
    t0 = time.time()
    me, deps = F.sha256_file(__file__), dep_hashes()
    now = lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    # ---- checks 1-4 and 6: no observation ----------------------------------------------------------------------------
    inv = [f"frozen file changed: {k}" for k, v in deps.items() if v != DEPS[k]]
    days = F.issue_days()
    sel = F.load_selection()
    if float(sel["R_km"]) != R_KM:
        inv.append(f"R of selection.json {sel['R_km']} != {R_KM}")
    with np.load(F.INPUTS) as z_:
        I = {k: z_[k] for k in ("issue", "issue_src", "src_off", "dday", "dstatic", "cat_id", "cat_heldout_place")}
    if [str(x) for x in I["issue"]] != days:
        inv.append("inputs.npz issue days differ from the record's 48")
    fsha = F.sha256_file(F.FORECASTS) if F.FORECASTS.exists() else None
    fc = None
    if fsha is None:
        inv.append(f"{F.FORECASTS} missing")
    elif not inv:
        finv, fc = check_forecasts(days, sel, I)
        inv += finv
    base = dict(scorer_sha256=me, forecasts_sha256=fsha, deps=deps, R_km=R_KM, checkpoint=sel["checkpoint"])
    F.RESULT.mkdir(parents=True, exist_ok=True)
    if inv:
        out = dict(cls="INVALID", stage="before the truth: no held-out value was read", invalid=inv, **base, written_utc=now())
        F.save_json(F.RESULT / "result.json", out)
        log_attempt(dict(utc=out["written_utc"], cls="INVALID", stage="before the truth", n_invalid=len(inv), forecasts_sha256=fsha))
        print(f"[score_final] class INVALID before the truth (nothing read): {inv[:20]}")
        return
    # ---- the one reading of the truth ----------------------------------------------------------------------------------
    stamp = dict(scorer_sha256=me, forecasts_sha256=fsha, utc=now(), host=os.uname().nodename, pid=os.getpid(),
                 job=os.environ.get("SLURM_JOB_ID"))
    if not take_truth_lock(stamp):
        print(f"[score_final] REFUSED: {lock} exists: a second reading is not allowed")
        sys.exit(4)
    log_attempt(dict(utc=stamp["utc"], stage="truth opened", forecasts_sha256=fsha))
    held, dday, isrc, dstatic = I["cat_heldout_place"], I["dday"], I["issue_src"], I["dstatic"]
    z = fc["z"]
    cat = np.load(F.CATALOGUE)
    catd = {k: cat[k] for k in ("id", "lat", "lon", "region")}
    catd["lat"], catd["lon"] = catd["lat"].astype(np.float64), catd["lon"].astype(np.float64)
    T = np.load(F.RECORD_PAIRS)
    rdays = [str(x) for x in T["issue_days"]]
    issue, lead, mon, obs, vday = T["issue"].astype(np.int64), T["lead"].astype(np.int64), T["mon"].astype(np.int64), T["obs"].astype(np.float64), T["vday"]
    iday = np.array(rdays)[issue]
    clean = iday >= "2026-09-06"
    fin = np.isfinite(obs)
    for a in ("km", "rt04", "cams", "clim", "bcsd"):
        fin &= np.isfinite(T[f"m_{a}"])
    rs = clean & fin                                          # the record's primary sample
    rk = float(np.sqrt(np.mean((T["m_km"][rs] - obs[rs]) ** 2)))
    rc = float(np.sqrt(np.mean((T["m_cams"][rs] - obs[rs]) ** 2)))
    rec = dict(n=int(rs.sum()), rmse_km=rk, rmse_cams=rc)
    g5 = []
    if rec["n"] != TOL["record_n"] or abs(rk - TOL["record_rmse_km"]) > TOL["record_tol"] or abs(rc - TOL["record_rmse_cams"]) > TOL["record_tol"]:
        g5.append(f"the truth reader does not reproduce the record: {rec}")
    di = {d: i for i, d in enumerate(days)}
    in_test = np.isin(iday, days) & held[mon]
    sel_rows = np.flatnonzero(in_test & rs)
    old_rule = in_test & np.isfinite(obs) & np.isfinite(T["m_km"]) & np.isfinite(T["m_cams"]) & np.isfinite(T["m_clim"])
    outside_primary = int((old_rule & ~rs).sum())
    ii = np.array([di[x] for x in iday[sel_rows]], np.int64)
    Ld, k = lead[sel_rows], mon[sel_rows]
    km_pts = np.full((sel_rows.size, 8), np.nan, np.float32)
    for i, d in enumerate(days):
        m = ii == i
        if m.any():
            km_pts[m] = np.maximum(np.load(F.POINTS / f"{d}.npz")["km"][Ld[m] - 1, :, k[m]], 0.0)
    ekm = T["e_km"][sel_rows]
    if not ((np.isnan(ekm) == np.isnan(km_pts)).all() and np.array_equal(ekm[np.isfinite(ekm)], km_pts[np.isfinite(km_pts)])):
        g5.append("the record's km at the held-out rows differs from points/<d>.npz")
    if g5:
        out = dict(cls="INVALID", stage="after the truth: check 5 failed, no table written", invalid=g5, record_identity=rec, **base,
                   written_utc=now())
        F.save_json(F.RESULT / "result.json", out)
        log_attempt(dict(utc=out["written_utc"], cls="INVALID", stage="after the truth (check 5)", forecasts_sha256=fsha))
        print(f"[score_final] class INVALID (check 5, no table): {g5}")
        return
    m1 = z["m1"][ii, Ld - 1, :, k]
    m2 = z["m2"][ii, Ld - 1, :, k]
    f0 = z["free0"][ii, Ld - 1, :, k]
    dist1 = dday[isrc[ii, 0], k]
    dist2 = dday[isrc[ii, 1], k]
    near1 = (dist1 <= R_KM)[:, None]
    near2 = ((dist2 <= R_KM) & (Ld <= 4))[:, None]
    arms = dict(mfed=np.where(near1, m1, km_pts), km=km_pts, meve=m1, msens=np.where(near2, m2, km_pts), fz=np.where(near1, f0, km_pts),
                cams=T["m_cams"][sel_rows], clim=T["m_clim"][sel_rows])
    D = write_pairs(F.RESULT / "pairs_heldout.npz", days, ii, Ld, k, vday[sel_rows], obs[sel_rows], catd, arms)
    mask = D.finite(ARMS) & D.issues(days[0], days[-1])
    lost = {a: int((D.finite(["km", "cams", "clim"]) & ~np.isfinite(D.m[a])).sum()) for a in ("mfed", "meve", "msens", "fz")}
    dcls = F.dist_class(dist1)
    far10 = dstatic[k] > 10.0
    dec, smp = decide(D, mask)
    md, js = tables(D, mask, dcls, far10, **(dict(Bt=50, B_=200) if fast_tables else {}))
    out = dict(cls=dec["cls"], meaning=MEANING[dec["cls"]], decision=dec, record_identity=rec, guard=fc["guard"],
               distance_agreement=fc["distance_agreement"], rows_lost_by_the_monitor_fed_arms=lost,
               rows_outside_the_record_primary_sample_excluded=outside_primary, tables=js, fast_tables=bool(fast_tables), **base,
               seconds=round(time.time() - t0, 1), written_utc=now())
    l1, l25 = dec["lead1"], dec["lead2_5"]
    st25 = "shown better" if shown_better(l25) else ("worse (interval entirely above 0)" if l25["ci"][0] > 0 else "not shown better (interval reaches 0)")
    head = (f"# Final test of the monitor-fed layer: class **{dec['cls']}**\n\n{MEANING[dec['cls']]}.\n\n"
            f"Lead day 1, RMSE of mfed vs km: {fci(l1)} % ({l1['n']:,} station-days, {l1['cells']} cells; "
            f"RMSE {l1['rmse_a']:.2f} vs {l1['rmse_b']:.2f}). Lead days 2-5 pooled: {fci(l25)} % ({l25['n']:,}; {st25}). "
            f"Invalidating checks: {dec['invalid'] or 'none'}.\n\n"
            "What it means: a reforecast test on the 48 issue days of the autumn record with archived monitor values (not forecasts "
            "issued live), at held-out monitors only; the monitor-fed layer acts near the assimilated monitors (R km) and is the "
            "station-free system elsewhere; the blind real-time claim of the paper stays that of the station-free system, whose numbers "
            "do not change. Both systems are reported side by side; 'M everywhere', the d - 2 arm and the no-prior diagnostic beside.\n\n")
    F.save_json(F.RESULT / "result.json", out)
    F.save_text(F.RESULT / "RESULT.md", head + md)
    log_attempt(dict(utc=out["written_utc"], cls=dec["cls"], stage="decision written", forecasts_sha256=fsha))
    print(head + f"[score_final] {time.time() - t0:.0f} s -> {F.RESULT}")


# ---- known-answer tests (synthetic truths only) ------------------------------------------------------------------------
def synthetic(seed=0, n_mon=1500):
    """a synthetic record: monitors in 2-degree cells of the eight regions, the 48 issue days x 5 lead days, an observed
    series with day-to-day structure, a station-free 8-member ensemble around it, a CAMS and a climatology arm"""
    rng = np.random.default_rng(seed)
    boxes = {"Europe": (35, 60, -10, 35), "East Asia": (20, 50, 100, 135), "South Asia": (8, 30, 68, 90), "North America": (25, 50, -125, -70),
             "Africa": (-30, 30, -15, 40), "South America": (-40, 5, -75, -40), "North": (61, 70, 10, 30), "Other": (-40, -15, 115, 150)}
    reg = rng.choice(list(boxes), n_mon)
    lat = np.array([rng.uniform(boxes[r][0], boxes[r][1]) for r in reg])
    lon = np.array([rng.uniform(boxes[r][2], boxes[r][3]) for r in reg])
    days = F.issue_days()
    d0 = dt.date(2026, 1, 1)
    vd = sorted({(dt.date.fromisoformat(d) + dt.timedelta(L)) for d in days for L in range(5)})
    vix = {v: i for i, v in enumerate(vd)}
    base = rng.normal(2.6, 0.5, n_mon)
    ar = np.zeros((len(vd), n_mon))
    for t in range(1, len(vd)):
        ar[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.35, n_mon)
    truth = np.exp(base[None] + ar)
    rows = []
    for i, d in enumerate(days):
        for Ld in range(1, 6):
            v = dt.date.fromisoformat(d) + dt.timedelta(Ld - 1)
            ok = rng.random(n_mon) < 0.7
            k = np.flatnonzero(ok)
            rows.append((np.full(k.size, i), np.full(k.size, Ld), k, np.full(k.size, (v - d0).days), truth[vix[v], k]))
    issue, lead, mon, vday, obs = (np.concatenate([r[j] for r in rows]) for j in range(5))
    nr = obs.size
    err = obs * (np.exp(rng.normal(0, 0.35, nr) + 0.05 * lead) - 1)
    km = (obs + err)[:, None] * np.exp(rng.normal(0, 0.2, (nr, 8)))
    cams = obs * np.exp(rng.normal(0.1, 0.5, nr))
    clim = np.exp(base[mon] + 0.15 * np.sin(vday / 20.0)) * 1.1          # the record's climatology varies with the day of year
    dist = rng.choice([5.0, 30.0, 100.0, 400.0], nr, p=[0.1, 0.25, 0.5, 0.15])
    cat = dict(id=np.array([f"syn:{j}" for j in range(n_mon)]), lat=lat, lon=lon, region=reg)
    return dict(days=days, issue=issue, lead=lead, mon=mon, vday=vday, obs=obs, km=km, cams=cams, clim=clim, dist=dist, cat=cat,
                far10=rng.random(nr) < 0.85)


def shrink(km, obs, f, where, spread=0.3, seed=1):
    """members moved toward (f < 1) or away from (f > 1) the observation at the rows `where`, by a factor drawn per row
    in f +- spread (so that the bootstrap interval is not degenerate)"""
    fr = f + np.random.default_rng(seed).uniform(-spread, spread, obs.size)
    x = obs[:, None] + fr[:, None] * (km - obs[:, None])
    return np.where(where[:, None], np.maximum(x, 0.0), km)


def unit_tests(tmp, res):
    """the class rule on a small synthetic record through decide() (and every table once)"""
    ok_all = True
    y = synthetic()
    L1, near = y["lead"] == 1, y["dist"] <= 50.0
    rows_cell = np.floor(y["cat"]["lat"][y["mon"]] / 2).astype(np.int64) * 1000 + np.floor(y["cat"]["lon"][y["mon"]] / 2).astype(np.int64)
    zc = np.random.default_rng(5).standard_normal(np.unique(rows_cell).size)[np.unique(rows_cell, return_inverse=True)[1]]
    small_loss = np.clip(1.03 + 0.3 * zc, 0.3, None)                  # a small, cell-heterogeneous loss at lead days 2-5
    x_small = y["obs"][:, None] + small_loss[:, None] * (y["km"] - y["obs"][:, None])
    cases = [("no gain (identical arms)", y["km"], "FAIL"),
             ("no gain on average (row factors 1 +- 0.3)", shrink(y["km"], y["obs"], 1.0, np.ones_like(L1)), "FAIL"),
             ("gain at lead day 1 only (lead days 2-5 unchanged)", shrink(y["km"], y["obs"], 0.6, L1), "PARTIAL"),
             ("gain at every lead day", shrink(y["km"], y["obs"], 0.6, np.ones_like(L1)), "SUCCESS"),
             ("gain near monitors only (<= 50 km), every lead day", shrink(y["km"], y["obs"], 0.6, near), "SUCCESS"),
             ("gain at lead day 1, loss at lead days 2-5", shrink(shrink(y["km"], y["obs"], 0.6, L1), y["obs"], 1.4, ~L1), "PARTIAL"),
             ("gain at lead day 1, small cell-heterogeneous loss at lead days 2-5",
              np.where(L1[:, None], shrink(y["km"], y["obs"], 0.6, L1), np.maximum(x_small, 0.0)), "PARTIAL"),
             ("loss at every lead day", shrink(y["km"], y["obs"], 1.3, np.ones_like(L1)), "FAIL"),
             ("no gain at lead day 1, gain at lead days 2-5", shrink(y["km"], y["obs"], 0.6, ~L1), "FAIL")]
    for name, mfed, want in cases:
        arms = dict(mfed=mfed, km=y["km"], meve=mfed, msens=y["km"], fz=y["km"], cams=y["cams"], clim=y["clim"])
        D = write_pairs(tmp / "pairs.npz", y["days"], y["issue"], y["lead"], y["mon"], y["vday"], y["obs"], y["cat"], arms)
        mask = D.finite(ARMS)
        dec, smp = decide(D, mask)
        extra, ok_near = {}, True
        if name.startswith("gain near"):
            sd = Sample(D, mask, group=F.dist_class(y["dist"]), gnames=DCLS, B=B, seed=SEED)
            rd = S.contrast(sd, "mfed", "km", "rmse")
            extra = {g: rd["all"][g]["est"] for g in DCLS if g in rd["all"]}
            ok_near = extra["50-150km"] == 0.0 and extra[">150km"] == 0.0 and extra["0-10km"] < -10 and extra["10-50km"] < -10
        ok = dec["cls"] == want and ok_near
        ok_all &= ok
        res.append(dict(case=f"unit: {name}", expected=want, got=dec["cls"], lead1=dec["lead1"], lead2_5=dec["lead2_5"], by_distance=extra, ok=bool(ok)))
        print(f"[{'OK' if ok else 'FAIL'}] unit: {name}: expected {want}, got {dec['cls']}; lead 1 {fci(dec['lead1'])}, lead 2-5 {fci(dec['lead2_5'])}"
              + (f"; by distance {', '.join(f'{g} {v:+.1f}' for g, v in extra.items())}" if extra else ""), flush=True)
    # INVALID overrides the numbers
    arms = dict(mfed=shrink(y["km"], y["obs"], 0.6, np.ones_like(L1)), km=y["km"], meve=y["km"], msens=y["km"], fz=y["km"], cams=y["cams"], clim=y["clim"])
    D = write_pairs(tmp / "pairs.npz", y["days"], y["issue"], y["lead"], y["mon"], y["vday"], y["obs"], y["cat"], arms)
    dec, _ = decide(D, D.finite(ARMS), invalid=["station-free identity failed (synthetic)"])
    ok = dec["cls"] == "INVALID"
    ok_all &= ok
    res.append(dict(case="unit: an invalidating check overrides a SUCCESS", expected="INVALID", got=dec["cls"], ok=bool(ok)))
    print(f"[{'OK' if ok else 'FAIL'}] unit: invalid check: expected INVALID, got {dec['cls']}", flush=True)
    # every table runs on the format, with fewer resamples
    md, js = tables(D, D.finite(ARMS), F.dist_class(y["dist"]), y["far10"], Bt=100, B_=400)
    ok = len(md) > 1000 and "SEDI" in md and np.isfinite(js["temporal"]["groups"]["all"]["mfed"]["r"]["est"]) and \
        md.index("Inside one period") < md.index("The record's T04")
    ok_all &= ok
    res.append(dict(case="unit: all tables run (per-period day-to-day tables before the record's T04)", ok=bool(ok), md_chars=len(md)))
    print(f"[{'OK' if ok else 'FAIL'}] unit: all tables run ({len(md):,} characters)", flush=True)
    # FILED: missing or wrong -> refusal; the right hash -> accepted
    f1 = tmp / "FILED"
    a, _ = check_filed(filed=f1)
    f1.write_text("0" * 64 + "\n")
    b, _ = check_filed(filed=f1)
    f1.write_text(F.sha256_file(__file__) + "\n")
    c, _ = check_filed(filed=f1)
    f1.unlink()
    ok = (not a) and (not b) and c
    ok_all &= ok
    res.append(dict(case="unit: FILED gate", ok=bool(ok), missing=a, wrong=b, right=c))
    print(f"[{'OK' if ok else 'FAIL'}] unit: FILED gate (missing -> refused, wrong hash -> refused, right hash -> accepted)", flush=True)
    return ok_all


def e2e_tests(tmp, res):
    """the UNCHANGED decision path (run_decision: checks, lock, truth reader, blend, decide, tables) on synthetic truths, the
    ft_lib paths RECORD_PAIRS, FORECASTS, FILED, RESULT redirected to tmp (review findings 1, 2, 3, 11; harness of
    review/scorer_e2e.py). The real inputs.npz, selection.json, catalogue and the record's forecast files points/<d>.npz are
    read (no observation); the synthetic truth is the record's km members plus noise scaled so that check 5 holds exactly."""
    saved = (F.RECORD_PAIRS, F.FORECASTS, F.FILED, F.RESULT)
    F.RECORD_PAIRS, F.FORECASTS, F.FILED, F.RESULT = tmp / "pairs.npz", tmp / "forecasts.npz", tmp / "FILED", tmp / "result"
    ok_all = True
    try:
        days = F.issue_days()
        sel = F.load_selection()
        with np.load(F.INPUTS) as z_:
            I = {k: z_[k] for k in ("cat_heldout_place", "dday", "issue_src", "src_off")}
        held, dday, isrc, cnt = I["cat_heldout_place"], I["dday"], I["issue_src"], np.diff(I["src_off"])
        n = held.size
        made = made_mask(days)
        rng = np.random.default_rng(20261121)
        KM = np.empty((len(days), 5, 8, n), np.float32)
        for i, d in enumerate(days):
            KM[i] = np.maximum(np.load(F.POINTS / f"{d}.npz")["km"], 0.0)
        ii, ll, kk = np.nonzero(np.isfinite(KM).all(2))
        NP = TOL["record_n"]
        perm = rng.permutation(ii.size)
        pick = np.concatenate([perm[:NP], perm[NP:NP + 60000]])
        extra = np.r_[np.zeros(NP, bool), np.ones(60000, bool)]       # rows outside the record's primary sample (rt04 NaN)
        o = np.lexsort((kk[pick], ll[pick], ii[pick]))
        pick, extra = pick[o], extra[o]
        issue, lead, mon = ii[pick], ll[pick] + 1, kk[pick]
        vday = np.array([(dt.date.fromisoformat(d) - dt.date(2026, 1, 1)).days for d in days])[issue] + lead - 1
        e_km = KM[issue, lead - 1, :, mon]
        m_km = e_km.mean(1)
        prim = ~extra
        e = rng.standard_normal(m_km.size) * (0.4 * m_km.astype(np.float64) + 4.0)
        obs = (m_km + e * TOL["record_rmse_km"] / np.sqrt(np.mean(e[prim] ** 2))).astype(np.float32)
        e2 = rng.standard_normal(m_km.size) * (0.4 * m_km.astype(np.float64) + 4.0)
        m_cams = (obs + e2 * TOL["record_rmse_cams"] / np.sqrt(np.mean(e2[prim] ** 2))).astype(np.float32)

        def write_record(drop_one=False):
            keep = np.ones(obs.size, bool)
            if drop_one:
                keep[np.flatnonzero(prim)[0]] = False                     # one primary row fewer: check 5 must fail
            F.save_npz(F.RECORD_PAIRS, issue_days=np.array(days), arms=np.array(["km", "rt04", "cams", "clim", "bcsd"]),
                       issue=issue[keep].astype(np.int16), lead=lead[keep].astype(np.int8), mon=mon[keep].astype(np.int32),
                       vday=vday[keep].astype(np.int16), obs=obs[keep], e_km=e_km[keep], m_km=m_km[keep],
                       m_rt04=np.where(extra, np.nan, m_km).astype(np.float32)[keep], m_cams=m_cams[keep],
                       m_clim=(0.9 * m_km + 3).astype(np.float32)[keep], m_bcsd=m_km[keep].copy())
        write_record()
        dist1 = dday[isrc[issue, 0], mon]
        F.FILED.write_text(F.sha256_file(__file__) + "\n")

        def prov(i, d, mut):
            lm = [Ld + 1 for Ld in range(5) if made[i, Ld]]
            dd = dt.date.fromisoformat(d)
            p = dict(issue=d, leads_made=lm, src_days={"1": str(dd - dt.timedelta(days=1)), "2": str(dd - dt.timedelta(days=2))},
                     n_monitors=[int(cnt[isrc[i, 0]]), int(cnt[isrc[i, 1]])],
                     identity={str(Ld): dict(n=8 * n, same_nan=True, max_abs=0.0, mean_abs=0.0, max_rel=0.0, r=1.0, identical=1.0) for Ld in lm},
                     guard_hits={str(Ld): {a: dict(at_bound=0, z0_trunc=0 if (a == "m2" and Ld == 5) else 200, at_pmax=0)
                                           for a in EXPECT["arms"]} for Ld in lm},
                     sha256={**{k: DEPS[v] for k, v in PROV_SHA.items()}, "ckpt_m": sel["checkpoint"]["sha256"], **FREE_SHA},
                     step_m=sel["checkpoint"]["step"], **{k: v for k, v in EXPECT.items()})
            return json.dumps(mut(i, p) if mut else p)

        def base_members():
            m1 = KM.copy()
            m1[~made] = np.nan
            m2 = m1.copy()
            m2[:, 4] = np.nan
            return m1, m2, m1.copy()

        def write_fc(m1, m2, f0, done=None, mut=None, dmin1=None):
            nm = np.array([[cnt[isrc[i, 0]], cnt[isrc[i, 1]]] for i in range(len(days))], np.int64)
            F.save_npz(F.FORECASTS, days=np.array(days), done=np.ones(len(days), bool) if done is None else done, free0=f0, m1=m1, m2=m2,
                       dmin1=dday[isrc[:, 0]] if dmin1 is None else dmin1, dmin2=dday[isrc[:, 1]], nmon=nm,
                       prov=np.array([prov(i, d, mut) for i, d in enumerate(days)]))

        def move(m, rows, f, spread=0.3, seed=1, fr=None):
            """members of m at the synthetic truth rows `rows` moved toward (f < 1) / away from (f > 1) the observation"""
            r = np.flatnonzero(rows)
            fr = (f + np.random.default_rng(seed).uniform(-spread, spread, r.size)) if fr is None else fr[r]
            x = obs[r, None] + fr[:, None] * (e_km[r] - obs[r, None])
            m[issue[r], lead[r] - 1, :, mon[r]] = np.maximum(x, 0.0)
            return m

        def run(name, want, m1, m2, f0, stage=None, fresh=True, fast=True, **kw):
            if fresh:
                shutil.rmtree(F.RESULT, ignore_errors=True)
            if m1 is not None:
                write_fc(m1, m2, f0, **kw)
            n0 = len(F._TRACE)
            t0, code = time.time(), None
            try:
                run_decision(fast_tables=fast)
            except SystemExit as ex:
                code = ex.code
            opened = any(p.endswith("pairs.npz") and str(tmp) in p and "heldout" not in p for p in F._TRACE[n0:])
            r = json.loads((F.RESULT / "result.json").read_text()) if (F.RESULT / "result.json").exists() else {}
            got = {3: "REFUSED (FILED)", 4: "REFUSED (read before)"}.get(code, r.get("cls"))
            o = dict(case=f"e2e: {name}", expected=want, got=got, truth_opened=opened, tables_written=(F.RESULT / "RESULT.md").exists(),
                     pairs_written=(F.RESULT / "pairs_heldout.npz").exists(), lock=(F.RESULT / "TRUTH_READ").exists(),
                     stage=r.get("stage"), seconds=round(time.time() - t0, 1))
            if "decision" in r:
                dc = r["decision"]
                o.update(lead1=[round(dc["lead1"]["est"], 3)] + [round(x, 3) for x in dc["lead1"]["ci"]],
                         lead2_5=[round(dc["lead2_5"]["est"], 3)] + [round(x, 3) for x in dc["lead2_5"]["ci"]],
                         n1=dc["lead1"]["n"], n25=dc["lead2_5"]["n"], outside_primary=r["rows_outside_the_record_primary_sample_excluded"],
                         by_distance={kk_: {g: round(r["tables"]["by_distance"][kk_]["all"][g]["est"], 3) for g in DCLS
                                            if g in r["tables"]["by_distance"][kk_]["all"]} for kk_ in ("mfed_vs_km|rmse", "meve_vs_km|rmse", "fz_vs_km|rmse", "mfed_vs_fz|rmse")})
            elif r.get("invalid"):
                o["invalid"] = [x[:200] for x in r["invalid"][:4]]
            ok = got == want
            if stage == "pre":                                   # nothing of the truth may be opened, no lock, no table
                ok &= not opened and not o["lock"] and not o["tables_written"] and not o["pairs_written"] and                     (o["stage"] or "").startswith("before the truth")
            elif stage == "post":                                # read once, with the lock
                ok &= opened and o["lock"]
            elif stage == "refused":
                ok &= not opened
            o["ok"] = bool(ok)
            res.append(o)
            print(f"[{'OK' if ok else 'FAIL'}] {o['case']}: expected {want}, got {got}; truth opened {opened}, tables {o['tables_written']}, "
                  f"lock {o['lock']}" + (f"; lead 1 {o['lead1']}, lead 2-5 {o['lead2_5']}" if "lead1" in o else "")
                  + (f"; invalid {o['invalid'][:2]}" if "invalid" in o else "") + f" ({o['seconds']} s)", flush=True)
            return o, r

        # -- classes through the real path ------------------------------------------------------------------------------
        m1, m2, f0 = base_members()
        o, r = run("identical arms (M = free0 = the record's km)", "FAIL", m1, m2, f0, stage="post")
        Z = np.load(F.RESULT / "pairs_heldout.npz")
        zm = Z["mon"].astype(np.int64)
        smp_ok = bool(held[zm].all()) and Z["obs"].size == int((held[mon] & prim).sum()) and o.get("outside_primary") == int((held[mon] & extra).sum())
        o["sample"] = dict(rows=int(zm.size), all_heldout=bool(held[zm].all()), heldout_primary_rows=int((held[mon] & prim).sum()),
                           heldout_rows_outside_primary=int((held[mon] & extra).sum()))
        o["ok"] = o["ok"] and smp_ok
        m1, m2, f0 = base_members()
        run("gain at lead day 1 only (lead days 2-5 unchanged), FULL tables", "PARTIAL", move(m1, lead == 1, 0.6), m2, f0, stage="post", fast=False)
        m1, m2, f0 = base_members()
        f0 = move(f0, (lead == 1) & (dist1 <= 150), 0.8, seed=4)
        o, r = run("gain within 50 km of an assimilated monitor, every lead day; free0 better at lead 1 within 150 km", "SUCCESS",
                   move(m1, dist1 <= 50, 0.6), m2, f0, stage="post")
        Z = np.load(F.RESULT / "pairs_heldout.npz")
        zm, zi, zl = Z["mon"].astype(np.int64), Z["issue"].astype(np.int64), Z["lead"].astype(np.int64)
        zd = dday[isrc[zi, 0], zm]
        far, nr_ = zd > 150.0, zd <= 150.0
        m1z = np.maximum(move(base_members()[0], dist1 <= 50, 0.6)[zi, zl - 1, :, zm], 0.0)   # the same m1 (move is seeded)
        f0z = np.maximum(f0[zi, zl - 1, :, zm], 0.0)
        blend = dict(far_rows=int(far.sum()), near_rows=int(nr_.sum()),
                     mfed_is_km_beyond_R=bool(np.array_equal(Z["e_mfed"][far], Z["e_km"][far])),
                     mfed_is_m1_within_R=bool(np.array_equal(Z["e_mfed"][nr_], m1z[nr_])),
                     fz_is_km_beyond_R=bool(np.array_equal(Z["e_fz"][far], Z["e_km"][far])),
                     fz_is_free0_within_R=bool(np.array_equal(Z["e_fz"][nr_], f0z[nr_])))
        bd = o.get("by_distance", {})
        o["blend_rows"] = blend
        o["ok"] = o["ok"] and all(v for k_, v in blend.items() if not k_.endswith("rows")) and \
            bd["mfed_vs_km|rmse"].get(">150km") == 0.0 and bd["mfed_vs_km|rmse"].get("50-150km") == 0.0 and \
            bd["fz_vs_km|rmse"].get(">150km") == 0.0 and bd["fz_vs_km|rmse"].get("50-150km", 0.0) < 0
        m1, m2, f0 = base_members()
        m1 = move(m1, dist1 > 150, 1.6, seed=2)
        o, _ = run("M worse beyond 150 km (f 1.6) + gain at every lead day within 150 km", "SUCCESS",
                   move(m1, dist1 <= 150, 0.6), m2, f0, stage="post")
        bd = o.get("by_distance", {})
        o["ok"] = o["ok"] and bd["meve_vs_km|rmse"].get(">150km", 0) > 0 and bd["mfed_vs_km|rmse"].get(">150km") == 0.0
        cat_ = np.load(F.CATALOGUE)
        cell_row = ((np.floor(cat_["lat"].astype(np.float64) / 2).astype(np.int64) + 90) * 1000 +
                    (np.floor(cat_["lon"].astype(np.float64) / 2).astype(np.int64) + 180))[mon]
        ucell, cinv = np.unique(cell_row, return_inverse=True)
        zc = np.random.default_rng(5).standard_normal(ucell.size)
        m1, m2, f0 = base_members()
        fr = np.clip(1.0 - 0.03 + 0.3 * zc[cinv], 0.3, None)                # review case 5: SUCCESS under the lenient reading
        o, _ = run("gain at lead day 1 + small cell-heterogeneous loss at lead days 2-5 (review case 5)", "PARTIAL",
                   move(move(m1, lead == 1, 0.6), lead >= 2, None, fr=fr), m2, f0, stage="post")
        o["ok"] = o["ok"] and o["lead2_5"][0] > 0
        m1, m2, f0 = base_members()
        run("gain at lead day 1 + clear loss at lead days 2-5 (f 1.4)", "PARTIAL", move(move(m1, lead == 1, 0.6), lead >= 2, 1.4, seed=3), m2, f0, stage="post")
        m1, m2, f0 = base_members()
        run("gain at lead days 2-5 only", "FAIL", move(m1, lead >= 2, 0.6), m2, f0, stage="post")
        # -- INVALID before the truth: nothing opened ----------------------------------------------------------------
        g1 = lambda: move(base_members()[0], lead == 1, 0.6)
        m1, m2, f0 = g1(), base_members()[1], base_members()[2]
        a1, a2, a0 = m1.copy(), m2.copy(), f0.copy()
        a1[20], a2[20], a0[20] = np.nan, np.nan, np.nan
        done = np.ones(len(days), bool)
        done[20] = False
        run("one issue day not done", "INVALID", a1, a2, a0, stage="pre", done=done)
        a1 = m1.copy()
        a1[5, 2, :, np.flatnonzero(held)[:50]] = np.nan
        run("M members NaN at 50 held-out monitors on one made lead day", "INVALID", a1, m2, f0, stage="pre")
        dm = dday[isrc[:, 0]].copy()
        dm[:, :n // 100] += 200.0
        run("network distances differ from the inputs' on 1 % of the monitors", "INVALID", m1, m2, f0, stage="pre", dmin1=dm)

        def mut_11a(i, p):
            p["identity"], p["arms"] = {}, ["m1", "m2"]
            return p
        run("provenance: free arm not run (identity empty; review 11a)", "INVALID", m1, m2, f0, stage="pre", mut=mut_11a)

        def mut_11(i, p):
            p["seed"], p["ode_steps"] = 7, 3
            return p
        run("provenance: seed 7 and 3 Euler steps (review 11)", "INVALID", m1, m2, f0, stage="pre", mut=mut_11)

        def mut_ck(i, p):
            p["sha256"]["ckpt_m"] = "0" * 64
            return p
        run("provenance: another checkpoint of M", "INVALID", m1, m2, f0, stage="pre", mut=mut_ck)

        def mut_arms(i, p):
            p["arms"] = ["free", "m1", "m2"]
            for hl in p["guard_hits"].values():
                hl.pop("free0")
            return p
        run("provenance: the pre-review arms (no free0)", "INVALID", m1, m2, f0, stage="pre", mut=mut_arms)

        def mut_noise(i, p):
            if i == 7:
                p["guard_hits"]["3"]["free0"]["z0_trunc"] += 1
            return p
        run("free0 does not share the base noise of m1 on one lead day", "INVALID", m1, m2, f0, stage="pre", mut=mut_noise)
        k0 = "a11/stageM/final_test/select_2021.py"
        DEPS[k0], keep = "0" * 64, DEPS[k0]
        try:
            run("a frozen file changed", "INVALID", m1, m2, f0, stage="pre")
        finally:
            DEPS[k0] = keep
        F.FILED.unlink()
        run("FILED missing", "REFUSED (FILED)", m1, m2, f0, stage="refused")
        F.FILED.write_text(F.sha256_file(__file__) + "\n")
        # -- the one reading --------------------------------------------------------------------------------------------
        run("first decision", "PARTIAL", m1, m2, f0, stage="post")
        run("the decision started again with the same FILED (lock present)", "REFUSED (read before)", None, None, None, stage="refused", fresh=False)
        write_record(drop_one=True)
        o, _ = run("check 5: the truth reader does not reproduce the record", "INVALID", m1, m2, f0, stage="post")
        o["ok"] = o["ok"] and not o["tables_written"] and not o["pairs_written"] and (o["stage"] or "").startswith("after the truth")
        run("... then started again (lock present)", "REFUSED (read before)", None, None, None, stage="refused", fresh=False)
        ok_all = all(x["ok"] for x in res if x["case"].startswith("e2e:"))
    finally:
        F.RECORD_PAIRS, F.FORECASTS, F.FILED, F.RESULT = saved
    return ok_all


def run_tests():
    F.install_truth_firewall(trace=True)                     # after the imports: no observation file can be opened now
    t0 = time.time()
    res = []
    real_result = sorted(p.name for p in F.RESULT.glob("*")) if F.RESULT.exists() else []
    tmp = Path(tempfile.mkdtemp(prefix="ft_score_tests_", dir=os.environ.get("FT_TMP", "/tmp")))
    try:
        ok_u = unit_tests(tmp, res)
        # constants of the provenance check = the files on disk (no observation)
        prior = F.CKPT_FREE.parent.parent / json.loads((F.CKPT_FREE.parent.parent / "PRIOR.json").read_text())["file"]
        sel = F.load_selection()
        cst = dict(ckpt_free=F.sha256_file(F.CKPT_FREE) == FREE_SHA["ckpt_free"], prior=F.sha256_file(prior) == FREE_SHA["prior"],
                   ckpt_m=F.sha256_file(sel["checkpoint"]["path"]) == sel["checkpoint"]["sha256"],
                   deps=dep_hashes() == DEPS)
        ok_c = all(cst.values())
        res.append(dict(case="frozen constants = the files on disk", ok=bool(ok_c), **cst))
        print(f"[{'OK' if ok_c else 'FAIL'}] frozen constants = the files on disk {cst}", flush=True)
        ok_e = e2e_tests(tmp, res)
        # the firewall refuses the truth files in this process
        refused = []
        for p in (F.RECORD_PAIRS, F.P / "a11/test2026/obs/obs_2026.npz", F.P / "hindcast/2026/st_truth_2026.npz"):
            try:
                open(p, "rb").close()
                refused.append(False)
            except PermissionError:
                refused.append(True)
        bad = [p for p in F.opened_paths() if any(m in p for m in F.TRUTH_MARKERS)]
        ok_f = all(refused)
        res.append(dict(case="firewall refuses the truth files (and nothing else of them was opened)", ok=bool(ok_f), attempts_refused=bad))
        print(f"[{'OK' if ok_f else 'FAIL'}] firewall refuses the truth files; truth-marker paths seen: {bad}", flush=True)
        after = sorted(p.name for p in F.RESULT.glob("*")) if F.RESULT.exists() else []
        ok_r = not F.FILED.exists() and after == real_result
        res.append(dict(case="the real FILED absent and the real result/ untouched", ok=bool(ok_r), result_before=real_result, result_after=after))
        print(f"[{'OK' if ok_r else 'FAIL'}] the real FILED absent, the real result/ untouched", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok_all = ok_u and ok_c and ok_e and ok_f and ok_r
    out = dict(all_ok=bool(ok_all), cases=res, scorer_sha256=F.sha256_file(__file__), seconds=round(time.time() - t0, 1), B=B,
               written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    F.save_json(F.OUT / "score_tests.json", out)
    print(f"[score_final --tests] {'ALL PASS' if ok_all else 'FAILURES'} in {time.time() - t0:.0f} s -> {F.OUT / 'score_tests.json'}")
    sys.exit(0 if ok_all else 1)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tests", action="store_true")
    g.add_argument("--hashes", action="store_true")
    a = ap.parse_args()
    if a.tests:
        run_tests()
    elif a.hashes:
        print(json.dumps(dict(score_final=F.sha256_file(__file__), deps=dep_hashes()), indent=1))
    else:
        run_decision()


if __name__ == "__main__":
    main()
