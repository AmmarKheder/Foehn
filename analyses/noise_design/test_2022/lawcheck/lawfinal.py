#!/usr/bin/env python3
"""A16 law check, final step (imported by lawcheck.py final): reads pop_stats.npz, ncp_stats.npz, part_noise.json and the
part files of F1 (eval_final_crps_e01) and G1 (eval_final_crps_e01_ar); writes lawcheck.json.

Uncertainty that CONTAINS the draw of the fixed fields: the units are the 16 member indices. Under the stored design index m
carries one fixed field for every forecast; under 'ar' index m carries fresh fields at every start. Both runs use the same
forecasts (same weather) and the same particle subset per (forecast, index). A statistic of the two runs is jackknifed by
deleting index j from both runs at once (paired delete-one jackknife over the 16 indices; t quantile with 15 dof).
Quartets: all 1,820 quartets of the 16 indices; a fresh quartet varies sqrt(16/12) more than a quartet of 16 fixed fields."""
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from lawcheck import LC, RUN, P, THR, NB, LO, STEP, RNAMES, save_json, sha_file

QUARTETS = np.array(list(combinations(range(16), 4)))
Q0 = int(np.flatnonzero((QUARTETS == [0, 1, 2, 3]).all(1))[0])
T15 = 2.1314
ALL = np.arange(16)


# ------------------------------------------------------------------ statistics of an index set from aggregated sums
class Run:
    """aggregated sums of one run over a set of forecasts; N, S1 (16), G (16, 16), C (3, 16), optional LS1, LG, H"""

    def __init__(self, N, S1, G, C, LS1=None, LG=None, H=None):
        self.N, self.S1, self.G, self.C, self.LS1, self.LG, self.H = float(N), S1, G, C, LS1, LG, H

    def level(self, I):
        return float(self.S1[I].mean() / self.N)

    def share(self, I, k):
        return float(self.C[k, I].mean() / self.N)

    def var_between(self, I, log=False):
        Gm = (self.LG if log else self.G) / self.N
        I = list(I); n = len(I)
        d = np.diag(Gm)[I]
        sub = Gm[np.ix_(I, I)]
        tot = (n * d.sum() - sub.sum())                      # sum over a<b of (Gaa + Gbb - 2 Gab)
        return float(tot / (n * (n - 1)))                    # = pixel mean of the ddof-1 variance over the members of I

    def dep(self, m, I=ALL):
        """pixel mean of (x_m - mean of the other members of I)^2"""
        o = [i for i in I if i != m]
        Gm = self.G / self.N
        return float(Gm[m, m] - 2 * Gm[m, o].mean() + Gm[np.ix_(o, o)].mean())

    def quantile(self, I, q):
        h = self.H[I].sum(0).astype(np.float64)
        c = np.cumsum(h) / h.sum()
        k = int(np.searchsorted(c, q))                       # bin k: 0 underflow, 1..NB, NB+1 overflow
        k = min(max(k, 1), NB)
        c0 = c[k - 1]; frac = (q - c0) / max(c[k] - c0, 1e-300)
        return float(np.exp(LO + (k - 1 + frac) * STEP))


def var_between_check():
    """V_I = sum_{a<b} mean (x_a - x_b)^2 / (n (n - 1)) is the pixel mean of the ddof=1 variance: test on random data"""
    x = np.random.default_rng(0).gamma(2.0, 10.0, (5000, 16))
    r = Run(len(x), x.sum(0), x.T @ x, np.zeros((3, 16)))
    return abs(r.var_between(ALL) - x.var(1, ddof=1).mean()) < 1e-8 * x.var(1, ddof=1).mean()


def jk_paired(fn):
    """fn(I) -> value with index set I used in both runs; paired delete-one jackknife over the 16 indices"""
    full = fn(ALL)
    reps = np.array([fn(np.delete(ALL, j)) for j in range(16)])
    se = float(np.sqrt(15 / 16 * ((reps - reps.mean()) ** 2).sum()))
    return full, se


def ratio_jk(fa, fs):
    """ratio fa(I)/fs(I) with paired jackknife on the log; 95 % interval (t, 15 dof)"""
    lr, se = jk_paired(lambda I: np.log(fa(I) / fs(I)))
    return dict(ratio=float(np.exp(lr)), ci95=[float(np.exp(lr - T15 * se)), float(np.exp(lr + T15 * se))],
                se_log=se, ar=float(fa(ALL)), stored=float(fs(ALL)))


def quartet_block(fs, fa):
    """distribution of a statistic over the 1,820 quartets of each run; rank of the stored quartet 0-3"""
    vs = np.array([fs(q) for q in QUARTETS]); va = np.array([fa(q) for q in QUARTETS])
    sd_fresh = vs.std(ddof=1) * np.sqrt(16 / 12)
    ref = fa(ALL)
    null_ratio = va[Q0] / vs                                     # paper ratio if the stored quartet were any other
    obs = va[Q0] / vs[Q0]
    return dict(
        stored=dict(q025=np.quantile(vs, 0.025), q50=np.median(vs), q975=np.quantile(vs, 0.975), min=vs.min(), max=vs.max(),
                    sd=vs.std(ddof=1), sd_fresh_quartet=sd_fresh, quartet_0_3=vs[Q0],
                    rank_quartet_0_3=float((vs < vs[Q0]).mean()), all16=fs(ALL)),
        ar=dict(q025=np.quantile(va, 0.025), q50=np.median(va), q975=np.quantile(va, 0.975), sd=va.std(ddof=1),
                quartet_0_3=va[Q0], all16=ref, rank_of_ar_median_in_stored=float((vs < np.median(va)).mean()),
                rank_of_ar_quartet_0_3_in_stored=float((vs < va[Q0]).mean())),
        z_stored_quartet_0_3_vs_ar16=float((vs[Q0] - ref) / sd_fresh) if sd_fresh > 0 else None,
        ratio_quartet_0_3_ar_over_stored=obs,
        null_ratio_if_stored_quartet_were_any=dict(q025=np.quantile(null_ratio, 0.025), q50=np.median(null_ratio),
                                                   q975=np.quantile(null_ratio, 0.975),
                                                   share_as_far_from_1_as_observed=float((np.abs(np.log(null_ratio)) >=
                                                                                         abs(np.log(obs))).mean())))


def index_spread(ys, ya):
    """per-index values of the two runs: across-index sd, field-draw variance estimate, t-test of the means"""
    ys, ya = np.asarray(ys, float), np.asarray(ya, float)
    d = ya - ys                                                    # paired by index (same particles)
    se = d.std(ddof=1) / 4.0
    return dict(stored_mean=ys.mean(), ar_mean=ya.mean(), stored_cv=ys.std(ddof=1) / ys.mean(), ar_cv=ya.std(ddof=1) / ya.mean(),
                field_draw_sd_rel=float(np.sqrt(max(ys.var(ddof=1) - ya.var(ddof=1), 0)) / ys.mean()),
                ratio_of_means=ya.mean() / ys.mean(),
                ratio_ci95=[(ya.mean() - T15 * se) / ys.mean(), (ya.mean() + T15 * se) / ys.mean()],
                paired_t=float(d.mean() / se) if se > 0 else None)


# ------------------------------------------------------------------ populated pixels
def pop_section(z):
    keys = z["keys"]
    out = {"source": str(LC / "pop_stats.npz") + " (from a13/warn16/data/run16 + a14/data/run16_13 + a13/warn16/data/run16all "
                     "task 0 = stored; a16/data/run16 = ar)",
           "sample": f"{len(keys)} forecasts (179 starts x leads 1, 3), pixels finite in all 32 members: "
                     f"min {int(z['ok_n'].min())} of 8112762; non-finite rows stored/ar {z['nonfinite'].sum(0).tolist()}"}
    for lab, sel in (("lead1", keys[:, 1] == 1), ("lead3", keys[:, 1] == 3), ("leads1_3", np.ones(len(keys), bool))):
        def mk(t, pre=""):
            if pre == "":
                return Run(z[f"{t}_N"][sel].sum(), z[f"{t}_S1"][sel].sum(0), z[f"{t}_G"][sel].sum(0), z[f"{t}_C"][sel].sum(0),
                           z[f"{t}_LS1"][sel].sum(0), z[f"{t}_LG"][sel].sum(0), z[f"{t}_H"][sel].sum(0))
            return Run(z[f"{t}_STN"][sel].sum(), z[f"{t}_STS1"][sel].sum(0), z[f"{t}_STG"][sel].sum(0), z[f"{t}_STC"][sel].sum(0))
        s, a = mk("s"), mk("a")
        sec = {"forecasts": int(sel.sum())}
        sec["level"] = ratio_jk(a.level, s.level)
        sec["var_between_members"] = ratio_jk(a.var_between, s.var_between)
        sec["var_between_members_log1p"] = ratio_jk(lambda I: a.var_between(I, True), lambda I: s.var_between(I, True))
        for k, t in enumerate(THR):
            sec[f"share_above_{t}"] = ratio_jk(lambda I, k=k: a.share(I, k), lambda I, k=k: s.share(I, k))
        for q in (0.9, 0.99, 0.999):
            sec[f"quantile_{q}"] = ratio_jk(lambda I, q=q: a.quantile(I, q), lambda I, q=q: s.quantile(I, q))
        sec["quartets"] = {"level": quartet_block(s.level, a.level), "var_between_members": quartet_block(s.var_between, a.var_between),
                           "share_above_75": quartet_block(lambda I: s.share(I, 1), lambda I: a.share(I, 1)),
                           "share_above_150": quartet_block(lambda I: s.share(I, 2), lambda I: a.share(I, 2)),
                           "quantile_0.999": quartet_block(lambda I: s.quantile(I, 0.999), lambda I: a.quantile(I, 0.999))}
        sec["per_index"] = {"level": index_spread([s.level([m]) for m in ALL], [a.level([m]) for m in ALL]),
                            "share_above_75": index_spread([s.share([m], 1) for m in ALL], [a.share([m], 1) for m in ALL]),
                            "departure_from_mean_of_others": index_spread([s.dep(m) for m in ALL], [a.dep(m) for m in ALL]),
                            "stored_level_by_index": [s.level([m]) for m in ALL], "ar_level_by_index": [a.level([m]) for m in ALL],
                            "stored_dep_by_index": [s.dep(m) for m in ALL], "ar_dep_by_index": [a.dep(m) for m in ALL]}
        ss, sa = mk("s", "ST"), mk("a", "ST")
        sec["station_pixels"] = {"pixels": int(z["n_station_pixels"]), "level": ratio_jk(sa.level, ss.level),
                                 "var_between_members": ratio_jk(sa.var_between, ss.var_between),
                                 "share_above_75": ratio_jk(lambda I: sa.share(I, 1), lambda I: ss.share(I, 1)),
                                 "quartets_var_between_members": quartet_block(ss.var_between, sa.var_between),
                                 "quartets_level": quartet_block(ss.level, sa.level)}
        reg = {}
        for r, nm in enumerate(RNAMES):
            rs = Run(z["s_RN"][sel, r].sum(), z["s_RS1"][sel, r].sum(0), z["s_RG"][sel, r].sum(0), z["s_RC"][sel, r].sum(0))
            ra = Run(z["a_RN"][sel, r].sum(), z["a_RS1"][sel, r].sum(0), z["a_RG"][sel, r].sum(0), z["a_RC"][sel, r].sum(0))
            qv = quartet_block(rs.var_between, ra.var_between); ql = quartet_block(rs.level, ra.level)
            reg[nm] = {"pixels": int(z["region_pixels"][r]), "level": ratio_jk(ra.level, rs.level),
                       "var_between_members": ratio_jk(ra.var_between, rs.var_between),
                       "share_above_75": ratio_jk(lambda I: ra.share(I, 1), lambda I: rs.share(I, 1)),
                       "quartet_0_3_var_ratio": qv["ratio_quartet_0_3_ar_over_stored"],
                       "quartet_0_3_var_null_q025_q975": [qv["null_ratio_if_stored_quartet_were_any"]["q025"],
                                                          qv["null_ratio_if_stored_quartet_were_any"]["q975"]],
                       "quartet_0_3_var_rank_in_stored": qv["stored"]["rank_quartet_0_3"],
                       "quartet_0_3_level_ratio": ql["ratio_quartet_0_3_ar_over_stored"],
                       "quartet_0_3_level_null_q025_q975": [ql["null_ratio_if_stored_quartet_were_any"]["q025"],
                                                            ql["null_ratio_if_stored_quartet_were_any"]["q975"]]}
        sec["regions"] = reg
        out[lab] = sec
    # lead drift between leads 1 and 3 (e): ratio(3)/ratio(1), paired jackknife
    def run_for(t, lead):
        sel = keys[:, 1] == lead
        return Run(z[f"{t}_N"][sel].sum(), z[f"{t}_S1"][sel].sum(0), z[f"{t}_G"][sel].sum(0), z[f"{t}_C"][sel].sum(0))
    s1, a1, s3, a3 = run_for("s", 1), run_for("a", 1), run_for("s", 3), run_for("a", 3)
    def drift(fn):
        lr, se = jk_paired(lambda I: np.log(fn(a3, I) / fn(s3, I)) - np.log(fn(a1, I) / fn(s1, I)))
        return dict(ratio3_over_ratio1=float(np.exp(lr)), ci95=[float(np.exp(lr - T15 * se)), float(np.exp(lr + T15 * se))])
    def st_for(t, lead):
        sel = keys[:, 1] == lead
        return Run(z[f"{t}_STN"][sel].sum(), z[f"{t}_STS1"][sel].sum(0), z[f"{t}_STG"][sel].sum(0), z[f"{t}_STC"][sel].sum(0))
    qd = {}
    for nm, (S1_, A1_, S3_, A3_) in (("populated_pixels", (s1, a1, s3, a3)),
                                     ("station_pixels", (st_for("s", 1), st_for("a", 1), st_for("s", 3), st_for("a", 3)))):
        dq = np.array([np.log(S3_.var_between(q) / S1_.var_between(q)) for q in QUARTETS])
        da = np.array([np.log(A3_.var_between(q) / A1_.var_between(q)) for q in QUARTETS])
        qd[nm] = dict(what="log(V lead 3 / V lead 1) of the variance between members, per quartet",
                      stored_q025=float(np.quantile(dq, 0.025)), stored_q975=float(np.quantile(dq, 0.975)), stored_quartet_0_3=float(dq[Q0]),
                      ar_q025=float(np.quantile(da, 0.025)), ar_q975=float(np.quantile(da, 0.975)), ar_quartet_0_3=float(da[Q0]),
                      ratio_of_ratios_quartet_0_3=float(np.exp(da[Q0] - dq[Q0])))
    out["lead_drift_1_to_3_quartets"] = qd
    exc = []
    for lab in ("lead1", "lead3"):
        for nm, v in out[lab]["regions"].items():
            for k in ("level", "var_between_members", "share_above_75"):
                lo_, hi_ = v[k]["ci95"]
                if lo_ > 1 or hi_ < 1:
                    exc.append(f"{lab}/{nm}/{k}: {v[k]['ratio']:.3f} [{lo_:.3f}, {hi_:.3f}]")
    out["regional_intervals_excluding_1"] = dict(tested=2 * len(RNAMES) * 3, expected_at_5pct=0.05 * 2 * len(RNAMES) * 3, list=exc)
    out["lead_drift_1_to_3"] = {"var_between_members": drift(lambda r, I: r.var_between(I)), "level": drift(lambda r, I: r.level(I)),
                                "share_above_75": drift(lambda r, I: r.share(I, 1))}
    # fingerprints: persistence of a member's departure pattern across start days (the fixed field) and leads
    FPs, FPa = z["s_FP"].astype(np.float32), z["a_FP"].astype(np.float32)
    pos = {(int(d), int(h)): i for i, (d, h) in enumerate(keys)}
    def corr_pairs(FP, pairs, same_member=True):
        vals = []
        for i, j in pairs:
            A, B = FP[i], FP[j]
            ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
            A = A[:, ok] - A[:, ok].mean(1, keepdims=True); B = B[:, ok] - B[:, ok].mean(1, keepdims=True)
            Bm = B if same_member else np.roll(B, 1, axis=0)
            vals.append((A * Bm).sum(1) / np.sqrt((A * A).sum(1) * (Bm * Bm).sum(1)))
        v = np.array(vals)                                            # (pairs, 16)
        per_m = v.mean(0)
        return dict(mean=float(per_m.mean()), se_over_indices=float(per_m.std(ddof=1) / 4), pairs=len(pairs))
    starts = sorted({int(d) for d, h in keys})
    cons = {h: [(pos[(d, h)], pos[(d2, h)]) for d, d2 in zip(starts[:-1], starts[1:])] for h in (1, 3)}
    far = {h: [(pos[(d, h)], pos[(d + 120, h)]) for d in starts if (d + 120, h) in pos] for h in (1, 3)}
    lead13 = [(pos[(d, 1)], pos[(d, 3)]) for d in starts]
    fp = {}
    for nm, FP in (("stored", FPs), ("ar", FPa)):
        fp[nm] = {"same_index_consecutive_starts_lead1": corr_pairs(FP, cons[1]),
                  "same_index_consecutive_starts_lead3": corr_pairs(FP, cons[3]),
                  "same_index_starts_120_days_apart_lead1": corr_pairs(FP, far[1]),
                  "same_index_same_start_lead1_vs_lead3": corr_pairs(FP, lead13),
                  "other_index_consecutive_starts_lead1": corr_pairs(FP, cons[1], same_member=False)}
    out["fingerprint_persistence"] = {"what": "pixel correlation of a member's departure from the mean of the other 15 "
                                              "(log1p values) at 60,000 fixed random populated pixels, between two forecasts",
                                      **fp}
    return out


# ------------------------------------------------------------------ NCP
def ncp_section(z, noise):
    keys = z["keys"]
    bn = [str(b) for b in z["band_names"]]
    out = {"source": str(LC / "ncp_stats.npz") + " (RUN/eval_case_eastasia_crps_e01 and _ar, crop0_d<start>_h<lead>.npz, "
                     "rows 50:650 cols 150:950 = N 41.5 S 35.5 W 113.5 E 121.5)",
           "sample": f"{len(keys)} forecasts = 28 starts x leads 1-7, 16 members, {int(z['ok'].min())} pixels finite in all 32",
           "bands": "normalized box filters: <5 = x - B5, 5-20 = B5 - B20, 20-80 = B20 - B80, 80-300 = B80 - B300, >300 = B300; "
                    "variance over the box per member, mean over forecasts"}
    def per_index(t, sel):
        return z[f"{t}_bands"][sel].mean(0)                          # (16, 6)
    def runs(sel):
        mk = lambda t: Run(z["ok"][sel].sum(), z[f"{t}_S1"][sel].sum(0), z[f"{t}_G"][sel].sum(0), z[f"{t}_C"][sel].sum(0))
        return mk("s"), mk("a")
    leads = {}
    for lab, sel in [("all_leads", np.ones(len(keys), bool))] + [(f"lead{h}", keys[:, 1] == h) for h in range(1, 8)]:
        bs, ba = per_index("s", sel), per_index("a", sel)
        s, a = runs(sel)
        sec = {"forecasts": int(sel.sum())}
        for b, nm in enumerate(bn):
            sec[f"band_{nm}"] = ratio_jk(lambda I, b=b: ba[I, b].mean(), lambda I, b=b: bs[I, b].mean())
        sec["level"] = ratio_jk(a.level, s.level)
        sec["var_between_members"] = ratio_jk(a.var_between, s.var_between)
        sec["share_above_75"] = ratio_jk(lambda I: a.share(I, 1), lambda I: s.share(I, 1))
        sec["share_above_150"] = ratio_jk(lambda I: a.share(I, 2), lambda I: s.share(I, 2))
        if lab == "all_leads":
            sec["quartets"] = {f"band_{nm}": quartet_block(lambda I, b=b: bs[I, b].mean(), lambda I, b=b: ba[I, b].mean())
                               for b, nm in enumerate(bn) if nm in ("5-20km", "20-80km", "80-300km", "total")}
            sec["quartets"]["level"] = quartet_block(s.level, a.level)
            sec["quartets"]["var_between_members"] = quartet_block(s.var_between, a.var_between)
            sec["per_index"] = {f"band_{nm}": index_spread(bs[:, b], ba[:, b]) for b, nm in enumerate(bn)}
            sec["per_index"]["level"] = index_spread([s.level([m]) for m in ALL], [a.level([m]) for m in ALL])
            sec["stored_band_5-20km_by_index"] = bs[:, 1].tolist()
            sec["ar_band_5-20km_by_index"] = ba[:, 1].tolist()
            # mechanism: stored per-index output vs the large-scale power of that index's fixed noise in the box
            pw = noise["ncp_box_noise_power"]["stored_per_member"]
            mech = {}
            rng = np.random.default_rng(1)
            for k, v in pw.items():
                v = np.asarray(v, float)
                for b, nm in enumerate(bn):
                    if nm not in ("5-20km", "20-80km", "80-300km", "total"):
                        continue
                    y = bs[:, b]
                    r = float(np.corrcoef(v, y)[0, 1])
                    perm = np.array([np.corrcoef(rng.permutation(v), y)[0, 1] for _ in range(4000)])
                    mech[f"{k}__vs__band_{nm}"] = dict(r=r, p_perm_two_sided=float((np.abs(perm) >= abs(r)).mean()))
                y = np.array([s.level([m]) for m in ALL]); r = float(np.corrcoef(v, y)[0, 1])
                perm = np.array([np.corrcoef(rng.permutation(v), y)[0, 1] for _ in range(4000)])
                mech[f"{k}__vs__level"] = dict(r=r, p_perm_two_sided=float((np.abs(perm) >= abs(r)).mean()))
            sec["stored_output_vs_stored_noise_power_16_indices"] = mech
            # the same relation under ar at lead 1, per (start, member): 448 fields, each used once
            pa = noise["ncp_box_noise_power"]["ar_lead1_per_start_member"]
            l1 = keys[:, 1] == 1
            dd = keys[l1, 0]
            bands_a1 = z["a_bands"][l1]                               # (28, 16, 6) ordered as keys
            S1a = z["a_S1"][l1] / z["ok"][l1][:, None]
            bands_s1 = z["s_bands"][l1]; S1s = z["s_S1"][l1] / z["ok"][l1][:, None]
            ar_mech = {}
            for k, v in pa.items():
                V = np.array([v[int(d)] for d in dd], float)          # (28, 16)
                Vc = V - V.mean(1, keepdims=True)
                for b, nm in enumerate(bn):
                    if nm not in ("5-20km", "20-80km", "80-300km", "total"):
                        continue
                    Y = bands_a1[:, :, b]; Yc = Y - Y.mean(1, keepdims=True)   # within-start centring removes the weather
                    ar_mech[f"{k}__vs__band_{nm}"] = float((Vc * Yc).sum() / np.sqrt((Vc ** 2).sum() * (Yc ** 2).sum()))
                Yc = S1a - S1a.mean(1, keepdims=True)
                ar_mech[f"{k}__vs__level"] = float((Vc * Yc).sum() / np.sqrt((Vc ** 2).sum() * (Yc ** 2).sum()))
            sec["ar_lead1_output_vs_noise_power_within_start"] = dict(n=int(V.size), r=ar_mech,
                                                                     r_crit_0p05=float(1.96 / np.sqrt(V.size - 28)))
            # persistence of the per-index effect across starts: within-start centred band, correlation between starts
            def persist(B):
                Bc = B - B.mean(1, keepdims=True)                      # (28, 16)
                c = np.corrcoef(Bc)                                    # starts x starts, across indices
                iu = np.triu_indices(len(Bc), 1)
                return float(c[iu].mean())
            sec["index_effect_persistence_across_starts_lead1"] = {
                nm: dict(stored=persist(bands_s1[:, :, b]), ar=persist(bands_a1[:, :, b]))
                for b, nm in enumerate(bn)}
            sec["index_effect_persistence_across_starts_lead1"]["level"] = dict(stored=persist(S1s), ar=persist(S1a))
            # fit on ar lead 1 (448 fields): log y = a_start + b u + c u^2, u = box mean of the noise x sqrt(N); then predict the
            # stored run from its 16 fixed u: does this one feature of the stored draw explain the stored/ar ratio at lead 1?
            ua = np.array([pa["mean_x_sqrtN"][int(d)] for d in dd]); us = np.asarray(noise["ncp_box_noise_power"]["stored_per_member"]["mean_x_sqrtN"])
            D = len(dd); X = np.zeros((D * 16, D + 2))
            for i in range(D):
                X[i * 16:(i + 1) * 16, i] = 1
            X[:, D], X[:, D + 1] = ua.ravel(), ua.ravel() ** 2
            pred = {}
            for nm_, ya, ys in [("level", S1a, S1s)] + [(nm, bands_a1[:, :, b], bands_s1[:, :, b]) for b, nm in enumerate(bn)]:
                ly = np.log(ya).ravel()
                coef = np.linalg.lstsq(X, ly, rcond=None)[0]
                res_ = ly - X @ coef
                ps = np.exp(coef[:D][:, None] + coef[D] * us[None] + coef[D + 1] * us[None] ** 2 + res_.var() / 2)
                pred[nm_] = dict(b=float(coef[D]), c=float(coef[D + 1]),
                                 within_start_R2=float(1 - res_.var() / (np.log(ya) - np.log(ya).mean(1, keepdims=True)).var()),
                                 observed_ratio_ar_over_stored=float(ya.mean() / ys.mean()),
                                 predicted_ratio_from_stored_u=float(ya.mean() / ps.mean()),
                                 corr_predicted_vs_observed_stored_by_index=float(np.corrcoef(ps.mean(0), ys.mean(0))[0, 1]))
            sec["box_mean_noise_prediction_lead1"] = dict(stored_u_mean=float(us.mean()), stored_u_sd=float(us.std(ddof=1)), items=pred)
        leads[lab] = sec
    out.update(leads)
    # (e) drift over leads 1..7 of log(ar/stored): OLS slope per lead step, paired jackknife (fields are the units)
    def slope_fn(fn):
        def f(I):
            y = np.array([np.log(fn("a", h, I) / fn("s", h, I)) for h in range(1, 8)])
            x = np.arange(1, 8) - 4.0
            return float((x * y).sum() / (x * x).sum())
        v, se = jk_paired(f)
        return dict(slope_log_ratio_per_lead=v, ci95=[v - T15 * se, v + T15 * se],
                    ratio_lead7_over_lead1_implied=float(np.exp(6 * v)))
    cache = {}
    def bandf(b):
        def fn(t, h, I):
            sel = keys[:, 1] == h
            k = (t, h)
            if k not in cache:
                cache[k] = z[f"{t}_bands"][sel].mean(0)
            return cache[k][I, b].mean()
        return fn
    rcache = {}
    def runf(stat):
        def fn(t, h, I):
            k = (t, h)
            if k not in rcache:
                sel = keys[:, 1] == h
                rcache[k] = Run(z["ok"][sel].sum(), z[f"{t}_S1"][sel].sum(0), z[f"{t}_G"][sel].sum(0), z[f"{t}_C"][sel].sum(0))
            return stat(rcache[k], I)
        return fn
    out["lead_drift"] = {f"band_{nm}": slope_fn(bandf(b)) for b, nm in enumerate(bn)}
    out["lead_drift"]["level"] = slope_fn(runf(lambda r, I: r.level(I)))
    out["lead_drift"]["var_between_members"] = slope_fn(runf(lambda r, I: r.var_between(I)))
    out["lead_drift"]["share_above_75"] = slope_fn(runf(lambda r, I: r.share(I, 1)))
    # luck of the lead drift for the stored quartet 0-3 (as in the 4-member paper runs): over stored quartets
    q = {}
    for b, nm in enumerate(bn):
        if nm not in ("5-20km", "total"):
            continue
        sl = []
        for Qd in QUARTETS:
            y = np.array([np.log(bandf(b)("s", h, Qd)) for h in range(1, 8)]); x = np.arange(1, 8) - 4.0
            sl.append((x * y).sum() / (x * x).sum())
        sl = np.array(sl)
        q[f"band_{nm}"] = dict(stored_quartet_slope_q025=float(np.quantile(sl, 0.025)), q975=float(np.quantile(sl, 0.975)),
                               stored_quartet_0_3=float(sl[Q0]), stored_all16=float(np.mean(sl)))
    out["lead_drift_stored_quartet_luck"] = q
    return out


# ------------------------------------------------------------------ F1 / G1 (4 members, leads 1-7)
def fg_section():
    out = {"source": [str(RUN / "eval_final_crps_e01/part_*.npz"), str(RUN / "eval_final_crps_e01_ar/part_*.npz")],
           "sample": "179 starts x leads 1-7, 4 members (indices 0-3); recs: land pixels with a valid target, 9 regions; st_ens: "
                     "8,110 station pixels (model values only)"}
    def load(d):
        recs, cols, st, keys = [], None, [], []
        for f in sorted(glob.glob(str(RUN / d / "part_*.npz"))):
            z = np.load(f)
            recs.append(z["recs"]); cols = [str(c) for c in z["rec_cols"]]; st.append(z["st_ens"]); keys.append(z["keys"])
        return np.concatenate(recs), cols, np.concatenate(st).astype(np.float32), np.concatenate(keys)
    Rs, cols, STs, Ks = load("eval_final_crps_e01")
    Ra, _, STa, Ka = load("eval_final_crps_e01_ar")
    ci = {c: i for i, c in enumerate(cols)}
    pos_a = {tuple(k): i for i, k in enumerate(Ka)}
    order = np.array([pos_a[tuple(k)] for k in Ks]); STa = STa[order]
    per = {}
    for h in range(1, 8):
        rs = Rs[Rs[:, ci["lead"]] == h]; ra = Ra[Ra[:, ci["lead"]] == h]
        sel = Ks[:, 1] == h
        A, B = STs[sel], STa[sel]
        ok = np.isfinite(A).all(2) & np.isfinite(B).all(2)
        per[str(h)] = dict(var_land=float(ra[:, ci["var"]].sum() / rs[:, ci["var"]].sum() * rs[:, ci["n"]].sum() / ra[:, ci["n"]].sum()),
                           level_mean_land=float(ra[:, ci["sum_f"]].sum() / rs[:, ci["sum_f"]].sum()),
                           var_station_pixels=float(B.var(2, ddof=1)[ok].mean() / A.var(2, ddof=1)[ok].mean()),
                           level_station_pixels=float(B.mean(2)[ok].mean() / A.mean(2)[ok].mean()),
                           share75_station_pixels=float((B > 75).mean(2)[ok].mean() / max((A > 75).mean(2)[ok].mean(), 1e-12)))
    out["ar_over_stored_per_lead"] = per
    for k in ("var_land", "var_station_pixels", "level_mean_land"):
        y = np.log([per[str(h)][k] for h in range(1, 8)]); x = np.arange(1, 8) - 4.0
        out[f"slope_log_{k}_per_lead"] = float((x * y).sum() / (x * x).sum())
    return out


def main():
    noise = json.loads((LC / "part_noise.json").read_text())
    res = {"what": __doc__, "checks": {"var_between_formula_ok": var_between_check()},
           "inputs_sha256": {n: sha_file(LC / n) for n in ("pop_stats.npz", "ncp_stats.npz", "part_noise.json")}}
    res["a_noise"] = {"source": str(LC / "part_noise.json"), **noise}
    res["c_d_pop"] = pop_section(np.load(LC / "pop_stats.npz"))
    res["d_e_ncp"] = ncp_section(np.load(LC / "ncp_stats.npz"), noise)
    res["e_f1_g1"] = fg_section()
    save_json(LC / "lawcheck.json", res)
    print("final done")
