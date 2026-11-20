#!/usr/bin/env python3
"""Tables of the issuance analysis, from the files of ../data (nothing else is read except the submitted result files of
experiments/figures/data, for the reproduction checks). Prints every table and writes ../data/results.json.
Intervals: 95 %, moving-block bootstrap over start days (blocks of seven consecutive start days, 10,000 resamples, seed 0;
common.block_index, the recipe of compare_fair.py), paired between systems and between lead times (the same resampled
start days for both terms of a difference); thresholds are kept fixed in the resamples.
    python summarise.py [label]"""
import json
import sys
from pathlib import Path

import numpy as np

from common import block_index, boot_counts, ci, rates, sedi

D = Path(__file__).resolve().parent.parent / "data"
FIGD = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures/data")
L = sys.argv[1] if len(sys.argv) > 1 else "crps_e01"
R = {}
f2 = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in np.atleast_1d(a)]
np.set_printoptions(linewidth=200, suppress=True)


def fmt(v, lo=None, hi=None, p=3):
    return f"{v:.{p}f}" if lo is None else f"{v:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


# =========================================================================================== 1. reproduction checks
print("=" * 110, "\n1. REPRODUCTION OF THE SUBMITTED FILES WITH ISSUANCE=d\n", "=" * 110, sep="")
old_te = np.load(FIGD / f"tail_events_{L}.npz"); te_d = np.load(D / f"tail_events_{L}_d.npz"); te_n = np.load(D / f"tail_events_{L}_d1.npz")
eq = np.array_equal(old_te["C"], te_d["C"])
print(f"tail_events: counts of the copy with ISSUANCE=d identical to data/tail_events_{L}.npz: {eq} "
      f"(max abs diff {np.abs(old_te['C'] - te_d['C']).max():.0f}; systems {list(map(str, te_d['systems']))})")
R["reproduction"] = {"tail_events_counts_identical": bool(eq)}
old_cg = np.load(FIGD / f"compare_ghap_global_{L}.npz"); cg_d = np.load(D / f"compare_ghap_global_{L}_d.npz"); cg_n = np.load(D / f"compare_ghap_global_{L}_d1.npz")
with np.errstate(all="ignore"):
    rel = np.nanmax(np.abs(old_cg["S"] - cg_d["S"]) / np.maximum(np.abs(old_cg["S"]), 1e-30))
print(f"compare_ghap_global: sums of the copy with ISSUANCE=d against data/compare_ghap_global_{L}.npz: identical "
      f"{np.array_equal(old_cg['S'], cg_d['S'])}, max relative difference {rel:.2e}; U75 identical {np.array_equal(old_cg['U75'], cg_d['U75'])}")
R["reproduction"]["compare_ghap_global_max_rel_diff"] = float(rel)
old_st = np.load(FIGD / f"extremes_stations_own_{L}.npz"); st = np.load(D / f"extremes_stations_own_eqiss_{L}.npz")
eqs = np.array_equal(old_st["C"], st["C"][:, :, :4])
print(f"extremes_stations_own: counts of leads 1-4 identical to data/extremes_stations_own_{L}.npz: {eqs}")
R["reproduction"]["stations_counts_identical_leads_1_4"] = bool(eqs)

# =========================================================================================== 2. land: r, bias, RMSE
print("\n" + "=" * 110, "\n2. LAND, AGAINST THE 1 KM ESTIMATES (0.25 deg block means): RMSE / MAE / bias / r, identical cells and days\n", "=" * 110, sep="")
SYSG = [str(s) for s in cg_n["systems"]]; REGG = [str(r) for r in cg_n["regions"]]


def sc(S):
    """S (..., 12) -> rmse, mae, bias, r"""
    n = S[..., 0]
    with np.errstate(all="ignore"):
        mx, my = S[..., 4] / n, S[..., 5] / n
        r = (S[..., 8] / n - mx * my) / np.sqrt(np.maximum(S[..., 6] / n - mx * mx, 1e-9) * np.maximum(S[..., 7] / n - my * my, 1e-9))
        return dict(rmse=np.sqrt(S[..., 2] / n), mae=S[..., 3] / n, bias=S[..., 1] / n, r=r, n=n)


so, sn = sc(cg_d["S"]), sc(cg_n["S"])
vd = np.flatnonzero(cg_n["SD"][:, 0, 0, 0, 0] > 0); idx = block_index(len(vd))
bs = sc(boot_counts(cg_n["SD"][vd], idx))                              # (n, region, lead, system)
R["land_scores"] = {}
for g, rn in enumerate(REGG):
    R["land_scores"][rn] = {}
    print(f"\n--- {rn} (cells x days per lead: {int(sn['n'][g, 0, 0])}; start days {len(vd)})")
    for k, nm in enumerate(SYSG):
        R["land_scores"][rn][nm] = {}
        for key in ("r", "bias", "rmse", "mae"):
            o, n_ = so[key][g, :, k], sn[key][g, :, k]; lo, hi = ci(bs[key][:, g, :, k])
            R["land_scores"][rn][nm][key] = {"submitted_run_d": f2(o), "equal_issuance": f2(n_), "ci_low": f2(lo), "ci_high": f2(hi)}
        if g == 0 or nm in ("CAMS", "Aurora", L):
            for key in ("r", "bias", "rmse"):
                o, n_ = so[key][g, :, k], sn[key][g, :, k]; lo, hi = ci(bs[key][:, g, :, k])
                print(f"  {nm:12s} {key:5s} submitted (run d): " + " ".join(f"{v:6.3f}" if np.isfinite(v) else "   -  " for v in o)
                      + "  | equal issuance: " + " ".join(f"{v:6.3f} [{a:.3f},{b:.3f}]" if np.isfinite(v) else "   -  " for v, a, b in zip(n_, lo, hi)))
# relative RMSE change of Foehn against each baseline, equal issuance
kF, kC, kA, kCl = SYSG.index(L), SYSG.index("CAMS"), SYSG.index("Aurora"), SYSG.index("climatology")
print("\n  Foehn RMSE relative to CAMS / Aurora / climatology over all land, equal issuance (%, with intervals):")
R["land_rmse_gain"] = {}
for nm, kk in (("CAMS", kC), ("Aurora", kA), ("climatology", kCl)):
    pt = 100 * (sn["rmse"][0, :, kF] / sn["rmse"][0, :, kk] - 1); b_ = 100 * (bs["rmse"][:, 0, :, kF] / bs["rmse"][:, 0, :, kk] - 1); lo, hi = ci(b_)
    old = 100 * (so["rmse"][0, :, kF] / so["rmse"][0, :, kk] - 1)
    R["land_rmse_gain"][nm] = {"submitted_run_d": f2(old), "equal_issuance": f2(pt), "ci_low": f2(lo), "ci_high": f2(hi)}
    print(f"    vs {nm:11s} submitted: " + " ".join(f"{v:+6.1f}" if np.isfinite(v) else "   -  " for v in old) + " | equal issuance: "
          + " ".join(f"{v:+.1f} [{a:+.1f},{b:+.1f}]" if np.isfinite(v) else "-" for v, a, b in zip(pt, lo, hi)))

# =========================================================================================== 3. land SEDI, the paper's thresholds
print("\n" + "=" * 110, "\n3. LAND, DETECTION OF LOCAL EXTREMES WITH THE PAPER'S THRESHOLDS (GHAP 2017-2020 local percentiles)\n", "=" * 110, sep="")
TS = [str(s) for s in te_n["systems"]]; TR = [str(r) for r in te_n["regions"]]; TT = ["P90", "P95", "P99"]
assert np.array_equal(te_d["C"][:, :, :, :4], te_n["C"][:, :, :, :4]), "Foehn's counts must not depend on the issuance of the baselines"
print("Foehn's counts (ensemble mean, one member, runs A and C) identical under both issuances: True")
print("persistence counts identical:", np.array_equal(te_d["C"][:, :, :, -1], te_n["C"][:, :, :, -1]))
vt = np.flatnonzero(te_n["CD"][:, 0, 0, 2, 0].sum(-1) > 0); idt = block_index(len(vt))
BT = boot_counts(te_n["CD"][vt], idt)                                  # (n, region, lead, tier, system, 4)
BS = sedi(BT)
iF, iC, iA = TS.index(f"{L} (mean)"), TS.index("CAMS"), TS.index("Aurora")
R["land_sedi_paper_thresholds"] = {}
for g, rn in enumerate(TR):
    R["land_sedi_paper_thresholds"][rn] = {}
    for kt in (2, 1, 0):
        d_ = {}
        if g == 0:
            print(f"\n--- {rn} | days above the local {TT[kt]}")
        for nm, k in (("Foehn (mean)", iF), ("CAMS", iC), ("Aurora", iA)):
            ro, rn_ = rates(te_d["C"][g, :, kt, k]), rates(te_n["C"][g, :, kt, k]); lo, hi = ci(BS[:, g, :, kt, k])
            d_[nm] = {"sedi_submitted_run_d": f2(ro["sedi"]), "sedi_equal_issuance": f2(rn_["sedi"]), "ci_low": f2(lo), "ci_high": f2(hi),
                      "pod": f2(rn_["pod"]), "far": f2(rn_["far"]), "freq_bias": f2(rn_["fb"]),
                      "pod_submitted": f2(ro["pod"]), "far_submitted": f2(ro["far"]), "freq_bias_submitted": f2(ro["fb"])}
            if g == 0:
                print(f"  {nm:13s} SEDI submitted: " + " ".join(f"{v:.3f}" if np.isfinite(v) else "  -  " for v in ro["sedi"])
                      + " | equal issuance: " + " ".join(f"{v:.3f}" if np.isfinite(v) else "  -  " for v in rn_["sedi"]))
                print(f"  {'':13s} equal issuance: hit rate " + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in rn_["pod"]) + " | false-alarm ratio "
                      + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in rn_["far"]) + " | frequency bias " + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in rn_["fb"]))
        # differences
        for nm, k in (("CAMS", iC), ("Aurora", iA)):
            pt = sedi(te_n["C"][g, :, kt, iF]) - sedi(te_n["C"][g, :, kt, k]); lo, hi = ci(BS[:, g, :, kt, iF] - BS[:, g, :, kt, k])
            d_[f"Foehn minus {nm}, equal lead"] = {"diff": f2(pt), "ci_low": f2(lo), "ci_high": f2(hi)}
            if g == 0:
                print(f"  Foehn - {nm:6s} at equal lead:      " + "  ".join(fmt(a, b, c) if np.isfinite(a) else "-" for a, b, c in zip(pt, lo, hi)))
            pt1 = sedi(te_n["C"][g, :, kt, iF]) - sedi(te_n["C"][g, 0, kt, k]); lo, hi = ci(BS[:, g, :, kt, iF] - BS[:, g, 0, kt, k][:, None])
            d_[f"Foehn at lead h minus {nm} at 1 day"] = {"diff": f2(pt1), "ci_low": f2(lo), "ci_high": f2(hi)}
            if g == 0:
                print(f"  Foehn(h) - {nm:6s}(1 day), h = 1..7: " + "  ".join(fmt(a, b, c) for a, b, c in zip(pt1, lo, hi)))
            if nm == "CAMS":
                po = sedi(te_d["C"][g, :, kt, iF]) - sedi(te_d["C"][g, 0, kt, k])
                d_["submitted: Foehn at lead h minus CAMS (run d) at 1 day"] = f2(po)
                if g == 0:
                    print("  submitted: Foehn(h) - CAMS run d (1 day): " + " ".join(f"{v:+.3f}" for v in po))
        R["land_sedi_paper_thresholds"][rn][TT[kt]] = d_
print("\n--- by region, days above the local P99, 1 day ahead (Extended Data Fig. 4h): SEDI submitted -> equal issuance; Foehn minus baseline with interval")
for g, rn in enumerate(TR):
    row = []
    for nm, k in (("Foehn", iF), ("CAMS", iC), ("Aurora", iA)):
        row.append(f"{nm} {sedi(te_d['C'][g, 0, 2, k]):.3f}->{sedi(te_n['C'][g, 0, 2, k]):.3f}")
    for nm, k in (("CAMS", iC), ("Aurora", iA)):
        a = sedi(te_n["C"][g, 0, 2, iF]) - sedi(te_n["C"][g, 0, 2, k]); lo, hi = ci(BS[:, g, 0, 2, iF] - BS[:, g, 0, 2, k])
        row.append(f"F-{nm} {fmt(a, lo, hi)}")
    print(f"  {rn:14s} " + " | ".join(row))
print("\n--- by region and lead, P99: Foehn minus CAMS / Aurora (equal lead, equal issuance); * interval excludes zero")
R["land_sedi_region_lead_P99"] = {}
for g, rn in enumerate(TR):
    R["land_sedi_region_lead_P99"][rn] = {}
    for nm, k in (("CAMS", iC), ("Aurora", iA)):
        a = sedi(te_n["C"][g, :5, 2, iF]) - sedi(te_n["C"][g, :5, 2, k]); lo, hi = ci(BS[:, g, :5, 2, iF] - BS[:, g, :5, 2, k])
        R["land_sedi_region_lead_P99"][rn][nm] = {"diff": f2(a), "ci_low": f2(lo), "ci_high": f2(hi)}
        print(f"  {rn:14s} F-{nm:6s} " + " ".join(f"{x:+.3f}{'*' if (l_ > 0 or h_ < 0) else ' '}" for x, l_, h_ in zip(a, lo, hi)))
# the transport sentence (A against C), not affected by the issuance
iA_, iC_ = TS.index("A (mean)"), TS.index("C (mean)")
print("\n--- 'without the transport the same network detects fewer extremes in every region 3 days ahead': SEDI P99, lead 3, A (with) / C (without)")
R["transport_P99_lead3"] = {}
for g, rn in enumerate(TR):
    a, c = sedi(te_n["C"][g, 2, 2, iA_]), sedi(te_n["C"][g, 2, 2, iC_]); lo, hi = ci(BS[:, g, 2, 2, iA_] - BS[:, g, 2, 2, iC_])
    R["transport_P99_lead3"][rn] = {"with": round(float(a), 3), "without": round(float(c), 3), "diff_ci": [round(float(lo), 3), round(float(hi), 3)]}
    print(f"  {rn:14s} {a:.3f} / {c:.3f}  diff {fmt(a - c, lo, hi)}")

# =========================================================================================== 4. land, frequency-matched
fmf = D / f"land_freqmatch_{L}.npz"
if fmf.exists():
    print("\n" + "=" * 110, "\n4. LAND, THRESHOLDS MATCHED IN FREQUENCY (equal issuance)\n", "=" * 110, sep="")
    fm = np.load(fmf); FS = [str(s) for s in fm["systems"]]; FK = [str(s) for s in fm["kinds"]]; FR = [str(s) for s in fm["regions"]]
    jF, jC, jA = FS.index(f"{L} (mean)"), FS.index("CAMS"), FS.index("Aurora")
    chk = np.array_equal(fm["C"][0][:, :, [jF, jC, jA]].transpose(3, 1, 0, 2, 4), te_n["C"][:, :, :, [iF, iC, iA]])
    print("kind 'ghap' recomputed from the dumped fields identical to tail_events (equal issuance):", chk)
    R["reproduction"]["freqmatch_ghap_kind_identical_to_tail_events"] = bool(chk)
    vf = np.flatnonzero(fm["CD"][:, 0, 2, 0, jF, 0].sum(-1) > 0); idf = block_index(len(vf))
    R["land_sedi_other_thresholds"] = {}
    for kind in (1, 2, 3):
        R["land_sedi_other_thresholds"][FK[kind]] = {}
        BF = sedi(boot_counts(fm["CD"][vf][:, kind], idf))             # (n, tier, lead, system, region)
        Ck = fm["C"][kind]
        for kt in (2, 1, 0):
            d_ = {}
            print(f"\n--- thresholds '{FK[kind]}' | above {TT[kt]} | all land")
            for nm, k in (("Foehn (mean)", jF), ("CAMS", jC), ("Aurora", jA)):
                r_ = rates(Ck[kt, :, k, 0]); lo, hi = ci(BF[:, kt, :, k, 0])
                d_[nm] = {"sedi": f2(r_["sedi"]), "ci_low": f2(lo), "ci_high": f2(hi), "pod": f2(r_["pod"]), "far": f2(r_["far"]), "freq_bias": f2(r_["fb"])}
                print(f"  {nm:13s} SEDI " + " ".join(f"{v:.3f}" if np.isfinite(v) else "  -  " for v in r_["sedi"]) + " | hit rate "
                      + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in r_["pod"]) + " | frequency bias " + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in r_["fb"]))
            for nm, k in (("CAMS", jC), ("Aurora", jA)):
                pt = sedi(Ck[kt, :, jF, 0]) - sedi(Ck[kt, :, k, 0]); lo, hi = ci(BF[:, kt, :, jF, 0] - BF[:, kt, :, k, 0])
                d_[f"Foehn minus {nm}, equal lead"] = {"diff": f2(pt), "ci_low": f2(lo), "ci_high": f2(hi)}
                print(f"  Foehn - {nm:6s} at equal lead:      " + "  ".join(fmt(a, b, c) if np.isfinite(a) else "-" for a, b, c in zip(pt, lo, hi)))
                pt1 = sedi(Ck[kt, :, jF, 0]) - sedi(Ck[kt, 0, k, 0]); lo, hi = ci(BF[:, kt, :, jF, 0] - BF[:, kt, 0, k, 0][:, None])
                d_[f"Foehn at lead h minus {nm} at 1 day"] = {"diff": f2(pt1), "ci_low": f2(lo), "ci_high": f2(hi)}
                print(f"  Foehn(h) - {nm:6s}(1 day), h = 1..7: " + "  ".join(fmt(a, b, c) for a, b, c in zip(pt1, lo, hi)))
            reg = {}
            for g, rn in enumerate(FR):
                reg[rn] = {nm: f2(sedi(Ck[kt, :, k, g])) for nm, k in (("Foehn (mean)", jF), ("CAMS", jC), ("Aurora", jA))}
                for nm, k in (("CAMS", jC), ("Aurora", jA)):
                    a = sedi(Ck[kt, :5, jF, g]) - sedi(Ck[kt, :5, k, g]); lo, hi = ci(BF[:, kt, :5, jF, g] - BF[:, kt, :5, k, g])
                    reg[rn][f"Foehn minus {nm}"] = {"diff": f2(a), "ci_low": f2(lo), "ci_high": f2(hi)}
                    if kt == 2:
                        print(f"    {rn:14s} F-{nm:6s} leads 1-5: " + " ".join(f"{x:+.3f}{'*' if (l_ > 0 or h_ < 0) else ' '}" for x, l_, h_ in zip(a, lo, hi)))
            d_["regions"] = reg
            R["land_sedi_other_thresholds"][FK[kind]][TT[kt]] = d_

# =========================================================================================== 5. stations
print("\n" + "=" * 110, "\n5. STATIONS OF THE SIX OFFICIAL NETWORKS (equal issuance, as submitted), THREE KINDS OF THRESHOLDS\n", "=" * 110, sep="")
SS = [str(s) for s in st["systems"]]; SK = [str(s) for s in st["kinds"]]; SN = [str(s) for s in st["nets"]]
CS, CDS = st["C"], st["CD"]                                            # C[kind, tier, lead, system, net, 4]
ids = block_index(CDS.shape[0])
R["stations_sedi"] = {}
for kind in range(3):
    R["stations_sedi"][SK[kind]] = {}
    BSt = sedi(boot_counts(CDS[:, kind], ids))                         # (n, tier, lead, system, net)
    for kt in (2, 1, 0):
        d_ = {}
        print(f"\n--- thresholds '{SK[kind]}' | above {TT[kt]} | all official stations ({int(st['NST'][kind, 0, -1])} stations at 1 day, "
              f"{int(CS[kind, kt, 0, 0, -1].sum())} station-days)")
        for s, nm in enumerate(SS):
            r_ = rates(CS[kind, kt, :, s, -1]); lo, hi = ci(BSt[:, kt, :, s, -1])
            d_[nm] = {"sedi": f2(r_["sedi"]), "ci_low": f2(lo), "ci_high": f2(hi), "pod": f2(r_["pod"]), "far": f2(r_["far"]), "freq_bias": f2(r_["fb"])}
            print(f"  {nm:13s} SEDI " + " ".join(f"{v:.3f}" if np.isfinite(v) else "  -  " for v in r_["sedi"]) + " | hit rate "
                  + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in r_["pod"]) + " | FAR " + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in r_["far"])
                  + " | freq. bias " + " ".join(f"{v:.2f}" if np.isfinite(v) else " - " for v in r_["fb"]))
        for nm, k in (("CAMS", 1), ("Aurora", 2)):
            pt = sedi(CS[kind, kt, :, 0, -1]) - sedi(CS[kind, kt, :, k, -1]); lo, hi = ci(BSt[:, kt, :, 0, -1] - BSt[:, kt, :, k, -1])
            d_[f"Foehn minus {nm}, equal lead"] = {"diff": f2(pt), "ci_low": f2(lo), "ci_high": f2(hi)}
            print(f"  Foehn - {nm:6s} at equal lead:      " + "  ".join(fmt(a, b, c) if np.isfinite(a) else "-" for a, b, c in zip(pt, lo, hi)))
            pt1 = sedi(CS[kind, kt, :, 0, -1]) - sedi(CS[kind, kt, 0, k, -1]); lo, hi = ci(BSt[:, kt, :, 0, -1] - BSt[:, kt, 0, k, -1][:, None])
            d_[f"Foehn at lead h minus {nm} at 1 day"] = {"diff": f2(pt1), "ci_low": f2(lo), "ci_high": f2(hi)}
            print(f"  Foehn(h) - {nm:6s}(1 day), h = 1..7: " + "  ".join(fmt(a, b, c) for a, b, c in zip(pt1, lo, hi)))
        nets = {}
        for n_, nn in enumerate(SN):
            nets[nn] = {nm: f2(sedi(CS[kind, kt, :, s, n_])) for s, nm in enumerate(SS)}
            for nm, k in (("CAMS", 1), ("Aurora", 2)):
                a = sedi(CS[kind, kt, :5, 0, n_]) - sedi(CS[kind, kt, :5, k, n_]); lo, hi = ci(BSt[:, kt, :5, 0, n_] - BSt[:, kt, :5, k, n_])
                nets[nn][f"Foehn minus {nm}"] = {"diff": f2(a), "ci_low": f2(lo), "ci_high": f2(hi)}
                if kt == 2:
                    print(f"    {nn:9s} F-{nm:6s} leads 1-5: " + " ".join(f"{x:+.3f}{'*' if (l_ > 0 or h_ < 0) else ' '}" for x, l_, h_ in zip(a, lo, hi))
                          + f"   (SEDI Foehn {sedi(CS[kind, kt, 0, 0, n_]):.2f}, {nm} {sedi(CS[kind, kt, 0, k, n_]):.2f} at 1 day)")
        d_["networks"] = nets
        R["stations_sedi"][SK[kind]][TT[kt]] = d_
(D / "results.json").write_text(json.dumps(R, indent=1))
print("\nsaved", D / "results.json")
