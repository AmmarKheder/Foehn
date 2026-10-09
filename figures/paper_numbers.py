#!/usr/bin/env python3
"""Every number of the main text that depends on the final model, from the analysis files (to update the text after a
change of model). python paper_numbers.py   (model: final_model.LABEL)"""
import glob

import numpy as np

import final_model as FM
from fig_gc3_scorecards import block_bootstrap
from fig_gc4_extremes import rev_curves, sedi
from nature_style import HERE

L = FM.LABEL
print("=== final model:", L, FM.M["run"], FM.M["eval"])
# ---- stations
fz = np.load(HERE / "data" / f"compare_fair_{L}.npz"); SYS = [str(s) for s in fz["systems"]]; S, PDN, SST = fz["S"], fz["PDN"], fz["SST"]
NETS = list(fz["nets"]); g = S.shape[0] - 1
I = {k: SYS.index(v) for k, v in (("F", f"{L} (1 km)"), ("F25", f"{L} at 0.25 deg"), ("F50", f"{L} at 0.5 deg"),
                                   ("C", "CAMS (run d+1, equal issuance)"), ("A", "Aurora (00 UTC d+1, equal issuance)"),
                                   ("P", "persistence (day d)"), ("G", "GHAP persistence (day d)"))}
r = lambda g_, h, i: np.sqrt(S[g_, h, i, 2] / S[g_, h, i, 0]); b = lambda g_, h, i: S[g_, h, i, 1] / S[g_, h, i, 0]
print("RMSE all stations J+1..4:", {k: [round(r(g, h, i), 1) for h in range(4)] for k, i in I.items()})
print("station-days per lead:", [int(S[g, h, I["F"], 0]) for h in range(4)])
for k in ("C", "A", "G", "P", "F25"):
    print(f"Foehn vs {k}:", [tuple(np.round(block_bootstrap(PDN[I['F'], g, :, h], PDN[I[k], g, :, h]), 1)) for h in range(4)])
for gi, nm in enumerate(NETS[:-1]):
    print(f"  {nm:18s} RMSE F/C/A J+1 {r(gi, 0, I['F']):.1f}/{r(gi, 0, I['C']):.1f}/{r(gi, 0, I['A']):.1f} J+4 {r(gi, 3, I['F']):.1f}/{r(gi, 3, I['C']):.1f}/"
          f"{r(gi, 3, I['A']):.1f} | bias F/C/A J+1 {b(gi, 0, I['F']):+.1f}/{b(gi, 0, I['C']):+.1f}/{b(gi, 0, I['A']):+.1f} | "
          f"vs C J+1 {block_bootstrap(PDN[I['F'], gi, :, 0], PDN[I['C'], gi, :, 0])[0]:+.0f}% vs A J+1 {block_bootstrap(PDN[I['F'], gi, :, 0], PDN[I['A'], gi, :, 0])[0]:+.0f}%")
n = SST[I["F"], 0, :, 1]; ok = n >= 30
with np.errstate(all="ignore"):
    ch = np.sqrt(SST[I["F"], 0, :, 0] / np.maximum(n, 1)) / np.sqrt(SST[I["C"], 0, :, 0] / np.maximum(SST[I["C"], 0, :, 1], 1)) - 1
print(f"better than CAMS at {100 * np.mean(ch[ok] < 0):.0f}% of {ok.sum()} stations (J+1)")
# ---- extremes at the stations
es = np.load(HERE / "data" / f"extremes_stations_{L}.npz"); CE = es["C"]; ES = [str(s) for s in es["systems"]]
al = np.logspace(np.log10(0.003), np.log10(0.6), 200)
for tier, nm in ((1, "P95"), (2, "P99")):
    print(nm, "REV at alpha 0.05 J+1:", {k: round(float(np.interp(0.05, al, rev_curves(CE[0, tier, 0, ES.index(k)], al))), 3) for k in ES},
          "max REV J+1:", {k: round(float(np.max(rev_curves(CE[0, tier, 0, ES.index(k)], al))), 3) for k in ES})
def roc(cnt):
    H = np.concatenate([[1.0], np.cumsum(cnt[::-1, 1])[::-1][1:] / cnt[:, 1].sum(), [0.0]])
    F = np.concatenate([[1.0], np.cumsum(cnt[::-1, 0])[::-1][1:] / cnt[:, 0].sum(), [0.0]])
    return -np.trapezoid(H, F)
print("ROC P95 stations J+1..4:", {k: [round(roc(CE[0, 1, h, ES.index(k)]), 3) for h in range(4)] for k in ES})
# ---- extremes over land
te = np.load(HERE / "data" / f"tail_events_{L}.npz"); CT = te["C"]; TS = [str(s) for s in te["systems"]]; TR = [str(x) for x in te["regions"]]
gg = TR.index("Global")
for k in (f"{L} (mean)", "CAMS", "Aurora", "persistence"):
    h_, m_, f_, c_ = [CT[gg, 0, 2, TS.index(k), q] for q in range(4)]
    print(f"P99 land {k:14s} SEDI by lead", np.round(sedi(CT[gg, :, 2, TS.index(k)]), 2), f"| J+1 POD {h_ / (h_ + m_):.2f} FAR {f_ / (h_ + f_):.2f}")
# ---- calibration
for nm, ev in (("2021", FM.SEL21_DIR), ("2022", FM.EVAL_DIR)):
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    if not parts:
        continue
    z0 = np.load(parts[0]); C = list(z0["rec_cols"]); M = int(z0["info"][5]); R = np.concatenate([np.load(p)["recs"] for p in parts])
    out = []
    for h in range(1, 8):
        rr = R[R[:, C.index("lead")] == h]; n_, sse, var, crps = (rr[:, C.index(c)].sum() for c in ("n", "sse", "var", "crps"))
        out.append(f"J+{h} ss {np.sqrt((M + 1) / M * var / n_) / np.sqrt(sse / n_):.2f} CRPS {crps / n_:.2f} RMSE {np.sqrt(sse / n_):.2f}")
    print(f"GHAP {nm}:", " | ".join(out))
# ---- January case
cz = glob.glob(str(HERE / "data" / f"case_eastasia_{FM.M['run']}.npz"))
if cz:
    z = np.load(cz[0]); W = z["warn"]
    print("January warnings CSI by lead (Foehn/CAMS/Aurora):", [tuple(round(W[h, k, 0] / W[h, k].sum(), 3) if W[h, k].sum() else None for k in range(3)) for h in range(7)])
    print("peak", int(z["peak"]), "BTH obs", round(float(z["convergence"][0, 0]), 1), "Foehn mean by lead", np.round(z["convergence"][:, 3:].mean(1), 1))
