#!/usr/bin/env python3
"""Steps 3b and 4: calibration against the 1 km estimates.
(3b) populated pixels (data/sums_pop.npz, s3_pop.py; leads 1 and 3; >= 100 residents): spread-to-error, rank histogram,
     member range, reliability of the exceedance of 15 / 37.5 / 75 ug m-3, Brier skill against the pooled frequency of the
     region and against the frequency of 2022 at each pixel; plain and weighted by residents; by region.
(4)  land pixels of the nine regions (the 'recs' of the evaluation, as figures/paper_numbers.py l.52-61): spread-to-error
     by lead and region, 2022 (final model and the model before the CRPS training) and 2021 (validation year).
Intervals: 95 %, moving-block bootstrap over the start days (blocks of 7 start days, 10,000 resamples).
    python s34_estimates.py   -> results/estimates_scores.json, results/estimates_scores.txt"""
import glob
import json

import numpy as np

from rel_common import BEFORE, DATA, FINAL, HERE, block_weights

RES = HERE / "results"; RES.mkdir(exist_ok=True)
M = 4; F = (M + 1) / M
OUT = {}
lines = []
P = lines.append


def div(a, b):
    with np.errstate(all="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


def tl(a):
    a = np.asarray(a, float)
    return (None if not np.isfinite(a) else float(a)) if a.ndim == 0 else [tl(x) for x in a]


def ci(b):
    with np.errstate(all="ignore"):
        return [tl(np.nanpercentile(b, 2.5, 0)), tl(np.nanpercentile(b, 97.5, 0))]


# ------------------------------------------------------------------ (3b) populated pixels
Z = np.load(DATA / "sums_pop.npz")
K = Z["keys"]; REG = [str(r) for r in Z["regions"]]; THR = Z["thr"]; LEADS = [int(h) for h in Z["leads"]]
pk = np.arange(M + 1) / M


def pop_scores(S, pre=""):
    """S: dict of sums with leading batch dims; pre = '' (pixels) or 'w' (residents)"""
    g = lambda q: S[pre + q]
    n = g("n"); o = {}
    o["n"] = n
    o["ss"] = np.sqrt(F * div(g("var"), g("sse"))); o["rmse"] = np.sqrt(div(g("sse"), n)); o["bias"] = div(g("err"), n)
    o["crps"] = div(g("crps"), n); o["mae"] = div(g("sae"), n); o["spread"] = np.sqrt(F * div(g("var"), n))
    rk = g("rank"); o["rank"] = div(rk, rk.sum(-1, keepdims=True))
    o["cover"] = o["rank"][..., 1:M].sum(-1); o["below"] = o["rank"][..., 0]; o["above"] = o["rank"][..., M]
    rel = g("rel")                                               # (..., 3, k, o)
    nk = rel.sum(-1); ok_ = rel[..., 1]
    o["rel_n"] = nk; o["rel_freq"] = div(ok_, nk)
    base = div(ok_.sum(-1), nk.sum(-1)); o["base"] = base
    bs = div((nk * pk ** 2 - 2 * pk * ok_ + ok_).sum(-1), nk.sum(-1))        # sum (p - o)^2 = sum p^2 n - 2 p o + o
    o["bs"] = bs; o["bss_base"] = 1 - div(bs, base * (1 - base))
    bsf = bs - div((nk * pk * (1 - pk)).sum(-1), nk.sum(-1)) / (M - 1)
    o["bss_base_fair"] = 1 - div(bsf, base * (1 - base))
    fr = o["rel_freq"]
    o["rel_term"] = div((nk * np.nan_to_num(pk - fr) ** 2).sum(-1), nk.sum(-1))
    o["res_term"] = div((nk * np.nan_to_num(fr - base[..., None]) ** 2).sum(-1), nk.sum(-1))
    o["unc_term"] = base * (1 - base)
    # three members -> the member left out (a perfectly reliable ensemble) and -> the estimate
    j = np.arange(M)
    hoc = (j + 1) * nk[..., 1:] + (M - j) * nk[..., :-1]
    o["ho_ref"] = div((j + 1) * nk[..., 1:], hoc)
    o["ho_freq"] = div((j + 1) * ok_[..., 1:] + (M - j) * ok_[..., :-1], hoc)
    return o


NAMES = [k for k in Z.files if k not in ("keys", "regions", "thr", "sd_edges", "leads", "pixref", "pixref_loo", "n_pixels", "residents")
         and not k.startswith("sd_")]
OUT["pop"] = {"regions": REG, "thresholds": [float(t) for t in THR], "n_pixels": Z["n_pixels"].tolist(),
              "residents": Z["residents"].tolist(), "leads": {}}
P("CALIBRATION AGAINST THE 1 km ESTIMATES. Source: data/sums_pop.npz (scripts/s3_pop.py) and the recs of the evaluation part files; "
  "scored by scripts/s34_estimates.py. Intervals: 95 %, block bootstrap over start days (7-start-day blocks, 10,000).")
P("")
P(f"== P. Populated 1 km pixels (>= 100 residents; {int(Z['n_pixels'][0]):,} pixels, {Z['residents'][0] / 1e9:.2f} billion residents), four members, 179 start days")
for li, h in enumerate(LEADS):
    rows = np.where(K[:, 1] == h)[0]; rows = rows[np.argsort(K[rows, 0])]
    W = block_weights(len(rows), B=7, n=10000, seed=0)
    OUT["pop"]["leads"][h] = {}
    for r, rname in enumerate(REG):
        S = {k: Z[k][rows, r] for k in NAMES}
        pt = {pre: pop_scores({k: v.sum(0) for k, v in S.items()}, pre) for pre in ("", "w")}
        bo = {pre: pop_scores({k: np.tensordot(W, v, axes=(1, 0)) for k, v in S.items()}, pre) for pre in ("", "w")}
        rec = {}
        for pre, nm in (("", "pixels"), ("w", "residents")):
            rec[nm] = {k: {"v": tl(v), "ci_day": ci(bo[pre][k])} for k, v in pt[pre].items()}
        # Brier skill against the frequency of 2022 at each pixel (in-sample: the best constant forecast of each pixel)
        bs_sum = np.asarray(pt[""]["bs"]) * np.asarray(pt[""]["rel_n"]).sum(-1)
        rec["pixels"]["bss_pixel_frequency"] = {"v": tl(1 - div(bs_sum, Z["pixref"][li, :, r]))}
        rec["pixels"]["bss_pixel_frequency_loo"] = {"v": tl(1 - div(bs_sum, Z["pixref_loo"][li, :, r]))}
        OUT["pop"]["leads"][h][rname] = rec
for rname in REG[::-1]:
    P(f"-- {rname}")
    for h in LEADS:
        for nm in ("pixels", "residents"):
            r = OUT["pop"]["leads"][h][rname][nm]
            f2 = lambda e, d=3: f"{e['v']:.{d}f} [{e['ci_day'][0]:.{d}f}, {e['ci_day'][1]:.{d}f}]"
            P(f"  lead {h} ({nm}): n {r['n']['v']:.4g} | spread/error {f2(r['ss'])} | RMSE {r['rmse']['v']:.2f} bias {r['bias']['v']:+.2f} CRPS {r['crps']['v']:.2f} | rank "
              + " / ".join(f"{x:.3f}" for x in r["rank"]["v"]) + f" | inside {f2(r['cover'])} below {r['below']['v']:.3f} above {r['above']['v']:.3f}")
            for j, T in enumerate(THR):
                s = (f"      > {T:g}: base {r['base']['v'][j]:.4f} | n per bin " + " / ".join(f"{x:.3g}" for x in r["rel_n"]["v"][j]) + " | observed "
                     + " / ".join("n/a" if x is None else f"{x:.3f}" for x in r["rel_freq"]["v"][j])
                     + " | CI low " + " / ".join("n/a" if x is None else f"{x:.3f}" for x in r["rel_freq"]["ci_day"][0][j])
                     + " high " + " / ".join("n/a" if x is None else f"{x:.3f}" for x in r["rel_freq"]["ci_day"][1][j])
                     + f" | BSS vs pooled frequency {r['bss_base']['v'][j]:+.3f} [{r['bss_base']['ci_day'][0][j]:+.3f}, {r['bss_base']['ci_day'][1][j]:+.3f}] fair {r['bss_base_fair']['v'][j]:+.3f}")
                if nm == "pixels":
                    s += f" | BSS vs the 2022 frequency of each pixel {r['bss_pixel_frequency']['v'][j]:+.3f} (each day left out: {r['bss_pixel_frequency_loo']['v'][j]:+.3f})"
                s += (f" | rel/res/unc {r['rel_term']['v'][j]:.4f} / {r['res_term']['v'][j]:.4f} / {r['unc_term']['v'][j]:.4f} | three members -> fourth "
                      + " / ".join("n/a" if x is None else f"{x:.3f}" for x in r["ho_ref"]["v"][j]) + " -> estimate "
                      + " / ".join("n/a" if x is None else f"{x:.3f}" for x in r["ho_freq"]["v"][j]))
                P(s)
P("")
P("== P2. Spread against error in bins of ensemble spread, all populated pixels: bin (ug m-3), share of pixel-days, spread, RMSE against the estimates")
OUT["pop"]["spread_bins"] = {}
for h in LEADS:
    k = K[:, 1] == h
    sn, sv, se = (Z[q][k].sum(0) for q in ("sd_n", "sd_var", "sd_sse"))
    OUT["pop"]["spread_bins"][h] = {"edges": Z["sd_edges"].tolist(), "share": tl(sn / sn.sum()), "spread": tl(np.sqrt(F * div(sv, sn))), "rmse": tl(np.sqrt(div(se, sn)))}
    P(f"-- lead {h}: " + " | ".join(f"{Z['sd_edges'][b]:g}-{Z['sd_edges'][b + 1]:g}: {100 * sn[b] / sn.sum():.1f} % {np.sqrt(F * sv[b] / sn[b]):.2f} / {np.sqrt(se[b] / sn[b]):.2f}"
                                    for b in range(len(sn)) if sn[b] > 0))

# ------------------------------------------------------------------ (4) land pixels of the nine regions (recs)
P("")
P("== L. Land pixels of the nine regions (recs; the ratio of the manuscript = all nine regions pooled, a pixel in two boxes counted twice)")
OUT["land"] = {}
for tag, ev in (("2022 final", FINAL / "eval_final_crps_e01"), ("2021 final (validation year)", FINAL / "eval2021_sel_crps_e01"),
                ("2022 before CRPS training", BEFORE / "eval_final_e02"), ("2021 before CRPS training", BEFORE / "eval2021_sel_e02")):
    parts = sorted(glob.glob(str(ev / "part_*.npz")))
    if not parts:
        P(f"-- {tag}: no part files in {ev}"); continue
    z0 = np.load(parts[0]); C = [str(c) for c in z0["rec_cols"]]; Mx = int(z0["info"][5]); regs = [str(r) for r in z0["regions"]]
    R = np.concatenate([np.load(p)["recs"] for p in parts])
    c = lambda q: C.index(q)
    fac = (Mx + 1) / Mx
    days = np.unique(R[:, c("start_doy")]).astype(int)
    OUT["land"][tag] = {"members": Mx, "start_days": int(len(days)), "regions": regs + ["Nine regions pooled"], "leads": {}}
    P(f"-- {tag}: {ev.name}, {Mx} members, {len(days)} start days, {len(R)} rows")
    for h in range(1, 8):
        rh = R[R[:, c("lead")] == h]
        # (start day, region) sums
        T = np.zeros((len(days), len(regs) + 1, 5))                        # n, sse, var, crps, err
        di = np.searchsorted(days, rh[:, c("start_doy")].astype(int)); ri = rh[:, c("region")].astype(int)
        vals = np.stack([rh[:, c(q)] for q in ("n", "sse", "var", "crps", "sum_err")], 1)
        np.add.at(T, (di, ri), vals); T[:, -1] = T[:, :-1].sum(1)
        ok = T[:, -1, 0] > 0
        W = block_weights(int(ok.sum()), B=7, n=10000, seed=0) if ok.sum() >= 14 else None
        rec = {}
        for r, rname in enumerate(regs + ["Nine regions pooled"]):
            s = T[ok, r].sum(0)
            e = {"n": float(s[0]), "ss": float(np.sqrt(fac * s[2] / s[1])), "rmse": float(np.sqrt(s[1] / s[0])), "crps": float(s[3] / s[0]),
                 "bias": float(s[4] / s[0]), "spread": float(np.sqrt(fac * s[2] / s[0]))}
            if W is not None:
                b = W @ T[ok, r]
                e["ss_ci"] = ci(np.sqrt(fac * b[:, 2] / b[:, 1]))
            rec[rname] = e
        OUT["land"][tag]["leads"][h] = rec
    L_ = OUT["land"][tag]["leads"]
    for rname in ["Nine regions pooled"] + regs:
        P(f"  {rname:20s} spread/error by lead: " + "  ".join(f"{L_[h][rname]['ss']:.2f}" + (f" [{L_[h][rname]['ss_ci'][0]:.2f}, {L_[h][rname]['ss_ci'][1]:.2f}]" if "ss_ci" in L_[h][rname] else "")
                                                               for h in range(1, 8)))
    P("  pooled CRPS by lead: " + "  ".join(f"{L_[h]['Nine regions pooled']['crps']:.2f}" for h in range(1, 8))
      + " | RMSE: " + "  ".join(f"{L_[h]['Nine regions pooled']['rmse']:.2f}" for h in range(1, 8)))

with open(RES / "estimates_scores.json", "w") as f:
    json.dump(OUT, f)
(RES / "estimates_scores.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
