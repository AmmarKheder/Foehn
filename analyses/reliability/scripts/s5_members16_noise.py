#!/usr/bin/env python3
"""Step 5: (A) the 16-member evaluations at the stations (case-study runs of the final model: East Asia, 28 start days of
January 2022, leads 1-7; Europe window of the Gironde case, 12 start days of July 2022 at leads 1-3 and 15 at leads 4-7):
member range against its expected value (M - 1) / (M + 1) for 4 and for 16 members, spread-to-error, rank histogram,
reliability with 17 probability values; (B) the reuse of the same noise fields for every forecast:
how fixed the order of the members is at a station, and what the choice of the noise fields changes for the pooled scores
(every quartet of the 16 noise fields as a four-member ensemble).
Sample of (A): official stations with >= 30 valid days, all members finite (the station lies in a window that was run),
measurement and 1 km estimate of the target day available. v = 0 against the measurement, v = 1 against the estimate.
    python s5_members16_noise.py   -> results/members16_noise.json, results/members16_noise.txt"""
import itertools
import json

import numpy as np

from rel_common import DATA, HERE, NETS, SFX, THR, block_weights, fair_crps, stations

RES = HERE / f"results{SFX}"; RES.mkdir(exist_ok=True)
ST = stations(); OBS, KEEP, SRC = ST["obs"], ST["keep"], ST["src"]
lines = []
P = lines.append
OUT = {"A": {}, "B": {}}


def tl(a):
    a = np.asarray(a, float)
    return (None if not np.isfinite(a) else float(a)) if a.ndim == 0 else [tl(x) for x in a]


def scores(E, tv, w):
    """E (n, s, M), tv (n, s), w (n, s) bool -> sums per forecast (n, ...)"""
    M = E.shape[-1]
    Ez = np.where(w[..., None], E, 0.0); t = np.where(w, tv, 0.0)
    mean = Ez.mean(-1); var = Ez.var(-1, ddof=1)
    lt = (Ez < t[..., None]).sum(-1)
    S = {"n": w.sum(1).astype(float), "var": (var * w).sum(1), "sse": (((mean - t) ** 2) * w).sum(1), "err": ((mean - t) * w).sum(1),
         "crps": (fair_crps(Ez, t) * w).sum(1), "sae": (np.abs(mean - t) * w).sum(1)}
    S["rank"] = np.stack([((lt == k) & w).sum(1) for k in range(M + 1)], 1).astype(float)
    rel = []
    for T in THR:
        kT = (Ez > T).sum(-1); o = t > T
        rel.append(np.stack([np.stack([((kT == k) & w & ~o).sum(1), ((kT == k) & w & o).sum(1)], 1) for k in range(M + 1)], 1))
    S["rel"] = np.stack(rel, 1).astype(float)                    # (n, 3, k, 2)
    return S


def derive(S, M):
    o = {}
    f = (M + 1) / M
    with np.errstate(all="ignore"):
        o["n"] = S["n"]; o["ss"] = np.sqrt(f * S["var"] / S["sse"]); o["rmse"] = np.sqrt(S["sse"] / S["n"]); o["bias"] = S["err"] / S["n"]
        o["crps"] = S["crps"] / S["n"]; o["mae"] = S["sae"] / S["n"]; o["spread"] = np.sqrt(f * S["var"] / S["n"])
        rk = S["rank"] / S["rank"].sum(-1, keepdims=True)
        o["rank"] = rk; o["cover"] = rk[..., 1:M].sum(-1); o["below"] = rk[..., 0]; o["above"] = rk[..., M]
        nk = S["rel"].sum(-1); ok_ = S["rel"][..., 1]
        o["rel_n"] = nk; o["rel_freq"] = ok_ / nk
        base = ok_.sum(-1) / nk.sum(-1); pk = np.arange(M + 1) / M
        bs = (nk * pk ** 2 - 2 * pk * ok_ + ok_).sum(-1) / nk.sum(-1)
        o["base"] = base; o["bss_base"] = 1 - bs / (base * (1 - base))
    return o


SETS = {"ea16": "East Asia window, start days 1-28 January 2022", "gir16": "Europe window (Gironde case), 12 start days of July 2022, leads 1-3",
        "girlong16": "Europe window (Gironde case), 15 start days of July 2022, leads 4-7"}
P("16-MEMBER EVALUATIONS AT THE STATIONS AND THE REUSE OF THE NOISE FIELDS. Source: data/st_{ea16,gir16,girlong16,final}.npz (s1_assemble.py), "
  "scored by scripts/s5_members16_noise.py. v=0 against the measurements, v=1 against the 1 km estimate at the same pixel.")
P("")
P("== A. Member range, spread-to-error, rank histogram: 16 members and their first four (the four noise fields of the 4-member evaluation)")
QUART = np.array(list(itertools.combinations(range(16), 4)))
for tag, desc in SETS.items():
    z = np.load(DATA / f"st_{tag}.npz"); K = z["keys"]
    E = z["ens"].astype(np.float32); G_ = z["ghap"].astype(np.float32)
    Y = OBS[:, K[:, 2]].T
    col = KEEP & np.isfinite(E).all((0, 2))                      # stations inside the windows that were run
    E, G_, Y, src = E[:, col], G_[:, col], Y[:, col], SRC[col]
    w = np.isfinite(Y) & np.isfinite(G_)
    nets = {NETS[s]: int(((src == s)).sum()) for s in NETS if (src == s).any()}
    OUT["A"][tag] = {"what": desc, "stations": int(col.sum()), "networks": nets, "forecasts": int(len(K)), "leads": {}}
    P(f"-- {tag}: {desc}; {int(col.sum())} official stations ({', '.join(f'{k} {v}' for k, v in nets.items())}), {len(K)} forecasts")
    leads = sorted(set(K[:, 1].tolist()))
    for hsel, hname in [([h], str(h)) for h in leads] + [(leads, "all")]:
        rows = np.where(np.isin(K[:, 1], hsel))[0]
        rec = {}
        for v, tv in enumerate((Y, G_)):
            for M_, Esub in ((16, E), (4, E[..., :4])):
                S = scores(Esub[rows], tv[rows], w[rows])
                pt = derive({k: x.sum(0) for k, x in S.items()}, M_)
                e = {k: tl(x) for k, x in pt.items()}
                # bootstrap over start days (blocks of 7), all leads of a start day kept together
                days = np.unique(K[rows, 0]); di = np.searchsorted(days, K[rows, 0])
                if len(days) >= 14:
                    Wd = block_weights(len(days), B=7, n=4000, seed=0)[:, di]            # (n, forecasts)
                    bo = derive({k: np.tensordot(Wd, x, axes=(1, 0)) for k, x in S.items()}, M_)
                    for q in ("ss", "cover", "crps", "below", "above"):
                        e[q + "_ci"] = [float(np.nanpercentile(bo[q], 2.5)), float(np.nanpercentile(bo[q], 97.5))]
                rec[f"M{M_}_v{v}"] = e
        OUT["A"][tag]["leads"][hname] = rec
        if hname in ("1", "3", "7", "all", "4"):
            for v, nm in ((0, "measurements"), (1, "estimates   ")):
                a, b = rec[f"M16_v{v}"], rec[f"M4_v{v}"]
                c = lambda e, q, d=3: f"{e[q]:.{d}f}" + (f" [{e[q + '_ci'][0]:.{d}f}, {e[q + '_ci'][1]:.{d}f}]" if q + "_ci" in e else "")
                P(f"  lead {hname:>3} vs {nm}: n {int(a['n']):,} | 16 members: inside the range {c(a, 'cover')} (expected 0.882), below {a['below']:.3f} above {a['above']:.3f}, spread/error {c(a, 'ss')}, CRPS {a['crps']:.2f}"
                  f" | first 4 members: inside {c(b, 'cover')} (expected 0.600), below {b['below']:.3f} above {b['above']:.3f}, spread/error {c(b, 'ss')}, CRPS {b['crps']:.2f}")
    # 17-bin rank histogram and reliability, all leads pooled
    for v, nm in ((0, "measurements"), (1, "estimates")):
        a = OUT["A"][tag]["leads"]["all"][f"M16_v{v}"]
        P(f"  all leads vs {nm}: rank histogram of 17 bins (0.059 each if exchangeable): " + " ".join(f"{x:.3f}" for x in a["rank"]))
        for j, T in enumerate(THR):
            nk = np.array(a["rel_n"][j]); fr = np.array([np.nan if x is None else x for x in a["rel_freq"][j]])
            grp = [(0, 0), (1, 5), (6, 10), (11, 15), (16, 16)]
            with np.errstate(all="ignore"):
                P(f"      > {T:g}: base {a['base'][j]:.3f}, BSS vs pooled frequency {a['bss_base'][j]:+.3f} | members above 0 / 1-5 / 6-10 / 11-15 / 16: n "
                  + " / ".join(f"{int(nk[lo:hi + 1].sum()):,}" for lo, hi in grp) + " | mean forecast probability "
                  + " / ".join(f"{(nk[lo:hi + 1] * np.arange(lo, hi + 1) / 16).sum() / max(nk[lo:hi + 1].sum(), 1):.3f}" for lo, hi in grp) + " | observed "
                  + " / ".join(f"{np.nansum(nk[lo:hi + 1] * fr[lo:hi + 1]) / max(nk[lo:hi + 1].sum(), 1):.3f}" for lo, hi in grp))
    # ---- (B) every quartet of the 16 noise fields as a four-member ensemble (all leads pooled)
    wq = w; t0 = np.where(wq, Y, 0.0); t1 = np.where(wq, G_, 0.0); n = wq.sum()
    res = np.zeros((len(QUART), 2, 4))                           # per quartet and verification: crps, rmse of the mean, spread/error, range
    Ez = np.where(wq[..., None], E, 0.0)
    for qi, q in enumerate(QUART):
        Eq = Ez[..., list(q)]
        mean = Eq.mean(-1); var = (Eq.var(-1, ddof=1) * wq).sum()
        for v, t in enumerate((t0, t1)):
            sse = (((mean - t) ** 2) * wq).sum()
            inside = (((t >= Eq.min(-1)) & (t <= Eq.max(-1))) & wq).sum()
            res[qi, v] = [(fair_crps(Eq, t) * wq).sum() / n, np.sqrt(sse / n), np.sqrt(1.25 * var / sse), inside / n]
    first = int(np.where((QUART == np.arange(4)).all(1))[0][0])
    disj = [int(np.where((QUART == np.arange(4 * k, 4 * k + 4)).all(1))[0][0]) for k in range(4)]
    OUT["B"][tag] = {"quartets": int(len(QUART)), "stat": ["crps", "rmse_mean", "spread_error", "member_range"],
                     "mean": tl(res.mean(0)), "sd": tl(res.std(0)), "sd_independent": tl(res.std(0) / np.sqrt(1 - 4 / 16)),
                     "min": tl(res.min(0)), "max": tl(res.max(0)), "members_0_3": tl(res[first]), "disjoint_quartets": tl(res[disj]),
                     "note": "sd over the 1,820 quartets of 16 fields drawn without replacement; sd_independent = sd / sqrt(1 - 4/16), the value for independent draws"}
    P(f"  noise fields ({tag}, all leads pooled, {int(n):,} station-days): every quartet of the 16 fields as a 4-member ensemble (1,820 quartets)")
    for v, nm in ((0, "measurements"), (1, "estimates   ")):
        r = res[:, v]
        P(f"      vs {nm}: " + " | ".join(f"{q} mean {r[:, i].mean():.3f}, sd {r[:, i].std():.3f} ({100 * r[:, i].std() / r[:, i].mean():.1f} %; independent draws {100 * r[:, i].std() / np.sqrt(0.75) / r[:, i].mean():.1f} %), "
                                             f"range {r[:, i].min():.3f}-{r[:, i].max():.3f}, members 0-3: {res[first, v, i]:.3f}, four disjoint quartets: " + " ".join(f"{res[d, v, i]:.3f}" for d in disj)
                                             for i, q in enumerate(("CRPS", "RMSE of the mean", "spread/error", "member range"))))

# ------------------------------------------------------------------ (B) the fixed order of the members, 4-member evaluation of the year
P("")
P("== B. The same noise fields for every forecast (4-member evaluation of 2022, official stations with >= 30 valid days)")
z = np.load(DATA / "st_final.npz"); K = z["keys"]; E = z["ens"][:, KEEP].astype(np.float32)
mean = E.mean(-1, keepdims=True); sd = E.std(-1, ddof=1, keepdims=True)
with np.errstate(all="ignore"):
    Zs = (E - mean) / np.where(sd > 0, sd, np.nan)               # standardised departure of each member (forecast, station, member)
OUT["B"]["year"] = {}
rng = np.random.default_rng(0)
for h in (1, 4, 7):
    rows = np.where(K[:, 1] == h)[0]
    top = E[rows].argmax(-1)                                     # (forecasts, stations)
    share = np.stack([(top == m).mean(0) for m in range(4)]).max(0)   # share of the forecasts in which the most frequent top member is top
    zt = Zs[rows]
    zm = np.nanmean(zt, 0)                                       # time mean of the standardised departure (station, member)
    tot = np.nanvar(zt); fixed = np.nanvar(zm)
    # pattern correlation of the standardised departures between two different start days (pairs at random)
    pc = []
    for _ in range(300):
        a, b = rng.choice(len(rows), 2, replace=False)
        x, y = zt[a].ravel(), zt[b].ravel(); ok = np.isfinite(x) & np.isfinite(y)
        pc.append(np.corrcoef(x[ok], y[ok])[0, 1])
    OUT["B"]["year"][h] = {"same_top_member_share_median": float(np.median(share)), "p10": float(np.percentile(share, 10)), "p90": float(np.percentile(share, 90)),
                           "mean": float(share.mean()), "fixed_share_of_variance": float(fixed / tot), "sd_of_time_mean_departure": float(np.sqrt(fixed)),
                           "pattern_correlation_between_start_days": [float(np.mean(pc)), float(np.percentile(pc, 2.5)), float(np.percentile(pc, 97.5))]}
    P(f"  lead {h}: share of the {len(rows)} forecasts in which the same member is the highest of the four at a station: median station {np.median(share):.3f}, "
      f"10th percentile {np.percentile(share, 10):.3f}, mean {share.mean():.3f} (exchangeable members: about 0.25-0.33) | the time mean of the standardised departure of a member "
      f"holds {100 * fixed / tot:.1f} % of its variance (sd across stations and members {np.sqrt(fixed):.2f}) | pattern correlation of the departures between two start days {np.mean(pc):.3f} "
      f"[{np.percentile(pc, 2.5):.3f}, {np.percentile(pc, 97.5):.3f}]")
# between lead 1 and lead 7 of the same start day
d1 = {int(d): n for n, (d, h, t) in enumerate(K) if h == 1}; d7 = {int(d): n for n, (d, h, t) in enumerate(K) if h == 7}
pc = []
for d in sorted(set(d1) & set(d7))[::4]:
    x, y = Zs[d1[d]].ravel(), Zs[d7[d]].ravel(); ok = np.isfinite(x) & np.isfinite(y)
    pc.append(np.corrcoef(x[ok], y[ok])[0, 1])
OUT["B"]["year"]["lead1_vs_lead7_same_start_day"] = [float(np.mean(pc)), float(np.min(pc)), float(np.max(pc))]
P(f"  pattern correlation of the departures between lead 1 and lead 7 of the same start day: {np.mean(pc):.3f} (range {np.min(pc):.3f}-{np.max(pc):.3f}, {len(pc)} start days)")
with open(RES / "members16_noise.json", "w") as f:
    json.dump(OUT, f)
(RES / "members16_noise.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
