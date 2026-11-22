#!/usr/bin/env python3
"""A16 law check, regional add-on (CPU, reads only): for small boxes (Gironde, Korea, NCP) the member's box-level departure is
driven by the mean of its noise over the box. Under the stored design each index has ONE such mean u_m for the whole year
(16 values); under 'ar' a fresh one per start. Measures, per box: the 16 stored u_m (white noise: u ~ N(0, 1), so
15 s^2 ~ chi2_15), the within-start relation of the ar member box level to u at lead 1 (179 starts x 16 members), and
whether the dispersion of the stored u_m predicts the observed ar/stored ratio of the between-member variance of box
levels. Adds the key 'f_region_box_noise' to lawcheck.json (temporary file, fsync, read-back, rename)."""
import json
from pathlib import Path

import numpy as np

from lawcheck import G, LC, REGIONS, RNAMES, innovation, jsonable, save_json

BOXES = ("GirondeBox", "Korea", "NCP")


def main():
    z = np.load(LC / "pop_stats.npz")
    keys = z["keys"]
    pop = np.load(G / "pop_pixels.npz")["idx"]
    s16 = np.load(G / "noise_tiles16.npy", mmap_mode="r")
    out = {"what": __doc__, "source": [str(LC / "pop_stats.npz"), str(G / "noise_tiles16.npy"), "Philox innovations eps_1 "
                                        "(a16/arnoise22.py keys [1521, 2022, d, m, 1, tr, tc])"]}
    for nm in BOXES:
        n, s, w, e = REGIONS[nm]
        r = RNAMES.index(nm)
        rows, cols = pop // 36000, pop % 36000
        lat = 84.0 - 0.01 * (rows + 0.5); lon = -180.0 + 0.01 * (cols + 0.5)
        sel = (lat <= n) & (lat > s) & (lon >= w) & (lon < e)
        pr, pc = rows[sel], cols[sel]
        tr, tc = pr // 512, pc // 512
        tiles = sorted(set(zip(tr.tolist(), tc.tolist())))
        grp = {t: np.flatnonzero((tr == t[0]) & (tc == t[1])) for t in tiles}
        npx = int(sel.sum())

        def u_of(get):
            acc = 0.0
            for t, ix in grp.items():
                f = np.asarray(get(*t), np.float32)
                acc += float(f[pr[ix] % 512, pc[ix] % 512].astype(np.float64).sum())
            return acc / np.sqrt(npx)                                # N(0, 1) for white noise
        us = np.array([u_of(lambda a, b, m=m: s16[m, a, b]) for m in range(16)])
        l1 = np.flatnonzero(keys[:, 1] == 1)
        starts = keys[l1, 0]
        ua = np.array([[u_of(lambda a, b, d=d, m=m: innovation(2022, int(d), m, 1, a, b).astype(np.float16)) for m in range(16)]
                       for d in starts])                             # (179, 16)
        res = {"populated_pixels": npx, "tiles": len(tiles)}
        res["stored_u"] = dict(values=us, mean=us.mean(), sd=us.std(ddof=1),
                               chi2_15_lower_tail_p=None)
        try:
            from scipy.stats import chi2
            q = 15 * us.var(ddof=1)
            res["stored_u"]["chi2_15_lower_tail_p"] = float(chi2.cdf(q, 15))
        except Exception:
            pass
        res["ar_u_lead1"] = dict(mean=ua.mean(), sd=ua.std(ddof=1), n=ua.size)
        for lead in (1, 3):
            li = np.flatnonzero(keys[:, 1] == lead)
            assert np.array_equal(keys[li, 0], starts)
            for t in ("s", "a"):
                L = z[f"{t}_RS1"][li, r] / z[f"{t}_RN"][li, r][:, None]     # (179, 16) member box level
                res[f"between_member_var_of_box_level_lead{lead}_{t}"] = float(L.var(1, ddof=1).mean())
            res[f"ratio_ar_over_stored_box_level_var_lead{lead}"] = res[f"between_member_var_of_box_level_lead{lead}_a"] / \
                res[f"between_member_var_of_box_level_lead{lead}_s"]
        # lead 1: within-start relation of the ar member box level to u (relative departure = b u + residual)
        La = z["a_RS1"][l1, r] / z["a_RN"][l1, r][:, None]
        Ls = z["s_RS1"][l1, r] / z["s_RN"][l1, r][:, None]
        da = La / La.mean(1, keepdims=True) - 1; ds = Ls / Ls.mean(1, keepdims=True) - 1
        uc = ua - ua.mean(1, keepdims=True)
        b = float((uc * da).sum() / (uc * uc).sum())
        resid = da - b * uc
        res["ar_lead1_relative_departure_on_u"] = dict(slope_b=b, r=float(np.corrcoef(uc.ravel(), da.ravel())[0, 1]),
                                                       resid_var=float(resid.var(ddof=1)))
        # stored: same relation with the 16 fixed u_m
        usc = us - us.mean()
        bs = float((usc[None] * ds).sum() / ((usc ** 2).sum() * len(l1)))
        res["stored_lead1_relative_departure_on_u"] = dict(slope_b=bs, r=float(np.corrcoef(np.tile(usc, len(l1)), ds.ravel())[0, 1]))
        # prediction of the ratio of between-member variances of the relative box level from var(u):
        #   ar: b^2 * var_ar(u) + resid ; stored: b^2 * var(us) + resid (resid assumed the same law)
        va, vs = ua.var(1, ddof=1).mean(), us.var(ddof=1)
        rv = float(resid.var(1, ddof=1).mean())
        res["predicted_ratio_relative_box_level_var_lead1"] = float((b * b * va + rv) / (b * b * vs + rv))
        res["observed_ratio_relative_box_level_var_lead1"] = float(da.var(1, ddof=1).mean() / ds.var(1, ddof=1).mean())
        out[nm] = res
        print(nm, {k: v for k, v in res.items() if not isinstance(v, dict)}, res["stored_u"]["sd"], res["stored_u"]["chi2_15_lower_tail_p"],
              flush=True)
    # weather-free single-draw test, every region: per forecast, log(v_ar / v_stored) of a between-member variance. The stored
    # members' departures persist across starts (fingerprints), so log X_stored is ONE draw for the year, while log X_ar(d)
    # is a fresh draw at every start: under the same law, z = mean_d / sd_d of the per-forecast log ratio is about N(0, 1)
    # and the share of starts with v_ar < v_stored is the rank of the stored draw (uniform under the same law).
    sd_test = {}
    for nm in ["Global"] + RNAMES:
        for lead in (1, 3):
            li = np.flatnonzero(keys[:, 1] == lead)
            item = {}
            for kind in ("pixel_var_between_members", "box_level_var_between_members"):
                v = {}
                for t in ("s", "a"):
                    if nm == "Global":
                        N, S1, Gm = z[f"{t}_N"][li], z[f"{t}_S1"][li], z[f"{t}_G"][li]
                    else:
                        r = RNAMES.index(nm)
                        N, S1, Gm = z[f"{t}_RN"][li, r], z[f"{t}_RS1"][li, r], z[f"{t}_RG"][li, r]
                    if kind == "pixel_var_between_members":
                        Gn = Gm / N[:, None, None]
                        d_ = np.einsum("fii->fi", Gn)
                        v[t] = (16 * d_.sum(1) - Gn.sum((1, 2))) / (16 * 15)
                    else:
                        L = S1 / N[:, None]
                        v[t] = (L / L.mean(1, keepdims=True)).var(1, ddof=1)
                x = np.log(v["a"] / v["s"])
                item[kind] = dict(ratio_of_means=float(v["a"].mean() / v["s"].mean()), mean_log_ratio=float(x.mean()),
                                  sd_log_ratio_over_starts=float(x.std(ddof=1)), z_single_draw=float(x.mean() / x.std(ddof=1)),
                                  share_starts_ar_below_stored=float((x < 0).mean()), starts=int(len(x)))
            sd_test[f"{nm}/lead{lead}"] = item
    out["single_draw_test"] = sd_test
    for k, v in sd_test.items():
        print(k, {kk: (round(vv["ratio_of_means"], 3), round(vv["z_single_draw"], 2), round(vv["share_starts_ar_below_stored"], 3))
                  for kk, vv in v.items()}, flush=True)
    lc = json.loads((LC / "lawcheck.json").read_text())
    lc["f_region_box_noise"] = jsonable(out)
    save_json(LC / "lawcheck.json", lc)
    print("region done")


if __name__ == "__main__":
    main()
