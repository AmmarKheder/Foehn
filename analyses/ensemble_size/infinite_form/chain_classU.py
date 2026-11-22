#!/usr/bin/env python3
"""A14 class U for the downscaler and the two-stage system at the official stations of 2022 (Supplementary Note chain2022,
Supplementary Table chain2022; D4: blind record, as-run first) and for Foehn-CAMS read at the stations (its 'coarse' arm).
Scorer test2022_official/code/score_official.py (A7): its functions and classes (Data, Table, reduce_ens, det, take, bil,
coarse_fields) are imported unchanged and its main() is replayed for the station pixel (where = 'pixel'; the rings are not
redone), without its checks and without writing anything outside a14/classU. Identity: every contrast and arm score of
tables S, C and F equals results/scores_official_2022.json. The infinite form uses the arm's own sums with the squared
error replaced by (sum e^2 - sum s2 / M) (column 'var' of Table._sums, ddof 1), through the unchanged Table.contrast (same
moving-block bootstrap); the eight-member form equals the as-run one for the eight-member arms and is
sum e^2 - sum s2/4 + sum s2/8 for Foehn's four members. V_draw: the chain, the coarse forecast and the same-day downscaler
draw their noise afresh for every day (seeds keyed by the valid or issue day, predict_official.py l. 307, 345; foehn_cams.py
l. 376): V_draw = 0; Foehn's four fixed fields: max(jackknife over its four members, the station quartet CV of
st_cf_classU.json carried, NOTES.md item 4).
    python chain_classU.py   -> results/chain_classU.json"""
import argparse
import json
import os
import sys
import time

import numpy as np

from u_common import DESIGN_NOTE, ISFX, P, RES, inflate, jk_var, safe_json, upstream_env

O = P / "test2022_official"
sys.path.insert(0, str(O / "code"))
with upstream_env():                                                  # design: its Foehn arm is eval_final_crps_e01_ar under 'ar' only
    import score_official as SO  # noqa: E402  (unchanged; main() not run)

t0 = time.time()
a = argparse.Namespace(preds=str(O / "preds"), selftest=False, no_paper_check=True, min_days=30, force=False, only_paper_checks=False)
Pp = SO.Path(a.preds)
D = SO.Data(a)
D.load_aurora(); D.load_foehn()
REF = json.loads((O / f"results{ISFX}" / "scores_official_2022.json").read_text())["tables"]["pixel"]   # design: S1 step 7
done_s, done_c = np.load(Pp / "done_sameday.npy").astype(bool), np.load(Pp / "done_chain.npy").astype(bool)
names_s = {"ds_sameday": "sameday_ds", "cams0_bil": "sameday_cams0", "bcsd": "sameday_bcsd", "clim_ens": "sameday_climens",
           "clim_med": "sameday_climmed"}
names_c = ("chain", "chain_t6fc", "chain_rtseed", "camsfc_ds", "bcsd_fc")
mm = {k: np.load(Pp / f"{k}.npy", mmap_mode="r") for k in list(names_s.values()) + list(names_c)}
ghap_all = np.load(O / "truth/ghap_official_2022.npy")
NI, NL, I0 = SO.NI, SO.NL, SO.I0
obs_v = D.obs[D.vidx]
iss = np.where(done_c)[0]
coarse = np.full((NI, NL, 8, D.S), np.nan, np.float32)
camsfc = np.full((NI, NL, D.S), np.nan, np.float32)
aurora = np.full((NI, NL, D.S), np.nan, np.float32)
foehn = np.full((NI, NL, 4, D.S), np.nan, np.float32)
for i in iss:
    I = I0 + int(i)
    coarse[i] = SO.bil(SO.coarse_fields(int(i)), 84.0, 0.4, -180.0, 0.4, D.lat, D.lon)
    camsfc[i] = D.cams_daily(I)[:, D.id]
    for L in range(1, NL + 1):
        aurora[i, L - 1] = D.aurora_daily(I, L)[D.id]
        if (I - 1, L) in D.foehn:
            foehn[i, L - 1] = D.foehn[(I - 1, L)][D.id].T
print(f"coarse systems done {time.time() - t0:.0f} s", flush=True)
clim_paper_s = D.CLIM[:, D.id]
A_coarse = SO.reduce_ens(coarse, obs_v)
A_camsfc, A_aurora, A_foehn = SO.det(camsfc, obs_v), SO.det(aurora, obs_v), SO.reduce_ens(foehn, obs_v)
C_S = [("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"), ("ds_sameday", "clim_paper"), ("ds_sameday", "clim_med"),
       ("ds_sameday", "clim_ens"), ("ds_sameday", "ghap"), ("bcsd", "cams0_bil"), ("cams0_bil", "clim_paper"), ("cams0_bil", "clim_med")]
C_C = [("chain", b) for b in ("coarse", "camsfc", "aurora", "clim_paper", "clim_med", "clim_ens", "bcsd_fc", "ds_sameday",
                              "camsfc_ds", "persist_ds", "ghap", "foehn")] + \
      [("chain_t6fc", b) for b in ("chain", "coarse", "camsfc", "aurora", "bcsd_fc", "clim_paper", "clim_med", "clim_ens", "foehn")] + \
      [("chain_rtseed", "chain"), ("coarse", "camsfc"), ("coarse", "aurora"), ("camsfc_ds", "camsfc"), ("camsfc_ds", "coarse"),
       ("bcsd_fc", "coarse"), ("bcsd_fc", "camsfc"), ("aurora", "camsfc"), ("persist_ds", "camsfc"),
       ("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"), ("ds_sameday", "clim_paper"), ("ds_sameday", "clim_med"),
       ("ds_sameday", "clim_ens"), ("ds_sameday", "camsfc"), ("foehn", "camsfc"), ("foehn", "aurora"), ("foehn", "coarse"),
       ("foehn", "clim_paper"), ("foehn", "ds_sameday"), ("camsfc", "clim_paper"), ("coarse", "clim_paper")]
w = 0
sv = lambda k: np.asarray(mm[k][..., w])[..., D.pix].astype(np.float32)      # noqa: E731
sd_raw = {n: sv(f) for n, f in names_s.items()}
SD = {n: SO.reduce_ens(x, D.obs) for n, x in sd_raw.items()}
SD["ghap"] = SO.det(ghap_all[:, D.pix, w], D.obs)
SD["clim_paper"] = SO.det(clim_paper_s, D.obs)
armsS = {n: {q: (x[q][:, None, :] if q != "M" else x[q]) for q in x} for n, x in SD.items()}
tS = SO.Table("S", armsS, D.obs[:, None, :], D.gmask, "cams0_bil", a.min_days, ["sameday"], "same-day")
armsC = {n: SO.reduce_ens(sv(n), obs_v) for n in names_c}
armsC["coarse"], armsC["camsfc"], armsC["aurora"] = A_coarse, A_camsfc, A_aurora
pers = [SO.reduce_ens(sd_raw["ds_sameday"][I0 - 1 + np.arange(NI)], obs_v[:, L]) for L in range(NL)]
armsC["persist_ds"] = {q: (np.stack([p_[q] for p_ in pers], 1) if q != "M" else pers[0]["M"]) for q in pers[0]}
for n in SD:
    armsC[n] = SO.take(SD[n], D.vidx)
for v_ in armsC.values():
    for q in ("mean", "crps", "var"):
        v_[q] = np.where(done_c[:, None, None], v_[q], np.nan).astype(np.float32)
LEADS = [str(L) for L in range(1, NL + 1)]
tC = SO.Table("C", armsC, obs_v, D.gmask, "camsfc", a.min_days, LEADS, "forecast")
armsF = dict(armsC); armsF["foehn"] = A_foehn
tF = SO.Table("F", armsF, obs_v, D.gmask, "camsfc", a.min_days, LEADS, "F")
print(f"tables built {time.time() - t0:.0f} s", flush=True)
FRESH = {"chain", "chain_t6fc", "camsfc_ds", "bcsd_fc", "coarse", "ds_sameday", "persist_ds", "bcsd", "cams0_bil", "clim_ens", "clim_med"}
# Foehn's carried quartet CV (station pixels, compare_fair station-days, 'all'), leads 1 and 3 of st_cf_classU.json
SUFX = os.environ.get("A14U_OUT", "")
K2 = json.loads((RES / (f"st_cf_classU{SUFX}.json" if (RES / f"st_cf_classU{SUFX}.json").exists() else "st_cf_classU.json")).read_text())["k2_station_pixels"]
cvq = {L: np.sqrt(K2[str(L)]["all"]["rmse_inf"]["v_quart"]) / K2[str(L)]["all"]["rmse_inf"]["quartet_mean"] for L in (1, 3)}
cv_lead = {1: cvq[1], 2: 0.5 * (cvq[1] + cvq[3]), 3: cvq[3], 4: cvq[3], 5: cvq[3]}
cvm = {L: np.sqrt(K2[str(L)]["all"]["rmse_m"]["v_quart"]) / K2[str(L)]["all"]["rmse_m"]["quartet_mean"] for L in (1, 3)}   # as-run (4-member) form
cv_lead_m = {1: cvm[1], 2: 0.5 * (cvm[1] + cvm[3]), 3: cvm[3], 4: cvm[3], 5: cvm[3]}
EXTRA_PAIRS = {"foehn - bcsd_fc"}                                    # fix 12: Foehn against the quantile mapping (table F)


def form_sums(T, name, form):
    s = T.sums[name].copy(); M = T.arms[name]["M"]
    if M > 1 and form in ("inf", "M8"):
        c8 = 1 / 8 if (form == "M8" and M != 8) else (1 / M if form == "M8" else 0.0)
        s[..., 1] = s[..., 1] - s[..., 4] / M + c8 * s[..., 4]
    return s


def foehn_jk_sums(T, form):
    """the Foehn arm's sums with each of its four members left out (3-member estimator), shape (4, G, U, K, 7)"""
    out = []
    for i in range(4):
        Fi = SO.reduce_ens(np.delete(foehn, i, 2), obs_v)
        s = T._sums(Fi, T.ok)
        if form != "as_run":
            c8 = 1 / 8 if form == "M8" else 0.0
            s[..., 1] = s[..., 1] - s[..., 4] / 3 + c8 * s[..., 4]
        out.append(s)
    return out


def table_forms(T, pairs, ref_res):
    res = {"identity": {}, "contrasts": {}, "arms": {}}
    asr = T.result(pairs)
    for k, v in asr["contrasts"].items():                         # identity with the paper's JSON (its own pairs only)
        rr = ref_res["contrasts"].get(k)
        if rr is None and k in EXTRA_PAIRS:
            continue
        res["identity"][k] = bool(rr is not None and json.dumps(v, sort_keys=True, default=float) == json.dumps(rr, sort_keys=True, default=float))
    for nm in T.arms:
        res["identity"]["arm:" + nm] = bool(json.dumps(asr["arms"][nm], sort_keys=True, default=float) == json.dumps(ref_res["arms"][nm], sort_keys=True, default=float))
    jkF = {f: foehn_jk_sums(T, f) for f in ("inf", "M8", "as_run")} if "foehn" in T.arms else None
    for A, B in pairs:
        if A not in T.arms or B not in T.arms:
            continue
        e = {"as_run": asr["contrasts"][f"{A} - {B}"]}
        for f in ("inf", "M8", "as_run_widened"):
            fs = "as_run" if f == "as_run_widened" else f
            if f == "as_run_widened" and "foehn" not in (A, B):
                continue
            c = T.contrast(A, B, form_sums(T, A, fs), form_sums(T, B, fs))
            f_ = fs
            if "foehn" in (A, B):                                   # V_draw of Foehn's fixed fields: jackknife and carried CV
                for gn in c:
                    for ln, r in c[gn].items():
                        if r is None or "rel_rmse_ci" not in r:
                            continue
                        g = SO.GROUPS.index(gn); k = None if ln == "all" else LEADS.index(ln)
                        reps = []
                        for i in range(4):
                            sa = jkF[f_][i] if A == "foehn" else form_sums(T, A, f_)
                            sb = jkF[f_][i] if B == "foehn" else form_sums(T, B, f_)
                            a_ = sa[g, :, k] if k is not None else sa[g].sum(1); b_ = sb[g, :, k] if k is not None else sb[g].sum(1)
                            a_, b_ = a_[T.units], b_[T.units]
                            reps.append(np.sqrt(a_[:, 1].sum() / b_[:, 1].sum()) - 1)
                        vj = float(jk_var(np.array(reps)))
                        Lc = 1 if ln == "all" else int(ln)
                        vq = float(((cv_lead_m if f_ == "as_run" else cv_lead)[Lc] * (1 + r["rel_rmse"])) ** 2)
                        ix = T.idx
                        sa_ = form_sums(T, A, f_); sb_ = form_sums(T, B, f_)
                        a_ = (sa_[g, :, k] if k is not None else sa_[g].sum(1))[T.units]; b_ = (sb_[g, :, k] if k is not None else sb_[g].sum(1))[T.units]
                        bs = np.sqrt(a_[ix, 1].sum(1) / b_[ix, 1].sum(1)) - 1
                        ci, fac = inflate(r["rel_rmse"], r["rel_rmse_ci"][0], r["rel_rmse_ci"][1], float(np.nanvar(bs, ddof=1)), max(vj, vq))
                        r.update(v_jack4=vj, v_quart=vq, rel_rmse_ci_inflated=ci, factor=fac)
            else:
                for gn in c:
                    for ln, r in c[gn].items():
                        if r is not None:
                            r["v_draw"] = 0.0 if (A in FRESH or T.arms[A]["M"] == 1) and (B in FRESH or T.arms[B]["M"] == 1) else None
            e[f] = c
        res["contrasts"][f"{A} - {B}"] = e
    for nm, arm in T.arms.items():
        M = arm["M"]
        if M == 1:
            continue
        o = {}
        for gn in SO.GROUPS:
            g = SO.GROUPS.index(gn)
            for k, ln in enumerate(LEADS + ["all"] if len(T.leads) > 1 else T.leads):
                q = T.sums[nm][g, :, k].sum(0) if ln != "all" else T.sums[nm][g].sum((0, 1))
                n = q[0]
                if n == 0:
                    continue
                mse = q[1] / n; v = q[4] / n
                o[f"{gn}|{ln}"] = dict(n=int(n), M=M, rmse_as_run=float(np.sqrt(mse)), rmse_inf=float(np.sqrt(mse - v / M)),
                                       rmse_8=float(np.sqrt(mse - v / M + v / 8)), ssr_fortin_as_run=float(np.sqrt((M + 1) / M * v) / np.sqrt(mse)),
                                       ssr_inf=float(np.sqrt(v) / np.sqrt(mse - v / M)), ssr_fortin_8=float(np.sqrt(9 / 8 * v) / np.sqrt(mse - v / M + v / 8)),
                                       crps_fair=float(q[3] / n))
        res["arms"][nm] = o
    return res


OUT = {"S": table_forms(tS, C_S, REF["S"]), "C": table_forms(tC, C_C, REF["C"]), "F": table_forms(tF, C_C + [("foehn", "bcsd_fc")], REF["F"]),
       "v_draw": "0 for the arms with fresh noise every day; Foehn's fixed fields: jackknife over its four members and the carried station quartet CV"}
if DESIGN_NOTE:
    OUT["design"] = DESIGN_NOTE
OUT["identity_all"] = {t: all(OUT[t]["identity"].values()) for t in ("S", "C", "F")}
OUT["identity_failures"] = {t: [k for k, v in OUT[t]["identity"].items() if not v] for t in ("S", "C", "F")}
print("identity:", OUT["identity_all"], OUT["identity_failures"], flush=True)
safe_json(RES / f"chain_classU{SUFX}.json", OUT)
print(f"done {time.time() - t0:.0f} s", flush=True)
