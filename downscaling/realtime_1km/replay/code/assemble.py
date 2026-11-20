#!/usr/bin/env python3
"""WP1a step D (CPU): the per-day deliverable replay/points/<d>.npz = the downscaled members of replay/km/<d>.npz plus the
station-free baselines at the monitors, with the definitions of realtime_1km/rc/coupling_live.py (its functions basis,
regrid_weights, regrid, bilinear and qmap are IMPORTED from that file, not rewritten). No observation is read.

Lead day L = 1..5, valid UTC day v = d + L - 1; instants a = 24(L-1), b = 24L - 12, c = 24L hours after 00 UTC of d;
daily mean = max(1/4 x(a) + 1/2 x(b) + 1/4 x(c), 0), the 0 h instant being the CAMS analysis of t0 (cams.npy[1, 1]).
  rt04  (5, M, n)  each coarse member's daily mean (coarse/<d>/pred.npy, 361 x 900, 84N..60S), bilinear at the monitor
  cams  (5, n)     the CAMS forecast of the same cycle (cams_fc.npz, leads 12..120 h, 451 x 900, 90N..90S), same daily
                   mean, bilinear; NaN when the file does not hold the ten leads
  clim  (5, n)     expm1(mu_p(doy of v)), mu_p = harmonic climatology of log1p of the 1 km target at the monitor pixel
  bcsd  (5, n)     expm1(mu_p - mubar_c + log1p(mean_m expm1(QM_c,month(log1p x_c,m)))): per-cell monthly quantile map of
                   each member's daily mean at the 0.25 deg cell point (bilinear in log1p from 0.4 deg, float16 as prep.py),
                   member mean in linear space, plus the climatological pattern (ghap_clim/bcsd_G.npz, 2017-2020)
  an    (5, M, n)  ANALYSIS-mode downscaled members of the valid day at the monitor pixel
                   (hindcast/2026/preds_st/pmG/<v>.npz, positions of hindcast/2026/st_ring_2026.npz), NaN when absent
Extras (not in the task list, same definitions): bcsd_m (5, M, n) the same arm per member (no member mean);
qm_cell (5, M, n) QM_c(log1p x_c,m) in log1p units (the cell level of the 'cons' coupling); mu_cell (5, n) mubar_c.
    python -I assemble.py 2026-09-01 [2026-09-02 ...] | --list days_all.txt   [--check-pairs]"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")
R = P / "realtime_1km/replay"
GH = P / "data/processed/ghap"
HC = P / "hindcast/2026"
sys.path.insert(0, str(P / "realtime_1km/rc"))
import coupling_live as CL  # noqa: E402  (definitions of the first live comparison; main() guarded)

NLEAD = 5
# the coarse model was fine-tuned on the start days 2026-01-01..08-31 with targets at +12..+120 h: every analysis valid up
# to 2026-09-05 00 UTC was a training target (for an earlier start day). The issue days 09-01.. are not training start days.
LAST_TARGET = dt.datetime(2026, 9, 5, 0)


def trained_instants(d, L):
    """number of the PREDICTED instants of lead day L of issue day d (24(L-1) for L > 1, 24L-12, 24L h) that were targets
    of the fine-tuning of the coarse model"""
    t0 = dt.datetime(d.year, d.month, d.day)
    hs = [h for h in (24 * (L - 1), 24 * L - 12, 24 * L) if h > 0]
    return sum(t0 + dt.timedelta(hours=h) <= LAST_TARGET for h in hs)
CODE = [R / "code" / f for f in ("coarse_replay.py", "prep_replay.py", "channels_replay.py", "infer_points.py", "positions.py", "assemble.py",
                                 "km_rank.py")] + \
       [P / "realtime_1km" / f for f in ("prep.py", "channels.py", "infer.py", "blh.py", "blh_calib.json", "rc/coupling_live.py")] + \
       [P / f for f in ("cdffm_aq/model_aq.py", "cdffm_aq/data_aq.py", "cdffm_aq_x/model_x.py", "cdffm_aq_x/copula_x.py",
                        "transport/flexgreen_pm.py", "scripts/build_global_t6.py")] + \
       [RT / "predict.py", RT.parent / "foehn_cams.py", RT.parent / "flexgreen.py"]


def sha(p, n=16):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:n]


def static():
    Z = np.load(R / "positions.npz")
    S = np.load(R / "positions_store.npz")
    lat, lon = Z["lat"].astype(np.float64), Z["lon"].astype(np.float64)
    mcell = Z["ucell"][Z["cell"]].astype(np.int64)
    b = np.load(GH / "ghap_clim/bcsd_G.npz")
    cells = b["cells"].astype(np.int64)
    cpos = np.searchsorted(cells, mcell)
    assert (cells[cpos] == mcell).all()
    ucell, uinv = np.unique(mcell, return_inverse=True)
    assert np.array_equal(ucell, Z["ucell"]) and np.array_equal(uinv, Z["cell"])
    ucpos = np.searchsorted(cells, ucell)
    ref = np.load(HC / "st_ring_2026.npz")
    rpix = np.load(HC / "st_pix_2026.npz")["pos"].astype(np.int64)
    j = np.searchsorted(rpix, Z["catalogue_pos"])
    assert (rpix[j] == Z["catalogue_pos"]).all()
    an_idx = ref["mon_idx"][j].astype(np.int64)
    assert np.array_equal(ref["pos"][an_idx], Z["catalogue_pos"])
    return dict(Z=Z, lat=lat, lon=lon, mu_coef=S["clim_mean"][Z["mon"]].astype(np.float64), ccoef=b["clim_cell_coef"][cpos],
                qx=b["qx"][ucpos], qy=b["qy"][ucpos], ucell=ucell, uinv=uinv, ci=ucell // 1440, cj=ucell % 1440,
                wts=CL.regrid_weights(84.0), an_idx=an_idx, n_an_pos=int(ref["pos"].size),
                code={str(f): sha(f) for f in CODE if f.exists()})


def cams_forecast(d):
    """(10, 451, 900) float32 on lon -180.., or None with the reason"""
    why = "no cams_fc.npz"
    for f in (R / "camsfc" / str(d) / "cams_fc.npz", RT / "data" / str(d) / "cams_fc.npz"):
        if not f.exists():
            continue
        c = np.load(f)
        if list(c["leads"]) != [12 * k for k in range(1, 11)]:
            why = f"{f}: leads {list(map(int, c['leads']))} h only"
            continue
        cl, co, cpm = c["lat"], c["lon"], c["pm25"].astype(np.float32)
        assert abs(cl[0] - 90.0) < 1e-6 and abs((cl[0] - cl[1]) - 0.4) < 1e-6 and cpm.shape == (10, 451, 900), (cl[:3], cpm.shape)
        if co[0] >= 0:                                                    # 0..360 -> -180..180 (as live_baselines.py)
            k0 = int(np.argmin(np.abs(co - 180.0)))
            cpm = np.roll(cpm, -k0, axis=-1)
            co = np.remainder(np.roll(co, -k0) + 180.0, 360.0) - 180.0
        assert abs(co[0] + 180.0) < 1e-6, co[:3]
        return cpm, str(f)
    return None, why


def one_day(d, st):
    t0 = time.time()
    Z = st["Z"]
    lat, lon, n = st["lat"], st["lon"], st["lat"].size
    kz = np.load(R / "km" / f"{d}.npz")
    kmeta = json.loads(str(kz["meta"]))
    M = kz["km"].shape[1]
    pp = R / "coarse" / str(d) / "pred.npy"
    pred = np.load(pp).astype(np.float32)                                     # (M, 10, 361, 900)
    assert pred.shape == (M, 10, 361, 900), pred.shape
    an0 = np.load(RT / "data" / str(d) / "cams.npy")[:, 1].astype(np.float32)[1]   # CAMS analysis PM2.5 at t0 (361, 900)
    cpm, cams_src = cams_forecast(d)
    an_full = None
    if cpm is not None:
        an_full = np.full(cpm.shape[1:], np.nan, np.float32)
        an_full[15:15 + 361] = an0                                           # rows 84..-60 of the 90..-90 grid
    out = {k: np.full((NLEAD, M, n), np.nan, np.float32) for k in ("rt04", "an", "bcsd_m", "qm_cell")}
    out.update({k: np.full((NLEAD, n), np.nan, np.float32) for k in ("cams", "clim", "bcsd", "mu_cell")})
    an_files = []
    for L in range(1, NLEAD + 1):
        v = d + dt.timedelta(L - 1)
        a_, b_, c_ = 24 * (L - 1), 24 * L - 12, 24 * L
        inst = lambda h: np.repeat(an0[None], M, 0) if h == 0 else pred[:, h // 12 - 1]
        day_m = np.maximum(0.25 * inst(a_) + 0.5 * inst(b_) + 0.25 * inst(c_), 0.0)          # (M, 361, 900), as prep.build_pm
        out["rt04"][L - 1] = np.stack([CL.bilinear(day_m[m], 84.0, 0.4, -180.0, 0.4, lat, lon) for m in range(M)])
        x025 = np.stack([CL.regrid(day_m[m], st["wts"]).astype(np.float16).astype(np.float64) for m in range(M)])
        xc = np.log1p(np.maximum(x025[:, st["ci"], st["cj"]], 0.0))                           # (M, n_ucell)
        qxm, qym = st["qx"][:, v.month - 1].astype(np.float64), st["qy"][:, v.month - 1].astype(np.float64)
        qm_m = np.stack([CL.qmap(xc[m], qxm, qym) for m in range(M)])
        qm_cell = np.log1p(np.expm1(qm_m).mean(0))                                            # member mean in linear space
        B = CL.basis(v.timetuple().tm_yday - 1)
        mu_p, mubar = st["mu_coef"] @ B, st["ccoef"] @ B
        out["clim"][L - 1] = np.expm1(mu_p)
        out["bcsd"][L - 1] = np.expm1(mu_p - mubar + qm_cell[st["uinv"]])
        out["bcsd_m"][L - 1] = np.expm1(mu_p[None] - mubar[None] + qm_m[:, st["uinv"]])
        out["qm_cell"][L - 1] = qm_m[:, st["uinv"]]
        out["mu_cell"][L - 1] = mubar
        if cpm is not None:
            F_cf = np.maximum(0.25 * (an_full if a_ == 0 else cpm[a_ // 12 - 1]) + 0.5 * cpm[b_ // 12 - 1] + 0.25 * cpm[c_ // 12 - 1], 0.0)
            out["cams"][L - 1] = CL.bilinear(F_cf, 90.0, 0.4, -180.0, 0.4, lat, lon)
        af = HC / "preds_st" / "pmG" / f"{v}.npz"
        if af.exists():
            mem = np.load(af)["members"]
            assert mem.shape == (M, st["n_an_pos"]), mem.shape
            out["an"][L - 1] = mem[:, st["an_idx"]].astype(np.float32)
            an_files.append(str(v))
    pm = {}
    for f in ("pred_meta.json", "pred_source.json"):
        if (pp.parent / f).exists():
            pm = json.loads((pp.parent / f).read_text())
    cmeta = pm.get("meta", pm)
    work = R / "work" / str(d)
    pj = json.loads((work / "prep.json").read_text()) if (work / "prep.json").exists() else {}
    cj = json.loads((work / "channels.json").read_text()) if (work / "channels.json").exists() else {}
    prov = dict(
        issue=str(d), valid=[str(d + dt.timedelta(L - 1)) for L in range(1, NLEAD + 1)], members=M, n_monitors=int(n),
        leads_made=kmeta["leads_made"],
        coarse=dict(pred=str(pp.resolve()), source="archived production forecast" if pp.is_symlink() else "replayed (FLEXGREEN + sampler recomputed)",
                    ckpt=cmeta.get("ckpt"), ckpt_sha256=pm.get("ckpt_sha256") or sha(RT / "runs/rt12_a/final.ckpt", 64), gi=cmeta.get("gi"),
                    steps=cmeta.get("steps"), refine_tail=cmeta.get("refine_tail"), inputs=cmeta.get("inputs"),
                    flexgreen_s=pm.get("flexgreen_s"), predict_s=pm.get("predict_s"), job=pm.get("job"),
                    predicted_instants_that_were_training_targets={str(L): trained_instants(d, L) for L in range(1, NLEAD + 1)},
                    all_leads_after_training_targets=bool(sum(trained_instants(d, L) for L in range(1, NLEAD + 1)) == 0),
                    note="the coarse model was fine-tuned on the start days 2026-01-01..08-31 (targets +12..+120 h): analyses valid up to "
                         "2026-09-05 00 UTC were training targets; issue days from 2026-09-05 on have no predicted instant among them"),
        downscaler=dict(ckpt=kmeta["ckpt"], ckpt_sha256=kmeta["ckpt_sha256"], run=kmeta["run"], step=kmeta["step"], prior=kmeta["prior"],
                        prior_sha256=kmeta["prior_sha256"], guard=kmeta["guard"], ode_steps=kmeta["ode_steps"], seed=kmeta["seed"],
                        noise=kmeta["noise"], device=kmeta["device"], torch=kmeta["torch"], job=kmeta["job"], seconds=kmeta["seconds"],
                        timing=kmeta["timing"], n_positions=kmeta["n_positions"]),
        inputs_1km=dict(work=str(work), prep_seconds_met=pj.get("seconds_met"), prep_seconds_pm=pj.get("seconds_pm"), ifs_source=pj.get("fetch_source"),
                        no_prev=pj.get("no_prev"), channels_seconds=cj.get("seconds"), flexgreen_seconds=cj.get("flexgreen_seconds"),
                        fires="GFAS of d - 1 persisted (RT/data/<d>/fires.npy)"),
        baselines=dict(cams_forecast=cams_src if cpm is not None else None, cams_missing_reason=None if cpm is not None else cams_src,
                       analysis_t0=str(RT / "data" / str(d) / "cams.npy"), bcsd=str(GH / "ghap_clim/bcsd_G.npz"),
                       clim=str(GH / "ghap_clim/clim_mean_G.npy"), an_valid_days=an_files, an_dir=str(HC / "preds_st/pmG"),
                       definitions=str(P / "realtime_1km/rc/coupling_live.py")),
        positions=dict(file=str(R / "positions.npz"), ring_rule="mean over the ring pixels inside the mask; ring<d>_n gives their number "
                       "(the 2022 station test required at least 4 d of the 8 d)", cell_sample="5 x 5 regular sample of the 25 x 25 pixel block"),
        code_sha256_16=st["code"], assembled_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        assemble_seconds=round(time.time() - t0, 1))
    arrays = dict(id=Z["id"], lat=Z["lat"], lon=Z["lon"], issue=np.array(str(d)), valid=np.array(prov["valid"]),
                  km=kz["km"], km_ring1=kz["km_ring1"], km_ring3=kz["km_ring3"], km_ring10=kz["km_ring10"], km_cell=kz["km_cell"],
                  rt04=out["rt04"], cams=out["cams"], clim=out["clim"], bcsd=out["bcsd"], an=out["an"],
                  km_cell_lin=kz["km_cell_lin"], bcsd_m=out["bcsd_m"], qm_cell=out["qm_cell"], mu_cell=out["mu_cell"],
                  ring1_n=Z["ring1_n"], ring3_n=Z["ring3_n"], ring10_n=Z["ring10_n"], cell_n=Z["cell_n"],
                  provenance=np.array(json.dumps(prov)))
    (R / "points").mkdir(exist_ok=True)
    tmp = R / "points" / f"{d}.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.rename(R / "points" / f"{d}.npz")
    fin = lambda x: float(np.isfinite(x).mean())
    print(f"{d}: points written ({time.time() - t0:.0f}s) leads {prov['leads_made']} coarse {'archived' if pp.is_symlink() else 'replayed'} | "
          f"means L1: km {np.nanmean(kz['km'][0]):.2f} rt04 {np.nanmean(out['rt04'][0]):.2f} cams {np.nanmean(out['cams'][0]):.2f} "
          f"clim {np.nanmean(out['clim'][0]):.2f} bcsd {np.nanmean(out['bcsd'][0]):.2f} an {np.nanmean(out['an'][0]) if an_files else float('nan'):.2f} | "
          f"finite: km {fin(kz['km']):.3f} cams {fin(out['cams']):.3f} an {fin(out['an']):.3f}", flush=True)
    return arrays


def check_pairs(days, st):
    """consistency with the first live comparison: the arms of rc/coupling_pairs.npz (issue days 2026-11-03..11-11) against
    the arrays written here, at the same (issue, lead, monitor); the observation column of that file is not used"""
    z = np.load(P / "realtime_1km/rc/coupling_pairs.npz")
    cols = [str(c) for c in z["cols"]]
    A = z["pairs"]
    res = {}
    for d in days:
        f = R / "points" / f"{d}.npz"
        doy0 = (d - dt.date(2026, 1, 1)).days
        s = A[:, cols.index("issue_doy0")].astype(int) == doy0
        if not f.exists() or not s.any():
            continue
        p = np.load(f)
        L_, k_ = A[s, cols.index("lead")].astype(int), A[s, cols.index("monitor")].astype(int)
        mine = dict(km=np.maximum(p["km"].astype(np.float64).mean(1), 0.0), rt04=p["rt04"].astype(np.float64).mean(1), cams=p["cams"], clim=p["clim"],
                    bcsd=p["bcsd"], qm=np.log1p(np.expm1(p["qm_cell"].astype(np.float64)).mean(1)))
        r = {}
        for arm, x in mine.items():
            ref = A[s, cols.index(arm)].astype(np.float64)
            y = (np.expm1(x[L_ - 1, k_]) if arm == "qm" else x[L_ - 1, k_]).astype(np.float64)
            ok = np.isfinite(y) & np.isfinite(ref)
            dd = np.abs(y[ok] - ref[ok])
            r[arm] = dict(n=int(ok.sum()), max_abs=float(dd.max()), mean_abs=float(dd.mean()), max_rel=float((dd / np.maximum(np.abs(ref[ok]), 1.0)).max()),
                          corr=float(np.corrcoef(y[ok], ref[ok])[0, 1]))
        res[str(d)] = r
        print(f"pairs check {d}: " + "; ".join(f"{k} max|d| {v['max_abs']:.4f} (rel {v['max_rel']:.1e})" for k, v in r.items()), flush=True)
    (R / "check" / "pairs_check.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--check-pairs", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    days = list(a.days) + ([x for x in Path(a.list).read_text().split() if x.strip()] if a.list else [])
    days = sorted(dt.date.fromisoformat(x) for x in set(days))
    st = static()
    done, missing = [], []
    for d in days:
        if not (R / "km" / f"{d}.npz").exists():
            missing.append(str(d))
            continue
        if (R / "points" / f"{d}.npz").exists() and not a.overwrite:
            done.append(str(d))
            continue
        one_day(d, st)
        done.append(str(d))
    print(f"assembled {len(done)} days; without km file: {missing}", flush=True)
    if a.check_pairs:
        check_pairs(days, st)


if __name__ == "__main__":
    main()
