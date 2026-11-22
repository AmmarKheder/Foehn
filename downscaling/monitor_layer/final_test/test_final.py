#!/usr/bin/env python3
"""Tests of the final test of the monitor-fed layer (draft Addendum 8 of A11). No held-out value and no observation used
as a truth is read by any of them (audit-hook firewall installed before any data file is opened).

  cpu           (numpy only, login node or CPU job; FT_TMP = a scratch directory for the perturbed copies)
    iii-a  leak test of the INPUTS: every row of a feed day file (and of assim_2026.npz) that is not an assimilated monitor
           by the place rule -- the held-out monitors, among them the twins inside the held-out cells -- gets its value
           (and valid-hour count) replaced by noise WITHOUT being read; build_inputs.py on these copies must give
           data/inputs.npz bit for bit. Keeps the perturbed inputs as --keep-leak (input of the forecast leak test, smoke).
    iv     no value of the issue day or later: (a) every issue day reads exactly the days d - 1 and d - 2; (b) for four
           issue days, every day file dated >= d (and every column >= d of assim_2026.npz) is replaced by noise: the
           inputs of that issue day are unchanged bit for bit, and the files opened by the builder are all dated < d.
    fw     the firewall refuses every truth file.
  gpu           (container, one GCD; job jobs/smoke_job.sh)
    i      M with its station features switched off against its own 'nostations' validation behaviour: on 2 SEL days of
           2021 x 4,096 pixels, members of this code path (infer_m.members_from_z0 with stn = None and with the explicit
           no-monitor vector) = cdffm_aq_m.lagval.validate_lag's 'M|nostations' members (same noise); the station
           features built by infer_m.mon_from_arrays from the 2021 assimilation file = LagStationSet.lagged's, bit for
           bit, and the members with them = validate_lag's 'M' members at lags 1 and 2; the production Sampler
           (prior None, guard on) = members_from_z0 with the same noise; the stored 2021 table of the selected checkpoint
           (lag 1, M and M|nostations, every distance class) is reproduced on the 24 SEL days x 100,000 pixels.
  check-smoke   (numpy only) the smoke outputs: ii identity of the station-free arm with the record's km on the smoke
           day; iii-b leak test of the FORECASTS (forecasts from the perturbed-input build = forecasts of the smoke);
           finiteness, guard hits, distances = inputs; the price of the full run.
  check-dry     (numpy only) the CPU dry run of infer_m.py (jobs/dry_cpu_job.sh: one issue day,
           2 members, lead days 1 and 5, every arm): the diagnostic arm free0 (check 4) runs, is finite, differs
           from free (the prior) and from m1 (the monitors), and shares the base noise of m1 / m2 (identical truncation
           counts); m2 lead day 5 empty; members not made empty; the full-run guard refused a non-frozen configuration.
           A CPU noise generator differs from the GPU's: the identity with the record is reported, not required.
"""
import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FT = Path(__file__).resolve().parent
sys.path.insert(0, str(FT))
import ft_lib as F  # noqa: E402

PY = sys.executable
RESULTS = []


def rec(name, ok, **kw):
    RESULTS.append(dict(test=name, ok=bool(ok), **kw))
    print(f"[{'OK' if ok else 'FAIL'}] {name} " + json.dumps(kw, default=F._jdef)[:600], flush=True)
    return ok


def load_npz(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def compare_inputs(a, b, keys=None):
    keys = keys or [k for k in a if k not in ("prov", "src_file")]
    diff = [k for k in keys if not (a[k].shape == b[k].shape and (np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == "f" else np.array_equal(a[k], b[k])))]
    return diff


# ---- perturbed copies (the held-out rows are never read) -------------------------------------------------------------
def u_ids():
    I = load_npz(F.INPUTS)
    return set(I["u_id"].tolist()), I


def noisy_dayfile(src, dst, keep_ids, rng, all_noise=False):
    """copy of a feed day file: rows whose id is not in keep_ids (and every row if all_noise) get noise in 'value' and
    'nhours'; kept rows are gathered through a memory map; the other members (ids, positions, date, rule, meta) are copied"""
    z = np.load(src)
    out = {k: z[k] for k in z.files if k not in ("value", "nhours")}
    ids = z["id"].astype(str)
    keep = np.zeros(ids.size, bool) if all_noise else np.isin(ids, sorted(keep_ids))
    r = np.flatnonzero(keep)
    v = rng.lognormal(3.0, 1.0, ids.size).astype(np.float32)
    h = rng.integers(18, 25, ids.size).astype(np.uint8)
    if r.size:
        v[r] = F.gather_rows(src, "value", r)
        h[r] = np.asarray(F.npz_memmap(src, "nhours")[r])
    out["value"], out["nhours"] = v, h
    F.save_npz(dst, **out)
    return int((~keep).sum())


def noisy_assim(src, dst, keep_ids, rng, from_col=None):
    """copy of assim_2026.npz: 'daily' is noise except the rows of keep_ids (gathered whole through a memory map), and,
    with from_col, every column >= from_col is noise for every row"""
    z = np.load(src)
    out = {k: z[k] for k in z.files if k != "daily"}
    ids = z["id"].astype(str)
    mm = F.npz_memmap(src, "daily")
    d = rng.lognormal(3.0, 1.0, mm.shape).astype(np.float32)
    r = np.flatnonzero(np.isin(ids, sorted(keep_ids)))
    d[r] = np.asarray(mm[r], np.float32)
    if from_col is not None:
        d[:, from_col:] = rng.lognormal(3.0, 1.0, (mm.shape[0], mm.shape[1] - from_col)).astype(np.float32)
    del mm
    out["daily"] = d
    F.save_npz(dst, **out)
    return int(ids.size - r.size)


def build(feed, assim, out, days=None, trace=None):
    cmd = [PY, "-I", "-B", str(FT / "build_inputs.py"), "--feed-dir", str(feed), "--assim", str(assim), "--out", str(out)]
    if days:
        cmd += ["--days", days]
    if trace:
        cmd += ["--trace", str(trace)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-4000:])
    return r.stdout.strip().splitlines()[-1]


def run_cpu(a):
    F.install_truth_firewall(trace=False)
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="ft_cpu_", dir=os.environ.get("FT_TMP", "/tmp")))
    rng = np.random.default_rng(20261121)
    keep, I = u_ids()
    src_days = [str(x) for x in I["src_day"]]
    try:
        # ---- iii-a --------------------------------------------------------------------------------------------------
        feed = tmp / "feed"
        feed.mkdir()
        nn = 0
        for s in src_days:
            f = F.FEED_REPLAY / f"stations_{s}.npz"
            if f.exists():
                nn += noisy_dayfile(f, feed / f.name, keep, rng)
        na = noisy_assim(F.ASSIM_2026, tmp / "assim_2026.npz", keep, rng)
        line = build(feed, tmp / "assim_2026.npz", tmp / "inputs_leak.npz")
        J = load_npz(tmp / "inputs_leak.npz")
        diff = compare_inputs(I, J)
        rec("iii-a leak test of the inputs (held-out rows replaced by noise, unread)", not diff, differing_arrays=diff,
            day_file_rows_replaced=nn, assim_rows_replaced=na, build=line)
        if a.keep_leak:
            shutil.copyfile(tmp / "inputs_leak.npz", a.keep_leak)
            os.chmod(a.keep_leak, 0o660)
        # ---- iv (a) -------------------------------------------------------------------------------------------------
        bad = []
        for i, d in enumerate(I["issue"]):
            dd = dt.date.fromisoformat(str(d))
            for k in (1, 2):
                s = dt.date.fromisoformat(str(I["src_day"][I["issue_src"][i, k - 1]]))
                if s != dd - dt.timedelta(days=k) or not s < dd:
                    bad.append((str(d), k, str(s)))
        rec("iv-a every issue day reads the days d - 1 and d - 2 only", not bad, bad=bad, issue_days=int(I["issue"].size))
        # ---- iv (b) -------------------------------------------------------------------------------------------------
        for d in ("2026-09-06", "2026-10-10", "2026-10-22", "2026-11-11"):
            dd = dt.date.fromisoformat(d)
            fd = tmp / f"feed_{d}"
            fd.mkdir()
            nlater = 0
            for f in sorted(F.FEED_REPLAY.glob("stations_*.npz")):
                fday = dt.date.fromisoformat(f.stem.split("_")[1])
                if fday >= dd:
                    noisy_dayfile(f, fd / f.name, keep, rng, all_noise=True)
                    nlater += 1
                else:
                    os.symlink(f, fd / f.name)
            col = (dd - dt.date(2026, 1, 1)).days
            noisy_assim(F.ASSIM_2026, tmp / f"assim_{d}.npz", set(I["u_id"].tolist()), rng, from_col=col)
            build(fd, tmp / f"assim_{d}.npz", tmp / f"in_{d}.npz", days=d, trace=tmp / f"tr_{d}.json")
            K = load_npz(tmp / f"in_{d}.npz")
            i = [str(x) for x in I["issue"]].index(d)
            same = True
            for k in (0, 1):
                si = int(I["issue_src"][i, k])
                sj = int(K["issue_src"][0, k])
                o0, o1 = int(I["src_off"][si]), int(I["src_off"][si + 1])
                p0, p1 = int(K["src_off"][sj]), int(K["src_off"][sj + 1])
                same &= np.array_equal(I["src_row"][o0:o1], K["src_row"][p0:p1]) and np.array_equal(I["src_val"][o0:o1], K["src_val"][p0:p1])
                same &= np.array_equal(I["dday"][si], K["dday"][sj]) and str(I["src_day"][si]) == str(K["src_day"][sj])
            tr = json.loads((tmp / f"tr_{d}.json").read_text())["opened"]
            opened_days = sorted({Path(p).stem.split("_")[1] for p in tr if "/stations_" in p})
            late = [x for x in opened_days if x >= d]
            rec(f"iv-b issue day {d}: inputs unchanged when every day >= d is noise", same and not late,
                later_files_replaced=nlater, assim_columns_from=col, day_files_opened=opened_days, opened_on_or_after_d=late)
        # ---- firewall ----------------------------------------------------------------------------------------------
        refused = {}
        for p in (F.RECORD_PAIRS, F.P / "a11/test2026/obs/obs_2026.npz", F.P / "hindcast/2026/st_truth_2026.npz",
                  F.P / "realtime_1km/verif/obs/2026-10-25.npz", F.P / "data/processed/stations/daily_2026.npz"):
            try:
                open(p, "rb").close()
                refused[str(p)] = False
            except PermissionError:
                refused[str(p)] = True
            except IsADirectoryError:
                refused[str(p)] = False
        rec("fw the firewall refuses every truth file", all(refused.values()), refused=refused)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = dict(mode="cpu", all_ok=all(r["ok"] for r in RESULTS), results=RESULTS, seconds=round(time.time() - t0, 1),
               written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    F.save_json(F.OUT / "test_cpu.json", out)
    print(f"[test_final cpu] {'ALL PASS' if out['all_ok'] else 'FAILURES'} in {time.time() - t0:.0f} s")
    sys.exit(0 if out["all_ok"] else 1)


# ---- GPU: test (i) ------------------------------------------------------------------------------------------------------
def run_gpu(a):
    import torch
    import infer_m as IM
    from cdffm_aq_m.data_aq import PixelArrays, assemble_inputs, doy_of, open_store, raw_to_device, store_kwargs, z_to_ug
    from cdffm_aq_m.lagval import CLASS_NAMES, validate_lag
    from cdffm_aq_m.model_aq import draw_sphere_noise, eval_sphere_noise, noise_chunk
    from cdffm_aq_m.stations import ASSIM_ROOT, LagStationSet, lag_features, none_features_lag
    from cdffm_aq_m.store_global import SEL_HOLD_IDX
    from cdffm_aq_m.train_aq import ValData, noise_spec
    from cdffm_aq_x.model_x import Sampler
    F.install_truth_firewall(trace=False)
    t0 = time.time()
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    sel = F.load_selection()
    margs, model, step = IM.load_m(sel["checkpoint"]["path"], dev)
    assert step == sel["checkpoint"]["step"]
    spec = noise_spec(margs)
    store = open_store(margs.store, **store_kwargs(margs))
    pix = PixelArrays(store, dev)
    stations = LagStationSet(ASSIM_ROOT, years=[2020, 2021], device=dev, clim_mean=store.clim_mean, pix_lat=store.lat, pix_lon=store.lon)
    t_load = time.time() - t0
    arm = [dict(name="M", model=model, kind="lag", grid=margs.grid, transport=margs.transport, flowmap=False)]
    days = [int(x) for x in a.sel_days.split(",")]
    val = ValData(store, a.pixels, days, heldout_file=None)
    mem = {}
    validate_lag(arm, store, pix, val, stations, [1, 2], 8, 10, spec, 0, dev, chunk=50000, baselines=False, nostations=True, members_out=mem)
    assim = {}
    for y in (2020, 2021):
        z = np.load(ASSIM_ROOT / f"assim_{y}.npz")
        assim[y] = dict(lat=np.asarray(z["lat"], np.float32), lon=np.asarray(z["lon"], np.float32), pos=np.asarray(z["pos"], np.int64),
                        daily=np.asarray(z["daily"], np.float32))
    worst = dict(nostations=0.0, none_vector=0.0, features=0.0, dmin=0.0, lag_members=0.0, sampler=0.0)
    with torch.no_grad():
        for t, d in zip(val.days, val.dates):
            k = torch.from_numpy(val.pix).to(dev)
            lat, lon, stat = pix.lat[k], pix.lon[k], pix.static[k]
            P_ = k.numel()
            doy = torch.tensor([doy_of(d)], device=dev)
            feat = model.encode(assemble_inputs(raw_to_device(store.raw_inputs(d), dev), pix, margs.grid, margs.transport))
            gen = torch.Generator(device=dev).manual_seed(int(t) * 1009 + 17)
            params = draw_sphere_noise(8, 1, spec, dev, gen)
            z0 = eval_sphere_noise(params, lat[None].expand(8, -1), lon[None].expand(8, -1), chunk=noise_chunk(8, spec))
            mu, sig = pix.mu_sigma(k[None], doy)
            ref0 = torch.from_numpy(mem[("M|nostations", 1, str(d))]).to(dev)
            x0 = z_to_ug(IM.members_from_z0(model, feat, lat, lon, stat, doy, z0, None, 10), mu, sig)
            x1 = z_to_ug(IM.members_from_z0(model, feat, lat, lon, stat, doy, z0, none_features_lag(P_, dev), 10), mu, sig)
            worst["nostations"] = max(worst["nostations"], float((x0 - ref0).abs().max()))
            worst["none_vector"] = max(worst["none_vector"], float((x1 - ref0).abs().max()))
            for lag in (1, 2):
                s = d - dt.timedelta(days=lag)
                A = assim[s.year]
                col = (s - dt.date(s.year, 1, 1)).days
                v = A["daily"][:, col]
                r = np.flatnonzero(np.isfinite(v))
                pos = A["pos"][r]
                clim = np.zeros((r.size, 5), np.float32)
                okp = pos >= 0
                clim[okp] = np.asarray(store.clim_mean[pos[okp]], np.float32)
                mo = IM.mon_from_arrays(A["lat"][r], A["lon"][r], v[r], clim, okp, s, lag, dev)
                fm, dm = lag_features(lat, lon, mo)
                fr, dr = lag_features(lat, lon, stations.lagged(d, lag))
                worst["features"] = max(worst["features"], float((fm - fr).abs().max()))
                worst["dmin"] = max(worst["dmin"], float((dm - dr).abs().max()))
                xl = z_to_ug(IM.members_from_z0(model, feat, lat, lon, stat, doy, z0, fm, 10), mu, sig)
                refl = torch.from_numpy(mem[("M", lag, str(d))]).to(dev)
                worst["lag_members"] = max(worst["lag_members"], float((xl - refl).abs().max()))
                if lag == 1:                                   # production Sampler (prior None, guard) = members_from_z0
                    base = 0 * 1_000_003 + 1 * 1009 + 3 * 7919
                    smp = Sampler(model, "base", spec, 1, 10, dev, cm=None, flowmap=False, prior=None, z_bound=10.0, z0_trunc=3.0,
                                  stn_fn=lambda la, lo, fm=fm: fm)
                    smp.start_day(feat, torch.Generator(device=dev).manual_seed(base + 23),
                                  gen_coarse=torch.Generator(device=dev).manual_seed(base + 37),
                                  gen_prior=torch.Generator(device=dev).manual_seed(base + 41))
                    zs = smp.members(lat, lon, stat, doy)
                    pb = draw_sphere_noise(1, 1, spec, dev, torch.Generator(device=dev).manual_seed(base + 23))
                    zb = eval_sphere_noise(pb, lat[None], lon[None], chunk=noise_chunk(1, spec))
                    zm = IM.members_from_z0(model, feat, lat, lon, stat, doy, zb, fm, 10, z_bound=10.0, z0_trunc=3.0)
                    worst["sampler"] = max(worst["sampler"], float((zs - zm).abs().max()))
    tol = 1e-4
    rec("i-a M with every monitor hidden = validate_lag 'M|nostations' (members, ug m-3)", worst["nostations"] <= tol, max_abs=worst["nostations"])
    rec("i-b explicit no-monitor vector = stn None", worst["none_vector"] <= tol, max_abs=worst["none_vector"])
    rec("i-c features of mon_from_arrays = LagStationSet.lagged (2021, lags 1, 2)", worst["features"] == 0.0 and worst["dmin"] == 0.0,
        max_abs_features=worst["features"], max_abs_dmin_km=worst["dmin"])
    rec("i-d members with those features = validate_lag 'M' members (lags 1, 2)", worst["lag_members"] <= tol, max_abs=worst["lag_members"])
    rec("i-e production Sampler (prior None, guard) = members_from_z0", worst["sampler"] <= 1e-6, max_abs_z=worst["sampler"])
    # stored 2021 table of the selected checkpoint, lag 1, on the full SEL set
    t1 = time.time()
    if a.skip_full:                                         # dry run on a CPU (its noise generator differs from the GPU's)
        print("[test_final gpu] --skip-full: the stored-table reproduction (i-f) is not run", flush=True)
        sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)
    valf = ValData(store, 100000, list(SEL_HOLD_IDX), heldout_file=None)
    rf = validate_lag(arm, store, pix, valf, stations, [1], 8, 10, spec, 0, dev, chunk=50000, baselines=False, nostations=True)
    stored = json.loads(F.SEL2021.read_text())["scores"]
    nm = f"M@{step}"
    devs = {}
    for lb, sl in (("M", nm), ("M|nostations", nm + "|nostations")):
        for c in CLASS_NAMES:
            for key in ("crps_fair", "rmse_ensmean"):
                a_, b_ = rf["scores"][lb]["1"][c][key], stored[sl]["1"][c][key]
                devs[f"{lb}|{c}|{key}"] = abs(a_ / b_ - 1)
    mx = max(devs.values())
    rec(f"i-f stored 2021 table of {nm} reproduced (lag 1 and |nostations, 5 classes, CRPS and RMSE)", mx <= 1e-4, max_rel_dev=mx,
        crps_all=rf["scores"]["M"]["1"]["all"]["crps_fair"], stored_crps_all=stored[nm]["1"]["all"]["crps_fair"],
        crps_nostations_all=rf["scores"]["M|nostations"]["1"]["all"]["crps_fair"], seconds=round(time.time() - t1, 1))
    out = dict(mode="gpu", all_ok=all(r["ok"] for r in RESULTS), results=RESULTS, load_seconds=round(t_load, 1),
               seconds=round(time.time() - t0, 1), job=os.environ.get("SLURM_JOB_ID"),
               written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    F.save_json(F.OUT / f"test_gpu_{os.environ.get('SLURM_JOB_ID', 'local')}.json", out)
    print(f"[test_final gpu] {'ALL PASS' if out['all_ok'] else 'FAILURES'} in {time.time() - t0:.0f} s")
    sys.exit(0 if out["all_ok"] else 1)


# ---- check of the smoke outputs, price ---------------------------------------------------------------------------------
def run_check_smoke(a):
    F.install_truth_firewall(trace=False)
    S = load_npz(a.smoke)
    Lk = load_npz(a.leak) if a.leak else None
    I = load_npz(F.INPUTS)
    days = [str(x) for x in S["days"]]
    assert len(days) == 1
    d = days[0]
    p = json.loads(str(S["prov"][0]))
    rec_km = np.load(F.POINTS / f"{d}.npz")["km"]
    ide = {}
    for Ld in p["leads_made"]:
        x, y = S["free"][0, Ld - 1], rec_km[Ld - 1]
        both = np.isfinite(x) & np.isfinite(y)
        dd = np.abs(x[both].astype(np.float64) - y[both])
        ide[Ld] = dict(max_abs=float(dd.max()), max_rel=float((dd / np.maximum(np.abs(y[both]), 1.0)).max()),
                       r=float(np.corrcoef(x[both], y[both])[0, 1]), same_nan=bool(np.array_equal(np.isfinite(x), np.isfinite(y))),
                       identical=float((dd == 0).mean()))
    ok = all(v["max_rel"] <= 1e-3 and v["r"] >= 0.999999 and v["same_nan"] for v in ide.values()) and len(ide) == 5
    rec(f"ii station-free arm through this code path = the record's km at the monitor pixels ({d})", ok, by_lead=ide)
    if Lk is not None:
        dm = {k: float(np.nanmax(np.abs(S[k] - Lk[k]))) for k in ("m1", "m2")}
        same = {k: bool(np.array_equal(S[k], Lk[k], equal_nan=True)) for k in ("m1", "m2")}
        rec("iii-b leak test of the forecasts: perturbed held-out rows -> identical M members", all(same.values()), bit_identical=same, max_abs=dm)
    made1 = [Ld - 1 for Ld in p["leads_made"]]
    made2 = [Ld - 1 for Ld in p["leads_made"] if Ld <= 4]
    fin = dict(m1=bool(np.isfinite(S["m1"][0][made1]).all()), m2=bool(np.isfinite(S["m2"][0][made2]).all()))
    rec("finite members of m1 (lead days made) and m2 (lead days 1-4)", all(fin.values()), finite=fin,
        m2_lead5_all_nan=bool(np.isnan(S["m2"][0, 4]).all()))
    i = [str(x) for x in I["issue"]].index(d)
    dref = I["dday"][I["issue_src"][i, 0]]
    dd1 = np.abs(S["dmin1"][0].astype(np.float64) - dref)
    agree = float((F.dist_class(S["dmin1"][0]) == F.dist_class(dref)).mean())
    rec("distances seen by the network = the inputs' distances", agree >= 0.999 and float(np.nanmax(dd1)) < 0.05,
        class_agreement=agree, max_abs_km=float(np.nanmax(dd1)))
    gh = p["guard_hits"]
    pts = p["points_per_member"] * p["members"]
    shares = {f"{Ld}|{arm}": dict(bound_or_cap=(v["at_bound"] + v["at_pmax"]) / pts, trunc=v["z0_trunc"] / pts, max_ug=v.get("max_ug"))
              for Ld, hl in gh.items() for arm, v in hl.items()}
    rec("guard: member points at the bound or the cap <= 1e-4, truncations <= 0.6 %",
        all(v["bound_or_cap"] <= 1e-4 and v["trunc"] <= 0.006 for k, v in shares.items() if not k.endswith("free")), shares=shares)
    held = I["cat_heldout_place"]
    near = dref <= F.load_selection()["R_km"]
    descr = dict(heldout_monitors=int(held.sum()), heldout_within_R=int((held & near).sum()),
                 mean_m1_minus_free_at_heldout_within_R_lead1=float(np.nanmean(S["m1"][0, 0][:, held & near] - S["free"][0, 0][:, held & near])),
                 note="forecast values only; no observation")
    # price: the full run = 48 issue days, arms m1 + m2 + free (identity), one GCD
    per_day = p["seconds"]
    load = p["load_seconds"]
    proj_h = (load + 48 * per_day) / 3600.0
    cap_h = math.ceil(2.0 * proj_h / 0.25) * 0.25
    price = dict(day=d, seconds_per_issue_day=per_day, timing_s=p["timing_s"], load_seconds=load,
                 projected_gcd_hours=round(proj_h, 3), cap_gcd_hours=max(cap_h, 0.5),
                 method="(model load + 48 x the measured seconds of the smoke issue day with the same arms) / 3600 on 1 GCD; "
                        "cap = twice the projection rounded up to 0.25 h (at least 0.5 h) = the job's time limit")
    rec("price of the full run", True, **price)
    out = dict(mode="check-smoke", all_ok=all(r["ok"] for r in RESULTS), results=RESULTS, price=price, description=descr,
               smoke=str(a.smoke), leak=str(a.leak), written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    F.save_json(Path(a.out), out)
    print(f"[test_final check-smoke] {'ALL PASS' if out['all_ok'] else 'FAILURES'}; projected {proj_h:.2f} GCD-h, cap {price['cap_gcd_hours']} -> {a.out}")
    sys.exit(0 if out["all_ok"] else 1)


# ---- check of the CPU dry run ------------------------------------------------------------------------
def run_check_dry(a):
    F.install_truth_firewall(trace=False)
    S = load_npz(a.dry)
    p = json.loads(str(S["prov"][0]))
    K, made = int(p["members"]), [Ld - 1 for Ld in p["leads_made"]]
    rec("full-run guard: infer_m.py --days all refused a non-frozen configuration (before loading anything)", a.refused_ok == 0,
        refused_ok_rc=a.refused_ok)
    rec("provenance of the dry run", p["arms"] == ["free", "free0", "m1", "m2"] and p.get("prior_free0", "x") is None and p["device"] == "cpu",
        arms=p["arms"], prior_free0=p.get("prior_free0"), device=p["device"], members=K, leads_made=p["leads_made"], seconds=p["seconds"])
    fin, empty = {}, {}
    for k in ("free", "free0", "m1", "m2"):
        mk = [x for x in made if not (k == "m2" and x == 4)]
        fin[k] = bool(np.isfinite(S[k][0][mk][:, :K]).all())
        rest = np.ones(S[k][0].shape[:2], bool)
        rest[np.ix_(mk, range(K))] = False
        empty[k] = bool(np.isnan(S[k][0][rest]).all())
    rec("finite members where made, NaN elsewhere (m2 lead day 5 empty by design)", all(fin.values()) and all(empty.values()),
        finite=fin, nan_elsewhere=empty)
    cmp = {}
    for x, y in (("free0", "free"), ("free0", "m1"), ("m1", "m2")):
        mk = [Ld - 1 for Ld in p["leads_made"] if not ("m2" in (x, y) and Ld == 5)]
        u, v = S[x][0][mk][:, :K].ravel().astype(np.float64), S[y][0][mk][:, :K].ravel().astype(np.float64)
        cmp[f"{x}-{y}"] = dict(max_abs=float(np.abs(u - v).max()), r=float(np.corrcoef(u, v)[0, 1]), mean_ratio=float(u.mean() / v.mean()))
    c = cmp["free0-free"]
    rec("free0 differs from free (the fitted prior) but is the same network: r >= 0.75 (two record members with different "
        "noise: r 0.88-0.90 on 2026-10-20), mean ratio within 0.8-1.25",
        c["max_abs"] > 0 and c["r"] >= 0.75 and 0.8 <= c["mean_ratio"] <= 1.25 and cmp["free0-m1"]["max_abs"] > 0, **cmp)
    gh = p["guard_hits"]
    tr = {Ld: {k: v["z0_trunc"] for k, v in hl.items()} for Ld, hl in gh.items()}
    same = all(t["free0"] == t["m1"] and (int(Ld) == 5 or t["m2"] == t["m1"]) for Ld, t in tr.items())
    rec("free0 shares the base noise of m1 / m2 (identical truncation counts of |z0| > 3)", same and all(t["m1"] > 0 for t in tr.values()), trunc=tr)
    ide = {Ld: v.get("max_rel") for Ld, v in p["identity"].items()}
    rec("identity of free with the record on CPU (reported only: the CPU noise generator differs from the GPU's)", True, max_rel=ide)
    out = dict(mode="check-dry", all_ok=all(r["ok"] for r in RESULTS), results=RESULTS, dry=str(a.dry),
               written_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    F.save_json(Path(a.out), out)
    print(f"[test_final check-dry] {'ALL PASS' if out['all_ok'] else 'FAILURES'} -> {a.out}")
    sys.exit(0 if out["all_ok"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cpu", "gpu", "check-smoke", "check-dry"])
    ap.add_argument("--keep-leak", default=None, help="cpu: keep the inputs built from the perturbed copies here")
    ap.add_argument("--sel-days", default="0,180", help="gpu: SEL hold indices of the member-level comparison")
    ap.add_argument("--pixels", type=int, default=4096)
    ap.add_argument("--skip-full", action="store_true", help="gpu: dry run without i-f (CPU)")
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--leak", default=None)
    ap.add_argument("--out", default=str(F.OUT / "smoke_check.json"))
    ap.add_argument("--dry", default=None, help="check-dry: the dry-run forecasts")
    ap.add_argument("--refused-ok", type=int, default=1, help="check-dry: 0 if the full-run guard refused the wrong configuration")
    a = ap.parse_args()
    {"cpu": run_cpu, "gpu": run_gpu, "check-smoke": run_check_smoke, "check-dry": run_check_dry}[a.mode](a)


if __name__ == "__main__":
    main()
