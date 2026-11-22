#!/usr/bin/env python3
"""Tests of A11 experiment M (lagged monitors as an input; a11/stageM/README.md). Sub-commands:

  cpu      no data file of the project is read (toy store + a synthetic assimilation set whose values ENCODE THEIR DATE):
           C1 loader: every value served for (valid day t, lag l) is a value of day t - l, across year ends and the leap
              day; lag >= 1 never serves a value of day t; lag 0 = StationSet.day; the A3 dropout is unchanged; days before
              the archive have no monitor; a missing year raises; a held-out id in an assimilation file is refused; the
              firewall refuses the held-out value files;
           C2 features: the first 41 entries are cdffm_aq.stations.station_features of the monitors of day t - l; anomaly
              = log1p(max(v, 1)) - climatology of the monitor's own pixel on day t - l; flags; lag one-hot; the no-monitor
              vector is the same for every lag; chunk independence;
           C3 model and training: with the option off the copy has the original's parameters; with it the station MLP
              takes 63 inputs; compute_losses draws the lag, asks the loader for day t - lag only, back-propagates;
           C4 faithfulness (test ii) on the toy store: with the lag option off the loss of one fixed batch is that of the
              ORIGINAL package (no stations, and --stations), same seed, same batch;
           C5 end to end on the toy store: train_aq with --stn-lag (6 steps, 2 validations, checkpoints), then val_lag.py;
              a --stations run with the option off validates without opening the held-out monitor file;
           C6 the same toy training on two ranks (CPU DDP): identical weights across ranks, lag validation on rank 0.
  real     the real assimilation files 2017-2021 (assimilated monitors only; CPU, the climatology is memory-mapped):
           R1 pos is the monitor's own pixel; R2 over random (day, lag, dropout) draws the loader returns no held-out id
              (checked here against split_v1_public.json, besides the loader's own assert) and exactly column t - lag;
           R3 the anomaly is the package's own climatology (PixelArrays.mu_sigma) at the monitor's pixel.
  ghap     GPU, the GHAP store: test (ii) with the real data: one fixed batch of 2 days x 16,384 points, original package
           against the copy with the lag option off, without and with --stations: same batch, same loss to 1e-5, same
           gradient norm.
  python tests/test_m.py cpu|real|ghap  -> tests/out/test_m_<sub>.json, exit 0 = pass
"""
import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
for p_ in (str(STAGE), str(ROOT)):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

RES = {}


def check(name, ok, **info):
    RES[name] = dict(ok=bool(ok), **info)
    print(f"[{'OK' if ok else 'FAIL'}] {name} {info}", flush=True)


def finish(sub):
    out = HERE / "out" / f"test_m_{sub}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = all(v["ok"] for v in RES.values())
    out.write_text(json.dumps({"passed": ok, "n_checks": len(RES), "checks": RES}, indent=1, default=str))
    print(f"[test_m {sub}] {'PASS' if ok else 'FAIL'} ({sum(v['ok'] for v in RES.values())}/{len(RES)}) -> {out}", flush=True)
    sys.exit(0 if ok else 1)


def n_days(y):
    return 366 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 365


def code_of(d):
    """the canary value of a date: exactly representable in float32, different for every day of 2017-2022"""
    return float((d.year - 2016) * 400 + (d - dt.date(d.year, 1, 1)).days + 1)


def make_canary(tmp, store, years=(2019, 2020, 2021), n=40, seed=0, heldout_inside=None, hole_frac=0.2):
    """synthetic assimilation root on the toy box: values = code_of(date); 3 monitors outside the mask (pos = -1), one
    monitor far away (> 300 km from the box)"""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(seed)
    pos = np.sort(g.choice(store.n_pix, n, replace=False)).astype(np.int64)
    lat, lon = store.lat[pos].astype(np.float32), store.lon[pos].astype(np.float32)
    lat = np.concatenate([lat, np.float32([45.123, 45.5, 45.9, 52.0])])
    lon = np.concatenate([lon, np.float32([10.3, 11.1, 11.9, 25.0])])
    pos = np.concatenate([pos, np.int64([-1, -1, -1, -1])])
    N = lat.size
    ids = [f"TOY:{i}" for i in range(N)]
    if heldout_inside:
        ids[5] = heldout_inside
    (tmp / "split_v1_public.json").write_text(json.dumps({"heldout_ids": ["TOY:H0", "TOY:H1"], "n_heldout": 2,
                                                          "split_v1_sha256": "synthetic-test"}))
    for y in years:
        nd = n_days(y)
        daily = np.tile(np.float32([code_of(dt.date(y, 1, 1) + dt.timedelta(k)) for k in range(nd)]), (N, 1))
        daily[g.random((N, nd)) < hole_frac] = np.nan
        np.savez(tmp / f"assim_{y}.npz", id=np.array(ids), network=np.array(["TOY"] * N), lat=lat, lon=lon, pos=pos,
                 daily=daily, year=y, split_sha256="synthetic-test")
    return dict(root=tmp, ids=ids, lat=lat, lon=lon, pos=pos, N=N)


# =================================================================================================================
# cpu
# =================================================================================================================

def sub_cpu():
    import cdffm_aq.data_aq as od
    import cdffm_aq.model_aq as om
    import cdffm_aq.stations as ost
    import cdffm_aq.train_aq as ot
    import cdffm_aq_m.data_aq as cd
    import cdffm_aq_m.model_aq as cm
    import cdffm_aq_m.stations as cst
    import cdffm_aq_m.store_global as csg
    import cdffm_aq_m.train_aq as ct
    torch.manual_seed(0)
    dev = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp(prefix="stageM_test_", dir=os.environ.get("STAGEM_TMP")))
    print(f"[test_m cpu] temporary files in {tmp}", flush=True)
    st = cd.ToyStore()
    pix = cd.PixelArrays(st, dev)
    can = make_canary(tmp / "assim_ok", st)
    # files that the firewall must refuse once a LagStationSet exists (written BEFORE the firewall is installed)
    fw1 = tmp / "assim_ok" / "heldout_sel2021.npz"
    fw2 = tmp / "data" / "processed" / "stations" / "daily_2021.npz"
    fw2.parent.mkdir(parents=True)
    for f in (fw1, fw2):
        np.savez(f, x=np.zeros(1))
    make_canary(tmp / "assim_bad", st, heldout_inside="TOY:H1")
    L = cst.LagStationSet(can["root"], years=(2019, 2020, 2021), device=dev, clim_mean=st.clim_mean, pix_lat=st.lat, pix_lon=st.lon)
    A3 = cst.StationSet(can["root"], years=(2019, 2020, 2021), device=dev)

    # ---- C1 loader
    days = [dt.date(2020, 12, 28) + dt.timedelta(k) for k in range(12)] + [dt.date(2020, 2, 27) + dt.timedelta(k) for k in range(6)] \
        + [dt.date(2019, 12, 30) + dt.timedelta(k) for k in range(5)] + [dt.date(2021, 6, 15), dt.date(2021, 12, 31)]
    n_calls = n_bad_src = n_same_day = n_vals = 0
    for d in days:
        for lag in range(0, 6):
            mon = L.lagged(d, lag)
            n_calls += 1
            v = mon["val"].numpy()
            n_vals += v.size
            n_bad_src += int((v != code_of(d - dt.timedelta(lag))).sum()) + int(mon["src"] != d - dt.timedelta(lag))
            if lag >= 1:
                n_same_day += int((v == code_of(d)).sum())
    check("C1_values_are_those_of_day_t_minus_lag", n_bad_src == 0 and n_vals > 0, calls=n_calls, values=n_vals, wrong=n_bad_src)
    check("C1_lag_ge_1_never_serves_the_valid_day", n_same_day == 0, values_of_day_t_found=n_same_day)
    same = True
    for d in days[:8]:
        for f in (None, 0.0, 0.37, 1.0):
            g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
            a_ = A3.day(d, hide=f, gen=g1)
            m_ = L.lagged(d, 0, hide=f, gen=g2)
            same &= torch.equal(a_[0], m_["lat"]) and torch.equal(a_[1], m_["lon"]) and torch.equal(a_[2], m_["val"]) \
                and torch.equal(a_[3], m_["xyz"])
    check("C1_lag0_equals_StationSet_day_with_same_dropout", same)
    d = dt.date(2020, 3, 3)
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    a_ = A3.day(d - dt.timedelta(3), hide=0.5, gen=g1)
    m_ = L.lagged(d, 3, hide=0.5, gen=g2)
    full = L.lagged(d, 3)
    check("C1_dropout_of_the_lagged_day_is_A3s", torch.equal(a_[2], m_["val"]) and torch.equal(a_[0], m_["lat"])
          and 0 < m_["val"].numel() < full["val"].numel(), kept=int(m_["val"].numel()), of=int(full["val"].numel()))
    L17 = cst.LagStationSet(can["root"], years=(2019,), device=dev, clim_mean=st.clim_mean)
    try:
        L17.lagged(dt.date(2021, 1, 2), 1)
        check("C1_missing_year_raises", False)
    except KeyError as e:
        check("C1_missing_year_raises", True, error=str(e)[:80])
    e_ = L.lagged(dt.date(2017, 1, 3), 5)
    f_none, dm_none = cst.lag_features(pix.lat[:50], pix.lon[:50], e_)
    check("C1_day_before_the_archive_has_no_monitor", e_["val"].numel() == 0 and e_["src"] == dt.date(2016, 12, 29)
          and torch.equal(f_none, cst.none_features_lag(50, dev)) and bool(torch.isinf(dm_none).all()))
    try:
        cst.LagStationSet(tmp / "assim_bad", years=(2019, 2020, 2021), device=dev, clim_mean=st.clim_mean)
        check("C1_heldout_id_in_a_file_is_refused", False)
    except AssertionError as e:
        check("C1_heldout_id_in_a_file_is_refused", "held-out" in str(e), error=str(e)[:90])
    ids_seen = set()
    for d in days:
        for lag in range(6):
            ids_seen.update(L.ids_of(L.lagged(d, lag, hide=0.3, gen=torch.Generator().manual_seed(lag))))
    check("C1_ids_served_are_assimilation_ids", ids_seen <= set(can["ids"]) and not ids_seen & L.heldout_ids and len(ids_seen) > 30,
          n_ids=len(ids_seen))
    blocked = []
    for f in (fw1, fw2, ROOT / "data/processed/assim/heldout_sel2021.npz", ROOT / "data/processed/stations/daily_2021.npz",
              ROOT / "data/processed/stations/daily_2026.npz", ROOT / "data/processed/stations/split_v1.json"):
        try:
            csg._FIREWALL[0]("open", (str(f),))              # the hook itself: the real files are not touched
            blocked.append(False)
        except PermissionError:
            blocked.append(True)
    opened = []
    for f in (fw1, fw2):                                     # synthetic files with the same markers: a real open()
        try:
            open(f, "rb").close()
            opened.append(True)
        except PermissionError:
            opened.append(False)
    ok_open = True
    try:
        np.load(can["root"] / "assim_2021.npz")["lat"]       # an assimilation file stays readable
    except PermissionError:
        ok_open = False
    check("C1_firewall_refuses_heldout_value_files", all(blocked) and not any(opened) and ok_open, hook_refused=blocked,
          synthetic_opened=opened, assimilation_file_readable=ok_open)

    # ---- C2 features
    qlat, qlon = pix.lat[::37][:300], pix.lon[::37][:300]
    eq41 = eqd = True
    for d in days[:10]:
        for lag in range(6):
            mon = L.lagged(d, lag)
            f63, dmin = cst.lag_features(qlat, qlon, mon)
            f41, dmin0 = ost.station_features(qlat, qlon, mon["lat"], mon["lon"], mon["val"], mxyz=mon["xyz"])   # ORIGINAL
            eq41 &= torch.equal(f63[:, :41], f41)
            eqd &= torch.equal(dmin, dmin0)
    check("C2_first_41_features_are_the_original_station_features", eq41 and eqd and f63.shape == (300, 63), shape=list(f63.shape))
    d, lag = dt.date(2021, 1, 2), 4                          # source day 2020-12-29 (previous, leap year)
    mon = L.lagged(d, lag)
    f63, dmin = cst.lag_features(mon["lat"], mon["lon"], mon)                 # queries AT the monitors: slot 0 = itself
    near = dmin < 1e-3
    src = d - dt.timedelta(lag)
    doy = torch.tensor([(src - dt.date(src.year, 1, 1)).days])
    yv = L.years[src.year]
    pos = torch.from_numpy(yv["pos"])[mon["idx"]]
    inm = pos >= 0
    mu = torch.zeros(pos.numel())
    mu[inm] = pix.mu_sigma(pos[inm][None], doy)[0][0]                         # the package's own climatology
    want = (torch.log1p(mon["val"].clamp(min=1.0)) - mu) * inm
    err = float((f63[:, cst.LAG_A0] - want)[near].abs().max())
    flags_ok = torch.equal(f63[near, cst.LAG_F0], inm[near].float())
    val_ok = bool((torch.expm1(f63[near, 0]) - code_of(src)).abs().max() < 0.5)
    check("C2_anomaly_is_log1p_value_minus_pixel_climatology_of_day_t_minus_lag", err < 1e-5 and flags_ok and val_ok
          and int(near.sum()) >= 25 and int((~inm[near]).sum()) >= 1, max_abs_err=err, n=int(near.sum()),
          n_outside_mask=int((~inm[near]).sum()), decoded_source_day=str(src))
    oh_ok = none_same = True
    far_lat, far_lon = torch.tensor([-30.0, 10.0]), torch.tensor([100.0, -60.0])
    for lag in range(6):
        mon = L.lagged(dt.date(2021, 6, 15), lag)
        f63, _ = cst.lag_features(qlat, qlon, mon)
        oh = f63[:, cst.LAG_L0:]
        anyv = f63[:, 40] == 0
        oh_ok &= bool((oh.sum(1) == anyv.float()).all()) and bool((oh[anyv].argmax(1) == lag).all()) and bool(anyv.all())
        ff, _ = cst.lag_features(far_lat, far_lon, mon)
        none_same &= torch.equal(ff, cst.none_features_lag(2, dev))
    check("C2_lag_one_hot_and_single_no_monitor_vector", oh_ok and none_same)
    mon = L.lagged(dt.date(2021, 6, 15), 2)
    f_a, d_a = cst.lag_features(qlat, qlon, mon)
    f_b, d_b = cst.lag_features(qlat, qlon, mon, chunk=7)
    check("C2_chunk_independent", torch.allclose(f_a, f_b, atol=1e-6) and torch.allclose(d_a, d_b, atol=1e-4) and bool(torch.isfinite(f_a).all()))
    spec_l = dict(lags=(1, 2, 3, 4, 5), p0=0.1)
    draws = np.array([cst.draw_lag(0, s, t, spec_l) for s in range(4000) for t in range(5)])
    freq = [float((draws == k).mean()) for k in range(6)]
    rep = [cst.draw_lag(3, 17, 5, spec_l) for _ in range(3)]
    check("C2_draw_lag_distribution_and_determinism", abs(freq[0] - 0.1) < 0.01 and all(abs(f - 0.18) < 0.012 for f in freq[1:])
          and len(set(rep)) == 1 and set(draws.tolist()) == set(range(6)), freq=[round(f, 4) for f in freq])
    draws0 = np.array([cst.draw_lag(0, s, 0, dict(lags=(1, 2, 3, 4, 5), p0=0.0)) for s in range(3000)])
    check("C2_draw_lag_without_lag0", draws0.min() == 1 and draws0.max() == 5)

    # ---- C3 model and training
    base = dict(grid="025", transport="t6", objective="fm", lifting_dim=None, use_checkpoint=False, doy_cond=True, sigma_k=0,
                seed=0, batch_size=2, n_points=256, fm_substeps=8, fm_diag_prob=0.25)
    spec = cm.NoiseSpec(64, 2.175, 4.3, 500.0)
    grid = cm.GRIDS["toy"]
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    k_ok = True
    for stn in (None, str(can["root"])):
        a_o = SimpleNamespace(**base, stations=stn)
        torch.manual_seed(0)
        m_o = om.build_model(a_o, 47, st.static_dim, grid)
        torch.manual_seed(0)
        m_c = cm.build_model(SimpleNamespace(**base, stations=stn, stn_lag=None, stn_lag0_prob=0.0), 47, st.static_dim, grid)
        k_ok &= shapes(m_o) == shapes(m_c) and all(torch.equal(v, m_c.state_dict()[k]) for k, v in m_o.state_dict().items())
    check("C3_option_off_same_parameters_and_init_as_the_original", k_ok)
    a_l = SimpleNamespace(**base, stations=str(can["root"]), stn_lag="1,2,3,4,5", stn_lag0_prob=0.1)
    torch.manual_seed(0)
    m_l = cm.build_model(a_l, 47, st.static_dim, grid)
    check("C3_lag_model_station_mlp_takes_63", m_l.stn_dim == 63 and m_l.stn_mlp[0].in_features == 63
          and m_l.velocity_net.net[0].in_features == 1 + 98 + 64 and cst.stn_dim_of(a_l) == 63
          and cst.stn_dim_of(SimpleNamespace(stations="x")) == 41 and cst.stn_dim_of(SimpleNamespace()) == 0,
          in_features=int(m_l.stn_mlp[0].in_features))
    stream = cd.BatchStream(st, 2, 256, 0, 0, dev)
    b = stream.next()
    stream.close()
    kk = b["k"]
    lat, lon = pix.lat[kk], pix.lon[kk]
    x = torch.randn(2, 47, 72, 144)
    with torch.no_grad():
        feat = m_l.encode(x)
        c_none = m_l.condition(feat, lat, lon, pix.static[kk], b["doy"])
        c_zero = m_l.condition(feat, lat, lon, pix.static[kk], b["doy"], stn=cst.none_features_lag(256, dev)[None].expand(2, -1, -1))
        f63, _ = cst.lag_features(lat[0], lon[0], L.lagged(dt.date(2020, 10, 3), 2))
        c_st = m_l.condition(feat[:1], lat[:1], lon[:1], pix.static[kk[:1]], b["doy"][:1], stn=f63[None])
    try:
        m_l.condition(feat[:1], lat[:1], lon[:1], pix.static[kk[:1]], b["doy"][:1], stn=f63[None, :, :41])
        refused = False
    except AssertionError:
        refused = True
    check("C3_no_monitor_condition_and_feature_size_check", torch.equal(c_none, c_zero) and c_none.shape[-1] == 98
          and not torch.equal(c_none[:1], c_st) and refused)

    class Spy:
        def __init__(self, inner):
            self.inner, self.calls = inner, []

        def lagged(self, d, lag, hide=None, gen=None):
            mon = self.inner.lagged(d, lag, hide=hide, gen=gen)
            self.calls.append((d, int(lag), mon["src"], hide, mon["val"].clone()))
            return mon

    Lt = cst.LagStationSet(can["root"], years=(2020, 2021), device=dev, clim_mean=st.clim_mean, pix_lat=st.lat, pix_lon=st.lon)
    spy = Spy(Lt)
    gen = torch.Generator().manual_seed(0)
    dates = [st.train_dates[int(t)] for t in b["day"]]
    loss = ct.compute_losses(m_l, pix, b, a_l, spec, gen, stations=spy, dates=dates, step=3)
    loss.backward()
    gn = float(sum(p.grad.norm() ** 2 for p in m_l.stn_mlp.parameters()) ** 0.5)
    for s_ in range(4, 64):
        with torch.no_grad():
            ct.compute_losses(m_l, pix, b, a_l, spec, gen, stations=spy, dates=dates, step=s_)
    ok_calls = all(src == d_ - dt.timedelta(lag) and 0 <= lag <= 5 and hide is not None and bool((v == code_of(d_ - dt.timedelta(lag))).all())
                   and (lag == 0 or not bool((v == code_of(d_)).any())) for d_, lag, src, hide, v in spy.calls)
    lags_seen = sorted({c[1] for c in spy.calls})
    want_lags = [cst.draw_lag(0, 3, int(b["day"][i]), cst.lag_spec(a_l)) for i in range(2)]
    check("C3_compute_losses_reads_only_day_t_minus_lag_and_backpropagates", bool(torch.isfinite(loss)) and gn > 0 and ok_calls
          and len(spy.calls) == 2 * 61 and [c[1] for c in spy.calls[:2]] == want_lags and len(lags_seen) >= 5,
          loss=float(loss), stn_grad_norm=gn, calls=len(spy.calls), lags_seen=lags_seen)
    for bad in (dict(stn_lag="1,2", stations=None), dict(stn_lag="0,1", stations="x"), dict(stn_lag="1,6", stations="x"),
                dict(stn_lag=None, stn_lag0_prob=0.1, stations="x")):
        try:
            cst.lag_spec(SimpleNamespace(**bad))
            check(f"C3_bad_options_refused_{bad}", False)
        except SystemExit:
            pass
    check("C3_bad_lag_options_refused", True)
    check("C3_lag_spec_off_by_default", cst.lag_spec(SimpleNamespace(stations="x")) is None
          and ct.parser().parse_args(["--name", "x"]).stn_lag is None and ct.parser().parse_args(["--name", "x"]).stn_lag0_prob == 0.0)

    # ---- C4 faithfulness on the toy store (test ii): original package vs the copy with the option off
    faithful_checks("C4_toy", faithful("toy", dev, {"no_stations": None, "stations": str(can["root"])}, n_points=256,
                                       noise=(64, 2.175, 4.3, 500.0), years=(2020, 2021)), 1e-5)

    # ---- C5 end to end on the toy store: training with the lag option, then val_lag.py
    runs = tmp / "runs"
    argv = ["--name", "toy_lag", "--runs-root", str(runs), "--store", "toy", "--device", "cpu", "--batch-size", "2", "--n-points", "512",
            "--max-steps", "6", "--val-every", "3", "--val-pixels", "1500", "--val-members", "4", "--val-ode-steps", "3",
            "--log-every", "2", "--noise-features", "64", "--warmup", "2", "--stations", str(can["root"]),
            "--stn-lag", "1,2,3,4,5", "--stn-lag0-prob", "0.1", "--val-lags", "1,5", "--sigma-k", "0"]
    module, trainer = ct.main(argv)
    rows = [json.loads(l) for l in open(runs / "toy_lag" / "metrics.jsonl") if l.strip()]
    args_j = json.loads((runs / "toy_lag" / "args.json").read_text())
    fin = all(math.isfinite(r["model"]["ug"]["crps_fair"]) and r["lag"]["info"]["members_finite"] and r["lag"]["info"]["scores_finite"]
              for r in rows)
    sc = rows[-1]["lag"]["scores"]
    check("C5_toy_training_with_lag_option", len(rows) == 2 and fin and args_j["stn_dim"] == 63 and args_j["stn_lag"] == "1,2,3,4,5"
          and set(sc) == {"model", "model|nostations"} and set(sc["model"]) == {"1", "5"} and "stations" not in rows[-1]
          and (runs / "toy_lag" / "ckpt" / "step000006.ckpt").exists() and module.val.held is None,
          steps=[r["step"] for r in rows], crps=[round(r["model"]["ug"]["crps_fair"], 3) for r in rows],
          crps_lag1_all=sc["model"]["1"]["all"]["crps_fair"], classes_n={c: (v or {}).get("n") for c, v in sc["model"]["1"].items()})
    # option off + --stations (the A3 path): the copy trains and validates WITHOUT opening <stations>/heldout_sel2021.npz
    # (the synthetic one above exists and is behind the firewall: opening it would raise)
    argv3 = ["--name", "toy_a3", "--runs-root", str(runs), "--store", "toy", "--device", "cpu", "--batch-size", "2", "--n-points", "512",
             "--max-steps", "3", "--val-every", "3", "--val-pixels", "1500", "--val-members", "4", "--val-ode-steps", "3",
             "--log-every", "2", "--noise-features", "64", "--warmup", "2", "--stations", str(can["root"]), "--sigma-k", "0"]
    mod3, _ = ct.main(argv3)
    r3 = [json.loads(l) for l in open(runs / "toy_a3" / "metrics.jsonl") if l.strip()][-1]
    check("C5_option_off_stations_run_never_opens_the_heldout_file", fw1.exists() and mod3.val.held is None and "lag" not in r3
          and r3["stations"]["heldout"] is None and r3["stations"]["model_uses_stations"] is True
          and math.isfinite(r3["model"]["ug"]["crps_fair"]) and json.loads((runs / "toy_a3" / "args.json").read_text())["stn_dim"] == 41,
          crps=r3["model"]["ug"]["crps_fair"], by_distance={k: v["crps_fair"] for k, v in r3["stations"]["by_distance"].items()})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cp = subprocess.run([sys.executable, "-u", str(STAGE / "val_lag.py"), "--run", "toy_lag", "--runs-root", str(runs), "--ckpt", "all",
                         "--free", "", "--naive", "", "--days", "0,1,2", "--pixels", "1500", "--lags", "0,1,5", "--members", "4",
                         "--ode-steps", "3", "--assim-root", str(can["root"]), "--save-members", "--tag", "toy", "--device", "cpu"],
                        capture_output=True, text=True, env=env)
    vj = runs / "toy_lag" / "val_lag" / "toy.json"
    ok5 = cp.returncode == 0 and vj.exists()
    info = {"rc": cp.returncode, "stderr_tail": cp.stderr[-600:] if cp.returncode else ""}
    if ok5:
        v = json.loads(vj.read_text())
        z = np.load(str(vj.with_suffix("")) + "_members.npz")
        ok5 = v["all_finite"] and set(v["scores"]) == {"M@3", "M@3|nostations", "M@6", "M@6|nostations", "climatology_median", "cams_bilinear"} \
            and all(np.isfinite(z[k]).all() and z[k].shape == (4, 1500) for k in z.files if k.startswith("M@"))
        # the in-training validation of step 6 (lags 1 and 5) and val_lag.py are two calls of the same function on the
        # same pixels and noise: equal scores
        d1 = abs(v["scores"]["M@6"]["1"]["all"]["crps_fair"] - sc["model"]["1"]["all"]["crps_fair"])
        d5 = abs(v["scores"]["M@6"]["5"]["all"]["crps_fair"] - sc["model"]["5"]["all"]["crps_fair"])
        # no monitor: the same members whatever the lag
        same_none = np.array_equal(z["M@6|nostations|lag1|2021-03-08"], z["M@6|nostations|lag5|2021-03-08"])
        differs = not np.array_equal(z["M@6|lag1|2021-03-08"], z["M@6|lag5|2021-03-08"])
        ok5 = ok5 and d1 < 1e-4 and d5 < 1e-4 and same_none and differs
        info.update(n_member_arrays=len(z.files), diff_vs_in_training=[d1, d5], nostations_same_for_all_lags=same_none,
                    lag1_differs_from_lag5=differs)
    check("C5_val_lag_on_the_toy_run", ok5, **info)

    # ---- C6 two ranks (CPU DDP, as the original package's own DDP test): the lag option under DDP
    cp = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "-m", "cdffm_aq_m.train_aq",
                         "--name", "toy_ddp"] + argv[2:], capture_output=True, text=True, cwd=str(STAGE),
                        env=dict(env, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
    ok6, info6 = cp.returncode == 0, {"rc": cp.returncode, "stderr_tail": cp.stderr[-600:] if cp.returncode else ""}
    if ok6:
        rd = [json.loads(l) for l in open(runs / "toy_ddp" / "metrics.jsonl") if l.strip()]
        ok6 = len(rd) == 2 and all(r["ddp_weights_max_abs_diff"] == 0.0 and r["lag"]["info"]["members_finite"]
                                   and r["lag"]["info"]["scores_finite"] and len(r["ddp_last_days_by_rank"]) == 2 for r in rd)
        info6.update(weights_max_abs_diff=[r["ddp_weights_max_abs_diff"] for r in rd], last_days_by_rank=rd[-1]["ddp_last_days_by_rank"],
                     crps=[round(r["model"]["ug"]["crps_fair"], 3) for r in rd])
    check("C6_two_ranks_ddp_with_the_lag_option", ok6, **info6)
    finish("cpu")


def faithful(store_name, dev, variants, n_points, noise, years):
    """test (ii): one fixed batch (BatchStream seed 0, rank 0, step 0), the same initial weights, the same generator seed:
    loss and gradient norm of the ORIGINAL package and of the copy with the lag option off. variants: {tag: assimilation
    root or None}. One store per package (the original's first, freed before the copy's is opened)."""
    import importlib
    out = {tag: {"stations": bool(root)} for tag, root in variants.items()}
    sd0, b0 = {}, {}
    for which, pkg in (("orig", "cdffm_aq"), ("copy", "cdffm_aq_m")):
        D, Mo, T, St = (importlib.import_module(f"{pkg}.{m}") for m in ("data_aq", "model_aq", "train_aq", "stations"))
        store = pix = None
        for tag, root in variants.items():
            a = SimpleNamespace(store=store_name, fires="injh", split="G" if store_name == "ghap" else None, sigma_k=0, grid="025",
                                transport="t6", objective="fm", lifting_dim=None, use_checkpoint=False, doy_cond=True, seed=0,
                                batch_size=2, n_points=n_points, fm_substeps=8, fm_diag_prob=0.25, stations=root)
            if which == "copy":
                a.stn_lag, a.stn_lag0_prob = None, 0.0                      # the option, explicitly off
            if store is None:
                store = D.open_store(store_name, **D.store_kwargs(a))
                pix = D.PixelArrays(store, dev)
            torch.manual_seed(0)
            model = Mo.build_model(a, D.in_channels(a.transport), store.static_dim, T.model_grid(store.name, a.grid)).to(dev)
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if which == "orig":
                sd0[tag] = sd
            else:
                out[tag]["same_init"] = list(sd) == list(sd0[tag]) and all(torch.equal(sd[k], sd0[tag][k]) for k in sd)
                model.load_state_dict(sd0[tag])
            stream = D.BatchStream(store, a.batch_size, a.n_points, a.seed, 0, dev, start_step=0)
            b = stream.next()
            stream.close()
            bc = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else {kk: vv.detach().cpu().clone() for kk, vv in v.items()})
                  for k, v in b.items()}
            if which == "orig":
                b0[tag] = bc
            else:
                out[tag]["same_batch"] = all(torch.equal(b0[tag][k], bc[k]) for k in ("k", "v", "day", "doy")) and \
                    all(torch.equal(b0[tag]["raw"][k], bc["raw"][k]) for k in bc["raw"])
            stn = None
            if root:
                stn = St.open_stations(a, list(years), dev) if which == "orig" else St.open_stations(a, list(years), dev, store=store)
                assert type(stn).__name__ == "StationSet"                   # option off: the A3 loader
            gen = torch.Generator(device=dev)
            gen.manual_seed(20261120)
            dates = [store.train_dates[int(t)] for t in b["day"]]
            loss = T.compute_losses(model, pix, b, a, Mo.NoiseSpec(*noise), gen, stations=stn, dates=dates, step=7)
            loss.backward()
            gn = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            out[tag][f"loss_{which}"], out[tag][f"grad_norm_{which}"] = float(loss.double()), gn
            out[tag]["days"] = [str(x) for x in dates]
            del model, stream, b, stn, loss
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del store, pix
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    for tag in variants:
        out[tag]["abs_diff"] = abs(out[tag]["loss_orig"] - out[tag]["loss_copy"])
    return out


def faithful_checks(prefix, res, gtol):
    for tag, r in res.items():
        check(f"{prefix}_loss_equals_original_{tag}", r["abs_diff"] <= 1e-5 and r["same_batch"] and r["same_init"]
              and abs(r["grad_norm_orig"] - r["grad_norm_copy"]) <= gtol * max(1.0, r["grad_norm_orig"]), **r)


# =================================================================================================================
# real: the assimilation files of 2017-2021 (assimilated monitors only), CPU
# =================================================================================================================

def sub_real():
    import cdffm_aq_m.stations as cst
    from cdffm_aq_m.data_aq import SIGMA_FLOOR
    dev = torch.device("cpu")
    G = ROOT / "data" / "processed" / "ghap"
    clim = np.load(G / "ghap_clim" / "clim_mean_G.npy", mmap_mode="r")
    mask = np.load(G / "grid" / "ghap_mask.npy", mmap_mode="r")
    years = (2017, 2018, 2019, 2020, 2021)
    L = cst.LagStationSet(cst.ASSIM_ROOT, years=years, device=dev, clim_mean=clim)
    held = set(json.loads((cst.ASSIM_ROOT / "split_v1_public.json").read_text())["heldout_ids"])
    check("R0_public_split_loaded", len(held) == 2049 and L.heldout_ids == held and L.n_total > 10000, n_heldout=len(held),
          n_monitor_years=L.n_total, split_sha256=L.split_sha256)
    raw, worst = {}, 0.0
    for y in years:
        z = np.load(cst.ASSIM_ROOT / f"assim_{y}.npz")
        raw[y] = dict(ids=z["id"].astype(str), daily=np.asarray(z["daily"], np.float32), pos=np.asarray(z["pos"], np.int64),
                      lat=np.asarray(z["lat"], np.float64), lon=np.asarray(z["lon"], np.float64))
        p = raw[y]["pos"]
        ok = p >= 0
        o = np.argsort(p[ok])
        mid = np.empty(int(ok.sum()), np.int64)
        mid[o] = np.asarray(mask[p[ok][o]], np.int64)
        plat, plon = 83.995 - 0.01 * (mid // 36000), -179.995 + 0.01 * (mid % 36000)
        worst = max(worst, float(np.abs(plat - raw[y]["lat"][ok]).max()), float(np.abs(plon - raw[y]["lon"][ok]).max()))
        raw[y]["clim"] = np.zeros((p.size, 5), np.float32)
        cr = np.empty((int(ok.sum()), 5), np.float32)
        cr[o] = np.asarray(clim[p[ok][o]], np.float32)
        raw[y]["clim"][ok] = cr
    check("R1_pos_is_the_monitors_own_pixel", worst <= 0.005 + 1e-4, max_abs_offset_deg=worst)
    g = np.random.default_rng(0)
    n_draw = n_val = n_held = n_wrong = n_same_as_t = n_cmp_t = 0
    anom_err, lags_used, ids_all = 0.0, set(), set()
    d0, d1 = dt.date(2017, 1, 1), dt.date(2021, 12, 31)
    special = [dt.date(2017, 1, 2), dt.date(2017, 1, 5), dt.date(2018, 1, 1), dt.date(2020, 3, 1), dt.date(2021, 1, 1), dt.date(2021, 1, 8)]
    draws = [(d, lag, None) for d in special for lag in range(6)] + \
            [(d0 + dt.timedelta(int(g.integers((d1 - d0).days + 1))), int(g.integers(6)), float(g.random()) if g.random() < 0.5 else None)
             for _ in range(600)]
    for d, lag, hide in draws:
        gen = torch.Generator().manual_seed(int(g.integers(2 ** 31)))
        mon = L.lagged(d, lag, hide=hide, gen=gen)
        n_draw += 1
        lags_used.add(lag)
        src = d - dt.timedelta(lag)
        if src.year < 2017:
            n_wrong += int(mon["val"].numel() != 0)
            continue
        idx = mon["idx"].numpy()
        ids = raw[src.year]["ids"][idx].tolist()
        assert ids == L.ids_of(mon)
        ids_all.update(ids)
        n_held += len(held.intersection(ids))
        col = (src - dt.date(src.year, 1, 1)).days
        v = mon["val"].numpy()
        n_val += v.size
        n_wrong += int((v != raw[src.year]["daily"][idx, col]).sum()) + int(not np.isfinite(v).all())
        if hide is None:                                    # without dropout: every valid monitor of day t - lag
            n_wrong += int(v.size != int(np.isfinite(raw[src.year]["daily"][:, col]).sum()))
        if lag >= 1 and d.year == src.year:                 # is anything served equal to the value of the valid day?
            vt = raw[d.year]["daily"][idx, (d - dt.date(d.year, 1, 1)).days]
            both = np.isfinite(vt)
            n_cmp_t += int(both.sum())
            n_same_as_t += int((v[both] == vt[both]).sum())
        ang = 2.0 * math.pi * col / 365.0
        basis = np.float32([1.0, math.cos(ang), math.sin(ang), math.cos(2 * ang), math.sin(2 * ang)])
        mu = (raw[src.year]["clim"][idx] * basis).sum(-1)
        inm = raw[src.year]["pos"][idx] >= 0
        want = np.log1p(np.maximum(v, 1.0)) - mu
        if inm.any():
            anom_err = max(anom_err, float(np.abs(mon["anom"].numpy()[inm] - want[inm]).max()))
        n_wrong += int((mon["anom_ok"].numpy() != inm).sum())
    check("R2_no_heldout_id_over_random_draws", n_held == 0 and not held.intersection(ids_all) and len(ids_all) > 3000,
          draws=n_draw, distinct_ids_served=len(ids_all), heldout_ids_served=n_held)
    check("R2_values_are_column_t_minus_lag", n_wrong == 0 and n_val > 100000 and lags_used == set(range(6)), values=n_val, wrong=n_wrong)
    # persistence makes a monitor repeat a value from one day to another only by coincidence of the float32 means
    check("R2_lag_ge_1_values_are_not_those_of_day_t", n_cmp_t > 100000 and n_same_as_t / n_cmp_t < 0.02, compared=n_cmp_t,
          equal_to_day_t=n_same_as_t, share=n_same_as_t / max(n_cmp_t, 1),
          note="equal float32 daily means on two different days: coincidences only (the hard check is R2_values_are_column_t_minus_lag)")
    check("R3_anomaly_is_value_minus_pixel_climatology", anom_err < 2e-5, max_abs_err=anom_err, sigma_floor_unused=SIGMA_FLOOR)
    # the anomaly against the package's own function on the monitors' rows (PixelArrays.mu_sigma) for one day
    from cdffm_aq_m.data_aq import PixelArrays
    d, lag = dt.date(2021, 7, 14), 3
    mon = L.lagged(d, lag)
    src = d - dt.timedelta(lag)
    idx = mon["idx"].numpy()
    p = raw[src.year]["pos"][idx]
    inm = p >= 0
    fake = SimpleNamespace(lat=np.zeros(int(inm.sum())), lon=np.zeros(int(inm.sum())), clim_mean=raw[src.year]["clim"][idx][inm],
                           clim_var=np.ones((int(inm.sum()), 5), np.float32), static=np.zeros((int(inm.sum()), 6), np.float16),
                           sigma_floor=0.1, sigma_k=0, norm=SimpleNamespace(t6_mean=np.zeros(6, np.float32), t6_std=np.ones(6, np.float32)))
    px = PixelArrays(fake, dev)
    mu_pkg = px.mu_sigma(torch.arange(int(inm.sum()))[None], torch.tensor([(src - dt.date(src.year, 1, 1)).days]))[0][0]
    e2 = float((mon["anom"][torch.from_numpy(inm)] - (torch.log1p(mon["val"][torch.from_numpy(inm)].clamp(min=1.0)) - mu_pkg)).abs().max())
    f63, dmin = cst.lag_features(mon["lat"][:2000], mon["lon"][:2000], mon)
    check("R3_anomaly_matches_PixelArrays_mu_sigma", e2 < 1e-5 and bool(torch.isfinite(f63).all()) and float(dmin.max()) < 0.05,
          max_abs_err=e2, n=int(inm.sum()), anomaly_mean=float(mon["anom"][torch.from_numpy(inm)].mean()),
          anomaly_std=float(mon["anom"][torch.from_numpy(inm)].std()))
    finish("real")


# =================================================================================================================
# ghap: test (ii) with the real data, GPU
# =================================================================================================================

def sub_ghap():
    dev = torch.device(f"cuda:{int(os.environ.get('SLURM_LOCALID', 0))}") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    assim = str(ROOT / "data" / "processed" / "assim")
    res = faithful("ghap", dev, {"stations": assim, "no_stations": None}, n_points=16384, noise=(1024, 2.175, 4.3, 500.0),
                   years=(2017, 2018, 2019, 2020, 2021))
    for r in res.values():
        r["device"] = str(dev)
    faithful_checks("G_ghap", res, 1e-4)
    finish("ghap")


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    {"cpu": sub_cpu, "real": sub_real, "ghap": sub_ghap}[sub]()
