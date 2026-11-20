#!/usr/bin/env python3
"""Unit test of cdffm_aq.store_ghap.GhapStore (PROTOCOL.md A2) and non-regression of the LGHAP store paths.
Reads GHAP 2017-2021 raw rows to verify the store values (this test process opens the store with firewall=False; a
child process checks that the firewall blocks raw GHAP and station files). Never reads 2022.
  python runs_ghap/test_store_ghap.py   -> runs_ghap/logs/test_store_ghap.json; exit 0 = all checks pass"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
from cdffm_aq.data_aq import open_store, store_kwargs  # noqa: E402
from cdffm_aq.store_ghap import GhapStore, GHAP_NC, GHAP_NR  # noqa: E402
from cdffm_aq.store_global import SEL_HOLD_IDX  # noqa: E402

SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/ghap1km_global.npy")
D0 = dt.date(2017, 1, 1)
RES = {}


def check(name, ok, **info):
    RES[name] = dict(ok=bool(ok), **info)
    print(f"[{'OK' if ok else 'FAIL'}] {name} {info}", flush=True)


def raw_at(d, gid):
    assert d.year <= 2021
    with open(SRC, "rb") as f:
        ver = np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f)
        hdr = f.tell()
    fd = os.open(SRC, os.O_RDONLY)
    out = np.empty(gid.size, np.float16)
    base = hdr + (d - D0).days * GHAP_NR * GHAP_NC * 2
    rows = gid // GHAP_NC
    for r in np.unique(rows):                       # one pread per lattice row
        m = rows == r
        buf = np.frombuffer(os.pread(fd, GHAP_NC * 2, base + int(r) * GHAP_NC * 2), np.float16)
        out[m] = buf[gid[m] % GHAP_NC]
    os.close(fd)
    return out.astype(np.float32)


def main():
    st = GhapStore(firewall=False)
    man = st.manifest
    mask = np.load(st.root / "grid/ghap_mask.npy").astype(np.int64)
    check("shapes", st.n_pix == man["mask"]["Nm"] == mask.size and st.clim_mean.shape == (st.n_pix, 5)
          and st.static.shape == (st.n_pix, 6) and st.hold.shape == (365, st.n_pix), n_pix=st.n_pix)
    exc = man["split_G"]["excluded_days"]
    excl = set(exc["incomplete_ghap_valid_count"]) | set(exc["inputs_not_finite"])
    check("splits", st.train_dates[0] >= dt.date(2017, 1, 1) and st.train_dates[-1] <= dt.date(2020, 12, 31)
          and len(st.train_dates) == 1461 - len(excl) and not ({str(x) for x in st.train_dates} & excl)
          and st.hold_dates[0] == dt.date(2021, 1, 8) and len(st.hold_dates) == 358
          and st.sel_dates == [dt.date(2021, 1, 8) + dt.timedelta(15 * k) for k in range(24)],
          sel_first_last=[str(st.sel_dates[0]), str(st.sel_dates[-1])], n_train=len(st.train_dates), n_excluded=len(excl))
    check("mask_sorted_in_grid", np.all(np.diff(mask) > 0) and mask.max() < GHAP_NR * GHAP_NC
          and (mask // GHAP_NC).max() <= 14386, lat=[float(st.lat.min()), float(st.lat.max())])
    cid = np.load(st.root / "grid/cell_id.npy").astype(np.int64)
    dl = np.abs(84.0 - 0.25 * (cid // 1440) - st.lat).max()
    dlo = np.abs(((-180.0 + 0.25 * (cid % 1440) - st.lon + 180.0) % 360.0) - 180.0).max()
    check("cell_id_nearest_point", dl <= 0.125 + 1e-6 and dlo <= 0.125 + 1e-6, max_dlat=float(dl), max_dlon=float(dlo))
    check("tile_ascending", np.allclose(st.tile_row_lat[st.tile_row], st.lat, atol=1e-6)
          and np.all(np.diff(st.tile_row_lat) > 0) and np.allclose(st.tile_col_lon0 + 0.01 * st.tile_col, st.lon, atol=1e-6))
    ice = np.load(st.root / "grid/ice_pct_lattice.npy", mmap_mode="r")
    ip = np.asarray(ice[mask // GHAP_NC, mask % GHAP_NC])
    check("no_permanent_ice_in_mask", not ((ip >= 50) & (ip <= 100)).any(), n_ice_removed=man["mask"]["ice_exclusion"]["n_removed"])
    check("statics_finite", np.isfinite(st.static.astype(np.float32)).all() and np.isfinite(st.clim_mean).all()
          and np.isfinite(st.clim_var).all(), sigma_floor=st.sigma_floor)
    sig_ann = np.sqrt(np.maximum(st.clim_var[:, 0], st.sigma_floor ** 2))
    check("sigma_floor_positive", st.sigma_floor > 0.01, annual_sigma_p1_p50_p99=np.percentile(sig_ann, [1, 50, 99]).tolist())
    # training points against the raw GHAP field
    pts_ok, info = True, {}
    for t in (0, 200, 700, len(st.train_dates) - 1):
        d = st.train_dates[t]
        idx, val = st.points(t)
        assert idx.dtype == np.uint32 and val.dtype == np.int16 and idx.size == st.n_points == 4_000_000
        srt = bool(np.all(np.diff(idx.astype(np.int64)) > 0)) and int(idx.max()) < st.n_pix
        sub = np.random.default_rng(t).choice(idx.size, 20000, replace=False)
        v = raw_at(d, mask[idx[sub].astype(np.int64)])
        exp = np.maximum(np.rint(v), 1)
        eq = bool(np.all(v > 0)) and np.array_equal(exp.astype(np.int16), val[sub])
        # cos-lat weighting: mean cos(lat) of the points vs the weighted expectation over the mask
        w = np.cos(np.deg2rad(st.lat))
        e_cos = float((w * w).sum() / w.sum())
        m_cos = float(np.cos(np.deg2rad(st.lat[idx.astype(np.int64)])).mean())
        ok = srt and eq and int(val.min()) >= 1 and abs(m_cos - e_cos) < 2e-3
        pts_ok &= ok
        info[str(d)] = dict(sorted_unique=srt, equal_raw=eq, min_val=int(val.min()), max_val=int(val.max()),
                            mean_cos_points=m_cos, expected=e_cos)
    check("training_points_vs_raw", pts_ok, **info)
    hold_ok, info = True, {}
    for t in (0, 165, 357):
        d = st.hold_dates[t]
        h = st.holdout(t)
        sub = np.sort(np.random.default_rng(100 + t).choice(st.n_pix, 50000, replace=False))
        v = raw_at(d, mask[sub])
        exp = np.where(v > 0, np.maximum(np.rint(v), 1), -1).astype(np.int16)
        eq = np.array_equal(exp, h[sub])
        hold_ok &= eq
        info[str(d)] = dict(equal_raw=eq, n_holes_day=int((h < 0).sum()), max=int(h.max()))
    check("holdout_2021_vs_raw", hold_ok, **info)
    raw = st.raw_inputs(dt.date(2021, 3, 1))
    raw17 = st.raw_inputs(dt.date(2017, 1, 1))
    check("raw_inputs", all(np.isfinite(raw[k].astype(np.float32)).all() for k in raw)
          and {k: v.shape for k, v in raw.items()} == {k: tuple(s) for k, s in st.raw_shapes().items()}
          and raw17["cams"].shape == (576, 1440), shapes={k: list(v.shape) for k, v in raw.items()})
    check("G0a_threshold", st.train_max_ug >= 100 and st.pmax_ug >= max(1000.0, st.train_max_ug),
          T_G=st.train_max_ug, P_max=st.pmax_ug)
    # open_store / store_kwargs dispatch (ghap) and LGHAP non-regression
    class A:
        store, fires, split, sigma_k = "ghap", "injh", None, None
    class B:
        store, fires, split, sigma_k = "global", "injh", "A", 0
    kw = store_kwargs(A())
    check("store_kwargs_ghap", kw == {"fires": "injh", "split": "G", "sigma_k": 0}, kw=kw)
    kwb = store_kwargs(B())
    check("store_kwargs_global_unchanged", kwb == {"fires": "injh", "split": "A", "sigma_k": 0}, kw=kwb)
    g = open_store("global", fires="injh", split="A", sigma_k=0, firewall=False)
    check("lghap_store_unchanged", g.n_pix == 114_305_974 and len(g.train_dates) == 1270 and g.hold_dates[0] == dt.date(2023, 1, 8)
          and not getattr(g, "has_holes", False) and not hasattr(g, "train_max_ug"), n_pix=g.n_pix)
    # firewall (child process): raw GHAP, sealed GHAP dir and station files unreadable with GhapStore(firewall=True)
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from cdffm_aq.store_ghap import GhapStore\n"
            "GhapStore()\n"
            "res = []\n"
            "for p in (%r, '/scratch/project_462001804/ammar/eccv/data/world/global1km/ghap1km_tiles.npy',"
            " %r, '/scratch/project_462001804/ammar/eccv/data/world/stations/x.npz'):\n"
            "    try:\n        open(p, 'rb'); res.append('OPEN')\n"
            "    except PermissionError:\n        res.append('BLOCKED')\n"
            "    except FileNotFoundError:\n        res.append('NOTFOUND')\n"
            "print(','.join(res))\n") % (str(ROOT), str(SRC), str(ROOT / "data/processed/ghap/sealed/x.npy"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    check("firewall", out.returncode == 0 and lines and lines[-1] == "BLOCKED,BLOCKED,BLOCKED,BLOCKED",
          result=lines[-1] if lines else out.stderr[-500:])
    ok = all(v["ok"] for v in RES.values())
    (ROOT / "runs_ghap/logs/test_store_ghap.json").write_text(json.dumps({"passed": ok, "checks": RES}, indent=1, default=str))
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
