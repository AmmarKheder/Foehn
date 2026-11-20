#!/usr/bin/env python3
"""z cell-mean table of the global store for the hcf variant (cdffm_aq_x), = data_x.build_cellmeans on the global store,
parallel over days (fork pool; the store and its per-pixel arrays are shared copy-on-write).

  training days (1,270, 2019-07-11..2022-12-31): mean of z over the stored 4 M points of the day falling in each
      0.25 deg cell (npts = their number; cells without a point that day: mean of the finite 3 x 3 neighbours, as
      data_x._fill_missing);
  validation days (358, 2023-01-08..12-31): exact mean of z over all mask pixels of the cell (full_2023.npy rows).
z = (log1p(max(LGHAP, 1)) - mu_p(doy)) / sigma_p(doy) (no dequantisation), computed with the same torch functions as
build_cellmeans (PixelArrays.mu_sigma, target_z), summed in float64. Cells = CellGeometry(store, 0.25 deg grid).cells.
Only GlobalStore.points (training days) and GlobalStore.holdout (2023 validation days) are read; the firewall audit hook
of GlobalStore is on (no sealed file can be opened).

  python build_global_cellmeans.py build [--procs 32]      -> data/processed/global/lghap_clim/cellmean_z_A.npz
  python build_global_cellmeans.py check                   -> direct float64 numpy recompute on a few cells / days
  --sigma-k 0 --out .../cellmean_z_A_K0.npz                -> the table under the annual sigma (PROTOCOL.md A1(e), A1(f))
Container: source env.sh; cdffm_run python -u scripts/build_global_cellmeans.py ...  (CPU, partition small)."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/processed/global/lghap_clim/cellmean_z_A.npz"
CHUNK = 2_000_000
G = {}


def setup():
    import torch
    from cdffm_aq.data_aq import PixelArrays
    from cdffm_aq.model_aq import GRIDS
    from cdffm_aq.store_global import GlobalStore
    from cdffm_aq_x.data_x import CellGeometry
    st = GlobalStore(fires="mami", sigma_k=G.get("sigma_k", 2))   # fires are not read here; mami avoids waiting for injh
    geom = CellGeometry(st, GRIDS["025"])
    pix = PixelArrays(st, torch.device("cpu"))
    cp_all = torch.from_numpy(geom.cell_pos[geom.cell_id])
    assert int(cp_all.min()) >= 0
    G.update(st=st, geom=geom, pix=pix, cp_all=cp_all)
    return st, geom


def means_of(k, v, doy):
    """as data_x.build_cellmeans.means_of (same torch formulas), accumulated chunk by chunk: float64 sums and counts
    per cell. k: (n,) int64 tensor of pixel indices, or (start, end) for the contiguous range start..end-1"""
    import torch
    from cdffm_aq.data_aq import target_z
    pix, cp_all, nc = G["pix"], G["cp_all"], G["geom"].n_cells
    n_tot = k[1] - k[0] if isinstance(k, tuple) else k.shape[0]
    s = torch.zeros(nc, dtype=torch.float64)
    n = torch.zeros(nc, dtype=torch.float64)
    for a in range(0, n_tot, CHUNK):
        e = min(n_tot, a + CHUNK)
        kk = torch.arange(k[0] + a, k[0] + e) if isinstance(k, tuple) else k[a:e]
        mu, sig = pix.mu_sigma(kk[None], torch.tensor([doy]))
        z = target_z(v[a:e][None], mu, sig)[0].double()
        cp = cp_all[kk]
        s.index_add_(0, cp, z)
        n.index_add_(0, cp, torch.ones_like(z))
    return s.numpy(), n.numpy()


def job(task):
    import torch
    from cdffm_aq.data_aq import doy_of
    torch.set_num_threads(1)
    kind, t = task
    st = G["st"]
    if kind == "train":
        idx, val = st.points(t)
        s, n = means_of(torch.from_numpy(idx.astype(np.int64)), torch.from_numpy(val), doy_of(st.train_dates[t]))
    else:
        s, n = means_of((0, st.n_pix), torch.from_numpy(st.holdout(t)), doy_of(st.hold_dates[t]))
    return kind, t, s, n


def cmd_build(a):
    from cdffm_aq_x.data_x import _fill_missing
    t0 = time.time()
    st, geom = setup()
    H, W, nc = geom.H, geom.W, geom.n_cells
    print(f"store ready: {st.n_pix:,} pixels, {nc:,} cells, {len(st.train_dates)} training + {len(st.hold_dates)} "
          f"validation days ({time.time() - t0:.0f} s)", flush=True)
    train = np.zeros((len(st.train_dates), nc), np.float32)
    npts = np.zeros((len(st.train_dates), nc), np.int32)
    hold = np.zeros((len(st.hold_dates), nc), np.float32)
    tasks = [("hold", t) for t in range(len(st.hold_dates))] + [("train", t) for t in range(len(st.train_dates))]
    done = 0
    with mp.get_context("fork").Pool(a.procs) as pool:
        for kind, t, s, n in pool.imap_unordered(job, tasks, chunksize=2):
            if kind == "train":
                g = np.full(H * W, np.nan)
                g[geom.cells] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
                g = _fill_missing(g.reshape(H, W), geom.mask_grid).ravel()
                train[t] = g[geom.cells]
                npts[t] = n
            else:
                assert n.min() > 0
                hold[t] = s / np.maximum(n, 1)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} days ({time.time() - t0:.0f} s)", flush=True)
    assert np.isfinite(train).all() and np.isfinite(hold).all()
    assert (npts.sum(1) == st.n_points).all(), "points per training day"
    tmp = OUT.with_name(OUT.stem + ".tmp.npz")
    np.savez(tmp, cells=geom.cells, train=train, hold=hold, npts=npts, H=H, W=W,
             train_dates=np.array([str(d) for d in st.train_dates]), hold_dates=np.array([str(d) for d in st.hold_dates]))
    os.replace(tmp, OUT)
    info = {"file": str(OUT), "n_cells": int(nc), "train_days": int(train.shape[0]), "hold_days": int(hold.shape[0]),
            "train_first_last": [str(st.train_dates[0]), str(st.train_dates[-1])],
            "hold_first_last": [str(st.hold_dates[0]), str(st.hold_dates[-1])],
            "npts_per_cell_day_median": float(np.median(npts)), "npts_zero_share": float((npts == 0).mean()),
            "train_cellmean_mean_std": [float(train.mean()), float(train.std())],
            "hold_cellmean_mean_std": [float(hold.mean()), float(hold.std())], "seconds": time.time() - t0,
            "procs": a.procs}
    info["sigma_k"] = G.get("sigma_k", 2)
    OUT.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def cmd_check(a):
    """independent float64 numpy recompute of z cell means on a few cells and days"""
    st, geom = setup()
    tb = np.load(OUT)
    assert np.array_equal(tb["cells"], geom.cells)
    rng = np.random.default_rng(11)
    sigma_floor = st.sigma_floor
    res = {"train": [], "hold": []}

    def z_np(k, v, d):
        doy = (d - d.replace(month=1, day=1)).days
        ang = 2 * np.pi * doy / 365.0
        b = np.array([1, np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], np.float64)
        mu = st.clim_mean[k].astype(np.float64) @ b
        cv = st.clim_var[k].astype(np.float64)
        var = cv @ b if st.sigma_k == 2 else cv[:, 0]            # annual sigma: the constant term (A1(e))
        sig = np.sqrt(np.maximum(var, sigma_floor ** 2))
        return (np.log1p(np.maximum(v.astype(np.float64), 1.0)) - mu) / sig

    cid = geom.cell_id
    for t in rng.choice(len(st.train_dates), 3, replace=False):
        idx, val = st.points(int(t))
        c_of = cid[idx.astype(np.int64)]
        cells = rng.choice(np.unique(c_of), 6, replace=False)
        for c in cells:
            m = c_of == c
            ref = z_np(idx[m].astype(np.int64), val[m], st.train_dates[int(t)]).mean()
            got = float(tb["train"][int(t), geom.cell_pos[c]])
            res["train"].append(dict(day=str(st.train_dates[int(t)]), cell=int(c), n=int(m.sum()), ref=float(ref),
                                     table=got, abs_err=abs(got - ref),
                                     npts_ok=int(tb["npts"][int(t), geom.cell_pos[c]]) == int(m.sum())))
    for t in rng.choice(len(st.hold_dates), 3, replace=False):
        v = st.holdout(int(t))
        cells = rng.choice(geom.cells, 6, replace=False)
        for c in cells:
            k = np.nonzero(cid == c)[0]
            ref = z_np(k, v[k], st.hold_dates[int(t)]).mean()
            got = float(tb["hold"][int(t), geom.cell_pos[c]])
            res["hold"].append(dict(day=str(st.hold_dates[int(t)]), cell=int(c), n=int(k.size), ref=float(ref),
                                    table=got, abs_err=abs(got - ref)))
    errs = [r["abs_err"] for r in res["train"] + res["hold"]]
    res["max_abs_err"] = float(max(errs))
    res["npts_all_ok"] = all(r["npts_ok"] for r in res["train"])
    res["ok"] = res["max_abs_err"] < 1e-4 and res["npts_all_ok"]
    OUT.with_name(OUT.stem + "_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("max_abs_err", "npts_all_ok", "ok")}), flush=True)
    for r in res["train"][:4] + res["hold"][:4]:
        print(r, flush=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--sigma-k", type=int, default=2, choices=(0, 2), help="target sigma: 2 seasonal (K = 2), 0 annual")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    G["sigma_k"] = a.sigma_k
    {"build": cmd_build, "check": cmd_check}[a.cmd](a)
