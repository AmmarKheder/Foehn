#!/usr/bin/env python3
"""PROTOCOL.md A2(d), GHAP 2022 single pass, step (b): the SCORING process (world venv, numpy only; no model code, no
GhapStore). The only process that opens GHAP 2022; every opening is logged in logs/UNSEALING.md.

  truth   GHAP V1 2022 (eccv/data/world/global1km/ghap1km_global.npy rows 1826..2190, read whole-day per worker) at the
          fixed subsample (every 8th GHAP mask pixel) and at the 2022 station pixels (test2022_ghap/st_pix_2022.npz)
          -> data/processed/ghap/sealed/ghap_sub_2022.npy (365, N_SUB) float16 and ghap_st_2022.npy (365, N_ST): the
          native GHAP value (no rounding, no floor; NaN = hole: GHAP <= 0 or not finite). Check first: the same
          extraction on 2 days of 2021 equals the stored hold-out field full_2021[::8] (as max(round(v), 1), -1 = hole).
  score   runs_global/score_2024.py's scoring (fair CRPS, ensemble-mean RMSE and bias, 1 deg block RMSE, Fortin
          spread/skill; cos(lat) weights; week-block bootstrap, 5,000 draws) on test2022_ghap/preds, with hole
          pixel-days given zero weight, GHAP geometry (1 deg blocks 60 S .. 84 N) and the region North (> 60 N) added
          to the six A1(f) regions. -> test2022_ghap/results/scores_ghap_2022.{json,md} + accum_2022.npz
  split   A2(e): preds_st (monitor + ring pixels) -> preds_st_mon/<arm> and preds_ring{1,3,10}/<arm> (ring means), incl.
          GHAP itself as a reference arm (the target product, not a model)
  pinning A2(e)-3: GHAP 2022 vs the 2022 monitors at the monitor pixel and on the rings -> results/pinning_2022.{json,md}
  python -I test2022_ghap/score_2022.py truth|score|split|pinning [--procs 32]"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT / "runs_global"))
import score_2024 as S  # noqa: E402

GH = ROOT / "data/processed/ghap"
O = ROOT / "test2022_ghap"
SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/ghap1km_global.npy")
TRUTH = GH / "sealed/ghap_sub_2022.npy"
TRUTH_ST = GH / "sealed/ghap_st_2022.npy"
NR, NC, D0 = 14400, 36000, dt.date(2017, 1, 1)
Y0, ND = dt.date(2022, 1, 1), 365
REGIONS = S.REGIONS + [("North", 60.0, 90.0, -180.0, 180.0)]
NB = 144 * 360
G_ = {}

S.REGIONS, S.RNAMES, S.NB = REGIONS, ["global"] + [r[0] for r in REGIONS], NB
S.OUT, S.RES, S.TRUTH, S.Y0, S.ND = O, O / "results", TRUTH, Y0, ND


def geometry():
    mask = np.load(GH / "grid/ghap_mask.npy")
    flat = mask[::S.SUB].astype(np.int64)
    r, c = flat // NC, flat % NC
    lat, lon = 83.995 - 0.01 * r, -179.995 + 0.01 * c
    reg = np.full(flat.size, -1, np.int64)
    for k, (_, la0, la1, lo0, lo1) in enumerate(REGIONS):
        m = (reg < 0) & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        reg[m] = k
    blk = (np.floor(lat).astype(np.int64) + 60) * 360 + np.floor(lon).astype(np.int64) + 180
    assert blk.min() >= 0 and blk.max() < NB
    return dict(flat=flat, r=r, c=c, lat=lat, lon=lon, w=np.cos(np.deg2rad(lat)), reg=reg, blk=blk)


S.geometry = geometry
_log0 = S.log_unseal
S.log_unseal = lambda text: _log0(text.replace("final2024/preds", "[ghap] test2022_ghap/preds").replace(
    "final2024/results/scores_lghap_2024.{json,md}", "[ghap] test2022_ghap/results (renamed scores_ghap_2022)"))


def _score_days(ts):
    """score_2024._score_days with hole pixel-days (NaN truth) at zero weight"""
    geo = S.G_["geo"]
    w0, reg1, blk = geo["w"], geo["reg"] + 1, geo["blk"]
    breg1 = S.G_["breg1"]
    n = w0.size
    acc = {k: np.zeros(n, np.float64) for k in ("mean_cdffm", "se_cdffm", "se_cams_bilinear", "y", "mean_cams_bilinear", "n_ok")}
    out = {}
    for t in ts:
        d = Y0 + dt.timedelta(t)
        y = S.read_row(TRUTH, t).astype(np.float32)
        okd = np.isfinite(y)
        y = np.where(okd, y, 0.0).astype(np.float32)
        w = np.where(okd, w0, 0.0)
        D = np.zeros((len(S.ARMS), len(S.RNAMES), len(S.STATS)))
        Yb = np.bincount(blk, w * y, NB)
        Wb = np.bincount(blk, w, NB)
        okb = Wb > 0
        for ai, arm in enumerate(S.ARMS):
            X = np.load(O / "preds" / arm / f"{d}.npz")["members"].astype(np.float32)
            assert X.shape[1] == n, (arm, d, X.shape)
            M = X.shape[0]
            m = X.mean(0)
            crps = S.fair_crps(X, y)
            e = m - y
            var = X.var(0) if M > 1 else np.zeros(n, np.float32)
            Pb = np.bincount(blk, w * m, NB)
            bse = np.where(okb, Wb * (Pb / np.where(okb, Wb, 1) - Yb / np.where(okb, Wb, 1)) ** 2, 0.0)
            for si, v in enumerate((w, w * crps, w * e * e, w * e, None, None, w * var)):
                if v is not None:
                    s = np.bincount(reg1, v, len(S.RNAMES))
                    D[ai, 1:, si] = s[1:]
                    D[ai, 0, si] = v.sum(dtype=np.float64)
            for si, v in ((4, Wb), (5, bse)):
                s = np.bincount(breg1, v, len(S.RNAMES))
                D[ai, 1:, si] = s[1:]
                D[ai, 0, si] = v.sum()
            if arm in ("cdffm", "cams_bilinear"):
                acc[f"se_{arm}"] += okd * e.astype(np.float64) ** 2
                acc[f"mean_{arm}"] += m
        acc["y"] += y
        acc["n_ok"] += okd
        out[t] = (D, {"max_cdffm_member": float(np.load(O / "preds" / "cdffm" / f"{d}.npz")["members"].max()),
                      "n_holes": int((~okd).sum())})
    return out, acc


S._score_days = _score_days


# ------------------------------------------------------------------------------------------------------------- truth
def _hdr():
    with open(SRC, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, _, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        assert tuple(shape) == (2191, NR, NC) and np.dtype(dtype) == np.float16
        return f.tell()


def _day(d):
    hdr = _hdr()
    a = np.empty(NR * NC, np.float16)
    mv = memoryview(a.view(np.uint8))
    fd = os.open(SRC, os.O_RDONLY)
    off = hdr + (d - D0).days * NR * NC * 2
    pos = 0
    while pos < len(mv):
        k = os.preadv(fd, [mv[pos:]], off + pos)
        assert k > 0
        pos += k
    os.close(fd)
    return a


def _truth_day(t):
    d = Y0 + dt.timedelta(t)
    a = _day(d)
    v = a[G_["flat"]].astype(np.float32)
    vs = a[G_["stflat"]].astype(np.float32)
    del a
    for x in (v, vs):
        x[~(np.isfinite(x) & (x > 0))] = np.nan
    _write(TRUTH, t, v.astype(np.float16))
    _write(TRUTH_ST, t, vs.astype(np.float16))
    return t, int(np.isnan(v).sum()), float(np.nanmax(v)), float(np.nanmean(v))


def _create(path, shape, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
        hdr = f.tell()
        f.truncate(hdr + int(np.prod(shape)) * np.dtype(dtype).itemsize)


def _write(path, i, row):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        hdr = f.tell()
    b = np.ascontiguousarray(row, dtype=dtype).view(np.uint8)
    fd = os.open(path, os.O_WRONLY)
    os.pwrite(fd, b.tobytes(), hdr + i * b.size)
    os.close(fd)


def cmd_truth(a):
    geo = geometry()
    mask = np.load(GH / "grid/ghap_mask.npy")
    st = np.load(O / "st_ring_2022.npz")["pos"].astype(np.int64)          # monitor + ring pixels (A2(e))
    G_["flat"], G_["stflat"] = geo["flat"], mask[st].astype(np.int64)
    n = geo["flat"].size
    sysj = json.loads((O / "SYSTEM.json").read_text())
    arms = [x for x in S.ARMS]
    miss = [(arm, t) for arm in arms for t in range(ND) if not (O / "preds" / arm / f"{Y0 + dt.timedelta(t)}.npz").exists()]
    assert not miss, f"predictions incomplete ({len(miss)} files missing, e.g. {miss[:3]}): the truth is not opened"
    chk21 = {}
    hold = GH / "ghap_eval/full_2021.npy"
    for d in (dt.date(2021, 1, 8), dt.date(2021, 7, 1)):
        a_ = _day(d)
        v = a_[geo["flat"]].astype(np.float32)
        ref = S.read_row(hold, (d - dt.date(2021, 1, 1)).days)[::S.SUB]
        mine = np.where(np.isfinite(v) & (v > 0), np.maximum(np.rint(v), 1), -1).astype(np.int16)
        chk21[str(d)] = bool(np.array_equal(mine, ref))
    print("2021 extraction check:", chk21, flush=True)
    assert all(chk21.values()), chk21
    S.log_unseal(f"[ghap] OPENING GHAP V1 2022 for the A2(d) single pass (user order 2026-10-28 ~13:40). File: "
                 f"{SRC} rows 1826..2190 (365 days, whole-day reads). Outputs: {TRUTH.relative_to(ROOT)} ({ND} x {n:,} float16, "
                 f"every 8th GHAP mask pixel, native values, NaN = hole) and {TRUTH_ST.relative_to(ROOT)} ({st.size} station "
                 f"pixels). System scored: {sysj['system']} ({sysj['system_dir']}), frozen {sysj.get('frozen_utc')}; "
                 f"predictions test2022_ghap/preds written before this opening under the GhapStore firewall.")
    _create(TRUTH, (ND, n), np.float16)
    _create(TRUTH_ST, (ND, st.size), np.float16)
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = pool.map(_truth_day, range(ND), chunksize=1)
    info = dict(file=str(TRUTH), file_st=str(TRUTH_ST), shape=[ND, n], seconds=time.time() - t0, check_2021=chk21,
                holes_per_day=[r[1] for r in rows], max_per_day=[r[2] for r in rows], mean_per_day=[r[3] for r in rows])
    (O / "results").mkdir(parents=True, exist_ok=True)
    (O / "results/truth_2022.json").write_text(json.dumps(info, indent=1))
    S.log_unseal(f"[ghap] GHAP 2022 truth extracted ({time.time() - t0:.0f} s; hole pixel-days {sum(r[1] for r in rows):,}); "
                 f"check 2021 extraction = full_2021[::8] {chk21}")
    print(f"[truth] done {time.time() - t0:.0f} s", flush=True)


def cmd_score(a):
    S.ARMS = tuple(x for x in S.ARMS if (O / "preds" / x).exists())
    S.cmd_score(a)
    # rename the outputs and fix the LGHAP wording of score_2024's writer
    res_dir = O / "results"
    for ext in ("json", "md"):
        p = res_dir / f"scores_lghap_2024.{ext}"
        txt = p.read_text()
        txt = (txt.replace("CD-FFM 1 km vs baselines on LGHAP v2, 2024 (single unsealing pass, PROTOCOL.md A1(h))",
                           "CD-FFM 1 km (GHAP system pmG_final) vs baselines on GHAP V1, 2022 (single pass, PROTOCOL.md A2(d))")
               .replace("max(LGHAP v2, 1) ug m-3 at every 8th mask pixel (14,288,247 pixels), 366 days",
                        "native GHAP V1 2022 ug m-3 at every 8th GHAP mask pixel, 365 days, hole pixel-days at zero weight")
               .replace("PROTOCOL.md A1(h) single 2024 pass", "PROTOCOL.md A2(d) single 2022 pass")
               .replace("week blocks (t // 7 from 2024-01-01)", "week blocks (t // 7 from 2022-01-01)")
               .replace("Largest CD-FFM member at the subsample over 2024", "Largest CD-FFM member at the subsample over 2022"))
        (res_dir / f"scores_ghap_2022.{ext}").write_text(txt)
        p.unlink()
    os.replace(res_dir / "accum_2024.npz", res_dir / "accum_2022.npz")
    S.log_unseal("[ghap] scores written: test2022_ghap/results/scores_ghap_2022.{json,md}")


def ring_means(X, idx, dd):
    """X (M, nU) members at the union positions (NaN = invalid) -> (M, npix) ring means over the valid ring pixels,
    NaN where fewer than 4 d of the 8 d ring pixels are valid (product_audit rule)"""
    ok = idx >= 0
    V = X[:, np.maximum(idx, 0)]                                         # (M, npix, 8 d)
    V = np.where(ok[None], V, np.nan)
    n = np.isfinite(V).sum(2)
    m = np.nansum(V, 2) / np.maximum(n, 1)
    return np.where(n >= 4 * dd, m, np.nan).astype(np.float32)


def _split_day(t):
    d = Y0 + dt.timedelta(t)
    R = G_["ring"]
    arms = {arm: np.load(O / "preds_st" / arm / f"{d}.npz")["members"].astype(np.float32) for arm in G_["arms"]}
    if G_["ghap"] is not None:
        arms["ghap"] = np.asarray(G_["ghap"][t], np.float32)[None]
    for arm, X in arms.items():
        meta = np.array(json.dumps({"date": str(d), "arm": arm, "members": int(X.shape[0])}))
        p = O / "preds_st_mon" / arm / f"{d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, members=X[:, R["mon_idx"]].astype(np.float16), meta=meta)
        for dd in R["rings"]:
            p = O / f"preds_ring{dd}" / arm / f"{d}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, members=ring_means(X, R[f"ring{dd}"], int(dd)).astype(np.float16), meta=meta)
    return t


def cmd_split(a):
    """A2(e): preds_st (monitor + ring pixels) -> preds_st_mon (monitor pixels) and preds_ring<d> (ring means), for every
    arm and for GHAP itself (from the truth step's station-position extraction; reference product, not a model)"""
    z = np.load(O / "st_ring_2022.npz")
    G_["ring"] = {k: z[k] for k in z.files}
    G_["arms"] = [p.name for p in sorted((O / "preds_st").iterdir()) if p.is_dir()]
    G_["ghap"] = np.load(TRUTH_ST, mmap_mode="r") if TRUTH_ST.exists() else None
    with Pool(min(a.procs, 16)) as pool:
        pool.map(_split_day, range(ND), chunksize=4)
    print(f"[split] arms {G_['arms']} + ghap={G_['ghap'] is not None} -> preds_st_mon, preds_ring{list(z['rings'])}", flush=True)


def cmd_pinning(a):
    """A2(e)-3: how pinned GHAP 2022 is to the 2022 monitors (inside the logged scoring step): per monitor, the share of
    its days with |GHAP - monitor| <= 0.5 ug m-3 and its daily R, at the monitor pixel and on the 1/3/10 km rings"""
    tr = np.load(O / "st_truth_2022.npz")
    obs = tr["daily"].astype(np.float64)                                 # (N monitors, 365)
    pix = tr["pix"]
    out = {"rule": "per monitor (monitors with >= 30 valid days in both): share of monitor-days with |GHAP - monitor UTC "
                   "daily mean| <= 0.5 ug m-3; daily Pearson R; GHAP at the monitor pixel and ring means (A2(e))"}
    for name in ["mon"] + [f"ring{dd}" for dd in (1, 3, 10)]:
        g = np.stack([np.load(O / ("preds_st_mon" if name == "mon" else f"preds_{name}") / "ghap" / f"{Y0 + dt.timedelta(t)}.npz")
                      ["members"][0].astype(np.float64) for t in range(ND)], 1)[pix]       # (N, 365)
        ok = np.isfinite(g) & np.isfinite(obs)
        n = ok.sum(1)
        good = n >= 30
        within = np.where(ok, np.abs(g - obs) <= 0.5, False).sum(1) / np.maximum(n, 1)
        r = np.full(len(n), np.nan)
        for i in np.where(good)[0]:
            r[i] = np.corrcoef(g[i, ok[i]], obs[i, ok[i]])[0, 1]
        nets = {}
        for net in ["All"] + sorted(set(tr["network"].tolist())):
            m = good & ((tr["network"] == net) if net != "All" else True)
            nets[net] = {"n_monitors": int(m.sum()),
                         "share_monitor_days_within_0p5": float((within[m] * n[m]).sum() / max(n[m].sum(), 1)),
                         "n_monitors_R_gt_0p999": int((r[m] > 0.999).sum()),
                         "share_monitors_R_gt_0p999": float((r[m] > 0.999).mean()) if m.any() else None,
                         "median_R": float(np.nanmedian(r[m])) if m.any() else None}
        out[name] = nets
    (O / "results").mkdir(parents=True, exist_ok=True)
    (O / "results/pinning_2022.json").write_text(json.dumps(out, indent=1))
    L = ["# GHAP V1 2022 pinning to the 2022 monitors (PROTOCOL.md A2(e), computed in the logged scoring step)", "",
         out["rule"], "", "| where | network | monitors | monitor-days within 0.5 µg m⁻³ | monitors with R > 0.999 | median R |",
         "|---|---|---|---|---|---|"]
    for name in ["mon", "ring1", "ring3", "ring10"]:
        for net, v in out[name].items():
            if v["n_monitors"]:
                L.append(f"| {'monitor pixel' if name == 'mon' else name.replace('ring', 'ring ') + ' km'} | {net} | "
                         f"{v['n_monitors']} | {v['share_monitor_days_within_0p5']:.1%} | {v['n_monitors_R_gt_0p999']} "
                         f"({v['share_monitors_R_gt_0p999']:.1%}) | {v['median_R']:.3f} |")
    (O / "results/pinning_2022.md").write_text("\n".join(L) + "\n")
    S.log_unseal("[ghap] GHAP 2022 pinning to the 2022 monitors computed -> test2022_ghap/results/pinning_2022.{json,md}")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["truth", "score", "split", "pinning"])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--test-dir", default=None, help="code test only: score DIR/preds against DIR/truth.npy (fake), "
                    "first --ndays days of 2022, log in DIR (never the GHAP 2022 truth)")
    ap.add_argument("--ndays", type=int, default=365)
    a = ap.parse_args()
    if a.test_dir:
        assert a.cmd == "score"
        O = Path(a.test_dir)
        TRUTH = O / "truth.npy"
        ND = a.ndays
        S.OUT, S.RES, S.TRUTH, S.ND, S.UNSEAL = O, O / "results", TRUTH, ND, O / "UNSEALING_test.md"
    {"truth": cmd_truth, "score": cmd_score, "split": cmd_split, "pinning": cmd_pinning}[a.cmd](a)
