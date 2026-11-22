#!/usr/bin/env python3
"""A17 diag, step 3 (CPU only): what the isotropic linear law of analyse.py leaves out. Same data (work/), same member folds.
  (a) non-local / anisotropic: 16 shifted box means of each noise field (widths 33, 65, 129, 257 px, centre moved by half
      a width to the north, south, east, west), added to the 12 isotropic features;
  (b) position-dependent kernel: the per-forecast law (static + level-scaled) fitted separately in each of the 9 regions
      of eval_global_green.REGIONS (pixels in no box: 'other'), with the isotropic and with the isotropic + shifted features;
  (c) log space (log x, all members > 0): static share and linear R2 of the log departures, oddness along u there.
Adds the key 'extra' to results.json (atomic write, read back).
    python extra.py [workers]"""
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as an  # noqa: E402
import collect as co  # noqa: E402  (audit hook: no station, monitor, _ar or a16 data path can be opened)

SHW = (33, 65, 129, 257)
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))          # north, south, east, west (rows grow southwards)
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
S = {}


def shifted(m):
    z = co.full_map(co.G / "noise_tiles16.npy", m)
    cs = co.integral(z)
    del z
    r, c = S["r"], S["c"]
    out = np.zeros((len(r), len(SHW) * len(DIRS)), np.float32)
    j = 0
    for w in SHW:
        for dr, dc in DIRS:
            s, n = co.box_sums(cs, r + dr * (w // 2), c + dc * (w // 2), w)
            out[:, j] = s / np.sqrt(np.maximum(n, 1)); j += 1
    return m, out


def acc_log(fc, li):
    _, T, N, M = fc.shape
    A = dict(S1=np.zeros((1, 3, N, M)), S2=np.zeros((1, 3, N, M)), SM=np.zeros((1, 3, N, M)), cnt=np.zeros((1, 3, N)),
             Smb=np.zeros((1, 3, N)), Smb2=np.zeros((1, 3, N)), bad=np.zeros(N, bool), T=T)
    for t in range(T):
        x = np.nan_to_num(np.asarray(fc[li, t], np.float32)).astype(np.float64)
        mb = x.mean(1)
        reg = np.digitize(mb, an.REG)
        pos = (x > 0).all(1)
        A["bad"] |= ~pos
        lx = np.log(np.where(pos[:, None], x, 1.0))
        D = lx - lx.mean(1, keepdims=True)
        for r in range(3):
            w = ((reg == r) & pos).astype(np.float64)
            Dw = D * w[:, None]
            A["S1"][0, r] += Dw; A["S2"][0, r] += Dw * D; A["SM"][0, r] += Dw * mb[:, None]
            A["cnt"][0, r] += w; A["Smb"][0, r] += w * mb; A["Smb2"][0, r] += w * mb * mb
    return A


def regional(F, A, pm, reg_id, use):
    a = np.zeros(an.tot(A, "S1").shape); g = np.zeros_like(a)
    for k in np.unique(reg_id):
        mk = pm & (reg_id == k)
        a_o, g_o, _ = an.m3_fit(F, A, mk, 16, use=use)
        a[mk], g[mk] = a_o[mk], g_o[mk]
    return an.m3_eval(A, pm, a, g, np.arange(16))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    meta = np.load(an.W / "meta.npz")
    fc16 = np.load(an.W / "fc16.npy", mmap_mode="r")
    feat = np.load(an.W / "feat.npy", mmap_mode="r")
    S["r"], S["c"] = meta["r"].astype(np.int64), meta["c"].astype(np.int64)
    N = len(S["r"])
    sh = np.zeros((16, N, len(SHW) * len(DIRS)), np.float32)
    with get_context("fork").Pool(workers) as pool:
        for m, o in pool.imap_unordered(shifted, range(16)):
            sh[m] = o
    an.log("shifted features", sh.std((0, 1)).round(3).tolist())
    raw = np.asarray(feat[:16], np.float32).transpose(1, 0, 2)
    Fs = an.feature_sets(raw)
    iso = Fs["lin"]
    ani = sh.transpose(1, 0, 2).astype(np.float64)
    ani = ani - ani.mean(1, keepdims=True)
    both = np.concatenate([iso, ani], -1)
    lat, lon = 84.0 - 0.01 * (meta["r"] + 0.5), -180.0 + 0.01 * (meta["c"] + 0.5)
    reg_id = np.full(N, len(REGIONS))
    for k, (n, s, w, e) in reversed(list(enumerate(REGIONS.values()))):
        reg_id[(lat <= n) & (lat > s) & (lon >= w) & (lon < e)] = k            # first box in the list wins
    R = dict(what=__doc__, region_names=list(REGIONS) + ["other"],
             pixels_by_region_id=[int((reg_id == k).sum()) for k in range(len(REGIONS) + 1)])
    for li, h in enumerate((1, 3)):
        doy = meta["keys16"][li, :, 0]
        A = an.accumulate(fc16, li, doy, 2)
        pm = ~A["bad"]
        o = {}
        for nm, F in (("iso", iso), ("iso+shifted", both)):
            o[f"global_{nm}"] = an.m3_fit(F, A, pm, 16, use=("a", "g"))[2]
            o[f"regional_{nm}"] = regional(F, A, pm, reg_id, ("a", "g"))
            an.log(f"lead {h} {nm}", o[f"global_{nm}"]["all"], o[f"regional_{nm}"]["all"])
        # static map: isotropic vs iso+shifted, global vs regional (target S, member folds)
        S_ = an.tot(A, "S1") / A["T"]
        for nm, F in (("iso", iso), ("iso+shifted", both)):
            p_, _ = an.ridge_cv(F, S_, pm, 16)
            o[f"static_R2_global_{nm}"] = an.r2(S_, p_, pm)
        del A
        L = acc_log(fc16, li)
        pl = pm & ~L["bad"]
        Sl1, Sl2 = an.tot(L, "S1"), an.tot(L, "S2")
        T = L["cnt"].sum((0, 1))
        Tp = T[pl]
        Sl = Sl1 / np.maximum(T, 1)[:, None]
        VR = (Sl2[pl] - Tp[:, None] * Sl[pl] ** 2).sum() / ((Tp - 1).sum() * 16)
        VSh = (Sl[pl] ** 2).mean(); VS = VSh - VR * (1.0 / Tp).mean()
        o["log_static_share"] = float(VS / (VS + VR))
        o["log_rms_departure"] = float(np.sqrt(VS + VR))
        o["log_valid_pixels"] = int(pl.sum())
        for nm, F in (("iso", iso), ("iso+shifted", both)):
            o[f"log_per_forecast_static_{nm}"] = an.m3_fit(F, L, pl, 16, use=("a",))[2]
            o[f"log_per_forecast_static+level_{nm}"] = an.m3_fit(F, L, pl, 16, use=("a", "g"))[2]
        p_, _ = an.ridge_cv(iso, Sl, pl, 16)
        o["log_static_R2_iso"] = an.r2(Sl, p_, pl)
        o["log_odd_static_iso"] = {k: v for k, v in an.oddness(Sl[pl].ravel(), p_[pl].ravel()).items()}
        an.log(f"lead {h} log", o["log_static_share"], o["log_per_forecast_static_iso"]["all"], o["log_odd_static_iso"]["even_over_odd"])
        R[f"lead{h}"] = o
    out = an.OUTF
    base = json.loads(out.read_text())
    base["extra"] = R
    txt = json.dumps(base, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v))
    tmp = out.parent / f"{out.name}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(txt); f.flush(); os.fsync(f.fileno())
    assert tmp.read_text() == txt
    os.replace(tmp, out)
    assert out.read_text() == txt
    an.log("written", out)


if __name__ == "__main__":
    main()
