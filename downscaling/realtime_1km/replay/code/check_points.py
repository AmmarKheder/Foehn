#!/usr/bin/env python3
"""WP1a check of step C: the pointwise members against the PUBLISHED full-field products of the same issue day.

  python -I check_points.py <day> [--tiles] [--json FILE]
(1) monitor pixels: member mean, 10th and 90th percentiles (linear interpolation over the 8 members, as infer.py's
    torch.quantile) of replay/km/<day>.npz 'km' against the snapshot realtime_1km/verif/fc/<day>.npz ('mean', 'p10', 'p90':
    values read on the day in the published float16 tiles). Reported per product: correlation, mean / 99.9th percentile /
    max absolute difference, share of values identical after the same float16 rounding, share within one float16 step.
(2) --tiles: the same at EVERY evaluated position (monitor pixels, rings, cell samples: replay/raw/<day>.npy) against the
    tiles realtime_1km/out/<day>/pmG_final_P3T3/lead<L>_part<r>.npz, plus the exceedance fractions prob15 / prob37_5 / prob75.
(3) when the production inputs realtime_1km/work/<day>/chan_L<L>.npz still exist: bitwise comparison with replay/work/<day>/."""
import argparse
import json
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
R = P / "realtime_1km/replay"
TAG = "pmG_final_P3T3"


def cmp(a, b):
    """a = replay (float32), b = published (float16 stored value)"""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok].astype(np.float64), b[ok].astype(np.float64)
    a16 = a.astype(np.float16).astype(np.float64)
    step = np.spacing(np.abs(b).astype(np.float16)).astype(np.float64)
    d = np.abs(a - b)
    return dict(n=int(ok.sum()), corr=float(np.corrcoef(a, b)[0, 1]), mean_abs=float(d.mean()), p999_abs=float(np.percentile(d, 99.9)),
                max_abs=float(d.max()), max_rel=float((d / np.maximum(np.abs(b), 1.0)).max()),
                identical_after_float16=float((a16 == b).mean()), within_1_float16_step=float((np.abs(a16 - b) <= step).mean()),
                mean_replay=float(a.mean()), mean_published=float(b.mean()))


def stats(km):
    """(M, n) members -> mean, p10, p90 as infer.py"""
    q = np.quantile(km.astype(np.float64), [0.1, 0.9], axis=0)
    return km.astype(np.float64).mean(0), q[0], q[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day")
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--km", default=str(R / "km"))
    ap.add_argument("--raw", default=str(R / "raw"))
    ap.add_argument("--work", default=str(R / "work"))
    a = ap.parse_args()
    d = a.day
    out = {"day": d}
    z = np.load(Path(a.km) / f"{d}.npz")
    km = z["km"]
    snap = np.load(P / "realtime_1km/verif/fc" / f"{d}.npz")
    Z = np.load(R / "positions.npz")
    assert np.array_equal(snap["id"], Z["id"]) and str(snap["tag"]) == TAG, (str(snap["tag"]),)
    out["snapshot"] = {"file": str(P / "realtime_1km/verif/fc" / f"{d}.npz"), "tag": str(snap["tag"]), "created": str(snap["created"])}
    for L in range(1, 6):
        if not np.isfinite(km[L - 1]).any():
            continue
        m, p10, p90 = stats(km[L - 1])
        out["snapshot"][f"lead{L}"] = {"mean": cmp(m, snap["mean"][L - 1]), "p10": cmp(p10, snap["p10"][L - 1]), "p90": cmp(p90, snap["p90"][L - 1])}
    allm = np.concatenate([stats(km[L])[0] for L in range(5)]); alls = snap["mean"].reshape(-1)
    out["snapshot"]["all_leads_mean"] = cmp(allm, alls)
    out["snapshot"]["all_leads_p10"] = cmp(np.concatenate([stats(km[L])[1] for L in range(5)]), snap["p10"].reshape(-1))
    out["snapshot"]["all_leads_p90"] = cmp(np.concatenate([stats(km[L])[2] for L in range(5)]), snap["p90"].reshape(-1))
    if a.tiles:
        raw = np.load(Path(a.raw) / f"{d}.npy")                                   # (5, M, U) float32
        pos = Z["pos"].astype(np.int64)
        flat = np.load(R / "positions_store.npz")["flat"].astype(np.int64)
        src = P / "realtime_1km/out" / d / TAG
        lat_row0 = float(json.loads((src / "part0.json").read_text())["lattice"]["lat_row0"])
        i = np.rint((83.995 - 0.01 * (flat // 36000) - lat_row0) / 0.01).astype(np.int64)
        j = flat % 36000
        rr, cc = i // 1000, j // 1000
        rows_, cols_ = 999 - (i - rr * 1000), j - cc * 1000
        tkey = rr * 100 + cc
        part_of = {}
        for pj in sorted(src.glob("part*.json")):
            jd = json.loads(pj.read_text())
            for t in jd["tiles"]:
                part_of[t] = int(jd["rank"])
        out["tiles"] = {"dir": str(src), "n_positions": int(pos.size)}
        prods = ("mean", "p10", "p90", "prob15", "prob37_5", "prob75")
        for L in range(1, 6):
            if not np.isfinite(raw[L - 1].astype(np.float32)).any():
                continue
            ref = {p: np.full(pos.size, np.nan, np.float32) for p in prods}
            npz = {}
            for t in np.unique(tkey):
                key = f"r{t // 100:02d}c{t % 100:02d}"
                r = part_of[key]
                if r not in npz:
                    npz[r] = np.load(src / f"lead{L}_part{r}.npz")
                s = tkey == t
                for p in prods:
                    ref[p][s] = npz[r][f"{p}/{key}"][rows_[s], cols_[s]].astype(np.float32)
            x = raw[L - 1].astype(np.float32)
            m, p10, p90 = stats(x)
            res = {"mean": cmp(m, ref["mean"]), "p10": cmp(p10, ref["p10"]), "p90": cmp(p90, ref["p90"])}
            for p, th in (("prob15", 15.0), ("prob37_5", 37.5), ("prob75", 75.0)):
                f = (x > th).mean(0)
                dd = np.abs(f - ref[p])
                res[p] = dict(n=int(np.isfinite(dd).sum()), identical=float((dd < 1e-3).mean()), differ_by_one_member=float(((dd > 1e-3) & (dd < 0.126)).mean()),
                              differ_more=float((dd >= 0.126).mean()), nan_in_tiles=int(np.isnan(ref[p]).sum()))
            out["tiles"][f"lead{L}"] = res
    chan = {}
    for L in range(1, 6):
        pf, rf = P / "realtime_1km/work" / d / f"chan_L{L}.npz", Path(a.work) / d / f"chan_L{L}.npz"
        if pf.exists() and rf.exists():
            x, y = np.load(rf), np.load(pf)
            chan[f"L{L}"] = {k: (bool(np.array_equal(x[k], y[k])) if x[k].shape == y[k].shape else False) for k in y.files}
            for k in ("cams", "met", "fires", "t6"):
                if not chan[f"L{L}"][k]:
                    dd = np.abs(x[k].astype(np.float32) - y[k].astype(np.float32))
                    chan[f"L{L}"][k + "_max_abs"] = float(dd.max())
                    chan[f"L{L}"][k + "_share_identical"] = float((x[k] == y[k]).mean())
    out["chan_vs_production_work"] = chan or "production work directory not on disk for this day"
    out["km_meta"] = {k: v for k, v in json.loads(str(z["meta"])).items() if k in ("ckpt_sha256", "prior_sha256", "guard", "step", "members", "leads_made", "seconds", "device", "job")}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
