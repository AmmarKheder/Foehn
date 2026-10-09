#!/usr/bin/env python3
"""Tables of a global evaluation (eval_global_fp.py parts): by region and lead over all valid 1 km land pixels (RMSE,
persistence RMSE, bias, correlation, fair CRPS, spread / error, CSI and Brier score at 25/35/75 ug/m3), at the stations
(daily means of stations_2022.npz, by network and lead; the GHAP product at the same pixels as a reference), and seams.
    python summarize_eval.py runs/<run>/<out> [--json]"""
import argparse
import json
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
NETS = {0: "EEA", 1: "CNEMC", 3: "EPA", 4: "AirKorea", 5: "Japan", 6: "India", 2: "OpenAQ"}


def load(d):
    parts = [np.load(p) for p in sorted(Path(d).glob("part_*.npz"))]
    assert parts, f"no part_*.npz in {d}"
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return dict(cols=[str(c) for c in parts[0]["rec_cols"]], recs=cat("recs"), keys=cat("keys"), seams=cat("seams"),
                st_ens=cat("st_ens").astype(np.float32), st_ghap=cat("st_ghap").astype(np.float32),
                regions=[str(r) for r in parts[0]["regions"]], thresholds=parts[0]["thresholds"], info=list(parts[0]["info"]))


def region_table(e):
    C = {c: k for k, c in enumerate(e["cols"])}
    R = e["recs"]
    out = {}
    for rid, rname in enumerate(e["regions"]):
        for h in sorted(set(R[:, C["lead"]].astype(int))):
            s = R[(R[:, C["region"]] == rid) & (R[:, C["lead"]] == h)].sum(0)
            n = s[C["n"]]
            if n == 0:
                continue
            mt, mf = s[C["sum_t"]] / n, s[C["sum_f"]] / n
            cov = s[C["sum_tf"]] / n - mt * mf
            r = cov / np.sqrt(max(s[C["sum_tt"]] / n - mt ** 2, 1e-12) * max(s[C["sum_ff"]] / n - mf ** 2, 1e-12))
            row = dict(n=int(n), rmse=np.sqrt(s[C["sse"]] / n), rmse_persist=np.sqrt(s[C["sse_persist"]] / n),
                       mae=s[C["sae"]] / n, bias=s[C["sum_err"]] / n, corr=r, crps=s[C["crps"]] / n,
                       spread_skill=np.sqrt(s[C["var"]] / n) / max(np.sqrt(s[C["sse"]] / n), 1e-12), mean_obs=mt)
            for th in e["thresholds"]:
                t = int(th)
                row[f"csi{t}"] = s[C[f"hits_{t}"]] / max(s[C[f"union_{t}"]], 1)
                row[f"brier{t}"] = s[C[f"brier_{t}"]] / n
            out[(rname, h)] = row
    return out


def station_table(e):
    st = np.load(G / "stations_2022.npz")
    obs_all, src = st["daily"], st["source"]
    doy_t, lead = e["keys"][:, 2], e["keys"][:, 1]
    out = {}
    for h in sorted(set(lead)):
        k = np.where(lead == h)[0]
        E = np.transpose(e["st_ens"][k], (1, 0, 2))                 # (stations, forecasts, members)
        gh = e["st_ghap"][k].T
        obs = obs_all[:, doy_t[k]]
        f = E.mean(-1)
        M = E.shape[-1]
        for s_id, net in list(NETS.items()) + [(None, "All")]:
            sel = np.ones(len(src), bool) if s_id is None else src == s_id
            ok = sel[:, None] & np.isfinite(obs) & np.isfinite(f) & np.isfinite(gh)
            if ok.sum() < 10:
                continue
            o, ff, g, EE = obs[ok], f[ok], gh[ok], E[ok]
            pair = sum(np.abs(EE[:, a] - EE[:, b]) for a in range(M) for b in range(a + 1, M)) / (M * (M - 1)) if M > 1 else 0
            crps = np.abs(EE - o[:, None]).mean(-1) - pair
            out[(net, int(h))] = dict(n=int(ok.sum()), stations=int(ok.any(1).sum()), rmse=np.sqrt(np.mean((ff - o) ** 2)),
                                      bias=np.mean(ff - o), corr=np.corrcoef(ff, o)[0, 1], crps=np.mean(crps),
                                      rmse_ghap=np.sqrt(np.mean((g - o) ** 2)), corr_ghap=np.corrcoef(g, o)[0, 1],
                                      mean_obs=np.mean(o))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    e = load(a.dir)
    print(" | ".join(e["info"]), f"| {len(e['keys'])} forecasts")
    rt = region_table(e)
    print(f"\n{'region':15s} lead {'n (M)':>7s} {'RMSE':>6s} {'pers.':>6s} {'bias':>6s} {'corr':>5s} {'CRPS':>6s} {'spr/err':>7s} "
          f"{'CSI25':>6s} {'CSI35':>6s} {'CSI75':>6s}")
    for (rn, h), r in rt.items():
        print(f"{rn:15s} {h:4d} {r['n'] / 1e6:7.1f} {r['rmse']:6.2f} {r['rmse_persist']:6.2f} {r['bias']:+6.2f} {r['corr']:5.2f} "
              f"{r['crps']:6.2f} {r['spread_skill']:7.2f} {r['csi25']:6.3f} {r['csi35']:6.3f} {r['csi75']:6.3f}")
    stt = station_table(e)
    print(f"\n{'network':8s} lead {'obs':>7s} {'stations':>8s} {'RMSE':>6s} {'bias':>6s} {'corr':>5s} {'CRPS':>6s} | GHAP: {'RMSE':>6s} {'corr':>5s}")
    for (net, h), r in stt.items():
        print(f"{net:8s} {h:4d} {r['n']:7d} {r['stations']:8d} {r['rmse']:6.2f} {r['bias']:+6.2f} {r['corr']:5.2f} {r['crps']:6.2f} | "
              f"      {r['rmse_ghap']:6.2f} {r['corr_ghap']:5.2f}")
    sm = e["seams"]
    print(f"\nseam ratio (1 = no seam): mean {np.nanmean(sm[:, 2]):.3f}, max {np.nanmax(sm[:, 2]):.3f}")
    if a.json:
        js = dict(regions={f"{k[0]}|{k[1]}": {kk: float(v) for kk, v in r.items()} for k, r in rt.items()},
                  stations={f"{k[0]}|{k[1]}": {kk: float(v) for kk, v in r.items()} for k, r in stt.items()},
                  seam_mean=float(np.nanmean(sm[:, 2])), info=e["info"])
        (Path(a.dir) / "summary.json").write_text(json.dumps(js, indent=1))
