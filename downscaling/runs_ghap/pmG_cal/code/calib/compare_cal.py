#!/usr/bin/env python3
"""calib.compare_cal -- before/after tables of PROTOCOL.md A6(g)-(i) from eval_x outputs (CPU, numpy only).

For each system (label=eval_dir; the first is the reference, normally the frozen system's full SEL evaluation): pooled
pixel scores (fair CRPS, RMSE of the ensemble mean, bias, Fortin spread/skill, D = RMSE^2 - V/M), 0.25 / 0.5 / 1 degree
block scores, rank histogram, log1p scores, spectra (power of member 0 and of the ensemble mean relative to GHAP: geometric
mean over 2-20 km, 20-100 km and the G2 band 10-140 km), extreme members (G0a / G0b counts, cells, new cells), gates.
Intervals: bootstrap over the selection days (5,000 draws, percentile 95 %) of each pooled score and of the paired
difference to the reference; the pooled values are rebuilt from the daily rows of metrics.json (every day has the same
pixel weights), and the rebuild is checked against the pooled values eval_x wrote.

  python compare_cal.py --out results/compare_full frozen=<dir> V2@1500=<dir> T1=<dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

M = 8
BANDS = {"2-20km": (2.0, 20.0), "20-100km": (20.0, 100.0), "G2_10-140km": (10.0, 140.0)}


def psd_ratios(f):
    if not Path(f).exists():
        return None
    z = np.load(f)
    k, psd, fields = z["k_cycles_per_km"], z["psd"], [str(x) for x in z["fields"]]
    lam = 1.0 / k
    out = {"n_box_days": float(z["n_box_days"])}
    for name in ("member0", "ensmean", "cams_bilinear"):
        num, den = psd[fields.index(name)], psd[fields.index("truth")]
        out[name] = {}
        for b, (lo, hi) in BANDS.items():
            sel = (lam >= lo) & (lam <= hi) & np.isfinite(num) & np.isfinite(den) & (den > 0)
            out[name][b] = float(np.exp(np.mean(np.log(num[sel] / den[sel])))) if sel.any() else None
    return out


def day_arrays(m):
    """per-day (crps, mse, var) of the pixel scores and of the 1 degree block scores, equal weights per day"""
    d = m["daily"]
    pix = np.array([[r["model"]["crps_fair"], r["model"]["rmse_ensmean"] ** 2,
                     (r["model"]["spread_skill_fortin"] * r["model"]["rmse_ensmean"]) ** 2 * M / (M + 1)] for r in d])
    blk = {}
    for s in ("1", "2", "4"):
        blk[s] = np.array([[r["model"]["blocks"][s]["ug"]["crps_fair"], r["model"]["blocks"][s]["ug"]["rmse_ensmean"] ** 2,
                            (r["model"]["blocks"][s]["ug"]["spread_skill_fortin"] * r["model"]["blocks"][s]["ug"]["rmse_ensmean"]) ** 2
                            * M / (M + 1)] for r in d])
    bias = np.array([r["model"]["bias"] for r in d])
    return [r["date"] for r in d], pix, blk, bias


def pooled(a):
    """a (..., n_days, 3) -> dict of pooled crps, rmse, ssF, D over the day axis"""
    c, mse, var = a[..., 0].mean(-1), a[..., 1].mean(-1), a[..., 2].mean(-1)
    rm = np.sqrt(mse)
    return {"crps_fair": c, "rmse_ensmean": rm, "spread_skill_fortin": np.sqrt((M + 1) / M * var) / rm, "D": mse - var / M}


def boot(a, ref=None, n_boot=5000, seed=0):
    nd = a.shape[0]
    idx = np.random.default_rng(seed).integers(0, nd, size=(n_boot, nd))
    bs = pooled(a[idx])
    out = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    if ref is not None:
        br = pooled(ref[idx])
        out["diff"] = {k: [float(np.percentile(bs[k] - br[k], 2.5)), float(np.percentile(bs[k] - br[k], 97.5))] for k in bs}
        out["rel_pct"] = {k: [float(np.percentile(100 * (bs[k] / br[k] - 1), 2.5)), float(np.percentile(100 * (bs[k] / br[k] - 1), 97.5))]
                          for k in bs}
    return out


def load(ed):
    ed = Path(ed)
    m = json.loads((ed / "metrics.json").read_text())
    dates, pix, blk, bias = day_arrays(m)
    u = m["scores"]["model"]["ug"]
    p = {k: float(v) for k, v in pooled(pix).items()}
    for k, kk in (("crps_fair", "crps_fair"), ("rmse_ensmean", "rmse_ensmean"), ("spread_skill_fortin", "spread_skill_fortin")):
        assert abs(p[k] - u[kk]) <= 2e-4 * abs(u[kk]), (str(ed), k, p[k], u[kk])        # the daily rebuild equals eval_x's pooled value
    ext = np.load(ed / "extremes.npz") if (ed / "extremes.npz").exists() else None
    g = json.loads((ed / "VALIDATION_GATES.json").read_text()) if (ed / "VALIDATION_GATES.json").exists() else None
    cal = json.loads((ed / "CAL.json").read_text()) if (ed / "CAL.json").exists() else {}
    return {"dir": str(ed), "m": m, "dates": dates, "pix": pix, "blk": blk, "bias": bias, "ext": ext, "gates": g,
            "psd": psd_ratios(ed / "psd.npz"), "tau": cal.get("tau", 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="label=eval_dir; the first is the reference")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = [(x.split("=", 1)[0], load(x.split("=", 1)[1])) for x in a.systems]
    ref_name, ref = S[0]
    res = {"reference": ref_name, "members": M, "n_boot": 5000, "systems": {}}
    md = [f"# A6 selection-year comparison (24 SEL days of 2021; reference = {ref_name})", "",
          "Intervals: bootstrap over the 24 selection days (5,000 draws, percentile 95 %); differences are paired "
          "(same days, same pixels). Selection-year numbers, not test results.", ""]
    rows = []
    for name, s in S:
        assert s["dates"] == ref["dates"], f"{name}: other days than the reference"
        m = s["m"]
        u, lg = m["scores"]["model"]["ug"], m["scores"]["model"]["log"]
        same_px = m["n_pixels_per_day"] == ref["m"]["n_pixels_per_day"]
        p = {k: float(v) for k, v in pooled(s["pix"]).items()}
        b = boot(s["pix"], ref["pix"] if (s is not ref and same_px) else None)
        r = {"dir": s["dir"], "step": m["step"], "tau": s["tau"], "n_pixels_per_day": m["n_pixels_per_day"], "pixel_stride": m["pixel_stride"],
             "pixel": {"crps_fair": u["crps_fair"], "rmse_ensmean": u["rmse_ensmean"], "bias": u["bias"],
                       "spread_skill_fortin": u["spread_skill_fortin"], "D": p["D"], "rmse_1member": u["rmse_1member"],
                       "rank_hist": u["rank_hist"], "outside_ensemble": u["rank_hist"][0] + u["rank_hist"][-1], "boot95": b},
             "log1p": {"crps_fair": lg["crps_fair"], "rmse_ensmean": lg["rmse_ensmean"], "bias": lg["bias"],
                       "spread_skill_fortin": lg["spread_skill_fortin"]},
             "blocks": {}, "sanity": m["sanity"], "psd": s["psd"],
             "gates": None if s["gates"] is None else {k.split("_")[0]: {"pass": v["pass"], "value": v.get("value", v.get("model"))}
                                                       for k, v in s["gates"]["gates"].items()},
             "daily_ssF": [float(x) for x in np.sqrt((M + 1) / M * s["pix"][:, 2] / s["pix"][:, 1])],
             "daily_rmse": [float(x) for x in np.sqrt(s["pix"][:, 1])], "daily_crps": [float(x) for x in s["pix"][:, 0]]}
        for sc, deg in (("1", "0.25"), ("2", "0.5"), ("4", "1")):
            bu = m["blocks"]["model"][sc]["ug"]
            bb = boot(s["blk"][sc], ref["blk"][sc] if (s is not ref and same_px) else None)
            r["blocks"][deg] = {"crps_fair": bu["crps_fair"], "rmse_ensmean": bu["rmse_ensmean"],
                                "spread_skill_fortin": bu["spread_skill_fortin"], "boot95": bb}
        if s["ext"] is not None and ref["ext"] is not None:
            c, c0 = s["ext"]["cells_pix_gt_440"] > 0, ref["ext"]["cells_pix_gt_440"] > 0
            if c.shape == c0.shape:
                new = np.argwhere(c & ~c0)
                r["extreme_cells"] = {"n": int(c.sum()), "n_reference": int(c0.sum()), "n_new": int(len(new)),
                                      "new_cells_lat_lon": [[round(84.0 - 0.25 * int(i), 2), round(-180.0 + 0.25 * int(j), 2)] for i, j in new[:200]],
                                      "pixel_days_in_new_cells": float(s["ext"]["cells_pix_gt_440"][c & ~c0].sum()),
                                      "daily_pix_gt_TG": [float(x) for x in s["ext"]["daily_pix_gt_440"]]}
        res["systems"][name] = r
        rows.append((name, r))

    def iv(x):
        return f"[{x[0]:.3f}, {x[1]:.3f}]"
    md += ["## Pixel scores (µg m⁻³, all evaluated pixels, cos-lat weights)", "",
           "| system | fair CRPS [95 %] | RMSE ens. mean [95 %] | bias | spread/skill [95 %] | D | truth outside ensemble |", "|---|---|---|---|---|---|---|"]
    for name, r in rows:
        p = r["pixel"]
        md.append(f"| {name} | {p['crps_fair']:.4f} {iv(p['boot95']['crps_fair'])} | {p['rmse_ensmean']:.4f} {iv(p['boot95']['rmse_ensmean'])} | "
                  f"{p['bias']:+.3f} | {p['spread_skill_fortin']:.4f} {iv(p['boot95']['spread_skill_fortin'])} | {p['D']:.2f} | {100 * p['outside_ensemble']:.1f} % |")
    md += ["", f"## Paired differences to {ref_name} (pixel; absolute and relative, 95 %)", "",
           "| system | Δ CRPS | Δ CRPS % | Δ RMSE | Δ RMSE % | Δ spread/skill | Δ D % |", "|---|---|---|---|---|---|---|"]
    r0 = rows[0][1]["pixel"]
    for name, r in rows[1:]:
        p = r["pixel"]
        if "diff" not in p["boot95"]:
            md.append(f"| {name} | (other pixel set: unpaired) | | | | | |")
            continue
        d, rl = p["boot95"]["diff"], p["boot95"]["rel_pct"]
        md.append(f"| {name} | {p['crps_fair'] - r0['crps_fair']:+.4f} {iv(d['crps_fair'])} | {100 * (p['crps_fair'] / r0['crps_fair'] - 1):+.2f} {iv(rl['crps_fair'])} | "
                  f"{p['rmse_ensmean'] - r0['rmse_ensmean']:+.4f} {iv(d['rmse_ensmean'])} | {100 * (p['rmse_ensmean'] / r0['rmse_ensmean'] - 1):+.2f} {iv(rl['rmse_ensmean'])} | "
                  f"{p['spread_skill_fortin'] - r0['spread_skill_fortin']:+.4f} {iv(d['spread_skill_fortin'])} | {100 * (p['D'] / r0['D'] - 1):+.2f} {iv(rl['D'])} |")
    md += ["", "## Block means (µg m⁻³): fair CRPS / RMSE / spread-skill [95 %]", "", "| system | 0.25° | 0.5° | 1° |", "|---|---|---|---|"]
    for name, r in rows:
        cells = []
        for deg in ("0.25", "0.5", "1"):
            b = r["blocks"][deg]
            cells.append(f"{b['crps_fair']:.3f} / {b['rmse_ensmean']:.3f} / {b['spread_skill_fortin']:.3f} {iv(b['boot95']['spread_skill_fortin'])}")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", "## log1p scores and rank histogram", "", "| system | log1p CRPS | log1p spread/skill | rank histogram (truth among 8 members) |", "|---|---|---|---|"]
    for name, r in rows:
        md.append(f"| {name} | {r['log1p']['crps_fair']:.4f} | {r['log1p']['spread_skill_fortin']:.4f} | " + " ".join(f"{x:.3f}" for x in r["pixel"]["rank_hist"]) + " |")
    md += ["", "## Spectra: power relative to GHAP (log1p fields, six regime boxes; geometric mean over the band)", "",
           "| system | member 2-20 km | member 20-100 km | member 10-140 km (G2) | ens. mean 2-20 km | ens. mean 20-100 km | box-days |", "|---|---|---|---|---|---|---|"]
    for name, r in rows:
        p = r["psd"]
        if p is None:
            md.append(f"| {name} | (no spectra) | | | | | |")
            continue
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        md.append(f"| {name} | {f(p['member0']['2-20km'])} | {f(p['member0']['20-100km'])} | {f(p['member0']['G2_10-140km'])} | {f(p['ensmean']['2-20km'])} | "
                  f"{f(p['ensmean']['20-100km'])} | {p['n_box_days']:.0f} |")
    md += ["", "## Extreme members and gates", "",
           "| system | pixel-days with a member > T_G (share) | members at state bound | members at cap | G0b share | largest member | cells > T_G (new) | gates failing |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        s = r["sanity"]
        ec = r.get("extreme_cells", {})
        gf = "n/a" if r["gates"] is None else (", ".join(k for k, v in r["gates"].items() if v["pass"] is False) or "none")
        md.append(f"| {name} | {s['n_pix_gt_440']:.0f} ({s['frac_pix_days_any_member_gt_440']:.2e}) | {s['n_member_at_bound']:.0f} | {s['n_member_at_pmax']:.0f} | "
                  f"{s['frac_member_pixel_days_guard_bound']:.2e} | {s['max_member_ug']:.0f} | {ec.get('n', s['n_cells_with_pix_gt_440'])} ({ec.get('n_new', '-')}) | {gf} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {out.with_suffix('.json')} , {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
