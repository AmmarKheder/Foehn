#!/usr/bin/env python3
"""A8(f)-(g): seeds 0 (frozen system, files read as they are), 1 and 2 side by side on the 24 selection days of 2021.

Reads only small json / npz result files (no field, no truth, no station, nothing of 2022):
  seed 0   runs_ghap/pmG_base/{metrics.jsonl, SELECTED.json, copula_s<step>/{fit,test}.json,
           eval_sel2021_M8_s10_<prior>_guard_T3/{metrics.json, psd.npz, extremes.npz}}, runs_ghap/pmG_final/{PRIOR.json,
           VALIDATION_GATES.json}
  seed k   runs_ghap/pmG_seed<k>/... (same files; gates and prior name in pmG_seed<k>/final/)
Writes a8/results/seed_compare.json (every number) and a8/results/seed_compare.md (tables). numpy only.

Noise statistics over the n available seeds: range = max - min, s = sample s.d. (n - 1), and the 95 % prediction
interval for ONE new seed, mean +- t(0.975, n - 1) * s * sqrt(1 + 1/n) (n = 3: 4.303 * s * 1.155 = 4.97 s). Rates of rare
events are treated on log10. Reading rules: protocol_amendments/A8_seed_replicates.md (g).

  python compare_seeds.py [--out-dir DIR]"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

RG = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap")
A8 = RG / "pmG_seed1" / "a8"
SEEDS = {0: dict(run="pmG_base", final="pmG_final"), 1: dict(run="pmG_seed1", final="pmG_seed1/final"),
         2: dict(run="pmG_seed2", final="pmG_seed2/final")}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
WL_TABLE = (239.0, 104.0, 52.0, 20.0, 9.8, 4.9, 2.1)       # wavelengths of the fact sheet's spectra table (km)


def jload(p):
    return json.loads(Path(p).read_text())


def load_seed(k):
    """everything of one seed, or None if its evaluation is not there yet"""
    s = SEEDS[k]
    rd, fd = RG / s["run"], RG / s["final"]
    out = {"seed": k, "run": s["run"], "run_dir": str(rd), "files": {}}
    mj = rd / "metrics.jsonl"
    if not mj.exists():
        return None
    rows = {}
    for line in open(mj):
        if line.strip():
            r = json.loads(line)
            rows[int(r["step"])] = r
    out["val"] = {st: {"crps": r["model"]["ug"]["crps_fair"], "rmse": r["model"]["ug"]["rmse_ensmean"],
                       "bias": r["model"]["ug"]["bias"], "ssF": r["model"]["ug"]["spread_skill_fortin"],
                       "train_loss": r["train_window"].get("loss"), "ms_per_step": r["train_window"].get("ms_per_step"),
                       "grad_norm_mean": r["train_window"].get("grad_norm_mean"), "elapsed_min": r["elapsed_min"]}
                  for st, r in sorted(rows.items())}
    out["files"]["metrics.jsonl"] = str(mj)
    base = next((r["baselines"] for r in rows.values() if "baselines" in r), None)
    out["val_baselines"] = None if base is None else {n: {"crps": b["ug"]["crps_fair"], "rmse": b["ug"]["rmse_ensmean"]} for n, b in base.items()}
    sj = rd / "SELECTED.json"
    if not sj.exists():
        out["stage"] = "trained"
        return out
    sel = jload(sj)
    out["selected_step"] = int(sel["selected_step"])
    out["sha256"] = sel.get("sha256")
    out["sel_model"] = sel["model"]
    out["files"]["SELECTED.json"] = str(sj)
    cd = rd / f"copula_s{out['selected_step']}"
    if (cd / "test.json").exists() and (cd / "fit.json").exists():
        t, f = jload(cd / "test.json"), jload(cd / "fit.json")
        out["prior"] = {"loglik": {n: v for n, v in t["loglik_nats_per_point"].items()},
                        "delta_P3_vs_P1r": t["delta_P3_vs_P1r"], "latent_std": t["latent_std"], "latent_mean": t["latent_mean"],
                        "rule_name": "P3" if t["delta_P3_vs_P1r"]["ci95"][0] > 0 else "P1r",
                        "band_variance": {n: f["priors"][n]["mean_band_variance"] for n in ("P1s", "P1r", "P3")},
                        "fit_val_ll": {n: f["priors"][n]["val_ll"] for n in f["priors"]},
                        "latent_corr_300km": {d: v["empirical"] for d, v in t["latent_correlation_by_distance"].get("300km", {}).items()}}
        out["files"]["copula"] = str(cd)
    pj = fd / "PRIOR.json"
    if not pj.exists():
        out["stage"] = "selected"
        return out
    name = jload(pj)["name"]
    out["prior_name"] = name
    ed = rd / f"eval_sel2021_M8_s10_{name}_guard_T3"
    if not (ed / "metrics.json").exists():
        out["stage"] = "prior"
        return out
    m = jload(ed / "metrics.json")
    out["files"]["eval"] = str(ed)
    out["eval"] = m
    out["gates_file"] = jload(fd / "VALIDATION_GATES.json") if (fd / "VALIDATION_GATES.json").exists() else None
    out["files"]["gates"] = str(fd / "VALIDATION_GATES.json")
    z = np.load(ed / "psd.npz")
    out["psd"] = {"k": z["k_cycles_per_km"], "fields": [str(x) for x in z["fields"]], "psd": z["psd"], "n_box_days": float(z["n_box_days"])}
    e = np.load(ed / "extremes.npz")
    out["ext_cells"] = e["cells_pix_gt_440"]
    out["stage"] = "evaluated"
    return out


def psd_ratio(P, field="member0", lo=10.0, hi=140.0):
    k, psd, fields = P["k"], P["psd"], P["fields"]
    lam = 1.0 / k
    sel = (lam >= lo) & (lam <= hi)
    num, den = psd[fields.index(field)][sel], psd[fields.index("truth")][sel]
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return float(np.exp(np.mean(np.log(num[ok] / den[ok]))))


def stats(vals, log=False):
    """values of the available seeds -> dict(mean, sd, range, min, max, pi95 for one new seed, n)"""
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return {"n": 0}
    w = np.log10(np.maximum(v, 1e-300)) if log else v
    r = {"n": int(n), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "range": float(v.max() - v.min())}
    if log:
        r["max_over_min"] = float(v.max() / max(v.min(), 1e-300))
    if n >= 2:
        s = float(w.std(ddof=1))
        h = T975[n - 1] * s * math.sqrt(1.0 + 1.0 / n)
        r["sd"] = float(v.std(ddof=1))
        if log:
            r["sd_log10"] = s
            r["pi95_new_seed"] = [float(10 ** (w.mean() - h)), float(10 ** (w.mean() + h))]
        else:
            r["pi95_new_seed"] = [float(w.mean() - h), float(w.mean() + h)]
            r["rel_range_pct"] = float(100.0 * r["range"] / abs(v.mean())) if v.mean() != 0 else None
            r["rel_sd_pct"] = float(100.0 * r["sd"] / abs(v.mean())) if v.mean() != 0 else None
    return r


def fmt(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e6):
        return f"{x:.3g}"
    return f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(A8 / "results"))
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    S = {k: load_seed(k) for k in SEEDS}
    S = {k: v for k, v in S.items() if v is not None}
    ev = [k for k, v in S.items() if v.get("stage") == "evaluated"]
    md = []
    J = {"seeds_available": {k: v.get("stage") for k, v in S.items()}, "files": {k: v["files"] for k, v in S.items()}}
    hdr = lambda cols: "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    md.append("# A8 — seeds 0 (frozen), 1, 2 on the 24 selection days of 2021\n")
    md.append("Written by `runs_ghap/pmG_seed1/a8/compare_seeds.py` from the result files listed at the end. Seed 0 = the frozen "
              "system's own files (`pmG_base`, `pmG_final`), not recomputed. Range = max − min over the seeds; s.d. with n − 1; "
              "PI = 95 % prediction interval for one new seed (mean ± 4.303 s √(1 + 1/3) for three seeds). Three seeds give a "
              "s.d. with 2 degrees of freedom (true s.d. between 0.52 and 6.3 times the estimate, 95 %).\n")
    stages = ", ".join("seed %d: %s" % (k, v.get("stage")) for k, v in S.items())
    md.append(f"Stage of each seed: {stages}.\n")

    # ---------------------------------------------------------------- 1. in-training validation and selection
    steps = sorted(set().union(*[set(v["val"]) for v in S.values()]))
    md.append("## 1. In-training validation (100,000 pixels, 24 selection days, training noise) and selected step\n")
    md.append(hdr(["step"] + [f"CRPS seed {k}" for k in S] + ["range"] + [f"RMSE seed {k}" for k in S] + [f"train loss seed {k}" for k in S]))
    J["val_by_step"] = {}
    for st in steps:
        c = [S[k]["val"].get(st, {}).get("crps") for k in S]
        r = [S[k]["val"].get(st, {}).get("rmse") for k in S]
        tl = [S[k]["val"].get(st, {}).get("train_loss") for k in S]
        ok = [x for x in c if x is not None]
        J["val_by_step"][st] = {"crps": dict(zip(S, c)), "rmse": dict(zip(S, r)), "train_loss": dict(zip(S, tl)),
                                "crps_range": (max(ok) - min(ok)) if len(ok) > 1 else None}
        md.append("| " + " | ".join([str(st)] + [fmt(x, 4) for x in c] + [fmt(J["val_by_step"][st]["crps_range"], 4)]
                                    + [fmt(x, 3) for x in r] + [fmt(x, 4) for x in tl]) + " |")
    J["selection"] = {}
    md.append("")
    md.append(hdr(["", *[f"seed {k}" for k in S], "range"]))
    sel_steps = [S[k].get("selected_step") for k in S]
    sel_crps = [S[k]["val"][S[k]["selected_step"]]["crps"] if S[k].get("selected_step") else None for k in S]
    late = {}
    for k in S:
        v = [S[k]["val"][st]["crps"] for st in steps if st >= 18000 and st in S[k]["val"]]
        late[k] = (float(np.std(v, ddof=1)) if len(v) > 1 else None, (max(v) - min(v)) if v else None)
    best24 = [S[k]["val"].get(24000, {}).get("crps") for k in S]
    md.append("| selected step | " + " | ".join(fmt(x, 0) if x else "—" for x in sel_steps) + " | |")
    md.append("| CRPS at the selected step | " + " | ".join(fmt(x, 4) for x in sel_crps) + f" | {fmt(stats(sel_crps).get('range'), 4)} |")
    md.append("| CRPS at step 22,000 | " + " | ".join(fmt(S[k]['val'].get(22000, {}).get('crps'), 4) for k in S)
              + f" | {fmt(stats([S[k]['val'].get(22000, {}).get('crps') for k in S]).get('range'), 4)} |")
    md.append("| CRPS at step 24,000 | " + " | ".join(fmt(x, 4) for x in best24) + f" | {fmt(stats(best24).get('range'), 4)} |")
    md.append("| within-run range of the CRPS over steps 18,000-24,000 | " + " | ".join(fmt(late[k][1], 4) for k in S) + " | |")
    md.append("| sha256 of the selected checkpoint (first 12) | " + " | ".join((S[k].get("sha256") or "—")[:12] for k in S) + " | |")
    J["selection"] = {"selected_step": dict(zip(S, sel_steps)), "crps_at_selected": dict(zip(S, sel_crps)),
                      "crps_at_selected_stats": stats(sel_crps), "late_sd_range_18k_24k": late,
                      "crps_at_24000": dict(zip(S, best24))}
    vb = {k: S[k].get("val_baselines") for k in S}
    J["val_baselines"] = vb
    md.append("\nBaselines on the validation pixels (must be identical across seeds: same pixels, same days): "
              + "; ".join(f"seed {k}: climatology median MAE {fmt(v['climatology_median']['crps'], 4)}, CAMS bilinear MAE {fmt(v['cams_bilinear']['crps'], 4)}"
                          for k, v in vb.items() if v) + ".\n")

    # ---------------------------------------------------------------- 2. prior
    P = {k: S[k]["prior"] for k in S if "prior" in S[k]}
    if P:
        md.append("## 2. Noise prior refitted on each seed (check days of 2021, nats per point)\n")
        md.append(hdr(["", *[f"seed {k}" for k in P], "range"]))
        rows = [("log-likelihood P1r", [P[k]["loglik"]["P1r"]["mean"] for k in P]),
                ("log-likelihood P3", [P[k]["loglik"]["P3"]["mean"] for k in P]),
                ("log-likelihood P1 (training noise)", [P[k]["loglik"]["P1"]["mean"] for k in P]),
                ("log-likelihood P0 (independent)", [P[k]["loglik"]["P0"]["mean"] for k in P]),
                ("P3 − P1r", [P[k]["delta_P3_vs_P1r"]["mean"] for k in P]),
                ("P3 − P1r, lower bound of the 95 % interval", [P[k]["delta_P3_vs_P1r"]["ci95"][0] for k in P]),
                ("P3 − P1r, upper bound", [P[k]["delta_P3_vs_P1r"]["ci95"][1] for k in P]),
                ("latent s.d. (1 = calibrated marginals)", [P[k]["latent_std"] for k in P]),
                ("latent mean", [P[k]["latent_mean"] for k in P])]
        J["prior"] = {"rule_name": {k: P[k]["rule_name"] for k in P}}
        for name, vals in rows:
            st = stats(vals)
            J["prior"][name] = {"values": dict(zip(P, vals)), "stats": st}
            md.append(f"| {name} | " + " | ".join(fmt(x, 4) for x in vals) + f" | {fmt(st.get('range'), 4)} |")
        md.append("| prior chosen by the rule | " + " | ".join(P[k]["rule_name"] for k in P) + " | |")
        for pn in ("P3", "P1r"):
            md.append(f"| {pn}: share of variance per band (1-4 / 4-20 / 20-100 / 100-500 / 500-3000 km) | "
                      + " | ".join(" / ".join(f"{100 * x:.0f}" for x in P[k]["band_variance"][pn]) for k in P) + " | |")
        J["prior"]["band_variance"] = {k: P[k]["band_variance"] for k in P}
        md.append("")

    if len(ev) == 0:
        (od / "seed_compare.md").write_text("\n".join(md) + "\n")
        (od / "seed_compare.json").write_text(json.dumps(J, indent=1, default=float))
        print("\n".join(md))
        return
    E = {k: S[k]["eval"] for k in ev}

    # ---------------------------------------------------------------- 3. evaluation scores
    def row(name, getter, d=3, log=False, section=None):
        vals = [getter(k) for k in ev]
        st = stats(vals, log=log)
        section[name] = {"values": dict(zip(ev, vals)), "stats": st}
        pi = st.get("pi95_new_seed")
        md.append(f"| {name} | " + " | ".join(fmt(x, d) for x in vals) + f" | {fmt(st.get('range'), d + 1)} | {fmt(st.get('sd'), d + 1)} | "
                  + (f"{fmt(st.get('rel_sd_pct'), 2)} %" if st.get("rel_sd_pct") is not None else "—") + " | "
                  + (f"[{fmt(pi[0], d)}, {fmt(pi[1], d)}]" if pi else "—") + " |")

    H = hdr(["", *[f"seed {k}" for k in ev], "range", "s.d.", "s.d. / mean", "PI for a new seed"])
    md.append("## 3. Evaluation on the 24 selection days, all 147,289,344 mask pixels (8 members × 10 Euler steps, fitted prior, guard, truncation)\n")
    md.append("Evaluated with: " + "; ".join(f"seed {k}: step {E[k]['step']}, prior {S[k]['prior_name']}, guard {E[k]['guard']}" for k in ev) + ".\n")
    md.append(H)
    J["scores"] = {}
    def g(k, *path):
        d = E[k]
        for key in path:
            d = d[key]
        return d
    row("pixel fair CRPS (µg m⁻³)", lambda k: g(k, "scores", "model", "ug", "crps_fair"), 3, section=J["scores"])
    row("RMSE of the ensemble mean", lambda k: g(k, "scores", "model", "ug", "rmse_ensmean"), 3, section=J["scores"])
    row("RMSE of one member", lambda k: g(k, "scores", "model", "ug", "rmse_1member"), 3, section=J["scores"])
    row("MAE of the ensemble mean", lambda k: g(k, "scores", "model", "ug", "mae_ensmean"), 3, section=J["scores"])
    row("bias of the ensemble mean", lambda k: g(k, "scores", "model", "ug", "bias"), 3, section=J["scores"])
    row("pixel spread/skill (Fortin)", lambda k: g(k, "scores", "model", "ug", "spread_skill_fortin"), 4, section=J["scores"])
    row("pixel spread / RMSE (plain)", lambda k: g(k, "scores", "model", "ug", "spread_skill"), 4, section=J["scores"])
    row("share of truths outside the ensemble", lambda k: g(k, "scores", "model", "ug", "rank_hist")[0] + g(k, "scores", "model", "ug", "rank_hist")[-1], 4, section=J["scores"])
    row("rank histogram, lowest bin", lambda k: g(k, "scores", "model", "ug", "rank_hist")[0], 4, section=J["scores"])
    row("rank histogram, highest bin", lambda k: g(k, "scores", "model", "ug", "rank_hist")[-1], 4, section=J["scores"])
    row("log1p fair CRPS", lambda k: g(k, "scores", "model", "log", "crps_fair"), 4, section=J["scores"])
    row("log1p RMSE of the ensemble mean", lambda k: g(k, "scores", "model", "log", "rmse_ensmean"), 4, section=J["scores"])
    row("log1p spread/skill (Fortin)", lambda k: g(k, "scores", "model", "log", "spread_skill_fortin"), 4, section=J["scores"])
    for b, lab in (("1", "0.25°"), ("2", "0.5°"), ("4", "1°")):
        row(f"{lab} block fair CRPS", lambda k, b=b: g(k, "blocks", "model", b, "ug", "crps_fair"), 3, section=J["scores"])
        row(f"{lab} block RMSE", lambda k, b=b: g(k, "blocks", "model", b, "ug", "rmse_ensmean"), 3, section=J["scores"])
        row(f"{lab} block spread/skill (Fortin)", lambda k, b=b: g(k, "blocks", "model", b, "ug", "spread_skill_fortin"), 4, section=J["scores"])
    row("share of the squared error in the 1° block means", lambda k: (g(k, "blocks", "model", "4", "ug", "rmse_ensmean") / g(k, "scores", "model", "ug", "rmse_ensmean")) ** 2, 4, section=J["scores"])
    thr = E[ev[0]]["scores"]["model"].get("thresholds_ug", {})
    for t in ("37.5", "75"):
        if t in thr:
            row(f"> {t} µg m⁻³: Brier score", lambda k, t=t: g(k, "scores", "model", "thresholds_ug", t, "brier"), 5, section=J["scores"])
            row(f"> {t} µg m⁻³: hit rate of the ensemble mean", lambda k, t=t: g(k, "scores", "model", "thresholds_ug", t, "pod_M"), 4, section=J["scores"])
            row(f"> {t} µg m⁻³: false-alarm ratio of the ensemble mean", lambda k, t=t: g(k, "scores", "model", "thresholds_ug", t, "far_M"), 4, section=J["scores"])
            row(f"> {t} µg m⁻³: SEDI of the ensemble mean", lambda k, t=t: g(k, "scores", "model", "thresholds_ug", t, "sedi_M"), 4, section=J["scores"])
    md.append("")
    md.append("Rank histograms (9 bins): " + "; ".join(f"seed {k}: " + ", ".join(f"{x:.3f}" for x in g(k, "scores", "model", "ug", "rank_hist")) for k in ev) + ".\n")
    J["rank_hist"] = {k: g(k, "scores", "model", "ug", "rank_hist") for k in ev}

    # baselines: identical across seeds by construction -> check of the evaluation itself
    md.append("### Baselines (independent of the trained model: a check that the three evaluations are the same evaluation)\n")
    md.append(hdr(["", *[f"seed {k}" for k in ev], "max |difference|"]))
    J["baselines"] = {}
    for n, lab in (("climatology_ens", "climatology ensemble"), ("climatology_median", "climatology median"), ("cams_bilinear", "CAMS bilinear")):
        for key, kl in (("crps_fair", "CRPS / MAE"), ("rmse_ensmean", "RMSE"), ("bias", "bias")):
            vals = [g(k, "scores", n, "ug", key) for k in ev]
            J["baselines"][f"{n}/{key}"] = {"values": dict(zip(ev, vals)), "max_abs_diff": float(max(vals) - min(vals))}
            md.append(f"| {lab}: {kl} | " + " | ".join(fmt(x, 4) for x in vals) + f" | {max(vals) - min(vals):.2e} |")
        vals = [g(k, "blocks", n, "4", "ug", "rmse_ensmean") for k in ev]
        md.append(f"| {lab}: 1° block RMSE | " + " | ".join(fmt(x, 4) for x in vals) + f" | {max(vals) - min(vals):.2e} |")
    vals = [g(k, "scores", "climatology_ens", "ug", "spread_skill_fortin") for k in ev]
    md.append("| climatology ensemble: pixel spread/skill | " + " | ".join(fmt(x, 4) for x in vals) + f" | {max(vals) - min(vals):.2e} |")
    md.append("")
    md.append("### Differences the fact sheet relies on (section 3.6), against the seed noise\n")
    md.append(hdr(["difference", *[f"seed {k}" for k in ev], "range", "s.d.", "smallest difference / s.d."]))
    J["contrasts"] = {}
    cons = [("CRPS: climatology ensemble − downscaler", lambda k: g(k, "scores", "climatology_ens", "ug", "crps_fair") - g(k, "scores", "model", "ug", "crps_fair")),
            ("CRPS: climatology-median MAE − downscaler", lambda k: g(k, "scores", "climatology_median", "ug", "crps_fair") - g(k, "scores", "model", "ug", "crps_fair")),
            ("CRPS: CAMS-bilinear MAE − downscaler", lambda k: g(k, "scores", "cams_bilinear", "ug", "crps_fair") - g(k, "scores", "model", "ug", "crps_fair")),
            ("RMSE: climatology median − downscaler", lambda k: g(k, "scores", "climatology_median", "ug", "rmse_ensmean") - g(k, "scores", "model", "ug", "rmse_ensmean")),
            ("RMSE: climatology ensemble mean − downscaler", lambda k: g(k, "scores", "climatology_ens", "ug", "rmse_ensmean") - g(k, "scores", "model", "ug", "rmse_ensmean")),
            ("RMSE: CAMS bilinear − downscaler", lambda k: g(k, "scores", "cams_bilinear", "ug", "rmse_ensmean") - g(k, "scores", "model", "ug", "rmse_ensmean")),
            ("1° block RMSE: climatology median − downscaler", lambda k: g(k, "blocks", "climatology_median", "4", "ug", "rmse_ensmean") - g(k, "blocks", "model", "4", "ug", "rmse_ensmean")),
            ("1° block RMSE: CAMS bilinear − downscaler", lambda k: g(k, "blocks", "cams_bilinear", "4", "ug", "rmse_ensmean") - g(k, "blocks", "model", "4", "ug", "rmse_ensmean"))]
    for name, fn in cons:
        vals = [fn(k) for k in ev]
        st = stats(vals)
        ratio = (min(vals) / st["sd"]) if st.get("sd") else None
        J["contrasts"][name] = {"values": dict(zip(ev, vals)), "stats": st, "min_over_sd": ratio}
        md.append(f"| {name} | " + " | ".join(fmt(x, 3) for x in vals) + f" | {fmt(st.get('range'), 4)} | {fmt(st.get('sd'), 4)} | {fmt(ratio, 0)} |")
    md.append("")

    # ---------------------------------------------------------------- 4. spectra
    md.append("## 4. Spectra (log1p fields, six 512 km boxes, 8 of the 24 days): power relative to GHAP\n")
    md.append(H)
    J["psd"] = {}
    row("member 0 / GHAP, geometric mean over 10-140 km (G2 statistic)", lambda k: psd_ratio(S[k]["psd"]), 4, section=J["psd"])
    row("ensemble mean / GHAP, geometric mean over 10-140 km", lambda k: psd_ratio(S[k]["psd"], "ensmean"), 4, section=J["psd"])
    row("member 0 / GHAP, geometric mean over 2-10 km", lambda k: psd_ratio(S[k]["psd"], "member0", 2.0, 10.0), 4, section=J["psd"])
    row("member 0 / GHAP, geometric mean over 140-500 km", lambda k: psd_ratio(S[k]["psd"], "member0", 140.0, 500.0), 4, section=J["psd"])
    lam = 1.0 / S[ev[0]]["psd"]["k"]
    for wl in WL_TABLE:
        i = int(np.argmin(np.abs(lam - wl)))
        row(f"member 0 / GHAP at {lam[i]:.3g} km", lambda k, i=i: float(S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("member0")][i] / S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("truth")][i]), 3, section=J["psd"])
    for wl in WL_TABLE:
        i = int(np.argmin(np.abs(lam - wl)))
        row(f"ensemble mean / GHAP at {lam[i]:.3g} km", lambda k, i=i: float(S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("ensmean")][i] / S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("truth")][i]), 3, section=J["psd"])
    tr = [S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("truth")] for k in ev]
    cm = [S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("cams_bilinear")] for k in ev]
    J["psd"]["truth_and_cams_identical_across_seeds"] = {"truth_max_rel_diff": float(np.nanmax(np.abs(np.array(tr) / tr[0] - 1))),
                                                         "cams_max_rel_diff": float(np.nanmax(np.abs(np.array(cm) / cm[0] - 1)))}
    r_all = np.array([S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("member0")] / S[k]["psd"]["psd"][S[k]["psd"]["fields"].index("truth")] for k in ev])
    J["psd"]["member0_ratio_by_wavelength"] = {"wavelength_km": lam.tolist(), **{str(k): r_all[i].tolist() for i, k in enumerate(ev)}}
    band = (lam >= 2.0) & (lam <= 120.0)
    md.append(f"\nMember-0 ratio between 2 and 120 km: " + "; ".join(f"seed {k}: {np.nanmin(r_all[i][band]):.2f}-{np.nanmax(r_all[i][band]):.2f}" for i, k in enumerate(ev))
              + f". GHAP and CAMS spectra across the seeds: max relative difference {J['psd']['truth_and_cams_identical_across_seeds']['truth_max_rel_diff']:.1e} and "
              f"{J['psd']['truth_and_cams_identical_across_seeds']['cams_max_rel_diff']:.1e} (same boxes, same days).\n")

    # ---------------------------------------------------------------- 5. sanity counts
    md.append("## 5. Extreme members (guard counters), 24 days × 147,289,344 pixels × 8 members\n")
    md.append(hdr(["", *[f"seed {k}" for k in ev], "max / min"]))
    J["sanity"] = {}
    for key, lab in (("n_pix_gt_440", "pixel-days with a member above T_G = 986 µg m⁻³"), ("n_member_at_bound", "member points at the state bound |z| = 10"),
                     ("n_member_at_pmax", "member points at the cap 1,000 µg m⁻³"), ("n_member_z0_trunc", "member points with truncated noise (|z0| > 3)"),
                     ("n_cells_with_pix_gt_440", "0.25° cells with a pixel-day above T_G"), ("max_member_ug", "largest member (µg m⁻³)"),
                     ("frac_pix_days_any_member_gt_440", "G0a rate (pixel-days above T_G)"), ("frac_member_pixel_days_guard_bound", "G0b rate (member points at bound or cap)")):
        vals = [E[k]["sanity"][key] for k in ev]
        J["sanity"][key] = dict(zip(ev, vals))
        mm = (max(vals) / min(vals)) if min(vals) > 0 else None
        md.append(f"| {lab} | " + " | ".join((f"{x:,.0f}" if x >= 1 or x == 0 else f"{x:.3g}") for x in vals) + f" | {fmt(mm, 2)} |")
    md.append("")
    dates = [r["date"] for r in E[ev[0]]["daily"]]
    md.append("By day (only days with a count in one seed): bound hits / cap hits / pixel-days above T_G\n")
    md.append(hdr(["day", *[f"seed {k}: bound / cap / > T_G" for k in ev]]))
    J["sanity_by_day"] = {}
    for i, d in enumerate(dates):
        cells = []
        anyc = False
        for k in ev:
            sn = E[k]["daily"][i]["sanity"]
            assert E[k]["daily"][i]["date"] == d
            c = (sn["n_member_at_bound"], sn["n_member_at_pmax"], sn["n_pix_gt_440"])
            anyc |= any(x > 0 for x in c)
            cells.append(c)
        J["sanity_by_day"][d] = {k: c for k, c in zip(ev, cells)}
        if anyc:
            md.append(f"| {d} | " + " | ".join(f"{c[0]:,.0f} / {c[1]:,.0f} / {c[2]:,.0f}" for c in cells) + " |")
    md.append("")
    # where the members above T_G are: 0.25 deg cells (encoder grid 576 x 1440, lat 84 - 0.25 r, lon -180 + 0.25 c)
    cellsets = {}
    J["extreme_cells"] = {}
    for k in ev:
        cc = S[k]["ext_cells"]
        rr, co = np.nonzero(cc > 0)
        cellsets[k] = set(zip(rr.tolist(), co.tolist()))
        boxes = {}
        for r_, c_ in zip(rr, co):
            key = (int(round((84.0 - 0.25 * r_) / 5.0)) * 5, int(round((-180.0 + 0.25 * c_) / 5.0)) * 5)
            boxes[key] = boxes.get(key, 0) + float(cc[r_, c_])
        J["extreme_cells"][k] = {"n_cells": len(cellsets[k]), "by_5deg_box_latlon": {f"{a},{b}": v for (a, b), v in sorted(boxes.items(), key=lambda x: -x[1])}}
    common = set.intersection(*cellsets.values()) if cellsets else set()
    union = set.union(*cellsets.values()) if cellsets else set()
    J["extreme_cells"]["n_common_all_seeds"] = len(common)
    J["extreme_cells"]["n_union"] = len(union)
    md.append("Cells of 0.25° with a pixel-day above T_G: " + "; ".join(f"seed {k}: {len(cellsets[k])}" for k in ev)
              + f"; in all seeds: {len(common)}; in at least one: {len(union)}. Pixel-days above T_G by 5° box (lat, lon of the box centre): "
              + " | ".join(f"seed {k}: " + ", ".join(f"({kk}) {v:.0f}" for kk, v in list(J['extreme_cells'][k]['by_5deg_box_latlon'].items())[:6]) for k in ev) + ".\n")

    # ---------------------------------------------------------------- 6. gates
    md.append("## 6. Validation gates (same script, same thresholds)\n")
    gd = [("G0a", "pixel-days above T_G ≤ 1e-6", lambda k: E[k]["sanity"]["frac_pix_days_any_member_gt_440"], 1e-6, "le", True),
          ("G0b", "member points at bound or cap ≤ 1e-7", lambda k: E[k]["sanity"]["frac_member_pixel_days_guard_bound"], 1e-7, "le", True),
          ("G0c", "RMSE < climatology-median RMSE", lambda k: g(k, "scores", "model", "ug", "rmse_ensmean"), None, "lt_ref", False),
          ("G0d", "pixel spread/skill ≥ 0.5 (and ≤ 1.5)", lambda k: g(k, "scores", "model", "ug", "spread_skill_fortin"), 0.5, "ge", False),
          ("G0e", "1° block spread/skill ≤ 1.5", lambda k: g(k, "blocks", "model", "4", "ug", "spread_skill_fortin"), 1.5, "le", False),
          ("G1", "1° block spread/skill ≥ 0.6", lambda k: g(k, "blocks", "model", "4", "ug", "spread_skill_fortin"), 0.6, "ge", False),
          ("G2", "member-0 / GHAP power, 10-140 km ≤ 2", lambda k: psd_ratio(S[k]["psd"]), 2.0, "le", False),
          ("G3", "CRPS < climatology-ensemble CRPS", lambda k: g(k, "scores", "model", "ug", "crps_fair"), None, "lt_ref", False),
          ("G4", "CRPS < CAMS-bilinear MAE", lambda k: g(k, "scores", "model", "ug", "crps_fair"), None, "lt_ref", False)]
    refs = {"G0c": lambda k: g(k, "scores", "climatology_median", "ug", "rmse_ensmean"),
            "G3": lambda k: g(k, "scores", "climatology_ens", "ug", "crps_fair"), "G4": lambda k: g(k, "scores", "cams_bilinear", "ug", "crps_fair")}
    md.append(hdr(["gate", "rule", *[f"seed {k}" for k in ev], "threshold", "result by seed", "range", "frozen margin", "threshold inside the PI for a new seed", "reading (A8(g))"]))
    J["gates"] = {}
    for name, rule, fn, thr_, kind, log in gd:
        vals = [fn(k) for k in ev]
        th = [refs[name](k) if kind == "lt_ref" else thr_ for k in ev]
        if kind in ("le",):
            ok = [bool(v <= t) for v, t in zip(vals, th)]
        elif kind == "ge":
            ok = [bool(v >= t) for v, t in zip(vals, th)]
        else:
            ok = [bool(v < t) for v, t in zip(vals, th)]
        if name == "G0d":
            ok = [bool(0.5 <= v <= 1.5) for v in vals]
        st = stats(vals, log=log)
        v0, t0 = vals[0], th[0]
        if log:
            margin = abs(math.log10(max(v0, 1e-300)) - math.log10(t0))            # in decades
            rng = math.log10(max(max(vals), 1e-300)) - math.log10(max(min(vals), 1e-300))
            mtxt = f"factor {10 ** margin:.2f}"
            rtxt = f"factor {10 ** rng:.2f}"
        else:
            margin = abs(v0 - t0)
            rng = st["range"]
            mtxt, rtxt = fmt(margin, 4), fmt(rng, 4)
        pi = st.get("pi95_new_seed")
        inside = None if pi is None else bool(pi[0] <= th[0] <= pi[1])
        same_side = len(set(ok)) == 1
        reading = ("property of the recipe" if same_side else "seed-dependent") + "; margin " + \
                  ("inside the seed noise" if (rng >= margin or not same_side) else "larger than the range of three seeds")
        J["gates"][name] = {"rule": rule, "values": dict(zip(ev, vals)), "threshold": dict(zip(ev, th)), "pass": dict(zip(ev, ok)),
                            "stats": st, "frozen_margin": margin, "range": rng, "log10": log, "threshold_inside_pi": inside,
                            "same_side": same_side, "reading": reading}
        md.append(f"| {name} | {rule} | " + " | ".join(fmt(x, 4) for x in vals) + f" | {fmt(th[0], 4)} | "
                  + " / ".join("pass" if o else "FAIL" for o in ok) + f" | {rtxt} | {mtxt} | {fmt(inside)} | {reading} |")
    gf = {k: S[k].get("gates_file") for k in ev}
    J["gates_files_passed"] = {k: (v or {}).get("passed") for k, v in gf.items()}
    J["gates_files_failed"] = {k: [n for n, x in (v or {}).get("gates", {}).items() if x.get("pass") is False] for k, v in gf.items()}
    md.append("\nOverall (gate files): " + "; ".join(f"seed {k}: {'passed' if J['gates_files_passed'][k] else 'FAILED ' + ', '.join(J['gates_files_failed'][k])}" for k in ev) + ".\n")

    # ---------------------------------------------------------------- 7. per-day scores: day effect against seed effect
    md.append("## 7. Day by day: what is the day, what is the seed\n")
    J["daily"] = {}
    for key, lab, getter in (("crps", "pixel fair CRPS", lambda r: r["model"]["crps_fair"]),
                             ("rmse", "RMSE of the ensemble mean", lambda r: r["model"]["rmse_ensmean"]),
                             ("ssF", "pixel spread/skill", lambda r: r["model"]["spread_skill_fortin"]),
                             ("ssF_1deg", "1° block spread/skill", lambda r: r["model"]["blocks"]["4"]["ug"]["spread_skill_fortin"]),
                             ("bias", "bias", lambda r: r["model"]["bias"])):
        A = np.array([[getter(r) for r in E[k]["daily"]] for k in ev])            # (seeds, days)
        day_sd = float(A.mean(0).std(ddof=1))
        seed_sd_within_day = float(np.sqrt((A.var(0, ddof=1)).mean())) if len(ev) > 1 else None
        cor = [float(np.corrcoef(A[i], A[j])[0, 1]) for i in range(len(ev)) for j in range(i + 1, len(ev))]
        J["daily"][key] = {"dates": dates, **{str(k): A[i].tolist() for i, k in enumerate(ev)}, "sd_between_days_of_seed_mean": day_sd,
                           "rms_seed_sd_within_day": seed_sd_within_day, "corr_between_seeds": cor,
                           "max_abs_seed_diff_by_day": (A.max(0) - A.min(0)).tolist()}
        md.append(f"- {lab}: s.d. between days {fmt(day_sd, 3)}; r.m.s. s.d. between seeds on a given day {fmt(seed_sd_within_day, 4)}; "
                  f"correlation of the 24 daily values between seeds {', '.join(f'{c:.3f}' for c in cor)}; largest seed range on one day "
                  f"{fmt(float((A.max(0) - A.min(0)).max()), 3)} ({dates[int((A.max(0) - A.min(0)).argmax())]}).")
    md.append("")
    md.append(hdr(["day", *[f"CRPS s{k}" for k in ev], *[f"RMSE s{k}" for k in ev], *[f"1° ssF s{k}" for k in ev]]))
    for i, d in enumerate(dates):
        md.append(f"| {d} | " + " | ".join(f"{J['daily']['crps'][str(k)][i]:.3f}" for k in ev) + " | "
                  + " | ".join(f"{J['daily']['rmse'][str(k)][i]:.2f}" for k in ev) + " | "
                  + " | ".join(f"{J['daily']['ssF_1deg'][str(k)][i]:.3f}" for k in ev) + " |")
    md.append("")
    # ---------------------------------------------------------------- 8. day-sampling noise next to the seed noise
    # The pooled scores are exact functions of the 24 daily rows with equal day weights (checked below), so the 24 days
    # can be resampled: this is the noise of evaluating on THESE 24 days, a different thing from the seed noise.
    M = E[ev[0]]["members"]
    rng = np.random.default_rng(20261119)
    B = 20000
    nd = len(dates)
    idx = rng.integers(0, nd, (B, nd))

    def comp(k, arm="model", blk=None):
        rows = E[k]["daily"]
        get = (lambda r: r[arm]) if blk is None else (lambda r: r[arm]["blocks"][blk]["ug"])
        rm = np.array([get(r)["rmse_ensmean"] for r in rows])
        cr = np.array([get(r)["crps_fair"] for r in rows])
        ssf = np.array([get(r).get("spread_skill_fortin") or np.nan for r in rows], float)
        V = (ssf * rm) ** 2 * M / (M + 1.0)
        return cr, rm ** 2, V

    def pooled(cr, se, V, ix=None):
        if ix is None:
            return float(cr.mean()), float(np.sqrt(se.mean())), float(np.sqrt((M + 1.0) / M * V.mean()) / np.sqrt(se.mean()))
        return cr[ix].mean(1), np.sqrt(se[ix].mean(1)), np.sqrt((M + 1.0) / M * V[ix].mean(1)) / np.sqrt(se[ix].mean(1))

    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    md.append("## 8. Noise of the 24-day sample (bootstrap of the days), next to the seed noise\n")
    md.append("The pooled scores are exact functions of the 24 daily rows (equal day weights; reconstruction error below). "
              f"Days resampled with replacement, {B:,} draws, the same draws for every seed (paired). This is the uncertainty of a "
              "statistic computed on these 24 days of 2021; the seed range of sections 3-6 is a different, additional source.\n")
    J["day_bootstrap"] = {"B": B, "reconstruction_error": {}, "per_seed": {}, "pairs": {}}
    md.append(hdr(["statistic", *[f"seed {k}: value [95 % interval over days]" for k in ev], "seed range (sections 3-6)"]))
    C = {k: {"pix": comp(k), "b4": comp(k, blk="4"), "clim_ens": comp(k, "climatology_ens"), "clim_med": comp(k, "climatology_median")} for k in ev}
    npd = np.array([r["sanity"]["n_pix"] for r in E[ev[0]]["daily"]])
    cnt = {k: {key: np.array([r["sanity"][key] for r in E[k]["daily"]]) for key in ("n_pix_gt_440", "n_member_at_bound", "n_member_at_pmax")} for k in ev}
    boot = {}
    for k in ev:
        c0, r0, s0 = pooled(*C[k]["pix"])
        J["day_bootstrap"]["reconstruction_error"][k] = {
            "crps": c0 - E[k]["scores"]["model"]["ug"]["crps_fair"], "rmse": r0 - E[k]["scores"]["model"]["ug"]["rmse_ensmean"],
            "ssF": s0 - E[k]["scores"]["model"]["ug"]["spread_skill_fortin"],
            "ssF_1deg": pooled(*C[k]["b4"])[2] - E[k]["blocks"]["model"]["4"]["ug"]["spread_skill_fortin"]}
        cb, rb, sb = pooled(*C[k]["pix"], idx)
        _, r4, s4 = pooled(*C[k]["b4"], idx)
        ce, _, _ = pooled(*C[k]["clim_ens"], idx)
        _, rmed, _ = pooled(*C[k]["clim_med"], idx)
        g0a = cnt[k]["n_pix_gt_440"][idx].sum(1) / npd[idx].sum(1)
        g0b = (cnt[k]["n_member_at_bound"][idx].sum(1) + cnt[k]["n_member_at_pmax"][idx].sum(1)) / (npd[idx].sum(1) * M)
        boot[k] = dict(crps=cb, rmse=rb, ssF=sb, rmse4=r4, ssF4=s4, g3=ce - cb, g0c=rmed - rb, g0a=g0a, g0b=g0b)
        J["day_bootstrap"]["per_seed"][k] = {
            "crps_ci": ci(cb), "rmse_ci": ci(rb), "ssF_ci": ci(sb), "ssF_1deg_ci": ci(s4), "rmse_1deg_ci": ci(r4),
            "G1_share_of_resamples_below_0.6": float((s4 < 0.6).mean()), "G0d_share_below_0.5": float((sb < 0.5).mean()),
            "G3_margin_ci": ci(ce - cb), "G0c_margin_ci": ci(rmed - rb),
            "G0a_rate_ci": ci(g0a), "G0a_share_of_resamples_failing": float((g0a > 1e-6).mean()),
            "G0b_rate_ci": ci(g0b), "G0b_share_of_resamples_failing": float((g0b > 1e-7).mean())}
    pe = J["day_bootstrap"]["per_seed"]
    vv = lambda k, key, d=3: f"{fmt(J['scores'][key]['values'][k], d)} [{fmt(pe[k][{'pixel fair CRPS (µg m⁻³)': 'crps_ci', 'RMSE of the ensemble mean': 'rmse_ci', 'pixel spread/skill (Fortin)': 'ssF_ci', '1° block spread/skill (Fortin)': 'ssF_1deg_ci', '1° block RMSE': 'rmse_1deg_ci'}[key]][0], d)}, {fmt(pe[k][{'pixel fair CRPS (µg m⁻³)': 'crps_ci', 'RMSE of the ensemble mean': 'rmse_ci', 'pixel spread/skill (Fortin)': 'ssF_ci', '1° block spread/skill (Fortin)': 'ssF_1deg_ci', '1° block RMSE': 'rmse_1deg_ci'}[key]][1], d)}]"
    for key, d in (("pixel fair CRPS (µg m⁻³)", 3), ("RMSE of the ensemble mean", 2), ("pixel spread/skill (Fortin)", 3),
                   ("1° block RMSE", 2), ("1° block spread/skill (Fortin)", 3)):
        md.append(f"| {key} | " + " | ".join(vv(k, key, d) for k in ev) + f" | {fmt(J['scores'][key]['stats'].get('range'), d + 1)} |")
    md.append("| G1: share of day resamples with 1° block spread/skill < 0.6 | " + " | ".join(f"{100 * pe[k]['G1_share_of_resamples_below_0.6']:.0f} %" for k in ev) + " | |")
    md.append("| G0a rate | " + " | ".join(f"{E[k]['sanity']['frac_pix_days_any_member_gt_440']:.2e} [{pe[k]['G0a_rate_ci'][0]:.1e}, {pe[k]['G0a_rate_ci'][1]:.1e}]" for k in ev) + " | |")
    md.append("| G0a: share of day resamples above 1e-6 | " + " | ".join(f"{100 * pe[k]['G0a_share_of_resamples_failing']:.0f} %" for k in ev) + " | |")
    md.append("| G0b rate | " + " | ".join(f"{E[k]['sanity']['frac_member_pixel_days_guard_bound']:.2e} [{pe[k]['G0b_rate_ci'][0]:.1e}, {pe[k]['G0b_rate_ci'][1]:.1e}]" for k in ev) + " | |")
    md.append("| G0b: share of day resamples above 1e-7 | " + " | ".join(f"{100 * pe[k]['G0b_share_of_resamples_failing']:.0f} %" for k in ev) + " | |")
    md.append("| G3 margin (climatology-ensemble CRPS − CRPS) | " + " | ".join(f"[{pe[k]['G3_margin_ci'][0]:.3f}, {pe[k]['G3_margin_ci'][1]:.3f}]" for k in ev) + " | |")
    md.append("| G0c margin (climatology-median RMSE − RMSE) | " + " | ".join(f"[{pe[k]['G0c_margin_ci'][0]:.3f}, {pe[k]['G0c_margin_ci'][1]:.3f}]" for k in ev) + " | |")
    md.append("")
    if len(ev) > 1:
        md.append("Paired differences between seeds (same resampled days): is a seed difference distinguishable from the choice of days?\n")
        md.append(hdr(["pair", "Δ pixel CRPS [95 %]", "Δ RMSE [95 %]", "Δ pixel spread/skill [95 %]", "Δ 1° block spread/skill [95 %]"]))
        for i_, a_ in enumerate(ev):
            for b_ in ev[i_ + 1:]:
                row_ = {}
                cells = []
                for key in ("crps", "rmse", "ssF", "ssF4"):
                    dlt = boot[b_][key] - boot[a_][key]
                    pt = {"crps": J["scores"]["pixel fair CRPS (µg m⁻³)"]["values"], "rmse": J["scores"]["RMSE of the ensemble mean"]["values"],
                          "ssF": J["scores"]["pixel spread/skill (Fortin)"]["values"], "ssF4": J["scores"]["1° block spread/skill (Fortin)"]["values"]}[key]
                    row_[key] = {"diff": pt[b_] - pt[a_], "ci": ci(dlt), "excludes_zero": bool(ci(dlt)[0] > 0 or ci(dlt)[1] < 0)}
                    cells.append(f"{pt[b_] - pt[a_]:+.4f} [{ci(dlt)[0]:+.4f}, {ci(dlt)[1]:+.4f}]")
                J["day_bootstrap"]["pairs"][f"seed {b_} - seed {a_}"] = row_
                md.append(f"| seed {b_} − seed {a_} | " + " | ".join(cells) + " |")
        md.append("")

    # ---------------------------------------------------------------- 9. reference checks on the frozen system
    md.append("## 9. Reference checks (is the seed-0 column, computed on 2026-10-28, comparable with columns computed today?)\n")
    R = od
    J["reference_checks"] = {}
    f = R / "smoke_seed0.json"
    if f.exists():
        x = jload(f)
        J["reference_checks"]["smoke_seed0"] = x
        md.append("- Seed 0, 200 steps with the A8 launcher against the frozen training log: day draws of the 8 ranks at steps 0-2 identical: "
                  f"{fmt(x['days_equal'])}; loss at steps 10 / 100 / 200: " + " / ".join(f"{x['steps'][t]['smoke']['loss']:.4f}" for t in ("10", "100", "200"))
                  + " against " + " / ".join(f"{x['steps'][t]['frozen']['loss']:.4f}" for t in ("10", "100", "200"))
                  + f"; model line identical: {fmt(x['model_line_equal'])}"
                  + (f" (identical once the token `stations=off`, printed by today's code, is removed: {fmt(x.get('model_line_equal_without_stations_token'))})" if 'model_line_equal_without_stations_token' in x else "")
                  + f" (`{f}`).")
    f = R / "valrepro_seed0.json"
    if f.exists():
        x = jload(f)
        J["reference_checks"]["valrepro"] = {st: {"diff": r["diff"]} for st, r in x["steps"].items()}
        md.append("- In-training validation of the frozen checkpoint recomputed today: "
                  + "; ".join(f"step {st}: CRPS {r['recomputed']['ug']['crps_fair']:.6f} (stored {r['stored']['ug']['crps_fair']:.6f}), RMSE {r['recomputed']['ug']['rmse_ensmean']:.6f} (stored {r['stored']['ug']['rmse_ensmean']:.6f})" for st, r in x["steps"].items())
                  + f" (`{f}`).")
    f = R / "psdrepro" / "psdrepro_compare.json"
    if f.exists():
        x = jload(f)
        J["reference_checks"]["psdrepro"] = {k: x[k] for k in ("fields", "G2_recomputed", "G2_stored", "G2_rel_diff", "pass_A8e2ii")}
        md.append(f"- Spectra of the frozen system recomputed today at the box pixels (8 days): member 0 median relative difference "
                  f"{x['fields']['member0']['median_rel_diff']:.1e} (max {x['fields']['member0']['max_rel_diff']:.1e}), ensemble mean "
                  f"{x['fields']['ensmean']['median_rel_diff']:.1e}; G2 {x['G2_recomputed']:.6f} against {x['G2_stored']:.6f} (`{f}`).")
    f = R / "refday_compare.json"
    if f.exists():
        x = jload(f)
        J["reference_checks"]["refday"] = {k: {kk: v[kk] for kk in ("eval_noise_seed", "what", "max_rel_diff_model_scores", "max_rel_diff_baseline_scores", "sanity_counts_equal", "minutes")} for k, v in x["days"].items()}
        for k, v in x["days"].items():
            rr = v["rows"]
            md.append(f"- Frozen system, full day {v['date']} re-evaluated today with the unmodified eval_x, evaluation noise seed {v['eval_noise_seed']} "
                      f"({v['what']}): CRPS {rr['model/crps_fair']['recomputed']:.6f} (stored {rr['model/crps_fair']['stored']:.6f}), RMSE "
                      f"{rr['model/rmse_ensmean']['recomputed']:.6f} ({rr['model/rmse_ensmean']['stored']:.6f}), pixel spread/skill "
                      f"{rr['model/spread_skill_fortin']['recomputed']:.6f} ({rr['model/spread_skill_fortin']['stored']:.6f}), 1° block spread/skill "
                      f"{rr['model/blocks4/spread_skill_fortin']['recomputed']:.6f} ({rr['model/blocks4/spread_skill_fortin']['stored']:.6f}); member points at the bound "
                      f"{rr['sanity/n_member_at_bound']['recomputed']:,.0f} ({rr['sanity/n_member_at_bound']['stored']:,.0f}), at the cap "
                      f"{rr['sanity/n_member_at_pmax']['recomputed']:,.0f} ({rr['sanity/n_member_at_pmax']['stored']:,.0f}), pixel-days above T_G "
                      f"{rr['sanity/n_pix_gt_440']['recomputed']:,.0f} ({rr['sanity/n_pix_gt_440']['stored']:,.0f}), truncated noise points "
                      f"{rr['sanity/n_member_z0_trunc']['recomputed']:,.0f} ({rr['sanity/n_member_z0_trunc']['stored']:,.0f}); largest relative difference of a model score "
                      f"{v['max_rel_diff_model_scores']:.1e}, of a baseline score {v['max_rel_diff_baseline_scores']:.1e} (`{v['file']}`).")
    for f in sorted(R.glob("refday_hits_day*.json")):
        x = jload(f)
        J["reference_checks"][f.stem] = x
        if x.get("n_member_points_at_bound", 0) > 0:
            md.append(f"- Bound hits of the frozen system on {x['date']} (evaluation noise seed {x.get('eval_noise_seed', 0)}): {x['n_member_points_at_bound']:,} member points on "
                      f"{x['n_pixels']:,} pixels in {x['n_cells_025']} cells of 0.25°; at z = +10: {x['n_at_plus_bound']:,}, at z = −10: {x['n_at_minus_bound']:,}; "
                      f"member value there (µg m⁻³, min / median / max): {x['member_value_ug_percentiles'][0]:.3g} / {x['member_value_ug_percentiles'][3]:.3g} / {x['member_value_ug_percentiles'][6]:.3g}; "
                      f"GHAP there: {x['truth_ug_percentiles'][0]:.0f} / {x['truth_ug_percentiles'][3]:.0f} / {x['truth_ug_percentiles'][6]:.0f}; latitude {x['lat_percentiles'][0]:.1f} to {x['lat_percentiles'][6]:.1f} "
                      f"(median {x['lat_percentiles'][3]:.1f}), longitude {x['lon_percentiles'][0]:.1f} to {x['lon_percentiles'][6]:.1f}; share of starting noise at the truncation "
                      f"|z0| = 3: {100 * x['share_start_noise_at_truncation_3']:.0f} %; 5° boxes (lat,lon: count): "
                      + ", ".join(f"({kk}): {vv_}" for kk, vv_ in list(x['by_5deg_box_latlon'].items())[:5]) + f" (`{f}`).")
        else:
            md.append(f"- Bound hits of the frozen system on {x['date']} (evaluation noise seed {x.get('eval_noise_seed', 0)}): none (`{f}`).")
    f = R / "valnoise_seed0.json"
    if f.exists():
        x = jload(f)
        tab = {}
        for st, r in x["steps"].items():
            vals = [r["recomputed"]["ug"]["crps_fair"]] + [v["crps_fair"] for v in r.get("val_noise_seeds", {}).values()]
            tab[st] = {"crps_by_val_noise_seed": vals, "sd": float(np.std(vals, ddof=1)), "range": float(max(vals) - min(vals))}
        J["reference_checks"]["valnoise"] = tab
        sds = [t["sd"] for t in tab.values()]
        md.append("- Sampling noise of the selection metric (frozen checkpoints re-validated with validation-noise seeds 0, 1, 2, 3; same pixels and days): "
                  + "; ".join(f"step {st}: CRPS " + ", ".join(f"{v:.4f}" for v in t["crps_by_val_noise_seed"]) + f" (s.d. {t['sd']:.4f})" for st, t in tab.items())
                  + f". Pooled s.d. {float(np.sqrt(np.mean(np.square(sds)))):.4f} µg m⁻³ (`{f}`).")
        J["reference_checks"]["valnoise_pooled_sd"] = float(np.sqrt(np.mean(np.square(sds))))
        # does the step chosen by the rule depend on the validation noise draw?
        seeds_ = ["0"] + list(next(iter(x["steps"].values())).get("val_noise_seeds", {}).keys())
        best = {}
        for j, vs in enumerate(seeds_):
            best[vs] = min(tab, key=lambda st: tab[st]["crps_by_val_noise_seed"][j])
        J["reference_checks"]["valnoise_best_step_by_noise_seed"] = best
        md.append("  Step with the lowest CRPS among those re-validated, by validation-noise seed: " + ", ".join(f"seed {vs}: {st}" for vs, st in best.items()) + ".")
    md.append("")
    md.append("## Files\n")
    for k in S:
        md.append(f"- seed {k}: " + "; ".join(f"`{v}`" for v in S[k]["files"].values()))
    (od / "seed_compare.md").write_text("\n".join(md) + "\n")
    (od / "seed_compare.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
