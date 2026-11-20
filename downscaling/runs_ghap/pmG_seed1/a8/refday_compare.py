#!/usr/bin/env python3
"""A8 reference check (added to (e)-2, see STATUS.md): full selection days of the FROZEN system re-evaluated today with
the unmodified evaluation script (cdffm_aq_x.eval_x, same options, one day per process, all 147,289,344 mask pixels),
against the per-day rows stored by the frozen evaluation of 2026-10-28
(runs_ghap/pmG_base/eval_sel2021_M8_s10_P3_guard_T3/metrics.json, key 'daily').
The re-evaluations are written under runs_ghap/pmG_seed1/a8/ref/pmG_base/eval_ref_day<t>/ (a view of pmG_base made of
symbolic links: nothing is written in pmG_base). stdlib only. Writes a8/results/refday_compare.json.

  python refday_compare.py"""
import glob
import json
from pathlib import Path

RG = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap")
A8 = RG / "pmG_seed1" / "a8"
FROZEN = RG / "pmG_base" / "eval_sel2021_M8_s10_P3_guard_T3" / "metrics.json"
SAN = ("n_pix", "n_pix_gt_440", "n_pix_gt_1000", "n_member_gt_1000", "n_member_at_bound", "n_member_at_pmax", "n_member_z0_trunc",
       "max_member_ug")


def flat(row):
    out = {f"sanity/{k}": row["sanity"][k] for k in SAN}
    for n in ("model", "climatology_median", "climatology_ens", "cams_bilinear"):
        for k in ("crps_fair", "rmse_ensmean", "bias", "crps_fair_log", "spread_skill_fortin"):
            if row[n].get(k) is not None:
                out[f"{n}/{k}"] = row[n][k]
        for s in ("1", "2", "4"):
            for k, v in row[n]["blocks"][s]["ug"].items():
                if v is not None:
                    out[f"{n}/blocks{s}/{k}"] = v
    return out


def main():
    stored = {r["date"]: r for r in json.loads(FROZEN.read_text())["daily"]}
    res = {"frozen": str(FROZEN), "days": {}}
    for f in sorted(glob.glob(str(A8 / "ref" / "pmG_base" / "eval_ref_day*" / "metrics.json"))):
        m = json.loads(Path(f).read_text())
        assert m["step"] == 22000 and m["pixel_stride"] == 1 and m["n_pixels_per_day"] == 147289344, (f, m["step"])
        tag = Path(f).parent.name[len("eval_"):]
        es = int(tag.split("_es")[1]) if "_es" in tag else 0          # evaluation-noise seed (0 = the frozen evaluation's)
        for r in m["daily"]:
            a, b = flat(r), flat(stored[r["date"]])
            rows = {}
            for k in b:
                d = a[k] - b[k]
                rows[k] = {"stored": b[k], "recomputed": a[k], "diff": d, "rel": (d / b[k]) if b[k] else None}
            model = {k: v for k, v in rows.items() if k.startswith("model/")}
            base = {k: v for k, v in rows.items() if not k.startswith("model/") and not k.startswith("sanity/")}
            mx = lambda dd: max((abs(v["rel"]) for v in dd.values() if v["rel"] is not None), default=0.0)
            res["days"][f"{r['date']} eval-seed {es}"] = {"file": f, "date": r["date"], "eval_noise_seed": es,
                                      "what": ("reproduction of the stored row (same evaluation noise)" if es == 0 else
                                               "same model and day under another evaluation noise draw (differences = evaluation sampling noise)"),
                                      "minutes": m["minutes"], "guard": m["guard"], "prior_file": m["prior_file"],
                                      "rows": rows, "max_rel_diff_model_scores": mx(model), "max_rel_diff_baseline_scores": mx(base),
                                      "sanity_counts_equal": all(rows[f"sanity/{k}"]["diff"] == 0 for k in SAN)}
            s = rows
            key = f"{r['date']} eval-seed {es}"
            print(f"{r['date']} [evaluation noise seed {es}]: model CRPS {s['model/crps_fair']['recomputed']:.6f} (stored {s['model/crps_fair']['stored']:.6f}), RMSE "
                  f"{s['model/rmse_ensmean']['recomputed']:.6f} ({s['model/rmse_ensmean']['stored']:.6f}), 1-deg block ssF "
                  f"{s['model/blocks4/spread_skill_fortin']['recomputed']:.6f} ({s['model/blocks4/spread_skill_fortin']['stored']:.6f}); "
                  f"bound {s['sanity/n_member_at_bound']['recomputed']:.0f} ({s['sanity/n_member_at_bound']['stored']:.0f}), cap "
                  f"{s['sanity/n_member_at_pmax']['recomputed']:.0f} ({s['sanity/n_member_at_pmax']['stored']:.0f}), >T_G "
                  f"{s['sanity/n_pix_gt_440']['recomputed']:.0f} ({s['sanity/n_pix_gt_440']['stored']:.0f}), truncated "
                  f"{s['sanity/n_member_z0_trunc']['recomputed']:.0f} ({s['sanity/n_member_z0_trunc']['stored']:.0f}); max rel diff model "
                  f"scores {res['days'][key]['max_rel_diff_model_scores']:.2e}, baselines {res['days'][key]['max_rel_diff_baseline_scores']:.2e}; "
                  f"{m['minutes']:.1f} min")
    (A8 / "results" / "refday_compare.json").write_text(json.dumps(res, indent=1))
    print(f"{len(res['days'])} day(s) -> {A8 / 'results' / 'refday_compare.json'}")


if __name__ == "__main__":
    main()
