#!/usr/bin/env python3
"""A7(g) check 3, diagnosis (2): per-day share of bit-identical same-day values against the frozen test, per member,
and the same against preds_smoke where available. Predictions only (no observation, no GHAP value)."""
import datetime as dt
import json
from pathlib import Path
import numpy as np
ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
F = ROOT / "test2022_ghap"
Y0 = dt.date(2022, 1, 1)
meta = np.load(O / "pos/st_meta_official.npz")
fidx = meta["frozen_idx"]
sh = fidx >= 0
S = np.load(O / "preds/sameday_ds.npy", mmap_mode="r")
rows = []
for t in range(365):
    d = Y0 + dt.timedelta(t)
    z = np.load(F / "preds_st" / "cdffm" / f"{d}.npz", allow_pickle=True)
    y = z["members"][:, fidx[sh]]
    x = np.asarray(S[t][:, sh, 0])
    idm = (x.view(np.uint16) == y.view(np.uint16)).mean(1)
    rel = np.abs(x.astype(np.float32) - y.astype(np.float32)) / np.maximum(y.astype(np.float32), 1.0)
    rows.append({"t": t, "date": str(d), "frac_identical_by_member": [round(float(v), 4) for v in idm],
                 "n_gt2steps_by_member": [int(v) for v in (rel > 0.0022).sum(1)], "max_rel_by_member": [round(float(v), 4) for v in rel.max(1)]})
    if t == 0:
        try:
            print("frozen meta of day 0:", z["meta"])
        except Exception as e:
            print("meta unreadable", e)
(O / "results/diag_check3_perday.json").write_text(json.dumps(rows))
low = [r for r in rows if min(r["frac_identical_by_member"]) < 0.998]
print("days with a member below 99.8 % identical:", len(low))
for r in low:
    print(r["t"], r["date"], r["frac_identical_by_member"], r["n_gt2steps_by_member"], r["max_rel_by_member"])
allm = np.array([r["frac_identical_by_member"] for r in rows])
print("member-days:", allm.size, "| >= 0.999:", int((allm >= 0.999).sum()), "| < 0.998:", int((allm < 0.998).sum()), "| < 0.9:", int((allm < 0.9).sum()),
      "| min:", allm.min())
print("quantiles of member-day identical share:", np.quantile(allm, [0, 0.001, 0.01, 0.05, 0.5]))
