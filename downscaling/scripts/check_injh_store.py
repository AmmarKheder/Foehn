"""GlobalStore(fires=injh) check (global-prep): channels 0-1 identical to mami, channel 2 = injh, real batch finite."""
import datetime as dt, json, sys
import numpy as np, torch
sys.path.insert(0, "/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
from cdffm_aq.data_aq import open_store, BatchStream, PixelArrays, assemble_inputs
si = open_store("global", fires="injh"); sm = open_store("global", fires="mami")
out = {"fires_name": si.fires_name}
for d in (dt.date(2019, 7, 11), dt.date(2021, 8, 15), dt.date(2022, 12, 31), dt.date(2023, 1, 1), dt.date(2023, 8, 20)):
    a, b = si.raw_inputs(d)["fires"], sm.raw_inputs(d)["fires"]
    fire = a[1].astype(np.float32) > -0.5
    out[str(d)] = {"ch01_identical": bool(np.array_equal(a[:2].view(np.uint16), b[:2].view(np.uint16))),
                   "ch2_differs": bool((a[2] != b[2]).any()), "fire_cells": int(fire.sum()),
                   "corr_ch2_fire": float(np.corrcoef(a[2][fire].astype(np.float32), b[2][fire].astype(np.float32))[0, 1]) if fire.sum() > 10 else None,
                   "finite": bool(np.isfinite(a.astype(np.float32)).all())}
pix = PixelArrays(si, torch.device("cpu"))
bs = BatchStream(si, 2, 4096, 0, 0, torch.device("cpu")); b = bs.next(); bs.close()
x = assemble_inputs(b["raw"], pix)
out["batch"] = {"shape": list(x.shape), "finite": bool(torch.isfinite(x).all()), "ch40_mean_std": [float(x[:, 40].mean()), float(x[:, 40].std())]}
print(json.dumps(out, indent=1))
ok = all(v["ch01_identical"] and v["ch2_differs"] and v["finite"] for k, v in out.items() if k.startswith("20")) and out["batch"]["finite"]
print("INJH_STORE_CHECK", "PASS" if ok else "FAIL")
