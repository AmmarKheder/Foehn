#!/usr/bin/env python3
"""A7: create the prediction arrays of predict_official.py (numpy only; run once before the srun; existing files with the
right shape are kept, so a resubmission continues).
    python -I init_preds.py <out dir>"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
ND, NI, NL = 365, 214, 5
ARMS_S = {"sameday_ds": 8, "sameday_cams0": 1, "sameday_bcsd": 1, "sameday_climens": 8, "sameday_climmed": 1}
ARMS_C = {"chain": 8, "chain_t6fc": 8, "chain_rtseed": 8, "camsfc_ds": 8, "bcsd_fc": 8}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    posf = O / "pos/st_ring_official.npz"
    n_pix = int(np.load(posf)["mon_idx"].size)
    files = {name: (ND, m, n_pix, 4) for name, m in ARMS_S.items()}
    files.update({name: (NI, NL, m, n_pix, 4) for name, m in ARMS_C.items()})
    made = []
    for name, shape in files.items():
        p = out / f"{name}.npy"
        if p.exists():
            assert np.load(p, mmap_mode="r").shape == shape, (p, shape)
            continue
        x = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=shape)
        x.flush()
        del x
        made.append(name)
    for name, shape in (("done_sameday", (ND,)), ("done_chain", (NI,))):
        p = out / f"{name}.npy"
        if not p.exists():
            np.save(p, np.zeros(shape, np.uint8))
            made.append(name)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "positions": str(posf), "positions_sha256": sha(posf), "n_pix": n_pix,
            "axes": "same-day arms (day of 2022, member, station pixel, where); forecast arms (issue day - 120, lead day - 1, "
                    "member, station pixel, where); where = station pixel, ring 1 km, ring 3 km, ring 10 km",
            "units": "ug m-3, float16", "shapes": {k: list(v) for k, v in files.items()},
            "code_sha256": {f: sha(O / "code" / f) for f in ("predict_official.py", "init_preds.py")}}
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        assert old["positions_sha256"] == meta["positions_sha256"], "positions changed since these arrays were created"
        meta["created"] = old["created"]
        meta["previous_code_sha256"] = old.get("previous_code_sha256", []) + ([old["code_sha256"]] if old["code_sha256"] != meta["code_sha256"] else [])
    mp.write_text(json.dumps(meta, indent=1))
    print(f"[init] {out}: created {made or 'nothing (all present)'}; n_pix {n_pix}", flush=True)


if __name__ == "__main__":
    main()
