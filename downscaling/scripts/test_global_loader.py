#!/usr/bin/env python3
"""CPU dataloader test of the global store (cdffm_aq.store_global.GlobalStore + data_aq.BatchStream, unchanged):
  1. the training sampler's reachable days = the 1,270 split-A training days; no 2023 / 2024 day (nor any sealed
     robustness day) is reachable: checked on the day list, on the point-file mapping of every training index, and on
     every row actually read while drawing batches (NpyRows.read is logged);
  2. no file under sealed/, no station file and no 2023 truth row is read by the training sampler; no such fd is open;
  3. batches assemble into finite 47-channel encoder inputs and finite z targets (PixelArrays, assemble_inputs,
     target_z of data_aq).
Training container:  python -u test_global_loader.py [--batches 20]"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
from cdffm_aq import data_aq  # noqa: E402
from cdffm_aq.store_global import GlobalStore, D2023  # noqa: E402

TRAIN0, TRAIN1 = dt.date(2019, 7, 11), dt.date(2022, 12, 31)


def row_date(path, row):
    """calendar date of a row of a global / Foehn store file, from the file's naming convention"""
    p = str(path)
    name = Path(p).name
    if name in ("met.npy", "fires.npy", "cams_pm25_025.npy"):
        return dt.date(2017, 1, 1) + dt.timedelta(row)
    if name.endswith("_2023_2024.npy"):
        return D2023 + dt.timedelta(row)
    if name.startswith("t6_"):
        return dt.date(int(name[3:7]), 1, 1) + dt.timedelta(row)
    if "lghap_points" in p:
        tag = name.split("_")[1][:-4]
        first = {"2019b": dt.date(2019, 7, 11), "2019a": dt.date(2019, 1, 1)}.get(tag, None) or dt.date(int(tag), 1, 1)
        return first + dt.timedelta(row)
    if name == "full_2023.npy":
        return D2023 + dt.timedelta(row)
    raise ValueError(p)


def main(a):
    t0 = time.time()
    log = []
    orig = data_aq.NpyRows.read

    def logged(self, i, out=None):
        log.append((self.path, int(i)))
        return orig(self, i, out)
    data_aq.NpyRows.read = logged

    st = GlobalStore()
    res = {"n_train_dates": len(st.train_dates), "train_first_last": [str(st.train_dates[0]), str(st.train_dates[-1])],
           "n_hold_dates": len(st.hold_dates), "hold_first_last": [str(st.hold_dates[0]), str(st.hold_dates[-1])]}
    # 1a. the reachable set
    assert len(st.train_dates) == 1270 and st.train_dates[0] == TRAIN0 and st.train_dates[-1] == TRAIN1
    assert len(set(st.train_dates)) == 1270
    assert not any(d >= D2023 for d in st.train_dates), "2023/2024 day reachable"
    assert not any(d <= dt.date(2019, 7, 10) for d in st.train_dates), "sealed robustness day reachable"
    assert not set(st.train_dates) & set(st.hold_dates)
    # 1b. every training index maps to a training point file and an in-range row (no read)
    for t, d in enumerate(st.train_dates):
        hit = [(f0, idx) for f0, f1, idx, _ in st._pts if f0 <= d <= f1]
        assert len(hit) == 1 and "sealed" not in hit[0][1].path and 0 <= (d - hit[0][0]).days < hit[0][1].shape[0]
    try:
        st.points(len(st.train_dates))
        raise AssertionError("index beyond the training days accepted")
    except IndexError:
        pass
    # 2. draw batches with the unchanged BatchStream of two ranks
    dev = torch.device("cpu")
    pix = data_aq.PixelArrays(st, dev)
    drawn = []
    for rank in range(2):
        bs = data_aq.BatchStream(st, B=2, N=16384, seed=0, rank=rank, device=dev, n_slots=2)
        for b in range(a.batches):
            batch = bs.next()
            days = [st.train_dates[int(t)] for t in batch["day"]]
            drawn += days
            if b == 0:
                x = data_aq.assemble_inputs(batch["raw"], pix, "025", "t6")
                mu, sg = pix.mu_sigma(batch["k"], batch["doy"])
                z = data_aq.target_z(batch["v"], mu, sg, u=torch.rand(batch["v"].shape) - 0.5)
                assert x.shape == (2, 47, 576, 1440) and torch.isfinite(x).all(), "non-finite encoder input"
                assert torch.isfinite(z).all(), "non-finite target"
                res.setdefault("first_batch", []).append({"rank": rank, "days": [str(d) for d in days],
                                                         "input_std_range": [float(x.std((0, 2, 3)).min()), float(x.std((0, 2, 3)).max())],
                                                         "z_mean_std": [float(z.mean()), float(z.std())]})
        bs.close()
    assert all(TRAIN0 <= d <= TRAIN1 for d in drawn)
    res["drawn_days"] = len(drawn)
    res["drawn_distinct"] = len(set(drawn))
    res["drawn_min_max"] = [str(min(drawn)), str(max(drawn))]
    # every row read while sampling belongs to a training date (or to D - 1 for nothing: T6 is per target day)
    reads = [(p, r, row_date(p, r)) for p, r in log]
    bad = [(p, r, str(d)) for p, r, d in reads if not (TRAIN0 <= d <= TRAIN1)]
    res["rows_read"] = len(reads)
    res["files_read"] = sorted({p for p, _, _ in reads})
    assert not bad, f"rows outside the training days read: {bad[:5]}"
    assert not any("sealed" in p or "stations" in p or "full_2023" in p for p, _, _ in reads)
    fds = [os.readlink(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd") if os.path.exists(f"/proc/self/fd/{f}")]
    res["open_fds_sealed_or_stations"] = [f for f in fds if "sealed" in f or "stations" in f]
    assert not res["open_fds_sealed_or_stations"]
    res["ok"] = True
    res["seconds"] = round(time.time() - t0)
    out = ROOT / "data/processed/global/manifest_parts/loader_test.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("LOADER TEST PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    main(ap.parse_args())
