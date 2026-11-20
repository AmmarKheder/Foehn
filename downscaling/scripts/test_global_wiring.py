#!/usr/bin/env python3
"""CPU checks of the --store global wiring (global-prep, 2026-10-26). Exit 1 if any check fails.

  store     open_store('global', fires='mami'): split A dates, pixel count 114,305,974, training points only;
            store_kwargs (smoke/toy -> {}; global -> fires, default injh); injh_status; fires rows per date
  firewall  the audit hook: open()/np.load/os.open of a sealed LGHAP file or a station file raises PermissionError;
            points() refuses a non-training day; holdout() only reaches 2023
  batch     one BatchStream batch (B = 2, N = 16,384): 47 x 576 x 1440 finite inputs, z statistics, days in split A
  cli       train_aq (base) and train_x (base) 2 steps each on CPU (B = 1, N = 1,024, 1 validation day x 2,000
            pixels, 2 members x 2 Euler), then eval_x on the train_x run (1 day, every 4,000th pixel)
  hcf       (only if the cell-mean table exists) train_x --variant hcf 2 steps on CPU + eval_x
  python scripts/test_global_wiring.py [store firewall batch cli hcf] --tmp DIR
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(check=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} " + json.dumps(info, default=str)[:800], flush=True)


def check_store():
    import argparse as ap_
    from cdffm_aq.data_aq import open_store, store_kwargs
    from cdffm_aq.store_global import injh_status, D2023
    st = open_store("global", fires="mami")
    record("store/open", st.n_pix == 114305974 and len(st.train_dates) == 1270 and len(st.hold_dates) == 358
           and str(st.train_dates[0]) == "2019-07-11" and str(st.train_dates[-1]) == "2022-12-31"
           and str(st.hold_dates[0]) == "2023-01-08" and str(st.hold_dates[-1]) == "2023-12-31"
           and st.fires_name == "mami", n_pix=st.n_pix, n_train=len(st.train_dates), n_hold=len(st.hold_dates))
    kw = {s: store_kwargs(ap_.Namespace(store=s, fires=None)) for s in ("smoke", "toy", "global")}
    kw["global_mami"] = store_kwargs(ap_.Namespace(store="global", fires="mami"))
    record("store/store_kwargs", kw["smoke"] == {} and kw["toy"] == {} and kw["global"] == {"fires": "injh", "split": "A"}
           and kw["global_mami"] == {"fires": "mami", "split": "A"}, kw=kw)
    ok, why = injh_status()
    record("store/injh_status_reported", True, injh_ready=ok, reason=why)
    import datetime as dt
    d = dt.date(2022, 12, 31)
    a = st.raw_inputs(d)
    b = st.raw_inputs(dt.date(2023, 1, 1))
    record("store/raw_inputs_both_sides_of_2023", all(np.isfinite(v.astype(np.float32)).all() for v in a.values())
           and all(np.isfinite(v.astype(np.float32)).all() for v in b.values()),
           shapes={k: list(v.shape) for k, v in a.items()})


def check_firewall():
    from cdffm_aq.store_global import GlobalStore, GLOBAL_ROOT
    st = GlobalStore(fires="mami")
    sealed = GLOBAL_ROOT / "sealed" / "lghap_points" / "idx_2024.npy"
    tries = {}
    for name, fn in (("open", lambda: open(sealed, "rb")), ("np.load", lambda: np.load(sealed, mmap_mode="r")),
                     ("os.open", lambda: os.open(str(sealed), os.O_RDONLY)),
                     ("stations", lambda: open(ROOT / "data" / "stations" / "x.txt", "rb")),
                     ("raw_lghap", lambda: open(ROOT / "data" / "raw" / "lghap_v2" / "x.zip", "rb"))):
        try:
            fn()
            tries[name] = "OPENED"
        except PermissionError:
            tries[name] = "blocked"
        except FileNotFoundError:
            tries[name] = "not found (NOT blocked)"
    record("firewall/sealed_and_stations_blocked", all(v == "blocked" for v in tries.values()), tries=tries)
    try:
        st.train_dates.append(st.hold_dates[0])
        st.points(len(st.train_dates) - 1)
        refused = False
    except (AssertionError, KeyError):
        refused = True
    finally:
        st.train_dates.pop()
    record("firewall/points_refuse_non_training_day", refused)
    record("firewall/holdout_only_2023", all(d.year == 2023 for d in st.hold_dates) and st.hold.shape[0] == 365)


def check_batch():
    import torch
    from cdffm_aq.data_aq import BatchStream, PixelArrays, assemble_inputs, target_z
    from cdffm_aq.store_global import GlobalStore
    st = GlobalStore(fires="mami")
    cpu = torch.device("cpu")
    pix = PixelArrays(st, cpu)
    bs = BatchStream(st, 2, 16384, 0, 0, cpu)
    t0 = time.time()
    b = bs.next()
    bs.close()
    x = assemble_inputs(b["raw"], pix)
    mu, sig = pix.mu_sigma(b["k"], b["doy"])
    z = target_z(b["v"], mu, sig)
    days = [str(st.train_dates[int(t)]) for t in b["day"]]
    record("batch/global", tuple(x.shape) == (2, 47, 576, 1440) and bool(torch.isfinite(x).all())
           and bool(torch.isfinite(z).all()) and abs(float(z.mean())) < 1 and 0.5 < float(z.std()) < 2,
           days=days, seconds=time.time() - t0, z_mean_std=[float(z.mean()), float(z.std())],
           ch_std_minmax=[float(x.std(dim=(0, 2, 3)).min()), float(x.std(dim=(0, 2, 3)).max())],
           loader_fill_s=bs.fill_s, loader_mb=bs.fill_bytes / 2 ** 20)


def _run(cmd, name, tmp, timeout=3600):
    t0 = time.time()
    log = Path(tmp) / f"{name}.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), timeout=timeout)
    tail = log.read_text()[-1500:]
    print(f"--- {name} (rc {r.returncode}, {time.time() - t0:.0f} s), tail:\n{tail}", flush=True)
    return r.returncode, tail


def check_cli(tmp, variants=("base",)):
    small = ["--store", "global", "--fires", "mami", "--batch-size", "1", "--n-points", "1024", "--max-steps", "2",
             "--val-every", "2", "--val-days", "0", "--val-pixels", "2000", "--val-members", "2", "--val-ode-steps",
             "2", "--warmup", "2", "--log-every", "1", "--device", "cpu", "--runs-root", str(tmp)]
    if "base" in variants:
        rc, tail = _run([sys.executable, "-u", "-m", "cdffm_aq.train_aq", "--name", "g_aq_base"] + small, "train_aq", tmp)
        mj = Path(tmp) / "g_aq_base" / "metrics.jsonl"
        args = json.loads((Path(tmp) / "g_aq_base" / "args.json").read_text()) if rc == 0 else {}
        record("cli/train_aq_global_base", rc == 0 and mj.exists() and args.get("fires") == "mami"
               and args.get("store") == "global", rc=rc)
    for v in variants:
        name = f"g_x_{v}"
        rc, tail = _run([sys.executable, "-u", "-m", "cdffm_aq_x.train_x", "--name", name, "--variant", v] + small,
                        f"train_x_{v}", tmp)
        ok = rc == 0 and (Path(tmp) / name / "ckpt" / "last.ckpt").exists()
        args = json.loads((Path(tmp) / name / "args.json").read_text()) if ok else {}
        record(f"cli/train_x_global_{v}", ok and args.get("fires") == "mami"
               and (v != "hcf" or args.get("cellmeans", "").endswith("cellmean_z_A.npz")), rc=rc)
        if ok:
            rc, tail = _run([sys.executable, "-u", "-m", "cdffm_aq_x.eval_x", "--run", name, "--runs-root", str(tmp),
                             "--ckpt", "last", "--members", "2", "--ode-steps", "2", "--days", "0", "--pixel-stride",
                             "4000", "--device", "cpu", "--chunk", "65536"], f"eval_x_{v}", tmp)
            mf = Path(tmp) / name / "eval_x_last_M2_s2" / "metrics.json"
            m = json.loads(mf.read_text()) if mf.exists() else {}
            record(f"cli/eval_x_global_{v}", rc == 0 and m.get("n_pixels_per_day", 0) > 20000
                   and "blocks" in m and m.get("pixel_stride") == 4000, rc=rc,
                   crps=m.get("scores", {}).get("model", {}).get("ug", {}).get("crps_fair"),
                   n_pix=m.get("n_pixels_per_day"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", default=["store", "firewall", "batch", "cli"])
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    Path(a.tmp).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for nm in a.checks:
        print(f"\n=== {nm}", flush=True)
        try:
            {"store": check_store, "firewall": check_firewall, "batch": check_batch,
             "cli": lambda: check_cli(a.tmp, ("base",)), "hcf": lambda: check_cli(a.tmp, ("hcf",))}[nm]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(f"{nm}/exception", False, error=repr(e)[:500])
    n_fail = sum(not r["ok"] for r in RESULTS)
    print(f"\nGLOBAL WIRING SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAILED: {r['check']}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
