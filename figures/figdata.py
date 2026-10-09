#!/usr/bin/env python3
"""Forecast fields for the paper figures (spectra, maps): every 7th start day of 2022 plus episode days, leads T+1 and
T+3, real HRES weather, 0.25 deg. One model per call (one GPU):
  flow runs (../dejavu/runs/<run>, 8 members, 4 steps): FOEHN (noise_t31_fpstab), no footprint (noise_t31);
  regression with the same network (../dejavu/runs/reg, ERA5 conditioning: shown only in the spectra);
  baselines (../retrain/runs/B_<m>_mettarget31, seed 42).
Output: data/<name>.npz with days (doy), leads, pred (n_days, n_leads, M, 168, 280) float16 in ug/m3; truth.npz.
    python figdata.py <name>"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
sys.path.insert(0, str(EXP / "dejavu"))
sys.path.insert(0, str(EXP / "retrain"))
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
DAYS = sorted(set(range(3, 360, 7)) | {70, 74, 78, 188, 194, 200, 218, 222, 298, 300})
LEADS = (1, 3)
dev = torch.device("cuda")


def flow(run, members=8, steps=4):
    from dejavu import DVData, DVModule, attach_hres, MU, SD
    model = DVModule.load_from_checkpoint(EXP / "dejavu/runs" / run / "final.ckpt", map_location="cpu").to(dev).eval()
    hp = model.hparams
    t31, fpr = hp.get("t31", False), hp.get("footprint", False)
    ds = DVData([2022], "noise", train=False, t31=t31, footprint=fpr, fp_met=hp.get("fp_met", "blh"))
    if t31:
        attach_hres(ds, t31, fpr)
    doy = np.load(EXP / "dejavu/data/meta.npz")["doy"]
    index = {(int(doy[i]), h): n for n, (i, h) in enumerate(ds.samples)}
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    M = members if hp.source != "reg" else 1
    pred, days = [], []
    for d in DAYS:
        per = []
        for h in LEADS:
            if (d, h) not in index:
                break
            it = ds[index[(d, h)]]
            cond = it["cond"][None].to(dev).expand(M, -1, -1, -1)
            lead = torch.full((M,), float(h), device=dev)
            fp = tuple(it[k][None].to(dev).expand(M, *([-1] * it[k].dim())) for k in ("wind", "rain", "stab") if k in it) if fpr else None
            x0 = torch.randn(M, 1, 168, 280, device=dev, generator=gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                per.append(model.sample(x0, cond, lead, steps, fp)[:, 0].float().cpu().numpy() * SD + MU)
        if len(per) == len(LEADS):
            pred.append(np.stack(per)); days.append(d)
    return np.array(days), np.stack(pred)


def baseline(name):
    from train_baselines import BaselineModule, CoarseDatasetV2
    from eval_coarse import swap_forecast_block
    from cranpm.data.europe_dataset_v2 import GHAP_MEAN, GHAP_STD
    ck = sorted((EXP / "retrain/runs" / f"B_{name}_mettarget31/ckpt").glob("best-*.ckpt"))[0]
    model = BaselineModule.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    ds = CoarseDatasetV2([2022], mode="fixed", train=False, extra=True)
    ds.met_day = "target31"
    swap_forecast_block(ds, "hres", 2022)
    index = {(d, h): n for n, (_, d, h) in enumerate(ds.samples)}
    pred, days = [], []
    for d in DAYS:
        if not all((d, h) in index for h in LEADS):
            continue
        items = [ds[index[(d, h)]] for h in LEADS]
        batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(batch).float().cpu().numpy()[:, 0] * GHAP_STD + GHAP_MEAN
        pred.append(p[:, None]); days.append(d)
    return np.array(days), np.stack(pred)


def truth():
    pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")
    meta = np.load(EXP / "dejavu/data/meta.npz")
    i0 = int(np.where(meta["year"] == 2022)[0][0])
    days = [d for d in DAYS if d + max(LEADS) < 365]
    t = np.stack([np.stack([np.asarray(pm[i0 + d + h], np.float32) for h in LEADS]) for d in days])
    return np.array(days), t[:, :, None]


def episode(target=83, leads=(1, 2, 3), members=32, steps=8, model="foehn"):
    """Forecasts of one target day (default: 25 March 2022, the day of 2022 with the largest area above 25 ug/m3) from
    several initialisation days: FOEHN members, or a deterministic baseline."""
    global DAYS, LEADS
    out = []
    for h in leads:
        DAYS, LEADS = [target - h], (h,)
        d, p = flow("noise_t31_fpstab", members=members, steps=steps) if model == "foehn" else baseline(model)
        out.append(p[0, 0])
    return np.array([target - h for h in leads]), np.stack(out)[:, None]


if __name__ == "__main__":
    name = sys.argv[1]
    if name.startswith("series"):           # series_foehn / series_convlstm: every start day of Feb-Apr 2022, lead 2
        model = name.split("_")[1]
        DAYS, LEADS = list(range(38, 112)), (2,)
        days, pred = flow("noise_t31_fpstab", members=16, steps=8) if model == "foehn" else baseline(model)
        np.savez_compressed(OUT / f"{name}.npz", days=days, leads=np.array(LEADS), pred=pred.astype(np.float16))
        print(name, pred.shape); sys.exit()
    if name.startswith("episode"):          # episode or episode_<baseline>
        model = name.split("_")[1] if "_" in name else "foehn"
        days, pred = episode(model=model)
        np.savez_compressed(OUT / f"{name}83.npz", days=days, leads=np.array([1, 2, 3]), pred=pred.astype(np.float16))
        print(name, pred.shape); sys.exit()
    if name == "truth":
        days, pred = truth()
    elif name in ("foehn", "nofp", "reg"):
        days, pred = flow({"foehn": "noise_t31_fpstab", "nofp": "noise_t31", "reg": "reg"}[name])
    else:
        days, pred = baseline(name)
    np.savez_compressed(OUT / f"{name}.npz", days=days, leads=np.array(LEADS), pred=pred.astype(np.float16))
    print(name, pred.shape)
