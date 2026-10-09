#!/usr/bin/env python3
"""Score 0.25 deg models (train_baselines.py checkpoints) against the GHAP 0.25 deg block means, per lead time, on every
day of a year: RMSE on land, persistence on the same samples, and CSI for exceedance of 25 ug/m3.
    python eval_coarse.py runs/B_convlstm_mettarget [--year 2022]
--met-source (target31 models only): era5 = perfect forecast-day weather; era5_smooth = the same smoothed to ~1.5 deg
(the resolution of the forecasts used); hres = the real ECMWF IFS HRES forecast issued at 00 UTC of the start day.
"""
import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baselines import LAND, BaselineModule, CoarseDatasetV2  # noqa: E402

from cranpm.data.europe_dataset_v2 import GHAP_MEAN, GHAP_STD  # noqa: E402


HRES = Path(__file__).resolve().parents[1] / "dejavu" / "data" / "hres"
WIND = [0, 1] + list(range(10, 20))


def box6(x):
    """6x6 box average (0.25 -> ~1.5 deg effective resolution), edges by replication."""
    p = np.pad(x, ((0, 0), (3, 2), (3, 2)), mode="edge")
    c = p.cumsum(1).cumsum(2)
    c = np.pad(c, ((0, 0), (1, 0), (1, 0)))
    return (c[:, 6:, 6:] - c[:, :-6, 6:] - c[:, 6:, :-6] + c[:, :-6, :-6]) / 36.0


def swap_forecast_block(ds, source, year):
    """Replace the forecast-day block of every sample by a smoothed ERA5 or by the HRES forecast issued on day d."""
    get = ds.__class__.__getitem__
    if source == "hres":
        table = {}
        for f in sorted(HRES.glob("*.npz")):
            with np.load(f) as m:
                blk, days = m["block"], m["day_of_year"]  # read each month once (npz members re-read on every access)
            table.update({int(dd): blk[k] for k, dd in enumerate(days)})
        em, es = ds.era5_mean[:, None, None], ds.era5_std[:, None, None]
        xm, xs = ds.extra_mean[3], ds.extra_std[3]

    def item(self, idx):
        y, d, h = self.samples[idx]
        out = get(self, idx)
        n = out["era5"].shape[0]
        if source == "era5_smooth":
            blk = out["era5"][n - 31:].numpy()
            out["era5"][n - 31:] = torch.from_numpy(box6(blk).astype(np.float32))
        else:
            raw = table[d][h - 1].astype(np.float32)[:, :168, :280]
            if os.environ.get("WIND_FLIP"):  # control: reverse the forecast wind direction, keep its speed
                raw = raw.copy(); raw[WIND] = -raw[WIND]
            sw = int(os.environ.get("WIND_SWAP", 0))
            if sw:  # control: the forecast wind of the forecast issued sw days later (or earlier), everything else true
                d2 = d + sw if (d + sw) in table else d - sw
                raw = raw.copy(); raw[WIND] = table[d2][h - 1].astype(np.float32)[WIND, :168, :280]
            era = (raw[:30] - em) / es  # raw = store convention (x - m) / s, then the dataset's standardisation
            tp = (np.log1p(np.maximum(raw[30], 0) * 1000.0) - xm) / xs
            out["era5"][n - 31:] = torch.from_numpy(np.concatenate([era, tp[None]]).astype(np.float32))
        return out

    ds.__class__ = type("SwappedDS", (ds.__class__,), {"__getitem__": item})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--met-source", choices=["era5", "era5_smooth", "hres"], default="era5")
    a = ap.parse_args()
    run = Path(a.run)
    ckpt = sorted(run.glob("ckpt/best-*.ckpt"))[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = BaselineModule.load_from_checkpoint(ckpt, map_location="cpu").to(dev).eval()
    met_day = module.hparams.get("met_day", "none")
    ds = CoarseDatasetV2([a.year], mode="fixed", train=False, extra=met_day != "none")
    ds.met_day = met_day
    if a.met_source != "era5":
        assert met_day == "target31", "real forecasts only replace the 31-channel forecast-day block"
        swap_forecast_block(ds, a.met_source, a.year)
    acc = {h: np.zeros(8) for h in (1, 2, 3, 4)}  # n, sse, sse_pers, hits, misses, false alarms, pers hits/misses/fa below
    ev = {h: np.zeros(6) for h in (1, 2, 3, 4)}
    records = []   # per (start day, lead): pixels, sse, sae, hits, union (CSI > 25), for the extreme-day figures
    for i in range(0, len(ds), a.batch):
        items = [ds[j] for j in range(i, min(i + a.batch, len(ds)))]
        batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = module(batch).float()
        for j, it in enumerate(items):
            h = int(it["lead_time"])
            tgt, p, g0 = it["target"][0].numpy(), pred[j, 0].cpu().numpy(), it["ghap_coarse"][0].numpy()
            m = tgt > LAND
            T, P, G = (x[m] * GHAP_STD + GHAP_MEAN for x in (tgt, p, g0))
            acc[h][:3] += [m.sum(), ((P - T) ** 2).sum(), ((G - T) ** 2).sum()]
            _, dd, _ = ds.samples[i + j]
            records.append([dd, h, m.sum(), ((P - T) ** 2).sum(), np.abs(P - T).sum(), ((T > 25) & (P > 25)).sum(), ((T > 25) | (P > 25)).sum()])
            for k, F in enumerate((P, G)):
                e, f = T > 25, F > 25
                ev[h][3 * k:3 * k + 3] += [(e & f).sum(), (e & ~f).sum(), (~e & f).sum()]
    out = {}
    csi = lambda v: v[0] / max(v.sum(), 1)
    print(f"{run.name} (met_day={met_day}, weather source {a.met_source}), {a.year}, 0.25 deg vs GHAP")
    for h in (1, 2, 3, 4):
        n, sse, ssp = acc[h][:3]
        out[f"T+{h}"] = dict(rmse=float(np.sqrt(sse / n)), persistence=float(np.sqrt(ssp / n)),
                             csi25=float(csi(ev[h][:3])), csi25_persistence=float(csi(ev[h][3:])))
        o = out[f"T+{h}"]
        print(f"  T+{h}: RMSE {o['rmse']:.2f} (persistence {o['persistence']:.2f}) | CSI>25 {o['csi25']:.3f} (persistence {o['csi25_persistence']:.3f})")
    tag = ("" if a.met_source == "era5" else f"_{a.met_source}") + (f"_windswap{os.environ['WIND_SWAP']}" if os.environ.get("WIND_SWAP") else "") + ("_windflip" if os.environ.get("WIND_FLIP") else "")
    (run / f"coarse_{a.year}{tag}.json").write_text(json.dumps(out, indent=1))
    np.save(run / f"records_coarse_{a.year}{tag}.npy", np.array(records, np.float64))


if __name__ == "__main__":
    main()
