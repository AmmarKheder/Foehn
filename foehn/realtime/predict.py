#!/usr/bin/env python3
"""RT step 3b: the forecast of start day d from the inputs of data/<yyyy-mm-dd>/ (build.py) and its Green's functions
(flexgreen.py --rt-dir), with the model, checkpoint, sampler and seeds of the paper's foehn_cams.py predict().
RtCamsData inherits CamsData and replaces only the reading of the inputs: the CAMS analysis from cams.npy, the weather
from hres_B.npy, the fires from fires.npy and the transport from green/ of the day directory; normalisation statistics,
static fields, particle subsets and noise seeds are those of CamsData / predict().
Output: out/<yyyy-mm-dd>/pred.npy (members, 10, 361, 900) float16, ug m-3, leads 12, 24, ..., 120 h, and meta.json.
    python predict.py 2026-10-10 [--members 8] [--steps 8] [--refine-tail 3]"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import common as C

from foehn_cams import (CMETA, FS_SCALE, G, GI_FIRST, HSTATS, NC, NCH, NE, NL, NR, NW, NX, NY, PX, PY, STATIC, CamsData,
                        CamsModule, Npy, clean_origins, collate, flow_grid, read, rewrap, wrap_pad)
from dejavu import MU, SD


class RtCamsData(CamsData):
    """CamsData whose inputs come from one real-time day directory (no scan of the training stores)."""

    def __init__(self, day_dir, psub=16, emis=False, no_fires=False):
        rt = Path(day_dir)
        self.no_fires = no_fires                                                  # hindcast test: the model sees no fire
        m = json.loads((rt / "meta.json").read_text())
        meta = np.load(G / "meta.npz")
        self.year, self.doy = meta["year"], meta["doy"]
        cm = np.load(CMETA)
        self.eps, self.cmu, self.csd = cm["eps"][:, None, None], cm["mean"][:, None, None], cm["std"][:, None, None]
        hs = np.load(HSTATS)
        self.hmu, self.hsd = hs["mean"][:, None, None], hs["std"][:, None, None]
        self.static = np.load(STATIC)["static"].astype(np.float32)
        self.psub, self.emis, self.train, self.lag = psub, emis, False, 0
        self.gi, self.ryear = int(m["gi"]), int(m["year"])
        self.cams_rt = np.load(rt / "cams.npy")                                   # (2, 13, NY, NX): t0 - 12 h, t0
        self.hb_rt = np.load(rt / "hres_B.npy")                                   # (33, NW, 181, 360)
        self.fire_rt = np.load(rt / "fires.npy")                                  # (3, 576, 1440), GFAS of d - 1
        if no_fires:                                                              # the standardised value of "no fire" (build_transport)
            self.fire_rt = np.broadcast_to(np.array([-0.5, -0.5, -1.0 / 6.0], np.float16)[:, None, None], self.fire_rt.shape).copy()
        green = rt / "green"
        self.org = Npy(green / f"origins_{self.ryear}.npy")
        self.fs = Npy(green / f"fire_{self.ryear}.npy")
        self.es = Npy(green / f"emis_{self.ryear}.npy") if emis else None
        self.samples = [(self.gi, k) for k in range(1, NL + 1)]

    # --- the readers of CamsData, on the day's files (same arithmetic, same normalisation)
    def cams_norm(self, t):
        x = self.cams_rt[1 if t == 2 * (self.gi - GI_FIRST) else 0].astype(np.float32)
        return (np.log(np.maximum(x, 0) + self.eps) - self.cmu) / self.csd

    def pm25(self, t):
        if t > 2 * (self.gi - GI_FIRST):                                           # the target: unknown in real time
            return np.zeros((NY, NX), np.float32)
        return self.cams_rt[1 if t == 2 * (self.gi - GI_FIRST) else 0, 1].astype(np.float32)

    def hres(self, gi, k):
        b = self.hb_rt
        st = [b[s].astype(np.float32) for s in (0, 2 * k)]
        st[0][NW - 1] = np.nan
        tp = b[2 * k - 2, NW - 1].astype(np.float32) + b[2 * k - 1, NW - 1].astype(np.float32)
        for a in st:
            a[NW - 1] = np.log1p(np.maximum(a[NW - 1], 0))
        x = np.concatenate([st[0], st[1], np.log1p(np.maximum(tp, 0))[None]])
        return np.nan_to_num((x - self.hmu) / self.hsd)

    def item(self, gi, k, member=0):
        sd = gi - GI_FIRST
        t0 = 2 * sd
        c0, cm = self.cams_norm(t0), self.cams_norm(t0 - 1)
        pm0 = (self.pm25(t0) - MU) / SD
        tgt = (self.pm25(t0 + k) - MU) / SD
        fine = wrap_pad(np.concatenate([c0, cm, self.static]))
        mask = wrap_pad(np.ones((1, NY, NX), np.float32)); mask[..., NX:] = 0
        o = self.org
        P = int(o.shape[4]); per = NR * NC * P * 4
        org = read(o, (0 * NL + k - 1) * per, per, (NR, NC, P, 4))               # the day store holds one day
        rng = np.random.default_rng((int(gi) * 31 + int(k)) * 101 + int(member))
        sub = np.sort(rng.choice(P, self.psub, replace=False))
        fs = read(self.fs, (0 * NL + k - 1) * 3 * NR * NC, 3 * NR * NC, (3, NR, NC))
        if self.no_fires:
            fs = np.zeros_like(fs)
        fire = np.nan_to_num(self.fire_rt.astype(np.float32))
        it = {"fine": torch.from_numpy(fine.astype(np.float16)), "x1": torch.from_numpy(wrap_pad(tgt[None])),
              "mask": torch.from_numpy(mask), "lead": torch.tensor(0.5 * k, dtype=torch.float32),
              "pm0": torch.from_numpy(pm0[None]), "wx": torch.from_numpy(self.hres(gi, k).astype(np.float32)),
              "fire": torch.from_numpy(fire.astype(np.float16)),
              "org": torch.from_numpy(clean_origins(org[:, :, sub])),
              "fsens": torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(fs), 0) * FS_SCALE[:, None, None]).astype(np.float32))}
        if self.emis:
            es = read(self.es, (0 * NL + k - 1) * NE * NR * NC, NE * NR * NC, (NE, NR, NC))
            it["esens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(es), 0)).astype(np.float32))
        return it


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine-tail", type=int, default=3)
    ap.add_argument("--ckpt", default=str(C.CKPT))
    ap.add_argument("--no-fires", action="store_true", help="hindcast test: no fire in the inputs (fires.npy and fire terms)")
    ap.add_argument("--out", default="pred.npy")
    a = ap.parse_args()
    d = C.parse_date(a.date)
    D, O = C.day_dir(d), C.out_dir(d)
    dev = torch.device("cuda", int(os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(dev)
    model = CamsModule.load_from_checkpoint(a.ckpt, map_location=dev).eval()
    ds = RtCamsData(D, psub=model.hparams.psub, emis=model.hparams.emis, no_fires=a.no_fires)
    gi = ds.gi
    grid = flow_grid(model, a.steps, a.refine_tail)
    C.log(d, f"predict {d}: gi {gi}, {a.members} members, {len(grid) - 1} Euler steps, emis {bool(model.hparams.emis)}, "
             f"ckpt {Path(a.ckpt).name}" + (", NO FIRES" if a.no_fires else ""))
    t0 = time.time()
    pred = np.zeros((a.members, NL, NY, NX), np.float16)
    for k in range(1, NL + 1):
        b = collate([ds.item(gi, k, member=m) for m in range(a.members)])
        b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
        gen = torch.Generator(device=dev).manual_seed(gi * 1000 + k)                 # the seeds of foehn_cams.predict
        x = rewrap(torch.randn(a.members, 1, PY, PX, device=dev, generator=gen))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(len(grid) - 1):
                tau = torch.full((a.members,), float(grid[s]), device=dev)
                x = rewrap(x + model.velocity(x, b, tau).float() * float(grid[s + 1] - grid[s]))
        pred[:, k - 1] = (x[:, 0, :NY, :NX] * SD + MU).cpu().numpy().astype(np.float16)
        print(f"   lead {12 * k:3d} h: mean {float(pred[:, k - 1].astype(np.float32).mean()):.2f} ug m-3", flush=True)
    np.save(O / a.out, pred)
    C.log(d, f"{a.out} {pred.shape} written in {time.time() - t0:.0f}s" + (" (no fires)" if a.no_fires else ""))
    (O / ("meta.json" if a.out == "pred.npy" else a.out.replace(".npy", "_meta.json"))).write_text(json.dumps(dict(date=str(d), gi=gi, members=a.members, steps=a.steps, refine_tail=a.refine_tail,
                                                 ckpt=a.ckpt, leads_h=[12 * k for k in range(1, NL + 1)],
                                                 inputs=C.meta(d)), indent=1, default=str))


if __name__ == "__main__":
    main()
