#!/usr/bin/env python3
"""Data of the architecture overview (fig_gc1_schematic.py): FLEXGREEN backward trajectories of the air reaching 25 cities
on the target day 25 January 2022, followed back 5 days to the start day 20 January in the 3-D wind of the HRES run of
20 January (lead 5), and the 850 hPa wind of that run at 00 UTC of the start day. Same physics loop as anim_green.py
(flexgreen.py: advection, boundary-layer turbulence, dry and wet removal); positions every hour.
    python figdata_arch2.py      (pytorch container, CPU or GPU) -> data/arch_overview.npz"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "global1km"))
import flexgreen as fg  # noqa: E402

D0, H, P = 19, 5, 64                       # start day 20 January 2022 (0-based day of year), lead 5 days, particles per city
CITIES = {"Delhi": (28.61, 77.21), "Karachi": (24.86, 67.01), "Dhaka": (23.81, 90.41), "Mumbai": (19.08, 72.88),
          "Tehran": (35.69, 51.39), "Riyadh": (24.71, 46.68), "Cairo": (30.04, 31.24), "Khartoum": (15.50, 32.56),
          "Addis Ababa": (9.03, 38.74), "Kano": (12.00, 8.52), "Lagos": (6.52, 3.38), "Istanbul": (41.01, 28.98),
          "Moscow": (55.76, 37.62), "Tashkent": (41.30, 69.24), "Chengdu": (30.57, 104.07), "Beijing": (39.90, 116.40),
          "Bangkok": (13.76, 100.50), "Nairobi": (-1.29, 36.82)}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = torch.Generator(device=dev); gen.manual_seed(11)
zs, z0 = fg.static(dev)
Bn = fg.load_run("hres", 2022, D0)                                           # (33, 53, 181, 360): steps 0..192 h, 1 deg
u850, v850 = Bn[0, fg.IU + 2].copy(), Bn[0, fg.IV + 2].copy()               # 850 hPa wind at 00 UTC of the start day
B = torch.tensor(Bn, device=dev); del Bn
fld = fg.Field(fg.derive(B, zs, z0)); del B
nc = len(CITIES)
lon = torch.cat([torch.full((P,), lo) for la, lo in CITIES.values()]).to(dev) + (torch.rand(P * nc, generator=gen, device=dev) - .5) * .3
lat = torch.cat([torch.full((P,), la) for la, lo in CITIES.values()]).to(dev) + (torch.rand(P * nc, generator=gen, device=dev) - .5) * .3
z = torch.rand(P * nc, generator=gen, device=dev) * 100.0
t_rel = 24.0 * H + 24.0 * torch.rand(P * nc, generator=gen, device=dev)        # release over the target day (hours of the run)
ts = t_rel / 6.0
surv = torch.ones_like(lon)
rec = [(lon.cpu().numpy().copy(), lat.cpu().numpy().copy(), z.cpu().numpy().copy(), surv.cpu().numpy().copy())]
n = int(round(24 * H * 3600 / fg.DT))
for k in range(n):                                                           # flexgreen.backward's loop, recorded every hour
    lon, lat, z, s = fg.advect(fld, lon, lat, z, ts, -fg.DT)
    ts = ts - fg.DT / (6 * 3600.0)
    h_, ust, invL, wst = s[fg.C_PBL], s[fg.C_UST], s[fg.C_IL], s[fg.C_WST]
    z, lon, lat = fg.turbulence(z, lon, lat, h_, ust, invL, wst, gen, kbar=s[fg.C_KBAR])
    lon = ((lon + 180) % 360) - 180
    surv = surv * torch.where(z < 2 * fg.HREF, math.exp(-fg.VDEP * fg.DT / (2 * fg.HREF)), 1.0)
    pr = s[fg.C_PR].clamp(min=0)
    surv = surv * torch.where((pr > 0.01) & (z < s[fg.C_CTOP]), torch.exp(-fg.WET_A * pr ** fg.WET_B * fg.DT), torch.ones_like(pr))
    if (k + 1) % 4 == 0:
        rec.append((lon.cpu().numpy().copy(), lat.cpu().numpy().copy(), z.cpu().numpy().copy(), surv.cpu().numpy().copy()))
pbl0 = s[fg.C_PBL].cpu().numpy()                                             # mixing height at the origins (start day)
L, A, Z, S = (np.stack([r[i] for r in rec]).astype(np.float32) for i in range(4))   # (hours + 1, particles)
city = np.repeat(np.arange(nc), P)
np.savez_compressed(HERE / "data" / "arch_overview.npz", lon=L, lat=A, z=Z, surv=S, city=city, pbl0=pbl0,
                    names=np.array(list(CITIES)), clat=np.array([v[0] for v in CITIES.values()]),
                    clon=np.array([v[1] for v in CITIES.values()]), u850=u850.astype(np.float32), v850=v850.astype(np.float32),
                    d0=D0, lead=H)
print("saved", L.shape, "origins below the mixing height:", float(np.mean(Z[-1] <= np.maximum(pbl0, 100.0))))
