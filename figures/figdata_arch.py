#!/usr/bin/env python3
"""Real thumbnails for the architecture figure (East Asia, forecast issued 22 January 2022 for 25 January, lead 3):
3-D HRES wind, FLEXGREEN trajectories, the global map of day d, learned latent tracers (encoder of Foehn-Green A,
checkpoint copied to data/greenA_epoch18.ckpt), gathered transport maps, 1 km inputs, 1 km members and the
exceedance probability.     python figdata_arch.py   (container) -> data/arch_thumbs.npz"""
import math, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "global1km")); sys.path.insert(0, str(HERE.parent / "dejavu"))
import flexgreen as fg  # noqa: E402
from dejavu import norm  # noqa: E402
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
D, H, I0 = 21, 3, 1826
out = {}
# --- 3-D wind of the HRES run of day d (1 deg), East Asia 15-55 N, 95-145 E, step +48 h
B = np.load(fg.WB / "hres_B_2022_01.npy", mmap_mode="r")[list(np.load(fg.WB / "hres_B_2022_01_doy.npy")).index(D)]
r0, r1, c0, c1 = 90 - 55, 90 - 15 + 1, 95 + 180, 145 + 180 + 1
for name, iu, iv in (("w10", fg.IU10, fg.IV10), ("w850", fg.IU + 2, fg.IV + 2), ("w500", fg.IU + 5, fg.IV + 5)):
    out[f"{name}_u"], out[f"{name}_v"] = np.asarray(B[8, iu, r0:r1, c0:c1], np.float32), np.asarray(B[8, iv, r0:r1, c0:c1], np.float32)
out["w700_w"] = np.asarray(B[8, fg.IW + 3, r0:r1, c0:c1], np.float32)
# --- FLEXGREEN trajectories (8 particles x 6 cities, lead 3, recorded every 3 h)
dev = torch.device("cpu"); gen = torch.Generator(device=dev); gen.manual_seed(11)
zs, z0 = fg.static(dev)
fld = fg.Field(fg.derive(torch.tensor(np.asarray(B, np.float32)), zs, z0))
C = {"Beijing": (39.9, 116.4), "Shijiazhuang": (38.04, 114.51), "Jinan": (36.65, 117.12), "Seoul": (37.57, 126.98),
     "Shenyang": (41.8, 123.43), "Zhengzhou": (34.75, 113.62)}
n = 8
lat = torch.tensor(np.repeat([v[0] for v in C.values()], n), dtype=torch.float32); lon = torch.tensor(np.repeat([v[1] for v in C.values()], n), dtype=torch.float32)
z = torch.rand(len(lat), generator=gen) * 100; ts = (24.0 * H + 24.0 * torch.rand(len(lat), generator=gen)) / 6.0
tr = [np.stack([lon.numpy(), lat.numpy(), z.numpy()], -1)]
for k in range(int(24 * H * 3600 / fg.DT)):
    lon, lat, z, s = fg.advect(fld, lon, lat, z, ts, -fg.DT); ts = ts - fg.DT / 21600.0
    z, lon, lat = fg.turbulence(z, lon, lat, s[fg.C_PBL], s[fg.C_UST], s[fg.C_IL], s[fg.C_WST], gen, kbar=s[fg.C_KBAR])
    if (k + 1) % 12 == 0:
        tr.append(np.stack([lon.numpy(), lat.numpy(), z.numpy()], -1))
out["traj"] = np.stack(tr, 1)                                       # (particles, steps, 3)
out["traj_city"] = np.repeat(np.arange(len(C)), n); out["cities"] = np.array(list(C)); out["city_ll"] = np.array(list(C.values()))
# --- global map of day d (0.25 deg) and the learned encoder of Foehn-Green A
pm = np.asarray(np.load(G / "pm.npy", mmap_mode="r")[I0 + D], np.float32)
with open(G / "fires.npy", "rb") as fh:
    np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
fire = np.fromfile(G / "fires.npy", dtype=dt, count=int(np.prod(sh[1:])), offset=hdr + (I0 + D) * int(np.prod(sh[1:])) * 2).reshape(sh[1:]).astype(np.float32)
out["pm_global"] = pm[::4, ::4]; out["fire0_global"] = fire[0, ::4, ::4]
gpm = np.concatenate([np.nan_to_num(norm(pm))[None], np.isfinite(pm)[None].astype(np.float32), np.nan_to_num(fire)])
sd = torch.load(HERE / "data" / "greenA_epoch18.ckpt", map_location="cpu", weights_only=False)["state_dict"]
enc = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1))
enc.load_state_dict({k.replace("net.enc.", ""): v for k, v in sd.items() if k.startswith("net.enc.")})
with torch.no_grad():
    f = F.avg_pool2d(enc(torch.tensor(gpm)[None]), 2)[0]               # (32, 288, 720)
fe = f[:, (84 - 55) * 2:(84 - 15) * 2, (95 + 180) * 2:(145 + 180) * 2].numpy()
var = fe.reshape(32, -1).std(1); top = np.argsort(var)[::-1][:4]
out["latent_ea"] = fe[top]; out["latent_idx"] = top
# --- gathered transport maps for East Asia (stored Green functions, day d, lead 3, 16 particles)
G2 = G / "green" / "hres" / "origins_2022.npy"
with open(G2, "rb") as fh:
    np.lib.format.read_magic(fh); sh2, _, dt2 = np.lib.format.read_array_header_1_0(fh); hdr2 = fh.tell()
rr0, rr1 = (84 - 55) * 2, (84 - 15) * 2
per = int(np.prod(sh2[3:]))
o = np.fromfile(G2, dtype=dt2, count=(rr1 - rr0) * per, offset=hdr2 + (((D * 7 + H - 1) * 288 + rr0) * per) * 2).reshape((rr1 - rr0,) + tuple(sh2[3:]))
o = torch.tensor(o[:, (95 + 180) * 2:(145 + 180) * 2, :16].astype(np.float32))[None]
sys.path.insert(0, str(HERE.parent / "global1km"))
from fullflow_global_green import gather  # noqa: E402
with torch.no_grad():
    g = gather(f[None], o)[0]                                          # (34, R, Q)
    pmt = gather(F.avg_pool2d(torch.tensor(gpm[:1])[None], 2), o)[0, :1]
out["tr_latent"] = g[top[0]].numpy(); out["tr_aloft"] = g[32].numpy(); out["tr_surv"] = g[33].numpy(); out["tr_pm"] = pmt[0].numpy()
out["computed"] = (o[0, ..., 3].abs().sum(-1) > 0).numpy()
# --- coarse weather of today and of the forecast day (0.25 deg), East Asia
with open(G / "met.npy", "rb") as fh:                                 # (D, 37, 576, 1440): one plane, no memory map
    np.lib.format.read_magic(fh); shm, _, dtm = np.lib.format.read_array_header_1_0(fh); hdrm = fh.tell()
plane = 576 * 1440
t2 = np.fromfile(G / "met.npy", dtype=dtm, count=plane, offset=hdrm + ((I0 + D) * shm[1] + 2) * plane * np.dtype(dtm).itemsize).reshape(576, 1440)
out["t2m_d"] = t2[(84 - 55) * 4:(84 - 15) * 4, (95 + 180) * 4:(145 + 180) * 4].astype(np.float32)
hb = np.load(G / "hres_block.npy", mmap_mode="r") if False else None
out["pm_ea"] = pm[(84 - 55) * 4:(84 - 15) * 4, (95 + 180) * 4:(145 + 180) * 4]
# --- 1 km inputs and outputs over the case box
c = np.load(HERE.parent / "global1km/runs/fp_flow_shift25_global_L7_noant/eval_case_eastasia/crop0_d21_h3.npz")
(cr0, cr1), (cc0, cc1) = c["rows"], c["cols"]
TILE, NTR, NTC = 512, 29, 71
def tiles_box(path, k, r0, r1, c0, c1):
    with open(path, "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
    tr0, tr1, tc0, tc1 = r0 // TILE, (r1 - 1) // TILE, c0 // TILE, (c1 - 1) // TILE
    nb, B_ = tc1 - tc0 + 1, TILE * TILE
    blk = np.stack([np.fromfile(path, dtype=dt, count=nb * B_, offset=hdr + (((k * NTR + t_) * NTC + tc0) * B_) * 2).reshape(nb, TILE, TILE)
                    for t_ in range(tr0, tr1 + 1)])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, nb * TILE)
    return blk[r0 - tr0 * TILE:r1 - tr0 * TILE, c0 - tc0 * TILE:c1 - tc0 * TILE].astype(np.float32)
for nm, k in (("ghap_d", I0 + D), ("ghap_dm1", I0 + D - 1), ("ghap_target", I0 + D + H)):
    a = tiles_box(G / "ghap1km_tiles.npy", k, cr0, cr1, cc0, cc1); out[nm] = np.where(a > 0, a, np.nan)[::2, ::2].astype(np.float16)
out["elev"] = tiles_box(G / "static1km_tiles.npy", 0, cr0, cr1, cc0, cc1)[::2, ::2].astype(np.float16)
land = tiles_box(G / "static1km_tiles.npy", 1, cr0, cr1, cc0, cc1) > 0
m = c["members"].astype(np.float32); m[:, ~land] = np.nan
out["member0"], out["member1"] = m[0, ::2, ::2].astype(np.float16), m[1, ::2, ::2].astype(np.float16)
out["p75"] = (m > 75).mean(0)[::2, ::2].astype(np.float16); out["p75"][~land[::2, ::2]] = np.nan
out["noise"] = np.random.default_rng(0).standard_normal((225, 475)).astype(np.float16)
out["box"] = c["box"]
np.savez_compressed(HERE / "data" / "arch_thumbs.npz", **out)
print("saved", {k: getattr(v, "shape", v) for k, v in out.items() if hasattr(v, "shape")})
