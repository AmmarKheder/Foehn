"""Tests of flexgreen.py: (a) exact advection in a uniform wind; (b) Thomson's well-mixed criterion of the boundary-layer
turbulence (unstable, neutral, stable); (c) real ERA5 fields: mixing heights and a backward cloud from Beijing."""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km")
import flexgreen as fg
dev = torch.device("cuda"); gen = torch.Generator(device=dev); gen.manual_seed(0)
# (a) uniform wind 10 m/s eastward everywhere (all levels and 10 m), w = 0
S = 33
f = torch.zeros(S, fg.NCH, 181, 360, device=dev)
f[:, fg.C_U:fg.C_U + 8] = 10.0; f[:, fg.C_U10] = 10.0
f[:, fg.C_H:fg.C_H + 8] = torch.tensor([100, 750, 1500, 3000, 4200, 5600, 7200, 9200.], device=dev).view(1, 8, 1, 1)
f[:, fg.C_PBL] = 1000.0; f[:, fg.C_UST] = 0.3; f[:, fg.C_WST] = 0.0
fld = fg.Field(f)
for lat0 in (0.0, 60.0):
    lon = torch.zeros(1000, device=dev); lat = torch.full((1000,), lat0, device=dev); z = torch.full((1000,), 50.0, device=dev)
    t_rel = torch.full((1000,), 48.0, device=dev)
    lo, la, zr, sv, _, _ = fg.backward(fld, None, lon, lat, z, t_rel, 24.0, gen, turb=False, removal=False)
    exp_ = -10 * 86400 / (fg.R_E * math.cos(math.radians(lat0))) * 180 / math.pi
    print(f"(a) lat {lat0:4.0f}: dlon {float(lo.mean()):8.4f} (expected {exp_:8.4f}) dlat {float(la.mean() - lat0):.2e} survival {float(sv.mean()):.3f}")
# (b) well-mixed criterion: start uniform in [0, h], 1 day of turbulence only, histogram must stay flat
for name, h, ust, il, wst in (("unstable", 1500., 0.4, -1 / 50., 2.0), ("neutral", 800., 0.4, 0.0, 0.0), ("stable", 200., 0.2, 1 / 50., 0.0)):
    n = 400000
    H = torch.full((n,), h, device=dev); U = torch.full((n,), ust, device=dev); IL = torch.full((n,), il, device=dev); W = torch.full((n,), wst, device=dev)
    z = torch.rand(n, generator=gen, device=dev) * h; lon = torch.zeros(n, device=dev); lat = torch.zeros(n, device=dev)
    zs = torch.zeros(n, device=dev)                                                     # a second set released at the ground
    t0 = time.time()
    for k in range(96):
        z, lon, lat = fg.turbulence(z, lon, lat, H, U, IL, W, gen)
        zs, _, _ = fg.turbulence(zs, lon * 0, lat * 0, H, U, IL, W, gen)
        if k in (3, 11):
            print(f"    {name}: after {(k + 1) * 15:4d} min, ground release mean z/h {float((zs / h).mean()):.3f} (well mixed 0.5)")
    hist = torch.histc(z / h, bins=10, min=0, max=1).cpu().numpy() / n * 10
    kbar = torch.stack([fg.kz(H[:1] * q, H[:1], U[:1], IL[:1], W[:1]) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]).mean()
    print(f"(b) {name:8s} h {h:5.0f} K_mean {float(kbar):7.1f} m2/s, h^2/K {float(h * h / kbar):7.0f} s | uniform start after 24 h, "
          f"bin density (1 = uniform): {np.round(hist, 3)} | max dev {np.abs(hist - 1).max():.3f} | ground start mean z/h {float((zs / h).mean()):.3f} | {time.time() - t0:.1f}s")
# (c) real ERA5 fields of 1 January 2017 (+8 days), if downloaded
chk = np.load(fg.WB / "era5_B_2017_01.npy", mmap_mode="r")
if not np.isfinite(np.asarray(chk[8, 3, 0], np.float32)).all() or float(np.abs(np.asarray(chk[8, 3, 0], np.float32)).max()) == 0:
    print("(c) skipped: ERA5 January 2017 not downloaded yet"); sys.exit()
zs_, z0 = fg.static(dev)
B = torch.tensor(fg.load_run("era5", 2017, 0), device=dev)
fd = fg.derive(B, zs_, z0)
h = fd[:, fg.C_PBL]
lat = torch.tensor(90.0 - np.arange(181), device=dev).view(181, 1).expand(181, 360); lon = torch.tensor(-180.0 + np.arange(360), device=dev).view(1, 360).expand(181, 360)
def at(la, lo):
    return int(90 - la), int(lo + 180)
for nm, (la, lo) in {"Beijing": (40, 116), "Sahara (Niger)": (17, 8), "Delhi": (29, 77), "Paris": (49, 2), "S. Pacific": (-20, -130)}.items():
    r, c = at(la, lo); hv = h[:12, r, c].cpu().numpy(); ws = fd[:12, fg.C_WST, r, c].cpu().numpy(); us = fd[:12, fg.C_UST, r, c].cpu().numpy()
    print(f"(c) {nm:15s} mixing height (m) 00/06/12/18 UTC x3 days: {np.round(hv).astype(int)} | u* {np.round(us[:4], 2)} | w* {np.round(ws[:4], 2)}")
print("(c) global mixing height percentiles (5/50/95 %):", np.percentile(h.cpu().numpy(), [5, 50, 95]).round(0))
fld = fg.Field(fd)
n = 2000
lo = torch.full((n,), 116.4, device=dev) + (torch.rand(n, generator=gen, device=dev) - .5) * .5
la = torch.full((n,), 39.9, device=dev) + (torch.rand(n, generator=gen, device=dev) - .5) * .5
z = torch.rand(n, generator=gen, device=dev) * 100
t_rel = 48.0 + 24 * torch.rand(n, generator=gen, device=dev)
for turb in (False, True):
    lo2, la2, zr, sv, _, _ = fg.backward(fld, None, lo, la, z, t_rel, 48.0, gen, turb=turb)
    print(f"(c) Beijing 2 days back, turbulence {turb}: origin lon {float(lo2.mean()):.2f} +- {float(lo2.std()):.2f}, lat {float(la2.mean()):.2f} +- {float(la2.std()):.2f}, "
          f"z/hmax median {float(zr.median()):.2f}, in mixed layer {float((zr < 1).float().mean()):.2f}, survival {float(sv.mean()):.3f}")
