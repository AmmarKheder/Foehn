"""convergence of the boundary-layer random-displacement model with the number of sub-steps (well-mixed criterion)"""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km")
import flexgreen as fg
dev = torch.device("cuda"); gen = torch.Generator(device=dev); gen.manual_seed(1)
for name, h, ust, il, wst in (("unstable", 1500., 0.4, -1 / 50., 2.0), ("unstable_shallow", 600., 0.3, -1 / 20., 1.2), ("neutral", 800., 0.4, 0.0, 0.0), ("stable", 200., 0.2, 1 / 50., 0.0)):
    for ns in (6, 18, 36, 90):
        n = 400000
        H = torch.full((n,), h, device=dev); U = torch.full((n,), ust, device=dev); IL = torch.full((n,), il, device=dev); W = torch.full((n,), wst, device=dev)
        z = torch.rand(n, generator=gen, device=dev) * h; lon = torch.zeros(n, device=dev); lat = torch.zeros(n, device=dev)
        torch.cuda.synchronize(); t0 = time.time()
        for k in range(96):
            z, lon, lat = fg.turbulence(z, lon, lat, H, U, IL, W, gen, nsub=ns)
        torch.cuda.synchronize()
        hist = torch.histc(z / h, bins=20, min=0, max=1).cpu().numpy() / n * 20
        print(f"{name:16s} nsub {ns:3d} ({900 / ns:5.1f} s) | density lowest 3 of 20 bins {np.round(hist[:3], 3)} top {hist[-1]:.3f} | max dev {np.abs(hist - 1).max():.3f} | {time.time() - t0:.1f}s / day / 0.4M", flush=True)
