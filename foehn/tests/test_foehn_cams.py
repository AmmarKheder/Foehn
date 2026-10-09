#!/usr/bin/env python3
"""Checks of foehn_cams.py without the data (CPU): geometry of every resampling (weather 1 deg, fires 0.25 deg, receptors
0.5 deg, tracers 0.4 deg) on fields equal to their own latitude / longitude, then a forward pass and the loss of CamsModule
on a synthetic batch of the real shapes.
    tests/cpu_container.sh "python3 tests/test_foehn_cams.py" """
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import foehn_cams as fc  # noqa: E402

torch.manual_seed(0)
net = fc.CamsNet(emis=True)
LAT04 = torch.tensor(84.0 - 0.4 * np.arange(fc.PY), dtype=torch.float32)[:, None].expand(fc.PY, fc.PX)
LON04 = torch.tensor(-180.0 + 0.4 * np.arange(fc.PX), dtype=torch.float32)[None].expand(fc.PY, fc.PX)
inside = (LAT04 >= -59.6) & (LAT04 <= 83.8)                 # rows inside every source grid (receptors: 83.875 .. -59.625)
lonw = ((LON04 + 180.0) % 360.0) - 180.0


def field(lat, lon):
    """(1, 3, n, m): latitude, cos(lon), sin(lon) on a source grid"""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return torch.tensor(np.stack([la, np.cos(np.deg2rad(lo)), np.sin(np.deg2rad(lo))])[None], dtype=torch.float32)


def check(name, f, g, tol):
    s = net.smp(f, g)[0]
    e_lat = (s[0] - LAT04)[inside].abs().max().item()
    e_lon = torch.sqrt((s[1] - torch.cos(torch.deg2rad(lonw))) ** 2 + (s[2] - torch.sin(torch.deg2rad(lonw))) ** 2)[inside].max().item()
    print(f"{name}: max |lat error| {e_lat:.4f} deg, max lon error {np.rad2deg(e_lon):.4f} deg")
    assert e_lat < tol and np.rad2deg(e_lon) < tol, name


check("weather 1 deg", field(90.0 - np.arange(181), -180.0 + np.arange(360)), net.g_wx, 0.01)
check("fires 0.25 deg", field(84.0 - 0.25 * np.arange(576), -180.0 + 0.25 * np.arange(1440)), net.g_fire, 0.01)
check("receptors 0.5 deg", field(84.0 - 0.125 - 0.5 * np.arange(288), -180.0 + 0.125 + 0.5 * np.arange(720)), net.g_rec, 0.01)
# gather on the 0.4 deg tracer grid: particles at random positions, tracer = own latitude / cos, sin of longitude
f = field(84.0 - 0.4 * np.arange(fc.NY), -180.0 + 0.4 * np.arange(fc.NX))
R, Q, P = 4, 5, 6
lon = torch.rand(1, R, Q, P) * 360.0 - 180.0
lat = torch.rand(1, R, Q, P) * 140.0 - 58.0
org = torch.stack([lon, lat, torch.full_like(lon, 0.5), torch.ones_like(lon)], -1)
for p in range(P):                                          # one particle per receptor at a time: exact values
    g = fc.gather(f, org[:, :, :, p:p + 1], 84.0, 0.4)[0]
    assert (g[0] - lat[0, :, :, p]).abs().max() < 0.01, "gather latitude"
    assert (g[1] - torch.cos(torch.deg2rad(lon[0, :, :, p]))).abs().max() < 1e-3, "gather longitude"
    assert (g[3] == 0).all() and (g[4] == 1).all(), "aloft / survival"
print("gather on the 0.4 deg grid: exact at single particles")
# forward + loss on a synthetic batch of the real shapes
B = 1
b = {"fine": torch.randn(B, 2 * fc.NCH + fc.NST, fc.PY, fc.PX).half(), "x1": torch.randn(B, 1, fc.PY, fc.PX),
     "mask": torch.ones(B, 1, fc.PY, fc.PX), "lead": torch.full((B,), 2.5), "pm0": torch.randn(B, 1, fc.NY, fc.NX),
     "wx": torch.randn(B, fc.NWX, 181, 360), "fire": torch.randn(B, 3, 576, 1440).half(),
     "org": torch.stack([torch.rand(B, fc.NR, fc.NC, 16) * 360 - 180, torch.rand(B, fc.NR, fc.NC, 16) * 140 - 58,
                         torch.rand(B, fc.NR, fc.NC, 16) * 2, torch.rand(B, fc.NR, fc.NC, 16)], -1).half(),
     "fsens": torch.rand(B, 3, fc.NR, fc.NC), "esens": torch.rand(B, fc.NE, fc.NR, fc.NC)}
m = fc.CamsModule(emis=True)
print(f"parameters {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
t0 = time.time()
with torch.no_grad():
    loss = m.step(b)
print(f"loss {loss.item():.3f} (zero-initialised output: about 2 = E|x1 - x0|^2), forward {time.time() - t0:.0f}s on CPU")
assert 1.5 < loss.item() < 2.5
x = fc.rewrap(torch.randn(1, 1, fc.PY, fc.PX))
assert torch.equal(x[..., fc.NX:], x[..., :fc.PX - fc.NX])
print("all checks passed")
