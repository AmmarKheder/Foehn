#!/usr/bin/env python3
"""Discretization error of the sampler of eval_global_green.py (CPU). Gaussian toy problem with the exact velocity of
the linear path (x1 ~ N(MU, SIG^2), x0 ~ N(0, 1), x_t = t x1 + (1 - t) x0): the ODE preserves the marginals exactly, so
any shrinkage of the sampled standard deviation is discretization error. Euler (the sampler used so far) against Heun
(second order) on Foehn's quantile grid, at matched numbers of network evaluations.
    tests/cpu_container.sh "python3 tests/test_sampler_eta.py" """
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_global_green as eg  # noqa: E402

MU, SIG = 0.7, 0.4


class Toy:
    def __init__(self, shift=1.0, mix=0.5):
        self.shift, self.mix = shift, mix

    def tau_grid(self, steps):                            # FullModule.tau_grid (quantiles of the training tau mixture)
        q = np.linspace(0, 1, steps + 1)
        if self.shift == 1.0:
            return q
        t = np.linspace(0, 1, 100001); y = 1 - t
        cdf = self.mix * (1 - y / (self.shift - (self.shift - 1) * y)) + (1 - self.mix) * t
        return np.interp(q, cdf, t)

    def velocity(self, x, b, tau):
        t = float(tau[0]); var = t * t * SIG ** 2 + (1 - t) ** 2
        return MU + t * SIG ** 2 / var * (x - t * MU) - (1 - t) / var * (x - t * MU)    # E[x1 | x_t] - E[x0 | x_t]


g = torch.Generator().manual_seed(0)
print(f"target: mean {MU}, std {SIG}  (network evaluations in brackets)")
for shift in (1.0, 25.0):
    for solver, steps, tail in (("euler", 8, 3), ("heun", 5, 2), ("heun", 8, 3), ("euler", 30, 1), ("euler", 60, 3)):
        m = Toy(shift)
        n = len(eg.flow_grid(m, steps, tail)) - 1
        nfe = n if solver == "euler" else 2 * n - 1
        out = eg.sample(m, None, torch.randn(1, 1, 300, 1000, generator=g), steps, tail, solver)
        print(f"grid shift {shift:4.0f} {solver:5s} {steps:2d} steps + tail {tail} [{nfe:3d}]: mean {out.mean():.3f} std {out.std():.3f}"
              f" ({100 * (out.std() / SIG - 1):+.1f} %)")
