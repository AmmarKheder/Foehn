"""CPU checks of the CRPS fine-tuning (fullflow_global_green.py --crps-steps): the fair CRPS against an exact value, its
unbiasedness for M = 2, and the gradient through K checkpointed Euler steps equal to the one without checkpointing.
    tests/cpu_container.sh "python tests/test_crps_finetune.py\""""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fullflow_global_green import FullModuleGreen, fair_crps  # noqa: E402

torch.manual_seed(0)
# 1. fair CRPS of two members: 0.5 (|a - y| + |b - y|) - 0.5 |a - b|
a, b, y = torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([2.5])
assert torch.allclose(fair_crps([a, b], y), torch.tensor([0.5 * (1.5 + 0.5) - 0.5 * 2.0])), fair_crps([a, b], y)
# 2. unbiased: mean over draws of the 2-member fair CRPS of N(0, 1) members against y = 0.7 = CRPS of N(0, 1) at 0.7
from math import erf, exp, pi, sqrt  # noqa: E402
z = 0.7
exact = z * erf(z / sqrt(2)) + 2 * exp(-z * z / 2) / sqrt(2 * pi) - 1 / sqrt(pi)
X = torch.randn(2, 400000)
est = fair_crps([X[0], X[1]], torch.full((400000,), z)).mean().item()
assert abs(est - exact) < 0.005, (est, exact)
# 3. gradient through K checkpointed Euler steps = gradient without checkpointing (a small linear velocity stands in)
m = FullModuleGreen(shift=25.0, shift_mix=0.5, crps_steps=4, crps_members=2, psub_member=2)
lin = torch.nn.Conv2d(2, 1, 3, padding=1)
m.velocity = lambda x, b_, tau: lin(torch.cat([x, b_["fine"][:, :1]], 1)) * (1 + tau.view(-1, 1, 1, 1))
batch = {"x1": torch.randn(1, 1, 8, 8), "mask": torch.ones(1, 1, 8, 8), "fine": torch.randn(1, 6, 8, 8),
         "org": torch.randn(1, 4, 4, 4, 4)}
torch.manual_seed(1); l1 = m.crps_loss(batch, grad=True); g1 = torch.autograd.grad(l1, lin.weight)[0]
import fullflow_global_green as FG  # noqa: E402
FG.checkpoint = lambda f, *args, **kw: f(*args)
torch.manual_seed(1); l2 = m.crps_loss(batch, grad=True); g2 = torch.autograd.grad(l2, lin.weight)[0]
assert torch.allclose(l1, l2) and torch.allclose(g1, g2, atol=1e-6), (l1, l2, (g1 - g2).abs().max())
print(f"ok: fair CRPS exact; unbiased ({est:.4f} vs {exact:.4f}); checkpointed gradient = plain gradient "
      f"(|g| {g1.abs().mean():.3e}); tau grid K=4: {np.round(m.tau_grid(4), 4)}")
