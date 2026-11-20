"""env_check.py -- checks of the cdffm environment (inside the SIF, after `source env.sh`; jobs/env_check_job.sh).

  python env_check.py gpu                  1 process: versions, GCD visible, CD-FFM's model (paper config
                                           w2e_static_pl24_l8, native 71 x 113 ERA5 grid) forward/backward on
                                           random data with CD-FFM's own loss code, CPU/GPU agreement, ms/step
  srun -n 2 ... env_check.py ddp           torch.distributed over RCCL on 2 GCDs: all_reduce, bandwidth, 3 DDP
                                           steps of that model; gradients and weights identical across ranks
  srun -n 2 ... env_check.py lightning     Lightning 2.4 DDP (SLURMEnvironment) fit of CD-FFM's FastFFM, 3 steps

Random data only: nothing is read from or written to the project folders.
"""
import copy
import os
import sys
import time
from types import SimpleNamespace

# CD-FFM's paths.py only builds path strings from CDFFM_ROOT; point it at a throw-away place anyway.
os.environ.setdefault("CDFFM_ROOT", os.path.join(os.environ.get("CDFFM_TMP", "/tmp"), "cdffm_root_envcheck"))
sys.path.insert(0, os.environ.get(
    "CDFFM_SCRIPTS", "/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/CD-FFM/scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# CD-FFM's final model w2e_static_pl24_l8 (configs/train/w2e_static_pl24_l8.json), batch as in the paper.
ARGS = SimpleNamespace(
    name="envcheck", backbone="flower", objective="fm", noise="field", noise_ls=0.5, noise_features=1024,
    noise_kernel="powerlaw", noise_beta=2.4, noise_lmax_km=500.0, noise_lmin_km=8.0, lifting_dim=None,
    mc_weight=0.0, doy_cond=True, static="clim", target="temp", holdout=None, batch_size=16, n_scattered=2048,
    n_clusters=16, points_per_cluster=32, lr=3e-4, warmup=1000, max_steps=60000, ema=0.999, grad_clip=1.0,
    val_every=3, val_members=8, val_ode_steps=10, ckpt_every=5000, seed=0, teacher=None)
H, W = 71, 113  # CD-FFM's Nordic ERA5 0.25 deg grid


class RandomData:
    """Stands in for CD-FFM's GPUData: same batch keys and shapes, random values."""

    def __init__(self, device, seed=0):
        self.device = device
        self.era5_lat = np.linspace(54.75, 72.25, H).astype(np.float32)
        self.era5_lon = np.linspace(4.0, 32.0, W).astype(np.float32)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)

    def sample_batch(self, B, n_scattered, n_clusters, points_per_cluster, generator=None):
        dev, g = self.device, self.g
        N = n_scattered + n_clusters * points_per_cluster
        x1 = torch.randn(B, N, 3, device=dev, generator=g)
        return dict(lat=54.75 + 17.5 * torch.rand(B, N, device=dev, generator=g),
                    lon=4.0 + 28.0 * torch.rand(B, N, device=dev, generator=g),
                    x1=x1, valid=torch.ones_like(x1, dtype=torch.bool),
                    era5=torch.randn(B, 4, H, W, device=dev, generator=g),
                    doy=torch.randint(1, 366, (B,), device=dev, generator=g),
                    static=torch.randn(B, N, 17, device=dev, generator=g))


def flat_params(model, grads=False):
    ps = [p for p in model.parameters() if p.requires_grad]
    if grads:
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float()
                          for p in ps])
    return torch.cat([p.detach().reshape(-1).float() for p in ps])


def mode_gpu():
    import einops
    import lightning_utilities
    import matplotlib
    import pytorch_lightning as pl
    import scipy
    import tensorboard
    import torchmetrics
    from fast_train import build_model, compute_losses, condition

    print(f"python {sys.version.split()[0]} ({sys.executable})")
    print(f"torch {torch.__version__}  hip {torch.version.hip}  rccl {torch.cuda.nccl.version()}  "
          f"pytorch_lightning {pl.__version__}  torchmetrics {torchmetrics.__version__}  "
          f"lightning_utilities {lightning_utilities.__version__}")
    print(f"numpy {np.__version__}  scipy {scipy.__version__}  einops {einops.__version__}  "
          f"matplotlib {matplotlib.__version__}  tensorboard {tensorboard.__version__}")
    print(f"pytorch_lightning from {os.path.dirname(pl.__file__)}")
    print(f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}  HOME={os.environ.get('HOME')}  "
          f"XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')}  MIOPEN_USER_DB_PATH={os.environ.get('MIOPEN_USER_DB_PATH')}")
    assert torch.cuda.is_available(), "torch sees no GPU"
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} {getattr(p, 'gcnArchName', '?')} {p.total_memory / 2**30:.1f} GiB")
    dev = torch.device("cuda:0")
    a = torch.randn(2048, 2048, device=dev)
    err = (a @ a.T - (a.cpu() @ a.cpu().T).to(dev)).abs().max().item()
    print(f"matmul 2048^2 GPU vs CPU max abs diff {err:.2e}")

    # CPU/GPU agreement of the full conditioning + velocity path (same weights, same inputs).
    torch.manual_seed(0)
    data_cpu = RandomData(torch.device("cpu"), seed=1)
    m_cpu = build_model(ARGS, data_cpu.era5_lat, data_cpu.era5_lon).eval()
    m_gpu = copy.deepcopy(m_cpu).to(dev)
    b = data_cpu.sample_batch(2, 512, 0, 0)
    t = torch.rand(2, 512)
    with torch.no_grad():
        outs = []
        for m, d in ((m_cpu, "cpu"), (m_gpu, dev)):
            bb = {k: v.to(d) for k, v in b.items()}
            cond = condition(m, bb["era5"], bb["lat"], bb["lon"], bb["doy"], True, bb["static"])
            outs.append(m.velocity(bb["x1"], cond, t.to(d)).cpu())
    rel = ((outs[0] - outs[1]).abs().max() / outs[0].abs().max()).item()
    print(f"CD-FFM velocity CPU vs GPU: max rel diff {rel:.2e} " + ("OK" if rel < 1e-3 else "MISMATCH"))

    # Training steps with CD-FFM's own loss code (compute_losses, objective fm, power-law noise field).
    data = RandomData(dev, seed=0)
    model = build_model(ARGS, data.era5_lat, data.era5_lon).to(dev)
    print(f"CD-FFM model params {sum(p.numel() for p in model.parameters()):,} "
          f"(B={ARGS.batch_size}, N={ARGS.n_scattered + ARGS.n_clusters * ARGS.points_per_cluster} points/day)")
    opt = torch.optim.AdamW(model.parameters(), lr=ARGS.lr, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats(dev)
    times, losses = [], []
    for it in range(23):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        main, _ = compute_losses(model, data, ARGS)
        opt.zero_grad(set_to_none=True)
        main.backward()
        if it == 0:
            for name in ("encoder", "query", "velocity_net"):
                gs = [p.grad for p in getattr(model, name).parameters()]
                ok = all(g is not None and torch.isfinite(g).all() for g in gs)
                gn = torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)).item()
                print(f"  grad {name:12s} finite={ok} norm={gn:.3e} " + ("OK" if ok and gn > 0 else "BAD"))
        torch.nn.utils.clip_grad_norm_(model.parameters(), ARGS.grad_clip)
        opt.step()
        torch.cuda.synchronize(dev)
        times.append(time.time() - t0)
        losses.append(main.item())
    assert all(np.isfinite(losses)), losses
    print(f"loss step0 {losses[0]:.4f} -> step22 {losses[-1]:.4f} (random targets: ~2 expected)")
    print(f"first step {times[0]:.1f} s (MIOpen kernel compile), then {1000 * np.median(times[3:]):.0f} ms/step "
          f"(median of 20), peak memory {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
    print("GPU CHECK PASSED")


def _dist_setup():
    import torch.distributed as dist
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    t0 = time.time()
    dist.init_process_group("nccl", rank=rank, world_size=world)
    x = torch.ones(1, device=dev)
    dist.all_reduce(x)
    torch.cuda.synchronize(dev)
    print(f"[rank {rank}/{world}] init_process_group + first all_reduce {time.time() - t0:.1f} s, "
          f"backend {dist.get_backend()}, cuda:{local} = {torch.cuda.get_device_name(dev)}, "
          f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}, host {os.uname().nodename}", flush=True)
    return dist, rank, world, local, dev


def _same_across_ranks(dist, vec, world):
    got = [torch.empty_like(vec) for _ in range(world)]
    dist.all_gather(got, vec)
    return max((g - got[0]).abs().max().item() for g in got)


def mode_ddp():
    from torch.nn.parallel import DistributedDataParallel as DDP

    from fast_train import build_model, compute_losses
    dist, rank, world, local, dev = _dist_setup()

    x = torch.full((1 << 20,), float(rank + 1), device=dev)
    dist.all_reduce(x)
    want = world * (world + 1) / 2
    ok_sum = bool(torch.all(x == want))
    big = torch.ones(64 << 20, device=dev)  # 256 MB fp32
    for _ in range(2):
        dist.all_reduce(big)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(big)
    torch.cuda.synchronize(dev)
    dt = (time.time() - t0) / 5
    algbw = big.numel() * 4 / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world

    class LossWrap(torch.nn.Module):  # DDP needs the loss computed inside forward()
        def __init__(self, model, data):
            super().__init__()
            self.model, self.data = model, data

        def forward(self, _unused):  # torch 2.4 DDP with device_ids fails on a forward() without inputs
            return compute_losses(self.model, self.data, ARGS)[0]

    torch.manual_seed(0)
    data = RandomData(dev, seed=100 + rank)  # each rank its own random batches
    model = build_model(ARGS, data.era5_lat, data.era5_lon).to(dev)
    ddp = DDP(LossWrap(model, data), device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=ARGS.lr, weight_decay=0.0)
    losses, gdiff = [], None
    dummy = torch.zeros(1, device=dev)
    for step in range(3):
        loss = ddp(dummy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            gdiff = _same_across_ranks(dist, flat_params(model, grads=True), world)
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize(dev)
    wdiff = _same_across_ranks(dist, flat_params(model), world)
    lt = torch.tensor(losses[0], device=dev)
    lall = [torch.empty_like(lt) for _ in range(world)]
    dist.all_gather(lall, lt)
    if rank == 0:
        print(f"all_reduce sum of ranks: {'OK' if ok_sum else 'WRONG'}; 256 MB all_reduce {dt * 1000:.1f} ms "
              f"(algbw {algbw:.0f} GB/s, busbw {busbw:.0f} GB/s)")
        print(f"DDP 3 steps: step-0 losses per rank {[round(v.item(), 4) for v in lall]} (different batches), "
              f"grad max diff across ranks {gdiff:.1e}, weight max diff after 3 steps {wdiff:.1e}")
        ok = ok_sum and gdiff == 0 and wdiff == 0 and all(np.isfinite(losses))
        print("DDP CHECK PASSED" if ok else "DDP CHECK FAILED")
    dist.barrier()
    dist.destroy_process_group()


def mode_lightning():
    import pytorch_lightning as pl

    from fast_train import FastFFM

    class CrossRank(pl.Callback):  # at the end of fit, while the process group is surely alive
        def on_train_end(self, trainer, pl_module):
            out = {}
            for k, vec in (("model", flat_params(pl_module.model)),
                           ("ema", torch.cat([p.detach().reshape(-1) for p in pl_module.ema.parameters()]))):
                g = pl_module.all_gather(vec)  # (world, n)
                out[k] = (g - g[0]).abs().max().item()
            self.diff = out

    rank, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_LOCALID"])
    dev = torch.device("cuda", local)
    torch.manual_seed(0)
    data = RandomData(dev, seed=200 + rank)
    module = FastFFM(ARGS, data)
    cb = CrossRank()
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ["SLURM_NTASKS_PER_NODE"]), num_nodes=1,
                         strategy="ddp", max_steps=3, limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[cb])
    t0 = time.time()
    trainer.fit(module)
    env = type(trainer.strategy.cluster_environment).__name__
    print(f"[rank {trainer.global_rank}/{trainer.world_size}] Lightning {pl.__version__} "
          f"{type(trainer.strategy).__name__} env={env} root_device={trainer.strategy.root_device} "
          f"steps={trainer.global_step} fit {time.time() - t0:.1f} s", flush=True)
    if trainer.global_rank == 0:
        ok = (env == "SLURMEnvironment" and trainer.world_size == 2 and trainer.global_step == 3
              and cb.diff["model"] == 0 and cb.diff["ema"] == 0)
        print(f"weights max diff across ranks {cb.diff['model']:.1e}, EMA {cb.diff['ema']:.1e}")
        print("LIGHTNING CHECK PASSED" if ok else "LIGHTNING CHECK FAILED")


if __name__ == "__main__":
    {"gpu": mode_gpu, "ddp": mode_ddp, "lightning": mode_lightning}[sys.argv[1]]()
