"""cdffm_aq.bench_s1 -- S1 encoder benchmark of PLAN.md section 7 (dev-g, 1 GCD, <= 30 min; jobs/bench_encoder.sh).

`run` (one process per grid x checkpointing pair): for each B in --batches, the full training step of train_aq
(objective fm, train_aq defaults: fp32, N = 16,384 points per day, AdamW 3e-4, no clip, EMA 0.999) on real smoke
batches. train_aq.compute_losses is called as is:
  47-channel input assembly -> whole-globe periodic Flower encoder (576 x 1440 at 0.25 deg, 288 x 720 at 0.5 deg)
  -> periodic LIIF query + doy at the N points -> sphere noise -> velocity MLP -> loss -> backward -> pre-clip
  gradient norm (as train_aq.on_before_optimizer_step) -> AdamW -> EMA.
  * s/step = wall time of a CUDA-synchronised block of steps / steps (throughput, like the training log's ms/step).
    Per-step median/min/max and the phase split come from CUDA events, with no host synchronisation inside the block:
    enc = H2D copy + input assembly + encoder forward; query = LIIF + noise + velocity + loss; bwd = backward;
    opt = gradient norm + AdamW + EMA.
  * Memory: max_memory_allocated / max_memory_reserved over the timed steps. The cache is emptied and the peaks are
    reset after the warm-up steps; the first warm-up step includes MIOpen's kernel search and compilation. An OOM is
    caught and recorded.
  * Batches: --n-batches distinct real batches, drawn once by BatchStream (seed 0) and kept in pinned host memory.
    They are copied to the GCD at every step, as BatchStream.next does. Not included: the disk read (prefetch thread;
    'data wait' in the train_aq log) and Lightning's per-step overhead.
  * Informational extras: --extra-n adds one config at B = --profile-b with more points per day; --profile-b
    profiles 2 steps (top GPU kernels).
`gate`: the PLAN.md S1 rule, keep 0.25 deg if B = 2 fits and takes <= 1.2 s/step, otherwise 0.5 deg.
  * B = 2 is taken without checkpointing if that fits, else with it (PLAN.md 5: checkpointing rather than a smaller
    batch).
  * 'Fits' = no OOM here AND the projected full-run peak fits the GCD. The projection is the reserved memory, minus
    the smoke per-pixel arrays, plus the global ones (115,203,319 pixels x 66 bytes in data_aq.PixelArrays = 7.1 GiB),
    plus a 2 GiB margin (HIP context, RCCL buffers, validation).
  * It writes the rows, the decision and the cost projection to one JSON, and appends the decision to
    SMOKE_STATUS.md.

  python -m cdffm_aq.bench_s1 run --grid 025 [--ckpt] --batches 1,2,4 --out RESULTS.jsonl
  python -m cdffm_aq.bench_s1 gate --results RESULTS.jsonl --json logs/s1_bench.json [--status logs/SMOKE_STATUS.md]
"""
import argparse
import copy
import datetime as dt
import gc
import json
import os
import socket
import sys
import time
import traceback

import numpy as np
import torch

from . import NOISE_FIT_SMOKE, PROJECT
from .data_aq import BatchStream, PixelArrays, in_channels, open_store
from .model_aq import build_model
from .train_aq import compute_losses, model_grid, resolve_noise

GIB = 2 ** 30
GATE_B = 2
GATE_S_PER_STEP = 1.2
FULL_RUN_PIXELS = 115_203_319                       # LGHAP batch-B common mask (PLAN.md 2.3)
FULL_RUN_PIXEL_BYTES = 4 + 4 + 4 + 20 + 20 + 2 * 7   # PixelArrays: lat, lon, w_area f32; clim mean, var (5) f32; s1-s7 f16
MARGIN_GIB = 2.0
MAIN_RUN_STEPS = 60000                              # PLAN.md 5 budget table
GPU_H_PER_NODE_H = 4                                # LUMI-G billing, 8 GCDs = 4 GPUs (PLAN.md 5)
PHASES = ("enc", "query", "bwd", "opt")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def run_args(grid_mode, ckpt):
    """the train_aq arguments that build_model / compute_losses read (train_aq defaults; noise = the smoke fit)"""
    a = argparse.Namespace(objective="fm", grid=grid_mode, transport="t6", doy_cond=True, lifting_dim=None,
                           use_checkpoint=ckpt, noise_fit=str(NOISE_FIT_SMOKE), noise_features=None, noise_beta=None,
                           noise_lmin_km=None, noise_lmax_km=None)
    return a, resolve_noise(a)


def draw_batches(store, n, B, N):
    """n real batches of B days x N points (BatchStream on the CPU side, seed 0, rank 0). The prefetch thread is
    joined, so that it is no longer reading the disk while the steps are timed."""
    bs = BatchStream(store, B, N, 0, 0, torch.device("cpu"))
    try:
        return [bs.next() for _ in range(n)]
    finally:
        bs.close()
        bs.thread.join(timeout=300)


def slice_batch(b, B, N, pin):
    """first B days and N points of a drawn batch, contiguous (and pinned for asynchronous copies)"""
    p = (lambda x: x.contiguous().pin_memory()) if pin else (lambda x: x.contiguous())
    return dict(raw={k: p(v[:B]) for k, v in b["raw"].items()}, k=p(b["k"][:B, :N]), v=p(b["v"][:B, :N]),
                day=b["day"][:B].clone(), doy=p(b["doy"][:B]))


def to_device(hb, dev):
    """as BatchStream.next: non-blocking copies from pinned host memory on the current stream"""
    nb = dev.type == "cuda"
    mv = lambda x: x.to(dev, non_blocking=nb)
    return dict(raw={k: mv(v) for k, v in hb["raw"].items()}, k=mv(hb["k"]), v=mv(hb["v"]), day=hb["day"],
                doy=mv(hb["doy"]))


class Stepper:
    """train_aq's step without Lightning: AQModule.training_step (compute_losses), backward,
    on_before_optimizer_step (pre-clip gradient norm), AdamW (wd 0, no clip), on_train_batch_end (EMA 0.999)"""

    def __init__(self, store, pix, grid_mode, ckpt, dev, seed=0):
        self.args, self.spec = run_args(grid_mode, ckpt)
        self.pix, self.dev = pix, dev
        torch.manual_seed(seed)
        self.grid = model_grid(store.name, grid_mode)
        self.model = build_model(self.args, in_channels("t6"), store.static_dim, self.grid).to(dev).train()
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=3e-4,
                                     weight_decay=0.0)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self._ev = None
        self.model.encoder.register_forward_hook(self._after_encoder)

    def _after_encoder(self, module, inputs, output):
        if self._ev is not None:
            self._ev.record()

    def step(self, hb, ev=None):
        """ev: 5 CUDA events (start, encoder done, loss done, backward done, step done) or None"""
        if ev:
            ev[0].record()
        self._ev = ev[1] if ev else None
        loss = compute_losses(self.model, self.pix, to_device(hb, self.dev), self.args, self.spec, self.gen)
        if ev:
            ev[2].record()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if ev:
            ev[3].record()
        sq = torch.zeros((), device=self.dev)
        for p in self.model.parameters():
            if p.grad is not None:
                sq = sq + p.grad.detach().float().pow(2).sum()
        self.opt.step()
        with torch.no_grad():
            for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=1 - 0.999)
        if ev:
            ev[4].record()
        return loss.detach(), sq.sqrt()


def timed_steps(st, batches, n, dev):
    cuda = dev.type == "cuda"
    evs = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(n)] if cuda else [None] * n
    losses, gns = [], []
    sync(dev)
    t0 = time.perf_counter()
    for i in range(n):
        loss, gn = st.step(batches[i % len(batches)], evs[i])
        losses.append(loss)
        gns.append(gn)
    sync(dev)
    wall = time.perf_counter() - t0
    out = dict(steps=n, s_per_step=wall / n)
    if cuda:
        per = np.array([e[0].elapsed_time(e[4]) for e in evs]) / 1000.0
        ph = np.array([[e[j].elapsed_time(e[j + 1]) for j in range(4)] for e in evs]) / 1000.0
        out.update(step_s_median=float(np.median(per)), step_s_min=float(per.min()), step_s_max=float(per.max()),
                   phases_s=dict(zip(PHASES, ph.mean(0).tolist())))
    L, G = torch.stack(losses).float().cpu(), torch.stack(gns).float().cpu()
    out.update(loss_first=float(L[0]), loss_last=float(L[-1]), grad_norm_mean=float(G.mean()),
               finite=bool(torch.isfinite(L).all() and torch.isfinite(G).all()))
    return out


def profile_steps(st, batches, dev, n=2):
    """torch.profiler over n steps: the GPU kernels with the largest device time (per step)"""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    sync(dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(n):
            st.step(batches[i % len(batches)])
        sync(dev)
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        t = e.self_cuda_time_total if t is None else t
        if getattr(e, "device_type", None) == DeviceType.CUDA and t > 0:
            rows.append((t, e.key, e.count))
    rows.sort(key=lambda r: -r[0])
    tot = sum(r[0] for r in rows)
    return dict(steps=n, kernel_ms_per_step=tot / 1000.0 / n,
                top=[dict(kernel=k[:120], ms_per_step=t / 1000.0 / n, share=t / max(tot, 1e-9), calls_per_step=c // n)
                     for t, k, c in rows[:15]])


def bench_config(store, pix, grid_mode, ckpt, B, N, drawn, a, dev, prof, deadline):
    if B == a.fake_oom:
        raise torch.cuda.OutOfMemoryError(f"fake OOM at B={B} (--fake-oom, CPU logic test)")
    cuda = dev.type == "cuda"
    batches = [slice_batch(b, B, N, cuda) for b in drawn]
    st = Stepper(store, pix, grid_mode, ckpt, dev, seed=a.seed)
    n_par = sum(p.numel() for p in st.model.parameters())
    row = dict(grid=grid_mode, H=st.grid.H, W=st.grid.W, in_ch=in_channels("t6"), ckpt=ckpt, B=B, N=N, params=n_par)
    warm = []
    for i in range(a.warmup):
        t0 = time.perf_counter()
        st.step(batches[i % len(batches)])
        sync(dev)
        warm.append(time.perf_counter() - t0)
    row["warmup_s"] = warm
    if cuda:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    est = max(warm[-1], 1e-3)
    n = max(a.min_steps, min(a.steps, int(a.timed_budget_s / est), int((deadline - time.time() - 10.0) / est)))
    row.update(timed_steps(st, batches, n, dev))
    if cuda:
        free, total = torch.cuda.mem_get_info(dev)
        row.update(max_alloc_gib=torch.cuda.max_memory_allocated(dev) / GIB,
                   max_reserved_gib=torch.cuda.max_memory_reserved(dev) / GIB,
                   device_used_gib=(total - free) / GIB, device_total_gib=total / GIB)
    if prof:
        try:
            row["profile"] = profile_steps(st, batches, dev)
        except Exception as e:  # noqa: BLE001  (informational only)
            row["profile"] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
    row["status"] = "ok" if row["finite"] else "nonfinite"
    return row


def write_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def fmt_row(r):
    if r.get("status") not in ("ok", "nonfinite"):
        return (f"{r['grid']:>4} ckpt={'on' if r['ckpt'] else 'off':<3} B={r['B']} N={r['N']}: {r['status']} "
                f"{r.get('error', r.get('reason', ''))[:160]}")
    s = (f"{r['grid']:>4} ckpt={'on' if r['ckpt'] else 'off':<3} B={r['B']} N={r['N']}: {r['s_per_step']:.3f} s/step "
         f"over {r['steps']} steps (first step {r['warmup_s'][0]:.1f} s)")
    if "phases_s" in r:
        p = r["phases_s"]
        s += (f" | median {r['step_s_median']:.3f} [{r['step_s_min']:.3f}, {r['step_s_max']:.3f}] | enc {p['enc']:.3f} "
              f"query {p['query']:.3f} bwd {p['bwd']:.3f} opt {p['opt']:.3f} | peak {r['max_alloc_gib']:.2f} GiB "
              f"allocated, {r['max_reserved_gib']:.2f} reserved")
    return s + f" | loss {r['loss_first']:.3f} -> {r['loss_last']:.3f}, grad norm {r['grad_norm_mean']:.3g}"


def cmd_run(a):
    T0 = time.time()
    dev = torch.device("cuda", torch.cuda.current_device()) if (a.device != "cpu" and torch.cuda.is_available()) else torch.device("cpu")
    cuda = dev.type == "cuda"
    env = dict(status="env", grid=a.grid, ckpt=a.ckpt, node=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"),
               date=dt.datetime.now().isoformat(timespec="seconds"), torch=torch.__version__,
               hip=getattr(torch.version, "hip", None), device=torch.cuda.get_device_name(dev) if cuda else "cpu",
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
               miopen_find_mode=os.environ.get("MIOPEN_FIND_MODE"), store=a.store)
    store = open_store(a.store)
    pix = PixelArrays(store, dev)
    env["pix_gib"] = sum(t.numel() * t.element_size() for t in (pix.lat, pix.lon, pix.w_area, pix.clim_mean,
                                                                pix.clim_var, pix.static)) / GIB
    Bs = [int(x) for x in a.batches.split(",")]
    plan = []
    for B in Bs:
        plan.append((B, a.n_points, B == a.profile_b))
        if B == a.profile_b and a.extra_n:
            plan.append((B, a.extra_n, False))
    t1 = time.time()
    drawn = draw_batches(store, a.n_batches, max(Bs), max(a.n_points, a.extra_n))
    env.update(draw_s=time.time() - t1, setup_s=time.time() - T0, n_batches=a.n_batches,
               drawn_days=[[str(store.train_dates[int(t)]) for t in b["day"]] for b in drawn])
    write_row(a.out, env)
    print(f"[bench] {env['device']} torch {env['torch']} hip {env['hip']} cudnn.benchmark={env['cudnn_benchmark']} | "
          f"grid {a.grid} ckpt={a.ckpt} | per-pixel arrays {env['pix_gib']:.2f} GiB | {a.n_batches} batches of "
          f"{max(Bs)} days drawn in {env['draw_s']:.1f} s (setup {env['setup_s']:.1f} s)", flush=True)
    per_day = None
    for B, N, prof in plan:
        base = dict(grid=a.grid, ckpt=a.ckpt, B=B, N=N, pix_gib=env["pix_gib"])
        # minimum need of the next config: warm-up + minimum timed steps at the per-day cost seen so far + 30 s for
        # model set-up and kernel search at a new shape
        need = None if per_day is None else (a.warmup + a.min_steps) * per_day * B * max(1.0, N / a.n_points) + 30.0
        if need is not None and time.time() - T0 + need > a.budget_s:
            row = dict(base, status="skipped", reason=f"time budget: {time.time() - T0:.0f} s used of {a.budget_s:.0f}, "
                                                      f"about {need:.0f} s needed")
        else:
            try:
                row = dict(base, **bench_config(store, pix, a.grid, a.ckpt, B, N, drawn, a, dev, prof,
                                                T0 + a.budget_s))
            except torch.cuda.OutOfMemoryError as e:
                row = dict(base, status="oom", error=str(e).strip().splitlines()[0][:400])
                if cuda:
                    row["max_reserved_gib_at_oom"] = torch.cuda.max_memory_reserved(dev) / GIB
            except Exception as e:  # noqa: BLE001  (recorded; the next config still runs)
                traceback.print_exc()
                row = dict(base, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        if row["status"] == "ok":
            per_day = max(per_day or 0.0, row["s_per_step"] / B)
        row["elapsed_s"] = time.time() - T0
        write_row(a.out, row)
        print("[bench] " + fmt_row(row), flush=True)
        if "profile" in row:
            pr = row["profile"]
            if "top" in pr:
                print(f"[bench] profile B={B}: {pr['kernel_ms_per_step']:.0f} ms of kernels per step; top kernels:",
                      flush=True)
                for k in pr["top"][:10]:
                    print(f"[bench]   {k['ms_per_step']:8.1f} ms {100 * k['share']:5.1f} %  x{k['calls_per_step']:<4} "
                          f"{k['kernel'][:100]}", flush=True)
            else:
                print(f"[bench] profile failed: {pr.get('error')}", flush=True)
    print(f"[bench] done in {time.time() - T0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------------------------------------------

def projected_full_run_gib(r):
    """reserved peak - smoke per-pixel arrays + global per-pixel arrays + margin (None without device statistics)"""
    if "max_reserved_gib" not in r:
        return None
    return r["max_reserved_gib"] - r["pix_gib"] + FULL_RUN_PIXELS * FULL_RUN_PIXEL_BYTES / GIB + MARGIN_GIB


def fits(r):
    if r.get("status") != "ok":
        return False
    p = r.get("projected_full_run_gib")
    return p is None or p <= r["device_total_gib"]


def pick_b2(rows, grid, n_points):
    """B = 2 at the default N: without checkpointing if it fits, else with it; (row or None, measured rows)"""
    cands = [r for r in rows if r["grid"] == grid and r["B"] == GATE_B and r["N"] == n_points]
    for ck in (False, True):
        r = next((r for r in cands if r["ckpt"] == ck), None)
        if r is not None and fits(r):
            return r, cands
    return None, cands


def cost(s_per_step):
    node_h = MAIN_RUN_STEPS * s_per_step / 3600.0
    return dict(s_per_step=s_per_step, steps=MAIN_RUN_STEPS, node_h=node_h, gpu_h=node_h * GPU_H_PER_NODE_H,
                note="one 60,000-step run on 1 node x 8 GCDs (2 days per GCD); validation, data wait, Lightning and "
                     "the DDP all-reduce excluded")


def cmd_gate(a):
    rows = [json.loads(line) for line in open(a.results) if line.strip()]
    envs = [r for r in rows if r.get("status") == "env"]
    cfg = [r for r in rows if r.get("status") != "env"]
    for r in cfg:
        r["projected_full_run_gib"] = projected_full_run_gib(r)
        r["fits"] = fits(r)
    r25, c25 = pick_b2(cfg, "025", a.n_points)
    r05, c05 = pick_b2(cfg, "05", a.n_points)
    desc = lambda r: (f"B=2 checkpointing {'on' if r['ckpt'] else 'off'}: {r['s_per_step']:.3f} s/step, peak "
                      f"{r['max_alloc_gib']:.1f} GiB allocated / {r['max_reserved_gib']:.1f} reserved, projected full "
                      f"run {r['projected_full_run_gib']:.1f} of {r['device_total_gib']:.0f} GiB"
                      if "max_alloc_gib" in r else f"B=2 checkpointing {'on' if r['ckpt'] else 'off'}: "
                                                   f"{r['s_per_step']:.3f} s/step")
    if r25 is not None and r25["s_per_step"] <= GATE_S_PER_STEP:
        choice, chosen = "025", r25
        reason = f"0.25 deg kept: {desc(r25)} (<= {GATE_S_PER_STEP} s/step)"
    elif r25 is not None or c25:
        choice, chosen = "05", r05
        why = (f"{desc(r25)} > {GATE_S_PER_STEP} s/step" if r25 is not None else
               "B=2 does not fit at 0.25 deg (" + ", ".join(f"ckpt {'on' if r['ckpt'] else 'off'}: {r['status']}"
                                                            for r in c25) + ")")
        reason = f"0.5 deg chosen: at 0.25 deg {why}" + (
            f"; 0.5 deg {desc(r05)}" if r05 is not None else
            "; WARNING: B=2 does not fit at 0.5 deg either" if c05 else "; 0.5 deg B=2 not measured")
    else:
        choice, chosen, reason = None, None, "undecided: 0.25 deg B=2 was not measured"
    flags = None
    if choice is not None and chosen is not None:
        flags = f"--grid {choice}" + (" --use-checkpoint" if chosen["ckpt"] else "")
    gate = dict(rule=f"PLAN.md S1: keep 0.25 deg if B = {GATE_B} fits and takes <= {GATE_S_PER_STEP} s/step, "
                     f"otherwise 0.5 deg; B = 2 without checkpointing if it fits, else with it",
                fits_definition=f"no OOM and projected full-run peak (reserved - smoke per-pixel arrays + "
                                f"{FULL_RUN_PIXELS:,} pixels x {FULL_RUN_PIXEL_BYTES} B + {MARGIN_GIB} GiB) <= GCD memory",
                grid=choice, use_checkpoint=None if chosen is None else bool(chosen["ckpt"]), reason=reason,
                train_flags=flags, b2_row=chosen)
    if chosen is not None:
        gate["cost_main_run"] = cost(chosen["s_per_step"])
    out = dict(step="S1 encoder benchmark (PLAN.md section 7)", job=a.job,
               written=dt.datetime.now().isoformat(timespec="seconds"), env=envs, gate=gate, rows=cfg)
    tmp = a.json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.json)
    print("[gate] rows:")
    for r in cfg:
        extra = (f" | projected full run {r['projected_full_run_gib']:.1f} GiB, fits={r['fits']}"
                 if r.get("projected_full_run_gib") is not None else f" | fits={r['fits']}")
        print("[gate]   " + fmt_row(r) + extra)
    print(f"[gate] {reason}")
    if chosen is not None:
        c = gate["cost_main_run"]
        print(f"[gate] main run projection: {c['steps']:,} steps x {c['s_per_step']:.3f} s = {c['node_h']:.1f} node-h = "
              f"{c['gpu_h']:.0f} GPU-h ({c['note']})")
    print(f"[gate] train_aq flags: {flags} -> {a.json}")
    if a.status:
        line = f"gate -> {reason}"
        if chosen is not None:
            line += (f"; main 60k-step run ~ {gate['cost_main_run']['gpu_h']:.0f} GPU-h at this s/step (validation and "
                     f"data wait excluded); train_aq flags '{flags}'")
        where = os.path.relpath(a.json, PROJECT) if os.path.abspath(a.json).startswith(str(PROJECT)) else a.json
        with open(a.status, "a") as f:
            f.write(f"- {dt.datetime.now():%Y-%m-%d %H:%M} [S1] job {a.job}: {line}. Table: {where}\n")
    return 0 if choice is not None else 1


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="time the training step for one grid x checkpointing pair")
    r.add_argument("--grid", choices=["025", "05"], default="025")
    r.add_argument("--ckpt", action="store_true", help="Flower activation checkpointing")
    r.add_argument("--batches", default="1,2,4", help="days per step (B), comma-separated")
    r.add_argument("--n-points", type=int, default=16384, help="points per day (train_aq default)")
    r.add_argument("--extra-n", type=int, default=0, help="one more config at B = --profile-b with this N")
    r.add_argument("--profile-b", type=int, default=0, help="profile 2 steps at this B (0 = off)")
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--steps", type=int, default=12, help="maximum timed steps per config")
    r.add_argument("--min-steps", type=int, default=3)
    r.add_argument("--timed-budget-s", type=float, default=40.0, help="target wall time of the timed block")
    r.add_argument("--budget-s", type=float, default=420.0, help="process budget; configs that would exceed it are "
                                                                 "skipped")
    r.add_argument("--n-batches", type=int, default=3)
    r.add_argument("--out", required=True, help="JSONL, one row per config (appended)")
    r.add_argument("--store", choices=["smoke", "toy"], default="smoke")
    r.add_argument("--device", default="auto", help="auto | cpu")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--fake-oom", type=int, default=0, help=argparse.SUPPRESS)   # CPU logic test of the OOM path
    g = sub.add_parser("gate", help="apply the PLAN.md S1 rule to the rows of one or more run processes")
    g.add_argument("--results", required=True)
    g.add_argument("--json", required=True)
    g.add_argument("--status", default=None, help="SMOKE_STATUS.md to append the decision to")
    g.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "?"))
    g.add_argument("--n-points", type=int, default=16384)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
        return 0
    return cmd_gate(a)


if __name__ == "__main__":
    sys.exit(main())
