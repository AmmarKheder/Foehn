"""calib.train_cal -- PROTOCOL.md A6(b): CRPS fine-tuning of the GHAP downscaler, as the calibration fine-tuning of the
1 km forecast model of the paper (global1km/fullflow_global_green.py --crps-steps / --crps-fm).

Per training day of the batch, M = 2 members are drawn exactly as the inference sampler of pmG_final draws them
(calib.core: prior P3 with the day's own condition, |z0| <= 3, 10 uniform Euler steps, |z| <= 10 before every velocity
evaluation and at the end); loss = fair CRPS of the members against the dequantised target (in z, or in ug m-3 for
--crps-space ug) + --crps-fm x the flow-matching loss of cdffm_aq.train_aq (unchanged). The gradient flows through all
the sampler steps. --train-part velocity: only the pointwise velocity net learns (encoder and LIIF query frozen, run
without gradient: the condition, hence the prior's weights, stay those of pmG_final); --train-part all: every weight.
cdffm_aq.train_aq.AQModule is subclassed, not modified: data stream, EMA, DDP, schedule and checkpoints are its own.
Validation (EMA weights, every --val-every steps, all ranks share the days): calib.core.validate_set on the sets of
--val-sets (V100k 'sel', hot-spot 'probe', in-sample 'ins2020') with the inference sampler; rank 0 appends the pooled
summaries to <run>/metrics.jsonl and writes the per-day sums to <run>/val_rows/step<NNNNNN>.npz.

  srun -n 8 ... python -u -m calib.train_cal --name V2 --runs-root runs_ghap/pmG_cal/runs --train-part velocity \
       --crps-space z --crps-fm 0.1 --lr 1e-4 --max-steps 3000 ...
"""
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cdffm_aq.data_aq import assemble_inputs, in_channels, open_store, store_kwargs, target_z, z_to_ug
from cdffm_aq.model_aq import sphere_noise
from cdffm_aq.train_aq import AQModule, env_rank, launch_env, parser as base_parser, resolve_noise

from .core import (FINAL, ValSet, band_fields, daily_table, draw_prior_params, euler_members, fair_crps_members, guard,
                   load_prior_module, prior_z0, reduce_rows, summarise, validate_set)


def cal_losses(model, pix, b, args, spec, gen, prior, frozen_cond):
    """(total, crps term, flow-matching loss) of one device batch (BatchStream.next())"""
    k = b["k"]
    B, N = k.shape
    dev = k.device
    lat, lon = pix.lat[k], pix.lon[k]
    mu, sig = pix.mu_sigma(k, b["doy"])
    u = torch.rand(B, N, device=dev, generator=gen) - 0.5                         # dequantisation of the integer
    x1 = target_z(b["v"], mu, sig, u)[..., None]
    x = assemble_inputs(b["raw"], pix, args.grid, args.transport)
    static = pix.static[k]
    with torch.set_grad_enabled(not frozen_cond):
        cond = model.condition(model.encode(x), lat, lon, static, b["doy"])
    # flow matching, exactly train_aq.compute_losses (objective fm)
    z0n = sphere_noise(lat, lon, 1, spec, generator=gen)
    t = torch.rand(B, N, device=dev, generator=gen)
    xt = (1 - t)[..., None] * z0n + t[..., None] * x1
    fm = ((model.velocity(xt, cond, t) - (x1 - z0n)) ** 2).mean()
    # M members per day, drawn as the inference sampler draws them (prior frozen, no gradient through z0)
    M = args.crps_members
    with torch.no_grad():
        cd = cond.detach()
        z0 = []
        for i in range(B):
            params = draw_prior_params(prior, M, dev, gen)
            xi = band_fields(prior, params, lat[i], lon[i], M)                                   # (M, N, K)
            z0.append(prior_z0(prior, xi, cd[i][None].expand(M, -1, -1), static[i][None].expand(M, -1, -1)))
        z0 = torch.stack(z0).clamp(-args.z0_trunc, args.z0_trunc)                                # (B, M, N, 1)
    C = cond.shape[-1]
    condM = cond[:, None].expand(B, M, N, C).reshape(B * M, N, C)
    xs = euler_members(model, z0.reshape(B * M, N, 1), condM, args.crps_steps, args.z_bound).reshape(B, M, N)
    y = x1[..., 0]
    if args.crps_space == "z":
        crps = fair_crps_members(xs, y)
    else:                                                                                        # ug m-3, capped as inference
        X = z_to_ug(xs, mu[:, None], sig[:, None], pmax=args.pmax_ug)
        crps = fair_crps_members(X, z_to_ug(y, mu, sig)) / args.crps_scale
    return crps + args.crps_fm * fm, crps, fm


class CalModule(AQModule):
    def __init__(self, args, store):
        super().__init__(args, store)
        self._prior = []                               # list: kept out of the state dict and of DDP (as the teacher)
        self.val_sets = {}
        self.vcache = {}
        self.frozen_cond = args.train_part == "velocity"
        self._guard = {"z_bound": args.z_bound, "pmax_ug": args.pmax_ug, "z0_trunc": args.z0_trunc}
        self._tstep = None

    @property
    def prior(self):
        return self._prior[0]

    def on_train_start(self):
        super().on_train_start()
        self._prior = [load_prior_module(self.device)]
        if self.args.val_sets:
            self.val_sets = {n: ValSet.load(n) for n in self.args.val_sets.split(",")}
        if self.global_rank == 0:
            a = self.args
            nt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[cal] arm {a.name}: train-part {a.train_part} ({nt:,} trainable parameters), CRPS in {a.crps_space} "
                  f"(M = {a.crps_members}, {a.crps_steps} Euler steps, |z0| <= {a.z0_trunc}, |z| <= {a.z_bound}, cap "
                  f"{a.pmax_ug}), lambda_fm {a.crps_fm}, lr {a.lr}, warm-up {a.warmup}, cosine to {a.max_steps}, EMA "
                  f"{a.ema}; prior {self.prior.kind} K = {self.prior.K}; validation sets "
                  f"{ {n: (len(v.days), int(v.pix.shape[-1])) for n, v in self.val_sets.items()} }", flush=True)

    def training_step(self, batch, batch_idx):
        b = self.stream.next()
        self._last_days = b["day"]
        if self.global_step < 3:
            print(f"[rank {self.global_rank}] step {self.global_step} days {b['day'].tolist()}", flush=True)
        loss, crps, fm = cal_losses(self.model, self.pix, b, self.args, self.spec, self.gen, self.prior, self.frozen_cond)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False)
        if self._win is not None:
            self._win["loss"] += loss.detach()
            self._win["n"] += 1
            self._win["crps"] = self._win.get("crps", 0.0) + crps.detach()
            self._win["fm"] = self._win.get("fm", 0.0) + fm.detach()
        if self.args.time_steps and self.global_rank == 0 and self.global_step < self.args.time_steps:
            torch.cuda.synchronize(self.device)
            now = time.time()
            if self._tstep is not None:
                print(f"[time] step {self.global_step}: {now - self._tstep:.3f} s since the previous step, peak memory "
                      f"{torch.cuda.max_memory_allocated(self.device) / 2 ** 30:.2f} GiB, loss {float(loss):.4f} "
                      f"(crps {float(crps):.4f}, fm {float(fm):.4f})", flush=True)
            self._tstep = now
        return loss

    def _log_window(self, step):
        w = self._win
        super()._log_window(step)
        if w is not None and w["n"] > 0 and self.global_rank == 0:
            c, f = float(w.get("crps", 0.0)) / w["n"], float(w.get("fm", 0.0)) / w["n"]
            self._last_window.update(crps=c, fm=f)
            print(f"[step {step}] crps term {c:.4f} | flow matching {f:.4f} (rank 0, window means)", flush=True)

    def _run_val(self):
        a = self.args
        step = int(self.global_step)
        if not self.val_sets or step in self._validated:
            return
        world = self.trainer.world_size
        t0 = time.time()
        res = {}
        for name, vs in self.val_sets.items():
            rows = validate_set(self.ema, self.store, self.pix, vs, self.prior, self.device, rank=self.global_rank,
                                world=world, M=a.val_members, ode_steps=a.val_ode_steps, seed=0, cache=self.vcache,
                                cond_frozen=self.frozen_cond, grid=a.grid, transport=a.transport, g=self._guard)
            res[name] = reduce_rows(rows, world)
        self._validated.add(step)
        if self.global_rank != 0:
            return
        dtv = time.time() - t0
        self._val_s += dtv
        rec = {"step": step, "arm": a.name, "elapsed_min": (time.time() - self.t0) / 60, "val_seconds": dtv,
               "train_window": self._last_window, "sets": {}}
        vdir = self.run_dir / "val_rows"
        vdir.mkdir(exist_ok=True)
        arrs = {}
        for name, rows in res.items():
            r = rows.cpu().numpy()
            arrs[name] = r
            rec["sets"][name] = summarise(r, a.val_members)
            if name == "sel":
                rec["sets"][name]["daily"] = daily_table(r, a.val_members, self.val_sets[name])
        np.savez(vdir / f"step{step:06d}.npz", **arrs)
        if "sel" in rec["sets"]:                              # the key eval_aq.best_val_step reads
            rec["model"] = {"ug": rec["sets"]["sel"]["ug"], "log": rec["sets"]["sel"]["log"]}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = [f"[val step {step}] ({dtv:.0f} s)"]
        for name, s in rec["sets"].items():
            u = s["ug"]
            msg.append(f"  {name:8s} crps {u['crps_fair']:.4f} rmseM {u['rmse_ensmean']:.4f} bias {u['bias']:+.3f} ssF "
                       f"{u['spread_skill_fortin']:.4f} | log ssF {s['log']['spread_skill_fortin']:.4f} | z crps "
                       f"{s['z']['crps_fair']:.4f} ssF {s['z']['spread_skill_fortin']:.4f} | pix>TG "
                       f"{s['sanity']['n_pix_gt_TG']:.0f} cap {s['sanity']['n_member_at_pmax']:.0f} bound "
                       f"{s['sanity']['n_member_at_bound']:.0f} max {s['sanity']['max_member_ug']:.0f}")
        print("\n".join(msg), flush=True)

    def on_train_end(self):
        if getattr(self.args, "no_final_val", False):          # smoke test: weights discarded, nothing scored
            if self.stream is not None:
                self.stream.close()
            if self.global_rank == 0:
                print(f"[done] {self.global_step} steps in {(time.time() - self.t0) / 60:.1f} min (smoke: no validation, "
                      f"no checkpoint)", flush=True)
            return
        super().on_train_end()


def parser():
    ap = base_parser()
    ap.add_argument("--train-part", choices=["velocity", "all"], default="velocity")
    ap.add_argument("--crps-space", choices=["z", "ug"], default="z")
    ap.add_argument("--crps-fm", type=float, default=1.0, help="lambda: weight of the flow-matching loss added to the CRPS")
    ap.add_argument("--crps-members", type=int, default=2)
    ap.add_argument("--crps-steps", type=int, default=10, help="Euler steps of the sampler in the loss (inference: 10)")
    ap.add_argument("--crps-scale", type=float, default=6.0, help="--crps-space ug: the CRPS is divided by this (ug m-3)")
    ap.add_argument("--val-sets", default="sel,probe,ins2020", help="calib.core validation sets ('' = none)")
    ap.add_argument("--no-final-val", action="store_true", help="smoke test: no validation and no checkpoint at the end")
    ap.add_argument("--time-steps", type=int, default=0, help="print the time and peak memory of the first N steps (rank 0)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    resolve_noise(args)
    g = guard()
    args.z_bound, args.z0_trunc, args.pmax_ug = g["z_bound"], g["z0_trunc"], g["pmax_ug"]
    plugins, devices, num_nodes = launch_env()
    pl.seed_everything(args.seed, workers=False, verbose=False)
    if args.store != "ghap" or args.objective != "fm" or args.stations or args.tune_stations:
        raise SystemExit("calib.train_cal: the GHAP store, objective fm, no stations (PROTOCOL.md A6)")
    args.fires, args.split, args.sigma_k = (store_kwargs(args)[k] for k in ("fires", "split", "sigma_k"))
    ref = json.loads((FINAL / "args.json").read_text())       # the recipe of the frozen system must be the same
    for k in ("store", "fires", "split", "grid", "transport", "sigma_k", "noise_features", "noise_beta", "noise_lmin_km",
              "noise_lmax_km", "doy_cond", "lifting_dim", "batch_size", "n_points", "precision"):
        if getattr(args, k) != ref[k]:
            raise SystemExit(f"--{k.replace('_', '-')} = {getattr(args, k)!r} differs from pmG_final ({ref[k]!r})")
    if not args.init_from:
        raise SystemExit("--init-from <pmG_final checkpoint> is required")
    store = open_store(args.store, **store_kwargs(args))     # GhapStore with its firewall on (no 2022, no station file)
    args.static_dim = store.static_dim
    args.in_ch = in_channels(args.transport)
    args.stn_dim = 0
    run_dir = Path(args.runs_root) / args.name
    if (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a run (metrics.jsonl): use another --name")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    if env_rank() == 0:
        with open(run_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=1)
    module = CalModule(args, store)
    module.load_init(args.init_from)
    if args.train_part == "velocity":
        for n_, p in module.model.named_parameters():
            p.requires_grad_(n_.startswith("velocity_net."))
    if env_rank() == 0:
        n = sum(p.numel() for p in module.model.parameters())
        nt = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
        print(f"[model] in_ch={args.in_ch} static={args.static_dim} params={n:,} trainable={nt:,} | noise beta="
              f"{args.noise_beta} {args.noise_lmin_km}-{args.noise_lmax_km} km F={args.noise_features} | world="
              f"{devices * num_nodes} x B={args.batch_size} days x N={args.n_points} | guard {g}", flush=True)
    ckpt_cb = ModelCheckpoint(dirpath=str(run_dir / "ckpt"), filename="step{step:06d}", auto_insert_metric_name=False,
                              save_top_k=-1, save_last=True)
    world = devices * num_nodes
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() and args.device != "cpu" else "cpu", devices=devices,
        num_nodes=num_nodes, strategy="ddp" if world > 1 else "auto", plugins=plugins,
        max_steps=getattr(args, "stop_at_step", None) or args.max_steps,
        max_time=timedelta(minutes=args.time_budget_min) if args.time_budget_min else None,
        val_check_interval=args.val_every, check_val_every_n_epoch=None, limit_val_batches=1,
        num_sanity_val_steps=0, gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        precision=args.precision, logger=False, callbacks=[] if args.no_final_val else [ckpt_cb],
        enable_checkpointing=not args.no_final_val, enable_progress_bar=False, enable_model_summary=False,
        use_distributed_sampler=False,
    )
    trainer.fit(module)
    return module, trainer


if __name__ == "__main__":
    main()
