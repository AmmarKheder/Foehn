"""cdffm_aq_x.train_x -- training of the variants: cdffm_aq.train_aq with --variant {base, estab, hcf}.

Everything else is the baseline's (losses, optimiser, schedule, EMA, DDP, validation protocol, checkpoints):
AQModuleX subclasses train_aq.AQModule and only (1) builds the variant model, (2) adds the coarse flow-matching loss
of hcf (fine FM loss + --coarse-weight x coarse FM loss; the fine flow is conditioned on the TRUE cell means of the
day, read from logs/x/cellmean_smoke.npz, built by data_x.build_cellmeans), (3) validates through model_x.Sampler
(identical to train_aq.validate for base/estab). metrics.jsonl keeps the baseline format (eval_aq.best_val_step
works), plus the coarse loss in the training window of hcf.

  srun ... python -u -m cdffm_aq_x.train_x --name hcf_s0 --variant hcf --seed 0 --grid 025 --max-steps 5000 ...
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from cdffm_aq.data_aq import (BatchStream, PixelArrays, Scores, assemble_inputs, bilinear_log1p_at, doy_of,
                              in_channels, open_store, raw_to_device, store_kwargs, target_z, z_to_ug)
from cdffm_aq.model_aq import sphere_noise, draw_sphere_noise
from cdffm_aq.train_aq import (AQModule, ValData, _short, compute_losses, model_grid, noise_spec, parser as base_parser,
                               resolve_noise, launch_env, env_rank)

from . import CELLMEAN_GLOBAL, CELLMEAN_SMOKE, RUNS_X
from .data_x import CellGeometry, CellMeans, build_cellmeans
from .model_x import Sampler, build_model_x, VARIANTS
from cdffm_aq.model_aq import NoiseSpec


def coarse_spec(args, spec=None):
    """noise law of the coarse flow (hcf): the fine law unless --coarse-lmin-km / --coarse-lmax-km are given"""
    spec = spec or noise_spec(args)
    lo = getattr(args, "coarse_lmin_km", None) or spec.lmin_km
    hi = getattr(args, "coarse_lmax_km", None) or spec.lmax_km
    return NoiseSpec(spec.n_features, spec.beta, float(lo), float(hi))


# ---------------------------------------------------------------------------------------------------------------
# hcf loss
# ---------------------------------------------------------------------------------------------------------------

def compute_losses_hcf(model, pix, b, args, spec, gen, cm):
    """fine FM loss (as train_aq.compute_losses, objective fm) with the LIIF fed the true cell means, plus the
    coarse FM loss on the grid (masked to the mask cells); returns (total, fine, coarse)"""
    k = b["k"]
    B, N = k.shape
    dev = k.device
    lat, lon = pix.lat[k], pix.lon[k]
    mu, sig = pix.mu_sigma(k, b["doy"])
    u = torch.rand(B, N, device=dev, generator=gen) - 0.5
    x1 = target_z(b["v"], mu, sig, u)[..., None]
    x = assemble_inputs(b["raw"], pix, args.grid, args.transport)
    feat = model.encode(x)
    zbar = cm.train_grid(b["day"])                                           # (B, 1, H, W) true cell means
    cond = model.condition(model.with_coarse(feat, zbar, cm.mask), lat, lon, pix.static[k], b["doy"])
    z0 = sphere_noise(lat, lon, 1, spec, generator=gen)
    t = torch.rand(B, N, device=dev, generator=gen)
    xt = (1 - t)[..., None] * z0 + t[..., None] * x1
    v = model.velocity(xt, cond, t)
    fine = ((v - (x1 - z0)) ** 2).mean()
    eps = cm.grid_noise(draw_sphere_noise(B, 1, coarse_spec(args, spec), dev, gen))   # (B, 1, H, W) coarse noise
    tc = torch.rand(B, device=dev, generator=gen)
    eta_t = (1 - tc)[:, None, None, None] * eps + tc[:, None, None, None] * zbar
    vc = model.coarse_velocity(eta_t, cm.mask, feat.detach() if getattr(args, 'coarse_detach', False) else feat, tc)
    coarse = (cm.mask * (vc - (zbar - eps)) ** 2).sum() / (cm.mask.sum() * B)
    return fine + args.coarse_weight * coarse, fine, coarse


# ---------------------------------------------------------------------------------------------------------------
# validation (all variants, one code path)
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def validate_x(model, store, pix, val, args, spec, device, baselines=False, cm=None):
    """as train_aq.validate, through model_x.Sampler (same generator use for base/estab)"""
    model.eval()
    k = torch.from_numpy(val.pix).to(device)
    lat, lon, w, stat = pix.lat[k], pix.lon[k], pix.w_area[k], pix.static[k]
    P = k.numel()
    M = args.val_members
    sc = Scores(M, device)
    base = {"climatology_median": Scores(1, device), "cams_bilinear": Scores(1, device)} if baselines else {}
    smp = Sampler(model, args.variant, spec, M, args.val_ode_steps, device, cm=cm, coarse_spec=coarse_spec(args, spec))
    for i, (t, d) in enumerate(zip(val.days, val.dates)):
        doy = torch.tensor([doy_of(d)], device=device)
        raw = raw_to_device(store.raw_inputs(d), device)
        feat = model.encode(assemble_inputs(raw, pix, args.grid, args.transport))
        mu, sig = pix.mu_sigma(k[None], doy)
        truth = torch.from_numpy(val.truth[i].astype(np.float32)).to(device).clamp(min=1.0)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) * 1_000_003 + int(t) * 1009 + 17)
        gc = torch.Generator(device=device)
        gc.manual_seed(int(args.seed) * 1_000_003 + int(t) * 1009 + 31)
        smp.start_day(feat, gen, gen_coarse=gc)
        ens = torch.empty(M, P, device=device)
        for a in range(0, P, args.val_chunk):
            e = min(P, a + args.val_chunk)
            ens[:, a:e] = smp.members(lat[a:e], lon[a:e], stat[a:e], doy)
        sc.add(z_to_ug(ens, mu, sig), truth, w)
        if baselines:
            base["climatology_median"].add(z_to_ug(torch.zeros(1, P, device=device), mu, sig), truth, w)
            cams = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native)
            base["cams_bilinear"].add(cams[None], truth, w)
    return sc.summary(), {n_: s.summary() for n_, s in base.items()}


# ---------------------------------------------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------------------------------------------

class AQModuleX(AQModule):
    def __init__(self, args, store):
        # AQModule.__init__ without its model construction (same fields)
        super(AQModule, self).__init__()
        self.args = args
        self.store = store
        self.spec = noise_spec(args)
        self.grid = model_grid(store.name, args.grid)
        self.model = build_model_x(args, in_channels(args.transport), store.static_dim, self.grid)
        self.teacher = None
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.run_dir = Path(args.runs_root) / args.name
        self.t0 = time.time()
        self.stream = self.pix = self.val = self.gen = None
        self.cm = None
        self._last_days = None
        self._baselines_done = False
        self._validated = set()
        self._win = None
        self._val_s = 0.0
        self._last_window = {}
        self._coarse = None

    def on_train_start(self):
        super().on_train_start()
        if self.args.variant == "hcf":
            geom = CellGeometry(self.store, self.grid)
            self.cm = CellMeans(geom, self.device, path=self.args.cellmeans, load_hold=False)  # training days only
            if self.global_rank == 0:
                print(f"[hcf] cell means {self.args.cellmeans}: {geom.n_cells} cells, box {geom.box}", flush=True)

    def training_step(self, batch, batch_idx):
        b = self.stream.next()
        self._last_days = b["day"]
        if self.global_step < 3:
            print(f"[rank {self.global_rank}] step {self.global_step} days {b['day'].tolist()}", flush=True)
        if self.args.variant == "hcf":
            loss, fine, coarse = compute_losses_hcf(self.model, self.pix, b, self.args, self.spec, self.gen, self.cm)
            self.log("train/fine", fine.detach(), on_step=True, on_epoch=False)
            self.log("train/coarse", coarse.detach(), on_step=True, on_epoch=False)
            if self._coarse is None:
                self._coarse = torch.zeros((), device=self.device)
            self._coarse += coarse.detach()
        else:
            loss = compute_losses(self.model, self.pix, b, self.args, self.spec, self.gen, None)
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False)
        if self._win is not None:
            self._win["loss"] += loss.detach()
            self._win["n"] += 1
        return loss

    def _log_window(self, step):
        w = self._win
        n = w["n"] if w is not None else 0
        super()._log_window(step)
        if self.args.variant == "hcf" and n > 0 and self.global_rank == 0 and self._coarse is not None:
            self._last_window["coarse_loss"] = float(self._coarse) / n
            print(f"[step {step}] coarse loss {self._last_window['coarse_loss']:.4f} (fine = loss - "
                  f"{self.args.coarse_weight} x coarse)", flush=True)
        self._coarse = torch.zeros((), device=self.device)

    def _run_val(self):
        info = self._ddp_checks()
        if self.global_rank != 0:
            return
        if int(self.global_step) in self._validated:
            print(f"[val step {self.global_step}] already in metrics.jsonl (resumed run): skipped", flush=True)
            return
        a = self.args
        if self.val is None:
            self.val = ValData(self.store, a.val_pixels, None if a.val_days is None else
                               [int(x) for x in a.val_days.split(",")])
        t0 = time.time()
        res, base = validate_x(self.ema, self.store, self.pix, self.val, a, self.spec, self.device,
                               baselines=not self._baselines_done, cm=self.cm)
        rec = dict(step=int(self.global_step), elapsed_min=(time.time() - self.t0) / 60,
                   val_seconds=time.time() - t0, val_days=len(self.val.days), val_pixels=len(self.val.pix),
                   model=res, train_window=self._last_window, **info)
        self._val_s += rec["val_seconds"]
        if base:
            rec["baselines"] = base
            self._baselines_done = True
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._validated.add(rec["step"])
        print(f"[val step {rec['step']}] model {_short(res)} ({rec['val_seconds']:.0f} s)", flush=True)
        for n_, r in base.items():
            print(f"[val step {rec['step']}] {n_} {_short(r)}", flush=True)
        if info:
            print(f"[val step {rec['step']}] DDP: weights max |diff| across ranks {info['ddp_weights_max_abs_diff']:.1e}"
                  f"; last days by rank {info['ddp_last_days_by_rank']}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def parser():
    ap = base_parser()
    ap.set_defaults(runs_root=str(RUNS_X))
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--coarse-weight", type=float, default=1.0, help="hcf: weight of the coarse FM loss")
    ap.add_argument("--coarse-hidden", type=int, default=64)
    ap.add_argument("--coarse-depth", type=int, default=3)
    ap.add_argument("--coarse-detach", action="store_true", help="hcf ablation: no coarse-loss gradient into the encoder")
    ap.add_argument("--coarse-dilations", default=None, help="hcf: dilations of the 3 x 3 coarse convs, e.g. 1:2:4 (no commas: sbatch --export splits on them)")
    ap.add_argument("--coarse-lmin-km", type=float, default=None, help="hcf: coarse noise law (default: the fine law)")
    ap.add_argument("--coarse-lmax-km", type=float, default=None)
    ap.add_argument("--cellmeans", default=str(CELLMEAN_SMOKE), help="hcf: cell-mean table (built if missing; "
                    "--store global: default CELLMEAN_GLOBAL, never built here)")
    return ap


def main(argv=None):
    import pytorch_lightning as pl
    from datetime import timedelta
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    args = parser().parse_args(argv)
    if args.objective != "fm":
        raise SystemExit("variants are trained with the flow-matching objective only")
    resolve_noise(args)
    plugins, devices, num_nodes = launch_env()
    use_gpu = args.device != "cpu" and torch.cuda.is_available()
    if args.store == "toy" and args.val_pixels > 5000:
        args.val_pixels = 5000
    pl.seed_everything(args.seed, workers=False, verbose=False)
    if args.store == "global":
        args.fires, args.split, args.sigma_k = (store_kwargs(args)[k] for k in ("fires", "split", "sigma_k"))
        if args.cellmeans == str(CELLMEAN_SMOKE):
            args.cellmeans = str(CELLMEAN_GLOBAL).replace("_A.npz", f"_{args.split}.npz")
        if args.variant == "hcf" and not Path(args.cellmeans).exists():
            raise SystemExit(f"[hcf] {args.cellmeans} missing: build it on CPU first (scripts/build_global_cellmeans.py)")
    store = open_store(args.store, **store_kwargs(args))
    args.static_dim = store.static_dim
    args.in_ch = in_channels(args.transport)
    run_dir = Path(args.runs_root) / args.name
    if not args.resume and (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a run (metrics.jsonl): use --resume or another --name")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    if args.variant == "hcf":
        if args.store == "toy":
            args.cellmeans = str(run_dir / "cellmean_toy.npz")
        if not Path(args.cellmeans).exists():
            if env_rank() == 0:
                print(f"[hcf] building the cell-mean table {args.cellmeans}", flush=True)
                build_cellmeans(store, CellGeometry(store, model_grid(store.name, args.grid)), args.cellmeans)
            else:
                while not Path(args.cellmeans).exists():
                    time.sleep(5)
                time.sleep(5)
    if env_rank() == 0:
        with open(run_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=1)
    module = AQModuleX(args, store)
    if env_rank() == 0:
        n = sum(p.numel() for p in module.model.parameters())
        ne = sum(p.numel() for p in module.model.encoder.parameters())
        nq = sum(p.numel() for p in module.model.query.parameters())
        nc = sum(p.numel() for p in module.model.coarse.parameters()) if args.variant == "hcf" else 0
        print(f"[model] variant={args.variant} objective={args.objective} grid={args.grid} in_ch={args.in_ch} "
              f"static={args.static_dim} params={n:,} (encoder {ne:,}, LIIF {nq:,}, coarse {nc:,}, velocity "
              f"{n - ne - nq - nc:,}) | noise beta={args.noise_beta} {args.noise_lmin_km}-{args.noise_lmax_km} km "
              f"F={args.noise_features} | world={devices * num_nodes} x B={args.batch_size} days x N={args.n_points} "
              f"| seed {args.seed}", flush=True)
    ckpt_cb = ModelCheckpoint(dirpath=str(run_dir / "ckpt"), filename="step{step:06d}", auto_insert_metric_name=False,
                              save_top_k=-1, save_last=True)
    world = devices * num_nodes
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu", devices=devices, num_nodes=num_nodes,
        strategy="ddp" if world > 1 else "auto", plugins=plugins,
        max_steps=getattr(args, "stop_at_step", None) or args.max_steps,
        max_time=timedelta(minutes=args.time_budget_min) if args.time_budget_min else None,
        val_check_interval=args.val_every, check_val_every_n_epoch=None, limit_val_batches=1,
        num_sanity_val_steps=0, gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        precision=args.precision,
        logger=TensorBoardLogger(save_dir=str(run_dir), name="tb"), log_every_n_steps=100,
        callbacks=[ckpt_cb], enable_progress_bar=False, enable_model_summary=False, use_distributed_sampler=False,
    )
    last = run_dir / "ckpt" / "last.ckpt"
    trainer.fit(module, ckpt_path=str(last) if (args.resume and last.exists()) else None)
    return module, trainer


if __name__ == "__main__":
    main()
