"""cdffm_aq_m.train_aq -- training of the CD-FFM adaptation (PLAN.md 5): CD-FFM fast_train.py + DDP on one LUMI node.

A11 experiment M copy of cdffm_aq/train_aq.py (protocol_amendments/A11_retraining_two_stages.md, Addendum 2 point 1;
a11/stageM/README.md). One option is added, OFF by default; with it off the loss of a batch is the original's (tested),
and ONE thing differs from the original on purpose: the in-training validation of a --stations run no longer scores at
the held-out monitors (the file of A3(d')-4 is never opened by this copy, whatever the options).
  --stn-lag 1,2,3,4,5 [--stn-lag0-prob P]   with --stations: for a training day t the lag l is drawn (stations.draw_lag:
        0 with probability P, otherwise uniform over the list), the station features are those of the assimilation
        monitors of day t - l with their anomaly and the lag (stations.lag_features, 63 floats), under the A3 dropout;
        the in-training validation becomes lagval.validate_lag on the same SEL days and pixels (--val-lags, default 1,5:
        scores by lag and by distance class, plus every monitor hidden). No held-out value is read by this code
        (firewall on the held-out value files as soon as a station set is opened; asserts in LagStationSet).

Losses exactly as CD-FFM's compute_losses / flowmap_loss, with the sphere noise prior:
  fm          linear-path flow matching, t ~ U(0, 1) per point: mean (v(x_t, c, t) - (x1 - z0))^2
  regression  baseline B3: the same net at x = 0, t = 0 predicts x1
  flowmap     distillation of a trained fm run (--teacher): RK4 teacher with --fm-substeps, s = t on
              --fm-diag-prob of the points, encoder and LIIF frozen
x1 = z of the dequantised LGHAP value (data_aq.target_z); no mass-conservation term (PLAN.md 4.2-6).
Optimiser and schedule as CD-FFM (AdamW 3e-4, wd 0, fast_train.lr_factor: warm-up then cosine to 10 %, EMA 0.999),
except gradient clipping, off by default (--grad-clip 0): in CD-FFM the clip at 1.0 never acts (initial gradient norm
0.05), but on the 0.25 deg globe the initial norm is ~3e5-7e5 (SelfWarp flow heads and lift at full resolution, real
input fields), so a clip at 1.0 would rescale every step by ~4e-6 and push 99.8-100 % of the velocity-net, LIIF,
bottleneck and decoder gradients below Adam's eps (1e-8), i.e. freeze them (S0 diagnostic). Adam itself is invariant to
the gradient scale. The pre-clip gradient norm is logged (mean / max per window) so a spike-only clip can be set later
(about 10 x the logged norm) if needed. Batch: --batch-size days per rank x --n-points points per day (scattered), drawn by one prefetch thread per
rank with its own generator (seed, rank): ranks draw different days. Validation (EMA weights) and metrics.jsonl on
rank 0 only, on the hold-out days: fair CRPS, RMSE (1 member, ensemble mean), MAE, bias, spread/skill in ug m-3 and
log1p, area-weighted; the first validation also scores the climatology and CAMS bilinear on the same pixels. All ranks
check at every validation that their weights are identical. Checkpoints (Lightning) at every validation + last.ckpt.

Launch:
  one process:      python -m cdffm_aq_m.train_aq --name RUN [...]            (LightningEnvironment, like CD-FFM)
  8 GCDs (srun):    srun ... python -u -m cdffm_aq_m.train_aq --name RUN ...  (SLURMEnvironment, one task per GCD)
  CPU DDP (tests):  python -m torch.distributed.run --nproc_per_node 2 -m cdffm_aq_m.train_aq --store toy ...
(run from a11/stageM, so that the copy is the package that is imported)
"""
import argparse
import copy
import datetime as dt
import json
import os
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment, TorchElasticEnvironment
from torch.utils.data import DataLoader, IterableDataset

from . import NOISE_FIT_SMOKE, RUNS_ROOT
from .data_aq import (BatchStream, PixelArrays, Scores, assemble_inputs, bilinear_log1p_at, doy_of, in_channels,
                      open_store, raw_to_device, store_kwargs, target_z, z_to_ug)
from .model_aq import (GRIDS, NoiseSpec, build_model, draw_sphere_noise, eval_sphere_noise, integrate, noise_chunk,
                       sphere_noise)
from .lagval import validate_lag
from .stations import (STN_DIM, draw_lag, dropout_generator, lag_features, lag_spec, open_stations, station_features,
                       stn_dim_of)
try:  # optional module of a later experiment (station fine-tuning): not part of this release, off in every result of the paper
    from .tune_stations import TUNE_SPLIT, TunedStore, TuneTargets, TuneVal, parse_years, validate_tune  # A4
except ImportError:
    TUNE_SPLIT = TuneTargets = TuneVal = parse_years = validate_tune = None

    class TunedStore:  # placeholder: without the optional module no store is of this type
        pass
from .store_global import D2025  # A4: the backfill boundary (tuning days need the store inputs)
from fast_train import lr_factor  # noqa: E402  (CD-FFM's schedule; model_aq put CD-FFM/scripts on sys.path)

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Tensor Cores.*")


class _Placeholder(IterableDataset):
    """Lightning needs a DataLoader; the real batch comes from BatchStream inside training_step (as CD-FFM)."""

    def __init__(self, n=None):
        self.n = n

    def __iter__(self):
        i = 0
        while self.n is None or i < self.n:
            yield torch.zeros(1)
            i += 1


def model_grid(store_name, grid_mode):
    if store_name == "toy":
        if grid_mode != "025":
            raise SystemExit("the toy store has a single grid (use --grid 025)")
        return GRIDS["toy"]
    return GRIDS[grid_mode]


def resolve_noise(args):
    """fill args.noise_* from --noise-fit (the fitted power law, scripts/measure_noise_beta.py) unless given"""
    fit = {}
    if args.noise_fit and Path(args.noise_fit).exists():
        fit = json.loads(Path(args.noise_fit).read_text())["chosen"]
    args.noise_beta = args.noise_beta if args.noise_beta is not None else float(fit.get("beta", 2.4))
    args.noise_lmin_km = args.noise_lmin_km if args.noise_lmin_km is not None else float(fit.get("lmin_km", 8.0))
    args.noise_lmax_km = args.noise_lmax_km if args.noise_lmax_km is not None else float(fit.get("lmax_km", 500.0))
    args.noise_features = args.noise_features if args.noise_features is not None else int(fit.get("n_features", 1024))
    args.noise_source = args.noise_fit if fit else "defaults (CD-FFM: beta 2.4, 8-500 km, F 1024)"
    return noise_spec(args)


def noise_spec(args):
    return NoiseSpec(int(args.noise_features), float(args.noise_beta), float(args.noise_lmin_km),
                     float(args.noise_lmax_km))


# ---------------------------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------------------------

def flowmap_loss(model, teacher, x1, lat, lon, cond, args, spec, gen):
    """CD-FFM fast_train.flowmap_loss with the sphere noise"""
    B, N, C = x1.shape
    dev = x1.device
    z0 = sphere_noise(lat, lon, C, spec, generator=gen)
    t = torch.rand(B, N, device=dev, generator=gen)
    diag = torch.rand(B, N, device=dev, generator=gen) < args.fm_diag_prob
    s = torch.where(diag, t, t + (1 - t) * torch.rand(B, N, device=dev, generator=gen))
    xt = (1 - t)[..., None] * z0 + t[..., None] * x1
    with torch.no_grad():
        v0 = teacher.velocity(xt, cond, t)
        h = (s - t) / args.fm_substeps
        x, tk = xt, t
        for _ in range(args.fm_substeps):
            k1 = teacher.velocity(x, cond, tk)
            k2 = teacher.velocity(x + 0.5 * h[..., None] * k1, cond, tk + 0.5 * h)
            k3 = teacher.velocity(x + 0.5 * h[..., None] * k2, cond, tk + 0.5 * h)
            k4 = teacher.velocity(x + h[..., None] * k3, cond, tk + h)
            x = x + h[..., None] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            tk = tk + h
        avg = (x - xt) / (s - t).clamp(min=1e-6)[..., None]
        target = torch.where(diag[..., None], v0, avg)
    u = model.velocity(xt, cond, t, s)
    return ((u - target) ** 2).mean()


def compute_losses(model, pix, b, args, spec, gen, teacher=None, stations=None, dates=None, step=0):
    """b: device batch from BatchStream.next(); returns the scalar loss. stations / dates / step (PROTOCOL.md A3(c)):
    for a --stations run, the station features of each day's points from the day's assimilation monitors with the
    dropout of a fraction f ~ U(0, 1) of them (generator seeded by (args.seed, step, day index))"""
    k = b["k"]
    B, N = k.shape
    dev = k.device
    lat, lon = pix.lat[k], pix.lon[k]
    mu, sig = pix.mu_sigma(k, b["doy"])
    u = torch.rand(B, N, device=dev, generator=gen) - 0.5                         # dequantisation of the integer
    x1 = target_z(b["v"], mu, sig, u)[..., None]
    x = assemble_inputs(b["raw"], pix, args.grid, args.transport)
    stn = None
    if stations is not None and getattr(model, "stations", False):
        lspec = lag_spec(args)                  # A11 experiment M: None = the A3 path below, unchanged
        stn = []
        for i_, d in enumerate(dates):
            g, f = dropout_generator(args.seed, step, int(b["day"][i_]), dev)
            if lspec is not None:               # monitors of day d - lag (never of day d when lag >= 1), A3 dropout
                lag = draw_lag(args.seed, step, int(b["day"][i_]), lspec)
                stn.append(lag_features(lat[i_], lon[i_], stations.lagged(d, lag, hide=f, gen=g))[0])
                continue
            mlat, mlon, mval, mxyz = stations.day(d, hide=f, gen=g)
            stn.append(station_features(lat[i_], lon[i_], mlat, mlon, mval, mxyz=mxyz)[0])
        stn = torch.stack(stn)
    cond = model.condition(model.encode(x), lat, lon, pix.static[k], b["doy"], stn=stn)
    if args.objective == "flowmap":
        return flowmap_loss(model, teacher, x1, lat, lon, cond, args, spec, gen)
    if args.objective == "fm":
        z0 = sphere_noise(lat, lon, 1, spec, generator=gen)
        t = torch.rand(B, N, device=dev, generator=gen)
        xt = (1 - t)[..., None] * z0 + t[..., None] * x1
        v = model.velocity(xt, cond, t)
        return ((v - (x1 - z0)) ** 2).mean()
    pred = model.velocity(torch.zeros_like(x1), cond, torch.zeros(B, N, device=dev))
    return ((pred - x1) ** 2).mean()


# ---------------------------------------------------------------------------------------------------------------
# Validation (rank 0)
# ---------------------------------------------------------------------------------------------------------------

class ValData:
    """fixed validation set: n_pix pixels drawn uniformly from the mask (seed 0; scores are cos(lat)-weighted) on the
    hold-out days; truth = raw LGHAP integers. heldout_file (PROTOCOL.md A3(d')-4): data/processed/assim/heldout_sel2021.npz,
    the held-out monitors (inside the mask) with their values on the SEL days: scoring only, monitors sharing a pixel
    averaged per pixel-day -> self.held = dict(pos (P,), truth (n_days, P) NaN = no value, n_monitors)"""

    def __init__(self, store, n_pix, days=None, seed=0, heldout_file=None):
        g = np.random.default_rng(seed)
        self.pix = np.sort(g.choice(store.n_pix, size=min(n_pix, store.n_pix), replace=False))
        self.days = list(range(len(store.hold_dates))) if days is None else list(days)
        self.dates = [store.hold_dates[t] for t in self.days]
        self.truth = np.stack([store.holdout(t)[self.pix] for t in self.days])
        self.held = None
        if heldout_file is not None:
            z = np.load(heldout_file)
            pos, daily, sel_doy = np.asarray(z["pos"], np.int64), np.asarray(z["daily"], np.float32), list(z["sel_doy"])
            assert (pos >= 0).all() and pos.max() < store.n_pix
            upos, inv = np.unique(pos, return_inverse=True)
            truth = np.full((len(self.days), upos.size), np.nan, np.float32)
            for i, d in enumerate(self.dates):
                col = sel_doy.index((d - dt.date(d.year, 1, 1)).days)
                v = daily[:, col]
                ok = np.isfinite(v)
                s_ = np.bincount(inv[ok], v[ok].astype(np.float64), upos.size)
                c_ = np.bincount(inv[ok], minlength=upos.size)
                truth[i] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
            self.held = dict(pos=upos, truth=truth, n_monitors=int(pos.size), network=z["network"].astype(str), inv=inv)


def _ug(summary):
    u = summary["ug"]
    return {k: u[k] for k in ("n", "crps_fair", "rmse_ensmean", "mae_ensmean", "bias") if k in u}


@torch.no_grad()
def validate(model, store, pix, val, args, spec, device, baselines=False, stations=None):
    """EMA model on the validation set: M members x args.val_ode_steps Euler steps (noise of member m on day t fixed
    across validations: generator seed (args.seed, t)); returns (model summary, {baseline: summary} or {}, extra).
    stations (PROTOCOL.md A3(d')-4; a StationSet with the validation year): the day's assimilation monitors give the
    station features (used by the condition only when the model has stations: a pmG run gets the distances only) ->
    extra = {"by_distance": CRPS of the validation pixels <= 10 km / > 10 km from the nearest assimilated monitor,
    "heldout": model, climatology median and CAMS bilinear at the held-out monitor pixels (val.held)}; else extra = {}"""
    model.eval()
    k = torch.from_numpy(val.pix).to(device)
    lat, lon, w, stat = pix.lat[k], pix.lon[k], pix.w_area[k], pix.static[k]
    P = k.numel()
    regression = args.objective == "regression"
    M = 1 if regression else args.val_members
    sc = Scores(M, device)
    base = {"climatology_median": Scores(1, device), "cams_bilinear": Scores(1, device)} if baselines else {}
    use_stn = stations is not None
    stn_model = use_stn and bool(getattr(model, "stations", False))
    if use_stn:
        assert not regression
        dist = {"near10km": Scores(M, device), "far10km": Scores(M, device)}
        held = {"model": Scores(M, device), "climatology_median": Scores(1, device), "cams_bilinear": Scores(1, device)}
        hk = torch.from_numpy(val.held["pos"]).to(device) if val.held is not None else None
        n_assim = []
    for i, (t, d) in enumerate(zip(val.days, val.dates)):
        doy = torch.tensor([doy_of(d)], device=device)
        raw = raw_to_device(store.raw_inputs(d), device)
        feat = model.encode(assemble_inputs(raw, pix, args.grid, args.transport))
        mu, sig = pix.mu_sigma(k[None], doy)
        truth = torch.from_numpy(val.truth[i].astype(np.float32)).to(device).clamp(min=1.0)
        if getattr(store, "has_holes", False):      # GHAP (PROTOCOL.md A2(a)): a hole (-1) gets zero weight
            w = pix.w_area[k] * torch.from_numpy(val.truth[i] >= 0).to(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) * 1_000_003 + int(t) * 1009 + 17)
        params = None if regression else draw_sphere_noise(M, 1, spec, device, gen)
        ens = torch.empty(M, P, device=device)
        if use_stn:
            mlat, mlon, mval, mxyz = stations.day(d)            # every valid assimilation monitor (no dropout)
            n_assim.append(int(mval.numel()))
            dmin_all = torch.empty(P, device=device)
        for a in range(0, P, args.val_chunk):
            e = min(P, a + args.val_chunk)
            stn = None
            if use_stn:
                f41, dmin = station_features(lat[a:e], lon[a:e], mlat, mlon, mval, mxyz=mxyz)
                dmin_all[a:e] = dmin
                if stn_model:
                    stn = f41[None]
            cond = model.condition(feat, lat[None, a:e], lon[None, a:e], stat[None, a:e], doy, stn=stn)
            if regression:
                ens[0, a:e] = model.velocity(torch.zeros(1, e - a, 1, device=device), cond,
                                             torch.zeros(1, e - a, device=device))[0, :, 0]
            else:
                z0 = eval_sphere_noise(params, lat[None, a:e].expand(M, -1), lon[None, a:e].expand(M, -1),
                                       chunk=noise_chunk(M, spec))
                z = integrate(model, z0, cond.expand(M, -1, -1), args.val_ode_steps,
                              flowmap=args.objective == "flowmap")
                ens[:, a:e] = z[..., 0]
        X = z_to_ug(ens, mu, sig)
        sc.add(X, truth, w)
        if baselines:
            base["climatology_median"].add(z_to_ug(torch.zeros(1, P, device=device), mu, sig), truth, w)
            cams = bilinear_log1p_at(raw["cams"][0], lat, lon, store.grid_native)
            base["cams_bilinear"].add(cams[None], truth, w)
        if use_stn:
            near = (dmin_all <= 10.0).float()
            dist["near10km"].add(X, truth, w * near)
            dist["far10km"].add(X, truth, w * (1.0 - near))
            if hk is not None:
                ht = torch.from_numpy(val.held["truth"][i]).to(device)
                hok = torch.isfinite(ht)
                if hok.any():
                    hlat, hlon, hstat = pix.lat[hk], pix.lon[hk], pix.static[hk]
                    hmu, hsig = pix.mu_sigma(hk[None], doy)
                    f41h, _ = station_features(hlat, hlon, mlat, mlon, mval, mxyz=mxyz)
                    condh = model.condition(feat, hlat[None], hlon[None], hstat[None], doy, stn=f41h[None] if stn_model else None)
                    z0h = eval_sphere_noise(params, hlat[None].expand(M, -1), hlon[None].expand(M, -1), chunk=noise_chunk(M, spec))
                    zh = integrate(model, z0h, condh.expand(M, -1, -1), args.val_ode_steps, flowmap=args.objective == "flowmap")
                    hw = hok.float()
                    hy = torch.where(hok, ht, torch.ones_like(ht)).clamp(min=0.0)
                    held["model"].add(z_to_ug(zh[..., 0], hmu, hsig), hy, hw)
                    held["climatology_median"].add(z_to_ug(torch.zeros(1, hk.numel(), device=device), hmu, hsig), hy, hw)
                    held["cams_bilinear"].add(bilinear_log1p_at(raw["cams"][0], hlat, hlon, store.grid_native)[None], hy, hw)
    extra = {}
    if use_stn:
        extra = {"n_assim_valid_per_day": n_assim, "model_uses_stations": stn_model,
                 "by_distance": {n_: _ug(s.summary()) for n_, s in dist.items()},
                 "heldout": {n_: _ug(s.summary()) for n_, s in held.items()} if hk is not None else None,
                 "heldout_n_pixels": None if hk is None else int(hk.numel()),
                 "heldout_n_monitors": None if hk is None else val.held["n_monitors"]}
    return sc.summary(), {n_: s.summary() for n_, s in base.items()}, extra


def _short(res):
    u, lg = res["ug"], res["log"]
    s = f"crps={u['crps_fair']:.3f} rmseM={u['rmse_ensmean']:.3f} rmse1={u['rmse_1member']:.3f} bias={u['bias']:+.3f}"
    if "spread_skill" in u:
        s += f" ss={u['spread_skill']:.2f} ssF={u['spread_skill_fortin']:.2f}"
    return s + f" | log1p crps={lg['crps_fair']:.4f} rmseM={lg['rmse_ensmean']:.4f}"


# ---------------------------------------------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------------------------------------------

class AQModule(pl.LightningModule):
    def __init__(self, args, store):
        super().__init__()
        self.args = args
        self.store = store                     # plain object: its arrays are not part of the state dict
        self.spec = noise_spec(args)
        self.grid = model_grid(store.name, args.grid)
        self.model = build_model(args, in_channels(args.transport), store.static_dim, self.grid)
        self.teacher = None
        if args.objective == "flowmap":
            self._init_flowmap()
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.run_dir = Path(args.runs_root) / args.name
        self.t0 = time.time()
        self.stream = self.pix = self.val = self.gen = None
        self.stations = None                   # A3: StationSet of the assimilation monitors (training + SEL years)
        self.tune_val = None                   # A4: TuneVal at the tune-held-out monitors (TunedStore runs)
        self._last_days = None
        self._baselines_done = False
        self._validated = set()
        self._win = None
        self._val_s = 0.0                      # validation seconds so far (rank 0), excluded from ms/step
        self._last_window = {}

    def _init_flowmap(self):
        from .eval_aq import load_run
        targs, teacher, step = load_run(self.args.teacher, torch.device("cpu"), "best", self.args.runs_root)
        for k in ("grid", "transport", "lifting_dim", "doy_cond", "noise_features", "noise_beta", "noise_lmin_km",
                  "noise_lmax_km", "store", "fires"):
            if getattr(targs, k, None) != getattr(self.args, k, None):
                raise SystemExit(f"--{k.replace('_', '-')} differs from teacher {self.args.teacher}: "
                                 f"{getattr(self.args, k, None)} vs {getattr(targs, k, None)}")
        sd = {k: v for k, v in teacher.state_dict().items() if not k.startswith("velocity_net.")}
        missing = self.model.load_state_dict(sd, strict=False)
        assert all(k.startswith("velocity_net.") for k in missing.missing_keys), missing
        from model import TwoTimeVelocityNet
        self.model.velocity_net = TwoTimeVelocityNet.from_teacher(teacher.velocity_net)
        for n_, p in self.model.named_parameters():
            p.requires_grad_(n_.startswith("velocity_net."))
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = [teacher.eval()]            # list: kept out of the state dict and of DDP
        print(f"[flowmap] teacher {self.args.teacher} (step {step})", flush=True)

    def load_init(self, path):
        """A4 (and any fine-tuning): start from the EMA weights of another run's checkpoint; model and EMA both get them,
        the optimiser starts fresh (used only when the run is not resumed from its own last.ckpt)"""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
        self.model.load_state_dict(sd)
        self.ema.load_state_dict(sd)
        print(f"[init] model and EMA weights from {path} (step {ck.get('global_step', '?')})", flush=True)

    # -- set-up on the rank's device, after DDP initialisation and checkpoint restore
    def on_train_start(self):
        a, dev, rank = self.args, self.device, self.global_rank
        self.pix = PixelArrays(self.store, dev)
        start = int(self.global_step)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(np.random.default_rng([a.seed, rank, start, 7]).integers(2 ** 62)))
        self.stream = BatchStream(self.store, a.batch_size, a.n_points, a.seed, rank, dev, start_step=start)
        if getattr(a, "stations", None):
            years = sorted({d.year for d in self.store.train_dates} | {d.year for d in self.store.hold_dates})
            self.stations = open_stations(a, years, dev, store=self.store)
            print(f"[rank {rank}] stations: {self.stations.n_total:,} assimilation monitors over {years} from "
                  f"{self.stations.root} (held-out ids refused: {len(self.stations.heldout_ids):,}; model.stations="
                  f"{self.model.stations})", flush=True)
            if lag_spec(a) is not None:
                print(f"[rank {rank}] A11 M lag option: {lag_spec(a)} | station MLP input {self.model.stn_dim} | "
                      f"validation lags {a.val_lags} | held-out value files firewalled", flush=True)
        if self.teacher:
            self.teacher[0].to(dev)
        if isinstance(self.store, TunedStore) and rank == 0:
            self.tune_val = TuneVal(self.store.targets, a.tune_val_days)
            print(f"[rank 0] A4 tuning: {json.dumps(self.store.targets.summary())} | replay {self.store.replay} | tune "
                  f"validation {len(self.tune_val.dates)} days x {self.tune_val.pos.size} tune-held-out pixels "
                  f"({self.tune_val.n_monitor_days} pixel-days)", flush=True)
        mfile = self.run_dir / "metrics.jsonl"
        if rank == 0 and mfile.exists():          # resumed run: steps already validated (Lightning re-validates the
            for line in open(mfile):              # step of a checkpoint written at validation end)
                if line.strip():
                    r = json.loads(line)
                    self._validated.add(int(r["step"]))
                    self._baselines_done |= "baselines" in r
        print(f"[rank {rank}/{self.trainer.world_size}] device={dev} start_step={start} store={self.store.name} "
              f"grid={self.grid.H}x{self.grid.W} in_ch={in_channels(a.transport)} B={a.batch_size} N={a.n_points}",
              flush=True)

    def training_step(self, batch, batch_idx):
        b = self.stream.next()
        self._last_days = b["day"]
        if self.global_step < 3:
            print(f"[rank {self.global_rank}] step {self.global_step} days {b['day'].tolist()}", flush=True)
        loss = compute_losses(self.model, self.pix, b, self.args, self.spec, self.gen,
                              self.teacher[0] if self.teacher else None, stations=self.stations,
                              dates=[self.store.train_dates[int(t)] for t in b["day"]], step=int(self.global_step))
        self.log("train/loss", loss.detach(), on_step=True, on_epoch=False)
        if self._win is not None:
            self._win["loss"] += loss.detach()
            self._win["n"] += 1
        return loss

    def on_before_optimizer_step(self, optimizer):
        """total gradient norm before any clipping (DDP: gradients already averaged), kept on the device"""
        if self._win is not None:
            sq = torch.zeros((), device=self.device)
            for p in self.model.parameters():
                if p.grad is not None:
                    sq = sq + p.grad.detach().float().pow(2).sum()
            gn = sq.sqrt()
            self._win["gn_sum"] += gn
            self._win["gn_max"] = torch.maximum(self._win["gn_max"], gn)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            d = self.args.ema
            for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        step = int(self.global_step)
        if step % self.args.log_every == 0 or step in (1, 10):
            self._log_window(step)

    def _log_window(self, step):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.time()
        w = self._win
        if w is not None and w["n"] > 0 and self.global_rank == 0:
            n = w["n"]
            mem = torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if self.device.type == "cuda" else float("nan")
            vs = self._val_s - w["val"]
            self._last_window = dict(step=step, loss=float(w["loss"]) / n, ms_per_step=(now - w["t"] - vs) / n * 1000,
                                     data_wait_ms=(self.stream.wait_s - w["wait"]) / n * 1000,
                                     fill_ms_per_batch=(self.stream.fill_s - w.get("fill", 0.0)) / n * 1000,
                                     fill_mb_s=(self.stream.fill_bytes - w.get("bytes", 0)) / 2 ** 20
                                     / max(self.stream.fill_s - w.get("fill", 0.0), 1e-9),
                                     grad_norm_mean=float(w["gn_sum"]) / n, grad_norm_max=float(w["gn_max"]),
                                     max_mem_gib=mem)
            lw = self._last_window
            print(f"[step {step}] loss {lw['loss']:.4f} | {lw['ms_per_step']:.0f} ms/step over {n} steps, validation "
                  f"excluded (data wait {lw['data_wait_ms']:.0f} ms/step) | grad norm mean {lw['grad_norm_mean']:.3g} "
                  f"max {lw['grad_norm_max']:.3g} (pre-clip) | max mem {mem:.2f} GiB | elapsed "
                  f"{(now - self.t0) / 60:.1f} min | loader {lw['fill_ms_per_batch']:.0f} ms/batch "
                  f"{lw['fill_mb_s']:.0f} MB/s (rank 0 prefetch thread)", flush=True)
        z = lambda: torch.zeros((), device=self.device)
        self._win = dict(loss=z(), n=0, t=now, wait=self.stream.wait_s, val=self._val_s, gn_sum=z(), gn_max=z(),
                         fill=self.stream.fill_s, bytes=self.stream.fill_bytes)

    def validation_step(self, batch, batch_idx):
        self._run_val()

    def _ddp_checks(self):
        """collective, on every rank: weights identical across ranks? days of the last batch per rank"""
        if self.trainer.world_size == 1:
            return {}
        vec = torch.stack([p.detach().double().sum() for p in self.model.parameters()])
        allv = self.all_gather(vec)
        days = self.all_gather(self._last_days.to(self.device) if self._last_days is not None
                               else torch.full((self.args.batch_size,), -1, device=self.device))
        return dict(ddp_weights_max_abs_diff=float((allv - allv[0:1]).abs().max()),
                    ddp_last_days_by_rank=days.tolist())

    def _run_val(self):
        info = self._ddp_checks()
        if self.global_rank != 0:
            return
        if int(self.global_step) in self._validated:
            print(f"[val step {self.global_step}] already in metrics.jsonl (resumed run): skipped", flush=True)
            return
        a = self.args
        lagged = lag_spec(a) is not None            # A11 experiment M
        if self.val is None:
            # A11 experiment M copy: the held-out monitor file of A3(d')-4 (<stations>/heldout_sel2021.npz, which the
            # original opened here to SCORE a --stations run) is never opened by this package, with the lag option on or
            # off: no code of the experiment reads a held-out value. It is also behind the firewall (open_stations).
            self.val = ValData(self.store, a.val_pixels, None if a.val_days is None else
                               [int(x) for x in a.val_days.split(",")], heldout_file=None)
        t0 = time.time()
        if lagged:
            lv = validate_lag([dict(name="model", model=self.ema, kind="lag", grid=a.grid, transport=a.transport,
                                    flowmap=a.objective == "flowmap")], self.store, self.pix, self.val, self.stations,
                              [int(x) for x in str(a.val_lags).split(",")], a.val_members, a.val_ode_steps, self.spec,
                              a.seed, self.device, chunk=a.val_chunk, baselines=not self._baselines_done)
            res, base, extra = lv["pooled"]["model"], lv["baselines"], {}
        else:
            res, base, extra = validate(self.ema, self.store, self.pix, self.val, a, self.spec, self.device,
                                        baselines=not self._baselines_done, stations=self.stations)
        rec = dict(step=int(self.global_step), elapsed_min=(time.time() - self.t0) / 60,
                   val_seconds=time.time() - t0, val_days=len(self.val.days), val_pixels=len(self.val.pix),
                   model=res, train_window=self._last_window, **info)
        if lagged:
            rec["lag"] = dict(lags=lv["lags"], classes=lv["classes"], scores=lv["scores"], info=lv["info"],
                              model_is="the lags added up, with the monitors of day t - lag (class 'all')",
                              nostations=lv["pooled"].get("model|nostations"))
        if extra:
            rec["stations"] = extra
        if self.tune_val is not None:
            rec["tune"] = validate_tune(self.ema, self.store, self.pix, self.tune_val, a, self.spec, self.device)
            rec["val_seconds"] = time.time() - t0
        self._val_s += rec["val_seconds"]
        if base:
            rec["baselines"] = base
            self._baselines_done = True
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._validated.add(rec["step"])
        print(f"[val step {rec['step']}] model {_short(res)} ({rec['val_seconds']:.0f} s)", flush=True)
        if "tune" in rec:
            th = rec["tune"]["tune_heldout"]
            print(f"[val step {rec['step']}] tune-held-out monitors ({rec['tune']['n_pixels']} px, {rec['tune']['n_pixel_days']} px-days): "
                  f"model crps {th['model']['crps_fair']:.3f} rmseM {th['model']['rmse_ensmean']:.3f} bias {th['model']['bias']:+.3f} | "
                  f"clim {th['climatology_median']['crps_fair']:.3f} | cams {th['cams_bilinear']['crps_fair']:.3f}", flush=True)
        for n_, r in base.items():
            print(f"[val step {rec['step']}] {n_} {_short(r)}", flush=True)
        if lagged:
            sc_ = rec["lag"]["scores"]
            g_ = lambda lb, lg, c, key="crps_fair": (f"{sc_[lb][str(lg)][c][key]:.3f}" if sc_[lb][str(lg)][c] else "-")
            for lg in rec["lag"]["lags"]:
                print(f"[val step {rec['step']}] lag {lg}: crps with monitors of day t-{lg} / every monitor hidden: "
                      + " | ".join(f"{c} {g_('model', lg, c)} / {g_('model|nostations', lg, c)}" for c in rec["lag"]["classes"])
                      + f" | rmseM all {g_('model', lg, 'all', 'rmse_ensmean')} ssF all "
                      f"{g_('model', lg, 'all', 'spread_skill_fortin')}", flush=True)
            if not (rec["lag"]["info"]["members_finite"] and rec["lag"]["info"]["scores_finite"]):
                print(f"[val step {rec['step']}] WARNING: non-finite members or scores in the lag validation", flush=True)
        if extra:
            bd, hd = extra["by_distance"], extra.get("heldout")
            print(f"[val step {rec['step']}] stations: crps near10km {bd['near10km']['crps_fair']:.3f} (w {bd['near10km']['n']:.0f}) "
                  f"far10km {bd['far10km']['crps_fair']:.3f} (w {bd['far10km']['n']:.0f})"
                  + (f" | held-out monitors ({extra['heldout_n_pixels']} px): model crps {hd['model']['crps_fair']:.3f} rmseM "
                     f"{hd['model']['rmse_ensmean']:.3f} | clim {hd['climatology_median']['crps_fair']:.3f} | cams "
                     f"{hd['cams_bilinear']['crps_fair']:.3f}" if hd else ""), flush=True)
        if info:
            print(f"[val step {rec['step']}] DDP: weights max |diff| across ranks {info['ddp_weights_max_abs_diff']:.1e}"
                  f"; last days by rank {info['ddp_last_days_by_rank']}", flush=True)

    def on_train_end(self):
        if self.global_step % self.args.val_every != 0:          # stopped between validations (max_time)
            self._run_val()
            ck = self.run_dir / "ckpt"
            self.trainer.save_checkpoint(str(ck / f"step{int(self.global_step):06d}.ckpt"))
            self.trainer.save_checkpoint(str(ck / "last.ckpt"))
        if self.stream is not None:
            self.stream.close()
        if self.global_rank == 0:
            print(f"[done] {self.global_step} steps in {(time.time() - self.t0) / 60:.1f} min -> {self.run_dir}",
                  flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.args.lr,
                                weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, self.args))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(_Placeholder(), batch_size=None)

    def val_dataloader(self):
        return DataLoader(_Placeholder(1), batch_size=None)


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs-root", default=str(RUNS_ROOT))
    ap.add_argument("--store", choices=["smoke", "toy", "global", "ghap"], default="smoke")
    ap.add_argument("--fires", choices=["injh", "mami"], default=None,
                    help="global store only: channel 40 (default injh from 2018-07-02, mami before; mami = tests)")
    ap.add_argument("--split", default=None, help="global store only: training split of the manifest (A = train "
                    "2019-07-11..2022-12-31, default; A21 = ..2021-12-31 once built)")
    ap.add_argument("--objective", choices=["fm", "regression", "flowmap"], default="fm")
    ap.add_argument("--teacher", default=None, help="flowmap: the trained fm run to distil (its best checkpoint)")
    ap.add_argument("--fm-substeps", type=int, default=8)
    ap.add_argument("--fm-diag-prob", type=float, default=0.25)
    ap.add_argument("--grid", choices=["025", "05"], default="025", help="encoder grid (05 = S1 fallback)")
    ap.add_argument("--transport", choices=["t6", "none"], default="t6", help="none = ablation A1 (41 channels)")
    ap.add_argument("--lifting-dim", type=int, default=None)
    ap.add_argument("--use-checkpoint", action="store_true", help="Flower activation checkpointing (memory)")
    ap.add_argument("--no-doy-cond", dest="doy_cond", action="store_false")
    ap.add_argument("--noise-fit", default=str(NOISE_FIT_SMOKE), help="JSON of scripts/measure_noise_beta.py")
    ap.add_argument("--noise-features", type=int, default=None)
    ap.add_argument("--noise-beta", type=float, default=None)
    ap.add_argument("--noise-lmin-km", type=float, default=None)
    ap.add_argument("--noise-lmax-km", type=float, default=None)
    ap.add_argument("--stations", default=None,
                    help="PROTOCOL.md A3: root of the assimilation monitor files (data/processed/assim) -> station-"
                         "conditioned flow (41 features -> MLP 32 -> condition) with training dropout; default off "
                         "(the A1/A2 code path is unchanged)")
    ap.add_argument("--stn-lag", default=None,
                    help="A11 experiment M (with --stations): comma-separated lags in 1..5, e.g. 1,2,3,4,5: the station "
                         "features of a training day t are those of the monitors of day t - lag (lag drawn per day and "
                         "step), with their anomaly and the lag (63 features); default off = the A3 path of pmS_base")
    ap.add_argument("--stn-lag0-prob", type=float, default=0.0,
                    help="A11 M: probability of the same-day lag 0 at training (keeps the A3 analysis in the same model)")
    ap.add_argument("--val-lags", default="1,5", help="A11 M: lags of the in-training validation (0..5)")
    ap.add_argument("--sigma-k", type=int, default=2, choices=(0, 2),
                    help="target sigma: 2 = seasonal (K = 2 fit; runs before 2026-10-27), 0 = annual (PROTOCOL.md A1(e))")
    ap.add_argument("--batch-size", type=int, default=2, help="days per rank")
    ap.add_argument("--n-points", type=int, default=16384, help="points per day")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--stop-at-step", type=int, default=None,
                    help="end this segment at that step; the LR schedule still runs to --max-steps (continue with "
                         "--resume and no --stop-at-step)")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="clip-by-norm threshold, 0 = off (default; see the module docstring: CD-FFM's 1.0 would "
                         "freeze most of the network on the 0.25 deg globe)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-days", default=None, help="comma-separated hold-out day indices (default: all)")
    ap.add_argument("--val-pixels", type=int, default=100000)
    ap.add_argument("--val-members", type=int, default=8)
    ap.add_argument("--val-ode-steps", type=int, default=10)
    ap.add_argument("--val-chunk", type=int, default=50000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--precision", default="32-true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <run>/ckpt/last.ckpt if present")
    ap.add_argument("--init-from", default=None, help="checkpoint whose EMA weights initialise model and EMA (fine-tuning; "
                    "ignored when the run resumes from its own last.ckpt)")
    ap.add_argument("--freeze-encoder", action="store_true", help="A5(b'): train only the pointwise head (LIIF query + velocity net)")
    ap.add_argument("--tune-stations", default=None, help="A4: assimilation root (data/processed/assim): the monitors' daily "
                    "means of --tune-years are the training targets (TunedStore); --stations stays off")
    ap.add_argument("--tune-years", default="2022-2026")
    ap.add_argument("--tune-end", default="2026-08-31", help="last tuning day; later days = temporal test")
    ap.add_argument("--tune-replay", type=float, default=0.0, help="share of the training days that are GHAP days (targets GHAP)")
    ap.add_argument("--tune-val-days", type=int, default=24)
    ap.add_argument("--tune-split", default=None, help="split file (default <tune-stations>/tune_split_v1.json); A5: tune_split_v2.json")
    ap.add_argument("--tune-min-monitors", type=int, default=50)
    ap.add_argument("--device", default="auto", help="auto (GPU if any) | cpu")
    return ap


def launch_env():
    """(plugins, devices, num_nodes): torchrun -> TorchElastic; srun with > 1 task -> SLURM (auto-detected);
    otherwise a plain single process (LightningEnvironment, as CD-FFM, also inside a 1-task Slurm job)"""
    e = os.environ
    if "TORCHELASTIC_RUN_ID" in e:
        local = int(e["LOCAL_WORLD_SIZE"])
        return [TorchElasticEnvironment()], local, int(e["WORLD_SIZE"]) // local
    if int(e.get("SLURM_NTASKS", "1")) > 1:
        return [], int(e["SLURM_NTASKS_PER_NODE"].split("(")[0]), int(e.get("SLURM_NNODES", "1"))
    return [LightningEnvironment()], 1, 1


def env_rank():
    e = os.environ
    return int(e.get("RANK", e.get("SLURM_PROCID", "0")))


def main(argv=None):
    args = parser().parse_args(argv)
    resolve_noise(args)
    plugins, devices, num_nodes = launch_env()
    use_gpu = args.device != "cpu" and torch.cuda.is_available()
    if args.store == "toy" and args.val_pixels > 5000:
        args.val_pixels = 5000
    pl.seed_everything(args.seed, workers=False, verbose=False)
    if args.store in ("global", "ghap"):
        args.fires, args.split, args.sigma_k = (store_kwargs(args)[k] for k in ("fires", "split", "sigma_k"))
    store = open_store(args.store, **store_kwargs(args))
    if args.tune_stations:                                                                 # A4: station targets
        if args.store != "ghap" or args.stations:
            raise SystemExit("--tune-stations: GHAP store only, without --stations (PROTOCOL.md A4)")
        targets = TuneTargets(args.tune_stations, parse_years(args.tune_years), dt.date.fromisoformat(args.tune_end),
                              Path(args.tune_split) if getattr(args, "tune_split", None) else Path(args.tune_stations) / TUNE_SPLIT.name,
                              args.tune_min_monitors,
                              available=lambda d: d < D2025 or (store.bf.last_day is not None and d <= store.bf.last_day))
        store = TunedStore(store, targets, args.tune_replay, args.seed)
    args.static_dim = store.static_dim
    args.in_ch = in_channels(args.transport)
    args.stn_dim = STN_DIM if args.stations else 0
    if lag_spec(args) is not None:                                                         # A11 experiment M
        if args.tune_stations or args.objective != "fm":
            raise SystemExit("--stn-lag: flow matching only, and never with --tune-stations (a monitor is an input of "
                             "experiment M, never a training target)")
        args.stn_dim = stn_dim_of(args)
        bad = [x for x in str(args.val_lags).split(",") if not x.strip().isdigit() or not 0 <= int(x) <= 5]
        if bad:
            raise SystemExit(f"--val-lags {args.val_lags!r}: lags in 0..5")
    run_dir = Path(args.runs_root) / args.name
    if not args.resume and (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a run (metrics.jsonl): use --resume or another --name")
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    if env_rank() == 0:
        with open(run_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=1)
    module = AQModule(args, store)
    last0 = run_dir / "ckpt" / "last.ckpt"
    if args.init_from and not (args.resume and last0.exists()):
        module.load_init(args.init_from)
    if getattr(args, "freeze_encoder", False):                                            # A5(b') ablation: the 0.25 deg encoder
        for p_ in module.model.encoder.parameters():                                      # keeps pmG's weights; only the pointwise
            p_.requires_grad_(False)                                                      # head (LIIF query + velocity net) learns
        if env_rank() == 0:
            print(f"[freeze] encoder frozen: {sum(p.numel() for p in module.model.encoder.parameters()):,} parameters; trainable "
                  f"{sum(p.numel() for p in module.model.parameters() if p.requires_grad):,}", flush=True)
    if env_rank() == 0:
        n = sum(p.numel() for p in module.model.parameters())
        ne = sum(p.numel() for p in module.model.encoder.parameters())
        nq = sum(p.numel() for p in module.model.query.parameters())
        print(f"[model] objective={args.objective} grid={args.grid} in_ch={args.in_ch} static={args.static_dim} "
              f"stations={'on (' + str(args.stations) + ')' if args.stations else 'off'} "
              f"{'lag ' + str(lag_spec(args)) + ' ' if lag_spec(args) else ''}"
              f"params={n:,} (encoder {ne:,}, LIIF {nq:,}, velocity {n - ne - nq:,}) | noise beta={args.noise_beta} "
              f"{args.noise_lmin_km}-{args.noise_lmax_km} km F={args.noise_features} ({args.noise_source}) | "
              f"world={devices * num_nodes} x B={args.batch_size} days x N={args.n_points}", flush=True)
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
