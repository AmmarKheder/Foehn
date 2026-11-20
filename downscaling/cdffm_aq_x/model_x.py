"""cdffm_aq_x.model_x -- the variant mechanisms (see the package docstring) and one member sampler for all of them.

estab  StableSelfWarp replaces Flower's SelfWarp: the flow head outputs displacements in CELL units (scaled by
       2 / (n - 1) per axis onto the align_corners=True normalised grid) and its last 1 x 1 conv is zero-initialised,
       so every warp starts as the identity. The function class is unchanged (a fixed linear reparametrisation of
       the flow-head output); what changes is the optimiser's geometry: at the baseline's init a unit output moves a
       sample by half the domain (720 cells at 0.25 deg) and the grid_sample coordinate gradient is ~1e5-1e7
       (S0/S2 diagnosis); here a unit output moves it by one cell.
hcf    CoarseVelocity: a convolutional velocity net on the encoder grid (periodic in longitude) for a flow that
       generates the 0.25 deg cell-mean field of z from the sphere noise evaluated at the cell centres. The
       pointwise flow is conditioned on the generated field through two extra LIIF feature channels (eta * mask,
       mask); training uses the true cell means (teacher forcing). The fine flow stays pointwise and resolution-free
       given eta; the coarse flow makes the coarse-scale uncertainty day-specific and spatially coherent.
Sampler  one code path per variant for validation and evaluation; for 'base' it consumes the generators exactly as
       train_aq.validate / eval_aq.main do (same members, so the baseline numbers are reproduced).
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cdffm_aq.model_aq import (AQFlowModel, PeriodicLIIFQuery, draw_sphere_noise, eval_sphere_noise, integrate,
                               noise_chunk)
from cdffm_aq import CDFFM_SCRIPTS

if str(CDFFM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CDFFM_SCRIPTS))
from flower_layers import FlowerBlock, SelfWarp, flow_fold_heads, grid_samples, value_fold_heads, value_unfold_heads  # noqa: E402
from model import SinusoidalTimeEmbed, TwoTimeVelocityNet  # noqa: E402

VARIANTS = ("base", "estab", "hcf")


# ---------------------------------------------------------------------------------------------------------------
# estab
# ---------------------------------------------------------------------------------------------------------------

class StableSelfWarp(SelfWarp):
    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads=32, padding_modes=("zeros", "zeros"),
                 meta_dim=0):
        super().__init__(in_channels, out_channels, spatial_resolution, num_heads=num_heads,
                         padding_modes=list(padding_modes), meta_dim=meta_dim)
        H, W = spatial_resolution
        # grid[..., 0] = x (width, W points), grid[..., 1] = y (height); align_corners=True: one cell = 2 / (n - 1)
        self.register_buffer("cell2norm", torch.tensor([2.0 / (W - 1), 2.0 / (H - 1)]).view(1, 1, 2, 1, 1))
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(self, u, meta=None):
        flow = self.flow_head(u)                                              # (B, heads * 2, H, W) in cells
        value = self.value_head(u)
        B, _, H, W = flow.shape
        flow = (flow.view(B, self.num_heads, 2, H, W) * self.cell2norm).reshape(B, self.num_heads * 2, H, W)
        flow = rearrange(flow, flow_fold_heads[2], dir=2, heads=self.num_heads)
        grid = self.base_grid.unsqueeze(0) + flow
        value = rearrange(value, value_fold_heads[2], heads=self.num_heads)
        u_warp = grid_samples[2](value, grid, mode="bilinear", padding_modes=self.padding_modes, align_corners=True)
        return rearrange(u_warp, value_unfold_heads[2], heads=self.num_heads)


def stabilise_flower(flower):
    """replace every SelfWarp of a Flower U-Net by a StableSelfWarp (fresh init); returns the number replaced"""
    n = 0
    for m in flower.modules():
        if isinstance(m, FlowerBlock):
            sw = m.self_warp
            m.self_warp = StableSelfWarp(sw.in_channels, sw.out_channels, sw.spatial_resolution, num_heads=sw.num_heads,
                                         padding_modes=sw.padding_modes)
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------
# hcf
# ---------------------------------------------------------------------------------------------------------------

class CoarseVelocity(nn.Module):
    """v(eta_t, mask, feat, t) on the encoder grid: 1 x 1 lift, `depth` residual 3 x 3 convs (circular padding in
    longitude, zeros in latitude), zero-initialised 1 x 1 output; t (B,) through a sinusoidal embedding"""

    def __init__(self, feat_dim=128, hidden=64, time_dim=16, depth=3, dilations=None):
        super().__init__()
        self.temb = SinusoidalTimeEmbed(time_dim)
        self.inp = nn.Conv2d(feat_dim + 2 + time_dim, hidden, 1)
        self.dilations = tuple(int(d) for d in (dilations or (1,) * depth))
        assert len(self.dilations) == depth, (self.dilations, depth)
        self.convs = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, dilation=d) for d in self.dilations])
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, eta_t, mask, feat, t):
        B, _, H, W = eta_t.shape
        te = self.temb(t.reshape(B)).to(feat.dtype)[:, :, None, None].expand(B, -1, H, W)
        x = F.gelu(self.inp(torch.cat([eta_t, mask.expand(B, -1, -1, -1), feat, te], 1)))
        for c, d in zip(self.convs, self.dilations):
            xp = F.pad(F.pad(x, (d, d, 0, 0), mode="circular"), (0, 0, d, d))
            x = x + F.gelu(c(xp))
        return self.out(x)


class AQFlowModelHCF(AQFlowModel):
    def __init__(self, in_ch, grid, static_dim, feat_dim=128, cond_mlp_dim=64, coarse_hidden=64, coarse_depth=3,
                 coarse_dilations=None, **kw):
        super().__init__(in_ch, grid, static_dim, feat_dim=feat_dim, cond_mlp_dim=cond_mlp_dim, **kw)
        self.query = PeriodicLIIFQuery(feat_dim + 2, cond_mlp_dim, grid, static_dim=static_dim)
        self.coarse = CoarseVelocity(feat_dim, hidden=coarse_hidden, depth=coarse_depth, dilations=coarse_dilations)

    @staticmethod
    def with_coarse(feat, eta, mask):
        """feature map seen by the LIIF: (B, C + 2, H, W) = [feat, eta * mask, mask]"""
        B = feat.shape[0]
        return torch.cat([feat, eta * mask, mask.expand(B, -1, -1, -1)], 1)

    def coarse_velocity(self, eta_t, mask, feat, t):
        return self.coarse(eta_t, mask, feat, t)

    @torch.no_grad()
    def sample_coarse(self, feat, mask, eps, n_steps):
        """Euler from the coarse noise eps (M, 1, H, W) with the day's features (1, C, H, W) -> eta_1 (M, 1, H, W)"""
        M = eps.shape[0]
        x = eps
        dt = 1.0 / n_steps
        f = feat.expand(M, -1, -1, -1)
        for s in range(n_steps):
            t = torch.full((M,), s * dt, device=x.device)
            x = x + self.coarse(x, mask, f, t) * dt
        return x


# ---------------------------------------------------------------------------------------------------------------
# construction / loading
# ---------------------------------------------------------------------------------------------------------------

def build_model_x(args, in_ch, static_dim, grid):
    """args: namespace with variant (default base), lifting_dim, use_checkpoint, doy_cond, objective"""
    variant = getattr(args, "variant", "base") or "base"
    kw = dict(doy_cond=getattr(args, "doy_cond", True), lifting_dim=getattr(args, "lifting_dim", None) or 64,
              use_checkpoint=getattr(args, "use_checkpoint", False), stations=bool(getattr(args, "stations", None)))
    if variant == "hcf":
        dil = getattr(args, "coarse_dilations", None)
        dil = [int(x) for x in str(dil).replace(":", ",").replace("x", ",").split(",")] if dil else None
        m = AQFlowModelHCF(in_ch, grid, static_dim, coarse_hidden=getattr(args, "coarse_hidden", 64),
                           coarse_depth=getattr(args, "coarse_depth", 3), coarse_dilations=dil, **kw)
    elif variant in ("base", "estab"):
        m = AQFlowModel(in_ch, grid, static_dim, **kw)
        if variant == "estab":
            stabilise_flower(m.encoder.flower)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if getattr(args, "objective", "fm") == "flowmap":
        m.velocity_net = TwoTimeVelocityNet.from_teacher(m.velocity_net)
    return m


def load_run_x(name, device, which="best", runs_root=None):
    """(args, EMA model in eval mode on device, step) of a train_x OR train_aq run (variant base if absent)"""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from cdffm_aq.eval_aq import ckpt_path
    from cdffm_aq.train_aq import model_grid
    from cdffm_aq.data_aq import in_channels
    from . import RUNS_X
    run_dir = Path(runs_root or RUNS_X) / name
    args = SimpleNamespace(**json.loads((run_dir / "args.json").read_text()))
    if not hasattr(args, "variant"):
        args.variant = "base"
    ck = torch.load(ckpt_path(run_dir, which), map_location=device, weights_only=False)
    sd = {k[len("ema."):]: v for k, v in ck["state_dict"].items() if k.startswith("ema.")}
    model = build_model_x(args, in_channels(args.transport), args.static_dim, model_grid(args.store, args.grid))
    model.load_state_dict(sd)
    return args, model.to(device).eval(), int(ck.get("global_step", -1))


# ---------------------------------------------------------------------------------------------------------------
# member sampler (validation and evaluation)
# ---------------------------------------------------------------------------------------------------------------

class Sampler:
    """One day: start_day(feat, gen, gen_coarse) draws the per-day noise parameters (and, for hcf, the M coarse
    members); members(lat, lon, static, doy) returns (M, n) z at the points. For 'base'/'estab' the generator use is
    exactly that of train_aq.validate / eval_aq.main (params drawn once per day, fields evaluated per chunk), so the
    baseline numbers are reproduced. For 'hcf' the FINE noise uses the same generator the same way (common random
    numbers with the baseline); the coarse noise comes from gen_coarse. prior: optional object with
    start_day(gen_prior) and field(lat, lon, cond, static) -> (M, n, 1) replacing the stationary fine noise (copula_x).
    z_bound (PROTOCOL.md A1(e)): flow state kept in [-z_bound, z_bound] (model_aq.integrate); n_bound counts the
    member points that ended at the bound since the last reset (reported by eval_x / realtime_1km).
    z0_trunc (A1(g)): the base noise is clamped to [-z0_trunc, z0_trunc] before the flow (truncated base measure:
    draws beyond it are outside the trained support and below the resolution of an 8-member ensemble); n_trunc counts
    the clamped member points.
    stn_fn (PROTOCOL.md A3(c)): for a stations model, a callable (lat, lon) -> (n, 41) station features of the day
    (cdffm_aq.stations.StationSet.features), set by the caller per day; None = the no-monitor state at every point (the
    day-level counterfactual pmS_nostations)."""

    def __init__(self, model, variant, spec, M, ode_steps, device, cm=None, flowmap=False, prior=None,
                 coarse_spec=None, z_bound=None, z0_trunc=None, stn_fn=None):
        self.model, self.variant, self.spec, self.M, self.ode_steps = model, variant, spec, M, ode_steps
        self.device, self.cm, self.flowmap, self.prior = device, cm, flowmap, prior
        self.coarse_spec = coarse_spec or spec
        self.z_bound = None if z_bound is None else float(z_bound)
        self.z0_trunc = None if z0_trunc is None else float(z0_trunc)
        self.n_bound = self.n_trunc = 0
        self.params = self.feats = None
        self.stn_fn = stn_fn

    def start_day(self, feat, gen, gen_coarse=None, gen_prior=None):
        M = self.M
        self.params = draw_sphere_noise(M, 1, self.spec, self.device, gen)
        if self.prior is not None:
            self.prior.start_day(gen_prior if gen_prior is not None else gen)
        if self.variant == "hcf":
            assert self.cm is not None and gen_coarse is not None
            pc = draw_sphere_noise(M, 1, self.coarse_spec, self.device, gen_coarse)
            eps = self.cm.grid_noise(pc, chunk=noise_chunk(M, self.coarse_spec))
            if self.z0_trunc is not None:                # A1(g): the truncated base measure at both levels
                self.n_trunc += int((eps.abs() > self.z0_trunc).sum())
                eps = eps.clamp(-self.z0_trunc, self.z0_trunc)
            eta = self.model.sample_coarse(feat, self.cm.mask, eps, self.ode_steps)
            self.eta = eta
            self.feats = [self.model.with_coarse(feat, eta[m:m + 1], self.cm.mask) for m in range(M)]
        else:
            self.feats = feat

    def condition(self, lat, lon, static, doy):
        """(M, n, c) or (1, n, c) for the pointwise variants"""
        m = self.model
        stn = None
        if self.stn_fn is not None and getattr(m, "stations", False):
            stn = self.stn_fn(lat, lon)[None]
        if self.variant == "hcf":
            return torch.cat([m.condition(f, lat[None], lon[None], static[None], doy, stn=stn) for f in self.feats], 0)
        return m.condition(self.feats, lat[None], lon[None], static[None], doy, stn=stn)

    def members(self, lat, lon, static, doy):
        M = self.M
        cond = self.condition(lat, lon, static, doy)
        if cond.shape[0] == 1:
            cond = cond.expand(M, -1, -1)
        if self.prior is None:
            z0 = eval_sphere_noise(self.params, lat[None].expand(M, -1), lon[None].expand(M, -1),
                                   chunk=noise_chunk(M, self.spec))
        else:
            z0 = self.prior.field(lat, lon, cond, static)
        if self.z0_trunc is not None:
            self.n_trunc += int((z0.abs() > self.z0_trunc).sum())
            z0 = z0.clamp(-self.z0_trunc, self.z0_trunc)
        z = integrate(self.model, z0, cond, self.ode_steps, flowmap=self.flowmap, bound=self.z_bound)[..., 0]
        if self.z_bound is not None:
            self.n_bound += int((z.abs() >= self.z_bound).sum())
        return z
