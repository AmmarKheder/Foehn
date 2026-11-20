"""cdffm_aq.model_aq -- the network of the minimal CD-FFM adaptation (PLAN.md 2.5, 4.1, 4.2).

Kept from CD-FFM, imported from CD-FFM/scripts (not copied): the Flower U-Net (flower_layers.Flower: lifting 64,
4 levels, 8 heads, 8 groups, 128 output features), the velocity MLPs (model.FlowVelocityNet 4 x 256 GELU with the
sinusoidal time embedding, model.TwoTimeVelocityNet for the flow map), the doy (cos, sin) features and the
inverse-CDF power-law radial law of fast_data._rff_frequencies.

Changed, each forced by the global domain or the new target (PLAN.md 4.2):
  * PeriodicFlowerEncoder: in_ch = 47 channels on the whole globe (576 x 1440 at 0.25 deg; 288 x 720 at 0.5 deg),
    boundary_condition_types ['NONE', 'PERIODIC'] (Flower's own periodic SelfWarp in longitude). Flower's coordinate
    channels (x, y) = linspace(-1, 1) are replaced by (y, cos lon, sin lon): the x channel jumps from +1 to -1 at the
    dateline. The lift therefore takes in_ch + 3 channels. Convolutions are 1x1 or 2x2/stride 2 (no padding).
  * PeriodicLIIFQuery: CD-FFM's LIIF (3x3 feature unfold, 4-corner local ensemble with area weights, MLP
    9*128 + 2 + S -> 128 -> 128 -> 64) with (a) the cell-centre mapping of the point-registered grid (grid point r is
    the centre of cell r) and (b) periodic longitude (wrap-around neighbours, no clamping in longitude). It gathers
    the 4 x 4 cells around each query instead of unfolding the whole globe (9 x 128 channels x 829,440 cells would be
    3.8 GB per day); in the interior this is identical to CD-FFM's F.unfold + nearest lookups (test_s0 checks it).
  * n_out = 1 (z of log1p PM2.5).
  * stations (PROTOCOL.md A3(c), 2026-11-02, off by default so the A1/A2 code path is unchanged): the 41-float station
    feature vector of cdffm_aq.stations (K = 8 nearest assimilation monitors of the day) goes through a small MLP
    41 -> 32 -> 32 (GELU) and is concatenated to the pointwise condition after the LIIF output and the doy features
    (cond 64 + 2 + 32 = 98 -> the velocity net), so the flow learns at each point how much to trust the nearby
    monitors against CAMS/met. A point without features (stn=None) gets stations.none_features (the no-monitor state).
  * Sphere noise: CD-FFM's random-Fourier-feature power-law field with 3-D frequencies (directions uniform on S^2,
    radial density k^(1 - beta) on [1/lmax, 1/lmin] cycles/km) evaluated at p = R (cos phi cos lam, cos phi sin lam,
    sin phi): isotropic in km at every latitude and continuous across the dateline. A 3-D spectrum k^-(beta + 1)
    restricted to the surface has the 2-D per-mode slope -beta of CD-FFM's planar field (test_s0 checks it at 0, 30
    and 55 deg). Unit marginal variance, as CD-FFM.
"""
import math
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import CDFFM_SCRIPTS

if str(CDFFM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CDFFM_SCRIPTS))
from flower_layers import Flower  # noqa: E402  (CD-FFM, verbatim Flower)
from model import FlowVelocityNet, TwoTimeVelocityNet  # noqa: E402  (CD-FFM velocity nets)

R_EARTH_KM = 6371.0


# ---------------------------------------------------------------------------------------------------------------
# Grids: point (r, c) at (lat0 - dlat r, lon0 + dlon c); periodic in longitude (W * dlon = 360)
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    H: int
    W: int
    lat0: float
    lon0: float
    dlat: float
    dlon: float

    def __post_init__(self):
        assert abs(self.W * self.dlon - 360.0) < 1e-6, "the encoder grid must be global in longitude"

    def lats(self):
        return self.lat0 - self.dlat * np.arange(self.H)

    def lons(self):
        return self.lon0 + self.dlon * np.arange(self.W)


GRIDS = {
    "025": Grid(576, 1440, 84.0, -180.0, 0.25, 0.25),     # Foehn coarse grid (met.npy, fires.npy, CAMS regrid)
    "05": Grid(288, 720, 83.875, -179.875, 0.5, 0.5),     # 2 x 2 block means of the 0.25 grid = T6 cell centres
    "toy": Grid(72, 144, 84.0, -180.0, 2.0, 2.5),         # CPU shape test (Flower needs H, W divisible by 8)
}


def coord_channels(grid):
    """(1, 3, H, W): Flower's y channel (linspace -1 -> 1 from row 0) and (cos lon, sin lon) instead of x"""
    y = torch.linspace(-1.0, 1.0, grid.H)
    lam = torch.from_numpy(np.deg2rad(grid.lons())).float()
    ch = [y[:, None].expand(grid.H, grid.W), torch.cos(lam)[None].expand(grid.H, grid.W),
          torch.sin(lam)[None].expand(grid.H, grid.W)]
    return torch.stack(ch)[None].contiguous()


class PeriodicFlowerEncoder(nn.Module):
    """Flower over the whole globe, periodic in longitude, no pad/crop (H and W divide by 2**(n_levels-1))."""

    def __init__(self, in_ch, grid, feat_dim=128, lifting_dim=64, n_levels=4, num_heads=8, groups=8,
                 use_checkpoint=False):
        super().__init__()
        self.grid = grid
        self.in_ch = in_ch
        self.flower = Flower(dim_in=in_ch, dim_out=feat_dim, n_spatial_dims=2, spatial_resolution=[grid.H, grid.W],
                             lifting_dim=lifting_dim, n_levels=n_levels, num_heads=num_heads, groups=groups,
                             boundary_condition_types=["NONE", "PERIODIC"], dim_meta=0,
                             use_checkpoint=use_checkpoint)
        # dateline-continuous coordinates: (y, cos lon, sin lon) replace Flower's (x, y) linspace pair
        self.flower.coord_grid = coord_channels(grid)
        self.flower.lift = nn.Conv2d(in_ch + 3, lifting_dim, kernel_size=1)

    def forward(self, x):
        assert x.shape[1:] == (self.in_ch, self.grid.H, self.grid.W), x.shape
        return self.flower(x)


class PeriodicLIIFQuery(nn.Module):
    """CD-FFM's LIIFQuery with the cell-centre mapping and a periodic longitude (see the module docstring)."""

    def __init__(self, feat_dim, out_dim, grid, static_dim=0, hidden=128):
        super().__init__()
        self.grid = grid
        self.static_dim = static_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 9 + 2 + static_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def fractional_index(self, lat, lon):
        """fractional row/column of (lat, lon) on the grid (cell centres at integers), float64; column in [0, W)"""
        g = self.grid
        r = (g.lat0 - lat.double()) / g.dlat
        c = torch.remainder((lon.double() - g.lon0) / g.dlon, g.W)
        return r, c

    def forward(self, feat, lat, lon, static=None):
        """feat (B, C, H, W); lat, lon (B, N) degrees; static (B, N, S) or None -> (B, N, out_dim)"""
        B, C, H, W = feat.shape
        N = lat.shape[1]
        r, c = self.fractional_index(lat, lon)
        r = r.clamp(0.0, H - 1.0)                       # as CD-FFM's clamp of the corner look-ups (rows only)
        r0, c0 = torch.floor(r), torch.floor(c)
        fr, fc = (r - r0).to(feat.dtype), (c - c0).to(feat.dtype)
        r0, c0 = r0.long(), c0.long()
        off = torch.arange(-1, 3, device=feat.device)
        rows = r0[..., None] + off                      # (B, N, 4): the 4 x 4 cells around the query
        cols = torch.remainder(c0[..., None] + off, W)  # periodic longitude
        inside = ((rows >= 0) & (rows < H)).to(feat.dtype)   # rows beyond the poles: zero, as F.unfold's padding
        flat = rows.clamp(0, H - 1)[..., :, None] * W + cols[..., None, :]          # (B, N, 4, 4)
        fT = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        bidx = torch.arange(B, device=feat.device)[:, None]
        blk = fT[bidx, flat.reshape(B, N * 16)].reshape(B, N, 4, 4, C) * inside[..., :, None, None]
        if static is None and self.static_dim > 0:
            static = torch.zeros(B, N, self.static_dim, device=feat.device, dtype=feat.dtype)
        # corner (dy, dx) = cell (r0 + dy, c0 + dx); its 3 x 3 neighbourhood = blk[dy:dy+3, dx:dx+3], flattened
        # channel-major like F.unfold; rel = 2 x (query - corner) in cells (CD-FFM's rel * H scaling); the weight of
        # a corner is the area of the opposite one over the total (CD-FFM's swap), i.e. the bilinear weight
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        areas = [(2.0 * (fr - dy)).abs() * (2.0 * (fc - dx)).abs() + 1e-9 for dy, dx in corners]
        tot = areas[0] + areas[1] + areas[2] + areas[3]
        out = 0.0
        for i, (dy, dx) in enumerate(corners):
            q = blk[:, :, dy:dy + 3, dx:dx + 3, :].permute(0, 1, 4, 2, 3).reshape(B, N, C * 9)
            parts = [q, (2.0 * (fr - dy))[..., None], (2.0 * (fc - dx))[..., None]]
            if static is not None:
                parts.append(static.to(feat.dtype))
            pred = self.mlp(torch.cat(parts, dim=-1))
            out = out + pred * (areas[3 - i] / tot)[..., None]
        return out


def doy_features(doy):
    """(cos, sin) of 2 pi doy / 365 (CD-FFM fast_data.doy_features; doy 0-based)"""
    ang = 2.0 * math.pi * doy.float() / 365.0
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


class AQFlowModel(nn.Module):
    """encoder (once per day) -> LIIF condition per query point (+ doy) -> pointwise velocity net (n_out = 1)"""

    def __init__(self, in_ch, grid, static_dim, feat_dim=128, cond_mlp_dim=64, doy_cond=True, lifting_dim=64,
                 use_checkpoint=False, n_out=1, stations=False, stn_hidden=32):
        super().__init__()
        self.n_out = n_out
        self.doy_cond = doy_cond
        self.grid = grid
        self.stations = bool(stations)
        self.encoder = PeriodicFlowerEncoder(in_ch, grid, feat_dim=feat_dim, lifting_dim=lifting_dim,
                                             use_checkpoint=use_checkpoint)
        self.query = PeriodicLIIFQuery(feat_dim, cond_mlp_dim, grid, static_dim=static_dim)
        cond_dim = cond_mlp_dim + (2 if doy_cond else 0)
        if self.stations:                       # A3(c): station features 41 -> 32 -> 32, concatenated to the condition
            from .stations import STN_DIM
            self.stn_mlp = nn.Sequential(nn.Linear(STN_DIM, stn_hidden), nn.GELU(), nn.Linear(stn_hidden, stn_hidden))
            cond_dim += stn_hidden
        self.velocity_net = FlowVelocityNet(n_out, cond_dim)

    def encode(self, x):
        return self.encoder(x)

    def condition(self, feat, lat, lon, static=None, doy=None, stn=None):
        """stn (B, N, 41) station features (cdffm_aq.stations) of a stations model, None = no monitor at any point"""
        cond = self.query(feat, lat, lon, static)
        if self.doy_cond:
            cond = torch.cat([cond, doy_features(doy)[:, None, :].expand(-1, lat.shape[1], -1).to(cond.dtype)], -1)
        if self.stations:
            if stn is None:
                from .stations import none_features
                stn = none_features(lat.shape[1], cond.device, cond.dtype)[None].expand(lat.shape[0], -1, -1)
            cond = torch.cat([cond, self.stn_mlp(stn.to(cond.dtype))], -1)
        else:
            assert stn is None, "station features given to a model trained without --stations"
        return cond

    def velocity(self, x_t, cond, t, s=None):
        if s is None:
            return self.velocity_net(x_t, cond, t)
        return self.velocity_net(x_t, cond, t, s)


def build_model(args, in_ch, static_dim, grid):
    """args: namespace with lifting_dim (None = 64), use_checkpoint, doy_cond, objective, stations (A3: truthy = on)"""
    m = AQFlowModel(in_ch, grid, static_dim, doy_cond=getattr(args, "doy_cond", True),
                    lifting_dim=getattr(args, "lifting_dim", None) or 64,
                    use_checkpoint=getattr(args, "use_checkpoint", False),
                    stations=bool(getattr(args, "stations", None)))
    if getattr(args, "objective", "fm") == "flowmap":
        m.velocity_net = TwoTimeVelocityNet.from_teacher(m.velocity_net)
    return m


# ---------------------------------------------------------------------------------------------------------------
# Sphere noise prior
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseSpec:
    n_features: int = 1024
    beta: float = 2.4
    lmin_km: float = 8.0
    lmax_km: float = 500.0


def sphere_xyz_km(lat, lon):
    """(...,) degrees -> (..., 3) float64 km on the 6371 km sphere"""
    phi, lam = torch.deg2rad(lat.double()), torch.deg2rad(lon.double())
    return R_EARTH_KM * torch.stack([torch.cos(phi) * torch.cos(lam), torch.cos(phi) * torch.sin(lam),
                                     torch.sin(phi)], dim=-1)


def radial_wavenumbers(shape, spec, device, generator=None):
    """|k| in cycles/km with density ~ k^(1 - beta) on [1/lmax, 1/lmin] (inverse CDF, as fast_data._rff_frequencies)"""
    k0, k1 = 1.0 / spec.lmax_km, 1.0 / spec.lmin_km
    u = torch.rand(shape, device=device, generator=generator, dtype=torch.float64)
    e = 2.0 - spec.beta
    if abs(e) < 1e-6:
        return k0 * (k1 / k0) ** u
    return (k0 ** e + u * (k1 ** e - k0 ** e)) ** (1.0 / e)


def draw_sphere_noise(B, n_out, spec, device, generator=None):
    """parameters of B independent fields: 3-D frequencies (B, F, 3) cycles/km (float64), cos/sin amplitudes (B, F, n_out)"""
    F_ = spec.n_features
    k = radial_wavenumbers((B, F_), spec, device, generator)
    d = torch.randn(B, F_, 3, device=device, generator=generator, dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    phase = torch.rand(B, F_, n_out, device=device, generator=generator) * (2 * math.pi)
    coeff = torch.randn(B, F_, n_out, device=device, generator=generator)
    return dict(freq=k[..., None] * d, a=coeff * torch.cos(phase), b=coeff * torch.sin(phase))


def eval_sphere_noise(params, lat, lon, chunk=None):
    """field of the drawn parameters at lat, lon (B, N) degrees -> (B, N, n_out), unit marginal variance. The phase
    k.p (up to ~800 cycles for lmin = 8 km) is reduced modulo 1 in float64 before the float32 cos/sin."""
    B, N = lat.shape
    freq, a, b = params["freq"], params["a"], params["b"]
    F_, n_out = a.shape[1], a.shape[2]
    out = torch.empty(B, N, n_out, device=lat.device, dtype=torch.float32)
    step = N if chunk is None else chunk
    for s in range(0, N, step):
        e = min(N, s + step)
        cyc = torch.bmm(sphere_xyz_km(lat[:, s:e], lon[:, s:e]), freq.transpose(1, 2))   # (B, n, F) cycles
        ang = (2 * math.pi) * torch.frac(cyc).float()
        out[:, s:e] = torch.bmm(torch.cos(ang), a) - torch.bmm(torch.sin(ang), b)
    return out * math.sqrt(2.0 / F_)


def noise_chunk(n_rows, spec, budget=2 ** 27):
    """points per eval_sphere_noise chunk so that the (rows, points, F) float64 phase array stays ~1 GB"""
    return max(1024, budget // max(1, n_rows * spec.n_features))


def sphere_noise(lat, lon, n_out, spec, generator=None, chunk=None):
    """one independent field per batch row (as CD-FFM's noise_field), evaluated at that row's points"""
    return eval_sphere_noise(draw_sphere_noise(lat.shape[0], n_out, spec, lat.device, generator), lat, lon, chunk)


# ---------------------------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def integrate(model, x, cond, n_steps, flowmap=False, bound=None):
    """Euler (or flow-map jumps) from t = 0 to 1 at fixed points: x (B, N, n_out) noise, cond (B, N, c).
    bound (PROTOCOL.md A1(e)): the state is kept in [-bound, bound] before every velocity evaluation and at the end, so
    the velocity MLP is never queried outside the |z| range of the training targets (its linear extrapolation there
    ran members away: z - z0 of +4 for tail noise draws, then expm1(mu + sigma z) of 1e4..1e7 ug m-3)"""
    B, N = x.shape[:2]
    dt = 1.0 / n_steps
    if bound is not None:
        x = x.clamp(-bound, bound)
    for s in range(n_steps):
        t = torch.full((B, N), s * dt, device=x.device)
        v = model.velocity(x, cond, t, t + dt) if flowmap else model.velocity(x, cond, t)
        x = x + v * dt
        if bound is not None:
            x = x.clamp(-bound, bound)
    return x
