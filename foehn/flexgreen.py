#!/usr/bin/env python3
"""FLEXGREEN: a FLEXPART-like 3-D backward Lagrangian particle model on GPU, receptor-oriented. It computes, once and
outside training, the Green's function of the transport for every 0.5 deg cell of the model grid, every start day and
every lead: where the air reaching the cell on day d+h was on day d (particle origins, the support on which Foehn gathers
its learned latent tracers) and how much fire emission it took up on the way. The transport has no learnable parameter.

Physics after FLEXPART (Stohl et al. 2005, ACP 5:2461; Pisso et al. 2019, GMD 12:4955), re-implemented from the
equations (FLEXPART is GPL-3; no code copied):
* advection by the 3-D wind of the driving weather (the HRES run of day d, or ERA5), linear in time between 6-hourly
  fields, bilinear in the horizontal (1 deg), linear in height above ground between the 10 m wind and the pressure
  levels 1000..300 hPa; Petterssen predictor-corrector, 15-min steps (FLEXPART's default synchronisation time);
* mixing height h: critical bulk Richardson number 0.25 with b = 100, refined on 20 sub-levels between pressure levels
  (theta, u, v linear), and, in unstable conditions, one iteration with the thermal excess temperature 8.5 w'theta'/w*
  (Vogelezang & Holtslag 1996; FLEXPART's 'richardson'), bounded to 100-4500 m;
* surface-layer stability: WeatherBench2 has no surface fluxes, so u*, L and w* come from the profiles (bulk Richardson
  number of the lowest layer, Businger-Dyer), as HYSPLIT does without fluxes (Draxler & Hess 1998);
* boundary-layer turbulence: Hanna (1982) sigmas and Lagrangian time scales (FLEXPART's 'hanna'); vertically, where the
  layer mixes faster than the 15-min step (h^2 / K < dt: convective) the exact long-time limit, a uniform position in
  [0, h]; elsewhere (stable, neutral, residual layers) a random-displacement model with drift dK/dz and 18 sub-steps
  (Thomson 1987 well-mixed condition), reflection at the ground and at h; horizontally a random displacement with
  K = sigma_u^2 T_Lu; above h: diffusivities 50 m2/s (horizontal) and 0.1 m2/s (vertical) (FLEXPART's d_trop/d_strat);
* removal: dry deposition v_d = 0.05 cm/s in the lowest 2 href = 30 m (FLEXPART's probability form); wet scavenging
  Lambda = A I^B, A = 1e-4 s-1, B = 0.8 (I in mm/h) below the top of the precipitating cloud (RH >= 80 %).
Backward runs: P particles per receptor (0.5 deg cell), released at uniform random times over the target day d+h
(the target is a daily mean) and heights 0-100 m, integrated back exactly h days (to the same time of day d).
Outputs per receptor, lead and particle: origin lon, lat, z / (maximum mixing height of the start day at the origin:
the day's mixed layer including its residual layer), survival; per receptor and lead: fire
terms of the GFAS emissions of day d spread from the ground to max(injection height, h): sum over the steps spent
below that top of survival x dt / top x [particulate emission (-> kg m-3), radiative power, particulate emission taken
up above the mixed layer (smoke aloft)]. With --emis, per receptor and lead: the sensitivity to the anthropogenic
surface emissions (FLEXPART's footprint: residence time in the 0-100 m layer / 100 m) times the CAMS-GLOB-ANT v6.2
fluxes of the month of day d (BC, OC, SO2, NOx, NH3, CO), with the survival: the concentration (ug m-3) that the
emissions between day d and the target time add to the air of the receptor, as an aerosol-like tracer.
Stores: <store>/<src>/origins_<year>.npy (days, nL, 288, 720, P, 4) float16, fire_<year>.npy (days, nL, 3, 288, 720)
float32, emis_<year>.npy (days, nL, 6, 288, 720) float32 (nL = 7 daily leads, or 10 twelve-hour leads with --instant);
<store>/<src>/done_<year>/<doy> marks finished days (store: green = 32 particles, green256 = 256 particles + --emis).
    python flexgreen.py run --src hres --days 0 1 2 ... [--year 2022] [--particles 256 --store green256 --emis --chunk 40000]"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
WB = G / "wb2"
L8 = [1000, 925, 850, 700, 600, 500, 400, 300]
IU, IV, IW, IT, IQ, IZ, IU10, IV10, IT2, ISP, ITP = 0, 8, 16, 24, 32, 40, 48, 49, 50, 51, 52
G0, RD, KAPPA, VK, R_E = 9.81, 287.05, 0.286, 0.4, 6.371e6
RIC, BRI, HMIN, HMAX = 0.25, 100.0, 100.0, 4500.0
HREF, VDEP, WET_A, WET_B = 15.0, 0.0005, 1e-4, 0.8      # v_d 0.05 cm/s: accumulation mode (0.1-1 um)
BS = 8.5                                    # thermal excess temperature constant (Vogelezang & Holtslag 1996, FLEXPART)
KH_FREE, KZ_FREE = 50.0, 0.1
H_FOOT = 100.0                              # m, FLEXPART's footprint layer: sensitivity to surface emissions
SPECIES = ("BC", "OC", "SO2", "NOx", "NH3", "CO")   # CAMS-GLOB-ANT v6.2, as emis_ant_fixed.npy (build_transport.py)
DT = 900.0                                  # s, synchronisation step
NSUB = 18                                   # random-displacement sub-steps (50 s) where not well mixed: the well-mixed
                                            # criterion holds to < 5 % (tests/test_wellmixed.py; 6 sub-steps: 14 %)
ZTOP = 9500.0
# derived channels sampled at the particles
C_U, C_V, C_W, C_H = 0, 8, 16, 24           # u, v, w (m/s), level heights above ground (m), 8 each
C_U10, C_V10, C_PBL, C_UST, C_IL, C_WST, C_CTOP, C_PR, C_KBAR = 32, 33, 34, 35, 36, 37, 38, 39, 40
NCH = 41


# ------------------------------------------------------------------------------------------------ meteorology
def psim(zeta):
    """Businger-Dyer stability function for momentum"""
    x = (1 - 16 * zeta.clamp(max=0)).clamp(min=1e-6) ** 0.25
    unst = 2 * torch.log((1 + x) / 2) + torch.log((1 + x * x) / 2) - 2 * torch.atan(x) + math.pi / 2
    return torch.where(zeta < 0, unst, -5 * zeta)


def derive(Bt, zs_km, z0):
    """Bt (S, 53, 181, 360) float32 (dl_wb2 channel order) -> fields (S, NCH, 181, 360) sampled by the particles."""
    S, dev = Bt.shape[0], Bt.device
    u, v, om, tc, q, zk = (Bt[:, i:i + 8] for i in (IU, IV, IW, IT, IQ, IZ))
    hl = (zk - zs_km) * 1000.0                                            # level heights above ground (m)
    p = torch.tensor(L8, device=dev, dtype=torch.float32).view(1, 8, 1, 1) * 100.0
    qk = q.clamp(min=0) / 1000.0
    T = tc + 273.15
    Tv = T * (1 + 0.608 * qk)
    wms = -om * RD * Tv / (p * G0)                                        # omega (Pa/s) -> w (m/s)
    thv = Tv * (1e5 / p) ** KAPPA
    u10, v10 = Bt[:, IU10], Bt[:, IV10]
    t2 = Bt[:, IT2] + 273.15
    sp = Bt[:, ISP] * 100.0
    valid = hl > 10.0
    first = valid.float().argmax(1, keepdim=True)                         # lowest level above ground
    g1 = lambda a: torch.gather(a, 1, first).squeeze(1)
    thref = t2 * (1 + 0.608 * g1(qk)) * (1e5 / sp) ** KAPPA
    U10 = torch.sqrt(u10 ** 2 + v10 ** 2).clamp(min=0.1)
    # surface layer: bulk Richardson between 2/10 m and the lowest level -> zeta, L, u*, heat flux, w*
    z1 = g1(hl).clamp(min=20.0)
    rib = G0 / thref * (g1(thv) - thref) * (z1 - 2) / ((g1(u) - u10) ** 2 + (g1(v) - v10) ** 2).clamp(min=0.1)
    lnz = torch.log(z1 / z0)
    zeta1 = torch.where(rib < 0, rib * lnz, rib * lnz / (1 - 5 * rib).clamp(min=0.04)).clamp(-5, 5)
    invL = zeta1 / z1
    ust = (VK * U10 / (torch.log(10.0 / z0) - psim(10.0 * invL) + psim(z0 * invL)).clamp(min=0.5)).clamp(min=0.01)
    # mixing height: critical bulk Richardson number (FLEXPART 'richardson', reference 2 m / 10 m), with one
    # iteration adding the thermal excess temperature BS w'theta' / w* in unstable conditions (as FLEXPART)
    du2 = (u - u10[:, None]) ** 2 + (v - v10[:, None]) ** 2
    wth = -ust ** 3 * thref * invL / (VK * G0)                            # kinematic heat flux, > 0 unstable

    def mixing_height(th0):
        """first level with Ri > 0.25, then, as FLEXPART, 20 sub-levels between it and the level below (or 2 m) with
        theta, u, v interpolated linearly and Ri recomputed: h = first sub-level above the critical value."""
        ri = G0 / th0[:, None] * (thv - th0[:, None]) * (hl - 2) / (du2 + BRI * ust[:, None] ** 2).clamp(min=0.1)
        ri = torch.where(valid, ri, torch.full_like(ri, -1.0))
        cross = (ri > RIC) & valid
        k = cross.float().argmax(1, keepdim=True)
        none = ~cross.any(1)
        km1 = (k - 1).clamp(min=0)
        prev = (k > first).squeeze(1)
        gk = lambda a_: torch.gather(a_, 1, k).squeeze(1)
        gp = lambda a_, surf: torch.where(prev, torch.gather(a_, 1, km1).squeeze(1), surf)
        z_k, th_k, u_k, v_k = gk(hl), gk(thv), gk(u), gk(v)
        z_p, th_p, u_p, v_p = gp(hl, torch.full_like(z_k, 2.0)), gp(thv, th0), gp(u, u10), gp(v, v10)
        fr = torch.arange(1, 21, device=dev, dtype=torch.float32).view(1, 20, 1, 1) / 20.0
        zl = z_p[:, None] + fr * (z_k - z_p)[:, None]
        thl = th_p[:, None] + fr * (th_k - th_p)[:, None]
        ul = u_p[:, None] + fr * (u_k - u_p)[:, None]; vl = v_p[:, None] + fr * (v_k - v_p)[:, None]
        ril = G0 / th0[:, None] * (thl - th0[:, None]) * (zl - 2) / (
            (ul - u10[:, None]) ** 2 + (vl - v10[:, None]) ** 2 + BRI * ust[:, None] ** 2).clamp(min=0.1)
        c2 = ril > RIC
        j = torch.where(c2.any(1), c2.float().argmax(1), torch.full_like(z_k, 19, dtype=torch.long))
        hh = torch.gather(zl, 1, j[:, None]).squeeze(1)
        return torch.where(none, torch.full_like(hh, HMAX), hh).clamp(HMIN, HMAX)
    h = mixing_height(thref)
    wst = (G0 / thref * wth.clamp(min=0) * h) ** (1.0 / 3.0)
    excess = torch.where(wth > 0, BS * wth / wst.clamp(min=0.01), torch.zeros_like(wth)).clamp(max=3.0)
    h = mixing_height(thref + excess)
    wst = (G0 / thref * wth.clamp(min=0) * h) ** (1.0 / 3.0)
    # precipitating cloud top (RH >= 80 %), precipitation rate of the 6 h starting at the step (mm/h)
    es = 611.2 * torch.exp(17.67 * tc / (tc + 243.5))
    rh = qk / (0.622 * es / (p - 0.378 * es)).clamp(min=1e-7)
    ctop = torch.where((rh >= 0.8) & valid, hl, torch.zeros_like(hl)).amax(1)
    tp = torch.nan_to_num(Bt[:, ITP], nan=0.0)
    pr = torch.cat([tp[1:], tp[-1:]], 0) / 6.0
    ctop = torch.where((pr > 0.01) & (ctop < 500.0), torch.full_like(ctop, 3000.0), ctop)
    # levels below ground: degenerate points at 10 m with the 10 m wind (monotonic heights)
    lvl = torch.arange(8, device=dev, dtype=torch.float32).view(1, 8, 1, 1)
    hl_m = torch.where(valid, hl, 10.0 + 0.01 * (lvl + 1))
    hl_m = torch.cummax(hl_m, 1).values + 0.001 * lvl
    u_m = torch.where(valid, u, u10[:, None].expand_as(u)); v_m = torch.where(valid, v, v10[:, None].expand_as(v))
    w_m = torch.where(valid, wms, torch.zeros_like(wms))
    kbar = torch.stack([kz(h * f, h, ust, invL, wst) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]).mean(0)   # column-mean K
    out = torch.cat([u_m, v_m, w_m, hl_m, u10[:, None], v10[:, None], h[:, None], ust[:, None], invL[:, None], wst[:, None],
                     ctop[:, None], pr[:, None], kbar[:, None]], 1)
    return torch.nan_to_num(out)


def wrap(a):
    """(..., 181, 360) -> (..., 181, 361): duplicated 180 E column for periodic bilinear sampling"""
    return torch.cat([a, a[..., :1]], -1)


class Field:
    """time-space interpolation of (S, C, 181, 360) fields (lat 90..-90, lon -180..179) at particles."""

    def __init__(self, f):
        self.S = f.shape[0]
        self.f = wrap(f).permute(1, 0, 2, 3).unsqueeze(0).contiguous()     # (1, C, S, 181, 361)

    def __call__(self, lon, lat, ts, ch=None):
        gx = (lon + 180.0) / 180.0 - 1.0
        gy = (90.0 - lat) / 90.0 - 1.0
        gz = ts / (self.S - 1) * 2.0 - 1.0
        grid = torch.stack([gx, gy, gz], -1).view(1, 1, 1, -1, 3)
        src = self.f if ch is None else self.f[:, ch]
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True).view(src.shape[1], -1)


def vinterp(prof_h, prof_x, z):
    """linear interpolation in height: prof_h (9, N) increasing, prof_x (k, 9, N), z (N) -> (k, N)"""
    idx = ((prof_h < z[None]).sum(0) - 1).clamp(0, 7)
    zl, zh = prof_h.gather(0, idx[None])[0], prof_h.gather(0, (idx + 1)[None])[0]
    fr = ((z - zl) / (zh - zl).clamp(min=1e-3)).clamp(0, 1)
    xl = prof_x.gather(1, idx[None, None].expand(prof_x.shape[0], 1, -1))[:, 0]
    xh = prof_x.gather(1, (idx + 1)[None, None].expand(prof_x.shape[0], 1, -1))[:, 0]
    return xl + fr * (xh - xl)


def wind3(s, z):
    """mean (u, v, w) at height z from the sampled channels s (NCH, N)"""
    ph = torch.cat([torch.full_like(s[:1], 10.0), s[C_H:C_H + 8]], 0)
    px = torch.stack([torch.cat([s[C_U10:C_U10 + 1], s[C_U:C_U + 8]], 0), torch.cat([s[C_V10:C_V10 + 1], s[C_V:C_V + 8]], 0),
                      torch.cat([torch.zeros_like(s[:1]), s[C_W:C_W + 8]], 0)])
    return vinterp(ph, px, z)


def hanna(z, h, ust, invL, wst):
    """FLEXPART's Hanna (1982) parameterisation: sigma_u, sigma_v, sigma_w, T_Lu, T_Lw (vectorised)."""
    z = z.clamp(min=1.0)
    zeta = (z / h).clamp(1e-3, 1.0)
    ust = ust.clamp(min=1e-4)
    ol = 1.0 / torch.where(invL.abs() < 1e-6, torch.full_like(invL, 1e-6), invL)
    neutral = (h * invL.abs()) < 1.0
    unstable = (~neutral) & (invL < 0)
    corr = z / ust                                                         # neutral
    sw_n = 1.3 * ust * torch.exp(-2e-4 * corr) + 1e-2
    su_n = 1e-2 + 2.0 * ust * torch.exp(-3e-4 * corr)
    tl_n = 0.5 * z / sw_n / (1 + 1.5e-3 * corr)
    su_u = 1e-2 + ust * (12 - 0.5 * h / ol).clamp(min=0) ** (1.0 / 3.0)  # unstable
    sw_u = torch.sqrt((1.2 * wst ** 2 * (1 - 0.9 * zeta) * zeta ** (2.0 / 3.0) + (1.8 - 1.4 * zeta) * ust ** 2).clamp(min=0)) + 1e-2
    tlu_u = 0.15 * h / su_u
    azl = (z / ol).abs()
    tlw_u = torch.where(z < ol.abs(), 0.1 * z / (sw_u * (0.55 - 0.38 * azl).clamp(min=0.05)),
                        torch.where(zeta < 0.1, 0.59 * z / sw_u, 0.15 * h / sw_u * (1 - torch.exp(-5 * zeta))))
    su_s = 1e-2 + 2.0 * ust * (1 - zeta)                                   # stable
    sv_s = 1e-2 + 1.3 * ust * (1 - zeta)
    tlu_s = 0.15 * h / su_s * torch.sqrt(zeta)
    tlw_s = 0.1 * h / sv_s * zeta ** 0.8
    su = torch.where(neutral, su_n, torch.where(unstable, su_u, su_s))
    sv = torch.where(neutral, su_n, torch.where(unstable, su_u, sv_s))
    sw = torch.where(neutral, sw_n, torch.where(unstable, sw_u, sv_s))
    tlu = torch.where(neutral, tl_n, torch.where(unstable, tlu_u, tlu_s)).clamp(min=10.0)
    tlw = torch.where(neutral, tl_n, torch.where(unstable, tlw_u, tlw_s)).clamp(min=30.0)
    return su, sv, sw, tlu, tlw


def kz(z, h, ust, invL, wst):
    _, _, sw, _, tlw = hanna(z, h, ust, invL, wst)
    return sw ** 2 * tlw


# ------------------------------------------------------------------------------------------------ particles
def turbulence(z, lon, lat, h, ust, invL, wst, gen, dt=DT, nsub=None, kbar=None):
    """one step of turbulent displacement (backward and forward have the same form for the random displacement model)"""
    inpbl = z < h
    su, sv, sw, tlu, tlw = hanna(z, h, ust, invL, wst)
    # vertical, in the boundary layer (kbar: column-mean K, from the fields or computed here)
    if kbar is None:
        kbar = torch.stack([kz(h * f, h, ust, invL, wst) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]).mean(0)
    wellmixed = inpbl & (h ** 2 / kbar.clamp(min=1e-3) < dt)
    znew = z.clone()
    zu = torch.rand(z.shape, generator=gen, device=z.device) * h
    znew = torch.where(wellmixed, zu, znew)
    rdm = inpbl & ~wellmixed
    if rdm.any():
        zr, hr, ur, ir, wr = z[rdm], h[rdm], ust[rdm], invL[rdm], wst[rdm]
        ns = nsub or NSUB
        ds = dt / ns
        for _ in range(ns):
            dz_ = 2.0
            k0 = kz(zr, hr, ur, ir, wr)
            dk = (kz(zr + dz_, hr, ur, ir, wr) - kz((zr - dz_).clamp(min=0.5), hr, ur, ir, wr)) / (2 * dz_)
            step = dk * ds + torch.sqrt(2 * k0 * ds) * torch.randn(zr.shape, generator=gen, device=z.device)
            zr = zr + step.clamp(-0.5 * hr, 0.5 * hr)
            zr = torch.where(zr < 0, -zr, zr)                              # reflection at the ground
            zr = torch.where(zr > hr, 2 * hr - zr, zr).clamp(0.0, None)     # and at the mixing height
            zr = torch.minimum(zr, hr)
        znew[rdm] = zr
    # above the boundary layer: free-tropospheric diffusion
    free = ~inpbl
    znew = torch.where(free, z + math.sqrt(2 * KZ_FREE * dt) * torch.randn(z.shape, generator=gen, device=z.device), znew)
    znew = znew.abs().clamp(max=ZTOP)
    # horizontal random displacement
    kx = torch.where(inpbl, su ** 2 * tlu.clamp(max=3 * 3600.0), torch.full_like(su, KH_FREE))
    ky = torch.where(inpbl, sv ** 2 * tlu.clamp(max=3 * 3600.0), torch.full_like(sv, KH_FREE))
    dx = torch.sqrt(2 * kx * dt) * torch.randn(z.shape, generator=gen, device=z.device)
    dy = torch.sqrt(2 * ky * dt) * torch.randn(z.shape, generator=gen, device=z.device)
    lat_n = (lat + dy / R_E * 180.0 / math.pi).clamp(-89.9, 89.9)
    lon_n = lon + dx / (R_E * torch.cos(torch.deg2rad(lat)).clamp(min=0.05)) * 180.0 / math.pi
    return znew, lon_n, lat_n


def advect(fld, lon, lat, z, ts, dt):
    """Petterssen predictor-corrector for the mean 3-D wind; dt < 0 backward. Returns new lon, lat, z and the sample."""
    s0 = fld(lon, lat, ts)
    u0, v0, w0 = wind3(s0, z)
    k = 180.0 / math.pi
    cl = torch.cos(torch.deg2rad(lat)).clamp(min=0.05)
    lon1 = lon + u0 * dt / (R_E * cl) * k
    lat1 = (lat + v0 * dt / R_E * k).clamp(-89.9, 89.9)
    z1 = (z + w0 * dt).clamp(0.0, ZTOP)
    ts1 = ts + dt / (6 * 3600.0)
    s1 = fld(((lon1 + 180) % 360) - 180, lat1, ts1)
    u1, v1, w1 = wind3(s1, z1)
    lon2 = lon + 0.5 * (u0 + u1) * dt / (R_E * cl) * k
    lat2 = (lat + 0.5 * (v0 + v1) * dt / R_E * k).clamp(-89.9, 89.9)
    z2 = (z + 0.5 * (w0 + w1) * dt).clamp(0.0, ZTOP)
    return ((lon2 + 180) % 360) - 180, lat2, z2, s1


def backward(fld, fire, lon, lat, z, t_rel, hours, gen, turb=True, removal=True, emis=None):
    """integrate particles back `hours` from their release times t_rel (hours since 00 UTC of the run's day 0).
    fire: callable (lon, lat) -> (3, N) emissions of day d, or None; emis: callable (lon, lat) -> (S, N) anthropogenic
    fluxes, or None. Returns lon, lat, z/h, survival at the origin, the fire sensitivity (3, N) and the emission term
    (S, N, kg m-3) or None."""
    n = int(round(hours * 3600 / DT))
    ts = t_rel / 6.0                                                      # in 6-h field steps
    surv = torch.ones_like(lon)
    fsens = torch.zeros((3,) + lon.shape, device=lon.device) if fire is not None else None
    esens = None
    for _ in range(n):
        lon, lat, z, s = advect(fld, lon, lat, z, ts, -DT)
        ts = ts - DT / (6 * 3600.0)
        h, ust, invL, wst = s[C_PBL], s[C_UST], s[C_IL], s[C_WST]
        if turb:
            z, lon, lat = turbulence(z, lon, lat, h, ust, invL, wst, gen, kbar=s[C_KBAR])
            lon = ((lon + 180) % 360) - 180
        if removal:
            surv = surv * torch.where(z < 2 * HREF, math.exp(-VDEP * DT / (2 * HREF)), 1.0)
            pr = s[C_PR].clamp(min=0)
            surv = surv * torch.where((pr > 0.01) & (z < s[C_CTOP]), torch.exp(-WET_A * pr ** WET_B * DT), torch.ones_like(pr))
        if fsens is not None:                       # emissions spread from the ground to max(injection height, h)
            fz = fire(lon, lat)
            top = torch.maximum(fz[2], h)
            w = surv * (z < top).float() * DT / top
            fsens[0] += fz[0] * w; fsens[1] += fz[1] * w; fsens[2] += fz[0] * w * (z > h).float()
        if emis is not None:                        # surface emissions: residence in FLEXPART's 0-100 m footprint layer
            ez = emis(lon, lat) * (surv * (z < H_FOOT).float() * (DT / H_FOOT))[None]
            esens = ez if esens is None else esens + ez
    # air of the start day's mixed layer: below the day's maximum mixing height at the origin (the residual layer of
    # a collapsed nocturnal boundary layer still holds the day's mixed air; the target and the input are daily means)
    d0 = torch.floor(ts / 4.0) * 4.0
    hmax = torch.stack([fld(lon, lat, (d0 + k).clamp(0, fld.S - 1), ch=[C_PBL])[0] for k in range(5)]).amax(0)
    return lon, lat, z / hmax, surv, fsens, esens


# ------------------------------------------------------------------------------------------------ driver
NR, NC = 288, 720                                                         # 0.5 deg model grid (84 N .. 60 S)
REC_LAT = 84.0 - 0.5 * np.arange(NR) - 0.125
REC_LON = -180.0 + 0.5 * np.arange(NC) + 0.125


def static(dev):
    st = np.load(WB / "static_1deg.npz")
    zs = torch.tensor(st["zs_km"], device=dev, dtype=torch.float32)
    lsm = torch.tensor(st["lsm"], device=dev, dtype=torch.float32).clamp(0, 1)
    z0 = torch.exp(lsm * math.log(0.1) + (1 - lsm) * math.log(2e-4))
    return zs, z0


def npy_block(path, i):
    """array[i] of a .npy file, read with pread (memmap page faults are slow on Lustre under many readers)"""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = int(np.prod(sh[1:])) * dt.itemsize
    fd = os.open(path, os.O_RDONLY)
    try:
        parts, got = [], 0
        while got < n:
            b = os.pread(fd, min(n - got, 1 << 30), hdr + i * n + got)
            if not b:
                raise IOError(f"short read of {path} block {i}")
            parts.append(b); got += len(b)
    finally:
        os.close(fd)
    return np.frombuffer(b"".join(parts), dt).reshape(sh[1:])


def npy_write_block(path, i, block):
    """array[i] = block in a .npy file with sequential pwrite (memmap writes fault every page in: slow on Lustre)"""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    block = np.ascontiguousarray(block, dtype=dt)
    assert block.shape == tuple(sh[1:]), (path, block.shape, sh)
    mv = memoryview(block.reshape(-1).view(np.uint8)); n = len(mv); done = 0
    fd = os.open(path, os.O_WRONLY)
    try:
        while done < n:
            done += os.pwrite(fd, mv[done:done + (1 << 30)], hdr + i * n + done)
        os.fsync(fd)
    finally:
        os.close(fd)


def load_run(src, year, doy):
    """(33, 53, 181, 360) float32 fields of the run of day doy: HRES steps 0..192 h, or ERA5 at the same times."""
    d0 = np.datetime64(f"{year}-01-01") + np.timedelta64(int(doy), "D")
    if src == "hres":
        m = int(str(d0)[5:7])
        doys = np.load(WB / f"hres_B_{year}_{m:02d}_doy.npy")
        return npy_block(WB / f"hres_B_{year}_{m:02d}.npy", int(np.where(doys == doy)[0][0])).astype(np.float32)
    out = []
    for k in range(9):                                                    # ERA5: days d .. d+8, 00/06/12/18 UTC
        dk = d0 + np.timedelta64(k, "D"); y, m, dd = int(str(dk)[:4]), int(str(dk)[5:7]), int(str(dk)[8:10])
        out.append(npy_block(WB / f"era5_B_{y}_{m:02d}.npy", dd - 1).astype(np.float32))
    return np.concatenate(out)[:33]


def fire_sampler(day_index, dev):
    """GFAS fires of the start day on the model's 0.25 deg grid (84 N .. 59.75 S), sampled bilinearly (0 outside)."""
    with open(G / "fires.npy", "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
    n = int(np.prod(sh[1:]))
    a = np.fromfile(G / "fires.npy", dtype=dt, count=n, offset=hdr + day_index * n * np.dtype(dt).itemsize).reshape(sh[1:])
    x = torch.tensor(np.nan_to_num(a.astype(np.float32), nan=-30.0), device=dev)        # (3, 576, 1440), build_transport.py
    f = torch.stack([(10 ** (x[0] - 13.5) - 1e-14).clamp(min=0),                          # particulate matter, kg m-2 s-1
                     (10 ** (x[1] - 5.5) - 1e-6).clamp(min=0),                            # radiative power, W m-2
                     (10 ** (0.6 * x[2] + 0.1) - 1.0).clamp(min=0)])                      # injection height, m
    f = torch.cat([f, f[..., :1]], -1)[None]

    def sample(lon, lat):
        gx = (lon + 180.0) / 180.0 - 1.0
        gy = (84.0 - lat) / (0.25 * 575) * 2.0 - 1.0
        out = F.grid_sample(f, torch.stack([gx, gy], -1).view(1, 1, -1, 2), mode="bilinear", padding_mode="zeros",
                            align_corners=True).view(3, -1)
        return torch.where(((lat > 84.0) | (lat < -59.75))[None], torch.zeros_like(out), out)
    return sample


def emis_sampler(year, doy, dev):
    """CAMS-GLOB-ANT v6.2 total anthropogenic fluxes (kg m-2 s-1) of the month of the start day on the model's 0.25 deg
    grid (84 N .. 59.75 S), sampled bilinearly (0 outside); emis_ant_fixed.npy holds log10(flux + 1e-14), standardised."""
    k = 12 * (int(year) - 2017) + int((np.datetime64(f"{year}-01-01") + np.timedelta64(int(doy), "D")).astype("datetime64[M]").astype(int) % 12)
    st = np.load(G / "emis_stats.npz")
    with open(G / "emis_ant_fixed.npy", "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
    n = int(np.prod(sh[1:]))
    a = np.fromfile(G / "emis_ant_fixed.npy", dtype=dt, count=n, offset=hdr + k * n * np.dtype(dt).itemsize).reshape(sh[1:])
    x = torch.tensor(a.astype(np.float32), device=dev) * torch.tensor(st["std"], device=dev)[:, None, None] \
        + torch.tensor(st["mean"], device=dev)[:, None, None]
    f = (10.0 ** x - 1e-14).clamp(min=0)
    f = torch.cat([f, f[..., :1]], -1)[None]
    nsp = f.shape[1]

    def sample(lon, lat):
        gx = (lon + 180.0) / 180.0 - 1.0
        gy = (84.0 - lat) / (0.25 * 575) * 2.0 - 1.0
        out = F.grid_sample(f, torch.stack([gx, gy], -1).view(1, 1, -1, 2), mode="bilinear", padding_mode="zeros",
                            align_corners=True).view(nsp, -1)
        return torch.where(((lat > 84.0) | (lat < -59.75))[None], torch.zeros_like(out), out)
    return sample


def run(a):
    dev = torch.device("cuda", int(os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(dev)
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    zs, z0 = static(dev)
    out = G / a.store / a.src
    out.mkdir(parents=True, exist_ok=True)
    meta = np.load(G / "meta.npz"); years, doys_all = meta["year"], meta["doy"]
    days = (list(a.days)[::-1] if a.reverse else list(a.days))[rank::world]   # --reverse: a second job from the end
    gen = torch.Generator(device=dev)
    lat_r = torch.tensor(REC_LAT, device=dev, dtype=torch.float32); lon_r = torch.tensor(REC_LON, device=dev, dtype=torch.float32)
    LA, LO = torch.meshgrid(lat_r, lon_r, indexing="ij")
    LA, LO = LA.reshape(-1), LO.reshape(-1)
    # receptors: 0.5 deg cells with land, plus a one-cell coastal margin (the loss is on land pixels only)
    land = np.load(G / "meta.npz")["land"].astype(bool)                   # (576, 1440) at 0.25 deg
    l5 = land.reshape(NR, 2, NC, 2).any((1, 3))
    l5 = l5 | np.roll(l5, 1, 0) | np.roll(l5, -1, 0) | np.roll(l5, 1, 1) | np.roll(l5, -1, 1)
    REC = torch.tensor(np.where(l5.reshape(-1))[0], device=dev)
    if a.all_cells:                                 # every 0.5 deg cell, sea included (targets defined everywhere, e.g. CAMS)
        REC = torch.arange(NR * NC, device=dev)
    # leads (hours): daily mode, releases spread over the target day d+h (daily-mean target), h = 1..7 days; instantaneous
    # mode (--instant, Aurora's 12-hour steps), releases at the valid time t0 + L, L = 12, 24, ..., 120 h, t0 = 00 UTC of d
    LEADS_H = [12.0 * k for k in range(1, 11)] if a.instant else [24.0 * h for h in range(1, 8)]
    nL = len(LEADS_H)
    if rank == 0:
        print(f"{len(REC)} receptors of {NR * NC} ({len(REC) / (NR * NC):.0%})", flush=True)
    nd = 366 if a.year % 4 == 0 else 365
    fo, ff, fe = out / f"origins_{a.year}.npy", out / f"fire_{a.year}.npy", out / f"emis_{a.year}.npy"  # raw stores: partial reads
    stores = [(fo, (nd, nL, NR, NC, a.particles, 4), np.float16), (ff, (nd, nL, 3, NR, NC), np.float32)]
    if a.emis:
        stores.append((fe, (nd, nL, len(SPECIES), NR, NC), np.float32))
    if rank == 0:
        for f_, sh, dt_ in stores:
            if not f_.exists():
                np.lib.format.open_memmap(f_, mode="w+", dtype=dt_, shape=sh).flush()
    while not all(f_.exists() for f_, _, _ in stores):
        time.sleep(5)
    time.sleep(5)
    with open(fo, "rb") as f_:
        np.lib.format.read_magic(f_); sh_o = np.lib.format.read_array_header_1_0(f_)[0]
    assert sh_o[4] == a.particles, f"{fo} holds {sh_o[4]} particles, not {a.particles}: use another --store"
    done = out / f"done_{a.year}"
    done.mkdir(exist_ok=True)
    for doy in days:
        if (done / f"{doy:03d}").exists() and not a.overwrite:
            continue
        run_day = np.datetime64(f"{a.year}-01-01") + np.timedelta64(int(doy) - a.lag, "D")    # --lag j: the run of day d - j
        ry, rdoy = int(str(run_day)[:4]), int((run_day - np.datetime64(str(run_day)[:4] + "-01-01")).astype(int))
        if a.src == "hres":                         # only complete months (the doy list is written at the end of a month)
            m_ = int(str(run_day)[5:7])
            if not (WB / f"hres_B_{ry}_{m_:02d}_doy.npy").exists():
                print(f"rank {rank}: {a.year} day {doy}: HRES month {m_} not downloaded yet, skipped", flush=True)
                continue
        t0 = time.time()
        gen.manual_seed(int(a.year) * 100000 + int(doy) * 100 + int(a.seed) * 7 + (0 if a.src == "hres" else 50) + 13 * a.lag)
        Bt = torch.tensor(load_run(a.src, ry, rdoy), device=dev)
        fld = Field(derive(Bt, zs, z0)); del Bt
        di = int(np.where((years == a.year) & (doys_all == doy))[0][0])
        fire = fire_sampler(di - 1 if a.instant else di, dev)   # --instant (t0 = 00 UTC of d): GFAS of day d-1, the last
                                                               # complete day at t0, persisted (as the CAMS forecasts do)
        emis = emis_sampler(a.year, doy, dev) if a.emis else None
        org = np.zeros((nL, NR * NC, a.particles, 4), np.float16)
        fs = np.zeros((nL, 3, NR * NC), np.float32)
        es = np.zeros((nL, len(SPECIES), NR * NC), np.float32)
        for li, L in enumerate(LEADS_H):
            if L + (0 if a.instant else 24.0) + 24.0 * a.lag > 192.0:   # beyond the 192 h of the stored runs: left empty
                continue
            th = time.time()
            for q0 in range(0, len(REC), a.chunk):
                rc = REC[q0:q0 + a.chunk]; n = len(rc) * a.particles
                la = LA[rc].repeat_interleave(a.particles) + (torch.rand(n, generator=gen, device=dev) - 0.5) * 0.5
                lo = LO[rc].repeat_interleave(a.particles) + (torch.rand(n, generator=gen, device=dev) - 0.5) * 0.5
                z = torch.rand(n, generator=gen, device=dev) * 100.0
                if a.instant:
                    t_rel = torch.full((n,), L, device=dev)
                else:
                    t_rel = L + 24.0 * torch.rand(n, generator=gen, device=dev)
                t_rel = t_rel + 24.0 * a.lag                                  # hours since 00 UTC of the run's day
                lo2, la2, zr, sv, fsn, esn = backward(fld, fire, lo, la, z, t_rel, L, gen, emis=emis)
                o = torch.stack([lo2, la2, zr, sv], -1).view(len(rc), a.particles, 4)
                ri = rc.cpu().numpy()
                org[li, ri] = o.half().cpu().numpy()
                fs[li][:, ri] = fsn.view(3, len(rc), a.particles).mean(-1).cpu().numpy()
                if esn is not None:                                                       # kg m-3 -> ug m-3
                    es[li][:, ri] = (esn.view(len(SPECIES), len(rc), a.particles).mean(-1) * 1e9).cpu().numpy()
            if a.verbose:
                print(f"   lead {L:.0f} h: {time.time() - th:.0f}s", flush=True)
        npy_write_block(fo, doy, org.reshape(nL, NR, NC, a.particles, 4)); npy_write_block(ff, doy, fs.reshape(nL, 3, NR, NC))
        if a.emis:
            npy_write_block(fe, doy, es.reshape(nL, len(SPECIES), NR, NC))
        (done / f"{doy:03d}").touch()
        print(f"rank {rank}: {a.src} {a.year} day {doy} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--src", choices=["hres", "era5"], default="hres")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--days", type=int, nargs="+", default=[0])
    ap.add_argument("--particles", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=100000, help="receptors per batch (all land receptors: one batch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="green", help="output directory under global1km (green: the 32-particle store)")
    ap.add_argument("--emis", action="store_true", help="also the anthropogenic emission term (emis_<year>.npy)")
    ap.add_argument("--reverse", action="store_true", help="days from the last (a second job meets the first one)")
    ap.add_argument("--instant", action="store_true", help="Aurora protocol: releases at the valid time 00 UTC + 12..120 h")
    ap.add_argument("--all-cells", action="store_true", help="receptors in every 0.5 deg cell, sea included")
    ap.add_argument("--lag", type=int, default=0, help="drive the transport of start day d with the HRES run of day d - lag "
                    "(lagged weather ensemble); leads beyond the run's 192 h stay empty")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())
