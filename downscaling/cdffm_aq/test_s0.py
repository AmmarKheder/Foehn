"""cdffm_aq.test_s0 -- S0 CPU checks of PLAN.md section 7 for the model code. (S0.1, orientation of LGHAP, statics,
CAMS and the x8/7 step, was run by the smoke-data build: data/processed/smoke/manifest.json.)

  shapes    model sizes (0.25 deg: 3.30 M), every parameter gets a finite gradient (fm, regression, flowmap), input
            assembly (47 channels; periodic x 2 upsampling of T6 against an explicit formula), fair CRPS formula
  liif      PeriodicLIIFQuery == CD-FFM's LIIFQuery (F.unfold + nearest look-ups) in the interior (copied weights);
            exact roll-equivariance across the dateline (no absolute position in the query)
  dateline  S0.2: with random weights, the jump across lambda = +-179.995 deg is no larger than the largest jump between
            any two neighbouring 0.01 deg pixels: encoder columns (full 0.25 deg globe), condition along latitude
            circles, sphere noise; negative controls with Flower's linspace x channel, CD-FFM's clamped LIIF and
            CD-FFM's planar noise must fail
  noise     S0.3: sphere-noise PSD slope = -beta +- 0.2 at 0, 30 and 55 deg (fitted power law of
            noise_fit_smoke.json); one 4,000 x 4,000 km member at 2 km shows no stripes at F = 1024 (else 4096):
            stripe = in some octave band a single plane wave carries > 0.20 more of the band's power than the
            brightest spot of an isotropic Gaussian field with the same spectrum; F = 64 must show stripes (control);
            the verdict (n_features) is written to noise_fit_smoke.json, the maps to logs/s0_noise_maps.png
  train     S0.4: 2 training steps of compute_losses on the 72 x 144 toy grid; then the train_aq CLI on the toy
            store (validation, checkpoints, --resume), a regression run, a flowmap run distilled from it, and
            eval_aq on the toy run (metrics.json, gates, maps)
  ddp       2 CPU processes (torchrun, gloo) of train_aq: ranks draw different days, weights identical
  store     the real smoke store: one real training batch (47 x 576 x 1440 inputs, targets, z round trip), the
            hold-out truth against CAMS bilinear, and one forward + backward of the full-size model on CPU

  python -m cdffm_aq.test_s0 [all | shapes liif dateline noise train ddp store ...] [--quick] [--tmp DIR]
Exit code 1 if any check fails.
"""
import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import NOISE_FIT_SMOKE, PROJECT
from .data_aq import (BatchStream, PixelArrays, Scores, SmokeStore, ToyStore, assemble_inputs, bilinear_log1p_at,
                      fair_crps, raw_to_device, target_z, upsample2_periodic, z_to_ug)
from .model_aq import (GRIDS, R_EARTH_KM, AQFlowModel, NoiseSpec, PeriodicFlowerEncoder, PeriodicLIIFQuery,
                       build_model, draw_sphere_noise, eval_sphere_noise)
from .spectra import fit_slope, radial_psd

RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(check=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: " + json.dumps(info, default=lambda o: float(o)), flush=True)


def noise_fit():
    if NOISE_FIT_SMOKE.exists():
        return json.loads(NOISE_FIT_SMOKE.read_text())
    print(f"WARNING: {NOISE_FIT_SMOKE} missing: CD-FFM defaults", flush=True)
    return {"chosen": {"beta": 2.4, "lmin_km": 8.0, "lmax_km": 500.0, "n_features": 1024}}


def chosen_spec(n_features=None):
    c = noise_fit()["chosen"]
    return NoiseSpec(int(n_features or c.get("n_features", 1024)), float(c["beta"]), float(c["lmin_km"]),
                     float(c["lmax_km"]))


# ---------------------------------------------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------------------------------------------

def check_shapes():
    for gname, expect in (("025", 3.30e6), ("05", None), ("toy", None)):
        m = AQFlowModel(47, GRIDS[gname], 6)
        n = sum(p.numel() for p in m.parameters())
        info = dict(grid=gname, params=n, encoder=sum(p.numel() for p in m.encoder.parameters()),
                    liif=sum(p.numel() for p in m.query.parameters()),
                    velocity=sum(p.numel() for p in m.velocity_net.parameters()))
        record(f"shapes/params_{gname}", expect is None or abs(n - expect) / expect < 0.01, **info)
    # every parameter receives a finite gradient, for each objective (toy grid, random batch)
    from .train_aq import compute_losses
    st = ToyStore()
    pix = PixelArrays(st, torch.device("cpu"))
    bs = BatchStream(st, 2, 512, 0, 0, torch.device("cpu"))
    b = bs.next()
    bs.close()
    spec = chosen_spec()
    for obj in ("fm", "regression", "flowmap"):
        args = argparse.Namespace(objective=obj, grid="025", transport="t6", fm_diag_prob=0.25, fm_substeps=2,
                                  doy_cond=True, lifting_dim=None, use_checkpoint=False)
        torch.manual_seed(0)
        m = AQFlowModel(47, GRIDS["toy"], 6)
        teacher = None
        if obj == "flowmap":
            from model import TwoTimeVelocityNet
            teacher = AQFlowModel(47, GRIDS["toy"], 6)
            m.load_state_dict(teacher.state_dict())
            m.velocity_net = TwoTimeVelocityNet.from_teacher(teacher.velocity_net)
            for n_, p in m.named_parameters():
                p.requires_grad_(n_.startswith("velocity_net."))
        gen = torch.Generator().manual_seed(1)
        loss = compute_losses(m, pix, b, args, spec, gen, teacher)
        loss.backward()
        params = [(n_, p) for n_, p in m.named_parameters() if p.requires_grad]
        no_grad = [n_ for n_, p in params if p.grad is None]
        bad = [n_ for n_, p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
        zero = [n_ for n_, p in params if p.grad is not None and float(p.grad.abs().max()) == 0.0]
        record(f"shapes/grads_{obj}", math.isfinite(float(loss)) and not no_grad and not bad and not zero,
               loss=float(loss), trainable=len(params), unused=no_grad[:5], nonfinite=bad[:5], all_zero=zero[:5])
    # encoder input: 47 channels; T6 periodic x 2 upsampling == explicit bilinear formula (incl. the dateline)
    x = assemble_inputs(b["raw"], pix)
    t6 = torch.randn(1, 2, 6, 10, dtype=torch.float64)
    up = upsample2_periodic(t6)
    H5, W5 = t6.shape[-2:]
    ref = torch.empty_like(up)
    for r in range(2 * H5):
        for c in range(2 * W5):
            fr, fc = min(max(0.5 * r - 0.25, 0.0), H5 - 1.0), 0.5 * c - 0.25          # input coordinates
            r0, c0 = min(int(math.floor(fr)), H5 - 2), int(math.floor(fc))
            wr, wc = fr - r0, fc - c0
            v = lambda rr, cc: t6[0, :, rr, cc % W5]
            ref[0, :, r, c] = ((1 - wr) * (1 - wc) * v(r0, c0) + (1 - wr) * wc * v(r0, c0 + 1)
                               + wr * (1 - wc) * v(r0 + 1, c0) + wr * wc * v(r0 + 1, c0 + 1))
    record("shapes/inputs", x.shape[1] == 47 and bool(torch.isfinite(x).all())
           and float((up - ref).abs().max()) < 1e-12, in_shape=list(x.shape),
           upsample_max_err=float((up - ref).abs().max()))
    # fair CRPS: sorted-member formula == brute force over pairs; Scores of a perfect point forecast
    g = torch.Generator().manual_seed(3)
    ens, y = torch.randn(8, 1000, generator=g, dtype=torch.float64), torch.randn(1000, generator=g, dtype=torch.float64)
    brute = (ens - y).abs().mean(0) - (ens[:, None] - ens[None]).abs().sum((0, 1)) / (2 * 8 * 7)
    sc = Scores(1, torch.device("cpu"))
    yy = torch.rand(500, dtype=torch.float32) * 50 + 1
    sc.add(yy[None], yy, torch.ones(500))
    s = sc.summary()
    err = float((fair_crps(ens, y) - brute).abs().max())
    record("shapes/fair_crps_and_scores", err < 1e-10 and s["ug"]["crps_fair"] == 0.0 and s["ug"]["rmse_ensmean"] == 0.0
           and s["thresholds_ug"]["25"]["csi_1"] == 1.0, max_err=err, perfect_crps=s["ug"]["crps_fair"],
           perfect_csi25=s["thresholds_ug"]["25"]["csi_1"])


# ---------------------------------------------------------------------------------------------------------------
# liif
# ---------------------------------------------------------------------------------------------------------------

def check_liif():
    """float32 like CD-FFM (its feat_coord grid is float32); tolerance 1e-4 covers its float32 coordinate rounding"""
    from model import LIIFQuery
    torch.manual_seed(0)
    g = GRIDS["toy"]
    C, S = 16, 6
    q = PeriodicLIIFQuery(C, 64, g, static_dim=S)
    lat_edges = np.array([g.lat0 + g.dlat / 2, g.lat0 - (g.H - 0.5) * g.dlat])     # CD-FFM maps [0] -> -1, [-1] -> +1
    lon_edges = np.array([g.lon0 - g.dlon / 2, g.lon0 + (g.W - 0.5) * g.dlon])
    ref = LIIFQuery(C, 64, lat_edges, lon_edges, static_dim=S)
    ref.mlp.load_state_dict(q.mlp.state_dict())
    feat = torch.randn(2, C, g.H, g.W)
    gen = torch.Generator().manual_seed(1)
    N = 4000
    r = 2 + torch.rand(2, N, generator=gen, dtype=torch.float64) * (g.H - 5)
    c = 2 + torch.rand(2, N, generator=gen, dtype=torch.float64) * (g.W - 5)
    keep = ((r - r.floor() - 0.5).abs() < 0.499) & ((c - c.floor() - 0.5).abs() < 0.499)   # away from cell edges
    lat, lon = (g.lat0 - g.dlat * r).float(), (g.lon0 + g.dlon * c).float()
    st = torch.randn(2, N, S, generator=gen)
    with torch.no_grad():
        a, b = q(feat, lat, lon, st), ref(feat, lat, lon, st)
    err = float((a - b).abs()[keep].max())
    record("liif/equals_cdffm_interior", err < 1e-4, max_abs_diff=err, out_std=float(a.std()), points=int(keep.sum()))
    # roll equivariance: rolling the map by s columns = shifting the query by s cells, also across the dateline
    s_ = g.W // 2
    lon_d = torch.cat([179.0 + torch.rand(1, N // 2, generator=gen, dtype=torch.float64),
                       -180.0 + torch.rand(1, N // 2, generator=gen, dtype=torch.float64)], 1).expand(2, -1)
    lat_d = lat[:, :N].double()
    with torch.no_grad():
        a = q(feat, lat_d, lon_d, st)
        b = q(torch.roll(feat, shifts=s_, dims=-1), lat_d, lon_d + s_ * g.dlon, st)
    err2 = float((a - b).abs().max())
    record("liif/roll_equivariant_dateline", err2 < 1e-5, max_abs_diff=err2)


# ---------------------------------------------------------------------------------------------------------------
# dateline (S0.2)
# ---------------------------------------------------------------------------------------------------------------

def smooth_periodic_inputs(in_ch, grid, seed=0):
    """smooth fields that are continuous across the dateline (low-order Fourier modes in longitude)"""
    g = torch.Generator().manual_seed(seed)
    lam = torch.from_numpy(np.deg2rad(grid.lons()))
    phi = torch.from_numpy(np.deg2rad(grid.lats()))
    x = torch.zeros(in_ch, grid.H, grid.W, dtype=torch.float64)
    for m in range(5):
        A = torch.randn(in_ch, 1, 1, generator=g, dtype=torch.float64)
        B = torch.randn(in_ch, 1, 1, generator=g, dtype=torch.float64)
        ph = float(torch.rand(1, generator=g)) * 2 * math.pi
        x += (A * torch.cos(m * lam) + B * torch.sin(m * lam)) * torch.cos((m + 1) * phi + ph)[:, None]
    return x.float()[None]


def column_jumps(feat):
    d = (feat[..., 1:] - feat[..., :-1]).pow(2).mean(dim=(0, 1, 2)).sqrt()
    wrap = float((feat[..., 0] - feat[..., -1]).pow(2).mean().sqrt())
    return d, wrap


def circle_jumps(values):
    """values (36000, C) along a latitude circle at 0.01 deg: jumps between neighbours, and across the dateline"""
    d = (values[1:] - values[:-1]).pow(2).mean(-1).sqrt()
    return d, float((values[0] - values[-1]).pow(2).mean().sqrt())


def ens_circle_jumps(values):
    """values (R, 36000) realisations along a latitude circle: RMS over realisations of each neighbour difference"""
    d = (values[:, 1:] - values[:, :-1]).pow(2).mean(0).sqrt()
    return d, float((values[:, 0] - values[:, -1]).pow(2).mean().sqrt())


def check_dateline(quick=False):
    from flower_layers import Flower
    from model import LIIFQuery
    from fast_data import noise_field
    torch.manual_seed(0)
    grid = GRIDS["toy"] if quick else GRIDS["025"]
    enc = PeriodicFlowerEncoder(47, grid).eval()
    x = smooth_periodic_inputs(47, grid)
    t0 = time.time()
    with torch.no_grad():
        feat = enc(x)
    d, wrap = column_jumps(feat)
    record(f"dateline/encoder_{grid.H}x{grid.W}", wrap <= float(d.max()), wrap=wrap, max_inner=float(d.max()),
           median_inner=float(d.median()), ratio_wrap_to_median=wrap / float(d.median()), seconds=time.time() - t0)
    torch.manual_seed(0)
    ctl = Flower(dim_in=47, dim_out=128, n_spatial_dims=2, spatial_resolution=[grid.H, grid.W], lifting_dim=64,
                 n_levels=4, num_heads=8, groups=8, boundary_condition_types=["NONE", "PERIODIC"], dim_meta=0).eval()
    with torch.no_grad():
        fc = ctl(x)
    dc, wc = column_jumps(fc)
    record("dateline/control_flower_linspace_x_fails", wc > float(dc.max()), wrap=wc, max_inner=float(dc.max()),
           ratio_wrap_to_median=wc / float(dc.median()))
    # condition (encoder + LIIF) along latitude circles at 0.01 deg
    q = PeriodicLIIFQuery(128, 64, grid, static_dim=6).eval()
    lat_e = np.array([grid.lat0 + grid.dlat / 2, grid.lat0 - (grid.H - 0.5) * grid.dlat])
    lon_e = np.array([grid.lon0 - grid.dlon / 2, grid.lon0 + (grid.W - 0.5) * grid.dlon])
    qref = LIIFQuery(128, 64, lat_e, lon_e, static_dim=6).eval()
    qref.mlp.load_state_dict(q.mlp.state_dict())
    lon = (-179.995 + 0.01 * torch.arange(36000, dtype=torch.float64))[None]
    spec = chosen_spec()
    for la in (-40.005, 0.005, 45.005):
        lat = torch.full_like(lon, la)
        st = torch.zeros(1, 36000, 6)
        with torch.no_grad():
            cnd = q(feat, lat.float(), lon.float(), st)[0]
            cref = qref(feat, lat.float(), lon.float(), st)[0]
        d, wrap = circle_jumps(cnd)
        record(f"dateline/condition_lat{la:+.0f}", wrap <= float(d.max()), wrap=wrap, max_inner=float(d.max()),
               median_inner=float(d.median()))
        d, wrap = circle_jumps(cref)
        record(f"dateline/control_cdffm_liif_fails_lat{la:+.0f}", wrap > float(d.max()), wrap=wrap,
               max_inner=float(d.max()))
        # random fields: RMS over R realisations of each neighbour difference (one draw is too noisy a statistic)
        R = 4 if quick else 16
        gen = torch.Generator().manual_seed(5)
        nz = eval_sphere_noise(draw_sphere_noise(R, 1, spec, torch.device("cpu"), gen), lat.expand(R, -1),
                               lon.expand(R, -1), chunk=4096)[..., 0]                        # (R, 36000)
        d, wrap = ens_circle_jumps(nz)
        record(f"dateline/sphere_noise_lat{la:+.0f}", wrap <= float(d.max()), rms_wrap=wrap, max_rms_inner=float(d.max()),
               median_rms_inner=float(d.median()), realisations=R)
        gen = torch.Generator().manual_seed(5)
        nz = noise_field(lat.float().expand(R, -1), lon.float().expand(R, -1), 1, n_features=spec.n_features,
                         kernel="powerlaw", beta=spec.beta, lmax_km=spec.lmax_km, lmin_km=spec.lmin_km, generator=gen,
                         chunk=4096)[..., 0]
        d, wrap = ens_circle_jumps(nz)
        record(f"dateline/control_cdffm_planar_noise_fails_lat{la:+.0f}", wrap > float(d.max()), rms_wrap=wrap,
               max_rms_inner=float(d.max()), realisations=R)


# ---------------------------------------------------------------------------------------------------------------
# noise (S0.3)
# ---------------------------------------------------------------------------------------------------------------

def tangent_box(lat_c, lon_c, size_km, n):
    """lat, lon (n, n) of an n x n gnomonic grid of side size_km centred at (lat_c, lon_c): rows north, cols east"""
    phi, lam = math.radians(lat_c), math.radians(lon_c)
    cvec = np.array([math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)])
    e = np.array([-math.sin(lam), math.cos(lam), 0.0])
    nn_ = np.array([-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)])
    s = (np.arange(n) - n / 2 + 0.5) * (size_km / n)
    X, Y = np.meshgrid(s, s)
    P = R_EARTH_KM * cvec[None, None] + X[..., None] * e + Y[..., None] * nn_
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(P[..., 2])), np.degrees(np.arctan2(P[..., 1], P[..., 0]))


def noise_on_box(lat, lon, spec, seed, chunk=32768):
    gen = torch.Generator().manual_seed(seed)
    la = torch.from_numpy(lat.reshape(1, -1))
    lo = torch.from_numpy(lon.reshape(1, -1))
    p = draw_sphere_noise(1, 1, spec, torch.device("cpu"), gen)
    return eval_sphere_noise(p, la, lo, chunk=chunk)[0, :, 0].numpy().reshape(lat.shape)


def wave_shares(field, dx_km, bands_km):
    """per wavelength band: fraction of the band's power in its brightest plane wave (5 x 5 Fourier pixels around the
    periodogram maximum, x 2 for the conjugate peak of a real field)"""
    f = field - field.mean()
    n = f.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fft2(f * w)) ** 2
    k = np.fft.fftfreq(n, d=dx_km)
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    out = {}
    for lo, hi in bands_km:
        m = (kr >= 1.0 / hi) & (kr < 1.0 / lo)
        S = P[m].sum()
        iy, ix = np.unravel_index(np.argmax(np.where(m, P, 0.0)), P.shape)
        ys, xs = (np.arange(iy - 2, iy + 3) % n), (np.arange(ix - 2, ix + 3) % n)
        peak = P[np.ix_(ys, xs)].sum()
        out[f"{lo:.0f}-{hi:.0f}km"] = float(min(1.0, 2.0 * peak / S))
    return out


def angular_index(field, dx_km, lam_lo, lam_hi, nsec=18):
    """diagnostic of orientation: power of wavelengths in [lam_lo, lam_hi] (periodogram whitened by its radial mean in
    log-k rings) in 180/nsec deg sectors; nsec x the largest sector share (1 = isotropic)"""
    f = field - field.mean()
    n = f.shape[0]
    w = np.outer(np.hanning(n), np.hanning(n))
    P = np.abs(np.fft.fft2(f * w)) ** 2
    k = np.fft.fftfreq(n, d=dx_km)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2)
    m = (kr >= 1.0 / lam_hi) & (kr < 1.0 / lam_lo)
    rings = np.digitize(np.log(kr[m]), np.linspace(np.log(1.0 / lam_hi), np.log(1.0 / lam_lo), 9))
    pw = P[m] / np.bincount(rings, weights=P[m])[rings] * np.bincount(rings)[rings]
    th = np.mod(np.arctan2(ky[m], kx[m]), np.pi)
    sec = np.minimum((th / np.pi * nsec).astype(int), nsec - 1)
    a = np.bincount(sec, weights=pw, minlength=nsec)
    return float(nsec * a.max() / a.sum())


def grf_field(n, dx_km, spec, seed):
    """isotropic Gaussian random field on a periodic n x n grid with per-mode PSD k^-beta on [1/lmax, 1/lmin]"""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n, d=dx_km)
    kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    amp = np.where((kr >= 1 / spec.lmax_km) & (kr <= 1 / spec.lmin_km), np.maximum(kr, 1e-12) ** (-spec.beta / 2), 0.0)
    f = np.real(np.fft.ifft2(amp * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))))
    return f / f.std()


def check_noise(quick=False):
    spec = chosen_spec(1024)
    # (a) PSD slope at 0, 30, 55 deg on 512 x 512 km boxes at 1 km, 16 realisations
    lo_km = max(5.0, 2 * spec.lmin_km)
    band = (lo_km, 100.0)
    nreal = 4 if quick else 16
    for la in (0.0, 30.0, 55.0):
        lat, lon = tangent_box(la, 20.0, 512.0, 512)
        psd, var = 0.0, []
        for r in range(nreal):
            f = noise_on_box(lat, lon, spec, 100 + r)
            var.append(f.var())
            k, p, _ = radial_psd(f, 1.0, 1.0)
            psd = psd + p
        fit = fit_slope(k, psd / nreal, *band)
        record(f"noise/slope_lat{la:.0f}", abs(fit["slope"] + spec.beta) <= 0.2, slope=fit["slope"], beta=spec.beta,
               band_km=band, r2=fit["r2"], box_variance=float(np.mean(var)), realisations=nreal)
    # (b) stripes: one 4,000 x 4,000 km member at 2 km (F = 1024; F = 64 control; F = 4096 if 1024 fails)
    n, L = (1000, 2000.0) if quick else (2000, 4000.0)
    dx = L / n
    lat, lon = tangent_box(47.0, 10.0, L, n)
    lo_b = max(2 * spec.lmin_km, 2 * 2 * dx)
    bands, b0 = [], lo_b
    while b0 * 2 <= spec.lmax_km * 1.0001:
        bands.append((b0, b0 * 2))
        b0 *= 2
    grfs = [grf_field(n, dx, spec, s) for s in range(3)]
    grf = np.mean([np.array(list(wave_shares(g_, dx, bands).values())) for g_ in grfs], 0)
    grf_ang = float(np.mean([angular_index(g_, dx, bands[0][0], 50.0) for g_ in grfs]))
    del grfs
    fields, verdict = {}, {}
    for F_ in (64, 1024, 4096):
        if F_ == 4096 and verdict.get(1024, {}).get("stripes") is False:
            continue
        sp = NoiseSpec(F_, spec.beta, spec.lmin_km, spec.lmax_km)
        t0 = time.time()
        f = noise_on_box(lat, lon, sp, 7)
        fields[F_] = f
        sh = wave_shares(f, dx, bands)
        excess = {b: s - g for (b, s), g in zip(sh.items(), grf)}
        stripes = bool(max(excess.values()) > 0.20)
        ang = angular_index(f, dx, bands[0][0], 50.0)
        verdict[F_] = dict(stripes=stripes, max_excess=float(max(excess.values())), shares=sh,
                           angular_index_lt50km=float(ang), seconds=time.time() - t0)
        if F_ == 64:
            record("noise/stripes_control_F64_detected", stripes, max_excess=max(excess.values()), shares=sh,
                   angular_index_lt50km=ang)
        else:
            record(f"noise/no_stripes_F{F_}", not stripes, max_excess=max(excess.values()), shares=sh,
                   grf_shares=dict(zip(sh, grf.round(4).tolist())), angular_index_lt50km=ang,
                   angular_index_grf=grf_ang)
    nf = 1024 if not verdict[1024]["stripes"] else (4096 if not verdict.get(4096, {}).get("stripes", True) else None)
    record("noise/feature_count_chosen", nf is not None, n_features=nf)
    _save_noise_maps(fields, grf_field(n, dx, spec, 0), dx, spec, L)
    if not quick and NOISE_FIT_SMOKE.exists() and nf is not None:
        res = json.loads(NOISE_FIT_SMOKE.read_text())
        res["chosen"]["n_features"] = nf
        res["stripe_test"] = {"box": "4000 x 4000 km at 2 km, 47N 10E", "rule": "stripes if max over octave bands of "
                              "(single-wave share - isotropic GRF share) > 0.20; angular index (wavelengths < 50 km, "
                              "1 = isotropic) reported as a diagnostic; maps in logs/s0_noise_maps.png",
                              "bands_km": [list(map(float, b_)) for b_ in bands], "grf_shares": grf.tolist(),
                              "grf_angular_index": float(grf_ang),
                              "verdict": {str(k_): {kk: vv for kk, vv in v.items() if kk != "seconds"}
                                          for k_, v in verdict.items()}, "n_features": nf}
        tmp = NOISE_FIT_SMOKE.with_name(NOISE_FIT_SMOKE.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(NOISE_FIT_SMOKE)


def _save_noise_maps(fields, grf, dx, spec, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(f"sphere noise F = {k}", v) for k, v in fields.items()] + [("isotropic GRF (FFT), same spectrum", grf)]
    hp = {}
    for name, v in panels:                       # high-pass view: wavelengths < 50 km
        n = v.shape[0]
        k = np.fft.fftfreq(n, d=dx)
        kr = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
        hp[name] = np.real(np.fft.ifft2(np.fft.fft2(v) * (kr > 1 / 50.0)))
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 8.4), squeeze=False)
    for j, (name, v) in enumerate(panels):
        axes[0, j].imshow(v[::-1], cmap="RdBu_r", vmin=-3, vmax=3, extent=[0, L, 0, L])
        axes[0, j].set_title(name, fontsize=9)
        h = hp[name]
        s = 3 * h.std()
        axes[1, j].imshow(h[::-1][:500, :500], cmap="RdBu_r", vmin=-s, vmax=s, extent=[0, 500 * dx, 0, 500 * dx])
        axes[1, j].set_title(f"high-pass < 50 km (zoom {500 * dx:.0f} km)", fontsize=9)
    fig.suptitle(f"S0 noise prior: beta {spec.beta}, {spec.lmin_km}-{spec.lmax_km} km, box {L:.0f} km at {dx:.0f} km")
    fig.tight_layout()
    out = PROJECT / "logs" / "s0_noise_maps.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"maps -> {out}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# train / eval (toy store) and DDP
# ---------------------------------------------------------------------------------------------------------------

def _run(cmd, tmp, name, env_extra=None, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.update(env_extra or {})
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=timeout)
    log = Path(tmp) / f"{name}.log"
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        print(f"--- {name} failed (rc {p.returncode}); tail of the log:\n" + (p.stdout + p.stderr)[-4000:], flush=True)
    return p, time.time() - t0


def check_train(tmp):
    from .train_aq import compute_losses
    # (a) 2 optimiser steps of the training loss on the toy grid, plain loop
    st = ToyStore()
    pix = PixelArrays(st, torch.device("cpu"))
    args = argparse.Namespace(objective="fm", grid="025", transport="t6", doy_cond=True, lifting_dim=None,
                              use_checkpoint=False)
    torch.manual_seed(0)
    m = build_model(args, 47, 6, GRIDS["toy"])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.0)
    w0 = torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
    bs = BatchStream(st, 2, 1024, 0, 0, torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(2):
        b = bs.next()
        loss = compute_losses(m, pix, b, args, chosen_spec(), gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    bs.close()
    moved = float((torch.cat([p.detach().flatten() for p in m.parameters()]) - w0).abs().max())
    record("train/two_steps_toy", all(math.isfinite(x) for x in losses) and moved > 0, losses=losses,
           max_weight_change=moved)
    # (b) CLI: fm with validation + checkpoints, resume; regression; flowmap from the fm run; eval
    py = [sys.executable, "-u", "-m"]
    common = ["--store", "toy", "--runs-root", str(tmp), "--batch-size", "2", "--n-points", "256", "--val-members", "4",
              "--val-ode-steps", "3", "--device", "cpu", "--log-every", "2", "--warmup", "2"]
    p, sec = _run(py + ["cdffm_aq.train_aq", "--name", "toy_fm", "--max-steps", "4", "--val-every", "2"] + common,
                  tmp, "train_toy_fm")
    rd = Path(tmp) / "toy_fm"
    rows = [json.loads(x) for x in open(rd / "metrics.jsonl")] if (rd / "metrics.jsonl").exists() else []
    ck = sorted(os.listdir(rd / "ckpt")) if (rd / "ckpt").exists() else []
    record("train/cli_fm_val_ckpt", p.returncode == 0 and [r["step"] for r in rows] == [2, 4]
           and {"step000002.ckpt", "step000004.ckpt", "last.ckpt"} <= set(ck) and "baselines" in rows[0],
           rc=p.returncode, steps=[r["step"] for r in rows], ckpts=ck, seconds=sec,
           val_crps=[r["model"]["ug"]["crps_fair"] for r in rows])
    p, sec = _run(py + ["cdffm_aq.train_aq", "--name", "toy_fm", "--max-steps", "6", "--val-every", "2", "--resume"]
                  + common, tmp, "train_toy_fm_resume")
    rows = [json.loads(x) for x in open(rd / "metrics.jsonl")] if (rd / "metrics.jsonl").exists() else []
    record("train/cli_resume", p.returncode == 0 and [r["step"] for r in rows] == [2, 4, 6]
           and (rd / "ckpt" / "step000006.ckpt").exists(), rc=p.returncode, steps=[r["step"] for r in rows],
           seconds=sec)
    p, sec = _run(py + ["cdffm_aq.train_aq", "--name", "toy_fm_stop", "--max-steps", "3", "--val-every", "2",
                        "--time-budget-min", "30"] + common, tmp, "train_toy_stop_between_val")
    sd = Path(tmp) / "toy_fm_stop" / "ckpt"
    ck = sorted(os.listdir(sd)) if sd.exists() else []
    record("train/cli_final_ckpt_between_validations", p.returncode == 0 and "step000003.ckpt" in ck,
           rc=p.returncode, ckpts=ck, seconds=sec)
    p, sec = _run(py + ["cdffm_aq.train_aq", "--name", "toy_reg", "--objective", "regression", "--max-steps", "2",
                        "--val-every", "2"] + common, tmp, "train_toy_reg")
    record("train/cli_regression", p.returncode == 0 and (Path(tmp) / "toy_reg" / "ckpt" / "last.ckpt").exists(),
           rc=p.returncode, seconds=sec)
    p, sec = _run(py + ["cdffm_aq.train_aq", "--name", "toy_fmap", "--objective", "flowmap", "--teacher", "toy_fm",
                        "--fm-substeps", "2", "--max-steps", "2", "--val-every", "2"] + common, tmp, "train_toy_flowmap")
    record("train/cli_flowmap", p.returncode == 0 and (Path(tmp) / "toy_fmap" / "ckpt" / "last.ckpt").exists(),
           rc=p.returncode, seconds=sec)
    p, sec = _run(py + ["cdffm_aq.eval_aq", "--run", "toy_fm", "--runs-root", str(tmp), "--ckpt", "best", "--members",
                        "4", "--ode-steps", "3", "--device", "cpu", "--map-dates", "2021-03-09", "--zoom",
                        "45.5,10.5,60,100", "--chunk", "5000"], tmp, "eval_toy_fm")
    out = list((rd).glob("eval_*/metrics.json"))
    met = json.loads(out[0].read_text()) if out else {}
    maps = list(rd.glob("eval_*/maps_*_box.png")) + list(rd.glob("eval_*/maps_*_zoom.png"))
    record("train/eval_toy", p.returncode == 0 and bool(met) and set(met.get("scores", {})) ==
           {"model", "climatology_median", "climatology_ens", "cams_bilinear"} and len(maps) == 2,
           rc=p.returncode, gates=met.get("gates"), model_crps=met.get("scores", {}).get("model", {}).get("ug", {})
           .get("crps_fair"), maps=[m_.name for m_ in maps], seconds=sec)
    p, sec = _run(py + ["cdffm_aq.eval_aq", "--run", "toy_reg", "--runs-root", str(tmp), "--ckpt", "last",
                        "--device", "cpu", "--map-dates", "", "--chunk", "5000"], tmp, "eval_toy_reg")
    record("train/eval_toy_regression", p.returncode == 0, rc=p.returncode, seconds=sec)


def check_ddp(tmp):
    cmd = [sys.executable, "-u", "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29541",
           "-m", "cdffm_aq.train_aq", "--name", "toy_ddp", "--store", "toy", "--runs-root", str(tmp), "--max-steps",
           "4", "--val-every", "2", "--batch-size", "2", "--n-points", "256", "--val-members", "2", "--val-ode-steps",
           "2", "--device", "cpu", "--log-every", "2"]
    p, sec = _run(cmd, tmp, "ddp_toy", env_extra={"OMP_NUM_THREADS": "2"})
    rd = Path(tmp) / "toy_ddp"
    rows = [json.loads(x) for x in open(rd / "metrics.jsonl")] if (rd / "metrics.jsonl").exists() else []
    diffs = [r.get("ddp_weights_max_abs_diff") for r in rows]
    days = [r.get("ddp_last_days_by_rank") for r in rows]
    first = [ln for ln in p.stdout.splitlines() if "] step " in ln and " days " in ln]
    distinct = any(d is not None and d[0] != d[1] for d in days)
    record("ddp/cpu_2_ranks", p.returncode == 0 and len(rows) == 2 and all(x == 0.0 for x in diffs) and distinct,
           rc=p.returncode, weights_max_abs_diff=diffs, last_days_by_rank=days, first_steps=first[:6], seconds=sec)


# ---------------------------------------------------------------------------------------------------------------
# store (real smoke data)
# ---------------------------------------------------------------------------------------------------------------

def check_store(quick=False):
    t0 = time.time()
    st = SmokeStore()
    cpu = torch.device("cpu")
    pix = PixelArrays(st, cpu)
    record("store/open", st.n_pix == 6864209 and len(st.train_dates) == 151 and len(st.hold_dates) == 24,
           n_pix=st.n_pix, train=[str(st.train_dates[0]), str(st.train_dates[-1])],
           hold=[str(st.hold_dates[0]), str(st.hold_dates[-1])], seconds=time.time() - t0)
    bs = BatchStream(st, 2, 16384, 0, 0, cpu)
    t1 = time.time()
    b = bs.next()
    bs.close()
    x = assemble_inputs(b["raw"], pix)
    ch_mean = x.mean(dim=(0, 2, 3))
    ch_std = x.std(dim=(0, 2, 3))
    mu, sig = pix.mu_sigma(b["k"], b["doy"])
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(b["k"].shape, generator=gen) - 0.5
    z = target_z(b["v"], mu, sig, u)
    back = z_to_ug(z, mu, sig)
    want = b["v"].float().clamp(min=1) + u
    rt = float(((back - want).abs() / want).max())
    record("store/real_batch", tuple(x.shape) == (2, 47, 576, 1440) and bool(torch.isfinite(x).all())
           and float(ch_mean.abs().max()) < 3 and float(ch_std.min()) > 0.01 and rt < 1e-4,
           days=[str(st.train_dates[int(t)]) for t in b["day"]], read_seconds=time.time() - t1,
           channel_mean_minmax=[float(ch_mean.min()), float(ch_mean.max())],
           channel_std_minmax=[float(ch_std.min()), float(ch_std.max())],
           cams_ch0_mean_std=[float(ch_mean[0]), float(ch_std[0])], t6_mean_std=[ch_mean[41:].tolist(),
                                                                                 ch_std[41:].tolist()],
           z_mean_std=[float(z.mean()), float(z.std())], z_minmax=[float(z.min()), float(z.max())],
           v_minmax=[int(b["v"].min()), int(b["v"].max())], roundtrip_max_rel_err=rt)
    # hold-out truth against CAMS bilinear (B1) on 200k pixels of one day
    t = 7
    d = st.hold_dates[t]
    truth = torch.from_numpy(st.holdout(t).astype(np.float32)).clamp(min=1)
    raw = raw_to_device(st.raw_inputs(d), cpu)
    sel = torch.from_numpy(np.random.default_rng(0).choice(st.n_pix, 200000, replace=False))
    cams = bilinear_log1p_at(raw["cams"][0], pix.lat[sel], pix.lon[sel], st.grid_native)
    yt = truth[sel]
    corr = float(np.corrcoef(np.log1p(cams.numpy()), np.log1p(yt.numpy()))[0, 1])
    record("store/holdout_vs_cams_bilinear", corr > 0.3, date=str(d), corr_log1p=corr,
           mean_truth=float(yt.mean()), mean_cams=float(cams.mean()), rmse=float((cams - yt).pow(2).mean().sqrt()))
    if quick:
        return
    # one forward + backward of the full-size model on CPU (B = 1, N = 4096)
    from .train_aq import compute_losses
    args = argparse.Namespace(objective="fm", grid="025", transport="t6", doy_cond=True, lifting_dim=None,
                              use_checkpoint=False)
    torch.manual_seed(0)
    m = build_model(args, 47, st.static_dim, GRIDS["025"])
    b1 = {"raw": {k_: v[:1] for k_, v in b["raw"].items()}, "k": b["k"][:1, :4096], "v": b["v"][:1, :4096],
          "doy": b["doy"][:1], "day": b["day"][:1]}
    t2 = time.time()
    loss = compute_losses(m, pix, b1, args, chosen_spec(), torch.Generator().manual_seed(0))
    t3 = time.time()
    loss.backward()
    t4 = time.time()
    gn = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    record("store/full_model_fwd_bwd_cpu", math.isfinite(float(loss)) and math.isfinite(gn) and gn > 0,
           loss=float(loss), grad_norm=gn, forward_s=t3 - t2, backward_s=t4 - t3, max_rss_gib=rss,
           threads=torch.get_num_threads())
    # why train_aq does not clip by default: share of each module's gradient elements that a clip at 1.0 (CD-FFM)
    # would push below Adam's eps (1e-8), i.e. freeze; without clipping (default) the same share
    c = min(1.0, 1.0 / gn)
    mods = {"velocity_net": "velocity_net.", "liif_mlp": "query.", "decoder": "encoder.flower.decoder_blocks.",
            "flow_heads": "flow_head"}
    frozen = {}
    for k_, key in mods.items():
        g_ = torch.cat([p.grad.abs().flatten() for n_, p in m.named_parameters() if key in n_])
        frozen[k_] = [float((g_ * c < 1e-8).float().mean()), float((g_ < 1e-8).float().mean())]
    record("store/no_module_frozen_without_clip", all(v[1] < 0.05 for v in frozen.values()), clip1_factor=c,
           frozen_share_clip1_vs_noclip=frozen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", default=["all"])
    ap.add_argument("--quick", action="store_true", help="toy-size dateline/noise boxes, no full-model backward")
    ap.add_argument("--tmp", default=None, help="scratch dir for the toy runs (deleted at the end)")
    a = ap.parse_args()
    nt = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    torch.set_num_threads(nt)
    names = ["shapes", "liif", "dateline", "noise", "train", "ddp", "store"] if a.checks == ["all"] else a.checks
    tmp = Path(a.tmp or tempfile.mkdtemp(prefix="cdffm_s0_"))
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"S0 checks {names} | torch {torch.__version__} | {nt} threads | tmp {tmp}", flush=True)
    for nm in names:
        t1 = time.time()
        print(f"\n=== {nm}", flush=True)
        try:
            {"shapes": check_shapes, "liif": check_liif, "dateline": lambda: check_dateline(a.quick),
             "noise": lambda: check_noise(a.quick), "train": lambda: check_train(tmp), "ddp": lambda: check_ddp(tmp),
             "store": lambda: check_store(a.quick)}[nm]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(f"{nm}/exception", False, error=repr(e)[:500])
        print(f"=== {nm} done in {time.time() - t1:.0f} s", flush=True)
    n_fail = sum(not r["ok"] for r in RESULTS)
    print(f"\nS0 SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAILED: {r['check']}", flush=True)
    if a.tmp is None:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
