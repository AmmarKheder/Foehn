#!/usr/bin/env python3
"""CPU checks of the HCF variant ON THE GLOBAL STORE (global-train agent, 2026-10-26; PROTOCOL.md Amendment A1).
Complements scripts/test_global_wiring.py (global-prep: store, firewall, batch, CLI) and cdffm_aq_x.test_x (toy grid).
Exit 1 if any check fails. Random-init model (geometry checks, not skill), smoke-default HCF sizes (64 x 3).

  shapes    real global training batch (B = 1, N = 4,096): compute_losses_hcf finite, backward reaches encoder / LIIF /
            velocity / coarse; cell-mean grid (1, 1, 576, 1440) equals the table row; coarse field eta of the Sampler
            (M, 1, 576, 1440) finite, the LIIF's eta channel zero outside the mask cells; members (M, n) finite
  resfree   a real 2023 validation day: points queried alone, in a chunk and in a 20,000-point set spread over the
            globe give the same members (|diff| < 1e-4 in z), for base and hcf
  dateline  real 2023 day inputs: (1) encoder features, (2) the coarse velocity (output conv perturbed) and (3) the hcf
            condition along a latitude circle at 0.01 deg (36,000 points, 4 latitudes) have no jump across the
            antimeridian larger than the largest jump between neighbours; (4) hcf members 0.01 deg apart across the
            antimeridian (230 latitudes, 8 members) differ no more than 2 x members 0.01 deg apart elsewhere
  injh      (only when the injh store passed its check) GlobalStore(fires='injh'): channel 2 differs from mami on fire
            days from 2018-07-02, channels 0-1 identical, finite
  python scripts/test_global_hcf.py [shapes resfree dateline injh] --tmp DIR
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(ROOT))
RESULTS = []


def record(name, ok, **info):
    RESULTS.append(dict(check=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} " + json.dumps(info, default=str)[:800], flush=True)


_CACHE = {}


def setup(variant):
    import torch
    from cdffm_aq.data_aq import PixelArrays, in_channels
    from cdffm_aq.model_aq import GRIDS, NoiseSpec
    from cdffm_aq.store_global import GlobalStore, injh_status
    from cdffm_aq_x import CELLMEAN_GLOBAL
    from cdffm_aq_x.data_x import CellGeometry, CellMeans
    from cdffm_aq_x.model_x import build_model_x
    if "store" not in _CACHE:
        fires = "injh" if injh_status()[0] else "mami"
        st = GlobalStore(fires=fires)
        _CACHE.update(store=st, pix=PixelArrays(st, torch.device("cpu")), fires=fires,
                      geom=CellGeometry(st, GRIDS["025"]))
    st, pix, geom = _CACHE["store"], _CACHE["pix"], _CACHE["geom"]
    args = argparse.Namespace(objective="fm", grid="025", transport="t6", doy_cond=True, lifting_dim=None,
                              use_checkpoint=False, variant=variant, coarse_hidden=64, coarse_depth=3,
                              coarse_weight=1.0, coarse_dilations=None)
    torch.manual_seed(0)
    m = build_model_x(args, in_channels("t6"), st.static_dim, GRIDS["025"])
    m.eval()
    cm = CellMeans(geom, torch.device("cpu"), path=CELLMEAN_GLOBAL) if variant == "hcf" else None
    spec = NoiseSpec(256, 2.175, 4.3, 500.0)
    return st, pix, args, m, geom, cm, spec


def day_features(m, st, pix, t):
    import torch
    from cdffm_aq.data_aq import assemble_inputs, raw_to_device, doy_of
    d = st.hold_dates[t]
    raw = raw_to_device(st.raw_inputs(d), torch.device("cpu"))
    with torch.no_grad():
        feat = m.encode(assemble_inputs(raw, pix, "025", "t6"))
    return feat, torch.tensor([doy_of(d)]), d


def check_shapes():
    import torch
    from cdffm_aq.data_aq import BatchStream
    from cdffm_aq_x.model_x import Sampler
    from cdffm_aq_x.train_x import compute_losses_hcf
    st, pix, args, m, geom, cm, spec = setup("hcf")
    m.train()
    bs = BatchStream(st, 1, 4096, 0, 0, torch.device("cpu"))
    b = bs.next()
    bs.close()
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    loss, fine, coarse = compute_losses_hcf(m, pix, b, args, spec, gen, cm)
    loss.backward()
    gn = {k: float(torch.sqrt(sum(p.grad.pow(2).sum() for p in mod.parameters() if p.grad is not None)))
          for k, mod in (("encoder", m.encoder), ("query", m.query), ("velocity", m.velocity_net), ("coarse", m.coarse))}
    # the coarse output conv is zero at init: its gradient is nonzero, the earlier coarse layers' may be 0 at step 0
    ok = all(math.isfinite(float(x)) for x in (loss, fine, coarse)) and gn["encoder"] > 0 and gn["query"] > 0 \
        and gn["velocity"] > 0 and gn["coarse"] > 0
    record("shapes/global_hcf_loss_backward", ok, loss=float(loss), fine=float(fine), coarse=float(coarse),
           grad_norms=gn, day=str(st.train_dates[int(b["day"][0])]), seconds=time.time() - t0,
           params=sum(p.numel() for p in m.parameters()), coarse_params=sum(p.numel() for p in m.coarse.parameters()),
           n_cells=geom.n_cells, fires=_CACHE["fires"])
    g = cm.train_grid(b["day"])
    tab_row = cm.train[int(b["day"][0])]
    record("shapes/global_cellmean_grid", tuple(g.shape) == (1, 1, 576, 1440)
           and float((g[0, 0].flatten()[cm.cells] - tab_row).abs().max()) == 0.0
           and float(g[0, 0].flatten()[cm.mask[0, 0].flatten() == 0].abs().max()) == 0.0
           and bool(torch.isfinite(cm.train).all()), shape=tuple(g.shape), n_train_rows=int(cm.train.shape[0]),
           table_std=float(cm.train.std()))
    m.eval()
    feat, doy, d = day_features(m, st, pix, 0)
    M = 2
    smp = Sampler(m, "hcf", spec, M, 2, torch.device("cpu"), cm=cm, coarse_spec=spec)
    smp.start_day(feat, torch.Generator().manual_seed(1), gen_coarse=torch.Generator().manual_seed(2))
    eta = smp.eta
    k = torch.arange(0, st.n_pix, st.n_pix // 5000)
    with torch.no_grad():
        z = smp.members(pix.lat[k], pix.lon[k], pix.static[k], doy)
    outside = cm.mask[0, 0] == 0
    lf = smp.feats[0][:, -2:]                         # [eta * mask, mask] channels seen by the LIIF
    record("shapes/global_sampler_eta_members", tuple(eta.shape) == (M, 1, 576, 1440) and bool(torch.isfinite(eta).all())
           and float(lf[0, 0][outside].abs().max()) == 0.0 and tuple(z.shape) == (M, k.numel())
           and bool(torch.isfinite(z).all()), eta_shape=tuple(eta.shape), z_shape=tuple(z.shape),
           eta_std_inside=float(eta[:, 0][:, ~outside].std()), z_std=float(z.std()), date=str(d))


def check_resfree():
    import torch
    from cdffm_aq_x.model_x import Sampler
    for variant in ("base", "hcf"):
        st, pix, args, m, geom, cm, spec = setup(variant)
        feat, doy, d = day_features(m, st, pix, 15)
        smp = Sampler(m, variant, spec, 3, 3, torch.device("cpu"), cm=cm, coarse_spec=spec)
        smp.start_day(feat, torch.Generator().manual_seed(3), gen_coarse=torch.Generator().manual_seed(5))
        rng = np.random.default_rng(0)
        kk = torch.from_numpy(np.sort(rng.choice(st.n_pix, 20000, replace=False)))
        lat, lon, stt = pix.lat[kk], pix.lon[kk], pix.static[kk]
        with torch.no_grad():
            full = smp.members(lat, lon, stt, doy)
            part = smp.members(lat[7000:9000], lon[7000:9000], stt[7000:9000], doy)
            one = smp.members(lat[12345:12346], lon[12345:12346], stt[12345:12346], doy)
        e1 = float((full[:, 7000:9000] - part).abs().max())
        e2 = float((full[:, 12345:12346] - one).abs().max())
        record(f"resfree/global_{variant}_point_alone_equals_in_set", e1 < 1e-4 and e2 < 1e-4, err_chunk=e1,
               err_single=e2, member_std=float(full.std()), date=str(d))


def check_dateline():
    import torch
    from cdffm_aq.test_s0 import column_jumps, circle_jumps
    from cdffm_aq_x.model_x import Sampler
    st, pix, args, m, geom, cm, spec = setup("hcf")
    feat, doy, d = day_features(m, st, pix, 30)
    dd, wrap = column_jumps(feat)
    record("dateline/global_encoder_features", wrap <= float(dd.max()), wrap=wrap, max_inner=float(dd.max()),
           median_inner=float(dd.median()), date=str(d))
    H, W = 576, 1440
    lons = torch.linspace(-180 + 0.125, 180 - 0.125, W)
    eta = torch.sin(torch.deg2rad(lons))[None, None, None].expand(1, 1, H, -1).clone() * cm.mask
    with torch.no_grad():
        for p in m.coarse.out.parameters():
            p.data.normal_(0, 0.1)
        v = m.coarse(eta, cm.mask, feat, torch.tensor([0.3]))
    dd, wrap = column_jumps(v)
    record("dateline/global_coarse_velocity_periodic", wrap <= float(dd.max()), wrap=wrap, max_inner=float(dd.max()),
           median_inner=float(dd.median()))
    f2 = m.with_coarse(feat, eta, cm.mask)
    lon = (-179.995 + 0.01 * torch.arange(36000, dtype=torch.float64))[None].float()
    res = {}
    for la in (-17.005, 45.005, 65.005, 0.005):
        lat = torch.full_like(lon, la)
        with torch.no_grad():
            cnd = m.condition(f2, lat, lon, torch.zeros(1, 36000, st.static_dim), doy)[0]
        dd, wrap = circle_jumps(cnd)
        res[la] = (wrap, float(dd.max()))
    record("dateline/global_hcf_condition_4_latitudes", all(w <= mx for w, mx in res.values()),
           wrap_vs_max_inner={str(k): v for k, v in res.items()})
    # members across the antimeridian (resolution-free queries, so any (lat, lon) is valid): points 0.01 deg apart
    # across the dateline (-179.995 vs +179.995) must differ like points 0.01 deg apart inside (+179.985 vs +179.995);
    # same statics for all three (those of a real Fiji/Pacific-row pixel when there is one, else of pixel 0)
    lonp = pix.lon.numpy()
    near = np.nonzero(np.abs(lonp) > 179.0)[0]
    src = int(near[0]) if near.size else 0
    lats = torch.arange(-54.995, 60.0, 0.5)
    n = lats.numel()
    stt = pix.static[src][None].expand(n, -1)
    smp = Sampler(m, "hcf", spec, 8, 3, torch.device("cpu"), cm=cm, coarse_spec=spec)
    smp.start_day(feat, torch.Generator().manual_seed(7), gen_coarse=torch.Generator().manual_seed(11))
    with torch.no_grad():
        zw = smp.members(lats, torch.full((n,), -179.995), stt, doy)
        ze = smp.members(lats, torch.full((n,), 179.995), stt, doy)
        ze2 = smp.members(lats, torch.full((n,), 179.985), stt, doy)
        zm = smp.members(lats, torch.full((n,), 0.005), stt, doy)
        zm2 = smp.members(lats, torch.full((n,), 0.015), stt, doy)
    across = float((zw - ze).pow(2).mean().sqrt())
    inner_e = float((ze - ze2).pow(2).mean().sqrt())
    inner_m = float((zm - zm2).pow(2).mean().sqrt())
    ref = max(inner_e, inner_m, 1e-6)
    record("dateline/global_members_across_antimeridian", across <= 2 * ref and math.isfinite(across),
           rms_across_0p01deg=across, rms_inside_east_0p01deg=inner_e, rms_greenwich_0p01deg=inner_m,
           n_lat=n, member_std=float(zw.std()), statics_from_pixel=src)


def check_injh():
    import torch  # noqa: F401
    from cdffm_aq.store_global import GlobalStore, injh_status
    ok, why = injh_status()
    if not ok:
        record("injh/store_ready", False, reason=why)
        return
    si = GlobalStore(fires="injh", firewall=False)
    sm = GlobalStore(fires="mami", firewall=False)
    diffs = {}
    for d in (dt.date(2019, 8, 15), dt.date(2021, 9, 1), dt.date(2022, 12, 31), dt.date(2023, 7, 15)):
        a = si.raw_inputs(d)["fires"].astype(np.float32)
        b = sm.raw_inputs(d)["fires"].astype(np.float32)
        diffs[str(d)] = dict(ch01_equal=bool(np.array_equal(a[:2], b[:2])), ch2_diff_frac=float((a[2] != b[2]).mean()),
                             finite=bool(np.isfinite(a).all()), corr_ch2=float(np.corrcoef(a[2].ravel(), b[2].ravel())[0, 1]))
    record("injh/global_store_channels", all(v["ch01_equal"] and v["finite"] and v["ch2_diff_frac"] > 0
                                              for v in diffs.values()), per_day=diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", default=["shapes", "resfree", "dateline", "injh"])
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    t0 = time.time()
    for nm in a.checks:
        print(f"\n=== {nm}", flush=True)
        try:
            {"shapes": check_shapes, "resfree": check_resfree, "dateline": check_dateline, "injh": check_injh}[nm]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(f"{nm}/exception", False, error=repr(e)[:500])
    n_fail = sum(not r["ok"] for r in RESULTS)
    print(f"\nGLOBAL HCF SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAILED: {r['check']}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
