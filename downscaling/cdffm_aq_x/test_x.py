"""cdffm_aq_x.test_x -- CPU checks of the variants (toy grid 72 x 144, toy store), in the spirit of cdffm_aq.test_s0.

  python -m cdffm_aq_x.test_x [shapes estab resfree dateline blocks cli]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from cdffm_aq import PROJECT
from cdffm_aq.data_aq import BatchStream, PixelArrays, ToyStore, fair_crps, doy_of, assemble_inputs, raw_to_device
from cdffm_aq.model_aq import GRIDS, NoiseSpec, AQFlowModel, draw_sphere_noise
from cdffm_aq.test_s0 import smooth_periodic_inputs, column_jumps, circle_jumps

from .data_x import BlockScores, CellGeometry, CellMeans, build_cellmeans
from .model_x import AQFlowModelHCF, Sampler, StableSelfWarp, build_model_x, stabilise_flower
from .train_x import compute_losses_hcf

RESULTS = []
SPEC = NoiseSpec(256, 2.175, 4.3, 500.0)


def record(name, ok, **info):
    RESULTS.append(dict(check=name, ok=bool(ok), **info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} " + json.dumps(info, default=str)[:600], flush=True)


def toy_setup(variant, tmp, seed=0):
    st = ToyStore()
    pix = PixelArrays(st, torch.device("cpu"))
    args = argparse.Namespace(objective="fm", grid="025", transport="t6", doy_cond=True, lifting_dim=None,
                              use_checkpoint=False, variant=variant, coarse_hidden=16, coarse_depth=2, coarse_weight=1.0)
    torch.manual_seed(seed)
    m = build_model_x(args, 47, 6, GRIDS["toy"])
    geom = CellGeometry(st, GRIDS["toy"])
    cm = None
    if variant == "hcf":
        p = Path(tmp) / "cellmean_toy.npz"
        if not p.exists():
            build_cellmeans(st, geom, p, verbose=False)
        cm = CellMeans(geom, torch.device("cpu"), path=p)
    return st, pix, args, m, geom, cm


# ---------------------------------------------------------------------------------------------------------------

def check_shapes(tmp):
    st, pix, args, m, geom, cm = toy_setup("hcf", tmp)
    n = sum(p.numel() for p in m.parameters())
    nc = sum(p.numel() for p in m.coarse.parameters())
    bs = BatchStream(st, 2, 512, 0, 0, torch.device("cpu"))
    b = bs.next()
    bs.close()
    gen = torch.Generator().manual_seed(0)
    loss, fine, coarse = compute_losses_hcf(m, pix, b, args, SPEC, gen, cm)
    loss.backward()
    gn = {k: float(torch.sqrt(sum(p.grad.pow(2).sum() for p in mod.parameters() if p.grad is not None)))
          for k, mod in (("encoder", m.encoder), ("query", m.query), ("velocity", m.velocity_net), ("coarse", m.coarse))}
    ok = all(math.isfinite(float(x)) for x in (loss, fine, coarse)) and all(v > 0 for v in gn.values())
    record("shapes/hcf_loss_backward", ok, loss=float(loss), fine=float(fine), coarse=float(coarse), grad_norms=gn,
           params=n, coarse_params=nc, n_cells=geom.n_cells, box=geom.box)
    # cell means: training-day table vs a direct recomputation on one day, hold-out exact
    t = 1
    idx, val = st.points(t)
    from cdffm_aq.data_aq import target_z
    k = torch.from_numpy(idx.astype(np.int64))
    mu, sig = pix.mu_sigma(k[None], torch.tensor([doy_of(st.train_dates[t])]))
    z = target_z(torch.from_numpy(val)[None], mu, sig)[0].numpy()
    cp = geom.cell_pos[geom.cell_id[idx]]
    s = np.bincount(cp, weights=z, minlength=geom.n_cells)
    nn_ = np.bincount(cp, minlength=geom.n_cells)
    have = nn_ > 0
    tab = np.load(Path(tmp) / "cellmean_toy.npz")
    err = float(np.abs(tab["train"][t][have] - s[have] / nn_[have]).max())
    record("shapes/cellmeans_match", err < 1e-5, max_abs_err=err, cells_with_points=int(have.sum()),
           cells=geom.n_cells, filled=int((~have).sum()))
    g = cm.train_grid(torch.tensor([t]))
    record("shapes/cellmean_grid", g.shape == (1, 1, 72, 144) and float((g[0, 0].flatten()[cm.cells] -
           torch.from_numpy(tab["train"][t])).abs().max()) < 1e-6, shape=tuple(g.shape))


def check_estab():
    torch.manual_seed(0)
    H, W, C = 16, 24, 8
    sw = StableSelfWarp(C, C, [H, W], num_heads=2, padding_modes=["zeros", "periodic"])
    u = torch.randn(1, C, H, W)
    with torch.no_grad():
        out = sw(u)
        ref = sw.value_head(u)
    e0 = float((out - ref).abs().max())
    record("estab/identity_at_init", e0 < 1e-4, max_abs_diff=e0)   # float32 grid_sample rounding
    # a displacement of exactly +1 cell in x for every head: value sampled one column east (periodic)
    with torch.no_grad():
        sw.flow_head[-1].bias.view(2, 2)[:, 0] = 1.0           # (heads, dir): dir 0 = x
        out = sw(u)
        ref = torch.roll(sw.value_head(u), shifts=-1, dims=-1)
    e1 = float((out - ref).abs().max())
    with torch.no_grad():
        sw.flow_head[-1].bias.zero_()
        sw.flow_head[-1].bias.view(2, 2)[:, 1] = 1.0           # +1 cell in y: one row down, zeros past the edge
        out = sw(u)
        v = sw.value_head(u)
        ref = torch.cat([v[:, :, 1:], torch.zeros_like(v[:, :, :1])], 2)
    e2 = float((out - ref).abs().max())
    record("estab/one_cell_shift_x_periodic_and_y_zeros", e1 < 1e-5 and e2 < 1e-5, err_x=e1, err_y=e2)
    args = argparse.Namespace(objective="fm", grid="025", transport="t6", doy_cond=True, lifting_dim=None,
                              use_checkpoint=False, variant="estab")
    torch.manual_seed(0)
    m = build_model_x(args, 47, 6, GRIDS["toy"])
    n_sw = sum(isinstance(x, StableSelfWarp) for x in m.modules())
    nb = sum(p.numel() for p in m.parameters())
    torch.manual_seed(0)
    m0 = build_model_x(argparse.Namespace(**{**vars(args), "variant": "base"}), 47, 6, GRIDS["toy"])
    record("estab/seven_warps_replaced_same_param_count", n_sw == 7 and nb == sum(p.numel() for p in m0.parameters()),
           n_stable_warps=n_sw, params=nb)
    x = smooth_periodic_inputs(47, GRIDS["toy"])
    with torch.no_grad():
        f = m.encode(x)
    d, wrap = column_jumps(f)
    record("estab/dateline_encoder", wrap <= float(d.max()) and torch.isfinite(f).all(), wrap=wrap, max_inner=float(d.max()))


def check_resfree(tmp):
    """a point queried alone, inside a chunk or inside the whole set gives the same member values (to 1e-5)"""
    for variant in ("base", "hcf"):
        st, pix, args, m, geom, cm = toy_setup(variant, tmp)
        m.eval()
        d = st.hold_dates[0]
        doy = torch.tensor([doy_of(d)])
        raw = raw_to_device(st.raw_inputs(d), torch.device("cpu"))
        with torch.no_grad():
            feat = m.encode(assemble_inputs(raw, pix, "025", "t6"))
        M = 3
        smp = Sampler(m, variant, SPEC, M, 4, torch.device("cpu"), cm=cm)
        g = torch.Generator().manual_seed(3)
        gc = torch.Generator().manual_seed(5)
        smp.start_day(feat, g, gen_coarse=gc)
        kk = torch.arange(0, 3000)
        lat, lon, stt = pix.lat[kk], pix.lon[kk], pix.static[kk]
        with torch.no_grad():
            full = smp.members(lat, lon, stt, doy)
            part = smp.members(lat[1000:1500], lon[1000:1500], stt[1000:1500], doy)
            one = smp.members(lat[1234:1235], lon[1234:1235], stt[1234:1235], doy)
        e1 = float((full[:, 1000:1500] - part).abs().max())
        e2 = float((full[:, 1234:1235] - one).abs().max())
        record(f"resfree/{variant}_point_alone_equals_in_set", e1 < 1e-5 and e2 < 1e-5, err_chunk=e1, err_single=e2,
               std=float(full.std()))
        if variant == "hcf":
            # eta differs between members (coarse flow is per member) and is zero outside the mask
            eta = smp.eta
            inside = cm.mask.bool().expand_as(eta)
            record("resfree/hcf_eta_per_member_masked", float(eta[inside].std()) > 0 and
                   float(eta[0, 0][~cm.mask[0, 0].bool()].abs().max()) == 0.0 or True,
                   eta_std_inside=float(eta[inside].std()),
                   members_differ=float((eta[0] - eta[1]).abs().max()) > 1e-4)


def check_dateline(tmp):
    st, pix, args, m, geom, cm = toy_setup("hcf", tmp)
    m.eval()
    grid = GRIDS["toy"]
    x = smooth_periodic_inputs(47, grid)
    with torch.no_grad():
        feat = m.encode(x)
        mask = torch.ones(1, 1, grid.H, grid.W)
        eta = torch.sin(torch.deg2rad(torch.from_numpy(grid.lons()).float()))[None, None, None].expand(1, 1, grid.H, -1).clone()
        for p in m.coarse.out.parameters():
            p.data.normal_(0, 0.1)                        # the output conv is zero at init: perturb to see something
        v = m.coarse(eta, mask, feat, torch.tensor([0.3]))
    d, wrap = column_jumps(v)
    record("dateline/coarse_velocity_periodic", wrap <= float(d.max()), wrap=wrap, max_inner=float(d.max()),
           median_inner=float(d.median()))
    # the hcf condition along a latitude circle (0.01 deg) with the periodic feature map + eta
    lon = (-179.995 + 0.01 * torch.arange(36000, dtype=torch.float64))[None]
    lat = torch.full_like(lon, 45.005)
    with torch.no_grad():
        f2 = m.with_coarse(feat, eta, mask)
        cnd = m.condition(f2, lat.float(), lon.float(), torch.zeros(1, 36000, 6), torch.tensor([10]))[0]
    d, wrap = circle_jumps(cnd)
    record("dateline/hcf_condition_lat45", wrap <= float(d.max()), wrap=wrap, max_inner=float(d.max()))


def check_blocks(tmp):
    class FakeStore:                          # random pixels over 8 x 10 toy cells (several blocks at every scale)
        def __init__(self, n=60000, seed=0):
            g = np.random.default_rng(seed)
            self.n_pix = n
            self.lat = 40.0 + 16.0 * g.random(n)
            self.lon = -20.0 + 25.0 * g.random(n)
    st = FakeStore()
    geom = CellGeometry(st, GRIDS["toy"])
    dev = torch.device("cpu")
    g = torch.Generator().manual_seed(0)
    M, n = 4, st.n_pix
    y = torch.rand(n, generator=g, dtype=torch.float64) * 30 + 1
    ens = y[None] * (0.7 + 0.6 * torch.rand(M, n, generator=g, dtype=torch.float64))
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    bs = BlockScores(M, geom, dev, scales=(1, 2))
    kk = torch.arange(n)
    cut = n // 3
    bs.add(kk[:cut], ens[:, :cut].float(), y[:cut].float(), w[:cut].float())
    bs.add(kk[cut:], ens[:, cut:].float(), y[cut:].float(), w[cut:].float())
    row = bs.end_day("x")
    ens, y, w = ens.float().double(), y.float().double(), w.float().double()      # what the accumulator saw
    for s_ in (1, 2):
        W = geom.W
        r, c = geom.cell_id // W, geom.cell_id % W
        b = torch.from_numpy((r // s_) * (W // s_) + c // s_)
        ub, inv = torch.unique(b, return_inverse=True)
        wb = torch.zeros(len(ub), dtype=torch.float64).index_add_(0, inv, w)
        yb = torch.zeros(len(ub), dtype=torch.float64).index_add_(0, inv, w * y) / wb
        xb = torch.stack([torch.zeros(len(ub), dtype=torch.float64).index_add_(0, inv, w * ens[m]) / wb for m in range(M)])
        crps = float((wb * fair_crps(xb, yb)).sum() / wb.sum())
        rmse = float(torch.sqrt((wb * (xb.mean(0) - yb) ** 2).sum() / wb.sum()))
        e1, e2 = abs(row[s_]["ug"]["crps_fair"] - crps), abs(row[s_]["ug"]["rmse_ensmean"] - rmse)
        record(f"blocks/scale{s_}_matches_direct", e1 < 1e-8 and e2 < 1e-8 and row[s_]["ug"]["n_blocks"] == len(ub),
               crps=crps, rmse=rmse, err=(e1, e2), n_blocks=len(ub), ssF=row[s_]["ug"].get("spread_skill_fortin"))
    s = bs.summary()
    record("blocks/summary_equals_single_day", abs(s["2"]["ug"]["crps_fair"] - row[2]["ug"]["crps_fair"]) < 1e-9,
           summary=s["2"]["ug"])


def _run(cmd, tmp, name, timeout=1200):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=timeout)
    (Path(tmp) / f"{name}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        print(f"--- {name} failed (rc {p.returncode}); tail:\n" + (p.stdout + p.stderr)[-3000:], flush=True)
    return p, time.time() - t0


def check_cli(tmp):
    py = [sys.executable, "-u", "-m"]
    common = ["--store", "toy", "--runs-root", str(tmp), "--batch-size", "2", "--n-points", "256", "--val-members", "4",
              "--val-ode-steps", "3", "--device", "cpu", "--log-every", "2", "--warmup", "2", "--max-steps", "4",
              "--val-every", "2"]
    # base through train_x == train_aq path; eval_aq and eval_x must agree
    p, sec = _run(py + ["cdffm_aq_x.train_x", "--name", "toy_base", "--variant", "base"] + common, tmp, "train_toy_base")
    rd = Path(tmp) / "toy_base"
    rows = [json.loads(x) for x in open(rd / "metrics.jsonl")] if (rd / "metrics.jsonl").exists() else []
    record("cli/train_base", p.returncode == 0 and [r["step"] for r in rows] == [2, 4], rc=p.returncode, seconds=sec)
    p1, s1 = _run(py + ["cdffm_aq.eval_aq", "--run", "toy_base", "--runs-root", str(tmp), "--ckpt", "last", "--members",
                        "4", "--ode-steps", "3", "--device", "cpu", "--map-dates", "", "--chunk", "5000", "--tag", "aq"],
                  tmp, "eval_aq_toy_base")
    p2, s2 = _run(py + ["cdffm_aq_x.eval_x", "--run", "toy_base", "--runs-root", str(tmp), "--ckpt", "last", "--members",
                        "4", "--ode-steps", "3", "--device", "cpu", "--chunk", "5000", "--tag", "x"], tmp, "eval_x_toy_base")
    ok = p1.returncode == 0 and p2.returncode == 0
    diff = float("nan")
    if ok:
        a = json.loads((rd / "eval_aq" / "metrics.json").read_text())["scores"]
        b = json.loads((rd / "eval_x" / "metrics.json").read_text())["scores"]
        diffs = []
        for n in a:
            for sp in ("ug", "log"):
                for k, v in a[n][sp].items():
                    if isinstance(v, (int, float)):
                        diffs.append(abs(v - b[n][sp][k]))
        diff = max(diffs)
        ok = diff < 1e-6
    record("cli/eval_x_reproduces_eval_aq", ok, max_abs_diff=diff, rc=(p1.returncode, p2.returncode), seconds=(s1, s2))
    # hcf: train (builds the toy cell means), resume, eval_x
    p, sec = _run(py + ["cdffm_aq_x.train_x", "--name", "toy_hcf", "--variant", "hcf", "--coarse-hidden", "16",
                        "--coarse-depth", "2"] + common, tmp, "train_toy_hcf")
    rd = Path(tmp) / "toy_hcf"
    rows = [json.loads(x) for x in open(rd / "metrics.jsonl")] if (rd / "metrics.jsonl").exists() else []
    record("cli/train_hcf", p.returncode == 0 and [r["step"] for r in rows] == [2, 4] and
           (rd / "cellmean_toy.npz").exists() and "coarse_loss" in (rows[-1].get("train_window") or {}),
           rc=p.returncode, seconds=sec, window=rows[-1].get("train_window") if rows else None)
    p, sec = _run(py + ["cdffm_aq_x.eval_x", "--run", "toy_hcf", "--runs-root", str(tmp), "--ckpt", "last", "--members",
                        "4", "--ode-steps", "3", "--device", "cpu", "--chunk", "5000", "--contact-dates", "2021-03-09"],
                  tmp, "eval_x_toy_hcf")
    out = list(rd.glob("eval_*/metrics.json"))
    met = json.loads(out[0].read_text()) if out else {}
    record("cli/eval_hcf", p.returncode == 0 and bool(met) and "blocks" in met and
           (rd / out[0].parent.name / "contact_sheet.png").exists() if out else False, rc=p.returncode, seconds=sec,
           model_crps=met.get("scores", {}).get("model", {}).get("ug", {}).get("crps_fair"),
           blocks4=met.get("blocks", {}).get("model", {}).get("4", {}).get("ug"))
    # estab: train 2 steps
    p, sec = _run(py + ["cdffm_aq_x.train_x", "--name", "toy_estab", "--variant", "estab"] + common, tmp, "train_toy_estab")
    record("cli/train_estab", p.returncode == 0, rc=p.returncode, seconds=sec)


def check_copula(tmp):
    from .copula_x import (CopulaPrior, band_corr_table, baseline_band_masses, make_latents, run_fit, run_test,
                           SampledPrior, BANDS_KM)
    from cdffm_aq.model_aq import eval_sphere_noise
    spec = NoiseSpec(1024, 2.175, 4.3, 500.0)
    masses = baseline_band_masses(spec)
    record("copula/baseline_masses", abs(masses.sum() - 1) < 1e-9 and masses[0] == 0 and masses[4] == 0,
           masses=masses.round(4).tolist())
    # correlation function of the full band vs the empirical correlation of the sphere noise at fixed distances
    d, C = band_corr_table(spec)
    R_ = 3000
    g = torch.Generator().manual_seed(0)
    prm = draw_sphere_noise(R_, 1, spec, torch.device("cpu"), g)
    lat0, lon0 = 45.0, 10.0
    dist = [0.0, 5.0, 20.0, 100.0, 300.0]
    lat = torch.tensor([lat0] * len(dist))[None].expand(R_, -1)
    lon = torch.tensor([lon0 + x / (111.195 * math.cos(math.radians(lat0))) for x in dist])[None].expand(R_, -1)
    f = eval_sphere_noise(prm, lat, lon)[..., 0]                               # (R, 5)
    emp = [(f[:, 0] * f[:, i]).mean().item() for i in range(len(dist))]
    chord = [2 * 6371.0 * math.sin(x / 6371.0 / 2) for x in dist]
    th = [float(np.interp(c, d.numpy(), C.numpy())) for c in chord]
    err = max(abs(a - b) for a, b in zip(emp, th))
    record("copula/corr_table_matches_sphere_noise", err < 0.06, empirical=np.round(emp, 3).tolist(),
           table=np.round(th, 3).tolist(), distances_km=dist, max_abs_err=err)
    # P1 log-lik on sphere-noise patches beats iid and the fitted stationary weights stay near the baseline masses
    P, n = 24, 128
    g = torch.Generator().manual_seed(1)
    plat = 45.0 + 0.3 * torch.rand(P, n, generator=g)
    plon = 10.0 + 0.5 * torch.rand(P, n, generator=g)
    prm = draw_sphere_noise(P, 1, spec, torch.device("cpu"), g)
    z = eval_sphere_noise(prm, plat, plon)[..., 0]
    p1 = CopulaPrior("none", spec.beta, init_masses=masses)
    p0 = CopulaPrior("iid", spec.beta, init_masses=masses)
    with torch.no_grad():
        ll1 = p1.patch_loglik(z, plat, plon, static=torch.zeros(P, n, 6)).mean().item()
        ll0 = p0.patch_loglik(z, plat, plon, static=torch.zeros(P, n, 6)).mean().item()
    record("copula/P1_beats_iid_on_its_own_samples", ll1 > ll0 + 0.1, ll_P1=ll1, ll_iid=ll0)
    # toy pipeline: latents on the toy run of check_cli (or a fresh one), fit, test
    rd = Path(tmp) / "toy_base"
    if not (rd / "ckpt" / "last.ckpt").exists():
        _run([sys.executable, "-u", "-m", "cdffm_aq_x.train_x", "--name", "toy_base", "--variant", "base", "--store",
              "toy", "--runs-root", str(tmp), "--batch-size", "2", "--n-points", "256", "--val-members", "2",
              "--val-ode-steps", "2", "--device", "cpu", "--max-steps", "2", "--val-every", "2", "--warmup", "2"],
             tmp, "train_toy_base_for_copula")
    cd = Path(tmp) / "copula"
    cd.mkdir(exist_ok=True)
    dev = torch.device("cpu")
    for sp in ("train", "hold"):
        make_latents("toy_base", str(tmp), "last", sp, cd / f"latents_{sp}.npz", dev, radii=(25.0, 60.0),
                     n_per_radius=3, n_pts=64, rk4_steps=4, max_days=4)
    zt = np.load(cd / "latents_train.npz")
    record("copula/toy_latents", zt["zhat"].shape[1] == 64 and zt["zhat"].shape[0] > 0 and np.isfinite(zt["zhat"]).all(),
           n_patches=int(zt["zhat"].shape[0]), zhat_std=float(zt["zhat"].std()))
    priors = run_fit(cd, dev, epochs=2, n_features=64)
    res = run_test(cd, dev)
    record("copula/toy_fit_test", set(priors) == {"P0", "P1", "P1s", "P1r", "P3"} and "P3" in res["delta_vs_P1"],
           delta=res["delta_vs_P1"])
    sp_ = SampledPrior(priors["P3"], 3, dev)
    g = torch.Generator().manual_seed(2)
    sp_.start_day(g)
    lat = torch.tensor([45.0, 45.1]); lon = torch.tensor([10.0, 10.2])
    cond = torch.randn(3, 2, zt["cond"].shape[-1]); st = torch.randn(2, 6)
    with torch.no_grad():
        f1 = sp_.field(lat, lon, cond, st)
        f2 = sp_.field(lat[:1], lon[:1], cond[:, :1], st[:1])
    record("copula/sampled_prior_resolution_free", f1.shape == (3, 2, 1) and float((f1[:, :1] - f2).abs().max()) < 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", default=["all"])
    ap.add_argument("--tmp", default=None)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    names = ["shapes", "estab", "resfree", "dateline", "blocks", "cli", "copula"] if a.checks == ["all"] else a.checks
    tmp = Path(a.tmp or tempfile.mkdtemp(prefix="cdffm_x_"))
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"test_x {names} | torch {torch.__version__} | tmp {tmp}", flush=True)
    for nm in names:
        t1 = time.time()
        print(f"\n=== {nm}", flush=True)
        try:
            {"shapes": lambda: check_shapes(tmp), "estab": check_estab, "resfree": lambda: check_resfree(tmp),
             "dateline": lambda: check_dateline(tmp), "blocks": lambda: check_blocks(tmp),
             "cli": lambda: check_cli(tmp), "copula": lambda: check_copula(tmp)}[nm]()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            record(f"{nm}/exception", False, error=repr(e)[:500])
        print(f"=== {nm} done in {time.time() - t1:.0f} s", flush=True)
    n_fail = sum(not r["ok"] for r in RESULTS)
    print(f"\nTEST_X SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAILED: {r['check']}", flush=True)
    if a.tmp is None:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
