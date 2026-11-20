"""calib.test_cal -- CPU plumbing checks of the A6 code on the synthetic ToyStore (no project data, no score):
  1. euler_members (with gradient) == model_aq.integrate (the inference Euler loop with the state bound);
  2. the members of calib.core (band fields + weights, truncation, integrate) == those of model_x.Sampler +
     copula_x.SampledPrior with the same generator (what eval_x and the real-time adapter draw);
  3. cal_losses: finite loss in z and in ug; with --train-part velocity only the velocity net gets a gradient, with
     all every weight does; the fair CRPS of M = 2 equals data_aq.fair_crps;
  4. validate_set + summarise run and agree with data_aq.Scores on the pooled numbers; temperature 1.0 is the identity.
    python -m calib.test_cal            (inside the container, working directory = project root)
"""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import torch

from cdffm_aq.data_aq import BatchStream, PixelArrays, Scores, ToyStore, assemble_inputs, doy_of, fair_crps, raw_to_device, z_to_ug
from cdffm_aq.model_aq import GRIDS, NoiseSpec, build_model, integrate
from cdffm_aq_x.copula_x import SampledPrior
from cdffm_aq_x.model_x import Sampler

from .core import (ValSet, band_fields, draw_prior_params, euler_members, fair_crps_members, guard, load_prior_module,
                   prior_z0, stat_layout, summarise, validate_set)
from .train_cal import cal_losses


def main():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    store = ToyStore()
    grid = GRIDS["toy"]
    args = SimpleNamespace(grid="025", transport="t6", lifting_dim=None, use_checkpoint=False, doy_cond=True, objective="fm",
                           stations=None, crps_members=2, crps_steps=10, crps_space="z", crps_fm=1.0, crps_scale=6.0,
                           z_bound=10.0, z0_trunc=3.0, pmax_ug=1000.0)
    model = build_model(args, 47, store.static_dim, grid)
    pix = PixelArrays(store, dev)
    spec = NoiseSpec(64, 2.175, 4.3, 500.0)
    prior = load_prior_module(dev)
    for s in prior.specs:                                   # fewer Fourier features: CPU test
        object.__setattr__(s, "n_features", 64)
    g = guard()
    print("guard", g, "| prior", prior.kind, "K", prior.K)
    M, n = 8, 3000
    # ---- 1 and 2
    d = store.hold_dates[1]
    doy = torch.tensor([doy_of(d)])
    kk = torch.arange(0, n)
    lat, lon, stat = pix.lat[kk], pix.lon[kk], pix.static[kk]
    with torch.no_grad():
        feat = model.encode(assemble_inputs(raw_to_device(store.raw_inputs(d), dev), pix, "025", "t6"))
        smp = Sampler(model, "base", spec, M, 10, dev, prior=SampledPrior(prior, M, dev), z_bound=g["z_bound"], z0_trunc=1.5)
        g1, gp = torch.Generator(), torch.Generator()
        g1.manual_seed(23)
        gp.manual_seed(41)
        smp.start_day(feat, g1, gen_prior=gp)
        z_ref = smp.members(lat, lon, stat, doy)
        gp2 = torch.Generator()
        gp2.manual_seed(41)
        params = draw_prior_params(prior, M, dev, gp2)
        xi = band_fields(prior, params, lat, lon, M)
        cond = model.condition(feat, lat[None], lon[None], stat[None], doy)
        z0 = prior_z0(prior, xi, cond.expand(M, -1, -1), stat[None].expand(M, -1, -1)).clamp(-1.5, 1.5)
        z_mine = integrate(model, z0, cond.expand(M, -1, -1), 10, bound=g["z_bound"])[..., 0]
        z_grad = euler_members(model, z0, cond.expand(M, -1, -1), 10, g["z_bound"])[..., 0]
    print(f"2. members vs Sampler: max |diff| {float((z_ref - z_mine).abs().max()):.2e} (z std {float(z_ref.std()):.3f}, "
          f"z0 std {float(z0.std()):.3f}, truncated {int((z0.abs() >= 1.5).sum())})")
    print(f"1. euler_members vs integrate: max |diff| {float((z_grad - z_mine).abs().max()):.2e}")
    assert float((z_ref - z_mine).abs().max()) < 1e-5 and float((z_grad - z_mine).abs().max()) < 1e-6
    # bound active: tiny bound
    with torch.no_grad():
        a_ = integrate(model, z0 * 5, cond.expand(M, -1, -1), 10, bound=0.7)
        b_ = euler_members(model, z0 * 5, cond.expand(M, -1, -1), 10, 0.7)
    assert float((a_ - b_).abs().max()) < 1e-6 and float(a_.abs().max()) <= 0.7 + 1e-6
    print("   with an active bound: identical, max |z|", float(a_.abs().max()))
    # ---- 3
    xs = torch.randn(3, 2, 50)
    y = torch.randn(3, 50)
    ref = torch.stack([fair_crps(xs[i], y[i]) for i in range(3)]).mean()
    assert abs(float(fair_crps_members(xs, y) - ref)) < 1e-6
    xs5 = torch.randn(3, 5, 50)
    ref5 = torch.stack([fair_crps(xs5[i], y[i]) for i in range(3)]).mean()
    assert abs(float(fair_crps_members(xs5, y) - ref5)) < 1e-6
    print("3. fair CRPS (M = 2 and 5) equals data_aq.fair_crps")
    stream = BatchStream(store, 2, 512, 0, 0, dev)
    b = stream.next()
    gen = torch.Generator()
    gen.manual_seed(1)
    for space in ("z", "ug"):
        for part in ("velocity", "all"):
            args.crps_space = space
            model.zero_grad(set_to_none=True)
            for n_, p in model.named_parameters():
                p.requires_grad_(part == "all" or n_.startswith("velocity_net."))
            loss, crps, fm = cal_losses(model, pix, b, args, spec, gen, prior, part == "velocity")
            loss.backward()
            got = {n_.split(".")[0] for n_, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
            none = [n_ for n_, p in model.named_parameters() if p.requires_grad and p.grad is None]
            print(f"   space {space} part {part}: loss {float(loss):.4f} crps {float(crps):.4f} fm {float(fm):.4f} | gradient "
                  f"in {sorted(got)}; trainable without gradient: {len(none)}")
            assert torch.isfinite(loss) and not none
            assert got == ({"velocity_net"} if part == "velocity" else {"encoder", "query", "velocity_net"}), got
    stream.close()
    for n_, p in model.named_parameters():
        p.requires_grad_(True)
    # ---- 4
    days = [0, 1, 2]
    pixs = np.sort(np.random.default_rng(0).choice(store.n_pix, 2000, replace=False))
    vs = ValSet("toy", "hold", days, [store.hold_dates[t] for t in days], pixs, np.stack([store.holdout(t)[pixs] for t in days]))
    cache = {}
    rows = validate_set(model, store, pix, vs, prior, dev, M=M, cache=cache, cond_frozen=True, g=g).numpy()
    rows2 = validate_set(model, store, pix, vs, prior, dev, M=M, cache=cache, cond_frozen=True, g=g, tau=1.0).numpy()
    rows3 = validate_set(model, store, pix, vs, prior, dev, M=M, cache={}, cond_frozen=False, g=g).numpy()
    assert np.allclose(rows, rows2) and np.allclose(rows, rows3, rtol=1e-6, atol=1e-9), "cache changes the result"
    s = summarise(rows, M)
    # independent recomputation with Scores through the Sampler
    sc = Scores(M, dev)
    with torch.no_grad():
        for i, t in enumerate(days):
            d = store.hold_dates[t]
            doy = torch.tensor([doy_of(d)])
            feat = model.encode(assemble_inputs(raw_to_device(store.raw_inputs(d), dev), pix, "025", "t6"))
            smp = Sampler(model, "base", spec, M, 10, dev, prior=SampledPrior(prior, M, dev), z_bound=g["z_bound"],
                          z0_trunc=g["z0_trunc"])
            g1, gp = torch.Generator(), torch.Generator()
            g1.manual_seed(t * 1009 + 23)
            gp.manual_seed(t * 1009 + 41)
            smp.start_day(feat, g1, gen_prior=gp)
            kk = torch.from_numpy(pixs)
            z = smp.members(pix.lat[kk], pix.lon[kk], pix.static[kk], doy)
            mu, sig = pix.mu_sigma(kk[None], doy)
            sc.add(z_to_ug(z, mu, sig, pmax=g["pmax_ug"]), torch.from_numpy(vs.truth[i].astype(np.float32)).clamp(min=1.0), pix.w_area[kk])
    r = sc.summary()["ug"]
    print(f"4. validate_set vs Scores through the Sampler: crps {s['ug']['crps_fair']:.6f} / {r['crps_fair']:.6f}, rmse "
          f"{s['ug']['rmse_ensmean']:.6f} / {r['rmse_ensmean']:.6f}, ssF {s['ug']['spread_skill_fortin']:.6f} / "
          f"{r['spread_skill_fortin']:.6f}; layout {len(stat_layout(M))} sums per day; regions {list(s['regions'])}")
    for k in ("crps_fair", "rmse_ensmean", "spread_skill_fortin"):
        assert abs(s["ug"][k] - r[k]) < 1e-5 * max(1.0, abs(r[k])), k
    rows_t = validate_set(model, store, pix, vs, prior, dev, M=M, cache=cache, cond_frozen=True, g=g, tau=1.5).numpy()
    st = summarise(rows_t, M)
    print(f"   temperature 1.5: ssF {st['ug']['spread_skill_fortin']:.4f} (1.0: {s['ug']['spread_skill_fortin']:.4f}), "
          f"truncated {st['sanity']['n_member_z0_trunc']:.0f} (1.0: {s['sanity']['n_member_z0_trunc']:.0f})")
    assert st["z"]["member_variance"] > s["z"]["member_variance"]
    # ---- 5: tapered temperature
    from .core import temper
    x = torch.linspace(-4, 4, 8001)
    assert torch.equal(temper(x, 1.0, 1.5), x)
    assert torch.allclose(temper(x, 1.5, None), 1.5 * x)
    y = temper(x, 1.5, 2.0)                                  # knee = 3 / tau: the plain temperature after the truncation
    assert torch.allclose(y.clamp(-3, 3), (1.5 * x).clamp(-3, 3), atol=1e-6)
    y = temper(x, 1.6, 1.5)
    assert bool((y[1:] >= y[:-1] - 1e-6).all()) and torch.allclose(y, -y.flip(0), atol=1e-6)        # monotone, odd
    assert abs(float(temper(torch.tensor(3.0), 1.6, 1.5)) - 3.0) < 1e-6 and abs(float(temper(torch.tensor(1.5), 1.6, 1.5)) - 2.4) < 1e-6
    assert int((y.abs() > 3).sum()) == int((x.abs() > 3).sum())                                   # the truncation acts on the same draws
    rows_k = validate_set(model, store, pix, vs, prior, dev, M=M, cache=cache, cond_frozen=True, g=g, tau=1.5, knee=2.0).numpy()
    keep = [i for i, n_ in enumerate(stat_layout(M)) if n_ != "san/n_member_z0_trunc"]     # the count of |z0| > 3 differs by construction
    assert np.allclose(rows_k[:, keep], rows_t[:, keep], rtol=1e-5, atol=1e-6), "knee = 3 / tau must reproduce the plain temperature"
    rows_k = validate_set(model, store, pix, vs, prior, dev, M=M, cache=cache, cond_frozen=True, g=g, tau=1.5, knee=1.5).numpy()
    sk = summarise(rows_k, M)
    print(f"5. tapered temperature: identities hold; tau 1.5 knee 1.5: ssF {sk['ug']['spread_skill_fortin']:.4f}, truncated "
          f"{sk['sanity']['n_member_z0_trunc']:.0f} (frozen {s['sanity']['n_member_z0_trunc']:.0f}, plain 1.5: {st['sanity']['n_member_z0_trunc']:.0f})")
    assert sk["sanity"]["n_member_z0_trunc"] == s["sanity"]["n_member_z0_trunc"]
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
