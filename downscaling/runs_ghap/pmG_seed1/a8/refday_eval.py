#!/usr/bin/env python3
"""A8 reference check: one or more full selection days of the FROZEN system re-evaluated with the UNMODIFIED
evaluation script (cdffm_aq_x.eval_x.main, same options as the frozen evaluation), plus the location of the member
points that end at the state bound (which eval_x counts but does not locate).

This wrapper changes no number: it wraps model_x.integrate and model_x.Sampler.members only to READ, for the member
points whose final state is at the bound (|z| >= z_bound), their latitude, longitude, member index, starting noise z0
and final z, then calls eval_x.main with the remaining arguments. Everything is written by eval_x in its own output
directory (<runs-root>/<run>/eval_<tag>/); the hits go to --hits-out (npz).

Single process: launch with WORLD_SIZE=1 RANK=0 LOCAL_RANK=<gcd> (eval_x.dist_setup reads them).
  python refday_eval.py --hits-out FILE -- --run pmG_base --runs-root <a8>/ref --ckpt 22000 ... --days 210 --tag ref_day210
"""
import sys
from pathlib import Path

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(W))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cdffm_aq_x.model_x as mx  # noqa: E402

HITS = {k: [] for k in ("lat", "lon", "member", "z0", "z", "call")}
_last = []
_ncall = [0]
_orig_integrate = mx.integrate
_orig_members = mx.Sampler.members


def integrate(model, x, cond, n_steps, flowmap=False, bound=None):
    out = _orig_integrate(model, x, cond, n_steps, flowmap=flowmap, bound=bound)
    if bound is not None:
        hit = out[..., 0].abs() >= bound
        if bool(hit.any()):
            m, i = torch.nonzero(hit, as_tuple=True)
            _last.append((m, i, x[m, i, 0].float(), out[m, i, 0].float()))
    return out


def members(self, lat, lon, static, doy):
    del _last[:]
    z = _orig_members(self, lat, lon, static, doy)
    _ncall[0] += 1
    for m, i, z0, zf in _last:
        HITS["lat"].append(lat[i].double().cpu().numpy())
        HITS["lon"].append(lon[i].double().cpu().numpy())
        HITS["member"].append(m.cpu().numpy())
        HITS["z0"].append(z0.cpu().numpy())
        HITS["z"].append(zf.cpu().numpy())
        HITS["call"].append(np.full(int(m.numel()), _ncall[0], np.int64))
    del _last[:]
    return z


def main():
    argv = sys.argv[1:]
    assert argv[0] == "--hits-out" and argv[2] == "--", "usage: refday_eval.py --hits-out FILE -- <eval_x arguments>"
    hits_out = Path(argv[1])
    mx.integrate = integrate
    mx.Sampler.members = members
    from cdffm_aq_x import eval_x
    assert eval_x.Sampler is mx.Sampler
    eval_x.main(argv[3:])
    hits_out.parent.mkdir(parents=True, exist_ok=True)
    cat = lambda v, dt: (np.concatenate(v) if v else np.zeros(0, dt))
    np.savez(hits_out, lat=cat(HITS["lat"], np.float64), lon=cat(HITS["lon"], np.float64), member=cat(HITS["member"], np.int64),
             z0=cat(HITS["z0"], np.float32), z=cat(HITS["z"], np.float32), call=cat(HITS["call"], np.int64),
             n_member_calls=_ncall[0], argv=np.array(" ".join(argv[3:])))
    n = sum(len(v) for v in HITS["z"])
    print(f"[refday_eval] {n} member points at the bound located -> {hits_out}", flush=True)


if __name__ == "__main__":
    main()
