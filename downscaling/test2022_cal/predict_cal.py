#!/usr/bin/env python3
"""2022 protocol of the frozen downscaler (PROTOCOL.md A2(d)) with the CALIBRATED sampler of runs_ghap/pmG_cal/final
(PROTOCOL.md A6 + draft amendment protocol_amendments/A9_calibrated_2022.md).

This is a wrapper, not a copy: it runs test2022_ghap/predict_2022.py UNMODIFIED (same subsample, station positions, arms,
seeds, guard, file formats, firewall: the model process cannot open raw GHAP, sealed or station files). Two hooks only:
  1. cdffm_aq_x.copula_x.load_prior, which predict_2022.main looks up when it runs, returns the fitted prior with its field
     passed through calib.core.temper(field, z0_tau, z0_knee, z0_trunc) before the sampler's unchanged truncation (the same
     wrapping as calib.eval_cal, which made the selection evaluation); z0_tau, z0_knee, z0_trunc are read from
     <system>/SAMPLER.json and must be there;
  2. predict_2022.save_npz adds the calibration to the meta of the downscaler's files (arm cdffm); arrays untouched.
--out is REQUIRED and may not be inside test2022_ghap (the frozen system's results are never written).
  srun -n 8N python -u test2022_cal/predict_cal.py --system runs_ghap/pmG_cal/final --out test2022_cal --full-every 3
  srun -n 8  python -u test2022_cal/predict_cal.py --system runs_ghap/pmG_cal/final --check 2021-05-08 --out <dir>   (no 2022)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
for p_ in (ROOT / "runs_ghap/pmG_cal/code", ROOT / "test2022_ghap", ROOT):
    if str(p_) not in sys.path:
        sys.path.insert(0, str(p_))


def main():
    argv = sys.argv[1:]
    assert "--system" in argv and "--out" in argv, "--system and --out are required"
    sysd = Path(argv[argv.index("--system") + 1]).resolve()
    out = Path(argv[argv.index("--out") + 1]).resolve()
    frozen_dir = (ROOT / "test2022_ghap").resolve()
    assert out != frozen_dir and frozen_dir not in out.parents, f"--out {out} is inside the frozen system's test2022_ghap"
    sj = json.loads((sysd / "SAMPLER.json").read_text())
    assert "z0_tau" in sj and "z0_knee" in sj and sj.get("z0_trunc") is not None, f"{sysd}/SAMPLER.json has no calibration (z0_tau, z0_knee)"
    tau, trunc = float(sj["z0_tau"]), float(sj["z0_trunc"])
    knee = None if sj["z0_knee"] is None else float(sj["z0_knee"])
    assert tau != 1.0, "z0_tau = 1 is the frozen system: use test2022_ghap/predict_2022.py"
    cal = {"z0_tau": tau, "z0_knee": knee, "z0_trunc": trunc, "map": sj.get("z0_map"), "protocol": sj.get("protocol")}

    import cdffm_aq_x.copula_x as cx
    from calib.core import temper
    temper(__import__("torch").zeros(1), tau, knee, trunc)          # raises on an invalid (tau, knee)
    orig = cx.load_prior

    def load_prior_cal(spec_str, model, spec, M, device):
        sp = orig(spec_str, model, spec, M, device)
        field0 = sp.field

        def field(lat, lon, cond, static):
            return temper(field0(lat, lon, cond, static), tau, knee, trunc)
        sp.field = field
        return sp
    cx.load_prior = load_prior_cal

    import predict_2022 as P                                         # the frozen system's script; main() guarded
    save0 = P.save_npz

    def save_npz(path, **arrays):
        if "meta" in arrays:
            m = json.loads(str(arrays["meta"]))
            if m.get("arm") == "cdffm":
                m["calibration"] = cal
                arrays["meta"] = np.array(json.dumps(m))
        save0(path, **arrays)
    P.save_npz = save_npz
    P.main()


if __name__ == "__main__":
    main()
