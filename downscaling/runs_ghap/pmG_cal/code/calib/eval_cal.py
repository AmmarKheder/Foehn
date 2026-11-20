"""calib.eval_cal -- the existing selection evaluation, cdffm_aq_x.eval_x, run UNMODIFIED on an A6 system.

For a fine-tuned checkpoint nothing is changed: this module only calls eval_x.main with the given arguments (the run
directory lives under runs_ghap/pmG_cal/runs, so eval_x writes its eval_<tag>/ there). For the temperature baseline of
A6(e) (environment variable CAL_TAU = tau != 1) the field returned by the fitted prior is multiplied by tau before the
sampler's unchanged truncation |z0| <= 3: copula_x.load_prior is wrapped (eval_x looks it up when it runs), nothing
else differs. Rank 0 writes eval_<tag>/CAL.json (tau, arguments) next to metrics.json.

  srun -n 8 ... python -u -m calib.eval_cal --run V2 --runs-root runs_ghap/pmG_cal/runs --ckpt 1500 --members 8 \
       --ode-steps 10 --days ... --psd-centres ... --prior-file runs_ghap/pmG_final/prior.pt:P3 --z-bound 10 \
       --pmax-ug 1000 --z0-trunc 3 --tag sel2021_M8_s10_P3_guard_T3
"""
import json
import os
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tau = float(os.environ.get("CAL_TAU", "1.0"))
    knee = os.environ.get("CAL_KNEE", "")
    knee = float(knee) if knee not in ("", "none", "None") else None       # tapered temperature (post hoc addendum)
    if tau != 1.0:
        import cdffm_aq_x.copula_x as cx
        from calib.core import temper
        orig = cx.load_prior
        trunc = float(argv[argv.index("--z0-trunc") + 1]) if "--z0-trunc" in argv else None
        if knee is not None and trunc is None:
            raise SystemExit("CAL_KNEE needs --z0-trunc (the taper ends at the truncation bound)")

        def load_prior_tau(spec_str, model, spec, M, device):
            sp = orig(spec_str, model, spec, M, device)
            field0 = sp.field

            def field(lat, lon, cond, static):
                return temper(field0(lat, lon, cond, static), tau, knee, trunc if trunc is not None else 3.0)
            sp.field = field
            return sp
        cx.load_prior = load_prior_tau
    if os.environ.get("CAL_NOHOLES", "") == "1":
        # eval_x --psd-boxes-only fails on a store with has_holes (it mixes the full hold-out row with the box-pixel field in
        # np.where). The 2021 hold-out days are complete (PROTOCOL.md A2(a)), so for the regime-box screening the flag is
        # switched off: a hole, if any, would count as 1 ug m-3 with its weight; the full evaluation never uses this switch.
        from cdffm_aq.store_ghap import GhapStore
        GhapStore.has_holes = False
    from cdffm_aq_x import eval_x
    eval_x.main(argv)
    if int(os.environ.get("SLURM_PROCID", "0")) == 0:
        def val(flag, default=None):
            return argv[argv.index(flag) + 1] if flag in argv else default
        run, root, tag = val("--run"), val("--runs-root"), val("--tag")
        if run and root and tag:
            d = Path(root) / run / f"eval_{tag}"
            if d.is_dir():
                (d / "CAL.json").write_text(json.dumps({"tau": tau, "knee": knee, "holes_flag_off": os.environ.get("CAL_NOHOLES", "") == "1",
                                                        "argv": argv, "protocol": "PROTOCOL.md A6"}, indent=1))


if __name__ == "__main__":
    main()
