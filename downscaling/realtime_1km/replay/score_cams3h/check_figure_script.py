#!/usr/bin/env python3
"""cams3h: does this directory work as a scorer directory for the figure of the real-time skill?

The data script of the figure (paper/natcomms/figures_new/fig_rt_skill/rt_skill_data.py) writes beside itself, so it is
NOT run in place: a copy is run in a temporary directory under work/ of this directory with `--score <this directory>`
(the scorer's code is taken from ../score). Nothing is written under the paper folder. Kept from its output: its own checks
against the tables, the shares of cells / monitors / issue days against CAMS (compared with side_by_side.json), the 0.4
degree input against CAMS on the rings sample. Evaluation only.
    python -I -B check_figure_script.py   -> figure_script_check.json"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
SRC = Path("/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/figures_new/fig_rt_skill/rt_skill_data.py")
sys.path.insert(0, str(R / "cams3h/code"))
import safe_io  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp(prefix="figcheck_", dir=str(HERE / "work")))
    try:
        shutil.copy(SRC, tmp / "rt_skill_data.py")
        r = subprocess.run([sys.executable, "-I", "-B", "rt_skill_data.py", "--score", str(HERE), "--code", str(R / "score"), "--tag", "c3h"],
                           cwd=str(tmp), capture_output=True, text=True, timeout=1200)
        log = (r.stdout + r.stderr)[-4000:]
        assert r.returncode == 0, log
        a = json.loads((tmp / "numbers_c3h.json").read_text())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    s = json.loads((HERE / "side_by_side.json").read_text())
    fc, fm, fd = a["a_map"]["share_lower_rmse_pct"]["cells_vs_cams"], a["a_map"]["share_lower_rmse_pct"]["monitors_vs_cams"], a["e_days"]["counts"]["cams"]
    mc, mm, md = s["share_lower_rmse"]["new"]["cells"], s["share_lower_rmse"]["new"]["monitors"], s["by_issue_day"]["counts"]["new"]
    close = lambda x, y: abs(float(x) - float(y)) < 1e-9  # noqa: E731
    same = dict(cells=close(fc["est"], mc["est"]) and close(fc["ci"][0], mc["ci"][0]) and close(fc["ci"][1], mc["ci"][1]) and fc["units"] == mc["units"] and fc["units_lower"] == mc["units_lower"],
                monitors=close(fm["est"], mm["est"]) and close(fm["ci"][0], mm["ci"][0]) and close(fm["ci"][1], mm["ci"][1]),
                issue_days=fd["pooled_lower"] == md["pooled_lower"] and fd["pooled_interval_below_zero"] == md["pooled_interval_below_zero_cells_only"]
                and fd["regmean_lower"] == md["regmean_lower"] and close(fd["pooled_min"], md["pooled_min"]) and close(fd["pooled_max"], md["pooled_max"]))
    out = dict(script=str(SRC), run_as=f"copy in a temporary directory, --score {HERE} --code {R / 'score'}",
               script_checks={k: v["identical"] for k, v in a["checks"].items()}, script_checks_all_identical=all(v["identical"] for v in a["checks"].values()),
               shares_from_the_figure_script=dict(cells_vs_cams=fc, monitors_vs_cams=fm, issue_days_vs_cams=fd,
                                                  cells_below_minus2=a["a_map"]["cells_below_minus2"], cells_within_2=a["a_map"]["cells_within_2"],
                                                  cells_above_plus2=a["a_map"]["cells_above_plus2"], cells_coloured=a["a_map"]["cells_coloured"]),
               identical_to_side_by_side=same, legend_numbers_vs_cams=dict(pooled=a["legend_numbers"]["change_pooled"]["cams"], regmean=a["legend_numbers"]["change_regmean"]["cams"]),
               input_vs_cams_on_the_rings_sample=a["d_rings"]["input_vs_cams_same_station_days"], log_tail=log[-1500:])
    safe_io.write_json(HERE / "figure_script_check.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "log_tail"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
