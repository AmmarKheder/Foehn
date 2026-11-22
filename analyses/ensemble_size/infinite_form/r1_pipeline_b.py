#!/usr/bin/env python3
"""A14 class U on all 179 start days, after R1 (a14/data/run16_13) has passed check13b (a14/results/check13b.txt 'passed';
amendment A14a, the rounding tolerance of check 13). A copy of r1_pipeline.py (whose gate, check13.txt, cannot pass as coded:
check 0) with the fixes of 2026-11-21: the gate reads check13b.txt only (no 'passed' value is ever read from
check13.txt or check13.json); the scorers are the fixed ones; fix 17: the pixel-days that have the four stored members but not
the sixteen are counted here from the pixel files (n4 - n per forecast), and the station-days of the same kind are reported by
st_cf_score.py / rel_score.py and printed in the 179-day table. Nothing is run unless the gate passed. Steps (CPU only):
  1. pop_pass.py run16 r1 a14/data/run16_13 (294 forecasts) and run16 t0 a13/warn16/data/run16all --only 2:1,2:3,4:1,4:3
     (the 4 forecasts of R1 written by task 0 of job 23898998; its leads 5 and 7 are not used, A14 section 5)
  2. fix 17: dropped pixel-days (n4 - n) of every sixteen-member pixel file; no forecast in two files
  3. st_cf_score.py, rel_score.py, chain_classU.py with run16 + r1 + t0 (sixteen members on all 179 start days, K2 on all),
     outputs *_179.json (the 30-day files are kept)
  4. make_table.py -> results/classU_table_179.{md,json}
    python r1_pipeline_b.py              the real run
    python r1_pipeline_b.py --dry-run    test of steps 2-4 without R1: run16 stands in (A14U_DRY179=1 forces the 179-day
                                         branches on its 30 days), outputs *_dry.json and classU_table_dry.{md,json}, deleted
                                         after the check; R1 is not read"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from u_common import AR, DATA, DESIGN, P, RES, RUN16_AR, SFX

DRY = "--dry-run" in sys.argv
chk = P / "a14" / f"results{SFX}" / "check13b.txt"                    # design: the check13b of the same design (a14/results_<D>)
if not chk.exists():
    sys.exit(f"{chk} missing: R1 not checked, nothing run")
r = json.loads(chk.read_text())
comp = r.get("completeness", {})
print("check13b:", {"passed": r.get("passed"), "fresh_process_ok": r.get("fresh_process_ok"),
                    **{k: comp.get(k) for k in ("forecasts_planned", "done", "missing", "extra", "wrong_target")}}, flush=True)
if r.get("passed") is not True:
    sys.exit("check13b did not pass: nothing run (see a14/results/check13b.txt)")
sfx = "_dry" if DRY else "_179"
env = dict(os.environ, A14U_OUT=sfx, A14U_PREFER=sfx)
if DRY:
    env["A14U_DRY179"] = "1"
tags = ["run16"] if (DRY or AR) else ["run16", "r1", "t0"]          # 'ar': the 358 forecasts in one folder, one tag
py = [sys.executable, "-u", "-B"]
if DESIGN:                                                             # design: every step under the write guard of the S1 chain
    py.append((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/run.py"))


def run(s):
    t0 = time.time()
    print("[r1_pipeline_b] run:", " ".join(s[3:]), flush=True)
    subprocess.run(s, check=True, env=env)
    print(f"[r1_pipeline_b] done in {time.time() - t0:.0f} s", flush=True)


if AR and not DRY:                                                     # A16: all 179 start days (the 30 seen ones among them)
    run(py + ["pop_pass.py", "run16", "run16", str(RUN16_AR), "--workers", "8"])
elif not DRY:
    run(py + ["pop_pass.py", "run16", "r1", str(P / "a14" / "data" / "run16_13"), "--workers", "8"])
    run(py + ["pop_pass.py", "run16", "t0", str(P / "a13" / "warn16" / "data" / "run16all"), "--workers", "4", "--only", "2:1,2:3,4:1,4:3"])
# fix 17: pixel-days with the four stored members finite but not the sixteen, per forecast, in every sixteen-member pixel file
keys_all = []
for tag in tags:
    z = np.load(DATA / f"pop16_{tag}.npz")
    d = z["n4"][:, 0] - z["n"][:, 0]
    keys_all += [tuple(k) for k in z["keys"].tolist()]
    print(f"[r1_pipeline_b] fix 17, pop16_{tag}: {len(d)} forecasts; pixel-days dropped (n4 - n, unweighted) {d.sum():.0f}, "
          f"forecasts with drops {int((d > 0).sum())}" + (f", worst {d.max():.0f} at {tuple(z['keys'][int(d.argmax())])}" if d.max() > 0 else ""), flush=True)
    assert (d >= 0).all(), f"pop16_{tag}: more pixels with sixteen members than with four"
assert len(set(keys_all)) == len(keys_all), "a forecast appears in two sixteen-member pixel files"
print(f"[r1_pipeline_b] sixteen-member forecasts in the pixel files: {len(keys_all)} ({len({k[0] for k in keys_all})} start days, "
      f"leads {sorted({k[1] for k in keys_all})})", flush=True)
if AR and not DRY:                                                     # A16 Addendum 2 point 2: member-0 ratios of the new run
    run(py + ["j_m0_design.py"])
for s in (py + ["st_cf_score.py"] + tags, py + ["rel_score.py"] + tags, py + ["chain_classU.py"], py + ["make_table.py"]):
    run(s)
st = json.loads((RES / f"st_cf_classU{sfx}.json").read_text())
rl = json.loads((RES / f"rel_classU{sfx}.json").read_text())
print("[r1_pipeline_b] fix 17, station-days (compare_fair sample):", st.get("dropped_16_station_days"), flush=True)
print("[r1_pipeline_b] fix 17, station-days (reliability samples):", rl.get("stations_reliability_16", {}).get("dropped"), flush=True)
print("[r1_pipeline_b] fix 17, populated pixels:", rl.get("populated_pixels", {}).get("dropped_16"), flush=True)
print(f"[r1_pipeline_b] all steps done -> results/classU_table{sfx}.md", flush=True)
