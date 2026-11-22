#!/usr/bin/env python3
"""A13 warn16, CPU checks before any GPU job (writes results/check_inputs.json):
  1. final.ckpt and ckpts/epoch=01.ckpt (the checkpoint the stored members were made with, logs/pop_crps_e01_22867011.out)
     hold the same weights and hyper-parameters;
  2. members 0-3 of noise_tiles16.npy equal noise_tiles.npy (bytes);
  3. the populated-pixel accumulation of sample16.py equals the dense float32 map of eval_global_green.run at the populated
     pixels BIT FOR BIT (real windows, real blend weights, random window outputs; one member);
  4. the start days and keys of the test (every sixth start day of the 179 of analyses/hidden/v3).
    python check_inputs.py"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sample16 as s16  # noqa: E402  (installs the audit hook; imports eval_global_green unchanged)
import torch  # noqa: E402

egg = s16.egg
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
EVP = RUN / "eval_pop_crps_e01"
DOYS = list(range(0, 349, 12))
R = {}

# 1 ------------------------------------------------------------------
t0 = time.time()
a = torch.load(RUN / "final.ckpt", map_location="cpu", weights_only=False)
b = torch.load(RUN / "ckpts" / "epoch=01.ckpt", map_location="cpu", weights_only=False)
sa, sb = a["state_dict"], b["state_dict"]
same_keys = sorted(sa) == sorted(sb)
same_w = same_keys and all(torch.equal(sa[k], sb[k]) for k in sa)
R["ckpt"] = dict(final_equals_epoch01_weights=bool(same_w), same_keys=bool(same_keys), n_tensors=len(sa),
                 hparams_equal=a.get("hyper_parameters") == b.get("hyper_parameters"),
                 top_keys_final=sorted(a), top_keys_epoch01=sorted(b), epoch_final=a.get("epoch"), epoch_epoch01=b.get("epoch"))
del a, b, sa, sb
print("1.", R["ckpt"], f"{time.time() - t0:.0f} s", flush=True)

# 2 ------------------------------------------------------------------
t0 = time.time()
n4, n16 = egg.Npy(egg.G / "noise_tiles.npy"), egg.Npy(egg.G / "noise_tiles16.npy")
per = int(np.prod(n4.shape[1:]))
eq, h4, h16 = [], [], []
for m in range(4):
    x, y = n4._read(m * per, per), n16._read(m * per, per)
    eq.append(bool(np.array_equal(x.view(np.uint16), y.view(np.uint16))))
    h4.append(hashlib.md5(x.tobytes()).hexdigest()); h16.append(hashlib.md5(y.tobytes()).hexdigest())
R["noise"] = dict(shape4=list(n4.shape), shape16=list(n16.shape), dtype=str(n4.dtype), members0_3_equal=eq, md5_4=h4, md5_16=h16)
print("2.", R["noise"], f"{time.time() - t0:.0f} s", flush=True)

# 3 ------------------------------------------------------------------
t0 = time.time()
win = egg.windows(np.load(egg.G / "meta.npz")["land"])
pop_idx = np.load(egg.G / "pop_pixels.npz")["idx"]
assert np.array_equal(pop_idx, np.load(EVP / "pop_idx.npy")), "pixel list differs from the stored one"
FH, FW, GH, GW = egg.FH, egg.FW, egg.GH, egg.GW
wgt = np.outer(egg.ramp(FH, egg.RAMP), egg.ramp(FW, egg.RAMP))
wsum = np.zeros((GH, GW), np.float32)
for k0, j0 in win:
    wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] += wgt
covered = wsum > 0
inv_full = np.where(covered, 1.0 / np.maximum(wsum, 1e-12), 0.0).astype(np.float32)
del wsum
pw = s16.pop_windows(pop_idx, win)
rng = np.random.default_rng(0)
dense = np.zeros((GH, GW), np.float32)
accp = np.zeros(len(pop_idx), np.float32)
for w, (k0, j0) in enumerate(win):
    v = (wgt * (rng.standard_normal((FH, FW), dtype=np.float32) * 20 + 30)).astype(np.float32)   # float32 product as on the GPU
    r0, c0 = 25 * k0, 25 * j0
    dense[r0:r0 + FH, c0:c0 + FW] += v
    sel, loc = pw[w]
    accp[sel] += v.reshape(-1)[loc]
dense *= inv_full
accp *= inv_full.ravel()[pop_idx]
d16, p16 = dense.ravel()[pop_idx].astype(np.float16), accp.astype(np.float16)
R["accumulation"] = dict(windows=len(win), pop_pixels=int(len(pop_idx)), pixels_in_no_window=int((~covered.ravel()[pop_idx]).sum()),
                         float32_bit_identical=bool(np.array_equal(dense.ravel()[pop_idx].view(np.uint32), accp.view(np.uint32))),
                         float16_bit_identical=bool(np.array_equal(d16.view(np.uint16), p16.view(np.uint16))),
                         window_memberships=int(sum(len(s) for s, _ in pw)))
del dense
print("3.", R["accumulation"], f"{time.time() - t0:.0f} s", flush=True)

# 4 ------------------------------------------------------------------
import glob  # noqa: E402
K = {}
for p in glob.glob(str(EVP / "part_*.npz")):
    k = int(p.split("_")[-1].split(".")[0])
    for n, (d, h, t) in enumerate(np.load(p)["keys"]):
        K[(int(d), int(h))] = (k, n, int(t))
s1 = sorted(d for d, h in K if h == 1)
assert s1 == sorted(d for d, h in K if h == 3) and len(s1) == 179
assert s1[::6] == DOYS, (s1[::6], DOYS)
R["start_days"] = dict(analysis_start_days=len(s1), every_sixth=DOYS, n=len(DOYS),
                       dates=[str(np.datetime64("2022-01-01") + int(d)) for d in DOYS],
                       stored_location={f"{d},{h}": list(K[(d, h)]) for d in DOYS for h in (1, 3)})
print("4.", len(DOYS), "start days:", DOYS, flush=True)
txt = json.dumps(R, indent=1)
out = HERE / "results" / "check_inputs.json"; tmp = out.with_suffix(".tmp")
with open(tmp, "w") as f_:
    f_.write(txt); f_.flush(); os.fsync(f_.fileno())
assert tmp.read_text() == txt; os.replace(tmp, out)
ok = R["ckpt"]["final_equals_epoch01_weights"] and all(R["noise"]["members0_3_equal"]) and R["accumulation"]["float32_bit_identical"]
print("ALL CHECKS PASS" if ok else "A CHECK FAILED", flush=True)
