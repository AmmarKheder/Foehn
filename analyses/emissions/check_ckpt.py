#!/usr/bin/env python3
"""What differs between the control checkpoint and the checkpoint with emissions (read only, CPU).
  - hyper-parameters of the two checkpoints (the control must differ by emis_scale only);
  - the input weights of the six emission channels (appended last to the coarse inputs, zero-initialised by init_from):
    exactly zero in the control (the channels are multiplied by 0, so their weights receive no gradient), non-zero with
    emissions;
  - how far the other weights moved from the base model in each fine-tuning (same optimiser, schedule and seed).
    singularity exec ... python check_ckpt.py   -> out/check_ckpt.txt"""
import hashlib
import sys
from pathlib import Path

import torch

R = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/runs")
CK = {"base (A)": R / "green_flow_shift25_global_L7_noant_whres_ghres" / "final.ckpt",
      "with emissions (e02)": R / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12" / "ckpts" / "epoch=02.ckpt",
      "control (ctrl)": R / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12ctrl0" / "ckpts" / "epoch=02.ckpt"}
out = open(Path(__file__).resolve().parent / "out" / "check_ckpt.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n")


S = {}
for k, p in CK.items():
    st = torch.load(p, map_location="cpu", weights_only=False)
    S[k] = st
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    say(f"== {k}: {p}\n   sha256 {h}  size {p.stat().st_size}  epoch {st.get('epoch')}  global_step {st.get('global_step')}")
    say("   hyper-parameters:", dict(st["hyper_parameters"]))
hp_e, hp_c = dict(S["with emissions (e02)"]["hyper_parameters"]), dict(S["control (ctrl)"]["hyper_parameters"])
say("\nhyper-parameters that differ between e02 and the control:", {k: (hp_e.get(k), hp_c.get(k)) for k in set(hp_e) | set(hp_c) if hp_e.get(k) != hp_c.get(k)})
sd_a, sd_e, sd_c = (S[k]["state_dict"] for k in CK)
grown = [k for k in sd_e if k in sd_a and sd_e[k].shape != sd_a[k].shape]
say("\ntensors whose shape grew with the emission channels:", [(k, tuple(sd_a[k].shape), tuple(sd_e[k].shape)) for k in grown])
for k in grown:
    na = sd_a[k].shape[1]
    for name, sd in (("with emissions (e02)", sd_e), ("control (ctrl)", sd_c)):
        w = sd[k][:, na:].float()
        say(f"   {name:22s} {k}: weights of the {w.shape[1]} added input channels: max |w| = {w.abs().max().item():.6g}, "
            f"norm = {w.norm().item():.6g}, non-zero = {(w != 0).sum().item()} of {w.numel()}")
# movement of the shared weights from the base model
def moved(sd):
    num = den = 0.0
    for k, v in sd_a.items():
        if not torch.is_floating_point(v) or k not in sd:
            continue
        w = sd[k].float(); v = v.float()
        if w.shape != v.shape:
            w = w[:, :v.shape[1]]
        num += (w - v).pow(2).sum().item(); den += v.pow(2).sum().item()
    return (num / den) ** 0.5
say(f"\nrelative change of the weights shared with the base model, ||w - w_base|| / ||w_base||: with emissions {moved(sd_e):.5f}, "
    f"control {moved(sd_c):.5f}")
num = den = 0.0
for k, v in sd_c.items():
    if torch.is_floating_point(v) and k in sd_e and k not in grown:
        num += (sd_e[k].float() - v.float()).pow(2).sum().item(); den += v.float().pow(2).sum().item()
say(f"relative difference between e02 and the control on the weights other than the grown tensor: {(num / den) ** 0.5:.5f}")
for name, key in (("with emissions (e02)", "with emissions (e02)"), ("control (ctrl)", "control (ctrl)")):
    st = S[key]
    lr = [g.get("lr") for o in st.get("optimizer_states", []) for g in o.get("param_groups", [])]
    say(f"{name}: learning rate stored in the optimiser state {lr}; lr_schedulers {[ (s.get('last_epoch'), s.get('_last_lr')) for s in st.get('lr_schedulers', [])]}")
out.close()
