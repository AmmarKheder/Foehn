"""(1) the edited fullflow_global_fp.py gives the same velocity as the version before the fp_physics option for an
existing checkpoint; (2) the FLEXPART-like option has no learnable parameter in the transport and trains (finite grads)."""
import sys, time
from pathlib import Path
import torch
GK = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km")
sys.path.insert(0, str(GK)); sys.path.insert(0, str(GK / "tests"))
import fullflow_global_fp as new
import fullflow_global_fp_before_physics as old
dev = torch.device("cuda")
ck = GK / "runs/fp_flow_shift25_global_L7_fix/final.ckpt"
ds = new.GlobalDataFP([2021], train=False, seed=1, max_lead=7)
b = {k: (v[None] if v.dim() > 0 else v.reshape(1)).to(dev) for k, v in ds[5].items()}
torch.manual_seed(0); x = torch.randn_like(b["x1"].float()); tau = torch.full((1,), 0.6, device=dev)
out = {}
for name, mod in (("old", old), ("new", new)):
    m = mod.FullModuleFP.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    with torch.no_grad():
        out[name] = m.velocity(x, b, tau).float()
    del m; torch.cuda.empty_cache()
print("same checkpoint, old vs new code: max |dv| =", float((out["old"] - out["new"]).abs().max()), "| max |v| =", float(out["old"].abs().max()))
m = new.FullModuleFP(lr=2e-4, epochs=60, shift=25.0, max_lead=7, fp_physics=True).to(dev)
fp_params = {n: p.numel() for n, p in m.net.fp.named_parameters()}
print("physics transport, learnable parameters inside the footprint layer:", fp_params, "| buffers:", {n: tuple(t.shape) for n, t in m.net.fp.named_buffers()})
t0 = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    v = m.velocity(x, b, tau)
loss = (v.float() - b["x1"].float()).pow(2).mean(); loss.backward()
g = [p.grad for p in m.parameters() if p.grad is not None]
print(f"physics: loss {float(loss):.4f}, {len(g)} tensors with grads, all finite {all(bool(torch.isfinite(t).all()) for t in g)}, step {time.time() - t0:.1f}s")
print("hparams of the physics model:", dict(m.hparams))
