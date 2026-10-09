#!/usr/bin/env python3
"""MOSAIC-style spectral check: radially averaged power spectra of forecasts vs truth (GHAP 0.25 deg) over a
land-dominated box of central Europe (96 x 128 cells), 2022, every 7th start day, leads T+1 and T+3.
Deterministic models (reg, ensemble means) should lose small-scale energy (damping); members of a good generative
model should match the truth at every scale.  python spectra.py  -> runs/spectra_2022.json"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dejavu import MU, SD, SIGMA, DVData, DVModule  # noqa: E402

ROOT = Path(__file__).resolve().parent
R0, R1, C0, C1 = 40, 136, 100, 228  # ~62N-38N, 0E-32E


def spectrum(f):
    x = f[R0:R1, C0:C1].astype(np.float64)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    x = x - x.mean()
    p = np.abs(np.fft.fft2(x * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1])))) ** 2
    ky, kx = np.meshgrid(np.fft.fftfreq(x.shape[0]), np.fft.fftfreq(x.shape[1]), indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2)
    bins = np.linspace(0, 0.5, 25)
    idx = np.digitize(k.ravel(), bins)
    return np.array([p.ravel()[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))])


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    models = {s: DVModule.load_from_checkpoint(ROOT / "runs" / s / "final.ckpt", map_location="cpu").to(dev).eval()
              for s in ("dejavu", "noise", "reg") if (ROOT / "runs" / s / "final.ckpt").exists()}
    ds = DVData([2022], "dejavu", train=False)
    acc = {}
    rng = torch.Generator(device=dev); rng.manual_seed(0)
    for n, (i, h) in enumerate(ds.samples):
        if h not in (1, 3) or i % 7:
            continue
        it = ds[n]
        cond = it["cond"][None].to(dev).expand(8, -1, -1, -1); lead = torch.full((8,), float(h), device=dev)
        fields = {"truth": [ds.pm[i + h]], "persistence": [ds.pm[i]]}
        an = torch.from_numpy(ds.analog_sources(i, h)).to(dev)[:, None]
        for s, m in models.items():
            x0 = an + SIGMA * torch.randn(an.shape, device=dev, generator=rng) if s == "dejavu" else \
                torch.randn(8, 1, *an.shape[-2:], device=dev, generator=rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m.sample(x0, cond, lead, 4)[:, 0].float().cpu().numpy() * SD + MU
            fields[f"{s} members"] = list(out) if s != "reg" else [out[0]]
            if s != "reg":
                fields[f"{s} ensemble mean"] = [out.mean(0)]
        fields["analog successors"] = list(ds.analog_sources(i, h)[:, 0] * SD + MU) if ds.analog_sources(i, h).ndim == 4 else list(ds.analog_sources(i, h) * SD + MU)
        for name, fs in fields.items():
            for f in fs:
                acc.setdefault(f"T+{h}|{name}", []).append(spectrum(np.asarray(f)))
    res = {k: np.nanmean(v, 0).tolist() for k, v in acc.items()}
    (ROOT / "runs" / "spectra_2022.json").write_text(json.dumps(res))
    for h in (1, 3):
        t = np.array(res[f"T+{h}|truth"])
        print(f"T+{h}: power / truth, by wavenumber band (large scales -> small scales)")
        for k, v in res.items():
            if k.startswith(f"T+{h}|") and "truth" not in k:
                r = np.array(v) / t
                print(f"  {k.split('|')[1]:24s} large {np.nanmean(r[1:4]):5.2f}  medium {np.nanmean(r[6:12]):5.2f}  small {np.nanmean(r[16:]):5.2f}")


if __name__ == "__main__":
    main()
