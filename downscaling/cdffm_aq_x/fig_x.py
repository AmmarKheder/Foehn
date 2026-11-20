"""cdffm_aq_x.fig_x -- one diagnostic PNG: (a) member / LGHAP and ensemble-mean / LGHAP power ratios against
wavelength for several runs; (b) spread/skill (Fortin) of block means at 1 km (pixels), 0.25, 0.5 and 1 deg.

  python -m cdffm_aq_x.fig_x --out logs/x/fig_psd_blocks.png base_s0=<eval dir> hcf_s0=<eval dir> ...
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=eval_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = plt.cm.tab10.colors
    for i, s in enumerate(a.runs):
        lab, path = s.split("=", 1)
        p = Path(path)
        c = colors[i % 10]
        if (p / "psd.npz").exists():
            z = np.load(p / "psd.npz")
            k, psd, f = z["k_cycles_per_km"], z["psd"], list(z["fields"])
            lam = 1.0 / k
            tr = psd[f.index("truth")]
            axes[0].plot(lam, psd[f.index("member0")] / tr, color=c, label=lab)
            axes[1].plot(lam, psd[f.index("ensmean")] / tr, color=c, label=lab)
            if i == 0:
                axes[0].plot(lam, psd[f.index("cams_bilinear")] / tr, color="k", ls=":", label="CAMS bilinear")
        m = json.loads((p / "metrics.json").read_text())
        if "blocks" in m:
            u = m["scores"]["model"]["ug"]
            b = m["blocks"]["model"]
            xs = ["1 km", "0.25°", "0.5°", "1°"]
            ys = [u.get("spread_skill_fortin", np.nan)] + [b[s]["ug"].get("spread_skill_fortin", np.nan) for s in ("1", "2", "4")]
            axes[2].plot(xs, ys, "o-", color=c, label=lab)
            if i == 0 and "climatology_ens" in m["blocks"]:
                bc = m["blocks"]["climatology_ens"]
                uc = m["scores"]["climatology_ens"]["ug"]
                axes[2].plot(xs, [uc.get("spread_skill_fortin", np.nan)] + [bc[s]["ug"].get("spread_skill_fortin", np.nan)
                                                                          for s in ("1", "2", "4")],
                             "s--", color="grey", label="climatology iid ens.")
    for ax, t in zip(axes[:2], ("member 0 / LGHAP power", "ensemble mean / LGHAP power")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xlabel("wavelength (km)")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[2].axhline(1, color="k", lw=0.8)
    axes[2].set_title("spread/skill (Fortin) of block means, ug m-3", fontsize=10)
    axes[2].set_ylim(0, None)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
