#!/usr/bin/env python3
"""Extended Data figure: who a 40 km grid hides, by income (2022; no forecast involved). Days of 2022 above 37.5 and 75 ug m-3
at the residents' 1 km pixel whose 40 km cell is below the threshold (figdata_exposure_income.py -> data/exposure_income_2022.npz,
the exact computation of the paper's 186 million residents a day, with every 1 km pixel labelled by income):
a, per resident, by World Bank income group of the country (FY2024, 2022 GNI per capita); numbers: share of the world's hidden
   person-days above 37.5 ug m-3;
b, per resident, by global decile of GDP per capita PPP 2022 of the second-level administrative unit (Kummu et al. 2025),
   deciles of the world's residents;
c, per resident, by quintile of GDP per capita within each country (the 134 countries whose quintiles each hold 10-30 % of their
   residents, 97 % of the world's population);
d, the share of the person-days above the threshold that the 40 km cells hide, by global decile.
    python fig_income.py"""
import numpy as np

from gc_style import clean, letter_before
from nature_style import DOUBLE, HERE, MM, plt, save

C375, C75 = "#C2185B", "#5C0A2C"                          # the thresholds as in Fig. 2e (> 37.5, > 75 ug m-3)
GROUPS = ["Low", "Lower-middle", "Upper-middle", "High"]


def split(a, k):
    """person-days above threshold k at 1 km, and hidden by the 40 km cells, from counts (..., level 1 km, level 40 km)"""
    return a[..., k + 1:, :].sum((-1, -2)), a[..., k + 1:, :k + 1].sum((-1, -2))


def main():
    z = np.load(HERE / "data" / "exposure_income_2022.npz"); T = list(z["thr"]); nd = int(z["ndays"])
    TH = ((37.5, C375, "o", "-"), (75.0, C75, "s", (0, (3, 1.6))))
    W, H = DOUBLE, 118 * MM
    fig = plt.figure(figsize=(W, H))
    fx = lambda mm: mm / (W / MM); fy = lambda mm: mm / (H / MM)
    axa = fig.add_axes([fx(13), fy(67), fx(72), fy(38)])
    axb = fig.add_axes([fx(107), fy(67), fx(73), fy(38)])
    axc = fig.add_axes([fx(13), fy(12), fx(72), fy(38)])
    axd = fig.add_axes([fx(107), fy(12), fx(73), fy(38)])

    # a: between countries
    pw = z["pop_wb"][:4]; x = np.arange(4)
    for j, (thr, col, *_ ) in enumerate(TH):
        e, h = split(z["wb"], T.index(thr)); v = h[:4] / pw
        axa.bar(x + (j - 0.5) * 0.37, v, width=0.35, color=col, zorder=3)
        print(f"a > {thr:g}: days per resident", dict(zip(GROUPS, np.round(v, 2))), "| M/day", np.round(h[:4] / nd / 1e6, 1),
              "| share of world hidden %", np.round(100 * h[:4] / h.sum(), 1))
        if thr == 37.5:
            for xi, vi, si in zip(x, v, 100 * h[:4] / h.sum()):
                axa.text(xi - 0.185, vi + 0.35, f"{si:.0f}%", ha="center", va="bottom", fontsize=5.6, color="#1B1E23")
    axa.set_xticks(x); axa.set_xticklabels([f"{g}\nincome" for g in GROUPS], fontsize=5.6); axa.tick_params(axis="x", length=0)
    axa.set_ylim(0, 15.5); axa.set_ylabel("Hidden days per resident, 2022", fontsize=6)
    axa.set_xlabel("World Bank income group of the country", fontsize=6, labelpad=2)
    axa.text(0.99, 0.97, "numbers: share of the world's\nhidden person-days (> 37.5)", transform=axa.transAxes, ha="right", va="top",
             fontsize=5.2, color="#5F6873", linespacing=1.15)
    axa.set_title("Per resident, lower-income countries lose more warning days", fontsize=6.3, loc="left", pad=5)

    # b, d: by global decile of local GDP per capita
    pd_ = z["pop_decile"][:10]; xd = np.arange(1, 11); edges = z["decile_edges"]
    for thr, col, mk, ls in TH:
        e, h = split(z["decile"], T.index(thr))
        axb.plot(xd, h[:10] / pd_, color=col, marker=mk, ms=2.6, lw=1.3, ls=ls, zorder=3)
        axd.plot(xd, 100 * h[:10] / e[:10], color=col, marker=mk, ms=2.6, lw=1.3, ls=ls, zorder=3)
        print(f"b > {thr:g}: days per resident by decile", np.round(h[:10] / pd_, 2), "| d share hidden %", np.round(100 * h[:10] / e[:10], 1))
    for ax_ in (axb, axd):
        ax_.set_xticks(xd); ax_.set_xlim(0.5, 10.5)                     # the GDP per capita of the poorest and richest tenth
        ax_.set_xticklabels([f"1\n< ${edges[0] / 1000:.1f}k"] + [str(k) for k in range(2, 10)] + [f"10\n> ${edges[-1] / 1000:.0f}k"])
        ax_.set_xlabel("Global decile of local GDP per capita (poorest to richest)", fontsize=6, labelpad=2)
    axb.set_ylim(0, 15.5); axb.set_ylabel("Hidden days per resident, 2022", fontsize=6)
    axb.set_title("By local income, the middle of the world is hidden most", fontsize=6.3, loc="left", pad=5)
    axd.set_ylim(0, 50); axd.set_ylabel("Share of the exposure hidden (%)", fontsize=6)
    axd.set_title("The share hidden rises with income: exceedances become local", fontsize=6.3, loc="left", pad=5)

    # c: within countries (quintiles of each country's residents; countries with well-defined quintiles)
    P = z["pop_cq"].reshape(-1, 6); nc = P.shape[0]
    share = P[:, :5] / np.maximum(P[:, :5].sum(1, keepdims=True), 1)
    good = (share.min(1) > 0.10) & (share.max(1) < 0.30) & (P[:, :5].sum(1) > 1e6); good[0] = False
    xq = np.arange(1, 6)
    for thr, col, mk, ls in TH:
        e, h = split(z["cq"], T.index(thr)); h = h.reshape(nc, 6)
        v = h[good, :5].sum(0) / P[good, :5].sum(0)
        axc.plot(xq, v, color=col, marker=mk, ms=2.6, lw=1.3, ls=ls, zorder=3)
        r = h[good, :5] / np.maximum(P[good, :5], 1); ok = r.sum(1) > 0
        nup, nall = int((r[ok, 4] > r[ok, 0]).sum()), int(ok.sum())
        print(f"c > {thr:g}: days per resident by quintile", np.round(v, 2), f"| richest > poorest in {nup} of {nall} countries")
        if thr == 37.5:
            axc.text(0.03, 0.97, f"richest fifth above the poorest\nin {nup} of {nall} countries (> 37.5)", transform=axc.transAxes,
                     ha="left", va="top", fontsize=5.2, color="#5F6873", linespacing=1.15)
    print(f"c: {int(good.sum())} countries, {100 * P[good, :5].sum() / P[1:, :5].sum():.0f} % of the world's residents")
    axc.set_xticks(xq); axc.set_xticklabels(["poorest\nfifth", "2nd", "3rd", "4th", "richest\nfifth"], fontsize=5.6)
    axc.set_xlim(0.6, 5.4); axc.set_ylim(0, 17.5); axc.set_ylabel("Hidden days per resident, 2022", fontsize=6)
    axc.set_xlabel("Fifth of each country's residents by local GDP per capita", fontsize=6, labelpad=2)
    axc.set_title("Within countries, the wealthier, urban districts are hidden", fontsize=6.3, loc="left", pad=5)

    for ax_ in (axa, axb, axc, axd):
        clean(ax_); ax_.tick_params(labelsize=5.5)
    axa.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([0], [0], color=col, marker=mk, ms=2.6, lw=1.3, ls=ls, label=f"> {thr:g} µg m$^{{-3}}$, WHO interim target {3 if thr == 37.5 else 1}")
          for thr, col, mk, ls in TH]
    fig.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, fy(117.6)), ncol=2, fontsize=5.6, frameon=False, handlelength=2.4,
               columnspacing=2.0)
    fig.canvas.draw()
    for ax_, s_ in zip((axa, axb, axc, axd), "abcd"):
        letter_before(fig, ax_._left_title, s_)
    save(fig, "figED_income")


if __name__ == "__main__":
    main()
