#!/usr/bin/env python3
"""Figures 1-4 of the manuscript, from out/results.json and the 10^9 window histograms."""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("out/results.json"))
OUT = "out/figures"
os.makedirs(OUT, exist_ok=True)
RG = R["ranges"]
WIN = {int(k): v for k, v in R["windows"].items()}
DRAFT = [10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10]
FRESH = [125894656, 199524352, 316227584, 501186560, 794329088, 1258921984, 1995259904, 3162275840]

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "cm",
})
BLACK, BLUE, ORANGE, GREEN, GREY = "#111111", "#1f5fa8", "#d9661f", "#2a8a57", "#888888"


def xs(rows):
    return np.array([r["nmid"] for r in rows])


def col(rows, key, i=0):
    def g(r):
        v = r.get(key)
        if v is None:
            return np.nan
        return v[i] if isinstance(v, list) else v
    return np.array([g(r) for r in rows], dtype=float)


def series(ax, key, color, label, fmt="o", ms=3.2, mfc=None, shift=1.0, with_windows=True):
    """Dyadic ranges (circles/fmt), first-draft windows (squares), fresh windows (triangles)."""
    ax.errorbar(xs(RG) * shift, col(RG, key), yerr=col(RG, key, 1), fmt=fmt, ms=ms, color=color,
                mfc=mfc or color, capsize=1.5, elinewidth=0.8, label=label)
    if not with_windows:
        return
    for group, mk in ((DRAFT, "s"), (FRESH, "^")):
        rows = [WIN[A] for A in group if A in WIN]
        if rows:
            ax.errorbar(xs(rows) * shift, col(rows, key), yerr=col(rows, key, 1), fmt=mk, ms=ms,
                        color=color, mfc="white", capsize=1.5, elinewidth=0.8)


def all_rows():
    return RG + [WIN[A] for A in DRAFT + FRESH if A in WIN]


def fig1():
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 5.6), sharex=True)
    rows = sorted(all_rows(), key=lambda r: r["nmid"])
    series(a, "shortfall_R", GREY, r"$\delta_R=\sqrt{n}\,(1-R/\mathrm{HLi})$")
    series(a, "tau", BLUE, r"$\tau=\sqrt{n}\,T/\mathrm{HLi}$ (measured)", shift=1.06)
    a.plot(xs(rows), col(rows, "tau_pred"), "-", color=BLUE, label=r"$\tau_{\rm pred}=\sqrt{n}\,S\varepsilon I_T/\mathrm{HLi}$")
    a.plot(xs(rows), [r["granville_term"] for r in rows], "--", color=ORANGE,
           label=r"Granville: $4\langle\varepsilon\rangle$")
    a.set_ylabel(r"units of $1/\sqrt{n}$")
    a.set_ylim(0, 5.2)
    a.legend(loc="center", bbox_to_anchor=(0.5, 0.7), ncol=2, frameon=False)
    a.set_title("(a) shortfall of $R$ and the prime-square count", loc="left", fontsize=8.5)
    series(b, "offset_rw", BLACK, r"$\sqrt{n}\,(R_w/\mathrm{HLi}-1)$ (measured)")
    b.plot(xs(rows), col(rows, "zeta"), "o", ms=3.5, mfc="none", color=GREEN,
           label=r"$\sqrt{n}\,\mathcal{L}(n)$, uniform weights")
    if "zeta_w" in rows[0]:
        b.plot(xs(rows), col(rows, "zeta_w"), "x", ms=4, color=ORANGE,
               label=r"$\sqrt{n}\,\mathcal{L}_w(n)$, logarithmic weights")
    b.axhline(0, color=GREY, lw=0.6)
    b.set_ylabel(r"units of $1/\sqrt{n}$")
    b.set_xscale("log")
    b.set_xlabel(r"$n$")
    b.legend(loc="lower right", frameon=False)
    b.set_title(r"(b) offset of $R_w$ and the linear zeta term", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_shortfall.pdf")
    plt.close(fig)


def fig2():
    fig, axs = plt.subplots(2, 3, figsize=(6.6, 4.4), sharex=True)
    rows = sorted(all_rows(), key=lambda r: r["nmid"])
    for ax, q in zip(axs.flat, (3, 5, 7, 11, 13)):
        for key, color, lab, sh, mk in (("raw", BLACK, r"$R/\mathrm{HLi}$ (raw)", 1.0, "o"),
                                        ("ours", BLUE, r"$R/P$ (corrected)", 1.08, "D"),
                                        ("granville", ORANGE, r"$R/I^*$ (Granville)", 0.92, "s")):
            y = np.array([r["gaps"][str(q)][key][0] for r in rows])
            e = np.array([r["gaps"][str(q)][key][1] for r in rows])
            ax.errorbar(xs(rows) * sh, y, yerr=e, fmt=mk, ms=2.4, color=color, capsize=1, elinewidth=0.6,
                        label=lab)
        pred = np.array([r["gaps"][str(q)]["pred"] for r in rows])
        ax.plot(xs(rows), pred, "-", color=BLACK, lw=0.8)
        ax.plot(xs(rows), -pred, "--", color=ORANGE, lw=0.8)
        ax.axhline(0, color=BLUE, lw=0.6)
        ax.set_xscale("log")
        ax.set_ylim(-14, 14)
        ax.set_title(f"$q={q}$", fontsize=8.5)
    for ax in axs[1]:
        ax.set_xlabel("$n$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$(q-2)\sqrt{n}\cdot\mathrm{gap}$")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[1, 2].axis("off")
    axs[1, 2].legend(h, l, loc="center", frameon=False, fontsize=7.5,
                     title="solid: predicted raw gap\ndashed: Granville's implied gap", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_residue_gaps.pdf")
    plt.close(fig)


def fig3():
    f = np.load("out/window_1000000000.npz")
    edges = np.linspace(-6.0, 6.0, 241)
    mids = 0.5 * (edges[1:] + edges[:-1])
    wd = edges[1] - edges[0]
    w = WIN[10 ** 9]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, key, mom, title, color in ((axs[0], "zo", "mom_zo", r"corrected: $z=(R-P)/\sqrt{P}$", BLUE),
                                       (axs[1], "zg", "mom_zg", r"Granville: $z_G=(R-I^*)/\sqrt{I^*}$", ORANGE)):
        h = f[f"w_hist_{key}"].astype(float)
        dens = h / (h.sum() * wd)
        ax.bar(mids, dens, width=wd, color=color, alpha=0.45, lw=0)
        mu, sd = w[mom]["mean"][0], w[mom]["sd"][0]
        g = np.exp(-0.5 * ((mids - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
        ax.plot(mids, g, "-", color=BLACK, lw=1, label=f"Normal fit, sd {sd:.3f}")
        cs = w["cramer_sd"]
        gc = np.exp(-0.5 * ((mids - mu) / cs) ** 2) / (cs * math.sqrt(2 * math.pi))
        ax.plot(mids, gc, ":", color=GREY, lw=1, label=f"Cramér width {cs:.2f}")
        ax.set_xlim(-4.5, 4.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("value")
        ax.legend(frameon=False, loc="upper left", fontsize=6.5)
        sk, ku = w[mom]["skew"][0], w[mom]["exkurt"][0]
        ax.text(0.98, 0.95, f"mean {mu:+.3f}\nskew {sk:+.3f}\nexc. kurt. {ku:+.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5)
    axs[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_z_hist_1e9.pdf")
    plt.close(fig)


def fig4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax in (a, b):
        series(ax, "resid_draft", GREY, r"truncated: $k\leq 5$, no $E(n)$, $\mathcal{L}$", shift=0.94)
        series(ax, "resid", GREEN, r"all prime powers, $\mathcal{L}$")
        if "resid_w" in RG[0]:
            series(ax, "resid_w", BLUE, r"all prime powers, $\mathcal{L}_w$", shift=1.06)
        ax.axhline(0, color=BLACK, lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
    x = np.geomspace(1e3, 2e10, 200)
    # weighted least-squares fit c n^{-1/4} to the first-draft residual over the dyadic ranges
    y, e, n = col(RG, "resid_draft"), col(RG, "resid_draft", 1), xs(RG)
    wts = 1 / e ** 2
    c = float(np.sum(wts * y * n ** -0.25) / np.sum(wts * n ** -0.5))
    R["fig4_fit_c"] = c
    for ax in (a, b):
        ax.plot(x, c * x ** -0.25, ":", color=BLACK, lw=0.9, label=rf"refit: ${c:.2f}\,n^{{-1/4}}$")
    a.set_ylabel(r"$\sqrt{n}\,\langle R_w/\mathrm{HLi}-1-\text{zeta term}\rangle$")
    a.set_ylim(-0.75, 0.45)
    a.legend(frameon=False, fontsize=6.3, loc="lower right")
    b.set_xlim(8e5, 2e10)
    b.set_ylim(-0.2, 0.12)
    a.set_title("(a) all ranges", loc="left", fontsize=8.5)
    b.set_title(r"(b) $n\geq 10^6$ and windows", loc="left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_residual.pdf")
    plt.close(fig)
    json.dump({"fig4_fit_c": c}, open("out/fig4_fit.json", "w"))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures written to", OUT)
