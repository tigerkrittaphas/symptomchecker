"""Grouped bar chart of Top-K diagnostic accuracy with 95% Wilson CIs (manuscript Fig).

x-axis = Top-K level (1..5); within each group, one bar per symptom checker with a 95%
Wilson confidence-interval whisker. Colours use the validated categorical palette in a
fixed per-entity order (CVD-safe: worst adjacent deltaE 24.2). GPT-4o is the focal series
(palette slot 1, blue). The accompanying Table 1 is the table view that satisfies the
contrast-relief rule for the lighter fills.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import load_checked_response, topk_success_wide, wilson_ci

# Fixed left-to-right order = palette slot order (keeps adjacent bars CVD-optimal).
APPS = ["GPT-4o", "Ada", "Symptomate", "DoctorAtHome", "Agnos", "Buoy"]
COLORS = {
    "GPT-4o": "#2a78d6",       # slot 1 blue  (focal)
    "Ada": "#1baf7a",          # slot 2 aqua
    "Symptomate": "#eda100",   # slot 3 yellow
    "DoctorAtHome": "#008300", # slot 4 green
    "Agnos": "#4a3aa7",        # slot 5 violet
    "Buoy": "#e34948",         # slot 6 red
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
WHISKER = "#52514e"


def darken(hex_color, f=0.72):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def main():
    cr = load_checked_response()
    ks = [1, 2, 3, 4, 5]
    # acc[app][k] and asymmetric CI errors
    acc = {a: [] for a in APPS}
    lo = {a: [] for a in APPS}
    hi = {a: [] for a in APPS}
    for k in ks:
        wide = topk_success_wide(cr, k, APPS)
        for a in APPS:
            s = wide[a].dropna().astype(int)
            n, c = len(s), int(s.sum())
            p = 100 * c / n
            cl, cu = wilson_ci(c, n)
            acc[a].append(p); lo[a].append(p - cl); hi[a].append(cu - p)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.labelcolor": INK,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.4))  # <= PLOS max width (7.5 in at 300 dpi = 2250 px)
    x = np.arange(len(ks))
    nb = len(APPS)
    total_w = 0.82
    bw = total_w / nb
    for i, a in enumerate(APPS):
        offs = -total_w / 2 + bw * (i + 0.5)
        ax.bar(x + offs, acc[a], width=bw * 0.92, color=COLORS[a],
               edgecolor=darken(COLORS[a]), linewidth=0.6,
               label=a, zorder=3)
        ax.errorbar(x + offs, acc[a], yerr=[lo[a], hi[a]], fmt="none",
                    ecolor=WHISKER, elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Diagnostic accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    ax.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handlelength=1.1, handleheight=1.1, columnspacing=1.3, fontsize=10)
    fig.tight_layout()

    out_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"),
        "/Users/tiger/Projects/symptomchecker/symptom_checker_manuscript/images",
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, "fig_topk_accuracy_ci")
        # PDF: vector, used by \includegraphics in the LaTeX draft (pdflatex can't embed TIFF).
        fig.savefig(base + ".pdf", bbox_inches="tight")
        # PNG: quick preview.
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        # TIFF: PLOS submission asset — 300 dpi, LZW compression.
        fig.savefig(base + ".tiff", dpi=300, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
    print("Saved fig_topk_accuracy_ci.{pdf,png,tiff} to:", *out_dirs, sep="\n  ")


if __name__ == "__main__":
    main()
