#!/usr/bin/env python3
"""Plot AMX utilization and L1D hit rate vs K for sweep_k_result.txt."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

K = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880, 2944, 3008, 3072, 3136, 3200, 3264, 3328, 3392, 3456, 3520, 3584, 3648, 3712, 3776, 3840, 3904, 3968, 4032, 4096]
Util = [76.00, 86.78, 88.29, 90.23, 90.60, 92.55, 92.38, 94.27, 93.49, 94.42, 94.74, 95.11, 95.21, 95.76, 95.26, 96.15, 95.58, 95.30, 96.19, 94.96, 95.16, 96.02, 87.43, 82.34, 88.78, 81.83, 79.82, 71.95, 74.33, 70.19, 73.88, 80.47, 74.12, 68.37, 74.40, 68.43, 73.75, 68.36, 74.59, 68.76, 74.86, 68.75, 74.38, 67.43, 74.04, 68.55, 74.61, 68.59, 75.14, 69.13, 75.17, 68.92, 75.17, 69.27, 70.06, 69.05, 75.27, 69.40, 75.27, 69.18, 75.38, 69.31, 75.18, 66.78]
L1D_Hit_Rate = [94.43, 90.36, 87.04, 84.14, 81.75, 80.07, 78.02, 76.23, 74.91, 73.62, 72.40, 71.29, 69.09, 68.57, 67.03, 66.73, 66.24, 64.32, 64.23, 62.43, 59.13, 62.37, 46.08, 45.92, 23.63, 16.59, 15.90, 14.79, 14.52, 14.12, 14.12, 13.39, 13.17, 13.00, 12.97, 12.52, 12.21, 12.00, 11.51, 11.62, 11.21, 11.30, 10.95, 11.03, 10.45, 10.48, 10.29, 10.78, 9.87, 10.01, 9.67, 9.71, 9.36, 9.92, 9.35, 9.22, 8.85, 8.98, 8.85, 8.84, 8.51, 8.46, 8.34, 8.63]

x = np.array(K)
y_util = np.array(Util)
y_hit = np.array(L1D_Hit_Rate)

# ---------- style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.top": True,
    "ytick.right": False,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "legend.framealpha": 0.9,
})

COLOR_UTIL = "#1f77b4"
COLOR_HIT  = "#d62728"

fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

# ----- left axis: AMX util -----
ax1.set_xlabel("$K$", fontsize=12)
ax1.set_ylabel("AMX Utilization (%)", fontsize=12, color=COLOR_UTIL)
ax1.tick_params(axis="y", labelcolor=COLOR_UTIL)

ln1, = ax1.plot(x, y_util, color=COLOR_UTIL, linewidth=1.6,
                label="AMX Utilization", zorder=3)
ax1.fill_between(x, y_util, alpha=0.07, color=COLOR_UTIL)

ax1.set_ylim(0, 115)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))

# ----- right axis: L1D hit rate -----
ax2 = ax1.twinx()
ax2.set_ylabel("L1D Hit Rate (%)", fontsize=12, color=COLOR_HIT)
ax2.tick_params(axis="y", labelcolor=COLOR_HIT, direction="in",
                length=4, width=0.8)
ax2.tick_params(axis="y", which="minor", direction="in", length=2, width=0.8)

ln2, = ax2.plot(x, y_hit, color=COLOR_HIT, linewidth=1.6,
                label="L1D Hit Rate", zorder=3)
ax2.fill_between(x, y_hit, alpha=0.07, color=COLOR_HIT)

ax2.set_ylim(0, 115)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))

# ----- x axis -----
ax1.set_xlim(0, 4200)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1024))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(256))
ax1.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda v, _: f"{int(v/1024)}K" if v >= 1024 else str(int(v)))
)
ax1.tick_params(axis="x", which="both", top=True, direction="in")

# ----- cache boundary annotation -----
boundary_k = 1408
ax1.axvline(boundary_k, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
ax1.text(boundary_k + 30, 105, "L1D capacity\nboundary",
         fontsize=8.5, color="#444444", va="top", ha="left", linespacing=1.4)

# ----- grid -----
ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, color="#cccccc", zorder=0)
ax1.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.4, color="#cccccc", zorder=0)

# ----- legend -----
lns = [ln1, ln2]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=10, loc="center right",
           bbox_to_anchor=(0.98, 0.35))

fig.tight_layout()
out = Path(__file__).parent / "sweep_k_plot.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
out_png = out.with_suffix(".png")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
print(f"Saved: {out_png}")
