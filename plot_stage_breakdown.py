#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MNK_PAT    = re.compile(r"M N K =\s*(\d+)\s+\d+\s+\d+")
STAGE_PAT  = re.compile(r"(\w+)\s+([\d.]+) s\s+\(\s*([\d.]+)%\)")

STAGE_KEYS = ["compute", "unpackC", "packB", "packA"]   # bottom → top
STAGE_COLORS = {
    "compute":  "#5B9BD5",   # steel blue
    "unpackC":  "#70AD47",   # green
    "packB":    "#ED7D31",   # orange
    "packA":    "#FFC000",   # gold
}
ONLINE_COLOR  = "#C00000"   # dark red
OVERHEAD_COLOR = "#FF9999"  # light red for overhead fill


def parse_log(log_path: Path) -> list[dict]:
    records: list[dict] = []
    cur: dict = {}

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            m = MNK_PAT.search(line)
            if m:
                cur = {"M": int(m.group(1))}
                continue

            s = STAGE_PAT.search(line)
            if s and cur:
                name, pct = s.group(1), float(s.group(3))
                cur[name] = pct
                continue

            if line.startswith("---") and cur.get("M"):
                needed = STAGE_KEYS + ["online_packing"]
                if all(k in cur for k in needed):
                    records.append(cur)
                cur = {}

    if not records:
        raise ValueError(f"No valid records found in {log_path}")
    return records


def plot_breakdown(records: list[dict], output_path: Path, title: str) -> None:
    x      = np.array([r["M"]              for r in records])
    layers = [np.array([r[k]               for r in records]) for k in STAGE_KEYS]
    online = np.array([r["online_packing"] for r in records])

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Stacked area (bottom → top: compute, unpackC, packB, packA)
    polys = ax.stackplot(
        x, *layers,
        labels=STAGE_KEYS,
        colors=[STAGE_COLORS[k] for k in STAGE_KEYS],
        alpha=0.88,
        zorder=2,
    )
    # Thin white edge between layers for visual separation
    for poly in polys:
        poly.set_edgecolor("white")
        poly.set_linewidth(0.6)

    # Overhead fill: gap between online_packing and 100 %
    ax.fill_between(x, 100, online, where=(online >= 100),
                    color=OVERHEAD_COLOR, alpha=0.35, zorder=3,
                    label="packing overhead")
    ax.fill_between(x, online, 100, where=(online < 100),
                    color=OVERHEAD_COLOR, alpha=0.35, zorder=3)

    # Online packing line — prominent, no markers (63 pts is dense enough)
    ax.plot(x, online, color=ONLINE_COLOR, linewidth=2.2,
            label="online packing / stage total", zorder=5)

    # 100 % reference
    ax.axhline(100, color="#555555", linewidth=1.2, linestyle="--", zorder=4)
    ax.text(x[-1] + 80, 100, "100%", va="center", ha="left",
            fontsize=10, color="#555555")

    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    ax.set_xlabel("M = N = K", fontsize=13, labelpad=8)
    ax.set_ylabel("Time share (%)", fontsize=13, labelpad=8)
    ax.set_xlim(left=0)
    y_top = max(online.max(), 100) * 1.18
    ax.set_ylim(0, y_top)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v/1000)}K" if v >= 1000 else str(int(v)))
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))

    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.7,
            alpha=0.5, color="#cccccc", zorder=1)

    ax.tick_params(axis="both", labelsize=11, color="#cccccc")

    # Legend inside lower-right corner (no data fluctuation there)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              fontsize=11, framealpha=0.92,
              loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-stage time breakdown from GEMM profile log."
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path(__file__).resolve().parent / "gemm-i8-1core-time-log.txt",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("stage-breakdown.png"),
    )
    parser.add_argument(
        "--title",
        default="GEMM Stage Time Breakdown vs Matrix Size",
    )
    args = parser.parse_args()

    records = parse_log(args.input)
    plot_breakdown(records, args.output, args.title)


if __name__ == "__main__":
    main()
