#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


LOG_PATTERN = re.compile(
    r"M N K =\s*(?P<M>\d+)\s+(?P<N>\d+)\s+(?P<K>\d+),.*?Utilization =\s*(?P<util>[\d.]+)%"
)

SERIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def parse_log(log_path: Path) -> list[list[tuple[int, float]]]:
    """Split log into runs: a new run starts when M resets to a smaller value."""
    runs: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    prev_m = -1

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            m = int(match.group("M"))
            util = float(match.group("util"))
            if m <= prev_m and current:
                runs.append(current)
                current = []
            current.append((m, util))
            prev_m = m

    if current:
        runs.append(current)

    if not runs:
        raise ValueError(f"No valid GEMM records found in {log_path}")

    return runs


def resolve_default_log_path():
    candidates = [
        Path(__file__).resolve().parent / "gemm-i8-1core.txt",
        Path("gemm-i8-1core.txt"),
        Path(__file__).resolve().parent.parent / "gemm-i8-1core.txt",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return candidates[0]


def plot_amx_util(
    series: list[tuple[list[tuple[int, float]], str]], output_path: Path, title: str
):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    all_y = []
    all_x_max = 0
    for idx, (points, label) in enumerate(series):
        x = np.array([m for m, _ in points])
        y = np.array([util for _, util in points])
        color = SERIES_COLORS[idx % len(SERIES_COLORS)]

        ax.fill_between(x, y, alpha=0.08, color=color)
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=label,
            zorder=3,
        )

        all_y.extend(y.tolist())
        all_x_max = max(all_x_max, x[-1])

    # Reference line at 50%
    ax.axhline(50, color="#888888", linewidth=1, linestyle=":", zorder=2)
    ax.text(
        all_x_max + 80, 50, "50%", va="center", ha="left", fontsize=10, color="#888888"
    )

    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    ax.set_xlabel("M = N = K", fontsize=13, labelpad=8)
    ax.set_ylabel("AMX Utilization (%)", fontsize=13, labelpad=8)
    ax.set_ylim(0, max(all_y) * 1.25)
    ax.set_xlim(left=0)

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: f"{int(v / 1000)}K" if v >= 1000 else str(int(v))
        )
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5, color="#cccccc")
    ax.tick_params(axis="both", labelsize=11, color="#cccccc")
    ax.legend(fontsize=11, framealpha=0.9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


DEFAULT_LABELS = ["GEMM (compute only)", 
                  "GEMM (online packing)",
                  "GEPP (online packing)", 
                  "GEPB(online packing)"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot AMX utilization against matrix size from GEMM log."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=resolve_default_log_path(),
        help="Path to log file (runs are auto-detected by M-value resets)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("amx-util-vs-mnk.png"),
        help="Output image path (PNG/SVG/PDF supported by matplotlib)",
    )
    parser.add_argument(
        "--title",
        default="AMX Utilization vs Matrix Size",
        help="Plot title",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated series labels (one per detected run)",
    )
    args = parser.parse_args()

    runs = parse_log(args.input)
    labels = [s.strip() for s in args.labels.split(",")]
    # Pad labels if fewer were provided than runs
    while len(labels) < len(runs):
        labels.append(f"Run {len(labels) + 1}")

    series = [(run, labels[i]) for i, run in enumerate(runs)]
    plot_amx_util(series, args.output, args.title)
    print(f"Saved plot to: {args.output} ({len(runs)} series)")


if __name__ == "__main__":
    main()
