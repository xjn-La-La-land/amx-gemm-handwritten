#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


STAGE_KEYS = ["compute", "unpackC", "packB", "packA"]   # bottom → top
STAGE_COLORS = {
    "compute":  "#5B9BD5",   # steel blue
    "unpackC":  "#70AD47",   # green
    "packB":    "#ED7D31",   # orange
    "packA":    "#FFC000",   # gold
}


def parse_log(log_path: Path) -> list[dict]:
    """Read the stages CSV (long format: M,N,K,stage,seconds,share_pct) and
    pivot into one record per (M,N,K) with each stage's share_pct."""
    by_size: dict[tuple[int, int, int], dict] = {}

    with log_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["M"]), int(row["N"]), int(row["K"]))
                stage = row["stage"]
                share = float(row["share_pct"])
            except (KeyError, ValueError):
                continue
            if stage == "total":
                continue  # total 恒为 100%，仅四个 stage 参与堆叠
            by_size.setdefault(key, {"M": key[0]})[stage] = share

    # 仅保留四个 stage 齐全的尺寸，按 M 升序
    records = [r for r in by_size.values() if all(k in r for k in STAGE_KEYS)]
    records.sort(key=lambda r: r["M"])
    if not records:
        raise ValueError(f"No valid records found in {log_path}")
    return records


def plot_breakdown(records: list[dict], output_path: Path, title: str) -> None:
    x      = np.array([r["M"] for r in records])
    layers = [np.array([r[k]  for r in records]) for k in STAGE_KEYS]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Stacked area (bottom → top: compute, unpackC, packB, packA)，四个 stage 合计 100%
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

    # 100 % reference
    ax.axhline(100, color="#555555", linewidth=1.2, linestyle="--", zorder=4)

    ax.set_title(title, fontsize=17, fontweight="bold", pad=18)
    ax.set_xlabel("M = N = K", fontsize=13, labelpad=8)
    ax.set_ylabel("Time share (%)", fontsize=13, labelpad=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

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
    # stage CSV 由 online 的 --profile-single 生成(默认名 gemm-i8-<N>core-stages.csv)
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path("gemm-i8-1core-stages.csv"),
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
