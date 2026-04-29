# -*- coding: utf-8 -*-
"""Plot pulse resistance (R0, R10, R30) from a pulse resistance CSV.

Usage:
    python plot_pulse_resistance.py <input_csv> <output_dir>

Example:
    python plot_pulse_resistance.py /data/GOLD/VTC/cell_pulse_resistance.csv /data/GOLD/VTC/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


BINS   = [-5, 5, 15, 25, 35, 45, 52.5, 60]
LABELS = [0, 10, 20, 30, 40, 50, 55]
R_COLS = ["R0", "R10", "R30"]


def load_and_prepare(input_csv):
    df = pd.read_csv(input_csv)
    df["T_group"] = pd.cut(df["Temperature"], bins=BINS, labels=LABELS).astype(int)
    for col in R_COLS:
        df[col] = df[col] * 1000  # Ω → mΩ
    agg = (
        df.groupby(["T_group", "Zustand", "SOC"])[R_COLS]
        .mean()
        .reset_index()
        .sort_values("SOC")
    )
    return agg


def _colors():
    cmap = cm.coolwarm
    norm = Normalize(vmin=min(LABELS), vmax=max(LABELS))
    return {t: cmap(norm(t)) for t in LABELS}


def make_figure(agg, zustand, output_png, soc_min=None):
    colors = _colors()
    data = agg[agg["Zustand"] == zustand]
    if soc_min is not None:
        data = data[data["SOC"] >= soc_min]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, r_col in zip(axes, R_COLS):
        for temp in LABELS:
            sub = data[data["T_group"] == temp].sort_values("SOC")
            if sub.empty:
                continue
            ax.plot(sub["SOC"], sub[r_col], color=colors[temp],
                    marker="o", markersize=4, linewidth=1.4, label=str(temp))
        ax.set_title(r_col, fontsize=11, fontweight="bold")
        ax.set_xlabel("SoC in %", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("R in mΩ", fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles, lbls = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, lbls, title="T in °C", fontsize=8,
                   title_fontsize=8, loc="upper right", framealpha=0.85)

    direction = "Charge" if zustand == "CHA" else "Discharge"
    soc_label = f" (SOC ≥ {soc_min}%)" if soc_min is not None else ""
    fig.suptitle(f"Pulse Resistance — {direction}{soc_label}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved {output_png}")
    plt.close()


def make_figure_combined(agg, output_png, soc_min=None):
    colors = _colors()

    def _filter(zustand):
        d = agg[agg["Zustand"] == zustand]
        return d[d["SOC"] > soc_min] if soc_min is not None else d

    cha = _filter("CHA")
    dch = _filter("DCH")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, r_col in zip(axes, R_COLS):
        for temp in LABELS:
            sub_cha = cha[cha["T_group"] == temp].sort_values("SOC")
            sub_dch = dch[dch["T_group"] == temp].sort_values("SOC")
            if not sub_cha.empty:
                ax.plot(sub_cha["SOC"], sub_cha[r_col], color=colors[temp],
                        marker="o", markersize=4, linewidth=1.4, linestyle="-")
            if not sub_dch.empty:
                ax.plot(sub_dch["SOC"], sub_dch[r_col], color=colors[temp],
                        marker="s", markersize=4, linewidth=1.4, linestyle="--")
        ax.set_title(r_col, fontsize=11, fontweight="bold")
        ax.set_xlabel("SoC in %", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("R in mΩ", fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    color_handles = [Line2D([0], [0], color=colors[t], marker="o", markersize=4,
                            linewidth=1.4, label=str(t)) for t in LABELS]
    dir_handles = [
        Line2D([0], [0], color="gray", linestyle="-",  marker="o", markersize=4, linewidth=1.4, label="Charge"),
        Line2D([0], [0], color="gray", linestyle="--", marker="s", markersize=4, linewidth=1.4, label="Discharge"),
    ]
    leg1 = axes[2].legend(handles=color_handles, title="T in °C", fontsize=8,
                          title_fontsize=8, loc="upper right", framealpha=0.85)
    axes[2].add_artist(leg1)
    axes[1].legend(handles=dir_handles, fontsize=8, loc="upper right", framealpha=0.85)

    soc_label = f" (SOC > {soc_min}%)" if soc_min is not None else ""
    fig.suptitle(f"Pulse Resistance — Charge & Discharge{soc_label}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved {output_png}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pulse resistance R0/R10/R30")
    parser.add_argument("input_csv", help="Path to pulse resistance CSV")
    parser.add_argument("output_dir", help="Directory to save PNG files")
    parser.add_argument("--soc-min", type=float, default=None,
                        help="Minimum SOC for filtered plots (default: no filter)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input_csv))[0]
    agg = load_and_prepare(args.input_csv)

    make_figure(agg, "CHA", os.path.join(args.output_dir, f"{stem}_CHA.png"))
    make_figure(agg, "DCH", os.path.join(args.output_dir, f"{stem}_DCH.png"))
    if args.soc_min is not None:
        make_figure(agg, "DCH", os.path.join(args.output_dir, f"{stem}_DCH_SOCgt{int(args.soc_min)}.png"),
                    soc_min=args.soc_min)
        make_figure_combined(agg, os.path.join(args.output_dir, f"{stem}_CHA_DCH_SOCgt{int(args.soc_min)}.png"),
                             soc_min=args.soc_min)
    else:
        make_figure_combined(agg, os.path.join(args.output_dir, f"{stem}_CHA_DCH.png"))
