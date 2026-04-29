# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

INPUT_CSV = "GOLD/VTC/VTC_resilite_pulse_resistance.csv"

bins   = [-5, 5, 15, 25, 35, 45, 52.5, 60]
labels = [0, 10, 20, 30, 40, 50, 55]
r_cols = ["R0", "R10", "R30"]

df = pd.read_csv(INPUT_CSV)
df["T_group"] = pd.cut(df["Temperature"], bins=bins, labels=labels).astype(int)
for col in r_cols:
    df[col] = df[col] * 1000

agg = (
    df.groupby(["T_group", "Zustand", "SOC"])[r_cols]
    .mean()
    .reset_index()
    .sort_values("SOC")
)

cmap   = cm.coolwarm
norm   = Normalize(vmin=min(labels), vmax=max(labels))
colors = {t: cmap(norm(t)) for t in labels}


def make_figure(zustand, output_png, soc_min=None):
    data = agg[agg["Zustand"] == zustand]
    if soc_min is not None:
        data = data[data["SOC"] >= soc_min]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})

    for ax, r_col in zip(axes, r_cols):
        for temp in labels:
            sub = data[data["T_group"] == temp].sort_values("SOC")
            if sub.empty:
                continue
            ax.plot(sub["SOC"], sub[r_col],
                    color=colors[temp], marker="o", markersize=4,
                    linewidth=1.4, label=str(temp))
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
    fig.suptitle(f"Pulse Resistance — {direction}{soc_label}", fontsize=11, fontweight="bold", y=1.01)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved {output_png}")
    plt.close()


def make_figure_combined(output_png, soc_min=None):
    def _filter(zustand):
        d = agg[agg["Zustand"] == zustand]
        return d[d["SOC"] > soc_min] if soc_min is not None else d

    cha = _filter("CHA")
    dch = _filter("DCH")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})

    for ax, r_col in zip(axes, r_cols):
        for temp in labels:
            sub_cha = cha[cha["T_group"] == temp].sort_values("SOC")
            sub_dch = dch[dch["T_group"] == temp].sort_values("SOC")
            if not sub_cha.empty:
                ax.plot(sub_cha["SOC"], sub_cha[r_col],
                        color=colors[temp], marker="o", markersize=4,
                        linewidth=1.4, linestyle="-", label=f"{temp}°C CHA")
            if not sub_dch.empty:
                ax.plot(sub_dch["SOC"], sub_dch[r_col],
                        color=colors[temp], marker="s", markersize=4,
                        linewidth=1.4, linestyle="--", label=f"{temp}°C DCH")
        ax.set_title(r_col, fontsize=11, fontweight="bold")
        ax.set_xlabel("SoC in %", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("R in mΩ", fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    color_handles = [Line2D([0], [0], color=colors[t], marker="o", markersize=4,
                            linewidth=1.4, label=str(t)) for t in labels]
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


make_figure("CHA", "GOLD/VTC/VTC_resilite_pulse_resistance_CHA.png")
make_figure("DCH", "GOLD/VTC/VTC_resilite_pulse_resistance_DCH.png")
make_figure("DCH", "GOLD/VTC/VTC_resilite_pulse_resistance_DCH_SOC40to100.png", soc_min=40)
make_figure_combined("GOLD/VTC/VTC_resilite_pulse_resistance_CHA_DCH_SOCgt40.png", soc_min=40)
