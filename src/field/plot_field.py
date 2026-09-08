"""Figures for the field-data section: what the CAP cluster selects, and what
selecting it buys.

Two figures, both print-oriented (vector PDF, no interactivity, legible in
greyscale by position and marker as well as by hue):

``timeline``  one vehicle's capacity estimates over its record, every session
              against the CAP-cluster subset, each with its own rolling-median
              ageing trend. Shows directly what the scatter numbers mean.
``fleet``     residual scatter about the ageing trend for all 20 vehicles,
              author's extraction against ours, as a paired plot.

Usage (from src/):
    python -m field.plot_field --out-dir <dir> [--vehicle N] [--base-dir D]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from field import benchmark_shiyunliu as bs
from field import io_shiyunliu

DEFAULT_BASE = "/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev"

# Validated categorical pair (dataviz slots 1-2): CVD dE 24.7, all checks pass.
C_ALL = "#8a8f98"   # every session - recessive, it is context not subject
C_CAP = "#2a78d6"   # the CAP cluster - the subject
C_AUT = "#eb6834"   # the author's reference, where two methods are compared
INK, MUTED, GRID = "#1a1c1f", "#5b6169", "#dcdfe3"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "figure.dpi": 200,
    })


def _trend(t: np.ndarray, y: np.ndarray, frac: float = 0.15):
    """Centred rolling-median ageing trend, same estimator the benchmark uses."""
    order = np.argsort(t)
    ts, ys = t[order], y[order]
    window = max(5, int(len(ys) * frac)) | 1
    tr = pd.Series(ys).rolling(window, center=True, min_periods=3).median().to_numpy()
    return ts, ys, tr


def timeline(base_dir: str, vehicle: str, our_dir: str, out_path: str) -> str:
    """One vehicle: all sessions vs the CAP cluster, each with its trend."""
    author = bs.author_capacities(base_dir, vehicle)
    ours = pd.read_csv(os.path.join(our_dir, f"{vehicle}_capacity.csv"))

    ta = pd.to_datetime(author["start_time"], errors="coerce", utc=True)
    to = pd.to_datetime(ours["CAP_start_time"], errors="coerce", utc=True)
    t0 = min(ta.min(), to.min())
    da = (ta - t0).dt.total_seconds().to_numpy() / 86400.0
    do = (to - t0).dt.total_seconds().to_numpy() / 86400.0

    _style()
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.scatter(da, author["Capacity_py_author"], s=3, c=C_ALL, alpha=0.35,
               linewidths=0, label=f"all charging sessions (n={len(author)})",
               rasterized=True)
    ax.scatter(do, ours["Capacity_py"], s=9, c=C_CAP, alpha=0.85, linewidths=0,
               marker="o", label=f"CAP cluster (n={len(ours)})")
    for d, y, c, lw in ((da, author["Capacity_py_author"].to_numpy(), C_ALL, 1.2),
                        (do, ours["Capacity_py"].to_numpy(), C_CAP, 1.6)):
        ts, _, tr = _trend(d, y)
        ax.plot(ts, tr, color=c, lw=lw, solid_capstyle="round", zorder=3)

    ra = bs.trend_residual_scatter(author["start_time"], author["Capacity_py_author"])
    ro = bs.trend_residual_scatter(ours["CAP_start_time"], ours["Capacity_py"])
    ax.set_xlabel("days since first session")
    ax.set_ylabel("capacity estimate (Ah)")
    ax.set_title(f"Vehicle {vehicle}: scatter about the ageing trend "
                 f"{ra:.2f}\\% $\\rightarrow$ {ro:.2f}\\%".replace("\\%", "%"),
                 loc="left", fontsize=9)
    lo = np.nanpercentile(author["Capacity_py_author"], 1)
    hi = np.nanpercentile(author["Capacity_py_author"], 99)
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.legend(loc="lower left", fontsize=7.5, handletextpad=0.4, borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fleet(base_dir: str, our_dir: str, out_path: str) -> str:
    """All vehicles: residual scatter, author's extraction vs the CAP cluster."""
    rows = []
    for v in io_shiyunliu.list_vehicles(base_dir):
        p = os.path.join(our_dir, f"{v}_capacity.csv")
        if not os.path.exists(p):
            continue
        a = bs.author_capacities(base_dir, v)
        o = pd.read_csv(p)
        if a.empty or o.empty:
            continue
        rows.append({
            "vehicle": int(v),
            "author": bs.trend_residual_scatter(a["start_time"], a["Capacity_py_author"]),
            "ours": bs.trend_residual_scatter(o["CAP_start_time"], o["Capacity_py"]),
        })
    df = pd.DataFrame(rows).dropna().sort_values("author").reset_index(drop=True)
    y = np.arange(len(df))

    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.hlines(y, df["ours"], df["author"], color=GRID, lw=1.6, zorder=1)
    ax.scatter(df["author"], y, s=26, c=C_AUT, zorder=3, linewidths=0,
               label="all sessions (published extraction)")
    ax.scatter(df["ours"], y, s=26, c=C_CAP, zorder=3, linewidths=0,
               label="CAP cluster (this method)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{v}" for v in df["vehicle"]], fontsize=7)
    ax.set_xlabel("scatter about the ageing trend (\\% of median capacity)".replace("\\%", "%"))
    ax.set_ylabel("vehicle")
    ratio = (df["author"] / df["ours"])
    ax.set_title(f"Improved on {int((ratio > 1).sum())}/{len(df)} vehicles "
                 f"(median {ratio.median():.2f}$\\times$)", loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Field-data figures")
    ap.add_argument("--base-dir", default=DEFAULT_BASE)
    ap.add_argument("--our-dir", default=None,
                    help="dir of <vehicle>_capacity.csv (default <base-dir>/40_capacity_monitore)")
    ap.add_argument("--vehicle", default="3", help="vehicle for the timeline figure")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    our_dir = a.our_dir or os.path.join(a.base_dir, "40_capacity_monitore")
    os.makedirs(a.out_dir, exist_ok=True)
    print(timeline(a.base_dir, a.vehicle, our_dir,
                   os.path.join(a.out_dir, "field_timeline.pdf")))
    print(fleet(a.base_dir, our_dir, os.path.join(a.out_dir, "field_fleet.pdf")))


if __name__ == "__main__":
    main()
