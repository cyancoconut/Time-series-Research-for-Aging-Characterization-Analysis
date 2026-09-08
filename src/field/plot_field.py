"""Figures for the field-data section: what the CAP cluster selects, and what
selecting it buys.

Four figures, all print-oriented (vector PDF, no interactivity, legible in
greyscale by position and marker as well as by hue):

``timeline``  one vehicle's capacity estimates over its record, every session
              against the CAP-cluster subset, each with its own rolling-median
              ageing trend. Shows directly what the scatter numbers mean.
``fleet``     residual scatter about the ageing trend for all 20 vehicles,
              author's extraction against ours, as a paired plot.
``medians``     per-vehicle median capacity, all charging sessions against the CAP
              cluster, as paired dumbbells - the comparison against doing no
              selection at all.
``fleet_timelines``
              all 20 vehicles as small multiples: the published monthly-median
              curve against the CAP cluster this method selects, each
              normalised to its own first level so the panels share an axis.

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


def _residual_scatter(times, values, frac: float = 0.15):
    """Scatter of capacity estimates about their own ageing trend, in percent.

    Defined here rather than imported so these figures stay independent of the
    benchmark module, which carries the same measure on its own branch. A field
    cell fades over the record, so the spread of a capacity timeline mixes
    degradation with estimator noise; detrending with a centred rolling median
    leaves the noise, summarised by its median absolute deviation and normalised
    by median capacity.
    """
    t = pd.to_datetime(pd.Series(times), errors="coerce", utc=True)
    y = pd.Series(values, dtype=float).to_numpy()
    ok = t.notna().to_numpy() & np.isfinite(y)
    if ok.sum() < 8:
        return None
    order = np.argsort(t[ok].astype("int64").to_numpy())
    y = y[ok][order]
    window = max(5, int(len(y) * frac)) | 1
    trend = pd.Series(y).rolling(window, center=True, min_periods=3).median().to_numpy()
    resid = y - trend
    resid = resid[np.isfinite(resid)]
    med = np.median(y)
    if resid.size == 0 or not np.isfinite(med) or med == 0:
        return None
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(1.4826 * mad / med * 100)


def _trend(t: np.ndarray, y: np.ndarray, frac: float = 0.15):
    """Centred rolling-median ageing trend, same estimator the benchmark uses."""
    order = np.argsort(t)
    ts, ys = t[order], y[order]
    window = max(5, int(len(ys) * frac)) | 1
    tr = pd.Series(ys).rolling(window, center=True, min_periods=3).median().to_numpy()
    return ts, ys, tr


def _monthly_median(days: np.ndarray, y: np.ndarray, min_n: int = 3):
    """Median capacity per 30-day bin — the dataset author's own aggregation.

    Deng et al. reduce the per-session estimates to a monthly mean or median
    ("the mean values are almost equal to the median values, indicating the
    calculated capacity points during a month are symmetrically distributed")
    and it is that curve, not the session cloud, that carries the degradation
    trend.

    Equal weight per month is also what makes a comparison between two
    populations honest here. Pooling estimates across the whole record instead
    weights each population by *when* it charged: the CAP cluster puts 71% of
    its sessions in the first half of the record against 42% for all sessions,
    so on cells that fade ~13% a pooled median flatters the CAP cluster by
    ~4% — an artifact of the time distribution, not a difference in what is
    being measured. Binned by month the two agree to ~0.1%.
    """
    b = pd.DataFrame({"bin": (np.asarray(days) // 30).astype(int), "y": np.asarray(y, float)})
    g = b.groupby("bin")["y"].agg(["median", "size"])
    g = g[g["size"] >= min_n]
    return g.index.to_numpy() * 30.0 + 15.0, g["median"].to_numpy()


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

    ra = _residual_scatter(author["start_time"], author["Capacity_py_author"])
    ro = _residual_scatter(ours["CAP_start_time"], ours["Capacity_py"])
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
            "author": _residual_scatter(a["start_time"], a["Capacity_py_author"]),
            "ours": _residual_scatter(o["CAP_start_time"], o["Capacity_py"]),
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


def fleet_timelines(base_dir: str, our_dir: str, out_path: str,
                    ncols: int = 5, show_sessions: bool = False) -> str:
    """All 20 vehicles: ageing trend from all sessions vs from the CAP cluster.

    One line per method per panel: the author's monthly median against the CAP
    cluster at its native per-session resolution. The comparison the figure is
    for: whether selecting changes the *shape* of the fade curve, not
    just its scatter. ``show_sessions=True`` puts the individual estimates back
    underneath as faint points — useful for one-off inspection, but at 2700
    sessions per vehicle they bury the lines they are meant to support.

    Capacity is normalised to each vehicle's own initial level so the panels
    share a y axis and the fade is comparable; the absolute pack capacities
    differ between vehicles and would otherwise force 20 different scales.
    """
    vehicles = io_shiyunliu.list_vehicles(base_dir)
    data = []
    for v in vehicles:
        p_ours = os.path.join(our_dir, f"{v}_capacity.csv")
        if not os.path.exists(p_ours):
            continue
        a = bs.author_capacities(base_dir, v)
        o = pd.read_csv(p_ours)
        if a.empty or o.empty:
            continue
        ta = pd.to_datetime(a["start_time"], errors="coerce", utc=True)
        to = pd.to_datetime(o["CAP_start_time"], errors="coerce", utc=True)
        t0 = ta.min()
        # Reference level: median of the CAP cluster's first 20 estimates - the
        # cleanest early measurement available, so 100% means "as first seen".
        ref = float(o["Capacity_py"].head(20).median())
        data.append({
            "v": int(v),
            "da": (ta - t0).dt.total_seconds().to_numpy() / 86400.0,
            "ya": a["Capacity_py_author"].to_numpy() / ref * 100.0,
            "do": (to - t0).dt.total_seconds().to_numpy() / 86400.0,
            "yo": o["Capacity_py"].to_numpy() / ref * 100.0,
            "span": (ta.max() - t0).total_seconds() / 86400.0,
        })
    data.sort(key=lambda d: d["v"])
    nrows = int(np.ceil(len(data) / ncols))

    _style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.4, 1.85 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, data):
        if show_sessions:
            ax.scatter(d["da"], d["ya"], s=1.4, c=C_ALL, alpha=0.22, linewidths=0,
                       rasterized=True)
            ax.scatter(d["do"], d["yo"], s=3.2, c=C_CAP, alpha=0.55, linewidths=0,
                       rasterized=True)
        # Author's curve: monthly median, their published construction — the
        # binning is what makes their session cloud readable.
        ma_x, ma_y = _monthly_median(d["da"], d["ya"])
        ax.plot(ma_x, ma_y, color=C_AUT, lw=1.2, zorder=3, solid_capstyle="round")
        # Ours at native resolution. Selection already did the noise reduction
        # that their monthly binning does, so binning ours too would smooth
        # twice and discard ~14 estimates per month for no gain.
        ts_o, _, tr_o = _trend(d["do"], d["yo"])
        ax.plot(ts_o, tr_o, color=C_CAP, lw=1.2, zorder=4, solid_capstyle="round")
        # where the selected sessions stop, against where the record stops
        ax.axvline(d["do"].max(), color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.set_title(f"#{d['v']}", loc="left", fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
    for ax in axes[len(data):]:
        ax.set_visible(False)
    axes[0].set_ylim(78, 104)
    for i, ax in enumerate(axes[:len(data)]):
        if i % ncols == 0:
            ax.set_ylabel("capacity (\\% of first)".replace("\\%", "%"), fontsize=7.5)
        if i >= len(data) - ncols:
            ax.set_xlabel("days", fontsize=7.5)

    handles = [
        plt.Line2D([], [], color=C_AUT, lw=1.6, label="all charging sessions"),
        plt.Line2D([], [], color=C_CAP, lw=1.6, label="CAP cluster (this method)"),
        plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)),
                   label="last selected session"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.005), handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def medians(base_dir: str, out_path: str) -> str:
    """All vehicles: median capacity from every session vs from the CAP cluster.

    The comparison against doing no selection at all. Both sides are the median
    of the per-month medians, following the dataset author's aggregation, so
    each month counts once.

    That equal weighting is the point. Pooling every estimate across the record
    instead makes all-sessions look ~4.3% low, but that is the time
    distribution talking: the CAP cluster puts 71% of its sessions in the first
    half of the record against 42% for all sessions, and these cells fade ~13%,
    so a pooled median simply weights the CAP cluster toward its fresher months.
    Weighted by month the two populations agree to ~0.1%, which is the honest
    result — selection buys precision, not a different capacity.

    Paired dumbbells rather than two sorted series: the quantity of interest is
    the gap within a vehicle, not the fleet ranking.
    """
    from field import cluster_sessions as cs, sessions as fs
    from field.extract_capacity import _trapz_abs_Ah

    rows = []
    for v in io_shiyunliu.list_vehicles(base_dir):
        df = fs.split_sessions(io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(base_dir, v)))
        feats = fs.session_features(df, vehicle=v)
        if feats.empty:
            continue
        labeled = cs.cluster_sessions(feats)
        cap_label = cs.pick_cap_cluster(labeled)
        cap_ids = set(labeled.loc[labeled["cluster_label"] == cap_label, "session_id"]) \
            if cap_label is not None else set()
        caps = []
        for sid, sub in df.groupby("session_id"):
            sf = labeled[labeled["session_id"] == sid]
            if sf.empty:
                continue
            dsoc = float(sf["dSOC"].iloc[0])
            ah = _trapz_abs_Ah(sub["Time"], sub["Current"])
            if not (np.isfinite(dsoc) and abs(dsoc) > 1e-6 and np.isfinite(ah)):
                continue
            caps.append((sf["start_time"].iloc[0], abs(ah) / (abs(dsoc) / 100.0),
                         sid in cap_ids))
        c = pd.DataFrame(caps, columns=["t", "cap", "is_cap"])
        if c.empty or not c["is_cap"].any():
            continue
        t0 = c["t"].min()
        c["days"] = (c["t"] - t0).dt.total_seconds() / 86400.0
        k = c[c["is_cap"]]
        _, my_all = _monthly_median(c["days"].to_numpy(), c["cap"].to_numpy())
        _, my_ours = _monthly_median(k["days"].to_numpy(), k["cap"].to_numpy())
        if my_all.size == 0 or my_ours.size == 0:
            continue
        rows.append({"vehicle": int(v),
                     "all": float(np.median(my_all)),
                     "ours": float(np.median(my_ours))})

    df = pd.DataFrame(rows).sort_values("vehicle").reset_index(drop=True)
    y = np.arange(len(df))

    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.hlines(y, df["all"], df["ours"], color=GRID, lw=1.6, zorder=1)
    ax.scatter(df["all"], y, s=26, c=C_AUT, zorder=3, linewidths=0,
               label="all charging sessions")
    ax.scatter(df["ours"], y, s=26, c=C_CAP, zorder=3, linewidths=0,
               label="CAP cluster (this method)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{v}" for v in df["vehicle"]], fontsize=7)
    ax.set_xlabel("median capacity estimate (Ah)")
    ax.set_ylabel("vehicle")
    ax.set_ylim(-0.8, len(df) + 1.4)   # blank row up top so the legend clears the marks
    gap = 100 * (df["all"] - df["ours"]) / df["ours"]
    ax.set_title("Median of monthly medians: the two agree to "
                 f"{abs(gap.median()):.1f}%", loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=7.5, handletextpad=0.4)
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
    print(fleet_timelines(a.base_dir, our_dir,
                          os.path.join(a.out_dir, "field_fleet_timelines.pdf")))
    print(medians(a.base_dir, os.path.join(a.out_dir, "field_medians.pdf")))


if __name__ == "__main__":
    main()
