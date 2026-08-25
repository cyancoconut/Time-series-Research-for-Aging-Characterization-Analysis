"""Plot quasi-OCV (qOCV) curves and their differential capacity from the exports.

Reads the per-check-up qOCV parquets written by ``output/export_qocv.py``
(``30_export_qocv/<cell>/<cell>_qocv_{cha,dch}_BM<n>_<SOH>SOH.parquet``) — slow
(~C/20) charge / discharge sweeps — and produces three views:

1. **qOCV curve** (``plot_qocv``) — terminal voltage vs throughput-normalised SOC
   for one matched charge+discharge pair, plus their mean (≈ the true
   thermodynamic OCV). The charge branch sits above discharge by the qOCV
   hysteresis (path dependence + the small residual C/20 overpotential).

2. **Differential capacity** (``plot_dqdv``) — dQ/dV for the same pair. A plateau
   in V-vs-Q (a two-phase electrode transition, where lots of charge is absorbed
   at nearly constant potential) becomes a **peak** in dQ/dV: dQ/dV = ΔQ/ΔV, so a
   small ΔV carrying a large ΔQ spikes. The reciprocal dV/dQ shows the same
   plateau as a *valley*. The peak pattern is the graphite-staging fingerprint.

3. **dQ/dV vs SOH** (``plot_dqdv_vs_soh``) — the differential overlaid across
   check-ups, coloured fresh→aged. Peaks that **shift to lower voltage** track
   loss of lithium inventory (LLI, electrodes going out of registration); peaks
   that **shrink** track loss of active material (LAM). Usually far more
   diagnostic than the raw qOCV drift or the SOH number alone.

Capacity is trapezoidally integrated from the logged current; the SOC axis is
each sweep's own throughput normalised 0–100 %, so charge and discharge align.

Standalone analysis utility — does not touch the pipeline. Run from ``src/``::

    # a whole cell folder: auto-picks a matched pair + overlays dQ/dV vs SOH
    python -m analysis.qocv_curve /path/to/30_export_qocv/<cell>
    # an explicit charge+discharge pair
    python -m analysis.qocv_curve <cell>_qocv_cha_BM12_90.3SOH.parquet \
                                  <cell>_qocv_dch_BM12_90.3SOH.parquet
    # pick a specific check-up by SOH (nearest match), set overlay count
    python -m analysis.qocv_curve <folder> --soh 90.3 --n-overlay 6
"""

import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

NOM_CAPACITY_DEFAULT = 3.0  # Ah (Sony/Murata US18650VTC6)

# Differential-capacity smoothing. The C/20 voltage is quantised (~0.1 mV steps),
# so raw dQ/dV is unusably noisy: interpolate Q onto a uniform V grid at ``DV`` and
# Savitzky-Golay-smooth before differentiating.
DV = 0.004            # V grid spacing for dQ/dV
SG_WINDOW = 51        # Savitzky-Golay window (odd)
SG_POLY = 3


# ---------------------------------------------------------------------------
# Loading / integration
# ---------------------------------------------------------------------------
# These four live in util.io_qocv so util.soc_from_qocv (the single source of
# SOC) can use them without util depending on analysis. Re-exported here so
# this module's own CLI and existing `qocv_curve.load_sweep`-style callers are
# unaffected by the move.
from util.io_qocv import _parse_soh, _parse_bm, load_sweep, soc_axis  # noqa: F401,E402


def differential_capacity(v, q, dv=DV, window=SG_WINDOW, poly=SG_POLY):
    """dQ/dV on a uniform voltage grid. Returns ``(v_grid, dqdv, soc_grid)``.

    Q is interpolated onto a uniform V grid, Savitzky-Golay smoothed, then
    differentiated — necessary because the raw C/20 voltage is quantised. ``soc``
    is the normalised charge state at each grid point (for a vs-SOC view).
    """
    o = np.argsort(v)
    v, q = v[o], q[o]
    v, idx = np.unique(v, return_index=True)     # strictly increasing for interp
    q = q[idx]
    if v[-1] - v[0] < 3 * dv:
        raise ValueError("voltage span too small for a dQ/dV grid")
    grid = np.arange(v.min() + 0.01, v.max() - 0.01, dv)
    qg = np.interp(grid, v, q)
    win = min(window, len(qg) - (1 - len(qg) % 2))   # keep odd & <= len
    if win >= 5:
        qg = savgol_filter(qg, win, poly)
    dqdv = np.gradient(qg, dv)
    return grid, dqdv, soc_axis(qg)


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------
def find_pairs(folder):
    """Map ``BM_Programm -> {'cha': path, 'dch': path, 'soh': value}`` in a folder."""
    pairs = {}
    for f in glob.glob(os.path.join(folder, "*_qocv_*_BM*.parquet")):
        base = os.path.basename(f)
        bm = _parse_bm(base)
        if bm is None:
            continue
        kind = "cha" if "_qocv_cha_" in base else "dch" if "_qocv_dch_" in base else None
        if kind is None:
            continue
        pairs.setdefault(bm, {})["soh"] = _parse_soh(base)
        pairs[bm][kind] = f
    return {bm: d for bm, d in pairs.items() if "cha" in d and "dch" in d}


def _pick_pair(pairs, soh=None):
    """Choose one ``(bm, dict)`` — nearest to ``soh`` if given, else freshest."""
    if soh is not None:
        bm = min(pairs, key=lambda b: abs(pairs[b]["soh"] - soh))
    else:
        bm = max(pairs, key=lambda b: (np.nan_to_num(pairs[b]["soh"], nan=-1)))
    return bm, pairs[bm]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_qocv(cha_path, dch_path, out_png, nom_capacity=NOM_CAPACITY_DEFAULT):
    """qOCV: charge + discharge terminal voltage vs SOC, plus their mean (≈ OCV)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vc, qc = load_sweep(cha_path, discharge=False)
    vd, qd = load_sweep(dch_path, discharge=True)
    soc_c, soc_d = soc_axis(qc), soc_axis(qd)
    crate = 1.0 / nom_capacity  # nominal C-rate label context only

    grid = np.linspace(2, 98, 400)
    vc_i = np.interp(grid, soc_c, vc)
    vd_i = np.interp(grid, soc_d, vd)
    hyst = float(np.mean((vc_i - vd_i)[(grid >= 20) & (grid <= 80)]) * 1000)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(soc_c, vc, color="C3", lw=1.6, label=f"charge  ({qc.max():.2f} Ah)")
    ax.plot(soc_d, vd, color="C0", lw=1.6, label=f"discharge  ({qd.max():.2f} Ah)")
    ax.plot(grid, (vc_i + vd_i) / 2, color="0.35", lw=1.0, ls="--", label="mean")
    ax.set_xlabel("SOC (%)  [throughput-normalised]")
    ax.set_ylabel("Voltage (V)")
    soh = _parse_soh(os.path.basename(cha_path))
    ax.set_title(f"qOCV — {os.path.basename(os.path.dirname(cha_path))} "
                 f"@ {soh:.1f}% SOH  (hyst ≈ {hyst:.0f} mV over 20–80%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    logging.info("qOCV plot -> %s", out_png)


def plot_dqdv(cha_path, dch_path, out_png):
    """Differential capacity dQ/dV for one pair — vs voltage and vs SOC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vc, qc = load_sweep(cha_path, discharge=False)
    vd, qd = load_sweep(dch_path, discharge=True)
    gc, dqc, socc = differential_capacity(vc, qc)
    gd, dqd, socd = differential_capacity(vd, qd)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(gc, dqc, "C3", lw=1.4, label="charge")
    ax[0].plot(gd, dqd, "C0", lw=1.4, label="discharge")
    ax[0].set_xlabel("Voltage (V)")
    ax[0].set_ylabel("dQ/dV (Ah/V)")
    ax[0].set_title("Incremental capacity (peaks = plateaus)")
    ax[1].plot(socc, dqc, "C3", lw=1.4, label="charge")
    ax[1].plot(socd, dqd, "C0", lw=1.4, label="discharge")
    ax[1].set_xlabel("SOC (%)")
    ax[1].set_ylabel("dQ/dV (Ah/V)")
    ax[1].set_title("dQ/dV vs SOC")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    soh = _parse_soh(os.path.basename(cha_path))
    fig.suptitle(f"qOCV differential — {os.path.basename(os.path.dirname(cha_path))} "
                 f"@ {soh:.1f}% SOH")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    logging.info("dQ/dV plot -> %s", out_png)

    # report charge-branch peak positions (the plateau locations)
    pk, _ = find_peaks(dqc, prominence=0.5)
    if len(pk):
        logging.info(
            "charge dQ/dV peaks: %s",
            ", ".join(f"{gc[p]:.3f} V (~{socc[p]:.0f}% SOC)" for p in pk),
        )


def plot_dqdv_vs_soh(folder, out_png, n_overlay=6):
    """Overlay dQ/dV across ``n_overlay`` SOH check-ups (charge + discharge)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcol

    pairs = find_pairs(folder)
    if not pairs:
        logging.warning("no matched cha/dch qOCV pairs in %s", folder)
        return
    # order by SOH (fresh -> aged) and pick an even spread
    bms = sorted(pairs, key=lambda b: np.nan_to_num(pairs[b]["soh"], nan=-1), reverse=True)
    idx = np.unique(np.linspace(0, len(bms) - 1, min(n_overlay, len(bms))).round().astype(int))
    bms = [bms[i] for i in idx]
    sohs = [pairs[b]["soh"] for b in bms]
    norm = mcol.Normalize(min(sohs), max(sohs))
    cmap = cm.viridis

    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    for j, (kind, dis) in enumerate([("cha", False), ("dch", True)]):
        for b, s in zip(bms, sohs):
            v, q = load_sweep(pairs[b][kind], discharge=dis)
            try:
                g, dqdv, _ = differential_capacity(v, q)
            except ValueError:
                continue
            ax[j].plot(g, dqdv, color=cmap(norm(s)), lw=1.4, label=f"{s:.1f}%")
        ax[j].set_xlabel("Voltage (V)")
        ax[j].set_ylabel("dQ/dV (Ah/V)")
        ax[j].set_title("charge" if kind == "cha" else "discharge")
        ax[j].grid(alpha=0.3)
        ax[j].legend(title="SOH", fontsize=8)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax[j], label="SOH (%)")
    fig.suptitle(f"dQ/dV vs SOH — {os.path.basename(os.path.normpath(folder))} "
                 "(C/20 qOCV)  [fresh=yellow → aged=purple]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    logging.info("dQ/dV vs SOH plot -> %s", out_png)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot qOCV curves + differential capacity.")
    ap.add_argument("inputs", nargs="+",
                    help="a folder of qOCV exports, or an explicit cha + dch parquet pair")
    ap.add_argument("--soh", type=float, default=None,
                    help="pick the check-up nearest this SOH (folder mode; default freshest)")
    ap.add_argument("--n-overlay", type=int, default=6,
                    help="number of SOH check-ups in the dQ/dV-vs-SOH overlay (folder mode)")
    ap.add_argument("--nom-capacity", type=float, default=NOM_CAPACITY_DEFAULT, help="Ah")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="output directory for PNGs (default: alongside the inputs)")
    args = ap.parse_args()

    # explicit cha + dch pair
    if len(args.inputs) == 2 and all(os.path.isfile(p) for p in args.inputs):
        cha = next((p for p in args.inputs if "_qocv_cha_" in os.path.basename(p)), args.inputs[0])
        dch = next((p for p in args.inputs if "_qocv_dch_" in os.path.basename(p)), args.inputs[1])
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(cha))
        stem = re.sub(r"_qocv_cha_", "_qocv_", os.path.splitext(os.path.basename(cha))[0])
        plot_qocv(cha, dch, os.path.join(out_dir, f"{stem}_curve.png"), args.nom_capacity)
        plot_dqdv(cha, dch, os.path.join(out_dir, f"{stem}_dqdv.png"))
        return

    # folder mode
    folder = args.inputs[0]
    if not os.path.isdir(folder):
        ap.error("give a folder, or exactly two files (a cha + dch parquet pair)")
    pairs = find_pairs(folder)
    if not pairs:
        logging.warning("no matched cha/dch qOCV pairs in %s", folder)
        return
    out_dir = args.out_dir or folder
    bm, pair = _pick_pair(pairs, args.soh)
    logging.info("single-pair plots from BM%s (SOH %.1f%%)", bm, pair["soh"])
    tag = f"BM{bm}_{pair['soh']:.1f}SOH"
    plot_qocv(pair["cha"], pair["dch"], os.path.join(out_dir, f"qocv_curve_{tag}.png"),
              args.nom_capacity)
    plot_dqdv(pair["cha"], pair["dch"], os.path.join(out_dir, f"qocv_dqdv_{tag}.png"))
    plot_dqdv_vs_soh(folder, os.path.join(out_dir, "qocv_dqdv_vs_SOH.png"), args.n_overlay)


if __name__ == "__main__":
    main()
