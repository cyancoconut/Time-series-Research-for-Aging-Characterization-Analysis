"""Prototype — voltage-window ICA capacity proxy vs. ΔSOC coulomb counting.

Motivation: both the dataset author's method (Deng et al.) and our HDBSCAN
pipeline estimate capacity as ``∫I dt / ΔSOC``. That divides by the *BMS-reported*
SOC change, so both inherit the BMS SOC-estimation error — the suspected
dominant noise source in the per-session capacity scatter.

This prototype tests an alternative that never divides by SOC. It works in
*voltage* space: for each charging session it integrates the charge throughput
``ΔQ = ∫I dt`` accumulated while the cell voltage rises across a fixed window
``[V_lo, V_hi]`` (a coarse incremental-capacity / DVA feature). ``ΔQ`` in a fixed
voltage window shrinks as the cell ages, so it is a SOH proxy decoupled from the
BMS SOC estimate.

Because a fixed-voltage-window ΔQ is C-rate sensitive (ohmic drop shifts the
voltage trajectory), only sessions whose in-window mean current sits in a
controlled band are kept.

The harness then builds three SOH timelines for one vehicle — (A) Deng ΔSOC,
(B) our HDBSCAN CAP timeline, (C) this ICA ΔQ — fits a smooth fade trend to each
and reports the **detrended residual scatter** (residual CV %). Lower residual CV
= a cleaner SOH tracker once real aging drift is removed. This is the
apples-to-apples test the pooled-CV benchmark could not give.

Usage::

    python -m field.ica_prototype [base_dir] [--vehicle 1] \
        [--v-lo 3.80] [--v-hi 3.90] [--i-band 0.15] [--plot out.png]
"""
from __future__ import annotations

import os
import argparse
import logging

import numpy as np
import pandas as pd

from field import io_shiyunliu, sessions, benchmark_shiyunliu


# Default cell-voltage ICA window (V). 3.80–3.90 sits in the CC region for this
# NMC pack (cell V spans ~3.5–4.26) and is crossed by ~960 of vehicle #1's
# sessions — well below the CV knee, so dQ/dV is well-behaved.
DEFAULT_V_LO = 3.80
DEFAULT_V_HI = 3.90
DEFAULT_I_BAND = 0.15          # keep sessions within ±15 % of the median in-window current
DEFAULT_CELL_V_COL = "Cell_V_max"


def _cumulative_Ah(time: pd.Series, current: pd.Series) -> np.ndarray:
    """Cumulative trapezoidal ∫|I|dt (Ah), one entry per row (first = 0)."""
    t = time.to_numpy()
    i = np.abs(current.to_numpy(dtype=float))
    dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
    step = 0.5 * (i[:-1] + i[1:]) * dt_s / 3600.0
    return np.concatenate([[0.0], np.cumsum(step)])


def ica_delta_q(
    sub: pd.DataFrame,
    *,
    v_lo: float,
    v_hi: float,
    cell_v_col: str = DEFAULT_CELL_V_COL,
) -> tuple[float, float]:
    """ΔQ (Ah) accumulated while cell voltage rises from ``v_lo`` to ``v_hi``,
    and the mean charge current over that window.

    Returns ``(nan, nan)`` if the session does not cross the window monotonically
    on the rising (charge) leg.
    """
    v = sub[cell_v_col].to_numpy(dtype=float)
    cur = sub["Current"].to_numpy(dtype=float)
    if len(v) < 5 or np.isnan(v).all():
        return float("nan"), float("nan")

    # Restrict to the rising leg: rows up to peak cell voltage. Charging only.
    peak = int(np.nanargmax(v))
    if peak < 2:
        return float("nan"), float("nan")
    t = sub["Time"].iloc[: peak + 1]
    vv = v[: peak + 1]
    cc = cur[: peak + 1]
    if np.nanmin(vv) > v_lo or np.nanmax(vv) < v_hi:
        return float("nan"), float("nan")

    q = _cumulative_Ah(t, sub["Current"].iloc[: peak + 1])

    # Enforce a monotone voltage axis for interpolation (running max), so np.interp
    # maps each window edge to a single charge value on the rising leg.
    vmono = np.maximum.accumulate(np.nan_to_num(vv, nan=-np.inf))
    if vmono[0] > v_lo or vmono[-1] < v_hi:
        return float("nan"), float("nan")
    q_lo = float(np.interp(v_lo, vmono, q))
    q_hi = float(np.interp(v_hi, vmono, q))
    dq = q_hi - q_lo
    if not (np.isfinite(dq) and dq > 0):
        return float("nan"), float("nan")

    # Mean current over the in-window rows (for C-rate control).
    in_win = (vmono >= v_lo) & (vmono <= v_hi)
    i_win = float(np.nanmean(np.abs(cc[in_win]))) if in_win.any() else float("nan")
    return dq, i_win


def ica_timeline(
    base_dir: str,
    vehicle: str,
    *,
    v_lo: float = DEFAULT_V_LO,
    v_hi: float = DEFAULT_V_HI,
    i_band: float = DEFAULT_I_BAND,
    cell_v_col: str = DEFAULT_CELL_V_COL,
) -> pd.DataFrame:
    """Per-session ICA ΔQ timeline for one vehicle.

    Columns: ``session_id, start_time, delta_q_Ah, i_win_A``. C-rate controlled:
    only sessions whose in-window current is within ``±i_band`` of the median are
    kept.
    """
    df = io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(base_dir, vehicle))
    df = sessions.split_sessions(df)

    rows = []
    for sid, sub in df.groupby("session_id", sort=True):
        if len(sub) < 100:
            continue
        dq, i_win = ica_delta_q(sub, v_lo=v_lo, v_hi=v_hi, cell_v_col=cell_v_col)
        if not np.isfinite(dq):
            continue
        rows.append({
            "session_id": int(sid),
            "start_time": sub["Time"].iloc[0],
            "delta_q_Ah": dq,
            "i_win_A": i_win,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # C-rate control: keep the band around the median in-window current.
    med = out["i_win_A"].median()
    lo, hi = med * (1 - i_band), med * (1 + i_band)
    kept = out[(out["i_win_A"] >= lo) & (out["i_win_A"] <= hi)].reset_index(drop=True)
    logging.info(
        f"vehicle #{vehicle}: {len(out)} sessions cross [{v_lo},{v_hi}]V; "
        f"{len(kept)} within current band {lo:.1f}-{hi:.1f} A (median {med:.1f})"
    )
    return kept


# --------------------------------------------------------------------------- #
# Comparison harness
# --------------------------------------------------------------------------- #
def _days(times: pd.Series) -> np.ndarray:
    t = pd.to_datetime(times, utc=True)
    return (t - t.min()).dt.total_seconds().to_numpy() / 86400.0


def _residual_cv(days: np.ndarray, value: np.ndarray, deg: int = 2) -> dict:
    """Fit a degree-``deg`` polynomial fade trend vs. time and report scatter.

    Returns raw CV (pooled, mixes aging + noise) and detrended residual CV
    (scatter around the fade curve = the noise proxy).
    """
    m = np.isfinite(days) & np.isfinite(value)
    days, value = days[m], value[m]
    n = len(value)
    if n < deg + 2:
        return {"n": n, "raw_cv": float("nan"), "resid_cv": float("nan"), "r2": float("nan")}
    med = float(np.median(value))
    coef = np.polyfit(days, value, deg)
    trend = np.polyval(coef, days)
    resid = value - trend
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((value - value.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": n,
        "raw_cv": float(np.std(value) / med * 100) if med else float("nan"),
        "resid_cv": float(np.std(resid) / med * 100) if med else float("nan"),
        "r2": r2,
    }


def compare_vehicle(
    base_dir: str,
    vehicle: str,
    our_csv_path: str,
    *,
    v_lo: float = DEFAULT_V_LO,
    v_hi: float = DEFAULT_V_HI,
    i_band: float = DEFAULT_I_BAND,
) -> dict:
    """Build the three SOH timelines and report detrended residual scatter."""
    # (A) Deng ΔSOC coulomb counting (every clean charge).
    author = benchmark_shiyunliu.author_capacities(base_dir, vehicle)
    a = _residual_cv(_days(author["start_time"]), author["Capacity_py_author"].to_numpy()) \
        if not author.empty else {"n": 0}

    # (B) Our HDBSCAN CAP timeline (from the emitted capacity CSV).
    if os.path.exists(our_csv_path):
        ours = pd.read_csv(our_csv_path)
        b = _residual_cv(_days(ours["CAP_start_time"]), ours["Capacity_py"].to_numpy()) \
            if not ours.empty else {"n": 0}
    else:
        b = {"n": 0}

    # (C) ICA ΔQ in a fixed voltage window (no SOC division).
    ica = ica_timeline(base_dir, vehicle, v_lo=v_lo, v_hi=v_hi, i_band=i_band)
    c = _residual_cv(_days(ica["start_time"]), ica["delta_q_Ah"].to_numpy()) \
        if not ica.empty else {"n": 0}

    return {"vehicle": vehicle, "deng": a, "ours": b, "ica": c,
            "ica_timeline": ica, "deng_timeline": author}


def _fmt(label: str, d: dict) -> str:
    if not d or d.get("n", 0) < 4:
        return f"  {label:<6} n={d.get('n', 0):>4}  (too few points)"
    return (f"  {label:<6} n={d['n']:>4}  raw_CV={d['raw_cv']:>5.1f}%  "
            f"resid_CV={d['resid_cv']:>5.1f}%  trend_R²={d['r2']:>5.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_dir", nargs="?",
                        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev")
    parser.add_argument("--vehicle", default="1")
    parser.add_argument("--our-dir", default=None,
                        help="Dir with <vehicle>_capacity.csv (default <base_dir>/40_capacity_monitore)")
    parser.add_argument("--v-lo", type=float, default=DEFAULT_V_LO)
    parser.add_argument("--v-hi", type=float, default=DEFAULT_V_HI)
    parser.add_argument("--i-band", type=float, default=DEFAULT_I_BAND)
    parser.add_argument("--plot", default=None, help="Optional PNG path for the three timelines")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    our_dir = args.our_dir or os.path.join(args.base_dir, "40_capacity_monitore")
    our_csv = os.path.join(our_dir, f"{args.vehicle}_capacity.csv")

    res = compare_vehicle(args.base_dir, args.vehicle, our_csv,
                          v_lo=args.v_lo, v_hi=args.v_hi, i_band=args.i_band)

    print(f"\n=== Vehicle #{args.vehicle} — SOH-tracker scatter "
          f"(ICA window {args.v_lo}-{args.v_hi} V cell) ===")
    print("  Lower resid_CV = cleaner once aging drift is removed.\n")
    print(_fmt("Deng", res["deng"]))
    print(_fmt("ours", res["ours"]))
    print(_fmt("ICA", res["ica"]))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            au = res["deng_timeline"]
            ax[0].scatter(_days(au["start_time"]), au["Capacity_py_author"], s=6, alpha=0.4)
            ax[0].set_title(f"Deng ΔSOC  (resid_CV={res['deng'].get('resid_cv', float('nan')):.1f}%)")
            ax[0].set_ylabel("Capacity (Ah)")
            if os.path.exists(our_csv):
                ou = pd.read_csv(our_csv)
                ax[1].scatter(_days(ou["CAP_start_time"]), ou["Capacity_py"], s=10, c="tab:orange")
            ax[1].set_title(f"Ours HDBSCAN  (resid_CV={res['ours'].get('resid_cv', float('nan')):.1f}%)")
            ax[1].set_ylabel("Capacity (Ah)")
            ic = res["ica_timeline"]
            ax[2].scatter(_days(ic["start_time"]), ic["delta_q_Ah"], s=10, c="tab:green")
            ax[2].set_title(f"ICA ΔQ  (resid_CV={res['ica'].get('resid_cv', float('nan')):.1f}%)")
            ax[2].set_ylabel("ΔQ in window (Ah)")
            ax[2].set_xlabel("days since first session")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=110)
            print(f"\nplot → {args.plot}")
        except Exception as e:
            logging.warning(f"plot failed: {e}")


if __name__ == "__main__":
    main()
