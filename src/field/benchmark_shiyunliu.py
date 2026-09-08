"""F5 — benchmark our HDBSCAN-filtered capacity timeline against the
dataset author's published extraction (Deng et al. 2023).

The author's ``capacity_extract.py`` runs the same formula we do
(``∫I dt / ΔSOC × 100``) on every session that passes a coarse validity
check (≥100 rows, monotonic SOC, current NaN < 10 %). Our pipeline adds
HDBSCAN clustering on top to keep only the full CC-CV sessions.

The benchmark therefore checks two things:

1. **Numerical agreement**: on the sessions both methods retain, capacity
   values should be identical (modulo rounding) — proves our integration
   matches the author's reference.
2. **Variance reduction**: the variance of the SOH timeline should be
   meaningfully lower on our HDBSCAN-filtered subset, demonstrating that
   the cluster restricts the estimator to its high-confidence regime.
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd

from field import io_shiyunliu, sessions


def author_capacities(base_dir: str, vehicle: str) -> pd.DataFrame:
    """Per-session capacity table emulating the author's capacity_extract.py.

    Same filters: session ≥100 rows; per-row dSOC stays within (-0.1, 2)%;
    current NaN fraction < 10%; ΔSOC != 0. Same formula:
    ``Capacity_py = ∫|I|dt / (|ΔSOC|/100)``.

    Operates on our canonical (post-negation) Current, so the integrand
    is ``np.abs(Current)`` rather than the author's ``-Current``.
    """
    df = io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(base_dir, vehicle))
    df = sessions.split_sessions(df)

    rows = []
    for sid, sub in df.groupby("session_id", sort=True):
        if len(sub) < 100:
            continue
        dsoc_per_row = sub["SOC"].diff().iloc[1:]
        if (dsoc_per_row > 2).any() or (dsoc_per_row < -0.1).any():
            continue
        cur = sub["Current"]
        if cur.isna().mean() > 0.1:
            continue
        cur = cur.ffill()
        t = sub["Time"].to_numpy()
        i = np.abs(cur.to_numpy(dtype=float))
        dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
        net_ah = float(np.sum(0.5 * (i[:-1] + i[1:]) * dt_s) / 3600.0)
        dsoc = float(sub["SOC"].iloc[-1] - sub["SOC"].iloc[0])
        if dsoc == 0:
            continue
        rows.append({
            "session_id": int(sid),
            "start_time": sub["Time"].iloc[0],
            "end_time": sub["Time"].iloc[-1],
            "SOC_start": float(sub["SOC"].iloc[0]),
            "SOC_end": float(sub["SOC"].iloc[-1]),
            "dSOC": dsoc,
            "Capacity_py_author": abs(net_ah) / (abs(dsoc) / 100.0),
        })
    return pd.DataFrame(rows)


def compare_vehicle(base_dir: str, vehicle: str, our_csv_path: str) -> dict:
    author = author_capacities(base_dir, vehicle)
    ours = pd.read_csv(our_csv_path) if os.path.exists(our_csv_path) else pd.DataFrame()

    n_author = len(author)
    n_ours = len(ours)

    if n_author == 0 or n_ours == 0:
        return {
            "vehicle": vehicle, "n_author": n_author, "n_ours": n_ours,
            "n_matched": 0, "subset": None, "median_abs_diff_Ah": None,
            "author_cap_iqr_Ah": None, "ours_cap_iqr_Ah": None,
            "author_soh_std": None, "ours_soh_std": None,
        }

    matched = ours.merge(
        author[["session_id", "Capacity_py_author"]],
        left_on="BM_Programm", right_on="session_id", how="inner",
    )
    abs_diff = (matched["Capacity_py"] - matched["Capacity_py_author"]).abs()

    return {
        "vehicle": vehicle,
        "n_author": n_author,
        "n_ours": n_ours,
        "n_matched": int(len(matched)),
        "subset": bool(len(matched) == n_ours),
        "median_abs_diff_Ah": float(abs_diff.median()) if len(matched) else None,
        "max_abs_diff_Ah": float(abs_diff.max()) if len(matched) else None,
        "author_cap_iqr_Ah": float(author["Capacity_py_author"].quantile(0.75) - author["Capacity_py_author"].quantile(0.25)),
        "ours_cap_iqr_Ah": float(ours["Capacity_py"].quantile(0.75) - ours["Capacity_py"].quantile(0.25)),
        "author_soh_std": float(author["Capacity_py_author"].std() / author["Capacity_py_author"].median() * 100),
        "ours_soh_std": float(ours["Capacity_py"].std() / ours["Capacity_py"].median() * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark our F4 capacity vs. the author's extraction")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev",
    )
    parser.add_argument(
        "--our-dir",
        default=None,
        help="Directory holding our <vehicle>_capacity.csv (default: <base_dir>/40_capacity_monitore)",
    )
    parser.add_argument("--vehicle", default=None)
    args = parser.parse_args()

    our_dir = args.our_dir or os.path.join(args.base_dir, "40_capacity_monitore")
    vehicles = [args.vehicle] if args.vehicle else io_shiyunliu.list_vehicles(args.base_dir)

    rows = []
    for v in vehicles:
        our_csv = os.path.join(our_dir, f"{v}_capacity.csv")
        result = compare_vehicle(args.base_dir, v, our_csv)
        rows.append(result)
        print(
            f"#{v:>3} | author={result['n_author']:>4} ours={result['n_ours']:>3} "
            f"matched={result['n_matched']:>3} subset={result['subset']!s:>5} | "
            f"median|Δ|={result['median_abs_diff_Ah']:.3f} Ah  "
            f"max|Δ|={result['max_abs_diff_Ah']:.3f} Ah | "
            f"author CV%={result['author_soh_std']:.1f}  ours CV%={result['ours_soh_std']:.1f}"
            if result['n_matched'] else
            f"#{v:>3} | author={result['n_author']:>4} ours={result['n_ours']:>3} no overlap"
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        ratio = (df["author_soh_std"] / df["ours_soh_std"]).dropna()
        print(f"\nFleet: HDBSCAN cuts capacity CV by a factor of "
              f"{ratio.median():.1f}× (median across vehicles, IQR "
              f"{ratio.quantile(0.25):.1f}–{ratio.quantile(0.75):.1f}×)")


if __name__ == "__main__":
    main()
