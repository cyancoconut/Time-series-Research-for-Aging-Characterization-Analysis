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
2. **Scatter reduction**: individual capacity estimates should be less
   noisy on our HDBSCAN-filtered subset, demonstrating that the cluster
   restricts the estimator to its high-confidence regime.

   This must be measured as scatter *about the ageing trend*, not as the
   spread of the timeline as a whole. These cells fade 20-30% over the
   record, so ``std/median`` across all sessions is dominated by real
   degradation and is nearly blind to estimator noise: on this dataset it
   reports a 1.1x improvement and is worse on 9 of 20 vehicles, while the
   trend-residual measure below reports 1.27x and is better on 20 of 20.
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


def trend_residual_scatter(times, values, frac: float = 0.15) -> float | None:
    """Scatter of capacity estimates about their own ageing trend, in percent.

    A field cell fades over the record, so the spread of a capacity timeline
    measures degradation plus estimator noise. Detrending with a centred
    rolling median leaves the noise. The residual is summarised by its median
    absolute deviation (scaled to a standard-deviation equivalent) and
    normalised by the median capacity, giving a robust, trend-free coefficient
    of variation that is comparable between two session subsets of different
    size.
    """
    t = pd.to_datetime(pd.Series(times), errors="coerce", utc=True)
    y = pd.Series(values, dtype=float).to_numpy()
    ok = t.notna().to_numpy() & np.isfinite(y)
    if ok.sum() < 8:
        return None
    order = np.argsort(t[ok].astype("int64").to_numpy())
    y = y[ok][order]
    window = max(5, int(len(y) * frac)) | 1  # odd, so the median is centred
    trend = pd.Series(y).rolling(window, center=True, min_periods=3).median().to_numpy()
    resid = y - trend
    resid = resid[np.isfinite(resid)]
    if resid.size == 0 or not np.isfinite(np.median(y)) or np.median(y) == 0:
        return None
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(1.4826 * mad / np.median(y) * 100)


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
        # Trend-inclusive spread, kept only for transparency: it is dominated
        # by real ageing and should not be read as an estimator-noise measure.
        "author_soh_std": float(author["Capacity_py_author"].std() / author["Capacity_py_author"].median() * 100),
        "ours_soh_std": float(ours["Capacity_py"].std() / ours["Capacity_py"].median() * 100),
        # The actual noise measure.
        "author_resid_pct": trend_residual_scatter(
            author["start_time"], author["Capacity_py_author"]),
        "ours_resid_pct": trend_residual_scatter(
            ours["CAP_start_time"], ours["Capacity_py"]),
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
            f"resid% author={result['author_resid_pct']:.2f} ours={result['ours_resid_pct']:.2f}"
            f"  (trend-inclusive CV% {result['author_soh_std']:.1f}/{result['ours_soh_std']:.1f})"
            if result['n_matched'] else
            f"#{v:>3} | author={result['n_author']:>4} ours={result['n_ours']:>3} no overlap"
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        ratio = (df["author_resid_pct"] / df["ours_resid_pct"]).dropna()
        cv_ratio = (df["author_soh_std"] / df["ours_soh_std"]).dropna()
        print(f"\nFleet: HDBSCAN cuts capacity scatter about the ageing trend by "
              f"{ratio.median():.2f}× (median, IQR {ratio.quantile(0.25):.2f}–"
              f"{ratio.quantile(0.75):.2f}×); better on {int((ratio > 1).sum())}/"
              f"{len(ratio)} vehicles.")
        print(f"       The trend-inclusive CV would report {cv_ratio.median():.2f}× "
              f"(better on {int((cv_ratio > 1).sum())}/{len(cv_ratio)}), because it "
              f"is dominated by real ageing rather than estimator noise.")


if __name__ == "__main__":
    main()
