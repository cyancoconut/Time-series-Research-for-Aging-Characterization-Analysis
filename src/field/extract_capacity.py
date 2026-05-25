"""F4 — extract opportunistic capacity from CAP-cluster sessions, emit per-vehicle CSV.

End-to-end pipeline for the shiyunliu field-data track:

    load_vehicle  →  split_sessions  →  session_features  →
        cluster_sessions  →  pick_cap_cluster  →  this module

For each session in the picked CAP cluster, coulomb-count
``Capacity_py = ∫|I|dt / (|ΔSOC|/100)``. Cumulative ``Ah_throughput`` is
integrated across **all** sessions chronologically (not just CAP ones) so the
emitted CSV matches the shape of ``40_capacity_monitore/<cell>_capacity.csv``
written by the CU pipeline, letting the existing aging-status monitor and
aging matrix work unchanged.
"""
from __future__ import annotations

import os
import argparse
import logging

import numpy as np
import pandas as pd

from field import io_shiyunliu, sessions, cluster_sessions


OUTPUT_COLUMNS = ["BM_Programm", "Capacity_py", "Ah_throughput", "SOH", "CAP_start_time"]
DEFAULT_OUTPUT_DIRNAME = "40_capacity_monitore"


def _trapz_abs_Ah(time: pd.Series, current: pd.Series) -> float:
    """Trapezoidal ∫|I|dt over a single session, returned in Ah."""
    m = current.notna() & time.notna()
    if m.sum() < 2:
        return float("nan")
    t = time[m].to_numpy()
    i = np.abs(current[m].to_numpy(dtype=float))
    dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
    return float(np.sum(0.5 * (i[:-1] + i[1:]) * dt_s) / 3600.0)


def extract_for_vehicle(
    base_dir: str,
    vehicle: str,
    *,
    nom_capacity: float | None = None,
) -> pd.DataFrame:
    """Run F1→F4 on one vehicle and return the per-CAP-session capacity table.

    If ``nom_capacity`` is None, SOH is normalised to the vehicle's own peak
    extracted capacity (self-referential).
    """
    df = io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(base_dir, vehicle))
    df = sessions.split_sessions(df)
    feats = sessions.session_features(df, vehicle=vehicle)
    if feats.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    labeled = cluster_sessions.cluster_sessions(feats)
    cap_label = cluster_sessions.pick_cap_cluster(labeled)
    if cap_label is None:
        logging.warning(f"vehicle #{vehicle}: no CAP cluster found, returning empty table")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cap_session_ids = set(labeled.loc[labeled["cluster_label"] == cap_label, "session_id"].tolist())

    # Walk sessions chronologically to accumulate Ah throughput; integrate ∫|I|dt
    # for every session so the throughput counter reflects real cell use, not
    # just CAP events. Capacity is computed only for CAP-cluster sessions.
    feats_sorted = labeled.sort_values("start_time").reset_index(drop=True)
    sid_to_session_rows = dict(tuple(df.groupby("session_id")))

    cum_ah = 0.0
    rows = []
    for _, sf in feats_sorted.iterrows():
        sid = sf["session_id"]
        sub = sid_to_session_rows.get(sid)
        if sub is None or sub.empty:
            continue
        net_ah = _trapz_abs_Ah(sub["Time"], sub["Current"])
        # Ah throughput captured BEFORE this session — matches the CU pipeline's
        # "throughput at the start of the CAP segment" convention.
        ah_at_start = cum_ah
        if np.isfinite(net_ah):
            cum_ah += net_ah

        if sid not in cap_session_ids:
            continue

        dsoc = sf["dSOC"]
        if not (np.isfinite(dsoc) and abs(dsoc) > 0 and np.isfinite(net_ah)):
            continue
        cap = abs(net_ah) / (abs(dsoc) / 100.0)
        rows.append({
            "BM_Programm": int(sid),
            "Capacity_py": cap,
            "Ah_throughput": ah_at_start,
            "SOH": np.nan,                            # filled below
            "CAP_start_time": sf["start_time"],
        })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(rows)
    ref = nom_capacity if (nom_capacity and nom_capacity > 0) else float(out["Capacity_py"].max())
    out["SOH"] = (out["Capacity_py"] / ref * 100).round(1)
    return out[OUTPUT_COLUMNS]


def write_capacity_csv(df: pd.DataFrame, out_dir: str, vehicle: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{vehicle}_capacity.csv")
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-vehicle opportunistic capacity tables from shiyunliu data"
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev",
    )
    parser.add_argument("--vehicle", default=None,
                        help="Single vehicle id (e.g. 1). Omit to process all.")
    parser.add_argument("--out-dir", default=None,
                        help=f"Override output dir (default: <base_dir>/{DEFAULT_OUTPUT_DIRNAME})")
    parser.add_argument("--nom-capacity", type=float, default=None,
                        help="Nominal capacity in Ah used as the SOH denominator. "
                             "Default: per-vehicle peak Capacity_py.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = args.out_dir or os.path.join(args.base_dir, DEFAULT_OUTPUT_DIRNAME)
    vehicles = [args.vehicle] if args.vehicle else io_shiyunliu.list_vehicles(args.base_dir)

    print(f"{'vehicle':>8}  {'n_CAP':>5}  {'peak_Ah':>8}  {'last_SOH':>8}  {'span_days':>10}")
    for v in vehicles:
        df = extract_for_vehicle(args.base_dir, v, nom_capacity=args.nom_capacity)
        path = write_capacity_csv(df, out_dir, v)
        if df.empty:
            print(f"{v:>8}  {0:>5}  {'--':>8}  {'--':>8}  {'--':>10}  → {path} (empty)")
            continue
        peak = df["Capacity_py"].max()
        last_soh = df.iloc[-1]["SOH"]
        span = pd.to_datetime(df["CAP_start_time"].iloc[-1]) - pd.to_datetime(df["CAP_start_time"].iloc[0])
        print(f"{v:>8}  {len(df):>5}  {peak:>8.1f}  {last_soh:>8.1f}  {span.days:>10}  → {path}")


if __name__ == "__main__":
    main()
