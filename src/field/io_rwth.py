"""Adapter for the RWTH Aachen EV field-data dataset.

DOI 10.18154/RWTH-2024-01907 (CC BY 4.0). Loads per-vehicle parquet files
into the canonical ``Time / Voltage / Current / Temperature / SOC / Speed /
Odometer / Power`` schema used by the field-data pipeline.

Source layout under ``base_dir``::

    field_test/<vehicle>.parquet                 raw real-world time series
    capacity_test/<vehicle>_capacity_tests.parquet  periodic dyno SOH refs
    charging_curves/<model>_<power>.parquet      characteristic charging curves
    trip_data/*.parquet                          pre-segmented trip aggregates
"""
from __future__ import annotations

import os
import glob
import pandas as pd


# MATLAB datenum: days since 0000-01-01. Python ordinal (0001-01-01) is 719529
# days later, so MATLAB_datenum - 719529 days = days since the Unix-friendly
# 1970-01-01 once we offset by the right epoch. We use the standard offset that
# makes ``pd.to_datetime(time_num - 719529, unit='D')`` produce wall-clock UTC.
MATLAB_EPOCH_OFFSET_DAYS = 719529

FIELD_TEST_RENAME = {
    "Temp_Ambient": "Temperature",
    "SoC_Real": "SOC",
}

CANONICAL_COLUMNS = [
    "Time", "Voltage", "Current", "Temperature", "SOC",
    "Speed", "Odometer", "Power",
]


def matlab_datenum_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a MATLAB datenum series (days since 0000-01-01) to UTC datetimes."""
    return pd.to_datetime(series - MATLAB_EPOCH_OFFSET_DAYS, unit="D", utc=True)


def load_field_test(path: str, drop_leading_nans: bool = True) -> pd.DataFrame:
    """Load one ``field_test/<vehicle>.parquet`` into canonical schema.

    Parameters
    ----------
    path : str
        Absolute path to the parquet file.
    drop_leading_nans : bool
        Drop the early rows where Voltage/Current/Temperature are all NaN
        (BMS signals lag SOC at log start). Default True.
    """
    df = pd.read_parquet(path)
    df = df.rename(columns=FIELD_TEST_RENAME)
    df["Time"] = matlab_datenum_to_datetime(df["time_num"])
    df = df.drop(columns=["time_num"])

    if drop_leading_nans:
        signal_cols = [c for c in ("Voltage", "Current", "Temperature") if c in df.columns]
        if signal_cols:
            first_valid = df[signal_cols].dropna(how="all").index.min()
            if pd.notna(first_valid):
                df = df.loc[first_valid:].reset_index(drop=True)

    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def load_capacity_test(path: str) -> pd.DataFrame:
    """Load one ``capacity_test/<vehicle>_capacity_tests.parquet`` into canonical schema.

    Same canonical columns as ``load_field_test`` plus ``test_number`` and
    ``test_direction`` (1 = charge, 2 = discharge per the dataset Readme).
    ``Power_AC`` is preserved.
    """
    df = pd.read_parquet(path)
    df = df.rename(columns=FIELD_TEST_RENAME)
    df["Time"] = matlab_datenum_to_datetime(df["time_num"])
    df = df.drop(columns=["time_num"])
    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def list_field_test_vehicles(base_dir: str) -> list[str]:
    """Return sorted vehicle ids that have a ``field_test/<id>.parquet`` file."""
    paths = sorted(glob.glob(os.path.join(base_dir, "field_test", "*.parquet")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def field_test_path(base_dir: str, vehicle: str) -> str:
    return os.path.join(base_dir, "field_test", f"{vehicle}.parquet")


def capacity_test_path(base_dir: str, vehicle: str) -> str:
    return os.path.join(base_dir, "capacity_test", f"{vehicle}_capacity_tests.parquet")


def _summarize(df: pd.DataFrame, label: str) -> None:
    span = df["Time"].max() - df["Time"].min()
    nn = {c: int(df[c].notna().sum()) for c in df.columns if c != "Time"}
    print(f"  {label}: rows={len(df):,}  span={span}  non-null={nn}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the RWTH field-data adapter")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/rwth_aachen",
        help="Directory containing field_test/, capacity_test/, etc.",
    )
    args = parser.parse_args()

    vehicles = list_field_test_vehicles(args.base_dir)
    print(f"Found {len(vehicles)} vehicles: {vehicles}")

    for v in vehicles:
        print(f"\n=== {v} ===")
        df = load_field_test(field_test_path(args.base_dir, v))
        _summarize(df, "field_test")
        cap_path = capacity_test_path(args.base_dir, v)
        if os.path.exists(cap_path):
            cap = load_capacity_test(cap_path)
            _summarize(cap, "capacity_test")
            if "test_number" in cap.columns:
                n_tests = cap["test_number"].dropna().nunique()
                print(f"    capacity_test: {n_tests} distinct test_number values")
