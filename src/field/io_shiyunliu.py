"""Adapter for the shiyunliu on-road EV charging dataset.

20 production EVs over ~29 months each, charging-only BMS logs in CSV form
inside ``#N.rar`` archives. Repo: shiyunliu-battery/battery-charging-data-of-
on-road-electric-vehicles (MIT). Accompanies Deng et al., Applied Energy 339
(2023) 120954.

Source layout under ``base_dir`` (after ``unrar x #N.rar``)::

    #1.csv, #2.csv, ..., #20.csv

Each CSV has unit-suffixed columns and an integer ``record_time`` formatted
``YYYYMMDDhhmmss``. Sign convention: ``charge_current`` is **negative when
charging** (verified at runtime by ``check_sign_convention`` against dSOC/dt).
This is the opposite of the RWTH dataset, so the adapter negates current at
load time so the canonical schema follows the convention used elsewhere in
the field track (positive Current = charging).
"""
from __future__ import annotations

import os
import glob
import re
import pandas as pd


CANONICAL_COLUMNS = [
    "Time", "Voltage", "Current", "Temperature", "SOC",
    "Cell_V_max", "Cell_V_min", "Cell_T_min",
    "Available_Energy_kWh", "Available_Capacity_Ah",
]

# Source-name → canonical-name (after the unit suffix is stripped).
RAW_RENAME = {
    "soc": "SOC",
    "pack_voltage": "Voltage",
    "charge_current": "Current",       # sign is negated at load (see module docstring)
    "max_cell_voltage": "Cell_V_max",
    "min_cell_voltage": "Cell_V_min",
    "max_temperature": "Temperature",
    "min_temperature": "Cell_T_min",
    "available_energy": "Available_Energy_kWh",
    "available_capacity": "Available_Capacity_Ah",
}

_UNIT_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _strip_units(name: str) -> str:
    return _UNIT_RE.sub("", name).strip()


def load_vehicle(path: str, *, negate_current: bool = True) -> pd.DataFrame:
    """Load one ``#N.csv`` into the canonical field-data schema.

    The raw CSV's ``record_time`` is decoded from ``YYYYMMDDhhmmss`` into a
    tz-aware UTC datetime. Column unit suffixes (e.g. ``pack_voltage (V)``)
    are stripped before renaming. ``charge_current`` is negated by default so
    positive Current means charging — matches the project-wide convention.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns=lambda c: _strip_units(c))
    df = df.rename(columns=RAW_RENAME)

    df["Time"] = pd.to_datetime(df["record_time"].astype(str), format="%Y%m%d%H%M%S", utc=True)
    df = df.drop(columns=["record_time"])

    if negate_current and "Current" in df.columns:
        df["Current"] = -df["Current"]

    cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras].sort_values("Time").reset_index(drop=True)


def list_vehicles(base_dir: str) -> list[str]:
    """Return sorted vehicle ids (e.g. ``['1', '2', ...]``) that have a CSV present."""
    paths = sorted(glob.glob(os.path.join(base_dir, "#*.csv")))
    return [os.path.splitext(os.path.basename(p))[0].lstrip("#") for p in paths]


def vehicle_path(base_dir: str, vehicle_id: str) -> str:
    return os.path.join(base_dir, f"#{vehicle_id}.csv")


def check_sign_convention(df: pd.DataFrame) -> None:
    """Assert positive Current means charging (SOC rises) — after negation.

    Same logic as field.segment.check_sign_convention: dSOC/dt is unambiguous.
    Window of dt < 30 s is generous because shiyunliu samples at ~8 s.
    """
    needed = {"Time", "SOC", "Current"}
    if not needed.issubset(df.columns):
        return
    d = df[["Time", "SOC", "Current"]].dropna().sort_values("Time")
    dt = d["Time"].diff().dt.total_seconds()
    dsoc = d["SOC"].diff()
    m = (dt > 0) & (dt < 30) & dsoc.notna()
    if m.sum() < 100:
        return
    s = d.loc[m].assign(dsoc=dsoc[m])
    pos_i = s.loc[s["dsoc"] > 0, "Current"].mean()
    neg_i = s.loc[s["dsoc"] < 0, "Current"].mean()
    if pd.isna(pos_i) or pd.isna(neg_i):
        return
    if pos_i <= 0:
        raise ValueError(
            f"Current sign convention looks wrong after adapter negation: "
            f"mean Current where SOC rises is {pos_i:.2f} A (expected > 0)."
        )


def _summarize(df: pd.DataFrame, label: str) -> None:
    span = df["Time"].max() - df["Time"].min()
    nn = {c: int(df[c].notna().sum()) for c in df.columns if c != "Time"}
    soc_range = (df["SOC"].min(), df["SOC"].max())
    i_range = (df["Current"].min(), df["Current"].max())
    print(f"  {label}: rows={len(df):,}  span={span}  "
          f"SOC=[{soc_range[0]:.1f}..{soc_range[1]:.1f}]  "
          f"I=[{i_range[0]:.1f}..{i_range[1]:.1f}]  "
          f"non-null={nn}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the shiyunliu adapter")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev",
    )
    args = parser.parse_args()

    vehicles = list_vehicles(args.base_dir)
    print(f"Found {len(vehicles)} vehicles: {vehicles}")

    for v in vehicles:
        print(f"\n=== vehicle #{v} ===")
        df = load_vehicle(vehicle_path(args.base_dir, v))
        check_sign_convention(df)
        _summarize(df, f"#{v}")
