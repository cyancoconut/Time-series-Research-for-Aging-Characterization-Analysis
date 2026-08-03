"""
Adapter: ISU-ILCC per-cell JSON -> BRONZE_CU-shaped parquet.

Two source files per cell (both double-encoded JSON — a JSON string containing JSON):

  * RPT file (``.../Release X.0/G<g>C<c>.json``): the periodic reference tests.
    Per cell it holds N RPTs; each RPT has four QV sweeps (C/2 & C/5, charge &
    discharge) as time series {Q, V, t, E, I}. This drives the pipeline: each RPT
    becomes one ``BM_Programm`` (so BM_Programm counts the RPTs), split into four
    clean segments by a unique per-sweep ``Zustand``.

  * Cycling file (``.../Cycling_json/Release X.0/G<g>C<c>.json``): the aging cycles
    between RPTs. Used ONLY to accumulate ``Ah_throughput`` (trapezoidal integral of
    |current| over the full cycling history, exactly like util.add_ah_throughput),
    which is merged onto each RPT row by timestamp so every RPT's CAP carries the
    real cumulative usage — not the tiny Ah of the RPT sweeps themselves.

Usage (run from anywhere; it adds ../src to sys.path for pipeline utils):
    python paper/build_bronze_isu.py \
        --rpt "/path/Release 1.0/G14C1.json" \
        --cycling "/path/Cycling_json/Release 1.0/G14C1.json" \
        --cell G14C1 \
        --out "/path/ISU_pipeline/BRONZE_CU/G14C1.parquet"
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

# Lives in paper/ (outside the pipeline); reach into ../src for pipeline utils.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from util.add_ah_throughput import add_ah_throughput

# (RPT QV field key, unique Zustand tag). Unique tags make dismember cut a boundary
# on every Zustand change (with qocv_procedure_filter matching), so each RPT splits
# into four clean segments.
RPT_SWEEPS = [
    ("QV_charge_C_2", "CHA_C2"),
    ("QV_discharge_C_2", "DCH_C2"),
    ("QV_charge_C_5", "CHA_C5"),
    ("QV_discharge_C_5", "DCH_C5"),
]


def load_isu_json(path: str) -> dict:
    """Decode the double-encoded ISU JSON to a dict."""
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, str):
        d = json.loads(d)
    return d


def build_rpt_frame(cell: str, rpt: dict, r: int):
    """One RPT -> a concatenated 4-sweep frame, or None if no usable sweep."""
    frames = []
    for key, zustand in RPT_SWEEPS:
        qv = rpt.get(key)
        if qv is None or r >= len(qv.get("V", [])):
            continue
        t = pd.to_datetime(qv["t"][r])
        current = np.asarray(qv["I"][r], dtype=float)
        voltage = np.asarray(qv["V"][r], dtype=float)
        n = min(len(t), len(current), len(voltage))
        if n == 0:
            continue
        frames.append(pd.DataFrame({
            "Zeit": t[:n],
            "Strom": current[:n],
            "Spannung": voltage[:n],
            "Zustand": zustand,
        }))
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame["T1"] = 25.0
    frame["AhAkku"] = np.nan
    frame["Prozedur"] = "ISU_RPT"
    frame["Ahjo_Test_ID"] = f"{cell}_RPT{r:03d}"
    return frame


def build_rpt_bronze(cell: str, rpt: dict):
    """All RPTs -> one BRONZE frame (sorted by Zeit), or None. One BM_Programm per
    RPT is realised downstream via the zero-padded Ahjo_Test_ID (dismember's
    groupby(...).ngroup() then numbers them in RPT order)."""
    n_rpt = len(rpt.get("capacity_discharge_C_2", []))
    frames = [build_rpt_frame(cell, rpt, r) for r in range(n_rpt)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    bronze = pd.concat(frames, ignore_index=True)
    return bronze.sort_values("Zeit").reset_index(drop=True)


def cycling_ah_series(cyc: dict) -> pd.DataFrame:
    """Cumulative Ah throughput over the full cycling history.

    Concatenates every cycle's charge + discharge (t, I), integrates |I| dt with the
    same trapezoidal method the pipeline uses (util.add_ah_throughput), and returns a
    time-sorted frame with columns ``Zeit`` + ``Ah_throughput``.
    """
    frames = []
    for key in ("QV_charge", "QV_discharge"):
        qv = cyc.get(key)
        if qv is None:
            continue
        ts_list, i_list = qv.get("t", []), qv.get("I", [])
        for c in range(min(len(ts_list), len(i_list))):
            t = pd.to_datetime(ts_list[c])
            current = np.asarray(i_list[c], dtype=float)
            n = min(len(t), len(current))
            if n == 0:
                continue
            frames.append(pd.DataFrame({"Zeit": t[:n], "Current": current[:n]}))
    if not frames:
        return pd.DataFrame(columns=["Zeit", "Ah_throughput"])
    allc = pd.concat(frames, ignore_index=True)
    allc = allc.drop_duplicates("Zeit").sort_values("Zeit").reset_index(drop=True)
    allc["Time_UTC"] = allc["Zeit"]
    if allc["Time_UTC"].dt.tz is None:
        allc["Time_UTC"] = allc["Time_UTC"].dt.tz_localize("UTC")
    allc = add_ah_throughput(allc)
    return allc[["Zeit", "Ah_throughput"]]


def merge_ah(bronze: pd.DataFrame, ah_series: pd.DataFrame) -> pd.DataFrame:
    """Assign each RPT row the cumulative cycling Ah_throughput at its timestamp
    (last cycling sample at/before the row). RPT rows before any cycling (e.g. the
    initial RPT0) get 0."""
    if ah_series.empty:
        bronze = bronze.copy()
        bronze["Ah_throughput"] = 0.0
        return bronze
    left = bronze.sort_values("Zeit").reset_index(drop=True)
    right = ah_series.sort_values("Zeit").reset_index(drop=True)
    merged = pd.merge_asof(left, right, on="Zeit", direction="backward")
    merged["Ah_throughput"] = merged["Ah_throughput"].fillna(0.0)
    return merged


def build_cell(cell: str, rpt_path: str, cycling_path: str) -> pd.DataFrame:
    """Full BRONZE_CU frame for one cell: per-RPT stream + cycling-derived Ah."""
    rpt = load_isu_json(rpt_path)
    bronze = build_rpt_bronze(cell, rpt)
    if bronze is None:
        return None
    ah_series = cycling_ah_series(load_isu_json(cycling_path))
    return merge_ah(bronze, ah_series)


def main():
    parser = argparse.ArgumentParser(description="Build BRONZE_CU from ISU-ILCC JSON")
    parser.add_argument("--rpt", required=True, help="RPT JSON path")
    parser.add_argument("--cycling", required=True, help="Cycling JSON path (Ah only)")
    parser.add_argument("--cell", required=True, help="Cell stem, e.g. G14C1")
    parser.add_argument("--out", required=True, help="Output BRONZE_CU parquet path")
    args = parser.parse_args()

    bronze = build_cell(args.cell, args.rpt, args.cycling)
    if bronze is None:
        print(f"{args.cell} - no usable RPT data.")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bronze.to_parquet(args.out, index=False)
    n_rpt = bronze["Ahjo_Test_ID"].nunique()
    print(f"{args.cell} - wrote {len(bronze)} rows, {n_rpt} RPTs, "
          f"Ah_throughput {bronze['Ah_throughput'].min():.3f}..{bronze['Ah_throughput'].max():.3f} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
