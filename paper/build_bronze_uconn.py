"""
Adapter: UConn-ILCC NMC RPT CSVs -> BRONZE_CU-shaped parquet.

Two source layouts per cell (sibling of the ISU adapter, but tabular CSV):

  * RPT files (``<rpt_dir>/rpt_cell_XX_part{0,1,2}.csv``): the periodic reference
    tests. RPT Number is globally continuous across parts (part0: 0..11, part1:
    12..23, ...), dates monotonic, so the parts concatenate in order and each RPT
    becomes one ``BM_Programm`` (Ahjo_Test_ID = ``cell_XX_RPT<n>``). Each RPT holds
    reference C/2.9 charge/discharge sweeps (``ref_chg`` / ``ref_dchg``) plus slow-
    and fast-pulse trains at 9 SoC points. Segmentation is driven by the ``State``
    column (mapped to ``Zustand``): with ``qocv_procedure_filter`` matching, every
    State change cuts a boundary, and the ~10-min ``Rest`` blocks become PAU stubs.

  * Cycling zip (``<cycling_zip>::cycling_data/cycling_cell_XX_part*.csv``): the aging
    cycles between RPTs. Read ONLY to accumulate ``Ah_throughput`` (trapezoidal
    integral of |current| over the full cycling history via util.add_ah_throughput),
    merged onto each RPT row by timestamp so every RPT's CAP carries the real
    cumulative usage. Members are streamed straight out of the zip (never extracted).

The ground-truth ``Segment_Key`` and ``Pulse_SOC`` columns are carried through into
BRONZE (the pipeline ignores unknown columns) so the blind HDBSCAN labels can be
validated against them later.

Usage (run from anywhere; it adds ../src to sys.path for pipeline utils):
    python paper/build_bronze_uconn.py \
        --rpt-dir "/path/UConn-ILCC NMC_rpt_data" \
        --cycling-zip "/path/cycling_data.zip" \
        --out-dir "/path/UConn_pipeline/BRONZE_CU" \
        --cells cell_01 cell_02 cell_03
"""

import os
import re
import sys
import glob
import zipfile
import argparse

import numpy as np
import pandas as pd

# Lives in paper/ (outside the pipeline); reach into ../src for pipeline utils.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
from util.add_ah_throughput import add_ah_throughput

DATE_COL = "Date (yyyy.mm.dd hh.mm.ss)"

# Source column -> BRONZE German column. Everything else in RPT_KEEP is carried
# through untouched (extra columns are inert in the pipeline). Zeit is created
# from DATE_COL in load_rpt, so it is not remapped here.
RPT_RENAME = {
    "Current (A)": "Strom",
    "Voltage (V)": "Spannung",
    "Capacity (Ah)": "AhAkku",
    "State": "Zustand",
    "Segment Key": "Segment_Key",
    "Pulse SOC": "Pulse_SOC",
}
RPT_KEEP = ["RPT Number", DATE_COL] + list(RPT_RENAME)


def _part_index(path: str) -> int:
    m = re.search(r"_part(\d+)\.csv$", path)
    return int(m.group(1)) if m else 0


def load_rpt(rpt_dir: str, cell: str) -> pd.DataFrame:
    """Concatenate a cell's RPT parts in part order, sorted by timestamp."""
    parts = sorted(glob.glob(os.path.join(rpt_dir, f"rpt_{cell}_part*.csv")),
                   key=_part_index)
    if not parts:
        return None
    frames = [pd.read_csv(p, usecols=RPT_KEEP) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["Zeit"] = pd.to_datetime(df[DATE_COL])
    return df.sort_values("Zeit").reset_index(drop=True)


def build_rpt_bronze(cell: str, df: pd.DataFrame) -> pd.DataFrame:
    """RPT stream -> BRONZE frame. One BM_Programm per RPT via a zero-padded
    Ahjo_Test_ID (dismember's groupby(...).ngroup() numbers them in RPT order)."""
    bronze = df.rename(columns=RPT_RENAME).copy()
    bronze["T1"] = 22.0  # no cell-surface temp channel; room held at ~22 C (README)
    bronze["Prozedur"] = "UConn_RPT"
    bronze["Ahjo_Test_ID"] = [f"{cell}_RPT{int(n):03d}" for n in df["RPT Number"]]
    keep = ["Zeit", "Strom", "Spannung", "T1", "AhAkku", "Zustand",
            "Prozedur", "Ahjo_Test_ID", "Segment_Key", "Pulse_SOC"]
    return bronze[keep]


def cycling_ah_series(cycling_zip: str, cell: str) -> pd.DataFrame:
    """Cumulative Ah throughput over the full cycling history for one cell.

    Streams every ``cycling_<cell>_part*.csv`` out of the zip (Date + Current only),
    concatenates in time order, and integrates |I| dt with the pipeline's own
    trapezoidal method (util.add_ah_throughput). Returns ``Zeit`` + ``Ah_throughput``.
    """
    frames = []
    with zipfile.ZipFile(cycling_zip) as zf:
        members = sorted(
            (n for n in zf.namelist()
             if re.search(rf"cycling_{cell}_part\d+\.csv$", n)),
            key=_part_index,
        )
        for name in members:
            with zf.open(name) as fh:
                frames.append(pd.read_csv(fh, usecols=[DATE_COL, "Current (A)"]))
    if not frames:
        return pd.DataFrame(columns=["Zeit", "Ah_throughput"])
    allc = pd.concat(frames, ignore_index=True)
    allc["Zeit"] = pd.to_datetime(allc[DATE_COL])
    allc = allc.rename(columns={"Current (A)": "Current"})
    allc = allc.drop_duplicates("Zeit").sort_values("Zeit").reset_index(drop=True)
    allc["Time_UTC"] = allc["Zeit"]
    allc = add_ah_throughput(allc)
    return allc[["Zeit", "Ah_throughput"]]


def merge_ah(bronze: pd.DataFrame, ah_series: pd.DataFrame) -> pd.DataFrame:
    """Assign each RPT row the cumulative cycling Ah_throughput at its timestamp
    (last cycling sample at/before the row). RPT rows before any cycling get 0."""
    if ah_series.empty:
        bronze = bronze.copy()
        bronze["Ah_throughput"] = 0.0
        return bronze
    left = bronze.sort_values("Zeit").reset_index(drop=True)
    right = ah_series.sort_values("Zeit").reset_index(drop=True)
    merged = pd.merge_asof(left, right, on="Zeit", direction="backward")
    merged["Ah_throughput"] = merged["Ah_throughput"].fillna(0.0)
    return merged


def build_cell(cell: str, rpt_dir: str, cycling_zip: str) -> pd.DataFrame:
    """Full BRONZE_CU frame for one cell: per-RPT stream + cycling-derived Ah."""
    df = load_rpt(rpt_dir, cell)
    if df is None:
        return None
    bronze = build_rpt_bronze(cell, df)
    ah_series = cycling_ah_series(cycling_zip, cell)
    return merge_ah(bronze, ah_series)


def main():
    parser = argparse.ArgumentParser(description="Build BRONZE_CU from UConn-ILCC CSVs")
    parser.add_argument("--rpt-dir", required=True, help="Dir with rpt_cell_XX_part*.csv")
    parser.add_argument("--cycling-zip", required=True, help="cycling_data.zip path")
    parser.add_argument("--out-dir", required=True, help="BRONZE_CU output dir")
    parser.add_argument("--type-cell", default="UConn", help="Prefix for output stem")
    parser.add_argument("--cells", nargs="+", default=["cell_01", "cell_02", "cell_03"],
                        help="Cell stems, e.g. cell_01 cell_02")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for cell in args.cells:
        bronze = build_cell(cell, args.rpt_dir, args.cycling_zip)
        if bronze is None:
            print(f"{cell} - no RPT files found under {args.rpt_dir}.")
            continue
        out = os.path.join(args.out_dir, f"{args.type_cell}_{cell}.parquet")
        bronze.to_parquet(out, index=False)
        n_rpt = bronze["Ahjo_Test_ID"].nunique()
        print(f"{cell} - wrote {len(bronze)} rows, {n_rpt} RPTs, "
              f"Ah_throughput {bronze['Ah_throughput'].min():.2f}.."
              f"{bronze['Ah_throughput'].max():.2f} -> {out}")


if __name__ == "__main__":
    main()
