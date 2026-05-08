"""
Single-pass MinIO fetch: builds BRONZE_CU and an Ah-throughput sidecar
for each cell by downloading each test file exactly once.

For each file:
  - All rows, Zeit + Strom  -> Ah throughput accumulator
  - CU file: all rows, all columns -> BRONZE_CU
  - Non-CU file: first row only, all columns + Prozedur name -> BRONZE_CU stub

Usage:
    cd src
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json --cells VTC_cell01
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json --overwrite

Required config keys (battery_config.json):
    working_path, type_cell, minio_endpoint, bucket_name, minio_prefix
    (e.g. minio_prefix = "j8005-metabatt/Metabatt/VTC")

Credentials via env vars:
    MINIO_ACCESS_KEY, MINIO_SECRET_KEY
"""

import io
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import urllib3
from minio import Minio
from minio.error import S3Error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.add_ah_throughput import add_ah_throughput


def _connect_minio(cfg: dict) -> Minio:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return Minio(
        cfg["minio_endpoint"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        secure=True,
        cert_check=False,
    )


def _is_cu(object_name: str) -> bool:
    parts = os.path.basename(object_name).split("=")
    return len(parts) > 3 and "jri_CU" in parts[3]


def _programme_name(object_name: str) -> str:
    parts = os.path.basename(object_name).split("=")
    return parts[3] if len(parts) > 3 else ""


def _combine_tests(dfs: list) -> pd.DataFrame:
    combined = pd.concat(dfs, ignore_index=True)

    zeit_columns = combined.filter(like="Zeit").columns.tolist()
    if len(zeit_columns) > 1:
        combined["Zeit"] = combined[zeit_columns[0]].fillna(combined[zeit_columns[1]])
    else:
        combined["Zeit"] = combined[zeit_columns[0]]

    col = combined.pop("Zeit")
    combined.insert(0, "Zeit", col)
    combined.drop(columns=zeit_columns, inplace=True)
    combined.sort_values("Zeit", inplace=True)
    combined = combined.rename(columns=lambda x: x.split("#")[0] if "#" in x else x)
    combined.reset_index(drop=True, inplace=True)
    combined["BM_Programm"] = combined.groupby("Ahjo_Test_ID").ngroup()
    return combined


def process_cell(
    minio_client: Minio,
    bucket_name: str,
    prefix: str,
    type_cell: str,
    cell: str,
    out_bronze_cu: str,
    out_ah_sidecar: str,
    overwrite: bool = False,
) -> None:
    if not overwrite and os.path.exists(out_bronze_cu) and os.path.exists(out_ah_sidecar):
        print(f"{cell} - already exists, skipping.")
        return

    objects = minio_client.list_objects(bucket_name, prefix=f"{prefix}/{type_cell}/{cell}/", recursive=True)
    cell_tests = [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]

    if not cell_tests:
        print(f"{cell} - no parquet files found.")
        return

    if not any(_is_cu(t) for t in cell_tests):
        print(f"{cell} - no CU files found, skipping.")
        return

    tests = []
    ah_frames = []

    for object_name in cell_tests:
        try:
            response = minio_client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            print(f"  Error fetching {object_name}: {e}")
            continue

        is_cu = _is_cu(object_name)

        if is_cu:
            try:
                df = pd.read_parquet(io.BytesIO(data))
            except Exception as e:
                print(f"  Error reading {object_name}: {e}")
                continue
            ah_frames.append(df[["Zeit", "Strom"]].copy())
            tests.append(df)
        else:
            # stub: first row only, all columns
            try:
                stub = pq.read_table(io.BytesIO(data)).slice(0, 1).to_pandas()
            except Exception as e:
                print(f"  Error reading stub from {object_name}: {e}")
                continue
            stub["Prozedur"] = _programme_name(object_name)
            tests.append(stub)

            # Ah: all rows, two columns only
            try:
                ah_frames.append(pd.read_parquet(io.BytesIO(data), columns=["Zeit", "Strom"]))
            except Exception as e:
                print(f"  Error reading Zeit/Strom from {object_name}: {e}")

    if not tests:
        print(f"{cell} - no data loaded.")
        return

    # --- BRONZE_CU ---
    os.makedirs(os.path.dirname(out_bronze_cu), exist_ok=True)
    _combine_tests(tests).to_parquet(out_bronze_cu, index=False)
    print(f"  Saved BRONZE_CU:  {out_bronze_cu}")

    # --- Ah throughput sidecar ---
    if not ah_frames:
        print(f"{cell} - no Zeit/Strom data for Ah throughput.")
        return

    df_all = pd.concat(ah_frames, ignore_index=True)
    df_all = df_all.drop_duplicates("Zeit").sort_values("Zeit").reset_index(drop=True)
    df_all[df_all.select_dtypes(np.float64).columns] = (
        df_all.select_dtypes(np.float64).astype(np.float32)
    )
    df_all = df_all.rename(columns={"Strom": "Current", "Zeit": "Time_UTC"})
    if df_all["Time_UTC"].dt.tz is None:
        df_all["Time_UTC"] = df_all["Time_UTC"].dt.tz_localize("UTC")
    else:
        df_all["Time_UTC"] = df_all["Time_UTC"].dt.tz_convert("UTC")
    df_all = add_ah_throughput(df_all)

    os.makedirs(os.path.dirname(out_ah_sidecar), exist_ok=True)
    df_all[["Time_UTC", "Ah_throughput"]].to_parquet(out_ah_sidecar, index=False)
    print(f"  Saved Ah sidecar: {out_ah_sidecar}")


def _list_cells(minio_client: Minio, bucket_name: str, prefix: str) -> list:
    objects = minio_client.list_objects(bucket_name, prefix=f"{prefix}/", recursive=True)
    cells = set()
    strip_len = len(f"{prefix}/")
    for obj in objects:
        remainder = obj.object_name[strip_len:]
        if "/" in remainder:
            cells.add(remainder.split("/")[0])
    return sorted(cells)


def run(cfg: dict, target_cells: list = None, overwrite: bool = False) -> None:
    minio_client = _connect_minio(cfg)
    bucket_name = cfg["bucket_name"]
    prefix = cfg["minio_prefix"]
    working_path = cfg["working_path"]
    type_cell = cfg.get("type_cell", "")

    cells = target_cells if target_cells else _list_cells(minio_client, bucket_name, prefix)

    for cell in cells:
        if type_cell and type_cell not in cell:
            continue
        print(f"Processing {cell}...")
        process_cell(
            minio_client=minio_client,
            bucket_name=bucket_name,
            prefix=prefix,
            cell=cell,
            type_cell=type_cell,
            out_bronze_cu=os.path.join(working_path, "BRONZE_CU", f"{cell}.parquet"),
            out_ah_sidecar=os.path.join(working_path, "Ah_throughput", f"{cell}.parquet"),
            overwrite=overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BRONZE_CU and Ah sidecar from MinIO")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cells")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cfg["minio_access_key"] = os.environ.get("MINIO_ACCESS_KEY", "")
    cfg["minio_secret_key"] = os.environ.get("MINIO_SECRET_KEY", "")

    run(cfg, target_cells=args.cells, overwrite=args.overwrite)
