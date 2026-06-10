"""
Single-pass MinIO fetch: builds BRONZE_CU (with Ah_throughput column)
for each cell by downloading each test file exactly once.

For each file:
  - All rows, Zeit + Strom  -> Ah throughput accumulator (full timeline)
  - CU file: all rows, all columns -> BRONZE_CU
  - Non-CU file: first + last row, all columns + Prozedur name -> BRONZE_CU stub

Ah_throughput is computed over the full timeline (all files) and merged
into BRONZE_CU by Zeit, so stubs receive the Ah value at each stub row.

Usage:
    cd src
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json --cells VTC_cell01
    python download/build_bronze_cu_with_ah.py /path/to/battery_config.json --overwrite

Required config keys (battery_config.json):
    working_path, type_cell, minio_endpoint, bucket_name, minio_prefix
    (e.g. minio_prefix = "j8005-metabatt/Metabatt/VTC")

Source is controlled by `download_from`: "local" | "minio" (default "minio").
    When "local", per-test parquets are read from <working_path>/<cell>/*.parquet.

Destination is controlled by `upload_to` (same semantics as main.py):
    "local" | "minio" | "both". Legacy `save_local` / `upload_s3` keys are
    still honored when `upload_to` is absent.

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
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from minio.error import S3Error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.add_ah_throughput import add_ah_throughput
from util import io_router


def _is_cu(object_name: str, cu_marker: str) -> bool:
    parts = os.path.basename(object_name).split("=")
    return len(parts) > 3 and cu_marker in parts[3]


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
    combined.drop(columns=[c for c in zeit_columns if c != "Zeit"], inplace=True)
    combined.sort_values("Zeit", inplace=True)
    combined = combined.rename(columns=lambda x: x.split("#")[0] if "#" in x else x)
    combined.reset_index(drop=True, inplace=True)
    combined["BM_Programm"] = combined.groupby("Ahjo_Test_ID").ngroup()
    return combined


def _save_local_parquet(df: pd.DataFrame, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_parquet(local_path, index=False)
    print(f"  Saved locally:    {local_path}")


def _list_cell_tests_local(working_path: str, cell: str) -> list:
    cell_dir = os.path.join(working_path, cell)
    if not os.path.isdir(cell_dir):
        return []
    return sorted(
        os.path.join(cell_dir, f)
        for f in os.listdir(cell_dir)
        if f.endswith(".parquet")
    )


def _list_cell_tests_minio(minio_client: Minio, bucket: str, prefix: str, cell: str) -> list:
    objects = minio_client.list_objects(bucket, prefix=f"{prefix}/{cell}/", recursive=True)
    return [o.object_name for o in objects if o.object_name.endswith(".parquet")]


def _read_test_bytes_local(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        print(f"  Error fetching {path}: {e}")
        return None


def _read_test_bytes_minio(client: Minio, bucket: str, object_name: str) -> bytes | None:
    try:
        response = client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    except S3Error as e:
        print(f"  Error fetching {object_name}: {e}")
        return None


def process_cell(
    cfg: dict,
    cell: str,
    out_bronze_cu: str | None,
    overwrite: bool = False,
    upload_minio: bool = False,
    minio_client: Minio | None = None,
    download_from: str = "minio",
) -> None:
    bucket_name = cfg["bucket_name"]
    prefix = cfg["minio_prefix"]
    working_path = cfg.get("working_path")
    cell_file = f"{cell}.parquet"
    # CU-file detection follows the config's procedure_filter (the check-up
    # programme name in the 4th '='-delimited filename field).
    cu_marker = cfg.get("procedure_filter")
    if not cu_marker:
        raise ValueError(
            "procedure_filter must be set in the battery config — it is the "
            "check-up programme name used to detect CU test files."
        )

    if not overwrite:
        if out_bronze_cu and os.path.exists(out_bronze_cu):
            print(f"{cell} - local BRONZE_CU already exists, skipping.")
            return
        if upload_minio and io_router.bronze_exists_on_minio(minio_client, cfg, cell_file):
            print(f"{cell} - MinIO BRONZE_CU already exists, skipping.")
            return

    if download_from == "local":
        cell_tests = _list_cell_tests_local(working_path, cell)
    else:
        cell_tests = _list_cell_tests_minio(minio_client, bucket_name, prefix, cell)

    if not cell_tests:
        print(f"{cell} - no parquet files found.")
        return

    if not any(_is_cu(t, cu_marker) for t in cell_tests):
        print(f"{cell} - no CU files found, skipping.")
        return

    tests = []
    ah_frames = []

    for object_name in cell_tests:
        if download_from == "local":
            data = _read_test_bytes_local(object_name)
        else:
            data = _read_test_bytes_minio(minio_client, bucket_name, object_name)
        if data is None:
            continue

        is_cu = _is_cu(object_name, cu_marker)

        if is_cu:
            try:
                df = pd.read_parquet(io.BytesIO(data))
            except Exception as e:
                print(f"  Error reading {object_name}: {e}")
                continue
            ah_frames.append(df[["Zeit", "Strom"]].copy())
            tests.append(df)
        else:
            # Non-CU: open via ParquetFile on the in-memory bytes. Read
            # Zeit+Strom for the Ah accumulator (full cycling timeline,
            # column-projected) and first+last rows (all columns) for the
            # BRONZE_CU stub — no full pandas materialization of the file.
            try:
                pf = pq.ParquetFile(io.BytesIO(data))
            except Exception as e:
                print(f"  Error opening {object_name}: {e}")
                continue

            n_rows = pf.metadata.num_rows
            n_rg = pf.num_row_groups
            if n_rows == 0 or n_rg == 0:
                continue

            try:
                ah_frames.append(pf.read(columns=["Zeit", "Strom"]).to_pandas())
            except Exception as e:
                print(f"  Error reading Zeit/Strom from {object_name}: {e}")

            try:
                first_rg = pf.read_row_group(0)
                if n_rows == 1:
                    stub = first_rg.slice(0, 1).to_pandas()
                else:
                    last_rg = (
                        first_rg if n_rg == 1 else pf.read_row_group(n_rg - 1)
                    )
                    stub = pa.concat_tables([
                        first_rg.slice(0, 1),
                        last_rg.slice(last_rg.num_rows - 1, 1),
                    ]).to_pandas()
            except Exception as e:
                print(f"  Error reading stub from {object_name}: {e}")
                continue
            stub["Prozedur"] = _programme_name(object_name)
            tests.append(stub)

    if not tests:
        print(f"{cell} - no data loaded.")
        return

    bronze = _combine_tests(tests)

    # --- Ah throughput computed over full timeline, merged into BRONZE_CU ---
    if ah_frames:
        df_all = pd.concat(ah_frames, ignore_index=True)
        df_all = df_all.drop_duplicates("Zeit").sort_values("Zeit").reset_index(drop=True)
        df_all[df_all.select_dtypes(np.float64).columns] = (
            df_all.select_dtypes(np.float64).astype(np.float32)
        )
        # add_ah_throughput needs Time_UTC + Current; keep original Zeit for merge key
        df_ah = df_all.rename(columns={"Strom": "Current"})
        df_ah["Time_UTC"] = df_ah["Zeit"]
        if df_ah["Time_UTC"].dt.tz is None:
            df_ah["Time_UTC"] = df_ah["Time_UTC"].dt.tz_localize("UTC")
        else:
            df_ah["Time_UTC"] = df_ah["Time_UTC"].dt.tz_convert("UTC")
        df_ah = add_ah_throughput(df_ah)

        bronze = bronze.merge(df_ah[["Zeit", "Ah_throughput"]], on="Zeit", how="left")
    else:
        print(f"{cell} - no Zeit/Strom data for Ah throughput; column omitted.")

    if out_bronze_cu:
        _save_local_parquet(bronze, out_bronze_cu)
    if upload_minio:
        io_router.upload_parquet(
            minio_client, cfg, bronze,
            io_router.bronze_object_key(f"{cell}.parquet"),
            include_tag=False,
        )


def _list_cells_minio(minio_client: Minio, bucket_name: str, prefix: str) -> list:
    objects = minio_client.list_objects(bucket_name, prefix=f"{prefix}/", recursive=True)
    cells = set()
    strip_len = len(f"{prefix}/")
    for obj in objects:
        remainder = obj.object_name[strip_len:]
        if "/" in remainder:
            cells.add(remainder.split("/")[0])
    return sorted(cells)


def _list_cells_local(working_path: str) -> list:
    if not working_path or not os.path.isdir(working_path):
        return []
    reserved = {"BRONZE_CU", "preSILVER", "SILVER", "GOLD",
                "with_features_pre_labeled", "with_features_post_labeled",
                "20_export_pulse", "30_export_qocv", "40_capacity_monitore",
                "50_evaluation"}
    cells = []
    for name in sorted(os.listdir(working_path)):
        full = os.path.join(working_path, name)
        if not os.path.isdir(full) or name in reserved:
            continue
        if any(f.endswith(".parquet") for f in os.listdir(full)):
            cells.append(name)
    return cells


def run(cfg: dict, target_cells: list = None, overwrite: bool = False) -> None:
    bucket_name = cfg["bucket_name"]
    prefix = cfg["minio_prefix"]
    working_path = cfg.get("working_path")

    # Honor `upload_to` (same semantics as main.py): "local" | "minio" | "both".
    # Fall back to legacy `save_local` / `upload_s3` keys when `upload_to` is absent.
    if cfg.get("upload_to") is not None:
        save_local = io_router.writes_local(cfg)
        upload_minio = io_router.writes_minio(cfg)
    else:
        save_local = cfg.get("save_local", True)
        upload_minio = bool(cfg.get("upload_s3", False))

    if not save_local and not upload_minio:
        raise ValueError(
            "upload_to must be one of 'local', 'minio', 'both' "
            "(or set legacy save_local/upload_s3)."
        )

    download_from = (cfg.get("download_from") or "minio").lower().strip()
    if download_from not in ("local", "minio"):
        raise ValueError(f"download_from must be 'local' or 'minio', got: {download_from!r}")
    if download_from == "local" and not working_path:
        raise ValueError("working_path required when download_from='local'")

    needs_minio = download_from == "minio" or upload_minio
    minio_client = io_router.make_minio_client(cfg) if needs_minio else None

    if target_cells:
        cells = target_cells
    elif download_from == "minio":
        cells = _list_cells_minio(minio_client, bucket_name, prefix)
    else:
        cells = _list_cells_local(working_path)

    for cell in cells:
        print(f"Processing {cell}...")
        process_cell(
            cfg=cfg,
            cell=cell,
            out_bronze_cu=os.path.join(working_path, "BRONZE_CU", f"{cell}.parquet") if save_local else None,
            overwrite=overwrite,
            upload_minio=upload_minio,
            minio_client=minio_client,
            download_from=download_from,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BRONZE_CU and Ah sidecar from MinIO")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cells")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(cfg, target_cells=args.cells, overwrite=args.overwrite)
