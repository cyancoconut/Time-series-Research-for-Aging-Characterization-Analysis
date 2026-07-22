"""
Concatenate check-up (CU) parquet files for each cell.

Two modes:
  minio  — download from MinIO, save locally (default export_type="local")
           or re-upload to MinIO under 00_BRONZE_CU/
  local  — read from a local raw data root, concatenate CU files, save locally

Run from src/:
    python concat_CUs_only.py /path/to/battery_config.json --mode minio
    python concat_CUs_only.py /path/to/battery_config.json --mode local
"""

import argparse
import glob
import io
import json
import logging
import os

import duckdb
import pandas as pd
import urllib3
from minio import Minio
from minio.error import S3Error

from util.procedure_filter import matches_any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

MINIO_ENDPOINT = "iseadocker.isea.rwth-aachen.de:9000"
BUCKET_NAME = "projects"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def build_minio_client(access_key: str, secret_key: str) -> Minio:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return Minio(
        MINIO_ENDPOINT,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        cert_check=False,
    )


def upload_to_minio(minio_client: Minio, df: pd.DataFrame, object_name: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    try:
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=buf,
            length=len(buf.getvalue()),
        )
        logging.info(f"Uploaded {object_name}")
    except S3Error as err:
        logging.error(f"Upload error for {object_name}: {err}")


def list_cells_minio(minio_client: Minio, type_cell: str) -> list[str]:
    """Return sorted list of cell folder names under the cell-type prefix."""
    prefix = f"j8005-metabatt/Metabatt/{type_cell}/"
    objects = minio_client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)
    folders = set()
    for obj in objects:
        remainder = obj.object_name[len(prefix):]
        if "/" in remainder:
            folders.add(remainder.split("/")[0])
    return sorted(folders)


def download_from_minio(cfg: dict, access_key: str, secret_key: str, export_type: str = "local") -> None:
    """Download CU parquet files from MinIO and concatenate per cell."""
    type_cell = cfg["type_cell"]
    working_path = cfg["working_path"]
    procedure_filter = cfg.get("procedure_filter", "jri_CU")

    minio_client = build_minio_client(access_key, secret_key)
    cells = list_cells_minio(minio_client, type_cell)
    logging.info(f"Found {len(cells)} cells on MinIO for type '{type_cell}'")

    for cell in cells:
        savepath = os.path.join(working_path, "BRONZE_CU", f"{cell}.parquet")
        if os.path.exists(savepath):
            logging.info(f"Skipping {cell} — BRONZE_CU already exists locally")
            continue

        prefix = f"j8005-metabatt/Metabatt/{type_cell}/{cell}/"
        objects = minio_client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)
        cell_tests = [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]

        if not any(matches_any(t, procedure_filter) for t in cell_tests):
            logging.info(f"Skipping {cell} — no '{procedure_filter}' files found")
            continue

        tests = []
        for test_file in cell_tests:
            programme_name = os.path.basename(test_file).split("=")[3]
            try:
                response = minio_client.get_object(BUCKET_NAME, test_file)
                data = response.read()
                response.close()
                response.release_conn()
            except Exception as e:
                logging.warning(f"Error fetching {test_file}: {e}. Skipping.")
                continue

            df_first_row = pd.read_parquet(io.BytesIO(data)).head(1)
            if matches_any(programme_name, procedure_filter):
                df = pd.read_parquet(io.BytesIO(data))
            else:
                df = df_first_row
                df["Prozedur"] = programme_name

            tests.append(df)

        if not tests:
            logging.warning(f"No data loaded for {cell}, skipping.")
            continue

        cell_df = pd.concat(tests, ignore_index=True)

        if export_type == "server":
            object_name = f"j8005-metabatt/Metabatt/{type_cell}/00_BRONZE_CU/{cell}.parquet"
            upload_to_minio(minio_client, cell_df, object_name)
        else:
            os.makedirs(os.path.join(working_path, "BRONZE_CU"), exist_ok=True)
            cell_df.to_parquet(savepath)
            logging.info(f"Saved {savepath}")


def concat_from_local(cfg: dict, rootpath: str) -> None:
    """Concatenate CU parquet files from a local raw data root."""
    procedure_filter = cfg.get("procedure_filter", "jri_CU")
    working_path = cfg["working_path"]
    out_dir = os.path.join(working_path, "BRONZE_CU")
    os.makedirs(out_dir, exist_ok=True)

    cells = sorted(
        d for d in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, d))
    )
    logging.info(f"Found {len(cells)} cell folders in {rootpath}")

    for cell in cells:
        loadpath = os.path.join(rootpath, cell)
        cell_tests = glob.glob("*.parquet", root_dir=loadpath)

        if not any(matches_any(t, procedure_filter) for t in cell_tests):
            logging.info(f"Skipping {cell} — no '{procedure_filter}' files found")
            continue

        tests = []
        for test_file in cell_tests:
            programme_name = os.path.basename(test_file).split("=")[3]
            filepath = os.path.join(loadpath, test_file)
            try:
                df_first_row = duckdb.sql(f"SELECT * FROM '{filepath}' LIMIT 1").df()
                if matches_any(programme_name, procedure_filter):
                    df = pd.read_parquet(filepath)
                else:
                    df = df_first_row
                tests.append(df)
            except Exception as e:
                logging.warning(f"Error loading {filepath}: {e}. Removing file.")
                os.remove(filepath)
                continue

        if not tests:
            logging.warning(f"No data loaded for {cell}, skipping.")
            continue

        cell_df = pd.concat(tests, ignore_index=True)
        save_path = os.path.join(out_dir, f"{cell}.parquet")
        cell_df.to_parquet(save_path)
        logging.info(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Concatenate CU parquet files per cell.")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "--mode",
        choices=["minio", "local"],
        default="minio",
        help="Source mode: download from MinIO or read from local raw root (default: minio)",
    )
    parser.add_argument(
        "--export-type",
        choices=["local", "server"],
        default="local",
        help="[minio mode] Save locally or re-upload to MinIO (default: local)",
    )
    parser.add_argument(
        "--rootpath",
        help="[local mode] Root directory containing per-cell raw parquet folders",
    )
    parser.add_argument("--access-key", default=os.environ.get("MINIO_ACCESS_KEY"), help="MinIO access key")
    parser.add_argument("--secret-key", default=os.environ.get("MINIO_SECRET_KEY"), help="MinIO secret key")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "minio":
        if not args.access_key or not args.secret_key:
            parser.error("--access-key and --secret-key (or MINIO_ACCESS_KEY/MINIO_SECRET_KEY env vars) required for minio mode")
        download_from_minio(cfg, args.access_key, args.secret_key, export_type=args.export_type)
    else:
        rootpath = args.rootpath or os.path.join(cfg["working_path"], "raw")
        concat_from_local(cfg, rootpath)


if __name__ == "__main__":
    main()
