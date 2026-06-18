"""Aggregate per-cell capacity results into a fleet-wide CSV for evaluation.

Capacity-only port of the legacy `Export_cap_pulse.ipynb` notebook. Pulse
aggregation will be handled by a separate evaluation script.

Inputs (driven by `download_from` in the battery config):
    <working_path or minio_prefix>/40_capacity_monitore/<cell_stem>_capacity.csv
    <working_path or minio_prefix>/BRONZE_CU/<cell_stem>_manifest.json
        (the `processed` list of test filenames; the aging procedure name is
        the 4th '='-delimited filename field of the cyclic files)

Outputs (driven by `upload_to`):
    <working_path>/50_evaluation/capacity_results.csv
    MinIO: <minio_prefix>/50_evaluation/... (untagged)

Usage (from src/):
    python -m evaluation.export_cap_pulse /path/to/battery_config.json
    python -m evaluation.export_cap_pulse /path/to/battery_config.json -o /tmp/out.csv
"""

import argparse
import glob
import io
import json
import logging
import os

import pandas as pd

from output.add_information_Hina import add_additional_information
from util import io_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

EVAL_DIRNAME = "50_evaluation"


def _make_readers(cfg, source):
    """Return (cells, fetch_capacity, fetch_procedures).

    cells: list of `<stem>_capacity.csv` filenames.
    fetch_capacity(name) -> DataFrame (the capacity CSV).
    fetch_procedures(stem) -> list[str] procedure names from the BRONZE_CU
        manifest's `processed` filenames (aging + check-up), or [].
    """
    if source == "local":
        wp = cfg["working_path"]
        cap_dir = os.path.join(wp, "40_capacity_monitore")
        files = sorted(glob.glob(os.path.join(cap_dir, "*_capacity.csv")))
        cells = [os.path.basename(p) for p in files]

        def fetch_capacity(name):
            return pd.read_csv(os.path.join(cap_dir, name))

        def fetch_procedures(stem):
            path = os.path.join(wp, "BRONZE_CU", f"{stem}_manifest.json")
            if not os.path.exists(path):
                logging.warning(f"{stem}: no BRONZE_CU manifest at {path}")
                return []
            with open(path) as f:
                return _procedure_names_from_manifest(json.load(f))

        return cells, fetch_capacity, fetch_procedures

    client = io_router.make_minio_client(cfg)
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/40_capacity_monitore/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    cells = sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith("_capacity.csv")
    )

    def fetch_capacity(name):
        key = f"{base}{name}"
        response = client.get_object(bucket, key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        return pd.read_csv(io.BytesIO(data))

    def fetch_procedures(stem):
        key = f"{cfg['minio_prefix']}/BRONZE_CU/{stem}_manifest.json"
        try:
            response = client.get_object(bucket, key)
            try:
                manifest = json.loads(response.read())
            finally:
                response.close()
                response.release_conn()
        except Exception as e:
            logging.warning(f"{stem}: no BRONZE_CU manifest ({type(e).__name__})")
            return []
        return _procedure_names_from_manifest(manifest)

    return cells, fetch_capacity, fetch_procedures


def _procedure_names_from_manifest(manifest):
    """Procedure names from a BRONZE_CU manifest's `processed` filenames.

    Each processed test filename is '='-delimited; the 4th field is the
    procedure name (cyclic/aging or check-up). The aging name is selected
    downstream by `add_information.select_aging_name`.
    """
    names = []
    for fname in manifest.get("processed", []):
        parts = os.path.basename(fname).split("=")
        if len(parts) > 3:
            names.append(parts[3])
    return names


def build_capacity_table(cfg, source="local"):
    cells, fetch_capacity, fetch_procedures = _make_readers(cfg, source)
    logging.info(f"Found {len(cells)} capacity CSVs ({source})")

    frames = []
    for cell in cells:
        stem = cell.replace("_capacity.csv", "")
        try:
            df = fetch_capacity(cell)
        except Exception as e:
            logging.warning(f"{cell}: read failed: {type(e).__name__}: {e}")
            continue

        if df.empty:
            logging.info(f"{cell}: empty capacity CSV, skipping")
            continue

        procedures = fetch_procedures(stem)
        df = df.copy()
        df["Name"] = stem
        df["Procedures"] = [procedures] * len(df)
        if "CAP_start_time" in df.columns:
            df["Time"] = pd.to_datetime(df["CAP_start_time"], errors="coerce")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    add_additional_information(df_all)
    return df_all


def _write_outputs(df_all, cfg, out_dir, out_name):
    if io_router.writes_local(cfg) and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        all_path = os.path.join(out_dir, out_name)
        df_all.to_csv(all_path, index=False)
        logging.info(f"Wrote {all_path}")

    if io_router.writes_minio(cfg):
        client = io_router.make_minio_client(cfg)
        all_key = f"{EVAL_DIRNAME}/{out_name}"
        io_router.upload_csv(client, cfg, df_all, all_key, include_tag=False)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-cell capacity results across the fleet"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Override local output CSV path",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    source = cfg.get("download_from", "local")
    df_all = build_capacity_table(cfg, source=source)
    if df_all.empty:
        logging.warning("No capacity data found")
        return

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        out_name = os.path.basename(args.output)
    else:
        out_dir = os.path.join(cfg.get("working_path", "."), EVAL_DIRNAME)
        out_name = "capacity_results.csv"

    _write_outputs(df_all, cfg, out_dir, out_name)


if __name__ == "__main__":
    main()
