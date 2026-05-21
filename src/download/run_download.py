"""Headless download runner.

Same logic as ``download_GUI.py`` but takes the config JSON as a CLI argument
instead of opening the configuration GUI. Used by ``pipeline_ui.py``.

Usage (from src/):
    python download/run_download.py /path/to/download_config.json

The JSON has the same shape as the file ``pipeline_ui.py`` saves:
    project, target_cell, cell_type, testformat,
    ahjo_endpoint, ahjo_key,
    minio_endpoint, access_key, secret_key, bucket_name, minio_prefix,
    export_type, export_path,
    include_unfinished, update_unfinished

Legacy configs that still use ``target_specimen`` instead of ``target_cell``
are accepted on load.
"""

import argparse
import datetime
import json
import os
import sys

from ahjo_dl.source.ahjo_source import AhjoSource

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_from_specimen import SpecimenDownloader


def run(cfg: dict) -> None:
    project = cfg["project"]
    target_cell = cfg.get("target_cell", cfg.get("target_specimen", [""]))
    cell_type = (cfg.get("cell_type") or "").strip()
    if not cell_type:
        raise SystemExit(
            "cell_type is required (MinIO path segment, e.g. 'VTC' or 'JGNE')."
        )
    testformat = cfg["testformat"]
    ahjo_endpoint = cfg["ahjo_endpoint"]
    ahjo_key = cfg["ahjo_key"]
    minio_endpoint = cfg["minio_endpoint"]
    access_key = cfg["access_key"]
    secret_key = cfg["secret_key"]
    bucket_name = cfg["bucket_name"]
    minio_prefix = (cfg.get("minio_prefix") or "").strip().strip("/")
    export_type = cfg["export_type"]
    if export_type in ("minio", "both", "server") and not minio_prefix:
        raise SystemExit(
            "minio_prefix is required when export_type uploads to MinIO "
            "(e.g. 'j8005-metabatt/Metabatt/VTC')."
        )
    export_path = cfg["export_path"]
    include_unfinished = bool(cfg.get("include_unfinished", False))
    update_unfinished = bool(cfg.get("update_unfinished", True))
    redownload = bool(cfg.get("redownload", False))
    temperature_column = cfg.get("temperature_column")

    ahjo = AhjoSource(ahjo_endpoint, ahjo_key)
    ahjo_project = ahjo.get_project(project)

    specimens = list(ahjo.list_specimens(ahjo_project))
    name_fragments = [t for t in (target_cell or []) if t]
    target_subset = [
        s for s in specimens
        if cell_type in s.name
        and (not name_fragments or any(t in s.name for t in name_fragments))
    ]

    print(f"Project: {project}")
    print(f"Cell type (MinIO prefix segment): {cell_type}")
    print(f"Target cell filter: {target_cell}")
    print(f"Matched {len(target_subset)} / {len(specimens)} specimens")
    print(f"include_unfinished={include_unfinished}, update_unfinished={update_unfinished}")

    downloader = SpecimenDownloader(
        ahjo,
        project,
        export_path,
        testformat,
        access_key=access_key,
        secret_key=secret_key,
        minio_endpoint=minio_endpoint,
        minio_secure=False,
        bucket_name=bucket_name,
    )

    prefix = f"{minio_prefix}/"
    print(f"MinIO prefix: {prefix}")

    for specimen in target_subset:
        print(f"--- Downloading {specimen.name} ---")
        try:
            downloader.download_single_tests(
                specimen,
                export_type,
                prefix=prefix,
                include_unfinished=include_unfinished,
                update_unfinished=update_unfinished,
                redownload=redownload,
                temperature_column=temperature_column,
            )
        except Exception as e:
            print(f"Download failed for {specimen}: {e}")
            log_path = os.path.join(export_path, "download_errors.log")
            try:
                os.makedirs(export_path, exist_ok=True)
                with open(log_path, "a") as log_file:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{ts}] Error downloading {specimen}: {e}\n")
            except Exception as log_err:
                print(f"  (also failed to write error log: {log_err})")
            continue

    print("Download run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless download from Ahjo / MinIO")
    parser.add_argument("config", help="Path to download config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(cfg)
