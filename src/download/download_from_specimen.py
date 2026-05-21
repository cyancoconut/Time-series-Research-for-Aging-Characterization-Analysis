import glob
import io
import os
import sys
from datetime import datetime

import pandas as pd
import urllib3
from ahjo_dl.entities.test import TestFormat
from minio.error import S3Error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import io_router  # noqa: E402


def _normalize_export_type(value):
    v = (value or "").lower().strip()
    if v == "server":
        return "minio"
    if v in ("local", "minio", "both"):
        return v
    return "local"


class SpecimenDownloader:
    def __init__(
        self,
        ahjo,
        project_name,
        export_path,
        test_format,
        access_key=None,
        secret_key=None,
        minio_endpoint=None,
        minio_secure=True,
        bucket_name=None,
    ):
        self.ahjo = ahjo
        self.project = project_name
        self.export_path = export_path
        self.test_format = test_format
        self.bucket_name = bucket_name
        self.minio_client = None
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if access_key and secret_key and minio_endpoint:
            self.minio_client = io_router.make_minio_client(
                {
                    "minio_endpoint": minio_endpoint,
                    "minio_access_key": access_key,
                    "minio_secret_key": secret_key,
                }
            )

    def _export_test(
        self,
        df,
        *,
        specimen_name,
        filename,
        prefix,
        writes_local,
        writes_minio,
    ):
        if writes_local:
            local_dir = f"{self.export_path}/{specimen_name}"
            os.makedirs(local_dir, exist_ok=True)
            local_path = f"{local_dir}/{filename}"
            df.to_parquet(local_path, index=False)
            print(f"  Saved locally:    {local_path}")

        if writes_minio:
            if self.minio_client is None:
                print("  Skipping S3 upload: no MinIO client configured.")
                return
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            object_name = f"{prefix}{specimen_name}/{filename}"
            try:
                self.minio_client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    data=buf,
                    length=len(buf.getvalue()),
                )
                print(f"  Uploaded to S3:   {self.bucket_name}/{object_name}")
            except S3Error as err:
                print(f"  Upload error for {object_name}: {err}")

    def download_single_tests(
        self,
        specimen,
        export_type,
        prefix,
        include_unfinished,
        update_unfinished,
        redownload=False,
        temperature_column=None,
    ):
        # redownload=True forces a fresh fetch of every test: existing parquets
        # (finished and unfinished) are deleted so they drop out of the
        # existing_test skip-list and are downloaded again. Use it to re-pull
        # data after a downloader fix (e.g. a newly retained column).
        print(f"Processing {specimen.name}")

        cfg = {"upload_to": _normalize_export_type(export_type)}
        writes_local = io_router.writes_local(cfg)
        writes_minio = io_router.writes_minio(cfg)
        replace_unfinished = update_unfinished and include_unfinished

        existing_test = []

        if writes_local:
            specimen_dir = f"{self.export_path}/{specimen.name}"
            for path in glob.glob(f"{specimen_dir}/**/*.parquet", recursive=True):
                file_name = os.path.basename(path)
                segs = file_name.split("=")
                if len(segs) < 5:
                    continue
                if redownload or (
                    replace_unfinished
                    and file_name.endswith("=unfinished.parquet")
                ):
                    reason = "redownload" if redownload else "unfinished refresh"
                    try:
                        os.remove(path)
                        print(f"  removed local file ({reason}): {path}")
                    except OSError as e:
                        print(f"  could not remove {path}: {e}")
                    continue
                existing_test.append(segs[4])

        if writes_minio:
            objects = self.minio_client.list_objects(
                bucket_name=self.bucket_name,
                prefix=f"{prefix}{specimen.name}/",
                recursive=True,
            )
            for obj in objects:
                file_name = os.path.basename(obj.object_name)
                segs = file_name.split("=")
                if len(segs) < 5:
                    continue
                if redownload or (
                    replace_unfinished
                    and file_name.endswith("=unfinished.parquet")
                ):
                    reason = "redownload" if redownload else "unfinished refresh"
                    print(f"  removing MinIO file ({reason}): {obj.object_name}")
                    try:
                        self.minio_client.remove_object(
                            bucket_name=self.bucket_name,
                            object_name=obj.object_name,
                        )
                    except S3Error as e:
                        print(f"  remove_object error for {obj.object_name}: {e}")
                    continue
                existing_test.append(segs[4])

        print(f"  Listing tests for {specimen.name} ...")
        all_tests = list(self.ahjo.get_tests_from_specimen(specimen.id))
        candidates = [
            t
            for t in all_tests
            if (t.name.replace("|", "_") not in existing_test)
            and (include_unfinished or t.finished)
            and ("TS" in t.name)
            and (self.test_format in t.name.split("|"))
        ]
        print(
            f"  {len(all_tests)} tests returned, "
            f"{len(candidates)} new candidates to fetch"
        )

        for i, test in enumerate(candidates, 1):
            sanitized_test_name = test.name.replace("|", "_")
            print(f"  [{i}/{len(candidates)}] Fetching {test.name} ...")
            file, file_size = self.ahjo.get_test(test, TestFormat.PARQUET)
            if file is None:
                continue
            df = pd.read_parquet(file)

            zeit_columns = df.filter(like="Zeit").columns
            if len(zeit_columns) > 1:
                df["Zeit"] = df[zeit_columns[0]].fillna(df[zeit_columns[1]])
            else:
                df["Zeit"] = df[zeit_columns[0]]

            df["Ahjo_Test_ID"] = test.id
            column_to_move = df.pop("Zeit")
            df.insert(0, "Zeit", column_to_move)
            df.drop(columns=zeit_columns, inplace=True)
            df.sort_values("Zeit", inplace=True)
            df = df.rename(columns=lambda x: x.split("#")[0] if "#" in x else x)

            # The temperature channel is exported under inconsistent names
            # ("T1", or the German "Temperatur" — sometimes with a sensor-
            # channel suffix, e.g. "Temperatur_ / PBOC1"). Exact-name matching
            # misses those, so the column was being dropped by the
            # desired_columns whitelist below. Normalise the chosen temperature
            # column to "T1" so it survives the whitelist and is renamed to
            # "Temperature" downstream in read_and_fix_format.
            #
            # A test may carry several temperature-like columns (e.g.
            # "Temperatur / C" alongside "Temperatur_ / PBOC1") where only one
            # is the cell temperature. Set the `temperature_column` config key
            # to the exact raw column name to pick it explicitly; otherwise the
            # first temperature-like column is used and a warning is logged.
            if "T1" not in df.columns:
                chosen = None
                if temperature_column:
                    wanted = str(temperature_column).strip()
                    matches = [
                        c for c in df.columns if str(c).strip() == wanted
                    ]
                    if matches:
                        chosen = matches[0]
                    else:
                        print(
                            f"  temperature_column {temperature_column!r} not "
                            f"found in test columns; falling back to heuristic"
                        )
                if chosen is None:
                    temp_cols = [
                        c
                        for c in df.columns
                        if str(c).strip().lower().startswith("temp")
                    ]
                    if temp_cols:
                        if len(temp_cols) > 1:
                            print(
                                f"  multiple temperature columns {temp_cols}; "
                                f"using {temp_cols[0]!r} as T1 — set the "
                                f"'temperature_column' config key to choose"
                            )
                        chosen = temp_cols[0]
                if chosen is not None:
                    df = df.rename(columns={chosen: "T1"})

            desired_columns = [
                "Zeit",
                "Spannung",
                "Strom",
                "T1",
                "Prozedur",
                "Zustand",
                "AhAkku",
                "Ahjo_Test_ID",
            ]

            existing_columns = [col for col in desired_columns if col in df.columns]
            df = df[existing_columns]
            df.reset_index(inplace=True, drop=True)

            status = "finished" if test.finished else "unfinished"
            object_name = f"{self.project}={specimen.name}={datetime.fromtimestamp(test.startDate).strftime('%Y-%m-%d_%H%M%S')}={test.parent}={sanitized_test_name}={test.equipment.name}=filesize-{file_size}={status}.parquet"

            self._export_test(
                df,
                specimen_name=specimen.name,
                filename=object_name,
                prefix=prefix,
                writes_local=writes_local,
                writes_minio=writes_minio,
            )
