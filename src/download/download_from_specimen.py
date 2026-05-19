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
        self, specimen, export_type, prefix, include_unfinished, update_unfinished
    ):
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
                if replace_unfinished and file_name.endswith("=unfinished.parquet"):
                    try:
                        os.remove(path)
                        print(f"  removed local unfinished file: {path}")
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
                if replace_unfinished and file_name.endswith("=unfinished.parquet"):
                    print(f"  removing MinIO unfinished file: {obj.object_name}")
                    try:
                        self.minio_client.remove_object(
                            bucket_name=self.bucket_name,
                            object_name=obj.object_name,
                        )
                    except S3Error as e:
                        print(f"  remove_object error for {obj.object_name}: {e}")
                    continue
                existing_test.append(segs[4])

        for test in self.ahjo.get_tests_from_specimen(specimen.id):
            sanitized_test_name = test.name.replace("|", "_")

            if (
                (sanitized_test_name not in existing_test)
                and (include_unfinished or test.finished)
                and ("TS" in test.name)
                and (self.test_format in test.name.split("|"))
            ):

                file, file_size = self.ahjo.get_test(test, TestFormat.PARQUET)
                if file is not None:
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

                    existing_columns = [
                        col for col in desired_columns if col in df.columns
                    ]
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
