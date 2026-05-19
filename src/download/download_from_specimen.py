from tqdm.notebook import tqdm
import pandas as pd
from datetime import datetime
import glob

# import reload
import io
import os
import sys

from ahjo_dl.entities.test import TestFormat
import duckdb

from minio import Minio
from minio.error import S3Error

import urllib3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import io_router  # noqa: E402


def _normalize_export_type(value: str | None) -> str:
    """Map UI / legacy export_type values to io_router's upload_to vocabulary."""
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
        self.export_path = export_path + "/BRONZE"
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

    def load_tests(self, filtered_tests_name):
        """Load test data and return list of DataFrames"""
        dfs = []
        for test in tqdm(filtered_tests_name):
            file, file_size = self.ahjo.get_test(test, TestFormat.PARQUET)
            if file is not None:
                df = pd.read_parquet(file)
                df["Ahjo_Test_ID"] = test.id
                df["Ahjo_Test_Size"] = file_size
                dfs.append(df)
        return dfs

    def load_tests_incremental(self, filtered_tests_name, Downloaded_Tests):
        """Load test data incrementally and return list of DataFrames"""
        dfs = []
        for test in tqdm(filtered_tests_name):
            file, file_size = self.ahjo.get_test_incremental_downloaded(
                test, TestFormat.PARQUET, Downloaded_Tests
            )
            if file is not None:
                df = pd.read_parquet(file)
                df["Ahjo_Test_ID"] = test.id
                df["Ahjo_Test_Size"] = file_size
                dfs.append(df)
        return dfs

    def combine_tests(self, dfs):
        print("Combining DataFrames...")
        """Combine DataFrames and save to parquet"""
        combined = pd.concat(dfs)

        # Zeit column handling
        zeit_columns = combined.filter(like="Zeit").columns
        if len(zeit_columns) > 1:
            combined["Zeit"] = combined[zeit_columns[0]].fillna(
                combined[zeit_columns[1]]
            )
        else:
            combined["Zeit"] = combined[zeit_columns[0]]

        # Reorder, rename columns, and reset index
        column_to_move = combined.pop("Zeit")
        combined.insert(0, "Zeit", column_to_move)
        combined.drop(columns=zeit_columns, inplace=True)
        combined.sort_values("Zeit", inplace=True)
        combined = combined.rename(columns=lambda x: x.split("#")[0] if "#" in x else x)
        combined.reset_index(inplace=True, drop=True)

        # add BM_Programm column
        # This will create a unique identifier for each test based on Ahjo_Test_ID
        combined["BM_Programm"] = combined.groupby("Ahjo_Test_ID").ngroup()

        return combined

    def export_tests(self, dfs, specimen_name, check_up_only, export_type):
        object_name = f"{specimen_name}-PROJECT-{self.project}.parquet"

        if export_type == "local":
            if check_up_only:
                self.combine_tests(dfs).to_parquet(
                    f"{self.export_path}/CU/{object_name}"
                )
            else:
                self.combine_tests(dfs).to_parquet(f"{self.export_path}/{object_name}")

        elif export_type == "server":
            self.export_combined_to_server(dfs, object_name)

    def export_tests_incremental(
        self, dfs, df_original, specimen_name, check_up_only, export_type
    ):
        object_name = f"{specimen_name}-PROJECT-{self.project}.parquet"

        if export_type == "local":
            combined_df = pd.concat([df_original, self.combine_tests(dfs)])
            combined_df.sort_values("Zeit", inplace=True)
            combined_df = combined_df.drop_duplicates(
                subset=["Ahjo_Test_ID", "Zeit"], keep="last"
            )
            combined_df.reset_index(inplace=True, drop=True)

            if check_up_only:
                combined_df.to_parquet(f"{self.export_path}/CU/{object_name}")
            else:
                combined_df.to_parquet(f"{self.export_path}/{object_name}")

        elif export_type == "server":
            combined_df = pd.concat([df_original, self.combine_tests(dfs)])
            combined_df.sort_values("Zeit", inplace=True)
            combined_df = combined_df.drop_duplicates(
                subset=["Ahjo_Test_ID", "Zeit"], keep="last"
            )
            combined_df.reset_index(inplace=True, drop=True)
            self.export_combined_to_server(combined_df, object_name)

    def export_combined_to_server(self, dfs, object_name):

        parquet_buffer = io.BytesIO()
        self.combine_tests(dfs).to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        try:
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=parquet_buffer,
                length=len(parquet_buffer.getvalue()),
            )
            print("File uploaded successfully.")

        except S3Error as err:
            print("Upload error:", err)

    def export_to_server(self, df, object_name):

        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        try:
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=parquet_buffer,
                length=len(parquet_buffer.getvalue()),
            )
            print("File uploaded successfully.")

        except S3Error as err:
            print("Upload error:", err)

    def _export_test(
        self,
        df,
        *,
        specimen_name: str,
        filename: str,
        prefix: str,
        writes_local: bool,
        writes_minio: bool,
    ) -> None:
        """Write a single test file to local disk, MinIO, or both.

        Mirrors the routing logic of io_router.writes_local / writes_minio.
        ``prefix`` is the MinIO-side bucket-relative prefix (e.g.
        ``j8005-metabatt/Metabatt/VTC/``); ``self.export_path`` is the local
        ``<export_path>/BRONZE`` root configured in ``__init__``.
        """
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

    def download_tests(
        self, specimen, initial_download, check_up_only, export_type="local"
    ):
        # using sets assures there are no duplicates
        tests = []
        object_name = f"{specimen.name}-PROJECT-{self.project}.parquet"

        if check_up_only:
            savepath = f"{self.export_path}/CU/{object_name}"
        else:
            savepath = f"{self.export_path}/{object_name}"

        if os.path.exists(savepath):
            print(f"{specimen.name} - already exists.")

            if initial_download:
                print("Initial download is set to True, skipping this test.")

            else:
                print("Checking for new tests...")
                try:
                    Downloaded_Tests = duckdb.execute(
                        "SELECT Ahjo_Test_ID, Ahjo_Test_Size FROM read_parquet(?)",
                        [savepath],
                    ).fetchall()
                except Exception as e:
                    print(f"Error reading {savepath}: {e}. Removing file.")
                    os.remove(savepath)

                # get all tests from one specimen
                for test in self.ahjo.get_tests_from_specimen(specimen.id):
                    # only keep tests with the specified testformat
                    if (self.test_format in test.name.split("|")) and (
                        "TS" in test.name
                    ):
                        if check_up_only:
                            if test.parent:
                                if "CU" in test.parent:
                                    tests.append(test)
                        else:
                            tests.append(test)

                if tests:
                    dfs = self.load_tests_incremental(tests, Downloaded_Tests)
                    if dfs:
                        print("Found {} new tests for this specimen.".format(len(dfs)))
                        df_original = pd.read_parquet(savepath)
                        # combine the new tests with the original DataFrame
                        self.export_tests_incremental(
                            dfs, df_original, specimen.name, check_up_only, export_type
                        )

                    else:
                        print("No new tests found for this specimen.")

                else:
                    print("No tests found for this specimen.")

        else:
            print(f"Processing {specimen.name}")
            # get all tests from one specimen
            for test in self.ahjo.get_tests_from_specimen(specimen.id):
                # only keep tests with the specified testformat
                if (self.test_format in test.name.split("|")) and ("TS" in test.name):
                    if check_up_only:
                        if test.parent:
                            if "CU" in test.parent:
                                tests.append(test)
                    else:
                        tests.append(test)

            if tests:
                print("Found {} tests for this specimen.".format(len(tests)))
                dfs = self.load_tests(tests)
                self.export_tests(dfs, specimen.name, check_up_only, export_type)

            else:
                print("No tests found for this specimen.")

    def download_single_tests(
        self, specimen, export_type, prefix, include_unfinished, update_unfinished
    ):
        print(f"Processing {specimen.name}")

        # Map export_type to io_router's upload_to vocabulary, then drive
        # which location(s) we read existing files from and upload to.
        cfg = {"upload_to": _normalize_export_type(export_type)}
        writes_local = io_router.writes_local(cfg)
        writes_minio = io_router.writes_minio(cfg)

        # Replacement only happens when *both* flags are set — the UI prevents
        # update_unfinished from being checked unless include_unfinished is on,
        # but we belt-and-suspenders the check here too.
        replace_unfinished = bool(update_unfinished and include_unfinished)

        existing_test: list[str] = []

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

        # get all tests from one specimen
        for test in self.ahjo.get_tests_from_specimen(specimen.id):
            sanitized_test_name = test.name.replace("|", "_")

            if (
                (sanitized_test_name not in existing_test)  # New test
                and (
                    include_unfinished or test.finished
                )  # Download unfinished tests based on flag
                and ("TS" in test.name)
                and (self.test_format in test.name.split("|"))
            ):

                file, file_size = self.ahjo.get_test(test, TestFormat.PARQUET)
                if file is not None:
                    df = pd.read_parquet(file)

                    # Zeit column handling
                    zeit_columns = df.filter(like="Zeit").columns
                    if len(zeit_columns) > 1:
                        df["Zeit"] = df[zeit_columns[0]].fillna(df[zeit_columns[1]])
                    else:
                        df["Zeit"] = df[zeit_columns[0]]

                    df["Ahjo_Test_ID"] = test.id
                    # Reorder, rename columns, and reset index
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
