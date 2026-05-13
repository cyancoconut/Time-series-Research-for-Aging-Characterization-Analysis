"""I/O routing for the main pipeline: local disk and/or MinIO.

Battery-config keys:
    download_from : "local" | "minio"           (default "local")
    upload_to     : "local" | "minio" | "both"  (default "local")

When uploading to MinIO, outputs land under
    <bucket>/<minio_prefix>/10_TRACY/<layer>/...
"""

import io
import os
import tempfile
from contextlib import contextmanager

import urllib3
from minio import Minio
from minio.error import S3Error

UPLOAD_PREFIX_TAG = "10_TRACY"


def needs_minio(cfg: dict) -> bool:
    return cfg.get("download_from") == "minio" or cfg.get("upload_to") in (
        "minio",
        "both",
    )


def writes_local(cfg: dict) -> bool:
    return cfg.get("upload_to", "local") in ("local", "both")


def writes_minio(cfg: dict) -> bool:
    return cfg.get("upload_to") in ("minio", "both")


def make_minio_client(cfg: dict) -> Minio:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return Minio(
        cfg["minio_endpoint"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        secure=True,
        cert_check=False,
    )


def list_bronze_cells(client: Minio, cfg: dict) -> list:
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/BRONZE_CU/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    return sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith(".parquet")
    )


def list_gold_cells(client: Minio, cfg: dict) -> list:
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/{UPLOAD_PREFIX_TAG}/GOLD/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    return sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith(".parquet")
    )


def list_gold_cells_local(working_path: str) -> list:
    import glob as _glob

    gold_dir = os.path.join(working_path, "GOLD")
    return sorted(
        os.path.basename(p) for p in _glob.glob(os.path.join(gold_dir, "*.parquet"))
    )


def fetch_gold_bytes(client: Minio, cfg: dict, cell: str) -> bytes:
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/{UPLOAD_PREFIX_TAG}/GOLD/{cell}"
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def gold_local_path(working_path: str, cell: str) -> str:
    return os.path.join(working_path, "GOLD", cell)


@contextmanager
def fetch_bronze(client: Minio, cfg: dict, cell: str):
    """Stream a BRONZE_CU object from MinIO into a tempfile; yield its path."""
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/BRONZE_CU/{cell}"
    response = client.get_object(bucket, key)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _upload_bytes(
    client: Minio, cfg: dict, key: str, payload: bytes, include_tag: bool = True
) -> None:
    bucket = cfg["bucket_name"]
    if include_tag:
        full_key = f"{cfg['minio_prefix']}/{UPLOAD_PREFIX_TAG}/{key}"
    else:
        full_key = f"{cfg['minio_prefix']}/{key}"
    try:
        client.put_object(
            bucket_name=bucket,
            object_name=full_key,
            data=io.BytesIO(payload),
            length=len(payload),
        )
        print(f"  Uploaded to S3:   {bucket}/{full_key}")
    except S3Error as e:
        print(f"  Upload error for {bucket}/{full_key}: {e}")


def upload_parquet(
    client: Minio, cfg: dict, df, key: str, include_tag: bool = True
) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    _upload_bytes(client, cfg, key, buf.getvalue(), include_tag=include_tag)


def upload_csv(client: Minio, cfg: dict, df, key: str) -> None:
    payload = df.to_csv(index=False).encode("utf-8")
    _upload_bytes(client, cfg, key, payload)


def gold_object_key(cell: str) -> str:
    return f"GOLD/{cell}"


def x_silver_object_key(cell: str) -> str:
    stem = cell.split(".")[0]
    return f"with_features_post_labeled/{stem}.csv"


def export_pulse_object_key(cell: str, filename: str) -> str:
    stem = cell.split(".")[0]
    return f"20_export_pulse/{stem}/{filename}"


def export_qocv_object_key(cell: str, filename: str) -> str:
    stem = cell.split(".")[0]
    return f"30_export_qocv/{stem}/{filename}"
