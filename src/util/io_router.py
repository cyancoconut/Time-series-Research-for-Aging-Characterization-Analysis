"""I/O routing for the main pipeline: local disk and/or MinIO.

Battery-config keys:
    download_from : "local" | "minio"           (default "local")
    upload_to     : "local" | "minio" | "both"  (default "local")

When uploading to MinIO, outputs land under
    <bucket>/<minio_prefix>/TRACY/<layer>/...
"""

import io
import logging
import os
import tempfile
from contextlib import contextmanager

import urllib3
from minio import Minio
from minio.error import S3Error

UPLOAD_PREFIX_TAG = "TRACY"

#: Pre-rename tag. Objects uploaded before the rename still live under this
#: prefix; readers fall back to it when the new prefix is empty, so nothing
#: already on MinIO becomes unreachable. Writes always use UPLOAD_PREFIX_TAG.
LEGACY_PREFIX_TAG = "10_TRACY"


def tagged_rel(rel: str) -> str:
    """Relative object dir under the current tag, e.g. `TRACY/GOLD`."""
    return f"{UPLOAD_PREFIX_TAG}/{rel.strip('/')}"


def _prefix_has_objects(client: Minio, cfg: dict, rel: str) -> bool:
    base = f"{cfg['minio_prefix']}/{rel.strip('/')}/"
    objs = client.list_objects(cfg["bucket_name"], prefix=base, recursive=False)
    return any(True for _ in objs)


def resolve_tagged_rel(client: Minio, cfg: dict, rel: str) -> str:
    """`TRACY/<rel>` if anything is there, else the legacy `10_TRACY/<rel>`.

    Read-side only. Lets a bucket written before the rename keep working
    without a bulk server-side copy.
    """
    current = tagged_rel(rel)
    if client is None:
        return current
    if _prefix_has_objects(client, cfg, current):
        return current
    legacy = f"{LEGACY_PREFIX_TAG}/{rel.strip('/')}"
    if _prefix_has_objects(client, cfg, legacy):
        logging.info("MinIO: %s empty, falling back to %s", current, legacy)
        return legacy
    return current


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
    rel = resolve_tagged_rel(client, cfg, "GOLD")
    base = f"{cfg['minio_prefix']}/{rel}/"
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
    rel = resolve_tagged_rel(client, cfg, "GOLD")
    key = f"{cfg['minio_prefix']}/{rel}/{cell}"
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


class _MinioRangeFile:
    """Seekable, read-only file-like over a MinIO object using HTTP range GETs.

    pyarrow.ParquetFile only needs read/seek/tell, so this lets the parquet
    reader fetch just the footer + the row groups it actually wants — orders
    of magnitude less network I/O than downloading the whole object.
    """

    def __init__(self, client: Minio, bucket: str, key: str):
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = client.stat_object(bucket, key).size
        self._pos = 0
        self.closed = False

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def read(self, n=-1):
        if self._pos >= self._size:
            return b""
        if n is None or n < 0:
            length = self._size - self._pos
        else:
            length = min(n, self._size - self._pos)
        if length <= 0:
            return b""
        response = self._client.get_object(
            self._bucket, self._key, offset=self._pos, length=length
        )
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        self._pos += len(data)
        return data

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def open_gold_range(client: Minio, cfg: dict, cell: str) -> _MinioRangeFile:
    """Open a GOLD parquet on MinIO as a range-read file-like object."""
    bucket = cfg["bucket_name"]
    rel = resolve_tagged_rel(client, cfg, "GOLD")
    key = f"{cfg['minio_prefix']}/{rel}/{cell}"
    return _MinioRangeFile(client, bucket, key)


def open_bronze_range(client: Minio, cfg: dict, cell: str) -> _MinioRangeFile:
    """Open a BRONZE_CU parquet on MinIO as a range-read file-like object.

    Used to peek at a single column (e.g. Prozedur) without downloading the
    whole bronze file — the procedure-filter gate can then skip cells that
    don't match before fetch_bronze pulls the full payload.
    """
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/BRONZE_CU/{cell}"
    return _MinioRangeFile(client, bucket, key)


def gold_local_path(working_path: str, cell: str) -> str:
    return os.path.join(working_path, "GOLD", cell)


def bronze_object_key(cell: str) -> str:
    return f"BRONZE_CU/{cell}"


def bronze_exists_on_minio(client: Minio, cfg: dict, cell: str) -> bool:
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/BRONZE_CU/{cell}"
    try:
        client.stat_object(bucket, key)
        return True
    except S3Error:
        return False


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


def upload_csv(client: Minio, cfg: dict, df, key: str, include_tag: bool = True) -> None:
    payload = df.to_csv(index=False).encode("utf-8")
    _upload_bytes(client, cfg, key, payload, include_tag=include_tag)


def gold_object_key(cell: str) -> str:
    return f"GOLD/{cell}"


def fetch_model_bytes(client: Minio, cfg: dict, filename: str) -> bytes:
    """Fetch a classifier artifact from `<prefix>/60_classifier/models/<filename>`.

    The trainer uploads model + meta there untagged; this is the read side used
    when `classifier_model_path` is not present locally (cross-machine inference).
    """
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/60_classifier/models/{filename}"
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def list_csv_objects(client: Minio, cfg: dict, rel_dir: str) -> list:
    """List `*.csv` basenames under `<prefix>/<rel_dir>/` on MinIO.

    Generic counterpart of `list_x_silver_cells` for arbitrary relative dirs
    (e.g. `10_TRACY/with_features_post_labeled`, `60_classifier/with_features_post_labeled`).
    """
    bucket = cfg["bucket_name"]
    base = f"{cfg['minio_prefix']}/{rel_dir.rstrip('/')}/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    return sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith(".csv")
    )


def fetch_csv_object(client: Minio, cfg: dict, rel_dir: str, name: str) -> bytes:
    """Fetch one `<prefix>/<rel_dir>/<name>` CSV payload from MinIO."""
    bucket = cfg["bucket_name"]
    key = f"{cfg['minio_prefix']}/{rel_dir.rstrip('/')}/{name}"
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def x_silver_object_key(cell: str, classifier: bool = False) -> str:
    stem = cell.split(".")[0]
    # Classifier-path CSVs go to 60_classifier/ (untagged, caller passes
    # include_tag=False) so they sit beside the model and stay out of the tagged
    # 10_TRACY/with_features_post_labeled/ that train_classifier consumes.
    if classifier:
        return f"60_classifier/with_features_post_labeled/{stem}.csv"
    return f"with_features_post_labeled/{stem}.csv"


def list_x_silver_cells(client: Minio, cfg: dict) -> list:
    """List the `with_features_post_labeled/*.csv` object names on MinIO.

    These are uploaded tagged (`upload_csv` default `include_tag=True`), so they
    live under `<prefix>/10_TRACY/with_features_post_labeled/`.
    """
    bucket = cfg["bucket_name"]
    rel = resolve_tagged_rel(client, cfg, "with_features_post_labeled")
    base = f"{cfg['minio_prefix']}/{rel}/"
    objs = client.list_objects(bucket, prefix=base, recursive=False)
    return sorted(
        os.path.basename(o.object_name)
        for o in objs
        if o.object_name.endswith(".csv")
    )


def fetch_x_silver_bytes(client: Minio, cfg: dict, name: str) -> bytes:
    """Fetch one `with_features_post_labeled/<name>.csv` payload from MinIO."""
    bucket = cfg["bucket_name"]
    rel = resolve_tagged_rel(client, cfg, "with_features_post_labeled")
    key = f"{cfg['minio_prefix']}/{rel}/{name}"
    response = client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def export_pulse_object_key(cell: str, filename: str) -> str:
    stem = cell.split(".")[0]
    return f"20_export_pulse/{stem}/{filename}"


def export_qocv_object_key(cell: str, filename: str) -> str:
    stem = cell.split(".")[0]
    return f"30_export_qocv/{stem}/{filename}"


def export_eis_object_key(cell: str, filename: str) -> str:
    stem = cell.split(".")[0]
    return f"25_export_eis/{stem}/{filename}"


def export_capacity_object_key(cell: str, filename: str) -> str:
    return f"40_capacity_monitore/{filename}"
