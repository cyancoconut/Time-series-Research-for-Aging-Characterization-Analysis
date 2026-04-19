import io

from minio import Minio
from minio.error import S3Error

import urllib3


class minioConnector:
    def __init__(
        self,
        access_key=None,
        secret_key=None,
        minio_endpoint=None,
        bucket_name=None,
    ):

        self.bucket_name = bucket_name
        self.minio_client = None

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if access_key and secret_key and minio_endpoint:
            self.minio_client = Minio(
                minio_endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=True,
                cert_check=False,
            )


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
