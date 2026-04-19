import influxdb_client
from influxdb_client import InfluxDBClient, WriteOptions
import pandas as pd


bucket = "J8005_BMWK_METABatt_Silver"
org = "BST"
token = "eAfKEG-O_Fv6-GjFlHVTPyvyB9sF4buLXAOxd0WbJElZ6-hlmIUdKpy-2pG3mfBy7gHL_StrkYPZ45ktRaUaMg=="
df = pd.read_parquet(
    r"D:\Data\METABatt\disassembled\METABatt_A123_APR18650M1B_262-PROJECT-J8005_BMWK_METABatt.parquet"
)
# df.set_index("Time", inplace=True)
df = df[
    [
        "Time",
        "Voltage",
        "Current",
        "Power",
        "Label_Procedure",
        "BM_Programm",
        "BM_Programm_procedure",
    ]
].copy()


def upload_df_to_influx(bucket, organization, token, df, measurement_name, tag_columns):
    bucket = bucket
    org = organization
    token = token

    with InfluxDBClient(url="http://localhost:8086", token=token, org=org) as _client:

        with _client.write_api(
            write_options=WriteOptions(
                batch_size=500,
                flush_interval=10_000,
                jitter_interval=2_000,
                retry_interval=5_000,
                max_retries=5,
                max_retry_delay=30_000,
                max_close_wait=300_000,
                exponential_base=2,
            )
        ) as _write_client:

            # Write the dataframe to InfluxDB
            _write_client.write(
                bucket,
                org,
                record=df,
                data_frame_measurement_name=measurement_name,
                data_frame_tag_columns=tag_columns,
                data_frame_timestamp_column="Time",
            )


upload_df_to_influx(
    bucket,
    org,
    token,
    df,
    measurement_name="METABatt_A123_APR18650M1B_262",
    tag_columns=["BM_Programm", "BM_Programm_procedure"],
)
