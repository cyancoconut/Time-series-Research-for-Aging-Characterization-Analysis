import os
from ahjo_dl.source.ahjo_source import AhjoSource
from get_user_input import get_config_from_gui, load_config_from_file
from download_from_specimen import SpecimenDownloader
import datetime
import glob

config_file_path = None  # set value to None -> opens a GUI that creates a config file
if config_file_path is None:

    result = get_config_from_gui()

else:
    result = load_config_from_file(config_file_path)

if result:
    (
        project,
        target_specimen,
        testformat,
        ahjo_endpoint,
        ahjo_key,
        minio_endpoint,
        access_key,
        secret_key,
        bucket_name,
        minio_prefix,
        export_type,
        export_path,
        test_type_filter,
        test_name_filter,
    ) = result
    minio_prefix = (minio_prefix or "").strip().strip("/")
    if export_type in ("minio", "both", "server") and not minio_prefix:
        raise SystemExit(
            "minio_prefix is required when export_type uploads to MinIO "
            "(e.g. 'j8005-metabatt/Metabatt/VTC')."
        )

ahjo = AhjoSource(ahjo_endpoint, ahjo_key)
ahjo_project = ahjo.get_project(project)
testsets = []
specimens = []

for testset in ahjo.list_testsets(ahjo_project):
    testsets.append(testset.name)

for specimen in ahjo.list_specimens(ahjo_project):
    specimens.append(specimen)

target_subset = [
    items
    for items in specimens
    if not target_specimen or any(target in items.name for target in target_specimen)
]


SpecimenDownloader = SpecimenDownloader(
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

for specimen in target_subset:

    try:
        initial_download = 0
        include_unfinished = False
        update_unfinished = True

        SpecimenDownloader.download_single_tests(
            specimen,
            export_type,
            prefix=f"{minio_prefix}/",
            include_unfinished=include_unfinished,
            update_unfinished=update_unfinished,
            test_type_filter=test_type_filter,
            test_name_filter=test_name_filter,
        )

    except Exception as e:
        print(f"Download failed: {str(e)}")
        # log the error
        log_path = os.path.join(export_path, "download_errors.log")
        with open(log_path, "a") as log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] Error downloading {specimen}: {str(e)}\n")
        continue
