"""Build BRONZE_PARA — the parametrization (initial characterization) layer.

Same builder as :mod:`build_bronze_cu_with_ah`, pointed at a different set of
test files: the ones whose 4th '='-delimited filename field matches
``para_procedure_filter`` (a list of substrings; one element is the normal
case). Output goes to ``<working_path>/BRONZE_PARA/`` and, when uploading,
``<minio_prefix>/BRONZE_PARA/``.

    python download/build_bronze_para.py <battery_cfg> [--cells …] [--overwrite]

No ``--incremental`` here (unlike the CU builder): a parametrization run is a
one-off BOL measurement, not a growing aging series, so there is nothing to
append to. Rebuild with ``--overwrite``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download.build_bronze_cu_with_ah import run

LAYER = "BRONZE_PARA"
FILTER_KEY = "para_procedure_filter"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BRONZE_PARA (initial-characterization layer) and Ah sidecar"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--cells", nargs="*", help="Optional subset of cells")
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild even if output exists"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if not cfg.get(FILTER_KEY):
        parser.error(
            f"{FILTER_KEY} must be set in the battery config — it lists the "
            "programme-name substrings that mark a parametrization test file."
        )

    run(
        cfg,
        target_cells=args.cells,
        overwrite=args.overwrite,
        layer=LAYER,
        filter_key=FILTER_KEY,
    )


if __name__ == "__main__":
    main()
