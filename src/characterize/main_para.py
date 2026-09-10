"""Initial characterization pipeline — the BOL parametrization run.

Same pipeline as :mod:`main` (dismember → features → clustering → calculate),
reading BRONZE_PARA instead of BRONZE_CU and writing every artefact under
``10_initial_characterization/<cell_stem>/`` instead of the shared GOLD /
20_export_pulse / 25_export_eis / 30_export_qocv / 40_capacity_monitore
folders. Pulse, EIS and qOCV exports are forced on — they are the point of the
run — and the BOL capacity CSV stays out of the aging monitor's folder.

    python -m characterize.main_para <battery_cfg> [--cells …]
                                     [--overwrite] [--clustering …]

Fitting is a separate step: see :mod:`characterize.fit_characterization`.
"""

import argparse
import logging

from main import load_config, run_pipeline
from util.run_context import CHARACTERIZATION


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial characterization pipeline (BRONZE_PARA → "
                    "10_initial_characterization/)"
    )
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument(
        "--cells", nargs="*", help="Optional subset of cell names to process"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess cells even if the characterization GOLD already exists",
    )
    parser.add_argument(
        "--clustering",
        choices=["auto", "hdbscan", "classifier"],
        default="auto",
        help="Clustering path: 'auto' (config classifier_model_path decides), "
             "'hdbscan' (force HDBSCAN, ignore classifier_model_path), or "
             "'classifier' (require classifier_model_path).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.clustering == "hdbscan":
        cfg["classifier_model_path"] = None
    elif args.clustering == "classifier" and not cfg.get("classifier_model_path"):
        parser.error("--clustering classifier needs classifier_model_path in the config")

    if not cfg.get(CHARACTERIZATION.procedure_filter_key):
        parser.error(
            f"{CHARACTERIZATION.procedure_filter_key} must be set in the battery "
            "config — it selects the parametrization procedures."
        )

    # Interpretation is a CU-path affordance (label-only, no GOLD/exports) and
    # would defeat the purpose here, where the exports are the deliverable.
    cfg["llm_interpret"] = False

    logging.info(
        "initial characterization: %s -> %s/<cell>/",
        CHARACTERIZATION.bronze_layer, CHARACTERIZATION.export_root_prefix,
    )
    run_pipeline(
        cfg,
        target_specimen=args.cells,
        overwrite=args.overwrite,
        run_ctx=CHARACTERIZATION,
    )


if __name__ == "__main__":
    main()
