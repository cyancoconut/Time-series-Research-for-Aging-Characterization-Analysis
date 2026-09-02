"""Export EIS impedance spectra, one bundled parquet per BM_Programm.

EIS measurements live as standalone files (csv/parquet) in the cell's download
folder — the cycler timeline only carries an ``EIS`` *label* marking when a
measurement was triggered, not the impedance data itself. This stage:

  1. discovers the cell's EIS files (``util.io_eis`` marker) from the configured
     source (local ``<working_path>/<cell_stem>/`` or MinIO
     ``<minio_prefix>/<cell_stem>/``),
  2. reduces each to a settled per-frequency spectrum,
  3. matches each measurement to a ``BM_Programm`` by nearest EIS-labelled
     segment time (the ``EIS`` target / EIS-procedure rows in GOLD), and
  4. writes one bundled parquet per program with every matched spectrum stacked,
     each tagged by ``eis_number`` / ``Time`` / ``U`` so they can be split back
     out downstream.

Filename: <cell_stem>_eis_BM<BM_Programm>_<SOH>SOH.parquet
Local:    <working_path>/25_export_eis/<cell_stem>/
MinIO:    <minio_prefix>/25_export_eis/<cell_stem>/  (untagged)

Gated by the ``export_eis`` config flag (default off). Mirrors the pulse/qOCV
export routing (``download_from`` for the source, ``upload_to`` for writes).
"""

import logging
import os

import pandas as pd

from util import io_eis
from util import io_router
from util.run_context import CU, RunContext


def _discover_and_load(cell_stem, cfg, minio_client, marker):
    """Load every EIS spectrum for a cell. Returns ``(specs, metas)`` lists."""
    source = cfg.get("download_from", "local")
    specs, metas = [], []

    if source == "minio":
        names = io_eis.list_eis_files_minio(minio_client, cfg, cell_stem, marker)
        loaders = [
            (n, io_eis.fetch_eis_bytes(minio_client, cfg, n)) for n in names
        ]
    else:
        root = cfg.get("working_path")
        if not root:
            logging.warning("export_eis: working_path unset — cannot read EIS files")
            return specs, metas
        loaders = [
            (p, p) for p in io_eis.list_eis_files_local(root, cell_stem, marker)
        ]

    for name, src in loaders:
        spec, meta = io_eis.load_eis_spectrum(src, filename=name, marker=marker)
        if spec is None:
            logging.warning(
                f"export_eis: {os.path.basename(str(name))} has no EIS spectrum "
                f"columns / no settled points — skipping"
            )
            continue
        specs.append(spec)
        metas.append(meta)
    return specs, metas


def _eis_anchors(df_gold, proc_filter):
    """EIS-labelled segments as ``(BM_Programm, Time, ID)`` anchors for matching.

    Uses the ``EIS`` target rows and, as a fallback/complement, any segment
    whose ``Prozedur`` contains the EIS procedure filter (some cells carry the
    EIS marker only in the procedure name, not as an ``EIS`` target).

    ``ID`` rides along so the bundle can record *which* segment each spectrum
    was measured in: ``util.soc_from_steps`` reads the segment one procedure
    number earlier to get the SOC step that preceded the measurement. Absent
    from GOLD (older exports), the column is simply not carried and the SOC
    falls back to the qOCV mapping.
    """
    cols = ["BM_Programm", "Time"]
    if df_gold is None or df_gold.empty:
        return pd.DataFrame(columns=cols)
    mask = df_gold["target"].astype(str) == "EIS"
    if "Prozedur" in df_gold.columns and proc_filter:
        mask = mask | df_gold["Prozedur"].astype(str).str.contains(
            proc_filter, na=False
        )
    if "ID" in df_gold.columns:
        cols.append("ID")
    return df_gold.loc[mask, cols]


def export_eis(df_gold, soh, cell, cfg, paths, minio_client, run_ctx: RunContext = CU):
    stem = cell.split(".")[0]
    marker = cfg.get("eis_file_marker", io_eis.DEFAULT_EIS_FILE_MARKER)
    proc_filter = cfg.get("eis_procedure_filter", "EIS")
    tol = cfg.get("eis_match_tolerance_minutes", 120)

    specs, metas = _discover_and_load(stem, cfg, minio_client, marker)
    if not specs:
        logging.info(f"{cell}: no EIS measurements found to export")
        return

    anchors = _eis_anchors(df_gold, proc_filter)
    matched = io_eis.match_spectra_to_anchors(metas, anchors, tol)

    n_unmatched = 0
    keep = []
    for spec, meta, anchor in zip(specs, metas, matched):
        if anchor is None or anchor.get("BM_Programm") is None:
            n_unmatched += 1
            logging.warning(
                f"{cell}: EIS {meta.get('eis_number')} at {meta.get('meas_time')} "
                f"matched no EIS-labelled segment within {tol} min — skipping"
            )
            continue
        spec = spec.copy()
        spec["BM_Programm"] = int(anchor["BM_Programm"])
        if anchor.get("ID") is not None:
            # The segment this spectrum was measured in — the anchor for the
            # step-counted SOC (util.soc_from_steps).
            spec["segment_ID"] = str(anchor["ID"])
        keep.append(spec)

    if not keep:
        logging.warning(f"{cell}: no EIS measurements matched a BM_Programm")
        return

    all_spec = pd.concat(keep, ignore_index=True)

    local_dir = paths["export_eis_dir"] if paths else None
    write_local = io_router.writes_local(cfg) and local_dir
    write_minio = io_router.writes_minio(cfg)
    if write_local:
        os.makedirs(local_dir, exist_ok=True)

    for bm_prog, group in all_spec.groupby("BM_Programm"):
        soh_val = soh.get(bm_prog, "NA")
        if soh_val == "NA":
            logging.warning(
                f"{cell}: BM_Programm={bm_prog} has no CAP capacity, EIS SOH=NA"
            )
        filename = f"{stem}_eis_BM{bm_prog}_{soh_val}SOH.parquet"
        group = group.sort_values(["eis_number", "frequency"]).reset_index(drop=True)

        if write_local:
            local_path = os.path.join(local_dir, filename)
            group.to_parquet(local_path, index=False)
            logging.info(
                f"{cell}: export_eis -> {local_path} "
                f"({group['eis_number'].nunique()} measurement(s))"
            )
        if write_minio:
            key = io_router.export_eis_object_key(
                cell, filename, root=run_ctx.export_root(cell)
            )
            io_router.upload_parquet(minio_client, cfg, group, key, include_tag=False)

    if n_unmatched:
        logging.info(f"{cell}: export_eis — {n_unmatched} EIS measurement(s) unmatched")
