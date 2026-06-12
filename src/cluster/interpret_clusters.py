"""LLM cluster interpretation — name HDBSCAN clusters with an LLM (advisory).

Reads every per-segment ``with_features_post_labeled/<cell>.csv`` (HDBSCAN
path), groups rows by ``(cell, cluster_id)``, builds a compact per-cluster
feature signature, and asks the configured LLM backend (``llm_provider`` in
the battery config — OpenAI default, Anthropic available; see
:mod:`util.llm_client`) to name each cluster. The LLM invents its **own**
free-form label (not constrained to the pipeline taxonomy), so the result is
an independent second opinion next to ``target``. **Augment-only**: results land
in three new columns (``llm_label`` / ``llm_confidence`` / ``llm_rationale``)
written back to the same CSV — ``target``, ``cluster_id``, and every numeric
result stay untouched. All clusters are interpreted (CAP/PUL/qOCV too); the
audit CSV puts ``llm_label`` next to the majority ``target`` per cluster for
human review of the rule labels.

Identical (rounded) signatures are deduplicated across cells, so one API call
covers every cluster that looks the same.

Outputs:
- ``with_features_post_labeled/<cell>.csv``: ``llm_*`` columns added in place
  (local file rewritten / same MinIO key re-uploaded, per ``upload_to``).
- ``<working_path>/50_evaluation/cluster_interpretation.csv``: one row per
  (cell, cluster) — n, majority target, llm_label, confidence, rationale —
  always written locally; uploaded untagged when ``upload_to`` includes minio.

Usage (from src/):
    python -m cluster.interpret_clusters /path/to/battery_config.json
    # --source {local,minio}   (default: config's download_from)
    # --cells FRAGMENT [...]   subset of cells by name fragment
    # --overwrite              re-interpret cells whose CSV already has llm_label
    # --dry-run                build + print signatures, no API calls, no writes
    # -o, --out-dir DIR        audit CSV dir (default <working_path>/50_evaluation)

Credentials: ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` env var (or
``openai_api_key`` / ``anthropic_api_key`` in the gitignored root
``config.json``).
"""

import argparse
import json
import logging
import os

import pandas as pd

from cluster.train_classifier import _iter_cell_csvs, bootstrap_leftover_labels
from util import io_router
from util.llm_client import ClusterLabel, make_llm_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Per-cluster signature basis — the same scale-free columns the classifier
# uses, summarized as distributions over the cluster's members.
_SIGNATURE_COLS = [
    "Current_mean",
    "abs_Current_mean",
    "Voltage_max",
    "Voltage_min",
    "Voltage_range",
    "Duration_minutes",
    "Duration_quartile",
    "prev_end_voltage_norm",
]

_LLM_COLS = ["llm_label", "llm_confidence", "llm_rationale"]


def build_signature(group: pd.DataFrame) -> dict:
    """Compact, rounded signature of one cluster (used as the dedupe key too)."""
    sig = {"n_segments": int(len(group))}
    for col in _SIGNATURE_COLS:
        if col not in group.columns:
            continue
        values = group[col].dropna()
        if values.empty:
            continue
        sig[col] = {
            "mean": round(float(values.mean()), 3),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
        }
    sig["majority_target"] = str(group["target"].astype(str).mode().iat[0])
    if "bootstrap_label" in group.columns:
        sig["bootstrap_label"] = str(group["bootstrap_label"].mode().iat[0])
    return sig


def _signature_key(sig: dict) -> str:
    """Dedupe key: identical rounded signatures share one API call."""
    return json.dumps(sig, sort_keys=True)


def _add_bootstrap_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the rule-bootstrap name on a copy without touching ``target``."""
    tmp = df.copy()
    tmp["target"] = tmp["target"].astype(str)
    tmp = bootstrap_leftover_labels(tmp)
    df["bootstrap_label"] = tmp["target"].to_numpy()
    return df


def interpret_cell(df: pd.DataFrame, client, cache: dict) -> pd.DataFrame:
    """Fill the ``llm_*`` columns for one cell's segments, cluster by cluster."""
    for col in _LLM_COLS:
        df[col] = None
    for cluster_id, group in df.groupby("cluster_id", dropna=False):
        sig = build_signature(group)
        key = _signature_key(sig)
        if key not in cache:
            cache[key] = client.interpret_cluster(sig)
        result: ClusterLabel = cache[key]
        df.loc[group.index, "llm_label"] = result.label
        df.loc[group.index, "llm_confidence"] = result.confidence
        df.loc[group.index, "llm_rationale"] = result.rationale
        logging.info(
            f"  cluster {cluster_id}: n={len(group)} "
            f"majority={sig['majority_target']} -> llm={result.label} "
            f"({result.confidence:.2f})"
        )
    return df


def _write_cell_csv(cfg: dict, source: str, cell: str, df: pd.DataFrame) -> None:
    """Write the augmented CSV back where it came from (in-place augment)."""
    if source == "minio":
        client = io_router.make_minio_client(cfg)
        io_router.upload_csv(client, cfg, df, f"with_features_post_labeled/{cell}.csv")
        return
    path = os.path.join(cfg["working_path"], "with_features_post_labeled", f"{cell}.csv")
    df.to_csv(path, index=False)


def main(config_path, source=None, cells=None, overwrite=False, dry_run=False, out_dir=None):
    with open(config_path) as f:
        cfg = json.load(f)
    source = source or cfg.get("download_from", "local")
    wp = cfg.get("working_path")
    out_dir = out_dir or (os.path.join(wp, "50_evaluation") if wp else ".")

    client = None if dry_run else make_llm_client(cfg)
    cache: dict = {}
    audit_rows = []
    n_done = n_skipped = 0

    for cell, df in _iter_cell_csvs(cfg, source):
        if cells and not any(fragment in cell for fragment in cells):
            continue
        if "cluster_id" not in df.columns or "target" not in df.columns:
            logging.warning(
                f"{cell}: no cluster_id/target column — skipping. CSVs predating "
                "#37 must be regenerated (re-run main.py) before interpretation."
            )
            continue
        if not overwrite and "llm_label" in df.columns and df["llm_label"].notna().any():
            logging.info(f"{cell}: already interpreted (use --overwrite) — skipping")
            n_skipped += 1
            continue

        df = _add_bootstrap_column(df)
        logging.info(f"{cell}: {df['cluster_id'].nunique(dropna=False)} clusters")

        if dry_run:
            for cluster_id, group in df.groupby("cluster_id", dropna=False):
                sig = build_signature(group)
                print(f"--- {cell} / cluster {cluster_id} ---")
                print(json.dumps(sig, indent=2))
            continue

        df = interpret_cell(df, client, cache)
        df = df.drop(columns=["bootstrap_label"])
        _write_cell_csv(cfg, source, cell, df)
        n_done += 1

        for cluster_id, group in df.groupby("cluster_id", dropna=False):
            majority = str(group["target"].astype(str).mode().iat[0])
            audit_rows.append({
                "cell": cell,
                "cluster_id": cluster_id,
                "n_segments": len(group),
                "majority_target": majority,
                "llm_label": group["llm_label"].iat[0],
                "llm_confidence": group["llm_confidence"].iat[0],
                "llm_rationale": group["llm_rationale"].iat[0],
            })

    if dry_run:
        logging.info("dry run — no API calls made, nothing written")
        return

    if not audit_rows:
        logging.info(f"nothing interpreted ({n_skipped} cells skipped) — audit CSV left untouched")
        return
    audit = pd.DataFrame(audit_rows)
    os.makedirs(out_dir, exist_ok=True)
    audit_path = os.path.join(out_dir, "cluster_interpretation.csv")
    audit.to_csv(audit_path, index=False)
    logging.info(
        f"{n_done} cells interpreted ({n_skipped} skipped), "
        f"{len(cache)} unique signatures -> {len(cache)} API calls"
    )
    logging.info(f"wrote {audit_path}")

    if io_router.writes_minio(cfg):
        mc = io_router.make_minio_client(cfg)
        io_router.upload_csv(
            mc, cfg, audit, "50_evaluation/cluster_interpretation.csv", include_tag=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM cluster interpretation (advisory)")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--source", choices=["local", "minio"], default=None,
                        help="Where to read/write the label CSVs (default: config download_from)")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Subset of cells by name fragment")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-interpret cells whose CSV already carries llm_label")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print per-cluster signatures without calling the LLM")
    parser.add_argument("-o", "--out-dir", default=None,
                        help="Audit CSV dir (default <working_path>/50_evaluation)")
    args = parser.parse_args()

    main(args.config, args.source, args.cells, args.overwrite, args.dry_run, args.out_dir)
