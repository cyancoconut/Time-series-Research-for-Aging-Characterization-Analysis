"""Compare HDBSCAN vs classifier segment labels for the same cells.

Both paths write a per-segment ``with_features_post_labeled/<stem>.csv`` keyed
on ``ID``; the classifier path is routed to ``60_classifier/`` (see main.py
``_build_paths``) so the two label sets sit side by side without overwriting.
This script joins them per cell and surfaces *where they disagree* — in
particular whether the classifier recovers CAP segments (check-ups) that
HDBSCAN missed.

Why this matters: the classifier was trained on HDBSCAN's own labels, so a
metric scored against those labels (e.g. the LOCO report) cannot show a
recovered CAP — it counts as a false positive. The only way to see improvement
is to diff the two label sets against the structural ground truth that *each
BM_Programm with a check-up yields exactly one CAP*.

Usage (from src/):
    python -m evaluation.compare_labels /path/to/battery_config.json
    # --source {local,minio}  (default: config's download_from)
    # local overrides:
    #   --hdbscan-dir DIR   (default <working_path>/with_features_post_labeled)
    #   --classifier-dir DIR(default <working_path>/60_classifier/with_features_post_labeled)
    #   -o, --out-dir DIR   (default <working_path>/50_evaluation)

On ``--source minio`` the two label sets are read from their MinIO keys
(HDBSCAN from the tagged ``<prefix>/10_TRACY/with_features_post_labeled/``,
classifier from the untagged ``<prefix>/60_classifier/with_features_post_labeled/``)
via io_router; the diff CSVs are still written locally to --out-dir.

Outputs (under --out-dir):
    label_diff_segments.csv  — every segment where the two labels disagree,
                               with the features that explain the call.
    cap_count_diff.csv       — per (cell, BM_Programm) CAP counts for both
                               paths and their delta.
"""

import argparse
import io
import json
import logging
import os

import pandas as pd

from util import io_router

# MinIO relative dirs for the two label sets (under <minio_prefix>/).
_HDBSCAN_REL = f"{io_router.UPLOAD_PREFIX_TAG}/with_features_post_labeled"  # 10_TRACY/...
_CLASSIFIER_REL = "60_classifier/with_features_post_labeled"

# Features that make a disagreement interpretable (a real CAP is a long,
# CAP-rate discharge preceded by a full charge -> high prev_end_voltage_norm).
_CONTEXT_COLS = [
    "BM_Programm",
    "Duration_minutes",
    "Current_mean",
    "abs_Current_mean",
    "Voltage_max",
    "Voltage_min",
    "Voltage_range",
    "prev_end_voltage_norm",
]


def _coerce(stem: str, df: pd.DataFrame, out: dict) -> None:
    if "ID" not in df.columns or "target" not in df.columns:
        logging.warning(f"{stem}.csv: missing ID/target column, skipping")
        return
    df["target"] = df["target"].astype(str)
    out[stem] = df


def _load_dir(path: str) -> dict:
    """Map cell stem -> DataFrame for every <stem>.csv under local ``path``."""
    out = {}
    if not path or not os.path.isdir(path):
        return out
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".csv"):
            continue
        _coerce(fname[: -len(".csv")], pd.read_csv(os.path.join(path, fname)), out)
    return out


def _load_minio(client, cfg: dict, rel_dir: str) -> dict:
    """Map cell stem -> DataFrame for every <stem>.csv under <prefix>/<rel_dir>/."""
    out = {}
    for name in io_router.list_csv_objects(client, cfg, rel_dir):
        data = io_router.fetch_csv_object(client, cfg, rel_dir, name)
        _coerce(name[: -len(".csv")], pd.read_csv(io.BytesIO(data)), out)
    return out


def _merge_cell(h: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Inner-join one cell's HDBSCAN and classifier rows on ID."""
    keep_ctx = [col for col in _CONTEXT_COLS if col in h.columns]
    left = h[["ID", "target", *keep_ctx]].rename(columns={"target": "target_hdbscan"})
    right = c[["ID", "target"]].rename(columns={"target": "target_classifier"})
    return left.merge(right, on="ID", how="inner")


def compare(h_cells: dict, c_cells: dict) -> dict:
    both = sorted(set(h_cells) & set(c_cells))
    classifier_only = sorted(set(c_cells) - set(h_cells))  # HDBSCAN skipped these
    hdbscan_only = sorted(set(h_cells) - set(c_cells))

    diff_frames, cap_frames = [], []
    for cell in both:
        merged = _merge_cell(h_cells[cell], c_cells[cell])
        merged.insert(0, "cell", cell)

        disagree = merged[merged["target_hdbscan"] != merged["target_classifier"]]
        if not disagree.empty:
            diff_frames.append(disagree)

        # CAP count per BM_Programm for each path (expect 1 per check-up program).
        if "BM_Programm" in merged.columns:
            h_cap = (
                merged[merged["target_hdbscan"] == "CAP"]
                .groupby("BM_Programm").size()
            )
            c_cap = (
                merged[merged["target_classifier"] == "CAP"]
                .groupby("BM_Programm").size()
            )
            cap = pd.DataFrame({"cap_hdbscan": h_cap, "cap_classifier": c_cap})
            cap = cap.fillna(0).astype(int).reset_index()
            cap.insert(0, "cell", cell)
            cap_frames.append(cap)

    # CAP counts for classifier-only cells (HDBSCAN wrote no CSV at all, so its
    # CAP count is 0 by construction). This is the headline recovery number:
    # whether the cells HDBSCAN gave up on actually got check-ups labeled.
    recovered_frames = []
    for cell in classifier_only:
        c = c_cells[cell]
        if "BM_Programm" not in c.columns:
            continue
        c_cap = c[c["target"] == "CAP"].groupby("BM_Programm").size()
        rec = c_cap.reset_index(name="cap_classifier")
        rec.insert(0, "cell", cell)
        recovered_frames.append(rec)

    diffs = (
        pd.concat(diff_frames, ignore_index=True)
        if diff_frames
        else pd.DataFrame(columns=["cell", "ID", "target_hdbscan", "target_classifier"])
    )
    caps = (
        pd.concat(cap_frames, ignore_index=True)
        if cap_frames
        else pd.DataFrame(columns=["cell", "BM_Programm", "cap_hdbscan", "cap_classifier"])
    )
    if not caps.empty:
        caps["cap_delta"] = caps["cap_classifier"] - caps["cap_hdbscan"]

    caps_recovered = (
        pd.concat(recovered_frames, ignore_index=True)
        if recovered_frames
        else pd.DataFrame(columns=["cell", "BM_Programm", "cap_classifier"])
    )

    # Label-transition counts (hdbscan -> classifier) over disagreements.
    if not diffs.empty:
        transitions = (
            diffs.groupby(["target_hdbscan", "target_classifier"])
            .size().reset_index(name="n").sort_values("n", ascending=False)
        )
    else:
        transitions = pd.DataFrame(columns=["target_hdbscan", "target_classifier", "n"])

    return {
        "both": both,
        "classifier_only": classifier_only,
        "hdbscan_only": hdbscan_only,
        "diffs": diffs,
        "caps": caps,
        "caps_recovered": caps_recovered,
        "transitions": transitions,
    }


def _print_report(res: dict) -> None:
    print("\n=== Cell coverage ===")
    print(f"compared (in both)      : {len(res['both'])}")
    print(f"classifier-only cells   : {len(res['classifier_only'])}"
          "  <- HDBSCAN wrote no CSV (skipped: no CAP cluster) — pure recoveries")
    for cell in res["classifier_only"]:
        print(f"    + {cell}")
    print(f"hdbscan-only cells      : {len(res['hdbscan_only'])}")
    for cell in res["hdbscan_only"]:
        print(f"    - {cell}")

    caps = res["caps"]
    print("\n=== CAP-count diff per (cell, BM_Programm) ===")
    if caps.empty:
        print("(no BM_Programm column / no shared cells)")
    else:
        recovered = caps[caps["cap_delta"] > 0]
        lost = caps[caps["cap_delta"] < 0]
        print(f"programs where classifier found MORE CAP (recovered): {len(recovered)}")
        if not recovered.empty:
            print(recovered.to_string(index=False))
        print(f"\nprograms where classifier found FEWER CAP (dropped): {len(lost)}")
        if not lost.empty:
            print(lost.to_string(index=False))

    rec = res["caps_recovered"]
    print("\n=== CAP counts in classifier-only cells (HDBSCAN skipped these) ===")
    if rec.empty:
        print("(no classifier-only cells, or no CAP labeled in them)")
    else:
        n_cells = rec["cell"].nunique()
        n_caps = int(rec["cap_classifier"].sum())
        print(f"{n_cells} recovered cells with {n_caps} CAP check-ups the classifier "
              "found where HDBSCAN produced nothing:")
        print(rec.to_string(index=False))
        cells_with_cap = set(rec["cell"])
        empty = [c for c in res["classifier_only"] if c not in cells_with_cap]
        if empty:
            print(f"\nclassifier-only cells with NO CAP found ({len(empty)}):")
            for c in empty:
                print(f"    ! {c}")

    print("\n=== Label transitions (HDBSCAN -> classifier) over disagreements ===")
    if res["transitions"].empty:
        print("(no disagreements)")
    else:
        print(res["transitions"].to_string(index=False))

    print(f"\ntotal disagreeing segments: {len(res['diffs'])}")


def main(config_path: str, source, hdbscan_dir, classifier_dir, out_dir) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    wp = cfg.get("working_path")
    source = source or cfg.get("download_from", "local")
    out_dir = out_dir or (os.path.join(wp, "50_evaluation") if wp else ".")

    if source == "minio":
        client = io_router.make_minio_client(cfg)
        logging.info(f"HDBSCAN    : minio <prefix>/{_HDBSCAN_REL}/")
        logging.info(f"classifier : minio <prefix>/{_CLASSIFIER_REL}/")
        h_cells = _load_minio(client, cfg, _HDBSCAN_REL)
        c_cells = _load_minio(client, cfg, _CLASSIFIER_REL)
    else:
        hdbscan_dir = hdbscan_dir or (
            os.path.join(wp, "with_features_post_labeled") if wp else None
        )
        classifier_dir = classifier_dir or (
            os.path.join(wp, "60_classifier", "with_features_post_labeled") if wp else None
        )
        logging.info(f"HDBSCAN dir   : {hdbscan_dir}")
        logging.info(f"classifier dir: {classifier_dir}")
        h_cells = _load_dir(hdbscan_dir)
        c_cells = _load_dir(classifier_dir)

    res = compare(h_cells, c_cells)
    _print_report(res)

    os.makedirs(out_dir, exist_ok=True)
    diff_path = os.path.join(out_dir, "label_diff_segments.csv")
    cap_path = os.path.join(out_dir, "cap_count_diff.csv")
    rec_path = os.path.join(out_dir, "cap_recovered_cells.csv")
    res["diffs"].to_csv(diff_path, index=False)
    res["caps"].to_csv(cap_path, index=False)
    res["caps_recovered"].to_csv(rec_path, index=False)
    print(f"\nwrote {diff_path}")
    print(f"wrote {cap_path}")
    print(f"wrote {rec_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Diff HDBSCAN vs classifier labels")
    parser.add_argument("config", help="Path to battery config JSON")
    parser.add_argument("--source", choices=["local", "minio"], default=None,
                        help="Where to read the label CSVs (default: config download_from)")
    parser.add_argument("--hdbscan-dir", default=None, help="local source only")
    parser.add_argument("--classifier-dir", default=None, help="local source only")
    parser.add_argument("-o", "--out-dir", default=None)
    args = parser.parse_args()
    main(args.config, args.source, args.hdbscan_dir, args.classifier_dir, args.out_dir)
