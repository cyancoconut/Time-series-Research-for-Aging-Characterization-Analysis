"""
Validate pipeline labels against the UConn-ILCC ground truth.

The UConn RPT CSVs carry a ``Segment Key`` column (ref_chg / ref_dchg / slowpulse /
fastpulse / '-') that build_bronze_uconn preserves into BRONZE_CU as ``Segment_Key``.
The pipeline itself ignores it (it clusters/classifies blind), so this script scores
the blind ``target`` against that ground truth *after the fact*.

GOLD is a per-row time series with ``Time`` + ``ID`` + ``target`` but no Segment_Key
(the pipeline drops unknown columns). We recover the truth by joining GOLD back to
BRONZE on the timestamp, then reduce to one row per segment (majority Segment_Key,
the segment's ``target``) and cross-tabulate.

Usage:
    python paper/validate_uconn_labels.py \
        --working-path "/path/UConn_pipeline" \
        --cells cell_01 cell_02 cell_03
"""

import os
import argparse

import pandas as pd


def _majority(s: pd.Series):
    vc = s.value_counts()
    return vc.index[0] if len(vc) else None


def validate_cell(working_path: str, type_cell: str, cell: str):
    """Return a per-segment frame (ID, target, Segment_Key, Pulse_SOC) for one cell."""
    stem = f"{type_cell}_{cell}"
    gold_path = os.path.join(working_path, "GOLD", f"{stem}.parquet")
    bronze_path = os.path.join(working_path, "BRONZE_CU", f"{stem}.parquet")
    if not (os.path.exists(gold_path) and os.path.exists(bronze_path)):
        return None

    gold = pd.read_parquet(gold_path, columns=["Time", "ID", "target"])
    bronze = pd.read_parquet(bronze_path, columns=["Zeit", "Segment_Key", "Pulse_SOC"])
    # Segment_Key is constant within a segment; one row per timestamp is enough.
    bronze = bronze.drop_duplicates("Zeit").rename(columns={"Zeit": "Time"})

    merged = gold.merge(bronze, on="Time", how="left")
    seg = merged.groupby("ID").agg(
        target=("target", "first"),
        Segment_Key=("Segment_Key", _majority),
        Pulse_SOC=("Pulse_SOC", _majority),
        n_rows=("Time", "size"),
    ).reset_index()
    seg["cell"] = cell
    return seg


def main():
    ap = argparse.ArgumentParser(description="Validate UConn labels vs Segment_Key")
    ap.add_argument("--working-path", required=True)
    ap.add_argument("--type-cell", default="UConn")
    ap.add_argument("--cells", nargs="+", default=["cell_01", "cell_02", "cell_03"])
    ap.add_argument("-o", "--out", default=None,
                    help="CSV for the per-segment table (default 50_evaluation/)")
    args = ap.parse_args()

    frames = []
    for cell in args.cells:
        seg = validate_cell(args.working_path, args.type_cell, cell)
        if seg is None:
            print(f"{cell} - GOLD or BRONZE missing, skipped.")
            continue
        frames.append(seg)
    if not frames:
        print("No cells validated.")
        return
    allseg = pd.concat(frames, ignore_index=True)

    # Fill unlabeled ground-truth ('-' = CV tails / inter-pulse / initial dchg).
    allseg["Segment_Key"] = allseg["Segment_Key"].fillna("-").replace("-", "(unlabeled)")

    ct = pd.crosstab(allseg["Segment_Key"], allseg["target"])
    print("\n=== Segment_Key (ground truth, rows) x pipeline target (cols) ===")
    print(ct.to_string())

    # Headline: how cleanly does the pipeline's CAP map to ref_dchg?
    print("\n=== CAP purity ===")
    cap = allseg[allseg["target"] == "CAP"]
    if len(cap):
        print(cap["Segment_Key"].value_counts().to_string())
        pure = (cap["Segment_Key"] == "ref_dchg").mean() * 100
        print(f"CAP segments that are ref_dchg: {pure:.1f}%  (n={len(cap)})")
    ref_d = allseg[allseg["Segment_Key"] == "ref_dchg"]
    if len(ref_d):
        recall = (ref_d["target"] == "CAP").mean() * 100
        print(f"ref_dchg segments labeled CAP:  {recall:.1f}%  (n={len(ref_d)})")

    out = args.out or os.path.join(args.working_path, "50_evaluation",
                                   "uconn_label_validation.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allseg.to_csv(out, index=False)
    print(f"\nPer-segment table -> {out}")


if __name__ == "__main__":
    main()
