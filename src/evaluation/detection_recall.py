"""Check-up detection recall, and the cost of a fixed-threshold baseline.

Covers validation items 3 and 4 of the paper plan. Read-only: it runs no
pipeline stage and writes only its own CSVs.

Ground truth
------------
There is no separate test-plan document for this fleet, so the schedule is
taken from the data. ``build_bronze_cu`` decides per *test file* whether it is
a check-up (``_is_cu``: the procedure field matches ``procedure_filter``) and
stores non-check-up files as 1--2 row stubs. BRONZE_CU therefore records, for
every test file the cycler produced, whether it was a check-up --- and one test
file is one scheduled check-up. Grouping BRONZE_CU by ``Ahjo_Test_ID`` and
testing ``Prozedur`` against ``procedure_filter`` recovers that set.

Using the procedure name as ground truth to score a method that never reads it
is the correct use of metadata, not circular: the metadata says *where the
answer is*, and the method has to find it in the signal.

**This is weaker than a written schedule and the paper must say so.** It counts
check-ups *present in the data*, so a check-up that was scheduled but never ran,
or ran but failed to upload, is invisible here and inflates recall. What it does
measure honestly is: given a check-up file, does the pipeline find the check-up
inside it?

Ground truth is taken from BRONZE_CU rather than GOLD deliberately. GOLD is
downstream of segmentation, so a programme the segmenter dropped entirely would
disappear from the denominator and inflate recall.

Detections
----------
A check-up is *detected* when the pipeline emitted a capacity value for its
programme (``40_capacity_monitore/<stem>_capacity.csv``, one row per CAP).
Completeness additionally requires a pulse and both quasi-OCV branches, read
from the GOLD target column.

Fixed-threshold baseline (item 4)
---------------------------------
The foils detect the capacity test with a hard-coded absolute current --- in
BatteryLife's MICH script, a threshold near \\SI{2.37}{\\ampere}, used to
*delete* the RPT. We reproduce the structure of that approach, not the constant:
a single absolute current calibrated on one chemistry and applied unchanged to
the others. A segment is called CAP when it is a discharge whose mean current is
within ``--baseline-tol`` of ``--baseline-current`` and which lasts at least
``--baseline-min-minutes``.

This is not a defeat of prior work --- the threshold was written for a different
purpose, on one source. It quantifies what a fixed constant costs when the
chemistry changes, which is the paper's separating axis.

Usage (from src/):
    python -m evaluation.detection_recall <battery_cfg> [--cells FRAG ...]
        [--baseline-current A] [--baseline-tol F] [--baseline-min-minutes M]
        [-o OUT_DIR]
    python -m evaluation.detection_recall --pool -o OUT_DIR

Outputs (default ``<working_path>/50_evaluation``):
    detection_recall_<type_cell>_checkups.csv  one row per check-up test file
    detection_recall_<type_cell>_summary.csv   one row per cell
    detection_recall_fleet_{checkups,summary}.csv   from --pool
"""

import argparse
import glob
import json
import logging
import os

import pandas as pd
import pyarrow.parquet as pq

from util.procedure_filter import as_filter_list

# Labels a complete check-up must contain, beyond the capacity test.
_PULSE_LABELS = {"PUL", "PUL*RES"}
_QOCV_LABELS = {"qOCV_DCH", "qOCV_CHA"}

# A file matching the procedure filter but far smaller than a real check-up is
# an aborted run or a setup stub sharing the check-up's procedure name (the
# jri_TempCriterion_25 steps are ~0.5% of a check-up). Misses are reported both
# ways so that "recall" and "did a real check-up go undetected" stay separate,
# checkable statements. The primary recall figure counts every matching file.
SUBSTANTIAL_FRACTION = 0.10


def bronze_ground_truth(bronze_path, procedure_filter):
    """Per test file: is it a check-up, and which BM_Programm did it become?

    ``BM_Programm`` is reproduced exactly as ``read_and_fix_format`` assigns it
    (``groupby("Ahjo_Test_ID").ngroup()``), so the ids line up with GOLD and the
    capacity export without re-reading the payload.
    """
    filters = as_filter_list(procedure_filter)
    if filters is None:
        raise ValueError("procedure_filter is required to identify check-ups")

    cols = ["Ahjo_Test_ID", "Prozedur"]
    have = pq.ParquetFile(bronze_path).schema_arrow.names
    if "Ahjo_Test_ID" not in have:
        raise ValueError(f"{bronze_path}: no Ahjo_Test_ID column")
    df = pd.read_parquet(bronze_path, columns=cols)

    # same grouping as read_and_fix_format
    df["BM_Programm"] = df.groupby("Ahjo_Test_ID").ngroup()

    rows = []
    for (test_id, bm), grp in df.groupby(["Ahjo_Test_ID", "BM_Programm"],
                                         sort=True):
        procs = sorted({str(p) for p in grp["Prozedur"].dropna().unique()})
        is_cu = any(f in p for p in procs for f in filters)
        rows.append(dict(
            Ahjo_Test_ID=test_id, BM_Programm=bm, n_rows=len(grp),
            is_checkup=is_cu, procedures="|".join(procs),
        ))
    return pd.DataFrame(rows)


def gold_programme_labels(gold_path):
    """``{BM_Programm: set(target)}`` from GOLD, or ``{}`` if absent."""
    if not os.path.exists(gold_path):
        return {}
    df = pd.read_parquet(gold_path, columns=["BM_Programm", "target"])
    return {int(k): set(v.astype(str))
            for k, v in df.groupby("BM_Programm")["target"]}


def capacity_detections(capacity_path):
    """Set of BM_Programm for which the pipeline emitted a capacity."""
    if not os.path.exists(capacity_path):
        return set()
    df = pd.read_csv(capacity_path)
    if "BM_Programm" not in df.columns or df.empty:
        return set()
    df = df[df["Capacity_py"].notna()] if "Capacity_py" in df else df
    return set(df["BM_Programm"].astype(int))


def baseline_detections(segments_path, nom_capacity, current_a, tol,
                        min_minutes):
    """Fixed absolute-current CAP detector, applied to the segment table.

    ``Current_mean`` in the segment CSV is normalised by ``Nom_Capacity``
    (see ``feature_extraction.classification``), so it is multiplied back to
    amperes here --- the whole point of the baseline is that it reasons in
    absolute current rather than in C-rate.
    """
    if not os.path.exists(segments_path):
        return None
    df = pd.read_csv(segments_path)
    need = {"BM_Programm", "Current_mean", "Duration_minutes"}
    if not need.issubset(df.columns):
        return None
    amps = df["Current_mean"] * nom_capacity
    hit = (
        (amps < 0)                                        # a discharge
        & (amps.abs() >= current_a * (1 - tol))
        & (amps.abs() <= current_a * (1 + tol))
        & (df["Duration_minutes"] >= min_minutes)
    )
    return set(df.loc[hit, "BM_Programm"].astype(int))


def audit_cell(stem, paths, cfg, baseline):
    gt = bronze_ground_truth(paths["bronze"], cfg["procedure_filter"])
    detected = capacity_detections(paths["capacity"])
    labels = gold_programme_labels(paths["gold"])
    base = baseline_detections(
        paths["segments"], cfg["nom_capacity"], baseline["current_a"],
        baseline["tol"], baseline["min_minutes"])

    gt["cell"] = stem
    gt["cap_detected"] = gt["BM_Programm"].isin(detected)
    gt["baseline_cap_detected"] = (
        gt["BM_Programm"].isin(base) if base is not None else pd.NA)

    tgt = gt["BM_Programm"].map(lambda b: labels.get(int(b), set()))
    gt["has_pulse"] = tgt.map(lambda s: bool(s & _PULSE_LABELS))
    gt["has_qocv_both"] = tgt.map(lambda s: _QOCV_LABELS <= s)
    gt["complete"] = gt["cap_detected"] & gt["has_pulse"] & gt["has_qocv_both"]

    cu = gt[gt["is_checkup"]]
    non_cu = gt[~gt["is_checkup"]]
    n = len(cu)

    # A file matching the procedure filter but a tiny fraction of the usual
    # size is an aborted run or a setup stub (e.g. a temperature-criterion
    # step that shares the check-up procedure name), not a check-up that was
    # missed. Report misses split by size so "recall" and "did we miss a real
    # check-up" are separate, checkable statements rather than one number the
    # reader has to interpret.
    median_rows = cu.loc[cu["cap_detected"], "n_rows"].median() if n else float("nan")
    gt["frac_of_median"] = gt["n_rows"] / median_rows
    cu = gt[gt["is_checkup"]]
    substantial = cu["frac_of_median"] >= SUBSTANTIAL_FRACTION
    sub = cu[substantial]
    summary = dict(
        cell=stem,
        n_test_files=len(gt),
        n_checkups=n,
        n_cap_detected=int(cu["cap_detected"].sum()),
        recall=cu["cap_detected"].mean() if n else float("nan"),
        n_substantial=int(len(sub)),
        n_substantial_missed=int((~sub["cap_detected"]).sum()),
        recall_substantial=(sub["cap_detected"].mean() if len(sub)
                            else float("nan")),
        median_checkup_rows=median_rows,
        n_complete=int(cu["complete"].sum()),
        completeness=cu["complete"].mean() if n else float("nan"),
        n_false_detections=int(non_cu["cap_detected"].sum()),
    )
    if base is not None:
        summary.update(
            baseline_n_cap=int(cu["baseline_cap_detected"].sum()),
            baseline_recall=cu["baseline_cap_detected"].mean() if n else float("nan"),
            baseline_n_false=int(non_cu["baseline_cap_detected"].sum()),
        )
    return gt, summary


def main(config_path, cells, baseline, out_dir):
    with open(config_path) as f:
        cfg = json.load(f)
    wp = cfg["working_path"]
    type_cell = cfg.get("type_cell", "")
    out_dir = out_dir or os.path.join(wp, "50_evaluation")

    bronze_paths = sorted(glob.glob(os.path.join(wp, "BRONZE_CU", "*.parquet")))
    bronze_paths = [
        p for p in bronze_paths
        if type_cell in os.path.basename(p) and "eis" not in os.path.basename(p)
    ]
    if cells:
        bronze_paths = [p for p in bronze_paths
                        if any(c in os.path.basename(p) for c in cells)]
    if not bronze_paths:
        raise SystemExit(f"no BRONZE_CU cells matched (type_cell={type_cell!r})")

    all_cu, all_sum = [], []
    for bp in bronze_paths:
        stem = os.path.basename(bp)[: -len(".parquet")]
        # Which labelling route produced this cell's outputs. GOLD and the
        # capacity export are written to shared paths by both routes, so the
        # only marker is where the segment CSV landed: the classifier path is
        # namespaced to 60_classifier/. Recorded per cell so a mixed fleet is
        # visible rather than silently averaged.
        hdbscan_seg = os.path.join(wp, "with_features_post_labeled",
                                   stem + ".csv")
        clf_seg = os.path.join(wp, "60_classifier",
                               "with_features_post_labeled", stem + ".csv")
        if os.path.exists(hdbscan_seg):
            route, segments = "hdbscan", hdbscan_seg
        elif os.path.exists(clf_seg):
            route, segments = "classifier", clf_seg
        else:
            route, segments = "unknown", hdbscan_seg
        paths = dict(
            bronze=bp,
            gold=os.path.join(wp, "GOLD", stem + ".parquet"),
            capacity=os.path.join(wp, "40_capacity_monitore",
                                  stem + "_capacity.csv"),
            segments=segments,
        )
        if not os.path.exists(paths["capacity"]):
            logging.warning(f"{stem}: no capacity export — cell never completed "
                            "a pipeline run; excluded")
            continue
        logging.info(f"scoring {stem} (route: {route})")
        try:
            cu, summary = audit_cell(stem, paths, cfg, baseline)
        except Exception as e:
            logging.error(f"{stem}: {type(e).__name__}: {e}")
            continue
        cu["route"] = route
        summary["route"] = route
        all_cu.append(cu)
        all_sum.append(summary)

    if not all_sum:
        raise SystemExit("no cell could be scored")

    tag = type_cell or "all"
    cu_df = pd.concat(all_cu, ignore_index=True)
    sum_df = pd.DataFrame(all_sum)
    cu_df["type_cell"] = tag
    sum_df["type_cell"] = tag
    sum_df["baseline_current_a"] = baseline["current_a"]

    os.makedirs(out_dir, exist_ok=True)
    p_cu = os.path.join(out_dir, f"detection_recall_{tag}_checkups.csv")
    p_sum = os.path.join(out_dir, f"detection_recall_{tag}_summary.csv")
    cu_df.to_csv(p_cu, index=False)
    sum_df.to_csv(p_sum, index=False)

    _report(cu_df, sum_df, baseline)
    print(f"\nwrote {p_cu}\nwrote {p_sum}")


def _report(cu_df, sum_df, baseline):
    pd.set_option("display.width", 200)
    print("\n=== check-up detection, per cell ===")
    cols = ["cell", "route", "n_test_files", "n_checkups", "n_cap_detected",
            "recall", "n_complete", "completeness", "n_false_detections"]
    print(sum_df[[c for c in cols if c in sum_df]].round(3).to_string(index=False))
    if sum_df.get("route", pd.Series(dtype=str)).nunique() > 1:
        print("  NOTE: cells were labelled by different routes; see 'route'.")

    if "baseline_recall" in sum_df:
        print(f"\n=== fixed-threshold baseline "
              f"({baseline['current_a']:.3f} A "
              f"+/- {baseline['tol']:.0%}, >= {baseline['min_minutes']} min) ===")
        b = sum_df[["cell", "n_checkups", "baseline_n_cap", "baseline_recall",
                    "baseline_n_false"]]
        print(b.round(3).to_string(index=False))

    cu = cu_df[cu_df["is_checkup"]]
    missed = cu[~cu["cap_detected"]]
    if len(missed):
        print("\n=== check-ups with no capacity detected ===")
        out = missed[["cell", "BM_Programm", "n_rows", "procedures",
                      "frac_of_median"]].copy()
        out["frac_of_median"] = out["frac_of_median"].round(4)
        print(out.to_string(index=False))
        big = int((missed["frac_of_median"] >= SUBSTANTIAL_FRACTION).sum())
        print(f"\nof these, {big} exceed {SUBSTANTIAL_FRACTION:.0%} of the "
              "median check-up size; the rest are aborted runs or setup stubs "
              "sharing the check-up procedure name.")
    else:
        print("\nno check-up was missed")


def pool(out_dir):
    """Pool per-config outputs into cross-chemistry tables."""
    sums = sorted(glob.glob(os.path.join(out_dir,
                                         "detection_recall_*_summary.csv")))
    sums = [p for p in sums if "_fleet_" not in p]
    if not sums:
        raise SystemExit(f"no per-config outputs found in {out_dir}")
    S = pd.concat([pd.read_csv(p) for p in sums], ignore_index=True)
    C = pd.concat([pd.read_csv(p.replace("_summary.csv", "_checkups.csv"))
                   for p in sums], ignore_index=True)

    agg = {"cells": ("cell", "nunique"), "checkups": ("n_checkups", "sum"),
           "detected": ("n_cap_detected", "sum"),
           "substantial": ("n_substantial", "sum"),
           "substantial_missed": ("n_substantial_missed", "sum"),
           "complete": ("n_complete", "sum"),
           "false_detections": ("n_false_detections", "sum")}
    if "baseline_n_cap" in S:
        agg["baseline_detected"] = ("baseline_n_cap", "sum")
        agg["baseline_false"] = ("baseline_n_false", "sum")
    F = S.groupby("type_cell").agg(**agg)
    F["recall_pct"] = (100 * F.detected / F.checkups).round(1)
    F["recall_substantial_pct"] = (
        100 * (F.substantial - F.substantial_missed) / F.substantial).round(1)
    F["completeness_pct"] = (100 * F.complete / F.checkups).round(1)
    if "baseline_detected" in F:
        F["baseline_recall_pct"] = (
            100 * F.baseline_detected / F.checkups).round(1)

    for name, df in (("summary", F), ("checkups", C)):
        path = os.path.join(out_dir, f"detection_recall_fleet_{name}.csv")
        df.to_csv(path, index=(name == "summary"))
        print(f"wrote {path}")

    pd.set_option("display.width", 200)
    print("\n=== check-up detection recall, by chemistry ===")
    print(F.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Check-up detection recall and fixed-threshold baseline")
    ap.add_argument("config", nargs="?", help="Path to battery config JSON")
    ap.add_argument("--pool", action="store_true",
                    help="pool existing per-config outputs in --out-dir")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--baseline-current", type=float, default=1.5,
                    help="fixed absolute CAP current in A (default 1.5 = the "
                         "NMC fleet's C/2 x 3.0 Ah, i.e. calibrated on NMC)")
    ap.add_argument("--baseline-tol", type=float, default=0.05)
    ap.add_argument("--baseline-min-minutes", type=float, default=30.0)
    ap.add_argument("-o", "--out-dir", default=None)
    a = ap.parse_args()
    bl = dict(current_a=a.baseline_current, tol=a.baseline_tol,
              min_minutes=a.baseline_min_minutes)
    if a.pool:
        if not a.out_dir:
            ap.error("--pool needs -o/--out-dir")
        pool(a.out_dir)
    elif a.config:
        main(a.config, a.cells, bl, a.out_dir)
    else:
        ap.error("a battery config is required unless --pool is given")
