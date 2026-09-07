"""Measure how much of METAbatt's *boundary detection* depends on protocol
metadata (``Prozedur`` / ``Zustand``) rather than on signal + physical config.

Why this exists
---------------
The paper claims the pipeline finds check-up structure from the measured signal,
needing only a handful of physical cell/test scalars — not a step-by-step
protocol description (PyProBE's ``README.yaml``) and not per-source code
(BatteryLife). Segmentation is where that claim is most exposed:
``DismemblerFunctions.dismembling`` fires a boundary on four rules, and one of
them (``Prozedur != Prozedur.shift()``) is the cycler's own procedure-name
column telling the segmenter where the cuts are.

This module quantifies the exposure with an ablation ladder, read-only. It runs
no pipeline stage and writes nothing outside its output directory.

    L0  production: long-PAU start, long-PAU exit, Prozedur change,
        Zustand change inside a qocv_procedure_filter match.
    L1  L0 minus the Prozedur-change rule. Isolates ``Prozedur``.
    L2  L1 with ``Zustand`` replaced by a state derived from the current:
        |I| < i_tol -> PAU, else sign(I) -> CHA/DCH, and the qOCV rule
        ungated (fires on every state change). No protocol column is read.

L2 is not simply "L1 with more removed". It removes the last protocol column
*and* promotes the qOCV rule to a general signal rule, so it can place
boundaries L1 cannot — which is the point: the question is not how much a
stripped-down segmenter loses, but whether the signal carries the same
structure the protocol columns were supplying.

Fidelity gate
-------------
``_dismember_at_level`` is a level-parameterised **copy** of
``DismemblerFunctions.dismembling``; the production segmenter is deliberately
left untouched. The copy is kept honest by a hard gate: its L0 output is
compared row for row against ``dismember_raw_cell`` run live on the same input,
and any single differing row aborts the cell. Comparing against a live run
rather than a stored artifact makes the gate immune to stale CSVs — the label
CSVs on disk predate the current segmenter, which is exactly the trap this
avoids.

Production labels are joined in afterwards, for reporting only: they say which
L0 segments are CAP / PUL / qOCV, and never feed the gate. Segments carrying a
dismember-time pre-label (PAU / EIS / AGING) never reach
``with_features_post_labeled`` at all, so ``pre_target`` is carried alongside to
tell a legitimately-absent label apart from a stale one.

Residual metadata dependencies NOT ablated here (stated for honesty, they sit
outside boundary detection):
  * ``prefiltering`` drops rows with ``Zustand in {SAVE, REST}`` — row
    filtering, not boundary placement.
  * ``BM_Programm`` comes from ``Ahjo_Test_ID`` — file identity, not protocol
    semantics (``dismember_raw_cell`` already falls back to a single group).
  * L2 preserves ``EIS``/``FLOATER`` states, which ``read_and_fix_format``
    assigns from the presence of instrument channel columns, not from any
    procedure text. Row counts are reported so this can be judged.

Usage (from src/):
    python -m evaluation.segmentation_audit /path/to/battery_config.json
    #   --cells FRAG [FRAG ...]   subset by name fragment
    #   --i-tol-frac F            L2 pause threshold as a fraction of
    #                             nom_capacity (default 1e-3, i.e. C/1000)
    #   --boundary-tol N          boundary match window in rows (default 2)
    #   -o, --out-dir DIR         default <working_path>/50_evaluation

Outputs (under --out-dir):
    segmentation_audit_<type_cell>_summary.csv   one row per (cell, level)
    segmentation_audit_<type_cell>_segments.csv  one row per L0 segment, its
                                                 fate at L1 and L2
    segmentation_audit_<type_cell>_zustand.csv   Zustand vs sign(I), i_tol sweep

Run it once per battery config, then pool the results across chemistries:
    python -m evaluation.segmentation_audit --pool -o <dir>
which writes segmentation_audit_fleet_{boundary,checkup,rule3,
zustand_confusion}.csv — the tables the ablation is reported as.
"""

import argparse
import glob
import json
import logging
import os

import contextlib
import io as _io

import numpy as np
import pandas as pd

from dismember.cluster_preparation import DismemblerFunctions, allocate_IDs
from dismember.dismember_raw_cell import dismember_raw_cell, read_and_fix_format

@contextlib.contextmanager
def _quiet():
    """Swallow the production segmenter's per-procedure print() chatter."""
    with contextlib.redirect_stdout(_io.StringIO()):
        yield


PAU_COLUMNS = ["PAU", "PAUO", "..."]
LEVELS = ("L0", "L1", "L2")

# Final labels (post target-sync) that the paper's claim actually rests on.
# EIS/PAU/AGING segments never reach with_features_post_labeled (they carry a
# dismember-time pre-label, and feature extraction keeps only target == -1), so
# they cannot appear here.
CHECKUP_TARGETS = ("CAP", "PUL", "PUL*RES", "qOCV_DCH", "qOCV_CHA")

# i_tol sweep for the Zustand-vs-signal report, as fractions of nom_capacity.
I_TOL_SWEEP = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def derive_state(df, i_tol, preserve=("EIS", "FLOATER")):
    """State column from the current alone: |I| < i_tol -> PAU, else sign.

    ``preserve`` keeps states that ``read_and_fix_format`` assigned from the
    presence of instrument channel columns (EIS / floater hardware), which are
    not protocol text. Everything else is overwritten from the signal.
    """
    cur = df["Current"].astype(float)
    state = np.where(cur.abs() < i_tol, "PAU", np.where(cur > 0, "CHA", "DCH"))
    state = pd.Series(state, index=df.index, dtype=object)
    keep = df["Zustand"].astype(str).isin(preserve)
    state[keep] = df.loc[keep, "Zustand"].astype(str)
    return state


def _dismember_at_level(df_cell, min_rows, pau_duration, qocv_filter, level, i_tol):
    """Level-parameterised copy of ``DismemblerFunctions.dismembling``.

    Returns ``(dismembered_df, rule_counts)``. ``rule_counts`` accumulates how
    many rows each L0 boundary rule fired on, and how many of the Prozedur-rule
    firings no other rule would have produced.

    Kept byte-faithful to production at L0 — see the module docstring's
    fidelity gate. Differences by level are confined to the three marked
    ``level`` branches.
    """
    use_prozedur_rule = level == "L0"          # rule 3, the ablated leak
    use_prozedur_state = level in ("L0", "L1")  # Zustand + "_EIS_" relabel
    ungated_state_rule = level == "L2"

    processed = []
    rules = {"pau_start": 0, "pau_exit": 0, "prozedur": 0, "qocv": 0,
             "prozedur_unique": 0, "total": 0}
    # rows where the Prozedur rule fired alone, collected to check whether the
    # boundary survives the discard-bucket / min_rows post-processing at all

    for _, programm_df in df_cell.groupby("BM_Programm"):
        if len(programm_df) <= 2:
            programm_df = programm_df.copy()
            programm_df["BM_Programm_procedure"] = 0
            programm_df["pre_target"] = "AGING"
            processed.append(programm_df)
            continue
        if programm_df.empty or len(programm_df) < min_rows:
            continue

        programm_df = programm_df.copy()
        # Level-dependent state column. L0/L1 use the cycler's Zustand; L2
        # rebuilds it from the current.
        if not use_prozedur_state:
            programm_df["Zustand"] = derive_state(programm_df, i_tol)

        if (programm_df["Zustand"].isin(PAU_COLUMNS)).all():
            continue

        programm_df["ZUSTAND_group"] = (
            programm_df["Zustand"] != programm_df["Zustand"].shift()
        ).cumsum()

        zustand_informations = (
            programm_df.groupby("ZUSTAND_group")
            .agg({
                "Zustand": "first",
                "Time_UTC": lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds() / 60,
            })
            .reset_index()
        )
        zustand_informations.columns = [
            "ZUSTAND_group", "Zustand", "ZUSTAND_Duration_minutes",
        ]
        programm_df = programm_df.merge(
            zustand_informations[["ZUSTAND_group", "ZUSTAND_Duration_minutes"]],
            on="ZUSTAND_group", how="left",
        ).reset_index(drop=True)

        mask_pau = programm_df["Zustand"].isin(PAU_COLUMNS)
        if use_prozedur_state:
            # Production relabels rests inside an "_EIS_" procedure as EIS.
            # That reads Prozedur, so L2 cannot do it.
            mask_eis = programm_df["Prozedur"].str.contains("_EIS_", na=False)
            programm_df.loc[mask_pau & mask_eis, ["Zustand"]] = "EIS"
            mask_pau = programm_df["Zustand"].isin(PAU_COLUMNS)

        group_change = (
            programm_df["ZUSTAND_group"] != programm_df["ZUSTAND_group"].shift()
        )

        pau_start = (
            mask_pau
            & (programm_df["ZUSTAND_Duration_minutes"] > pau_duration)
            & (group_change | (programm_df.index == 0))
        )
        pau_exit = (
            group_change
            & programm_df["Zustand"].shift().isin(PAU_COLUMNS)
            & (programm_df["ZUSTAND_Duration_minutes"].shift() > pau_duration)
        )
        proz_change = programm_df["Prozedur"] != programm_df["Prozedur"].shift()

        if ungated_state_rule:
            qocv_boundary = group_change
        elif qocv_filter:
            qocv_boundary = (
                programm_df["Prozedur"].str.contains(qocv_filter, na=False)
                & group_change
            )
        else:
            qocv_boundary = pd.Series(False, index=programm_df.index)

        if level == "L0":
            rules["pau_start"] += int(pau_start.sum())
            rules["pau_exit"] += int(pau_exit.sum())
            rules["prozedur"] += int(proz_change.sum())
            rules["qocv"] += int(qocv_boundary.sum())
            uniq = proz_change & ~(pau_start | pau_exit | qocv_boundary)
            rules["prozedur_unique"] += int(uniq.sum())
            rules["_uniq_idx"] = rules.get("_uniq_idx", [])
            rules["_uniq_idx"].append(programm_df.loc[uniq, "_row"].tolist())

        start = pau_start | pau_exit | qocv_boundary
        if use_prozedur_rule:
            start = start | proz_change
        rules["total"] += int(start.sum())

        programm_df["BM_Programm_procedure"] = start.cumsum()

        for _, pau_group in programm_df[mask_pau].groupby("ZUSTAND_group"):
            duration = pau_group["ZUSTAND_Duration_minutes"].iloc[0]
            if duration <= pau_duration:
                programm_df.loc[pau_group.index, "BM_Programm_procedure"] = 0
            else:
                programm_df.loc[pau_group.index[1:-1], "BM_Programm_procedure"] = 0

        pure_pau_proc = programm_df.groupby("BM_Programm_procedure")[
            "Zustand"
        ].transform(lambda x: x.isin(PAU_COLUMNS).all())
        if "pre_target" not in programm_df.columns:
            programm_df["pre_target"] = pd.NA
        programm_df.loc[pure_pau_proc, "pre_target"] = "PAU"

        pure_eis_proc = programm_df.groupby("BM_Programm_procedure")[
            "Zustand"
        ].transform(lambda x: (x == "EIS").all())
        programm_df.loc[pure_eis_proc, "pre_target"] = "EIS"

        for df_name, df_procedure in programm_df.groupby("BM_Programm_procedure"):
            if df_procedure["Zustand"].isin(PAU_COLUMNS).all():
                continue
            if (df_procedure.shape[0] < min_rows) and (
                not (df_procedure["Zustand"] == "EIS").any()
            ):
                programm_df.loc[
                    programm_df["BM_Programm_procedure"] == df_name,
                    "BM_Programm_procedure",
                ] = 0

        programm_df.drop(["ZUSTAND_group"], axis=1, inplace=True)
        processed.append(programm_df)

    if not processed:
        return pd.DataFrame(), rules
    out = allocate_IDs(pd.concat(processed, ignore_index=True))
    return out, rules


# ---------------------------------------------------------------- metrics ---

def _boundary_rows(seg_by_row):
    """Row positions where the segment ID changes, on the raw row sequence."""
    ids = seg_by_row.sort_index()
    changed = ids.ne(ids.shift())
    changed.iloc[0] = False  # the first row is not a boundary
    return set(ids.index[changed].tolist())


def boundary_scores(ref, test, tol):
    """Precision / recall / F1 of ``test`` boundaries against ``ref``.

    A test boundary counts as matched when a reference boundary sits within
    ``tol`` rows of it, absorbing the one-sample ambiguity between "last row of
    A" and "first row of B".
    """
    if not ref:
        return dict(precision=np.nan, recall=np.nan, f1=np.nan,
                    n_ref=0, n_test=len(test))
    ref_arr = np.array(sorted(ref))

    def _near(xs):
        if not xs:
            return 0
        idx = np.searchsorted(ref_arr, sorted(xs))
        hits = 0
        for x, i in zip(sorted(xs), idx):
            for j in (i - 1, i):
                if 0 <= j < len(ref_arr) and abs(int(ref_arr[j]) - int(x)) <= tol:
                    hits += 1
                    break
        return hits

    tp_test = _near(test)
    # recall is measured the other way round so a many-to-one collapse cannot
    # inflate it
    test_arr = np.array(sorted(test)) if test else np.array([])
    hit_ref = 0
    for r in ref_arr:
        if test_arr.size and np.min(np.abs(test_arr - r)) <= tol:
            hit_ref += 1
    precision = tp_test / len(test) if test else np.nan
    recall = hit_ref / len(ref_arr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else 0.0
    )
    return dict(precision=precision, recall=recall, f1=f1,
                n_ref=len(ref_arr), n_test=len(test))


def _segment_rowsets(dis, drop_bucket0=True):
    """``{ID: set(row)}`` from a dismembered frame, keyed on the stable _row."""
    out = {}
    for seg_id, grp in dis.groupby("ID", sort=False):
        if drop_bucket0 and str(seg_id).endswith("_0"):
            continue
        out[seg_id] = set(grp["_row"].tolist())
    return out


def segment_fates(ref_sets, test_sets):
    """Fate of every reference segment under a test segmentation.

    ``exact``  the same rows survive as one segment (IoU >= 0.99)
    ``merged`` its best match also swallows rows from other reference segments
    ``split``  two or more test segments carve it up
    ``both``   merged and split at once
    ``lost``   no test segment overlaps it (all its rows fell to the discard
               bucket)
    """
    # row -> reference segment, to detect contamination
    row_owner = {}
    for sid, rows in ref_sets.items():
        for r in rows:
            row_owner[r] = sid

    rows_out = []
    for sid, rows in ref_sets.items():
        overlaps = {}
        for tid, trows in test_sets.items():
            n = len(rows & trows)
            if n:
                overlaps[tid] = n
        if not overlaps:
            rows_out.append(dict(ID=sid, n_pieces=0, best_id=None, iou=0.0,
                                 overlap=0, contamination=0, fate="lost"))
            continue
        best_id = max(overlaps, key=overlaps.get)
        best = test_sets[best_id]
        inter = len(rows & best)
        iou = inter / len(rows | best)
        # rows the match adds that belong to a *different* reference segment
        contamination = sum(
            1 for r in best - rows if row_owner.get(r) not in (None, sid)
        )
        n_pieces = len(overlaps)
        if iou >= 0.99:
            fate = "exact"
        elif contamination and n_pieces >= 2:
            fate = "both"
        elif contamination:
            fate = "merged"
        elif n_pieces >= 2:
            fate = "split"
        else:
            fate = "exact" if iou >= 0.99 else "trimmed"
        rows_out.append(dict(ID=sid, n_pieces=n_pieces, best_id=best_id,
                             iou=iou, overlap=inter,
                             contamination=contamination, fate=fate))
    return pd.DataFrame(rows_out)


def zustand_vs_signal(df, nom_capacity):
    """Row-level agreement between ``Zustand`` and a signal-derived state.

    Returns (confusion long-form, i_tol sweep). The sweep reports how well the
    zero-current threshold reproduces the cycler's PAU class, which is what the
    pause boundary rules key on.
    """
    truth = df["Zustand"].astype(str)
    conf_rows, sweep_rows = [], []
    for frac in I_TOL_SWEEP:
        pred = derive_state(df, frac * nom_capacity)
        is_pau_true = truth.isin(PAU_COLUMNS)
        is_pau_pred = pred.isin(PAU_COLUMNS)
        tp = int((is_pau_true & is_pau_pred).sum())
        fp = int((~is_pau_true & is_pau_pred).sum())
        fn = int((is_pau_true & ~is_pau_pred).sum())
        sweep_rows.append(dict(
            i_tol_frac=frac, i_tol_A=frac * nom_capacity,
            pau_precision=tp / (tp + fp) if tp + fp else np.nan,
            pau_recall=tp / (tp + fn) if tp + fn else np.nan,
            n_pau_true=int(is_pau_true.sum()), n_pau_pred=int(is_pau_pred.sum()),
        ))
        if frac == 1e-3:
            conf = (
                pd.crosstab(truth, pred)
                .stack()
                .reset_index()
            )
            conf.columns = ["Zustand", "signal_state", "n_rows"]
            conf = conf[conf["n_rows"] > 0]
            conf_rows.append(conf)
    conf_df = conf_rows[0] if conf_rows else pd.DataFrame()
    return conf_df, pd.DataFrame(sweep_rows)


# ----------------------------------------------------------------- driver ---

def audit_cell(bronze_path, prod_seq, cfg, i_tol_frac, boundary_tol):
    """Run the L0/L1/L2 ladder for one cell. Returns (summary rows, segment
    rows, zustand frames, rule counts) or raises RuntimeError if the L0
    fidelity gate fails."""
    min_rows = cfg.get("min_rows", 20)
    pau_duration = cfg.get("pau_duration", 9.9)
    qocv_filter = cfg.get("qocv_procedure_filter")
    nom_capacity = cfg["nom_capacity"]
    i_tol = i_tol_frac * nom_capacity

    df = read_and_fix_format(bronze_path, cfg["v_max"])
    helper = DismemblerFunctions(min_rows, pau_duration, qocv_filter)
    df = helper.prefiltering(df, ["SAVE", "REST"])
    df = helper.add_ah_throughput(df)
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))

    dis, seg, rules = {}, {}, {}
    for level in LEVELS:
        dis[level], rules[level] = _dismember_at_level(
            df, min_rows, pau_duration, qocv_filter, level, i_tol
        )
        if dis[level].empty:
            raise RuntimeError(f"{level}: dismember produced no segments")
        seg[level] = dis[level].set_index("_row")["ID"]

    # --- fidelity gate -------------------------------------------------
    # Compare L0 against production run live, row for row. Both consume the
    # same prefiltered frame in the same order, so an identical ID sequence is
    # the strongest available proof that the level-parameterised copy has not
    # drifted from DismemblerFunctions.dismembling.
    mine = dis["L0"]["ID"].astype(str).tolist()
    if len(mine) != len(prod_seq):
        raise RuntimeError(
            f"L0 fidelity gate FAILED: {len(mine)} rows vs production "
            f"{len(prod_seq)}"
        )
    diff = [i for i, (a, b) in enumerate(zip(mine, prod_seq)) if a != b]
    if diff:
        ex = [(i, mine[i], prod_seq[i]) for i in diff[:5]]
        raise RuntimeError(
            f"L0 fidelity gate FAILED: {len(diff)} rows differ from production "
            f"(row, audit, production): {ex}"
        )

    ref_bounds = _boundary_rows(seg["L0"])
    ref_sets = _segment_rowsets(dis["L0"])
    # PAU / EIS / AGING segments never reach with_features_post_labeled by
    # design (feature extraction keeps only target == -1), so carrying the
    # dismember-time label lets a missing production label be told apart from
    # a stale one.
    pre = (dis["L0"].groupby("ID")["target"].first().astype(str)
           .rename("pre_target").reset_index())

    summary, segments = [], None
    for level in LEVELS:
        row = dict(level=level, n_rows=len(df))
        row.update(boundary_scores(ref_bounds, _boundary_rows(seg[level]),
                                   boundary_tol))
        test_sets = _segment_rowsets(dis[level])
        row["n_segments"] = len(test_sets)
        fates = segment_fates(ref_sets, test_sets)
        for fate in ("exact", "trimmed", "split", "merged", "both", "lost"):
            row[f"n_{fate}"] = int((fates["fate"] == fate).sum())
        row["mean_iou"] = float(fates["iou"].mean()) if len(fates) else np.nan
        fates = fates.rename(columns={
            c: f"{level}_{c}" for c in fates.columns if c != "ID"
        })
        segments = fates if segments is None else segments.merge(fates, on="ID")
        summary.append(row)
    segments = segments.merge(pre, on="ID", how="left")

    r0 = rules["L0"]
    uniq_rows = [r for chunk in r0.pop("_uniq_idx", []) for r in chunk]
    if uniq_rows:
        bucket0 = dis["L0"].set_index("_row")["ID"].astype(str)
        landed = bucket0.reindex(uniq_rows).fillna("")
        r0["prozedur_unique_in_bucket0"] = int(landed.str.endswith("_0").sum())
    else:
        r0["prozedur_unique_in_bucket0"] = 0

    conf, sweep = zustand_vs_signal(df, nom_capacity)
    return summary, segments, conf, sweep, r0


def main(config_path, cells, i_tol_frac, boundary_tol, out_dir):
    with open(config_path) as f:
        cfg = json.load(f)
    wp = cfg["working_path"]
    type_cell = cfg.get("type_cell", "")
    out_dir = out_dir or os.path.join(wp, "50_evaluation")
    label_dir = os.path.join(wp, "with_features_post_labeled")

    paths = sorted(glob.glob(os.path.join(wp, "BRONZE_CU", "*.parquet")))
    paths = [p for p in paths
             if type_cell in os.path.basename(p) and "eis" not in os.path.basename(p)]
    if cells:
        paths = [p for p in paths
                 if any(c in os.path.basename(p) for c in cells)]
    if not paths:
        raise SystemExit(f"no BRONZE_CU cells matched (type_cell={type_cell!r})")

    all_summary, all_segments, all_conf, all_sweep = [], [], [], []
    failures = []
    for path in paths:
        stem = os.path.basename(path)[: -len(".parquet")]
        label_path = os.path.join(label_dir, stem + ".csv")
        prod = None
        if os.path.exists(label_path):
            prod = pd.read_csv(label_path, usecols=["ID", "target"])
            prod["ID"] = prod["ID"].astype(str)
        else:
            logging.warning(f"{stem}: no production label CSV — check-up "
                            "fates will be unlabelled")

        logging.info(f"auditing {stem}")
        try:
            with _quiet():
                prod_dis = dismember_raw_cell(
                    stem, path, cfg.get("min_rows", 20),
                    cfg.get("pau_duration", 9.9), cfg["v_max"],
                    cfg.get("procedure_filter"),
                    cfg.get("qocv_procedure_filter"),
                )
            if prod_dis is None or prod_dis.empty:
                raise RuntimeError("production dismember returned nothing")
            summary, segments, conf, sweep, rules = audit_cell(
                path, prod_dis["ID"].astype(str).tolist(), cfg,
                i_tol_frac, boundary_tol
            )
        except Exception as e:
            logging.error(f"{stem}: {type(e).__name__}: {e}")
            failures.append((stem, str(e)))
            continue

        for r in summary:
            r["cell"] = stem
            r.update({f"rule_{k}": v for k, v in rules.items()})
            r["prozedur_unique_frac"] = (
                rules["prozedur_unique"] / rules["prozedur"]
                if rules["prozedur"] else np.nan
            )
        segments["cell"] = stem
        if prod is not None:
            segments = segments.merge(prod, on="ID", how="left")
            labelable = segments["pre_target"] == "-1"
            cov = (segments.loc[labelable, "target"].notna().mean()
                   if labelable.any() else float("nan"))
            logging.info(
                f"{stem}: {int(labelable.sum())} labelable L0 segments "
                f"({len(segments)} total), production label coverage {cov:.1%}")
            if cov < 0.98:
                logging.warning(
                    f"{stem}: label CSV covers only {cov:.1%} of labelable "
                    "segments — it predates the current segmenter")
        conf["cell"] = stem
        sweep["cell"] = stem
        all_summary.extend(summary)
        all_segments.append(segments)
        all_conf.append(conf)
        all_sweep.append(sweep)

    if not all_summary:
        raise SystemExit("no cell completed the audit")

    summary_df = pd.DataFrame(all_summary)
    segments_df = pd.concat(all_segments, ignore_index=True)
    zust_df = pd.concat(
        [pd.concat(all_conf, ignore_index=True).assign(kind="confusion"),
         pd.concat(all_sweep, ignore_index=True).assign(kind="i_tol_sweep")],
        ignore_index=True,
    )

    # check-up-restricted fates: the metric the paper claim rests on
    checkup = segments_df[segments_df.get("target", pd.Series(dtype=str))
                          .astype(str).isin(CHECKUP_TARGETS)]
    for level in LEVELS:
        col = f"{level}_fate"
        for fate in ("exact", "trimmed", "split", "merged", "both", "lost"):
            counts = (checkup[checkup[col] == fate]
                      .groupby("cell").size().to_dict())
            summary_df.loc[summary_df["level"] == level, f"checkup_n_{fate}"] = (
                summary_df.loc[summary_df["level"] == level, "cell"]
                .map(counts).fillna(0).values
            )
    summary_df["checkup_n_total"] = summary_df["cell"].map(
        checkup.groupby("cell").size().to_dict()
    ).fillna(0)

    os.makedirs(out_dir, exist_ok=True)
    # Namespaced by type_cell so runs for different chemistries coexist in one
    # 50_evaluation/ instead of overwriting each other; --pool reads them back.
    tag = type_cell or "all"
    p_sum = os.path.join(out_dir, f"segmentation_audit_{tag}_summary.csv")
    p_seg = os.path.join(out_dir, f"segmentation_audit_{tag}_segments.csv")
    p_zus = os.path.join(out_dir, f"segmentation_audit_{tag}_zustand.csv")
    summary_df["type_cell"] = tag
    summary_df.to_csv(p_sum, index=False)
    segments_df["type_cell"] = tag
    segments_df.to_csv(p_seg, index=False)
    zust_df["type_cell"] = tag
    zust_df.to_csv(p_zus, index=False)

    _report(summary_df, checkup, failures)
    print(f"\nwrote {p_sum}\nwrote {p_seg}\nwrote {p_zus}")


def pool(out_dir):
    """Pool per-config audit outputs in ``out_dir`` into cross-chemistry tables.

    Run the audit once per battery config first; this reads back every
    ``segmentation_audit_<type_cell>_summary.csv`` (and the matching segment /
    zustand files) and writes the tables the ablation is actually reported as.
    """
    sums = sorted(glob.glob(os.path.join(out_dir, "segmentation_audit_*_summary.csv")))
    sums = [p for p in sums if not p.endswith("_fleet_summary.csv")]
    if not sums:
        raise SystemExit(f"no per-config audit outputs found in {out_dir}")
    S = pd.concat([pd.read_csv(p) for p in sums], ignore_index=True)
    G = pd.concat([pd.read_csv(p.replace("_summary.csv", "_segments.csv"))
                   for p in sums], ignore_index=True)
    Z = pd.concat([pd.read_csv(p.replace("_summary.csv", "_zustand.csv"))
                   for p in sums], ignore_index=True)

    boundary = S.groupby(["type_cell", "level"]).agg(
        cells=("cell", "nunique"), segments=("n_segments", "sum"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), exact=("n_exact", "sum"), merged=("n_merged", "sum"),
        split=("n_split", "sum"), trimmed=("n_trimmed", "sum"),
        lost=("n_lost", "sum"))

    checkup = S.groupby(["type_cell", "level"]).agg(
        total=("checkup_n_total", "sum"), exact=("checkup_n_exact", "sum"),
        split=("checkup_n_split", "sum"), merged=("checkup_n_merged", "sum"),
        lost=("checkup_n_lost", "sum")).astype(int)
    checkup["exact_pct"] = (100 * checkup.exact / checkup.total).round(1)
    # a split segment still exists, just cut into pieces; a merged or lost one
    # has been swallowed and cannot be recovered downstream
    checkup["intact_pct"] = (
        100 * (checkup.exact + checkup.split) / checkup.total).round(1)

    rule3 = S[S.level == "L0"].groupby("type_cell").agg(
        firings=("rule_prozedur", "sum"), unique=("rule_prozedur_unique", "sum"),
        unique_discarded=("rule_prozedur_unique_in_bucket0", "sum"))
    rule3["unique_pct"] = (100 * rule3["unique"] / rule3.firings).round(1)
    rule3["discarded_pct"] = (
        100 * rule3.unique_discarded / rule3["unique"]).round(1)

    conf = Z[Z.kind == "confusion"].groupby(
        ["type_cell", "Zustand", "signal_state"])["n_rows"].sum().reset_index()

    for name, df in (("boundary", boundary), ("checkup", checkup),
                     ("rule3", rule3), ("zustand_confusion", conf)):
        path = os.path.join(out_dir, f"segmentation_audit_fleet_{name}.csv")
        df.to_csv(path)
        print(f"wrote {path}")

    pd.set_option("display.width", 200)
    print("\n=== boundary agreement vs L0 ===")
    print(boundary.round(3).to_string())
    print("\n=== check-up segment recovery (CAP/PUL/PUL*RES/qOCV_*) ===")
    print(checkup.to_string())
    print("\n=== rule 3 (Prozedur-change) firings ===")
    print(rule3.to_string())
    print("\n=== labelled segments that break, by level ===")
    for lv in LEVELS[1:]:
        bad = G[(G[f"{lv}_fate"] != "exact") & G["target"].notna()]
        bad = bad[bad["target"].astype(str).isin(CHECKUP_TARGETS)]
        print(f"\n{lv}:")
        print(pd.crosstab(bad["target"], [bad["type_cell"], bad[f"{lv}_fate"]])
              .to_string() if len(bad) else "  no check-up segment breaks")


def _report(summary_df, checkup, failures):
    pd.set_option("display.width", 200)
    print("\n=== boundary agreement vs L0 ===")
    cols = ["cell", "level", "n_segments", "precision", "recall", "f1",
            "n_exact", "n_trimmed", "n_split", "n_merged", "n_both", "n_lost"]
    print(summary_df[cols].round(3).to_string(index=False))

    print("\n=== check-up segment fate (CAP / PUL / PUL*RES / qOCV_*) ===")
    ccols = ["cell", "level", "checkup_n_total", "checkup_n_exact",
             "checkup_n_trimmed", "checkup_n_split", "checkup_n_merged",
             "checkup_n_both", "checkup_n_lost"]
    print(summary_df[[c for c in ccols if c in summary_df]].to_string(index=False))

    print("\n=== rule-3 (Prozedur) uniqueness, per cell ===")
    r = (summary_df[summary_df["level"] == "L0"]
         [["cell", "rule_prozedur", "rule_prozedur_unique",
           "rule_prozedur_unique_in_bucket0", "prozedur_unique_frac",
           "rule_pau_start", "rule_pau_exit", "rule_qocv"]])
    print(r.round(3).to_string(index=False))
    tot = r["rule_prozedur"].sum()
    uniq = r["rule_prozedur_unique"].sum()
    b0 = r["rule_prozedur_unique_in_bucket0"].sum()
    print(f"\nFLEET: {uniq}/{tot} Prozedur-change boundaries ({uniq / tot:.1%}) "
          f"fire where no other rule does;")
    print(f"       of those, {b0} ({b0 / uniq:.1%}) land in the discard bucket, "
          "so they cut nothing that survives.")

    if failures:
        print("\n=== FAILED CELLS ===")
        for stem, err in failures:
            print(f"  {stem}: {err}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Audit segmentation's dependence on protocol metadata")
    ap.add_argument("config", nargs="?", help="Path to battery config JSON")
    ap.add_argument("--pool", action="store_true",
                    help="pool existing per-config outputs in --out-dir into "
                         "cross-chemistry tables (no config needed)")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="subset by name fragment")
    ap.add_argument("--i-tol-frac", type=float, default=1e-3,
                    help="L2 pause threshold as a fraction of nom_capacity")
    ap.add_argument("--boundary-tol", type=int, default=2,
                    help="boundary match window, in rows")
    ap.add_argument("-o", "--out-dir", default=None)
    a = ap.parse_args()
    if a.pool:
        if not a.out_dir:
            ap.error("--pool needs -o/--out-dir")
        pool(a.out_dir)
    elif a.config:
        main(a.config, a.cells, a.i_tol_frac, a.boundary_tol, a.out_dir)
    else:
        ap.error("a battery config is required unless --pool is given")
