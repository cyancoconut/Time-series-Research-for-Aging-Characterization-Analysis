"""Does cross-chemistry transfer come from the features or from the label space?

Covers validation item 5. Read-only with respect to the pipeline: it trains
throwaway models in memory and writes only its own CSVs.

What this measures
------------------
A classifier trained on some chemistries and applied to an unseen one has two
things that could stop it working: the *features* could be chemistry-bound, or
the *label space* could be. This separates them.

Two label spaces over identical features:

``absolute``
    The pipeline's own labels --- ``CAP``, ``PUL``, ``qOCV_DCH``, ``qOCV_CHA``,
    ``PREP_CHA``, ``SOC_ADJUST``, ``-1``. "CAP" here means "a full discharge at
    *this fleet's* capacity C-rate", so the rate is baked into the class and the
    model must learn it from the training chemistries.

``shape``
    What the segment *is*, independent of any rate: ``full_discharge``,
    ``full_charge``, ``partial_dch``, ``partial_cha``, ``pulse``, ``other``. No
    C-rate appears in the class. A prediction is resolved to CAP afterwards, by
    testing the segment's own measured ``abs_Current_mean`` against the
    **held-out chemistry's configured** ``cap_rate`` --- the same test
    ``predict_classifier._map_llm_label_to_tagged`` applies at inference.

The comparison is the point: in ``absolute`` the chemistry-specific quantity is
learned, in ``shape`` it is supplied as configuration. Both use the same
features, the same model and the same folds, so a difference between them is a
property of the label space alone.

Why it is framed this way
-------------------------
Measured first, then framed. In the ``absolute`` space held-out CAP recall is
zero on LFP and on sodium-ion, and no feature set changes it --- not dropping
the voltage-range feature, not adding or substituting the coulombic duration.
The reason is visible in the training distribution: with LFP held out, every
training CAP sits at ``abs_Current_mean`` 0.500 (NMC and Na-ion are both C/2)
while LFP's is 1.000, so "CAP" has been learned as "a full discharge at C/2".
That is covariate shift across most of the current features at once, not a
deficiency of any one of them, which is why feature engineering cannot repair
it.

Note on a hypothesis this replaced: the flat LFP discharge plateau was expected
to break the voltage-range feature. It does not. ``Voltage_range`` separates
full from partial discharges on LFP with ROC AUC 1.0 (medians 0.90 against
0.04) --- a full LFP discharge still crosses both knees, and the partial
discharges in this protocol are small SoC-adjust steps. The separability report
below is retained as the evidence for that negative result.

Usage (from src/):
    python -m evaluation.feature_ablation CONFIG [CONFIG ...] [-o OUT_DIR]

One config per chemistry; each supplies ``type_cell`` (which cells belong to
it), ``cap_rate`` and ``qocv_crate``. Example:

    python -m evaluation.feature_ablation \\
        .../battery_config_VTC_linux.json \\
        .../battery_config_APR_linux.json \\
        .../battery_config_Hina_linux.json

Outputs (default ``<working_path>/50_evaluation``):
    feature_ablation_separability.csv  per (chemistry, feature) full-vs-partial AUC
    feature_ablation_loco.csv          per (label space, feature set, held-out chemistry)
"""

import argparse
import glob
import json
import logging
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from cluster.train_classifier import FEATURE_COLS, _new_model

# Cell-stem fragment -> chemistry label, for the cells this fleet contains.
CHEMISTRY_PATTERNS = (
    ("VTC6", "NMC"),
    ("APR18650M1B", "LFP"),
    ("Hina", "SIB"),
)

# Candidate features for the full-vs-partial decision, in the order reported.
SWING_FEATURES = ("Voltage_range", "true_voltage_range", "norm_duration")

COULOMBIC = "norm_duration"
VOLTAGE_SWING = "Voltage_range"
DURATION_COLS = ("Duration_minutes", "Duration_quartile")

FEATURE_SETS = {
    "shipped": list(FEATURE_COLS),
    "minus_voltage_range": [c for c in FEATURE_COLS if c != VOLTAGE_SWING],
    "plus_coulombic": list(FEATURE_COLS) + [COULOMBIC],
    "swap_duration_for_coulombic": (
        [c for c in FEATURE_COLS if c not in DURATION_COLS] + [COULOMBIC]),
}

# The pipeline's final label vocabulary, after the post-labeling target sync and
# the PUL*RES -> PUL merge. Anything else means a cell's labels were never
# synced; see _check_label_vocabulary.
CANONICAL_LABELS = {
    "CAP", "PUL", "qOCV_DCH", "qOCV_CHA", "PREP_CHA", "SOC_ADJUST", "-1",
}

# Same full-vs-partial threshold the LLM prompt and _canonicalize_llm_labels use.
FULL_VOLTAGE_RANGE_THRESHOLD = 0.9
CAP_RATE_TOL = 0.05

FULL_LABEL = "CAP"
_PARTIAL_RE = re.compile(r"^(SOC_ADJUST|partial_dch(_|$))")


def chemistry_of(stem):
    for frag, chem in CHEMISTRY_PATTERNS:
        if frag in stem:
            return chem
    return None


def _check_label_vocabulary(df):
    """Refuse to train across cells whose label vocabularies disagree.

    A cell whose post-labeling target sync did not complete keeps intermediate
    tagged labels (``CAP*``, ``PUL*``) or raw cluster ids (``3.0``,
    ``cap_layer_1``). Those are cell-private strings: with such a cell in the
    training split the model can assign them to held-out segments, driving
    held-out recall to zero in a way that looks like a feature effect rather
    than a data defect. That exact failure happened before this guard existed,
    so it fails loudly rather than warning.
    """
    seen = set(df["target"].unique())
    unknown = seen - CANONICAL_LABELS
    if not unknown:
        return
    offenders = (df[df["target"].isin(unknown)]
                 .groupby("cell")["target"].agg(lambda s: sorted(set(s))[:6]))
    lines = "\n".join(f"    {c}: {v}" for c, v in offenders.items())
    raise SystemExit(
        "non-canonical target labels found — these cells need a pipeline "
        f"re-run before they can be used for training:\n{lines}\n"
        f"  unexpected labels: {sorted(unknown)}\n"
        f"  expected: {sorted(CANONICAL_LABELS)}")


def load_segments(configs):
    """Load every cell CSV, tagged with its chemistry and that chemistry's rates.

    ``configs`` maps chemistry -> config dict. Cells are matched to a chemistry
    by stem fragment, so one working_path holding several chemistries is fine.
    """
    working_paths = {c["working_path"] for c in configs.values()}
    frames = []
    for wp in sorted(working_paths):
        for p in sorted(glob.glob(os.path.join(
                wp, "with_features_post_labeled", "*.csv"))):
            stem = os.path.basename(p)[: -len(".csv")]
            chem = chemistry_of(stem)
            if chem is None or chem not in configs:
                logging.warning(f"{stem}: no config for its chemistry, skipped")
                continue
            df = pd.read_csv(p)
            df["cell"] = stem
            df["chemistry"] = chem
            df["cap_rate"] = configs[chem].get("cap_rate")
            df["qocv_rate"] = configs[chem].get("qocv_crate")
            frames.append(df)
    if not frames:
        raise SystemExit("no segment CSVs found for the given configs")
    out = pd.concat(frames, ignore_index=True)
    out["target"] = out["target"].astype(str)
    # The classifier need not tell a restore pulse from a test pulse; the split
    # is recomputed downstream, and train_classifier merges them for the same
    # reason. Keeping the spaces identical makes the numbers comparable.
    out.loc[out["target"] == "PUL*RES", "target"] = "PUL"
    _check_label_vocabulary(out)
    return out


def shape_label(row):
    """What the segment is, with no C-rate in the class name.

    Pulses are taken from the pipeline label because a pulse is defined by its
    duration relative to the protocol, not by its shape in these features.
    Everything else is rebuilt from the segment's own sign and corrected SoC
    swing, so the class carries no chemistry-specific quantity.
    """
    t = str(row["target"])
    if t == "PUL":
        return "pulse"
    tvr, cm = row.get("true_voltage_range"), row.get("Current_mean")
    if not (np.isfinite(tvr) and np.isfinite(cm)) or cm == 0:
        return "other"
    full = tvr >= FULL_VOLTAGE_RANGE_THRESHOLD
    charge = cm > 0
    if full:
        return "full_charge" if charge else "full_discharge"
    return "partial_cha" if charge else "partial_dch"


def resolve_cap(pred, abs_current, cap_rate, tol=CAP_RATE_TOL):
    """Resolve a shape prediction to CAP using the target chemistry's cap_rate.

    Mirrors ``predict_classifier._map_llm_label_to_tagged``: a full discharge
    counts as the capacity test when its *measured* C-rate matches the rate the
    config declares for that cell. The chemistry-specific number enters here,
    at inference, instead of being learned.
    """
    if pred != "full_discharge" or not cap_rate or not np.isfinite(abs_current):
        return False
    return cap_rate * (1 - tol) <= abs_current <= cap_rate * (1 + tol)


def separability(df):
    """ROC AUC of each swing feature for full vs partial discharge, per chemistry."""
    rows = []
    for chem, grp in df.groupby("chemistry"):
        is_full = grp["target"] == FULL_LABEL
        is_partial = grp["target"].str.match(_PARTIAL_RE).fillna(False)
        sub = grp[is_full | is_partial]
        y = (sub["target"] == FULL_LABEL).astype(int)
        if y.nunique() < 2:
            logging.warning(f"{chem}: only one class present, skipping AUC")
            continue
        for feat in SWING_FEATURES:
            if feat not in sub.columns:
                continue
            v = sub[feat]
            ok = v.notna()
            if ok.sum() < 2 or y[ok].nunique() < 2:
                continue
            auc = roc_auc_score(y[ok], v[ok])
            vf, vp = v[ok][y[ok] == 1], v[ok][y[ok] == 0]
            rows.append(dict(
                chemistry=chem, feature=feat, auc=auc,
                auc_abs=max(auc, 1 - auc),
                n_full=int(y[ok].sum()), n_partial=int((1 - y[ok]).sum()),
                full_median=float(vf.median()),
                partial_median=float(vp.median()),
                # The hypothesised plateau failure needs a partial discharge
                # that is LARGE in charge but small in voltage. These say
                # whether such a segment exists in the data at all.
                partial_p95=float(vp.quantile(0.95)),
                partial_max=float(vp.max()),
                full_min=float(vf.min()),
                gap=float(vf.min() - vp.max()),
            ))
    return pd.DataFrame(rows)


def loco(df):
    """Leave-one-chemistry-out in both label spaces, over every feature set."""
    rows = []
    df = df.copy()
    df["shape"] = df.apply(shape_label, axis=1)
    chems = sorted(df["chemistry"].unique())

    for space in ("absolute", "shape"):
        ycol = "target" if space == "absolute" else "shape"
        for name, cols in FEATURE_SETS.items():
            usable = [c for c in cols if c in df.columns]
            if len(usable) != len(cols):
                logging.warning(
                    f"{name}: missing {sorted(set(cols) - set(usable))}, skipped")
                continue
            d = df.dropna(subset=usable + [ycol, "abs_Current_mean"])
            for held in chems:
                tr = d[d["chemistry"] != held]
                te = d[d["chemistry"] == held]
                if te.empty or tr.empty or tr[ycol].nunique() < 2:
                    continue
                model = _new_model()
                model.fit(tr[usable], tr[ycol])
                pred = model.predict(te[usable])

                truth_cap = (te["target"] == FULL_LABEL).to_numpy()
                if space == "absolute":
                    pred_cap = pred == FULL_LABEL
                else:
                    rate = te["cap_rate"].iloc[0]
                    pred_cap = np.array([
                        resolve_cap(p, a, rate)
                        for p, a in zip(pred, te["abs_Current_mean"])])

                tp = int((pred_cap & truth_cap).sum())
                rows.append(dict(
                    label_space=space, feature_set=name, held_out=held,
                    n_train=len(tr), n_test=len(te),
                    n_cap_true=int(truth_cap.sum()),
                    n_cap_pred=int(pred_cap.sum()),
                    cap_recall=tp / truth_cap.sum() if truth_cap.sum() else np.nan,
                    cap_precision=tp / pred_cap.sum() if pred_cap.sum() else 0.0,
                    macro_f1=f1_score(te[ycol], pred, average="macro",
                                      zero_division=0),
                ))
    return pd.DataFrame(rows)


def main(config_paths, out_dir):
    configs = {}
    for p in config_paths:
        with open(p) as f:
            cfg = json.load(f)
        # type_cell ("VTC", "APR", "Hina") is a prefix of the cell-stem
        # fragment the chemistry map keys on ("VTC6", "APR18650M1B", "Hina").
        tc = cfg.get("type_cell", "")
        chem = next((c for frag, c in CHEMISTRY_PATTERNS if tc and tc in frag),
                    None)
        if chem is None:
            raise SystemExit(
                f"{p}: type_cell {tc!r} matches no chemistry in "
                f"{[f for f, _ in CHEMISTRY_PATTERNS]}")
        configs[chem] = cfg
    logging.info(f"chemistries: {sorted(configs)}")

    out_dir = out_dir or os.path.join(
        next(iter(configs.values()))["working_path"], "50_evaluation")

    df = load_segments(configs)
    logging.info(f"{len(df)} segments, {df['cell'].nunique()} cells, "
                 f"{dict(df.groupby('chemistry')['cell'].nunique())}")

    sep = separability(df)
    lo = loco(df)

    os.makedirs(out_dir, exist_ok=True)
    p_sep = os.path.join(out_dir, "feature_ablation_separability.csv")
    p_loco = os.path.join(out_dir, "feature_ablation_loco.csv")
    sep.to_csv(p_sep, index=False)
    lo.to_csv(p_loco, index=False)

    pd.set_option("display.width", 220)
    print("\n=== full vs partial discharge: separability by feature (ROC AUC) ===")
    if len(sep):
        print(sep.pivot(index="feature", columns="chemistry",
                        values="auc_abs").round(4).to_string())
        print("\nmedians (full | partial), and the largest partial seen:")
        print(sep[["chemistry", "feature", "full_median", "partial_median",
                   "partial_max", "n_full", "n_partial"]]
              .round(4).to_string(index=False))

    print("\n=== leave-one-chemistry-out: CAP recall on the held-out chemistry ===")
    for space in ("absolute", "shape"):
        s = lo[lo.label_space == space]
        if s.empty:
            continue
        print(f"\nlabel space = {space}"
              + ("   (CAP learned from the training chemistries)"
                 if space == "absolute"
                 else "   (shape learned; CAP resolved by the held-out "
                      "chemistry's configured cap_rate)"))
        print(s.pivot(index="feature_set", columns="held_out",
                      values="cap_recall").round(3).to_string())

    print(f"\nwrote {p_sep}\nwrote {p_loco}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Do features or the label space carry cross-chemistry transfer?")
    ap.add_argument("configs", nargs="+",
                    help="one battery config per chemistry")
    ap.add_argument("-o", "--out-dir", default=None)
    a = ap.parse_args()
    main(a.configs, a.out_dir)
