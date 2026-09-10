"""Cross-laboratory validation: does the method transfer off our own bench?

Covers validation item 6 of the paper plan. Read-only: trains throwaway models
in memory and writes only its own CSVs.

Every cell in the in-house fleet was measured at one institute on one cycler
family, so nothing in the other results supports a cross-laboratory claim. This
module runs the method against reference performance tests from two independent
laboratories --- ISU-ILCC and UConn-ILCC --- ingested through thin format
adapters and given nothing but their published cell parameters. No label from
either laboratory is used for training.

What is and is not measurable on adapted data
---------------------------------------------
The adapters synthesise the columns the pipeline expects. For ISU the
``Prozedur`` is a single constant and ``Zustand`` is a per-sweep tag the adapter
invents, so the metadata ablation of the in-house results is **meaningless
here** --- one cannot ablate protocol columns one has written oneself. That
experiment is deliberately not run on these datasets. What the external data
does test, honestly, is the part that matters most: given a foreign
laboratory's raw record and its datasheet scalars, are the check-ups found?

Ground truth
------------
One ``BM_Programm`` is one reference performance test, which both adapters
guarantee by construction (ISU maps each RPT to a programme; UConn's source
files are named ``<cell>_RPT<nnn>``). So the denominator is the programme count
and needs no labelling.

Three detectors, same segments
------------------------------
``pipeline``
    Whatever the pipeline actually labelled CAP when it was run --- the
    unsupervised clustering route.

``config_rule``
    The configured criterion applied directly: a full sweep
    (``true_voltage_range >= 0.9``) that is a discharge within
    ``cap_tol`` of the laboratory's declared ``cap_rate``. This is the mechanism
    the in-house results attribute transfer to, expressed as a rule.

``learned_shape``
    A random forest trained on the **in-house** fleet only, predicting segment
    shape with no C-rate in any class, whose predictions are resolved to CAP
    against the external laboratory's ``cap_rate``. This is the same procedure
    the in-house leave-one-chemistry-out uses, with a laboratory boundary
    instead of a chemistry boundary.

Usage (from src/):
    python -m evaluation.external_validation \\
        --external CFG [CFG ...] --in-house CFG [CFG ...] [-o OUT_DIR]

Outputs:
    external_validation_programmes.csv  one row per (lab, cell, programme)
    external_validation_summary.csv     one row per (lab, detector)
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from cluster.train_classifier import FEATURE_COLS, _new_model
from evaluation.feature_ablation import (
    CAP_RATE_TOL,
    FULL_VOLTAGE_RANGE_THRESHOLD,
    load_segments as load_in_house_segments,
    resolve_cap,
    shape_label,
)

# The coulombic feature is carried because the in-house ablation showed it is
# what keeps capacity precision at 1.0 once recall is secured by the rate test.
EXTERNAL_FEATURES = list(FEATURE_COLS) + ["norm_duration"]


_ROUTE_DIRS = {
    "hdbscan": ("with_features_post_labeled",),
    "classifier": ("60_classifier", "with_features_post_labeled"),
}


def load_external(cfg):
    """Segments for one external laboratory, one frame per labelling route.

    Both routes are loaded when both have been run. They are kept separate on
    purpose: the two routes write the same segments with the same features and
    differ only in ``target``, so comparing a laboratory labelled by one route
    against a laboratory labelled by the other would confound the route with
    the laboratory. That mistake is easy to make, because which routes a
    dataset happens to have been run through is an accident of its history.
    """
    lab = cfg.get("type_cell", "external")
    out = {}
    for route, parts in _ROUTE_DIRS.items():
        seg_dir = os.path.join(cfg["working_path"], *parts)
        paths = sorted(glob.glob(os.path.join(seg_dir, "*.csv")))
        if not paths:
            continue
        frames = []
        for p in paths:
            df = pd.read_csv(p)
            df["cell"] = os.path.basename(p)[: -len(".csv")]
            frames.append(df)
        d = pd.concat(frames, ignore_index=True)
        d["lab"] = lab
        d["route"] = route
        d["cap_rate"] = cfg["cap_rate"]
        d["target"] = d["target"].astype(str)
        out[route] = d
    if not out:
        raise SystemExit(f"no per-segment CSVs under {cfg['working_path']}")
    logging.info(f"{lab}: routes available = {sorted(out)}")
    return out


def detect(df, model=None, features=None):
    """Per programme, which detectors found a capacity test.

    Returns one row per (lab, cell, BM_Programm) with a boolean per detector,
    plus the number of segments each detector selected --- a detector that finds
    the programme by tagging six segments in it is not the same as one that
    tags one, and the paper reports both.
    """
    rows = []
    if model is not None:
        d = df.dropna(subset=features)
        pred = pd.Series(model.predict(d[features]), index=d.index)
        shape_pred = pd.Series("", index=df.index, dtype=object)
        shape_pred.loc[pred.index] = pred.values
    else:
        shape_pred = pd.Series("", index=df.index, dtype=object)

    for (lab, route, cell, prog), g in df.groupby(
            ["lab", "route", "cell", "BM_Programm"]):
        rate = g["cap_rate"].iloc[0]
        pipeline = g["target"] == "CAP"
        rule = (
            (g["Current_mean"] < 0)
            & (g["Current_mean"].abs().between(rate * (1 - CAP_RATE_TOL),
                                               rate * (1 + CAP_RATE_TOL)))
            & (g["true_voltage_range"] >= FULL_VOLTAGE_RANGE_THRESHOLD)
        )
        learned = pd.Series(False, index=g.index)
        if model is not None:
            learned = pd.Series([
                resolve_cap(shape_pred.get(i, ""), a, rate)
                for i, a in zip(g.index, g["abs_Current_mean"])], index=g.index)
        rows.append(dict(
            lab=lab, route=route, cell=cell, BM_Programm=prog, n_segments=len(g),
            pipeline=bool(pipeline.any()), n_pipeline=int(pipeline.sum()),
            config_rule=bool(rule.any()), n_config_rule=int(rule.sum()),
            learned_shape=bool(learned.any()), n_learned_shape=int(learned.sum()),
        ))
    return pd.DataFrame(rows)


def train_in_house(cfg_paths):
    """Shape-space model trained on the in-house fleet only."""
    from evaluation.feature_ablation import CHEMISTRY_PATTERNS

    configs = {}
    for p in cfg_paths:
        with open(p) as f:
            cfg = json.load(f)
        tc = cfg.get("type_cell", "")
        chem = next((c for frag, c in CHEMISTRY_PATTERNS if tc and tc in frag),
                    None)
        if chem is None:
            raise SystemExit(f"{p}: type_cell {tc!r} is not an in-house chemistry")
        configs[chem] = cfg
    df = load_in_house_segments(configs)
    df["shape"] = df.apply(shape_label, axis=1)
    d = df.dropna(subset=EXTERNAL_FEATURES + ["shape"])
    model = _new_model()
    model.fit(d[EXTERNAL_FEATURES], d["shape"])
    logging.info(
        f"trained shape model on {len(d)} in-house segments, "
        f"{d['cell'].nunique()} cells, chemistries {sorted(df.chemistry.unique())}")
    return model


def main(external_cfgs, in_house_cfgs, out_dir):
    model, features = None, EXTERNAL_FEATURES
    if in_house_cfgs:
        model = train_in_house(in_house_cfgs)

    frames = []
    for p in external_cfgs:
        with open(p) as f:
            cfg = json.load(f)
        for route, d in load_external(cfg).items():
            logging.info(f"{cfg.get('type_cell')} [{route}]: {len(d)} segments, "
                         f"{d['cell'].nunique()} cells")
            frames.append(d)
    ext = pd.concat(frames, ignore_index=True)

    progs = detect(ext, model, features)
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    p_prog = os.path.join(out_dir, "external_validation_programmes.csv")
    progs.to_csv(p_prog, index=False)

    rows = []
    for (lab, route), g in progs.groupby(["lab", "route"]):
        for det in ("pipeline", "config_rule", "learned_shape"):
            found = int(g[det].sum())
            sel = g[f"n_{det}"]
            rows.append(dict(
                lab=lab, route=route, detector=det, cells=g["cell"].nunique(),
                programmes=len(g), found=found,
                recall_pct=round(100 * found / len(g), 1),
                segments_selected=int(sel.sum()),
                median_per_programme=float(sel.median()),
            ))
    summary = pd.DataFrame(rows)
    p_sum = os.path.join(out_dir, "external_validation_summary.csv")
    summary.to_csv(p_sum, index=False)

    pd.set_option("display.width", 200)
    print("\n=== capacity test recovered, per laboratory ===")
    print(summary.to_string(index=False))
    print("\n=== per cell ===")
    per = (progs.groupby(["lab", "route", "cell"])
           .agg(programmes=("BM_Programm", "size"),
                pipeline=("pipeline", "sum"),
                config_rule=("config_rule", "sum"),
                learned_shape=("learned_shape", "sum")))
    print(per.to_string())
    print(f"\nwrote {p_prog}\nwrote {p_sum}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Cross-laboratory validation on external RPT datasets")
    ap.add_argument("--external", nargs="+", required=True,
                    help="config per external laboratory")
    ap.add_argument("--in-house", nargs="*", default=None,
                    help="in-house configs to train the shape model on")
    ap.add_argument("-o", "--out-dir", default=None)
    a = ap.parse_args()
    main(a.external, a.in_house, a.out_dir)
