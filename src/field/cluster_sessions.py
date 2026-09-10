"""F3 — HDBSCAN clustering on per-session features to find the CAP cluster.

The CAP cluster is the analog of ``post_cluster_filter.find_capacity`` for the
field-data track: rather than detecting controlled C/2 discharges in a cycler,
we identify the group of charging sessions whose shape (large ΔSOC, low
current variability, CC→CV transition) makes them suitable for an opportunistic
capacity estimate.

Workflow::

    feats = sessions.session_features(split_sessions(load_vehicle(...)), vehicle=v)
    labeled = cluster_sessions(feats)
    summary = summarize_clusters(labeled)
    cap_labels = pick_cap_clusters(labeled)
    cap_sessions = labeled[labeled["cluster_label"].isin(cap_labels)]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler


# Per-session feature fed to HDBSCAN: the state-of-charge excursion, alone.
#
# The wider set this replaced — duration, dSOC, I_mean, I_cv, V_max and the
# has_cv_tail flag — turned out to be circular. Clustering on it reproduced
# has_cv_tail to 99.7–100% agreement on all 20 vehicles: the flag was one of
# the inputs, so HDBSCAN was relabelling an answer it had been handed rather
# than discovering the capacity population. Removing the flag and clustering on
# the remaining physical features failed on 11 of 20 vehicles.
#
# Clustering on dSOC alone is both non-circular and better. pick_cap_clusters
# then keeps every cluster that ended full and covered at least half the SOC
# window. Measured against the previous default, per vehicle:
#
#     variant                      found   med n   resid%   coverage%
#     previous (has_cv_tail)       20/20     290     1.33          79
#     [V_max, SOC_end]             20/20      74     0.93          20
#     dSOC alone                   20/20      57     0.67          97
#
# (measured at the 1% min_cluster_size in force at the time; the comparison
# between feature sets holds, the absolute figures have since moved.)
#
# Better on scatter and on record coverage on 20 of 20 vehicles each, and only
# ~37% of the sessions it selects carry a CV tail — so it finds a different and
# cleaner population, not the flag under another name.
#
# Two caveats. With one feature the "clustering" is a fine partition of a single
# axis, so HDBSCAN may add little over quantile binning; and the selected count
# varies widely between vehicles (28–565).
DEFAULT_FEATURE_COLUMNS = [
    "dSOC",
]


# min_cluster_size as a fraction of a vehicle's session count.
#
# 5% suits 19 of the 20 vehicles. Vehicle 9 needs 2%, and the reason is
# structural rather than a fitted constant: this parameter does two different
# jobs depending on how a driver charges. Where the deep charges fall into
# discrete habitual bands separated by real gaps in the dSOC axis — vehicle 9
# has a literal empty gap from 69.2 to 71.2, some 3.5x the epsilon merge
# distance — the parameter decides *which band* is admitted. Where the deep
# range is a smooth plateau instead (vehicle 10: 38-64 sessions in every
# 5-point bin from 35 to 85, no dip anywhere), it decides *how much of the
# continuum* is taken. One value cannot mean the same thing in both cases,
# which is why the fleet result is not monotonic in it.
#
# At 5% all 167 of vehicle 9's sessions with dSOC >= 70 fall into noise and the
# only cluster on offer is a shallow one (median dSOC 28.4, ending at SOC 76.8)
# that pick_cap_clusters now rejects outright — so at 5% that vehicle yields an
# empty table, not a bad one. 2% lets its deep bands reach cluster status.
#
# The fleet-level cost of the override is nil: median scatter reduction is
# 1.71x either way, and it recovers the 20th vehicle.
DEFAULT_MIN_CLUSTER_FRACTION = 0.05
MIN_CLUSTER_FRACTION_OVERRIDES: dict[str, float] = {"9": 0.02}


def _min_cluster_fraction(feats: pd.DataFrame) -> float:
    """Per-vehicle min_cluster_size fraction, read off the ``vehicle`` column."""
    if "vehicle" not in feats.columns or feats.empty:
        return DEFAULT_MIN_CLUSTER_FRACTION
    vehicle = str(feats["vehicle"].iloc[0])
    return MIN_CLUSTER_FRACTION_OVERRIDES.get(vehicle, DEFAULT_MIN_CLUSTER_FRACTION)


def _build_feature_matrix(feats: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=feats.index)
    out["duration_s_log"] = np.log1p(feats["duration_s"].clip(lower=0))
    out["dSOC"] = feats["dSOC"]
    out["I_mean"] = feats["I_mean"]
    i_mean_safe = feats["I_mean"].replace(0, np.nan).abs()
    out["I_cv"] = (feats["I_std"] / i_mean_safe).fillna(0)
    out["V_max"] = feats["V_max"]
    out["has_cv_tail"] = feats["has_cv_tail"].astype(float)
    return out


# HDBSCAN's cluster_selection_epsilon merges clusters separated by less than this
# distance in the standardised feature space. It matters here because dSOC is
# quantised to 0.4-point steps: at epsilon 0 every discrete value with enough
# members becomes its own cluster, giving ~50 clusters that are bins rather than
# structure. 0.03 merges those into ~7 real bands at no cost - measured over the
# 20 vehicles, median residual scatter 0.65% against 0.67%, coverage unchanged
# at 97%, and the same sessions selected (55 against 57). Above ~0.05 the bands
# merge too far: the selected population broadens to 114 then 478 sessions and
# scatter degrades to 0.71% and 1.02%.
#
# Those numbers were measured at min_cluster_size = 1% of sessions, which was
# the default at the time; the default is now 5% (2% on vehicle 9), which
# already merges the bands structurally, so epsilon does less work than it did.
# It has not been re-swept against the coarser partition.
def cluster_sessions(
    feats: pd.DataFrame,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.03,
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Add a ``cluster_label`` column (-1 = noise) to the per-session features.

    Default ``min_cluster_size`` is ``max(10, f * session count)``, where ``f``
    is ``DEFAULT_MIN_CLUSTER_FRACTION`` (5%) unless the vehicle appears in
    ``MIN_CLUSTER_FRACTION_OVERRIDES`` — see the note on those constants for why
    vehicle 9 needs 2%. Pass ``min_cluster_size`` explicitly to bypass both.

    Measured over the fleet, under the multi-cluster ``pick_cap_clusters`` rule:

        setting          found   total n   med dSOC   med resid%   ratio   cov%
        5%, v9 at 2%     20/20      7367       60.4         1.02    1.71x   99.9
        5% flat          19/20      7083       60.0         1.02    1.70x   99.9
        3% flat          20/20      7324       60.8         0.97    1.75x   99.9
        2% flat          20/20      7739       61.8         0.98    1.71x   99.8

    ``ratio`` is the reduction in capacity scatter about the ageing trend
    against taking every charging session. Note it is nearly flat across these
    settings: what moves it is how *deep* the admitted population is, not how
    the partition is cut. Taking only the single deepest admissible cluster
    reaches ~2.1x, at 4397 sessions instead of 7367 — precision bought with
    density. The multi-cluster rule is preferred because it samples the record
    far more finely (consecutive estimates ~1 day apart against ~130).

    The consequence to keep in mind is that a sparse deep tail can still be
    stranded in noise on a plateau vehicle. Vehicle 10 is the live example: its
    60 deepest sessions (median dSOC 85.2, SOC_end 97.6, the highest CV-tail
    rate in the vehicle at 55%) are all label -1, and the admitted clusters take
    the plateau instead. Selecting on a dSOC threshold rather than on cluster
    density would address this, but is a larger change to the method than a
    parameter choice.

    Cluster membership counts are in any case an accident of how often a given
    driver charged from empty, not a property of the cell, so tying selection to
    a size threshold is inherently soft.
    """
    if feats.empty:
        out = feats.copy()
        out["cluster_label"] = pd.Series(dtype=int)
        return out

    if min_cluster_size is not None:
        mcs = min_cluster_size
    else:
        mcs = max(10, int(len(feats) * _min_cluster_fraction(feats)))
    ms = min_samples if min_samples is not None else max(5, mcs // 2)

    X = _build_feature_matrix(feats)[feature_columns].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    labels = np.full(len(feats), -1, dtype=int)
    if mask.sum() < mcs:
        out = feats.copy()
        out["cluster_label"] = labels
        return out

    Xs = StandardScaler().fit_transform(X[mask])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels[mask] = clusterer.fit_predict(Xs)

    out = feats.copy()
    out["cluster_label"] = labels
    return out


def summarize_clusters(labeled: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster summary: size + medians of the discriminating features."""
    g = labeled.groupby("cluster_label", sort=True)
    out = pd.DataFrame({
        "cluster_label": g.size().index,
        "n": g.size().to_numpy(),
        "median_duration_s": g["duration_s"].median().to_numpy(),
        "median_dSOC": g["dSOC"].median().to_numpy(),
        "median_I_mean": g["I_mean"].median().to_numpy(),
        "median_I_cv": (g["I_std"].median() / g["I_mean"].median().abs().replace(0, np.nan)).to_numpy(),
        "cv_tail_rate": g["has_cv_tail"].mean().to_numpy(),
        "median_V_max": g["V_max"].median().to_numpy(),
        "median_SOC_end": g["SOC_end"].median().to_numpy(),
    })
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def pick_cap_clusters(
    labeled: pd.DataFrame,
    *,
    min_median_dsoc: float = 50.0,
    min_median_soc_end: float = 95.0,
) -> list[int]:
    """Return every cluster that qualifies as a capacity population, deepest first.

    Admissibility is the whole rule. A cluster is a capacity population when

    * ``median_dSOC > min_median_dsoc`` (default 50) — it covers at least half
      the SOC window, and
    * ``median_SOC_end > min_median_soc_end`` (default 95) — it ends essentially
      full.

    Every cluster meeting both is returned; there is no single winner. The list
    is ordered by ``median_dSOC`` descending so the deepest population comes
    first, but callers are expected to use all of them.

    **Why all of them, rather than the best one.** The previous rule sorted on
    ``median_SOC_end`` and took the top cluster, breaking ties on
    ``median_dSOC``. SOC_end saturates — qualifying clusters share a median of
    exactly 97.6 — so that sort was decided entirely by the tie-break, and it
    discarded clusters that were equally valid capacity events merely for being
    slightly shallower. On a *modal* vehicle that is a real loss: vehicle 9 has
    three admissible bands at median dSOC 75.6, 59.4 and 41.6, of which the old
    rule kept only the first. Taking all qualifying clusters uses the whole
    measurable record instead of one band of it.

    The floors are what keeps that safe. They are deliberately loose — 50 and 95
    admit anything that plausibly is a capacity event — but they are hard, so a
    shallow cluster cannot enter simply because nothing better exists. At the 5%
    ``min_cluster_size`` this setting replaced, vehicle 9's only pick was a
    cluster at median dSOC 28.4 ending at SOC 76.8; it fails both floors and now
    yields an empty list rather than a capacity table with 2.70% scatter at a
    depth that reads several percent high.

    Why a depth floor at all: with the charge endpoint held fixed,
    capacity-vs-dSOC is flat above ~70 points and inflates below — roughly +1%
    at dSOC 50-60 against the >=70 reference, rising to ~+7% below 20. A cluster
    whose median dSOC sits below 50 is measuring the inflation, not the cell.
    Admitting the 50-70 band therefore accepts a known ~1% upward bias on those
    sessions in exchange for coverage.

    ``median_V_max`` was an earlier completeness criterion and is still computed
    in ``summarize_clusters``. SOC_end says the same thing in SOC points and is
    not lifted by the current-dependent IR offset that inflates V_max on fast
    charges. The CV-tail rate is deliberately unused — it was the criterion
    before that and made the stage circular (it is also a clustering input).
    Both remain useful as independent checks on what was selected.

    Note the floors cannot rescue a stranded population — they only reject a bad
    cluster. If a vehicle's deep sessions are all label -1, as vehicle 10's 60
    deepest still are, no admissible cluster contains them.
    """
    summary = summarize_clusters(labeled)
    candidates = summary[
        (summary["cluster_label"] != -1)
        & (summary["median_dSOC"] > min_median_dsoc)
        & (summary["median_SOC_end"] > min_median_soc_end)
    ]
    if candidates.empty:
        return []
    return [
        int(c)
        for c in candidates.sort_values("median_dSOC", ascending=False)["cluster_label"]
    ]


if __name__ == "__main__":
    import argparse
    from field import io_shiyunliu, sessions

    parser = argparse.ArgumentParser(description="Smoke-test HDBSCAN on shiyunliu session features")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="/home/ann/Documents/Data_Metabatt/field_data/shiyunliu_20ev",
    )
    parser.add_argument("--vehicle", default="1")
    parser.add_argument("--all", action="store_true", help="Run on all 20 vehicles")
    args = parser.parse_args()

    vehicles = io_shiyunliu.list_vehicles(args.base_dir) if args.all else [args.vehicle]
    for v in vehicles:
        print(f"\n=== vehicle #{v} ===")
        df = io_shiyunliu.load_vehicle(io_shiyunliu.vehicle_path(args.base_dir, v))
        feats = sessions.session_features(sessions.split_sessions(df), vehicle=v)
        if feats.empty:
            print("  no sessions kept — skipping")
            continue
        labeled = cluster_sessions(feats)
        n_noise = int((labeled["cluster_label"] == -1).sum())
        n_clusters = int(labeled.loc[labeled["cluster_label"] != -1, "cluster_label"].nunique())
        print(f"  sessions={len(labeled)}  clusters={n_clusters}  noise={n_noise}")
        summary = summarize_clusters(labeled)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        cap = pick_cap_clusters(labeled)
        if not cap:
            print("  ← no cluster selected")
        else:
            cap_rows = labeled[labeled["cluster_label"].isin(cap)]
            print(f"  ← CAP clusters: {cap}  (n={len(cap_rows)}, "
                  f"median dSOC={cap_rows['dSOC'].median():.1f}, "
                  f"SOC_end={cap_rows['SOC_end'].median():.1f}, "
                  f"CV-tail rate={cap_rows['has_cv_tail'].mean():.1%})")
