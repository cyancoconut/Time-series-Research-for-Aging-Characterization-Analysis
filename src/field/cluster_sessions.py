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
    cap_label = pick_cap_cluster(labeled)
    cap_sessions = labeled[labeled["cluster_label"] == cap_label]
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
# Clustering on dSOC alone is both non-circular and better. pick_cap_cluster
# then takes the cluster with the highest median_SOC_end, ties broken by the
# broadest median_dSOC — "of the charges that ended full, the ones that came
# furthest". Measured against the previous default, per vehicle:
#
#     variant                      found   med n   resid%   coverage%
#     previous (has_cv_tail)       20/20     290     1.33          79
#     [V_max, SOC_end]             20/20      74     0.93          20
#     dSOC alone                   20/20      57     0.67          97
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
# the default at the time; the default is now 5% and already merges the bands
# structurally, so epsilon does much less work than it used to. It has not been
# re-swept against the coarser partition.
def cluster_sessions(
    feats: pd.DataFrame,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.03,
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Add a ``cluster_label`` column (-1 = noise) to the per-session features.

    Default ``min_cluster_size`` is ``max(10, 5% of session count)`` — 112-153
    on these vehicles. This is a coarse partition on purpose: it yields 2-3
    clusters per vehicle, one of which is the broad "charged most of the way and
    ended full" population, and every vehicle has one. At 1% (26-30) the same
    population splits into 5-10 narrow dSOC bands, and on vehicles 3 and 16 the
    deepest bands failed to reach cluster status at all and fell into noise,
    leaving those two with no CAP cluster and an empty capacity table.

    The trade is scatter for coverage — see ``pick_cap_cluster``. Below ~15 the
    partition fragments further and the pick turns unstable: vehicle 16 finds a
    CAP population at 20, loses it at 15, and finds a much smaller one at 10.

    Cluster membership counts are in any case an accident of how often a given
    driver charged from empty, not a property of the cell, so tying selection to
    a size threshold is inherently soft.
    """
    if feats.empty:
        out = feats.copy()
        out["cluster_label"] = pd.Series(dtype=int)
        return out

    mcs = min_cluster_size if min_cluster_size is not None else max(10, int(len(feats) * 0.05))
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


def pick_cap_cluster(
    labeled: pd.DataFrame,
    *,
    min_median_dsoc: float = 0.0,
) -> int | None:
    """Pick the cluster that ended fullest, and among those, charged broadest.

    * ``median_SOC_end`` — the completeness signal. A session that ends at the
      top of the SOC window reached full charge; a partial top-up stops lower.

      SOC saturates: the qualifying clusters share a median SOC_end of exactly
      97.6, so on its own this sort is decided by row order rather than by the
      criterion. ``median_dSOC`` therefore breaks the tie, descending — among
      the clusters that end full, the broadest charge. (``median_V_max``
      saturates the same way, ~380 V, so the criterion this replaced was equally
      tie-bound.)
    * ``median_dSOC >= min_median_dsoc`` — an optional depth floor, off by
      default. See the note on its removal below.

    ``median_V_max`` was the previous completeness criterion and is still
    computed in ``summarize_clusters``. SOC_end says the same thing in SOC
    points and is not lifted by the current-dependent IR offset that inflates
    V_max on fast charges. The CV-tail rate is deliberately unused — it was the
    criterion before that and made the stage circular (it is also a clustering
    input). Both remain useful as independent checks on what was selected.

    **The depth floor is off by default because it cannot coexist with the 5%
    min_cluster_size.** At that coarser partition the winning clusters sit at
    median dSOC 54-70, so a floor of 70 rejects every one of them and no vehicle
    yields a CAP cluster at all. Paired with 5%, this rule finds a cluster on
    20 of 20 vehicles where 1%-plus-floor-70 found 18.

    What that costs, measured over the fleet — 1% with a floor of 70 against 5%
    with none:

        variant           vehicles   total n   med n   med resid%   med cov%   med dSOC
        1%, floor 70         18/20      1219      52         0.61       96.6       79.9
        5%, no floor         20/20      7317     390         1.02       99.9       60.0

    Six times the sessions and near-total record coverage, at 1.7x the
    trend-residual scatter. The scatter is the price of a broad cluster: the
    selected population spans ~28 dSOC points p10-p90 against ~4 before, and its
    dSOC mix wanders 9-15 points from quarter to quarter. That wandering is
    non-monotonic, so it inflates scatter rather than faking a trend, and on
    trend precision the trade is favourable — the standard error of the ageing
    trend improves from 0.085 to 0.052.

    The bias it accepts is real but small. With the charge endpoint held fixed,
    capacity-vs-dSOC is flat above 70 points and inflates below: ~+1% at dSOC
    50-60 against the >=70 reference, rising to ~+7% below 20. A median dSOC of
    60 therefore reads roughly 1% high.

    **Vehicle 9 is the case to watch.** Without a floor it selects a cluster at
    median dSOC 28.4 — its deep sessions are all in noise at this
    min_cluster_size, 49% of the vehicle — giving 2.70% residual scatter, four
    times the fleet median, at a depth where the estimate reads several percent
    high. It is *found* rather than *usable*. A floor around 40 would return
    None there instead, at the cost of dropping back to 19 of 20.
    """
    summary = summarize_clusters(labeled)
    candidates = summary[
        (summary["cluster_label"] != -1)
        & (summary["median_dSOC"] >= min_median_dsoc)
    ]
    if candidates.empty:
        return None
    return int(
        candidates.sort_values(
            ["median_SOC_end", "median_dSOC"], ascending=[False, False]
        ).iloc[0]["cluster_label"]
    )


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
        cap = pick_cap_cluster(labeled)
        if cap is None:
            print("  ← no cluster selected")
        else:
            cap_rows = labeled[labeled["cluster_label"] == cap]
            print(f"  ← CAP cluster: {cap}  (n={len(cap_rows)}, "
                  f"median dSOC={cap_rows['dSOC'].median():.1f}, "
                  f"SOC_end={cap_rows['SOC_end'].median():.1f}, "
                  f"CV-tail rate={cap_rows['has_cv_tail'].mean():.1%})")
