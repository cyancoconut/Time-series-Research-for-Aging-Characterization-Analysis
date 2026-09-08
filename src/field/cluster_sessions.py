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
# Clustering on dSOC alone is both non-circular and better. It partitions the
# excursion axis finely (~50 clusters); pick_cap_cluster then takes the slice
# with the highest median_V_max above a dSOC floor — "among deep charges, the
# ones that ended highest". Measured against the previous default, per vehicle:
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


def cluster_sessions(
    feats: pd.DataFrame,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Add a ``cluster_label`` column (-1 = noise) to the per-session features.

    Default ``min_cluster_size`` is ``max(10, 1% of session count)`` — small
    enough to surface a CAP cluster on a single vehicle (~50–90 candidates)
    but big enough to suppress micro-clusters of noise.
    """
    if feats.empty:
        out = feats.copy()
        out["cluster_label"] = pd.Series(dtype=int)
        return out

    mcs = min_cluster_size if min_cluster_size is not None else max(10, int(len(feats) * 0.01))
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
    })
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def pick_cap_cluster(
    labeled: pd.DataFrame,
    *,
    min_median_dsoc: float = 20.0,
) -> int | None:
    """Pick the cluster best matching 'a charge that ended full'.

    ``median_V_max`` is the completeness signal: a charge that ends high reached
    the top of charge, whereas a partial top-up stops at a lower peak voltage.
    Neither duration nor dSOC can stand in for it — both measure how much charge
    was added, not where the charge ended, so a deep charge starting from empty
    looks identical to a full one.

    Selection is therefore the cluster with the highest ``median_V_max``, subject
    to a ``median_dSOC`` floor that rules out shallow top-up slices (a sliver of
    charge can reach a high voltage without being a usable capacity sample).

    The CV-tail rate is deliberately not used here. It was the previous
    criterion and made the whole stage circular; it is still computed in
    ``summarize_clusters`` and is useful as an independent check on what was
    selected.
    """
    summary = summarize_clusters(labeled)
    candidates = summary[
        (summary["cluster_label"] != -1)
        & (summary["median_dSOC"] >= min_median_dsoc)
    ]
    if candidates.empty:
        return None
    return int(
        candidates.sort_values("median_V_max", ascending=False)
        .iloc[0]["cluster_label"]
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
            print("  ← no cluster passes min_dsoc=60")
        else:
            cap_rows = labeled[labeled["cluster_label"] == cap]
            print(f"  ← CAP cluster: {cap}  (n={len(cap_rows)}, "
                  f"median dSOC={cap_rows['dSOC'].median():.1f}, "
                  f"CV-tail rate={cap_rows['has_cv_tail'].mean():.1%})")
