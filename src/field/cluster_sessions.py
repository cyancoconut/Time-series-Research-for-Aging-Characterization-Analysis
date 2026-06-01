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


# Per-session features fed to HDBSCAN. duration is log-scaled before standardising
# because it spans roughly 3 orders of magnitude (~30 s … ~100 ks).
DEFAULT_FEATURE_COLUMNS = [
    "duration_s_log",
    "dSOC",
    "I_mean",
    "I_cv",          # std / mean — shape sensitivity, dimensionless
    "V_max",
    "has_cv_tail",   # 0/1
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
    min_cv_tail_rate: float = 0.5,
    min_median_dsoc: float = 30.0,
) -> int | None:
    """Pick the cluster best matching 'full CC-CV charge'.

    CV tail is the defining signature — a session that reaches CV has completed
    its CC ramp and is suitable for coulomb-counting. Clusters with
    ``cv_tail_rate < min_cv_tail_rate`` are excluded as 'mostly partial top-ups'.
    Among the remainder, rank by ``median_dSOC * cv_tail_rate``. A floor on
    ``median_dSOC`` prevents picking a tiny end-of-charge top-up cluster that
    happens to have many CV-tail samples.
    """
    summary = summarize_clusters(labeled)
    candidates = summary[
        (summary["cluster_label"] != -1)
        & (summary["cv_tail_rate"] >= min_cv_tail_rate)
        & (summary["median_dSOC"] >= min_median_dsoc)
    ]
    if candidates.empty:
        return None
    candidates = candidates.copy()
    candidates["score"] = candidates["median_dSOC"] * candidates["cv_tail_rate"]
    return int(candidates.sort_values("score", ascending=False).iloc[0]["cluster_label"])


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
