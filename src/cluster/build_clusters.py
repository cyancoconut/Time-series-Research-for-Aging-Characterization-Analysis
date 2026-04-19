import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

import matplotlib.pyplot as plt

# Scale the features
scaler = StandardScaler()


def define_clusters(cluster):
    print(len(cluster))


def merge_target(df, X_unlabeled):
    if "target" in df.columns:
        df_merged = df.merge(
            X_unlabeled[["ID", "target"]], on="ID", how="left", suffixes=("_x", "_y")
        )
        df_merged["target"] = df_merged["target_y"].fillna(df_merged["target_x"])

        df_final = df_merged.drop(["target_x", "target_y"], axis=1)

    else:
        df_final = df.merge(X_unlabeled[["ID", "target"]], on="ID", how="left")
    return df_final
