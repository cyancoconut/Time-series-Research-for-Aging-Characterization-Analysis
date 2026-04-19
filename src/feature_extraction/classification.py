# classification.py
# -*- coding: utf-8 -*-

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

import matplotlib.pyplot as plt

# Scale the features
scaler = StandardScaler()


class FeatureExtraction:
    def __init__(self, max_voltage, min_voltage, nom_voltage, nominal_capacity):
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.nom_voltage = nom_voltage
        self.nominal_capacity = nominal_capacity

    def extract_features(self, group, columns):
        """Extract relevant features from a group of time series data"""
        features = {}
        for col in columns:
            if col == "Voltage":
                normalize_max = self.max_voltage
                normalize_min = self.min_voltage
            elif col == "Power":
                normalize_max = self.nom_voltage * self.nominal_capacity
                normalize_min = self.nom_voltage * self.nominal_capacity
            else:
                normalize_max = self.nominal_capacity
                normalize_min = self.nominal_capacity

            if col == "Voltage":
                df = pd.DataFrame()
                df["voltage_normalized"] = (group[col] - normalize_min) / (
                    normalize_max - normalize_min
                )
                features[f"{col}_mean"] = df["voltage_normalized"].mean()
                features[f"{col}_std"] = df["voltage_normalized"].std()
                features[f"{col}_max"] = df["voltage_normalized"].max()
                features[f"{col}_min"] = df["voltage_normalized"].min()
                features[f"{col}_range"] = (
                    df["voltage_normalized"].max() - df["voltage_normalized"].min()
                )

            elif col == "Duration_minutes":
                features[col] = group[col].mean() / 1200

            elif col == "Temperature":
                features[f"{col}_mean"] = group[col].mean()

            else:  # current can be faulty because of stationary values
                features[f"{col}_mean"] = (group[col][2:-1].mean()) / (
                    (normalize_max + normalize_min) / 2
                )
                features[f"{col}_std"] = group[col][2:-1].std() / (
                    (normalize_max + normalize_min) / 2
                )
                max_and_min = [abs(group[col][2:-1]).max(), abs(group[col][2:-1]).min()]
                features[f"{col}_max"] = (
                    np.sign(features[f"{col}_mean"]) * max(max_and_min) / normalize_max
                )
                features[f"{col}_min"] = (
                    np.sign(features[f"{col}_mean"]) * min(max_and_min) / normalize_min
                )
                features[f"{col}_range"] = (
                    abs(group[col][2:-1]).max() / normalize_max
                ) - (abs(group[col][2:-1]).min() / normalize_min)

            # features[f'{col}_trend'] = np.polyfit(np.arange(len(group)), group[col], 1)[0]
        return pd.Series(features)

    def get_X_unlabled(self, df, feature_columns):
        features = (
            df.groupby("ID")[feature_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, feature_columns)
            )
            .reset_index()
        )

        return features

    def get_X_labled(self, df, feature_columns):
        features = (
            df.groupby("ID")[feature_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, feature_columns)
            )
            .reset_index()
        )
        return features

    def get_y_labled(self, df, target_column):
        y_labeled = df.groupby("ID")[target_column].first().reset_index()[target_column]
        return y_labeled

    def isolation_forest(self, df, df_classified, outlier_columns):
        # Create an Isolation Forest model
        iso_forest = IsolationForest(
            contamination=0.006, random_state=42
        )  # Adjust contamination based on your dataset
        # X_outliers = df.groupby("ID")[outlier_columns].apply(lambda x: extract_features(x, outlier_columns)).reset_index()

        X_normal = (
            df_classified.groupby("ID")[outlier_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, outlier_columns)
            )
            .reset_index(drop=True)
        )
        X_test = (
            df.groupby("ID")[outlier_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, outlier_columns)
            )
            .reset_index(drop=True)
        )

        X_all_scaled = scaler.fit_transform(pd.concat([X_normal, X_test]))

        # Split the scaled data back into normal and test sets
        X_normal_scaled = X_all_scaled[: len(X_normal)]
        X_test_scaled = X_all_scaled[len(X_normal) :]

        # Train the model
        iso_forest.fit(X_normal_scaled)
        outlier_labels = iso_forest.predict(X_test_scaled)
        outliers = np.where(outlier_labels == -1)[0]

        return outliers

    def cluster_outlier_detection(self, df, df_classified, outlier_columns):
        # Combine labeled and unlabeled data
        X_labeled_outliers = (
            df_classified.groupby("ID")[outlier_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, outlier_columns)
            )
            .reset_index()
        )
        X_unlabeled_outliers = (
            df.groupby("ID")[outlier_columns]
            .apply(
                lambda x: FeatureExtraction.extract_features(self, x, outlier_columns)
            )
            .reset_index()
        )

        X_combined = np.vstack((X_labeled_outliers, X_unlabeled_outliers))
        X_combined_scaled = scaler.fit_transform(X_combined)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_combined_scaled)
        print(f"Cluster labels: {np.unique(cluster_labels)}")
        # Calculate silhouette score to evaluate cluster quality
        silhouette = silhouette_score(X_combined_scaled, cluster_labels)

        # Identify the cluster with the lowest density (i.e., the outliers)
        cluster_sizes = np.bincount(cluster_labels)
        outlier_cluster = np.argmin(cluster_sizes)
        print(f"Outlier cluster: {outlier_cluster}")
        # Get the indices of the samples in the outlier cluster
        outlier_indices = np.where(cluster_labels == outlier_cluster)[0]

        n_labeled = X_labeled_outliers.shape[0]

        # Filter the outlier indices to only include samples from the unlabeled data
        unlabeled_outliers = outlier_indices[outlier_indices[n_labeled:]]

        return unlabeled_outliers


def random_forest(X_labeled, X_unlabeled, y_labeled):

    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    # Initialize the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(clf, X_labeled_scaled, y_labeled.tolist(), cv=3)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Train the model on all labeled data
    clf.fit(X_labeled_scaled, y_labeled.tolist())

    # Predict unlabeled data
    y_unlabeled_pred = clf.predict(X_unlabeled_scaled)

    return y_unlabeled_pred


class label_encoder:

    def __init__(self):
        self.le = LabelEncoder()
        self.is_fitted = False

    def encoding(self, df):
        if not self.is_fitted:
            self.le.fit(df.loc[~df["target"].isin([-1, "EIS", "EIS-Diga"]), "target"])
            self.is_fitted = True
        df.loc[df["target"] != -1, "target"] = self.le.transform(
            df.loc[df["target"] != -1, "target"]
        )
        print(dict(zip(self.le.classes_, self.le.transform(self.le.classes_))))

        return df

    def inverse_encoding(self, arr):
        if not self.is_fitted:
            raise ValueError(
                "LabelEncoder has not been fitted. Call encoding() method first."
            )
        arr = arr.astype(int)
        return self.le.inverse_transform(arr)


def get_ID_from_index(arr, X_unlabeled):
    df_y = pd.DataFrame({"index": arr})
    df_y = df_y.merge(X_unlabeled["ID"], on="index", how="left")
    return df_y


def merge_target(df, X_unlabeled, y_unlabeled_pred):

    df_y = pd.DataFrame({"target": y_unlabeled_pred})
    df_y = df_y.join(X_unlabeled["ID"].reset_index()["ID"])

    df_merged = df.merge(df_y, on="ID", how="left", suffixes=("_x", "_y"))
    df_merged["target"] = df_merged["target_y"].fillna(df_merged["target_x"])

    df_final = df_merged.drop(["target_x", "target_y"], axis=1)

    return df_final
