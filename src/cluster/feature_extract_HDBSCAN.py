import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics.pairwise import euclidean_distances

np.random.seed(42)

def _get_tf():
    import tensorflow as tf
    tf.random.set_seed(42)
    return tf

def _get_keras():
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, LeakyReLU
    return Model, Sequential, Dense, Input, LeakyReLU

class TabularAutoencoderHDBSCAN:
    def __init__(self, input_dim, encoding_dim=10, hdbscan_params=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hdbscan_params = hdbscan_params or {
            'min_cluster_size': 5,
            'min_samples': 5
            #'cluster_selection_epsilon': 0.5
        }
        self.scaler = StandardScaler()

        self.clusterer = HDBSCAN(**self.hdbscan_params)
        self.autoencoder = None  # Will be set in fit
        
    def fit(self, df, autoencoder_epochs=50, batch_size=32):
        tf = _get_tf()
        Model, Sequential, Dense, Input, LeakyReLU = _get_keras()
        if "target" in df.columns:
            # print("Target column dropped")
            df.drop("target", axis=1, inplace=True)

        input_features = df.drop("ID", axis=1)
        self.autoencoder = Sequential([
            Dense(int(self.input_dim/2), activation="relu", input_shape=(self.input_dim,)),
            Dense(int(self.input_dim/4), activation="relu"),
            Dense(2, activation="relu"),
            Dense(int(self.input_dim/4), activation="relu"),
            Dense(int(self.input_dim/2), activation="relu"),
            Dense(int(self.input_dim), activation="relu"),
        ])

        self.autoencoder.compile(optimizer='adam', loss='mse')
        history = self.autoencoder.fit(input_features, input_features, 
                                     epochs=autoencoder_epochs, 
                                     batch_size=batch_size, 
                                     verbose=0)
        self.autoencoder.summary()

        # Create a model that outputs layer_4 activations
        Model, *_ = _get_keras()
        intermediate_layer_model = Model(inputs=self.autoencoder.layers[0].input,
                                      outputs=self.autoencoder.layers[5].output)

        # Get layer_4 activations
        layer_4_output = intermediate_layer_model.predict(input_features)
        
        # Perform HDBSCAN clustering
        self.labels_ = self.clusterer.fit_predict(layer_4_output)

        df["target"] = np.nan
        for cluster in set(self.labels_):
            # print(f"\nCluster {cluster}:")
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
            # print(df[mask])

        return self, df
    
    
    def fit_cluster_only(self, df, autoencoder_epochs=50, batch_size=32):
        if "target" in df.columns:
            # print("Target column dropped")
            df.drop("target", axis=1, inplace=True)

        input_features = df.drop("ID", axis=1)

        # Perform HDBSCAN clustering
        self.labels_ = self.clusterer.fit_predict(input_features)

        df["target"] = np.nan
        for cluster in set(self.labels_):
            # print(f"\nCluster {cluster}:")
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
            # print(df[mask])

        return self, df
    
    def fit_cluster_with_partial_init(self, df, n_clusters=4, known_centers=None, autoencoder_epochs=50, batch_size=32):
        if "target" in df.columns:
            df.drop("target", axis=1, inplace=True)
        
        input_features = df.drop("ID", axis=1)
        

        
        # Handle known centers if provided
        if known_centers is not None:
            # Convert to numpy array if it's not already
            known_centers = np.array(known_centers).reshape(-1, 1)
            # Generate random centers for the remaining clusters
            remaining_centers = np.random.uniform(input_features.min(), input_features.max(), 
                                                (n_clusters - len(known_centers), 1))
            # Combine known and random centers
            initial_centers = np.vstack([known_centers, remaining_centers])
        else:
            initial_centers = 'k-means++'  # Use default initialization if no centers provided
        
        # Initialize KMeans clusterer
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=initial_centers if known_centers is not None else 'k-means++',
            n_init=1 if known_centers is not None else 10,
            max_iter=300,
            random_state=42
        )
        
        # Fit and predict
        self.labels_ = kmeans.fit_predict(input_features)
        self.clusterer = kmeans
        
        # Assign targets
        df["target"] = np.nan
        for cluster in set(self.labels_):
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
        
        return self, df
    
    def fit_with_autoscaler(self, df, autoencoder_epochs=50, batch_size=32):
        _get_tf()
        Model, Sequential, Dense, Input, LeakyReLU = _get_keras()
        if "target" in df.columns:
            # print("Target column dropped")
            df.drop("target", axis=1, inplace=True)

        #input_features = df.drop("ID", axis=1)

        X_scaled = df.drop('ID', axis=1) #self.scaler.fit_transform(df.drop('ID', axis=1))

        self.autoencoder = Sequential([
            Dense(int(self.input_dim/1), activation="relu", input_shape=(self.input_dim,)),
            Dense(int(self.input_dim/1), activation="relu"),
            Dense(2, activation="relu"),
            Dense(int(self.input_dim/1), activation="relu"),
            Dense(int(self.input_dim/1), activation="relu"),
            Dense(int(self.input_dim), activation="relu"),
        ])

        self.autoencoder.compile(optimizer='adam', loss='mse')
        history = self.autoencoder.fit(X_scaled, X_scaled, 
                                     epochs=autoencoder_epochs, 
                                     batch_size=batch_size, 
                                     verbose=0)
        self.autoencoder.summary()

        # Create a model that outputs layer_4 activations
        Model, *_ = _get_keras()
        intermediate_layer_model = Model(inputs=self.autoencoder.layers[0].input,
                                      outputs=self.autoencoder.layers[0].output)

        # Get layer_4 activations
        layer_0_output = intermediate_layer_model.predict(X_scaled)
        
        # Perform HDBSCAN clustering
        self.labels_ = self.clusterer.fit_predict(layer_0_output)
        df["target"] = np.nan
        for cluster in set(self.labels_):
            print(f"\nCluster {cluster}:")
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
            print(df[mask])

        return self, df
        
    def predict(self, df):
        Model, *_ = _get_keras()
        input_features = df.drop("ID", axis=1)
        intermediate_layer_model = Model(inputs=self.autoencoder.layers[0].input,
                                      outputs=self.autoencoder.layers[3].output)
        layer_4_output = intermediate_layer_model.predict(input_features)

        self.labels_ = self.clusterer.fit_predict(layer_4_output)

        df["target"] = np.nan
        for cluster in set(self.labels_):
            # print(f"\nCluster {cluster}:")
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
            # print(df[mask])

        return self, df
    
    def predict_hdbscan_only(self, df):
        input_features = df.drop("ID", axis=1)

        self.labels_ = self.clusterer.fit_predict(input_features)

        df["target"] = np.nan
        for cluster in set(self.labels_):
            # print(f"\nCluster {cluster}:")
            mask = self.labels_ == cluster
            df.loc[mask, "target"] = cluster
            # print(df[mask])

        return self, df
    
    def evaluate_reconstruction(self, df, n_features=5):
        """Compare original vs reconstructed values for selected features"""
        input_features = df.drop("ID", axis=1)
        reconstructed = self.autoencoder.predict(input_features)
        
        features = input_features.columns[:n_features]
        
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
        for i, feature in enumerate(features):
            sns.scatterplot(x=input_features[feature], y=reconstructed[:, i], ax=axes[i])
            axes[i].set_title(f'{feature}: Original vs Reconstructed')
            axes[i].set_xlabel('Original Value')
            axes[i].set_ylabel('Reconstructed Value')
            # Add perfect reconstruction line
            lims = [
                min(axes[i].get_xlim()[0], axes[i].get_ylim()[0]),
                max(axes[i].get_xlim()[1], axes[i].get_ylim()[1]),
            ]
            axes[i].plot(lims, lims, 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Return reconstruction error
        mse = np.mean((input_features.values - reconstructed) ** 2, axis=0)
        return pd.Series(mse, index=input_features.columns, name='MSE')
    
    def visualize_clusters(self, df):
        """Visualize clusters in encoded space"""
        input_features = df.drop(['target', 'ID'], axis=1)
        
        # Get the bottleneck layer (layer 3) for visualization
        bottleneck_model = Model(inputs=self.autoencoder.layers[0].input,
                               outputs=self.autoencoder.layers[4].output)
        encoded_features = bottleneck_model.predict(input_features)
        
        # plt.figure(figsize=(10, 6))
        # scatter = plt.scatter(encoded_features[:, 0], encoded_features[:, 1],
        #                     c=self.labels_, cmap='viridis')
        # plt.colorbar(scatter)
        # plt.title('Clusters in Encoded Space')
        # plt.xlabel('First Encoded Dimension')
        # plt.ylabel('Second Encoded Dimension')
        # plt.show()
        
        # Feature importance for clusters
        df_with_clusters = input_features.copy()
        df_with_clusters['Cluster'] = self.labels_
        
        cluster_means = df_with_clusters.groupby('Cluster').mean()
        cluster_std = df_with_clusters.groupby('Cluster').std()
        cluster_size = df_with_clusters.groupby('Cluster').size()
        
        return cluster_means, cluster_std, cluster_size

    def visualize_clusters_only(self, df):
        """Visualize clusters in encoded space"""
        input_features = df.drop(['target', 'ID'], axis=1)
    
        
        # Feature importance for clusters
        df_with_clusters = input_features.copy()
        df_with_clusters['Cluster'] = self.labels_
        
        cluster_means = df_with_clusters.groupby('Cluster').mean()
        cluster_std = df_with_clusters.groupby('Cluster').std()
        cluster_size = df_with_clusters.groupby('Cluster').size()
        
        return cluster_means, cluster_std, cluster_size
    
    def save_model(self, path):
        """Save all model components"""
        self.autoencoder.save(f'{path}_autoencoder.keras')
        joblib.dump(self.clusterer, f'{path}_hdbscan.joblib')

    def save_model_hdbscan(self, path):
        """Save all model components"""
        joblib.dump(self.clusterer, f'{path}_hdbscan.joblib')
        
    def load_model(self, path):
        """Load all model components"""
        tf = _get_tf()
        self.autoencoder = tf.keras.models.load_model(f'{path}_autoencoder.keras')
        self.clusterer = joblib.load(f'{path}_hdbscan.joblib')

    def load_model_hdbscan(self, path):
        """Load all model components"""
        self.clusterer = joblib.load(f'{path}_hdbscan.joblib')

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# class TabularAutoencoderKMeans:
#     def __init__(self, input_dim, encoding_dim=10, n_clusters=5):
#         self.input_dim = input_dim
#         self.encoding_dim = encoding_dim
#         self.n_clusters = n_clusters
#         self.scaler = StandardScaler()
#         self.autoencoder = self._build_autoencoder()
#         self.encoder = self._build_encoder()
#         self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        
#     def _build_autoencoder(self):
#         # Encoder
#         input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
#         encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
#         encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
#         encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
#         encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
#         encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
#         # Decoder
#         decoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
#         decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
#         decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
#         decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
#         decoded = tf.keras.layers.Dense(self.input_dim, activation='linear')(decoded)
        
#         autoencoder = tf.keras.Model(input_layer, decoded)
#         autoencoder.compile(optimizer='adam', loss='mse')
#         return autoencoder
    
#     def _build_encoder(self):
#         return tf.keras.Model(self.autoencoder.input,
#                             self.autoencoder.layers[5].output)
    
#     def fit(self, df, autoencoder_epochs=50, batch_size=32):
#         if "target" in df.columns:
#             # print("Target column dropped")
#             df.drop("target", axis=1, inplace=True)
        
#         # Scale data
#         X_scaled = self.scaler.fit_transform(df.drop('ID', axis=1))
        
#         # Train autoencoder
#         history = self.autoencoder.fit(X_scaled, X_scaled,
#                                      epochs=autoencoder_epochs,
#                                      batch_size=batch_size,
#                                      verbose=0)
        
#         # Get encoded features
#         encoded_features = self.encoder.predict(X_scaled)
        
#         # Perform K-means clustering
#         self.labels_ = self.clusterer.fit_predict(encoded_features)
        
#         df["target"] = self.labels_
        
#         for cluster in range(self.n_clusters):
#             # print(f"\nCluster {cluster}:")
#             mask = self.labels_ == cluster
#             # print(df[mask])
            
#         return self, history, df
        
#     def predict(self, df):
#         X_scaled = self.scaler.transform(df)
#         encoded_features = self.encoder.predict(X_scaled)
#         return self.clusterer.predict(encoded_features)
    
#     def evaluate_reconstruction(self, df, n_features=5):
#         """Compare original vs reconstructed values for selected features"""
#         X_scaled = self.scaler.transform(df)
#         reconstructed_scaled = self.autoencoder.predict(X_scaled)
#         reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        
#         features = df.columns[:n_features]
        
#         fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
#         for i, feature in enumerate(features):
#             sns.scatterplot(x=df[feature], y=reconstructed[:, i], ax=axes[i])
#             axes[i].set_title(f'{feature}: Original vs Reconstructed')
#             axes[i].set_xlabel('Original Value')
#             axes[i].set_ylabel('Reconstructed Value')
#             lims = [
#                 min(axes[i].get_xlim()[0], axes[i].get_ylim()[0]),
#                 max(axes[i].get_xlim()[1], axes[i].get_ylim()[1]),
#             ]
#             axes[i].plot(lims, lims, 'r--', alpha=0.5)
        
#         plt.tight_layout()
#         plt.show()
        
#         mse = np.mean((df.values - reconstructed) ** 2, axis=0)
#         return pd.Series(mse, index=df.columns, name='MSE')
    
#     def visualize_clusters(self, df):
#         """Visualize clusters in encoded space"""
#         X_scaled = self.scaler.transform(df.drop(['target', 'ID'], axis=1))
#         encoded_features = self.encoder.predict(X_scaled)
        
#         plt.figure(figsize=(10, 6))
#         scatter = plt.scatter(encoded_features[:, 0], encoded_features[:, 1],
#                             c=self.labels_, cmap='viridis')
#         plt.colorbar(scatter)
#         plt.title('Clusters in Encoded Space')
#         plt.xlabel('First Encoded Dimension')
#         plt.ylabel('Second Encoded Dimension')
#         plt.show()
        
#         # Feature importance for clusters
#         df_with_clusters = df.drop(['target', 'ID'], axis=1).copy()
#         df_with_clusters['Cluster'] = self.labels_
        
#         cluster_means = df_with_clusters.groupby('Cluster').mean()
#         plt.figure(figsize=(12, 8))
#         sns.heatmap(cluster_means, cmap='RdBu_r', center=0)
#         plt.title('Feature Values Across Clusters')
#         plt.show()
        
#         return cluster_means
    
#     def save_model(self, path):
#         """Save all model components"""
#         self.autoencoder.save(f'{path}_autoencoder.keras')
#         np.save(f'{path}_scaler_mean.npy', self.scaler.mean_)
#         np.save(f'{path}_scaler_scale.npy', self.scaler.scale_)
#         np.save(f'{path}_kmeans_centers.npy', self.clusterer.cluster_centers_)
        
#     def load_model(self, path):
#         """Load all model components"""
#         self.autoencoder = tf.keras.models.load_model(f'{path}_autoencoder.keras')
#         self.encoder = tf.keras.Model(self.autoencoder.input,
#                                     self.autoencoder.layers[5].output)
#         self.scaler.mean_ = np.load(f'{path}_scaler_mean.npy')
#         self.scaler.scale_ = np.load(f'{path}_scaler_scale.npy')
#         self.clusterer.cluster_centers_ = np.load(f'{path}_kmeans_centers.npy')