from sklearn_som.som import SOM
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import folium
from folium.plugins import MarkerCluster
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min
import re
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
import torch
from config import TRAIN_CONFIG, ALL_PFAS_TARGETS, ADMIN_COLS

def _get_env_feature_columns(df):
    """Returns columns that are not PFAS targets or admin fields."""
    cols_exclude_prefixes = [*ALL_PFAS_TARGETS, *ADMIN_COLS, 'PFAS_total']
    cols_exclude_prefixes = [col.lower() for col in cols_exclude_prefixes]

    def should_exclude(col_name):
        col_name = col_name.lower()
        for excl_col in cols_exclude_prefixes:
            if col_name == excl_col or col_name.startswith(f"{excl_col}_"):
                return True
        return False

    return [col for col in df.columns if not should_exclude(col)]


def _fit_env_transform(df):
    """
    Fits the full environmental feature transform pipeline on a training DataFrame.
    Applies feature selection, importance weighting, and PCA reduction.
 
    Returns (X_reduced, artifacts) on success, or (None, error_string) if the data
    is too sparse, static, or small to cluster meaningfully.
 
    Parameters
    df : training DataFrame
    """
    feature_cols = _get_env_feature_columns(df)
    X_df = df[feature_cols].copy()

    # Numeric only, same rule as clustering.
    X_num = X_df.select_dtypes(include=['number']).fillna(0)

    if X_num.shape[1] == 0:
        return None, "ANOMALY_NO_NUM"

    variances = X_num.var()
    variable_cols = variances[variances > 1e-8].index.tolist()

    if len(variable_cols) < 5:
        return None, "ANOMALY_STATIC"

    X_num = X_num[variable_cols]
    X_raw = X_num.to_numpy(dtype='float64')

    if len(X_raw) < 75:
        return None, "ANOMALY_SIZE"

    if np.std(X_raw, axis=0).sum() < 1e-6:
        return None, "ANOMALY_STATIC"

    feat_weights = TRAIN_CONFIG['env_cluster_weighted']
    weights = np.array([np.sqrt(feat_weights.get(col, 1.0)) for col in variable_cols])

    X_weighted = X_raw * weights

    n_components = min(8, X_weighted.shape[1], X_weighted.shape[0] - 1)
    if n_components < 1:
        return None, "ANOMALY_STATIC"

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_weighted)

    artifacts = {
        'feature_cols': feature_cols,
        'variable_cols': variable_cols,
        'weights': weights,
        'pca': pca,
    }
    return (X_reduced, artifacts), None


def _transform_env_with_artifacts(df, artifacts):
    """
    Applies a previously fit transform pipeline to a new DataFrame.
 
    Parameters
    df        : DataFrame to transform
    artifacts : dict returned by _fit_env_transform
    """
    variable_cols = artifacts['variable_cols']
    weights = artifacts['weights']
    pca = artifacts['pca']

    X_num = df.reindex(columns=variable_cols, fill_value=0)
    X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_raw = X_num.to_numpy(dtype='float64')
    X_weighted = X_raw * weights
    X_reduced = pca.transform(X_weighted)
    return X_reduced

def check_intra_cluster_divergence(agents, threshold=0.85):
    """
    Identifies clusters whose internal weight similarity has dropped below threshold.
 
    Parameters
    agents    : agent dict
    threshold : cosine similarity below which a cluster is considered diverging
 
    Returns
    diverging_clusters : list of cluster_ids with avg internal similarity < threshold
    """
    diverging_clusters = []
    
    # Group agents by their assigned cluster_id.
    clusters = {}
    for nid, data in agents.items():
        cid = data['train_df']['cluster_id'].iloc[0] 
        if cid not in clusters: clusters[cid] = []
        clusters[cid].append(nid)

    for cid, node_ids in clusters.items():
        if len(node_ids) < 2: continue
        
        # Calculate mean pairwise similarity within the cluster.
        sims = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                # Flatten weights to vectors.
                w1 = torch.cat([p.flatten() for p in agents[node_ids[i]]['model'].parameters()])
                w2 = torch.cat([p.flatten() for p in agents[node_ids[j]]['model'].parameters()])
                
                sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0)
                sims.append(sim.item())
        
        avg_internal_sim = np.mean(sims)
        
        # If the cluster is drifting apart internally, it's diverging.
        if avg_internal_sim < threshold:
            diverging_clusters.append(cid)
            
    return diverging_clusters

def env_clustering(df):
    """
    Clusters wells by environmental features using a SOM followed by
    agglomerative clustering on the SOM codebook. Assigns node tiers
    by cluster size quantile (top third = tier 1, middle = tier 2, bottom = tier 3).
 
    Also saves transform artifacts needed to assign test points to the same space.
 
    Parameters
    df : training DataFrame with environmental features
 
    Returns
    df        : DataFrame with added 'cluster_id' and 'node_tier' columns
    artifacts : transform and clustering artifacts for test assignment, or None on failure
    """
    df = df.copy()

    fit_result, err = _fit_env_transform(df)
    if err is not None:
        df['cluster_id'] = -1
        df['node_tier'] = 3
        return df, None

    (X_reduced, artifacts) = fit_result

    n_samples = X_reduced.shape[0]
    grid_size = int(np.sqrt(5 * np.sqrt(n_samples)))
    grid_size = max(grid_size, 1)

    som = SOM(m=grid_size, n=grid_size, dim=X_reduced.shape[1], lr=0.05, sigma=0.3)
    som.fit(X_reduced)
    winner_neurons = som.predict(X_reduced)

    codebook = som.weights
    threshold = 0.6
    print(f"Unique SOM neurons activated: {len(np.unique(winner_neurons))}")

    aggCluster = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='cosine',
        linkage='average'
    )
    neuron_group_labels = aggCluster.fit_predict(codebook)

    df['cluster_id'] = neuron_group_labels[winner_neurons]

    cluster_sizes = df.groupby('cluster_id').size()
    print(f"Clusters formed: {len(cluster_sizes)}")
    print(f"Cluster size — min: {cluster_sizes.min()}, median: {int(cluster_sizes.median())}, max: {cluster_sizes.max()}")

    size_33 = cluster_sizes.quantile(0.33)
    size_66 = cluster_sizes.quantile(0.66)

    def assign_tier(cluster_id):
        s = cluster_sizes[cluster_id]
        if s >= size_66:
            return 1
        if s >= size_33:
            return 2
        return 3

    df['node_tier'] = df['cluster_id'].map(assign_tier)

    artifacts['som'] = som
    artifacts['winner_neurons'] = winner_neurons
    artifacts['neuron_group_labels'] = neuron_group_labels

    # Store per-cluster centroids in PCA-reduced weighted space for test assignment.
    reduced_centroids = {}
    for cluster_id, group_df in df.groupby('cluster_id', sort=False):
        row_pos = df.index.get_indexer(group_df.index)
        reduced_centroids[cluster_id] = X_reduced[row_pos].mean(axis=0)

    artifacts['reduced_centroids'] = reduced_centroids

    return df, artifacts


def test_env_clustering(train_df, test_data, cluster_artifacts):
    """
    Assigns test rows to the nearest training cluster by cosine similarity
    to stored cluster centroids. Warns on low-confidence assignments.
 
    Parameters
    train_df          : training DataFrame (used for context, not re-fitted)
    test_data         : test DataFrame to assign
    cluster_artifacts : artifacts dict returned by env_clustering
 
    Returns
    test_data : test DataFrame with 'cluster_id' column added
    """
    test_data = test_data.copy()

    if cluster_artifacts is None:
        test_data['cluster_id'] = -1
        return test_data

    X_test_reduced = _transform_env_with_artifacts(test_data, cluster_artifacts)
    cluster_centroids = cluster_artifacts['reduced_centroids']

    cluster_ids = list(cluster_centroids.keys())
    if not cluster_ids:
        test_data['cluster_id'] = -1
        return test_data

    test_assignments = []

    for i, test_vec in enumerate(X_test_reduced):
        first_cluster = cluster_ids[0]
        best_cluster = first_cluster
        max_sim = cosineSimilarity(test_vec, cluster_centroids[first_cluster])

        for cluster_id in cluster_ids[1:]:
            sim = cosineSimilarity(test_vec, cluster_centroids[cluster_id])
            if sim > max_sim:
                max_sim = sim
                best_cluster = cluster_id

        if max_sim < 0.4:
            print(f"Warning: Row {i} has low similarity ({max_sim:.3f}) with all clusters.")

        test_assignments.append(best_cluster)

    test_data['cluster_id'] = test_assignments

    print("\nTest cluster assignment counts:")
    print(test_data['cluster_id'].value_counts(dropna=False).sort_values(ascending=False))

    plt.show()

    return test_data

def cosineSimilarity(a, b):
    """
    Computes the cosine similarity between two vectors. Handles zero-norm 
    vectors by returning 0.0 and warns if inputs are None.
 
    Parameters
    a : first vector (numpy array or list)
    b : second vector (numpy array or list)
 
    Returns
    similarity : float (ranging from -1.0 to 1.0) or None if inputs are invalid
    """
    if a is not None and b is not None: # Not none.
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    else:
        print("Centroid evaluated the 2 vectors to None. No basis for cosine simi comparison")
        return None