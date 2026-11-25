"""
Evaluation and Experimental Comparison Module
Part of BEACHMIND (SIH26054)

Executes benchmark experiments:
- Experiment 1: Standardized Raw Features + K-Means
- Experiment 2: PCA-Reduced Features + K-Means
- Experiment 3: Standardized Raw Features + DBSCAN
- Experiment 4: PCA-Reduced Features + DBSCAN

Calculates:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Number of Clusters & Noise Points
- Dimensionality Reduction Ratio
- Execution Latency (ms)
- PCA Component Selection sensitivity testing (2, 3, 5, 10, 95% variance)
"""

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, List, Optional, Tuple

from modules.dimensionality_reduction import run_pca_analysis


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous internal clustering validation indices.
    Returns np.nan or 'N/A' when metrics are mathematically undefined.
    Excludes noise samples (-1) for density clustering metrics where appropriate.
    """
    unique_labels = set(labels)
    non_noise_mask = labels != -1
    clean_labels = labels[non_noise_mask]
    clean_X = X[non_noise_mask]
    n_clean_clusters = len(set(clean_labels))

    # Metric validation checks
    # Silhouette requires at least 2 distinct clusters and at least 1 sample in each
    if n_clean_clusters >= 2 and len(clean_labels) > n_clean_clusters:
        try:
            sil = float(silhouette_score(clean_X, clean_labels))
        except Exception:
            sil = np.nan
        try:
            db = float(davies_bouldin_score(clean_X, clean_labels))
        except Exception:
            db = np.nan
        try:
            ch = float(calinski_harabasz_score(clean_X, clean_labels))
        except Exception:
            ch = np.nan
    else:
        sil = np.nan
        db = np.nan
        ch = np.nan

    return {
        "silhouette_score": sil,
        "davies_bouldin_index": db,
        "calinski_harabasz_index": ch,
        "n_clusters": int(n_clean_clusters),
        "n_noise": int(np.sum(labels == -1))
    }


def run_four_way_benchmark(
    X_scaled: np.ndarray,
    X_pca: np.ndarray,
    k_clusters: int = 3,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Executes the 4 standard benchmark experiments under controlled conditions.
    """
    n_features_raw = X_scaled.shape[1]
    n_features_pca = X_pca.shape[1]

    experiments = [
        {"id": "Exp 1", "name": "Raw Features + K-Means", "space": "Standardized (Raw)", "X": X_scaled, "algo": "kmeans"},
        {"id": "Exp 2", "name": "PCA Features + K-Means", "space": "PCA-Reduced", "X": X_pca, "algo": "kmeans"},
        {"id": "Exp 3", "name": "Raw Features + DBSCAN", "space": "Standardized (Raw)", "X": X_scaled, "algo": "dbscan"},
        {"id": "Exp 4", "name": "PCA Features + DBSCAN", "space": "PCA-Reduced", "X": X_pca, "algo": "dbscan"}
    ]

    results = []

    for exp in experiments:
        X_curr = exp["X"]
        t0 = time.perf_counter()

        if exp["algo"] == "kmeans":
            km = KMeans(n_clusters=k_clusters, init="k-means++", n_init=10, random_state=random_state)
            labels = km.fit_predict(X_curr)
        else:
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            labels = db.fit_predict(X_curr)

        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = compute_clustering_metrics(X_curr, labels)
        retained_features = X_curr.shape[1]

        results.append({
            "Experiment_ID": exp["id"],
            "Experiment_Name": exp["name"],
            "Feature_Space": exp["space"],
            "Retained_Features": retained_features,
            "Dimensionality_Reduction": f"{round((1 - retained_features / n_features_raw) * 100, 1)}%" if retained_features < n_features_raw else "0.0% (Original)",
            "Clusters_Found": metrics["n_clusters"],
            "Noise_Samples": metrics["n_noise"],
            "Silhouette_Score": round(metrics["silhouette_score"], 4) if not np.isnan(metrics["silhouette_score"]) else "N/A",
            "Davies_Bouldin_Index": round(metrics["davies_bouldin_index"], 4) if not np.isnan(metrics["davies_bouldin_index"]) else "N/A",
            "Calinski_Harabasz_Index": round(metrics["calinski_harabasz_index"], 2) if not np.isnan(metrics["calinski_harabasz_index"]) else "N/A",
            "Computation_Time_ms": round(t_elapsed_ms, 2)
        })

    return pd.DataFrame(results)


def run_pca_component_selection_experiment(
    X_scaled: np.ndarray,
    feature_names: List[str],
    k_clusters: int = 3,
    candidate_components: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Evaluates impact of retained components on explained variance, clustering quality, and runtime.
    Tests: 2, 3, 5, 10, and 95% variance cutoff.
    """
    n_samples, n_features = X_scaled.shape
    max_k = min(n_samples, n_features)

    if candidate_components is None:
        raw_candidates = [2, 3, 5, 10]
        candidate_components = [c for c in raw_candidates if c < max_k]

    results = []

    # 1. Evaluate fixed component counts
    for c in candidate_components:
        t0 = time.perf_counter()
        pca_res = run_pca_analysis(X_scaled, feature_names, n_components=c)
        X_p = pca_res["X_pca"]
        
        km = KMeans(n_clusters=k_clusters, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_p)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        cum_var = pca_res["cumulative_explained_variance"][-1] * 100.0
        metrics = compute_clustering_metrics(X_p, labels)

        results.append({
            "Configuration": f"{c} Components",
            "Components_Retained": c,
            "Explained_Variance_Percent": round(cum_var, 2),
            "Feature_Reduction_Percent": round(pca_res["reduction_ratio_percent"], 1),
            "KMeans_Silhouette": round(metrics["silhouette_score"], 4) if not np.isnan(metrics["silhouette_score"]) else "N/A",
            "KMeans_Davies_Bouldin": round(metrics["davies_bouldin_index"], 4) if not np.isnan(metrics["davies_bouldin_index"]) else "N/A",
            "Execution_Time_ms": round(elapsed_ms, 2)
        })

    # 2. Evaluate 95% Variance threshold configuration
    t0 = time.perf_counter()
    pca_95 = run_pca_analysis(X_scaled, feature_names, variance_threshold=0.95)
    X_p95 = pca_95["X_pca"]
    k95 = pca_95["k_selected"]

    km95 = KMeans(n_clusters=k_clusters, init="k-means++", n_init=10, random_state=42)
    labels95 = km95.fit_predict(X_p95)
    elapsed_95_ms = (time.perf_counter() - t0) * 1000.0

    cum_var_95 = pca_95["cumulative_explained_variance"][-1] * 100.0
    metrics95 = compute_clustering_metrics(X_p95, labels95)

    results.append({
        "Configuration": "95% Variance Threshold",
        "Components_Retained": k95,
        "Explained_Variance_Percent": round(cum_var_95, 2),
        "Feature_Reduction_Percent": round(pca_95["reduction_ratio_percent"], 1),
        "KMeans_Silhouette": round(metrics95["silhouette_score"], 4) if not np.isnan(metrics95["silhouette_score"]) else "N/A",
        "KMeans_Davies_Bouldin": round(metrics95["davies_bouldin_index"], 4) if not np.isnan(metrics95["davies_bouldin_index"]) else "N/A",
        "Execution_Time_ms": round(elapsed_95_ms, 2)
    })

    return pd.DataFrame(results)
