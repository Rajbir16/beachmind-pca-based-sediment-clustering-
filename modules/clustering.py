"""
Clustering and Sediment Grouping Module
Part of BEACHMIND (SIH26054)

Implements:
- K-Means++ with configurable k, inertia (Elbow), silhouette scores, and centroids
- DBSCAN with configurable epsilon, min_samples, and noise identification (-1)
- Support for clustering on both Standardized Raw Features and PCA-Reduced Features
- Cluster characteristics summarization (morphology, texture, color)
- Neutral cluster naming policy to prevent unfounded geological assumptions
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Dict, Any, Tuple, Optional, List


def compute_kmeans_elbow(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Evaluates inertia (WCSS) and silhouette scores across a range of k values for Elbow curve analysis.
    """
    n_samples = X.shape[0]
    actual_k_max = min(k_max, n_samples - 1)
    
    k_values = []
    inertias = []
    silhouettes = []

    if actual_k_max < k_min:
        return {"k_values": [], "inertias": [], "silhouettes": []}

    for k in range(k_min, actual_k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        k_values.append(k)
        inertias.append(float(km.inertia_))
        
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = np.nan
        silhouettes.append(sil)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes
    }


def run_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes K-Means++ clustering on provided feature matrix (Standardized or PCA).
    """
    n_samples = X.shape[0]
    k = max(2, min(n_clusters, n_samples - 1)) if n_samples > 2 else 2

    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # Calculate per-sample silhouette scores
    try:
        sample_silhouettes = silhouette_samples(X, labels).tolist()
        overall_silhouette = float(silhouette_score(X, labels))
    except Exception:
        sample_silhouettes = [0.0] * n_samples
        overall_silhouette = np.nan

    return {
        "model": km,
        "labels": labels,
        "n_clusters": int(k),
        "centroids": centroids,
        "inertia": float(km.inertia_),
        "silhouette_score": overall_silhouette,
        "sample_silhouettes": sample_silhouettes
    }


def run_dbscan_clustering(
    X: np.ndarray,
    eps: float = 1.5,
    min_samples: int = 3
) -> Dict[str, Any]:
    """
    Executes DBSCAN density-based clustering.
    Identifies natural density clusters and labels isolated samples as noise (-1).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    # Silhouette score is only valid if there are at least 2 non-noise clusters or 2 clusters including noise
    valid_clusters = len(unique_labels)
    if valid_clusters >= 2 and n_clusters >= 1 and (len(labels) - n_noise) >= 2:
        try:
            # Silhouette on non-noise samples
            non_noise_mask = labels != -1
            if len(set(labels[non_noise_mask])) >= 2:
                overall_silhouette = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
            else:
                overall_silhouette = np.nan
        except Exception:
            overall_silhouette = np.nan
    else:
        overall_silhouette = np.nan

    return {
        "model": db,
        "labels": labels,
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "silhouette_score": overall_silhouette,
        "unique_labels": sorted(list(unique_labels))
    }


def summarize_cluster_characteristics(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    has_calibration: bool = False
) -> pd.DataFrame:
    """
    Produces descriptive scientific summaries for each cluster.
    Assigns neutral names (Cluster 0, Cluster 1...) unless physical calibration
    and sedimentological standards allow Wentworth classification.
    """
    df = features_df.copy()
    df["Cluster_Label"] = labels

    cluster_summaries = []
    unique_labels = sorted(df["Cluster_Label"].unique())

    for cl in unique_labels:
        subset = df[df["Cluster_Label"] == cl]
        count = len(subset)

        # Neutral label naming
        if cl == -1:
            name = "Noise / Outlier Samples"
            desc = "Atypical grains, boundary artifacts, or anomalous texture"
        else:
            name = f"Sediment Cluster {cl}"
            desc = "Sediment group with consistent texture and size characteristics"

        # Morphology metrics if present
        mean_diam = subset["morph_d50"].mean() if "morph_d50" in subset else np.nan
        mean_circ = subset["morph_mean_circularity"].mean() if "morph_mean_circularity" in subset else np.nan
        mean_aspect = subset["morph_mean_aspect_ratio"].mean() if "morph_mean_aspect_ratio" in subset else np.nan

        # Texture metrics
        glcm_cont = subset["glcm_contrast_mean"].mean() if "glcm_contrast_mean" in subset else np.nan
        glcm_homo = subset["glcm_homogeneity_mean"].mean() if "glcm_homogeneity_mean" in subset else np.nan

        # Color metrics
        r_mean = subset["color_r_mean"].mean() if "color_r_mean" in subset else np.nan
        g_mean = subset["color_g_mean"].mean() if "color_g_mean" in subset else np.nan
        b_mean = subset["color_b_mean"].mean() if "color_b_mean" in subset else np.nan

        # Reference grain size if available
        ref_d50 = subset["reference_d50"].mean() if "reference_d50" in subset and not subset["reference_d50"].isna().all() else np.nan

        cluster_summaries.append({
            "Cluster_ID": cl,
            "Sediment_Name": name,
            "Sample_Count": count,
            "Percent_Of_Dataset": round((count / len(df)) * 100.0, 1),
            "Mean_Grain_Diameter": round(mean_diam, 3) if not np.isnan(mean_diam) else "N/A",
            "Reference_D50_mm": round(ref_d50, 3) if not np.isnan(ref_d50) else "N/A",
            "Mean_Circularity": round(mean_circ, 3) if not np.isnan(mean_circ) else "N/A",
            "Mean_Aspect_Ratio": round(mean_aspect, 3) if not np.isnan(mean_aspect) else "N/A",
            "GLCM_Contrast": round(glcm_cont, 3) if not np.isnan(glcm_cont) else "N/A",
            "GLCM_Homogeneity": round(glcm_homo, 3) if not np.isnan(glcm_homo) else "N/A",
            "Color_RGB_Avg": f"({int(r_mean)}, {int(g_mean)}, {int(b_mean)})" if not np.isnan(r_mean) else "N/A",
            "Geological_Interpretation": desc
        })

    return pd.DataFrame(cluster_summaries)
