"""
Dimensionality Reduction and PCA Module
Part of BEACHMIND (SIH26054)

Implements:
- StandardScaler feature scaling to prevent scale dominance and data leakage
- Principal Component Analysis (PCA) with configurable components or variance threshold (95%)
- Explained variance ratio and cumulative explained variance
- Eigenvalue analysis and Scree plot calculations
- Component loadings (correlation of original features with principal components)
- Latent feature projection and inverse transformation
- Empirical correlation analysis between Principal Components and reference grain size
- Joblib serialization for scaler and PCA models
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, Optional, List


def fit_and_scale_features(
    features_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Standardizes features to zero mean and unit variance.
    Saves fitted scaler to disk to guarantee consistency and prevent data leakage.
    """
    if feature_cols is None:
        exclude = ["image_id", "filename", "beach_name", "sample_id", "latitude", "longitude", "reference_d50", "dataset_source", "is_synthetic", "coastal_facies", "grain_type_label", "acquisition_conditions", "Cluster_Label"]
        feature_cols = [c for c in features_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(features_df[c])]

    if not feature_cols:
        raise ValueError("No numeric features found to scale.")

    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)

    return X_scaled, scaler, feature_cols


def run_pca_analysis(
    X_scaled: np.ndarray,
    feature_names: List[str],
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = 0.95,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes PCA on standardized feature matrix.
    Supports either explicit component count or variance cutoff (e.g., 0.95).
    Calculates eigenvalues, loadings, and cumulative variance.
    """
    n_samples, n_features = X_scaled.shape
    max_possible_components = min(n_samples, n_features)

    if n_components is not None:
        actual_k = min(int(n_components), max_possible_components)
        pca = PCA(n_components=actual_k)
    elif variance_threshold is not None:
        actual_k = min(variance_threshold, 0.999)
        pca = PCA(n_components=actual_k)
    else:
        pca = PCA(n_components=min(10, max_possible_components))

    X_pca = pca.fit_transform(X_scaled)
    k_selected = pca.n_components_

    # Calculate eigenvalues: lambda_i = variance_i * (n_samples - 1) or singular_values^2 / (n - 1)
    eigenvalues = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var_ratio)

    # Component Loadings: loading = eigenvector_i * sqrt(eigenvalue_i)
    # Reflects the Pearson correlation between original standardized features and principal components
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(k_selected)]
    )

    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(k_selected)]
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pca, save_path)

    reduction_ratio = (1.0 - (k_selected / n_features)) * 100.0

    return {
        "pca_model": pca,
        "X_pca": X_pca,
        "pca_df": pca_df,
        "k_selected": int(k_selected),
        "n_original_features": int(n_features),
        "reduction_ratio_percent": float(reduction_ratio),
        "explained_variance_ratio": explained_var_ratio.tolist(),
        "cumulative_explained_variance": cum_explained_var.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "loadings_df": loadings_df,
        "max_possible_components": int(max_possible_components)
    }


def analyze_pc_grain_size_correlation(
    pca_df: pd.DataFrame,
    reference_grain_sizes: pd.Series
) -> pd.DataFrame:
    """
    Investigates empirical correlation between Principal Components and ground-truth grain size.
    Prevents the erroneous assumption that high variance PCs automatically encode grain size.
    """
    valid_mask = ~reference_grain_sizes.isna()
    if valid_mask.sum() < 3:
        return pd.DataFrame()

    ref = reference_grain_sizes[valid_mask].astype(float)
    corr_records = []

    for col in pca_df.columns:
        pc_vals = pca_df.loc[valid_mask, col].astype(float)
        corr = float(np.corrcoef(pc_vals, ref)[0, 1])
        corr_records.append({
            "Principal_Component": col,
            "Pearson_Correlation_r": corr,
            "Absolute_Correlation": abs(corr),
            "Variance_Explained_Indicator": "Relevant to Grain Size" if abs(corr) >= 0.4 else "Subtle / Secondary Signal"
        })

    corr_df = pd.DataFrame(corr_records).sort_values("Absolute_Correlation", ascending=False)
    return corr_df
