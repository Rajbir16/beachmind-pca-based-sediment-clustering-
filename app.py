"""
BEACHMIND: Intelligent Beach Sediment Mapping
PCA-Based Feature Reduction of Camera-Derived Grain-Size Data
Smart India Hackathon SIH26054

Main Application Entry Point
"""

import streamlit as st
import pandas as pd
import numpy as np

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="BEACHMIND – Intelligent Beach Sediment Mapping",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar,
    load_demo_dataset_to_session
)

# Initialize Session State
init_session_state()
apply_scientific_theme()
render_sidebar()

# Page Title & Header
render_header(
    "BEACHMIND – Intelligent Beach Sediment Mapping",
    "PCA-Based Feature Reduction of Camera-Derived Grain-Size Data | SIH26054"
)

# Auto-offer Demo Mode if empty
reg_df = st.session_state.get("dataset_registry", pd.DataFrame())
features_df = st.session_state.get("features_df", pd.DataFrame())
pca_res = st.session_state.get("pca_result")
cluster_res = st.session_state.get("clustering_result")

if reg_df.empty:
    st.info(
        "👋 **Welcome to BEACHMIND!** No active dataset found in memory. "
        "You can immediately explore the full system using the built-in **Demonstration Mode** "
        "(12 procedurally generated synthetic beach sediment images with full pipeline results) "
        "or navigate to **2_Image_Processing** in the sidebar to upload real field imagery."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Launch Synthetic Demonstration Mode", type="primary", use_container_width=True):
            load_demo_dataset_to_session()
            st.rerun()

st.markdown("---")

# Scientific Metric Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Uploaded Samples</div>
            <div class="metric-value">{}</div>
        </div>
    """.format(len(reg_df) if not reg_df.empty else 0), unsafe_allow_html=True)

with m_col2:
    feat_count = len([c for c in features_df.columns if c not in ["image_id", "filename", "beach_name", "sample_id", "latitude", "longitude", "reference_d50", "dataset_source", "Cluster_Label"]]) if not features_df.empty else 0
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Extracted Features</div>
            <div class="metric-value">{}</div>
        </div>
    """.format(feat_count), unsafe_allow_html=True)

with m_col3:
    k_comp = pca_res["k_selected"] if pca_res else "N/A"
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">PCA Components</div>
            <div class="metric-value">{}</div>
        </div>
    """.format(k_comp), unsafe_allow_html=True)

with m_col4:
    n_cl = cluster_res["kmeans"]["n_clusters"] if cluster_res and "kmeans" in cluster_res else "N/A"
    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sediment Clusters</div>
            <div class="metric-value">{}</div>
        </div>
    """.format(n_cl), unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Project Introduction & Scientific Methodology
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🔬 Problem Statement & Research Objective")
    st.markdown("""
    Coastal sediment mapping provides vital baseline intelligence for coastal engineering, sediment transport modeling,
    and beach erosion monitoring. Traditional laboratory mechanical sieve analysis is time-consuming, expensive, and spatially sparse.
    
    **BEACHMIND** implements a pure **Computer Vision & Classical Machine Learning** framework (zero deep neural networks) to:
    1. **Preprocess Optical Beach Imagery**: Apply CLAHE, noise filtering, and contour morphological segmentation.
    2. **Extract Multi-Modal Handcrafted Features**:
       - *Texture*: Gray Level Co-occurrence Matrix (GLCM: Contrast, Homogeneity, Energy, ASM, Correlation).
       - *Local Micro-patterns*: Rotation-invariant uniform Local Binary Patterns (LBP).
       - *Spatial Frequency*: Multi-orientation Gabor filter banks.
       - *Sediment Color*: Multi-channel RGB/HSV moments and color histograms.
       - *Grain Morphology*: Equivalent diameter ($D_{eq}$), circularity, aspect ratio, and grain size distributions ($D_{10}, D_{50}, D_{90}$).
    3. **Dimensionality Reduction via PCA**: Eliminate multicollinearity and reduce feature dimensionality from 50+ variables to dominant orthogonal principal components while preserving $\\ge 95\\%$ explained variance.
    4. **Unsupervised Clustering**: Partition sediments into natural facies using **K-Means++** and **DBSCAN** across both raw and PCA-reduced feature spaces.
    5. **Empirical Ground-Truth Validation**: Cross-validate optical grain diameters against mechanical laboratory sieve analysis.
    6. **Geospatial Mapping**: Project sediment cluster assignments onto interactive coastal maps using Folium.
    """)

with col_right:
    st.markdown("### 📋 End-to-End System Workflow")
    st.markdown("""
    ```
    [1. Raw Camera Imagery / SandSnap]
                  │
                  ▼
    [2. Computer Vision Preprocessing]
       (CLAHE, Denoising, Normalization)
                  │
                  ▼
    [3. Comprehensive Feature Extraction]
       (GLCM, LBP, Gabor, Color, Morphology)
                  │
                  ▼
    [4. StandardScaler Normalization]
                  │
                  ▼
    [5. Principal Component Analysis (PCA)]
       (Variance Scree Analysis & Loadings)
                  │
                  ▼
    [6. Dual Clustering: K-Means & DBSCAN]
       (Raw Space vs PCA Latent Space)
                  │
                  ▼
    [7. Model Comparison & Sieve Validation]
       (Silhouette, Davies-Bouldin, MAE, R²)
                  │
                  ▼
    [8. Interactive Coastal Folium Mapping]
    ```
    """)

st.markdown("---")

# Quick Navigation & Architectural Guide
st.markdown("### 🧭 Modular Application Pages")
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    st.info("**2. Image Processing**\n\nUpload images (single, batch, ZIP), inspect preprocessing stages, adjust CLAHE & segmentation.")

with nav_col2:
    st.info("**3. PCA Analysis**\n\nConfigure variance threshold, analyze scree plots, 2D/3D projections, and component loadings.")

with nav_col3:
    st.info("**4. Clustering Analysis**\n\nRun K-Means & DBSCAN, inspect cluster centroids, silhouette profiles, and representative samples.")

with nav_col4:
    st.info("**5. Model Comparison**\n\nCompare 4 standard experimental setups, runtime latency, and component selection sensitivity.")

nav_col5, nav_col6, nav_col7, nav_col8 = st.columns(4)

with nav_col5:
    st.info("**6. Sediment Validation**\n\nUpload laboratory mechanical sieve ground truth, compute MAE, RMSE, MAPE, and residual plots.")

with nav_col6:
    st.info("**7. Beach Sediment Map**\n\nInteractive Folium coastal map with cluster color-coded markers and popup sample cards.")

with nav_col7:
    st.info("**8. Reports & Downloads**\n\nExport full research report in PDF and download features, PCA, and cluster CSV datasets.")

with nav_col8:
    st.info("**Dataset Registry**\n\nReview metadata table, coordinates, reference measurements, and acquisition condition logs.")
