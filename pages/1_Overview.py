"""
Page 1: Overview Dashboard
Part of BEACHMIND (SIH26054)
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar,
    load_demo_dataset_to_session
)
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Overview Dashboard",
    "Dataset Statistics, Algorithmic Architecture, and Pipeline Workflow | SIH26054"
)

reg_df = st.session_state.get("dataset_registry", pd.DataFrame())
features_df = st.session_state.get("features_df", pd.DataFrame())
pca_res = st.session_state.get("pca_result")
cluster_res = st.session_state.get("clustering_result")

# Metric Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sample Images</div>
            <div class="metric-value">{len(reg_df)}</div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    feat_count = len([c for c in features_df.columns if c not in ["image_id", "filename", "beach_name", "sample_id", "latitude", "longitude", "reference_d50", "dataset_source", "Cluster_Label"]]) if not features_df.empty else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Extracted Features</div>
            <div class="metric-value">{feat_count}</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    k_comp = pca_res["k_selected"] if pca_res else "N/A"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PCA Components</div>
            <div class="metric-value">{k_comp}</div>
        </div>
    """, unsafe_allow_html=True)

with m4:
    n_cl = cluster_res["kmeans"]["n_clusters"] if cluster_res and "kmeans" in cluster_res else "N/A"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sediment Clusters</div>
            <div class="metric-value">{n_cl}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Dataset Registry Display
st.markdown("### 🗃️ Active Dataset Registry")

if reg_df.empty:
    st.warning("No dataset is currently registered in session memory.")
    if st.button("Load Synthetic Demonstration Dataset", type="primary"):
        load_demo_dataset_to_session()
        st.rerun()
else:
    display_cols = [c for c in ["image_id", "filename", "dataset_source", "beach_name", "sample_id", "latitude", "longitude", "reference_d50", "acquisition_conditions"] if c in reg_df.columns]
    st.dataframe(reg_df[display_cols], use_container_width=True)

    csv_data = convert_df_to_csv_bytes(reg_df)
    st.download_button(
        label="📥 Download Dataset Registry as CSV",
        data=csv_data,
        file_name="beachmind_dataset_registry.csv",
        mime="text/csv"
    )

st.markdown("---")

# Architectural Framework & Compliance Check
st.markdown("### 🛡️ Smart India Hackathon Compliance & Research Ethics")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    **Zero Deep Learning Constraint**:
    - **OpenCV & Scipy** for all spatial transformations, morphological contour analysis, and filtering.
    - **Scikit-Image** for GLCM texture matrices and rotation-invariant Local Binary Patterns (LBP).
    - **Scikit-Learn** for standard scaling, Principal Component Analysis, K-Means++, and DBSCAN.
    - No black-box convolutional networks, transfer learning, or pretrained models are used.
    """)

with c2:
    st.markdown("""
    **Geological & Statistical Scientific Integrity**:
    - Internal clustering metrics (Silhouette, Davies-Bouldin) do not prove geological sediment facies.
    - Cluster numbers are neutral indices ($0, 1, 2$) unless rigorous laboratory calibration is verified.
    - When geographic coordinates or sieve measurements are absent, functions gracefully disable instead of inventing data.
    """)
