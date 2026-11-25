"""
Page 3: Principal Component Analysis (PCA)
Part of BEACHMIND (SIH26054)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar
)
from modules.dimensionality_reduction import (
    fit_and_scale_features,
    run_pca_analysis,
    analyze_pc_grain_size_correlation
)
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Principal Component Analysis (PCA)",
    "Feature Scaling, Eigenvalue Scree Analysis, Dimensionality Reduction & Latent Projections | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())
X_scaled = st.session_state.get("X_scaled")
feat_cols = st.session_state.get("feature_cols", [])

if features_df.empty or X_scaled is None or len(feat_cols) == 0:
    st.warning("⚠️ No feature matrix found. Please upload images in 'Image Processing' or load the Demo dataset from the sidebar.")
    st.stop()

# Section 1: Scientific Motivation for StandardScaler
with st.expander("ℹ️ Scientific Rationale: Why StandardScaler is Mathematically Essential Before PCA", expanded=False):
    st.markdown("""
    PCA seeks orthogonal projection axes that maximize sample variance. However, our extracted features span fundamentally incompatible units and dynamic ranges:
    - **Contour Area**: Ranges from $10^1$ to $10^5$ square pixels.
    - **GLCM Homogeneity & ASM**: Constrained strictly within $[0, 1]$.
    - **RGB/HSV Color Moments**: Span $[0, 255]$ and $[0, 360]$.
    - **LBP Histogram Frequencies**: Small fractional probabilities $[0, 0.25]$.

    Without zero-mean and unit-variance standardization ($z = (x - \\mu) / \\sigma$), variables with numerically larger magnitudes 
    would dominate the covariance matrix and hijack the principal components, regardless of their actual geological significance.
    StandardScaler projects all features into an equitable, dimensionless statistical metric space.
    """)

# Section 2: PCA Hyperparameter Configuration
st.markdown("### ⚙️ PCA Configuration & Parameter Selection")
c_cfg1, c_cfg2, c_cfg3 = st.columns(3)

max_k = min(len(features_df), len(feat_cols))

with c_cfg1:
    pca_mode = st.radio("Dimensionality Reduction Mode:", ["Target Variance Cutoff", "Fixed Component Count"])

with c_cfg2:
    if pca_mode == "Target Variance Cutoff":
        var_target = st.slider("Target Cumulative Variance Threshold:", 0.80, 0.99, 0.95, 0.01)
        k_input = None
    else:
        var_target = None
        k_input = st.slider("Select Principal Components (k):", 2, max(2, max_k), min(3, max_k))

with c_cfg3:
    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🔄 Recompute PCA Projections", type="primary", use_container_width=True):
        pca_res = run_pca_analysis(
            X_scaled, feat_cols,
            n_components=k_input,
            variance_threshold=var_target
        )
        st.session_state["pca_result"] = pca_res
        st.success(f"PCA updated: Selected {pca_res['k_selected']} principal components!")

pca_res = st.session_state.get("pca_result")
if not pca_res:
    pca_res = run_pca_analysis(X_scaled, feat_cols, n_components=min(3, max_k))
    st.session_state["pca_result"] = pca_res

# Display Reduction Metrics
k_sel = pca_res["k_selected"]
orig_dim = pca_res["n_original_features"]
red_pct = pca_res["reduction_ratio_percent"]
cum_var = pca_res["cumulative_explained_variance"]
cum_var_pct = cum_var[-1] * 100.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Original Features", f"{orig_dim}")
with m2:
    st.metric("Retained Components", f"{k_sel}")
with m3:
    st.metric("Dimensionality Reduced", f"{red_pct:.1f}%")
with m4:
    st.metric("Preserved Variance", f"{cum_var_pct:.2f}%")

st.markdown("---")

# Visualizations Tab
viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
    "📈 Scree & Cumulative Variance",
    "🌐 2D & 3D Latent Projections",
    "🔍 Component Loadings Analysis",
    "🎯 Grain-Size Correlation Check"
])

# ----------------- TAB 1: SCREE & VARIANCE -----------------
with viz_tab1:
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        # Scree Plot (Eigenvalues)
        eigenvals = pca_res["eigenvalues"]
        scree_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(eigenvals))],
            "Eigenvalue": eigenvals
        })
        
        fig_scree = px.bar(
            scree_df.head(15), x="Component", y="Eigenvalue",
            title="Eigenvalue Scree Plot (Kaiser Criterion: λ ≥ 1.0)",
            labels={"Eigenvalue": "Eigenvalue (Variance Magnitude)"},
            color="Eigenvalue",
            color_continuous_scale="Viridis"
        )
        fig_scree.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Kaiser Criterion (λ = 1.0)")
        fig_scree.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_scree, use_container_width=True)

    with v_col2:
        # Cumulative Explained Variance
        cum_df = pd.DataFrame({
            "Component_Index": list(range(1, len(cum_var) + 1)),
            "Cumulative_Variance_Pct": [v * 100.0 for v in cum_var]
        })
        
        fig_cum = px.line(
            cum_df, x="Component_Index", y="Cumulative_Variance_Pct",
            markers=True,
            title="Cumulative Explained Variance Curve",
            labels={"Component_Index": "Number of Components", "Cumulative_Variance_Pct": "Cumulative Variance (%)"}
        )
        fig_cum.add_hline(y=95.0, line_dash="dash", line_color="green", annotation_text="95% Target Threshold")
        fig_cum.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_cum, use_container_width=True)

# ----------------- TAB 2: 2D & 3D PROJECTIONS -----------------
with viz_tab2:
    pca_df = pca_res["pca_df"].copy()
    pca_df["image_id"] = features_df["image_id"].values
    pca_df["filename"] = features_df["filename"].values
    pca_df["beach_name"] = features_df.get("beach_name", "Beach Sample").values

    cl_labels = st.session_state.get("clustering_result", {}).get("active_labels")
    if cl_labels is not None and len(cl_labels) == len(pca_df):
        pca_df["Cluster"] = [f"Cluster {c}" if c != -1 else "Noise" for c in cl_labels]
    else:
        pca_df["Cluster"] = "Sample"

    p2d_col, p3d_col = st.columns(2)

    with p2d_col:
        # 2D Projection
        fig_2d = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="Cluster",
            hover_data=["image_id", "filename", "beach_name"],
            title=f"2D PCA Projection: PC1 ({pca_res['explained_variance_ratio'][0]*100:.1f}%) vs PC2 ({pca_res['explained_variance_ratio'][1]*100:.1f}%)" if len(pca_res['explained_variance_ratio']) > 1 else "2D PCA Projection",
            template="plotly_white"
        )
        fig_2d.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
        fig_2d.update_layout(height=480)
        st.plotly_chart(fig_2d, use_container_width=True)

    with p3d_col:
        # 3D Projection if at least 3 components
        if k_sel >= 3:
            fig_3d = px.scatter_3d(
                pca_df, x="PC1", y="PC2", z="PC3",
                color="Cluster",
                hover_data=["image_id", "filename", "beach_name"],
                title="3D Orthogonal PCA Space (PC1, PC2, PC3)",
                template="plotly_white"
            )
            fig_3d.update_traces(marker=dict(size=6, opacity=0.85))
            fig_3d.update_layout(height=480, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D visualization requires at least 3 principal components (currently k = 2).")

# ----------------- TAB 3: LOADINGS -----------------
with viz_tab3:
    st.markdown("### 🔍 Principal Component Loadings (Feature-Component Correlations)")
    st.caption("Loadings quantify the extent to which each original Computer Vision feature contributes to the principal components.")

    loadings_df = pca_res["loadings_df"]
    
    selected_pc = st.selectbox("Select Principal Component to Inspect:", loadings_df.columns)
    
    sorted_loadings = loadings_df[selected_pc].sort_values()
    top_pos = sorted_loadings.tail(8)
    top_neg = sorted_loadings.head(8)
    top_combined = pd.concat([top_neg, top_pos])

    fig_load = px.bar(
        x=top_combined.values, y=top_combined.index,
        orientation='h',
        title=f"Top Influential Feature Loadings on {selected_pc}",
        labels={"x": "Factor Loading (-1 to +1)", "y": "Computer Vision Feature"},
        color=top_combined.values,
        color_continuous_scale="RdBu_r"
    )
    fig_load.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_load, use_container_width=True)

# ----------------- TAB 4: GRAIN SIZE CORRELATION -----------------
with viz_tab4:
    st.markdown("### 🎯 Empirical Principal Component vs. Reference Grain Size Correlation")
    st.markdown("""
    > ⚠️ **Scientific Integrity Note**: Retaining 95% overall geometric variance does **not** inherently guarantee 
    > that the principal components capture sediment grain size. PCA is completely unsupervised; it maximizes overall variance, 
    > which could be driven by lighting contrast, mineral color, or image noise. We must explicitly verify the correlation 
    > between PCs and ground-truth grain size when reference measurements exist.
    """)

    if "reference_d50" in features_df.columns and not features_df["reference_d50"].isna().all():
        corr_df = analyze_pc_grain_size_correlation(pca_res["pca_df"], features_df["reference_d50"])
        if not corr_df.empty:
            st.dataframe(corr_df, use_container_width=True)
            
            # Scatter plot of top correlated PC vs Reference D50
            best_pc = corr_df.iloc[0]["Principal_Component"]
            best_r = corr_df.iloc[0]["Pearson_Correlation_r"]
            
            scat_df = pd.DataFrame({
                best_pc: pca_res["pca_df"][best_pc],
                "Reference_D50_mm": features_df["reference_d50"]
            }).dropna()

            fig_corr = px.scatter(
                scat_df, x=best_pc, y="Reference_D50_mm",
                trendline="ols",
                title=f"Empirical Grain Size Relationship: {best_pc} vs. Laboratory Sieve D50 (Pearson r = {best_r:.3f})",
                template="plotly_white"
            )
            fig_corr.update_layout(height=420)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient non-null reference D50 measurements to compute correlation.")
    else:
        st.info("Reference sieve measurements are not present in current registry. Upload a reference CSV to perform correlation verification.")

st.markdown("---")
# Download PCA Transformed Matrix
csv_pca = convert_df_to_csv_bytes(pca_res["pca_df"])
st.download_button(
    "📥 Download PCA Transformed Features CSV",
    data=csv_pca,
    file_name="pca_transformed_features.csv",
    mime="text/csv"
)
