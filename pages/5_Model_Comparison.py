"""
Page 5: Advanced Model Comparison & Component Selection
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
from modules.evaluation import (
    run_four_way_benchmark,
    run_pca_component_selection_experiment
)
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Advanced Experimental Benchmark",
    "Four-Way Experimental Comparison & PCA Component Sensitivity Analysis | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())
X_scaled = st.session_state.get("X_scaled")
pca_res = st.session_state.get("pca_result")
feat_cols = st.session_state.get("feature_cols", [])

if features_df.empty or X_scaled is None or not pca_res:
    st.warning("⚠️ No active dataset found. Please upload images in 'Image Processing' or activate Demo Mode from the sidebar.")
    st.stop()

X_pca = pca_res["X_pca"]

# Scientific Integrity Notice
st.markdown("""
<div class="alert-box">
<b>Research Methodology Note:</b> Internal clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) quantify mathematical geometric cohesion and inter-cluster separation in latent space. A superior silhouette score indicates tighter geometric clusters, but does <i>not</i> by itself prove geological or sedimentological correctness. Mechanical sieve calibration remains the definitive ground-truth benchmark.
</div>
""", unsafe_allow_html=True)

comp_tab1, comp_tab2 = st.tabs([
    "🔬 Four-Way Experimental Benchmark",
    "📐 PCA Component Selection Sensitivity"
])

# ----------------- TAB 1: FOUR-WAY BENCHMARK -----------------
with comp_tab1:
    st.markdown("### 🏆 Controlled Four-Way Benchmark Matrix")
    
    # Allow user to configure comparison parameters
    with st.expander("⚙️ Benchmark Hyperparameters", expanded=False):
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            bench_k = st.slider("K-Means Clusters (k):", 2, max(2, min(8, len(features_df)-1)), 3)
        with b_c2:
            bench_eps = st.slider("DBSCAN Epsilon (ε):", 0.5, 4.0, 1.5, step=0.1)
        with b_c3:
            bench_min = st.slider("DBSCAN Min Samples:", 2, 8, 3)

        if st.button("🔄 Re-run Benchmark Matrix", type="primary"):
            b_df = run_four_way_benchmark(
                X_scaled, X_pca,
                k_clusters=bench_k,
                dbscan_eps=bench_eps,
                dbscan_min_samples=bench_min
            )
            st.session_state["benchmark_df"] = b_df
            st.success("Benchmark refreshed!")

    benchmark_df = st.session_state.get("benchmark_df", pd.DataFrame())
    if benchmark_df.empty:
        benchmark_df = run_four_way_benchmark(X_scaled, X_pca, k_clusters=3)
        st.session_state["benchmark_df"] = benchmark_df

    st.dataframe(benchmark_df, use_container_width=True)

    # Comparative Metric Charts
    st.markdown("#### Comparative Metric Visualizations")
    c_chart1, c_chart2 = st.columns(2)

    # Convert numeric columns for plotting
    plot_b_df = benchmark_df.copy()
    plot_b_df["Silhouette_Num"] = pd.to_numeric(plot_b_df["Silhouette_Score"], errors="coerce")
    plot_b_df["DaviesBouldin_Num"] = pd.to_numeric(plot_b_df["Davies_Bouldin_Index"], errors="coerce")
    plot_b_df["Latency_ms"] = pd.to_numeric(plot_b_df["Computation_Time_ms"], errors="coerce")

    with c_chart1:
        fig_sil_b = px.bar(
            plot_b_df.dropna(subset=["Silhouette_Num"]),
            x="Experiment_ID", y="Silhouette_Num",
            color="Feature_Space",
            text_auto=".3f",
            title="Silhouette Score by Experiment (Higher is Better)",
            labels={"Silhouette_Num": "Silhouette Coefficient", "Experiment_ID": "Experiment"}
        )
        fig_sil_b.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_sil_b, use_container_width=True)

    with c_chart2:
        fig_time_b = px.bar(
            plot_b_df,
            x="Experiment_ID", y="Latency_ms",
            color="Feature_Space",
            text_auto=".1f",
            title="Computational Execution Time (ms) (Lower is Faster)",
            labels={"Latency_ms": "Time (ms)", "Experiment_ID": "Experiment"}
        )
        fig_time_b.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_time_b, use_container_width=True)

    # Download Benchmark Results
    csv_bench = convert_df_to_csv_bytes(benchmark_df)
    st.download_button(
        "📥 Download Four-Way Benchmark Table (CSV)",
        data=csv_bench,
        file_name="beachmind_model_benchmark.csv",
        mime="text/csv"
    )

# ----------------- TAB 2: COMPONENT SELECTION SENSITIVITY -----------------
with comp_tab2:
    st.markdown("### 📐 PCA Component Selection Sensitivity Analysis")
    st.markdown("""
    Evaluates system performance across candidate component counts ($2, 3, 5, 10$) and the dynamic **95% Variance Threshold**.
    Helps researchers identify the parsimonious optimal dimensionality cutoff that retains grain-size information while stripping noise.
    """)

    sens_df = run_pca_component_selection_experiment(X_scaled, feat_cols, k_clusters=3)
    st.dataframe(sens_df, use_container_width=True)

    # Sensitivity Plots: Variance Retained vs Clustering Quality
    sens_plot_df = sens_df.copy()
    sens_plot_df["Sil_Num"] = pd.to_numeric(sens_plot_df["KMeans_Silhouette"], errors="coerce")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        fig_sens_var = px.line(
            sens_plot_df, x="Configuration", y="Explained_Variance_Percent",
            markers=True,
            title="Preserved Variance Across PCA Configurations (%)",
            labels={"Explained_Variance_Percent": "Cumulative Explained Variance (%)"}
        )
        fig_sens_var.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_sens_var, use_container_width=True)

    with s_col2:
        fig_sens_sil = px.line(
            sens_plot_df.dropna(subset=["Sil_Num"]), x="Configuration", y="Sil_Num",
            markers=True,
            title="K-Means Clustering Quality (Silhouette) vs. PCA Dimensionality",
            labels={"Sil_Num": "Silhouette Coefficient"}
        )
        fig_sens_sil.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_sens_sil, use_container_width=True)
