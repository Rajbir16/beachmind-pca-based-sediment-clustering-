"""
Page 4: Unsupervised Clustering Analysis
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
from modules.clustering import (
    compute_kmeans_elbow,
    run_kmeans_clustering,
    run_dbscan_clustering,
    summarize_cluster_characteristics
)
from modules.evaluation import run_four_way_benchmark
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Unsupervised Sediment Clustering",
    "K-Means++ & DBSCAN Clustering across Standardized and PCA Feature Spaces | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())
X_scaled = st.session_state.get("X_scaled")
pca_res = st.session_state.get("pca_result")
imgs_dict = st.session_state.get("images_dict", {})

if features_df.empty or X_scaled is None or not pca_res:
    st.warning("⚠️ No active dataset found. Please upload images in 'Image Processing' or activate Demo Mode from the sidebar.")
    st.stop()

X_pca = pca_res["X_pca"]

# Scientific Policy Banner
with st.expander("🛡️ Sedimentological Policy: Neutral Cluster Labeling & Integrity", expanded=False):
    st.markdown("""
    **Core Scientific Rule**: Cluster numbers ($0, 1, 2$) produced by unsupervised algorithms have **no inherent geological sediment meaning**.
    Assigning names such as *'Fine Sand'* or *'Gravel'* purely from mathematical cluster assignments without verified optical calibration 
    and ASTM/Wentworth sieve verification is scientifically invalid. Clusters are designated with neutral nomenclature (e.g., *Sediment Cluster 0*).
    """)

# Interactive Algorithm and Feature Space Selection
st.markdown("### ⚙️ Algorithm & Representation Setup")
cfg1, cfg2, cfg3 = st.columns(3)

with cfg1:
    algo_choice = st.radio("Clustering Algorithm:", ["K-Means++", "DBSCAN"])

with cfg2:
    space_choice = st.radio("Input Feature Representation:", ["PCA-Reduced Space (Recommended)", "Standardized Raw Feature Space"])

target_X = X_pca if "PCA" in space_choice else X_scaled

with cfg3:
    if algo_choice == "K-Means++":
        max_k_possible = max(2, min(8, len(features_df) - 1)) if len(features_df) > 2 else 2
        k_val = st.slider("Number of Clusters (k):", 2, max(2, max_k_possible), min(3, max_k_possible))
    else:
        eps_val = st.slider("DBSCAN Epsilon (ε):", 0.2, 5.0, 1.5, step=0.1)
        min_pts = st.slider("Min Samples (min_pts):", 2, 10, 3)

# Execution Button
if st.button("🚀 Run Clustering Pipeline", type="primary"):
    if algo_choice == "K-Means++":
        res = run_kmeans_clustering(target_X, n_clusters=k_val)
        labels = res["labels"]
        algo_label = f"K-Means (k={k_val}, {space_choice})"
    else:
        res = run_dbscan_clustering(target_X, eps=eps_val, min_samples=min_pts)
        labels = res["labels"]
        algo_label = f"DBSCAN (ε={eps_val}, min_pts={min_pts}, {space_choice})"

    # Update session clustering state
    summary = summarize_cluster_characteristics(features_df, labels)
    
    st.session_state["clustering_result"] = {
        algo_choice.lower().replace("+", "").replace("-", ""): res,
        "summary": summary,
        "active_labels": labels,
        "active_algo": algo_label
    }
    
    # Store cluster in features_df
    st.session_state["features_df"]["Cluster_Label"] = labels

    # Update benchmark
    bench_df = run_four_way_benchmark(X_scaled, X_pca, k_clusters=k_val if algo_choice == "K-Means++" else 3)
    st.session_state["benchmark_df"] = bench_df
    st.success(f"Clustering complete via {algo_label}!")

# Retrieve current active clustering result
cluster_state = st.session_state.get("clustering_result")
if not cluster_state:
    km_def = run_kmeans_clustering(X_pca, n_clusters=3)
    summary_def = summarize_cluster_characteristics(features_df, km_def["labels"])
    cluster_state = {
        "kmeans": km_def,
        "summary": summary_def,
        "active_labels": km_def["labels"],
        "active_algo": "K-Means++ (k=3, PCA Space)"
    }
    st.session_state["clustering_result"] = cluster_state
    st.session_state["features_df"]["Cluster_Label"] = km_def["labels"]

active_labels = cluster_state["active_labels"]
summary_df = cluster_state["summary"]

# Clustering Performance Metric Cards
sil_val = "N/A"
if "kmeans" in cluster_state and not np.isnan(cluster_state["kmeans"].get("silhouette_score", np.nan)):
    sil_val = f"{cluster_state['kmeans']['silhouette_score']:.4f}"
elif "dbscan" in cluster_state and not np.isnan(cluster_state["dbscan"].get("silhouette_score", np.nan)):
    sil_val = f"{cluster_state['dbscan']['silhouette_score']:.4f}"

unique_assigned = set(active_labels)
n_cl_found = len(unique_assigned - {-1})
n_noise = int(np.sum(active_labels == -1))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Active Algorithm", cluster_state.get("active_algo", "K-Means")[:18])
with m2:
    st.metric("Identified Clusters", f"{n_cl_found}")
with m3:
    st.metric("Noise / Outliers", f"{n_noise}")
with m4:
    st.metric("Silhouette Coefficient", sil_val)

st.markdown("---")

# Main Clustering Views: 1. Cluster Visualization, 2. Elbow & Diagnostics, 3. Characteristics Table, 4. Representative Images
cl_tab1, cl_tab2, cl_tab3, cl_tab4 = st.tabs([
    "📊 Latent Cluster Scatter",
    "📉 Elbow & Diagnostics",
    "📋 Descriptive Cluster Summaries",
    "🖼️ Representative Sediment Gallery"
])

# ----------------- TAB 1: SCATTER -----------------
with cl_tab1:
    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros(len(X_pca)),
        "Cluster": [f"Cluster {c}" if c != -1 else "Noise / Outlier" for c in active_labels],
        "image_id": features_df["image_id"],
        "filename": features_df["filename"],
        "beach": features_df.get("beach_name", "Beach Sample")
    })

    fig_scat = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="Cluster",
        hover_data=["image_id", "filename", "beach"],
        title=f"Sample Cluster Partitioning in 2D Latent Space ({cluster_state.get('active_algo', '')})",
        template="plotly_white"
    )
    fig_scat.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1.2, color='DarkSlateGrey')))
    fig_scat.update_layout(height=480)
    st.plotly_chart(fig_scat, use_container_width=True)

# ----------------- TAB 2: ELBOW & DIAGNOSTICS -----------------
with cl_tab2:
    st.markdown("#### K-Means Elbow Curve & Silhouette Optimization")
    elbow_data = compute_kmeans_elbow(target_X, k_min=2, k_max=min(8, len(features_df)-1) if len(features_df) > 3 else 2)
    
    if elbow_data["k_values"]:
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            fig_elb = px.line(
                x=elbow_data["k_values"], y=elbow_data["inertias"],
                markers=True,
                title="Elbow Method: Within-Cluster Sum of Squares (Inertia)",
                labels={"x": "Number of Clusters (k)", "y": "Inertia (WCSS)"}
            )
            fig_elb.update_layout(height=380, template="plotly_white")
            st.plotly_chart(fig_elb, use_container_width=True)

        with e_col2:
            fig_sil = px.line(
                x=elbow_data["k_values"], y=elbow_data["silhouettes"],
                markers=True,
                title="Silhouette Score Trajectory across k",
                labels={"x": "Number of Clusters (k)", "y": "Mean Silhouette Coefficient"}
            )
            fig_sil.update_layout(height=380, template="plotly_white")
            st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.info("Insufficient samples to calculate multi-k Elbow trajectory.")

# ----------------- TAB 3: SUMMARIES -----------------
with cl_tab3:
    st.markdown("#### Descriptive Morphological & Texture Cluster Characteristics")
    st.caption("Summarizes statistical central tendencies across grain dimensions, GLCM texture, and optical color.")
    st.dataframe(summary_df, use_container_width=True)

# ----------------- TAB 4: GALLERY -----------------
with cl_tab4:
    st.markdown("#### 🖼️ Representative Sample Images by Sediment Cluster")
    
    for cl_id in sorted(list(unique_assigned)):
        cl_name = f"Sediment Cluster {cl_id}" if cl_id != -1 else "Noise / Outlier Samples"
        st.markdown(f"##### {cl_name}")
        
        sample_indices = np.where(active_labels == cl_id)[0]
        # Show up to 4 representative images
        show_indices = sample_indices[:4]
        
        cols = st.columns(len(show_indices) if len(show_indices) > 0 else 1)
        for i, idx in enumerate(show_indices):
            row = features_df.iloc[idx]
            img_id = row["image_id"]
            with cols[i]:
                if img_id in imgs_dict:
                    st.image(imgs_dict[img_id], caption=f"{row['filename']} ({img_id})", use_container_width=True)
                else:
                    st.info(f"Image {img_id}")
                
                # Small card stats
                d50_val = row.get("morph_d50", np.nan)
                d50_str = f"{d50_val:.3f}" if not pd.isna(d50_val) else "N/A"
                st.caption(f"Beach: {row.get('beach_name', 'N/A')}\nOptical D50: {d50_str}")
        st.markdown("---")

# Download Cluster Assignments
csv_cl = convert_df_to_csv_bytes(features_df[["image_id", "filename", "Cluster_Label"]])
st.download_button(
    "📥 Download Cluster Assignments CSV",
    data=csv_cl,
    file_name="sediment_cluster_assignments.csv",
    mime="text/csv"
)
