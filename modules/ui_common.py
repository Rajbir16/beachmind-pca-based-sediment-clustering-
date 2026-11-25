"""
UI/UX Common Helpers and Theming
Part of BEACHMIND (SIH26054)

Provides:
- Consistent scientific styling (Tailwind-like research palette: navy, slate, teal)
- Shared session state initialization across all multi-page tabs
- Reusable UI widgets: metric cards, callout alerts, workflow progress bars
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.dataset_manager import generate_synthetic_demo_dataset
from modules.feature_extraction import extract_features_batch
from modules.dimensionality_reduction import fit_and_scale_features, run_pca_analysis
from modules.clustering import run_kmeans_clustering, run_dbscan_clustering, summarize_cluster_characteristics
from modules.evaluation import run_four_way_benchmark


def init_session_state():
    """Ensures all persistent session state variables exist with safe defaults."""
    if "is_initialized" not in st.session_state:
        st.session_state["is_initialized"] = True
        st.session_state["dataset_registry"] = pd.DataFrame()
        st.session_state["images_dict"] = {}
        st.session_state["features_df"] = pd.DataFrame()
        st.session_state["artifacts_map"] = {}
        st.session_state["scaler"] = None
        st.session_state["X_scaled"] = None
        st.session_state["feature_cols"] = []
        st.session_state["pca_result"] = None
        st.session_state["pca_k"] = 3
        st.session_state["kmeans_k"] = 3
        st.session_state["dbscan_eps"] = 1.5
        st.session_state["dbscan_min_samples"] = 3
        st.session_state["clustering_result"] = None
        st.session_state["benchmark_df"] = pd.DataFrame()
        st.session_state["validation_result"] = None
        st.session_state["is_demo_mode"] = False
        st.session_state["pixel_to_mm_ratio"] = None


def load_demo_dataset_to_session():
    """Loads procedural demo dataset into session state and executes feature extraction."""
    with st.spinner("Initializing demonstration mode with procedural sediment textures..."):
        reg_df, img_dict = generate_synthetic_demo_dataset()
        st.session_state["dataset_registry"] = reg_df
        st.session_state["images_dict"] = img_dict
        st.session_state["is_demo_mode"] = True

        # Process batch extraction
        batch_items = [
            (row["image_id"], row["filename"], img_dict[row["image_id"]])
            for _, row in reg_df.iterrows()
        ]
        feat_df, art_map, _ = extract_features_batch(batch_items)
        
        # Merge with registry columns
        merged_df = reg_df.merge(feat_df, on=["image_id", "filename"], how="left")
        st.session_state["features_df"] = merged_df
        st.session_state["artifacts_map"] = art_map

        # Standardize and compute PCA
        X_scaled, scaler, feat_cols = fit_and_scale_features(merged_df)
        st.session_state["X_scaled"] = X_scaled
        st.session_state["scaler"] = scaler
        st.session_state["feature_cols"] = feat_cols

        pca_res = run_pca_analysis(X_scaled, feat_cols, n_components=3)
        st.session_state["pca_result"] = pca_res

        # Initial clustering
        km_res = run_kmeans_clustering(pca_res["X_pca"], n_clusters=3)
        db_res = run_dbscan_clustering(pca_res["X_pca"], eps=1.5, min_samples=3)
        
        merged_df["Cluster_Label"] = km_res["labels"]
        cluster_summary = summarize_cluster_characteristics(merged_df, km_res["labels"])

        st.session_state["clustering_result"] = {
            "kmeans": km_res,
            "dbscan": db_res,
            "summary": cluster_summary,
            "active_labels": km_res["labels"],
            "active_algo": "K-Means (PCA Space)"
        }

        # Benchmark
        bench_df = run_four_way_benchmark(X_scaled, pca_res["X_pca"], k_clusters=3)
        st.session_state["benchmark_df"] = bench_df


def apply_scientific_theme():
    """Injects professional scientific CSS styling."""
    st.markdown("""
        <style>
        /* Modern scientific clean UI */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.05rem;
            font-weight: 400;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0.2rem 0;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 600;
            color: #64748b;
            letter-spacing: 0.05em;
        }
        .alert-box {
            background-color: #eff6ff;
            border-left: 4px solid #2563eb;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            color: #1e40af;
            font-size: 0.95rem;
        }
        .warning-box {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 0.8rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            color: #92400e;
            font-size: 0.95rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            white-space: pre-wrap;
            border-radius: 6px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Renders standardized academic page header."""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)
    
    if st.session_state.get("is_demo_mode", False):
        st.markdown(
            '<div class="warning-box"><b>DEMONSTRATION MODE ACTIVE:</b> Currently running on procedurally generated synthetic beach sediment images. '
            'Features, coordinates, and metrics are simulated for demonstration. Upload real field imagery in <b>Image Processing</b> to run authentic research mapping.</div>',
            unsafe_allow_html=True
        )


def render_sidebar():
    """Renders consistent scientific sidebar with project credits and dataset status."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/sand-timer.png", width=64)
        st.markdown("### BEACHMIND")
        st.caption("**SIH26054** | Smart India Hackathon")
        st.markdown("---")

        reg_df = st.session_state.get("dataset_registry", pd.DataFrame())
        n_imgs = len(reg_df)
        has_feat = not st.session_state.get("features_df", pd.DataFrame()).empty

        st.markdown("**Session Status**")
        st.markdown(f"- **Images Loaded:** `{n_imgs}`")
        st.markdown(f"- **Features Extracted:** `{'Yes' if has_feat else 'Pending'}`")
        if st.session_state.get("is_demo_mode", False):
            st.markdown("- **Data Source:** `Synthetic Demo`")
        elif n_imgs > 0:
            st.markdown("- **Data Source:** `User Uploaded`")
        else:
            st.markdown("- **Data Source:** `None (Empty)`")

        st.markdown("---")
        if st.button("Load Synthetic Demo Dataset", use_container_width=True, type="secondary"):
            load_demo_dataset_to_session()
            st.rerun()

        if st.button("Reset / Clear Workspace", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session_state()
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<small style='color: #94a3b8;'>Built with OpenCV, scikit-learn, scikit-image, and Streamlit. "
            "Strictly Classical ML & CV (Zero Neural Networks).</small>",
            unsafe_allow_html=True
        )
