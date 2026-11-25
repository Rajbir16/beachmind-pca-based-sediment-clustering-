"""
Page 8: Reports and Scientific Exports
Part of BEACHMIND (SIH26054)
"""

import streamlit as st
import pandas as pd
import numpy as np

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar
)
from modules.report_generator import (
    generate_research_pdf_report,
    convert_df_to_csv_bytes
)

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Research Reports & Data Exports",
    "Comprehensive PDF Technical Report and Structured Research Data Tables | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())
pca_res = st.session_state.get("pca_result")
bench_df = st.session_state.get("benchmark_df", pd.DataFrame())
val_res = st.session_state.get("validation_result")
reg_df = st.session_state.get("dataset_registry", pd.DataFrame())

if features_df.empty:
    st.warning("⚠️ No active dataset found. Please upload images in 'Image Processing' or activate Demo Mode from the sidebar.")
    st.stop()

# Section 1: Generate PDF Technical Report
st.markdown("### 📄 Complete Research & Technical PDF Report")
st.markdown("""
Download a publication-ready technical document summarizing the end-to-end Classical Computer Vision and Machine Learning experiment:
- Image Acquisition & Preprocessing Metrics
- GLCM, LBP, Gabor & Morphological Feature Formulations
- StandardScaler & PCA Dimensionality Reduction Results
- Four-Way Experimental Benchmark Comparison
- Sieve Analysis Validation Metrics (MAE, RMSE, MAPE, R²)
- SIH26054 Scientific Compliance Notes
""")

# Prepare summary dictionary for PDF
summary_meta = {
    "n_images": len(features_df),
    "n_features": len([c for c in features_df.columns if c not in ["image_id", "filename", "beach_name", "sample_id", "latitude", "longitude", "reference_d50", "dataset_source", "Cluster_Label"]]),
    "dataset_source": "Procedural Synthetic Demo" if st.session_state.get("is_demo_mode", False) else "User Uploaded Field Dataset",
    "has_calibration": st.session_state.get("pixel_to_mm_ratio") is not None
}

with st.spinner("Generating technical PDF document..."):
    try:
        pdf_bytes = generate_research_pdf_report(
            summary_data=summary_meta,
            benchmark_df=bench_df,
            pca_metrics=pca_res,
            validation_metrics=val_res
        )
        st.download_button(
            label="📕 Download Full Research Report (PDF)",
            data=pdf_bytes,
            file_name="BEACHMIND_SIH26054_Research_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
    except Exception as e:
        st.error(f"Error compiling PDF: {e}")

st.markdown("---")

# Section 2: CSV Data Tables Export
st.markdown("### 💾 Individual CSV Dataset Exports")
st.caption("Export intermediate representations for external geochemical analysis, GIS mapping, or peer review.")

c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    st.markdown("#### 1. Extracted Features Matrix")
    st.caption(f"{len(features_df)} samples × {len(features_df.columns)} dimensions")
    csv_feat = convert_df_to_csv_bytes(features_df)
    st.download_button(
        "📥 Download extracted_features.csv",
        data=csv_feat,
        file_name="extracted_features.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("#### 2. PCA Transformed Coordinates")
    if pca_res and "pca_df" in pca_res:
        st.caption(f"{len(pca_res['pca_df'])} samples × {pca_res['k_selected']} components")
        csv_pca = convert_df_to_csv_bytes(pca_res["pca_df"])
        st.download_button(
            "📥 Download pca_transformed_features.csv",
            data=csv_pca,
            file_name="pca_transformed_features.csv",
            mime="text/csv",
            use_container_width=True
        )

with c_exp2:
    st.markdown("#### 3. Sediment Cluster Assignments")
    if "Cluster_Label" in features_df.columns:
        cl_exp_df = features_df[["image_id", "filename", "Cluster_Label"]].copy()
        if "beach_name" in features_df.columns:
            cl_exp_df["beach_name"] = features_df["beach_name"]
        st.caption(f"Cluster labels assigned for {len(cl_exp_df)} samples")
        csv_cl = convert_df_to_csv_bytes(cl_exp_df)
        st.download_button(
            "📥 Download cluster_assignments.csv",
            data=csv_cl,
            file_name="cluster_assignments.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("#### 4. Model Comparison Benchmark Table")
    if not bench_df.empty:
        st.caption("Four-way experimental performance & latency metrics")
        csv_bench = convert_df_to_csv_bytes(bench_df)
        st.download_button(
            "📥 Download model_comparison_benchmark.csv",
            data=csv_bench,
            file_name="model_comparison_benchmark.csv",
            mime="text/csv",
            use_container_width=True
        )

if val_res and val_res.get("status") == "Success":
    st.markdown("---")
    st.markdown("#### 5. Sieve Analysis Validation Data")
    st.caption("Optical vs laboratory mechanical sieve metrics and residuals")
    csv_val = convert_df_to_csv_bytes(val_res["validation_df"])
    st.download_button(
        "📥 Download sieve_validation_metrics.csv",
        data=csv_val,
        file_name="sieve_validation_metrics.csv",
        mime="text/csv"
    )
