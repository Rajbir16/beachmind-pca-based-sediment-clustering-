"""
Page 6: Sieve Analysis Ground-Truth Validation
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
from modules.validation import evaluate_sieve_validation
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Laboratory Sieve Analysis Validation",
    "Quantitative Ground-Truth Validation: Optical vs. Mechanical Sieve Particle Diameters | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())

if features_df.empty:
    st.warning("⚠️ No active dataset found. Please upload images in 'Image Processing' or activate Demo Mode from the sidebar.")
    st.stop()

# Educational & Methodological Policy
st.markdown("""
<div class="alert-box">
<b>Strict Ground-Truth Validation Protocol:</b> This module quantitatively compares image-derived particle diameters (e.g. Optical D50 in mm) 
against standardized mechanical sieve analysis benchmarks (ASTM D422 / ISO 13320). 
<br/><b>Scientific Principle:</b> Unsupervised cluster labels are never conflated with ground-truth sediment size. 
Validation metrics (MAE, RMSE, MAPE, R²) are computed strictly against physical laboratory sieve measurements.
</div>
""", unsafe_allow_html=True)

# Check or Upload Sieve Data
st.markdown("### 📥 Sieve Ground-Truth Data")
c_up1, c_up2 = st.columns([2, 1])

with c_up1:
    st.markdown("**Optionally Upload Laboratory Sieve CSV to Pair with Current Images:**")
    uploaded_sieve = st.file_uploader("Upload Sieve CSV (Columns: image_id or filename, sieve_d50_mm)", type=["csv"])
    if uploaded_sieve:
        try:
            s_df = pd.read_csv(uploaded_sieve)
            st.success(f"Uploaded sieve table with {len(s_df)} rows.")
            # Merge with features_df
            # Look for image_id or filename
            id_col = next((c for c in s_df.columns if c.lower() in ["image_id", "id"]), None)
            d50_col = next((c for c in s_df.columns if "d50" in c.lower() or "grain" in c.lower() or "sieve" in c.lower()), None)
            
            if id_col and d50_col:
                s_map = dict(zip(s_df[id_col].astype(str), pd.to_numeric(s_df[d50_col], errors='coerce')))
                features_df["reference_d50"] = features_df["image_id"].astype(str).map(s_map).fillna(features_df.get("reference_d50", np.nan))
                st.session_state["features_df"] = features_df
                st.success("Successfully updated reference D50 measurements in session registry!")
            else:
                st.error("Uploaded CSV must contain an ID column and a D50/sieve column.")
        except Exception as e:
            st.error(f"Error reading sieve CSV: {e}")

with c_up2:
    has_ref = "reference_d50" in features_df.columns and not features_df["reference_d50"].isna().all()
    n_pairs = int((~features_df["reference_d50"].isna()).sum()) if "reference_d50" in features_df.columns else 0
    st.metric("Paired Ground-Truth Samples", f"{n_pairs}")
    st.caption("Status: " + ("Ready for validation" if n_pairs >= 2 else "Awaiting paired laboratory measurements"))

st.markdown("---")

# Execute Validation
if not has_ref or n_pairs < 2:
    st.warning("⚠️ Laboratory sieve analysis measurements are not available for the active images. Upload a reference CSV above, or switch to the Demonstration Dataset (which includes procedural ground truth) to inspect the validation module.")
else:
    # Run Validation
    # Compare morph_d50 or optical estimated grain size with reference_d50
    pred_col = "morph_d50" if "morph_d50" in features_df.columns else "morph_mean_diameter"
    actual_vals = features_df["reference_d50"].values
    pred_vals = features_df[pred_col].values

    val_res = evaluate_sieve_validation(actual_vals, pred_vals)
    st.session_state["validation_result"] = val_res

    # Metric Cards
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        st.metric("Mean Absolute Error (MAE)", f"{val_res['mae']} mm")
    with v2:
        st.metric("Root Mean Sq Error (RMSE)", f"{val_res['rmse']} mm")
    with v3:
        st.metric("Mean Abs % Error (MAPE)", f"{val_res['mape_percent']}%")
    with v4:
        r2_str = f"{val_res['r2_score']}" if val_res['r2_score'] != "N/A" else "N/A"
        st.metric("R² Score", r2_str)
    with v5:
        p_str = f"{val_res['pearson_r']}" if val_res['pearson_r'] != "N/A" else "N/A"
        st.metric("Pearson Correlation (r)", p_str)

    st.markdown("---")

    # Diagnostic Plots: Actual vs Predicted and Residuals
    g_col1, g_col2 = st.columns(2)
    val_table = val_res["validation_df"]

    with g_col1:
        # Actual vs Predicted Scatter with 1:1 Identity Line
        min_v = min(val_table["Reference_Sieve_mm"].min(), val_table["Optical_Estimated_mm"].min()) * 0.9
        max_v = max(val_table["Reference_Sieve_mm"].max(), val_table["Optical_Estimated_mm"].max()) * 1.1

        fig_reg = px.scatter(
            val_table,
            x="Reference_Sieve_mm",
            y="Optical_Estimated_mm",
            trendline="ols",
            title=f"Actual Sieve D50 vs. Camera-Estimated D50 (Pearson r = {val_res['pearson_r']})",
            labels={"Reference_Sieve_mm": "Laboratory Mechanical Sieve D50 (mm)", "Optical_Estimated_mm": "Computer Vision Estimated D50 (mm)"},
            template="plotly_white"
        )
        # Add 1:1 Identity reference line
        fig_reg.add_shape(
            type="line", line=dict(dash='dash', color='gray'),
            x0=min_v, y0=min_v, x1=max_v, y1=max_v
        )
        fig_reg.update_traces(marker=dict(size=10, opacity=0.85, color="#1e40af"))
        fig_reg.update_layout(height=420)
        st.plotly_chart(fig_reg, use_container_width=True)

    with g_col2:
        # Residual Plot (Residual vs True)
        fig_res = px.scatter(
            val_table,
            x="Reference_Sieve_mm",
            y="Residual_Error_mm",
            title="Residual Error Distribution (e = Optical - Sieve)",
            labels={"Reference_Sieve_mm": "Laboratory Sieve D50 (mm)", "Residual_Error_mm": "Residual Error (mm)"},
            template="plotly_white"
        )
        fig_res.add_hline(y=0.0, line_dash="dash", line_color="red")
        fig_res.update_traces(marker=dict(size=10, opacity=0.85, color="#b91c1c"))
        fig_res.update_layout(height=420)
        st.plotly_chart(fig_res, use_container_width=True)

    # Detailed Table
    st.markdown("#### 📋 Detailed Ground-Truth Comparison Table")
    st.dataframe(val_table, use_container_width=True)

    csv_val = convert_df_to_csv_bytes(val_table)
    st.download_button(
        "📥 Download Sieve Validation Data (CSV)",
        data=csv_val,
        file_name="sieve_validation_metrics.csv",
        mime="text/csv"
    )
