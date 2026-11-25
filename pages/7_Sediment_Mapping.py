"""
Page 7: Geographical Beach Sediment Mapping
Part of BEACHMIND (SIH26054)
"""

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar
)
from modules.mapping import (
    validate_coordinates_data,
    create_beach_sediment_map,
    get_cluster_color
)
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Geographical Beach Sediment Map",
    "Interactive Coastal Spatial Distribution of Unsupervised Sediment Clusters | SIH26054"
)

features_df = st.session_state.get("features_df", pd.DataFrame())

if features_df.empty:
    st.warning("⚠️ No active dataset found. Please upload images in 'Image Processing' or activate Demo Mode from the sidebar.")
    st.stop()

# Methodological Integrity Notice
st.markdown("""
<div class="alert-box">
<b>Geospatial Accuracy Policy:</b> To uphold strict scientific authenticity, coordinates are never fabricated. 
If geographic coordinates (Latitude / Longitude) are not present in your dataset or metadata CSV, 
spatial mapping is gracefully disabled rather than plotting samples at arbitrary or hallucinated locations.
</div>
""", unsafe_allow_html=True)

# Coordinate Validation Check
is_valid_geo, geo_df, geo_msg = validate_coordinates_data(features_df)

if not is_valid_geo:
    st.warning(f"⚠️ {geo_msg}")
    st.markdown("#### 🗺️ How to Enable Coastal Mapping:")
    st.markdown("""
    1. Navigate to **2_Image_Processing**.
    2. Upload a metadata CSV file alongside your images containing:
       - `latitude` (e.g. 13.0620)
       - `longitude` (e.g. 80.2825)
       - `beach_name` (e.g. Marina Beach North)
    3. Alternatively, click **'Load Synthetic Demo Dataset'** in the sidebar to view an authentic coastal demonstration along the Tamil Nadu coastline.
    """)
else:
    st.success(f"✅ {geo_msg}")

    # Cluster Filter Setup
    st.markdown("### 🎛️ Map Controls & Cluster Filters")
    c_m1, c_m2, c_m3 = st.columns([1, 1, 1])

    with c_m1:
        tile_choice = st.selectbox(
            "Base Map Imagery:",
            ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"]
        )

    # Detect Available Clusters
    cluster_col = "Cluster_Label" if "Cluster_Label" in geo_df.columns else None
    available_clusters = sorted(geo_df[cluster_col].unique().tolist()) if cluster_col else [0]

    with c_m2:
        selected_clusters = st.multiselect(
            "Filter Sediment Clusters:",
            options=available_clusters,
            default=available_clusters,
            format_func=lambda c: f"Cluster {c}" if c != -1 else "Noise / Outliers (-1)"
        )

    with c_m3:
        st.markdown("**Legend / Palette**")
        legend_html = "<div style='display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;'>"
        for cl in available_clusters:
            col_hex = get_cluster_color(cl)
            name = f"Cl {cl}" if cl != -1 else "Noise"
            legend_html += f"<span style='background-color: {col_hex}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{name}</span>"
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("---")

    # Render Folium Map
    f_map = create_beach_sediment_map(
        geo_df,
        cluster_column="Cluster_Label",
        selected_clusters=selected_clusters,
        tile_layer=tile_choice
    )

    st_folium(f_map, width="100%", height=550)

    st.markdown("---")

    # Spatial Transect Distribution Table
    st.markdown("### 📊 Coastal Transect Distribution Table")
    display_geo_cols = [c for c in ["image_id", "filename", "beach_name", "Cluster_Label", "lat", "lon", "morph_d50", "reference_d50"] if c in geo_df.columns]
    st.dataframe(geo_df[display_geo_cols], use_container_width=True)

    csv_geo = convert_df_to_csv_bytes(geo_df[display_geo_cols])
    st.download_button(
        "📥 Download Geospatial Sediment Registry (CSV)",
        data=csv_geo,
        file_name="beachmind_geospatial_sediment_data.csv",
        mime="text/csv"
    )
