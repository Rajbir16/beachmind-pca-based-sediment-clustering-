"""
Geographical Beach Sediment Mapping Module
Part of BEACHMIND (SIH26054)

Integrates Folium with interactive geospatial markers:
- Renders sample positions on satellite / openstreetmap tiles
- Color-codes markers strictly by unsupervised clustering assignment
- Displays rich popup cards (Image ID, Beach, Coordinates, Estimated D50, Reference D50)
- Enforces strict verification of real coordinates; NEVER fabricates coordinates
"""

import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Scientific Color Palette for Distinct Clusters
CLUSTER_COLORS = [
    "#1f77b4",  # Cluster 0 - Blue
    "#2ca02c",  # Cluster 1 - Green
    "#ff7f0e",  # Cluster 2 - Orange
    "#9467bd",  # Cluster 3 - Purple
    "#17becf",  # Cluster 4 - Cyan
    "#e377c2",  # Cluster 5 - Pink
    "#bcbd22",  # Cluster 6 - Olive
    "#8c564b",  # Cluster 7 - Brown
]
NOISE_COLOR = "#7f7f7f"  # Noise (-1) - Gray


def get_cluster_color(cluster_id: int) -> str:
    """Returns hexadecimal color for a given cluster ID."""
    if cluster_id == -1:
        return NOISE_COLOR
    idx = int(cluster_id) % len(CLUSTER_COLORS)
    return CLUSTER_COLORS[idx]


def validate_coordinates_data(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, str]:
    """
    Strict validation of geographical coordinate fields.
    Refuses to invent coordinates if absent.
    """
    lat_col = next((c for c in df.columns if c.lower() in ["latitude", "lat", "y_coord"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["longitude", "lon", "lng", "x_coord"]), None)

    if not lat_col or not lon_col:
        return False, pd.DataFrame(), "Geographical mapping disabled: Latitude and Longitude columns are absent from the dataset registry."

    valid_df = df.copy()
    valid_df["lat"] = pd.to_numeric(valid_df[lat_col], errors='coerce')
    valid_df["lon"] = pd.to_numeric(valid_df[lon_col], errors='coerce')

    valid_coords = valid_df.dropna(subset=["lat", "lon"])
    # Validate global geographic ranges
    valid_coords = valid_coords[(valid_coords["lat"].between(-90, 90)) & (valid_coords["lon"].between(-180, 180))]

    if len(valid_coords) == 0:
        return False, pd.DataFrame(), "Geographical mapping disabled: No valid numeric geographic coordinates found in the dataset."

    return True, valid_coords, f"Found {len(valid_coords)} valid geospatial samples."


def create_beach_sediment_map(
    geospatial_df: pd.DataFrame,
    cluster_column: str = "Cluster_Label",
    selected_clusters: Optional[List[int]] = None,
    tile_layer: str = "OpenStreetMap"
) -> folium.Map:
    """
    Creates an interactive Folium map populated with sediment cluster markers.
    """
    plot_df = geospatial_df.copy()

    if selected_clusters is not None and cluster_column in plot_df.columns:
        plot_df = plot_df[plot_df[cluster_column].isin(selected_clusters)]

    if len(plot_df) == 0:
        center_lat, center_lon = 13.0, 80.2  # Default coastal reference
        zoom_start = 6
    else:
        center_lat = float(plot_df["lat"].mean())
        center_lon = float(plot_df["lon"].mean())
        zoom_start = 12 if len(plot_df) > 1 else 14

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=tile_layer,
        control_scale=True
    )

    for _, row in plot_df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        img_id = str(row.get("image_id", "N/A"))
        fname = str(row.get("filename", "N/A"))
        beach = str(row.get("beach_name", "Coastal Segment"))
        cl_val = int(row.get(cluster_column, 0)) if cluster_column in row and not pd.isna(row[cluster_column]) else 0

        color = get_cluster_color(cl_val)
        cl_title = f"Cluster {cl_val}" if cl_val != -1 else "Noise / Outlier"

        d50_val = row.get("morph_d50", np.nan)
        d50_str = f"{round(float(d50_val), 3)}" if not pd.isna(d50_val) else "N/A"
        
        ref_d50 = row.get("reference_d50", np.nan)
        ref_str = f"{round(float(ref_d50), 3)} mm" if not pd.isna(ref_d50) else "Not Measured"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4; min-width: 180px;">
            <h4 style="margin: 0 0 6px 0; color: #1e3a8a; border-bottom: 2px solid {color}; padding-bottom: 3px;">
                {beach}
            </h4>
            <b>Image ID:</b> {img_id}<br/>
            <b>File:</b> {fname}<br/>
            <b>Assigned Cluster:</b> <span style="color: {color}; font-weight: bold;">{cl_title}</span><br/>
            <b>Optical D50:</b> {d50_str}<br/>
            <b>Reference Sieve:</b> {ref_str}<br/>
            <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{beach} | {cl_title} (ID: {img_id})",
            color="#333333",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85
        ).add_to(m)

    return m
