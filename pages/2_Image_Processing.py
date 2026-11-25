"""
Page 2: Image Processing & Feature Extraction
Part of BEACHMIND (SIH26054)
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from modules.ui_common import (
    init_session_state,
    apply_scientific_theme,
    render_header,
    render_sidebar
)
from modules.dataset_manager import (
    parse_uploaded_zip_file,
    align_metadata_with_images
)
from modules.preprocessing import (
    load_image_to_rgb,
    preprocess_image_pipeline,
    segment_sediment_grains
)
from modules.grain_analysis import calculate_contour_morphology
from modules.feature_extraction import extract_features_batch
from modules.dimensionality_reduction import fit_and_scale_features, run_pca_analysis
from modules.clustering import run_kmeans_clustering, run_dbscan_clustering, summarize_cluster_characteristics
from modules.evaluation import run_four_way_benchmark
from modules.report_generator import convert_df_to_csv_bytes

init_session_state()
apply_scientific_theme()
render_sidebar()

render_header(
    "Image Preprocessing & Feature Extraction",
    "Computer Vision Pipeline, Grain Morphology, and Multi-Modal Feature Extraction | SIH26054"
)

# Tab Layout: 1. Dataset Ingestion, 2. Preprocessing Inspector, 3. Feature Extraction Table
tab_ingest, tab_inspect, tab_table = st.tabs([
    "📥 Ingestion & Calibration",
    "🔍 Preprocessing Stage Inspector",
    "📊 Extracted Features Registry"
])

# ----------------- TAB 1: INGESTION -----------------
with tab_ingest:
    st.markdown("### 📂 Upload Sediment Imagery & Metadata")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        upload_type = st.radio(
            "Select Upload Format:",
            ["Multiple Image Files (JPG/PNG)", "ZIP Archive (.zip)", "Single Image File"],
            horizontal=True
        )

        uploaded_images = None
        uploaded_zip = None

        if upload_type == "Multiple Image Files (JPG/PNG)":
            uploaded_images = st.file_uploader(
                "Upload Beach Sand Images",
                type=["jpg", "jpeg", "png", "tif", "bmp"],
                accept_multiple_files=True
            )
        elif upload_type == "ZIP Archive (.zip)":
            uploaded_zip = st.file_uploader(
                "Upload ZIP Archive containing Sand Images",
                type=["zip"]
            )
        else:
            single_img = st.file_uploader("Upload Single Beach Sand Image", type=["jpg", "jpeg", "png", "tif"])
            if single_img:
                uploaded_images = [single_img]

    with col_up2:
        st.markdown("**Optional Metadata / Reference CSV (SandSnap format or custom)**")
        uploaded_meta = st.file_uploader(
            "Upload Metadata CSV (Beach name, Lat/Lon, Reference D50)",
            type=["csv"]
        )
        
        st.markdown("**Optical Scale Calibration**")
        use_calib = st.checkbox("Enable Physical Calibration (Convert pixels to mm)")
        calib_ratio = None
        if use_calib:
            calib_ratio = st.number_input(
                "Scale Ratio (mm per pixel):",
                min_value=0.0001,
                max_value=10.0,
                value=0.039,
                step=0.001,
                format="%.4f"
            )
            st.caption("Example: A 20mm scale marker spanning 512 pixels corresponds to ~0.039 mm/px.")
        else:
            st.caption("Operating in relative pixel dimensions. Results will be explicitly tagged as uncalibrated.")

    st.markdown("---")
    
    # Process Upload Trigger
    if st.button("⚡ Process Uploaded Images & Extract Features", type="primary"):
        meta_df = None
        if uploaded_meta:
            try:
                meta_df = pd.read_csv(uploaded_meta)
                st.success(f"Loaded metadata CSV with {len(meta_df)} records.")
            except Exception as e:
                st.error(f"Error reading metadata CSV: {e}")

        image_items = []
        if uploaded_zip is not None:
            with st.spinner("Extracting and parsing ZIP archive..."):
                image_items = parse_uploaded_zip_file(uploaded_zip.read())
        elif uploaded_images:
            for idx, uf in enumerate(uploaded_images):
                img_id = f"UPLOAD_{idx+1:03d}"
                image_items.append((img_id, uf.name, uf))

        if not image_items:
            st.error("Please select or upload at least one image file or ZIP archive.")
        else:
            with st.spinner(f"Executing CV Preprocessing & Feature Extraction for {len(image_items)} images..."):
                # Align with metadata
                reg_df = align_metadata_with_images(image_items, meta_df)
                
                # Extract features batch with detailed diagnostics
                feat_df, art_map, diagnostics = extract_features_batch(image_items, pixel_to_mm_ratio=calib_ratio)
                
                # Diagnostic evaluation
                diag_df = pd.DataFrame(diagnostics)
                success_count = sum(1 for d in diagnostics if d["status"] == "SUCCESS")
                failed_count = sum(1 for d in diagnostics if d["status"] == "FAILED")

                if failed_count > 0:
                    with st.expander(f"⚠️ Diagnostic Report: {failed_count} of {len(image_items)} images failed processing", expanded=(success_count == 0)):
                        st.markdown("**Per-Image Extraction Diagnostic Status:**")
                        display_cols = ["image_id", "filename", "file_size_bytes", "decoded_shape", "decode_success", "preprocess_success", "feature_success", "status", "error_message"]
                        available_cols = [c for c in display_cols if c in diag_df.columns]
                        st.dataframe(diag_df[available_cols], use_container_width=True)
                        
                        st.markdown("**Detailed Error Tracebacks:**")
                        for d in diagnostics:
                            if d["status"] == "FAILED":
                                st.error(f"❌ **{d['filename']}** ({d['image_id']}) - Size: {d['file_size_bytes']} bytes")
                                st.code(d.get("traceback") or d.get("error_message") or "Unknown error", language="python")

                if feat_df.empty:
                    st.error("Feature extraction produced no valid outputs. Please review the diagnostic breakdown above.")
                else:
                    # Store image arrays in memory from art_map (avoiding re-reading stream)
                    imgs_dict = {}
                    for img_id, a_data in art_map.items():
                        if "image_rgb" in a_data:
                            imgs_dict[img_id] = a_data["image_rgb"]
                        elif "prep" in a_data and "resized_rgb" in a_data["prep"]:
                            imgs_dict[img_id] = a_data["prep"]["resized_rgb"]

                    # Filter registry to only successfully processed images
                    successful_ids = set(feat_df["image_id"])
                    filtered_reg_df = reg_df[reg_df["image_id"].isin(successful_ids)].copy()

                    # Merge features and metadata
                    merged_df = filtered_reg_df.merge(feat_df, on=["image_id", "filename"], how="left")
                    
                    st.session_state["dataset_registry"] = filtered_reg_df
                    st.session_state["images_dict"] = imgs_dict
                    st.session_state["features_df"] = merged_df
                    st.session_state["artifacts_map"] = art_map
                    st.session_state["is_demo_mode"] = False
                    st.session_state["pixel_to_mm_ratio"] = calib_ratio

                    # Automatically fit scaler & PCA
                    X_scaled, scaler, feat_cols = fit_and_scale_features(merged_df)
                    st.session_state["X_scaled"] = X_scaled
                    st.session_state["scaler"] = scaler
                    st.session_state["feature_cols"] = feat_cols

                    pca_components = min(3, max(1, X_scaled.shape[0]), len(feat_cols))
                    pca_res = run_pca_analysis(X_scaled, feat_cols, n_components=pca_components)
                    st.session_state["pca_result"] = pca_res

                    # Initial clustering if >= 2 samples
                    if len(merged_df) >= 2:
                        k_use = min(3, max(2, len(merged_df) - 1)) if len(merged_df) > 2 else 2
                        km_res = run_kmeans_clustering(pca_res["X_pca"], n_clusters=k_use)
                        db_res = run_dbscan_clustering(pca_res["X_pca"], eps=1.5, min_samples=min(2, len(merged_df)))
                        
                        merged_df["Cluster_Label"] = km_res["labels"]
                        cl_sum = summarize_cluster_characteristics(merged_df, km_res["labels"])

                        st.session_state["clustering_result"] = {
                            "kmeans": km_res,
                            "dbscan": db_res,
                            "summary": cl_sum,
                            "active_labels": km_res["labels"],
                            "active_algo": "K-Means (PCA Space)"
                        }

                        bench_df = run_four_way_benchmark(X_scaled, pca_res["X_pca"], k_clusters=k_use)
                        st.session_state["benchmark_df"] = bench_df
                    else:
                        st.session_state["clustering_result"] = None
                        st.session_state["benchmark_df"] = pd.DataFrame()
                        st.info("Uploaded 1 image. Clustering and benchmarking require at least 2 images.")

                    st.success(f"Successfully processed {len(merged_df)} of {len(image_items)} images and extracted {len(feat_cols)} features!")
                    st.rerun()

# ----------------- TAB 2: PREPROCESSING INSPECTOR -----------------
with tab_inspect:
    st.markdown("### 🔬 Multi-Stage Preprocessing Pipeline Verification")
    
    imgs_dict = st.session_state.get("images_dict", {})
    art_map = st.session_state.get("artifacts_map", {})
    reg_df = st.session_state.get("dataset_registry", pd.DataFrame())

    if not imgs_dict:
        st.info("No images loaded. Upload images in the first tab or enable demonstration mode.")
    else:
        sample_ids = list(imgs_dict.keys())
        selected_id = st.selectbox("Select Sample Image to Inspect:", sample_ids)

        curr_img = imgs_dict[selected_id]
        
        # Interactive preprocessing parameters
        st.markdown("**Fine-Tune Preprocessing Hyperparameters:**")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            denoise_choice = st.selectbox("Denoising Filter", ["Gaussian", "Median", "Bilateral"])
        with p_col2:
            kernel_sz = st.slider("Filter Kernel Size", 3, 11, 5, step=2)
        with p_col3:
            clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, step=0.5)
        with p_col4:
            seg_method = st.selectbox("Segmentation Method", ["Adaptive Gaussian", "Otsu"])

        # Re-run pipeline dynamically for the inspected image
        prep = preprocess_image_pipeline(
            curr_img,
            denoise_method=denoise_choice.lower(),
            kernel_size=kernel_sz,
            apply_clahe=True,
            clahe_clip_limit=clahe_clip
        )
        
        bin_mask, contours, quality = segment_sediment_grains(
            prep["clahe_gray"],
            method="otsu" if seg_method == "Otsu" else "adaptive"
        )

        st.markdown("---")
        st.markdown("#### Side-by-Side Visual Verification")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.image(prep["resized_rgb"], caption="1. Original Resized RGB", use_container_width=True)
            st.image(prep["denoised_gray"], caption=f"2. {denoise_choice} Denoised Grayscale", use_container_width=True)

        with v_col2:
            st.image(prep["clahe_gray"], caption=f"3. Contrast Enhanced (CLAHE Clip: {clahe_clip})", use_container_width=True)
            st.image(prep["edges_canny"], caption="4. Canny Edge Detection", use_container_width=True)

        with v_col3:
            st.image(bin_mask, caption="5. Binary Grain Boundary Mask", use_container_width=True)
            
            # Draw contours on RGB for overlay visualization
            overlay = prep["resized_rgb"].copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            st.image(overlay, caption=f"6. Grain Contours Overlay ({len(contours)} grains)", use_container_width=True)

        # Segmentation Quality Check Card
        st.markdown("#### 🛡️ Segmentation Quality Assessment")
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            st.metric("Quality Score", f"{quality['quality_score']*100:.1f}%")
        with q_col2:
            st.metric("Grain Coverage Ratio", f"{quality['coverage_ratio']*100:.1f}%")
        with q_col3:
            st.metric("Status", quality["quality_status"])

        if not quality["is_acceptable"]:
            st.warning("⚠️ Segmentation quality is sub-optimal. Some contours may reflect illumination gradients or clumping rather than isolated grains.")
        else:
            st.success("✅ Segmentation quality satisfies morphological validity criteria.")

# ----------------- TAB 3: FEATURES TABLE -----------------
with tab_table:
    st.markdown("### 📋 Structured Extracted Features DataFrame")
    features_df = st.session_state.get("features_df", pd.DataFrame())

    if features_df.empty:
        st.info("Feature matrix is currently empty. Run feature extraction from Tab 1.")
    else:
        st.markdown(f"Total Samples: **{len(features_df)}** | Total Dimension Columns: **{len(features_df.columns)}**")
        st.dataframe(features_df, use_container_width=True)

        csv_bytes = convert_df_to_csv_bytes(features_df)
        st.download_button(
            label="📥 Download extracted_features.csv",
            data=csv_bytes,
            file_name="extracted_features.csv",
            mime="text/csv",
            type="primary"
        )
