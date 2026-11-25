"""
Integration Test Suite for BEACHMIND (SIH26054)
Tests end-to-end pipeline execution:
- Dataset generation / ingestion
- Multi-modal feature extraction
- Scaling & PCA dimensionality reduction
- PC-grain size empirical correlation
- K-Means and DBSCAN clustering
- Four-way benchmark evaluation
- Sieve analysis validation metrics
- Geospatial mapping
- PDF report generation
"""

import os
import sys

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from modules.dataset_manager import generate_synthetic_demo_dataset
from modules.feature_extraction import extract_features_batch
from modules.dimensionality_reduction import (
    fit_and_scale_features,
    run_pca_analysis,
    analyze_pc_grain_size_correlation
)
from modules.clustering import (
    run_kmeans_clustering,
    run_dbscan_clustering,
    summarize_cluster_characteristics
)
from modules.evaluation import (
    run_four_way_benchmark,
    run_pca_component_selection_experiment
)
from modules.validation import evaluate_sieve_validation
from modules.mapping import (
    validate_coordinates_data,
    create_beach_sediment_map
)
from modules.report_generator import generate_research_pdf_report


def run_all_tests():
    print("=================================================================")
    print("      BEACHMIND (SIH26054) - AUTOMATED INTEGRATION TEST SUITE     ")
    print("=================================================================")

    # 1. Dataset Generation
    print("[TEST 1/9] Testing Procedural Demonstration Dataset Generation...")
    reg_df, img_dict = generate_synthetic_demo_dataset()
    assert len(reg_df) == 12, "Expected 12 demo records"
    assert len(img_dict) == 12, "Expected 12 demo images"
    print(f"  --> PASS: Generated {len(reg_df)} procedural sediment samples.")

    # 2. Batch Extraction
    print("[TEST 2/10] Testing Batch Feature Extraction Pipeline...")
    items = [(row['image_id'], row['filename'], img_dict[row['image_id']]) for _, row in reg_df.iterrows()]
    feat_df, art_map, diagnostics = extract_features_batch(items, pixel_to_mm_ratio=0.04)
    assert not feat_df.empty, "Feature extraction produced empty DataFrame"
    assert len(feat_df) == 12, "Expected 12 extracted feature rows"
    assert len(diagnostics) == 12, "Expected 12 diagnostic records"
    assert all(d["status"] == "SUCCESS" for d in diagnostics), "All demo items should succeed"
    merged_df = reg_df.merge(feat_df, on=['image_id', 'filename'])
    print(f"  --> PASS: Extracted {feat_df.shape[1]} dimensional features per image.")

    # 3. Scaling & PCA
    print("[TEST 3/10] Testing StandardScaler and PCA Dimensionality Reduction...")
    X_scaled, scaler, feat_cols = fit_and_scale_features(merged_df)
    assert X_scaled.shape[0] == 12, "Expected 12 scaled samples"
    
    pca_res = run_pca_analysis(X_scaled, feat_cols, n_components=3)
    assert pca_res["k_selected"] == 3
    assert pca_res["pca_df"].shape == (12, 3)
    k_sel = pca_res['k_selected']
    cum_var = pca_res['cumulative_explained_variance'][-1] * 100.0
    print(f"  --> PASS: PCA reduced {len(feat_cols)} features -> {k_sel} components (Variance: {cum_var:.2f}%).")

    # 4. Correlation Check
    print("[TEST 4/10] Testing Empirical PC vs Reference Grain Size Correlation...")
    corr_df = analyze_pc_grain_size_correlation(pca_res['pca_df'], merged_df['reference_d50'])
    assert not corr_df.empty, "Expected correlation results for reference D50"
    top_pc = corr_df.iloc[0]['Principal_Component']
    top_r = corr_df.iloc[0]['Pearson_Correlation_r']
    print(f"  --> PASS: Top PC correlated with grain size: {top_pc} (r = {top_r:.3f}).")

    # 5. Clustering
    print("[TEST 5/10] Testing K-Means++ and DBSCAN Clustering...")
    km_res = run_kmeans_clustering(pca_res['X_pca'], n_clusters=3)
    db_res = run_dbscan_clustering(pca_res['X_pca'], eps=1.5, min_samples=2)
    merged_df['Cluster_Label'] = km_res['labels']
    summary_df = summarize_cluster_characteristics(merged_df, km_res['labels'])
    assert len(summary_df) > 0, "Cluster summary should not be empty"
    print(f"  --> PASS: K-Means Silhouette = {km_res['silhouette_score']:.3f}, DBSCAN clusters = {db_res['n_clusters']}.")

    # 6. Benchmarks
    print("[TEST 6/10] Testing Four-Way Benchmark & Component Sensitivity Matrix...")
    bench_df = run_four_way_benchmark(X_scaled, pca_res['X_pca'], k_clusters=3)
    assert len(bench_df) == 4, "Expected 4 benchmark experiments"
    sens_df = run_pca_component_selection_experiment(X_scaled, feat_cols, k_clusters=3)
    assert not sens_df.empty, "Sensitivity matrix should not be empty"
    print("  --> PASS: Four-Way Benchmark Matrix and PCA Sensitivity calculated successfully.")

    # 7. Sieve Validation
    print("[TEST 7/10] Testing Mechanical Sieve Analysis Validation...")
    val_res = evaluate_sieve_validation(merged_df['reference_d50'], merged_df['morph_d50'])
    assert val_res['status'] == 'Success', "Validation status should be Success"
    print(f"  --> PASS: MAE = {val_res['mae']} mm, RMSE = {val_res['rmse']} mm, MAPE = {val_res['mape_percent']}%, R2 = {val_res['r2_score']}.")

    # 8. Folium Map
    print("[TEST 8/10] Testing Geospatial Beach Sediment Mapping...")
    is_valid, geo_df, msg = validate_coordinates_data(merged_df)
    assert is_valid, "Coordinates should be valid"
    fmap = create_beach_sediment_map(geo_df, cluster_column='Cluster_Label')
    assert fmap is not None, "Map object should be created"
    print("  --> PASS: Folium map populated with color-coded sediment cluster markers.")

    # 9. PDF Generation
    print("[TEST 9/10] Testing Scientific PDF Report Compilation...")
    pdf_bytes = generate_research_pdf_report(
        summary_data={'n_images': 12, 'n_features': len(feat_cols), 'dataset_source': 'Test Procedural', 'has_calibration': True},
        benchmark_df=bench_df,
        pca_metrics=pca_res,
        validation_metrics=val_res
    )
    assert len(pdf_bytes) > 2000, "PDF byte length too small"
    print(f"  --> PASS: Generated PDF document ({len(pdf_bytes)} bytes).")

    # 10. Actual UploadedFile / Decoded JPG Bytes Pipeline
    print("[TEST 10/10] Testing Streamlit UploadedFile and JPG Byte Decoding Pipeline...")
    import cv2
    
    # Generate actual JPG files in memory and test UploadedFile simulation
    test_jpgs = []
    for i in range(3):
        # Create varied sand-like image
        h, w = 256, 256
        synthetic_sand = np.random.randint(100, 220, (h, w, 3), dtype=np.uint8)
        # Add grain features
        for _ in range(50):
            cx, cy = np.random.randint(10, w-10), np.random.randint(10, h-10)
            cv2.circle(synthetic_sand, (cx, cy), np.random.randint(3, 10), (np.random.randint(50, 250), 180, 140), -1)
        
        ok, enc = cv2.imencode('.jpg', synthetic_sand, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        assert ok, "cv2.imencode failed"
        jpg_bytes = enc.tobytes()

        class MockUploadedFile:
            def __init__(self, name, data):
                self.name = name
                self._data = data
                self.size = len(data)
                self.type = "image/jpeg"
            def getvalue(self):
                return self._data
            def read(self):
                return self._data
            def seek(self, pos):
                pass

        uf = MockUploadedFile(f"uploaded_sample_{i+1}.jpg", jpg_bytes)
        test_jpgs.append((f"UPLOAD_{i+1:03d}", uf.name, uf))

    up_df, up_artifacts, up_diags = extract_features_batch(test_jpgs)
    assert not up_df.empty, "Uploaded JPG feature extraction must not be empty"
    assert len(up_df) == 3, f"Expected 3 extracted rows from 3 uploaded JPGs, got {len(up_df)}"
    assert all(d["status"] == "SUCCESS" for d in up_diags), f"Expected all uploaded JPGs to succeed, got {up_diags}"
    print(f"  --> PASS: Successfully processed 3 real JPG UploadedFile objects. Extracted {up_df.shape[1]} features.")

    print("\n=================================================================")
    print("   ALL 9 TEST STAGES PASSED! SYSTEM FULLY FUNCTIONAL & VERIFIED  ")
    print("=================================================================")


if __name__ == "__main__":
    run_all_tests()
