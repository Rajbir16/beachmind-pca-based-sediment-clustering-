"""
Comprehensive Feature Extraction Subsystem
Part of BEACHMIND (SIH26054)

Extracts traditional Computer Vision and statistical features:
1. GLCM Texture: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM
2. Local Binary Patterns (LBP): Multiscale uniform histogram features (fixed bin size)
3. Gabor Filter Bank: Multi-orientation, multi-frequency mean and energy responses
4. Color Metrics: RGB and HSV channel means, standard deviations, and color histograms
5. Morphological & Grain-Size Metrics: Area, perimeter, circularity, aspect ratio, D10, D50, D90
6. Statistical Grayscale Moments: Mean, Median, Std, Variance, Min, Max, Skewness, Kurtosis
"""

import os
import io
import traceback
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import Dict, Any, List, Optional, Tuple

from modules.preprocessing import load_image_to_rgb, preprocess_image_pipeline, segment_sediment_grains
from modules.grain_analysis import calculate_contour_morphology


def extract_glcm_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts GLCM texture metrics across standard 4 angles (0, 45, 90, 135 deg) at distance=1 and distance=3.
    Returns averaged and directional scalar indicators with full NaN handling.
    """
    # Downscale intensity to 64 levels for stable co-occurrence density
    quantized = (gray_img // 4).astype(np.uint8)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')

    # Safe scalar conversion handling potential flat regions (where variance is zero)
    def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
        if arr is None or arr.size == 0 or np.isnan(arr).all():
            return default
        val = float(np.nanmean(arr))
        return val if not np.isnan(val) else default

    def _safe_std(arr: np.ndarray, default: float = 0.0) -> float:
        if arr is None or arr.size == 0 or np.isnan(arr).all():
            return default
        val = float(np.nanstd(arr))
        return val if not np.isnan(val) else default

    return {
        "glcm_contrast_mean": _safe_mean(contrast),
        "glcm_contrast_std": _safe_std(contrast),
        "glcm_dissimilarity_mean": _safe_mean(dissimilarity),
        "glcm_homogeneity_mean": _safe_mean(homogeneity, default=1.0),
        "glcm_energy_mean": _safe_mean(energy),
        "glcm_correlation_mean": _safe_mean(correlation, default=1.0),
        "glcm_asm_mean": _safe_mean(asm),
    }


def extract_lbp_features(gray_img: np.ndarray, num_points: int = 16, radius: float = 2.0) -> Dict[str, float]:
    """
    Extracts rotation-invariant uniform Local Binary Pattern (LBP) histogram features.
    Guarantees a fixed number of bins (num_points + 2) across all images.
    """
    lbp = local_binary_pattern(gray_img, P=num_points, R=radius, method="uniform")
    # Uniform LBP always has exactly num_points + 2 possible pattern codes
    fixed_n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=fixed_n_bins, range=(0, fixed_n_bins), density=True)

    features = {}
    for i in range(fixed_n_bins):
        features[f"lbp_bin_{i}"] = float(hist[i]) if i < len(hist) else 0.0

    features["lbp_energy"] = float(np.sum(hist ** 2))
    features["lbp_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-12)))
    return features


def extract_gabor_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Extracts Gabor filter bank features across 4 orientations and 3 frequencies.
    Captures multi-scale sediment grain alignment and granularity.
    """
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.08, 0.16, 0.32]  # coarse, medium, fine grain textures
    ksize = 21

    features = {}
    for i_theta, theta in enumerate(orientations):
        for i_freq, freq in enumerate(frequencies):
            wavelength = 1.0 / freq
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma=3.0, theta=theta, lambd=wavelength, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            filtered = cv2.filter2D(gray_img.astype(np.float32), cv2.CV_32F, kernel)
            
            mean_val = float(np.mean(filtered))
            energy_val = float(np.mean(filtered ** 2))
            
            features[f"gabor_mean_o{i_theta}_f{i_freq}"] = mean_val if not np.isnan(mean_val) else 0.0
            features[f"gabor_energy_o{i_theta}_f{i_freq}"] = energy_val if not np.isnan(energy_val) else 0.0

    return features


def extract_color_features(rgb_img: np.ndarray, hsv_img: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical color characteristics across RGB and HSV color spaces.
    Beach sediment composition strongly influences color (quartz, mineral sands, shell fragments).
    """
    r_chan, g_chan, b_chan = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    h_chan, s_chan, v_chan = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # Color histograms (4 bins per channel for compact representation)
    r_hist, _ = np.histogram(r_chan, bins=4, range=(0, 256), density=True)
    g_hist, _ = np.histogram(g_chan, bins=4, range=(0, 256), density=True)
    b_hist, _ = np.histogram(b_chan, bins=4, range=(0, 256), density=True)

    features = {
        "color_r_mean": float(np.mean(r_chan)),
        "color_r_std": float(np.std(r_chan)),
        "color_g_mean": float(np.mean(g_chan)),
        "color_g_std": float(np.std(g_chan)),
        "color_b_mean": float(np.mean(b_chan)),
        "color_b_std": float(np.std(b_chan)),
        "color_h_mean": float(np.mean(h_chan)),
        "color_h_std": float(np.std(h_chan)),
        "color_s_mean": float(np.mean(s_chan)),
        "color_s_std": float(np.std(s_chan)),
        "color_v_mean": float(np.mean(v_chan)),
        "color_v_std": float(np.std(v_chan)),
    }

    for i in range(4):
        features[f"color_r_hist_{i}"] = float(r_hist[i])
        features[f"color_g_hist_{i}"] = float(g_hist[i])
        features[f"color_b_hist_{i}"] = float(b_hist[i])

    return features


def extract_statistical_features(gray_img: np.ndarray) -> Dict[str, float]:
    """
    Computes first- to fourth-order intensity moments of the grayscale image.
    """
    flat = gray_img.ravel().astype(np.float64)

    mean_val = float(np.mean(flat))
    median_val = float(np.median(flat))
    std_val = float(np.std(flat))
    var_val = float(np.var(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    
    skew_raw = float(stats.skew(flat))
    skew_val = skew_raw if not np.isnan(skew_raw) else 0.0
    
    kurt_raw = float(stats.kurtosis(flat))
    kurt_val = kurt_raw if not np.isnan(kurt_raw) else 0.0

    return {
        "stat_mean": mean_val,
        "stat_median": median_val,
        "stat_std": std_val,
        "stat_variance": var_val,
        "stat_min": min_val,
        "stat_max": max_val,
        "stat_skewness": skew_val,
        "stat_kurtosis": kurt_val
    }


def extract_all_features_single_image(
    image_source: Any,
    image_id: str = "img_001",
    filename: str = "unknown.jpg",
    pixel_to_mm_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    End-to-end extraction from a single raw image input into a full feature vector dictionary.
    First decodes and verifies image using load_image_to_rgb.
    """
    # 1. Image decoding & format verification
    image_rgb = load_image_to_rgb(image_source)
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError(f"Failed to decode image from source {filename} (result is not a numpy array).")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Decoded image must be 3-channel RGB (H, W, 3). Got shape {image_rgb.shape}.")
    if image_rgb.shape[0] < 5 or image_rgb.shape[1] < 5:
        raise ValueError(f"Image dimensions {image_rgb.shape[:2]} are too small for processing.")

    # 2. Preprocessing pipeline
    prep = preprocess_image_pipeline(image_rgb)
    clahe_gray = prep["clahe_gray"]
    rgb = prep["resized_rgb"]
    hsv = prep["hsv"]

    # 3. Grain segmentation & morphology
    binary_mask, contours, quality = segment_sediment_grains(clahe_gray)
    morph_data = calculate_contour_morphology(contours, pixel_to_mm_ratio=pixel_to_mm_ratio)

    # 4. Individual feature modules
    glcm_feat = extract_glcm_features(clahe_gray)
    lbp_feat = extract_lbp_features(clahe_gray)
    gabor_feat = extract_gabor_features(clahe_gray)
    color_feat = extract_color_features(rgb, hsv)
    stat_feat = extract_statistical_features(clahe_gray)

    # Morphological features to numeric feature vector
    morph_feat = {
        "morph_particle_count": float(morph_data["particle_count"]),
        "morph_mean_area": float(morph_data["mean_area"]) if not np.isnan(morph_data["mean_area"]) else 0.0,
        "morph_median_area": float(morph_data["median_area"]) if not np.isnan(morph_data["median_area"]) else 0.0,
        "morph_mean_perimeter": float(morph_data["mean_perimeter"]) if not np.isnan(morph_data["mean_perimeter"]) else 0.0,
        "morph_mean_circularity": float(morph_data["mean_circularity"]) if not np.isnan(morph_data["mean_circularity"]) else 0.0,
        "morph_mean_aspect_ratio": float(morph_data["mean_aspect_ratio"]) if not np.isnan(morph_data["mean_aspect_ratio"]) else 1.0,
        "morph_mean_diameter": float(morph_data["mean_equivalent_diameter"]) if not np.isnan(morph_data["mean_equivalent_diameter"]) else 0.0,
        "morph_d10": float(morph_data["d10_diameter"]) if not np.isnan(morph_data["d10_diameter"]) else 0.0,
        "morph_d50": float(morph_data["d50_diameter"]) if not np.isnan(morph_data["d50_diameter"]) else 0.0,
        "morph_d90": float(morph_data["d90_diameter"]) if not np.isnan(morph_data["d90_diameter"]) else 0.0,
        "morph_sorting_index": float(morph_data["sorting_index"]) if not np.isnan(morph_data["sorting_index"]) else 1.0,
        "seg_quality_score": float(quality["quality_score"]),
        "seg_coverage_ratio": float(quality["coverage_ratio"])
    }

    # Combine into unified dictionary
    combined: Dict[str, Any] = {
        "image_id": image_id,
        "filename": filename,
        **glcm_feat,
        **lbp_feat,
        **gabor_feat,
        **color_feat,
        **stat_feat,
        **morph_feat
    }

    return {
        "features": combined,
        "image_rgb": image_rgb,
        "preprocessing_artifacts": prep,
        "segmentation_mask": binary_mask,
        "contours": contours,
        "morphology_details": morph_data,
        "quality_metrics": quality
    }


def extract_features_batch(
    image_items: List[Tuple[str, str, Any]],
    pixel_to_mm_ratio: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Processes a batch of images: [(image_id, filename, image_source), ...].
    Each image is processed independently with full diagnostic logging.

    Returns:
        features_df: Pandas DataFrame with feature rows for all successfully processed images
        artifacts_map: Dict mapping image_id to preprocessed artifacts and morphology
        diagnostics: List of diagnostic dicts reporting status, sizes, shapes, and error traces
    """
    records = []
    artifacts_map = {}
    diagnostics = []

    for img_id, fname, source in image_items:
        # Determine file size if accessible
        file_size = 0
        if hasattr(source, "size"):
            file_size = source.size
        elif hasattr(source, "getvalue"):
            file_size = len(source.getvalue())
        elif isinstance(source, (bytes, bytearray)):
            file_size = len(source)
        elif isinstance(source, str) and os.path.exists(source):
            file_size = os.path.getsize(source)
        elif isinstance(source, np.ndarray):
            file_size = source.nbytes

        diag: Dict[str, Any] = {
            "image_id": img_id,
            "filename": fname,
            "file_size_bytes": file_size,
            "decoded_shape": None,
            "decode_success": False,
            "preprocess_success": False,
            "feature_success": False,
            "status": "PENDING",
            "error_message": None,
            "traceback": None
        }

        try:
            res = extract_all_features_single_image(
                source, image_id=img_id, filename=fname, pixel_to_mm_ratio=pixel_to_mm_ratio
            )
            diag["decoded_shape"] = res["image_rgb"].shape
            diag["decode_success"] = True
            diag["preprocess_success"] = True
            diag["feature_success"] = True
            diag["status"] = "SUCCESS"

            records.append(res["features"])
            artifacts_map[img_id] = {
                "image_rgb": res["image_rgb"],
                "prep": res["preprocessing_artifacts"],
                "mask": res["segmentation_mask"],
                "morphology": res["morphology_details"],
                "quality": res["quality_metrics"]
            }
        except Exception as e:
            tb = traceback.format_exc()
            diag["status"] = "FAILED"
            diag["error_message"] = str(e)
            diag["traceback"] = tb
            # Try to determine if decode succeeded before failure
            try:
                probe = load_image_to_rgb(source)
                diag["decoded_shape"] = probe.shape
                diag["decode_success"] = True
            except Exception:
                diag["decode_success"] = False

        diagnostics.append(diag)

    if not records:
        return pd.DataFrame(), {}, diagnostics

    df = pd.DataFrame(records)
    
    # Clean any NaN/inf values safely using column medians or zero imputation
    feature_cols = [c for c in df.columns if c not in ["image_id", "filename"]]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            median_val = df[col].median()
            fill_val = median_val if not np.isnan(median_val) else 0.0
            df[col] = df[col].fillna(fill_val)

    return df, artifacts_map, diagnostics
