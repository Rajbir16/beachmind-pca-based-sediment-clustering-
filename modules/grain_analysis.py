"""
Grain Morphology and Particle Size Distribution Analysis Module
Part of BEACHMIND (SIH26054)
Calculates:
- Particle area, perimeter, circularity, aspect ratio
- Equivalent spherical diameter (Deq)
- Grain-size percentiles (D10, D50, D90, sorting)
- Physical calibration (mm/pixel) conversion or pixel-scale relative measurements
- Geological Wentworth scale classification (when calibrated)
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def calculate_contour_morphology(
    contours: List[np.ndarray],
    pixel_to_mm_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes rigorous morphological metrics across segmented contours.
    
    If pixel_to_mm_ratio is provided (mm per pixel), metrics are converted to millimeters.
    Otherwise, metrics are reported in pixel units, explicitly marking calibration as uncalibrated.
    """
    if not contours:
        return {
            "particle_count": 0,
            "mean_area": np.nan,
            "median_area": np.nan,
            "std_area": np.nan,
            "mean_perimeter": np.nan,
            "mean_circularity": np.nan,
            "mean_aspect_ratio": np.nan,
            "mean_equivalent_diameter": np.nan,
            "d10_diameter": np.nan,
            "d50_diameter": np.nan,
            "d90_diameter": np.nan,
            "sorting_index": np.nan,
            "units": "pixels (uncalibrated)",
            "is_calibrated": False,
            "areas": [],
            "diameters": [],
            "circularities": [],
            "aspect_ratios": []
        }

    areas_px = []
    perimeters_px = []
    circularities = []
    aspect_ratios = []
    equiv_diameters_px = []

    for c in contours:
        area = float(cv2.contourArea(c))
        if area <= 1.0:
            continue
        perimeter = float(cv2.arcLength(c, True))
        if perimeter <= 0.0:
            continue

        areas_px.append(area)
        perimeters_px.append(perimeter)

        # Circularity = 4 * pi * Area / (Perimeter^2). Perfect circle = 1.0
        circ = (4.0 * np.pi * area) / (perimeter ** 2)
        circularities.append(min(float(circ), 1.0))

        # Equivalent circular diameter = 2 * sqrt(Area / pi)
        eq_diam = 2.0 * np.sqrt(area / np.pi)
        equiv_diameters_px.append(eq_diam)

        # Fit ellipse or bounding rectangle for aspect ratio
        if len(c) >= 5:
            try:
                (x, y), (d1, d2), angle = cv2.fitEllipse(c)
                major = max(d1, d2)
                minor = min(d1, d2)
                ar = (major / minor) if minor > 0 else 1.0
            except Exception:
                x, y, w, h = cv2.boundingRect(c)
                ar = float(max(w, h)) / float(min(w, h)) if min(w, h) > 0 else 1.0
        else:
            x, y, w, h = cv2.boundingRect(c)
            ar = float(max(w, h)) / float(min(w, h)) if min(w, h) > 0 else 1.0

        aspect_ratios.append(min(float(ar), 10.0))

    if not areas_px:
        return calculate_contour_morphology([])

    # Scale conversion
    is_calibrated = (pixel_to_mm_ratio is not None and pixel_to_mm_ratio > 0)
    scale = pixel_to_mm_ratio if is_calibrated else 1.0
    scale_sq = (scale ** 2)

    areas = np.array(areas_px) * scale_sq
    perimeters = np.array(perimeters_px) * scale
    diameters = np.array(equiv_diameters_px) * scale

    # Percentiles for Grain Size Distribution
    d10 = float(np.percentile(diameters, 10))
    d50 = float(np.percentile(diameters, 50))  # Median grain size
    d90 = float(np.percentile(diameters, 90))
    # Trask sorting index or geometric sorting: sqrt(D90 / D10)
    sorting_index = float(np.sqrt(d90 / d10)) if d10 > 0 else 1.0

    units = "mm" if is_calibrated else "pixels (relative)"

    return {
        "particle_count": len(areas),
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "std_area": float(np.std(areas)),
        "mean_perimeter": float(np.mean(perimeters)),
        "mean_circularity": float(np.mean(circularities)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "mean_equivalent_diameter": float(np.mean(diameters)),
        "d10_diameter": d10,
        "d50_diameter": d50,
        "d90_diameter": d90,
        "sorting_index": sorting_index,
        "units": units,
        "is_calibrated": is_calibrated,
        "pixel_to_mm_ratio": pixel_to_mm_ratio if is_calibrated else None,
        "areas": areas.tolist(),
        "diameters": diameters.tolist(),
        "circularities": circularities,
        "aspect_ratios": aspect_ratios
    }


def classify_wentworth_sediment(d50_mm: float) -> str:
    """
    Standard Geological Udden-Wentworth grain-size scale classification.
    Only valid if diameter is accurately calibrated in millimeters.
    """
    if np.isnan(d50_mm) or d50_mm <= 0:
        return "Unclassified / Invalid Size"
    if d50_mm < 0.0039:
        return "Clay (< 0.0039 mm)"
    elif d50_mm < 0.0625:
        return "Silt (0.0039 - 0.0625 mm)"
    elif d50_mm < 0.125:
        return "Very Fine Sand (0.0625 - 0.125 mm)"
    elif d50_mm < 0.25:
        return "Fine Sand (0.125 - 0.25 mm)"
    elif d50_mm < 0.50:
        return "Medium Sand (0.25 - 0.50 mm)"
    elif d50_mm < 1.0:
        return "Coarse Sand (0.50 - 1.0 mm)"
    elif d50_mm < 2.0:
        return "Very Coarse Sand (1.0 - 2.0 mm)"
    elif d50_mm < 4.0:
        return "Granule (2.0 - 4.0 mm)"
    elif d50_mm < 64.0:
        return "Pebble / Gravel (4.0 - 64.0 mm)"
    else:
        return "Cobble / Boulder (> 64.0 mm)"
