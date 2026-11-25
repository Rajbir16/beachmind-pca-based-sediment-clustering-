"""
Sieve Analysis and Ground-Truth Validation Module
Part of BEACHMIND (SIH26054)

Validates optical/computer-vision grain-size estimations against
laboratory mechanical sieve-analysis measurements:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R-squared (Coefficient of Determination)
- Pearson correlation coefficient (r)
- Residual analysis (Residuals = Predicted - Actual)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def evaluate_sieve_validation(
    actual_sizes: np.ndarray,
    predicted_sizes: np.ndarray
) -> Dict[str, Any]:
    """
    Computes rigorous statistical regression error metrics between laboratory sieve ground truth
    and camera-derived optical grain sizes.
    """
    actual = np.asarray(actual_sizes, dtype=np.float64)
    predicted = np.asarray(predicted_sizes, dtype=np.float64)

    # Filter invalid/nan values
    valid_mask = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual > 0)
    y_true = actual[valid_mask]
    y_pred = predicted[valid_mask]

    n_valid = len(y_true)
    if n_valid < 2:
        return {
            "valid_samples_count": n_valid,
            "mae": np.nan,
            "rmse": np.nan,
            "mape_percent": np.nan,
            "r2_score": np.nan,
            "pearson_r": np.nan,
            "status": "Insufficient valid ground-truth pairs (minimum 2 required)."
        }

    # Error metrics
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # MAPE (only on strictly positive non-zero true values)
    mape = float(np.mean(np.abs(residuals / y_true))) * 100.0

    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan

    # Pearson r
    corr_matrix = np.corrcoef(y_true, y_pred)
    pearson_r = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else np.nan

    validation_df = pd.DataFrame({
        "Reference_Sieve_mm": y_true,
        "Optical_Estimated_mm": y_pred,
        "Residual_Error_mm": residuals,
        "Absolute_Error_mm": np.abs(residuals),
        "Percentage_Error": (np.abs(residuals / y_true)) * 100.0
    })

    return {
        "valid_samples_count": n_valid,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_percent": round(mape, 2),
        "r2_score": round(r2, 4) if not np.isnan(r2) else "N/A",
        "pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else "N/A",
        "validation_df": validation_df,
        "residuals": residuals.tolist(),
        "status": "Success"
    }
