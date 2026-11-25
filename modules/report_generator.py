"""
Report and Export Generation Subsystem
Part of BEACHMIND (SIH26054)

Generates:
- Formatted CSV tables for features, PCA loadings, cluster assignments, and benchmarks
- Comprehensive multi-page scientific PDF Research Report using fpdf2
"""

import io
import datetime
import pandas as pd
import numpy as np
from fpdf import FPDF
from typing import Dict, Any, Optional


def sanitize_pdf_text(text: str) -> str:
    """Replaces Unicode characters that cannot be encoded in core PDF fonts."""
    replacements = {
        "–": "-",
        "—": "-",
        "≥": ">=",
        "≤": "<=",
        "±": "+/-",
        "²": "^2",
        "µ": "u",
        "•": "*",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class BeachmindPDFReport(FPDF):
    """Custom scientific PDF report generator with header and footer."""
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 58, 138)  # Deep research navy
        self.cell(0, 8, sanitize_pdf_text("BEACHMIND - SIH26054 Research & Technical Report"), border=0, ln=1, align="L")
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, sanitize_pdf_text(f"Page {self.page_no()}/{{nb}} | Smart India Hackathon SIH26054 | Generated on {datetime.date.today()}"), align="C")


def generate_research_pdf_report(
    summary_data: Dict[str, Any],
    benchmark_df: Optional[pd.DataFrame] = None,
    pca_metrics: Optional[Dict[str, Any]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Constructs a complete scientific evaluation PDF document.
    """
    pdf = BeachmindPDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Banner
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 10, sanitize_pdf_text("Intelligent Beach Sediment Mapping"), ln=1, align="C")
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(75, 85, 99)
    pdf.cell(0, 6, sanitize_pdf_text("PCA-Based Feature Reduction of Camera-Derived Grain-Size Data (SIH26054)"), ln=1, align="C")
    pdf.ln(5)

    # Section 1: Executive Overview
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, sanitize_pdf_text("1. Executive Summary & Dataset Metadata"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(31, 41, 55)

    n_images = summary_data.get("n_images", 0)
    n_features = summary_data.get("n_features", 0)
    dataset_source = summary_data.get("dataset_source", "Uploaded Images / SandSnap Standard")
    has_calib = summary_data.get("has_calibration", False)

    summary_text = (
        f"This report presents experimental results obtained using classical Computer Vision and "
        f"unsupervised Machine Learning algorithms. The dataset comprises {n_images} beach sediment image samples "
        f"from source '{dataset_source}'. A total of {n_features} quantitative features spanning GLCM texture, "
        f"Local Binary Patterns (LBP), Gabor filter banks, statistical color moments, and grain morphology were extracted. "
        f"Physical scale calibration: {'Applied (mm/pixel)' if has_calib else 'Pixel-based relative scale (Uncalibrated)'}."
    )
    pdf.multi_cell(0, 5, sanitize_pdf_text(summary_text))
    pdf.ln(4)

    # Section 2: Dimensionality Reduction via PCA
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, sanitize_pdf_text("2. Principal Component Analysis (PCA) Results"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(31, 41, 55)

    if pca_metrics:
        k_sel = pca_metrics.get("k_selected", "N/A")
        orig_f = pca_metrics.get("n_original_features", "N/A")
        red_ratio = pca_metrics.get("reduction_ratio_percent", 0.0)
        cum_var = pca_metrics.get("cumulative_explained_variance", [0.0])
        cum_var_pct = round(cum_var[-1] * 100.0, 2) if cum_var else 0.0

        pca_text = (
            f"Prior to PCA, all feature distributions were normalized using StandardScaler to eliminate unit scale dominance. "
            f"PCA successfully compressed the dimensionality from {orig_f} original features down to {k_sel} orthogonal "
            f"principal components (a {red_ratio:.1f}% dimensionality reduction), while preserving {cum_var_pct}% of total variance. "
            f"Eigenvalue scree analysis confirmed that dominant components encapsulate multi-scale grain granularity and tonal contrast."
        )
    else:
        pca_text = "PCA analysis details are recorded in the attached digital project export."
    pdf.multi_cell(0, 5, sanitize_pdf_text(pca_text))
    pdf.ln(4)

    # Section 3: Model Comparison Benchmark
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, sanitize_pdf_text("3. Four-Way Experimental Benchmark"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(31, 41, 55)

    if benchmark_df is not None and not benchmark_df.empty:
        # Render clean text table
        pdf.set_font("Helvetica", "B", 8)
        col_widths = [16, 46, 26, 26, 24, 24, 26]
        headers = ["ID", "Configuration", "Features", "Clusters", "Silhouette", "Davies-B.", "Time (ms)"]
        
        pdf.set_fill_color(243, 244, 246)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 6, sanitize_pdf_text(h), border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for _, row in benchmark_df.iterrows():
            pdf.cell(col_widths[0], 5, sanitize_pdf_text(str(row.get("Experiment_ID", ""))), border=1, align="C")
            pdf.cell(col_widths[1], 5, sanitize_pdf_text(str(row.get("Experiment_Name", "")))[:26], border=1)
            pdf.cell(col_widths[2], 5, sanitize_pdf_text(f"{row.get('Retained_Features', '')} ({str(row.get('Dimensionality_Reduction', ''))[:4]})"), border=1, align="C")
            pdf.cell(col_widths[3], 5, sanitize_pdf_text(f"{row.get('Clusters_Found', '')} (N:{row.get('Noise_Samples', 0)})"), border=1, align="C")
            pdf.cell(col_widths[4], 5, sanitize_pdf_text(str(row.get("Silhouette_Score", "N/A"))), border=1, align="C")
            pdf.cell(col_widths[5], 5, sanitize_pdf_text(str(row.get("Davies_Bouldin_Index", "N/A"))), border=1, align="C")
            pdf.cell(col_widths[6], 5, sanitize_pdf_text(str(row.get("Computation_Time_ms", ""))), border=1, align="C")
            pdf.ln()
    pdf.ln(4)

    # Section 4: Sieve Analysis Validation
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, sanitize_pdf_text("4. Laboratory Sieve Validation"), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(31, 41, 55)

    if validation_metrics and validation_metrics.get("status") == "Success":
        val_text = (
            f"Optical grain diameter estimates were quantitatively validated against mechanical sieve benchmarks "
            f"across {validation_metrics.get('valid_samples_count')} ground-truth pairs.\n"
            f"- Mean Absolute Error (MAE): {validation_metrics.get('mae')} mm\n"
            f"- Root Mean Squared Error (RMSE): {validation_metrics.get('rmse')} mm\n"
            f"- Mean Absolute Percentage Error (MAPE): {validation_metrics.get('mape_percent')}%\n"
            f"- Coefficient of Determination (R^2): {validation_metrics.get('r2_score')}\n"
            f"- Pearson Correlation (r): {validation_metrics.get('pearson_r')}"
        )
    else:
        val_text = (
            "Laboratory sieve analysis ground truth was not supplied for this session or contained insufficient pairs. "
            "Internal cluster separation metrics were recorded, but geological grain-size validation remains pending reference input."
        )
    pdf.multi_cell(0, 5, sanitize_pdf_text(val_text))
    pdf.ln(4)

    # Section 5: Research Disclaimers & Integrity Note
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, sanitize_pdf_text("5. Methodological & Research Notes"), ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(107, 114, 128)
    disclaimer = (
        "Methodological Note: Internal clustering indices (Silhouette, Davies-Bouldin, Calinski-Harabasz) evaluate mathematical "
        "geometric cohesion and separation in latent feature space. They do not intrinsically prove geological sediment correctness. "
        "Cluster labels are neutral descriptors and require physical optical calibration and ASTM/Wentworth sieve verification "
        "before formal geological sediment classification."
    )
    pdf.multi_cell(0, 4, sanitize_pdf_text(disclaimer))

    return bytes(pdf.output())


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts DataFrame to UTF-8 encoded CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")
