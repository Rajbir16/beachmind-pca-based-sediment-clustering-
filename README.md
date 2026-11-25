# BEACHMIND – Intelligent Beach Sediment Mapping
### PCA-Based Feature Reduction of Camera-Derived Grain-Size Data for Beach Sediment Mapping
**Smart India Hackathon Problem Statement SIH26054**

---

## 📌 Executive Overview

**BEACHMIND** is a research-grade Computer Vision and Machine Learning web application designed to characterize beach sediments directly from close-up digital optical photographs and digital camera imagery. 

Traditional sediment analysis relies on laboratory mechanical sieve analysis (ASTM D422), which is destructive, labour-intensive, expensive, and spatially sparse. BEACHMIND extracts multidimensional texture, micro-pattern, spatial frequency, color, and grain morphological features using **purely classical Computer Vision and Machine Learning algorithms (strictly zero Deep Learning, CNNs, or pretrained models)**. High-dimensional feature matrices are compressed using **Principal Component Analysis (PCA)** to eliminate multicollinearity and isolate dominant orthogonal variance axes, followed by unsupervised clustering (**K-Means++** and **DBSCAN**) and interactive geospatial coastal mapping.

---

## ⚙️ Technology Stack

| Layer | Technologies Used |
|---|---|
| **Programming Language** | Python 3.11+ / 3.13 |
| **Web Dashboard** | Streamlit (Multi-page scientific dashboard) |
| **Computer Vision** | OpenCV (`cv2`), scikit-image (`skimage`) |
| **Dimensionality Reduction & ML** | scikit-learn (`StandardScaler`, `PCA`, `KMeans`, `DBSCAN`) |
| **Scientific Numerical Computing** | NumPy, SciPy, Pandas |
| **Visualization & Plots** | Plotly Express, Plotly Graph Objects, Matplotlib |
| **Geospatial Mapping** | Folium, `streamlit-folium` |
| **Model Serialization & Reports** | Joblib, `fpdf2` (Multi-page scientific PDF generation) |

---

## 🏗️ System Architecture

```
SIH26054/
│
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & execution instructions
├── TECHNICAL_DOCUMENTATION.md  # Detailed mathematical formulations
├── VIVA_PREPARATION.md         # SIH Defense and viva interview guide
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py        # OpenCV CLAHE, noise removal, edge detection, grain segmentation
│   ├── grain_analysis.py       # Morphology, Deq, circularity, aspect ratio, D10/D50/D90
│   ├── feature_extraction.py   # GLCM, LBP, Gabor filter bank, color moments, grayscale statistics
│   ├── dimensionality_reduction.py # StandardScaler, PCA, Scree plots, loadings, PC-grain correlation
│   ├── clustering.py           # K-Means++, DBSCAN, Elbow analysis, descriptive summaries
│   ├── evaluation.py           # Four-way benchmark matrix, PCA component sensitivity testing
│   ├── validation.py           # Mechanical sieve ground-truth validation (MAE, RMSE, MAPE, R²)
│   ├── mapping.py              # Interactive Folium coastal map with cluster markers
│   ├── report_generator.py     # PDF research report compiler & CSV data export
│   ├── dataset_manager.py      # Image ingestion (Single, Multiple, ZIP), metadata alignment & demo mode
│   └── ui_common.py            # Streamlit theme, session state, navigation widgets
│
├── pages/
│   ├── 1_Overview.py           # Dataset stats, project intro, registry table, workflow
│   ├── 2_Image_Processing.py   # Upload (Single, Batch, ZIP), preprocessing inspector, feature table
│   ├── 3_PCA_Analysis.py       # Scree plot, cumulative variance, 2D/3D projections, loadings
│   ├── 4_Clustering.py         # K-Means++ and DBSCAN across Raw & PCA space, gallery
│   ├── 5_Model_Comparison.py   # Four-way benchmark matrix & component selection sensitivity
│   ├── 6_Validation.py         # Mechanical sieve ground truth comparison & residual plots
│   ├── 7_Sediment_Mapping.py   # Folium interactive coastal map with cluster filtering
│   └── 8_Reports.py            # Download full PDF research report & structured CSVs
│
├── dataset/                    # Image storage and metadata registries
│   ├── images/
│   ├── metadata/
│   └── processed/
│
├── tests/
│   └── test_pipeline.py        # Automated end-to-end integration test suite
│
├── models/                     # Saved joblib scalers & PCA objects
└── outputs/                    # Exported CSVs and generated PDF reports
```

---

## 🚀 Installation and Execution (Windows PowerShell)

### 1. Clone or Open Workspace
Open Windows PowerShell in the project directory:
```powershell
cd "c:\Users\rajbi\OneDrive\Projects\PCA based feature reduction of grain size data for beach sediment mapping"
```

### 2. Set Up Virtual Environment (Recommended)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Verify that all algorithms, OpenCV routines, PCA, and clustering components pass:
```powershell
python tests\test_pipeline.py
```

### 5. Launch the Web Application
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🔬 Core Scientific Modules

### 1. Multi-Stage Computer Vision Preprocessing
- **Illumination Normalization**: Contrast Limited Adaptive Histogram Equalization (**CLAHE**) balances uneven coastal sunlight.
- **Noise Suppression**: Gaussian, Median, or Bilateral spatial filtering.
- **Contour Segmentation & Morphometrics**: Otsu/Adaptive Gaussian thresholding and morphological opening/closing isolate individual sediment grains.
- **Segmentation Quality Verification**: Assesses boundary regularity and grain coverage ratio ($15\% - 85\%$) to guard against false-positive contours.

### 2. Comprehensive Feature Extraction
Extracts 50+ quantitative features:
- **GLCM Texture**: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, Angular Second Moment (ASM) computed at 4 orientations ($\theta \in \{0, \pi/4, \pi/2, 3\pi/4\}$).
- **Local Binary Patterns (LBP)**: Rotation-invariant uniform histograms ($P=16, R=2$).
- **Gabor Filter Bank**: Spatial frequency responses across 4 orientations and 3 wavelengths.
- **Sediment Color**: Channel means, standard deviations, and histograms across RGB and HSV spaces.
- **Morphological Particle Distribution**: Equivalent diameter ($D_{eq} = 2\sqrt{A/\pi}$), circularity ($4\pi A / P^2$), aspect ratio ($a/b$), and percentiles ($D_{10}, D_{50}, D_{90}$).
- **Grayscale Statistical Moments**: Mean, median, variance, skewness, and kurtosis.

### 3. Feature Scaling & Principal Component Analysis (PCA)
- **StandardScaler**: Enforces zero mean and unit variance ($z = (x - \mu)/\sigma$) to prevent scale dominance.
- **PCA Dimensionality Reduction**: Retains 95% cumulative explained variance while reducing dimensionality by $70\% - 90\%$.
- **Scree & Loadings Analysis**: Eigenvalues evaluated against Kaiser's criterion ($\lambda \ge 1.0$); factor loadings identify dominant physical feature drivers.
- **Grain-Size Relevance Verification**: Empirical Pearson correlation checks between individual PCs and reference laboratory $D_{50}$ to guarantee the retention of grain-size information.

### 4. Dual Unsupervised Clustering
- **K-Means++**: Fast partition clustering with Elbow method (WCSS) and Silhouette profile optimization.
- **DBSCAN**: Density-based spatial clustering capable of discovering arbitrary cluster shapes and isolating anomalous noise points ($-1$).
- **Neutral Naming Policy**: Adheres to scientific integrity; cluster numbers are designated with neutral labels (*Sediment Cluster 0*) unless calibrated with physical scale (mm/px) and verified against Udden-Wentworth standards.

### 5. Four-Way Controlled Benchmark Matrix
Compares:
1. *Standardized Features + K-Means*
2. *PCA-Reduced Features + K-Means*
3. *Standardized Features + DBSCAN*
4. *PCA-Reduced Features + DBSCAN*

Evaluated on:
- Silhouette Coefficient
- Davies-Bouldin Index (Lower is Better)
- Calinski-Harabasz Index (Higher is Better)
- Execution Latency (ms)
- Dimensionality Reduction Ratio (%)

### 6. Mechanical Sieve Analysis Ground-Truth Validation
For datasets paired with laboratory sieve tests (e.g., SandSnap), calculates:
- Mean Absolute Error (**MAE**)
- Root Mean Squared Error (**RMSE**)
- Mean Absolute Percentage Error (**MAPE**)
- Coefficient of Determination (**$R^2$**)
- Pearson Correlation Coefficient (**$r$**)
- Residual Plots ($e_i = y_{\text{pred}, i} - y_{\text{true}, i}$)

### 7. Interactive Geospatial Coastal Mapping
- Interactive **Folium** maps rendered through `streamlit-folium`.
- Color-coded markers mapped to cluster assignments.
- Rich popup cards displaying Image ID, Beach name, Coordinates, Optical $D_{50}$, and Reference Sieve measurement.
- **Strict Coordinate Policy**: Refuses to fabricate coordinates; gracefully disables mapping if coordinates are missing.

---

## 📊 Dataset Ingestion & Demonstration Mode
The application supports:
1. **Multiple Image Upload**: Upload hundreds of images simultaneously.
2. **ZIP Archives**: Extracts and registers image contents automatically.
3. **Metadata CSV**: Ingests SandSnap or custom registries containing geographic coordinates and sieve $D_{50}$.
4. **Built-in Demonstration Mode**: 12 procedurally generated synthetic beach sand samples with authentic sediment textures, coordinates along the Tamil Nadu coastline, and laboratory sieve benchmarks.
