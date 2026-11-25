# BEACHMIND – Technical Documentation & Mathematical Formulations
## SIH26054: PCA-Based Feature Reduction of Camera-Derived Grain-Size Data for Beach Sediment Mapping

---

## 1. Architectural Philosophy: Strictly Classical Computer Vision & Machine Learning

In accordance with the Smart India Hackathon guidelines for problem statement **SIH26054**, this system operates strictly on **interpretable, classical, non-deep-learning algorithms**:
- **Zero Deep Learning**: No CNNs, ResNets, Vision Transformers, transfer learning, or black-box neural networks are utilized.
- **Deterministic & Interpretable**: Every extracted dimension corresponds directly to verifiable physical or mathematical properties (e.g., co-occurrence texture entropy, harmonic spatial frequency, geometrical circularity, pixel aspect ratios).
- **Lightweight & Laptop Compatible**: Highly optimized for low-latency CPU inference on standard laptops with 16 GB RAM.

---

## 2. Image Preprocessing Pipeline

### 2.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)
Optical beach imagery suffers from non-uniform natural illumination (sun glare, cloud shadows, varying moisture content). Global histogram equalization amplifies background noise in flat regions. CLAHE mitigates this by:
1. Dividing the image into small contextual rectangular tiles (default: $8 \times 8$).
2. Computing the local histogram for each grid tile.
3. Clipping histogram bin counts that exceed a preset threshold $\beta$ (Clip Limit = 2.0).
4. Redistributing clipped probability density uniformly across all gray levels.
5. Performing bilinear interpolation across tile boundaries to eliminate artificial block borders.

### 2.2 Spatial Noise Suppression
- **Gaussian Smoothing**:
  $$G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
  Suppresses high-frequency sensor noise while preserving macroscopic grain contours.
- **Median / Bilateral Filtering**: Removes impulse salt-and-pepper noise while strictly preserving sharp sediment grain boundaries.

### 2.3 Contour Segmentation and Quality Index
- **Adaptive Thresholding**:
  $$T(x, y) = \text{mean}(N(x, y)) - C$$
  Threshold adapts to local neighborhood $N(x, y)$, isolating individual grains even under variable illumination.
- **Morphological Operations**:
  $$\text{Opening: } A \circ B = (A \ominus B) \oplus B$$
  $$\text{Closing: } A \bullet B = (A \oplus B) \ominus B$$
  Removes spurious micro-pixel noise and seals minor internal holes in grain contours.
- **Quality Check**: Computes the valid grain area ratio and contour count to ensure valid segmentation and avoid illumination artifacts.

---

## 3. Feature Extraction Formulations

### 3.1 Gray Level Co-occurrence Matrix (GLCM)
The GLCM matrix $P(i, j | d, \theta)$ tabulates the joint probability of gray levels $i$ and $j$ occurring at distance $d$ and angle $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.

1. **Contrast**:
   $$\sum_{i,j} |i - j|^2 P(i, j)$$
   Measures local intensity variations and micro-scale grain shadow contrast.

2. **Dissimilarity**:
   $$\sum_{i,j} |i - j| P(i, j)$$

3. **Homogeneity**:
   $$\sum_{i,j} \frac{P(i, j)}{1 + (i - j)^2}$$
   Higher for smooth, well-sorted fine sand; lower for heterogeneous coarse gravel.

4. **Energy (Uniformity)**:
   $$\sum_{i,j} P(i, j)^2$$

5. **Angular Second Moment (ASM)**:
   $$\text{ASM} = \text{Energy} = \sum_{i,j} P(i, j)^2$$

6. **Correlation**:
   $$\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$$

---

### 3.2 Local Binary Patterns (LBP)
Computes micro-texture representations invariant to monotonic illumination changes:
$$LBP_{P, R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p, \quad s(x) = \begin{cases} 1 & x \ge 0 \\ 0 & x < 0 \end{cases}$$
We employ **rotation-invariant uniform patterns** with $P=16, R=2.0$, constructing a compact histogram of local micro-surface roughness.

---

### 3.3 2D Gabor Filter Bank
Models multi-scale spatial frequency and directional grain orientation:
$$g(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right)$$
where $x' = x\cos\theta + y\sin\theta$ and $y' = -x\sin\theta + y\cos\theta$.
Evaluated across 4 orientations ($\theta \in \{0, \pi/4, \pi/2, 3\pi/4\}$) and 3 frequencies ($\lambda^{-1} \in \{0.08, 0.16, 0.32\}$).

---

### 3.4 Morphological Particle Characterization
When individual grains are segmented:
1. **Equivalent Circular Diameter**:
   $$D_{eq} = 2 \sqrt{\frac{\text{Area}}{\pi}}$$
2. **Circularity (Shape Factor)**:
   $$C = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2} \quad (C \le 1.0)$$
3. **Aspect Ratio**:
   $$\text{AR} = \frac{\text{Major Axis}}{\text{Minor Axis}}$$
4. **Grain Size Percentiles**: $D_{10}, D_{50}$ (Median), $D_{90}$, and Trask Sorting Index $S_0 = \sqrt{D_{90}/D_{10}}$.

---

## 4. Dimensionality Reduction: Principal Component Analysis (PCA)

### 4.1 Feature Standardization (StandardScaler)
Because features span drastically different dimensions and ranges (Area $\sim 10^4 \text{ px}^2$, GLCM $\in [0, 1]$, RGB $\in [0, 255]$), Z-score standardization is mandatory:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

### 4.2 Covariance and Spectral Decomposition
1. Compute the empirical covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{d \times d}$:
   $$\mathbf{\Sigma} = \frac{1}{n - 1} \mathbf{X}_{std}^T \mathbf{X}_{std}$$
2. Compute eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$ and orthogonal eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_d$:
   $$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
3. Retain $k$ components satisfying the cumulative explained variance threshold:
   $$\frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j} \ge 0.95$$
4. Latent projection: $\mathbf{Z} = \mathbf{X}_{std} \mathbf{W}_k$, where $\mathbf{W}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k]$.

### 4.3 Factor Loadings Analysis
The loading of feature $j$ on component $i$ is:
$$L_{ji} = v_{ji} \sqrt{\lambda_i}$$
This equals the Pearson correlation between original feature $j$ and principal component $i$.

---

## 5. Clustering Algorithms

### 5.1 K-Means++
- **Initialization**: Selects the first centroid randomly; subsequent centroids are chosen with probability proportional to the squared distance from the closest existing centroid:
  $$P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$$
- **Objective (WCSS Minimization)**:
  $$\arg\min_{\mathbf{S}} \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$$

### 5.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- For each point $p$, defines its $\varepsilon$-neighborhood:
  $$N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \varepsilon\}$$
- Core points: $|N_\varepsilon(p)| \ge \text{MinPts}$.
- Directly density-reachable and density-connected points form clusters. Points not reachable from any core point are flagged as **Noise (-1)**.

---

## 6. Evaluation and Ground-Truth Validation Metrics

### 6.1 Clustering Metrics
1. **Silhouette Coefficient**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, +1]$$
   where $a(i)$ is mean intra-cluster distance, and $b(i)$ is mean nearest-cluster distance.
2. **Davies-Bouldin Index**:
   $$DB = \frac{1}{k} \sum_{i=1}^k \max_{j \ne i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)$$
   Lower indicates tighter, better-separated clusters.
3. **Calinski-Harabasz Index (Variance Ratio)**:
   $$CH = \frac{\text{Tr}(\mathbf{B}_k) / (k - 1)}{\text{Tr}(\mathbf{W}_k) / (n - k)}$$
   Higher indicates superior cluster separation.

### 6.2 Mechanical Sieve Regression Metrics
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_{\text{pred}, i} - y_{\text{true}, i}|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_{\text{pred}, i} - y_{\text{true}, i})^2}$$
- **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_{\text{pred}, i} - y_{\text{true}, i}}{y_{\text{true}, i}}\right|$$
- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_i (y_{\text{true}, i} - y_{\text{pred}, i})^2}{\sum_i (y_{\text{true}, i} - \bar{y}_{\text{true}})^2}$$
