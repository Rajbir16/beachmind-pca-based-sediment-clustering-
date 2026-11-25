"""
Preprocessing Pipeline Module for BEACHMIND (SIH26054)
Implements Computer Vision algorithms using OpenCV, NumPy, and SciPy:
- Image loading & color space transformations (RGB, Gray, HSV)
- Robust decoding for Streamlit UploadedFile, bytes, PIL Images, and file paths
- Noise removal (Gaussian, Median, Bilateral)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Normalization [0, 1] and [0, 255]
- Edge detection (Canny & Sobel)
- Mathematical morphology (Opening, Closing, Top-hat)
- Grain boundary segmentation & quality metrics
"""

import io
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List


def load_image_to_rgb(image_source: Any) -> np.ndarray:
    """
    Safely loads and decodes an image from:
    - Streamlit UploadedFile (via .getvalue())
    - raw bytes / bytearray
    - io.BytesIO or file-like streams
    - filepath string
    - PIL.Image.Image
    - existing numpy array (2D or 3D)

    Returns:
        uint8 RGB numpy array of shape (H, W, 3)
    Raises:
        ValueError if decoding fails or dimensions are invalid.
        TypeError if input type is unsupported.
    """
    if image_source is None:
        raise ValueError("Image source is None.")

    # 1. Existing numpy array
    if isinstance(image_source, np.ndarray):
        if image_source.size == 0 or image_source.ndim < 2:
            raise ValueError(f"Invalid numpy array with shape {image_source.shape}")
        
        # Ensure uint8
        if image_source.dtype != np.uint8:
            if np.issubdtype(image_source.dtype, np.floating):
                if image_source.max() <= 1.0:
                    arr = np.clip(image_source * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(image_source, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(image_source, 0, 255).astype(np.uint8)
        else:
            arr = image_source.copy()

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif channels == 3:
                return arr
            elif channels == 1:
                return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Unsupported number of image channels: {channels}")
        else:
            raise ValueError(f"Unsupported image array dimensions: {arr.ndim}")

    raw_bytes: Optional[bytes] = None

    # 2. Streamlit UploadedFile object (has .getvalue())
    if hasattr(image_source, "getvalue"):
        try:
            raw_bytes = image_source.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to read bytes from uploaded file via getvalue(): {e}")

    # 3. Raw bytes or bytearray
    elif isinstance(image_source, (bytes, bytearray)):
        raw_bytes = bytes(image_source)

    # 4. File-like stream (io.BytesIO or open file handle)
    elif hasattr(image_source, "read"):
        if hasattr(image_source, "seek"):
            try:
                image_source.seek(0)
            except Exception:
                pass
        raw_bytes = image_source.read()
        if hasattr(image_source, "seek"):
            try:
                image_source.seek(0)
            except Exception:
                pass

    # 5. Filepath string
    elif isinstance(image_source, str):
        img_bgr = cv2.imread(image_source, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback to PIL in case of unicode path or special JPEG variant
            try:
                pil_img = Image.open(image_source).convert("RGB")
                return np.array(pil_img, dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Failed to read image from path '{image_source}': {e}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 6. PIL Image
    elif isinstance(image_source, Image.Image):
        pil_rgb = image_source.convert("RGB")
        return np.array(pil_rgb, dtype=np.uint8)

    else:
        raise TypeError(f"Unsupported image input type: {type(image_source).__name__}")

    # If raw_bytes were extracted, decode via OpenCV and fallback to PIL
    if raw_bytes is not None:
        if len(raw_bytes) == 0:
            raise ValueError("Image byte buffer is empty (0 bytes).")

        nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is not None and img_bgr.size > 0:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Fallback to PIL (handles progressive JPEGs, CMYK, WebP, EXIF orientations)
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            decoded_arr = np.array(pil_img, dtype=np.uint8)
            if decoded_arr.size > 0:
                return decoded_arr
        except Exception as e:
            raise ValueError(f"OpenCV and PIL both failed to decode image buffer ({len(raw_bytes)} bytes): {e}")

        raise ValueError(f"Could not decode valid image pixels from buffer ({len(raw_bytes)} bytes).")

    raise ValueError("Failed to obtain valid image bytes from source.")


def preprocess_image_pipeline(
    image_rgb: np.ndarray,
    target_size: Optional[Tuple[int, int]] = (512, 512),
    denoise_method: str = "gaussian",
    kernel_size: int = 5,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    canny_thresh1: int = 50,
    canny_thresh2: int = 150
) -> Dict[str, Any]:
    """
    Executes complete multi-stage CV preprocessing pipeline.
    Validates image_rgb before proceeding.
    Returns structured dictionary containing all intermediate representations.
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"preprocess_image_pipeline requires a numpy.ndarray, got {type(image_rgb).__name__}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"preprocess_image_pipeline expects a 3-channel RGB image (H, W, 3), got shape {image_rgb.shape}")
    if image_rgb.shape[0] < 5 or image_rgb.shape[1] < 5:
        raise ValueError(f"Image dimensions too small to preprocess: {image_rgb.shape[:2]}")

    # 1. Resize if requested (maintaining high interpolation quality)
    if target_size is not None and (image_rgb.shape[1], image_rgb.shape[0]) != target_size:
        resized_rgb = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    else:
        resized_rgb = image_rgb.copy()

    # 2. Color conversions
    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)

    # 3. Noise removal
    if denoise_method.lower() == "median":
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        denoised_gray = cv2.medianBlur(gray, k)
    elif denoise_method.lower() == "bilateral":
        denoised_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    else:  # default Gaussian
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        denoised_gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 4. Contrast Enhancement via CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        clahe_gray = clahe.apply(denoised_gray)
    else:
        clahe_gray = denoised_gray.copy()

    # 5. Normalization
    normalized_float = clahe_gray.astype(np.float32) / 255.0
    normalized_uint8 = cv2.normalize(clahe_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 6. Edge Detection
    edges_canny = cv2.Canny(clahe_gray, canny_thresh1, canny_thresh2)
    sobelx = cv2.Sobel(clahe_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(clahe_gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))

    # 7. Morphological operations (Morphological Gradient & Opening)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_opened = cv2.morphologyEx(clahe_gray, cv2.MORPH_OPEN, morph_kernel)
    morph_gradient = cv2.morphologyEx(clahe_gray, cv2.MORPH_GRADIENT, morph_kernel)

    return {
        "original_rgb": image_rgb,
        "resized_rgb": resized_rgb,
        "gray": gray,
        "hsv": hsv,
        "denoised_gray": denoised_gray,
        "clahe_gray": clahe_gray,
        "normalized_float": normalized_float,
        "normalized_uint8": normalized_uint8,
        "edges_canny": edges_canny,
        "edges_sobel": edges_sobel,
        "morph_opened": morph_opened,
        "morph_gradient": morph_gradient,
    }


def segment_sediment_grains(
    clahe_gray: np.ndarray,
    method: str = "adaptive",
    min_area: int = 15,
    max_area: int = 25000,
    clean_morphology: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Segments individual or clumped sediment grains from preprocessed grayscale image.
    Calculates segmentation quality indicators (do not assume all contours are clean grains).

    Returns:
        binary_mask: uint8 0/255 mask
        valid_contours: list of contours matching area bounds
        quality_metrics: dict with segmentation quality assessment
    """
    if clahe_gray is None or clahe_gray.ndim != 2:
        raise ValueError("segment_sediment_grains requires a 2D grayscale array.")

    h, w = clahe_gray.shape
    total_pixels = h * w

    if method == "otsu":
        _, thresh = cv2.threshold(clahe_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Adaptive Gaussian
        thresh = cv2.adaptiveThreshold(
            clahe_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )

    if clean_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = closed
    else:
        binary_mask = thresh

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    segmented_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            valid_contours.append(c)
            segmented_area += area

    # Quality Assessment
    grain_coverage_ratio = segmented_area / float(total_pixels) if total_pixels > 0 else 0.0
    detected_count = len(valid_contours)
    
    if detected_count < 5:
        quality_status = "Poor (Under-segmented: too few grains detected)"
        quality_score = 0.2
    elif grain_coverage_ratio > 0.95 or grain_coverage_ratio < 0.05:
        quality_status = "Poor (Illumination threshold artifact or extreme clutter)"
        quality_score = 0.3
    elif detected_count > 4000:
        quality_status = "Moderate (High noise / potential over-fragmentation)"
        quality_score = 0.65
    else:
        quality_status = "Good (Valid sediment grain morphology detected)"
        quality_score = 0.92

    quality_metrics = {
        "total_contours_found": len(contours),
        "valid_grain_contours": detected_count,
        "segmented_area_pixels": float(segmented_area),
        "coverage_ratio": float(grain_coverage_ratio),
        "quality_score": float(quality_score),
        "quality_status": quality_status,
        "is_acceptable": quality_score >= 0.5
    }

    return binary_mask, valid_contours, quality_metrics
