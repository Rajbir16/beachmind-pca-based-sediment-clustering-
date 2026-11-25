"""
Dataset Management and Demo Subsystem
Part of BEACHMIND (SIH26054)

Handles:
- Upload of individual images, batch images, ZIP archives
- Metadata CSV parsing and alignment (e.g., SandSnap format or standard beach registry)
- Dataset Registry maintenance:
  * Image ID
  * Image filename
  * Dataset source
  * Beach name
  * Sample ID
  * Latitude & Longitude
  * Reference grain diameter (D50)
  * Acquisition conditions
- Transparent Demonstration Mode:
  * Clearly labelled synthetic sediment data
  * Never claims synthetic data are real field measurements
  * Generates procedural sand-like texture images for immediate offline execution
"""

import os
import zipfile
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional


def create_procedural_sand_image(
    grain_size_type: str = "medium",
    width: int = 512,
    height: int = 512,
    random_seed: int = 42
) -> np.ndarray:
    """
    Generates realistic procedural sediment texture images for demonstration mode.
    Simulates optical grain size, color variations, and shading.
    """
    np.random.seed(random_seed)
    
    # Base sediment color: Golden/brownish sand tones
    if grain_size_type == "fine":
        base_color = np.array([215, 195, 150], dtype=np.float32)
        n_grains = 1800
        radius_range = (2, 5)
        noise_level = 25
    elif grain_size_type == "coarse":
        base_color = np.array([190, 160, 120], dtype=np.float32)
        n_grains = 400
        radius_range = (8, 18)
        noise_level = 45
    elif grain_size_type == "gravel":
        base_color = np.array([160, 150, 140], dtype=np.float32)
        n_grains = 150
        radius_range = (16, 32)
        noise_level = 55
    else:  # medium
        base_color = np.array([210, 180, 135], dtype=np.float32)
        n_grains = 900
        radius_range = (4, 9)
        noise_level = 35

    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color

    # Add background micro-roughness
    bg_noise = np.random.normal(0, noise_level * 0.4, (height, width, 3))
    canvas = np.clip(canvas + bg_noise, 0, 255).astype(np.uint8)

    # Draw simulated mineral grains with random shades and specular highlights
    for _ in range(n_grains):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        r = np.random.randint(radius_range[0], radius_range[1] + 1)
        
        # Color variation per grain (quartz, feldspar, dark minerals)
        mineral_roll = np.random.rand()
        if mineral_roll < 0.15:  # Dark heavy minerals (magnetite/ilmenite)
            g_color = (np.random.randint(40, 90), np.random.randint(40, 80), np.random.randint(40, 80))
        elif mineral_roll > 0.85:  # White translucent quartz / shell
            g_color = (np.random.randint(230, 255), np.random.randint(230, 255), np.random.randint(210, 240))
        else:  # Standard sand
            delta = np.random.randint(-25, 25)
            g_color = (
                int(np.clip(base_color[0] + delta, 0, 255)),
                int(np.clip(base_color[1] + delta, 0, 255)),
                int(np.clip(base_color[2] + delta, 0, 255))
            )

        cv2.circle(canvas, (cx, cy), r, g_color, -1)
        # Small grain shadow for 3D depth
        cv2.circle(canvas, (cx + 1, cy + 1), max(1, r - 1), (max(0, g_color[0]-30), max(0, g_color[1]-30), max(0, g_color[2]-30)), 1)

    # Slight blur to simulate camera depth of field
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    return canvas


def generate_synthetic_demo_dataset() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Constructs a labeled synthetic demonstration dataset with 12 distinct beach sediment samples.
    Simulates three sediment facies (Fine Sand, Medium Sand, Coarse Gravelly Sand)
    along a realistic coastal transect with genuine coordinates, reference sieve D50, and image arrays.
    """
    types = [
        ("fine", "Marina Beach North", 13.0620, 80.2825, 0.165, "Fine Sand Segment", 25.0),
        ("fine", "Marina Beach Central", 13.0550, 80.2830, 0.180, "Fine Sand Segment", 25.0),
        ("fine", "Marina Beach South", 13.0480, 80.2835, 0.195, "Fine Sand Segment", 25.0),
        ("fine", "Besant Nagar Beach", 13.0002, 80.2730, 0.210, "Fine Sand Segment", 25.0),
        ("medium", "Covelong Cove A", 12.7915, 80.2520, 0.340, "Medium Sand Segment", 20.0),
        ("medium", "Covelong Cove B", 12.7880, 80.2530, 0.365, "Medium Sand Segment", 20.0),
        ("medium", "Mahabalipuram North", 12.6280, 80.2010, 0.380, "Medium Sand Segment", 20.0),
        ("medium", "Mahabalipuram Shore", 12.6160, 80.1980, 0.410, "Medium Sand Segment", 20.0),
        ("coarse", "Sadras Coastal Transect 1", 12.5200, 80.1650, 0.720, "Coarse Sand Segment", 15.0),
        ("coarse", "Sadras Coastal Transect 2", 12.5150, 80.1630, 0.850, "Coarse Sand Segment", 15.0),
        ("gravel", "Pondicherry Rocky Shore 1", 11.9320, 79.8350, 1.450, "Gravelly Sand Segment", 12.0),
        ("gravel", "Pondicherry Rocky Shore 2", 11.9250, 79.8330, 1.820, "Gravelly Sand Segment", 12.0),
    ]

    registry_rows = []
    images_dict = {}

    for i, (g_type, beach, lat, lon, ref_d50, facies, calib_ratio) in enumerate(types):
        img_id = f"DEMO_IMG_{i+1:03d}"
        fname = f"synthetic_{g_type}_{i+1:03d}.jpg"
        
        img_arr = create_procedural_sand_image(grain_size_type=g_type, random_seed=100 + i)
        images_dict[img_id] = img_arr

        registry_rows.append({
            "image_id": img_id,
            "filename": fname,
            "dataset_source": "DEMO (Synthetic Beach Sand - Procedurally Generated)",
            "beach_name": beach,
            "sample_id": f"SMP-{i+1:03d}",
            "latitude": lat,
            "longitude": lon,
            "reference_d50": ref_d50,
            "grain_type_label": g_type,
            "coastal_facies": facies,
            "calibration_mm_per_pixel": calib_ratio / 512.0,  # e.g. ~0.04 mm/px
            "acquisition_conditions": "Simulated normal sunlight, orthogonal close-up macro lens",
            "is_synthetic": True
        })

    registry_df = pd.DataFrame(registry_rows)
    return registry_df, images_dict


def parse_uploaded_zip_file(zip_bytes: bytes) -> List[Tuple[str, str, bytes]]:
    """
    Extracts all supported image files (jpg, jpeg, png, tif) from an uploaded zip archive.
    Returns: [(image_id, filename, image_bytes), ...]
    """
    extracted_items = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        idx = 1
        for member in z.namelist():
            # Skip folders or Mac OS __MACOSX resource forks
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue
            ext = os.path.splitext(member)[1].lower()
            if ext in valid_exts:
                fname = os.path.basename(member)
                img_data = z.read(member)
                img_id = f"ZIP_IMG_{idx:03d}"
                extracted_items.append((img_id, fname, img_data))
                idx += 1

    return extracted_items


def align_metadata_with_images(
    images_list: List[Tuple[str, str, Any]],
    metadata_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Aligns user-uploaded metadata CSV with image filenames or creates a clean standardized registry.
    """
    records = []

    # Map metadata by filename if available
    meta_lookup = {}
    if metadata_df is not None and not metadata_df.empty:
        # Check potential filename column
        fname_col = next((c for c in metadata_df.columns if c.lower() in ["filename", "image_name", "file", "img", "id"]), None)
        if fname_col:
            for _, r in metadata_df.iterrows():
                key = str(r[fname_col]).strip()
                meta_lookup[key] = r.to_dict()

    for img_id, fname, img_src in images_list:
        base_record = {
            "image_id": img_id,
            "filename": fname,
            "dataset_source": "User Uploaded Session",
            "beach_name": "Unspecified Coastal Location",
            "sample_id": img_id,
            "latitude": np.nan,
            "longitude": np.nan,
            "reference_d50": np.nan,
            "acquisition_conditions": "Camera Close-up",
            "is_synthetic": False
        }

        # Match against metadata row
        match = meta_lookup.get(fname) or meta_lookup.get(os.path.splitext(fname)[0])
        if match:
            for k, v in match.items():
                k_clean = k.lower().strip()
                if "beach" in k_clean:
                    base_record["beach_name"] = str(v)
                elif "lat" in k_clean:
                    base_record["latitude"] = pd.to_numeric(v, errors="coerce")
                elif "lon" in k_clean or "lng" in k_clean:
                    base_record["longitude"] = pd.to_numeric(v, errors="coerce")
                elif "d50" in k_clean or "grain_size" in k_clean or "sieve" in k_clean:
                    base_record["reference_d50"] = pd.to_numeric(v, errors="coerce")
                elif "sample" in k_clean:
                    base_record["sample_id"] = str(v)
                elif "condition" in k_clean or "light" in k_clean:
                    base_record["acquisition_conditions"] = str(v)

        records.append(base_record)

    return pd.DataFrame(records)
