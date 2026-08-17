import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class CropQualitySelector:
    """
    Evaluates detected person crops to ensure only high-quality, sharp,
    well-lit, unoccluded frames with visible upper-body/face are accepted
    for Re-ID feature extraction and stored in canonical person galleries.
    """

    def __init__(
        self,
        min_width: int = 48,
        min_height: int = 96,
        min_area: int = 4800,
        min_laplacian_var: float = 60.0,
        min_head_laplacian_var: float = 40.0,
        min_contrast_std: float = 22.0,
        brightness_range: Tuple[int, int] = (35, 225),
        aspect_ratio_range: Tuple[float, float] = (1.2, 4.5),
        min_quality_score: float = 0.50,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.min_laplacian_var = min_laplacian_var
        self.min_head_laplacian_var = min_head_laplacian_var
        self.min_contrast_std = min_contrast_std
        self.brightness_range = brightness_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_quality_score = min_quality_score

    def assess_quality(self, crop_bgr: np.ndarray, bbox_in_frame: Optional[Tuple[int, int, int, int]] = None, frame_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Performs multi-factor quality inspection:
        1. Dimension & Area verification (rejects tiny pixelated crops).
        2. Aspect ratio verification (rejects abnormal non-pedestrian slices).
        3. Full-body Sharpness (Laplacian variance motion-blur check).
        4. Upper-Body / Head / Face Structural Visibility (checks upper 30% for texture & contrast).
        5. Lighting, Contrast & Dynamic Range (rejects dark silhouettes and washed out crops).
        6. Frame Boundary Clipping (penalizes bboxes clipped at image edges).
        """
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return {"approved": False, "reason": "empty_crop", "score": 0.0}

        h, w = crop_bgr.shape[:2]
        area = h * w

        # 1. Dimension & Area Check
        if w < self.min_width or h < self.min_height or area < self.min_area:
            return {
                "approved": False,
                "reason": f"too_small_{w}x{h}_area_{area}",
                "score": round(min(1.0, area / float(self.min_area * 2)) * 0.3, 3)
            }

        # 2. Pedestrian Aspect Ratio Check (height / width)
        aspect_ratio = float(h) / float(max(1, w))
        if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
            return {
                "approved": False,
                "reason": f"invalid_aspect_ratio_{aspect_ratio:.2f}",
                "score": 0.2
            }

        # 3. Grayscale conversion & Sharpness Evaluation
        if crop_bgr.ndim == 3 and crop_bgr.shape[2] == 3:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        elif crop_bgr.ndim == 3 and crop_bgr.shape[2] == 4:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            gray = crop_bgr.copy()

        full_lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if full_lap_var < self.min_laplacian_var:
            return {
                "approved": False,
                "reason": f"too_blurry_laplacian_{full_lap_var:.1f}",
                "score": round(min(1.0, full_lap_var / self.min_laplacian_var) * 0.4, 3)
            }

        # 4. Upper-Body / Head / Face Visibility & Contrast Check
        head_h = max(10, int(h * 0.30))
        head_region = gray[0:head_h, :]
        head_lap_var = float(cv2.Laplacian(head_region, cv2.CV_64F).var()) if head_region.size > 0 else 0.0
        head_std = float(np.std(head_region)) if head_region.size > 0 else 0.0

        if head_lap_var < self.min_head_laplacian_var:
            return {
                "approved": False,
                "reason": f"blurred_upper_body_laplacian_{head_lap_var:.1f}",
                "score": round(min(1.0, head_lap_var / self.min_head_laplacian_var) * 0.45, 3)
            }

        if head_std < 14.0:
            return {
                "approved": False,
                "reason": f"occluded_or_featureless_head_std_{head_std:.1f}",
                "score": 0.35
            }

        # 5. Lighting & Contrast (Dynamic Range)
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        if mean_brightness < self.brightness_range[0]:
            return {"approved": False, "reason": f"too_dark_{mean_brightness:.1f}", "score": 0.25}
        if mean_brightness > self.brightness_range[1]:
            return {"approved": False, "reason": f"too_bright_{mean_brightness:.1f}", "score": 0.25}
        if contrast_std < self.min_contrast_std:
            return {"approved": False, "reason": f"low_contrast_{contrast_std:.1f}", "score": 0.35}

        # 6. Edge Clipping Penalties (if bbox & frame dimensions provided)
        edge_penalty = 0.0
        if bbox_in_frame is not None and frame_size is not None:
            fh, fw = frame_size
            x1, y1, x2, y2 = bbox_in_frame
            clipped_edges = 0
            if x1 <= 2: clipped_edges += 1
            if y1 <= 2: clipped_edges += 1
            if x2 >= fw - 3: clipped_edges += 1
            if y2 >= fh - 3: clipped_edges += 1
            if clipped_edges >= 2:
                edge_penalty = 0.20
            elif clipped_edges == 1:
                edge_penalty = 0.08

        # 7. Compute Unified Continuous Quality Score [0.0 - 1.0]
        # Sharpness score (normalized against 150.0 reference)
        s_score = min(1.0, full_lap_var / 150.0)
        # Head clarity score (normalized against 100.0 reference)
        h_score = min(1.0, head_lap_var / 100.0)
        # Resolution score (normalized against 128x256 reference)
        r_score = min(1.0, area / (128.0 * 256.0))
        # Contrast score (normalized against 60.0 reference)
        c_score = min(1.0, contrast_std / 60.0)

        raw_score = (0.35 * s_score) + (0.25 * h_score) + (0.20 * r_score) + (0.20 * c_score) - edge_penalty
        final_score = round(max(0.0, min(1.0, raw_score)), 3)

        approved = final_score >= self.min_quality_score

        return {
            "approved": approved,
            "score": final_score,
            "metrics": {
                "width": w,
                "height": h,
                "aspect_ratio": round(aspect_ratio, 2),
                "full_laplacian_var": round(full_lap_var, 1),
                "head_laplacian_var": round(head_lap_var, 1),
                "contrast_std": round(contrast_std, 1),
                "mean_brightness": round(mean_brightness, 1),
            },
            "reason": "passed_quality_checks" if approved else "score_below_threshold"
        }

    @staticmethod
    def are_duplicate_crops(crop1: np.ndarray, crop2: np.ndarray, similarity_threshold: float = 0.88) -> bool:
        """
        Determines whether two crops are near-duplicates using normalized cross-correlation
        on normalized grayscale thumbnails.
        """
        if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0:
            return False

        try:
            g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY) if crop1.ndim == 3 else crop1
            g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY) if crop2.ndim == 3 else crop2

            # Resize both to common thumbnail for fast structural comparison
            t1 = cv2.resize(g1, (64, 128)).astype(np.float32)
            t2 = cv2.resize(g2, (64, 128)).astype(np.float32)

            t1_norm = (t1 - np.mean(t1)) / (np.std(t1) + 1e-6)
            t2_norm = (t2 - np.mean(t2)) / (np.std(t2) + 1e-6)

            corr = float(np.mean(t1_norm * t2_norm))
            return corr >= similarity_threshold
        except Exception:
            return False

