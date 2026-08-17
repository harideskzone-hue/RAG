import cv2
import numpy as np
from typing import Dict, Any

class CropQualitySelector:
    """
    Evaluates a crop to determine if it meets the minimum quality thresholds
    for generating a persistent canonical identity embedding.
    """
    def __init__(self, 
                 min_width: int = 32, 
                 min_height: int = 64,
                 min_laplacian_var: float = 30.0,
                 brightness_range: tuple = (30, 230)):
        self.min_width = min_width
        self.min_height = min_height
        self.min_laplacian_var = min_laplacian_var
        self.brightness_range = brightness_range

    def assess_quality(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Assess crop quality based on blur, size, and lighting.
        Occlusion and truncation are assumed to be handled partially by bbox shape/confidence,
        but advanced occlusion detection requires semantic segmentation (omitted for Phase 3).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {"approved": False, "reason": "empty_crop"}
            
        h, w = crop_bgr.shape[:2]
        if w < self.min_width or h < self.min_height:
            return {"approved": False, "reason": f"too_small_{w}x{h}"}

        # Check blur using Laplacian variance
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < self.min_laplacian_var:
            return {"approved": False, "reason": "too_blurry", "score": float(blur_val)}

        # Check lighting (brightness)
        mean_brightness = np.mean(gray)
        if mean_brightness < self.brightness_range[0]:
            return {"approved": False, "reason": "too_dark", "score": float(mean_brightness)}
        if mean_brightness > self.brightness_range[1]:
            return {"approved": False, "reason": "too_bright", "score": float(mean_brightness)}

        return {
            "approved": True, 
            "score": float(blur_val),  # higher is generally sharper
            "reason": "passed_quality_checks"
        }
