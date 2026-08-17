"""
person_quality.py — Continuous Person Crop Quality Assessment
==============================================================
Evaluates visual quality of person crops for downstream Re-ID feature extraction.
Computes a continuous quality score q in [0.0, 1.0] based on blur, resolution,
edge clipping, and orientation signals.

Key Principles:
- Configurable blur threshold (not hardcoded).
- Front, side, and back facing views are ALL treated as usable for Re-ID.
- Outputs continuous score q for quality-weighted tracklet aggregation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class QualityResult:
    """Dataclass holding detailed quality assessment results."""
    quality_score: float  # Continuous score in [0.0, 1.0]
    is_usable: bool       # True if quality_score >= min_usable_threshold
    blur_variance: float  # Raw Laplacian variance value
    blur_score: float     # Sub-score in [0.0, 1.0]
    resolution_score: float # Sub-score in [0.0, 1.0]
    edge_score: float     # Sub-score in [0.0, 1.0]
    orientation_score: float # Sub-score in [0.0, 1.0]
    penalties: float      # Deductions applied
    violations: List[str] = field(default_factory=list)
    metrics: Dict[str, Union[float, int, bool, List[str]]] = field(default_factory=dict)


class PersonQualityAssessor:
    """
    Continuous Person Quality Assessor for CCTV Person Crops.
    """

    def __init__(
        self,
        blur_threshold: float = 45.0,
        min_height: int = 64,
        min_width: int = 32,
        min_usable_threshold: float = 0.40,
        edge_threshold_pct: float = 0.02,
        w_blur: float = 0.35,
        w_res: float = 0.25,
        w_edge: float = 0.20,
        w_orient: float = 0.20,
    ) -> None:
        """
        Initialize quality assessor with configurable thresholds and weights.

        Args:
            blur_threshold: Reference Laplacian variance threshold (default 45.0)
            min_height: Minimum required crop height in pixels
            min_width: Minimum required crop width in pixels
            min_usable_threshold: Score cutoff for usability (default 0.40)
            edge_threshold_pct: Border proximity ratio to detect edge clipping
            w_blur: Weight for blur score
            w_res: Weight for resolution score
            w_edge: Weight for edge containment score
            w_orient: Weight for orientation / visibility score
        """
        if blur_threshold <= 0:
            raise ValueError("blur_threshold must be strictly positive")
        
        total_w = w_blur + w_res + w_edge + w_orient
        if abs(total_w - 1.0) > 1e-3:
            raise ValueError(f"Weights must sum to 1.0 (got {total_w:.4f})")

        self.blur_threshold = blur_threshold
        self.min_height = min_height
        self.min_width = min_width
        self.min_usable_threshold = min_usable_threshold
        self.edge_threshold_pct = edge_threshold_pct

        self.w_blur = w_blur
        self.w_res = w_res
        self.w_edge = w_edge
        self.w_orient = w_orient

    def assess_crop(
        self,
        crop: np.ndarray,
        bbox_in_frame: Optional[Union[Tuple[int, int, int, int], List[int]]] = None,
        frame_dimensions: Optional[Tuple[int, int]] = None,
        orientation: str = "unknown",
        is_severely_occluded: bool = False,
    ) -> QualityResult:
        """
        Assess crop quality and return continuous QualityResult.

        Args:
            crop: Image crop array (H, W, C)
            bbox_in_frame: Bounding box in original frame [x1, y1, x2, y2]
            frame_dimensions: Original frame dimensions (frame_h, frame_w)
            orientation: Subject facing direction ("front", "side", "back", "unknown")
            is_severely_occluded: Flag for extreme occlusion

        Returns:
            QualityResult object with quality_score in [0.0, 1.0]
        """
        violations: List[str] = []
        penalties: float = 0.0

        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return QualityResult(
                quality_score=0.0,
                is_usable=False,
                blur_variance=0.0,
                blur_score=0.0,
                resolution_score=0.0,
                edge_score=0.0,
                orientation_score=0.0,
                penalties=1.0,
                violations=["empty_crop"],
                metrics={"error": "Crop is empty or invalid"},
            )

        ch, cw = crop.shape[:2]

        # 1. Blur Score Evaluation (Laplacian Variance)
        if crop.ndim == 3 and crop.shape[2] == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        else:
            gray = crop.copy()

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Smooth continuous mapping for blur score
        blur_score = min(1.0, lap_var / self.blur_threshold)
        if lap_var < (0.3 * self.blur_threshold):
            violations.append(f"severe_blur (var={lap_var:.1f})")

        # 2. Resolution & Aspect Ratio Evaluation
        target_area = float(self.min_height * self.min_width * 4)  # e.g., 128x64 = 8192
        crop_area = float(ch * cw)
        res_score = min(1.0, max(0.0, crop_area / target_area))

        if ch < self.min_height or cw < self.min_width:
            violations.append(f"low_resolution ({cw}x{ch})")
            penalties += 0.20

        aspect_ratio = float(ch) / float(max(1, cw))
        if aspect_ratio < 0.8 or aspect_ratio > 5.0:
            violations.append(f"abnormal_aspect_ratio ({aspect_ratio:.2f})")
            penalties += 0.15

        # 3. Frame Edge Containment Evaluation
        edge_score = 1.0
        if bbox_in_frame is not None and frame_dimensions is not None:
            fh, fw = frame_dimensions
            if fh > 0 and fw > 0:
                x1, y1, x2, y2 = bbox_in_frame
                margin_x = int(round(fw * self.edge_threshold_pct))
                margin_y = int(round(fh * self.edge_threshold_pct))

                clipped_edges = 0
                if x1 <= margin_x:
                    clipped_edges += 1
                    violations.append("clipped_left")
                if x2 >= (fw - margin_x):
                    clipped_edges += 1
                    violations.append("clipped_right")
                if y1 <= margin_y:
                    clipped_edges += 1
                    violations.append("clipped_top")

                if clipped_edges == 1:
                    edge_score = 0.70
                elif clipped_edges >= 2:
                    edge_score = 0.30
                    penalties += 0.15

        # 4. Orientation & Visibility Signal
        # Front, side, and back views are ALL treated as valid and usable!
        orient_clean = str(orientation).lower().strip()
        if orient_clean in ("front", "side", "back"):
            orient_score = 1.0
        elif orient_clean == "unknown":
            orient_score = 0.85
        else:
            orient_score = 0.70

        if is_severely_occluded:
            violations.append("severe_occlusion")
            penalties += 0.35
            orient_score *= 0.5

        # Calculate weighted continuous score
        raw_score = (
            self.w_blur * blur_score
            + self.w_res * res_score
            + self.w_edge * edge_score
            + self.w_orient * orient_score
        )

        final_score = max(0.0, min(1.0, raw_score - penalties))
        is_usable = final_score >= self.min_usable_threshold

        metrics = {
            "crop_width": cw,
            "crop_height": ch,
            "aspect_ratio": round(aspect_ratio, 2),
            "laplacian_variance": round(lap_var, 2),
            "blur_threshold": self.blur_threshold,
            "orientation": orient_clean,
            "is_severely_occluded": is_severely_occluded,
            "violations_count": len(violations),
        }

        return QualityResult(
            quality_score=round(final_score, 4),
            is_usable=is_usable,
            blur_variance=round(lap_var, 2),
            blur_score=round(blur_score, 4),
            resolution_score=round(res_score, 4),
            edge_score=round(edge_score, 4),
            orientation_score=round(orient_score, 4),
            penalties=round(penalties, 4),
            violations=violations,
            metrics=metrics,
        )
