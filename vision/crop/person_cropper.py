"""
person_cropper.py — Configurable Asymmetric Person Crop Module
================================================================
Extracts person crops from video frames with configurable asymmetric padding.
Ensures full coverage of head/hair and feet without modifying detection bboxes.
"""
from typing import Dict, List, Tuple, Union
import numpy as np


class PersonCropper:
    """
    Configurable Person Cropper with asymmetric top/bottom/side padding.
    """

    def __init__(
        self,
        top_pad: float = 0.25,
        bottom_pad: float = 0.10,
        side_pad: float = 0.10,
    ) -> None:
        if top_pad < 0 or bottom_pad < 0 or side_pad < 0:
            raise ValueError("Padding percentages must be non-negative")
        self.top_pad = top_pad
        self.bottom_pad = bottom_pad
        self.side_pad = side_pad

    def crop(
        self,
        frame: np.ndarray,
        bbox: Union[Tuple[int, int, int, int], List[int]],
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float, bool, List[int]]]]:
        """
        Crop a person from frame using asymmetric padding while clamping to bounds.

        Args:
            frame: Input image array (H, W, C) or (H, W)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            crop: Extracted image crop
            meta: Metadata dictionary containing crop bounds, padding, and original size
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: Must be non-empty numpy array")

        if len(bbox) != 4:
            raise ValueError("Bbox must contain 4 integer elements [x1, y1, x2, y2]")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        img_h, img_w = frame.shape[:2]

        # Clamp raw bbox to image dimensions
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            # Return empty crop
            empty_crop = np.zeros((0, 0, 3) if frame.ndim == 3 else (0, 0), dtype=frame.dtype)
            meta = {
                "valid": False,
                "original_bbox": [int(v) for v in bbox],
                "padded_bbox": [0, 0, 0, 0],
                "crop_w": 0,
                "crop_h": 0,
                "frame_w": img_w,
                "frame_h": img_h,
            }
            return empty_crop, meta

        # Apply asymmetric padding
        pad_t = int(round(bh * self.top_pad))
        pad_b = int(round(bh * self.bottom_pad))
        pad_s = int(round(bw * self.side_pad))

        nx1 = max(0, x1 - pad_s)
        ny1 = max(0, y1 - pad_t)
        nx2 = min(img_w, x2 + pad_s)
        ny2 = min(img_h, y2 + pad_b)

        crop_img = frame[ny1:ny2, nx1:nx2].copy()
        ch, cw = crop_img.shape[:2]

        meta = {
            "valid": True,
            "original_bbox": [x1, y1, x2, y2],
            "padded_bbox": [nx1, ny1, nx2, ny2],
            "crop_w": cw,
            "crop_h": ch,
            "frame_w": img_w,
            "frame_h": img_h,
            "padding_applied": {
                "top_px": pad_t,
                "bottom_px": pad_b,
                "side_px": pad_s,
            },
        }

        return crop_img, meta


def crop_person(
    frame: np.ndarray,
    bbox: Union[Tuple[int, int, int, int], List[int]],
    top_pad: float = 0.25,
    bottom_pad: float = 0.10,
    side_pad: float = 0.10,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, bool, List[int]]]]:
    """Helper function to perform one-off person cropping."""
    cropper = PersonCropper(top_pad=top_pad, bottom_pad=bottom_pad, side_pad=side_pad)
    return cropper.crop(frame, bbox)
