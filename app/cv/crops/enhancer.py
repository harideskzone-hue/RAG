import cv2
import numpy as np
from typing import Tuple


class CropEnhancer:
    """
    Enhances crop visual quality:
    1. Increases sharpness and edge definition via unsharp masking.
    2. Improves illumination and contrast using CLAHE in LAB color space.
    3. Centers the enhanced person crop on a clean, professional white background canvas.
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 128), padding: int = 12):
        self.target_size = target_size  # (height, width)
        self.padding = padding

    def enhance(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Applies sharpening, contrast optimization, and white background canvas padding.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        # 1. CLAHE Contrast & Lighting Optimization in LAB Color Space
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 2. Unsharp Masking for Sharpness & Fine Detail Recovery
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0, sigmaY=2.0)
        sharpened = cv2.addWeighted(enhanced, 1.45, gaussian, -0.45, 0)

        # 3. Clean White Background Canvas with Aspect-Preserving Centering
        target_h, target_w = self.target_size
        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)

        h, w = sharpened.shape[:2]
        max_h = target_h - (2 * self.padding)
        max_w = target_w - (2 * self.padding)
        scale = min(max_h / h, max_w / w)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        resized = cv2.resize(sharpened, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        # Center on white background
        y_offset = (target_h - nh) // 2
        x_offset = (target_w - nw) // 2

        canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
        return canvas
