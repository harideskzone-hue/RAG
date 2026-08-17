import os
import numpy as np
import cv2
from typing import List
try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    FeatureExtractor = None

class OSNetExtractor:
    """
    Extracts 512-dimensional visual feature embeddings from quality-approved crops.
    Used for Person Re-Identification (Re-ID).
    """
    def __init__(self, model_name: str = "osnet_x1_0", model_path: str = None, device: str = "cpu"):
        self.vector_dim = 512
        if model_path is None:
            # Check default project models directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            msmt17_path = os.path.join(project_root, "models", "osnet_x1_0_msmt17.pth")
            if os.path.exists(msmt17_path):
                model_path = msmt17_path

        if FeatureExtractor is not None:
            self.extractor = FeatureExtractor(
                model_name=model_name,
                model_path=model_path if model_path and os.path.exists(model_path) else "",
                device=device,
                verbose=False
            )
        else:
            self.extractor = None
        
    def extract(self, crop_bgr: np.ndarray) -> List[float]:
        """
        Extracts the feature vector from the crop.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot extract embedding from empty crop")
            
        if self.extractor is None:
            vec = np.ones(512, dtype=np.float32) / np.sqrt(512)
            return vec.tolist()

        # FeatureExtractor expects list of images.
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = self.extractor([crop_rgb])
        
        # features is a tensor of shape (1, 512)
        embedding = features[0].detach().cpu().numpy()
        
        # Normalize to unit length (cosine similarity requirement)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
