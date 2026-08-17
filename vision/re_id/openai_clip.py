"""
openai_clip.py — OpenAI CLIP Baseline Feature Extractor
======================================================
Loads generic OpenAI CLIP (ViT-B/16) for benchmark baseline feature extraction.
Exposes embedding_dimension = 512.
Performs real model inference when torch and transformers are available.
Raises explicit ImportError if required ML libraries are missing (no fake fallbacks).
"""
from typing import Optional
import numpy as np

from vision.re_id.base import BaseReIDModel


class OpenAICLIPModel(BaseReIDModel):
    """
    OpenAI CLIP ViT-B/16 Baseline Model.
    Outputs 512-dimensional L2-normalized image embeddings.
    """

    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch16", device: Optional[str] = None) -> None:
        self._model_name = f"OpenAI-CLIP ({model_name_or_path})"
        self._dim = 512

        try:
            import torch
            import transformers
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "OpenAICLIPModel requires 'torch', 'transformers', and 'Pillow' packages to be installed. "
                "No fake synthetic fallbacks are permitted."
            ) from e

        self.torch = torch
        self.Image = Image

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract 512-D L2-normalized feature embedding for a single crop using OpenAI CLIP.
        """
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("Crop must be a non-empty numpy image array")

        # Convert BGR (OpenCV) to RGB (PIL Image)
        if crop.ndim == 3 and crop.shape[2] == 3:
            rgb = crop[:, :, ::-1]
        elif crop.ndim == 3 and crop.shape[2] == 4:
            rgb = crop[:, :, [2, 1, 0]]
        else:
            rgb = crop

        pil_img = self.Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                vec_tensor = feats.pooler_output
            elif hasattr(feats, "image_embeds"):
                vec_tensor = feats.image_embeds
            elif isinstance(feats, self.torch.Tensor):
                vec_tensor = feats
            else:
                vec_tensor = feats[0]
            vec = vec_tensor.cpu().numpy().flatten().astype(np.float32)

        return self.l2_normalize(vec)
