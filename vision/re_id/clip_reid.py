"""
clip_reid.py — Fine-Tuned CLIP-ReID Feature Extractor
=====================================================
Loads Re-ID fine-tuned CLIP checkpoint for surveillance person matching.
Dynamically exposes embedding_dimension (D) based on checkpoint architecture.
Performs real model inference; raises explicit errors if model file or ML dependencies are missing.
"""
import os
from typing import Optional
import numpy as np

from vision.re_id.base import BaseReIDModel


class CLIPReIDModel(BaseReIDModel):
    """
    Fine-Tuned CLIP-ReID Model.
    Exposes dynamic output dimension D depending on fine-tuned checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str,
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self._model_name = f"CLIP-ReID ({os.path.basename(checkpoint_path)})"

        if not os.path.exists(checkpoint_path) and not checkpoint_path.startswith("hf://"):
            raise FileNotFoundError(f"CLIP-ReID checkpoint file not found: '{checkpoint_path}'")

        try:
            import torch
            import transformers
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "CLIPReIDModel requires 'torch', 'transformers', and 'Pillow' packages to be installed. "
                "No fake synthetic fallbacks are permitted."
            ) from e

        self.torch = torch
        self.Image = Image

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        # Load PyTorch checkpoint / TorchScript / HuggingFace model
        from transformers import AutoModel, AutoProcessor
        try:
            self.processor = AutoProcessor.from_pretrained(checkpoint_path)
            self.model = AutoModel.from_pretrained(checkpoint_path).to(self.device)
            self.model.eval()
        except Exception:
            # Fallback to direct torch load
            self.model = torch.load(checkpoint_path, map_location=self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            self.processor = None

        # Determine dynamic embedding dimension
        if embedding_dim is not None:
            self._dim = embedding_dim
        elif hasattr(self.model, "config") and hasattr(self.model.config, "projection_dim"):
            self._dim = self.model.config.projection_dim
        elif hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            self._dim = self.model.config.hidden_size
        else:
            self._dim = 512

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract D-dimensional L2-normalized feature embedding for a single crop using CLIP-ReID.
        """
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            raise ValueError("Crop must be a non-empty numpy image array")

        if crop.ndim == 3 and crop.shape[2] == 3:
            rgb = crop[:, :, ::-1]
        elif crop.ndim == 3 and crop.shape[2] == 4:
            rgb = crop[:, :, [2, 1, 0]]
        else:
            rgb = crop

        pil_img = self.Image.fromarray(rgb)

        if self.processor is not None:
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                features = self.model.get_image_features(**inputs)
                vec = features.cpu().numpy().flatten().astype(np.float32)
        else:
            # Simple PyTorch tensor pass
            tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with self.torch.no_grad():
                features = self.model(tensor)
                vec = features.cpu().numpy().flatten().astype(np.float32)

        return self.l2_normalize(vec)
