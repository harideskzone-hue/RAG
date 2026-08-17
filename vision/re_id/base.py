"""
base.py — Abstract Base Class for VISTA Re-ID Feature Extractors
=================================================================
Defines the clean abstract contract for all Re-ID vision embedding models.
Requires dynamic reporting of embedding_dimension (D) to handle generic CLIP (512-D),
fine-tuned CLIP-ReID, or alternate visual encoders.
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseReIDModel(ABC):
    """
    Abstract Base Class for Person Re-ID Feature Extraction Models.
    """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the output feature embedding dimension (D)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model identification string."""
        pass

    @abstractmethod
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalized feature embedding for a single person crop.

        Args:
            crop: BGR/RGB image array (H, W, C)

        Returns:
            embedding: 1D L2-normalized float32 numpy array of shape (D,)
        """
        pass

    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract L2-normalized feature embeddings for a batch of person crops.

        Args:
            crops: List of BGR/RGB image arrays

        Returns:
            embeddings: 2D L2-normalized float32 numpy array of shape (N, D)
        """
        if not crops:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = [self.extract_embedding(c) for c in crops]
        return np.vstack(embeddings)

    @staticmethod
    def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Helper utility for L2 normalization of vectors or matrices."""
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            if norm < eps:
                return np.zeros_like(vec)
            return vec / norm
        elif vec.ndim == 2:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            return vec / norms
        else:
            raise ValueError(f"Expected 1D or 2D array for L2 normalization, got {vec.ndim}D")
