"""
test_re_id.py — Unit Tests for Re-ID Model Abstraction Layer (Group C)
"""
import pytest
import numpy as np
from vision.re_id.base import BaseReIDModel
from vision.re_id.openai_clip import OpenAICLIPModel
from vision.re_id.clip_reid import CLIPReIDModel


class ExplicitTestMockReIDModel(BaseReIDModel):
    """Explicit Mock Re-ID model for testing BaseReIDModel contract."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"ExplicitTestMock-{self._dim}D"

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            raise ValueError("Crop cannot be empty")
        # Deterministic normalized vector derived from crop size
        seed = int(crop.shape[0] * crop.shape[1]) % 1000
        rng = np.random.RandomState(seed)
        raw = rng.randn(self._dim).astype(np.float32)
        return self.l2_normalize(raw)


def test_base_reid_model_contract():
    model_512 = ExplicitTestMockReIDModel(dim=512)
    model_768 = ExplicitTestMockReIDModel(dim=768)

    assert model_512.embedding_dimension == 512
    assert model_768.embedding_dimension == 768

    crop = np.zeros((100, 50, 3), dtype=np.uint8)

    emb_512 = model_512.extract_embedding(crop)
    assert emb_512.shape == (512,)
    assert abs(np.linalg.norm(emb_512) - 1.0) < 1e-5

    emb_768 = model_768.extract_embedding(crop)
    assert emb_768.shape == (768,)
    assert abs(np.linalg.norm(emb_768) - 1.0) < 1e-5


def test_base_reid_batch_extraction():
    model = ExplicitTestMockReIDModel(dim=256)
    crops = [
        np.zeros((100, 50, 3), dtype=np.uint8),
        np.zeros((120, 60, 3), dtype=np.uint8),
        np.zeros((80, 40, 3), dtype=np.uint8),
    ]

    batch_emb = model.extract_batch(crops)

    assert batch_emb.shape == (3, 256)
    norms = np.linalg.norm(batch_emb, axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-5


def test_l2_normalize_utility():
    vec_1d = np.array([3.0, 4.0], dtype=np.float32)
    norm_1d = BaseReIDModel.l2_normalize(vec_1d)
    assert np.allclose(norm_1d, [0.6, 0.8])

    vec_2d = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    norm_2d = BaseReIDModel.l2_normalize(vec_2d)
    assert np.allclose(np.linalg.norm(norm_2d, axis=1), [1.0, 1.0])

    zero_vec = np.zeros(4, dtype=np.float32)
    assert np.allclose(BaseReIDModel.l2_normalize(zero_vec), zero_vec)


def test_clip_reid_missing_checkpoint_error():
    """Supervisor Correction 1: Explicit error handling when checkpoint missing."""
    with pytest.raises(FileNotFoundError, match="checkpoint file not found"):
        CLIPReIDModel(checkpoint_path="/path/to/nonexistent/checkpoint.pt")
