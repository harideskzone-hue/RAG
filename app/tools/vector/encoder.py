import os
from abc import ABC, abstractmethod

class VectorEncoder(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        pass

class ModelFreeVectorEncoder(VectorEncoder):
    def encode(self, text: str) -> list[float]:
        # Return a dummy embedding (MiniLM is 384 dimensions)
        return [0.1] * 384

class MiniLMVectorEncoder(VectorEncoder):
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            import logging
            logging.warning("sentence-transformers not installed, MiniLMVectorEncoder will fail on encode.")
            self.model = None

    def encode(self, text: str) -> list[float]:
        if not self.model:
            return [0.1] * 384
        return self.model.encode(text).tolist()

def get_vector_encoder() -> VectorEncoder:
    from app.platform.config.config import config
    if getattr(config, "model_free", False):
        return ModelFreeVectorEncoder()
    return MiniLMVectorEncoder()
