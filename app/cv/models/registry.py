import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Central configuration and validation for CV Models.
    Requires explicit selection and validation of models before use.
    """

    def __init__(self, config_overrides: dict = None):
        default_model_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "models")
        self.config = {
            "model_dir": os.environ.get("CV_MODEL_DIR") or default_model_dir,
            "detector_model": os.environ.get("CV_DETECTOR_MODEL", "yolo26n.pt"),
            "device": os.environ.get("CV_DEVICE", "cpu"),
            "tracker_config": os.environ.get("CV_TRACKER_CONFIG", "bytetrack.yaml")
        }
        
        if config_overrides:
            self.config.update(config_overrides)

    def validate(self):
        """
        Validates that required configuration is present and models physically exist.
        Fails fast if misconfigured.
        """
        model_dir = self.config.get("model_dir")
        if not model_dir:
            raise ValueError("CV_MODEL_DIR must be explicitly configured.")
            
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists() or not model_dir_path.is_dir():
            raise ValueError(f"CV_MODEL_DIR '{model_dir}' does not exist or is not a directory.")

        detector_model = self.config.get("detector_model")
        if not detector_model:
            raise ValueError("CV_DETECTOR_MODEL must be configured.")
            
        detector_path = model_dir_path / detector_model
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector model not found: {detector_path}. Will NOT download a fallback.")
            
        logger.info(f"ModelRegistry validated successfully. Using detector {detector_model} on {self.config.get('device')} from {model_dir}")

    def get_detector_path(self) -> Path:
        """Returns the absolute path to the detector model."""
        return Path(self.config["model_dir"]) / self.config["detector_model"]
        
    def get_tracker_config(self) -> str:
        """Returns the tracker config name (e.g. bytetrack.yaml)"""
        return self.config["tracker_config"]
        
    def get_device(self) -> str:
        """Returns the device to run inference on."""
        return self.config["device"]
