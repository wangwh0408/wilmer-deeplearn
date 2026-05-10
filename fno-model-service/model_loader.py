import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fno2_model import FNO2d
    FNO2D_AVAILABLE = True
except ImportError as e:
    logging.warning(f"fno2_model not found: {e}")
    FNO2D_AVAILABLE = False


from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    _instance = None
    _model = None
    _model_info = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._loaded = False

    @classmethod
    def get_model(cls) -> Optional[nn.Module]:
        if cls._model is None:
            cls.load_model()
        return cls._model

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        if cls._model_info is None:
            cls.load_model()
        return cls._model_info or {}

    @classmethod
    def load_model(cls, model_path: Optional[str] = None) -> Tuple[bool, str]:
        if model_path is None:
            model_path = Config.MODEL_PATH

        try:
            if not os.path.exists(model_path):
                error_msg = f"Model file not found: {model_path}"
                logger.error(error_msg)
                return False, error_msg

            logger.info(f"Loading model from: {model_path}")

            checkpoint = torch.load(model_path, map_location=torch.device(Config.DEVICE), weights_only=False)

            modes = checkpoint.get('modes', Config.DEFAULT_MODES)
            width = checkpoint.get('width', Config.DEFAULT_WIDTH)
            in_channels = checkpoint.get('config', {}).get('in_channels', Config.DEFAULT_IN_CHANNELS)
            out_channels = checkpoint.get('config', {}).get('out_channels', Config.DEFAULT_OUT_CHANNELS)
            resolution = checkpoint.get('resolution', Config.DEFAULT_RESOLUTION)

            logger.info(f"Model config from checkpoint: modes={modes}, width={width}, "
                       f"in_channels={in_channels}, out_channels={out_channels}, resolution={resolution}")

            if not FNO2D_AVAILABLE:
                error_msg = "FNO2d model class not available"
                logger.error(error_msg)
                return False, error_msg

            model = FNO2d(
                modes1=modes,
                modes2=modes,
                width=width,
                in_channels=in_channels,
                out_channels=out_channels
            ).to(Config.DEVICE)

            model.load_state_dict(checkpoint['model_state_dict'])

            model.eval()

            cls._model = model
            cls._model_info = {
                'path': model_path,
                'modes': modes,
                'width': width,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'resolution': resolution,
                'device': Config.DEVICE,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            cls._loaded = True

            logger.info(f"Model loaded successfully! Parameters: {cls._model_info['total_parameters']:,}")
            return True, "Model loaded successfully"

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def predict(
        cls,
        source: np.ndarray,
        return_numpy: bool = True
    ) -> Dict[str, Any]:
        model = cls.get_model()
        if model is None:
            raise RuntimeError("Model not loaded")

        if source.ndim == 2:
            source = source[np.newaxis, ..., np.newaxis]
        elif source.ndim == 3:
            source = source[np.newaxis, ...]
        elif source.ndim != 4:
            raise ValueError(f"Invalid input shape: {source.shape}. Expected 2D, 3D, or 4D array.")

        source_tensor = torch.tensor(source, dtype=torch.float32).to(Config.DEVICE)

        with torch.no_grad():
            output = model(source_tensor)

        result = {
            'input_shape': list(source.shape),
            'output_shape': list(output.shape),
        }

        if return_numpy:
            result['output'] = output.cpu().numpy()
            result['input'] = source
        else:
            result['output'] = output
            result['input'] = source_tensor

        return result

    @classmethod
    def predict_batch(
        cls,
        sources: List[np.ndarray],
        return_numpy: bool = True
    ) -> List[Dict[str, Any]]:
        results = []
        for source in sources:
            results.append(cls.predict(source, return_numpy))
        return results

    @classmethod
    def unload_model(cls):
        cls._model = None
        cls._model_info = None
        cls._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")
