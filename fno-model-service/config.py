import os
from typing import Optional, Dict, Any


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'fno-model-service-secret-key-2024')
    
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', 'fno2_model.pth'))
    
    DEVICE = os.environ.get('DEVICE', 'cpu')
    
    DEFAULT_MODES = int(os.environ.get('DEFAULT_MODES', 12))
    DEFAULT_WIDTH = int(os.environ.get('DEFAULT_WIDTH', 32))
    DEFAULT_IN_CHANNELS = int(os.environ.get('DEFAULT_IN_CHANNELS', 3))
    DEFAULT_OUT_CHANNELS = int(os.environ.get('DEFAULT_OUT_CHANNELS', 1))
    DEFAULT_RESOLUTION = int(os.environ.get('DEFAULT_RESOLUTION', 64))
    
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 32))
    MAX_RESOLUTION = int(os.environ.get('MAX_RESOLUTION', 256))
    
    CORS_ENABLED = os.environ.get('CORS_ENABLED', 'True').lower() == 'true'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'False').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 100))
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        return {
            'host': cls.HOST,
            'port': cls.PORT,
            'debug': cls.DEBUG,
            'model_path': cls.MODEL_PATH,
            'device': cls.DEVICE,
            'default_modes': cls.DEFAULT_MODES,
            'default_width': cls.DEFAULT_WIDTH,
            'default_in_channels': cls.DEFAULT_IN_CHANNELS,
            'default_out_channels': cls.DEFAULT_OUT_CHANNELS,
            'default_resolution': cls.DEFAULT_RESOLUTION,
            'max_batch_size': cls.MAX_BATCH_SIZE,
            'max_resolution': cls.MAX_RESOLUTION,
            'cors_enabled': cls.CORS_ENABLED,
            'rate_limit_enabled': cls.RATE_LIMIT_ENABLED,
        }
