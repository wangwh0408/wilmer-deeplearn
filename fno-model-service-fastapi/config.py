import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'fno-model-service-fastapi-secret-key-2024')
    
    HOST: str = os.environ.get('HOST', '0.0.0.0')
    PORT: int = int(os.environ.get('PORT', 8000))
    DEBUG: bool = os.environ.get('DEBUG', 'True').lower() == 'true'
    RELOAD: bool = os.environ.get('RELOAD', 'True').lower() == 'true'
    
    MODEL_PATH: str = os.environ.get(
        'MODEL_PATH', 
        os.path.join(os.path.dirname(__file__), '..', 'fno2_model.pth')
    )
    
    DEVICE: str = os.environ.get('DEVICE', 'cpu')
    
    DEFAULT_MODES: int = int(os.environ.get('DEFAULT_MODES', 12))
    DEFAULT_WIDTH: int = int(os.environ.get('DEFAULT_WIDTH', 32))
    DEFAULT_IN_CHANNELS: int = int(os.environ.get('DEFAULT_IN_CHANNELS', 3))
    DEFAULT_OUT_CHANNELS: int = int(os.environ.get('DEFAULT_OUT_CHANNELS', 1))
    DEFAULT_RESOLUTION: int = int(os.environ.get('DEFAULT_RESOLUTION', 64))
    
    MAX_BATCH_SIZE: int = int(os.environ.get('MAX_BATCH_SIZE', 32))
    MAX_RESOLUTION: int = int(os.environ.get('MAX_RESOLUTION', 256))
    
    CORS_ENABLED: bool = os.environ.get('CORS_ENABLED', 'True').lower() == 'true'
    CORS_ORIGINS: str = os.environ.get('CORS_ORIGINS', '*')
    
    RATE_LIMIT_ENABLED: bool = os.environ.get('RATE_LIMIT_ENABLED', 'False').lower() == 'true'
    RATE_LIMIT_PER_MINUTE: int = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 100))
    
    API_TITLE: str = "FNO Model Service"
    API_DESCRIPTION: str = "Web service for FNO (Fourier Neural Operator) model inference"
    API_VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def get_config_dict(self) -> Dict[str, Any]:
        return {
            'host': self.HOST,
            'port': self.PORT,
            'debug': self.DEBUG,
            'model_path': self.MODEL_PATH,
            'device': self.DEVICE,
            'default_modes': self.DEFAULT_MODES,
            'default_width': self.DEFAULT_WIDTH,
            'default_in_channels': self.DEFAULT_IN_CHANNELS,
            'default_out_channels': self.DEFAULT_OUT_CHANNELS,
            'default_resolution': self.DEFAULT_RESOLUTION,
            'max_batch_size': self.MAX_BATCH_SIZE,
            'max_resolution': self.MAX_RESOLUTION,
            'cors_enabled': self.CORS_ENABLED,
            'rate_limit_enabled': self.RATE_LIMIT_ENABLED,
            'api_title': self.API_TITLE,
            'api_version': self.API_VERSION,
        }


settings = Settings()
