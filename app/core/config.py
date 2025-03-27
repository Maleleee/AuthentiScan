from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AuthentiScan"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    TEXT_DETECTION_MODEL: str = "microsoft/deberta-v3-base"
    ZERO_SHOT_MODEL: str = "facebook/bart-large-mnli"
    IMAGE_DETECTION_MODEL: str = "stylegan-detector"
    VIDEO_DETECTION_MODEL: str = "deepfake-detector"
    
    # Confidence Thresholds
    TEXT_CONFIDENCE_THRESHOLD: float = 0.7
    IMAGE_CONFIDENCE_THRESHOLD: float = 0.8
    VIDEO_CONFIDENCE_THRESHOLD: float = 0.85
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        case_sensitive = True

settings = Settings() 