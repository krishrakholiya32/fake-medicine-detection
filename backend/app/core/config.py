from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "Fake Medicine Detection"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:80",
    ]

    DATABASE_URL: str = "postgresql+asyncpg://pillcheck:pillcheck@db:5432/pillcheck"

    MODELS_PATH: str = "/app/models"
    MODEL_FILENAME: str = "pill_classifier_effnetb0.onnx"
    LABELS_FILENAME: str = "labels.json"

    MATCH_CONFIDENCE_THRESHOLD: float = 0.5   # below this, verdict is "uncertain" even if names match

    class Config:
        env_file = ".env"


settings = Settings()
