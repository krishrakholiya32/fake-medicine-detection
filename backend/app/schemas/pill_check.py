from datetime import datetime
from typing import List

from pydantic import BaseModel


class TopPrediction(BaseModel):
    label: str
    confidence: float


class PillCheckResult(BaseModel):
    id: int
    claimed_drug: str
    predicted_drug: str
    confidence: float
    verdict: str
    top_predictions: List[TopPrediction]


class PillCheckOut(BaseModel):
    id: int
    original_filename: str
    claimed_drug: str
    predicted_drug: str
    confidence: float
    verdict: str
    created_at: datetime

    class Config:
        from_attributes = True
