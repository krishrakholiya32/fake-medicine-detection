from datetime import datetime

from sqlalchemy import String, DateTime, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.core.database import Base


class PillCheck(Base):
    __tablename__ = "pill_checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    claimed_drug: Mapped[str] = mapped_column(String(256))
    predicted_drug: Mapped[str] = mapped_column(String(256))
    confidence: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(16))          # match | mismatch | uncertain
    top_predictions: Mapped[dict] = mapped_column(JSON)        # [{"label": ..., "confidence": ...}, ...]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
