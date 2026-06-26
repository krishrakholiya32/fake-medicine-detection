import cv2
import numpy as np
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.pill_check import PillCheck
from app.schemas.pill_check import PillCheckResult
from app.services.classification.verification import run_pill_check

router = APIRouter(prefix="/api/check", tags=["check"])

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/pill", response_model=PillCheckResult)
async def check_pill(
    file: UploadFile, claimed_drug: str = Form(...), db: AsyncSession = Depends(get_db)
):
    if file.content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(400, f"unsupported image content type: {file.content_type}")

    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode image")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = run_pill_check(image_rgb, claimed_drug)

    check = PillCheck(
        original_filename=file.filename,
        claimed_drug=claimed_drug,
        predicted_drug=result["predicted_drug"],
        confidence=result["confidence"],
        verdict=result["verdict"],
        top_predictions=result["top_predictions"],
    )
    db.add(check)
    await db.commit()
    await db.refresh(check)

    return PillCheckResult(
        id=check.id,
        claimed_drug=claimed_drug,
        predicted_drug=result["predicted_drug"],
        confidence=result["confidence"],
        verdict=result["verdict"],
        top_predictions=result["top_predictions"],
    )
