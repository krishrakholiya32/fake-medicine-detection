from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.pill_check import PillCheck
from app.schemas.pill_check import PillCheckOut

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=List[PillCheckOut])
async def list_history(
    verdict: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(PillCheck).order_by(desc(PillCheck.created_at)).offset(offset).limit(limit)
    if verdict:
        q = q.where(PillCheck.verdict == verdict)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{check_id}", response_model=PillCheckOut)
async def get_history_detail(check_id: int, db: AsyncSession = Depends(get_db)):
    check = await db.get(PillCheck, check_id)
    if check is None:
        raise HTTPException(404, "check not found")
    return check
