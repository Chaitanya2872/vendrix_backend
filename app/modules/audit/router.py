from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.common.dependencies import require_roles
from app.db.session import get_db
from app.models import AuditLog, User
router = APIRouter(prefix="/audit-logs", tags=["audit"])
@router.get("")
def list_logs(limit: int = Query(100, le=500), db: Session = Depends(get_db), _: User = Depends(require_roles("ADMIN"))):
    return db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).all()
